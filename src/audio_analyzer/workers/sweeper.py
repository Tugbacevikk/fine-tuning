"""
Sweeper Service (Kuyruk Temizleyici ve Garanti Hizmeti).
Komut: python -m audio_analyzer.workers.sweeper
Redis kilidi (sweeper_lock) ile küme (cluster) ortamında tek bir aktif instance olarak çalışır.

Sorumluluklar:
1. PENDING Sweep: PENDING durumunda X dakikadır bekleyen kayıtları Redis Stream'e yeniden yayınlar.
2. PROCESSING Sweep: PROCESSING_STALE_SEC süresini aşan askıda/çökmüş kayıtları:
   - attempts < MAX_JOB_ATTEMPTS ise PENDING durumuna çekip yeniden akışa koyar.
   - attempts >= MAX_JOB_ATTEMPTS ise FAILED durumuna getirir, DLQ akışına atar ve FAILED webhook gönderir.
"""

import asyncio
import logging
import os
import signal
import uuid

from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.adapters.repository.session import AsyncSessionLocal
from audio_analyzer.adapters.repository.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)


class SweeperService:
    def __init__(
        self,
        adapter: RedisStreamAdapter | None = None,
        lock_name: str = "sweeper_lock",
        lock_ttl_sec: int = 60,
    ):
        self.adapter = adapter or RedisStreamAdapter()
        self.lock_name = lock_name
        self.lock_ttl_sec = lock_ttl_sec
        self.lock_id = uuid.uuid4().hex
        self.running = False

    async def _try_acquire_lock(self) -> bool:
        """Redis üzerinden dağıtık kilit edinmeye çalışır (`SET lock_name lock_id NX EX ttl`)."""
        client = self.adapter.get_client()
        try:
            acquired = await client.set(
                name=self.lock_name,
                value=self.lock_id,
                nx=True,
                ex=self.lock_ttl_sec,
            )
            return bool(acquired)
        except Exception as e:
            logger.warning("Sweeper kilit alma uyarısı: %s", e)
            return False

    async def _renew_lock(self) -> bool:
        """Kilit süresini uzatır."""
        client = self.adapter.get_client()
        try:
            val = await client.get(self.lock_name)
            if val == self.lock_id or (isinstance(val, bytes) and val.decode() == self.lock_id):
                await client.expire(self.lock_name, self.lock_ttl_sec)
                return True
            return False
        except Exception as e:
            logger.warning("Sweeper kilit yenileme hatası: %s", e)
            return False

    async def _release_lock(self):
        """Kilit bizdeyse serbest bırakır."""
        client = self.adapter.get_client()
        try:
            val = await client.get(self.lock_name)
            if val == self.lock_id or (isinstance(val, bytes) and val.decode() == self.lock_id):
                await client.delete(self.lock_name)
        except Exception:
            pass

    async def run_single_sweep(self):
        """Tek bir temizleme döngüsü yürütür."""
        pending_stale_sec = int(os.getenv("PENDING_STALE_SEC", "300"))
        processing_stale_sec = int(os.getenv("PROCESSING_STALE_SEC", "1800"))
        max_attempts = int(os.getenv("MAX_JOB_ATTEMPTS", "3"))

        uow = SqlAlchemyUnitOfWork(session_factory=AsyncSessionLocal)
        async with uow:
            repo = uow.repository

            # (a) PENDING Sweep: Lag > 0 ise atla, aksi halde yetim kayıtları yayınla (updated_at güncellenerek)
            lag = await self.adapter.get_consumer_group_lag()
            if lag is not None and lag > 0:
                logger.info("Sweeper: Consumer group lag = %d > 0. Kuyrukta bekleyen işler var, PENDING sweep atlanıyor.", lag)
            else:
                stale_pending = await repo.get_stale_pending_records(stale_seconds=pending_stale_sec, limit=50)
                if stale_pending:
                    logger.info("Sweeper: %d adet PENDING durumunda bekleyen kayıt bulundu, akışa yayınlanıyor.", len(stale_pending))
                    for record in stale_pending:
                        try:
                            await repo.touch_pending(record.id)
                            await self.adapter.publish_job(
                                job_id=str(record.id),
                                file_name=record.file_name,
                                callback_url=record.callback_url,
                            )
                            logger.info("Sweeper: PENDING Job %s yeniden akışa yayınlandı.", record.id)
                        except Exception as ex:
                            logger.error("Sweeper: Job %s yayınlanırken hata: %s", record.id, ex)

            # (b) PROCESSING Sweep: Süresi dolan PROCESSING kayıtları toparla
            stale_processing = await repo.get_stale_processing_records(stale_seconds=processing_stale_sec, limit=50)
            if stale_processing:
                logger.info("Sweeper: %d adet askıda kalmış (PROCESSING) kayıt bulundu.", len(stale_processing))
                for record in stale_processing:
                    if record.attempts < max_attempts:
                        # Yeniden denenebilir -> PENDING yap ve akışa fırlat
                        await repo.reset_record_to_pending(record.id)
                        await self.adapter.publish_job(
                            job_id=str(record.id),
                            file_name=record.file_name,
                            callback_url=record.callback_url,
                        )
                        logger.warning(
                            "Sweeper: Askıda kalan Job %s (attempts=%d) PENDING durumuna sıfırlanıp yeniden akışa verildi.",
                            record.id,
                            record.attempts,
                        )
                    else:
                        # Maksimum deneme aşıldı -> FAILED yap, DLQ'ya aktar ve Transactional Outbox'a yaz
                        err_msg = f"Job processing timed out after {processing_stale_sec}s (stale processing)"
                        attempts, _is_final = await repo.handle_job_failure(
                            record.id,
                            error_message=err_msg,
                            is_transient=False,
                            max_attempts=max_attempts,
                        )
                        await self.adapter.publish_to_dlq(
                            job_id=str(record.id),
                            error_message=err_msg,
                            attempts=attempts,
                        )
                        if record.callback_url:
                            import json

                            payload = {
                                "job_id": str(record.id),
                                "file_name": record.file_name,
                                "status": "FAILED",
                                "attempts": attempts,
                                "error_message": err_msg,
                            }
                            await repo.create_webhook_delivery(
                                job_id=record.id,
                                url=record.callback_url,
                                payload=json.dumps(payload),
                            )

                        logger.error(
                            "Sweeper: Askıda kalan Job %s (attempts=%d) maks limiti aştı, FAILED yapılıp DLQ'ya atıldı.",
                            record.id,
                            attempts,
                        )

            # (c) Retention Cleanup: Eski ses dosyaları ve eski veritabanı kayıtları temizliği
            try:
                from audio_analyzer.adapters.storage.storage_factory import (
                    get_storage_adapter,
                )
                from audio_analyzer.services.retention_service import RetentionService
                storage = get_storage_adapter()
                retention_svc = RetentionService(storage=storage, repository=repo)
                await retention_svc.cleanup_expired_audio_files()
                await retention_svc.cleanup_old_database_records()
            except Exception as ret_err:
                logger.warning("Sweeper: Retention temizliği uyarısı: %s", ret_err)

    async def run(self):
        """Sweeper ana döngüsü."""
        from audio_analyzer.config import get_settings
        interval_sec = get_settings().sweeper_interval_sec

        logger.info("Sweeper servisi başlatılıyor (interval=%.1fs, lock=%s)...", interval_sec, self.lock_name)

        while self.running:
            acquired = await self._try_acquire_lock()
            if acquired:
                try:
                    await self.run_single_sweep()
                except Exception as ex:
                    logger.error("Sweeper çalıştırma hatası: %s", ex, exc_info=True)
                finally:
                    await self._release_lock()
            else:
                logger.debug("Sweeper kilidi başkasında, bu döngü atlanıyor.")

            try:
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                break

        logger.info("Sweeper servisi durduruldu.")

    def stop(self):
        self.running = False


async def start_sweeper_main():
    sweeper = SweeperService()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, sweeper.stop)
        except NotImplementedError:
            pass

    await sweeper.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start_sweeper_main())
