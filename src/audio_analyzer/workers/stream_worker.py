import asyncio
import contextlib
import logging
import os
import signal
import time
import uuid

from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.adapters.repository.unit_of_work import SqlAlchemyUnitOfWork
from audio_analyzer.adapters.storage.storage_factory import get_storage_adapter
from audio_analyzer.services.job_service import JobService
from audio_analyzer.services.pipeline_factory import get_shared_pipeline

logger = logging.getLogger(__name__)


class RedisStreamWorker:
    """
    Redis Streams Tüketici Grubu (Consumer Group) Worker Hizmeti.
    100+ sunucuda eşzamanlı olarak çalışarak Redis Stream üzerinden gelen işleri
    sıfır kilitlenme (key contention) ve tam paralellikle tüketir.
    """

    def __init__(
        self,
        consumer_name: str | None = None,
        adapter: RedisStreamAdapter | None = None,
    ):
        self.consumer_name = consumer_name or f"worker_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        self.adapter = adapter or RedisStreamAdapter()
        self.running = False
        self._background_tasks: set[asyncio.Task] = set()
        self._liveness_task: asyncio.Task | None = None

    async def _liveness_heartbeat_loop(self, interval_sec: float | None = None):
        """Worker'ın hayatta olduğunu Redis'e periyodik olarak (TTL>=60s) bildirir."""
        from audio_analyzer.config import get_settings

        settings = get_settings()
        ttl = max(60, settings.worker_heartbeat_ttl_sec)
        interval = interval_sec or settings.worker_liveness_interval_sec
        client = self.adapter.get_client()
        try:
            while self.running:
                now_str = str(time.time())
                await client.set("audio_analyzer:worker_heartbeat", now_str, ex=ttl)
                await client.set(f"audio_analyzer:worker_heartbeat:{self.consumer_name}", now_str, ex=ttl)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Worker liveness heartbeat note: %s", e)

    async def _heartbeat_loop(self, msg_id: str, interval_sec: float = 15.0):
        try:
            while self.running:
                await asyncio.sleep(interval_sec)
                await self.adapter.claim_message_heartbeat(self.consumer_name, msg_id)
        except asyncio.CancelledError:
            pass

    async def process_single_message(self, msg_id: str, fields: dict) -> bool:
        """Tek bir Redis Stream mesajını çözer, JobService ile yürütür ve karara göre ACK / RETRY / DLQ yönetir."""
        job_id_str = fields.get("job_id")
        if not job_id_str:
            logger.warning("Eksik job_id içeren mesaj es geçiliyor: %s", msg_id)
            await self.adapter.ack_message(msg_id)
            return False

        hb_task = asyncio.create_task(self._heartbeat_loop(msg_id))
        try:
            from audio_analyzer.adapters.repository.session import AsyncSessionLocal

            record_id = uuid.UUID(job_id_str)
            uow = SqlAlchemyUnitOfWork(session_factory=AsyncSessionLocal)
            storage = get_storage_adapter()

            async with uow:
                pipeline = get_shared_pipeline()
                job_service = JobService(
                    storage=storage,
                    repository=uow.repository,
                    pipeline=pipeline,
                    session_factory=AsyncSessionLocal,
                )
                res_status, attempts, err_msg = await job_service.execute_job_detailed(record_id)

            if res_status in ("COMPLETED", "SKIPPED"):
                await self.adapter.ack_message(msg_id)
                logger.info(
                    "Worker %s job_id=%s mesajını başardı (%s) ve XACK onayladı.",
                    self.consumer_name,
                    job_id_str,
                    res_status,
                )
                return True
            elif res_status == "FAILED":
                # Kalıcı hata veya Maksimum deneme aşıldı -> Dead-Letter Stream'e aktar ve XACK et
                await self.adapter.publish_to_dlq(
                    job_id=job_id_str,
                    error_message=err_msg or "Unknown Failure",
                    attempts=attempts,
                )
                await self.adapter.ack_message(msg_id)
                logger.warning(
                    "Worker %s job_id=%s (attempts=%d) FAILED durumunda DLQ'ya atıldı ve XACK onaylandı.",
                    self.consumer_name,
                    job_id_str,
                    attempts,
                )
                return False
            elif res_status == "RETRY":
                # Geçici hata -> Worker loop'unu bloke etmeden arka planda üstel backoff sonrası yeniden akışa yayınla ve eski mesajı XACK et
                backoff_sec = min(30.0, float(2 ** max(0, attempts - 1)))
                logger.info(
                    "Worker %s job_id=%s (attempts=%d) geçici hata aldı. %.1f sn backoff arka planda beklenip yeniden akışa yayınlanacak.",
                    self.consumer_name,
                    job_id_str,
                    attempts,
                    backoff_sec,
                )

                async def _delayed_retry(b_sec: float, j_id: str, f_fields: dict, m_id: str):
                    try:
                        await asyncio.sleep(b_sec)
                        await self.adapter.publish_job(
                            job_id=j_id,
                            file_name=f_fields.get("file_name", ""),
                            callback_url=f_fields.get("callback_url", None),
                        )
                        await self.adapter.ack_message(m_id)
                    except Exception as retry_err:
                        logger.error("Gecikmeli yeniden deneme hatası (job_id=%s): %s", j_id, retry_err)

                retry_task = asyncio.create_task(_delayed_retry(backoff_sec, job_id_str, fields, msg_id))
                self._background_tasks.add(retry_task)
                retry_task.add_done_callback(self._background_tasks.discard)
                await asyncio.sleep(0)
                return False
            else:
                await self.adapter.ack_message(msg_id)
                return False

        except Exception as ex:
            logger.error(
                "Worker %s msg_id=%s beklenmeyen hata aldı: %s",
                self.consumer_name,
                msg_id,
                ex,
                exc_info=True,
            )
            try:
                max_attempts = int(os.getenv("MAX_JOB_ATTEMPTS", "3"))
                delivery_count = await self.adapter.get_pending_delivery_count(msg_id)
                if delivery_count >= max_attempts:
                    logger.error("Mesaj %s zehirli (poison pill, delivery_count=%d >= %d). DLQ'ya atılıp FAILED yapılıyor ve XACK ediliyor.", msg_id, delivery_count, max_attempts)
                    if job_id_str:
                        uow = SqlAlchemyUnitOfWork(session_factory=AsyncSessionLocal)
                        async with uow:
                            await uow.repository.handle_job_failure(
                                uuid.UUID(job_id_str),
                                error_message=f"Poison pill message error: {ex}",
                                is_transient=False,
                                max_attempts=max_attempts,
                            )
                    await self.adapter.publish_to_dlq(
                        job_id=job_id_str or "unknown",
                        error_message=str(ex),
                        attempts=delivery_count,
                    )
                    await self.adapter.ack_message(msg_id)
            except Exception as dlq_err:
                logger.error("DLQ poison pill işleme hatası: %s", dlq_err)

            return False
        finally:
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb_task

    async def run(self):
        """Worker döngüsünü başlatır (XREADGROUP + XAUTOCLAIM)."""
        self.running = True
        prefetch_count = int(os.getenv("WORKER_PREFETCH", "1"))
        min_idle_ms = int(os.getenv("STREAM_CLAIM_IDLE_MS", "900000"))
        claim_interval_sec = float(os.getenv("STREAM_CLAIM_INTERVAL_SEC", "60"))

        logger.info(
            "Redis Stream Worker başlatılıyor: consumer_name=%s, prefetch=%d, claim_idle_ms=%d",
            self.consumer_name,
            prefetch_count,
            min_idle_ms,
        )

        await self.adapter.create_consumer_group()
        self._liveness_task = asyncio.create_task(self._liveness_heartbeat_loop())
        self._background_tasks.add(self._liveness_task)
        self._liveness_task.add_done_callback(self._background_tasks.discard)

        last_claim_time = time.monotonic()

        while self.running:
            try:
                # 1. Okunmamış mesajları XREADGROUP ile çek (WORKER_PREFETCH)
                messages = await self.adapter.consume_messages(
                    consumer_name=self.consumer_name,
                    count=prefetch_count,
                    block_ms=1000,
                )

                for msg_id, fields in messages:
                    if not self.running:
                        break
                    await self.process_single_message(msg_id, fields)

                # 2. Periyodik olarak (zamana bağlı STREAM_CLAIM_INTERVAL_SEC) yetim mesajları XAUTOCLAIM ile devral
                now = time.monotonic()
                if now - last_claim_time >= claim_interval_sec:
                    last_claim_time = now
                    claimed = await self.adapter.claim_pending_messages(
                        consumer_name=self.consumer_name,
                        min_idle_time_ms=min_idle_ms,
                        count=prefetch_count,
                    )
                    for msg_id, fields in claimed:
                        logger.info(
                            "Worker %s sahipsiz kalan %s mesajını XAUTOCLAIM ile devraldı.",
                            self.consumer_name,
                            msg_id,
                        )
                        await self.process_single_message(msg_id, fields)

            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.error("Worker döngü hatası: %s", err, exc_info=True)
                await asyncio.sleep(1.0)

        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
        logger.info("Redis Stream Worker durduruldu: consumer_name=%s", self.consumer_name)

    def stop(self):
        self.running = False
        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()


async def start_worker_main():
    os.environ.setdefault("USE_REDIS_STREAM", "true")
    from audio_analyzer.adapters.storage.storage_factory import (
        assert_storage_shared_across_processes,
    )
    from audio_analyzer.utils.cpu_budget import setup_cpu_thread_budget

    setup_cpu_thread_budget()
    assert_storage_shared_across_processes()

    worker = RedisStreamWorker()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except NotImplementedError:
            pass  # Windows signals

    await worker.run()


def main():
    import sys

    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(start_worker_main())
    except KeyboardInterrupt:
        logger.info("Worker kullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as exc:
        logger.critical(
            "Worker başlatılırken veya çalışırken kritik hata oluştu: %s",
            exc,
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
