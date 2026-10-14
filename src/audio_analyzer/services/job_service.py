import asyncio
import contextlib
import logging
import re
import uuid

from audio_analyzer.domain.interfaces import IAudioStorage, ITranscriptRepository
from audio_analyzer.domain.models import AudioRecord, JobStatus
from audio_analyzer.services.pipeline import AudioAnalysisPipeline
from audio_analyzer.utils.metrics import JOB_STATUS_COUNTER

logger = logging.getLogger(__name__)


def sanitize_error_message(ex: Exception) -> str:
    """
    Sistem dosya yollarını, sunucu dizin yapısını ve iç detayları sızdırmamak
    için kullanıcıya dönecek hata mesajını temizler.
    """
    err_type = type(ex).__name__
    err_str = str(ex)

    # Windows ve Unix mutlak dosya yollarını maskele (örn. C:\Users\... veya /var/...)
    err_str = re.sub(r'[A-Za-z]:\\[^\s:]+', '[FILE_PATH]', err_str)
    err_str = re.sub(r'/(?:[^\s:]+/)+[^\s:]+', '[FILE_PATH]', err_str)

    return f"{err_type}: {err_str}"


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


from audio_analyzer.domain.errors import is_transient_error


class JobService:
    """
    Ses Analiz İşleri ve Görev Yaşam Döngüsü Servisi (Job Lifecycle Service).
    Kullanıcı taleplerini alır, veritabanı durumlarını ve dosya depolamayı yönetir.
    """

    def __init__(
        self,
        storage: IAudioStorage,
        repository: ITranscriptRepository,
        pipeline: AudioAnalysisPipeline | None = None,
        session_factory: object | None = None,
    ):
        self.storage = storage
        self.repo = repository
        self.pipeline = pipeline
        self.session_factory = session_factory

    async def create_job(
        self,
        file_name: str,
        file_bytes: bytes,
        callback_url: str | None = None,
        external_id: str | None = None,
        num_speakers: int | None = None,
    ) -> uuid.UUID:
        """
        Yeni bir analiz görevi oluşturur (status='PENDING').
        İdempotency: external_id verilmişse ve aynı kaydolmuş iş varsa tekrar oluşturma, var olan job_id'yi dön.
        """
        if external_id:
            existing = await self.repo.get_record_by_external_id(external_id)
            if existing:
                logger.info("External ID '%s' ile eşleşen kayıt (%s) bulundu, mevcut job_id dönülüyor.", external_id, existing.id)
                return existing.id

        storage_uri = await asyncio.to_thread(self.storage.save, file_bytes, file_name)
        record_id = uuid.uuid4()

        record = AudioRecord(
            id=record_id,
            external_id=external_id,
            storage_uri=storage_uri,
            file_name=file_name,
            status=JobStatus.PENDING,
            callback_url=callback_url,
            num_speakers=num_speakers,
        )
        try:
            saved = await self.repo.save_record(record)
            return saved.id
        except Exception as ex:
            if external_id and ("unique" in str(ex).lower() or "integrity" in str(ex).lower() or "duplicate" in str(ex).lower()):
                existing = await self.repo.get_record_by_external_id(external_id)
                if existing:
                    logger.info("External ID '%s' çakışmasında yarış durumu yakalandı, mevcut job_id (%s) dönülüyor.", external_id, existing.id)
                    return existing.id
            raise ex

    async def execute_job(
        self, record_id: uuid.UUID, file_bytes: bytes | None = None, num_speakers: int | None = None
    ) -> bool:
        """
        Geriye dönük uyumluluk için boolean sonuç döndüren execute_job sarmalayıcısı.
        """
        status_str, _, _ = await self.execute_job_detailed(
            record_id, file_bytes=file_bytes, num_speakers=num_speakers
        )
        return status_str in ("COMPLETED", "SKIPPED")

    async def execute_job_detailed(
        self, record_id: uuid.UUID, file_bytes: bytes | None = None, num_speakers: int | None = None
    ) -> tuple[str, int, str | None]:
        """
        Arka plan worker'ı tarafından çağrılır.
        Kısa transaction ile durumu 'PROCESSING' yapar, DB bağlantısını serbest bırakarak pipeline'ı çalıştırır.
        Hata durumunda hata sınıflandırmasına (kalıcı vs geçici) göre retry veya FAILED kararı verir.
        Webhook istekleri Outbox tablosuna yazılır (CPU worker'ı yavaş/kapalı alıcılar için bekletilmez).
        Döner: (status_str, attempts, error_message)
        """
        import json
        import os

        stale_sec = int(os.getenv("PROCESSING_STALE_SEC", "1800"))
        max_attempts = int(os.getenv("MAX_JOB_ATTEMPTS", "3"))

        # Kısa Transaction 1: Atomik olarak işi devral (PENDING -> PROCESSING) ve hemen commit et
        claimed, record, is_completed = await self.repo.claim_job_atomically(
            record_id, stale_seconds=stale_sec
        )

        if is_completed:
            logger.info("Job %s zaten COMPLETED durumunda, tekrar çalıştırılmıyor.", record_id)
            return "COMPLETED", getattr(record, "attempts", 0), None

        if not claimed or not record:
            logger.info("Job %s başka bir worker tarafından işleniyor veya zaman aşımına uğramamış, atlanıyor.", record_id)
            return "SKIPPED", 0, None

        if num_speakers is None and getattr(record, "num_speakers", None) is not None:
            num_speakers = record.num_speakers

        # Claim token (fencing token)
        claim_token = getattr(record, "processing_started_at", None)

        # Heartbeat loop (Item 4)
        async def _heartbeat_loop():
            try:
                while True:
                    await asyncio.sleep(60.0)
                    if self.session_factory is not None:
                        async with self.session_factory() as hb_session:
                            from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
                            hb_repo = PostgresRepository(hb_session)
                            await hb_repo.touch_processing(record_id, claim_token)
                    else:
                        await self.repo.touch_processing(record_id, claim_token)
            except asyncio.CancelledError:
                pass
            except Exception as hb_err:
                logger.warning("Heartbeat loop warning for job %s: %s", record_id, hb_err)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        completed_success = False

        # --- DB BAĞLANTISI SERBEST: AI Pipeline (CPU/GPU) Thread Pool'da Yürütülür ---
        try:
            if self.pipeline is None:
                from audio_analyzer.services.pipeline_factory import get_shared_pipeline

                self.pipeline = get_shared_pipeline()

            # file_bytes verilmemişse storage'dan oku (Lazy get_bytes / get_path)
            if not file_bytes:
                if hasattr(self.storage, "get_bytes"):
                    try:
                        file_bytes = await asyncio.to_thread(self.storage.get_bytes, record.storage_uri)
                    except Exception as ex:
                        logger.warning("Storage'dan RAM baytları okunamadı (%s): %s", record.storage_uri, ex)

                if not file_bytes:
                    try:
                        p = await asyncio.to_thread(self.storage.get_path, record.storage_uri)
                        if p and os.path.exists(p):
                            file_bytes = await asyncio.to_thread(_read_file, p)
                    except Exception as ex:
                        logger.warning("Dosya diski üzerinden RAM'e okunamadı: %s", ex)

            res = None
            if file_bytes and hasattr(self.pipeline, "process_bytes"):
                if num_speakers is not None:
                    res = await asyncio.to_thread(self.pipeline.process_bytes, file_bytes, num_speakers=num_speakers)
                else:
                    try:
                        res = await asyncio.to_thread(self.pipeline.process_bytes, file_bytes, num_speakers=None)
                    except TypeError:
                        res = await asyncio.to_thread(self.pipeline.process_bytes, file_bytes)
            else:
                local_path = await asyncio.to_thread(self.storage.get_path, record.storage_uri)
                if (not local_path or not os.path.exists(local_path)) and not file_bytes:
                    raise FileNotFoundError(f"Ses dosyası depolamada bulunamadı: {record.storage_uri}")
                temp_local_file = None
                try:
                    if (not local_path or not os.path.exists(local_path)) and file_bytes:
                        import tempfile
                        from pathlib import Path

                        suffix = Path(record.file_name or "audio.wav").suffix or ".wav"
                        temp_name = f"job_tmp_{uuid.uuid4().hex}{suffix}"
                        temp_path = os.path.join(tempfile.gettempdir(), temp_name)
                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)
                        local_path = temp_path
                        temp_local_file = temp_path

                    if num_speakers is not None:
                        res = await asyncio.to_thread(
                            self.pipeline.process, local_path, num_speakers=num_speakers
                        )
                    else:
                        try:
                            res = await asyncio.to_thread(
                                self.pipeline.process, local_path, num_speakers=None
                            )
                        except TypeError:
                            res = await asyncio.to_thread(
                                self.pipeline.process, local_path
                            )
                finally:
                    if temp_local_file and os.path.exists(temp_local_file):
                        try:
                            os.remove(temp_local_file)
                        except Exception as clean_err:
                            logger.warning("Geçici dosya silinemedi: %s", clean_err)

            utterances, language, overlap_summary = res[0], res[1], res[2]

            webhook_payload_str = None
            if record.callback_url:
                payload_dict = {
                    "job_id": str(record_id),
                    "file_name": record.file_name,
                    "status": "COMPLETED",
                    "language": language,
                    "overlap_summary": (
                        overlap_summary.model_dump()
                        if overlap_summary and hasattr(overlap_summary, "model_dump")
                        else (overlap_summary.dict() if overlap_summary and hasattr(overlap_summary, "dict") else None)
                    ),
                    "utterances": [
                        {
                            "speaker_id": u.speaker_id,
                            "start_time": u.start_time,
                            "end_time": u.end_time,
                            "text": u.text,
                        }
                        for u in utterances
                    ],
                }
                webhook_payload_str = json.dumps(payload_dict)

            # Atomik completion ve fencing token kontrolü (Item 2 & Item 3)
            success = await self.repo.complete_job(
                record_id=record_id,
                claim_token=claim_token,
                utterances=utterances,
                language=language,
                overlap_summary=overlap_summary,
                webhook_payload=webhook_payload_str,
            )

            if not success:
                logger.info("İş başka worker'a geçti veya fencing token eşleşmedi (job_id=%s), sonuç atılıyor.", record_id)
                return "SKIPPED", 0, None

            JOB_STATUS_COUNTER.labels(status="COMPLETED").inc()
            logger.info("Ses analizi başarıyla tamamlandı ve atomik kaydedildi (job_id=%s, dil=%s).", record_id, language)
            completed_success = True
            return "COMPLETED", record.attempts, None

        except Exception as ex:
            logger.error("Job execution failed for job_id=%s: %s", record_id, ex, exc_info=True)
            
            # Eğer tamamlama başarılı olduysa veya veritabanında zaten COMPLETED ise, FAILED veya PENDING'e çekme! (Item 3)
            current_rec = await self.repo.get_record_by_id(record_id)
            if completed_success or (current_rec and current_rec.status == JobStatus.COMPLETED):
                logger.warning("İş %s zaten COMPLETED durumunda, hata handler ile PENDING/FAILED'a çekilmiyor: %s", record_id, ex)
                return "COMPLETED", getattr(current_rec, "attempts", 0), None

            sanitized_msg = sanitize_error_message(ex)
            is_transient = is_transient_error(ex)

            attempts, is_final_failed = await self.repo.handle_job_failure(
                record_id,
                error_message=sanitized_msg,
                is_transient=is_transient,
                max_attempts=max_attempts,
                claim_token=claim_token,
            )

            JOB_STATUS_COUNTER.labels(status="FAILED" if is_final_failed else "PENDING").inc()

            if is_final_failed and record.callback_url:
                payload = {
                    "job_id": str(record_id),
                    "file_name": record.file_name,
                    "status": "FAILED",
                    "attempts": attempts,
                    "error_message": sanitized_msg,
                }
                await self.repo.create_webhook_delivery(
                    job_id=record_id, url=record.callback_url, payload=json.dumps(payload)
                )

            result_status = "FAILED" if is_final_failed else "RETRY"
            return result_status, attempts, sanitized_msg

        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task


