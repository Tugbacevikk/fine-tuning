import asyncio
import logging
import mimetypes
import os
import secrets
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.adapters.storage.storage_factory import get_storage_adapter
from audio_analyzer.api.dependencies import get_repository, get_uow
from audio_analyzer.api.rate_limiter import rate_limiter
from audio_analyzer.config import get_settings
from audio_analyzer.domain.interfaces import ITranscriptRepository
from audio_analyzer.domain.models import OverlapSummary, TranscriptUtterance
from audio_analyzer.services.job_service import JobService
from audio_analyzer.utils.file_validator import is_valid_audio_content

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Jobs & Analysis"])

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB



async def verify_api_key(api_key: str | None = Depends(api_key_header)):
    """
    İsteğin X-API-Key başlığını doğrular. Ortam değişkeninde API_KEY tanımlıysa kontrol eder.
    Tanımlı değilse APP_ENV=development haricinde erişimi engeller.
    """
    from audio_analyzer.config import get_settings
    settings = get_settings()
    expected_api_key = settings.api_key.strip()
    app_env = settings.app_env

    if expected_api_key:
        if not api_key or not secrets.compare_digest(api_key, expected_api_key):
            raise HTTPException(
                status_code=401, detail="Geçersiz veya eksik API Anahtarı (X-API-Key header)."
            )
    elif app_env != "development":
        raise HTTPException(
            status_code=401, detail="API Anahtarı yapılandırılmamış (Üretim ortamında API_KEY zorunludur)."
        )
    return api_key


async def run_pipeline_background(
    job_id_str: str, file_name: str, file_bytes: bytes, num_speakers: int | None = None
):
    """
    Arka plan asenkron worker fonksiyonu (Non-blocking HTTP 202 mimarisi).
    Redis Streams veya FastAPI BackgroundTasks için merkezi JobService.execute_job iş mantığını çağırır.
    """
    try:
        record_uuid = uuid.UUID(job_id_str)
        async with get_uow() as uow:
            from audio_analyzer.api.dependencies import AsyncSessionLocal
            storage = get_storage_adapter()
            job_service = JobService(
                storage=storage,
                repository=uow.repository,
                session_factory=AsyncSessionLocal,
            )
            await job_service.execute_job(record_uuid, file_bytes=file_bytes, num_speakers=num_speakers)
    except Exception as ex:
        logger.error("Background task execution error: %s", ex, exc_info=True)


# --- Schemas ---
class JobCreateResponse(BaseModel):
    job_id: str
    file_name: str
    status: str
    message: str


class UtteranceResponse(BaseModel):
    id: uuid.UUID | None = None
    speaker_id: str
    start_time: float
    end_time: float
    text: str


class JobStatusResponse(BaseModel):
    job_id: str
    file_name: str
    status: str
    language: str | None = None
    error_message: str | None = None
    overlap_summary: OverlapSummary | None = None
    utterances: list[UtteranceResponse] = []


class UtteranceUpdateRequest(BaseModel):
    speaker_id: str
    text: str


class UtteranceCreateRequest(BaseModel):
    speaker_id: str
    start_time: float
    end_time: float
    text: str

# --- Endpoints ---
@router.post("/analyze", response_model=JobCreateResponse, status_code=202)
async def upload_and_analyze_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    callback_url: str | None = Form(None),
    external_id: str | None = Form(None),
    num_speakers: int | None = Form(None, ge=1, le=10, description="Beklenen konuşmacı sayısı (1-10). 1 verilirse diarization baypas edilir."),
    speakers: int | None = Form(None, ge=1, le=10, description="Geriye dönük uyumluluk takma adı."),
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Ses dosyasını yükler, validasyondan geçirir, PENDING durumuyla kaydeder
    ve analizi ASENKRON başlatır (HTTP 202 Accepted).
    Redis tabanlı Dağıtık Rate-limiting ve external_id İdempotency uygulanır.
    'num_speakers=1' gönderilirse diarization bypass edilerek analiz süresi düşürülür.
    """
    target_speakers = num_speakers if num_speakers is not None else speakers

    await rate_limiter.check_rate_limit(request)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Ses dosyası adı boş olamaz.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen ses formatı '{ext}'. İzin verilen formatlar: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    file_bytes_buf = bytearray()
    chunk_size = 1024 * 1024  # 1 MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        file_bytes_buf.extend(chunk)
        if len(file_bytes_buf) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Dosya boyutu çok büyük ({len(file_bytes_buf) / (1024 * 1024):.1f} MB). Maksimum izin verilen limit: 100 MB.",
            )
    file_bytes = bytes(file_bytes_buf)

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Yüklenen ses dosyası boş (0 bayt) olamaz.")

    if callback_url:
        from audio_analyzer.utils.ssrf_validator import validate_callback_url
        is_valid_cb = await asyncio.to_thread(validate_callback_url, callback_url)
        if not is_valid_cb:
            raise HTTPException(
                status_code=400,
                detail="Geçersiz veya güvensiz callback_url (SSRF koruması: Özel/yerel IP adresleri ve güvensiz protokoller kabul edilmez).",
            )

    if not is_valid_audio_content(file_bytes, file.filename):
        raise HTTPException(
            status_code=400,
            detail="Yüklenen dosya geçerli ve bozulmamış bir ses dosyası içeriği (WAV, MP3, FLAC, M4A, OGG) taşımıyor.",
        )

    storage = get_storage_adapter()
    job_service = JobService(storage=storage, repository=repository)
    job_id = await job_service.create_job(
        file_name=file.filename,
        file_bytes=file_bytes,
        callback_url=callback_url,
        external_id=external_id,
        num_speakers=target_speakers,
    )

    use_redis_stream = os.getenv("USE_REDIS_STREAM", "false").lower() == "true" or os.getenv("USE_REDIS_QUEUE", "false").lower() == "true"

    if use_redis_stream:
        try:
            stream_adapter = RedisStreamAdapter()
            await stream_adapter.publish_job(
                job_id=str(job_id),
                file_name=file.filename,
                callback_url=callback_url,
                num_speakers=target_speakers,
            )

            logger.info("Görüşme görevi %s başarıyla Redis Stream (XADD) akışına fırlatıldı.", job_id)
        except Exception as e:
            logger.error("Redis Stream (XADD) yayını başarısız oldu (%s). Kayıt PENDING kaldı, Sweeper toparlayacak.", e)
            raise HTTPException(
                status_code=503,
                detail=f"Mesaj kuyruğu servisi geçici olarak yanıt vermiyor. İsteğiniz PENDING olarak kaydedildi (job_id: {job_id}).",
                headers={"Retry-After": "5", "X-Job-ID": str(job_id)},
            )
    else:
        env_mode = get_settings().app_env
        if env_mode in {"production", "prod"}:
            logger.warning(
                "UYARI: Production ortamında in-process BackgroundTasks kullanımı yüksek GPU/CPU yükünde HTTP sunucusunu kilitleyebilir. 'USE_REDIS_STREAM=true' yapılandırılması şiddetle önerilir (Job ID: %s).",
                job_id
            )
        background_tasks.add_task(run_pipeline_background, str(job_id), file.filename, file_bytes, target_speakers)


    return JobCreateResponse(
        job_id=str(job_id),
        file_name=file.filename,
        status="PENDING",
        message="Ses analizi görevi asenkron olarak kuyruğa alındı.",
    )


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(
    skip: int = Query(0, ge=0, description="Atlanacak kayıt sayısı"),
    limit: int = Query(20, ge=1, le=100, description="Getirilecek maksimum kayıt sayısı"),
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Geçmişte yüklenen tüm ses analizi görevlerini tarihe göre tersten sıralı (en yeni en üstte)
    ve sayfalamalı (Pagination: skip, limit) olarak getirir.
    """
    records = await repository.list_records(skip=skip, limit=limit)
    return [
        JobStatusResponse(
            job_id=str(r.id),
            file_name=r.file_name,
            status=r.status.value,
            language=r.language,
            error_message=r.error_message,
            utterances=[
                UtteranceResponse(
                    id=u.id,
                    speaker_id=u.speaker_id,
                    start_time=u.start_time,
                    end_time=u.end_time,
                    text=u.text,
                )
                for u in r.utterances
            ],
            overlap_summary=r.overlap_summary,
        )
        for r in records
    ]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Verilen job_id görevinin durumunu ve analiz sonuçlarını getirir.
    """
    try:
        record_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz UUID formatı.")

    record = await repository.get_record_by_id(record_uuid)

    if not record:
        raise HTTPException(status_code=404, detail="Ses analizi görevi bulunamadı.")

    utterances = [
        UtteranceResponse(
            id=u.id,
            speaker_id=u.speaker_id,
            start_time=u.start_time,
            end_time=u.end_time,
            text=u.text,
        )
        for u in record.utterances
    ]

    return JobStatusResponse(
        job_id=str(record.id),
        file_name=record.file_name,
        status=record.status.value,
        language=record.language,
        error_message=record.error_message,
        overlap_summary=record.overlap_summary,
        utterances=utterances,
    )


@router.get("/jobs/{job_id}/audio")
async def get_job_audio_file(
    job_id: str,
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Ses analizi görevine ait ham ses dosyasını tarayıcıda dinlenmek üzere sunar (Streaming / Audio Player).
    """
    try:
        record_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz UUID formatı.")

    record = await repository.get_record_by_id(record_uuid)

    if not record:
        raise HTTPException(status_code=404, detail="Ses analizi görevi bulunamadı.")

    storage = get_storage_adapter()

    # 1. Önce diski/yerel yolu dene (HTML5 <audio> Range requests & partial content streaming için FileResponse)
    try:
        local_path = storage.get_path(record.storage_uri)
        if local_path and os.path.exists(local_path):
            media_type, _ = mimetypes.guess_type(local_path)
            if not media_type:
                media_type = "audio/wav"
            return FileResponse(path=local_path, media_type=media_type, filename=record.file_name)
    except Exception as e:
        logger.warning("Storage get_path note: %s", e)

    # 2. Disk yolu bulunamazsa RAM bellek akışı olarak dön (RAM storage fallback)
    if hasattr(storage, "get_bytes"):
        try:
            audio_bytes = storage.get_bytes(record.storage_uri)
            if audio_bytes and len(audio_bytes) > 0:
                media_type, _ = mimetypes.guess_type(record.file_name)
                if not media_type:
                    media_type = "audio/wav"
                return Response(content=audio_bytes, media_type=media_type)
        except Exception as e:
            logger.warning("RAM storage get_bytes note: %s", e)

    raise HTTPException(status_code=404, detail="Ses dosyası depolamada veya diskte bulunamadı.")



@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Tüm ses analizi görevini, ilişkili veritabanı kayıtlarını ve fiziksel ses dosyasını siler.
    """
    try:
        record_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz UUID formatı.")

    record = await repository.get_record_by_id(record_uuid)
    if not record:
        raise HTTPException(status_code=404, detail="Ses analizi görevi bulunamadı.")

    # 1. Önce DB kaydını sil ve commit et
    success = await repository.delete_record(record_uuid)
    if not success:
        raise HTTPException(status_code=404, detail="Görevi veritabanından silme başarısız.")

    # 2. Sonra dosyayı best-effort sil
    storage = get_storage_adapter()
    try:
        storage.delete(record.storage_uri)
    except Exception as ex:
        logger.warning("File delete note: %s", ex)

    return {"status": "SUCCESS", "message": "Ses analizi kaydı ve dosyası başarıyla silindi."}


@router.put("/jobs/{job_id}/utterances/{utterance_id}")
async def update_utterance_speaker(
    job_id: str,
    utterance_id: uuid.UUID,
    req: UtteranceUpdateRequest,
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Kullanıcının Arayüzden (UI) manuel olarak konuşmacı etiketini veya metni değiştirmesini sağlar.
    """
    try:
        record_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz UUID formatı.")

    success = await repository.update_utterance_by_id(
        record_id=record_uuid,
        utterance_id=utterance_id,
        speaker_id=req.speaker_id,
        text=req.text,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Güncellenecek cümle bulunamadı.")

    return {"status": "SUCCESS", "message": "Konuşmacı etiketi başarıyla güncellendi."}


@router.delete("/jobs/{job_id}/utterances/{utterance_id}")
async def delete_utterance(
    job_id: str,
    utterance_id: uuid.UUID,
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Kullanıcının Arayüzden (UI) seçtiği konuşmacı kartını/kutusunu silmesini sağlar.
    """
    try:
        record_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz UUID formatı.")

    success = await repository.delete_utterance_by_id(record_uuid, utterance_id)

    if not success:
        raise HTTPException(status_code=404, detail="Silinecek cümle bulunamadı.")

    return {"status": "SUCCESS", "message": "Konuşmacı bloğu başarıyla silindi."}


@router.post("/jobs/{job_id}/utterances")
async def create_utterance(
    job_id: str,
    req: UtteranceCreateRequest,
    repository: ITranscriptRepository = Depends(get_repository),
    _api_key: str | None = Depends(verify_api_key),
):
    """
    Kullanıcının Arayüzden (UI) yeni bir konuşmacı kartı/kutusu eklemesini sağlar.
    """
    try:
        record_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz UUID formatı.")

    new_u = TranscriptUtterance(
        id=uuid.uuid4(),
        speaker_id=req.speaker_id,
        start_time=req.start_time,
        end_time=req.end_time,
        text=req.text,
    )
    success = await repository.add_utterance(record_id=record_uuid, utterance=new_u)
    if not success:
        raise HTTPException(status_code=404, detail="Ses görevi bulunamadı.")

    return {"status": "SUCCESS", "message": "Yeni konuşmacı bloğu başarıyla eklendi."}
