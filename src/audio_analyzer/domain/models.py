import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _detect_compute_type() -> str:
    if _detect_device() == "cuda":
        return "float16"
    return "int8"


class JobStatus(str, Enum):
    """Analiz görevinin yaşam döngüsü durumları."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DeviceConfig(BaseModel):
    """
    Dinamik donanım ve hassasiyet (Quantization) konfigürasyonu.
    GPU varsa CUDA + float16, yoksa CPU + int8 modunu seçer.
    """

    device: str = Field(default_factory=_detect_device)
    compute_type: str = Field(default_factory=_detect_compute_type)
    device_index: int = 0  # Çoklu GPU kartı seçimi için (0, 1, 2...)


class WordSegment(BaseModel):
    """Speech-to-Text motorundan çıkan kelime seviyesinde zaman damgalı metin birimi."""

    word: str
    start_time: float
    end_time: float
    probability: float = 1.0

    @property
    def midpoint(self) -> float:
        """Kelimenin zaman eksenindeki orta noktası."""
        return (self.start_time + self.end_time) / 2.0

    @property
    def duration(self) -> float:
        """Kelimenin telaffuz süresi (saniye)."""
        return self.end_time - self.start_time


class DiarizationSegment(BaseModel):
    """Speaker Diarization motorundan çıkan konuşmacı zaman aralığı."""

    speaker_id: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Konuşma aralığının süresi (saniye)."""
        return self.end_time - self.start_time

    def contains_timestamp(self, ts: float) -> bool:
        """Verilen zaman damgasının bu konuşma aralığı içinde olup olmadığını kontrol eder."""
        return self.start_time <= ts <= self.end_time


class TranscriptUtterance(BaseModel):
    """Final çıktı: Konuşmacı ile eşleştirilmiş cümle/paragraf bloğu."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OverlapSegment(BaseModel):
    """İki veya daha fazla konuşmacının aynı anda konuştuğu zaman aralığı."""

    speakers: list[str]
    start_time: float
    end_time: float
    duration: float


class UntranscribedSpeechSegment(BaseModel):
    """Whisper tarafindan transkript edilmemis konusma araligi."""

    start_time: float
    end_time: float
    duration: float
    speaker_id: str | None = None


class OverlapSummary(BaseModel):
    """Konuşma çakışması ve kalite kontrol metrik özeti."""

    supported: bool = False
    total_overlap_seconds: float = 0.0
    overlap_percentage: float = 0.0
    interrupt_count: int = 0
    overlaps: list[OverlapSegment] = Field(default_factory=list)
    untranscribed_speech_seconds: float = 0.0
    untranscribed_speech_count: int = 0
    untranscribed_speech: list[UntranscribedSpeechSegment] = Field(default_factory=list)


class AudioRecord(BaseModel):
    """Ses kaydı domain varlığı."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    external_id: str | None = None
    storage_uri: str
    file_name: str
    duration_seconds: float | None = None
    sample_rate: int = 16000
    channels: int = 1
    language: str | None = None
    status: JobStatus = JobStatus.PENDING
    error_message: str | None = None
    callback_url: str | None = None
    webhook_status: str | None = None
    attempts: int = 0
    processing_started_at: datetime | None = None
    last_error_at: datetime | None = None
    overlap_summary: OverlapSummary | None = None
    untranscribed_speech: list[UntranscribedSpeechSegment] = Field(default_factory=list)
    num_speakers: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    utterances: list[TranscriptUtterance] = Field(default_factory=list)


