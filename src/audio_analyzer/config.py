import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)


def _opt_int(name: str) -> int | None:
    val = os.getenv(name)
    if not val:
        return None
    try:
        parsed = int(val)
        return parsed if parsed > 0 else None
    except ValueError:
        return None


def _get_whisper_beam_size() -> int:
    env_val = os.getenv("WHISPER_BEAM_SIZE")
    if env_val:
        return int(env_val)
    profile = os.getenv("PIPELINE_PROFILE", "full").lower()
    return 1 if profile == "feedback" else 5


@dataclass
class Settings:
    """
    Sistem genelindeki tüm konfigürasyon ve ortam değişkenlerinin
    tek bir noktadan yönetildiği ve doğrulandığı merkezi Settings sınıfı.
    """

    # 1. API & Güvenlik
    app_env: str = field(
        default_factory=lambda: os.getenv("APP_ENV", os.getenv("ENV", "production")).lower()
    )
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    webhook_secret: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", ""))

    # 2. Veritabanı & Depolama
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://postgres@localhost:5432/audio_db"
        )
    )
    allow_sqlite_fallback: bool = field(
        default_factory=lambda: os.getenv("ALLOW_SQLITE_FALLBACK", "false").lower() == "true"
    )
    storage_type: str = field(default_factory=lambda: os.getenv("STORAGE_TYPE", "disk"))
    storage_dir: str = field(default_factory=lambda: os.getenv("STORAGE_DIR", "storage/raw"))
    memory_storage_max_bytes: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_STORAGE_MAX_BYTES", "536870912"))
    )
    audio_retention_hours: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_RETENTION_HOURS", "24"))
    )
    result_retention_days: int = field(
        default_factory=lambda: int(os.getenv("RESULT_RETENTION_DAYS", "30"))
    )

    # 3. Mesajlaşma & Kuyruk
    use_redis_stream: bool = field(
        default_factory=lambda: os.getenv("USE_REDIS_STREAM", "false").lower() == "true"
    )
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # 4. Yapay Zeka Model Yapılandırması
    model_dir: str = field(
        default_factory=lambda: os.getenv("MODEL_DIR", "storage/models")
    )
    whisper_model_size: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small"))
    whisper_beam_size: int = field(default_factory=_get_whisper_beam_size)
    whisper_batch_size: int = field(default_factory=lambda: int(os.getenv("WHISPER_BATCH_SIZE", "8")))
    worker_cpu_threads: int = field(default_factory=lambda: int(os.getenv("WORKER_CPU_THREADS", "4")))
    diarization_engine: str = field(default_factory=lambda: os.getenv("DIARIZATION_ENGINE", "pyannote").lower())
    diarization_step_sec: float = field(default_factory=lambda: float(os.getenv("DIARIZATION_STEP_SEC", "0.3")))
    diarization_threshold: float = field(default_factory=lambda: float(os.getenv("DIARIZATION_THRESHOLD", "0.55")))
    target_num_speakers: int | None = field(
        default_factory=lambda: _opt_int("TARGET_NUM_SPEAKERS") or _opt_int("NUM_SPEAKERS")
    )
    max_silence_threshold: float = field(default_factory=lambda: float(os.getenv("MAX_SILENCE_THRESHOLD", "1.5")))
    enable_denoiser: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DENOISER", "false").lower() == "true"
    )

    # Pipeline Yapılandırması
    pipeline_profile: str = field(default_factory=lambda: os.getenv("PIPELINE_PROFILE", "full").lower())
    pipeline_min_diarize_sec: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_MIN_DIARIZE_SEC", "2.0"))
    )
    pipeline_min_snr_db: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_MIN_SNR_DB", "15.0"))
    )
    vad_engine: str = field(default_factory=lambda: os.getenv("VAD_ENGINE", "faster_whisper").lower())
    run_pipeline_sequentially: bool = field(
        default_factory=lambda: os.getenv("RUN_PIPELINE_SEQUENTIALLY", "true").lower() == "true"
    )

    # 5. SemanticRefiner & Yerel LLM
    domain_mode: str | None = field(default_factory=lambda: os.getenv("DOMAIN_MODE"))
    use_llm: bool = field(default_factory=lambda: os.getenv("USE_LLM", "false").lower() == "true")
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))

    # 6. Sweeper & DLQ (Yeniden Deneme ve Garanti Hizmeti)
    sweeper_interval_sec: float = field(default_factory=lambda: float(os.getenv("SWEEPER_INTERVAL_SEC", "30.0")))
    pending_stale_sec: int = field(default_factory=lambda: int(os.getenv("PENDING_STALE_SEC", "300")))
    processing_stale_sec: int = field(default_factory=lambda: int(os.getenv("PROCESSING_STALE_SEC", "1800")))
    max_job_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_JOB_ATTEMPTS", "3")))

    # 7. S3 / Bulut Nesne Depolama (STORAGE_TYPE=s3 durumunda)
    s3_bucket_name: str = field(default_factory=lambda: os.getenv("S3_BUCKET_NAME", "ses-analizi-storage"))
    aws_access_key_id: str = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    aws_secret_access_key: str = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    s3_endpoint_url: str = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))


def get_settings() -> Settings:
    """Merkezi Settings nesnesinin dinamik erişim fonksiyonu."""
    return Settings()


def reset_settings():
    """Testler sırasında konfigürasyonu yeniden yüklemek için kullanılır."""
    pass
