import logging
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# Windows SYMLINK ve SpeechBrain önbellek uyarılarını bastır
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_analyzer.adapters.repository.models import Base
from audio_analyzer.api import dependencies, metrics
from audio_analyzer.api.routers import jobs

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("speechbrain.utils.fetching").setLevel(logging.ERROR)
logging.getLogger("speechbrain.utils.parameter_transfer").setLevel(logging.ERROR)

logger = logging.getLogger("audio_analyzer.api.main")
def get_cors_config(allowed_origins_raw: str | None = None) -> tuple[list[str], bool]:
    if allowed_origins_raw is None:
        allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]
    if not origins:
        return [], False
    if "*" in origins:
        return origins, False
    return origins, True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Uygulama Yaşam Döngüsü (Lifespan).
    Üretim (PostgreSQL) ortamlarında tek şema kaynağı Alembic'tir.
    `create_all` yalnızca yerel SQLite dev/test modunda tabloları otomatik oluşturur.
    """
    from audio_analyzer.config import get_settings
    settings = get_settings()
    app_env = settings.app_env
    api_key = settings.api_key.strip()
    webhook_secret = settings.webhook_secret.strip()
    if app_env != "development":
        if not api_key:
            raise ValueError("Üretim ortamında (APP_ENV!=development) API_KEY tanımlanması zorunludur!")
        if not webhook_secret:
            raise ValueError("Üretim ortamında (APP_ENV!=development) WEBHOOK_SECRET tanımlanması zorunludur!")

    from audio_analyzer.adapters.storage.storage_factory import (
        assert_storage_shared_across_processes,
    )

    assert_storage_shared_across_processes()

    allow_fallback = os.getenv("ALLOW_SQLITE_FALLBACK", "false").lower() == "true"
    db_is_sqlite = "sqlite" in str(dependencies.engine.url)

    try:
        from sqlalchemy import text

        async with dependencies.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        if db_is_sqlite:
            async with dependencies.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
    except Exception as db_err:
        if allow_fallback and not db_is_sqlite:
            logger.warning("PostgreSQL bağlantısı başarısız, ALLOW_SQLITE_FALLBACK aktif. SQLite'a geçiliyor... Hata: %s", db_err)
            fallback_url = "sqlite:///storage/dev_database.db"
            dependencies.engine = dependencies.create_async_db_engine(fallback_url)
            dependencies.AsyncSessionLocal.configure(bind=dependencies.engine)
            dependencies.SessionLocal = dependencies.AsyncSessionLocal
            async with dependencies.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        elif db_is_sqlite:
            logger.warning("SQLite tablo oluşturma uyarısı: %s", db_err)
        else:
            logger.error("PostgreSQL bağlantı hatası: %s", db_err)
            if app_env == "production":
                raise RuntimeError("Production ortamında veritabanı bağlantısı kurulamadı ve fallback kapalı!")

    use_redis_stream = os.getenv("USE_REDIS_STREAM", "false").lower() == "true"
    if not use_redis_stream:
        try:
            from audio_analyzer.services.pipeline_factory import get_shared_pipeline

            get_shared_pipeline()
        except Exception as e:
            logger.warning("Model ön yükleme uyarısı: %s", e)

    yield


app = FastAPI(
    title="Ses Analizi Platformu (STT + Speaker Diarization)",
    description="Gelişmiş Ses Analizi ve Konuşmacı Ayrıştırma Platformu Web Arayüzü",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (Kurumsal Üçüncü Parti İstemci Desteği)
allowed_origins, allow_credentials = get_cors_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def content_length_limit_middleware(request: Request, call_next):
    if request.url.path.endswith("/analyze") and request.method == "POST":
        content_length = request.headers.get("Content-Length")
        max_size = 100 * 1024 * 1024  # 100 MB
        if content_length and content_length.isdigit() and int(content_length) > max_size:
            return Response(
                content='{"detail":"Dosya boyutu çok büyük (Content-Length 100 MB üstünde)."}',
                status_code=413,
                media_type="application/json",
            )
    return await call_next(request)

# APIRouter Kaydı
app.include_router(jobs.router)
app.include_router(metrics.router)

# Statik Dosya Hizmeti
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
def get_web_ui():
    """
    Sürükle-Bırak Kolay Ses Analiz Web Arayüzü (Statik Frontend Dosyasından Sunulur).
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web arayüzü dosyası bulunamadı")
    return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Tarayıcıların otomatik favicon isteğine 204 No Content dönerek 404 loglarını engeller.
    """
    return Response(status_code=204)


@app.get("/health")
async def health_check():
    """
    Sistem Sağlık ve Kullanılabilirlik (Liveness Probe) Kontrolü.
    """
    db_status = "OK"
    try:
        from sqlalchemy import text

        async with dependencies.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("Health check database connection error: %s", e, exc_info=True)
        db_status = "ERROR"

    is_healthy = db_status == "OK"
    return Response(
        content=f'{{"status": "{ "HEALTHY" if is_healthy else "UNHEALTHY" }", "database": "{db_status}"}}',
        media_type="application/json",
        status_code=200 if is_healthy else 503,
    )


@app.get("/health/ready")
async def readiness_check():
    """
    Kubernetes / Docker Readiness Probe Kontrolü.
    Veritabanı (DB), Redis kuyruğu ve AI Model hazırlık durumunu doğrular.
    """
    import json

    db_ok = False
    redis_ok = False
    model_ok = False
    use_redis_stream = os.getenv("USE_REDIS_STREAM", "false").lower() == "true"

    # 1. DB Kontrolü
    try:
        from sqlalchemy import text

        async with dependencies.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as ex:
        logger.error("Readiness DB kontrol hatası: %s", ex)

    # 2. Redis & Worker Kontrolü
    worker_ok = True
    try:
        from audio_analyzer.config import get_settings

        settings = get_settings()
        redis_url = settings.redis_url
        import redis.asyncio as aioredis

        from audio_analyzer.adapters.messaging.redis_stream_adapter import (
            RedisStreamAdapter,
        )

        pool = RedisStreamAdapter.get_pool(redis_url)
        client = aioredis.Redis(connection_pool=pool)
        pong = await client.ping()
        redis_ok = bool(pong)

        if use_redis_stream:
            worker_ok = False
            # a) Worker heartbeat kontrolü
            hb = await client.get("audio_analyzer:worker_heartbeat")
            if hb:
                worker_ok = True
            else:
                # b) Consumer group aktiflik kontrolü (genişletilmiş tolerans: 5 dakika)
                try:
                    stream_key = settings.redis_stream_name
                    group_name = settings.redis_stream_group
                    consumers = await client.xinfo_consumers(stream_key, group_name)
                    if consumers:
                        idle_limit = settings.worker_consumer_idle_threshold_ms
                        for c in consumers:
                            idle = c.get("idle", 999999) if isinstance(c, dict) else 999999
                            if idle < idle_limit:
                                worker_ok = True
                                break
                except Exception:
                    pass
    except Exception as ex:
        logger.warning("Readiness Redis/Worker kontrol uyarısı: %s", ex)
        worker_ok = False
        redis_ok = False

    # 3. Model Yüklü Kontrolü (Senkron yükleme yapmadan kontrol eder)
    try:
        from audio_analyzer.services import pipeline_factory

        if pipeline_factory._cached_pipeline is not None:
            pipe = pipeline_factory._cached_pipeline
            model_ok = pipe.stt_engine is not None and getattr(pipe.stt_engine, "_model", None) is not None
    except Exception as ex:
        logger.warning("Readiness Model kontrol uyarısı: %s", ex)

    # Readiness probe Redis'i kontrol eder ama worker'ın meşguliyetinden bağımsızdır (meşgulken probe düşmez)
    is_ready = db_ok and (redis_ok if use_redis_stream else True)
    status_code = 200 if is_ready else 503

    return Response(
        content=json.dumps(
            {
                "status": "READY" if is_ready else "NOT_READY",
                "database": "OK" if db_ok else "ERROR",
                "redis": "OK" if redis_ok else ("DISABLED" if not use_redis_stream else "ERROR"),
                "worker": "OK" if (worker_ok or not use_redis_stream) else "DOWN",
                "use_redis_stream": use_redis_stream,
                "model_loaded": "OK" if model_ok else "PENDING",
            }
        ),
        media_type="application/json",
        status_code=status_code,
    )


def start():
    """Uvicorn sunucusu üzerinden FastAPI API'sini başlatır."""
    import uvicorn

    reload_env = os.getenv("RELOAD", "false").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run("audio_analyzer.api.main:app", host=host, port=port, reload=reload_env)


if __name__ == "__main__":
    start()

