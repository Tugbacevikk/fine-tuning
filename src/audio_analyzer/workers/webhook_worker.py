"""
Webhook Outbox Worker Servisi.
Görüşme analizi tamamlandığında veya nihai hataya düştüğünde veritabanı Outbox (webhook_deliveries)
tablosuna yazılan kayıtları harici sunucuları yormadan asenkron teslim eder.

Özellikler:
- HMAC SHA256 Güvenlik İmzası (WEBHOOK_SECRET ortam değişkeni ile)
- Kısa HTTP Timeout (5s)
- Host Bazlı Eşzamanlılık Sınırı (WEBHOOK_HOST_CONCURRENCY, varsayılan 5)
- Üstel Backoff ile Yeniden Deneme (Exponential Backoff)
- WEBHOOK_MAX_ATTEMPTS (varsayılan 5) aşılınca 'DEAD' durumuna geçiş

Kullanım:
    python -m audio_analyzer.workers.webhook_worker
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audio_analyzer.adapters.repository.session import get_uow_with_engine, init_engine

load_dotenv()
logger = logging.getLogger("webhook_worker")

# Per-host semaphores for rate limiting per domain
_host_semaphores: dict[str, asyncio.Semaphore] = {}


def get_host_semaphore(url: str, max_concurrent: int) -> asyncio.Semaphore:
    netloc = urlparse(url).netloc or "default"
    if netloc not in _host_semaphores:
        _host_semaphores[netloc] = asyncio.Semaphore(max_concurrent)
    return _host_semaphores[netloc]


def sign_payload(payload_str: str, secret: str, timestamp_str: str | None = None) -> str:
    """Payload ve timestamp verisini HMAC SHA256 ile imzalar."""
    from audio_analyzer.services.webhook_service import WebhookService

    svc = WebhookService(secret_key=secret)
    return svc.generate_signature(timestamp_str, payload_str)


async def deliver_single_webhook(delivery: dict, engine, max_attempts: int, timeout_sec: float, host_concurrency: int, secret: str | None) -> bool:
    """Tek bir webhook teslimatını gerçekleştirir."""
    delivery_id = delivery["id"]
    url = delivery["url"]
    payload_str = delivery["payload"]
    attempts = delivery["attempts"] + 1

    from audio_analyzer.utils.ssrf_validator import validate_callback_url
    if not validate_callback_url(url):
        error_msg = f"SSRF Protection: Invalid or unsafe webhook URL '{url}'"
        now = datetime.now(timezone.utc)
        async with get_uow_with_engine(engine) as uow:
            await uow.repository.update_webhook_delivery_status(
                delivery_id=delivery_id,
                status="DEAD",
                attempts=attempts,
                error_message=error_msg,
            )
            logger.warning("Webhook delivery %s -> %s BLOCKED BY SSRF VALIDATOR", delivery_id, url)
            return False

    timestamp_str = str(int(time.time()))
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "AudioAnalyzer-Webhook/1.0",
        "X-Timestamp": timestamp_str,
    }
    if secret:
        headers["X-Signature"] = sign_payload(payload_str, secret, timestamp_str)

    sem = get_host_semaphore(url, host_concurrency)
    success = False
    error_msg = None

    async with sem:
        try:
            async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=False) as client:
                resp = await client.post(url, content=payload_str.encode("utf-8"), headers=headers)
                if resp.is_success:
                    success = True
                else:
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as ex:
            error_msg = f"{type(ex).__name__}: {str(ex)[:200]}"

    now = datetime.now(timezone.utc)
    async with get_uow_with_engine(engine) as uow:
        if success:
            await uow.repository.update_webhook_delivery_status(
                delivery_id=delivery_id,
                status="SUCCESS",
                attempts=attempts,
                error_message=None,
            )
            logger.info("Webhook delivery %s -> %s SUCCESS (attempts=%d)", delivery_id, url, attempts)
            return True
        else:
            is_dead = attempts >= max_attempts
            next_status = "DEAD" if is_dead else "PENDING"
            # Üstel backoff: 5, 10, 20, 40, 80 saniye...
            backoff_sec = (2 ** (attempts - 1)) * 5
            next_attempt_at = now + timedelta(seconds=backoff_sec)

            await uow.repository.update_webhook_delivery_status(
                delivery_id=delivery_id,
                status=next_status,
                attempts=attempts,
                next_attempt_at=next_attempt_at if not is_dead else None,
                error_message=error_msg,
            )
            if is_dead:
                logger.error(
                    "Webhook delivery %s -> %s DEAD (max_attempts=%d reached). Error: %s",
                    delivery_id,
                    url,
                    max_attempts,
                    error_msg,
                )
            else:
                logger.warning(
                    "Webhook delivery %s -> %s FAILED (attempt %d/%d). Next retry in %ds. Error: %s",
                    delivery_id,
                    url,
                    attempts,
                    max_attempts,
                    backoff_sec,
                    error_msg,
                )
            return False


async def run_webhook_worker_loop(poll_interval: float = 2.0):
    """Webhook Outbox döngüsü."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WebhookWorker) %(message)s")
    logger.info("Webhook Outbox Worker başlatıldı.")

    from audio_analyzer.config import get_settings

    settings = get_settings()
    max_attempts = int(os.getenv("WEBHOOK_MAX_ATTEMPTS", "5"))
    timeout_sec = float(os.getenv("WEBHOOK_TIMEOUT_SEC", "5.0"))
    host_concurrency = int(os.getenv("WEBHOOK_HOST_CONCURRENCY", "5"))
    secret = settings.webhook_secret

    engine = init_engine()

    stop_event = asyncio.Event()

    def _handle_stop(sig, frame):
        logger.info("Webhook Worker durdurma sinyali (SIGINT/SIGTERM) alındı.")
        stop_event.set()

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: stop_event.set())

    try:
        while not stop_event.is_set():
            try:
                async with get_uow_with_engine(engine) as uow:
                    due_deliveries = await uow.repository.get_due_webhook_deliveries(limit=50)

                if due_deliveries:
                    tasks = [
                        deliver_single_webhook(
                            delivery=d,
                            engine=engine,
                            max_attempts=max_attempts,
                            timeout_sec=timeout_sec,
                            host_concurrency=host_concurrency,
                            secret=secret,
                        )
                        for d in due_deliveries
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    await asyncio.sleep(poll_interval)
            except Exception as ex:
                logger.error("Webhook Worker döngü hatası: %s", ex, exc_info=True)
                await asyncio.sleep(poll_interval)
    finally:
        await engine.dispose()
        logger.info("Webhook Worker kapandı.")


if __name__ == "__main__":
    asyncio.run(run_webhook_worker_loop())
