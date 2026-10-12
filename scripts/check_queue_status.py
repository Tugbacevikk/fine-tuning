import asyncio
import os
import uuid
from audio_analyzer.adapters.repository.unit_of_work import SqlAlchemyUnitOfWork
from audio_analyzer.api.dependencies import AsyncSessionLocal
from audio_analyzer.adapters.messaging.redis_stream_adapter import RedisStreamAdapter
from audio_analyzer.config import get_settings
from sqlalchemy import text

async def main():
    settings = get_settings()
    print("=== CONFIG CHECK ===")
    print("settings.use_redis_stream:", settings.use_redis_stream)
    print("settings.redis_url:", settings.redis_url)
    print("settings.storage_type:", settings.storage_type)
    print("os.getenv('USE_REDIS_STREAM'):", os.getenv("USE_REDIS_STREAM"))
    print("os.getenv('REDIS_URL'):", os.getenv("REDIS_URL"))

    print("\n=== REDIS STREAM CHECK ===")
    adapter = RedisStreamAdapter()
    client = adapter.get_client()
    try:
        stream_len = await client.xlen(adapter.stream_key)
        print(f"Stream: '{adapter.stream_key}' length: {stream_len}")
        groups = await client.xinfo_groups(adapter.stream_key)
        print(f"Consumer groups on '{adapter.stream_key}': {groups}")
        consumers = await client.xinfo_consumers(adapter.stream_key, adapter.group_name)
        print(f"Consumers on '{adapter.group_name}': {consumers}")
        pending = await client.xpending(adapter.stream_key, adapter.group_name)
        print(f"Pending messages: {pending}")
    except Exception as e:
        print("Redis error:", e)

    print("\n=== DATABASE CHECK ===")
    uow = SqlAlchemyUnitOfWork(session_factory=AsyncSessionLocal)
    async with uow:
        res = await uow.session.execute(text("SELECT status, count(*) FROM audio_records GROUP BY status"))
        print("Status counts:", res.fetchall())

        proc = await uow.session.execute(text("SELECT id, file_name, status, attempts, error_message, created_at, updated_at FROM audio_records WHERE status = 'PROCESSING'"))
        print("Stuck PROCESSING records:", proc.fetchall())

        # Check records that are PENDING
        p_res = await uow.session.execute(text("SELECT id, file_name, status, attempts, created_at FROM audio_records WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT 10"))
        print("First 10 PENDING records:")
        for row in p_res.fetchall():
            print(row)

if __name__ == "__main__":
    asyncio.run(main())
