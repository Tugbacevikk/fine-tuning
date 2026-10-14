import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

DEFAULT_STREAM_KEY = "audio_analysis_stream"
DEFAULT_GROUP_NAME = "audio_workers_group"


class RedisStreamAdapter:
    """
    High-throughput non-blocking Redis Streams (XADD / XREADGROUP / XACK / XAUTOCLAIM) Adaptörü.
    Saniyede 5.000+ eşzamanlı isteği 100+ worker arasında key contention olmadan sıfır çakışmayla dağıtır.
    """

    _pool: aioredis.ConnectionPool | None = None

    def __init__(
        self,
        redis_url: str | None = None,
        stream_key: str | None = None,
        group_name: str | None = None,
        redis_client: aioredis.Redis | None = None,
    ):
        from audio_analyzer.config import get_settings

        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.stream_key = stream_key or settings.redis_stream_name or DEFAULT_STREAM_KEY
        self.group_name = group_name or settings.redis_stream_group or DEFAULT_GROUP_NAME
        self._custom_client = redis_client

    @classmethod
    def get_pool(cls, redis_url: str) -> aioredis.ConnectionPool:
        """Yüksek eşzamanlılık (4k-5k req/sec) için paylaşımlı Redis bağlantı havuzu."""
        if cls._pool is None:
            from audio_analyzer.config import get_settings

            max_conn = get_settings().redis_max_connections
            cls._pool = aioredis.ConnectionPool.from_url(
                redis_url,
                max_connections=max_conn,
                decode_responses=True,
            )
        return cls._pool

    def get_client(self) -> aioredis.Redis:
        """Havuz üzerinden asenkron Redis istemcisi sağlar."""
        if self._custom_client is not None:
            return self._custom_client
        pool = self.get_pool(self.redis_url or "redis://localhost:6379/0")
        return aioredis.Redis(connection_pool=pool)

    async def publish_job(
        self,
        job_id: str,
        file_name: str,
        callback_url: str | None = None,
        num_speakers: int | None = None,
        max_len: int = 100000,
    ) -> str:
        """
        Saniyede 5.000+ isteği Redis Stream (XADD) akışına ekler.
        Non-blocking asenkron I/O kullanır.
        """
        client = self.get_client()
        payload: dict[str, Any] = {
            "job_id": str(job_id),
            "file_name": str(file_name),
            "callback_url": str(callback_url) if callback_url else "",
            "num_speakers": str(num_speakers) if num_speakers is not None else "",
        }
        msg_id = await client.xadd(
            name=self.stream_key,
            fields=payload,
            maxlen=max_len,
            approximate=True,
        )
        logger.debug("Redis Stream XADD msg_id=%s published for job_id=%s", msg_id, job_id)
        return str(msg_id)

    async def create_consumer_group(self, start_id: str = "0-0") -> bool:
        """
        Tüketici Grubu (Consumer Group) oluşturur (`XGROUP CREATE`).
        Grup daha önceden var ise BUSYGROUP uyarısını sessizce yutar.
        """
        client = self.get_client()
        try:
            await client.xgroup_create(
                name=self.stream_key,
                groupname=self.group_name,
                id=start_id,
                mkstream=True,
            )
            logger.info(
                "Redis Stream Consumer Group '%s' created for stream '%s'",
                self.group_name,
                self.stream_key,
            )
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Redis Stream Consumer Group '%s' already exists.", self.group_name)
                return True
            logger.error("Error creating Redis Stream group: %s", e)
            raise e

    async def consume_messages(
        self,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[str, dict[str, str]]]:
        """
        Tüketici Grubu (Consumer Group) üzerinden okunmamış mesajları çeker (`XREADGROUP`).
        100+ Worker düğümü aynı grubun parçası olarak sıfır çakışmayla mesajları paylaşır.
        Dönen format: [(msg_id, fields_dict), ...]
        """
        client = self.get_client()
        response: Any = await client.xreadgroup(
            groupname=self.group_name,
            consumername=consumer_name,
            streams={self.stream_key: ">"},
            count=count,
            block=block_ms,
        )

        messages: list[tuple[str, dict[str, str]]] = []
        if response and isinstance(response, list):
            for stream_item in response:
                if isinstance(stream_item, (tuple, list)) and len(stream_item) >= 2:
                    stream_messages = stream_item[1]
                    if isinstance(stream_messages, list):
                        for item in stream_messages:
                            if isinstance(item, (tuple, list)) and len(item) >= 2:
                                msg_id = str(item[0])
                                raw_fields = item[1]
                                fields_dict = dict(raw_fields) if isinstance(raw_fields, dict) else {}
                                messages.append((msg_id, fields_dict))
        return messages

    async def ack_message(self, message_id: str) -> int:
        """
        İşlenmiş mesajı onaylar (`XACK`).
        Mesaj Tüketici Grubunun işlenmeyi bekleyenler listesinden (PEL - Pending Entries List) düşer.
        """
        client = self.get_client()
        return await client.xack(self.stream_key, self.group_name, message_id)

    async def claim_pending_messages(
        self,
        consumer_name: str,
        min_idle_time_ms: int = 60000,
        count: int = 10,
    ) -> list[tuple[str, dict[str, str]]]:
        """
        Çöken worker'ların yarım kalan mesajlarını (Orphan/Pending entries) devralır (`XAUTOCLAIM`).
        `min_idle_time_ms` süresince yanıt alınamayan mesajlar aktif worker'a yeniden atanır.
        """
        client = self.get_client()
        try:
            res = await client.xautoclaim(
                name=self.stream_key,
                groupname=self.group_name,
                consumername=consumer_name,
                min_idle_time=min_idle_time_ms,
                start_id="0-0",
                count=count,
            )
            claimed_messages = []
            if res and len(res) >= 2:
                for msg in res[1]:
                    if isinstance(msg, (tuple, list)) and len(msg) >= 2:
                        msg_id, fields = msg[0], msg[1]
                        if fields:
                            claimed_messages.append((str(msg_id), dict(fields)))
            return claimed_messages
        except Exception as e:
            logger.warning("xautoclaim note: %s", e)
            return []

    async def claim_message_heartbeat(self, consumer_name: str, message_id: str) -> bool:
        """
        Uzun süren işlerde mesajın idle süresini sıfırlayarak (heartbeat)
        başka bir worker'ın XAUTOCLAIM ile devralmasını engeller (`XCLAIM min_idle=0`).
        """
        client = self.get_client()
        try:
            res = await client.xclaim(
                name=self.stream_key,
                groupname=self.group_name,
                consumername=consumer_name,
                min_idle_time=0,
                message_ids=[message_id],
            )
            return bool(res)
        except Exception as e:
            logger.warning("xclaim heartbeat note for msg_id=%s: %s", message_id, e)
            return False

    async def publish_to_dlq(
        self,
        job_id: str,
        error_message: str,
        attempts: int = 1,
        dlq_key: str = "audio_analysis_dlq",
    ) -> str:
        """
        Maksimum deneme sayısı aşılan veya kalıcı hata alan işleri Dead-Letter Stream (DLQ) akışına yazar (`XADD audio_analysis_dlq`).
        """
        client = self.get_client()
        payload = {
            "job_id": str(job_id),
            "error_message": str(error_message),
            "attempts": str(attempts),
        }
        msg_id = await client.xadd(
            name=dlq_key,
            fields=payload,
            maxlen=100000,
            approximate=True,
        )
        logger.warning(
            "Job %s (attempts=%d) Dead-Letter Stream (%s, msg_id=%s) akışına aktarıldı. Hata: %s",
            job_id,
            attempts,
            dlq_key,
            msg_id,
            error_message,
        )
        return str(msg_id)

    async def get_consumer_group_lag(self) -> int | None:
        """
        Redis Consumer Group'un tüketilmemiş mesaj sayısını (lag) döner.
        Grup bulunamazsa veya lag desteği yoksa None döner.
        """
        client = self.get_client()
        try:
            groups = await client.xinfo_groups(self.stream_key)
            if groups:
                for g in groups:
                    g_name = g.get("name") if isinstance(g, dict) else (g[1] if isinstance(g, (list, tuple)) else None)
                    if g_name == self.group_name or (isinstance(g_name, bytes) and g_name.decode() == self.group_name):
                        lag = g.get("lag") if isinstance(g, dict) else None
                        if lag is not None:
                            return int(lag)
            return None
        except Exception as e:
            logger.debug("xinfo_groups lag check note: %s", e)
            return None

    async def get_pending_delivery_count(self, message_id: str) -> int:
        """
        XPENDING sorgusu ile mesajın kaç kez teslim edildiğini (delivery count) döner.
        """
        client = self.get_client()
        try:
            res = await client.xpending_range(
                name=self.stream_key,
                groupname=self.group_name,
                min=message_id,
                max=message_id,
                count=1,
            )
            if res and len(res) > 0:
                item = res[0]
                if isinstance(item, dict):
                    return int(item.get("times_delivered", 1))
                elif isinstance(item, (tuple, list)) and len(item) >= 4:
                    return int(item[3])
            return 1
        except Exception as e:
            logger.debug("xpending_range delivery count note: %s", e)
            return 1
