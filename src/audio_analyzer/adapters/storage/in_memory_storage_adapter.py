import threading
import uuid
from collections import OrderedDict
from typing import ClassVar

from audio_analyzer.domain.interfaces import IAudioStorage


class InMemoryStorageAdapter(IAudioStorage):
    """
    RAM-Tabanlı (In-Memory Stream Buffer) Nesne Depolama Adaptörü.
    YALNIZCA tek süreçli (single-process) test, CLI ve BackgroundTasks ortamlarında kullanılır.
    Süreçler arası (web & worker) paylaşılamaz.
    """

    _storage: ClassVar[OrderedDict] = OrderedDict()
    _shared_buffer: ClassVar[OrderedDict] = _storage
    _total_bytes: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, max_items: int = 100000, max_bytes: int | None = None):
        self.max_items = max_items
        if max_bytes is None:
            from audio_analyzer.config import get_settings
            max_bytes = get_settings().memory_storage_max_bytes
        self.max_bytes = max_bytes

    def save(self, file_bytes: bytes, file_name: str) -> str:
        unique_prefix = uuid.uuid4().hex[:8]
        safe_name = file_name.replace("/", "_").replace("\\", "_")
        storage_uri = f"ram://{unique_prefix}_{safe_name}"

        with InMemoryStorageAdapter._lock:
            # Yeni kaydı ekle
            InMemoryStorageAdapter._storage[storage_uri] = file_bytes
            InMemoryStorageAdapter._total_bytes += len(file_bytes)

            # Sınır aşımında en eski kayıtları çıkar
            while (
                len(InMemoryStorageAdapter._storage) > self.max_items
                or InMemoryStorageAdapter._total_bytes > self.max_bytes
            ) and InMemoryStorageAdapter._storage:
                pop_key, pop_bytes = InMemoryStorageAdapter._storage.popitem(last=False)
                InMemoryStorageAdapter._total_bytes -= len(pop_bytes)
                InMemoryStorageAdapter._total_bytes = max(InMemoryStorageAdapter._total_bytes, 0)

        return storage_uri

    def get_bytes(self, storage_uri: str) -> bytes:
        with InMemoryStorageAdapter._lock:
            if storage_uri in InMemoryStorageAdapter._storage:
                return InMemoryStorageAdapter._storage[storage_uri]
        raise FileNotFoundError(f"RAM depolamasında ses URI bulunamadı: {storage_uri}")

    def get_path(self, storage_uri: str) -> str:
        with InMemoryStorageAdapter._lock:
            if storage_uri in InMemoryStorageAdapter._storage:
                return storage_uri
        raise FileNotFoundError(f"RAM depolamasında ses URI bulunamadı: {storage_uri}")

    def delete(self, storage_uri: str) -> bool:
        with InMemoryStorageAdapter._lock:
            if storage_uri in InMemoryStorageAdapter._storage:
                pop_bytes = InMemoryStorageAdapter._storage.pop(storage_uri)
                InMemoryStorageAdapter._total_bytes -= len(pop_bytes)
                InMemoryStorageAdapter._total_bytes = max(InMemoryStorageAdapter._total_bytes, 0)
                return True
        return False
