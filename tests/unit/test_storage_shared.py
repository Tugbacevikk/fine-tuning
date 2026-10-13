import concurrent.futures
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from audio_analyzer.adapters.storage.in_memory_storage_adapter import (
    InMemoryStorageAdapter,
)
from audio_analyzer.adapters.storage.local_disk_storage_adapter import (
    LocalDiskStorageAdapter,
)
from audio_analyzer.adapters.storage.s3_storage_adapter import S3StorageAdapter
from audio_analyzer.adapters.storage.storage_factory import (
    assert_storage_shared_across_processes,
    get_storage_adapter,
)
from audio_analyzer.domain.models import AudioRecord, JobStatus, OverlapSummary
from audio_analyzer.services.job_service import JobService


@pytest.mark.unit
def test_local_disk_storage_cross_instance_sharing(tmp_path):
    """API ve Worker gibi iki ayrı adaptör örneğinin aynı paylaşımlı disk dizininde sorunsuz çalıştığını doğrular."""
    base_dir = str(tmp_path / "shared_raw")

    adapter_api = LocalDiskStorageAdapter(base_dir=base_dir)
    adapter_worker = LocalDiskStorageAdapter(base_dir=base_dir)

    test_bytes = b"RIFF____WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    uri = adapter_api.save(test_bytes, "test_audio.wav")
    assert uri.startswith("disk://")

    fetched_bytes = adapter_worker.get_bytes(uri)
    assert fetched_bytes == test_bytes

    disk_path = adapter_worker.get_path(uri)
    assert os.path.exists(disk_path)

    deleted = adapter_worker.delete(uri)
    assert deleted is True

    with pytest.raises(FileNotFoundError):
        adapter_worker.get_bytes(uri)


@pytest.mark.unit
def test_local_disk_storage_path_traversal_prevention(tmp_path):
    """Path traversal ve geçersiz URI yönlendirme saldırılarının engellendiğini doğrular."""
    base_dir = tmp_path / "storage_raw"
    adapter = LocalDiskStorageAdapter(base_dir=str(base_dir))

    # 1. Dosya adı sızdırma girişimi (../../etc/passwd.wav)
    uri = adapter.save(b"test data", "../../etc/passwd.wav")
    assert ".." not in uri
    resolved_path = Path(adapter.get_path(uri)).resolve()
    assert resolved_path.parent == base_dir.resolve()

    # 2. Sahte URI denemeleri
    with pytest.raises(ValueError):
        adapter.get_bytes("disk://../secret.txt")

    with pytest.raises(ValueError):
        adapter.get_bytes("s3://my-bucket/audio.wav")


@pytest.mark.unit
def test_storage_factory_adapter_creation(monkeypatch):
    """STORAGE_TYPE env ayarına göre fabrika fonksiyonunun doğru adaptörü ürettiğini doğrular."""
    from audio_analyzer.config import reset_settings
    monkeypatch.setenv("STORAGE_TYPE", "disk")
    reset_settings()
    adapter = get_storage_adapter()
    assert isinstance(adapter, LocalDiskStorageAdapter)

    monkeypatch.setenv("STORAGE_TYPE", "memory")
    reset_settings()
    adapter = get_storage_adapter()
    assert isinstance(adapter, InMemoryStorageAdapter)

    monkeypatch.setenv("STORAGE_TYPE", "s3")
    reset_settings()
    adapter = get_storage_adapter()
    assert isinstance(adapter, S3StorageAdapter)
    reset_settings()


@pytest.mark.unit
def test_assert_storage_shared_across_processes_guard(monkeypatch):
    """USE_REDIS_STREAM=true iken STORAGE_TYPE=memory kullanımının engellendiğini doğrular."""
    from audio_analyzer.config import reset_settings
    monkeypatch.setenv("USE_REDIS_STREAM", "true")
    monkeypatch.setenv("STORAGE_TYPE", "memory")
    reset_settings()
    with pytest.raises(RuntimeError, match="RAM depolaması süreçler arası paylaşılmaz"):
        assert_storage_shared_across_processes()

    # Disk veya S3 ile akış modu sorunsuz geçmeli
    monkeypatch.setenv("STORAGE_TYPE", "disk")
    reset_settings()
    assert_storage_shared_across_processes()

    monkeypatch.setenv("STORAGE_TYPE", "s3")
    reset_settings()
    assert_storage_shared_across_processes()

    # Akış/Kuyruk modu kapalıysa memory sorunsuz geçmeli
    monkeypatch.setenv("USE_REDIS_STREAM", "false")
    monkeypatch.setenv("USE_REDIS_QUEUE", "false")
    monkeypatch.setenv("STORAGE_TYPE", "memory")
    reset_settings()
    assert_storage_shared_across_processes()
    reset_settings()


@pytest.mark.unit
def test_in_memory_storage_byte_limit_and_concurrency():
    """InMemoryStorageAdapter'ın bayt sınırı aşımında en eski kayıtları sildiğini ve thread-safe olduğunu doğrular."""
    # Sınıf değişkenini temizle
    InMemoryStorageAdapter._storage.clear()
    InMemoryStorageAdapter._total_bytes = 0

    adapter = InMemoryStorageAdapter(max_items=1000, max_bytes=100)

    # 1. 50'şer baytlık 3 kayıt ekle (Toplam 150 bayt > 100 bayt max_bytes)
    uri1 = adapter.save(b"x" * 50, "file1.wav")
    uri2 = adapter.save(b"y" * 50, "file2.wav")
    uri3 = adapter.save(b"z" * 50, "file3.wav")

    # En eski kayıt (uri1) silinmiş olmalı
    with pytest.raises(FileNotFoundError):
        adapter.get_bytes(uri1)

    assert adapter.get_bytes(uri2) == b"y" * 50
    assert adapter.get_bytes(uri3) == b"z" * 50
    assert InMemoryStorageAdapter._total_bytes <= 100

    # 2. Çoklu izlek (Eşzamanlı 8 thread x 200 işlem) eşzamanlılık testi
    def worker_task(thread_id):
        for i in range(200):
            u = adapter.save(f"thread_{thread_id}_data_{i}".encode(), f"t{thread_id}_{i}.wav")
            if i % 2 == 0:
                adapter.delete(u)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_task, t) for t in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert InMemoryStorageAdapter._total_bytes >= 0


@pytest.mark.unit
async def test_job_service_execute_job_lazy_get_path():
    """process_bytes çalıştığında storage.get_path çağrısının 0 kez tetiklendiğini (lazy) doğrular."""
    mock_storage = MagicMock()
    mock_storage.get_bytes.return_value = b"test_audio_bytes"
    mock_storage.get_path.return_value = "dummy/path.wav"

    mock_repo = AsyncMock()
    record_id = uuid.uuid4()
    mock_record = AudioRecord(
        id=record_id,
        storage_uri="s3://bucket/test.wav",
        file_name="test.wav",
        status=JobStatus.PENDING,
    )
    mock_repo.get_record_by_id.return_value = mock_record
    mock_repo.claim_job_atomically.return_value = (True, mock_record, False)
    mock_repo.update_status.return_value = True
    mock_repo.save_utterances.return_value = True

    mock_pipeline = MagicMock()
    mock_pipeline.process_bytes.return_value = ([], "tr", OverlapSummary())

    job_service = JobService(
        storage=mock_storage,
        repository=mock_repo,
        pipeline=mock_pipeline,
    )

    success = await job_service.execute_job(record_id)
    assert success is True

    # process_bytes çağrıldığı için get_path 0 kez tetiklenmeli
    mock_storage.get_path.assert_not_called()
    mock_pipeline.process_bytes.assert_called_once_with(b"test_audio_bytes", num_speakers=None)


@pytest.mark.unit
def test_get_storage_type_defaults_to_disk(monkeypatch):
    """STORAGE_TYPE env boşken get_storage_type() == 'disk' olmalı."""
    from audio_analyzer.config import reset_settings
    monkeypatch.delenv("STORAGE_TYPE", raising=False)
    reset_settings()
    from audio_analyzer.adapters.storage.storage_factory import get_storage_type, assert_storage_shared_across_processes
    assert get_storage_type() == "disk"

    monkeypatch.setenv("USE_REDIS_STREAM", "true")
    reset_settings()
    # assert_storage_shared_across_processes should not raise when STORAGE_TYPE is default (disk)
    assert_storage_shared_across_processes()
    reset_settings()

