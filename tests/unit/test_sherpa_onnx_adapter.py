import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxAudioAdapter
from audio_analyzer.domain.models import DiarizationSegment


def test_sherpa_onnx_adapter_initialization():
    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    assert adapter.num_speakers == 2
    assert adapter._initialized is False
    assert adapter._pipeline is None


def test_sherpa_onnx_adapter_lazy_load_real(tmp_path):
    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    adapter._lazy_load_pipeline()
    if adapter._pipeline is not None:
        assert adapter._initialized is True
        assert adapter._config is not None


def test_sherpa_onnx_adapter_missing_models(tmp_path):
    with patch("audio_analyzer.config.get_settings") as mock_settings:
        mock_settings.return_value.model_dir = str(tmp_path / "non_existent_models")
        adapter = SherpaOnnxAudioAdapter()
        adapter._lazy_load_pipeline()
        assert adapter._pipeline is None
        segments = adapter.diarize("dummy.wav")
        assert segments == []


def test_sherpa_onnx_adapter_diarize_sample():
    eval_wav = Path("tests/fixtures/diarization_eval/large/eval_sample_01_2spk.wav")
    if not eval_wav.exists():
        pytest.skip("Fixture file not found")

    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    segments = adapter.diarize(str(eval_wav))

    assert len(segments) > 0
    for seg in segments:
        assert isinstance(seg, DiarizationSegment)
        assert seg.speaker_id.startswith("SPEAKER_")
        assert seg.start_time >= 0.0
        assert seg.end_time >= seg.start_time


def test_sherpa_onnx_adapter_numpy_input():
    eval_wav = Path("tests/fixtures/diarization_eval/large/eval_sample_01_2spk.wav")
    if not eval_wav.exists():
        pytest.skip("Fixture file not found")

    import soundfile as sf
    audio, sr = sf.read(str(eval_wav), dtype="float32")

    adapter = SherpaOnnxAudioAdapter(num_speakers=2)
    segments = adapter.diarize(audio, num_speakers=2)

    assert len(segments) > 0
    assert isinstance(segments[0], DiarizationSegment)


def test_cluster_preservation_filter():
    adapter = SherpaOnnxAudioAdapter(min_duration_on=0.3)

    class MockRawSegment:
        def __init__(self, speaker, start, end):
            self.speaker = speaker
            self.start = start
            self.end = end

        @property
        def duration(self):
            return self.end - self.start

    raw_segs = [
        MockRawSegment(speaker=0, start=0.0, end=1.0),
        MockRawSegment(speaker=1, start=1.5, end=2.5),
        MockRawSegment(speaker=2, start=3.0, end=3.1),  # 0.1s duration < 0.3s
    ]

    # Without num_speakers: speaker 2 segment should be filtered out
    filtered = adapter._apply_cluster_preservation_filter(raw_segs, num_speakers_requested=None)
    spk_ids = {s.speaker_id for s in filtered}
    assert "SPEAKER_02" not in spk_ids

    # With num_speakers=3: speaker 2 segment should be preserved!
    filtered_preserved = adapter._apply_cluster_preservation_filter(raw_segs, num_speakers_requested=3)
    spk_ids_preserved = {s.speaker_id for s in filtered_preserved}
    assert "SPEAKER_02" in spk_ids_preserved


def test_sherpa_onnx_multithreaded_lock(tmp_path):
    adapter = SherpaOnnxAudioAdapter()

    mock_pipeline = MagicMock()
    mock_raw_seg = MagicMock()
    mock_raw_seg.speaker = 0
    mock_raw_seg.start = 0.0
    mock_raw_seg.end = 1.0
    mock_raw_seg.duration = 1.0
    mock_result = MagicMock()
    mock_result.sort_by_start_time.return_value = [mock_raw_seg]
    mock_pipeline.process.return_value = mock_result

    adapter._pipeline = mock_pipeline
    adapter._config = MagicMock()
    adapter._config.clustering.num_clusters = -1
    adapter._initialized = True

    results = {}

    def worker(num_spk, key):
        segs = adapter.diarize(np.zeros(16000, dtype=np.float32), num_speakers=num_spk)
        results[key] = segs

    import threading
    t1 = threading.Thread(target=worker, args=(2, "t1"))
    t2 = threading.Thread(target=worker, args=(3, "t2"))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert "t1" in results and "t2" in results
    assert len(results["t1"]) > 0
    assert len(results["t2"]) > 0


