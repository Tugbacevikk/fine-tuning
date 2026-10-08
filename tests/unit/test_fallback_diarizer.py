import pytest
from audio_analyzer.adapters.diarization.fallback_diarizer import FallbackDiarizer
from audio_analyzer.domain.interfaces import IDiarizer
from audio_analyzer.domain.models import DiarizationSegment
from audio_analyzer.services.pipeline import AudioAnalysisPipeline


class MockFailingDiarizer(IDiarizer):
    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[DiarizationSegment]:
        raise RuntimeError("Primary diarizer GPU out of memory")


class MockEmptyDiarizer(IDiarizer):
    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[DiarizationSegment]:
        return []


class MockSuccessfulDiarizer(IDiarizer):
    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[DiarizationSegment]:
        return [
            DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0),
            DiarizationSegment(speaker_id="SPEAKER_01", start_time=2.1, end_time=4.0),
        ]


def test_fallback_diarizer_primary_success():
    primary = MockSuccessfulDiarizer()
    fallback = MockEmptyDiarizer()
    diarizer = FallbackDiarizer(primary_diarizer=primary, fallback_diarizers=[fallback])

    res = diarizer.diarize("dummy.wav")
    assert len(res) == 2
    assert res[0].speaker_id == "SPEAKER_00"


def test_fallback_diarizer_triggers_fallback_on_exception():
    primary = MockFailingDiarizer()
    fallback = MockSuccessfulDiarizer()
    diarizer = FallbackDiarizer(primary_diarizer=primary, fallback_diarizers=[fallback])

    res = diarizer.diarize("dummy.wav")
    assert len(res) == 2
    assert res[0].speaker_id == "SPEAKER_00"


def test_fallback_diarizer_triggers_fallback_on_empty():
    primary = MockEmptyDiarizer()
    fallback = MockSuccessfulDiarizer()
    diarizer = FallbackDiarizer(primary_diarizer=primary, fallback_diarizers=[fallback])

    res = diarizer.diarize("dummy.wav")
    assert len(res) == 2
    assert res[0].speaker_id == "SPEAKER_00"


def test_conditional_diarization_single_speaker_bypasses_engine():
    class MockCallTrackerSTT:
        def transcribe(self, audio_path):
            from audio_analyzer.domain.models import WordSegment

            return [WordSegment(word="Test", start_time=0.0, end_time=1.0)], "tr"

    class MockTrackingDiarizer(IDiarizer):
        def __init__(self):
            self.called = False

        def diarize(self, audio_path: str, num_speakers: int | None = None):
            self.called = True
            return []

    stt = MockCallTrackerSTT()
    tracker_diarizer = MockTrackingDiarizer()
    pipeline = AudioAnalysisPipeline(stt_engine=stt, diarizer=tracker_diarizer)

    # single speaker mode (num_speakers=1)
    utterances, lang, _ = pipeline._execute_pipeline("test.wav", audio_duration=10.0, num_speakers=1)

    assert tracker_diarizer.called is False
    assert len(utterances) == 1
    assert utterances[0].speaker_id == "SPEAKER_00"


def test_failing_diarizer_called_exactly_once():
    """Doğrular: Çöken bir motor FallbackDiarizer tarafından tam olarak 1 kez çağrılır (çift çalıştırma yapılmaz)."""
    class CountingFailingDiarizer(IDiarizer):
        def __init__(self):
            self.call_count = 0

        def diarize(self, audio_path: str, num_speakers: int | None = None):
            self.call_count += 1
            raise RuntimeError("Engine CUDA OOM Failure")

    failing_primary = CountingFailingDiarizer()
    successful_fallback = MockSuccessfulDiarizer()
    diarizer = FallbackDiarizer(primary_diarizer=failing_primary, fallback_diarizers=[successful_fallback])

    res = diarizer.diarize("test.wav")
    assert failing_primary.call_count == 1
    assert len(res) == 2
    assert res[0].speaker_id == "SPEAKER_00"
