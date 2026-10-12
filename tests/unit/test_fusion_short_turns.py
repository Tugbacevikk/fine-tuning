import uuid
import pytest

from audio_analyzer.domain.models import TranscriptUtterance
from audio_analyzer.services.fusion_engine import FusionEngine


@pytest.mark.unit
def test_bind_short_unpunctuated_turn_to_previous_speaker():
    """Süresi <1.0s, önceki noktalama ile bitmeyen ve gap<0.3s olan parçanın önceki konuşmacıya bağlandığını doğrular."""
    fusion = FusionEngine()
    utterances = [
        # SPEAKER_00: 38.42 - 43.98s, noktalama yok ("...bu sohbetlerimize")
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_00",
            start_time=38.42,
            end_time=43.98,
            text="Yeşil çayımı aldım sohbete hazırım bu sohbetlerimize",
        ),
        # SPEAKER_01: 43.98 - 44.84s (0.86s, gap=0.0s), metin="bayılıyorum"
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_01",
            start_time=43.98,
            end_time=44.84,
            text="bayılıyorum",
        ),
    ]

    bound = fusion._bind_short_unpunctuated_turns(utterances)
    assert len(bound) == 2
    assert bound[0].speaker_id == "SPEAKER_00"
    assert bound[1].speaker_id == "SPEAKER_00"  # SPEAKER_00'a bağlandı
    assert bound[1].text == "bayılıyorum"  # Metin değişmedi


@pytest.mark.unit
def test_bind_short_turn_does_not_bind_if_previous_has_punctuation():
    """Önceki cümle nokta veya soru işaretiyle bitiyorsa kısa parça önceki konuşmacıya bağlanmaz."""
    fusion = FusionEngine()
    utterances = [
        # Nokta ile bitiyor
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_00",
            start_time=10.0,
            end_time=12.0,
            text="Senin keyfin nasıl?",
        ),
        # Kısa cevap (0.5s, gap=0.1s)
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_01",
            start_time=12.1,
            end_time=12.6,
            text="İyiyim.",
        ),
    ]

    bound = fusion._bind_short_unpunctuated_turns(utterances)
    assert bound[1].speaker_id == "SPEAKER_01"  # Değişmemeli


@pytest.mark.unit
def test_bind_short_turn_does_not_bind_if_gap_too_large():
    """Aralarındaki boşluk 0.3s veya üzerindeyse kısa parça bağlanmaz."""
    fusion = FusionEngine()
    utterances = [
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_00",
            start_time=10.0,
            end_time=12.0,
            text="Burada bekliyoruz",
        ),
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_01",
            start_time=12.5,  # gap = 0.5s >= 0.3s
            end_time=13.0,
            text="evet",
        ),
    ]

    bound = fusion._bind_short_unpunctuated_turns(utterances)
    assert bound[1].speaker_id == "SPEAKER_01"


@pytest.mark.unit
def test_bind_short_turn_does_not_bind_if_duration_ge_one_second():
    """Süresi 1.0s ve üzeri olan parçalar bağlanmaz."""
    fusion = FusionEngine()
    utterances = [
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_00",
            start_time=10.0,
            end_time=12.0,
            text="Burada bekliyoruz",
        ),
        TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id="SPEAKER_01",
            start_time=12.1,
            end_time=13.3,  # dur = 1.2s >= 1.0s
            text="ben de bekliyorum",
        ),
    ]

    bound = fusion._bind_short_unpunctuated_turns(utterances)
    assert bound[1].speaker_id == "SPEAKER_01"
