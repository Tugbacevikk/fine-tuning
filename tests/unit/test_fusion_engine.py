import pytest

from audio_analyzer.domain.models import DiarizationSegment, WordSegment
from audio_analyzer.services.fusion_engine import FusionEngine


@pytest.mark.unit
def test_fusion_engine_speaker_attribution(sample_stt_words, sample_diarization_segments):
    """Kelime orta noktası ve IoU algoritmasının konuşmacıları doğru eşlediğini doğrular."""
    engine = FusionEngine(max_silence_threshold=1.5, bind_short_turns=False)
    utterances = engine.align(sample_stt_words, sample_diarization_segments)

    assert len(utterances) == 3

    # 1. Cümle: SPEAKER_00 ("Merhaba nasılsınız")
    assert utterances[0].speaker_id == "SPEAKER_00"
    assert utterances[0].text == "Merhaba nasılsınız"

    # 2. Cümle: SPEAKER_01 ("Ben iyiyim")
    assert utterances[1].speaker_id == "SPEAKER_01"
    assert utterances[1].text == "Ben iyiyim"

    # 3. Cümle: SPEAKER_00 ("Teşekkürler") - 2.0 saniye üzerindeki sessizlik yüzünden ayrı paragraf oldu!
    assert utterances[2].speaker_id == "SPEAKER_00"
    assert utterances[2].text == "Teşekkürler"


@pytest.mark.unit
def test_fusion_engine_silence_threshold_trigger():
    """3.0 saniyenin üzerindeki sessizlik boşluğunun aynı konuşmacıyı 2 ayrı Utterances bloğuna böldüğünü doğrular."""
    words = [
        WordSegment(word="Birinci", start_time=0.0, end_time=0.5),
        WordSegment(word="cümle", start_time=0.6, end_time=1.0),
        # 1.0s - 4.5s arası 3.5 saniyelik boşluk Var (> 3.0s)
        WordSegment(word="İkinci", start_time=4.5, end_time=5.0),
        WordSegment(word="cümle", start_time=5.1, end_time=5.5),
    ]
    diarization = [DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=6.0)]

    engine = FusionEngine(max_silence_threshold=3.0)
    utterances = engine.align(words, diarization)

    assert len(utterances) == 2
    assert utterances[0].speaker_id == "SPEAKER_00"
    assert utterances[0].text == "Birinci cümle"
    assert utterances[1].speaker_id == "SPEAKER_00"
    assert utterances[1].text == "İkinci cümle"


@pytest.mark.unit
def test_fusion_engine_empty_input():
    """Boş kelime listesinde boş liste dönüldüğünü doğrular."""
    engine = FusionEngine()
    assert engine.align([], []) == []
