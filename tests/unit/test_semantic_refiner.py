import uuid
from unittest.mock import patch

from audio_analyzer.domain.models import TranscriptUtterance
from audio_analyzer.services.semantic_refiner import SemanticRefiner


def test_semantic_refiner_smoothes_orphan_and_sandwich_speakers():
    refiner = SemanticRefiner()

    u1 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0, text="Merhaba hastaneyi arıyorum.")
    u_punct = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_04", start_time=10.0, end_time=10.07, text="?")
    u2 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=10.1, end_time=20.0, text="Yok yok bu haftadan bahsediyorum.")

    res = refiner.refine([u1, u_punct, u2])

    # SPEAKER_04 should be smoothed and merged into SPEAKER_00!
    speakers = {u.speaker_id for u in res}
    assert "SPEAKER_04" not in speakers
    assert len(res) == 1
    assert res[0].speaker_id == "SPEAKER_00"
    assert "?" in res[0].text


def test_semantic_refiner_llm_mode_respects_feedback_profile(monkeypatch):
    # 1. LLM active: mock Ollama response, verify LLM is called
    monkeypatch.setenv("PIPELINE_PROFILE", "full")
    refiner = SemanticRefiner(use_llm=True)
    utt = TranscriptUtterance(
        id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0, text="Merhaba hastaneyi arıyorum."
    )

    with patch.object(refiner, "_query_ollama_llm", return_value='[{"idx": 0, "speaker": "SPEAKER_01"}]') as mock_query:
        res = refiner.refine([utt])
        mock_query.assert_called_once()
        assert res[0].speaker_id == "SPEAKER_01"

    # 2. PIPELINE_PROFILE=feedback: LLM must NOT be called
    monkeypatch.setenv("PIPELINE_PROFILE", "feedback")
    refiner_fb = SemanticRefiner(use_llm=True)
    with patch.object(refiner_fb, "_query_ollama_llm") as mock_query_fb:
        res_fb = refiner_fb.refine([utt])
        mock_query_fb.assert_not_called()
        assert res_fb[0].speaker_id == "SPEAKER_00"


def test_semantic_refiner_llm_json_parsing_and_fail_open(monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "full")
    refiner = SemanticRefiner(use_llm=True)

    u1 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0, text="Alo.")
    u2 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_01", start_time=2.5, end_time=5.0, text="Şehir Hastanesinden arıyorum.")
    u3 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_02", start_time=5.5, end_time=15.0, text="Cuma günü ultrason çekiminiz var.")

    # 1. Success case: LLM maps SPEAKER_01 and SPEAKER_02 to SPEAKER_00
    mock_llm_json = '[{"idx": 0, "speaker": "SPEAKER_00"}, {"idx": 1, "speaker": "SPEAKER_00"}, {"idx": 2, "speaker": "SPEAKER_00"}]'
    with patch.object(refiner, "_query_ollama_llm", return_value=mock_llm_json):
        res = refiner.refine([u1, u2, u3])
        # All three utterances should now belong to SPEAKER_00 and be merged into 1 utterance!
        assert len(res) == 1
        assert res[0].speaker_id == "SPEAKER_00"
        assert "Şehir Hastanesinden arıyorum" in res[0].text

    # 2. Fail-open case: Ollama offline / returns None
    with patch.object(refiner, "_query_ollama_llm", return_value=None):
        res_fail = refiner.refine([u1, u2, u3])
        # Should gracefully return without error
        assert len(res_fail) > 0
