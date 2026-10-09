import uuid

from audio_analyzer.domain.models import TranscriptUtterance
from audio_analyzer.services.semantic_refiner import SemanticRefiner


def test_semantic_refiner_splits_merged_utterance():
    refiner = SemanticRefiner(domain_mode="call_center")

    # Müşteri şikayeti ve Temsilci kapanışının tek bir SPEAKER_00 bloğunda birleştiği durum
    merged_text = "Merhaba, ben Elif Çevik. Bilgisayar yavaş çalışıyor. Şikayetinizi not aldım ve konuyu en kısa sürede çözmek için gerekli adımları atacağız."
    utt = TranscriptUtterance(
        id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=6.58, end_time=59.43, text=merged_text
    )

    result = refiner.refine([utt])

    # 2 ayrı parçaya bölünmüş olmalı
    assert len(result) == 2
    assert result[0].speaker_id == "SPEAKER_00"
    assert "Elif Çevik" in result[0].text
    assert result[1].speaker_id == "SPEAKER_01"
    assert "Şikayetinizi not aldım" in result[1].text


def test_semantic_refiner_domain_mode_greeting_lock_scope():
    u1 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=1.0, end_time=3.0, text="Hoş geldiniz efendim.")
    u2 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_01", start_time=3.2, end_time=5.0, text="Hoş geldiniz, nasılsınız?")

    # 1. domain_mode=None (genel amaçlı) modunda iki ayrı konuşmacı korunmalı
    refiner_general = SemanticRefiner(domain_mode=None)
    res_general = refiner_general.refine([u1, u2])
    assert len(res_general) == 2
    assert res_general[0].speaker_id == "SPEAKER_00"
    assert res_general[1].speaker_id == "SPEAKER_01"

    # 2. domain_mode="call_center" modunda açılış selamlaması tek temsilci kartına birleştirilmeli
    refiner_cc = SemanticRefiner(domain_mode="call_center")
    res_cc = refiner_cc.refine([u1, u2])
    assert len(res_cc) == 1
    assert res_cc[0].speaker_id == "SPEAKER_00"


def test_semantic_refiner_splits_dialogue_question_turns():
    refiner = SemanticRefiner(split_on_questions=True)
    utt = TranscriptUtterance(
        id=uuid.uuid4(),
        speaker_id="SPEAKER_00",
        start_time=10.0,
        end_time=20.0,
        text="Merhaba doktor bey, hasta ne zaman taburcu olacak? Hasta yakınlarını şimdi almaya başlıyoruz.",
    )

    result = refiner.refine([utt])

    assert len(result) == 2
    assert result[0].speaker_id == "SPEAKER_00"
    assert "hasta ne zaman taburcu olacak?" in result[0].text
    assert result[1].speaker_id == "SPEAKER_01"
    assert "Hasta yakınlarını şimdi almaya başlıyoruz." in result[1].text


def test_semantic_refiner_llm_mode_does_not_bypass_rules_and_respects_feedback_profile(monkeypatch):
    from unittest.mock import patch

    # 1. LLM active: mock Ollama response, verify rule-based steps still execute
    monkeypatch.setenv("PIPELINE_PROFILE", "full")
    refiner = SemanticRefiner(domain_mode="call_center", use_llm=True)
    merged_text = "Merhaba, ben Elif Çevik. Bilgisayar yavaş çalışıyor. Şikayetinizi not aldım ve konuyu en kısa sürede çözmek için gerekli adımları atacağız."
    utt = TranscriptUtterance(
        id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=6.58, end_time=59.43, text=merged_text
    )

    with patch.object(refiner, "_query_ollama_llm", return_value="Dummy LLM output") as mock_query:
        res = refiner.refine([utt])
        mock_query.assert_called_once()
        # Rule-based split should still happen!
        assert len(res) == 2
        assert res[0].speaker_id == "SPEAKER_00"
        assert res[1].speaker_id == "SPEAKER_01"

    # 2. PIPELINE_PROFILE=feedback: LLM must NOT be called
    monkeypatch.setenv("PIPELINE_PROFILE", "feedback")
    refiner_fb = SemanticRefiner(domain_mode="call_center", use_llm=True)
    with patch.object(refiner_fb, "_query_ollama_llm") as mock_query_fb:
        res_fb = refiner_fb.refine([utt])
        mock_query_fb.assert_not_called()
        assert len(res_fb) == 2


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


def test_semantic_refiner_llm_json_parsing_and_fail_open(monkeypatch):
    from unittest.mock import patch

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





