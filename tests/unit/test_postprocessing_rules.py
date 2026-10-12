import uuid
import numpy as np
import pytest

from audio_analyzer.domain.models import DiarizationSegment, TranscriptUtterance, WordSegment
from audio_analyzer.services.fusion_engine import FusionEngine
from audio_analyzer.services.semantic_refiner import SemanticRefiner
from audio_analyzer.services.speaker_merger import SpeakerClusterMerger


def test_speaker_cluster_merger_toggle(monkeypatch):
    """ENABLE_SPEAKER_MERGER=false iken birleştirme yapılmadığını doğrular."""
    segs = [
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0),
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=10.0, end_time=11.0), # 1s (<3s)
    ]
    merger = SpeakerClusterMerger()
    
    # 1. Varsayılan aktif: SPEAKER_01 SPEAKER_00'a devredilir
    monkeypatch.setenv("ENABLE_SPEAKER_MERGER", "true")
    res_on = merger.merge(segs)
    assert len({s.speaker_id for s in res_on}) == 1

    # 2. Kapalı: SPEAKER_01 korunur
    monkeypatch.setenv("ENABLE_SPEAKER_MERGER", "false")
    res_off = merger.merge(segs)
    assert len({s.speaker_id for s in res_off}) == 2


def test_fusion_smooth_words_toggle_and_short_responses(monkeypatch):
    """
    ENABLE_SMOOTH_WORDS=false iken kısa 1 kelimelik cevapların ('Evet.')
    karşı tarafa devredilmediğini ve zincirleme etki olmadığını doğrular.
    """
    words = [
        WordSegment(word="Nasılsınız", start_time=0.0, end_time=1.0),
        WordSegment(word="İyiyim", start_time=1.2, end_time=1.8), # B'nin cevabı
        WordSegment(word="Teşekkürler", start_time=2.0, end_time=3.0),
    ]
    diar = [
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=1.1),
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=1.15, end_time=1.9),
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=1.95, end_time=3.1),
    ]

    # Kapalıyken İyiyim SPEAKER_01 kalmalı
    engine_off = FusionEngine(enable_smooth_words=False, bind_short_turns=False)
    res_off = engine_off.align(words, diar)
    assert any(u.speaker_id == "SPEAKER_01" and "İyiyim" in u.text for u in res_off)

    # Açıkken _smooth_attributed_words yumuşatır
    engine_on = FusionEngine(enable_smooth_words=True, bind_short_turns=False)
    res_on = engine_on.align(words, diar)
    # Tek kelimelik sıçrama yumuşatılır
    assert len(res_on) == 1


def test_semantic_refiner_sandwich_rule_duration_only_and_no_cascading(monkeypatch):
    """
    Sandviç kuralında kelime sayısı koşulunun kalktığını, süresi >= 0.8s olan
    kısa cevapların (örn. 2 kelimelik 'Evet, doğru.' 1.0s) korunmasını ve
    kararların orijinal listeye göre alınarak zincirleme etki yapmadığını doğrular.
    """
    monkeypatch.setenv("ENABLE_SANDWICH_RULE", "true")
    monkeypatch.setenv("SANDWICH_MAX_DURATION_SEC", "0.8")

    refiner = SemanticRefiner(use_llm=False)

    u1 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=0.0, end_time=4.0, text="Yarın gelecek misiniz?")
    # 2 kelime ama süresi 1.2s (> 0.8s) -> Eski kuralda len<=3 yüzünden SPEAKER_00'a eziliyordu, artık KORUNMALI!
    u2 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_01", start_time=4.2, end_time=5.4, text="Evet, geleceğim.")
    u3 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=5.6, end_time=9.0, text="Tamam o zaman görüşürüz.")

    res = refiner.refine([u1, u2, u3])
    assert len(res) == 3
    assert res[1].speaker_id == "SPEAKER_01"
    assert res[1].text == "Evet, geleceğim."

    # Çok kısa (<0.8s) parazit sandviç ise ezilmeli
    u2_short = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_01", start_time=4.2, end_time=4.5, text="Hı")
    res_short = refiner.refine([u1, u2_short, u3])
    # SPEAKER_01 SPEAKER_00'a ezilmeli ve birleşmeli
    assert len(res_short) == 1
    assert res_short[0].speaker_id == "SPEAKER_00"


def test_semantic_refiner_sandwich_toggle(monkeypatch):
    """ENABLE_SANDWICH_RULE=false iken sandviç yumuşatmanın çalışmadığını doğrular."""
    monkeypatch.setenv("ENABLE_SANDWICH_RULE", "false")
    refiner = SemanticRefiner(use_llm=False)

    u1 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=0.0, end_time=4.0, text="A")
    u2_short = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_01", start_time=4.1, end_time=4.4, text="B")
    u3 = TranscriptUtterance(id=uuid.uuid4(), speaker_id="SPEAKER_00", start_time=4.5, end_time=8.0, text="A2")

    res = refiner.refine([u1, u2_short, u3])
    # Kural kapalı olduğu için SPEAKER_01 korunur
    assert any(u.speaker_id == "SPEAKER_01" for u in res)
