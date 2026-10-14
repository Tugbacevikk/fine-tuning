import numpy as np
import pytest

from audio_analyzer.domain.models import DiarizationSegment
from audio_analyzer.services.speaker_merger import SpeakerClusterMerger


@pytest.mark.unit
def test_speaker_merger_merges_small_speaker_to_temporally_nearest():
    """3.0s'den ve %5'ten az konuşan mikro konuşmacının zamanca en yakın büyük konuşmacıya bağlandığını doğrular."""
    segments = [
        # SPEAKER_00: toplam 10.0s (Büyük)
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0),
        # SPEAKER_01: toplam 10.0s (Büyük)
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=10.5, end_time=20.5),
        # SPEAKER_02: 0.8s (Küçük - parazit/bayılıyorum segmenti) -> SPEAKER_01'e yapışmalı
        DiarizationSegment(speaker_id="SPEAKER_02", start_time=20.5, end_time=21.3),
        # SPEAKER_00: toplam 10.0s (Büyük)
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=22.0, end_time=32.0),
    ]
    merger = SpeakerClusterMerger(min_speaker_sec=3.0, min_speaker_ratio=0.05)
    merged = merger.merge(segments)

    spk_ids = set(s.speaker_id for s in merged)
    assert "SPEAKER_02" not in spk_ids
    assert spk_ids == {"SPEAKER_00", "SPEAKER_01"}

    # 20.5 - 21.3 segmenti SPEAKER_01 ile bitişik olduğundan SPEAKER_01'e eklenmiş olmalı
    spk01_segs = [s for s in merged if s.speaker_id == "SPEAKER_01"]
    assert len(spk01_segs) == 1
    assert spk01_segs[0].start_time == 10.5
    assert spk01_segs[0].end_time == 21.3


@pytest.mark.unit
def test_speaker_merger_respects_num_speakers_floor():
    """num_speakers verilmişse küme sayısının num_speakers'ın altına düşürülmediğini doğrular."""
    segments = [
        # SPEAKER_00: 20.0s
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=20.0),
        # SPEAKER_01: 2.0s (normalde < 3.0s küçük)
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=21.0, end_time=23.0),
    ]
    merger = SpeakerClusterMerger(min_speaker_sec=3.0, min_speaker_ratio=0.05)

    # num_speakers=2 verildiğinde SPEAKER_01 yok edilmemeli!
    merged = merger.merge(segments, num_speakers=2)
    spk_ids = set(s.speaker_id for s in merged)
    assert spk_ids == {"SPEAKER_00", "SPEAKER_01"}


@pytest.mark.unit
def test_speaker_merger_env_variables_override(monkeypatch):
    """MIN_SPEAKER_SEC ve MIN_SPEAKER_RATIO ortam değişkenlerinin çalıştığını doğrular."""
    monkeypatch.setenv("MIN_SPEAKER_SEC", "1.0")
    monkeypatch.setenv("MIN_SPEAKER_RATIO", "0.01")

    merger = SpeakerClusterMerger()
    assert merger.min_speaker_sec == 1.0
    assert merger.min_speaker_ratio == 0.01


@pytest.mark.unit
def test_speaker_merger_with_embeddings():
    """Embedding merkezleri sağlandığında en yakın vektöre (kosinüs) atandığını doğrular."""
    segments = [
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0),
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=10.0, end_time=20.0),
        DiarizationSegment(speaker_id="SPEAKER_02", start_time=20.0, end_time=21.0),  # küçük
    ]

    # SPEAKER_02 vektörü SPEAKER_00'a çok yakın ([1, 0] vs [0.99, 0.01])
    embeddings = {
        "SPEAKER_00": np.array([1.0, 0.0]),
        "SPEAKER_01": np.array([0.0, 1.0]),
        "SPEAKER_02": np.array([0.98, 0.02]),
    }

    merger = SpeakerClusterMerger(min_speaker_sec=3.0, min_speaker_ratio=0.05)
    merged = merger.merge(segments, speaker_embeddings=embeddings)

    spk02_target = [s for s in merged if s.start_time >= 20.0][0]
    assert spk02_target.speaker_id == "SPEAKER_00"


@pytest.mark.unit
def test_speaker_merger_cancels_merge_when_similarity_below_threshold():
    """Benzerlik eşiğin altındaysa birleştirmenin iptal edildiğini ve konuşmacının korunduğunu doğrular."""
    segments = [
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0),
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=10.0, end_time=20.0),
        DiarizationSegment(speaker_id="SPEAKER_02", start_time=20.0, end_time=21.5),  # 1.5s < 3.0s
    ]

    # SPEAKER_02, SPEAKER_00 ve SPEAKER_01'e tamamen ortogonal ([0, 0, 1]) -> sim=0.0 < 0.50
    embeddings = {
        "SPEAKER_00": np.array([1.0, 0.0, 0.0]),
        "SPEAKER_01": np.array([0.0, 1.0, 0.0]),
        "SPEAKER_02": np.array([0.0, 0.0, 1.0]),
    }

    merger = SpeakerClusterMerger(min_speaker_sec=3.0, min_speaker_ratio=0.05, similarity_threshold=0.50)
    merged = merger.merge(segments, speaker_embeddings=embeddings)

    spk_ids = set(s.speaker_id for s in merged)
    # Düşük benzerlik sebebiyle SPEAKER_02 yutulmamalı, korunmalı!
    assert "SPEAKER_02" in spk_ids
    assert len(spk_ids) == 3


@pytest.mark.unit
def test_speaker_merger_enforces_target_num_speakers_using_cosine_similarity():
    """Hedef num_speakers=2 verildiğinde ve 3 büyük konuşmacı olduğunda benzer olan ikilinin birleştiğini doğrular."""
    segments = [
        # SPEAKER_00: 20s
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=20.0),
        # SPEAKER_01: 20s
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=20.0, end_time=40.0),
        # SPEAKER_02: 15s (Büyük, normalde <3s filtresine takılmaz)
        DiarizationSegment(speaker_id="SPEAKER_02", start_time=40.0, end_time=55.0),
    ]

    # SPEAKER_02, SPEAKER_00 ile %95 benzer (aynı kişi), SPEAKER_01 ile farklı
    embeddings = {
        "SPEAKER_00": np.array([1.0, 0.0]),
        "SPEAKER_01": np.array([0.0, 1.0]),
        "SPEAKER_02": np.array([0.95, 0.05]),
    }

    merger = SpeakerClusterMerger(min_speaker_sec=3.0, min_speaker_ratio=0.05)
    merged = merger.merge(segments, num_speakers=2, speaker_embeddings=embeddings)

    spk_ids = set(s.speaker_id for s in merged)
    assert len(spk_ids) == 2
    assert "SPEAKER_00" in spk_ids
    assert "SPEAKER_01" in spk_ids
    assert "SPEAKER_02" not in spk_ids


