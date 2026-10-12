import logging
import os
import numpy as np

from audio_analyzer.domain.models import DiarizationSegment

logger = logging.getLogger(__name__)


class SpeakerClusterMerger:
    """
    Küçük konuşmacıları birleştirme servisi (Diarization Post-Processing).
    
    Toplam konuşma süresi MIN_SPEAKER_SEC (varsayılan 3.0s) veya toplam konuşma
    süresine oranı MIN_SPEAKER_RATIO (varsayılan 0.05 / %5) altında kalan küçük
    konuşmacıları tespit eder.
    
    Bu konuşmacıların segmentlerini:
    1. Varsa embedding merkezine (cosine similarity) en yakın büyük konuşmacıya atar.
    2. Embedding yoksa zamanca en yakın komşu büyük segmentin konuşmacısına atar.
    
    Kural: num_speakers verilmişse küme sayısını asla num_speakers altına düşürmez.
    """

    def __init__(
        self,
        min_speaker_sec: float | None = None,
        min_speaker_ratio: float | None = None,
    ):
        from audio_analyzer.config import get_settings
        settings = get_settings()

        env_sec = os.getenv("MIN_SPEAKER_SEC")
        env_ratio = os.getenv("MIN_SPEAKER_RATIO")

        self.min_speaker_sec = (
            min_speaker_sec
            if min_speaker_sec is not None
            else float(env_sec) if env_sec is not None else settings.min_speaker_sec
        )
        self.min_speaker_ratio = (
            min_speaker_ratio
            if min_speaker_ratio is not None
            else float(env_ratio) if env_ratio is not None else settings.min_speaker_ratio
        )

    def merge(
        self,
        segments: list[DiarizationSegment],
        num_speakers: int | None = None,
        speaker_embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[DiarizationSegment]:
        if not segments:
            return []

        from audio_analyzer.config import get_settings
        if not get_settings().enable_speaker_merger:
            return segments

        # 1. Konuşmacı sürelerini topla
        spk_durations: dict[str, float] = {}
        for s in segments:
            dur = max(0.0, s.end_time - s.start_time)
            spk_durations[s.speaker_id] = spk_durations.get(s.speaker_id, 0.0) + dur

        unique_speakers = list(spk_durations.keys())
        if len(unique_speakers) <= 1:
            return segments

        # num_speakers verilmişse ve mevcut konuşmacı sayısı zaten <= num_speakers ise birleştirme yapma
        if num_speakers is not None and num_speakers > 0 and len(unique_speakers) <= num_speakers:
            return segments

        total_speech_sec = sum(spk_durations.values())

        # 2. Küçük konuşmacıları bul (Süre < MIN_SPEAKER_SEC VEYA Oran < MIN_SPEAKER_RATIO)
        small_speakers = set()
        for spk, dur in spk_durations.items():
            ratio = (dur / total_speech_sec) if total_speech_sec > 0 else 0.0
            if dur < self.min_speaker_sec or ratio < self.min_speaker_ratio:
                small_speakers.add(spk)

        # Eğer tüm konuşmacılar küçük çıkarsa, en uzun konuşan konuşmacıyı büyük (major) kabul et
        if len(small_speakers) == len(unique_speakers):
            top_spk = max(spk_durations.keys(), key=lambda k: spk_durations[k])
            small_speakers.remove(top_spk)

        # num_speakers kısıtı: Kalan küme sayısı num_speakers'ın altına düşemez
        if num_speakers is not None and num_speakers > 0:
            allowed_removals = len(unique_speakers) - num_speakers
            if len(small_speakers) > allowed_removals:
                # Süresi en az olandan başlayarak yalnızca izin verilen kadarını küçük say
                sorted_small = sorted(list(small_speakers), key=lambda k: spk_durations[k])
                small_speakers = set(sorted_small[:allowed_removals])

        if not small_speakers:
            return segments

        major_speakers = [s for s in unique_speakers if s not in small_speakers]
        major_segments = [s for s in segments if s.speaker_id in major_speakers]

        logger.info(
            "Küçük konuşmacılar birleştiriliyor: %s -> Büyük konuşmacılar: %s",
            list(small_speakers),
            major_speakers,
        )

        # 3. Yeniden atama (Embedding vs Zamansal Yakınlık)
        spk_target_map: dict[str, str] = {}
        if speaker_embeddings:
            for small_spk in small_speakers:
                if small_spk in speaker_embeddings:
                    s_emb = speaker_embeddings[small_spk]
                    s_norm = s_emb / (np.linalg.norm(s_emb) + 1e-12)
                    best_maj = None
                    best_sim = -1.0
                    for maj_spk in major_speakers:
                        if maj_spk in speaker_embeddings:
                            m_emb = speaker_embeddings[maj_spk]
                            m_norm = m_emb / (np.linalg.norm(m_emb) + 1e-12)
                            sim = float(np.dot(s_norm, m_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_maj = maj_spk
                    if best_maj:
                        spk_target_map[small_spk] = best_maj

        new_segments: list[DiarizationSegment] = []
        for s in segments:
            if s.speaker_id in small_speakers:
                target_spk = spk_target_map.get(s.speaker_id)
                if not target_spk:
                    target_spk = self._find_nearest_major_speaker(s, major_segments)
                new_segments.append(
                    DiarizationSegment(
                        speaker_id=target_spk,
                        start_time=s.start_time,
                        end_time=s.end_time,
                    )
                )
            else:
                new_segments.append(s)

        # Zamana göre sırala ve bitişik segmentleri birleştir
        new_segments.sort(key=lambda s: s.start_time)
        return self._merge_adjacent_segments(new_segments)

    def _find_nearest_major_speaker(
        self, seg: DiarizationSegment, major_segments: list[DiarizationSegment]
    ) -> str:
        """Zamanca en yakın komşu büyük segmentin konuşmacısını döner."""
        if not major_segments:
            return seg.speaker_id

        best_spk = major_segments[0].speaker_id
        min_dist = float("inf")

        for m in major_segments:
            # Örtüşme (overlap) durumu: mesafe 0
            if max(seg.start_time, m.start_time) < min(seg.end_time, m.end_time):
                return m.speaker_id

            if seg.end_time <= m.start_time:
                dist = m.start_time - seg.end_time
            else:
                dist = seg.start_time - m.end_time

            if dist < min_dist:
                min_dist = dist
                best_spk = m.speaker_id

        return best_spk

    def _merge_adjacent_segments(
        self, segments: list[DiarizationSegment]
    ) -> list[DiarizationSegment]:
        """Aynı konuşmacıya ait bitişik segmentleri tek parçada toplar."""
        if len(segments) <= 1:
            return segments

        merged: list[DiarizationSegment] = [segments[0]]
        for curr in segments[1:]:
            prev = merged[-1]
            if curr.speaker_id == prev.speaker_id and curr.start_time <= prev.end_time + 0.05:
                merged[-1] = DiarizationSegment(
                    speaker_id=prev.speaker_id,
                    start_time=prev.start_time,
                    end_time=max(prev.end_time, curr.end_time),
                )
            else:
                merged.append(curr)

        return merged
