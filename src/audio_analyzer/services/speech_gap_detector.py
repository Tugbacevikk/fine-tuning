import logging
from typing import Sequence

from audio_analyzer.domain.models import DiarizationSegment, UntranscribedSpeechSegment, WordSegment

logger = logging.getLogger(__name__)


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Kesişen veya bitişik zaman aralıklarını tek bir birleşik aralık listesinde birleştirir.
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = [list(sorted_intervals[0])]

    for cur_start, cur_end in sorted_intervals[1:]:
        prev = merged[-1]
        if cur_start <= prev[1]:
            prev[1] = max(prev[1], cur_end)
        else:
            merged.append([cur_start, cur_end])

    return [(m[0], m[1]) for m in merged]


def calculate_speech_coverage(
    diarization_segments: Sequence[DiarizationSegment],
    words: Sequence[WordSegment],
    min_safety_gap_sec: float = 2.0,
) -> tuple[list[UntranscribedSpeechSegment], dict]:
    """
    Diarization konuşma aralıkları ile Whisper kelimelerini karşılaştırır:
    (a) 2 sn eşiği olmaksızın kelime içermeyen tüm konuşma boşluklarının toplam süresi,
    (b) Toplam konuşma süresi ve kapsama yüzdesi,
    (c) 2 sn üzeri güvenlik ağı için tetiklenecek aralıklar (UntranscribedSpeechSegment).
    """
    if not diarization_segments:
        return [], {
            "total_speech_seconds": 0.0,
            "all_untranscribed_seconds": 0.0,
            "speech_coverage_percentage": 100.0,
            "untranscribed_speech_seconds": 0.0,
            "untranscribed_speech_count": 0,
        }

    raw_intervals = [(s.start_time, s.end_time) for s in diarization_segments if s.end_time > s.start_time]
    merged_speech_intervals = merge_intervals(raw_intervals)

    total_speech_seconds = sum(end - start for start, end in merged_speech_intervals)
    sorted_words = sorted(words, key=lambda w: w.start_time)

    all_untranscribed_seconds = 0.0
    safety_gaps: list[UntranscribedSpeechSegment] = []

    for s_start, s_end in merged_speech_intervals:
        speech_dur = s_end - s_start
        overlapping_words = [
            w for w in sorted_words if w.end_time > s_start and w.start_time < s_end
        ]

        if not overlapping_words:
            all_untranscribed_seconds += speech_dur
            if speech_dur >= min_safety_gap_sec:
                safety_gaps.append(
                    UntranscribedSpeechSegment(
                        start_time=round(s_start, 2),
                        end_time=round(s_end, 2),
                        duration=round(speech_dur, 2),
                    )
                )
            continue

        first_w = overlapping_words[0]
        init_gap = first_w.start_time - s_start
        if init_gap > 0:
            all_untranscribed_seconds += init_gap
            if init_gap >= min_safety_gap_sec:
                safety_gaps.append(
                    UntranscribedSpeechSegment(
                        start_time=round(s_start, 2),
                        end_time=round(first_w.start_time, 2),
                        duration=round(init_gap, 2),
                    )
                )

        current_cov_end = max(s_start, first_w.end_time)
        for w in overlapping_words[1:]:
            if w.start_time > current_cov_end:
                gap = w.start_time - current_cov_end
                all_untranscribed_seconds += gap
                if gap >= min_safety_gap_sec:
                    safety_gaps.append(
                        UntranscribedSpeechSegment(
                            start_time=round(current_cov_end, 2),
                            end_time=round(w.start_time, 2),
                            duration=round(gap, 2),
                        )
                    )
            current_cov_end = max(current_cov_end, w.end_time)

        trail_gap = s_end - current_cov_end
        if trail_gap > 0:
            all_untranscribed_seconds += trail_gap
            if trail_gap >= min_safety_gap_sec:
                safety_gaps.append(
                    UntranscribedSpeechSegment(
                        start_time=round(current_cov_end, 2),
                        end_time=round(s_end, 2),
                        duration=round(trail_gap, 2),
                    )
                )

    coverage_pct = 100.0
    if total_speech_seconds > 0:
        coverage_pct = max(
            0.0,
            (total_speech_seconds - all_untranscribed_seconds) / total_speech_seconds * 100.0,
        )

    stats = {
        "total_speech_seconds": round(total_speech_seconds, 2),
        "all_untranscribed_seconds": round(all_untranscribed_seconds, 2),
        "speech_coverage_percentage": round(coverage_pct, 2),
        "untranscribed_speech_seconds": round(sum(g.duration for g in safety_gaps), 2),
        "untranscribed_speech_count": len(safety_gaps),
    }

    return safety_gaps, stats


def detect_untranscribed_speech(
    diarization_segments: Sequence[DiarizationSegment],
    words: Sequence[WordSegment],
    min_gap_sec: float = 2.0,
) -> list[UntranscribedSpeechSegment]:
    """
    Geriye dönük uyumluluk: min_gap_sec üzerindeki aralıkları döndürür.
    """
    gaps, _ = calculate_speech_coverage(diarization_segments, words, min_safety_gap_sec=min_gap_sec)
    return gaps
