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


def detect_untranscribed_speech(
    diarization_segments: Sequence[DiarizationSegment],
    words: Sequence[WordSegment],
    min_gap_sec: float = 2.0,
) -> list[UntranscribedSpeechSegment]:
    """
    Diarization motorunun konuşma tespit ettiği zaman aralıkları ile
    Whisper'ın ürettiği kelime zaman damgalarını karşılaştırır.
    İçinde hiç kelime bulunmayan ve süresi min_gap_sec (varsayılan 2.0 sn)
    üzerinde olan konuşma aralıklarını 'transkriptsiz konuşma' olarak belirler.
    """
    if not diarization_segments:
        return []

    # 1. Diarization konuşma aralıklarını çıkar ve kesişenleri birleştir
    raw_intervals = [(s.start_time, s.end_time) for s in diarization_segments if s.end_time > s.start_time]
    merged_speech_intervals = merge_intervals(raw_intervals)

    # 2. Kelimeleri başlangıç zamanına göre sırala
    sorted_words = sorted(words, key=lambda w: w.start_time)

    untranscribed: list[UntranscribedSpeechSegment] = []

    for s_start, s_end in merged_speech_intervals:
        speech_dur = s_end - s_start
        if speech_dur < min_gap_sec:
            continue

        # Bu konuşma aralığıyla kesişen kelimeleri bul
        overlapping_words = [
            w for w in sorted_words if w.end_time > s_start and w.start_time < s_end
        ]

        if not overlapping_words:
            # Konuşma aralığında HİÇ kelime yok
            untranscribed.append(
                UntranscribedSpeechSegment(
                    start_time=round(s_start, 2),
                    end_time=round(s_end, 2),
                    duration=round(speech_dur, 2),
                )
            )
            continue

        # Kelimeler arasındaki ve baş/son boşlukları tara
        # a) İlk kelimeden önceki konuşma boşluğu
        first_w = overlapping_words[0]
        if (first_w.start_time - s_start) >= min_gap_sec:
            untranscribed.append(
                UntranscribedSpeechSegment(
                    start_time=round(s_start, 2),
                    end_time=round(first_w.start_time, 2),
                    duration=round(first_w.start_time - s_start, 2),
                )
            )

        # b) Ardışık kelimeler arasındaki boşluklar
        current_cov_end = max(s_start, first_w.end_time)
        for w in overlapping_words[1:]:
            if w.start_time > current_cov_end:
                gap = w.start_time - current_cov_end
                if gap >= min_gap_sec:
                    untranscribed.append(
                        UntranscribedSpeechSegment(
                            start_time=round(current_cov_end, 2),
                            end_time=round(w.start_time, 2),
                            duration=round(gap, 2),
                        )
                    )
            current_cov_end = max(current_cov_end, w.end_time)

        # c) Son kelimeden sonraki konuşma boşluğu
        if (s_end - current_cov_end) >= min_gap_sec:
            untranscribed.append(
                UntranscribedSpeechSegment(
                    start_time=round(current_cov_end, 2),
                    end_time=round(s_end, 2),
                    duration=round(s_end - current_cov_end, 2),
                )
            )

    return untranscribed
