import uuid

from audio_analyzer.domain.models import (
    DiarizationSegment,
    TranscriptUtterance,
    WordSegment,
)


class FusionEngine:
    """
    Speech-to-Text kelime zaman damgaları ile Speaker Diarization zaman aralıklarını
    birleştiren ve çakışma/sessizlik kurallarını uygulayan hizalama motoru.
    """

    def __init__(self, max_silence_threshold: float = 1.5):
        """
        :param max_silence_threshold: Aynı konuşmacı bu süreden (saniye) fazla susarsa yeni paragraf başlatır.
        """
        self.max_silence_threshold = max_silence_threshold

    def align(
        self, words: list[WordSegment], diarization_segments: list[DiarizationSegment]
    ) -> list[TranscriptUtterance]:
        """
        Kelime seviyesindeki STT çıktılarını konuşmacı aralıklarıyla IoU ve Midpoint kurallarına göre eşler.
        """
        if not words:
            return []

        if not diarization_segments:
            # Konuşmacı bilgisi yoksa tek bir varsayılan konuşmacı ("SPEAKER_UNKNOWN") atar.
            return self._build_single_speaker_utterances(words, "SPEAKER_UNKNOWN")

        # 1. Her kelimeye en uygun konuşmacıyı atama (Midpoint + IoU)
        attributed_words = []
        for word in words:
            best_speaker = self._find_best_speaker_for_word(word, diarization_segments)
            attributed_words.append((word, best_speaker))

        # 1b. Gürültü ve anlık sıçramaları önlemek için 1-kelimelik spikeleri yumuşatma (Smoothing Pass)
        attributed_words = self._smooth_attributed_words(attributed_words)

        # 2. Kelimeleri Konuşmacı ve Sessizlik eşiğine göre gruplayarak Utterance blokları oluşturma
        raw_utterances = self._group_words_into_utterances(attributed_words)

        # 2b. Kısa parça bağlama (<1.0s, gap<0.3s, noktalama yok -> önceki konuşmacı)
        bound_utterances = self._bind_short_unpunctuated_turns(raw_utterances)

        # 3. Konuşmacıları ilk görünme (kronolojik ses) sırasına göre SPEAKER_00, SPEAKER_01... olarak sıralama
        return self._normalize_speaker_ids(bound_utterances)

    def _find_best_speaker_for_word(
        self, word: WordSegment, diarization_segments: list[DiarizationSegment]
    ) -> str:
        # 1. Kelimenin orta noktası hangi segmente düşüyorsa öncelikli olarak o konuşmacıyı ata
        for seg in diarization_segments:
            if seg.contains_timestamp(word.midpoint):
                return seg.speaker_id

        # 2. Orta nokta eşleşmezse, en yüksek kesişim süresine (max overlap duration) sahip segmente ata
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0.0

        for seg in diarization_segments:
            overlap_start = max(word.start_time, seg.start_time)
            overlap_end = min(word.end_time, seg.end_time)
            overlap_len = max(0.0, overlap_end - overlap_start)

            if overlap_len > max_overlap:
                max_overlap = overlap_len
                best_speaker = seg.speaker_id

        return best_speaker

    def _group_words_into_utterances(
        self, attributed_words: list[tuple[WordSegment, str]]
    ) -> list[TranscriptUtterance]:
        utterances: list[TranscriptUtterance] = []
        if not attributed_words:
            return utterances

        current_speaker = attributed_words[0][1]
        current_words: list[WordSegment] = [attributed_words[0][0]]

        for word, speaker in attributed_words[1:]:
            last_word = current_words[-1]
            silence_gap = word.start_time - last_word.end_time

            # Bölümleme kriterleri: Konuşmacı değişimi VEYA 1.5s üzerindeki sessizlik
            if speaker != current_speaker or silence_gap > self.max_silence_threshold:
                # Mevcut bloğu sonlandırıp listeye ekle
                utt = self._create_utterance(current_speaker, current_words)
                if utt.text.strip():
                    utterances.append(utt)
                current_speaker = speaker
                current_words = [word]
            else:
                current_words.append(word)

        # Son kalan bloğu ekle
        if current_words:
            utt = self._create_utterance(current_speaker, current_words)
            if utt.text.strip():
                utterances.append(utt)

        return utterances

    def _create_utterance(self, speaker_id: str, words: list[WordSegment]) -> TranscriptUtterance:
        start_time = words[0].start_time
        end_time = words[-1].end_time
        text = " ".join(w.word.strip() for w in words if w.word and w.word.strip()).strip()
        return TranscriptUtterance(
            id=uuid.uuid4(),
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            text=text,
        )

    def _build_single_speaker_utterances(
        self, words: list[WordSegment], speaker_id: str
    ) -> list[TranscriptUtterance]:
        attributed = [(w, speaker_id) for w in words]
        return self._group_words_into_utterances(attributed)

    def _smooth_attributed_words(
        self, attributed_words: list[tuple[WordSegment, str]]
    ) -> list[tuple[WordSegment, str]]:
        if len(attributed_words) < 3:
            return attributed_words

        smoothed = list(attributed_words)
        for i in range(1, len(attributed_words) - 1):
            prev_spk = smoothed[i - 1][1]
            next_spk = smoothed[i + 1][1]
            curr_spk = smoothed[i][1]

            # Eğer sağındaki ve solundaki konuşmacı aynı ise ancak ortadaki tek kelime farklıysa ortadakini düzelt
            if prev_spk == next_spk and curr_spk != prev_spk:
                smoothed[i] = (smoothed[i][0], prev_spk)

        return smoothed

    def _normalize_speaker_ids(
        self, utterances: list[TranscriptUtterance]
    ) -> list[TranscriptUtterance]:
        """
        Gözlemlenme (kronolojik ses sırası) önceliğine göre konuşmacı isimlerini
        düzenli biçimde SPEAKER_00, SPEAKER_01, SPEAKER_02... olarak yeniden sıralar.
        """
        speaker_map = {}
        counter = 0

        for utt in utterances:
            if utt.speaker_id == "SPEAKER_UNKNOWN":
                continue
            if utt.speaker_id not in speaker_map:
                speaker_map[utt.speaker_id] = f"SPEAKER_{counter:02d}"
                counter += 1
            utt.speaker_id = speaker_map[utt.speaker_id]

        return utterances

    def _bind_short_unpunctuated_turns(
        self, utterances: list[TranscriptUtterance]
    ) -> list[TranscriptUtterance]:
        """
        Kısa parça bağlama (Step 3):
        Süresi 1.0 s'den kısa bir utterance, önceki utterance noktalama ile bitmiyorsa
        ve aralarındaki boşluk 0.3 s'den kısaysa önceki konuşmacıya bağlanır.
        Metin değişmez, sadece etiket (speaker_id) güncellenir.
        """
        if len(utterances) <= 1:
            return utterances

        result: list[TranscriptUtterance] = [utterances[0]]
        for i in range(1, len(utterances)):
            prev_utt = result[-1]
            curr_utt = utterances[i]

            dur = curr_utt.end_time - curr_utt.start_time
            gap = curr_utt.start_time - prev_utt.end_time
            prev_text = prev_utt.text.strip()
            prev_ends_with_punct = prev_text.endswith((".", "?", "!", ":", ";", "…", "..."))

            if dur < 1.0 and gap < 0.3 and not prev_ends_with_punct:
                curr_utt = TranscriptUtterance(
                    id=curr_utt.id,
                    speaker_id=prev_utt.speaker_id,
                    start_time=curr_utt.start_time,
                    end_time=curr_utt.end_time,
                    text=curr_utt.text,
                )

            result.append(curr_utt)

        return result
