import unittest
from unittest.mock import MagicMock
import numpy as np

from audio_analyzer.domain.models import DiarizationSegment, UntranscribedSpeechSegment, WordSegment
from audio_analyzer.services.pipeline import AudioAnalysisPipeline
from audio_analyzer.services.speech_gap_detector import calculate_speech_coverage


class TestSTTSafetyNet(unittest.TestCase):
    def test_rescue_untranscribed_speech_recovers_words(self):
        mock_stt = MagicMock()
        mock_diarizer = MagicMock()

        pipeline = AudioAnalysisPipeline(stt_engine=mock_stt, diarizer=mock_diarizer)

        # Simüle edilen kurtarma STT çıktısı (0.0 - 5.0 sn aralığında bir dilim için)
        mock_stt.transcribe.return_value = (
            [
                WordSegment(word="kurtarılan", start_time=1.0, end_time=2.0),
                WordSegment(word="cümle", start_time=2.1, end_time=3.0),
            ],
            "tr",
        )

        dummy_audio = np.zeros(16000 * 30, dtype=np.float32)  # 30 sn ses
        initial_words = [
            WordSegment(word="başlangıç", start_time=0.5, end_time=1.0),
            WordSegment(word="bitiş", start_time=28.0, end_time=29.0),
        ]
        untranscribed_gaps = [
            UntranscribedSpeechSegment(start_time=10.0, end_time=15.0, duration=5.0)
        ]

        rescued_words = pipeline._rescue_untranscribed_speech(
            audio_input=dummy_audio,
            words=initial_words,
            untranscribed=untranscribed_gaps,
            total_duration=30.0,
        )

        # Kurtarılan kelimelerin zaman damgaları (slice_start = 10.0 - 0.5 = 9.5)
        # 9.5 + 1.0 = 10.5, 9.5 + 2.1 = 11.6
        self.assertEqual(len(rescued_words), 4)
        words_text = [w.word for w in rescued_words]
        self.assertIn("kurtarılan", words_text)
        self.assertIn("cümle", words_text)

        rescued_word = next(w for w in rescued_words if w.word == "kurtarılan")
        self.assertAlmostEqual(rescued_word.start_time, 10.5, places=1)

    def test_calculate_speech_coverage_percentage(self):
        diar_segs = [
            DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=20.0)
        ]
        # 20 sn konuşmanın 10 sn'sinde kelime var
        words = [
            WordSegment(word="bir", start_time=0.0, end_time=10.0),
        ]
        safety_gaps, stats = calculate_speech_coverage(diar_segs, words, min_safety_gap_sec=2.0)
        self.assertEqual(stats["total_speech_seconds"], 20.0)
        self.assertAlmostEqual(stats["all_untranscribed_seconds"], 10.0, places=1)
        self.assertAlmostEqual(stats["speech_coverage_percentage"], 50.0, places=1)
        self.assertEqual(len(safety_gaps), 1)
        self.assertEqual(safety_gaps[0].duration, 10.0)


if __name__ == "__main__":
    unittest.main()
