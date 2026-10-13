import unittest
from audio_analyzer.domain.models import DiarizationSegment, WordSegment
from audio_analyzer.services.speech_gap_detector import (
    detect_untranscribed_speech,
    merge_intervals,
)
from audio_analyzer.config import Settings


class TestSpeechGapDetector(unittest.TestCase):
    def test_merge_intervals(self):
        intervals = [(1.0, 5.0), (3.0, 7.0), (10.0, 15.0), (14.0, 18.0)]
        merged = merge_intervals(intervals)
        self.assertEqual(merged, [(1.0, 7.0), (10.0, 18.0)])

    def test_detect_gap_greater_than_2s(self):
        diar_segs = [
            DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=30.0)
        ]
        # Kelimeler sadece 0.5 - 2.0 ve 28.0 - 29.5 aralığında (ortada 26 sn boşluk var)
        words = [
            WordSegment(word="merhaba", start_time=0.5, end_time=1.5, probability=0.95),
            WordSegment(word="görüşürüz", start_time=28.0, end_time=29.0, probability=0.95),
        ]
        gaps = detect_untranscribed_speech(diar_segs, words, min_gap_sec=2.0)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start_time, 1.5, places=1)
        self.assertAlmostEqual(gaps[0].end_time, 28.0, places=1)
        self.assertAlmostEqual(gaps[0].duration, 26.5, places=1)

    def test_ignore_gaps_less_than_2s(self):
        diar_segs = [
            DiarizationSegment(speaker_id="SPEAKER_00", start_time=1.0, end_time=10.0)
        ]
        # Boşluklar 1.2 sn (< 2.0 sn)
        words = [
            WordSegment(word="bir", start_time=1.5, end_time=2.5, probability=0.9),
            WordSegment(word="iki", start_time=3.5, end_time=4.5, probability=0.9),
            WordSegment(word="üç", start_time=5.5, end_time=6.5, probability=0.9),
            WordSegment(word="dört", start_time=7.5, end_time=8.5, probability=0.9),
            WordSegment(word="beş", start_time=9.0, end_time=9.8, probability=0.9),
        ]
        gaps = detect_untranscribed_speech(diar_segs, words, min_gap_sec=2.0)
        self.assertEqual(len(gaps), 0)

    def test_entire_segment_missing_words(self):
        diar_segs = [
            DiarizationSegment(speaker_id="SPEAKER_01", start_time=30.0, end_time=60.0)
        ]
        words = []
        gaps = detect_untranscribed_speech(diar_segs, words, min_gap_sec=2.0)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_time, 30.0)
        self.assertEqual(gaps[0].end_time, 60.0)
        self.assertEqual(gaps[0].duration, 30.0)

    def test_whisper_config_defaults(self):
        settings = Settings()
        self.assertEqual(settings.whisper_log_prob_threshold, -1.0)
        self.assertEqual(settings.whisper_no_speech_threshold, 0.6)
        self.assertTrue(settings.whisper_temperature_fallback)
        self.assertTrue(settings.enable_stt_safety_net)


if __name__ == "__main__":
    unittest.main()
