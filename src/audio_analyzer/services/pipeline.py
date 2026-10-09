import concurrent.futures
import io
import logging
import os
import time
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from audio_analyzer.api.metrics import STAGE_DURATION_HISTOGRAM
from audio_analyzer.config import get_settings
from audio_analyzer.domain.interfaces import (
    IAudioDenoiser,
    IAudioProcessor,
    IDiarizer,
    ISTTEngine,
    IVADProcessor,
)
from audio_analyzer.domain.models import OverlapSummary, TranscriptUtterance
from audio_analyzer.services.fusion_engine import FusionEngine
from audio_analyzer.services.overlap_detector import OverlapDetector
from audio_analyzer.services.semantic_refiner import SemanticRefiner

logger = logging.getLogger(__name__)

DEFAULT_UNBOUNDED_AUDIO_DURATION: float = 1000.0


def estimate_snr_db(audio_array: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Ses sinyalinin SNR (Signal-to-Noise Ratio) değerini dB cinsinden tahmin eder.
    Frame bazlı enerji dağılımını kullanarak sinyal ve gürültü seviyelerini karşılaştırır.
    """
    if audio_array is None or len(audio_array) == 0:
        return 0.0

    frame_size = int(0.02 * sample_rate)  # 20ms frame
    if len(audio_array) < frame_size:
        return 30.0

    num_frames = len(audio_array) // frame_size
    frames = audio_array[: num_frames * frame_size].reshape(num_frames, frame_size)
    energies = np.mean(frames ** 2, axis=1)

    top_k = max(1, int(0.3 * num_frames))
    signal_energy = float(np.mean(np.partition(energies, -top_k)[-top_k:]))

    low_k = max(1, int(0.2 * num_frames))
    noise_energy = float(np.mean(np.partition(energies, low_k - 1)[:low_k]))

    if noise_energy <= 1e-10:
        return 40.0

    snr_linear = signal_energy / noise_energy
    return float(10.0 * np.log10(max(snr_linear, 1e-5)))


class AudioAnalysisPipeline:
    """
    Ses Analizi Ana Pipeline Orkestratörü.
    Gelen ses dosyasını gürültüden arındırır (Denoiser), normalize eder,
    VAD, STT ve Diarization motorlarını çalıştırır, çakışmaları (Overlap Detection)
    tespit eder ve çıktıları FusionEngine ile birleştirir.
    """

    def __init__(
        self,
        stt_engine: ISTTEngine,
        diarizer: IDiarizer,
        audio_processor: IAudioProcessor | None = None,
        vad_processor: IVADProcessor | None = None,
        denoiser: IAudioDenoiser | None = None,
        fusion_engine: FusionEngine | None = None,
        semantic_refiner: SemanticRefiner | None = None,
    ):
        self.stt_engine = stt_engine
        self.diarizer = diarizer
        self.audio_processor = audio_processor
        self.vad_processor = vad_processor
        self.denoiser = denoiser
        self.fusion_engine = fusion_engine or FusionEngine()
        self.semantic_refiner = semantic_refiner or SemanticRefiner()

    def process(
        self, audio_path: str, num_speakers: int | None = None
    ) -> tuple[list[TranscriptUtterance], str | None, OverlapSummary]:
        """
        Ses dosyasını (MP3/WAV/FLAC vb.) analiz eder.
        Returns: (List[TranscriptUtterance], detected_language, overlap_summary)
        """
        settings = get_settings()
        profile = settings.pipeline_profile
        min_snr_db = settings.pipeline_min_snr_db

        working_path = audio_path
        created_temp_files: list[str] = []

        try:
            # 1. Ön Gürültü Temizleme (DeepFilterNet Denoiser)
            should_denoise = False
            denoiser_active = self.denoiser and self.denoiser.enabled and settings.enable_denoiser
            if denoiser_active:
                t0 = time.monotonic()
                if profile == "feedback":
                    try:
                        audio_data, sr = sf.read(audio_path)
                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1)
                        snr_db = estimate_snr_db(audio_data, sr)
                        if snr_db < min_snr_db:
                            should_denoise = True
                            logger.info("SNR düşük (%.1f dB < %.1f dB), denoiser çalıştırılıyor.", snr_db, min_snr_db)
                        else:
                            logger.info("SNR yeterli (%.1f dB >= %.1f dB), denoiser atlanıyor.", snr_db, min_snr_db)
                    except Exception as snr_err:
                        logger.warning("SNR ölçülemedi: %s. Denoiser varsayılan çalıştırılıyor.", snr_err)
                        should_denoise = True
                else:
                    should_denoise = True
                STAGE_DURATION_HISTOGRAM.labels(stage="denoise_check").observe(time.monotonic() - t0)

            import tempfile
            import uuid

            if should_denoise and denoiser_active:
                t0 = time.monotonic()
                temp_name = f"{Path(audio_path).stem}_{uuid.uuid4().hex[:8]}.denoised.wav"
                denoised_wav_path = os.path.join(tempfile.gettempdir(), temp_name)
                audio_after_denoise = self.denoiser.denoise(
                    input_path=audio_path, output_path=denoised_wav_path
                )
                if audio_after_denoise != audio_path:
                    created_temp_files.append(audio_after_denoise)
                    working_path = audio_after_denoise
                STAGE_DURATION_HISTOGRAM.labels(stage="denoise").observe(time.monotonic() - t0)

            # 2. Ses Ön İşleme & Normalizasyon (16kHz Mono WAV Dönüşümü)
            if self.audio_processor:
                t0 = time.monotonic()
                temp_name = f"{Path(working_path).stem}_{uuid.uuid4().hex[:8]}.processed.wav"
                processed_wav_path = os.path.join(tempfile.gettempdir(), temp_name)
                working_path = self.audio_processor.normalize_and_resample(
                    input_path=working_path,
                    output_path=processed_wav_path,
                    target_sample_rate=16000,
                )
                if working_path != audio_path and working_path not in created_temp_files:
                    created_temp_files.append(working_path)
                STAGE_DURATION_HISTOGRAM.labels(stage="normalize").observe(time.monotonic() - t0)

            # Pre-calculate audio duration directly from audio file before STT
            try:
                audio_duration = float(sf.info(working_path).duration)
            except Exception:
                audio_duration = 0.0

            return self._execute_pipeline(working_path, audio_duration, num_speakers=num_speakers)
        finally:
            for tmp_file in created_temp_files:
                if Path(tmp_file).exists():
                    try:
                        Path(tmp_file).unlink()
                    except Exception as cleanup_err:
                        logger.warning("Geçici ses dosyası temizleme uyarısı: %s", cleanup_err)

    def process_bytes(
        self, file_bytes: bytes, num_speakers: int | None = None
    ) -> tuple[list[TranscriptUtterance], str | None, OverlapSummary]:
        """
        0-Disk I/O: Ses dosyasını doğrudan RAM bellek üzerinden işler.
        """
        settings = get_settings()
        profile = settings.pipeline_profile
        min_snr_db = settings.pipeline_min_snr_db

        # 1. Ses Baytlarını RAM'de 16kHz Mono float32 NumPy dizisine çevir
        if self.audio_processor and hasattr(self.audio_processor, "convert_bytes_to_ndarray"):
            audio_array, _ = self.audio_processor.convert_bytes_to_ndarray(
                file_bytes, target_sample_rate=16000
            )
        else:
            audio_array, sr = sf.read(io.BytesIO(file_bytes))
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            if sr != 16000:
                g = gcd(int(sr), 16000)
                audio_array = scipy.signal.resample_poly(audio_array, 16000 // g, int(sr) // g)

        # 2. Ön Gürültü Temizleme (RAM Üzerinde Denoise)
        should_denoise = False
        denoiser_active = self.denoiser and self.denoiser.enabled and settings.enable_denoiser
        if denoiser_active:
            t0 = time.monotonic()
            if profile == "feedback":
                snr_db = estimate_snr_db(audio_array, 16000)
                if snr_db < min_snr_db:
                    should_denoise = True
                    logger.info("SNR düşük (%.1f dB < %.1f dB), RAM denoiser çalıştırılıyor.", snr_db, min_snr_db)
                else:
                    logger.info("SNR yeterli (%.1f dB >= %.1f dB), RAM denoiser atlanıyor.", snr_db, min_snr_db)
            else:
                should_denoise = True
            STAGE_DURATION_HISTOGRAM.labels(stage="denoise_check").observe(time.monotonic() - t0)

        if should_denoise and denoiser_active and hasattr(self.denoiser, "denoise_array"):
            t0 = time.monotonic()
            audio_array = self.denoiser.denoise_array(audio_array, sample_rate=16000)
            STAGE_DURATION_HISTOGRAM.labels(stage="denoise").observe(time.monotonic() - t0)

        # Pre-calculate audio duration directly from array before STT
        audio_duration = float(len(audio_array)) / 16000.0 if len(audio_array) > 0 else 0.0

        return self._execute_pipeline(audio_array, audio_duration, num_speakers=num_speakers)

    def _execute_pipeline(
        self, audio_input: str | np.ndarray, audio_duration: float, num_speakers: int | None = None
    ) -> tuple[list[TranscriptUtterance], str | None, OverlapSummary]:
        """
        Ortak pipeline yürütme motoru. Hem dosya yolu hem NumPy dizisi ile çalışır.
        STT öncesinde hesaplanan ses süresi ve konuşmacı sayısı ile diarization kararını verir.
        """
        from audio_analyzer.domain.models import DiarizationSegment

        settings = get_settings()
        profile = settings.pipeline_profile
        min_diarize_sec = settings.pipeline_min_diarize_sec
        domain_mode = settings.domain_mode
        vad_engine = settings.vad_engine
        run_sequentially = settings.run_pipeline_sequentially

        # 3. VAD İşleme
        speech_timestamps: list[tuple[float, float]] | None = None
        stt_has_vad = getattr(self.stt_engine, "has_vad_filter", True)
        enable_vad_filter = (
            os.getenv("ENABLE_VAD_FILTER", os.getenv("WHISPER_VAD_FILTER", "false")).lower() == "true"
        )
        if enable_vad_filter and self.vad_processor and (vad_engine == "silero" or not stt_has_vad):
            t0 = time.monotonic()
            try:
                speech_timestamps = self.vad_processor.get_speech_timestamps(audio_input)
            except Exception as e:
                logger.warning("VAD İşleme Hatası: %s. VAD filtresi atlanıyor.", e)
            STAGE_DURATION_HISTOGRAM.labels(stage="vad").observe(time.monotonic() - t0)

        # Diarization Kararı (STT ÖNCESİNDE KONTROL)
        should_diarize = True
        is_single_speaker = num_speakers == 1

        if is_single_speaker:
            logger.info("num_speakers=1 belirtildi: Diarization motoru by-pass ediliyor (Tek konuşmacı kipleri).")
            should_diarize = False
        elif profile == "feedback" and audio_duration < min_diarize_sec:
            logger.info("Ses süresi (%.1f sn < %.1f sn), diarization atlanıyor.", audio_duration, min_diarize_sec)
            should_diarize = False

        # 4 & 5. STT ve Diarization Motorlarını Çalıştır
        if run_sequentially:
            t0 = time.monotonic()
            words, detected_language = self.stt_engine.transcribe(audio_input)
            STAGE_DURATION_HISTOGRAM.labels(stage="stt").observe(time.monotonic() - t0)

            if should_diarize:
                t0 = time.monotonic()
                diarization_segments = self._call_diarizer(audio_input, num_speakers=num_speakers)
                STAGE_DURATION_HISTOGRAM.labels(stage="diarization").observe(time.monotonic() - t0)
            elif is_single_speaker:
                diarization_segments = self._create_single_speaker_segments(audio_duration)
            else:
                diarization_segments = []
        else:
            total_worker_threads = settings.worker_cpu_threads
            stt_threads = max(1, total_worker_threads // 2)
            diar_threads = max(1, total_worker_threads - stt_threads)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                t0_stt = time.monotonic()
                future_stt = executor.submit(self.stt_engine.transcribe, audio_input)

                future_diar = None
                if should_diarize:
                    future_diar = executor.submit(self._call_diarizer, audio_input, num_speakers)

                words, detected_language = future_stt.result()
                STAGE_DURATION_HISTOGRAM.labels(stage="stt").observe(time.monotonic() - t0_stt)

                if future_diar:
                    t0_diar = time.monotonic()
                    diarization_segments = future_diar.result()
                    STAGE_DURATION_HISTOGRAM.labels(stage="diarization").observe(time.monotonic() - t0_diar)
                elif is_single_speaker:
                    diarization_segments = self._create_single_speaker_segments(audio_duration)
                else:
                    diarization_segments = []

        # 4b. VAD Filtrelemesi
        if enable_vad_filter and speech_timestamps and words:
            words = self._filter_words_with_vad(words, speech_timestamps)

        # 6. Overlap Detection
        t0 = time.monotonic()
        total_duration = audio_duration
        if total_duration <= 0.0:
            if words:
                total_duration = max(w.end_time for w in words)
            elif diarization_segments:
                total_duration = max(s.end_time for s in diarization_segments)

        overlap_summary = OverlapDetector.detect_overlaps(
            diarization_segments=diarization_segments, total_audio_duration=total_duration
        )
        STAGE_DURATION_HISTOGRAM.labels(stage="overlap_detection").observe(time.monotonic() - t0)

        # 7. FusionEngine
        t0 = time.monotonic()
        raw_utterances = self.fusion_engine.align(words, diarization_segments)
        STAGE_DURATION_HISTOGRAM.labels(stage="fusion").observe(time.monotonic() - t0)

        # 8. SemanticRefiner (Her profilde anlamsal düzeltme ve imla kurallarını çalıştırır)
        t0 = time.monotonic()
        final_utterances = self.semantic_refiner.refine(raw_utterances)
        STAGE_DURATION_HISTOGRAM.labels(stage="semantic_refine").observe(time.monotonic() - t0)

        return final_utterances, detected_language, overlap_summary

    def _filter_words_with_vad(
        self, words: list, speech_timestamps: list[tuple[float, float]], tolerance: float = 0.6
    ) -> list:
        """
        VAD konuşma aralıklarının tamamen dışında kalan (sessizlikte türetilmiş)
        STT kelime halüsinasyonlarını eler.
        """
        if not speech_timestamps:
            return words

        filtered_words = []
        for word in words:
            is_speech = any(
                (start - tolerance) <= word.midpoint <= (end + tolerance)
                for start, end in speech_timestamps
            )
            if is_speech:
                filtered_words.append(word)

        return filtered_words

    def _call_diarizer(self, audio_input: str | np.ndarray, num_speakers: int | None = None) -> list:
        """IDiarizer.diarize adaptörünü çağırır."""
        return self.diarizer.diarize(audio_input, num_speakers=num_speakers)

    def _create_single_speaker_segments(self, audio_duration: float) -> list:
        """Tek konuşmacılı sesler için SPEAKER_00 kapsama segmenti üretir."""
        from audio_analyzer.domain.models import DiarizationSegment
        end_t = audio_duration if audio_duration > 0 else DEFAULT_UNBOUNDED_AUDIO_DURATION
        return [DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=end_t)]
