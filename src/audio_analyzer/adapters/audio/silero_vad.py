import logging

import numpy as np
import soundfile as sf

from audio_analyzer.domain.interfaces import IVADProcessor

logger = logging.getLogger(__name__)


class SileroVADProcessor(IVADProcessor):
    """
    Silero VAD (Voice Activity Detection) Motoru.
    Cümle aralarındaki ve nefes duraklamalarındaki kesin sessizlik noktalarını tespit ederek
    konuşmanın asla cümle ortasında bölünmemesini garanti eder.
    Rust (PyO3) FFI entegrasyonu ve Python native fallback ile tam uyumludur.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._get_speech_timestamps_fn = None

    def _load_model(self):
        if self._model is None:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                self._model = load_silero_vad()
                self._get_speech_timestamps_fn = get_speech_timestamps
            except Exception as e:
                logger.warning(
                    "Silero VAD model yükleme uyarısı: %s. Energy VAD kullanılıyor.", e
                )
                self._model = "ENERGY_FALLBACK"
                self._get_speech_timestamps_fn = None

    def get_speech_timestamps(
        self, audio_input: str | np.ndarray, min_silence_duration_ms: int = 400
    ) -> list[tuple[float, float]]:
        """
        Ses verisini (dosya yolu veya RAM tamponu) analiz eder ve konuşma aralıklarını döner.
        """
        if isinstance(audio_input, np.ndarray):
            data = audio_input
            sr = 16000
        else:
            data, sr = sf.read(audio_input)

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != 16000:
            from math import gcd

            import scipy.signal

            g = gcd(int(sr), 16000)
            data = scipy.signal.resample_poly(data, 16000 // g, int(sr) // g)
            sr = 16000

        self._load_model()

        if self._model != "ENERGY_FALLBACK" and self._get_speech_timestamps_fn is not None:
            try:
                import torch

                wav_tensor = torch.tensor(data, dtype=torch.float32)
                timestamps = self._get_speech_timestamps_fn(
                    wav_tensor,
                    self._model,
                    threshold=self.threshold,
                    sampling_rate=16000,
                    min_silence_duration_ms=min_silence_duration_ms,
                )
                return [
                    (round(ts["start"] / 16000, 2), round(ts["end"] / 16000, 2))
                    for ts in timestamps
                ]
            except Exception as ex:
                logger.warning("Silero VAD execution fallback: %s", ex)

        # Fallback: High-precision energy-based silence detection (Rust DSP compatible)
        return self._energy_vad_chunking(data, sr, min_silence_duration_ms)

    def _energy_vad_chunking(
        self, data: np.ndarray, sr: int, min_silence_ms: int
    ) -> list[tuple[float, float]]:
        frame_ms = 30
        frame_samples = int(sr * frame_ms / 1000)
        num_frames = len(data) // frame_samples

        if num_frames == 0:
            return [(0.0, round(len(data) / sr, 2))]

        energies = [
            np.sqrt(np.mean(data[i * frame_samples : (i + 1) * frame_samples] ** 2))
            for i in range(num_frames)
        ]

        thresh = max(1e-4, np.percentile(energies, 30) * 1.5)
        is_speech = [e > thresh for e in energies]

        min_silence_frames = max(1, int(min_silence_ms / frame_ms))
        speech_chunks: list[tuple[float, float]] = []

        in_speech = False
        start_frame = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    in_speech = True
                    start_frame = i
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        in_speech = False
                        end_frame = i - silence_count
                        start_sec = round(start_frame * frame_ms / 1000, 2)
                        end_sec = round(end_frame * frame_ms / 1000, 2)
                        if end_sec > start_sec:
                            speech_chunks.append((start_sec, end_sec))

        if in_speech:
            start_sec = round(start_frame * frame_ms / 1000, 2)
            end_sec = round(len(data) / sr, 2)
            speech_chunks.append((start_sec, end_sec))

        return speech_chunks if speech_chunks else [(0.0, round(len(data) / sr, 2))]
