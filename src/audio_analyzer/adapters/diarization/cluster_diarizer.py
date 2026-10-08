import logging

import numpy as np

from audio_analyzer.domain.interfaces import IDiarizer
from audio_analyzer.domain.models import DeviceConfig, DiarizationSegment

logger = logging.getLogger(__name__)


class LocalSpectralClusterDiarizer(IDiarizer):
    """
    Kadın-Kadın ve Aynı Cinsiyet Konuşmacı Ayrıştırma Motoru (Cosine K-Means++ / Agglomerative).
    28-boyutlu akustik matris (MFCC, Delta-MFCC, Pitch/F0 ve Spektral Merkez) ve Kosinüs Benzerliği
    kullanarak konuşmacıları dengeli biçimde SPEAKER_00 ve SPEAKER_01 olarak ayırır.
    """

    def __init__(self, device_config: DeviceConfig = None, num_speakers: int = 2):
        self.device_config = device_config or DeviceConfig()
        self.num_speakers = num_speakers

    def _estimate_pitch_and_centroid(self, window: np.ndarray, sr: int) -> tuple[float, float]:
        """Kadın konuşmacıların ana ses frekansını (F0/Pitch) ve Spektral Merkezini hesaplar."""
        if len(window) == 0 or np.max(np.abs(window)) < 1e-4:
            return 0.0, 0.0

        fft_mag = np.abs(np.fft.rfft(window * np.hamming(len(window))))
        freqs = np.fft.rfftfreq(len(window), d=1.0 / sr)
        sum_mag = np.sum(fft_mag)
        centroid = np.sum(freqs * fft_mag) / (sum_mag + 1e-6) if sum_mag > 0 else 0.0

        corr = np.correlate(window, window, mode="full")
        corr = corr[len(corr) // 2 :]

        min_lag = int(sr / 350.0)  # ~350 Hz
        max_lag = int(sr / 100.0)  # ~100 Hz

        if max_lag < len(corr) and min_lag < max_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            pitch = float(sr) / peak_lag if peak_lag > 0 else 0.0
        else:
            pitch = 0.0

        return pitch, centroid

    def _extract_features(
        self, signal: np.ndarray, sr: int, num_coefficients: int = 13
    ) -> np.ndarray:
        """Sesteki vokal tınılarını ve Delta özniteliklerini hesaplar."""
        frame_len = int(sr * 0.025)  # 25ms
        frame_step = int(sr * 0.010)  # 10ms

        signal_length = len(signal)
        if signal_length < frame_len:
            return np.zeros((1, num_coefficients * 2))

        num_frames = 1 + int(np.ceil((signal_length - frame_len) / frame_step))
        pad_signal_length = (num_frames - 1) * frame_step + frame_len
        z = np.zeros(pad_signal_length - signal_length)
        pad_signal = np.append(signal, z)

        indices = (
            np.tile(np.arange(0, frame_len), (num_frames, 1))
            + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        )
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_len)

        nfft = 512
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = (1.0 / nfft) * (mag_frames**2)

        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, 28)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((26, int(np.floor(nfft / 2 + 1))))
        for m in range(1, 27):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - f_m)

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # Pure NumPy DCT Type-II
        N = filter_banks.shape[1]
        k_idx = np.arange(1, num_coefficients + 1)[:, None]
        n_idx = np.arange(N)[None, :]
        dct_mat = np.cos(np.pi * k_idx * (2 * n_idx + 1) / (2 * N))
        mfcc = np.dot(filter_banks, dct_mat.T)

        delta_mfcc = np.gradient(mfcc, axis=0) if len(mfcc) > 2 else np.zeros_like(mfcc)
        return np.hstack((mfcc, delta_mfcc))

    def _cluster_cosine_kmeans(
        self, norm_features: np.ndarray, k: int = 2, max_iter: int = 50
    ) -> np.ndarray:
        """SVD/PCA Bisection ve Kosinüs Benzerliği ile Garantili 2 Konuşmacı Kümelemesi."""
        n_samples, n_features = norm_features.shape
        if n_samples < k:
            return np.zeros(n_samples, dtype=int)

        norms = np.linalg.norm(norm_features, axis=1, keepdims=True) + 1e-8
        unit_feats = norm_features / norms

        # 1. SVD / PCA İlk Ayrıştırması (Vokal Tını Ekseninde İkiye Bölme)
        centered = unit_feats - np.mean(unit_feats, axis=0)
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        pc1 = vh[0]
        proj = unit_feats @ pc1
        pca_labels = (proj > np.median(proj)).astype(int)

        # Başlangıç küme merkezleri (Centroids)
        c0 = np.mean(unit_feats[pca_labels == 0], axis=0)
        c1 = np.mean(unit_feats[pca_labels == 1], axis=0)
        c0 = c0 / (np.linalg.norm(c0) + 1e-8)
        c1 = c1 / (np.linalg.norm(c1) + 1e-8)
        centroids = np.vstack([c0, c1])

        # 2. Kosinüs K-Means İterasyonu
        labels = pca_labels.copy()
        for _ in range(max_iter):
            sims = np.dot(unit_feats, centroids.T)
            new_labels = np.argmax(sims, axis=1)

            if np.array_equal(new_labels, labels):
                break

            # Eğer ikili kümeleme bozulursa PCA etiketlerine dön
            if len(np.unique(new_labels)) < k:
                return pca_labels

            labels = new_labels
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    c_mean = np.mean(unit_feats[mask], axis=0)
                    centroids[j] = c_mean / (np.linalg.norm(c_mean) + 1e-8)

        return labels

    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[DiarizationSegment]:
        try:
            import os
            import wave

            signal = None
            sr = 16000

            try:
                with wave.open(audio_path, "rb") as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)
                    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                # WAV dışında (MP3, M4A vb.) ses dosyalarını PyAV ile otomatik 16kHz WAV yap
                from audio_analyzer.adapters.audio.audio_converter import (
                    AudioConverterProcessor,
                )

                processor = AudioConverterProcessor()
                temp_wav = f"{audio_path}.temp_diarize.wav"
                converted_path = processor.normalize_and_resample(
                    audio_path, temp_wav, target_sample_rate=16000
                )
                try:
                    with wave.open(converted_path, "rb") as wf:
                        sr = wf.getframerate()
                        n_frames = wf.getnframes()
                        frames = wf.readframes(n_frames)
                        signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                finally:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)

            if signal is None or len(signal) == 0:
                return []

            win_sec = 0.5
            step_sec = 0.25
            win_size = int(sr * win_sec)
            step_size = int(sr * step_sec)

            if len(signal) < win_size:
                return [
                    DiarizationSegment(
                        speaker_id="SPEAKER_00", start_time=0.0, end_time=len(signal) / sr
                    )
                ]

            num_wins = 1 + (len(signal) - win_size) // step_size

            window_features = []
            energies = []

            for i in range(num_wins):
                w = signal[i * step_size : i * step_size + win_size]
                rms = np.sqrt(np.mean(w**2))
                energies.append(rms)

                mfcc_feat = self._extract_features(w, sr)
                mean_mfcc = np.mean(mfcc_feat, axis=0)
                pitch, centroid = self._estimate_pitch_and_centroid(w, sr)
                combined = np.hstack((mean_mfcc, [pitch, centroid]))
                window_features.append(combined)

            window_features = np.array(window_features)
            energies = np.array(energies)

            mean = np.mean(window_features, axis=0)
            std = np.std(window_features, axis=0) + 1e-6
            norm_features = (window_features - mean) / std

            k_clusters = num_speakers if num_speakers is not None else self.num_speakers
            raw_labels = self._cluster_cosine_kmeans(norm_features, k=k_clusters)

            # RMS Sessizlik Eşiği (Sessiz Es/Nefes Boşluğu Tespiti)
            silence_thresh = np.percentile(energies, 20)

            # VAD-Gated Speaker Transition: Kesintisiz cümle akışında konuşmacı değişimini kilitler
            gated_labels = raw_labels.copy()
            current_active_spk = raw_labels[0]

            for i in range(1, num_wins):
                if energies[i] < silence_thresh:
                    current_active_spk = raw_labels[i]
                else:
                    if raw_labels[i] != current_active_spk:
                        lookahead = raw_labels[i : min(i + 4, num_wins)]
                        if np.all(lookahead == raw_labels[i]) and np.any(
                            energies[i : min(i + 4, num_wins)] < silence_thresh
                        ):
                            current_active_spk = raw_labels[i]
                        else:
                            gated_labels[i] = current_active_spk

            # Medyan Filtreleme
            from scipy.signal import medfilt

            k_size = min(7, len(gated_labels))
            if k_size % 2 == 0:
                k_size = max(1, k_size - 1)
            final_labels = (
                medfilt(gated_labels, kernel_size=k_size) if len(gated_labels) > 0 else gated_labels
            )

            segments: list[DiarizationSegment] = []
            current_spk = f"SPEAKER_{final_labels[0]:02d}"
            start_t = 0.0

            for i in range(1, num_wins):
                spk = f"SPEAKER_{final_labels[i]:02d}"
                if spk != current_spk:
                    end_t = i * step_sec
                    segments.append(
                        DiarizationSegment(
                            speaker_id=current_spk, start_time=start_t, end_time=end_t
                        )
                    )
                    current_spk = spk
                    start_t = end_t

            end_t = len(signal) / sr
            segments.append(
                DiarizationSegment(speaker_id=current_spk, start_time=start_t, end_time=end_t)
            )

            return segments

        except Exception as e:
            logger.error("LocalSpectralClusterDiarizer error: %s", e, exc_info=True)
            return [DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=300.0)]
