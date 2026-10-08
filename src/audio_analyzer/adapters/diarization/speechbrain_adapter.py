import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_analyzer.adapters.audio.rust_dsp_adapter import RustAudioDSPProcessor
from audio_analyzer.domain.interfaces import IAudioProcessor, IDiarizer
from audio_analyzer.domain.models import DeviceConfig, DiarizationSegment

logger = logging.getLogger(__name__)


def majority_vote_filter(labels: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Sayısal sıralı olmayan konuşmacı etiketleri (>2) için kayan pencere çoğunluk oyu (mode) filtresi."""
    if len(labels) < kernel_size:
        return labels
    out = np.copy(labels)
    half = kernel_size // 2
    for i in range(half, len(labels) - half):
        window = labels[i - half : i + half + 1]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


class SpeechBrainECAPADiarizer(IDiarizer):
    """
    SpeechBrain ECAPA-TDNN Derin Sinir Ağı Konuşmacı Ayrıştırma Motoru (%100 Token-Free & Çevrimdışı).
    192-boyutlu nöral ses embedding'leri çıkararak hiyerarşik kümeleme (Agglomerative Clustering) ile konuşmacıları ayrıştırır.
    """

    def __init__(
        self,
        device_config: DeviceConfig = None,
        num_speakers: int | None = None,
        max_speakers: int = 10,
        audio_processor: IAudioProcessor | None = None,
    ):
        from audio_analyzer.config import get_settings
        settings = get_settings()
        self.device_config = device_config or DeviceConfig()
        self.num_speakers = num_speakers if num_speakers is not None else settings.target_num_speakers
        self.max_speakers = max_speakers
        self.audio_processor = audio_processor or RustAudioDSPProcessor()
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is None:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            logging.getLogger("speechbrain").setLevel(logging.ERROR)
            logging.getLogger("speechbrain.utils.fetching").setLevel(logging.ERROR)
            logging.getLogger("speechbrain.utils.parameter_transfer").setLevel(logging.ERROR)
            logging.getLogger("speechbrain.utils.quirks").setLevel(logging.ERROR)

            from speechbrain.inference.speaker import EncoderClassifier

            from audio_analyzer.config import get_settings
            settings = get_settings()
            spk_dir = Path(settings.model_dir) / "diarization" / "speechbrain_ecapa"

            spk_dir.mkdir(parents=True, exist_ok=True)

            try:
                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(spk_dir),
                    run_opts={"device": self.device_config.device},
                )
            except Exception as ex:
                logger.debug("SpeechBrain önbellek yükleme notu: %s", ex)
                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device_config.device},
                )

    def diarize(self, audio_input: str | np.ndarray, num_speakers: int | None = None) -> list[DiarizationSegment]:
        try:
            self._load_classifier()
            from scipy.signal import medfilt

            from audio_analyzer.config import get_settings
            settings = get_settings()

            if isinstance(audio_input, np.ndarray):
                data = audio_input
                sr = 16000
            else:
                data, sr = sf.read(audio_input)
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            target_sr = 16000
            if sr != target_sr:
                from math import gcd
                import scipy.signal
                g = gcd(int(sr), target_sr)
                up = target_sr // g
                down = int(sr) // g
                data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
                sr = target_sr

            win_sec = 1.2
            step_sec = float(settings.diarization_step_sec)
            win_samples = int(sr * win_sec)
            step_samples = int(sr * step_sec)

            if len(data) < win_samples:
                return [
                    DiarizationSegment(
                        speaker_id="SPEAKER_00", start_time=0.0, end_time=len(data) / sr
                    )
                ]

            num_wins = 1 + (len(data) - win_samples) // step_samples
            valid_clips = []
            valid_indices = []

            # Energy gating to skip absolute silence (5th percentile to keep quiet speech)
            rms_energies = np.array(
                [
                    np.sqrt(np.mean(data[i * step_samples : i * step_samples + win_samples] ** 2))
                    for i in range(num_wins)
                ]
            )
            energy_thresh = max(1e-5, np.percentile(rms_energies, 5))

            for i in range(num_wins):
                clip = data[i * step_samples : i * step_samples + win_samples]
                if rms_energies[i] < energy_thresh:
                    continue
                valid_clips.append(clip)
                valid_indices.append(i)

            if not valid_clips:
                return [
                    DiarizationSegment(
                        speaker_id="SPEAKER_00", start_time=0.0, end_time=len(data) / sr
                    )
                ]

            # Dynamic Batching for SpeechBrain Embedding Extraction (8/16/32 Batch Slots)
            batch_size = int(os.getenv("DIARIZATION_BATCH_SIZE", "32"))
            raw_embeddings = []
            device = self.device_config.device

            logger.info("SpeechBrain ECAPA Konuşmacı Ayrıştırma başlatılıyor (%d pencere segmenti işleniyor)...", len(valid_clips))
            for b_idx in range(0, len(valid_clips), batch_size):
                b_chunk = valid_clips[b_idx : b_idx + batch_size]
                batch_arr = np.array(b_chunk, dtype=np.float32)
                tensor_batch = torch.tensor(batch_arr, dtype=torch.float32)

                with torch.no_grad():
                    if device == "cuda" and torch.cuda.is_available():
                        tensor_batch = tensor_batch.to("cuda")
                        with torch.amp.autocast("cuda"):
                            emb = self._classifier.encode_batch(tensor_batch)
                    else:
                        emb = self._classifier.encode_batch(tensor_batch)

                    emb_np = emb.squeeze(1).cpu().numpy()
                    if emb_np.ndim == 1:
                        emb_np = np.expand_dims(emb_np, axis=0)
                    raw_embeddings.append(emb_np)

            embeddings = np.vstack(raw_embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            unit_embs = embeddings / norms

            from audio_analyzer.config import get_settings
            settings = get_settings()
            dist_thresh = float(settings.diarization_threshold)

            from sklearn.cluster import AgglomerativeClustering

            if len(unit_embs) == 1:
                raw_valid_labels = np.zeros(1, dtype=int)
            else:
                effective_num_speakers = num_speakers if num_speakers is not None else self.num_speakers

                if effective_num_speakers is not None:
                    n_spk = min(effective_num_speakers, len(unit_embs))
                    if n_spk <= 1:
                        raw_valid_labels = np.zeros(len(unit_embs), dtype=int)
                    else:
                        model = AgglomerativeClustering(
                            n_clusters=n_spk, metric="cosine", linkage="average"
                        )
                        raw_valid_labels = model.fit_predict(unit_embs)
                else:
                    model = AgglomerativeClustering(
                        n_clusters=None,
                        metric="cosine",
                        linkage="average",
                        distance_threshold=dist_thresh,
                    )
                    raw_valid_labels = model.fit_predict(unit_embs)
                    n_clusters = len(np.unique(raw_valid_labels))
                    if n_clusters > self.max_speakers:
                        model_cap = AgglomerativeClustering(
                            n_clusters=self.max_speakers, metric="cosine", linkage="average"
                        )
                        raw_valid_labels = model_cap.fit_predict(unit_embs)

            # Map valid labels back to all windows
            raw_labels = np.zeros(num_wins, dtype=int)
            last_lbl = raw_valid_labels[0] if len(raw_valid_labels) > 0 else 0
            val_idx = 0
            for i in range(num_wins):
                if val_idx < len(valid_indices) and i == valid_indices[val_idx]:
                    if val_idx < len(raw_valid_labels):
                        last_lbl = raw_valid_labels[val_idx]
                    val_idx += 1
                raw_labels[i] = last_lbl

            effective_num_spk = num_speakers if num_speakers is not None else self.num_speakers

            k_size = 3 if len(raw_labels) >= 3 else 1
            unique_label_count = len(np.unique(raw_labels))

            if unique_label_count > 2:
                final_labels = majority_vote_filter(raw_labels, kernel_size=k_size)
            else:
                final_labels = (
                    medfilt(raw_labels, kernel_size=k_size) if len(raw_labels) > 0 else raw_labels
                )

            if (
                effective_num_spk is not None
                and len(np.unique(final_labels)) < min(effective_num_spk, len(unit_embs))
                and len(np.unique(raw_labels)) >= min(effective_num_spk, len(unit_embs))
            ):
                # Yaklaşım A (Seçilen Yöntem): Medfilt sonrası kaybolan kümelerin en uzun ardışık pencere grubunu koru
                raw_clusters = set(np.unique(raw_labels))
                final_clusters = set(np.unique(final_labels))
                missing_clusters = raw_clusters - final_clusters

                for c in missing_clusters:
                    runs = []
                    in_run = False
                    start_r = 0
                    for r_i, lbl in enumerate(raw_labels):
                        if lbl == c:
                            if not in_run:
                                in_run = True
                                start_r = r_i
                        else:
                            if in_run:
                                in_run = False
                                runs.append((start_r, r_i))
                    if in_run:
                        runs.append((start_r, len(raw_labels)))

                    if runs:
                        best_run = max(runs, key=lambda r: r[1] - r[0])
                        final_labels[best_run[0]:best_run[1]] = c

            segments: list[DiarizationSegment] = []
            current_spk = f"SPEAKER_{final_labels[0]:02d}"
            start_t = 0.0

            half_win = win_sec / 2.0
            for i in range(1, num_wins):
                spk = f"SPEAKER_{final_labels[i]:02d}"
                if spk != current_spk:
                    # Segment sınırlarını pencere merkezine göre hesapla
                    end_t = (i * step_sec) + half_win
                    segments.append(
                        DiarizationSegment(
                            speaker_id=current_spk, start_time=start_t, end_time=end_t
                        )
                    )
                    current_spk = spk
                    start_t = end_t

            end_t = len(data) / sr
            segments.append(
                DiarizationSegment(speaker_id=current_spk, start_time=start_t, end_time=end_t)
            )

            return segments

        except Exception as e:
            logger.error("SpeechBrainECAPADiarizer error, fallback: %s", e, exc_info=True)
            from audio_analyzer.adapters.diarization.cluster_diarizer import (
                LocalSpectralClusterDiarizer,
            )

            effective_num_spk = num_speakers if num_speakers is not None else self.num_speakers
            fallback_diarizer = LocalSpectralClusterDiarizer(
                device_config=self.device_config, num_speakers=effective_num_spk or 2
            )
            if isinstance(audio_input, str):
                return fallback_diarizer.diarize(audio_input, num_speakers=effective_num_spk)
            else:
                import tempfile
                import uuid
                tmp_path = os.path.join(tempfile.gettempdir(), f"diar_tmp_{uuid.uuid4().hex}.wav")
                sf.write(tmp_path, audio_input, 16000, subtype="PCM_16")
                try:
                    return fallback_diarizer.diarize(tmp_path, num_speakers=effective_num_spk)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception as clean_err:
                            logger.warning("Geçici dosya silinirken hata: %s", clean_err)
