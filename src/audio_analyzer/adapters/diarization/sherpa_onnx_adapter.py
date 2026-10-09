import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_analyzer.domain.interfaces import IDiarizer
from audio_analyzer.domain.models import DeviceConfig, DiarizationSegment
from audio_analyzer.utils.cpu_budget import setup_cpu_thread_budget

logger = logging.getLogger(__name__)


class SherpaOnnxAudioAdapter(IDiarizer):
    """
    Sherpa-ONNX Konuşmacı Ayrıştırma Motoru Adaptörü (%100 Çevrimdışı, CPU-Optimized, Token-Free).
    Pyannote segmentation-3.0 ONNX ve WeSpeaker/3DSpeaker ONNX embedding modellerini kullanarak
    yüksek başarımlı ve düşük bellek tüketimli diarization gerçekleştirir.
    """

    def __init__(
        self,
        device_config: DeviceConfig | None = None,
        num_speakers: int | None = None,
        cluster_threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
    ):
        self.device_config = device_config or DeviceConfig()
        self.num_speakers = num_speakers
        self.cluster_threshold = cluster_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self._pipeline = None
        self._config = None
        self._initialized = False
        self._lock = threading.Lock()

    def _lazy_load_pipeline(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

            try:
                import sherpa_onnx
                from audio_analyzer.config import get_settings

                settings = get_settings()
                base_dir = Path(settings.model_dir) / "diarization" / "sherpa_onnx"

                # 1. Segmentation modeli yolunu tespit et
                seg_path = None
                for candidate in ["model.onnx", "segmentation-3.0.onnx", "pyannote-segmentation-3.0.onnx"]:
                    p = base_dir / candidate
                    if p.exists():
                        seg_path = p
                        break

                if not seg_path and base_dir.exists():
                    onnx_files = list(base_dir.rglob("*.onnx"))
                    for f in onnx_files:
                        if "segmentation" in f.name.lower() or f.name == "model.onnx":
                            seg_path = f
                            break

                # 2. Embedding modeli yolunu tespit et
                emb_path = None
                for candidate in [
                    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                    "wespeaker_en_voxceleb_CAM++_LM.onnx",
                    "embedding.onnx",
                ]:
                    p = base_dir / candidate
                    if p.exists():
                        emb_path = p
                        break

                if not emb_path and base_dir.exists():
                    onnx_files = list(base_dir.rglob("*.onnx"))
                    for f in onnx_files:
                        if f != seg_path and ("embedding" in f.name.lower() or "speaker" in f.name.lower() or "eres2net" in f.name.lower() or "cam++" in f.name.lower()):
                            emb_path = f
                            break

                if not seg_path or not seg_path.exists():
                    logger.warning("Sherpa-ONNX segmentation modeli bulunamadı (%s). Fallback motoruna düşülecek.", base_dir / "model.onnx")
                    self._pipeline = None
                    return

                if not emb_path or not emb_path.exists():
                    logger.warning("Sherpa-ONNX embedding modeli bulunamadı (%s). Fallback motoruna düşülecek.", base_dir)
                    self._pipeline = None
                    return

                threads = setup_cpu_thread_budget()
                logger.info(
                    "Sherpa-ONNX konuşmacı motoru yükleniyor (Seg: %s, Emb: %s, Threads: %d, Threshold: %.2f)...",
                    seg_path.name,
                    emb_path.name,
                    threads,
                    self.cluster_threshold,
                )

                py_cfg = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_path)
                )
                seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=py_cfg, num_threads=threads, debug=False, provider="cpu"
                )
                emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(emb_path), num_threads=threads, debug=False, provider="cpu"
                )
                target_clusters = self.num_speakers if self.num_speakers and self.num_speakers > 0 else -1
                clu_cfg = sherpa_onnx.FastClusteringConfig(
                    num_clusters=target_clusters, threshold=self.cluster_threshold
                )

                # Python tarafında custom min_duration_on filtresi uygulamak için C++ min_duration_on=0.0 ve min_duration_off=0.0 geçilir
                self._config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=seg_cfg,
                    embedding=emb_cfg,
                    clustering=clu_cfg,
                    min_duration_on=0.0,
                    min_duration_off=0.0,
                )

                if not self._config.validate():
                    logger.error("Sherpa-ONNX konfigürasyon doğrulaması başarısız oldu!")
                    self._pipeline = None
                    return

                self._pipeline = sherpa_onnx.OfflineSpeakerDiarization(self._config)
                logger.info("Sherpa-ONNX konuşmacı motoru başarıyla aktifleştirildi.")

            except Exception as ex:
                logger.warning("Sherpa-ONNX motoru yüklenemedi (%s). Fallback motoruna düşülecek.", ex)
                self._pipeline = None

    def _prepare_samples(self, audio_input: Any) -> np.ndarray:
        """Ses girdisini float32 1D numpy dizisine ve 16000Hz örnekleme oranına dönüştürür."""
        if isinstance(audio_input, (str, Path)):
            audio, sr = sf.read(str(audio_input), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                import math
                gcd = math.gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)
            return audio.astype(np.float32)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio.astype(np.float32)
        elif isinstance(audio_input, dict):
            waveform = audio_input.get("waveform")
            sr = audio_input.get("sample_rate", 16000)
            if hasattr(waveform, "numpy"):
                waveform = waveform.numpy()
            audio = np.asarray(waveform, dtype=np.float32).squeeze()
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                import math
                gcd = math.gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)
            return audio.astype(np.float32)
        else:
            raise ValueError(f"Desteklenmeyen ses girdi tipi: {type(audio_input)}")

    def _apply_cluster_preservation_filter(
        self, raw_segments: list[Any], num_speakers_requested: int | None
    ) -> list[DiarizationSegment]:
        """
        Python tarafında min_duration_on filtresi uygular.
        num_speakers verildiğinde, filtre bir konuşmacının TÜM segmentlerini silecekse
        o konuşmacının en uzun segmentini korur (Yaklaşım A - Cluster Preservation).
        """
        if not raw_segments:
            return []

        # 1. Konuşmacılara göre grupla
        spk_groups: dict[Any, list[Any]] = {}
        for seg in raw_segments:
            spk_groups.setdefault(seg.speaker, []).append(seg)

        filtered_raw = []
        is_num_speakers_given = num_speakers_requested is not None and num_speakers_requested > 0

        for spk, seg_list in spk_groups.items():
            valid_segs = [s for s in seg_list if s.duration >= self.min_duration_on]
            if not valid_segs:
                if is_num_speakers_given:
                    # Konuşmacı kümesi tamamen silinmesin diye en uzun segmenti koru
                    longest_seg = max(seg_list, key=lambda s: float(s.duration))
                    filtered_raw.append(longest_seg)
            else:
                filtered_raw.extend(valid_segs)

        # 2. Başlangıç zamanına göre sırala
        filtered_raw.sort(key=lambda s: float(s.start))

        # 3. DiarizationSegment modeline dönüştür
        result_segments: list[DiarizationSegment] = []
        for seg in filtered_raw:
            spk_idx = seg.speaker
            speaker_id = f"SPEAKER_{spk_idx:02d}" if isinstance(spk_idx, int) else str(spk_idx)
            result_segments.append(
                DiarizationSegment(
                    speaker_id=speaker_id,
                    start_time=round(float(seg.start), 3),
                    end_time=round(float(seg.end), 3),
                )
            )

        return result_segments

    def diarize(self, audio_input: object, num_speakers: int | None = None) -> list[DiarizationSegment]:
        self._lazy_load_pipeline()

        if self._pipeline is None:
            logger.error("Sherpa-ONNX motoru yüklü değil. Konuşmacı ayrıştırma atlanıyor.")
            return []

        try:
            samples = self._prepare_samples(audio_input)
            if len(samples) == 0:
                return []

            target_num_speakers = num_speakers if num_speakers is not None else self.num_speakers
            target_clusters = target_num_speakers if target_num_speakers and target_num_speakers > 0 else -1

            # Thread-safe set_config + process çağrısı
            with self._lock:
                if self._config is not None and self._config.clustering.num_clusters != target_clusters:
                    self._config.clustering.num_clusters = target_clusters
                    self._pipeline.set_config(self._config)

                raw_result = self._pipeline.process(samples)
                raw_segments = raw_result.sort_by_start_time()

            # Python-side min_duration_on & cluster preservation filtresi
            return self._apply_cluster_preservation_filter(raw_segments, target_num_speakers)

        except Exception as err:
            logger.error("Sherpa-ONNX diarize işleminde hata oluştu (%s).", err, exc_info=True)
            return []
