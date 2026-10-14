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

_UNSET = object()


class _SegmentWrapper:
    __slots__ = ("start", "end", "speaker", "duration")

    def __init__(self, start: float, end: float, speaker: int | str):
        self.start = float(start)
        self.end = float(end)
        self.speaker = speaker
        self.duration = float(end - start)


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
        cluster_threshold: float | None = None,
        min_duration_on: float | None = None,
        min_duration_off: float = 0.5,
        num_threads: int | None = None,
    ):
        from audio_analyzer.config import get_settings

        settings = get_settings()
        env_th = os.getenv("SHERPA_CLUSTER_THRESHOLD")
        if cluster_threshold is not None:
            effective_th = cluster_threshold
        elif env_th:
            effective_th = float(env_th)
        else:
            effective_th = settings.sherpa_cluster_threshold

        if min_duration_on is not None:
            effective_min_dur = min_duration_on
        else:
            effective_min_dur = settings.sherpa_min_duration_on if settings.enable_min_duration_on else 0.0

        self.device_config = device_config or DeviceConfig()
        self.num_speakers = num_speakers
        self.cluster_threshold = effective_th
        self.min_duration_on = effective_min_dur
        self.min_duration_off = min_duration_off
        self.num_threads = num_threads
        self._pipeline = None
        self._config = None
        self._embedding_extractor = None
        self.last_speaker_embeddings = {}
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

                # 2. Embedding modeli yolunu tespit et (SHERPA_EMBEDDING_MODEL desteği)
                emb_path = None
                env_emb = os.getenv("SHERPA_EMBEDDING_MODEL") or getattr(settings, "sherpa_embedding_model", None)
                if env_emb and (base_dir / env_emb).exists():
                    emb_path = base_dir / env_emb
                else:
                    for candidate in [
                        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                        "wespeaker_en_voxceleb_resnet34_LM.onnx",
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

                threads = self.num_threads if self.num_threads is not None else settings.worker_cpu_threads
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
                try:
                    self._embedding_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(emb_cfg)
                except Exception as emb_err:
                    logger.warning("Sherpa-ONNX SpeakerEmbeddingExtractor başlatılamadı: %s", emb_err)
                    self._embedding_extractor = None
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

        filter_enabled = self.min_duration_on > 0.0

        for spk, seg_list in spk_groups.items():
            if filter_enabled:
                valid_segs = [s for s in seg_list if s.duration >= self.min_duration_on]
                if not valid_segs:
                    if is_num_speakers_given:
                        # Konuşmacı kümesi tamamen silinmesin diye en uzun segmenti koru
                        longest_seg = max(seg_list, key=lambda s: float(s.duration))
                        filtered_raw.append(longest_seg)
                else:
                    filtered_raw.extend(valid_segs)
            else:
                filtered_raw.extend(seg_list)

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

    def diarize(self, audio_input: object, num_speakers: int | None | object = _UNSET) -> list[DiarizationSegment]:
        self._lazy_load_pipeline()

        if self._pipeline is None:
            logger.error("Sherpa-ONNX motoru yüklü değil. Konuşmacı ayrıştırma atlanıyor.")
            return []

        try:
            samples = self._prepare_samples(audio_input)
            if len(samples) == 0:
                return []

            if num_speakers is _UNSET:
                target_num_speakers = self.num_speakers
            else:
                target_num_speakers = num_speakers

            target_clusters = (
                target_num_speakers if target_num_speakers is not None and target_num_speakers > 0 else -1
            )

            # Dinamik konuşmacı sayısı tahmini (Opt-in via ESTIMATE_NUM_SPEAKERS=true)
            estimate_speakers = os.getenv("ESTIMATE_NUM_SPEAKERS", "false").lower() == "true"
            if target_clusters == -1 and estimate_speakers and self._embedding_extractor is not None:
                try:
                    # 1. Pipeline'ı tek geçişle çalıştır (num_clusters=-1)
                    with self._lock:
                        if self._config is not None and self._config.clustering.num_clusters != -1:
                            self._config.clustering.num_clusters = -1
                            self._pipeline.set_config(self._config)
                        raw_result = self._pipeline.process(samples)
                        raw_segments = list(raw_result.sort_by_start_time())

                    # 2. Segment embedding'lerini çıkar (Tek geçiş, kümeleme ve estimator için ortak)
                    from audio_analyzer.services.speaker_count_estimator import SpeakerCountEstimator
                    from sklearn.cluster import AgglomerativeClustering
                    estimator = SpeakerCountEstimator()

                    embs = []
                    durs = []
                    valid_indices = []
                    sr = 16000
                    for idx, s in enumerate(raw_segments):
                        dur = float(s.end - s.start)
                        if dur >= 0.2:
                            st = max(0, int(s.start * sr))
                            en = min(len(samples), int(s.end * sr))
                            if en > st + 1600:
                                stream = self._embedding_extractor.create_stream()
                                stream.accept_waveform(sr, samples[st:en])
                                stream.input_finished()
                                v = np.array(self._embedding_extractor.compute(stream), dtype=np.float32)
                                n = np.linalg.norm(v)
                                if n > 1e-12:
                                    embs.append(v / n)
                                    durs.append(dur)
                                    valid_indices.append(idx)

                    if embs:
                        X_emb = np.array(embs)
                        estimated_k, details = estimator.estimate_from_embeddings(X_emb, durs)
                        logger.info("Sherpa-ONNX Dinamik Tahmin: k=%d seçildi (Kural: %s)", estimated_k, details.get("decision_rule"))

                        # 3. İkinci Sherpa C++ motoru çağrısını önle: Scikit-Learn ile doğrudan kümele (< 1ms)
                        clu = AgglomerativeClustering(n_clusters=estimated_k, metric="cosine", linkage="average")
                        cluster_labels = clu.fit_predict(X_emb)

                        # Konuşmacı merkez embedding'lerini downstream merger için önbelleğe al
                        spk_centroids: dict[str, np.ndarray] = {}
                        for k_idx in range(estimated_k):
                            mask = (cluster_labels == k_idx)
                            if np.any(mask):
                                c = np.mean(X_emb[mask], axis=0)
                                spk_centroids[f"SPEAKER_{k_idx:02d}"] = c / (np.linalg.norm(c) + 1e-12)
                        self.last_speaker_embeddings = spk_centroids

                        # Segmentleri yeni küme etiketleri ile sarmala
                        assigned = {valid_indices[i]: cluster_labels[i] for i in range(len(valid_indices))}
                        last_lbl = 0
                        adapted_segments = []
                        for idx, s in enumerate(raw_segments):
                            lbl = assigned.get(idx)
                            if lbl is None:
                                lbl = last_lbl
                            else:
                                last_lbl = lbl
                            adapted_segments.append(_SegmentWrapper(start=s.start, end=s.end, speaker=lbl))

                        return self._apply_cluster_preservation_filter(adapted_segments, estimated_k)
                except Exception as est_err:
                    logger.warning("Sherpa-ONNX dinamik konuşmacı sayısı tahmininde hata, standart akışa düşülüyor: %s", est_err)

            # Standart sabit eşikli tek geçişli çalıştırma
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

    def extract_speaker_embeddings(
        self, audio_input: object, segments: list[DiarizationSegment]
    ) -> dict[str, np.ndarray]:
        """
        Diarization segmentlerindeki her bir konuşmacının ses kesitlerinden
        SpeakerEmbeddingExtractor kullanarak normalize edilmiş ortalama embedding çıkarır.
        """
        if not segments:
            return {}

        # Önbellekte tüm konuşmacıların embedding'leri zaten mevcutsa doğrudan dön (Tekrar çıkarmayı önle)
        if self.last_speaker_embeddings:
            needed = {s.speaker_id for s in segments}
            if needed.issubset(self.last_speaker_embeddings.keys()):
                return self.last_speaker_embeddings
        try:
            samples = self._prepare_samples(audio_input)
            if len(samples) == 0:
                return {}

            self._lazy_load_pipeline()
            if self._embedding_extractor is None:
                return {}

            sr = 16000
            spk_samples: dict[str, list[np.ndarray]] = {}
            for s in segments:
                st_idx = max(0, int(s.start_time * sr))
                en_idx = min(len(samples), int(s.end_time * sr))
                if en_idx > st_idx + 800:  # En az 50ms
                    spk_samples.setdefault(s.speaker_id, []).append(samples[st_idx:en_idx])

            speaker_embeddings: dict[str, np.ndarray] = {}
            for spk_id, chunks in spk_samples.items():
                if not chunks:
                    continue
                # En fazla toplam 15 saniyelik ses parçasını birleştir
                merged_chunk = np.concatenate(chunks[:10])
                if len(merged_chunk) > sr * 15:
                    merged_chunk = merged_chunk[: sr * 15]

                if len(merged_chunk) >= 1600:  # >= 0.1s
                    stream = self._embedding_extractor.create_stream()
                    stream.accept_waveform(sr, merged_chunk)
                    stream.input_finished()
                    raw_emb = np.array(self._embedding_extractor.compute(stream), dtype=np.float32)
                    norm = np.linalg.norm(raw_emb)
                    if norm > 1e-12:
                        speaker_embeddings[spk_id] = raw_emb / norm

            self.last_speaker_embeddings = speaker_embeddings
            return speaker_embeddings
        except Exception as e:
            logger.warning("Konuşmacı embedding çıkarımı sırasında hata: %s", e)
            return {}
