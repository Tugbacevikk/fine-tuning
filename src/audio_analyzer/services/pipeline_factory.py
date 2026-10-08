import logging
import threading

from audio_analyzer.adapters.audio.silero_vad import SileroVADProcessor
from audio_analyzer.config import get_settings
from audio_analyzer.domain.models import DeviceConfig
from audio_analyzer.services.fusion_engine import FusionEngine
from audio_analyzer.services.pipeline import AudioAnalysisPipeline
from audio_analyzer.services.semantic_refiner import SemanticRefiner

logger = logging.getLogger(__name__)

_cached_pipeline: AudioAnalysisPipeline | None = None
_pipeline_lock = threading.Lock()


def reset_pipeline_cache():
    """Önbellekteki pipeline nesnesini sıfırlar, böylece güncel parametreler yeniden yüklenir."""
    global _cached_pipeline
    with _pipeline_lock:
        _cached_pipeline = None


def _build_pipeline() -> AudioAnalysisPipeline:
    settings = get_settings()
    device_config = DeviceConfig()

    # Ortam değişkenlerinden blueprint parametrelerini oku
    whisper_model_size = settings.whisper_model_size
    max_silence_threshold = settings.max_silence_threshold

    # 1. STT Engine (FasterWhisper)
    try:
        from audio_analyzer.adapters.stt.faster_whisper_adapter import (
            FasterWhisperAdapter,
        )

        stt_engine = FasterWhisperAdapter(
            model_size=whisper_model_size, device_config=device_config
        )
    except Exception as e:
        raise RuntimeError(
            f"STT Motoru (FasterWhisper) başlatılamadı: {e}. Lütfen model bağımlılıklarını kontrol edin."
        )

    # 2. Diarization Engine (Pyannote 3.1 SOTA / SpeechBrain ECAPA Fallback)
    from audio_analyzer.adapters.diarization.fallback_diarizer import FallbackDiarizer
    from audio_analyzer.adapters.diarization.speechbrain_adapter import SpeechBrainECAPADiarizer

    diar_engine_name = settings.diarization_engine
    fallback_diarizer = SpeechBrainECAPADiarizer(
        device_config=device_config, num_speakers=settings.target_num_speakers
    )

    if diar_engine_name in ("sherpa_onnx", "sherpa"):
        from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import (
            SherpaOnnxAudioAdapter,
        )

        primary_diarizer = SherpaOnnxAudioAdapter(
            device_config=device_config, num_speakers=settings.target_num_speakers
        )
    elif diar_engine_name == "pyannote":
        from audio_analyzer.adapters.diarization.pyannote_adapter import (
            PyannoteAudioAdapter,
        )

        primary_diarizer = PyannoteAudioAdapter(
            device_config=device_config, num_speakers=settings.target_num_speakers
        )
    else:
        primary_diarizer = fallback_diarizer
        fallback_diarizer = None

    fallbacks = [fallback_diarizer] if fallback_diarizer and fallback_diarizer != primary_diarizer else []
    diarizer = FallbackDiarizer(primary_diarizer=primary_diarizer, fallback_diarizers=fallbacks)

    from audio_analyzer.adapters.audio.denoiser import DeepFilterDenoiser
    from audio_analyzer.adapters.audio.rust_dsp_adapter import RustAudioDSPProcessor

    audio_processor = RustAudioDSPProcessor()
    vad_processor = SileroVADProcessor()
    enable_denoiser = settings.enable_denoiser
    denoiser = DeepFilterDenoiser(enabled=enable_denoiser)

    # 3. Modelleri sunucu açılışında belleğe ısındır (Warm-Up & Pre-Load)
    try:
        logger.info("Yapay Zeka modelleri (FasterWhisper & Pyannote.audio) belleğe yükleniyor...")
        if hasattr(stt_engine, "_lazy_load_model"):
            stt_engine._lazy_load_model()
        target_diar = diarizer.primary if hasattr(diarizer, "primary") else diarizer
        if hasattr(target_diar, "_lazy_load_pipeline"):
            target_diar._lazy_load_pipeline()
        elif hasattr(target_diar, "_load_classifier"):
            target_diar._load_classifier()
        logger.info("Yapay Zeka modelleri başarıyla yüklendi [HAZIR].")
    except Exception as warm_err:
        logger.warning("Model ön yükleme uyarısı: %s", warm_err)

    return AudioAnalysisPipeline(
        stt_engine=stt_engine,
        diarizer=diarizer,
        audio_processor=audio_processor,
        vad_processor=vad_processor,
        denoiser=denoiser,
        fusion_engine=FusionEngine(max_silence_threshold=max_silence_threshold),
        semantic_refiner=SemanticRefiner(),
    )


def get_shared_pipeline() -> AudioAnalysisPipeline:
    """
    Sıcak Yükleme (Warm-Loading) Singleton Fabrikası.
    Ağır GPU/CPU yapay zeka modellerini (Whisper, SpeechBrain/PyAnnote) disk/bellekten
    yalnızca İLK İSTEKTE yükler ve sonraki tüm analiz görevlerinde hazır modeli yeniden kullanır.
    Her istekte modeli sıfırdan yükleme gecikmesini (10-30sn) ortadan kaldırır.
    """
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline

    with _pipeline_lock:
        if _cached_pipeline is not None:
            return _cached_pipeline
        _cached_pipeline = _build_pipeline()
        return _cached_pipeline
