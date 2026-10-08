import logging
from typing import List, Optional

from audio_analyzer.api.metrics import DIARIZATION_FALLBACK_COUNTER
from audio_analyzer.domain.interfaces import IDiarizer
from audio_analyzer.domain.models import DiarizationSegment

logger = logging.getLogger(__name__)


class FallbackDiarizer(IDiarizer):
    """
    Diarization Fallback Orkestratörü.
    Öncelikli diarization motorunu (ör. Pyannote/Sherpa) çalıştırır.
    Başarısız olursa veya boş sonuç döndürürse, zincirdeki yedek motorlara (ör. SpeechBrain / ClusterDiarizer) geçer.
    """

    def __init__(
        self, primary_diarizer: IDiarizer, fallback_diarizers: Optional[List[IDiarizer]] = None
    ):
        self.primary = primary_diarizer
        self.fallbacks = fallback_diarizers or []

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[DiarizationSegment]:
        chain = [self.primary] + self.fallbacks
        for i, engine in enumerate(chain):
            engine_name = engine.__class__.__name__
            try:
                logger.info(
                    "Diarization motoru deneniyor: %s (Adım %d/%d)", engine_name, i + 1, len(chain)
                )
                if num_speakers is not None:
                    try:
                        segments = engine.diarize(audio_path, num_speakers=num_speakers)
                    except TypeError:
                        segments = engine.diarize(audio_path)
                else:
                    segments = engine.diarize(audio_path)
                if segments and len(segments) > 0:
                    return segments

                logger.warning(
                    "%s diarization motoru boş segment listesi döndürdü. Fallback tetikleniyor...",
                    engine_name,
                )
                next_engine = chain[i + 1].__class__.__name__ if i + 1 < len(chain) else "None"
                DIARIZATION_FALLBACK_COUNTER.labels(
                    from_engine=engine_name, to_engine=next_engine, reason="empty_result"
                ).inc()
            except Exception as e:
                logger.error(
                    "%s diarization hatası: %s. Fallback tetikleniyor...",
                    engine_name,
                    e,
                    exc_info=True,
                )
                next_engine = chain[i + 1].__class__.__name__ if i + 1 < len(chain) else "None"
                DIARIZATION_FALLBACK_COUNTER.labels(
                    from_engine=engine_name, to_engine=next_engine, reason="exception"
                ).inc()

        logger.error("Tüm diarization motorları (fallback dahil) başarısız oldu veya boş döndü.")
        return []
