import logging
from audio_analyzer.api.metrics import DIARIZATION_ESCALATION_COUNTER
from audio_analyzer.domain.interfaces import IDiarizer
from audio_analyzer.domain.models import DiarizationSegment

logger = logging.getLogger(__name__)


class EscalatingDiarizer(IDiarizer):
    """
    Diarization Yükseltme (Escalation) Katmanı.
    Konuşmacı sayısı (num_speakers) bilinmediğinde (DIARIZATION_ESCALATION=unknown_speakers)
    isteği doğrudan zor durum kademesine (Pyannote Community-1) yönlendirir ve Prometheus metriğini artırır.
    """

    def __init__(
        self,
        primary_diarizer: IDiarizer,
        escalation_diarizer: IDiarizer | None = None,
        escalation_mode: str = "off",
    ):
        self.primary_diarizer = primary_diarizer
        self.escalation_diarizer = escalation_diarizer
        self.escalation_mode = escalation_mode.lower()

    def diarize(self, audio_input: object, num_speakers: int | None = None) -> list[DiarizationSegment]:
        if (
            self.escalation_mode == "unknown_speakers"
            and num_speakers is None
            and self.escalation_diarizer is not None
        ):
            logger.info(
                "Diarization yükseltme politikası tetiklendi (reason=unknown_speakers -> target=community1)"
            )
            try:
                DIARIZATION_ESCALATION_COUNTER.labels(
                    reason="unknown_speakers", target_engine="community1"
                ).inc()
            except Exception as metric_err:
                logger.debug("Prometheus escalation metric uyarısı: %s", metric_err)

            res = self.escalation_diarizer.diarize(audio_input, num_speakers=num_speakers)
            if res:
                return res

        return self.primary_diarizer.diarize(audio_input, num_speakers=num_speakers)
