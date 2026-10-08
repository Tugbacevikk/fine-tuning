import logging
import os
from pathlib import Path

from audio_analyzer.domain.interfaces import IDiarizer
from audio_analyzer.domain.models import DeviceConfig, DiarizationSegment

logger = logging.getLogger(__name__)


class PyannoteAudioAdapter(IDiarizer):
    """
    Pyannote.audio 4.x Community-1 Zor Durum Kademe Konuşmacı Ayrıştırma Motoru Adaptörü.
    Zorlu konuşmacı ayrıştırma senaryoları için ikincil yüksek duyarlılıklı kademe olarak görev yapar.
    """

    def __init__(
        self,
        device_config: DeviceConfig | None = None,
        token: str | None = None,
        num_speakers: int | None = None,
    ):
        self.device_config = device_config or DeviceConfig()
        self.token = (
            token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.num_speakers = num_speakers
        self._pipeline = None
        self._initialized = False

    def _lazy_load_pipeline(self):
        if not self._initialized:
            self._initialized = True
            try:
                import torch
                from pyannote.audio import Pipeline
                from audio_analyzer.config import get_settings

                settings = get_settings()
                model_dir = Path(settings.model_dir)
                community_dir = model_dir / "diarization" / "pyannote-community-1"

                if (community_dir / "config.yaml").exists():
                    source = str(community_dir / "config.yaml")
                elif community_dir.exists():
                    source = str(community_dir)
                else:
                    source = "pyannote/speaker-diarization-3.1"

                tok = self.token
                logger.info("Pyannote.audio 4.x Community-1 hattı yükleniyor (Kaynak: %s)...", source)

                kwargs = {}
                if tok and not Path(source).exists():
                    kwargs["token"] = tok

                self._pipeline = Pipeline.from_pretrained(source, **kwargs)
                if self._pipeline is not None:
                    step = settings.diarization_step_sec
                    if hasattr(self._pipeline, "segmentation") and hasattr(self._pipeline.segmentation, "step"):
                        self._pipeline.segmentation.step = max(0.4, step)
                    if self.device_config.device == "cuda":
                        self._pipeline.to(torch.device("cuda"))
                logger.info("Pyannote.audio 4.x Community-1 motoru başarıyla aktifleştirildi.")
            except Exception as ex:
                logger.warning(
                    "Pyannote.audio 4.x Community-1 motoru yüklenemedi (%s).", ex
                )
                self._pipeline = None

    def diarize(self, audio_input: object, num_speakers: int | None = None) -> list[DiarizationSegment]:
        self._lazy_load_pipeline()

        if self._pipeline is None:
            logger.error("Pyannote.audio 4.x motoru yüklenemedi. Konuşmacı ayrıştırma atlanıyor.")
            return []

        try:
            from audio_analyzer.config import get_settings
            step = get_settings().diarization_step_sec
            if hasattr(self._pipeline, "segmentation"):
                self._pipeline.segmentation.step = max(0.4, step)
            import numpy as np
            import torch

            if isinstance(audio_input, np.ndarray):
                tensor = torch.from_numpy(audio_input).float()
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                audio_payload = {"waveform": tensor, "sample_rate": 16000}
            elif isinstance(audio_input, dict):
                audio_payload = audio_input
            else:
                audio_payload = str(audio_input)

            kwargs = {}
            target_num_speakers = num_speakers if num_speakers is not None else self.num_speakers
            if target_num_speakers:
                kwargs["num_speakers"] = target_num_speakers

            diarization_out = self._pipeline(audio_payload, **kwargs)
            
            # Pyannote 4.x exclusive annotation kontrolü
            annotation = getattr(diarization_out, "exclusive", None)
            if annotation is None:
                annotation = getattr(diarization_out, "speaker_diarization", diarization_out)

            segments: list[DiarizationSegment] = []

            if hasattr(annotation, "itertracks"):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segments.append(
                        DiarizationSegment(
                            speaker_id=str(speaker),
                            start_time=round(float(turn.start), 3),
                            end_time=round(float(turn.end), 3),
                        )
                    )

            return segments
        except Exception as err:
            logger.error("Pyannote Community-1 diarize hatası (%s).", err)
            return []
