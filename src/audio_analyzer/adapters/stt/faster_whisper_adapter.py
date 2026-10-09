import logging
import os

from audio_analyzer.domain.interfaces import ISTTEngine
from audio_analyzer.domain.models import DeviceConfig, WordSegment
from audio_analyzer.utils.cpu_budget import setup_cpu_thread_budget

logger = logging.getLogger(__name__)


class FasterWhisperAdapter(ISTTEngine):
    """
    Faster-Whisper (CTranslate2) Speech-to-Text Motor Adaptörü.
    Donanım bilincine (DeviceConfig) sahiptir: GPU'da CUDA+float16, CPU'da int8 modunda çalışır.
    """

    def __init__(
        self,
        model_size: str = "small",
        device_config: DeviceConfig | None = None,
        initial_prompt: str | None = None,
    ):
        self.model_size = model_size
        self.device_config = device_config or DeviceConfig()
        self.initial_prompt = initial_prompt
        self._model = None
        self._batched_model = None

    def _lazy_load_model(self):
        """Modeli ve BatchedInferencePipeline'ı ihtiyaç anında (lazy loading) bir kez yükler."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                cpu_threads = setup_cpu_thread_budget()
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device_config.device,
                    compute_type=self.device_config.compute_type,
                    device_index=self.device_config.device_index,
                    cpu_threads=cpu_threads,
                )

                try:
                    from faster_whisper import BatchedInferencePipeline

                    self._batched_model = BatchedInferencePipeline(model=self._model)
                    logger.info("FasterWhisper BatchedInferencePipeline tek seferlik oluşturuldu.")
                except Exception as b_err:
                    logger.info("BatchedInferencePipeline oluşturulamadı (%s), standart modele düşülüyor.", b_err)
                    self._batched_model = None

            except ImportError:
                raise ImportError(
                    "faster-whisper kütüphanesi yüklü değil. 'pip install faster-whisper' çalıştırın."
                )

    def transcribe(self, audio_path: str) -> tuple[list[WordSegment], str | None]:
        from audio_analyzer.config import get_settings
        settings = get_settings()
        language = os.getenv("WHISPER_LANGUAGE", "tr")
        beam_size = settings.whisper_beam_size
        batch_size = settings.whisper_batch_size
        vad_params = dict(min_silence_duration_ms=1000, speech_pad_ms=400)

        prompt_str = self.initial_prompt or (
            "Türkçe ses kaydı."
            if language == "tr"
            else None
        )

        use_vad_filter = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"

        if self._batched_model is not None and self.device_config.device == "cuda":
            segments, info = self._batched_model.transcribe(
                audio_path,
                batch_size=batch_size,
                language=language,
                initial_prompt=prompt_str,
                word_timestamps=True,
                beam_size=beam_size,
                condition_on_previous_text=False,
                temperature=0.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=None,
                vad_filter=use_vad_filter,
                vad_parameters=vad_params if use_vad_filter else None,
            )
        else:
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                initial_prompt=prompt_str,
                word_timestamps=True,
                beam_size=beam_size,
                condition_on_previous_text=False,
                temperature=0.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=None,
                vad_filter=use_vad_filter,
                vad_parameters=vad_params if use_vad_filter else None,
            )

        words: list[WordSegment] = []
        last_clean_words: list[str] = []

        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for w in segment.words:
                    clean_w = w.word.strip().lower()
                    # Ardışık 3'ten fazla kelime tekrarı döngüsünü filtrele
                    if len(last_clean_words) >= 2 and clean_w == last_clean_words[-1] == last_clean_words[-2]:
                        continue
                    last_clean_words.append(clean_w)
                    if len(last_clean_words) > 10:
                        last_clean_words.pop(0)

                    words.append(
                        WordSegment(
                            word=w.word,
                            start_time=w.start,
                            end_time=w.end,
                            probability=w.probability if hasattr(w, "probability") else 1.0,
                        )
                    )

        detected_language = info.language if info and hasattr(info, "language") else None
        return words, detected_language
