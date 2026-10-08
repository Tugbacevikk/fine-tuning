import os
import pytest
from pathlib import Path
from unittest.mock import patch

from audio_analyzer.api.metrics import DIARIZATION_FALLBACK_COUNTER
from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxAudioAdapter
from audio_analyzer.domain.models import DeviceConfig


@pytest.mark.integration
def test_airgapped_diarization_no_fallbacks():
    """
    Ağsız (Air-Gapped / Token-Free) Entegrasyon Testi.
    HF_HUB_OFFLINE=1 ve TRANSFORMERS_OFFLINE=1 ile gerçek bir 2-konuşmacılı kaydı işler.
    audio_diarization_fallbacks_total sayacının ARTMADIĞINI (%0 fallback) doğrular.
    """
    wav_path = Path("tests/fixtures/diarization_eval/large/eval_sample_01_2spk.wav")
    if not wav_path.exists():
        pytest.skip("Test için gerekli 2-konuşmacılı WAV fixtures dosyası bulunamadı.")

    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIARIZATION_ENGINE": "sherpa_onnx"}):
        device_config = DeviceConfig()
        adapter = SherpaOnnxAudioAdapter(device_config=device_config, num_speakers=2)
        adapter._lazy_load_pipeline()

        assert adapter._pipeline is not None, "Air-gapped ortamda Sherpa-ONNX yerel modeli bulunamadı!"

        # Başlangıç fallback sayacı değerini kaydet
        initial_fallbacks = sum(sample.value for sample in DIARIZATION_FALLBACK_COUNTER.collect()[0].samples if sample.name == "audio_diarization_fallbacks_total_sum") if DIARIZATION_FALLBACK_COUNTER.collect() else 0.0

        segments = adapter.diarize(str(wav_path), num_speakers=2)

        # Bitiş fallback sayacını kontrol et
        final_fallbacks = sum(sample.value for sample in DIARIZATION_FALLBACK_COUNTER.collect()[0].samples if sample.name == "audio_diarization_fallbacks_total_sum") if DIARIZATION_FALLBACK_COUNTER.collect() else 0.0

        assert len(segments) > 0, "Ağsız modda konuşmacı segmentleri başarıyla üretilmeli."
        assert final_fallbacks == initial_fallbacks, "HF_HUB_OFFLINE=1 iken fallback sayacı ARTMAMALIDIR."
