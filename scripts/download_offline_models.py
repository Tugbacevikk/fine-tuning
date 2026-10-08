import sys
from pathlib import Path

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from audio_analyzer.config import get_settings
    MODEL_DIR = Path(get_settings().model_dir)
except Exception:
    MODEL_DIR = Path(__file__).parent.parent / "storage" / "models"


def download_stt_model(model_size: str = "small"):
    """FasterWhisper modelini yerel depolama klasörüne indirir ve hazırlar."""
    stt_dir = MODEL_DIR / "stt" / model_size
    stt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STT] STT Modeli indiriliyor/kontrol ediliyor ({model_size}) -> {stt_dir}")

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, download_root=str(stt_dir), device="cpu", compute_type="int8")
        print(f"[OK] STT Modeli ({model_size}) yerel klasöre indirildi!")
    except Exception as e:
        print(f"[ERROR] STT Model indirme hatası: {e}")
        sys.exit(1)


def download_vad_model():
    """Silero VAD modelini yerel klasöre indirir ve önbelleğe alır."""
    vad_dir = MODEL_DIR / "vad"
    vad_dir.mkdir(parents=True, exist_ok=True)
    print(f"[VAD] Silero VAD Modeli indiriliyor -> {vad_dir}")

    try:
        from silero_vad import load_silero_vad

        model = load_silero_vad()
        print("[OK] Silero VAD Modeli yerel klasöre kaydedildi!")
    except Exception as e:
        print(f"[ERROR] Silero VAD indirme hatası: {e}")
        sys.exit(1)


def download_diarization_model():
    """SpeechBrain ECAPA-TDNN konuşmacı ayrıştırma modelini yerel depolama klasörüne indirir."""
    spk_dir = MODEL_DIR / "diarization" / "speechbrain_ecapa"
    spk_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] SpeechBrain ECAPA-TDNN Modeli indiriliyor/kontrol ediliyor -> {spk_dir}")

    try:
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(spk_dir),
            run_opts={"device": "cpu"},
        )
        print("[OK] SpeechBrain ECAPA-TDNN Modeli yerel klasöre kaydedildi!")
    except Exception as e:
        print(f"[ERROR] Standard SpeechBrain indirme hatası: {e}")
        sys.exit(1)


def download_pyannote_community1_model():
    """Pyannote 4.x Community-1 konuşmacı ayrıştırma modelini yerel depolama klasörüne indirir."""
    pyannote_dir = MODEL_DIR / "diarization" / "pyannote-community-1"
    pyannote_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Pyannote Community-1 Modeli indiriliyor/kontrol ediliyor -> {pyannote_dir}")

    try:
        import os
        from huggingface_hub import snapshot_download

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("[WARNING] Pyannote Community-1 indirmesi için HF_TOKEN ortam değişkeni gereklidir. Pyannote indirmesi atlanıyor.")
            return

        print("[Pyannote] Community-1 Pipeline reposu indiriliyor (pyannote/speaker-diarization-3.1)...")
        snapshot_download(
            repo_id="pyannote/speaker-diarization-3.1",
            local_dir=str(pyannote_dir),
            token=token,
        )
        print("[OK] Pyannote Community-1 yerel çevrimdışı modeller başarıyla yüklendi!")
    except Exception as e:
        print(f"[ERROR] Pyannote Community-1 indirme hatası: {e}")
        sys.exit(1)


def download_sherpa_onnx_model():
    """Sherpa-ONNX (pyannote segmentation-3.0 ONNX ve WeSpeaker/3DSpeaker ONNX embedding) modellerini yerel klasöre indirir."""
    sherpa_dir = MODEL_DIR / "diarization" / "sherpa_onnx"
    sherpa_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Sherpa-ONNX Modelleri indiriliyor/kontrol ediliyor -> {sherpa_dir}")

    try:
        from huggingface_hub import hf_hub_download

        print("[Sherpa-ONNX] Segmentation modeli indiriliyor (csukuangfj/sherpa-onnx-pyannote-segmentation-3-0)...")
        hf_hub_download(
            repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            filename="model.onnx",
            local_dir=str(sherpa_dir),
        )

        print("[Sherpa-ONNX] Embedding modeli indiriliyor (csukuangfj/speaker-embedding-models)...")
        hf_hub_download(
            repo_id="csukuangfj/speaker-embedding-models",
            filename="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            local_dir=str(sherpa_dir),
        )

        print("[OK] Sherpa-ONNX çevrimdışı ONNX modelleri başarıyla yüklendi!")
    except Exception as e:
        print(f"[ERROR] Sherpa-ONNX indirme hatası: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("[START] Çevrimdışı Yapay Zeka Model İndirme Başlatılıyor...")
    download_stt_model("small")
    download_vad_model()
    download_diarization_model()
    download_pyannote_community1_model()
    download_sherpa_onnx_model()
    print("[DONE] Çevrimdışı model paketleme tamamlandı!")

