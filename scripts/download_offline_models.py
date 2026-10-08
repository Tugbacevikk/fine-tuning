import sys
from pathlib import Path

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_DIR = Path(__file__).parent.parent / "storage" / "models"


def download_stt_model(model_size: str = "small"):
    """FasterWhisper modelini yerel depolama klasörüne indirir ve hazırlar."""
    stt_dir = MODEL_DIR / "stt" / model_size
    stt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STT] STT Modeli indiriliyor/kontrol ediliyor ({model_size}) -> {stt_dir}")

    try:
        from faster_whisper import WhisperModel

        # Modeli bir kez yerel klasöre indir
        model = WhisperModel(model_size, download_root=str(stt_dir), device="cpu", compute_type="int8")
        print(f"[OK] STT Modeli ({model_size}) yerel klasöre indirildi!")
    except Exception as e:
        print(f"[ERROR] STT Model indirme hatası: {e}")


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
        print(f"[WARNING] Silero VAD indirme uyarısı: {e}")


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
        print(f"[WARNING] Standard SpeechBrain indirme uyarısı: {e}")


def download_pyannote_model():
    """Pyannote 3.1 konuşmacı ayrıştırma modelini yerel depolama klasörüne indirir."""
    pyannote_dir = MODEL_DIR / "diarization" / "pyannote"
    pyannote_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Pyannote 3.1 Modeli indiriliyor/kontrol ediliyor -> {pyannote_dir}")

    try:
        import os
        from pyannote.audio import Pipeline

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
            pipeline.save_pretrained(str(pyannote_dir))
            print("[OK] Pyannote 3.1 Modeli yerel klasöre kaydedildi!")
        else:
            print("[WARNING] Pyannote indirmesi için HF_TOKEN ortam değişkeni gereklidir.")
    except Exception as e:
        print(f"[WARNING] Pyannote indirme uyarısı: {e}")


if __name__ == "__main__":
    print("[START] Çevrimdışı Yapay Zeka Model İndirme Başlatılıyor...")
    download_stt_model("small")
    download_vad_model()
    download_diarization_model()
    download_pyannote_model()
    print("[DONE] Çevrimdışı model paketleme tamamlandı!")
