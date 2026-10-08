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
        import yaml
        from huggingface_hub import snapshot_download

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("[WARNING] Pyannote indirmesi için HF_TOKEN ortam değişkeni gereklidir.")
            return

        print("[Pyannote] Pipeline reposu indiriliyor (pyannote/speaker-diarization-3.1)...")
        snapshot_download(
            repo_id="pyannote/speaker-diarization-3.1",
            local_dir=str(pyannote_dir),
            token=token,
        )

        seg_dir = pyannote_dir / "segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        print("[Pyannote] Segmentation modeli indiriliyor (pyannote/segmentation-3.0)...")
        snapshot_download(
            repo_id="pyannote/segmentation-3.0",
            local_dir=str(seg_dir),
            token=token,
        )

        emb_dir = pyannote_dir / "embedding"
        emb_dir.mkdir(parents=True, exist_ok=True)
        print("[Pyannote] Embedding modeli indiriliyor (speechbrain/spkrec-ecapa-voxceleb)...")
        snapshot_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            local_dir=str(emb_dir),
            token=token,
        )

        config_path = pyannote_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if "pipeline" in config_data and "params" in config_data["pipeline"]:
                params = config_data["pipeline"]["params"]
                seg_bin = seg_dir / "pytorch_model.bin"
                if not seg_bin.exists():
                    seg_bin = seg_dir / "model.safetensors"
                params["segmentation"] = str(seg_bin) if seg_bin.exists() else str(seg_dir)
                params["embedding"] = str(emb_dir)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False)

            print("[OK] Pyannote 3.1 config.yaml yerel model yollarıyla güncellendi!")
        print("[OK] Pyannote 3.1 yerel çevrimdışı modeller başarıyla yüklendi!")
    except Exception as e:
        print(f"[WARNING] Pyannote indirme uyarısı: {e}")


if __name__ == "__main__":
    print("[START] Çevrimdışı Yapay Zeka Model İndirme Başlatılıyor...")
    download_stt_model("small")
    download_vad_model()
    download_diarization_model()
    download_pyannote_model()
    print("[DONE] Çevrimdışı model paketleme tamamlandı!")
