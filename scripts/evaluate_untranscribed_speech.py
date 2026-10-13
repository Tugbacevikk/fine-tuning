import argparse
import os
import sys
import time
from pathlib import Path

# UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")

import soundfile as sf
from audio_analyzer.config import get_settings
from audio_analyzer.services.pipeline_factory import get_shared_pipeline, reset_pipeline_cache


def run_benchmark_for_config(
    file_path: str,
    log_prob_threshold: str,
    temp_fallback: str,
    vad_filter: str,
    safety_net: str,
) -> dict:
    os.environ["WHISPER_LOG_PROB_THRESHOLD"] = log_prob_threshold
    os.environ["WHISPER_TEMPERATURE_FALLBACK"] = temp_fallback
    os.environ["ENABLE_VAD_FILTER"] = vad_filter
    os.environ["ENABLE_STT_SAFETY_NET"] = safety_net
    reset_pipeline_cache()

    info = sf.info(file_path)
    audio_dur = info.duration

    pipeline = get_shared_pipeline()

    t0 = time.monotonic()
    utterances, detected_lang, overlap_summary = pipeline.process(file_path)
    elapsed = time.monotonic() - t0

    rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

    total_words = sum(len(u.text.strip().split()) for u in utterances)
    sample_text = " // ".join(f"[{u.start_time:04.1f}-{u.end_time:04.1f}] {u.speaker_id}: {u.text}" for u in utterances[:3])

    untranscribed_sec = overlap_summary.untranscribed_speech_seconds if overlap_summary else 0.0
    untranscribed_cnt = overlap_summary.untranscribed_speech_count if overlap_summary else 0

    return {
        "audio_dur": audio_dur,
        "elapsed": elapsed,
        "rtf": rtf,
        "words": total_words,
        "untranscribed_sec": untranscribed_sec,
        "untranscribed_cnt": untranscribed_cnt,
        "sample": sample_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Transkriptsiz konusma ve Whisper ayarlari karsilastirmasi")
    parser.add_argument("--folder", type=str, required=True, help="Ses dosyalarinin bulundugu klasor")
    parser.add_argument("--files", type=str, default=None, help="Virgulle ayrilmis dosya listesi (opsiyonel)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Hata: Klasor bulunamadi: {folder}")
        sys.exit(1)

    if args.files:
        filenames = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        filenames = [
            f.name
            for f in sorted(folder.iterdir())
            if f.suffix.lower() in (".wav", ".mp3") and not f.name.startswith(".")
        ]

    print(f"\nToplam {len(filenames)} dosya test edilecek: {filenames}\n")

    results = []

    for fname in filenames:
        fpath = str(folder / fname)
        if not os.path.exists(fpath):
            print(f"Atlandi: {fpath} bulunamadi.")
            continue

        print(f"--> Isleniyor: {fname}")

        # 0. Once (Baseline)
        print("    [1/4] Baseline (Once: log_prob=None, temp=0.0, vad=False, safety=False)...")
        res_0 = run_benchmark_for_config(
            fpath,
            log_prob_threshold="none",
            temp_fallback="false",
            vad_filter="false",
            safety_net="false",
        )

        # 1. Adim 1.2
        print("    [2/4] Adim 1.2 (log_prob=-1.0, temp_fallback=True, vad=False, safety=False)...")
        res_1 = run_benchmark_for_config(
            fpath,
            log_prob_threshold="-1.0",
            temp_fallback="true",
            vad_filter="false",
            safety_net="false",
        )

        # 2. Adim 1.3
        print("    [3/4] Adim 1.3 (Adim 1.2 + ENABLE_VAD_FILTER=True)...")
        res_2 = run_benchmark_for_config(
            fpath,
            log_prob_threshold="-1.0",
            temp_fallback="true",
            vad_filter="true",
            safety_net="false",
        )

        # 3. Adim 1.4
        print("    [4/4] Adim 1.4 (Adim 1.2 + Guvenlik Agi ENABLE_STT_SAFETY_NET=True)...")
        res_3 = run_benchmark_for_config(
            fpath,
            log_prob_threshold="-1.0",
            temp_fallback="true",
            vad_filter="false",
            safety_net="true",
        )

        results.append({
            "name": fname,
            "dur": res_0["audio_dur"],
            "base": res_0,
            "step_12": res_1,
            "step_13": res_2,
            "step_14": res_3,
        })

    # Markdown Tablosu
    print("\n" + "=" * 90)
    print("BOLUM 1 OLCUM VE KARSILASTIRMA TABLOSU")
    print("=" * 90)

    print("\n### 1. Transkriptsiz Konuşma Süresi (sn) Karşılaştırması")
    print("| Dosya Adı | Ses Süresi | Önce (Baseline) | Adım 1.2 | Adım 1.3 (VAD) | Adım 1.4 (Güvenlik Ağı) |")
    print("|-----------|------------|-----------------|----------|----------------|-------------------------|")
    for r in results:
        print(
            f"| `{r['name'][:35]}` | {r['dur']:.1f}s "
            f"| **{r['base']['untranscribed_sec']:.2f}s** ({r['base']['untranscribed_cnt']} aralık) "
            f"| **{r['step_12']['untranscribed_sec']:.2f}s** ({r['step_12']['untranscribed_cnt']} aralık) "
            f"| **{r['step_13']['untranscribed_sec']:.2f}s** ({r['step_13']['untranscribed_cnt']} aralık) "
            f"| **{r['step_14']['untranscribed_sec']:.2f}s** ({r['step_14']['untranscribed_cnt']} aralık) |"
        )

    print("\n### 2. Kelime Sayısı ve RTF Karşılaştırması")
    print("| Dosya Adı | Kelime (Önce) | Kelime (1.2) | Kelime (1.3) | Kelime (1.4) | RTF (Önce) | RTF (1.2) | RTF (1.3) | RTF (1.4) |")
    print("|-----------|---------------|--------------|--------------|--------------|------------|-----------|-----------|-----------|")
    for r in results:
        print(
            f"| `{r['name'][:35]}` "
            f"| {r['base']['words']} "
            f"| {r['step_12']['words']} "
            f"| {r['step_13']['words']} "
            f"| {r['step_14']['words']} "
            f"| {r['base']['rtf']:.2f} "
            f"| {r['step_12']['rtf']:.2f} "
            f"| {r['step_13']['rtf']:.2f} "
            f"| {r['step_14']['rtf']:.2f} |"
        )

    print("\n### 3. Örnek Transkriptler")
    for r in results:
        print(f"\n#### `{r['name']}`:")
        print(f"- **Önce (Baseline):** {r['base']['sample']}")
        print(f"- **Adım 1.2:** {r['step_12']['sample']}")
        print(f"- **Adım 1.3 (VAD):** {r['step_13']['sample']}")
        print(f"- **Adım 1.4 (Güvenlik Ağı):** {r['step_14']['sample']}")

    # Temizlik: Ortam degiskenlerini varsayilana dondur
    os.environ["WHISPER_LOG_PROB_THRESHOLD"] = "-1.0"
    os.environ["WHISPER_TEMPERATURE_FALLBACK"] = "true"
    os.environ["ENABLE_VAD_FILTER"] = "false"
    os.environ["ENABLE_STT_SAFETY_NET"] = "true"
    reset_pipeline_cache()


if __name__ == "__main__":
    main()
