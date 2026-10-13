import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")

import soundfile as sf
from audio_analyzer.config import get_settings
from audio_analyzer.services.pipeline_factory import get_shared_pipeline, reset_pipeline_cache

# Suppress excessive info logs during bulk benchmark
logging.getLogger("audio_analyzer").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


import gc


def analyze_file(pipeline, file_path: str, is_new_config: bool = True) -> dict:
    os.environ["WHISPER_LOG_PROB_THRESHOLD"] = "-1.0" if is_new_config else "none"
    os.environ["WHISPER_TEMPERATURE_FALLBACK"] = "true" if is_new_config else "false"
    os.environ["ENABLE_VAD_FILTER"] = "false"
    os.environ["ENABLE_STT_SAFETY_NET"] = "true" if is_new_config else "false"

    info = sf.info(file_path)
    audio_dur = info.duration

    gc.collect()
    max_retries = 2
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            utterances, detected_lang, overlap_summary = pipeline.process(file_path)
            elapsed = time.monotonic() - t0
            break
        except Exception as e:
            if attempt < max_retries - 1 and "mkl_malloc" in str(e).lower():
                print(f"\n[UYARI] Bellek tahsis hatasi, 3 sn beklenip gc.collect yapiliyor (Deneme {attempt+1})...")
                gc.collect()
                time.sleep(3)
                continue
            raise

    rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

    total_words = sum(len(u.text.strip().split()) for u in utterances)
    unique_speakers = len(set(u.speaker_id for u in utterances))
    num_blocks = len(utterances)

    all_untranscribed = getattr(overlap_summary, "all_untranscribed_seconds", 0.0)
    untranscribed_2s = getattr(overlap_summary, "untranscribed_speech_seconds", 0.0)
    coverage_pct = getattr(overlap_summary, "speech_coverage_percentage", 100.0)
    high_temp_cnt = getattr(overlap_summary, "high_temperature_segment_count", 0)
    low_conf_utts = getattr(overlap_summary, "low_confidence_utterance_count", 0)

    sample = " // ".join(f"[{u.start_time:04.1f}-{u.end_time:04.1f}] {u.speaker_id}: {u.text[:40]}" for u in utterances[:2])

    return {
        "audio_dur": audio_dur,
        "elapsed": elapsed,
        "rtf": rtf,
        "words": total_words,
        "speakers": unique_speakers,
        "blocks": num_blocks,
        "all_untranscribed": all_untranscribed,
        "untranscribed_2s": untranscribed_2s,
        "coverage_pct": coverage_pct,
        "high_temp_cnt": high_temp_cnt,
        "low_conf_utts": low_conf_utts,
        "sample": sample,
    }


def main():
    parser = argparse.ArgumentParser(description="24 dosyalik tam karsilastirma benchmarki")
    parser.add_argument("--folder", type=str, required=True, help="Ses dosyalarinin bulundugu klasor")
    parser.add_argument("--output", type=str, default="benchmark_24_results.json", help="Cikti JSON dosyasi")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Hata: Klasor bulunamadi: {folder}")
        sys.exit(1)

    audio_files = [
        f for f in sorted(folder.iterdir())
        if f.suffix.lower() in (".wav", ".mp3") and not f.name.startswith(".")
    ]

    print(f"\n==================================================================")
    print(f"  Toplam {len(audio_files)} dosya benchmarka alindi.")
    print(f"==================================================================\n")

    # Mevcut kaydedilmiş sonuçlar varsa oku (kaldığı yerden devam edebilme)
    results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Mevcut {len(results)} dosya kaydi yuklendi, kaldigi yerden devam ediliyor.\n")
        except Exception:
            results = {}

    pipeline = get_shared_pipeline()

    total_high_temp_segments = 0

    for idx, f in enumerate(audio_files, 1):
        fname = f.name
        fpath = str(f)

        if fname in results and "before" in results[fname] and "after" in results[fname]:
            print(f"[{idx}/{len(audio_files)}] {fname[:40]:40s} (Onceden islenmis, atlandi)")
            total_high_temp_segments += results[fname]["after"].get("high_temp_cnt", 0)
            continue

        print(f"[{idx}/{len(audio_files)}] Isleniyor: {fname}...")

        try:
            # 1. Once (Baseline)
            print(f"   -> [1/2] Once (Baseline)...", end="", flush=True)
            t_start = time.monotonic()
            before_res = analyze_file(pipeline, fpath, is_new_config=False)
            print(f" bitti ({time.monotonic() - t_start:.1f}s, RTF: {before_res['rtf']:.2f}, kelime: {before_res['words']})")

            # 2. Yeni Ayarlar
            print(f"   -> [2/2] Yeni Ayarlar...", end="", flush=True)
            t_start = time.monotonic()
            after_res = analyze_file(pipeline, fpath, is_new_config=True)
            print(f" bitti ({time.monotonic() - t_start:.1f}s, RTF: {after_res['rtf']:.2f}, kelime: {after_res['words']})")

            total_high_temp_segments += after_res.get("high_temp_cnt", 0)

            results[fname] = {
                "name": fname,
                "duration": before_res["audio_dur"],
                "before": before_res,
                "after": after_res,
            }

            # Anlik kaydet
            with open(args.output, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)

        except Exception as exc:
            print(f" HATA: {exc}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # TABLOLARI YAZDIR
    # =========================================================================
    print("\n" + "=" * 120)
    print("YENI KLASOR 24 DOSYA TAM KARSILASTIRMA BENCHMARK TABLOSU")
    print("=" * 120)

    print("\n### 1. Transkriptsiz Konuşma ve Kapsama Oranı (Önce vs Yeni)")
    print("| # | Dosya Adı | Süre | Tüm Boşluk (Önce) | Tüm Boşluk (Yeni) | 2s+ Boşluk (Önce) | 2s+ Boşluk (Yeni) | Kapsama (Önce) | Kapsama (Yeni) |")
    print("|---|-----------|------|-------------------|-------------------|-------------------|-------------------|----------------|----------------|")
    for i, (k, r) in enumerate(results.items(), 1):
        b = r["before"]
        a = r["after"]
        short_name = k[:30]
        print(
            f"| {i:02d} | `{short_name}` | {r['duration']:.1f}s "
            f"| {b['all_untranscribed']:.2f}s | **{a['all_untranscribed']:.2f}s** "
            f"| {b['untranscribed_2s']:.2f}s | **{a['untranscribed_2s']:.2f}s** "
            f"| %{b['coverage_pct']:.1f} | **%{a['coverage_pct']:.1f}** |"
        )

    print("\n### 2. Kelime, Konuşmacı, Blok Sayısı ve RTF (Önce vs Yeni)")
    print("| # | Dosya Adı | Kelime (Önce/Yeni) | Konuşmacı (Önce/Yeni) | Blok (Önce/Yeni) | RTF (Önce/Yeni) | Düşük Güven (Yeni) |")
    print("|---|-----------|--------------------|-----------------------|------------------|-----------------|--------------------|")
    for i, (k, r) in enumerate(results.items(), 1):
        b = r["before"]
        a = r["after"]
        short_name = k[:30]
        print(
            f"| {i:02d} | `{short_name}` "
            f"| {b['words']} / **{a['words']}** "
            f"| {b['speakers']} / **{a['speakers']}** "
            f"| {b['blocks']} / **{a['blocks']}** "
            f"| {b['rtf']:.2f} / **{a['rtf']:.2f}** "
            f"| {a['high_temp_cnt']} seg ({a['low_conf_utts']} blok) |"
        )

    print(f"\nToplam Yüksek Temperature (>=0.6) ile Çözülen Segment Sayısı: {total_high_temp_segments}")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
