import argparse
import gc
import json
import os
import sys
import time

# UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")

import soundfile as sf
from audio_analyzer.services.pipeline_factory import get_shared_pipeline


def analyze_file(pipeline, file_path: str, is_new_config: bool = True) -> dict:
    os.environ["WHISPER_LOG_PROB_THRESHOLD"] = "-1.0" if is_new_config else "none"
    os.environ["WHISPER_TEMPERATURE_FALLBACK"] = "true" if is_new_config else "false"
    os.environ["ENABLE_VAD_FILTER"] = "false"
    os.environ["ENABLE_STT_SAFETY_NET"] = "true" if is_new_config else "false"

    info = sf.info(file_path)
    audio_dur = info.duration

    gc.collect()
    t0 = time.monotonic()
    utterances, detected_lang, overlap_summary = pipeline.process(file_path)
    elapsed = time.monotonic() - t0

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--find", type=str, default=None)
    parser.add_argument("--output", type=str, default="benchmark_24_results.json")
    args = parser.parse_args()

    if args.find:
        matched = [
            os.path.join("Yeni klasör", f)
            for f in os.listdir("Yeni klasör")
            if args.find.lower() in f.lower()
        ]
        if not matched:
            print(f"Hata: '{args.find}' ile eslesen dosya bulunamadi.")
            sys.exit(1)
        fpath = matched[0]
    elif args.file:
        fpath = args.file
    else:
        print("Hata: --file veya --find belirtilmelidir.")
        sys.exit(1)

    fname = os.path.basename(fpath)
    print(f"Tek dosya isleniyor: {fname.encode('ascii', 'replace').decode('ascii')}")

    pipeline = get_shared_pipeline()

    print("   -> [1/2] Once (Baseline)...", flush=True)
    before_res = analyze_file(pipeline, fpath, is_new_config=False)
    print(f"      Bitti ({before_res['elapsed']:.1f}s, RTF: {before_res['rtf']:.2f}, kelime: {before_res['words']})")

    print("   -> [2/2] Yeni Ayarlar...", flush=True)
    after_res = analyze_file(pipeline, fpath, is_new_config=True)
    print(f"      Bitti ({after_res['elapsed']:.1f}s, RTF: {after_res['rtf']:.2f}, kelime: {after_res['words']})")

    results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    results[fname] = {
        "name": fname,
        "duration": before_res["audio_dur"],
        "before": before_res,
        "after": after_res,
    }

    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"[OK] {fname} basariyla kaydedildi.")


if __name__ == "__main__":
    main()
