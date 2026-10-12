"""
Toplu Ses Dosyası Analiz Scripti (scripts/batch_analyze_folder.py)
Kullanıcının belirlediği klasördeki tüm ses dosyalarını güncel pipeline (Sherpa-ONNX Threshold 0.75 + SpeakerClusterMerger + Fusion Turn Binding) ile analiz eder.
Çıktıları hem metin (TXT), altyazı (SRT), veri (JSON) hem de değerlendirme (RTTM) formatında kaydeder.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Proje kökünü sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_analyzer.config import get_settings
from audio_analyzer.services.pipeline_factory import _build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_analyze")


def format_timestamp_srt(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_text(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    return f"{int(mins):02d}:{secs:05.2f}"


def run_batch_analysis(
    input_folder: Path,
    output_folder: Path | None = None,
    num_speakers: int | None = None,
):
    settings = get_settings()
    logger.info("Analiz parametreleri:")
    logger.info(" - Sherpa Cluster Threshold: %s", settings.sherpa_cluster_threshold)
    logger.info(" - Diarization Engine: %s", settings.diarization_engine)
    logger.info(" - Whisper Model: %s", settings.whisper_model_size)
    logger.info(" - Sabit Konuşmacı Sayısı: %s", num_speakers if num_speakers else "Otomatik")

    if output_folder is None:
        output_folder = input_folder / "analiz_sonuclari"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Ses dosyalarını tara
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = [f for f in sorted(input_folder.glob("*")) if f.is_file() and f.suffix.lower() in extensions]

    if not audio_files:
        logger.error("Klasörde desteklenen ses dosyası bulunamadı: %s", input_folder)
        return

    logger.info("Toplam %d ses dosyası analiz edilecek.", len(audio_files))

    # Pipeline'ı oluştur
    logger.info("Yapay zeka pipeline'ı yükleniyor...")
    pipeline = _build_pipeline()
    logger.info("Pipeline hazır!\n")

    summary_records = []

    for idx, audio_path in enumerate(audio_files, 1):
        safe_stem = re.sub(r"[^\w\-]", "_", audio_path.stem).strip("_") or f"audio_{idx}"
        legacy_stem = audio_path.stem.replace(" ", "_")

        json_path = output_folder / f"{safe_stem}.json"
        legacy_json_path = output_folder / f"{legacy_stem}.json"

        target_json = json_path if json_path.exists() else legacy_json_path if legacy_json_path.exists() else json_path

        clean_name = audio_path.name.encode("ascii", "replace").decode("ascii")

        if target_json.exists():
            print(f"[{idx}/{len(audio_files)}] Önceden tamamlandı, atlanıyor: {clean_name}")
            try:
                with open(target_json, "r", encoding="utf-8") as f_prev:
                    prev_data = json.load(f_prev)
                    summary_records.append({
                        "file_name": prev_data.get("file_name", audio_path.name),
                        "duration_sec": prev_data.get("duration_seconds", 0.0),
                        "speaker_count": prev_data.get("speaker_count", 0),
                        "speakers": prev_data.get("speakers", []),
                        "utterance_count": len(prev_data.get("utterances", [])),
                        "execution_time_sec": prev_data.get("execution_time_seconds", 0.0),
                        "rtf": prev_data.get("rtf", 0.0),
                        "language": prev_data.get("detected_language", "tr"),
                    })
            except Exception as read_err:
                logger.warning("Önceki JSON okunamadı (%s): %s", target_json.name, read_err)
            continue

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(audio_files)}] İŞLENİYOR: {clean_name}")
        print(f"{'='*80}")

        start_t = time.perf_counter()
        try:
            utterances, detected_lang, overlap_summary = pipeline.process(
                str(audio_path), num_speakers=num_speakers
            )
            elapsed = time.perf_counter() - start_t
        except Exception as e:
            logger.error("'%s' işlenirken hata oluştu: %s", audio_path.name, e)
            continue

        unique_speakers = sorted(list({u.speaker_id for u in utterances}))
        try:
            import soundfile as sf
            total_audio_duration = float(sf.info(str(audio_path)).duration)
        except Exception:
            total_audio_duration = utterances[-1].end_time if utterances else 0.0
        rtf = elapsed / total_audio_duration if total_audio_duration > 0 else 0.0

        print(f" -> Tamamlandı: {elapsed:.2f} sn | RTF: {rtf:.2f}x | Konuşmacılar ({len(unique_speakers)}): {unique_speakers} | Dil: {detected_lang}")

        # 1. TXT Çıktısı
        txt_path = output_folder / f"{safe_stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(f"Dosya: {audio_path.name}\n")
            f_txt.write(f"Tespit Edilen Dil: {detected_lang}\n")
            f_txt.write(f"Konuşmacı Sayısı: {len(unique_speakers)} ({', '.join(unique_speakers)})\n")
            f_txt.write(f"Toplam Süre: {total_audio_duration:.2f} saniye\n")
            f_txt.write("=" * 60 + "\n\n")
            for u in utterances:
                start_str = format_timestamp_text(u.start_time)
                end_str = format_timestamp_text(u.end_time)
                f_txt.write(f"[{start_str} - {end_str}] {u.speaker_id}:\n  {u.text}\n\n")

        # 2. SRT Altyazı Çıktısı
        srt_path = output_folder / f"{safe_stem}.srt"
        with open(srt_path, "w", encoding="utf-8") as f_srt:
            for s_idx, u in enumerate(utterances, 1):
                start_srt = format_timestamp_srt(u.start_time)
                end_srt = format_timestamp_srt(u.end_time)
                f_srt.write(f"{s_idx}\n{start_srt} --> {end_srt}\n[{u.speaker_id}] {u.text}\n\n")

        # 3. JSON Çıktısı
        json_path = output_folder / f"{safe_stem}.json"
        json_payload = {
            "file_name": audio_path.name,
            "detected_language": detected_lang,
            "duration_seconds": round(total_audio_duration, 2),
            "speakers": unique_speakers,
            "speaker_count": len(unique_speakers),
            "execution_time_seconds": round(elapsed, 2),
            "rtf": round(rtf, 2),
            "overlap_summary": overlap_summary.model_dump() if hasattr(overlap_summary, "model_dump") else {},
            "utterances": [
                {
                    "speaker_id": u.speaker_id,
                    "start_time": round(u.start_time, 2),
                    "end_time": round(u.end_time, 2),
                    "text": u.text,
                }
                for u in utterances
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(json_payload, f_json, indent=2, ensure_ascii=False)

        # 4. RTTM Çıktısı
        rttm_path = output_folder / f"{safe_stem}.rttm"
        with open(rttm_path, "w", encoding="utf-8") as f_rttm:
            for u in utterances:
                dur = max(0.01, round(u.end_time - u.start_time, 3))
                f_rttm.write(
                    f"SPEAKER {safe_stem} 1 {u.start_time:.3f} {dur:.3f} <NA> <NA> {u.speaker_id} <NA> <NA>\n"
                )

        summary_records.append({
            "file_name": audio_path.name,
            "duration_sec": round(total_audio_duration, 2),
            "speaker_count": len(unique_speakers),
            "speakers": unique_speakers,
            "utterance_count": len(utterances),
            "execution_time_sec": round(elapsed, 2),
            "rtf": round(rtf, 2),
            "language": detected_lang,
        })

    # Genel Özet JSON
    summary_path = output_folder / "toplu_analiz_ozeti.json"
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sherpa_cluster_threshold": settings.sherpa_cluster_threshold,
            "total_files": len(summary_records),
            "results": summary_records,
        }, f_sum, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("TOPLU ANALİZ TAMAMLANDI!")
    print(f"Çıktı Dizini: {output_folder.absolute()}")
    print(f"İncelenen Dosya Sayısı: {len(summary_records)}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klasördeki tüm ses dosyalarını analiz eder.")
    parser.add_argument(
        "--folder",
        type=str,
        default=r"C:\Users\ADIL CEVIK\Desktop\Yeni klasör",
        help="Analiz edilecek ses dosyalarının bulunduğu klasör",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Çıktıların yazılacağı hedef klasör (varsayılan: <folder>/analiz_sonuclari)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Sabit konuşmacı sayısı (varsayılan: None / Otomatik)",
    )
    args = parser.parse_args()

    run_batch_analysis(
        input_folder=Path(args.folder),
        output_folder=Path(args.output) if args.output else None,
        num_speakers=args.num_speakers,
    )
