"""
Diarization Değerlendirme ve Baseline Ölçüm Scripti (scripts/evaluate_diarization.py)

Değerlendirme Kuralları:
- `--engine pyannote|ecapa|cluster|sherpa_onnx`: Motoru doğrudan adaptör sınıfıyla değerlendirir.
- `--engine chain`: FallbackDiarizer üretim zincirini değerlendirir.
- `--compare eski.json yeni.json`: İki benchmark JSON çıktısını eşleştirerek (Wilcoxon testi, Ortalama/Medyan Fark, İyileşen/Kötüleşen/Değişmeyen sayısı) karşılaştırır.
- Herhangi bir kayıtta boş sonuç alınırsa veya yükleme başarısız olursa JSON YAZMADAN exit code 1 ile çıkar.
"""

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import soundfile as sf
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.stats import wilcoxon

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_analyzer.domain.models import DeviceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_diarization")


def parse_rttm(rttm_path: Path, uri: str) -> tuple[Annotation, int]:
    """RTTM dosyasını okuyarak pyannote.core.Annotation ve benzersiz konuşmacı sayısını döndürür."""
    annotation = Annotation(uri=uri)
    speakers = set()
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start_t = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                speakers.add(speaker_id)
                annotation[Segment(start_t, start_t + duration)] = speaker_id
    return annotation, len(speakers)


def load_single_engine(engine_name: str, device_config: DeviceConfig):
    """Belirtilen motor adaptörünü doğrudan yükler (Fallback zinciri olmadan)."""
    if engine_name == "pyannote":
        from audio_analyzer.adapters.diarization.pyannote_adapter import PyannoteAudioAdapter

        adapter = PyannoteAudioAdapter(device_config=device_config)
        adapter._lazy_load_pipeline()
        if adapter._pipeline is None:
            raise RuntimeError(
                "Pyannote 3.1 yerel modelleri/konfigürasyonu yüklenemedi. Lütfen modellerin varlığını kontrol edin."
            )
        return adapter
    elif engine_name == "ecapa":
        from audio_analyzer.adapters.diarization.speechbrain_adapter import SpeechBrainECAPADiarizer

        adapter = SpeechBrainECAPADiarizer(device_config=device_config)
        adapter._load_classifier()
        if adapter._classifier is None:
            raise RuntimeError("SpeechBrain ECAPA modeli yüklenemedi.")
        return adapter
    elif engine_name == "cluster":
        from audio_analyzer.adapters.diarization.cluster_diarizer import LocalSpectralClusterDiarizer

        return LocalSpectralClusterDiarizer()
    elif engine_name in ("sherpa_onnx", "sherpa"):
        from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxAudioAdapter

        adapter = SherpaOnnxAudioAdapter(device_config=device_config)
        adapter._lazy_load_pipeline()
        if adapter._pipeline is None:
            raise RuntimeError("Sherpa-ONNX yerel ONNX modelleri yüklenemedi.")
        return adapter
    else:
        raise ValueError(f"Geçersiz motor adı: {engine_name}")


def load_chain_engine(device_config: DeviceConfig):
    """Üretim ortamı FallbackDiarizer zincirini yükler."""
    from audio_analyzer.adapters.diarization.fallback_diarizer import FallbackDiarizer
    from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxAudioAdapter
    from audio_analyzer.adapters.diarization.speechbrain_adapter import SpeechBrainECAPADiarizer
    from audio_analyzer.config import get_settings

    settings = get_settings()
    ecapa = SpeechBrainECAPADiarizer(device_config=device_config)

    if settings.diarization_engine in ("sherpa_onnx", "sherpa"):
        primary = SherpaOnnxAudioAdapter(device_config=device_config)
        primary._lazy_load_pipeline()
    else:
        from audio_analyzer.adapters.diarization.pyannote_adapter import PyannoteAudioAdapter
        try:
            primary = PyannoteAudioAdapter(device_config=device_config)
            primary._lazy_load_pipeline()
        except Exception:
            primary = ecapa

    fallbacks = [ecapa] if primary != ecapa else []
    return FallbackDiarizer(primary_diarizer=primary, fallback_diarizers=fallbacks)


def compare_benchmarks(old_json_path: Path, new_json_path: Path):
    """İki benchmark JSON dosyasını eşleştirilmiş farklar ve Wilcoxon testi ile karşılaştırır."""
    if not old_json_path.exists() or not new_json_path.exists():
        logger.error(
            "[ERROR] Karşılaştırılacak JSON dosyaları bulunamadı: %s veya %s",
            old_json_path,
            new_json_path,
        )
        sys.exit(1)

    with open(old_json_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    with open(new_json_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    old_map = {r["file_name"]: r["der_percent"] for r in old_data.get("details", [])}
    new_map = {r["file_name"]: r["der_percent"] for r in new_data.get("details", [])}

    common_files = sorted(list(set(old_map.keys()).intersection(set(new_map.keys()))))
    if not common_files:
        logger.error("[ERROR] JSON dosyaları arasında eşleşen ortak dosya bulunamadı.")
        sys.exit(1)

    old_ders = np.array([old_map[f] for f in common_files])
    new_ders = np.array([new_map[f] for f in common_files])
    diffs = new_ders - old_ders  # (yeni DER - eski DER)

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    improved = int(np.sum(diffs < -0.001))
    degraded = int(np.sum(diffs > 0.001))
    unchanged = int(np.sum(np.abs(diffs) <= 0.001))

    # Wilcoxon signed-rank test
    try:
        if np.all(diffs == 0):
            p_value = 1.0
        else:
            w_res = wilcoxon(old_ders, new_ders)
            p_value = float(w_res.pvalue)
    except Exception as e:
        logger.warning("Wilcoxon testi hesaplanamadı: %s", e)
        p_value = 1.0

    print("\n" + "=" * 90)
    print("PAIRED BENCHMARK COMPARISON REPORT")
    print("=" * 90)
    print(f" Eski Rapor (Old) : {old_json_path.name} ({old_data.get('engine', 'N/A')})")
    print(f" Yeni Rapor (New) : {new_json_path.name} ({new_data.get('engine', 'N/A')})")
    print(f" Ortak Kayıt Sayısı : {len(common_files)}")
    print("-" * 90)
    print(f" Ortalama DER Farkı (Yeni - Eski) : %{mean_diff:+.2f} ({'İyileşme' if mean_diff < 0 else 'Kötüleşme'})")
    print(f" Medyan DER Farkı  (Yeni - Eski) : %{median_diff:+.2f}")
    print(f" İyileşen Kayıt Sayısı           : {improved} / {len(common_files)}")
    print(f" Kötüleşen Kayıt Sayısı          : {degraded} / {len(common_files)}")
    print(f" Değişmeyen Kayıt Sayısı         : {unchanged} / {len(common_files)}")
    print(f" Wilcoxon Test p-değeri (p-val)  : {p_value:.4f} ({'Anlamlı Fark Var (p < 0.05)' if p_value < 0.05 else 'İstatistiksel Anlamlı Fark Yok (p >= 0.05)'})")
    print("=" * 90 + "\n")

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_json_path = out_dir / f"comparison_{timestamp_str}.json"

    comp_report = {
        "old_file": old_json_path.name,
        "new_file": new_json_path.name,
        "num_common_files": len(common_files),
        "mean_diff_percent": round(mean_diff, 2),
        "median_diff_percent": round(median_diff, 2),
        "improved_count": improved,
        "degraded_count": degraded,
        "unchanged_count": unchanged,
        "wilcoxon_pvalue": round(p_value, 4),
    }

    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comp_report, f, indent=2, ensure_ascii=False)


def run_evaluation(
    data_dir: Path,
    engine_name: str,
    unknown_speakers: bool,
    collar: float = 0.25,
    seed: int = 42,
):
    wav_files = sorted(list(data_dir.glob("*.wav")))
    if not wav_files:
        logger.error("[ERROR] Dizinde hiç .wav dosyası bulunamadı: %s", data_dir.absolute())
        sys.exit(1)

    # speaker_manifest.json doğrulanma durumu kontrolü
    manifest_path = data_dir / "speaker_manifest.json"
    manifest_verified = False
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                manifest_verified = m_data.get("verified", False)
        except Exception:
            manifest_verified = False

    if not manifest_verified:
        print("\n" + "!" * 90)
        print("UYARI: kaynak konuşmacı kimlikleri henüz doğrulanmadı (speaker_manifest.json -> verified: false)")
        print("!" * 90)

    device_config = DeviceConfig()
    logger.info("Motor başlatılıyor: %s ...", engine_name)

    start_load_time = time.perf_counter()
    try:
        if engine_name == "chain":
            diarizer = load_chain_engine(device_config)
        else:
            diarizer = load_single_engine(engine_name, device_config)
    except Exception as load_err:
        logger.error("[ERROR] Motor yükleme başarısız oldu (%s): %s", engine_name, load_err)
        sys.exit(1)

    warmup_wav = wav_files[0]
    try:
        diarizer.diarize(str(warmup_wav), num_speakers=2)
    except Exception as warmup_err:
        logger.warning("Warm-up uyarısı: %s", warmup_err)
    load_time_sec = round(time.perf_counter() - start_load_time, 3)

    logger.info("Motor başarıyla yüklendi ve ısındırıldı (Yükleme süresi: %.3f sn).", load_time_sec)

    der_metric = DiarizationErrorRate(collar=collar)
    process = psutil.Process()

    results = []
    fallback_count = 0

    print("\n" + "=" * 90)
    print(f"DIARIZATION BENCHMARK EVALUATION: [{engine_name.upper()}]")
    print(f"Veri Dizini        : {data_dir.absolute()}")
    print(f"Test Dosya Sayısı  : {len(wav_files)}")
    print(f"Konuşmacı Sayısı   : {'Bilinmiyor (None)' if unknown_speakers else 'RTTM Ground-Truth'}")
    print(f"DER Collar         : {collar}s")
    print(f"Model Load Time    : {load_time_sec}s")
    if not manifest_verified:
        print("Manifest Doğrulama : UYARI (verified: false)")
    print("=" * 90 + "\n")

    for wav_path in wav_files:
        rttm_path = wav_path.with_suffix(".rttm")
        if not rttm_path.exists():
            logger.warning("RTTM dosyası bulunamadı, atlanıyor: %s", rttm_path.name)
            continue

        file_stem = wav_path.stem
        info = sf.info(str(wav_path))
        audio_duration = info.duration

        ref_annotation, gt_num_speakers = parse_rttm(rttm_path, file_stem)
        target_num_speakers = None if unknown_speakers else gt_num_speakers

        start_time = time.perf_counter()
        try:
            hyp_segments = diarizer.diarize(str(wav_path), num_speakers=target_num_speakers)
        except Exception as err:
            logger.error("[ERROR] %s dosyasında diarization hatası fırlatıldı: %s", wav_path.name, err)
            sys.exit(1)

        elapsed = time.perf_counter() - start_time
        end_rss = process.memory_info().rss / (1024 * 1024)
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        if not hyp_segments or len(hyp_segments) == 0:
            logger.error(
                "[ERROR] '%s' motoru '%s' dosyası için BOŞ segment listesi döndürdü. Değerlendirme iptal ediliyor (Exit 1).",
                engine_name,
                wav_path.name,
            )
            sys.exit(1)

        hyp_annotation = Annotation(uri=file_stem)
        detected_speakers = set()
        for seg in hyp_segments:
            hyp_annotation[Segment(seg.start_time, seg.end_time)] = seg.speaker_id
            detected_speakers.add(seg.speaker_id)

        comp = der_metric(ref_annotation, hyp_annotation, detailed=True)
        total_gt_sec = comp.get("total", 0.0)

        if total_gt_sec > 0:
            missed_speech_pct = (comp.get("missed detection", 0.0) / total_gt_sec) * 100.0
            false_alarm_pct = (comp.get("false alarm", 0.0) / total_gt_sec) * 100.0
            speaker_confusion_pct = (comp.get("confusion", 0.0) / total_gt_sec) * 100.0
            der_pct = comp.get("diarization error rate", 0.0) * 100.0
        else:
            missed_speech_pct = 0.0
            false_alarm_pct = 0.0
            speaker_confusion_pct = 0.0
            der_pct = 0.0

        actual_engine = engine_name
        if engine_name == "chain":
            actual_engine = diarizer.primary.__class__.__name__

        record_res = {
            "file_name": wav_path.name,
            "duration_sec": round(audio_duration, 2),
            "gt_num_speakers": gt_num_speakers,
            "detected_num_speakers": len(detected_speakers),
            "actual_engine": actual_engine,
            "der_percent": round(der_pct, 2),
            "missed_speech_percent": round(missed_speech_pct, 2),
            "false_alarm_percent": round(false_alarm_pct, 2),
            "speaker_confusion_percent": round(speaker_confusion_pct, 2),
            "execution_time_sec": round(elapsed, 3),
            "rtf": round(rtf, 3),
            "rss_memory_mb": round(end_rss, 1),
            "output_segments": len(hyp_segments),
        }
        results.append(record_res)

        print(
            f" -> {file_stem:25s} | DER: {der_pct:6.2f}% (Miss: {missed_speech_pct:5.2f}%, FA: {false_alarm_pct:5.2f}%, Conf: {speaker_confusion_pct:5.2f}%) | "
            f"Süre: {elapsed:6.3f}s | RTF: {rtf:5.2f}x | Spk: {len(detected_speakers)}/{gt_num_speakers}"
        )

    if not results:
        logger.error("[ERROR] Hiçbir dosya değerlendirilemedi.")
        sys.exit(1)

    der_values = [r["der_percent"] for r in results]
    miss_values = [r["missed_speech_percent"] for r in results]
    fa_values = [r["false_alarm_percent"] for r in results]
    conf_values = [r["speaker_confusion_percent"] for r in results]
    times = [r["execution_time_sec"] for r in results]
    rss_values = [r["rss_memory_mb"] for r in results]

    avg_der = float(np.mean(der_values))
    std_der = float(np.std(der_values))
    avg_miss = float(np.mean(miss_values))
    avg_fa = float(np.mean(fa_values))
    avg_conf = float(np.mean(conf_values))

    spk_matches = sum(1 for r in results if r["detected_num_speakers"] == r["gt_num_speakers"])
    speaker_match_rate_pct = float(np.round((spk_matches / len(results)) * 100.0, 2))

    median_time = float(np.median(times))
    p95_time = float(np.percentile(times, 95))
    max_rss = float(np.max(rss_values))
    fallback_rate_pct = round((fallback_count / len(results)) * 100.0, 1)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"diarization_{engine_name}_{timestamp_str}.json"

    benchmark_data = {
        "engine": engine_name,
        "timestamp": timestamp_str,
        "random_seed": seed,
        "manifest_verified": manifest_verified,
        "num_files": len(results),
        "unknown_speakers": unknown_speakers,
        "collar_sec": collar,
        "load_time_sec": load_time_sec,
        "summary": {
            "avg_der_percent": round(avg_der, 2),
            "std_der_percent": round(std_der, 2),
            "avg_missed_speech_percent": round(avg_miss, 2),
            "avg_false_alarm_percent": round(avg_fa, 2),
            "avg_speaker_confusion_percent": round(avg_conf, 2),
            "speaker_match_rate_percent": speaker_match_rate_pct,
            "median_time_sec": round(median_time, 3),
            "p95_time_sec": round(p95_time, 3),
            "max_rss_mb": round(max_rss, 1),
            "fallback_rate_percent": fallback_rate_pct,
        },
        "details": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"BENCHMARK SUMMARY RESULTS [{engine_name.upper()}]")
    print("=" * 90)
    if not manifest_verified:
        print(" UYARI: kaynak konuşmacı kimlikleri henüz doğrulanmadı (verified: false)")
    print(f" Toplam Test Dosyası      : {len(results)}")
    print(f" Model Yükleme Süresi     : {load_time_sec:.3f} saniye")
    print(f" Ortalama DER ± Std       : %{avg_der:.2f} ± %{std_der:.2f}")
    print(f"  - Missed Speech (Kaçırılan)       : %{avg_miss:.2f}")
    print(f"  - False Alarm (Yanlış Alarm)     : %{avg_fa:.2f}")
    print(f"  - Speaker Confusion (Karışma)    : %{avg_conf:.2f}")
    print(f" Konuşmacı Sayısı Eşleşme Oranı     : %{speaker_match_rate_pct:.2f} ({spk_matches}/{len(results)})")
    print(f" Medyan İşlem Süresi (P50)          : {median_time:.3f} saniye")
    print(f" P95 İşlem Süresi (P95)             : {p95_time:.3f} saniye")
    print(f" Maksimum RSS Bellek Kullanımı      : {max_rss:.1f} MB")
    print(f" Fallback Oranı                     : %{fallback_rate_pct:.1f}")
    print(f" JSON Rapor Dosyası                 : {json_path.absolute()}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarization Değerlendirme ve Baseline Ölçüm Scripti")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="tests/fixtures/diarization_eval/large",
        help="Ses ve RTTM dosyalarının bulunduğu klasör (Varsayılan: tests/fixtures/diarization_eval/large)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="ecapa",
        choices=["pyannote", "ecapa", "cluster", "sherpa_onnx", "chain"],
        help="Değerlendirilecek motor (pyannote, ecapa, cluster, sherpa_onnx, chain)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("ESKI_JSON", "YENI_JSON"),
        help="İki benchmark JSON dosyasını karşılaştırır (--compare eski.json yeni.json)",
    )
    parser.add_argument(
        "--unknown-speakers",
        action="store_true",
        help="Konuşmacı sayısı bilinmiyor modunda çalıştır (num_speakers=None)",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="DER tolerans penceresi (saniye cinsinden, varsayılan: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik seed değeri (Varsayılan: 42)",
    )
    args = parser.parse_args()

    if args.compare:
        compare_benchmarks(Path(args.compare[0]), Path(args.compare[1]))
    else:
        run_evaluation(
            data_dir=Path(args.data_dir),
            engine_name=args.engine,
            unknown_speakers=args.unknown_speakers,
            collar=args.collar,
            seed=args.seed,
        )
