"""
Diarization Değerlendirme ve Baseline Ölçüm Scripti (scripts/evaluate_diarization.py)

Değerlendirme Kuralları:
- `--engine pyannote|ecapa|cluster|sherpa_onnx`: Motoru doğrudan adaptör sınıfıyla değerlendirir.
- `--engine chain`: FallbackDiarizer üretim zincirini değerlendirir.
- Herhangi bir kayıtta boş sonuç alınırsa veya yükleme başarısız olursa JSON YAZMADAN exit code 1 ile çıkar.
- Model yükleme süresi (`load_time_sec`) ayrı ölçülür ve warm-up yapılır.
- DER bileşenleri (missed speech, false alarm, speaker confusion) ayrı ayrı hesaplanır.
- Çıktı: docs/benchmarks/diarization_<engine>_<tarih>.json + Konsol Özet Tablosu
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

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_analyzer.api.metrics import DIARIZATION_FALLBACK_COUNTER
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
    elif engine_name == "sherpa_onnx":
        try:
            from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxDiarizer
            return SherpaOnnxDiarizer(device_config=device_config)
        except Exception as e:
            raise RuntimeError(f"Sherpa-ONNX motoru yüklenemedi: {e}")
    else:
        raise ValueError(f"Geçersiz motor adı: {engine_name}")


def load_chain_engine(device_config: DeviceConfig):
    """Üretim ortamı FallbackDiarizer zincirini yükler."""
    from audio_analyzer.adapters.diarization.fallback_diarizer import FallbackDiarizer
    from audio_analyzer.adapters.diarization.pyannote_adapter import PyannoteAudioAdapter
    from audio_analyzer.adapters.diarization.speechbrain_adapter import SpeechBrainECAPADiarizer

    ecapa = SpeechBrainECAPADiarizer(device_config=device_config)
    try:
        primary = PyannoteAudioAdapter(device_config=device_config)
        primary._lazy_load_pipeline()
    except Exception:
        primary = ecapa

    fallbacks = [ecapa] if primary != ecapa else []
    return FallbackDiarizer(primary_diarizer=primary, fallback_diarizers=fallbacks)


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

    device_config = DeviceConfig()
    logger.info("Motor başlatılıyor: %s ...", engine_name)

    # 1. Model Yükleme ve Warm-up Süresi Ölçümü
    start_load_time = time.perf_counter()
    try:
        if engine_name == "chain":
            diarizer = load_chain_engine(device_config)
        else:
            diarizer = load_single_engine(engine_name, device_config)
    except Exception as load_err:
        logger.error("[ERROR] Motor yükleme başarısız oldu (%s): %s", engine_name, load_err)
        sys.exit(1)

    # Warm-up (Süre ölçümünden muaf)
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

        # İşlem süresi ve RSS bellek ölçümü (Yükleme süresi dahil DEĞİL)
        start_time = time.perf_counter()
        try:
            hyp_segments = diarizer.diarize(str(wav_path), num_speakers=target_num_speakers)
        except Exception as err:
            logger.error("[ERROR] %s dosyasında diarization hatası fırlatıldı: %s", wav_path.name, err)
            sys.exit(1)

        elapsed = time.perf_counter() - start_time
        end_rss = process.memory_info().rss / (1024 * 1024)
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        # SIFIR SEGMENT KONTROLÜ (Exit 1 ve JSON yazmama garantisi)
        if not hyp_segments or len(hyp_segments) == 0:
            logger.error(
                "[ERROR] '%s' motoru '%s' dosyası için BOŞ segment listesi döndürdü. Değerlendirme iptal ediliyor (Exit 1).",
                engine_name,
                wav_path.name,
            )
            sys.exit(1)

        # Hipotez Annotation ve Konuşmacı Sayısı Tespiti
        hyp_annotation = Annotation(uri=file_stem)
        detected_speakers = set()
        for seg in hyp_segments:
            hyp_annotation[Segment(seg.start_time, seg.end_time)] = seg.speaker_id
            detected_speakers.add(seg.speaker_id)

        # Detaylı DER Bileşenleri Hesaplama
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

    # Özet İstatistikler ve Standart Sapma
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
    print(f" Toplam Test Dosyası  : {len(results)}")
    print(f" Model Yükleme Süresi : {load_time_sec:.3f} saniye")
    print(f" Ortalama DER ± Std   : %{avg_der:.2f} ± %{std_der:.2f}")
    print(f"  - Missed Speech (Kaçırılan)   : %{avg_miss:.2f}")
    print(f"  - False Alarm (Yanlış Alarm) : %{avg_fa:.2f}")
    print(f"  - Speaker Confusion (Karışma): %{avg_conf:.2f}")
    print(f" Medyan İşlem Süresi (P50)      : {median_time:.3f} saniye")
    print(f" P95 İşlem Süresi (P95)         : {p95_time:.3f} saniye")
    print(f" Maksimum RSS Bellek Kullanımı  : {max_rss:.1f} MB")
    print(f" Fallback Oranı                 : %{fallback_rate_pct:.1f}")
    print(f" JSON Rapor Dosyası             : {json_path.absolute()}")
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

    run_evaluation(
        data_dir=Path(args.data_dir),
        engine_name=args.engine,
        unknown_speakers=args.unknown_speakers,
        collar=args.collar,
        seed=args.seed,
    )
