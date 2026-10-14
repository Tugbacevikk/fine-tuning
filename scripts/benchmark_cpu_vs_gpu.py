#!/usr/bin/env python3
"""
CPU vs GPU (CUDA) Karşılaştırmalı Performans Benchmark Betiği.
Donanım: Intel Core i5-11400H vs NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)

Kullanım:
    python scripts/benchmark_cpu_vs_gpu.py --audio tests/fixtures/diarization_eval/sample1_2spk.wav --model medium --repeat 1
    python scripts/benchmark_cpu_vs_gpu.py --audio-dir "Yeni klasör" --model medium --repeat 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Windows console encoding fix
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run_worker_subcommand(device_target: str, audio_path: str, model_size: str, repeat: int) -> dict:
    """
    Belirli bir cihazda (CPU veya GPU) tek bir ses dosyası için benchmark'ı
    temiz bir Python alt sürecinde (subprocess) yürütür.
    Bu sayede bellek şişmesi (OOM / mkl_malloc) ve CUDA context çakışmaları önlenir.
    """
    python_exe = sys.executable
    cmd = [
        python_exe,
        __file__,
        "--internal-run-single",
        "--target-device", device_target,
        "--audio", str(audio_path),
        "--model", model_size,
        "--repeat", str(repeat),
    ]

    env = os.environ.copy()
    if device_target == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # GPU için CUDA görünür
        env.pop("CUDA_VISIBLE_DEVICES", None)

    # D sürücüsündeki önbellek ve temp klasörleri
    env["TMP"] = "D:\\temp"
    env["TEMP"] = "D:\\temp"
    env["PIP_CACHE_DIR"] = "D:\\pip_cache"

    res = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if res.returncode != 0:
        error_msg = res.stderr.strip() or res.stdout.strip() or f"Process exited with code {res.returncode}"
        return {"success": False, "error": error_msg, "device": device_target}

    try:
        # Son JSON satırını parse et
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                data = json.loads(line)
                data["success"] = True
                return data
        return {"success": False, "error": f"JSON çıktısı bulunamadı: {res.stdout[-300:]}"}
    except Exception as e:
        return {"success": False, "error": f"Çıktı çözümlenemedi ({e}): {res.stdout[-300:]}"}


def execute_single_device_run(device_target: str, audio_path: str, model_size: str, repeat: int):
    """Alt süreç içinde çağrılan tek cihazlık ölçüm fonksiyonu."""
    import soundfile as sf
    import numpy as np

    info = sf.info(audio_path)
    audio_duration = float(info.duration)

    # 1. Model Yükleme Süresi ve Cihaz Tespiti
    t_model_start = time.perf_counter()
    from audio_analyzer.domain.models import DeviceConfig
    from audio_analyzer.adapters.stt.faster_whisper_adapter import FasterWhisperAdapter
    from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import SherpaOnnxAudioAdapter
    from audio_analyzer.services.pipeline import AudioAnalysisPipeline
    from audio_analyzer.config import get_settings

    settings = get_settings()
    threads = settings.worker_cpu_threads

    if device_target == "gpu":
        device_cfg = DeviceConfig(device="cuda", compute_type="float16")
    else:
        device_cfg = DeviceConfig(device="cpu", compute_type="int8")
    actual_device = device_cfg.device
    compute_type = device_cfg.compute_type

    # STT Model
    stt_engine = FasterWhisperAdapter(
        model_size=model_size,
        device_config=device_cfg,
        cpu_threads=threads if actual_device == "cpu" else 4,
    )
    stt_engine._lazy_load_model()

    # Diarizer (Sherpa-ONNX kodda daima CPU)
    diarizer = SherpaOnnxAudioAdapter(num_threads=threads)
    diarizer._lazy_load_pipeline()

    pipeline = AudioAnalysisPipeline(
        stt_engine=stt_engine,
        diarizer=diarizer,
    )
    t_model_load = time.perf_counter() - t_model_start

    # GPU VRAM Ölçümü (CTranslate2 CUDA tahsisi dahil)
    peak_vram_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available() and actual_device == "cuda":
            free_b, total_b = torch.cuda.mem_get_info()
            peak_vram_mb = (total_b - free_b) / (1024 * 1024)
    except Exception:
        pass

    # 2. Isınma Turu (Hızlı Warm-up: Modeli ve CUDA çekirdeklerini ısıtmak için 1.5 sn kesit)
    try:
        if audio_duration > 2.0:
            sample_data, _ = sf.read(audio_path, stop=int(1.5 * 16000), dtype="float32")
            stt_engine.transcribe(sample_data)
        else:
            stt_engine.transcribe(audio_path)
    except Exception:
        pass

    # 3. Tekrarlı Ölçüm
    runs = []
    for rep in range(repeat):
        t0 = time.perf_counter()
        t_stt_0 = time.perf_counter()
        words, detected_lang = stt_engine.transcribe(audio_path)
        t_stt = time.perf_counter() - t_stt_0

        t_diar_0 = time.perf_counter()
        diar_segs = diarizer.diarize(audio_path)
        t_diar = time.perf_counter() - t_diar_0

        t_total = time.perf_counter() - t0

        try:
            import torch
            if torch.cuda.is_available() and actual_device == "cuda":
                free_b, total_b = torch.cuda.mem_get_info()
                peak_vram_mb = max(peak_vram_mb, (total_b - free_b) / (1024 * 1024))
        except Exception:
            pass

        runs.append({
            "total_time": t_total,
            "stt_time": t_stt,
            "diarization_time": t_diar,
            "rtf": t_total / audio_duration if audio_duration > 0 else 0.0,
        })

    # Ortalama Metrikler
    avg_total = float(np.mean([r["total_time"] for r in runs]))
    avg_stt = float(np.mean([r["stt_time"] for r in runs]))
    avg_diar = float(np.mean([r["diarization_time"] for r in runs]))
    avg_rtf = float(np.mean([r["rtf"] for r in runs]))

    output_data = {
        "device_target": device_target,
        "actual_device": actual_device,
        "compute_type": compute_type,
        "audio_path": audio_path,
        "audio_duration": audio_duration,
        "model_load_time_sec": round(t_model_load, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "avg_total_sec": round(avg_total, 2),
        "avg_stt_sec": round(avg_stt, 2),
        "avg_diarization_sec": round(avg_diar, 2),
        "avg_rtf": round(avg_rtf, 3),
        "repeats": repeat,
    }

    print(json.dumps(output_data))


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU Benchmark Runner")
    parser.add_argument("--audio", type=str, default=None, help="Test edilecek tek ses dosyası")
    parser.add_argument("--audio-dir", type=str, default=None, help="Test ses dosyalarının bulunduğu dizin")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model boyutu (varsayılan: medium)")
    parser.add_argument("--repeat", type=int, default=3, help="Tekrar sayısı (varsayılan: 3)")
    parser.add_argument("--output", type=str, default="benchmark_cpu_vs_gpu_results.json", help="Sonuç JSON dosyası")

    # İç kullanım bayrakları
    parser.add_argument("--internal-run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-device", type=str, default="cpu", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.internal_run_single:
        execute_single_device_run(args.target_device, args.audio, args.model, args.repeat)
        sys.exit(0)

    # 1. Test edilecek dosyaları belirle (Özellikle diyalog_parca_*.wav dosyaları)
    audio_files = []
    if args.audio:
        p = Path(args.audio)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            audio_files.append(str(p))
    elif args.audio_dir:
        dir_path = Path(args.audio_dir)
        if not dir_path.is_absolute():
            dir_path = PROJECT_ROOT / dir_path
        if dir_path.exists():
            for f in sorted(dir_path.glob("diyalog_parca_*.wav")):
                audio_files.append(str(f))
            if not audio_files:
                for f in sorted(dir_path.glob("*.wav")):
                    audio_files.append(str(f))

    if not audio_files:
        print("[HATA] Analiz edilecek ses dosyası bulunamadı!")
        sys.exit(1)

    print("=" * 80)
    print("        CPU vs GPU (CUDA) PERFORMANS BENCHMARK KARŞILAŞTIRMASI")
    print(f" Model: {args.model} | Tekrar: {args.repeat} | Toplam Dosya: {len(audio_files)}")
    print("=" * 80 + "\n")

    results = {
        "hardware": {
            "cpu": "Intel Core i5-11400H (6 Core / 12 Thread)",
            "gpu": "NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)",
            "ram_gb": 16,
        },
        "model_size": args.model,
        "repeats": args.repeat,
        "diarization_engine": "Sherpa-ONNX (CPU, provider='cpu')",
        "files": [],
        "summary": {},
    }

    for idx, audio_file in enumerate(audio_files, 1):
        file_name = Path(audio_file).name
        print(f"[{idx}/{len(audio_files)}] Dosya Analiz Ediliyor: {file_name}...")

        # 1. CPU Koşusu
        print(f"  -> CPU testi çalıştırılıyor ({args.repeat} tekrar)...")
        cpu_res = run_worker_subcommand("cpu", audio_file, args.model, args.repeat)

        # 2. GPU Koşusu
        print(f"  -> GPU testi çalıştırılıyor ({args.repeat} tekrar)...")
        gpu_res = run_worker_subcommand("gpu", audio_file, args.model, args.repeat)

        file_summary = {
            "file_name": file_name,
            "audio_duration_sec": cpu_res.get("audio_duration", 0.0),
            "cpu": cpu_res,
            "gpu": gpu_res,
        }

        if cpu_res.get("success") and gpu_res.get("success"):
            cpu_time = cpu_res["avg_total_sec"]
            gpu_time = gpu_res["avg_total_sec"]
            speedup = round(cpu_time / gpu_time, 2) if gpu_time > 0 else 1.0
            file_summary["speedup_ratio"] = speedup
            print(f"     ✅ Bitti: CPU = {cpu_time:.2f}s (RTF: {cpu_res['avg_rtf']:.2f}) | GPU = {gpu_time:.2f}s (RTF: {gpu_res['avg_rtf']:.2f}) | Hızlanma: {speedup}x")
        elif cpu_res.get("success") and not gpu_res.get("success"):
            file_summary["speedup_ratio"] = None
            file_summary["gpu_error"] = gpu_res.get("error")
            print(f"     ⚠️ CPU Bitti: {cpu_res['avg_total_sec']:.2f}s (RTF: {cpu_res['avg_rtf']:.2f}) | GPU Hatası: {gpu_res.get('error')[:80]}...")
        else:
            print(f"     ❌ Hata: CPU={cpu_res.get('error')} | GPU={gpu_res.get('error')}")

        results["files"].append(file_summary)

        # Her dosyadan sonra sonuçları JSON'a kaydet (Progressive save)
        out_path = PROJECT_ROOT / args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Genel Özet İstatistikleri
    valid_files = [f for f in results["files"] if f.get("cpu", {}).get("success") and f.get("gpu", {}).get("success")]
    if valid_files:
        avg_cpu_time = float(np.mean([f["cpu"]["avg_total_sec"] for f in valid_files]))
        avg_gpu_time = float(np.mean([f["gpu"]["avg_total_sec"] for f in valid_files]))
        avg_cpu_stt = float(np.mean([f["cpu"]["avg_stt_sec"] for f in valid_files]))
        avg_gpu_stt = float(np.mean([f["gpu"]["avg_stt_sec"] for f in valid_files]))
        avg_cpu_diar = float(np.mean([f["cpu"]["avg_diarization_sec"] for f in valid_files]))
        avg_gpu_diar = float(np.mean([f["gpu"]["avg_diarization_sec"] for f in valid_files]))
        avg_speedup = round(avg_cpu_time / avg_gpu_time, 2) if avg_gpu_time > 0 else 1.0
        stt_speedup = round(avg_cpu_stt / avg_gpu_stt, 2) if avg_gpu_stt > 0 else 1.0
        peak_vram = max([f["gpu"].get("peak_vram_mb", 0.0) for f in valid_files])

        results["summary"] = {
            "total_benchmark_files": len(valid_files),
            "avg_cpu_total_sec": round(avg_cpu_time, 2),
            "avg_gpu_total_sec": round(avg_gpu_time, 2),
            "avg_cpu_stt_sec": round(avg_cpu_stt, 2),
            "avg_gpu_stt_sec": round(avg_gpu_stt, 2),
            "avg_cpu_diarization_sec": round(avg_cpu_diar, 2),
            "avg_gpu_diarization_sec": round(avg_gpu_diar, 2),
            "overall_speedup_ratio": avg_speedup,
            "stt_speedup_ratio": stt_speedup,
            "peak_gpu_vram_mb": peak_vram,
            "cpu_target_met_rtf_le_1": all(f["cpu"]["avg_rtf"] <= 1.0 for f in valid_files),
            "gpu_target_met_rtf_le_1": all(f["gpu"]["avg_rtf"] <= 1.0 for f in valid_files),
        }

        # Güncel özeti kaydet
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Benchmark tamamlandı. Sonuçlar '{args.output}' dosyasına kaydedildi.")
    print("=" * 80)


if __name__ == "__main__":
    main()
