#!/usr/bin/env python3
"""
CPU Performans & Benchmark Betiği.
Ses dosyaları üzerinde her aşamanın (decode, denoise, VAD, STT, diarization, refine)
işlem süresini, RTF'yi ve ses saniyesi başına çekirdek-saniye (k) değerini ölçer.

Kullanım:
    python scripts/benchmark_cpu.py --audio-dir ./storage --models tiny,small --profiles feedback,full --threads 4
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Project root path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf


def measure_pipeline_stages(
    audio_path: str, model_size: str, profile: str, threads: int
) -> dict[str, float]:
    """Tek bir ses dosyası için pipeline aşamalarının sürelerini ve ses uzunluğunu ölçer."""
    os.environ["WHISPER_MODEL_SIZE"] = model_size
    os.environ["PIPELINE_PROFILE"] = profile
    os.environ["WORKER_CPU_THREADS"] = str(threads)
    from audio_analyzer.utils.cpu_budget import setup_cpu_thread_budget
    setup_cpu_thread_budget(threads)

    from audio_analyzer.adapters.audio.audio_converter import AudioConverterProcessor
    from audio_analyzer.adapters.audio.denoiser import DeepFilterDenoiser
    from audio_analyzer.adapters.audio.silero_vad import SileroVADProcessor
    from audio_analyzer.adapters.stt.faster_whisper_adapter import FasterWhisperAdapter
    from audio_analyzer.domain.models import DeviceConfig
    from audio_analyzer.services.fusion_engine import FusionEngine
    from audio_analyzer.services.pipeline import estimate_snr_db
    from audio_analyzer.services.semantic_refiner import SemanticRefiner

    device_config = DeviceConfig()
    stt_engine = FasterWhisperAdapter(model_size=model_size, device_config=device_config, cpu_threads=threads)
    diar_engine = os.getenv("DIARIZATION_ENGINE", "sherpa_onnx").lower()
    if diar_engine in ("sherpa_onnx", "sherpa"):
        from audio_analyzer.adapters.diarization.sherpa_onnx_adapter import (
            SherpaOnnxAudioAdapter,
        )
        diarizer = SherpaOnnxAudioAdapter(device_config=device_config, num_threads=threads)
    else:
        from audio_analyzer.adapters.diarization.speechbrain_adapter import (
            SpeechBrainECAPADiarizer,
        )
        diarizer = SpeechBrainECAPADiarizer(device_config=device_config)
    audio_processor = AudioConverterProcessor()
    denoiser = DeepFilterDenoiser(enabled=True)
    vad_processor = SileroVADProcessor()
    fusion_engine = FusionEngine()
    semantic_refiner = SemanticRefiner()

    # 1. Decode & Audio info
    t0 = time.perf_counter()
    info = sf.info(audio_path)
    audio_duration = float(info.duration)
    t_decode = time.perf_counter() - t0

    # 2. Denoise
    t0 = time.perf_counter()
    working_path = audio_path
    min_snr_db = float(os.getenv("PIPELINE_MIN_SNR_DB", "15.0"))
    temp_files = []

    should_denoise = True
    if profile == "feedback":
        try:
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            snr_val = estimate_snr_db(data, sr)
            should_denoise = snr_val < min_snr_db
        except Exception:
            should_denoise = True

    import tempfile
    import uuid

    if should_denoise and denoiser:
        denoised_path = os.path.join(tempfile.gettempdir(), f"{Path(audio_path).stem}_{uuid.uuid4().hex[:6]}.bm_denoised.wav")
        working_path = denoiser.denoise(audio_path, denoised_path)
        if working_path != audio_path:
            temp_files.append(working_path)
    t_denoise = time.perf_counter() - t0

    # 3. Audio conversion / resample
    t0 = time.perf_counter()
    processed_path = os.path.join(tempfile.gettempdir(), f"{Path(working_path).stem}_{uuid.uuid4().hex[:6]}.bm_proc.wav")
    working_path = audio_processor.normalize_and_resample(working_path, processed_path, 16000)
    if working_path != audio_path and working_path not in temp_files:
        temp_files.append(working_path)
    t_decode += time.perf_counter() - t0

    # 4. VAD
    t0 = time.perf_counter()
    vad_engine = os.getenv("VAD_ENGINE", "faster_whisper").lower()
    speech_timestamps = None
    if vad_engine == "silero":
        speech_timestamps = vad_processor.get_speech_timestamps(working_path)
    t_vad = time.perf_counter() - t0

    # 5. STT
    t0 = time.perf_counter()
    words, language = stt_engine.transcribe(working_path)
    t_stt = time.perf_counter() - t0

    # 6. Diarization
    t0 = time.perf_counter()
    min_diarize_sec = float(os.getenv("PIPELINE_MIN_DIARIZE_SEC", "10.0"))
    if profile == "feedback" and audio_duration < min_diarize_sec:
        diar_segments = []
    else:
        diar_segments = diarizer.diarize(working_path)
    t_diarize = time.perf_counter() - t0

    # Cleanup temp audio
    for tmp in temp_files:
        if Path(tmp).exists():
            try:
                Path(tmp).unlink()
            except Exception:
                pass

    # 7. Refine
    t0 = time.perf_counter()
    raw_utterances = fusion_engine.align(words, diar_segments)
    final_utterances = semantic_refiner.refine(raw_utterances)
    t_refine = time.perf_counter() - t0

    t_total = t_decode + t_denoise + t_vad + t_stt + t_diarize + t_refine

    return {
        "audio_duration": audio_duration,
        "t_decode": t_decode,
        "t_denoise": t_denoise,
        "t_vad": t_vad,
        "t_stt": t_stt,
        "t_diarize": t_diarize,
        "t_refine": t_refine,
        "t_total": t_total,
    }


def run_benchmark(audio_paths: list[str], models: list[str], profiles: list[str], threads: int):
    """Verilen sesler, modeller ve profiller için benchmark matrisini çalıştırır ve sonuçları tablo basar."""
    print("=======================================================================================")
    print("                    CPU PERFORMANS & BENCHMARK SONUÇLARI")
    print(f" CPU Thread Bütçesi: {threads} | Test Edilen Ses Sayısı: {len(audio_paths)}")
    print("=======================================================================================\n")

    header = (
        f"| {'Model':<7} | {'Profil':<8} | {'Ses Süresi':<10} | {'Decode':<8} | {'Denoise':<8} | "
        f"{'VAD':<7} | {'STT':<8} | {'Diarize':<8} | {'Refine':<7} | {'Toplam':<8} | {'RTF':<6} | {'k (çekirdek-sn/ses-sn)':<22} |"
    )
    separator = "|" + "-" * (len(header) - 2) + "|"

    print(header)
    print(separator)

    for model in models:
        for profile in profiles:
            total_audio_sec = 0.0
            sum_decode = 0.0
            sum_denoise = 0.0
            sum_vad = 0.0
            sum_stt = 0.0
            sum_diarize = 0.0
            sum_refine = 0.0
            sum_total = 0.0

            for audio_path in audio_paths:
                res = measure_pipeline_stages(audio_path, model, profile, threads)
                total_audio_sec += res["audio_duration"]
                sum_decode += res["t_decode"]
                sum_denoise += res["t_denoise"]
                sum_vad += res["t_vad"]
                sum_stt += res["t_stt"]
                sum_diarize += res["t_diarize"]
                sum_refine += res["t_refine"]
                sum_total += res["t_total"]

                file_name = Path(audio_path).name
                file_rtf = res["t_total"] / res["audio_duration"] if res["audio_duration"] > 0 else 0
                import ctypes
                from ctypes import wintypes
                class PMC(ctypes.Structure):
                    _fields_ = [('cb', wintypes.DWORD), ('pf', wintypes.DWORD), ('pws', ctypes.c_size_t), ('ws', ctypes.c_size_t)]
                pmc = PMC()
                pmc.cb = ctypes.sizeof(PMC)
                ctypes.windll.kernel32.K32GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
                ram_mb = pmc.ws / (1024 * 1024)

                print(f"   [Dosya: {file_name:<18}] Süre: {res['audio_duration']:>5.1f}s | İşlem: {res['t_total']:>5.2f}s | RTF: {file_rtf:.3f} | RAM: {ram_mb:.1f} MB")

            if total_audio_sec <= 0:
                continue

            rtf = sum_total / total_audio_sec
            k = (sum_total * threads) / total_audio_sec

            row = (
                f"| {model:<7} | {profile:<8} | {total_audio_sec:>8.2f}s | {sum_decode:>6.2f}s | {sum_denoise:>6.2f}s | "
                f"{sum_vad:>5.2f}s | {sum_stt:>6.2f}s | {sum_diarize:>6.2f}s | {sum_refine:>5.2f}s | {sum_total:>6.2f}s | "
                f"{rtf:>6.3f} | {k:>22.3f} |"
            )
            print(row)

    print(separator)
    print("\n[İPUCU] 'k' parametresi kapasite planlamasında kullanılır:")
    print("        Gereken Çekirdek = (Toplam Ses Süresi × k) / Hedef Süre\n")


def main():
    parser = argparse.ArgumentParser(description="CPU Performans Benchmark Betiği")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="./storage",
        help="Test ses dosyalarının bulunduğu dizin veya ses dosyası yolu",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tiny,small",
        help="Virgülle ayrılmış model isimleri (örn: tiny,base,small)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="feedback,full",
        help="Virgülle ayrılmış profiller (örn: feedback,full)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Kullanılacak CPU thread bütçesi (WORKER_CPU_THREADS)",
    )

    args = parser.parse_args()

    path = Path(args.audio_dir)
    audio_paths: list[str] = []

    if path.is_file():
        audio_paths.append(str(path))
    elif path.is_dir():
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            audio_paths.extend([str(p) for p in path.glob(ext)])
    else:
        print(f"HATA: '{args.audio_dir}' dizini veya dosyası bulunamadı.")
        sys.exit(1)

    if not audio_paths:
        print(f"HATA: '{args.audio_dir}' içinde ses dosyası (.wav, .mp3, .flac) bulunamadı.")
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    run_benchmark(audio_paths, models, profiles, args.threads)


if __name__ == "__main__":
    main()

