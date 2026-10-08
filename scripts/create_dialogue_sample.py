"""
2+ Konuşmacılı Sentetik Test Ses ve RTTM Dosyası Üretici (scripts/create_dialogue_sample.py)

Bu betik, diarization değerlendirme ve DER ölçümleri için sentetik konuşmacı diyalogları (.wav)
ve bunlara karşılık gelen RTTM (Ground Truth) dosyalarını üretir.
"""

import argparse
import math
import struct
import wave
from pathlib import Path


def generate_dialogue_pair(
    file_stem: str,
    output_dir: Path,
    segments: list[tuple[float, float, str, float]],
    sample_rate: int = 16000,
):
    """
    segments: list of (start_time, end_time, speaker_id, tone_freq)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{file_stem}.wav"
    rttm_path = output_dir / f"{file_stem}.rttm"

    max_end = max(end for _, end, _, _ in segments)
    num_samples = int(sample_rate * max_end)

    # 1. WAV Dosyası Üretimi
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)

        samples = []
        for i in range(num_samples):
            t = float(i) / sample_rate
            current_sample = 0.0

            for start, end, _, freq in segments:
                if start <= t < end:
                    # Fade in/out yumuşatması
                    fade = min(1.0, (t - start) * 20.0, (end - t) * 20.0)
                    current_sample += fade * 32767.0 * 0.3 * math.sin(2.0 * math.pi * freq * t)

            sample_val = max(-32768, min(32767, int(current_sample)))
            samples.append(struct.pack("<h", sample_val))

        wav_file.writeframes(b"".join(samples))

    # 2. RTTM Dosyası Üretimi
    rttm_lines = []
    for start, end, speaker_id, _ in sorted(segments, key=lambda x: x[0]):
        duration = round(end - start, 3)
        start_t = round(start, 3)
        rttm_lines.append(
            f"SPEAKER {file_stem} 1 {start_t:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
        )

    with open(rttm_path, "w", encoding="utf-8") as f:
        f.writelines(rttm_lines)

    print(f"[+] Diyalog ve RTTM Üretildi: {wav_path.name} ({max_end:.1f}s, {len(segments)} segment)")


def generate_sample_suite(output_dir: Path):
    """3 farklı 2 ve 3 konuşmacılı diyalog seti üretir."""
    # Örnek 1: 2 Konuşmacı, 6 saniye (0-3s SPEAKER_00, 3-6s SPEAKER_01)
    generate_dialogue_pair(
        "sample1_2spk",
        output_dir,
        [
            (0.0, 3.0, "SPEAKER_00", 440.0),
            (3.0, 6.0, "SPEAKER_01", 220.0),
        ],
    )

    # Örnek 2: 2 Konuşmacı, 10 saniye (Karşılıklı konuşma)
    generate_dialogue_pair(
        "sample2_2spk_turn",
        output_dir,
        [
            (0.0, 3.5, "SPEAKER_00", 440.0),
            (3.5, 7.0, "SPEAKER_01", 220.0),
            (7.0, 10.0, "SPEAKER_00", 440.0),
        ],
    )

    # Örnek 3: 3 Konuşmacı, 12 saniye
    generate_dialogue_pair(
        "sample3_3spk",
        output_dir,
        [
            (0.0, 4.0, "SPEAKER_00", 440.0),
            (4.0, 8.0, "SPEAKER_01", 261.63),
            (8.0, 12.0, "SPEAKER_02", 174.61),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentetik Diarization Test Verisi ve RTTM Üretici")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/fixtures/diarization_eval",
        help="Çıktı klasörü (Varsayılan: tests/fixtures/diarization_eval)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"[START] Sentetik Diarization Test Verileri Üretiliyor -> {out_dir.absolute()}")
    generate_sample_suite(out_dir)
    print("[DONE] Test verileri üretildi!")
