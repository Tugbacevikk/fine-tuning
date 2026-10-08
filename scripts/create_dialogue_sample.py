"""
Sentetik ve Gerçek Ses Kaynaklı Diarization Test Verisi ve RTTM Üretici (scripts/create_dialogue_sample.py)

Bu betik, diarization değerlendirmesi ve DER ölçümleri için test ses dosyalarını (.wav)
ve RTTM (Ground Truth) etiket dosyalarını üretir.
- 30-90 saniye uzunlukta
- 2 ve 3 konuşmacılı
- Kısa cevaplar (<1 sn) ve küçük konuşma örtüşmeleri (%5-10) içeren
- Sabit seed (42) ile tekrarlanabilir test verisi seti oluşturur.
"""

import argparse
import math
import random
import sys
from pathlib import Path
import numpy as np
import soundfile as sf


def create_eval_dataset(
    output_dir: Path,
    num_recordings: int = 40,
    seed: int = 42,
    use_raw_storage: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gerçek ses kaynaklarını storage/raw altından ara
    raw_dir = Path("storage/raw")
    raw_wavs = []
    if use_raw_storage and raw_dir.exists():
        for p in raw_dir.glob("*.wav"):
            try:
                info = sf.info(str(p))
                if info.duration >= 5.0 and info.channels == 1 and info.samplerate == 16000:
                    raw_wavs.append(p)
            except Exception:
                pass

    if raw_wavs:
        print(f"[DATASET] Bulunan gerçek kaynak ses dosyası sayısı: {len(raw_wavs)} (storage/raw)")
    else:
        print("[DATASET] Gerçek ses bulunamadı, sentetik harmonik ses dalgası jeneratörü kullanılacak.")

    for idx in range(1, num_recordings + 1):
        num_spk = 2 if (idx % 3 != 0) else 3
        target_duration = float(random.randint(30, 90))
        file_stem = f"eval_sample_{idx:02d}_{num_spk}spk"

        wav_path = output_dir / f"{file_stem}.wav"
        rttm_path = output_dir / f"{file_stem}.rttm"

        segments = []
        curr_t = 0.0
        speakers = [f"SPEAKER_{i:02d}" for i in range(num_spk)]
        freqs = [220.0, 440.0, 330.0]

        while curr_t < target_duration:
            spk_idx = random.randint(0, num_spk - 1)
            spk_id = speakers[spk_idx]
            freq = freqs[spk_idx]

            # Kısa cevaplar (<1s) ve normal konuşma turları (2s - 7s)
            is_short = random.random() < 0.25
            turn_dur = round(random.uniform(0.4, 0.9) if is_short else random.uniform(2.0, 7.0), 2)
            if curr_t + turn_dur > target_duration:
                turn_dur = round(target_duration - curr_t, 2)
            if turn_dur <= 0.1:
                break

            start_t = round(curr_t, 2)
            end_t = round(start_t + turn_dur, 2)
            segments.append((start_t, end_t, spk_id, freq, spk_idx))

            # Küçük konuşma çakışması / örtüşmesi (%5-10)
            if random.random() < 0.15 and len(segments) > 1:
                overlap_dur = round(random.uniform(0.3, 0.8), 2)
                curr_t += max(0.1, turn_dur - overlap_dur)
            else:
                pause_dur = round(random.uniform(0.1, 0.5), 2)
                curr_t += turn_dur + pause_dur

        # Sinyal sentezleme / birleştirme
        sr = 16000
        total_samples = int(sr * target_duration)
        signal = np.zeros(total_samples, dtype=np.float32)

        for start_t, end_t, spk_id, freq, spk_idx in segments:
            start_idx = int(start_t * sr)
            end_idx = min(total_samples, int(end_t * sr))
            seg_len = end_idx - start_idx
            if seg_len <= 0:
                continue

            if raw_wavs:
                source_wav = raw_wavs[spk_idx % len(raw_wavs)]
                data, source_sr = sf.read(str(source_wav), dtype="float32")
                if len(data) > seg_len:
                    start_crop = random.randint(0, len(data) - seg_len)
                    chunk = data[start_crop : start_crop + seg_len]
                else:
                    chunk = np.tile(data, math.ceil(seg_len / len(data)))[:seg_len]
            else:
                t_arr = np.linspace(0, (end_t - start_t), seg_len, endpoint=False)
                chunk = 0.3 * np.sin(2.0 * np.pi * freq * t_arr).astype(np.float32)

            # Kenar yumuşatma (Fade-in / Fade-out)
            fade_len = min(160, seg_len // 2)
            if fade_len > 0:
                window = np.ones(seg_len, dtype=np.float32)
                window[:fade_len] = np.linspace(0, 1, fade_len)
                window[-fade_len:] = np.linspace(1, 0, fade_len)
                chunk = chunk * window

            signal[start_idx:end_idx] += chunk

        # Normalize et
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = (signal / max_val) * 0.9

        sf.write(str(wav_path), signal, sr)

        # RTTM etiket dosyasını yaz
        rttm_lines = []
        for start_t, end_t, spk_id, _, _ in sorted(segments, key=lambda x: x[0]):
            duration = round(end_t - start_t, 3)
            rttm_lines.append(
                f"SPEAKER {file_stem} 1 {start_t:.3f} {duration:.3f} <NA> <NA> {spk_id} <NA> <NA>\n"
            )

        with open(rttm_path, "w", encoding="utf-8") as f:
            f.writelines(rttm_lines)

    print(f"[OK] {num_recordings} adet test kaydı (30-90s) '{output_dir}' klasörüne yazıldı. (Seed={seed})")


def generate_small_sample_suite(output_dir: Path):
    """Birim testler için 3 adet hızlı test kaydı üretir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Örnek 1
    create_eval_dataset(output_dir, num_recordings=3, seed=42, use_raw_storage=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentetik ve Gerçek Ses Kaynaklı Diarization Test Verisi Üretici")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/fixtures/diarization_eval/large",
        help="Çıktı klasörü (Varsayılan: tests/fixtures/diarization_eval/large)",
    )
    parser.add_argument(
        "--num-recordings",
        type=int,
        default=40,
        help="Üretilecek kayıt sayısı (Varsayılan: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik için sabit seed değeri (Varsayılan: 42)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"[START] Test Veri Seti Üretiliyor -> {out_dir.absolute()}")
    create_eval_dataset(out_dir, num_recordings=args.num_recordings, seed=args.seed)
    print("[DONE] Test veri seti üretimi tamamlandı!")
