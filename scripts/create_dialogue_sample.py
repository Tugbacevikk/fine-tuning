"""
Sentetik ve Gerçek Ses Kaynaklı Diarization Test Verisi ve RTTM Üretici (scripts/create_dialogue_sample.py)

Doğrulama ve Güvenlik Kuralları:
1. Manifest Eşleşmesi (speaker_manifest.json): storage/raw altındaki her gerçek ses dosyasına benzersiz
   bir kaynak konuşmacı kimliği (source_speaker_id) atanır. Aynı konuşmacıya ait iki ses, aynı diyalog içinde
   kesinlikle farklı konuşmacı etiketleri (SPEAKER_00 ve SPEAKER_01) olarak eşleştirilemez.
2. Örtüşme (Overlap): Gerçekçi çakışmalı konuşmalar (--overlap-ratio, varsayılan 0.08 / %8.0) RTTM dosyasında
   eşzamanlı iki ayrı SPEAKER satırı olarak doğru biçimde işaretlenir.
3. Gerçekleşen örtüşme oranı hedef orandan ±1 puandan fazla saparsa uyarı verilir.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
import numpy as np
import soundfile as sf


def build_speaker_manifest(raw_dir: Path, manifest_path: Path) -> dict:
    """storage/raw altındaki ses dosyalarına kaynak konuşmacı kimliği atar ve verified: false alanını ekler."""
    manifest_data = {"verified": False, "files": {}}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    manifest_data["verified"] = loaded.get("verified", False)
                    manifest_data["files"] = loaded.get("files", {})
        except Exception:
            pass

    wav_files = sorted(list(raw_dir.glob("*.wav"))) if raw_dir.exists() else []
    files_map = manifest_data["files"]
    updated = False

    for idx, wav in enumerate(wav_files, start=1):
        if wav.name not in files_map:
            files_map[wav.name] = f"SPEAKER_SRC_{idx:03d}"
            updated = True

    if updated or not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    return manifest_data


def create_eval_dataset(
    output_dir: Path,
    num_recordings: int = 40,
    seed: int = 42,
    target_overlap_ratio: float = 0.08,
    use_raw_storage: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("storage/raw")
    manifest_path = output_dir / "speaker_manifest.json"
    manifest_data = build_speaker_manifest(raw_dir, manifest_path)
    files_map = manifest_data.get("files", {})

    valid_source_speakers = {}
    if use_raw_storage and raw_dir.exists():
        for p in raw_dir.glob("*.wav"):
            try:
                info = sf.info(str(p))
                if info.duration >= 5.0 and info.channels == 1 and info.samplerate == 16000:
                    spk_id = files_map.get(p.name, f"SPEAKER_SRC_{p.name}")
                    if spk_id not in valid_source_speakers:
                        valid_source_speakers[spk_id] = []
                    valid_source_speakers[spk_id].append(p)
            except Exception:
                pass

    unique_src_speakers = list(valid_source_speakers.keys())
    print(f"[MANIFEST] Bulunan benzersiz kaynak konuşmacı sayısı: {len(unique_src_speakers)}")

    total_audio_duration = 0.0
    total_overlap_duration = 0.0

    for idx in range(1, num_recordings + 1):
        num_spk = 2 if (idx % 3 != 0) else 3
        target_duration = float(random.randint(15, 35))
        file_stem = f"eval_sample_{idx:02d}_{num_spk}spk"

        wav_path = output_dir / f"{file_stem}.wav"
        rttm_path = output_dir / f"{file_stem}.rttm"

        if len(unique_src_speakers) >= num_spk:
            selected_src_spks = random.sample(unique_src_speakers, num_spk)
        else:
            selected_src_spks = [f"DUMMY_SRC_{i}" for i in range(num_spk)]

        dialogue_spk_roles = [f"SPEAKER_{i:02d}" for i in range(num_spk)]
        freqs = [220.0, 440.0, 330.0]

        segments = []
        curr_t = 0.0

        while curr_t < target_duration:
            role_idx = random.randint(0, num_spk - 1)
            role_id = dialogue_spk_roles[role_idx]
            freq = freqs[role_idx]

            is_short = random.random() < 0.25
            turn_dur = round(random.uniform(0.4, 0.9) if is_short else random.uniform(2.0, 7.0), 2)
            if curr_t + turn_dur > target_duration:
                turn_dur = round(target_duration - curr_t, 2)
            if turn_dur <= 0.1:
                break

            start_t = round(curr_t, 2)
            end_t = round(start_t + turn_dur, 2)
            segments.append((start_t, end_t, role_id, freq, role_idx))

            # Örtüşme olasılığını hedef örtüşme oranına göre dinamik ayarla
            if random.random() < (target_overlap_ratio * 5.2) and len(segments) > 1:
                overlap_dur = round(random.uniform(1.0, 2.0), 2)
                curr_t += max(0.1, turn_dur - overlap_dur)
            else:
                pause_dur = round(random.uniform(0.1, 0.3), 2)
                curr_t += turn_dur + pause_dur

        sr = 16000
        total_samples = int(sr * target_duration)
        signal = np.zeros(total_samples, dtype=np.float32)

        rttm_lines = []
        for start_t, end_t, role_id, _, _ in sorted(segments, key=lambda x: x[0]):
            dur = round(end_t - start_t, 3)
            rttm_lines.append(
                f"SPEAKER {file_stem} 1 {start_t:.3f} {dur:.3f} <NA> <NA> {role_id} <NA> <NA>\n"
            )

        activity_matrix = np.zeros((num_spk, total_samples), dtype=bool)
        for start_t, end_t, _, _, role_idx in segments:
            s_idx = int(start_t * sr)
            e_idx = min(total_samples, int(end_t * sr))
            activity_matrix[role_idx, s_idx:e_idx] = True

        overlap_frames = np.sum(np.sum(activity_matrix, axis=0) > 1)
        file_overlap_sec = overlap_frames / float(sr)

        total_audio_duration += target_duration
        total_overlap_duration += file_overlap_sec

        for start_t, end_t, role_id, freq, role_idx in segments:
            start_idx = int(start_t * sr)
            end_idx = min(total_samples, int(end_t * sr))
            seg_len = end_idx - start_idx
            if seg_len <= 0:
                continue

            src_spk = selected_src_spks[role_idx]
            if src_spk in valid_source_speakers:
                src_file = random.choice(valid_source_speakers[src_spk])
                data, source_sr = sf.read(str(src_file), dtype="float32")
                if len(data) > seg_len:
                    start_crop = random.randint(0, len(data) - seg_len)
                    chunk = data[start_crop : start_crop + seg_len]
                else:
                    chunk = np.tile(data, math.ceil(seg_len / len(data)))[:seg_len]
            else:
                t_arr = np.linspace(0, (end_t - start_t), seg_len, endpoint=False)
                chunk = 0.3 * np.sin(2.0 * np.pi * freq * t_arr).astype(np.float32)

            fade_len = min(160, seg_len // 2)
            if fade_len > 0:
                window = np.ones(seg_len, dtype=np.float32)
                window[:fade_len] = np.linspace(0, 1, fade_len)
                window[-fade_len:] = np.linspace(1, 0, fade_len)
                chunk = chunk * window

            signal[start_idx:end_idx] += chunk

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = (signal / max_val) * 0.9

        sf.write(str(wav_path), signal, sr)

        with open(rttm_path, "w", encoding="utf-8") as f:
            f.writelines(rttm_lines)

    actual_overlap_pct = (total_overlap_duration / total_audio_duration) * 100.0 if total_audio_duration > 0 else 0.0
    target_overlap_pct = target_overlap_ratio * 100.0

    manifest_data["target_overlap_percent"] = round(target_overlap_pct, 2)
    manifest_data["actual_overlap_percent"] = round(actual_overlap_pct, 2)
    manifest_data["seed"] = seed

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] {num_recordings} adet test kaydı '{output_dir}' dizinine yazıldı (Seed={seed}).")
    print(f"     Toplam Ses Süresi  : {total_audio_duration / 60.0:.2f} dakika ({total_audio_duration:.1f}s)")
    print(f"     Gerçekleşen Örtüşme : %{actual_overlap_pct:.2f} ({total_overlap_duration:.2f}s) | Hedef: %{target_overlap_pct:.2f}")

    if abs(actual_overlap_pct - target_overlap_pct) > 1.0:
        print(
            f"[WARNING] Gerçekleşen örtüşme oranı (%{actual_overlap_pct:.2f}), hedef (%{target_overlap_pct:.2f}) orandan ±1 puandan fazla sapma gösterdi!"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarization Test Verisi ve RTTM Üretici")
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
        help="Rastgelelik seed değeri (Varsayılan: 42)",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.08,
        help="Hedef örtüşme oranı (Varsayılan: 0.08 / %8.0)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"[START] Test Veri Seti Üretiliyor -> {out_dir.absolute()}")
    create_eval_dataset(
        out_dir,
        num_recordings=args.num_recordings,
        seed=args.seed,
        target_overlap_ratio=args.overlap_ratio,
    )
    print("[DONE] Test veri seti üretimi tamamlandı!")
