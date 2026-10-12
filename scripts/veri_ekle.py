"""
Veri Ekleme ve Audacity Etiketleme Aracı (scripts/veri_ekle.py)

Kullanım Senaryosu:
1. Pipeline analiz çıktısındaki (.rttm) dosyalarını Audacity etiket (labels.txt) formatına çevirir:
   python scripts/veri_ekle.py hazirla --ses ses.wav --taslak-rttm-klasoru analiz_sonuclari/
   python scripts/veri_ekle.py toplu --ses-klasoru "Yeni klasör" --taslak-rttm-klasoru "Yeni klasör/analiz_sonuclari"

2. Kullanıcı Audacity'de sesi ve taslak etiketleri açıp (File -> Import -> Labels) kulaklıkla dinleyip düzeltir.
   Ardından Audacity'den File -> Export -> Export Labels diyerek kaydeder.

3. Düzeltilmiş Audacity etiketlerini doğrudan test setine RTTM olarak aktarır:
   python scripts/veri_ekle.py kaydet --ses hazirlik/ses.wav --etiket hazirlik/ses.txt --hedef-klasor tests/fixtures/diarization_eval/real_turkish
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("veri_ekle")


def rttm_to_audacity_labels(rttm_path: Path) -> list[tuple[float, float, str]]:
    """
    RTTM dosyasını okuyarak Audacity formatında (start_sec, end_sec, speaker_id) listesi döndürür.
    RTTM formatı: SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    labels = []
    if not rttm_path.exists():
        return labels

    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start_t = round(float(parts[3]), 6)
                duration = round(float(parts[4]), 6)
                end_t = round(start_t + duration, 6)
                speaker_id = parts[7]
                labels.append((start_t, end_t, speaker_id))

    labels.sort(key=lambda x: x[0])
    return labels


def audacity_labels_to_rttm(labels: list[tuple[float, float, str]], file_stem: str) -> list[str]:
    """
    Audacity (start_sec, end_sec, speaker_id) listesini standart NIST RTTM satırlarına dönüştürür.
    """
    rttm_lines = []
    sorted_labels = sorted(labels, key=lambda x: x[0])
    for start_t, end_t, spk in sorted_labels:
        duration = max(0.01, end_t - start_t)
        line = f"SPEAKER {file_stem} 1 {start_t:.3f} {duration:.3f} <NA> <NA> {spk} <NA> <NA>\n"
        rttm_lines.append(line)
    return rttm_lines


def write_audacity_label_file(output_txt: Path, labels: list[tuple[float, float, str]]):
    """Audacity etiket dosyasını (tab-separated) yazar."""
    with open(output_txt, "w", encoding="utf-8") as f:
        for start_t, end_t, spk in labels:
            f.write(f"{start_t:.6f}\t{end_t:.6f}\t{spk}\n")


def read_audacity_label_file(label_txt: Path) -> list[tuple[float, float, str]]:
    """Audacity etiket dosyasını okur."""
    labels = []
    with open(label_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                start_t = float(parts[0])
                end_t = float(parts[1])
                spk = parts[2].strip()
                labels.append((start_t, end_t, spk))
            elif len(parts) == 2:
                # Tek zaman damgalı nokta etiket fallback
                t = float(parts[0])
                labels.append((t, t + 0.5, parts[1].strip()))
    return sorted(labels, key=lambda x: x[0])


def normalize_audio_to_wav(input_path: Path, output_path: Path):
    """Ses dosyasını 16kHz mono WAV formatına dönüştürür."""
    import soundfile as sf
    import numpy as np
    import scipy.signal
    from math import gcd

    audio_array, sr = sf.read(str(input_path))
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sr != 16000:
        g = gcd(int(sr), 16000)
        audio_array = scipy.signal.resample_poly(audio_array, 16000 // g, int(sr) // g)
        sr = 16000

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_array, 16000, subtype="PCM_16")


def cmd_hazirla(
    ses_path: Path,
    taslak_rttm: Path | None = None,
    taslak_rttm_klasoru: Path | None = None,
    cikis_klasoru: Path | None = None,
):
    if not ses_path.exists():
        logger.error("Ses dosyası bulunamadı: %s", ses_path)
        sys.exit(1)

    if cikis_klasoru is None:
        cikis_klasoru = Path("tests/fixtures/diarization_eval/real_turkish/hazirlik")
    cikis_klasoru.mkdir(parents=True, exist_ok=True)

    safe_stem = re.sub(r"[^\w\-]", "_", ses_path.stem).strip("_") or "sample"
    target_wav = cikis_klasoru / f"{safe_stem}.wav"
    target_labels = cikis_klasoru / f"{safe_stem}.txt"

    logger.info("Ses 16kHz mono WAV formatına dönüştürülüyor: %s -> %s", ses_path.name, target_wav.name)
    normalize_audio_to_wav(ses_path, target_wav)

    # Taslak RTTM bulma
    found_rttm = None
    if taslak_rttm and taslak_rttm.exists():
        found_rttm = taslak_rttm
    elif taslak_rttm_klasoru and taslak_rttm_klasoru.exists():
        candidates = [
            taslak_rttm_klasoru / f"{safe_stem}.rttm",
            taslak_rttm_klasoru / f"{ses_path.stem}.rttm",
            taslak_rttm_klasoru / f"{ses_path.stem.replace(' ', '_')}.rttm",
        ]
        for c in candidates:
            if c.exists():
                found_rttm = c
                break

    if found_rttm and found_rttm.exists():
        logger.info("Taslak RTTM bulundu: %s", found_rttm.name)
        labels = rttm_to_audacity_labels(found_rttm)
        write_audacity_label_file(target_labels, labels)
        logger.info("Audacity taslak etiketleri oluşturuldu (%d etiket): %s", len(labels), target_labels)
    else:
        logger.warning("Taslak RTTM bulunamadı. Boş etiket şablonu oluşturuluyor.")
        write_audacity_label_file(target_labels, [])

    print(f"\n[HAZIR] Audacity'de açılacak dosyalar:")
    print(f" - Ses Dosyası     : {target_wav.absolute()}")
    print(f" - Audacity Etiketi: {target_labels.absolute()}")
    print("Nasıl Kullanılır: Audacity'de WAV dosyasını açın -> File -> Import -> Labels ile .txt dosyasını yükleyin.")


def cmd_toplu(
    ses_klasoru: Path,
    taslak_rttm_klasoru: Path | None = None,
    cikis_klasoru: Path | None = None,
):
    if not ses_klasoru.exists():
        logger.error("Ses klasörü bulunamadı: %s", ses_klasoru)
        sys.exit(1)

    if cikis_klasoru is None:
        cikis_klasoru = Path("tests/fixtures/diarization_eval/real_turkish/hazirlik")
    cikis_klasoru.mkdir(parents=True, exist_ok=True)

    extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = [f for f in sorted(ses_klasoru.glob("*")) if f.is_file() and f.suffix.lower() in extensions]

    logger.info("Toplu hazırlık: %d ses dosyası işlenecek.", len(audio_files))
    for f in audio_files:
        cmd_hazirla(
            ses_path=f,
            taslak_rttm_klasoru=taslak_rttm_klasoru,
            cikis_klasoru=cikis_klasoru,
        )


def cmd_kaydet(
    ses_path: Path,
    etiket_path: Path,
    hedef_klasor: Path | None = None,
):
    if not ses_path.exists() or not etiket_path.exists():
        logger.error("Ses veya etiket dosyası bulunamadı: %s veya %s", ses_path, etiket_path)
        sys.exit(1)

    if hedef_klasor is None:
        hedef_klasor = Path("tests/fixtures/diarization_eval/real_turkish")
    hedef_klasor.mkdir(parents=True, exist_ok=True)

    safe_stem = re.sub(r"[^\w\-]", "_", ses_path.stem).strip("_")
    dest_wav = hedef_klasor / f"{safe_stem}.wav"
    dest_rttm = hedef_klasor / f"{safe_stem}.rttm"

    logger.info("Audacity etiketleri okunuyor: %s", etiket_path.name)
    labels = read_audacity_label_file(etiket_path)
    if not labels:
        logger.warning("Etiket dosyasında hiçbir konuşmacı etiketi bulunamadı!")

    rttm_lines = audacity_labels_to_rttm(labels, safe_stem)
    with open(dest_rttm, "w", encoding="utf-8") as f_rttm:
        f_rttm.writelines(rttm_lines)

    # Ses dosyasını 16kHz mono WAV olarak test setine kopyala/normalize et
    normalize_audio_to_wav(ses_path, dest_wav)

    unique_speakers = sorted(list({lbl[2] for lbl in labels}))
    print(f"\n[BAŞARILI] Test setine kaydedildi:")
    print(f" - Hedef WAV : {dest_wav.absolute()}")
    print(f" - Hedef RTTM: {dest_rttm.absolute()} ({len(rttm_lines)} segment, Konuşmacılar: {unique_speakers})")


def main():
    parser = argparse.ArgumentParser(description="Veri ekleme ve Audacity etiket dönüştürme aracı")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # hazirla
    p_hazirla = subparsers.add_parser("hazirla", help="Tek bir ses dosyası için Audacity taslak etiketlerini hazırlar")
    p_hazirla.add_argument("--ses", type=Path, required=True, help="Kaynak ses dosyası")
    p_hazirla.add_argument("--taslak-rttm", type=Path, default=None, help="Taslak .rttm dosyası")
    p_hazirla.add_argument("--taslak-rttm-klasoru", type=Path, default=None, help="Taslak RTTM dosyalarının bulunduğu klasör")
    p_hazirla.add_argument("--cikis-klasoru", type=Path, default=None, help="Hazırlık çıktı klasörü")

    # toplu
    p_toplu = subparsers.add_parser("toplu", help="Bir klasördeki tüm ses dosyaları için Audacity taslak etiketlerini hazırlar")
    p_toplu.add_argument("--ses-klasoru", type=Path, required=True, help="Kaynak ses dosyaları klasörü")
    p_toplu.add_argument("--taslak-rttm-klasoru", type=Path, default=None, help="Taslak RTTM klasörü (analiz çıktıları)")
    p_toplu.add_argument("--cikis-klasoru", type=Path, default=None, help="Hazırlık çıktı klasörü")

    # kaydet
    p_kaydet = subparsers.add_parser("kaydet", help="Audacity'de düzeltilmiş etiketleri RTTM olarak test setine kaydeder")
    p_kaydet.add_argument("--ses", type=Path, required=True, help="Audacity'de etiketlenen ses dosyası")
    p_kaydet.add_argument("--etiket", type=Path, required=True, help="Audacity'den dışa aktarılan .txt etiket dosyası")
    p_kaydet.add_argument("--hedef-klasor", type=Path, default=None, help="Nihai test seti dizini (varsayılan: tests/fixtures/diarization_eval/real_turkish)")

    args = parser.parse_args()

    if args.subcommand == "hazirla":
        cmd_hazirla(
            ses_path=args.ses,
            taslak_rttm=args.taslak_rttm,
            taslak_rttm_klasoru=args.taslak_rttm_klasoru,
            cikis_klasoru=args.cikis_klasoru,
        )
    elif args.subcommand == "toplu":
        cmd_toplu(
            ses_klasoru=args.ses_klasoru,
            taslak_rttm_klasoru=args.taslak_rttm_klasoru,
            cikis_klasoru=args.cikis_klasoru,
        )
    elif args.subcommand == "kaydet":
        cmd_kaydet(
            ses_path=args.ses,
            etiket_path=args.etiket,
            hedef_klasor=args.hedef_klasor,
        )


if __name__ == "__main__":
    main()
