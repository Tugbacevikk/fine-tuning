import argparse
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from audio_analyzer.config import get_settings
from audio_analyzer.services.pipeline_factory import get_shared_pipeline, reset_pipeline_cache


def find_audio_file(filename: str) -> Path | None:
    candidate_dirs = [
        Path("storage/raw"),
        Path("tests/fixtures"),
        Path("storage"),
        Path("Yeni klasör"),
    ]
    for d in candidate_dirs:
        p = d / filename
        if p.exists() and p.is_file():
            return p
    return None


def run_benchmark(
    csv_path: str = "tests/fixtures/speaker_counts.csv",
    repeats: int = 3,
    split_eval: bool = False,
):
    if not os.path.exists(csv_path):
        print(f"Hata: CSV dosyası bulunamadı: {csv_path}")
        return

    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if not row.get("dosya_adi"):
                continue
            items.append({
                "row_index": idx,
                "dosya_adi": row["dosya_adi"].strip(),
                "gercek": int(row["gercek_kisi_sayisi"].strip()),
                "not": row.get("not", "").strip(),
            })

    if not items:
        print("CSV dosyasında geçerli satır bulunamadı.")
        return

    print("\n" + "=" * 90)
    print("KONUŞMACI SAYISI DOĞRULUK VE UÇTAN UCA RTF BENCHMARK TESTİ (GERÇEK PIPELINE)")
    print(f"Test Dosyası     : {csv_path} ({len(items)} kayıt)")
    print(f"RTF Tekrar Sayısı: {repeats}")
    print(f"Split Değerlendirme: {'Aktif (Tek: Train / Çift: Test)' if split_eval else 'Tüm Liste'}")
    print("=" * 90)

    # 1. Pipeline'ı pipeline_factory üzerinden oluştur
    pipeline = get_shared_pipeline()

    # Sonuç listeleri
    results_fixed = []
    results_dynamic = []

    for item in items:
        fname = item["dosya_adi"]
        true_k = item["gercek"]
        row_idx = item["row_index"]
        fpath = find_audio_file(fname)

        if not fpath:
            print(f"[UYARI] '{fname}' bulunamadı, atlanıyor.")
            continue

        audio_info = sf.info(str(fpath))
        audio_dur = float(audio_info.duration)
        audio_data, sr = sf.read(str(fpath), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        print(f"\n[{row_idx}] Dosya: {fname} (Süre: {audio_dur:.2f}s | Gerçek Konuşmacı: {true_k})", flush=True)

        # -------------------------------------------------------------
        # YÖNTEM 1: Sabit Eşik (SHERPA_CLUSTER_THRESHOLD = 0.55)
        # -------------------------------------------------------------
        os.environ["ESTIMATE_NUM_SPEAKERS"] = "false"
        
        # A. Sherpa Ham Çıktısı (ClusterMerger ve Refiner öncesi)
        raw_segs_fixed = pipeline.diarizer.diarize(audio_data, num_speakers=None)
        raw_k_fixed = len(set(s.speaker_id for s in raw_segs_fixed))

        # B. Uçtan Uca Final Pipeline (STT + Diarizer + ClusterMerger + Refiner)
        print("  - Sabit Eşik (0.55) analiz ediliyor...", end="", flush=True)
        t0 = time.monotonic()
        utts_fixed, lang_fixed, ov_fixed = pipeline.process(str(fpath), num_speakers=None)
        dt_fixed = time.monotonic() - t0
        final_k_fixed = len(set(u.speaker_id for u in utts_fixed))
        rtf_fixed = dt_fixed / audio_dur if audio_dur > 0 else 0.0
        print(f" Tamamlandı ({dt_fixed:.2f}s). Ham={raw_k_fixed} -> Final={final_k_fixed}", flush=True)

        results_fixed.append({
            "row_index": row_idx,
            "split": "Train (Tek)" if row_idx % 2 == 1 else "Test (Çift)",
            "dosya": fname,
            "sure": audio_dur,
            "gercek": true_k,
            "raw_k": raw_k_fixed,
            "final_k": final_k_fixed,
            "dogru": final_k_fixed == true_k,
            "hata": abs(final_k_fixed - true_k),
            "sure_ort": dt_fixed,
            "rtf": rtf_fixed,
            "gecikme_uyarisi": dt_fixed > 60.0 if audio_dur >= 55.0 else False,
        })

        # -------------------------------------------------------------
        # YÖNTEM 2: Dinamik Tahmin (ESTIMATE_NUM_SPEAKERS = true, Tek Geçiş)
        # -------------------------------------------------------------
        os.environ["ESTIMATE_NUM_SPEAKERS"] = "true"

        # A. Sherpa Ham Çıktısı (Dinamik)
        raw_segs_dyn = pipeline.diarizer.diarize(audio_data, num_speakers=None)
        raw_k_dyn = len(set(s.speaker_id for s in raw_segs_dyn))

        # B. Uçtan Uca Final Pipeline (Dinamik) - RTF için dosya başına N tekrar
        durations_dyn = []
        final_utts_dyn = []
        for r in range(repeats):
            print(f"  - Dinamik Yöntem (Tekrar {r+1}/{repeats}) analiz ediliyor...", end="", flush=True)
            t0 = time.monotonic()
            utts_dyn, lang_dyn, ov_dyn = pipeline.process(str(fpath), num_speakers=None)
            dt = time.monotonic() - t0
            durations_dyn.append(dt)
            final_utts_dyn = utts_dyn
            print(f" {dt:.2f}s", flush=True)

        mean_dt_dyn = float(np.mean(durations_dyn))
        rtf_dyn = mean_dt_dyn / audio_dur if audio_dur > 0 else 0.0
        final_k_dyn = len(set(u.speaker_id for u in final_utts_dyn))
        print(f"  -> Dinamik Ortalama: {mean_dt_dyn:.2f}s (RTF={rtf_dyn:.3f}) | Ham={raw_k_dyn} -> Final={final_k_dyn}", flush=True)

        results_dynamic.append({
            "row_index": row_idx,
            "split": "Train (Tek)" if row_idx % 2 == 1 else "Test (Çift)",
            "dosya": fname,
            "sure": audio_dur,
            "gercek": true_k,
            "raw_k": raw_k_dyn,
            "final_k": final_k_dyn,
            "dogru": final_k_dyn == true_k,
            "hata": abs(final_k_dyn - true_k),
            "sure_ort": mean_dt_dyn,
            "sureler": durations_dyn,
            "rtf": rtf_dyn,
            "gecikme_uyarisi": mean_dt_dyn > 60.0 if audio_dur >= 55.0 else False,
        })

    # Ortam değişkenini orijinal varsayılanına (false) geri al
    os.environ["ESTIMATE_NUM_SPEAKERS"] = "false"

    # -------------------------------------------------------------
    # RAPORLAMA VE METRİKLER
    # -------------------------------------------------------------
    def calculate_metrics(subset):
        if not subset:
            return {"acc": 0.0, "mae": 0.0, "avg_rtf": 0.0, "count": 0, "correct": 0}
        n = len(subset)
        correct = sum(1 for x in subset if x["dogru"])
        return {
            "count": n,
            "correct": correct,
            "acc": (correct / n) * 100.0,
            "mae": sum(x["hata"] for x in subset) / n,
            "avg_rtf": sum(x["rtf"] for x in subset) / n,
        }

    print("\n" + "=" * 90)
    print("GENEL PERFORMANS VE RTF KARŞILAŞTIRMASI")
    print("=" * 90)

    m_fix = calculate_metrics(results_fixed)
    m_dyn = calculate_metrics(results_dynamic)

    print(f"\n1. SABİT EŞİK (0.55):")
    print(f"   Doğruluk: %{m_fix['acc']:.1f} ({m_fix['correct']}/{m_fix['count']}) | MAE: {m_fix['mae']:.2f} | Ortalama RTF: {m_fix['avg_rtf']:.3f}")

    print(f"\n2. DİNAMİK YÖNTEM (Silhouette/Eigengap + Centroid):")
    print(f"   Doğruluk: %{m_dyn['acc']:.1f} ({m_dyn['correct']}/{m_dyn['count']}) | MAE: {m_dyn['mae']:.2f} | Ortalama RTF: {m_dyn['avg_rtf']:.3f}")

    # 1 dakikalık seslerde 60 sn uyarısı
    over_60_fixed = [r for r in results_fixed if r["gecikme_uyarisi"]]
    over_60_dyn = [r for r in results_dynamic if r["gecikme_uyarisi"]]
    if over_60_fixed or over_60_dyn:
        print("\n⚠️ [GECİKME UYARISI]: 1 dakikalık seslerde uçtan uca süre 60 saniyeyi aşan dosyalar:")
        for r in over_60_fixed:
            print(f"   - Sabit (0.55): {r['dosya']} -> {r['sure_ort']:.2f}s (Ses süresi: {r['sure']:.2f}s)")
        for r in over_60_dyn:
            print(f"   - Dinamik     : {r['dosya']} -> {r['sure_ort']:.2f}s (Ses süresi: {r['sure']:.2f}s)")
    else:
        print("\n✅ [RTF BAŞARISI]: Tüm dosyalarda uçtan uca süre (STT dahil) 60 saniyenin altında kalmıştır.")

    # Train / Validation Ayrımı (Eğer split aktifse veya liste > 2 ise)
    if split_eval or len(items) >= 4:
        train_fixed = [r for r in results_fixed if r["row_index"] % 2 == 1]
        test_fixed = [r for r in results_fixed if r["row_index"] % 2 == 0]
        train_dyn = [r for r in results_dynamic if r["row_index"] % 2 == 1]
        test_dyn = [r for r in results_dynamic if r["row_index"] % 2 == 0]

        mt_fix = calculate_metrics(train_fixed)
        me_fix = calculate_metrics(test_fixed)
        mt_dyn = calculate_metrics(train_dyn)
        me_dyn = calculate_metrics(test_dyn)

        print("\n" + "-" * 90)
        print("TRAIN / TEST (TEK / ÇİFT SATIRLAR) AYRIŞTIRMA RAPORU")
        print("-" * 90)
        print(f"Train Seti (Tek Satırlar  - Eşik Kalibrasyon Kümesi, {mt_fix['count']} dosya):")
        print(f"  - Sabit (0.55): Doğruluk %{mt_fix['acc']:.1f} | MAE: {mt_fix['mae']:.2f}")
        print(f"  - Dinamik     : Doğruluk %{mt_dyn['acc']:.1f} | MAE: {mt_dyn['mae']:.2f}")

        print(f"\nTest Seti (Çift Satırlar - Doğrulama / Genelleme Kümesi, {me_fix['count']} dosya):")
        print(f"  - Sabit (0.55): Doğruluk %{me_fix['acc']:.1f} | MAE: {me_fix['mae']:.2f}")
        print(f"  - Dinamik     : Doğruluk %{me_dyn['acc']:.1f} | MAE: {me_dyn['mae']:.2f}")

    # Detaylı Markdown Tablosu
    print("\n### Detaylı Karşılaştırma Tablosu (Markdown)\n")
    print("| No | Dosya Adı | Gerçek | Ham Sherpa (0.55) | Final Pipeline (0.55) | Ham Sherpa (Din) | Final Pipeline (Din) | Ort. Süre | RTF | Durum |")
    print("| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for i in range(len(results_fixed)):
        rf = results_fixed[i]
        rd = results_dynamic[i]
        st_fix = "✅" if rf["dogru"] else "❌"
        st_dyn = "✅" if rd["dogru"] else "❌"
        status = "✅ Dinamik Kazandı" if (rd["dogru"] and not rf["dogru"]) else ("ℹ️ Sabit Kazandı" if (rf["dogru"] and not rd["dogru"]) else ("✅ İkisi de Doğru" if rf["dogru"] else "❌ İkisi de Hatalı"))
        print(
            f"| {rf['row_index']} | `{rf['dosya']}` | **{rf['gercek']}** | "
            f"{rf['raw_k']} | {rf['final_k']} ({st_fix}) | "
            f"{rd['raw_k']} | {rd['final_k']} ({st_dyn}) | "
            f"{rd['sure_ort']:.2f}s | {rd['rtf']:.3f} | {status} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konuşmacı Sayısı Doğruluk ve RTF Benchmark")
    parser.add_argument("--csv", default="tests/fixtures/speaker_counts.csv", help="CSV dosya yolu")
    parser.add_argument("--repeats", type=int, default=3, help="RTF için dosya başına tekrar sayısı (varsayılan: 3)")
    parser.add_argument("--split", action="store_true", help="Tek/çift satır Train-Test ayrımı yap")
    args = parser.parse_args()

    run_benchmark(csv_path=args.csv, repeats=args.repeats, split_eval=args.split)
