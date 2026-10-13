import unittest
import tempfile
import shutil
import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle (PYTHONPATH=src durumunda scripts modülünün bulunabilmesi için)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.veri_ekle import (
    rttm_to_audacity_labels,
    audacity_labels_to_rttm,
    write_audacity_label_file,
    read_audacity_label_file,
    cmd_onayla,
)


class TestVeriEkle(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_rttm_and_audacity_conversion(self):
        # 1. RTTM Dosyası oluştur
        rttm_file = self.tmp_path / "test.rttm"
        with open(rttm_file, "w", encoding="utf-8") as f:
            f.write("SPEAKER sample 1 1.500 2.500 <NA> <NA> SPEAKER_00 <NA> <NA>\n")
            f.write("SPEAKER sample 1 4.200 3.100 <NA> <NA> SPEAKER_01 <NA> <NA>\n")

        # 2. RTTM -> Audacity dönüştür
        labels = rttm_to_audacity_labels(rttm_file)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0], (1.5, 4.0, "SPEAKER_00"))
        self.assertEqual(labels[1], (4.2, 7.3, "SPEAKER_01"))

        # 3. Audacity dosyasını yaz ve tekrar oku
        audacity_file = self.tmp_path / "test.txt"
        write_audacity_label_file(audacity_file, labels)
        read_labels = read_audacity_label_file(audacity_file)
        self.assertEqual(len(read_labels), 2)
        self.assertAlmostEqual(read_labels[0][0], 1.5, places=4)
        self.assertAlmostEqual(read_labels[0][1], 4.0, places=4)
        self.assertEqual(read_labels[0][2], "SPEAKER_00")

        # 4. Audacity -> RTTM dönüştür
        rttm_lines = audacity_labels_to_rttm(read_labels, "sample")
        self.assertEqual(len(rttm_lines), 2)
        self.assertIn("SPEAKER sample 1 1.500 2.500 <NA> <NA> SPEAKER_00 <NA> <NA>", rttm_lines[0])
        self.assertIn("SPEAKER sample 1 4.200 3.100 <NA> <NA> SPEAKER_01 <NA> <NA>", rttm_lines[1])

    def test_cmd_onayla_approval_workflow_and_collision_prevention(self):
        bekleyen = self.tmp_path / "_bekleyen"
        bekleyen.mkdir(parents=True, exist_ok=True)
        hedef = self.tmp_path / "real_turkish"
        hedef.mkdir(parents=True, exist_ok=True)

        # 1. Bekleyen kuyruğuna ilk dosyayı koy
        (bekleyen / "test_call.rttm").write_text(
            "SPEAKER test_call 1 0.000 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n", encoding="utf-8"
        )
        (bekleyen / "test_call.wav").write_bytes(b"dummy_wav_1")

        cmd_onayla(bekleyen_klasor=bekleyen, hedef_klasor=hedef)

        # Hedefe taşınmış ve bekleyenden silinmiş olmalı
        self.assertTrue((hedef / "test_call.rttm").exists())
        self.assertTrue((hedef / "test_call.wav").exists())
        self.assertEqual((hedef / "test_call.wav").read_bytes(), b"dummy_wav_1")
        self.assertFalse((bekleyen / "test_call.rttm").exists())
        self.assertFalse((bekleyen / "test_call.wav").exists())

        # 2. Aynı isimde ikinci bir dosya bekleyene gelirse: üzerine yazmamalı, test_call_1 yapmalı
        (bekleyen / "test_call.rttm").write_text(
            "SPEAKER test_call 1 1.000 3.000 <NA> <NA> SPEAKER_01 <NA> <NA>\n", encoding="utf-8"
        )
        (bekleyen / "test_call.wav").write_bytes(b"dummy_wav_2")

        cmd_onayla(bekleyen_klasor=bekleyen, hedef_klasor=hedef)

        self.assertTrue((hedef / "test_call.rttm").exists())
        self.assertEqual((hedef / "test_call.wav").read_bytes(), b"dummy_wav_1")  # İlk dosya korundu!
        self.assertTrue((hedef / "test_call_1.rttm").exists())
        self.assertEqual((hedef / "test_call_1.wav").read_bytes(), b"dummy_wav_2")
        self.assertFalse((bekleyen / "test_call.rttm").exists())



if __name__ == "__main__":
    unittest.main()
