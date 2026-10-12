import unittest
import tempfile
import shutil
from pathlib import Path
from scripts.veri_ekle import (
    rttm_to_audacity_labels,
    audacity_labels_to_rttm,
    write_audacity_label_file,
    read_audacity_label_file,
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


if __name__ == "__main__":
    unittest.main()
