import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from audio_analyzer.adapters.diarization.speechbrain_adapter import (
    SpeechBrainECAPADiarizer,
    majority_vote_filter,
)


import pytest


def test_majority_vote_filter():
    labels = np.array([0, 2, 0, 1, 1, 1, 0])
    filtered = majority_vote_filter(labels, kernel_size=3)
    # The [0, 2, 0] window at index 1 -> majority is 0, so 2 becomes 0
    assert filtered[1] == 0

@pytest.mark.torch
def test_model_dir_env_variable():
    with patch.dict(os.environ, {"MODEL_DIR": "/custom/model/path"}):
        diarizer = SpeechBrainECAPADiarizer()
        with patch("speechbrain.inference.speaker.EncoderClassifier.from_hparams") as mock_hparams:
            mock_hparams.return_value = MagicMock()
            diarizer._load_classifier()
            savedir_arg = mock_hparams.call_args[1].get("savedir")
            assert str(Path("/custom/model/path/diarization/speechbrain_ecapa")) in savedir_arg


def test_target_num_speakers_config_defaults(monkeypatch):
    from audio_analyzer.config import reset_settings, get_settings
    monkeypatch.delenv("TARGET_NUM_SPEAKERS", raising=False)
    monkeypatch.delenv("NUM_SPEAKERS", raising=False)
    reset_settings()
    assert get_settings().target_num_speakers is None

    monkeypatch.setenv("TARGET_NUM_SPEAKERS", "3")
    reset_settings()
    assert get_settings().target_num_speakers == 3
    reset_settings()


def test_diarizer_distance_threshold_branch(monkeypatch):
    from audio_analyzer.config import reset_settings
    monkeypatch.delenv("TARGET_NUM_SPEAKERS", raising=False)
    monkeypatch.delenv("NUM_SPEAKERS", raising=False)
    reset_settings()

    diarizer = SpeechBrainECAPADiarizer(num_speakers=None)
    diarizer._classifier = MagicMock()
    # Mock classifier encode_batch to return a dummy embedding tensor for 2 clips
    import torch
    dummy_emb = torch.zeros((2, 1, 192), dtype=torch.float32)
    dummy_emb[0, 0, 0] = 1.0
    dummy_emb[1, 0, 1] = 1.0
    diarizer._classifier.encode_batch.return_value = dummy_emb

    # Create dummy audio data (16000 Hz * 3 seconds)
    dummy_audio = np.random.randn(48000).astype(np.float32)

    with patch("sklearn.cluster.AgglomerativeClustering") as mock_clustering:
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.zeros(2, dtype=int)
        mock_clustering.return_value = mock_instance

        diarizer.diarize(dummy_audio)

        # Verify AgglomerativeClustering was called with distance_threshold
        mock_clustering.assert_called()
        kwargs = mock_clustering.call_args[1]
        assert "distance_threshold" in kwargs
        assert kwargs["n_clusters"] is None
    reset_settings()


def test_num_speakers_preserved_against_medfilt_reduction():
    """Doğrular: num_speakers=2 verildiğinde medfilt kısa konuşmacı turunu silip küme sayısını 1'e düşürürse raw_labels korunur."""
    diarizer = SpeechBrainECAPADiarizer(num_speakers=2)
    diarizer._classifier = MagicMock()
    import torch

    # 4 pencere segmenti, 2 ayrı konuşmacı
    dummy_emb = torch.zeros((4, 1, 192), dtype=torch.float32)
    dummy_emb[0:3, 0, 0] = 1.0  # SPEAKER_00
    dummy_emb[3, 0, 1] = 1.0    # SPEAKER_01 (1 pencere)
    diarizer._classifier.encode_batch.return_value = dummy_emb

    dummy_audio = np.random.randn(16000 * 4).astype(np.float32)

    with patch("sklearn.cluster.AgglomerativeClustering") as mock_clustering:
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 0, 1])
        mock_clustering.return_value = mock_instance

        segments = diarizer.diarize(dummy_audio, num_speakers=2)
        speakers = {s.speaker_id for s in segments}
        assert len(speakers) == 2
        assert "SPEAKER_00" in speakers
        assert "SPEAKER_01" in speakers


