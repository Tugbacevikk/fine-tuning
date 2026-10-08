import pytest
from unittest.mock import MagicMock

from audio_analyzer.adapters.diarization.escalating_diarizer import EscalatingDiarizer
from audio_analyzer.domain.models import DiarizationSegment


def test_escalating_diarizer_off_mode():
    primary_mock = MagicMock()
    primary_mock.diarize.return_value = [
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=1.0)
    ]
    escalation_mock = MagicMock()

    diarizer = EscalatingDiarizer(
        primary_diarizer=primary_mock,
        escalation_diarizer=escalation_mock,
        escalation_mode="off",
    )

    res = diarizer.diarize("dummy.wav", num_speakers=None)
    assert len(res) == 1
    primary_mock.diarize.assert_called_once_with("dummy.wav", num_speakers=None)
    escalation_mock.diarize.assert_not_called()


def test_escalating_diarizer_unknown_speakers_mode_triggered():
    primary_mock = MagicMock()
    escalation_mock = MagicMock()
    escalation_mock.diarize.return_value = [
        DiarizationSegment(speaker_id="SPEAKER_01", start_time=0.0, end_time=2.0)
    ]

    diarizer = EscalatingDiarizer(
        primary_diarizer=primary_mock,
        escalation_diarizer=escalation_mock,
        escalation_mode="unknown_speakers",
    )

    # num_speakers is None -> should trigger escalation
    res = diarizer.diarize("dummy.wav", num_speakers=None)
    assert len(res) == 1
    assert res[0].speaker_id == "SPEAKER_01"
    escalation_mock.diarize.assert_called_once_with("dummy.wav", num_speakers=None)
    primary_mock.diarize.assert_not_called()


def test_escalating_diarizer_unknown_speakers_mode_bypassed_when_num_speakers_given():
    primary_mock = MagicMock()
    primary_mock.diarize.return_value = [
        DiarizationSegment(speaker_id="SPEAKER_00", start_time=0.0, end_time=1.0)
    ]
    escalation_mock = MagicMock()

    diarizer = EscalatingDiarizer(
        primary_diarizer=primary_mock,
        escalation_diarizer=escalation_mock,
        escalation_mode="unknown_speakers",
    )

    # num_speakers is specified -> bypass escalation, call primary
    res = diarizer.diarize("dummy.wav", num_speakers=2)
    assert len(res) == 1
    primary_mock.diarize.assert_called_once_with("dummy.wav", num_speakers=2)
    escalation_mock.diarize.assert_not_called()
