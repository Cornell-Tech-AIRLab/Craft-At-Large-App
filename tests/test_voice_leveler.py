"""The voice leveler evens out quiet and loud voices without clipping."""

import numpy as np

from craft_at_large_app.audio.voice_leveler import VoiceLeveler


def _run(level: float) -> np.ndarray:
    rate = 24000
    t = np.arange(rate) / rate
    signal = (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    leveler = VoiceLeveler()
    return np.concatenate([leveler.process(signal[i : i + 480]) for i in range(0, rate, 480)])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x[len(x) // 2 :]))))


def test_quiet_voices_are_boosted_and_loud_ones_never_clip() -> None:
    """A quiet voice gets louder, a loud one a little quieter, and no sample passes the ceiling."""
    quiet, loud = _run(0.02), _run(0.6)
    assert _rms(quiet) > 3 * 0.02 / np.sqrt(2)
    assert _rms(loud) < 0.6 / np.sqrt(2)
    assert np.abs(quiet).max() <= VoiceLeveler.PEAK_CEILING
    assert np.abs(loud).max() <= VoiceLeveler.PEAK_CEILING


def test_pauses_keep_the_gain() -> None:
    """Silence between words does not pump the gain up."""
    leveler = VoiceLeveler()
    leveler.process(np.full(480, 0.3, dtype=np.float32))
    before = leveler.gain
    leveler.process(np.zeros(480, dtype=np.float32))
    assert leveler.gain == before
