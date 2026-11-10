"""Even out the loudness of the robot's speech.

Some voices are recorded much quieter than others, which is hard to hear in a crafting circle. The
leveler measures the speech level in short chunks and moves a gain toward a steady target: quiet
voices are boosted (up to MAX_GAIN), loud ones are turned down slightly, and peaks are never pushed
past PEAK_CEILING, so nothing clips. Near-silent chunks (pauses between words) leave the gain alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class VoiceLeveler:
    """Automatic gain for float32 speech audio in [-1, 1]."""

    TARGET_RMS = 0.12  # about -18 dBFS: clear speech level
    SILENCE_RMS = 0.004  # below this a chunk is a pause; the gain holds
    MIN_GAIN = 0.6
    MAX_GAIN = 4.0
    PEAK_CEILING = 0.95
    ATTACK = 0.5  # how fast the gain drops when speech gets loud (per chunk)
    RELEASE = 0.08  # how fast it rises when speech gets quiet (per chunk)

    def __init__(self) -> None:
        """Start at unity gain."""
        self.gain = 1.0

    def reset(self) -> None:
        """Forget the current gain, e.g. when a new reply starts."""
        self.gain = 1.0

    def process(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the chunk with the leveling gain applied."""
        if chunk.size == 0:
            return chunk
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        if rms >= self.SILENCE_RMS:
            wanted = min(self.MAX_GAIN, max(self.MIN_GAIN, self.TARGET_RMS / rms))
            rate = self.ATTACK if wanted < self.gain else self.RELEASE
            self.gain += (wanted - self.gain) * rate
        peak = float(np.max(np.abs(chunk)))
        gain = self.gain if peak == 0.0 else min(self.gain, self.PEAK_CEILING / peak)
        return (chunk * gain).astype(np.float32, copy=False)
