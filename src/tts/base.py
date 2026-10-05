from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TTSResult:
    """TTS output containing audio samples and sample rate."""
    audio: np.ndarray
    sample_rate: int


class TTSEngine(ABC):
    """Abstract interface for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        """Convert text to speech."""
        ...

    def save(self, result: TTSResult, path: str) -> None:
        """Save synthesized audio to a WAV file."""
        import soundfile as sf

        sf.write(path, result.audio, result.sample_rate)
