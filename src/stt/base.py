from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class STTResult:
    """STT output containing transcribed text."""

    text: str


class STTEngine(ABC):
    """Abstract interface for speech-to-text engines."""

    @abstractmethod
    def transcribe(
        self,
        audio: str | np.ndarray,
        sample_rate: int | None = None,
        **kwargs,
    ) -> STTResult:
        """Transcribe speech audio (file path or numpy array) to text."""
        ...
