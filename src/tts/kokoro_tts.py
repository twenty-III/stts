from __future__ import annotations

import numpy as np
import requests

from .base import TTSEngine, TTSResult

_OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"


class KokoroTTS(TTSEngine):
    """Kokoro TTS via OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        model: str = "hexgrad/kokoro-82m",
        default_voice: str = "af_heart",
        default_speed: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._default_voice = default_voice
        self._default_speed = default_speed

    def synthesize(self, text: str, **kwargs) -> TTSResult:
        voice = kwargs.get("voice", self._default_voice)
        speed = kwargs.get("speed", self._default_speed)

        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "pcm",
        }

        resp = requests.post(
            _OPENROUTER_TTS_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()

        audio, sample_rate = self._decode_pcm(resp.content)
        return TTSResult(audio=audio, sample_rate=sample_rate)

    @staticmethod
    def _decode_pcm(raw: bytes, sample_rate: int = 24_000) -> tuple[np.ndarray, int]:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sample_rate
