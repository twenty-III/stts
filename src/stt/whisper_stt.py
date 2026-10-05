from __future__ import annotations

import base64
import io

import numpy as np
import requests

from .base import STTEngine, STTResult

_OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"


class WhisperSTT(STTEngine):
    """Whisper V3 STT via OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/whisper-large-v3-turbo",
    ) -> None:
        self._api_key = api_key
        self._model = model

    def transcribe(
        self,
        audio: str | np.ndarray,
        sample_rate: int | None = None,
        **kwargs,
    ) -> STTResult:
        audio_bytes, fmt = self._prepare_audio(audio, sample_rate)
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        payload: dict = {
            "model": self._model,
            "input_audio": {
                "data": b64_audio,
                "format": fmt,
            },
        }

        if "language" in kwargs:
            payload["language"] = kwargs["language"]

        resp = requests.post(
            _OPENROUTER_STT_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()

        data = resp.json()
        return STTResult(text=data["text"].strip())

    @staticmethod
    def _prepare_audio(
        audio: str | np.ndarray,
        sample_rate: int | None,
    ) -> tuple[bytes, str]:
        if isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio is a numpy array")
            import soundfile as sf

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV")
            return buf.getvalue(), "wav"

        path = str(audio)
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else "wav"
        with open(path, "rb") as f:
            return f.read(), ext
