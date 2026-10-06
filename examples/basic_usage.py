import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.stt import WhisperSTT
from src.tts import KokoroTTS

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def text_to_speech_example():
    tts = KokoroTTS(api_key=API_KEY)
    result = tts.synthesize(
        "chandu ke chacha ne chadu ki chachi ko chandi ke chammach se chatni chatai"
    )
    tts.save(result, "output.wav")
    print(
        f"Saved audio -> output.wav ({len(result.audio)} samples @ {result.sample_rate} Hz)"
    )
    return result


def speech_to_text_example(audio_path: str = "output.wav"):
    stt = WhisperSTT(api_key=API_KEY)
    result = stt.transcribe(audio_path)
    print(f"Transcription: {result.text}")
    return result


def round_trip():
    tts_result = text_to_speech_example()
    stt = WhisperSTT(api_key=API_KEY)
    stt_result = stt.transcribe(tts_result.audio, sample_rate=tts_result.sample_rate)
    print(f"Round-trip transcription: {stt_result.text}")


if __name__ == "__main__":
    round_trip()
