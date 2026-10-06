import os
import sys

# Ensure we can import 'src' from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.stt import WhisperSTT

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def transcribe_local_file(audio_path: str = "input.wav"):
    """Reads a local audio file and transcribes it using Whisper via OpenRouter."""

    if not os.path.exists(audio_path):
        print(f"❌ Error: Could not find '{audio_path}'!")
        print(
            "Please place an audio file with that name in the directory and try again."
        )
        return

    print(f"🎧 Found '{audio_path}'. Sending to OpenRouter for transcription...")

    stt = WhisperSTT(api_key=API_KEY)

    # The transcribe method automatically reads the file and handles base64 encoding
    result = stt.transcribe(audio_path)

    print("\n📝 Transcription Result:")
    print("-" * 30)
    print(result.text)
    print("-" * 30)


if __name__ == "__main__":
    transcribe_local_file("input.wav")
