"""
Speech Intelligence (STT -> Agent -> TTS)
------------------------------------------
Provides the speech interfaces used by NeuroForgeAI.

Current implementation:
- Robust Base64 audio decoding
- Mock STT for end-to-end testing
- Mock TTS for end-to-end testing

Production STT/TTS can later replace MockSTT/MockTTS without
changing the rest of the application.
"""

from __future__ import annotations

import abc
import base64
import binascii
import hashlib
from dataclasses import dataclass
import base64
import binascii

# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    duration_sec: float


@dataclass
class SynthesisResult:
    audio_url: str
    voice: str
    duration_sec: float


# ============================================================
# ABSTRACT SPEECH INTERFACES
# ============================================================

class SpeechToText(abc.ABC):
    """Base interface for speech-to-text implementations."""

    @abc.abstractmethod
    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        ...


class TextToSpeech(abc.ABC):
    """Base interface for text-to-speech implementations."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
    ) -> SynthesisResult:
        ...


# ============================================================
# MOCK STT
# ============================================================

class MockSTT(SpeechToText):
    """
    Deterministic mock STT.

    This does NOT perform real speech recognition.

    It attempts to interpret incoming bytes as UTF-8 text.
    Real WAV/MP3/OGG audio will normally produce:

        <unintelligible audio>

    This implementation exists so the entire NeuroForge
    pipeline can be tested without an external STT service.
    """

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:

        if not audio_bytes:
            return TranscriptionResult(
                text="<empty audio>",
                confidence=0.0,
                duration_sec=0.0,
            )

        try:
            text = audio_bytes.decode("utf-8").strip()

        except UnicodeDecodeError:
            text = "<unintelligible audio>"

        # Prevent empty decoded text from looking like valid speech.
        if not text:
            text = "<unintelligible audio>"

        # Simple mock duration heuristic.
        words = max(1, len(text.split()))
        duration = round(words / 2.5, 2)

        return TranscriptionResult(
            text=text,
            confidence=0.97,
            duration_sec=duration,
        )


# ============================================================
# MOCK TTS
# ============================================================

class MockTTS(TextToSpeech):
    """
    Deterministic mock TTS.

    Does not generate real audio.

    Returns a stable pseudo URL based on a SHA-256 hash of
    the generated text.
    """

    def synthesize(
        self,
        text: str,
        voice: str = "default",
    ) -> SynthesisResult:

        if not text:
            text = ""

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:16]

        words = max(1, len(text.split()))
        duration = round(words / 2.5, 2)

        return SynthesisResult(
            audio_url=f"mock://tts/{voice}/{digest}.wav",
            voice=voice,
            duration_sec=duration,
        )


# ============================================================
# BASE64 AUDIO DECODER
# ============================================================

def decode_base64_audio(b64_audio: str) -> bytes:
    """
    Decode Base64 encoded audio safely.

    Supports:

    1. Normal Base64
       UklGR...

    2. Data URLs
       data:audio/wav;base64,UklGR...

    3. Base64 containing whitespace/newlines

    4. Missing '=' padding
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not b64_audio:
        raise ValueError(
            "audio_base64 is empty"
        )

    if not isinstance(b64_audio, str):
        raise ValueError(
            "audio_base64 must be a string"
        )

    # --------------------------------------------------------
    # Support browser-style data URLs
    # --------------------------------------------------------

    if (
        b64_audio.startswith("data:")
        and "," in b64_audio
    ):
        b64_audio = b64_audio.split(
            ",",
            1,
        )[1]

    # --------------------------------------------------------
    # Remove spaces/newlines/tabs
    # --------------------------------------------------------

    b64_audio = "".join(
        b64_audio.split()
    )

    if not b64_audio:
        raise ValueError(
            "audio_base64 contains no data"
        )

    # --------------------------------------------------------
    # Restore missing Base64 padding
    # --------------------------------------------------------

    padding = (-len(b64_audio)) % 4

    if padding:
        b64_audio += "=" * padding

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    try:

        audio_bytes = base64.b64decode(
            b64_audio,
            validate=True,
        )

    except binascii.Error as exc:

        raise ValueError(
            "Invalid Base64 audio payload"
        ) from exc

    # --------------------------------------------------------
    # Ensure decoded data isn't empty
    # --------------------------------------------------------

    if not audio_bytes:
        raise ValueError(
            "Decoded audio payload is empty"
        )

    return audio_bytes


# ============================================================
# SINGLETONS
# ============================================================

stt_singleton: SpeechToText = MockSTT()

tts_singleton: TextToSpeech = MockTTS()