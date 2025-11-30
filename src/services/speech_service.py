import io
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger("speech_service")


class SpeechService:
    def __init__(self):
        self.tts_model = None
        self.whisper_model = None
        self.device = settings.device
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        logger.info("Loading speech models")
        try:
            # Load TTS model (Coqui TTS)
            self._load_tts_model()

            # Load Whisper model for speech-to-text
            self._load_whisper_model()

            self._initialized = True
            logger.info("Speech models loaded successfully")
        except Exception as e:
            logger.error("Failed to load speech models", error=str(e))
            # Don't raise - allow service to run with partial functionality
            logger.warning("Speech service will have limited functionality")

    def _load_tts_model(self):
        """Load TTS model for text-to-speech generation"""
        try:
            from TTS.api import TTS

            # Use a fast, high-quality model
            self.tts_model = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False,
                gpu=(self.device == "cuda"),
            )
            logger.info("TTS model loaded")
        except Exception as e:
            logger.warning("TTS model not available", error=str(e))
            self.tts_model = None

    def _load_whisper_model(self):
        """Load Whisper model for speech-to-text"""
        try:
            import whisper

            # Use base model for balance of speed and accuracy
            model_size = "base"
            if Path(settings.whisper_model_path).exists():
                self.whisper_model = whisper.load_model(
                    model_size, download_root=settings.whisper_model_path
                )
            else:
                self.whisper_model = whisper.load_model(model_size)
            logger.info("Whisper model loaded", model=model_size)
        except Exception as e:
            logger.warning("Whisper model not available", error=str(e))
            self.whisper_model = None

    async def generate(
        self,
        prompt: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> dict:
        """Generate speech audio from text"""
        if not self._initialized:
            self.initialize()

        start = time.time()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{hash(prompt) & 0xFFFFFF:06x}.wav"
        output_path = settings.output_dir / "audio" / filename

        if self.tts_model is None:
            logger.error("TTS model not available")
            return {
                "status": "error",
                "error": "TTS model not initialized",
                "file_path": None,
                "generation_time": time.time() - start,
            }

        try:
            logger.info(
                "Generating speech",
                text_length=len(prompt),
                voice=voice,
                language=language,
            )

            # Generate speech
            self.tts_model.tts_to_file(
                text=prompt,
                file_path=str(output_path),
            )

            generation_time = time.time() - start
            logger.info("Speech generated", file=filename, time=generation_time)

            return {
                "status": "completed",
                "file_path": f"/outputs/audio/{filename}",
                "generation_time": generation_time,
            }

        except Exception as e:
            logger.error("Speech generation failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "file_path": None,
                "generation_time": time.time() - start,
            }

    async def to_text(self, audio_data: bytes) -> dict:
        """Convert speech audio to text using Whisper"""
        if not self._initialized:
            self.initialize()

        start = time.time()

        if self.whisper_model is None:
            logger.error("Whisper model not available")
            return {
                "text": "",
                "language": None,
                "confidence": 0.0,
                "error": "Whisper model not initialized",
            }

        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            logger.info("Transcribing audio", file_size=len(audio_data))

            # Transcribe
            result = self.whisper_model.transcribe(tmp_path)

            # Clean up temp file
            import os

            os.unlink(tmp_path)

            transcription_time = time.time() - start
            logger.info(
                "Audio transcribed",
                text_length=len(result["text"]),
                language=result.get("language"),
                time=transcription_time,
            )

            return {
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
                "confidence": 0.95,  # Whisper doesn't provide confidence directly
            }

        except Exception as e:
            logger.error("Speech-to-text failed", error=str(e))
            return {
                "text": "",
                "language": None,
                "confidence": 0.0,
                "error": str(e),
            }

    async def detect_language(self, audio_data: bytes) -> dict:
        """Detect the language of spoken audio"""
        if not self._initialized:
            self.initialize()

        if self.whisper_model is None:
            return {"language": None, "error": "Whisper model not initialized"}

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            import whisper

            # Load audio and detect language
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
            _, probs = self.whisper_model.detect_language(mel)

            import os

            os.unlink(tmp_path)

            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]

            return {
                "language": detected_lang,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error("Language detection failed", error=str(e))
            return {"language": None, "error": str(e)}


_service = None


def get_speech_service():
    global _service
    if _service is None:
        _service = SpeechService()
    return _service
