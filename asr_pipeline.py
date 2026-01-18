"""
ASR Pipeline for Depression Detection

This module implements the Automatic Speech Recognition pipeline using IndicWhisper
for transcribing Tamil and Malayalam audio to text.

Reference: Multimodal NLP Upgrade - Requirement 1
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import torch
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        pipeline
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

from config import ASRConfig, PreprocessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Language code mapping for IndicWhisper
LANGUAGE_CODES = {
    "tamil": "ta",
    "malayalam": "ml",
    "ta": "ta",
    "ml": "ml"
}


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio with timestamps."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class TranscriptResult:
    """Result of ASR transcription for a single audio file."""
    text: str
    language: str
    confidence: float
    segments: List[TranscriptSegment]
    audio_path: str
    speaker_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "audio_file": os.path.basename(self.audio_path),
            "speaker_id": self.speaker_id,
            "language": self.language,
            "transcript": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "segments": [
                {
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "confidence": seg.confidence
                }
                for seg in self.segments
            ],
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], audio_path: str = "") -> "TranscriptResult":
        """Create TranscriptResult from dictionary."""
        segments = [
            TranscriptSegment(
                text=seg["text"],
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                confidence=seg.get("confidence", 1.0)
            )
            for seg in data.get("segments", [])
        ]
        return cls(
            text=data.get("transcript", ""),
            language=data.get("language", ""),
            confidence=data.get("confidence", 0.0),
            segments=segments,
            audio_path=audio_path or data.get("audio_file", ""),
            speaker_id=data.get("speaker_id", ""),
            timestamp=data.get("timestamp", ""),
            error=data.get("error")
        )


class VADChunker:
    """Voice Activity Detection based chunker for long audio files.
    
    Segments audio into chunks based on speech activity to handle
    audio files longer than the model's context window.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_length_s: int = 30,
        frame_size_ms: int = 30,
        aggressiveness: int = 3
    ):
        self.sample_rate = sample_rate
        self.chunk_length_samples = chunk_length_s * sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_samples = int(sample_rate * frame_size_ms / 1000)
        self.aggressiveness = aggressiveness
        
        if HAS_WEBRTCVAD:
            self.vad = webrtcvad.Vad(aggressiveness)
        else:
            self.vad = None
            logger.warning("webrtcvad not available. Using simple chunking.")
    
    def _detect_speech_frames(self, audio: np.ndarray) -> List[bool]:
        """Detect which frames contain speech."""
        if self.vad is None:
            # Fallback: energy-based detection
            return self._energy_based_detection(audio)
        
        # Convert to 16-bit PCM for WebRTC VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        num_frames = len(audio_int16) // self.frame_samples
        
        speech_frames = []
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame_bytes = audio_int16[start:end].tobytes()
            
            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                speech_frames.append(is_speech)
            except Exception:
                speech_frames.append(True)  # Assume speech on error
        
        return speech_frames
    
    def _energy_based_detection(self, audio: np.ndarray) -> List[bool]:
        """Simple energy-based speech detection."""
        num_frames = len(audio) // self.frame_samples
        threshold = np.percentile(np.abs(audio), 30)  # 30th percentile as threshold
        
        speech_frames = []
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame_energy = np.sqrt(np.mean(audio[start:end] ** 2))
            speech_frames.append(frame_energy > threshold)
        
        return speech_frames
    
    def chunk_audio(
        self,
        audio: np.ndarray,
        use_vad: bool = True
    ) -> List[Dict[str, Any]]:
        """Chunk audio into segments suitable for ASR.
        
        Args:
            audio: Audio waveform as numpy array (mono, 16kHz)
            use_vad: Whether to use VAD for intelligent chunking
            
        Returns:
            List of dictionaries with 'audio', 'start_time', 'end_time' keys
        """
        total_samples = len(audio)
        total_duration = total_samples / self.sample_rate
        
        # If audio is short enough, return as single chunk
        if total_samples <= self.chunk_length_samples:
            return [{
                "audio": audio,
                "start_time": 0.0,
                "end_time": total_duration
            }]
        
        chunks = []
        
        if use_vad and self.vad is not None:
            # VAD-based chunking: find natural speech boundaries
            speech_frames = self._detect_speech_frames(audio)
            chunks = self._chunk_by_speech_boundaries(audio, speech_frames)
        else:
            # Simple fixed-length chunking with overlap
            chunks = self._simple_chunk(audio)
        
        return chunks
    
    def _chunk_by_speech_boundaries(
        self,
        audio: np.ndarray,
        speech_frames: List[bool]
    ) -> List[Dict[str, Any]]:
        """Chunk audio at speech boundaries."""
        chunks = []
        current_start = 0
        
        while current_start < len(audio):
            # Determine chunk end
            chunk_end = min(current_start + self.chunk_length_samples, len(audio))
            
            # If not at the end, try to find a silence boundary
            if chunk_end < len(audio):
                # Look for silence in the last 20% of the chunk
                search_start = int(current_start + 0.8 * self.chunk_length_samples)
                search_start_frame = search_start // self.frame_samples
                search_end_frame = chunk_end // self.frame_samples
                
                # Find last silence frame in search region
                best_boundary = chunk_end
                for frame_idx in range(min(search_end_frame, len(speech_frames)) - 1, 
                                       max(search_start_frame, 0), -1):
                    if frame_idx < len(speech_frames) and not speech_frames[frame_idx]:
                        best_boundary = (frame_idx + 1) * self.frame_samples
                        break
                
                chunk_end = best_boundary
            
            # Extract chunk
            chunk_audio = audio[current_start:chunk_end]
            start_time = current_start / self.sample_rate
            end_time = chunk_end / self.sample_rate
            
            chunks.append({
                "audio": chunk_audio,
                "start_time": start_time,
                "end_time": end_time
            })
            
            current_start = chunk_end
        
        return chunks
    
    def _simple_chunk(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Simple fixed-length chunking."""
        chunks = []
        current_start = 0
        overlap_samples = int(self.chunk_length_samples * 0.1)  # 10% overlap
        
        while current_start < len(audio):
            chunk_end = min(current_start + self.chunk_length_samples, len(audio))
            chunk_audio = audio[current_start:chunk_end]
            
            start_time = current_start / self.sample_rate
            end_time = chunk_end / self.sample_rate
            
            chunks.append({
                "audio": chunk_audio,
                "start_time": start_time,
                "end_time": end_time
            })
            
            # Move to next chunk with overlap
            current_start = chunk_end - overlap_samples
            if current_start >= len(audio) - overlap_samples:
                break
        
        return chunks


class ASRPipeline:
    """ASR Pipeline using IndicWhisper for Tamil and Malayalam transcription.
    
    Implements Requirement 1: Automatic Speech Recognition
    - Transcribes audio to text in Tamil or Malayalam
    - Handles code-mixed speech (Tanglish/Manglish)
    - Uses VAD-based chunking for long audio
    - Returns punctuated and normalized text
    """
    
    def __init__(self, config: ASRConfig):
        """Initialize ASR Pipeline.
        
        Args:
            config: ASRConfig with model and processing settings
        """
        self.config = config
        self.device = self._get_device(config.device)
        self.sample_rate = 16000  # Whisper expects 16kHz
        
        # Initialize VAD chunker
        self.vad_chunker = VADChunker(
            sample_rate=self.sample_rate,
            chunk_length_s=config.chunk_length_s
        )
        
        # Load model and processor
        self.model = None
        self.processor = None
        self.pipe = None
        self._load_model()
    
    def _get_device(self, device_str: str) -> str:
        """Determine the best available device."""
        if device_str == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device_str == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the IndicWhisper model and processor."""
        if not HAS_TRANSFORMERS:
            logger.error("transformers library not installed. ASR will not work.")
            return
        
        try:
            logger.info(f"Loading ASR model: {self.config.model_name}")
            
            # Try to load IndicWhisper, fall back to base Whisper if not available
            try:
                self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.config.model_name
                )
            except Exception as e:
                logger.warning(f"Could not load {self.config.model_name}: {e}")
                logger.info("Falling back to openai/whisper-base")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-base"
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=0 if self.device == "cuda" else -1,
                chunk_length_s=self.config.chunk_length_s,
                return_timestamps=self.config.return_timestamps
            )
            
            logger.info(f"ASR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            self.model = None
            self.processor = None
            self.pipe = None
    
    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array (mono, 16kHz) or None on error
        """
        try:
            if HAS_SOUNDFILE:
                audio, sr = sf.read(audio_path, dtype='float32')
            elif HAS_LIBROSA:
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
            else:
                import torchaudio
                waveform, sr = torchaudio.load(audio_path)
                audio = waveform.squeeze().numpy()
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sr != self.sample_rate:
                if HAS_LIBROSA:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                else:
                    # Simple resampling
                    ratio = self.sample_rate / sr
                    new_length = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
                    audio = audio[indices]
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None
    
    def _extract_speaker_id(self, audio_path: str) -> str:
        """Extract speaker ID from audio filename."""
        from preprocessing import SpeakerIDExtractor
        filename = os.path.basename(audio_path)
        return SpeakerIDExtractor.extract(filename)
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "tamil"
    ) -> TranscriptResult:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Target language ("tamil" or "malayalam")
            
        Returns:
            TranscriptResult with transcription and metadata
        """
        speaker_id = self._extract_speaker_id(audio_path)
        
        # Validate language
        if language.lower() not in LANGUAGE_CODES:
            return TranscriptResult(
                text="",
                language=language,
                confidence=0.0,
                segments=[],
                audio_path=audio_path,
                speaker_id=speaker_id,
                error=f"Unsupported language: {language}"
            )
        
        lang_code = LANGUAGE_CODES[language.lower()]
        
        # Check if model is loaded
        if self.model is None or self.processor is None:
            return TranscriptResult(
                text="",
                language=language,
                confidence=0.0,
                segments=[],
                audio_path=audio_path,
                speaker_id=speaker_id,
                error="ASR model not loaded"
            )
        
        # Load audio
        audio = self._load_audio(audio_path)
        if audio is None:
            return TranscriptResult(
                text="",
                language=language,
                confidence=0.0,
                segments=[],
                audio_path=audio_path,
                speaker_id=speaker_id,
                error="Failed to load audio file"
            )
        
        # Check audio quality (minimum duration)
        duration = len(audio) / self.sample_rate
        if duration < 0.5:
            return TranscriptResult(
                text="",
                language=language,
                confidence=0.0,
                segments=[],
                audio_path=audio_path,
                speaker_id=speaker_id,
                error=f"Audio too short: {duration:.2f}s"
            )
        
        try:
            return self._transcribe_audio(audio, audio_path, language, lang_code, speaker_id)
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return TranscriptResult(
                text="",
                language=language,
                confidence=0.0,
                segments=[],
                audio_path=audio_path,
                speaker_id=speaker_id,
                error=str(e)
            )
    
    def _transcribe_audio(
        self,
        audio: np.ndarray,
        audio_path: str,
        language: str,
        lang_code: str,
        speaker_id: str
    ) -> TranscriptResult:
        """Internal method to transcribe audio array."""
        duration = len(audio) / self.sample_rate
        
        # Use VAD-based chunking for long audio (> 30 seconds)
        if duration > self.config.chunk_length_s:
            return self._transcribe_with_vad(
                audio, audio_path, language, lang_code, speaker_id
            )
        
        # Direct transcription for short audio
        if self.pipe is not None:
            result = self.pipe(
                audio,
                generate_kwargs={"language": lang_code, "task": "transcribe"}
            )
            
            text = result.get("text", "").strip()
            chunks = result.get("chunks", [])
            
            segments = []
            if chunks:
                for chunk in chunks:
                    segments.append(TranscriptSegment(
                        text=chunk.get("text", "").strip(),
                        start_time=chunk.get("timestamp", [0, 0])[0] or 0,
                        end_time=chunk.get("timestamp", [0, duration])[1] or duration,
                        confidence=1.0
                    ))
            else:
                segments.append(TranscriptSegment(
                    text=text,
                    start_time=0.0,
                    end_time=duration,
                    confidence=1.0
                ))
            
            return TranscriptResult(
                text=text,
                language=language,
                confidence=1.0,
                segments=segments,
                audio_path=audio_path,
                speaker_id=speaker_id
            )
        
        # Fallback: manual processing
        input_features = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=lang_code,
                task="transcribe"
            )
        
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return TranscriptResult(
            text=text,
            language=language,
            confidence=1.0,
            segments=[TranscriptSegment(
                text=text,
                start_time=0.0,
                end_time=duration,
                confidence=1.0
            )],
            audio_path=audio_path,
            speaker_id=speaker_id
        )
    
    def _transcribe_with_vad(
        self,
        audio: np.ndarray,
        audio_path: str,
        language: str,
        lang_code: str,
        speaker_id: str
    ) -> TranscriptResult:
        """Transcribe long audio using VAD-based chunking.
        
        Implements Requirement 1.3: VAD segmentation for audio > 30 seconds
        """
        # Chunk audio using VAD
        chunks = self.vad_chunker.chunk_audio(audio, use_vad=True)
        
        all_segments = []
        all_text_parts = []
        
        for chunk_info in chunks:
            chunk_audio = chunk_info["audio"]
            start_time = chunk_info["start_time"]
            end_time = chunk_info["end_time"]
            
            # Skip very short chunks
            if len(chunk_audio) < self.sample_rate * 0.5:
                continue
            
            try:
                if self.pipe is not None:
                    result = self.pipe(
                        chunk_audio,
                        generate_kwargs={"language": lang_code, "task": "transcribe"}
                    )
                    chunk_text = result.get("text", "").strip()
                else:
                    input_features = self.processor(
                        chunk_audio,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    ).input_features.to(self.device)
                    
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            input_features,
                            language=lang_code,
                            task="transcribe"
                        )
                    
                    chunk_text = self.processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0].strip()
                
                if chunk_text:
                    all_text_parts.append(chunk_text)
                    all_segments.append(TranscriptSegment(
                        text=chunk_text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=1.0
                    ))
                    
            except Exception as e:
                logger.warning(f"Failed to transcribe chunk {start_time}-{end_time}: {e}")
        
        # Combine all text
        full_text = " ".join(all_text_parts)
        
        return TranscriptResult(
            text=full_text,
            language=language,
            confidence=1.0 if all_segments else 0.0,
            segments=all_segments,
            audio_path=audio_path,
            speaker_id=speaker_id
        )
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: str = "tamil",
        output_dir: Optional[str] = None
    ) -> List[TranscriptResult]:
        """Transcribe a batch of audio files.
        
        Implements Requirement 1.6: Batch transcription with matching identifiers
        
        Args:
            audio_paths: List of paths to audio files
            language: Target language for all files
            output_dir: Optional directory to save transcript JSON files
            
        Returns:
            List of TranscriptResult objects (same order as input)
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            logger.info(f"Transcribing [{i+1}/{len(audio_paths)}]: {audio_path}")
            result = self.transcribe(audio_path, language)
            results.append(result)
            
            # Save transcript if output directory specified
            if output_dir is not None:
                self._save_transcript(result, output_dir)
        
        return results
    
    def _save_transcript(
        self,
        result: TranscriptResult,
        output_dir: str
    ) -> str:
        """Save transcript result to JSON file.
        
        Args:
            result: TranscriptResult to save
            output_dir: Directory to save the JSON file
            
        Returns:
            Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        audio_basename = os.path.splitext(os.path.basename(result.audio_path))[0]
        json_filename = f"{audio_basename}_transcript.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # Save as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved transcript: {json_path}")
        return json_path
    
    def load_transcript(self, json_path: str) -> Optional[TranscriptResult]:
        """Load a transcript from JSON file.
        
        Args:
            json_path: Path to transcript JSON file
            
        Returns:
            TranscriptResult or None if loading fails
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TranscriptResult.from_dict(data, json_path)
        except Exception as e:
            logger.error(f"Failed to load transcript {json_path}: {e}")
            return None
    
    def get_transcript_path(self, audio_path: str, output_dir: str) -> str:
        """Get the expected transcript JSON path for an audio file.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory where transcripts are stored
            
        Returns:
            Expected path to the transcript JSON file
        """
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(output_dir, f"{audio_basename}_transcript.json")
    
    def transcript_exists(self, audio_path: str, output_dir: str) -> bool:
        """Check if a transcript already exists for an audio file.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory where transcripts are stored
            
        Returns:
            True if transcript exists, False otherwise
        """
        transcript_path = self.get_transcript_path(audio_path, output_dir)
        return os.path.exists(transcript_path)
    
    def transcribe_directory(
        self,
        audio_dir: str,
        language: str,
        output_dir: Optional[str] = None,
        label: Optional[int] = None,
        skip_existing: bool = True
    ) -> List[TranscriptResult]:
        """Transcribe all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            language: Target language ("tamil" or "malayalam")
            output_dir: Directory to save transcript JSON files
            label: Optional label to include in transcript metadata
            skip_existing: Skip files that already have transcripts
            
        Returns:
            List of TranscriptResult objects
        """
        if not os.path.exists(audio_dir):
            logger.error(f"Directory not found: {audio_dir}")
            return []
        
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_files = []
        
        for filename in sorted(os.listdir(audio_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in audio_extensions:
                audio_files.append(os.path.join(audio_dir, filename))
        
        logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
        
        # Filter out files with existing transcripts if requested
        if skip_existing and output_dir:
            files_to_process = []
            for audio_path in audio_files:
                if not self.transcript_exists(audio_path, output_dir):
                    files_to_process.append(audio_path)
                else:
                    logger.info(f"Skipping (transcript exists): {audio_path}")
            audio_files = files_to_process
            logger.info(f"Processing {len(audio_files)} files (skipped existing)")
        
        # Transcribe all files
        results = []
        for i, audio_path in enumerate(audio_files):
            logger.info(f"Transcribing [{i+1}/{len(audio_files)}]: {os.path.basename(audio_path)}")
            result = self.transcribe(audio_path, language)
            
            # Add label to result if provided
            if label is not None and output_dir:
                result_dict = result.to_dict()
                result_dict["label"] = label
                
                # Save with label
                os.makedirs(output_dir, exist_ok=True)
                json_path = self.get_transcript_path(audio_path, output_dir)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
            elif output_dir:
                self._save_transcript(result, output_dir)
            
            results.append(result)
        
        return results
    
    def load_all_transcripts(
        self,
        transcript_dir: str
    ) -> Dict[str, TranscriptResult]:
        """Load all transcripts from a directory.
        
        Args:
            transcript_dir: Directory containing transcript JSON files
            
        Returns:
            Dictionary mapping audio filename to TranscriptResult
        """
        transcripts = {}
        
        if not os.path.exists(transcript_dir):
            logger.warning(f"Transcript directory not found: {transcript_dir}")
            return transcripts
        
        for filename in os.listdir(transcript_dir):
            if filename.endswith('_transcript.json'):
                json_path = os.path.join(transcript_dir, filename)
                result = self.load_transcript(json_path)
                if result:
                    # Extract original audio filename
                    audio_filename = filename.replace('_transcript.json', '.wav')
                    transcripts[audio_filename] = result
        
        logger.info(f"Loaded {len(transcripts)} transcripts from {transcript_dir}")
        return transcripts


class TranscriptManager:
    """Manages transcript storage and retrieval.
    
    Provides utilities for:
    - Saving transcripts with metadata
    - Loading transcripts by audio file
    - Batch operations on transcript directories
    """
    
    def __init__(self, base_output_dir: str = "transcripts"):
        """Initialize TranscriptManager.
        
        Args:
            base_output_dir: Base directory for storing transcripts
        """
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
    
    def get_language_dir(self, language: str) -> str:
        """Get the transcript directory for a specific language."""
        return os.path.join(self.base_output_dir, language.lower())
    
    def get_class_dir(self, language: str, label: int) -> str:
        """Get the transcript directory for a specific language and class."""
        class_name = "depressed" if label == 1 else "non_depressed"
        return os.path.join(self.get_language_dir(language), class_name)
    
    def save_transcript(
        self,
        result: TranscriptResult,
        language: str,
        label: int
    ) -> str:
        """Save a transcript with proper directory structure.
        
        Args:
            result: TranscriptResult to save
            language: Language of the transcript
            label: Class label (0 or 1)
            
        Returns:
            Path to saved JSON file
        """
        output_dir = self.get_class_dir(language, label)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create transcript data with label
        data = result.to_dict()
        data["label"] = label
        data["normalized_text"] = result.text  # Placeholder for preprocessing
        data["morphemes"] = []  # Placeholder for morphological segmentation
        
        # Generate output path
        audio_basename = os.path.splitext(os.path.basename(result.audio_path))[0]
        json_path = os.path.join(output_dir, f"{audio_basename}_transcript.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return json_path
    
    def load_transcript(
        self,
        audio_path: str,
        language: str,
        label: int
    ) -> Optional[TranscriptResult]:
        """Load a transcript for a specific audio file.
        
        Args:
            audio_path: Path to the original audio file
            language: Language of the transcript
            label: Class label (0 or 1)
            
        Returns:
            TranscriptResult or None if not found
        """
        output_dir = self.get_class_dir(language, label)
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(output_dir, f"{audio_basename}_transcript.json")
        
        if not os.path.exists(json_path):
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TranscriptResult.from_dict(data, audio_path)
        except Exception as e:
            logger.error(f"Failed to load transcript {json_path}: {e}")
            return None
    
    def load_all_transcripts(
        self,
        language: Optional[str] = None,
        label: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load all transcripts matching the criteria.
        
        Args:
            language: Optional language filter
            label: Optional label filter
            
        Returns:
            List of transcript dictionaries with metadata
        """
        transcripts = []
        
        # Determine directories to search
        if language and label is not None:
            dirs = [self.get_class_dir(language, label)]
        elif language:
            dirs = [
                self.get_class_dir(language, 0),
                self.get_class_dir(language, 1)
            ]
        else:
            dirs = []
            for lang in ["tamil", "malayalam"]:
                dirs.extend([
                    self.get_class_dir(lang, 0),
                    self.get_class_dir(lang, 1)
                ])
        
        # Load transcripts from each directory
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            
            for filename in os.listdir(dir_path):
                if filename.endswith('_transcript.json'):
                    json_path = os.path.join(dir_path, filename)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        transcripts.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to load {json_path}: {e}")
        
        return transcripts
    
    def get_transcript_stats(self) -> Dict[str, Any]:
        """Get statistics about stored transcripts.
        
        Returns:
            Dictionary with transcript counts by language and class
        """
        stats = {
            "total": 0,
            "by_language": {},
            "by_class": {"depressed": 0, "non_depressed": 0}
        }
        
        for language in ["tamil", "malayalam"]:
            stats["by_language"][language] = {"depressed": 0, "non_depressed": 0}
            
            for label, class_name in [(1, "depressed"), (0, "non_depressed")]:
                dir_path = self.get_class_dir(language, label)
                if os.path.exists(dir_path):
                    count = len([f for f in os.listdir(dir_path) 
                                if f.endswith('_transcript.json')])
                    stats["by_language"][language][class_name] = count
                    stats["by_class"][class_name] += count
                    stats["total"] += count
        
        return stats


def create_asr_pipeline(config: Optional[ASRConfig] = None) -> ASRPipeline:
    """Factory function to create ASR pipeline.
    
    Args:
        config: Optional ASRConfig. Uses default if not provided.
        
    Returns:
        Configured ASRPipeline instance
    """
    if config is None:
        config = ASRConfig()
    return ASRPipeline(config)


if __name__ == "__main__":
    # Test ASR pipeline
    from config import get_config
    
    config = get_config("combined")
    
    print("Creating ASR Pipeline...")
    asr = create_asr_pipeline(config.asr)
    
    # Test with a sample file if available
    test_file = "NLP Dataset/Malayalam/Depressed/Train_set/D_A001_1.wav"
    if os.path.exists(test_file):
        print(f"\nTranscribing: {test_file}")
        result = asr.transcribe(test_file, language="malayalam")
        print(f"Text: {result.text}")
        print(f"Speaker ID: {result.speaker_id}")
        print(f"Confidence: {result.confidence}")
        if result.error:
            print(f"Error: {result.error}")
    else:
        print(f"\nTest file not found: {test_file}")
        print("ASR Pipeline initialized successfully.")
