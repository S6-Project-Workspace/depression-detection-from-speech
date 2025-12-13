"""
Audio Preprocessing Pipeline for Depression Detection

This module implements the signal preprocessing pipeline as specified in
Section 2 of the architectural blueprint:
- Audio Normalization and Format Conversion (2.1)
- Voice Activity Detection and Segmentation (2.2)
- Chunking and Variable-Length Padding (2.3)
"""

import os
import re
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging

# Use soundfile backend for compatibility
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
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    logging.warning("webrtcvad not installed. Using energy-based VAD.")

from config import PreprocessingConfig, PathConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a processed audio chunk."""
    waveform: torch.Tensor  # Shape: (1, num_samples)
    attention_mask: torch.Tensor  # Shape: (num_samples,)
    speaker_id: str
    file_path: str
    chunk_idx: int
    start_time: float
    end_time: float
    label: int  # 0: Non-depressed, 1: Depressed


class AudioNormalizer:
    """Handles audio format conversion and normalization.
    
    Implements Section 2.1: Audio Normalization and Format Conversion
    - Sinc Interpolation resampling with Hann window
    - Stereo to mono conversion
    - DC offset removal
    - Peak normalization
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._resamplers: Dict[int, Resample] = {}
    
    def _get_resampler(self, orig_sr: int) -> Resample:
        """Get or create a resampler for the given original sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = Resample(
                orig_freq=orig_sr,
                new_freq=self.config.sampling_rate,
                lowpass_filter_width=self.config.lowpass_filter_width,
                rolloff=self.config.rolloff,
                resampling_method="sinc_interp_hann"  # Sinc interpolation with Hann window
            )
        return self._resamplers[orig_sr]
    
    def normalize(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Apply full normalization pipeline to audio.
        
        Args:
            waveform: Input waveform tensor of shape (channels, samples)
            orig_sr: Original sampling rate
            
        Returns:
            Normalized mono waveform at target sampling rate
        """
        # Step 1: Resample to 16kHz if needed
        if orig_sr != self.config.sampling_rate:
            resampler = self._get_resampler(orig_sr)
            waveform = resampler(waveform)
        
        # Step 2: Convert stereo to mono by averaging channels
        # x_mono[t] = 0.5 * (x_left[t] + x_right[t])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Step 3: DC offset removal - subtract mean amplitude
        # x'[t] = x[t] - μ_x
        waveform = waveform - torch.mean(waveform)
        
        # Step 4: Peak normalization to 0.95
        # Prevents clipping during augmentation
        max_amp = torch.max(torch.abs(waveform))
        if max_amp > 0:
            waveform = waveform * (self.config.peak_normalize_target / max_amp)
        
        return waveform


class VoiceActivityDetector:
    """Voice Activity Detection for isolating target speaker utterances.
    
    Implements Section 2.2: Voice Activity Detection and Segmentation
    - Energy-based VAD or WebRTC VAD with high sensitivity
    - 30ms frame analysis
    - 200ms padding for prosodic preservation
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.frame_samples = int(config.sampling_rate * config.vad_frame_size_ms / 1000)
        self.padding_samples = int(config.sampling_rate * config.silence_padding_ms / 1000)
        
        if HAS_WEBRTCVAD:
            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        else:
            self.vad = None
    
    def _energy_vad(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """Energy-based VAD as fallback.
        
        Frames with energy below -40dB relative to signal peak are classified as silence.
        """
        waveform_np = waveform.squeeze().numpy()
        
        # Calculate signal peak in dB
        peak_amplitude = np.max(np.abs(waveform_np))
        if peak_amplitude == 0:
            return []
        
        # Process in frames
        num_frames = len(waveform_np) // self.frame_samples
        speech_frames = []
        
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = waveform_np[start:end]
            
            # Calculate frame energy in dB relative to peak
            frame_energy = np.sqrt(np.mean(frame ** 2))
            if frame_energy > 0:
                frame_db = 20 * np.log10(frame_energy / peak_amplitude)
            else:
                frame_db = -100
            
            if frame_db >= self.config.silence_threshold_db:
                speech_frames.append(i)
        
        if not speech_frames:
            return []
        
        # Convert frame indices to sample ranges and merge consecutive regions
        regions = []
        start_frame = speech_frames[0]
        prev_frame = speech_frames[0]
        
        for frame_idx in speech_frames[1:]:
            if frame_idx > prev_frame + 1:
                # Gap detected - save previous region
                regions.append((
                    max(0, start_frame * self.frame_samples - self.padding_samples),
                    min(len(waveform_np), (prev_frame + 1) * self.frame_samples + self.padding_samples)
                ))
                start_frame = frame_idx
            prev_frame = frame_idx
        
        # Add final region
        regions.append((
            max(0, start_frame * self.frame_samples - self.padding_samples),
            min(len(waveform_np), (prev_frame + 1) * self.frame_samples + self.padding_samples)
        ))
        
        return regions
    
    def _webrtc_vad(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """WebRTC VAD for neural-based voice activity detection."""
        # WebRTC VAD requires 16-bit PCM audio
        waveform_np = waveform.squeeze().numpy()
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        
        num_frames = len(waveform_int16) // self.frame_samples
        speech_frames = []
        
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame_bytes = waveform_int16[start:end].tobytes()
            
            try:
                is_speech = self.vad.is_speech(
                    frame_bytes,
                    sample_rate=self.config.sampling_rate
                )
                if is_speech:
                    speech_frames.append(i)
            except Exception:
                # Fallback if WebRTC fails on a frame
                speech_frames.append(i)
        
        if not speech_frames:
            return []
        
        # Merge consecutive speech frames with padding
        regions = []
        start_frame = speech_frames[0]
        prev_frame = speech_frames[0]
        
        for frame_idx in speech_frames[1:]:
            if frame_idx > prev_frame + 1:
                regions.append((
                    max(0, start_frame * self.frame_samples - self.padding_samples),
                    min(len(waveform_int16), (prev_frame + 1) * self.frame_samples + self.padding_samples)
                ))
                start_frame = frame_idx
            prev_frame = frame_idx
        
        regions.append((
            max(0, start_frame * self.frame_samples - self.padding_samples),
            min(len(waveform_int16), (prev_frame + 1) * self.frame_samples + self.padding_samples)
        ))
        
        return regions
    
    def detect(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """Detect voice activity regions in the waveform.
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            
        Returns:
            List of (start_sample, end_sample) tuples for speech regions
        """
        if self.vad is not None:
            return self._webrtc_vad(waveform)
        return self._energy_vad(waveform)
    
    def extract_speech(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate all speech regions.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            Concatenated speech regions
        """
        regions = self.detect(waveform)
        
        if not regions:
            return waveform  # Return original if no speech detected
        
        speech_segments = []
        for start, end in regions:
            speech_segments.append(waveform[:, start:end])
        
        return torch.cat(speech_segments, dim=1)


class AudioChunker:
    """Sliding window chunking for DNN input preparation.
    
    Implements Section 2.3: Chunking and Variable-Length Padding
    - 5-second chunks with configurable overlap
    - Zero-padding for short segments
    - Attention mask generation
    """
    
    def __init__(self, config: PreprocessingConfig, is_training: bool = True):
        self.config = config
        self.is_training = is_training
        
        self.chunk_samples = int(config.chunk_duration_sec * config.sampling_rate)
        
        if is_training:
            self.overlap_samples = int(config.chunk_overlap_train_sec * config.sampling_rate)
        else:
            self.overlap_samples = int(config.chunk_overlap_inference_sec * config.sampling_rate)
        
        self.hop_samples = self.chunk_samples - self.overlap_samples
    
    def chunk(
        self,
        waveform: torch.Tensor,
        speaker_id: str,
        file_path: str,
        label: int
    ) -> List[AudioChunk]:
        """Segment waveform into fixed-length chunks.
        
        Args:
            waveform: Input waveform of shape (1, samples)
            speaker_id: Unique identifier for the speaker
            file_path: Path to the original audio file
            label: Class label (0 or 1)
            
        Returns:
            List of AudioChunk objects
        """
        total_samples = waveform.shape[1]
        chunks = []
        
        # Handle short audio (less than one chunk)
        if total_samples < self.chunk_samples:
            # Zero-pad to chunk length
            padded = torch.zeros(1, self.chunk_samples)
            padded[:, :total_samples] = waveform
            
            # Create attention mask (1 for valid, 0 for padding)
            attention_mask = torch.zeros(self.chunk_samples)
            attention_mask[:total_samples] = 1.0
            
            chunks.append(AudioChunk(
                waveform=padded,
                attention_mask=attention_mask,
                speaker_id=speaker_id,
                file_path=file_path,
                chunk_idx=0,
                start_time=0.0,
                end_time=total_samples / self.config.sampling_rate,
                label=label
            ))
            return chunks
        
        # Sliding window chunking
        chunk_idx = 0
        start = 0
        
        while start < total_samples:
            end = start + self.chunk_samples
            
            if end <= total_samples:
                # Full chunk
                chunk_waveform = waveform[:, start:end]
                attention_mask = torch.ones(self.chunk_samples)
            else:
                # Partial chunk - pad to full length
                remaining = total_samples - start
                chunk_waveform = torch.zeros(1, self.chunk_samples)
                chunk_waveform[:, :remaining] = waveform[:, start:]
                
                attention_mask = torch.zeros(self.chunk_samples)
                attention_mask[:remaining] = 1.0
            
            chunks.append(AudioChunk(
                waveform=chunk_waveform,
                attention_mask=attention_mask,
                speaker_id=speaker_id,
                file_path=file_path,
                chunk_idx=chunk_idx,
                start_time=start / self.config.sampling_rate,
                end_time=min(end, total_samples) / self.config.sampling_rate,
                label=label
            ))
            
            chunk_idx += 1
            start += self.hop_samples
            
            # Stop if next chunk would start beyond audio length
            if start >= total_samples:
                break
        
        return chunks


class SpeakerIDExtractor:
    """Extract speaker IDs from filenames for speaker-independent CV.
    
    Based on the dataset structure:
    - Tamil Depressed: D_A00_X-Y.wav, D_FX00X-Y.wav, D_SXX_XX-Y.wav, D_F2001.wav
    - Tamil Non-depressed: ND1_XXXX.wav, ND2_XXXX.wav, etc.
    - Malayalam Depressed: D_AXXX_Y.wav, D_F001_XX_Y.wav, D_S001.wav
    - Malayalam Non-depressed: ND1_XXXX.wav, etc.
    """
    
    # Regex patterns for different filename formats
    PATTERNS = [
        # D_A00_1-1.wav -> speaker A00_1
        r'^D_([A-Z]\d+_\d+)-\d+\.wav$',
        # D_F1001-1.wav -> speaker F1001
        r'^D_([A-Z]\d+)-\d+\.wav$',
        # D_F10010-1.wav -> speaker F10010
        r'^D_([A-Z]\d+)-\d+\.wav$',
        # D_S00_06-1.wav -> speaker S00_06
        r'^D_([A-Z]\d+_\d+)-\d+\.wav$',
        # D_F2001.wav -> speaker F2001
        r'^D_([A-Z]\d+)\.wav$',
        # D_F2001_a.wav -> speaker F2001
        r'^D_([A-Z]\d+)_[a-z]\.wav$',
        # D_A001_1.wav (Malayalam) -> speaker A001
        r'^D_([A-Z]\d+)_\d+\.wav$',
        # D_F001_10_1.wav (Malayalam) -> speaker F001_10
        r'^D_([A-Z]\d+_\d+)_\d+\.wav$',
        # D_S001.wav -> speaker S001
        r'^D_([A-Z]\d+)\.wav$',
        # D_S0010_a.wav -> speaker S0010
        r'^D_([A-Z]\d+)_[a-z]\.wav$',
        # ND1_0001.wav -> speaker ND1
        r'^(ND\d+)_\d+\.wav$',
    ]
    
    @classmethod
    def extract(cls, filename: str) -> str:
        """Extract speaker ID from filename.
        
        Args:
            filename: Name of the audio file (not full path)
            
        Returns:
            Speaker ID string
        """
        for pattern in cls.PATTERNS:
            match = re.match(pattern, filename)
            if match:
                return match.group(1)
        
        # Fallback: use everything before the last underscore/dash
        base = os.path.splitext(filename)[0]
        
        # Try to find a pattern
        if '_' in base:
            parts = base.rsplit('_', 1)
            if parts[-1].isdigit() or parts[-1] in 'abcd':
                return parts[0]
        
        if '-' in base:
            parts = base.rsplit('-', 1)
            if parts[-1].isdigit():
                return parts[0]
        
        # Last resort: use the full basename
        return base


class AudioPreprocessor:
    """Main preprocessing pipeline combining all components.
    
    Implements the complete preprocessing flow:
    1. Load audio file
    2. Normalize (resample, mono, DC removal, peak normalization)
    3. Apply VAD to extract speech regions
    4. Chunk into fixed-length segments
    5. Generate attention masks
    """
    
    def __init__(
        self,
        config: PreprocessingConfig,
        is_training: bool = True,
        apply_vad: bool = True
    ):
        self.config = config
        self.is_training = is_training
        self.apply_vad = apply_vad
        
        self.normalizer = AudioNormalizer(config)
        self.vad = VoiceActivityDetector(config) if apply_vad else None
        self.chunker = AudioChunker(config, is_training)
    
    def process_file(
        self,
        file_path: str,
        label: int,
        speaker_id: Optional[str] = None
    ) -> List[AudioChunk]:
        """Process a single audio file.
        
        Args:
            file_path: Path to the audio file
            label: Class label (0: non-depressed, 1: depressed)
            speaker_id: Optional speaker ID (extracted from filename if not provided)
            
        Returns:
            List of processed AudioChunk objects
        """
        # Load audio using soundfile or librosa for better compatibility
        try:
            if HAS_SOUNDFILE:
                # Use soundfile directly
                audio_data, sample_rate = sf.read(file_path, dtype='float32')
                # Convert to torch tensor with shape (channels, samples)
                if len(audio_data.shape) == 1:
                    waveform = torch.from_numpy(audio_data).unsqueeze(0)
                else:
                    # Multi-channel: transpose from (samples, channels) to (channels, samples)
                    waveform = torch.from_numpy(audio_data.T)
            elif HAS_LIBROSA:
                # Use librosa as fallback
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
                waveform = torch.from_numpy(audio_data)
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                # Fallback to torchaudio
                waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return []
        
        # Extract speaker ID if not provided
        if speaker_id is None:
            filename = os.path.basename(file_path)
            speaker_id = SpeakerIDExtractor.extract(filename)
        
        # Normalize
        waveform = self.normalizer.normalize(waveform, sample_rate)
        
        # Check minimum duration
        duration = waveform.shape[1] / self.config.sampling_rate
        if duration < self.config.min_audio_duration:
            logger.warning(f"Audio too short ({duration:.2f}s): {file_path}")
            return []
        
        # Apply VAD
        if self.vad is not None:
            waveform = self.vad.extract_speech(waveform)
        
        # Chunk
        chunks = self.chunker.chunk(waveform, speaker_id, file_path, label)
        
        return chunks
    
    def process_directory(
        self,
        directory: str,
        label: int
    ) -> List[AudioChunk]:
        """Process all audio files in a directory.
        
        Args:
            directory: Path to directory containing audio files
            label: Class label for all files in this directory
            
        Returns:
            List of all processed AudioChunk objects
        """
        all_chunks = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return all_chunks
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in audio_extensions:
                continue
            
            file_path = os.path.join(directory, filename)
            chunks = self.process_file(file_path, label)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(all_chunks)} chunks from {directory}")
        return all_chunks


def build_dataset_manifest(
    paths: PathConfig,
    preprocessing_config: PreprocessingConfig,
    language: str = "combined",
    is_training: bool = True
) -> Tuple[List[AudioChunk], Dict[str, int]]:
    """Build the complete dataset manifest with speaker mapping.
    
    Args:
        paths: PathConfig with data directories
        preprocessing_config: PreprocessingConfig for processing
        language: "tamil", "malayalam", or "combined"
        is_training: Whether this is for training (affects chunking overlap)
        
    Returns:
        Tuple of (list of AudioChunks, speaker_id to speaker_idx mapping)
    """
    preprocessor = AudioPreprocessor(preprocessing_config, is_training=is_training)
    all_chunks = []
    
    if language.lower() in ["tamil", "combined"]:
        # Tamil Depressed
        chunks = preprocessor.process_directory(paths.tamil_depressed, label=1)
        all_chunks.extend(chunks)
        
        # Tamil Non-depressed
        chunks = preprocessor.process_directory(paths.tamil_non_depressed, label=0)
        all_chunks.extend(chunks)
    
    if language.lower() in ["malayalam", "combined"]:
        # Malayalam Depressed
        chunks = preprocessor.process_directory(paths.malayalam_depressed, label=1)
        all_chunks.extend(chunks)
        
        # Malayalam Non-depressed
        chunks = preprocessor.process_directory(paths.malayalam_non_depressed, label=0)
        all_chunks.extend(chunks)
    
    # Build speaker mapping
    unique_speakers = sorted(set(chunk.speaker_id for chunk in all_chunks))
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Unique speakers: {len(unique_speakers)}")
    logger.info(f"Depressed chunks: {sum(1 for c in all_chunks if c.label == 1)}")
    logger.info(f"Non-depressed chunks: {sum(1 for c in all_chunks if c.label == 0)}")
    
    return all_chunks, speaker_to_idx


if __name__ == "__main__":
    # Test preprocessing
    from config import get_config
    
    config = get_config("combined")
    
    # Test speaker ID extraction
    test_files = [
        "D_A00_1-1.wav",
        "D_F1001-1.wav",
        "D_S00_06-1.wav",
        "D_F2001.wav",
        "D_F2001_a.wav",
        "ND1_0001.wav",
        "D_A001_1.wav",
        "D_F001_10_1.wav",
    ]
    
    print("Speaker ID Extraction Test:")
    for filename in test_files:
        speaker_id = SpeakerIDExtractor.extract(filename)
        print(f"  {filename} -> {speaker_id}")
    
    # Build dataset manifest
    print("\nBuilding dataset manifest...")
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language="combined",
        is_training=True
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Unique speakers: {len(speaker_map)}")
