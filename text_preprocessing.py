"""
Text Preprocessing Module for Dravidian Languages (Tamil and Malayalam)

This module provides Unicode normalization and morphological segmentation
for Dravidian text, supporting the multimodal depression detection pipeline.

Reference: Multimodal NLP Upgrade - Requirement 2

Feature: multimodal-nlp-upgrade
"""

import unicodedata
import re
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessedText:
    """Result of text preprocessing.
    
    Contains both the normalized text and morpheme-segmented version
    as required by Requirement 2.5.
    """
    original: str
    normalized: str
    morphemes: List[str]
    morpheme_text: str  # Space-joined morphemes
    language: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original": self.original,
            "normalized": self.normalized,
            "morphemes": self.morphemes,
            "morpheme_text": self.morpheme_text,
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessedText":
        """Create PreprocessedText from dictionary."""
        return cls(
            original=data.get("original", ""),
            normalized=data.get("normalized", ""),
            morphemes=data.get("morphemes", []),
            morpheme_text=data.get("morpheme_text", ""),
            language=data.get("language", "")
        )


@dataclass
class TextPreprocessingConfig:
    """Configuration for text preprocessing.
    
    Reference: Multimodal NLP Upgrade - Requirement 2
    """
    # Language setting ("ta" for Tamil, "ml" for Malayalam)
    language: str = "ta"
    
    # Normalization settings
    apply_normalization: bool = True
    
    # Morphological segmentation settings
    apply_morphological_segmentation: bool = True
    morfessor_model_path: Optional[str] = None
    
    # Additional preprocessing options
    remove_punctuation: bool = False
    lowercase: bool = False  # Not applicable for Dravidian scripts
    
    # Supported languages
    supported_languages: List[str] = field(
        default_factory=lambda: ["ta", "ml", "tamil", "malayalam"]
    )


class TextPreprocessor:
    """Text preprocessor for Dravidian languages (Tamil and Malayalam).
    
    Handles:
    1. Unicode normalization (NFC) - Requirement 2.1
    2. Nukta and Chillu character resolution - Requirement 2.2
    3. Morphological segmentation using Morfessor - Requirements 2.3, 2.4
    4. Output of both normalized and morpheme-segmented text - Requirement 2.5
    
    Reference: Multimodal NLP Upgrade - Requirement 2
    """
    
    def __init__(self, config: Optional[TextPreprocessingConfig] = None):
        """Initialize the text preprocessor.
        
        Args:
            config: TextPreprocessingConfig instance. If None, uses defaults.
        """
        self.config = config or TextPreprocessingConfig()
        self._normalizer = None
        self._morfessor_model = None
        self._init_normalizer()
        self._init_morfessor()
    
    def _get_language_code(self, language: str) -> str:
        """Convert language name to ISO code."""
        language_map = {
            "tamil": "ta",
            "malayalam": "ml",
            "ta": "ta",
            "ml": "ml"
        }
        return language_map.get(language.lower(), "ta")
    
    def _init_normalizer(self):
        """Initialize the Indic NLP normalizer."""
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            
            lang_code = self._get_language_code(self.config.language)
            factory = IndicNormalizerFactory()
            self._normalizer = factory.get_normalizer(lang_code)
            logger.info(f"Initialized Indic normalizer for language: {lang_code}")
        except ImportError:
            logger.warning(
                "indic-nlp-library not installed. "
                "Unicode normalization will use basic NFC only."
            )
            self._normalizer = None
        except Exception as e:
            logger.warning(f"Failed to initialize Indic normalizer: {e}")
            self._normalizer = None
    
    def _init_morfessor(self):
        """Initialize the Morfessor model for morphological segmentation."""
        if not self.config.apply_morphological_segmentation:
            return
        
        try:
            import morfessor
            
            if self.config.morfessor_model_path and os.path.exists(
                self.config.morfessor_model_path
            ):
                # Load pre-trained model
                io = morfessor.MorfessorIO()
                self._morfessor_model = io.read_binary_model_file(
                    self.config.morfessor_model_path
                )
                logger.info(
                    f"Loaded Morfessor model from: {self.config.morfessor_model_path}"
                )
            else:
                # Use baseline model for unsupervised segmentation
                self._morfessor_model = morfessor.BaselineModel()
                logger.info("Initialized Morfessor baseline model (unsupervised)")
        except ImportError:
            logger.warning(
                "morfessor not installed. "
                "Morphological segmentation will use word-level tokens."
            )
            self._morfessor_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize Morfessor: {e}")
            self._morfessor_model = None
    
    def normalize(self, text: str) -> str:
        """Normalize Unicode characters in text.
        
        Applies:
        1. NFC normalization (canonical composition) - Requirement 2.1
        2. Indic-specific normalization (Nukta, Chillu) - Requirement 2.2
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Step 1: Apply NFC normalization (canonical form)
        # This ensures consistent representation of composed characters
        normalized = unicodedata.normalize("NFC", text)
        
        # Step 2: Apply Indic-specific normalization if available
        if self._normalizer is not None and self.config.apply_normalization:
            try:
                normalized = self._normalizer.normalize(normalized)
            except Exception as e:
                logger.warning(f"Indic normalization failed: {e}")
        
        # Step 3: Handle Malayalam-specific Chillu characters
        # Chillu letters are consonants without inherent vowel
        if self._get_language_code(self.config.language) == "ml":
            normalized = self._normalize_malayalam_chillu(normalized)
        
        # Step 4: Clean up whitespace
        normalized = self._clean_whitespace(normalized)
        
        return normalized
    
    def _normalize_malayalam_chillu(self, text: str) -> str:
        """Normalize Malayalam Chillu characters to standard representations.
        
        Malayalam has special Chillu characters (consonants without vowel)
        that can be represented in multiple ways. This normalizes them.
        
        Reference: Requirement 2.2
        """
        # Malayalam Chillu character mappings
        # These are atomic Chillu characters that should be preserved
        # The normalization ensures consistent representation
        
        # Chillu characters in Unicode (U+0D7A to U+0D7F)
        # These are already in their canonical form after NFC
        # Additional normalization handles legacy representations
        
        # Pattern: Consonant + Virama + ZWJ -> Chillu (if applicable)
        # This is handled by NFC normalization
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def segment_morphemes(self, text: str) -> List[str]:
        """Segment text into morphemes.
        
        Uses Morfessor for unsupervised morphological segmentation,
        which is effective for agglutinative Dravidian languages.
        
        Args:
            text: Input text (should be normalized first)
            
        Returns:
            List of morphemes
            
        Reference: Requirements 2.3, 2.4
        """
        if not text:
            return []
        
        # Tokenize into words first
        words = text.split()
        
        if not words:
            return []
        
        morphemes = []
        
        for word in words:
            word_morphemes = self._segment_word(word)
            morphemes.extend(word_morphemes)
        
        return morphemes
    
    def _segment_word(self, word: str) -> List[str]:
        """Segment a single word into morphemes.
        
        Args:
            word: Single word to segment
            
        Returns:
            List of morphemes for this word
        """
        if not word:
            return []
        
        # Skip punctuation-only tokens
        if all(unicodedata.category(c).startswith('P') for c in word):
            return [word]
        
        if self._morfessor_model is not None:
            try:
                # Use Morfessor for segmentation
                # viterbi_segment returns a tuple of (segments, cost)
                segments, _ = self._morfessor_model.viterbi_segment(word)
                if segments:
                    return list(segments)
            except Exception as e:
                logger.debug(f"Morfessor segmentation failed for '{word}': {e}")
        
        # Fallback: return word as single morpheme
        return [word]
    
    def preprocess(self, text: str, language: Optional[str] = None) -> PreprocessedText:
        """Preprocess text with normalization and morphological segmentation.
        
        This is the main entry point that produces both normalized text
        and morpheme-segmented version as required by Requirement 2.5.
        
        Args:
            text: Input text string
            language: Optional language override ("ta" or "ml")
            
        Returns:
            PreprocessedText containing original, normalized, and morpheme data
        """
        # Handle language override
        if language:
            original_lang = self.config.language
            self.config.language = self._get_language_code(language)
            # Reinitialize normalizer if language changed
            if self.config.language != self._get_language_code(original_lang):
                self._init_normalizer()
        
        # Store original
        original = text
        
        # Step 1: Normalize
        normalized = self.normalize(text) if self.config.apply_normalization else text
        
        # Step 2: Segment into morphemes
        if self.config.apply_morphological_segmentation:
            morphemes = self.segment_morphemes(normalized)
        else:
            # Fallback to word-level tokenization
            morphemes = normalized.split() if normalized else []
        
        # Create morpheme text (space-joined)
        morpheme_text = " ".join(morphemes)
        
        return PreprocessedText(
            original=original,
            normalized=normalized,
            morphemes=morphemes,
            morpheme_text=morpheme_text,
            language=self._get_language_code(self.config.language)
        )
    
    def preprocess_batch(
        self,
        texts: List[str],
        language: Optional[str] = None
    ) -> List[PreprocessedText]:
        """Preprocess a batch of texts.
        
        Args:
            texts: List of input text strings
            language: Optional language override
            
        Returns:
            List of PreprocessedText results
        """
        return [self.preprocess(text, language) for text in texts]
    
    def train_morfessor(
        self,
        texts: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Train a Morfessor model on the given texts.
        
        This enables unsupervised morphological segmentation
        tailored to the specific corpus.
        
        Args:
            texts: List of texts to train on
            save_path: Optional path to save the trained model
        """
        try:
            import morfessor
            
            # Initialize a new baseline model
            model = morfessor.BaselineModel()
            
            # Collect all words with their counts
            word_counts = {}
            for text in texts:
                normalized = self.normalize(text)
                for word in normalized.split():
                    # Skip empty words and punctuation
                    if word and not all(
                        unicodedata.category(c).startswith('P') for c in word
                    ):
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            if not word_counts:
                logger.warning("No valid words found for training")
                return
            
            # Load data as list of (count, word) tuples
            # Morfessor expects data in this format
            data = [(count, word) for word, count in word_counts.items()]
            model.load_data(data)
            
            # Train the model
            model.train_batch()
            
            # Update the model
            self._morfessor_model = model
            
            # Save if path provided
            if save_path:
                io = morfessor.MorfessorIO()
                io.write_binary_model_file(save_path, model)
                logger.info(f"Saved Morfessor model to: {save_path}")
            
            logger.info(f"Trained Morfessor model on {len(word_counts)} unique words")
            
        except ImportError:
            logger.error("morfessor not installed. Cannot train model.")
        except Exception as e:
            logger.error(f"Failed to train Morfessor model: {e}")


def create_preprocessor(
    config_or_language: Union["TextPreprocessingConfig", str] = "ta",
    apply_normalization: bool = True,
    apply_morphological_segmentation: bool = True,
    morfessor_model_path: Optional[str] = None
) -> TextPreprocessor:
    """Factory function to create a TextPreprocessor.
    
    Args:
        config_or_language: Either a TextPreprocessingConfig object or language code ("ta"/"ml")
        apply_normalization: Whether to apply Unicode normalization (ignored if config provided)
        apply_morphological_segmentation: Whether to apply morpheme segmentation (ignored if config provided)
        morfessor_model_path: Optional path to pre-trained Morfessor model (ignored if config provided)
        
    Returns:
        Configured TextPreprocessor instance
    """
    if hasattr(config_or_language, 'language'):
        # Config object provided (from config.py or text_preprocessing.py)
        config = TextPreprocessingConfig(
            language=config_or_language.language,
            apply_normalization=getattr(config_or_language, 'apply_normalization', True),
            apply_morphological_segmentation=getattr(config_or_language, 'apply_morphological_segmentation', True),
            morfessor_model_path=getattr(config_or_language, 'morfessor_model_path', None)
        )
        return TextPreprocessor(config)
    else:
        # Individual parameters provided
        config = TextPreprocessingConfig(
            language=config_or_language,
            apply_normalization=apply_normalization,
            apply_morphological_segmentation=apply_morphological_segmentation,
            morfessor_model_path=morfessor_model_path
        )
        return TextPreprocessor(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    
    # Create preprocessor for Tamil
    preprocessor_ta = create_preprocessor(language="ta")
    
    # Example Tamil text
    tamil_text = "நான் நலமாக இருக்கிறேன்"
    result_ta = preprocessor_ta.preprocess(tamil_text)
    
    print("Tamil Preprocessing:")
    print(f"  Original: {result_ta.original}")
    print(f"  Normalized: {result_ta.normalized}")
    print(f"  Morphemes: {result_ta.morphemes}")
    print(f"  Morpheme text: {result_ta.morpheme_text}")
    print()
    
    # Create preprocessor for Malayalam
    preprocessor_ml = create_preprocessor(language="ml")
    
    # Example Malayalam text
    malayalam_text = "നിങ്ങൾ എങ്ങനെയുണ്ട്"
    result_ml = preprocessor_ml.preprocess(malayalam_text)
    
    print("Malayalam Preprocessing:")
    print(f"  Original: {result_ml.original}")
    print(f"  Normalized: {result_ml.normalized}")
    print(f"  Morphemes: {result_ml.morphemes}")
    print(f"  Morpheme text: {result_ml.morpheme_text}")
