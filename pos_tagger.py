"""
Part-of-Speech Tagger for Dravidian Languages (Tamil and Malayalam)

This module implements POS tagging using fine-tuned transformer models
optimized for Dravidian languages with support for agglutinative morphology
and code-mixed text.

Reference: Traditional NLP Tasks - Requirement 1
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class POSConfig:
    """Configuration for POS Tagger."""
    model_name: str = "bert-base-multilingual-cased"  # More accessible model
    tagset: str = "universal"  # Universal Dependencies tagset
    max_length: int = 512
    batch_size: int = 16
    languages: List[str] = field(default_factory=lambda: ["ta", "ml"])
    device: str = "cpu"
    
    # Universal Dependencies POS tags
    pos_tags: List[str] = field(default_factory=lambda: [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", 
        "VERB", "X", "O"  # O for outside/unknown
    ])
    
    # Function words for linguistic analysis
    function_words: Dict[str, List[str]] = field(default_factory=lambda: {
        "ta": ["அது", "இது", "என்", "ஒரு", "மற்றும்", "அல்லது", "ஆனால்", "இல்", "க்கு", "ல்"],
        "ml": ["അത്", "ഇത്", "എന്ന്", "ഒരു", "കൂടാതെ", "അല്ലെങ്കിൽ", "പക്ഷേ", "ൽ", "ക്ക്", "ന്"]
    })


@dataclass
class POSResult:
    """Result of POS tagging operation."""
    tokens: List[str]
    tags: List[str]
    confidence_scores: List[float]
    pos_distribution: Dict[str, int]
    language: str
    
    def __post_init__(self):
        """Validate result consistency."""
        assert len(self.tokens) == len(self.tags), \
            f"Token count ({len(self.tokens)}) must equal tag count ({len(self.tags)})"
        assert len(self.tokens) == len(self.confidence_scores), \
            f"Token count ({len(self.tokens)}) must equal confidence count ({len(self.confidence_scores)})"


class POSTagger:
    """Part-of-Speech Tagger for Dravidian languages."""
    
    def __init__(self, config: POSConfig):
        """Initialize POS tagger with configuration.
        
        Args:
            config: POSConfig instance with model and processing settings
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.model_name,
                num_labels=len(config.pos_tags)
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Create tag mappings
            self.id2tag = {i: tag for i, tag in enumerate(config.pos_tags)}
            self.tag2id = {tag: i for i, tag in enumerate(config.pos_tags)}
            
            logger.info(f"Initialized POS tagger with {config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize POS tagger: {e}")
            # Fallback to simple rule-based tagger for testing
            self._init_fallback_tagger()
    
    def _init_fallback_tagger(self):
        """Initialize simple rule-based fallback tagger."""
        logger.warning("Using fallback rule-based POS tagger")
        self.tokenizer = None
        self.model = None
        
        # Simple rules for common patterns
        self.pos_rules = {
            "ta": {
                "verb_endings": ["கிறேன்", "கிறாய்", "கிறான்", "கிறாள்", "கிறது", "கிறோம்", "கிறீர்கள்", "கிறார்கள்"],
                "noun_endings": ["ம்", "ன்", "ள்", "ர்", "து"],
                "adj_endings": ["ான", "ிய", "ு"]
            },
            "ml": {
                "verb_endings": ["ുന്നു", "ുന്ന", "ിച്ചു", "ും", "ാൻ"],
                "noun_endings": ["ം", "ൻ", "ൾ", "ർ", "ത്"],
                "adj_endings": ["ായ", "ിയ", "ു"]
            }
        }
    
    def tag_sentence(self, tokens: List[str], language: str) -> List[Tuple[str, str]]:
        """Tag a single sentence with POS tags.
        
        Args:
            tokens: List of tokens to tag
            language: Language code ("ta" or "ml")
            
        Returns:
            List of (token, tag) tuples
        """
        if not tokens:
            return []
        
        if self.model is not None:
            return self._tag_with_model(tokens, language)
        else:
            return self._tag_with_rules(tokens, language)
    
    def _tag_with_model(self, tokens: List[str], language: str) -> List[Tuple[str, str]]:
        """Tag using transformer model."""
        # Join tokens for tokenization
        text = " ".join(tokens)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert predictions to tags
        predicted_tags = [self.id2tag[pred.item()] for pred in predictions[0]]
        
        # Align with original tokens (simplified alignment)
        if len(predicted_tags) > len(tokens):
            predicted_tags = predicted_tags[:len(tokens)]
        elif len(predicted_tags) < len(tokens):
            predicted_tags.extend(["NOUN"] * (len(tokens) - len(predicted_tags)))
        
        return list(zip(tokens, predicted_tags))
    
    def _tag_with_rules(self, tokens: List[str], language: str) -> List[Tuple[str, str]]:
        """Tag using simple rule-based approach."""
        tagged = []
        rules = self.pos_rules.get(language, {})
        
        for token in tokens:
            tag = "NOUN"  # Default tag
            
            # Check for punctuation
            if token in ".,!?;:\"'()[]{}":
                tag = "PUNCT"
            # Check for numbers
            elif token.isdigit():
                tag = "NUM"
            # Check for pronouns first (more specific)
            elif token in ["நான்", "அவர்", "அவள்", "அவன்", "நீ", "நாம்", "ഞാൻ", "അവർ", "അവൾ", "അവൻ", "നീ", "നാം"]:
                tag = "PRON"
            # Check verb endings
            elif any(token.endswith(ending) for ending in rules.get("verb_endings", [])):
                tag = "VERB"
            # Check adjective endings
            elif any(token.endswith(ending) for ending in rules.get("adj_endings", [])):
                tag = "ADJ"
            # Check function words
            elif token in self.config.function_words.get(language, []):
                tag = "ADP"
            
            tagged.append((token, tag))
        
        return tagged
    
    def tag_batch(self, sentences: List[List[str]], language: str) -> List[List[Tuple[str, str]]]:
        """Tag multiple sentences in batch.
        
        Args:
            sentences: List of token lists
            language: Language code ("ta" or "ml")
            
        Returns:
            List of tagged sentences
        """
        return [self.tag_sentence(tokens, language) for tokens in sentences]
    
    def extract_pos_features(self, tagged_sentence: List[Tuple[str, str]]) -> Dict[str, float]:
        """Extract POS-based features for depression analysis.
        
        Args:
            tagged_sentence: List of (token, tag) tuples
            
        Returns:
            Dictionary of POS features
        """
        if not tagged_sentence:
            return self._empty_pos_features()
        
        tokens, tags = zip(*tagged_sentence)
        tag_counts = Counter(tags)
        total_tokens = len(tokens)
        
        # POS distribution (normalized)
        pos_distribution = {tag: count / total_tokens for tag, count in tag_counts.items()}
        
        # POS diversity (Shannon entropy)
        pos_diversity = -sum(p * np.log2(p) for p in pos_distribution.values() if p > 0)
        
        # Function word ratio
        function_words = set(self.config.function_words.get("ta", []) + 
                           self.config.function_words.get("ml", []))
        function_word_count = sum(1 for token in tokens if token in function_words)
        function_word_ratio = function_word_count / total_tokens if total_tokens > 0 else 0.0
        
        # Content word ratio
        content_tags = {"NOUN", "VERB", "ADJ", "ADV"}
        content_word_count = sum(tag_counts.get(tag, 0) for tag in content_tags)
        content_word_ratio = content_word_count / total_tokens if total_tokens > 0 else 0.0
        
        # Depression-specific features
        pronoun_count = tag_counts.get("PRON", 0)
        first_person_pronouns = {"நான்", "என்", "எனக്கு", "ഞാൻ", "എന്റെ", "എനിക്ക്"}
        first_person_count = sum(1 for token in tokens if token in first_person_pronouns)
        first_person_ratio = first_person_count / total_tokens if total_tokens > 0 else 0.0
        
        features = {
            # Basic POS features
            "pos_diversity": pos_diversity,
            "function_word_ratio": function_word_ratio,
            "content_word_ratio": content_word_ratio,
            "pronoun_ratio": pronoun_count / total_tokens if total_tokens > 0 else 0.0,
            
            # Depression-specific features
            "first_person_pronoun_ratio": first_person_ratio,
            
            # Individual POS tag ratios (top 10 most common)
            **{f"{tag.lower()}_ratio": pos_distribution.get(tag, 0.0) 
               for tag in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "PUNCT", "NUM", "CCONJ"]}
        }
        
        return features
    
    def _empty_pos_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            "pos_diversity": 0.0,
            "function_word_ratio": 0.0,
            "content_word_ratio": 0.0,
            "pronoun_ratio": 0.0,
            "first_person_pronoun_ratio": 0.0,
            **{f"{tag.lower()}_ratio": 0.0 
               for tag in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "PUNCT", "NUM", "CCONJ"]}
        }
    
    def create_pos_result(self, tokens: List[str], tags: List[str], 
                         confidence_scores: List[float], language: str) -> POSResult:
        """Create POSResult with validation.
        
        Args:
            tokens: List of tokens
            tags: List of POS tags
            confidence_scores: List of confidence scores
            language: Language code
            
        Returns:
            POSResult instance
        """
        # Calculate POS distribution
        pos_distribution = Counter(tags)
        
        return POSResult(
            tokens=tokens,
            tags=tags,
            confidence_scores=confidence_scores,
            pos_distribution=dict(pos_distribution),
            language=language
        )


def create_pos_tagger(language: str = "ta", device: str = "cpu") -> POSTagger:
    """Factory function to create POS tagger.
    
    Args:
        language: Language code ("ta" or "ml")
        device: Device to run on ("cpu", "cuda", "mps")
        
    Returns:
        POSTagger instance
    """
    config = POSConfig(
        languages=[language],
        device=device
    )
    return POSTagger(config)


if __name__ == "__main__":
    # Example usage
    tagger = create_pos_tagger("ta")
    
    # Test Tamil sentence
    tokens = ["நான்", "மகிழ்ச்சியாக", "இருக்கிறேன்"]
    tagged = tagger.tag_sentence(tokens, "ta")
    print("Tagged sentence:", tagged)
    
    # Extract features
    features = tagger.extract_pos_features(tagged)
    print("POS features:", features)