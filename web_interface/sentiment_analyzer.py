"""
Sentiment Analysis Module for Web Interface

This module provides sentiment analysis capabilities with confidence measures
and linguistic feature extraction for the NLP web interface.

Reference: NLP Web Interface - Requirement 2.4
"""

import logging
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result with confidence measures."""
    overall_sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # Overall confidence score
    scores: Dict[str, float]  # Individual sentiment scores
    features: Dict[str, float]  # Sentiment-related features
    
    def __post_init__(self):
        """Validate sentiment result."""
        assert self.overall_sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= self.confidence <= 1.0
        assert all(0.0 <= score <= 1.0 for score in self.scores.values())


class SentimentAnalyzer:
    """Sentiment analyzer with support for Dravidian languages."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize sentiment analyzer.
        
        Args:
            device: Device to run on ("cpu", "cuda", "mps")
        """
        self.device = device
        self._initialize_lexicons()
    
    def _initialize_lexicons(self):
        """Initialize sentiment lexicons for different languages."""
        # Tamil sentiment words
        self.tamil_positive = {
            "மகிழ்ச்சி", "நல்ல", "அருமை", "சந்தோஷம்", "பிரமாதம்", "அழகு",
            "வெற்றி", "நன்மை", "பெருமை", "மகிழ்வு", "ஆனந்தம்", "நம்பிக்கை",
            "அன்பு", "கனவு", "வாழ்க்கை", "நட்பு", "உற்சாகம்", "புன்னகை"
        }
        
        self.tamil_negative = {
            "துக்கம்", "கவலை", "வருத்தம்", "கோபம்", "பயம்", "வேதனை",
            "தோல்வி", "நோய்", "வலி", "அழுகை", "சோகம்", "மன அழுத்தம்",
            "பிரச்சனை", "கஷ்டம்", "இழப்பு", "தனிமை", "நிராசை", "கடினம்"
        }
        
        # Malayalam sentiment words
        self.malayalam_positive = {
            "സന്തോഷം", "നല്ല", "സുന്ദരം", "വിജയം", "സ്നേഹം", "സന്തുഷ്ടി",
            "ആനന്ദം", "പ്രതീക്ഷ", "സൗന്ദര്യം", "സുഖം", "മനോഹരം", "അഭിമാനം",
            "സൗഹൃദം", "ഉത്സാഹം", "പുഞ്ചിരി", "സ്വപ്നം", "ജീവിതം", "സമാധാനം"
        }
        
        self.malayalam_negative = {
            "ദുഃഖം", "വിഷമം", "കോപം", "ഭയം", "വേദന", "പരാജയം",
            "രോഗം", "വേദന", "കരച്ചിൽ", "ദുഃഖം", "സമ്മർദ്ദം", "പ്രശ്നം",
            "കഷ്ടം", "നഷ്ടം", "ഏകാന്തത", "നിരാശ", "ബുദ്ധിമുട്ട്", "വിഷാദം"
        }
        
        # English sentiment words (basic set)
        self.english_positive = {
            "happy", "good", "great", "excellent", "wonderful", "amazing",
            "love", "joy", "success", "beautiful", "perfect", "fantastic",
            "awesome", "brilliant", "outstanding", "marvelous", "superb", "delighted"
        }
        
        self.english_negative = {
            "sad", "bad", "terrible", "awful", "horrible", "hate",
            "angry", "fear", "pain", "failure", "ugly", "disgusting",
            "disappointed", "frustrated", "worried", "anxious", "depressed", "miserable"
        }
        
        # Combine all lexicons
        self.positive_words = {
            "ta": self.tamil_positive,
            "ml": self.malayalam_positive,
            "en": self.english_positive
        }
        
        self.negative_words = {
            "ta": self.tamil_negative,
            "ml": self.malayalam_negative,
            "en": self.english_negative
        }
        
        # Intensity modifiers
        self.intensifiers = {
            "ta": {"மிகவும்", "அதிகம்", "நிறைய", "மிக", "பெரிதும்"},
            "ml": {"വളരെ", "അധികം", "ഏറെ", "മിക", "വളരെയധികം"},
            "en": {"very", "extremely", "highly", "really", "quite", "so", "too"}
        }
        
        self.diminishers = {
            "ta": {"கொஞ்சம்", "சிறிது", "அல்ப", "குறைவு"},
            "ml": {"കുറച്ച്", "അല്പം", "ചെറുത്", "കുറവ്"},
            "en": {"little", "slightly", "somewhat", "barely", "hardly"}
        }
    
    def analyze_sentiment(self, text: str, tokens: List[str], 
                         language: str = "ta") -> SentimentResult:
        """Analyze sentiment of text with confidence measures.
        
        Args:
            text: Original text
            tokens: Tokenized text
            language: Language code ("ta", "ml", "en")
            
        Returns:
            SentimentResult with sentiment scores and features
        """
        if not tokens:
            return self._create_neutral_result()
        
        # Get sentiment lexicons for language
        positive_words = self.positive_words.get(language, self.positive_words["en"])
        negative_words = self.negative_words.get(language, self.negative_words["en"])
        intensifiers = self.intensifiers.get(language, self.intensifiers["en"])
        diminishers = self.diminishers.get(language, self.diminishers["en"])
        
        # Count sentiment words
        positive_count = 0
        negative_count = 0
        intensity_modifier = 1.0
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Check for intensifiers/diminishers
            if token_lower in intensifiers:
                intensity_modifier = 1.5
                continue
            elif token_lower in diminishers:
                intensity_modifier = 0.5
                continue
            
            # Check sentiment words
            if token_lower in positive_words:
                positive_count += intensity_modifier
            elif token_lower in negative_words:
                negative_count += intensity_modifier
            
            # Reset intensity modifier
            intensity_modifier = 1.0
        
        # Calculate sentiment scores
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            # No sentiment words found - analyze patterns
            return self._analyze_patterns(text, tokens, language)
        
        # Calculate normalized scores
        positive_score = positive_count / total_sentiment_words
        negative_score = negative_count / total_sentiment_words
        neutral_score = max(0.0, 1.0 - positive_score - negative_score)
        
        # Determine overall sentiment
        if positive_score > negative_score and positive_score > 0.4:
            overall_sentiment = "positive"
            confidence = positive_score
        elif negative_score > positive_score and negative_score > 0.4:
            overall_sentiment = "negative"
            confidence = negative_score
        else:
            overall_sentiment = "neutral"
            confidence = max(neutral_score, 0.5)
        
        # Extract sentiment features
        features = self._extract_sentiment_features(
            text, tokens, positive_count, negative_count, language
        )
        
        return SentimentResult(
            overall_sentiment=overall_sentiment,
            confidence=min(confidence, 1.0),
            scores={
                "positive": min(positive_score, 1.0),
                "negative": min(negative_score, 1.0),
                "neutral": min(neutral_score, 1.0)
            },
            features=features
        )
    
    def _analyze_patterns(self, text: str, tokens: List[str], 
                         language: str) -> SentimentResult:
        """Analyze sentiment using linguistic patterns when no sentiment words found."""
        # Check for question marks (often neutral/negative)
        question_count = text.count("?")
        exclamation_count = text.count("!")
        
        # Check for negation patterns
        negation_patterns = {
            "ta": ["இல்லை", "அல்ல", "இல்லாத", "அல்லாத"],
            "ml": ["ഇല്ല", "അല്ല", "ഇല്ലാത്ത", "അല്ലാത്ത"],
            "en": ["not", "no", "never", "nothing", "none"]
        }
        
        negations = negation_patterns.get(language, negation_patterns["en"])
        negation_count = sum(1 for token in tokens if token.lower() in negations)
        
        # Simple pattern-based scoring
        if exclamation_count > 0 and negation_count == 0:
            # Exclamations without negation tend to be positive
            return SentimentResult(
                overall_sentiment="positive",
                confidence=0.6,
                scores={"positive": 0.6, "negative": 0.2, "neutral": 0.2},
                features={"exclamation_count": exclamation_count, "pattern_based": 1.0}
            )
        elif negation_count > 0:
            # Negations tend to be negative
            return SentimentResult(
                overall_sentiment="negative",
                confidence=0.6,
                scores={"positive": 0.2, "negative": 0.6, "neutral": 0.2},
                features={"negation_count": negation_count, "pattern_based": 1.0}
            )
        else:
            # Default to neutral
            return self._create_neutral_result()
    
    def _extract_sentiment_features(self, text: str, tokens: List[str],
                                  positive_count: float, negative_count: float,
                                  language: str) -> Dict[str, float]:
        """Extract sentiment-related features."""
        features = {}
        
        # Basic sentiment features
        features["positive_word_count"] = positive_count
        features["negative_word_count"] = negative_count
        features["sentiment_word_ratio"] = (positive_count + negative_count) / len(tokens) if tokens else 0.0
        
        # Punctuation features
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        features["capitalization_ratio"] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0
        
        # Emotional indicators
        emotional_punctuation = text.count("!") + text.count("?") + text.count("...")
        features["emotional_punctuation"] = emotional_punctuation
        
        # Length features (longer texts might be more neutral)
        features["text_length"] = len(text)
        features["token_count"] = len(tokens)
        features["avg_word_length"] = np.mean([len(token) for token in tokens]) if tokens else 0.0
        
        # Language-specific features
        if language in ["ta", "ml"]:
            # Dravidian languages might have specific patterns
            features["dravidian_language"] = 1.0
        else:
            features["dravidian_language"] = 0.0
        
        return features
    
    def _create_neutral_result(self) -> SentimentResult:
        """Create neutral sentiment result."""
        return SentimentResult(
            overall_sentiment="neutral",
            confidence=0.5,
            scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            features={"default_neutral": 1.0}
        )
    
    def extract_linguistic_features(self, text: str, tokens: List[str],
                                  pos_tags: Optional[List[str]] = None,
                                  entities: Optional[List] = None,
                                  language: str = "ta") -> Dict[str, float]:
        """Extract comprehensive linguistic features for depression analysis.
        
        Args:
            text: Original text
            tokens: Tokenized text
            pos_tags: POS tags (optional)
            entities: Named entities (optional)
            language: Language code
            
        Returns:
            Dictionary of linguistic features
        """
        features = {}
        
        if not tokens:
            return features
        
        # Basic text statistics
        features["sentence_count"] = max(1, text.count(".") + text.count("!") + text.count("?"))
        features["word_count"] = len(tokens)
        features["avg_sentence_length"] = len(tokens) / features["sentence_count"]
        features["char_count"] = len(text)
        features["avg_word_length"] = np.mean([len(token) for token in tokens])
        
        # Lexical diversity
        unique_tokens = set(token.lower() for token in tokens)
        features["lexical_diversity"] = len(unique_tokens) / len(tokens)
        features["type_token_ratio"] = len(unique_tokens) / len(tokens)
        
        # Function word analysis
        function_words = {
            "ta": {"அது", "இது", "என்", "ஒரு", "மற்றும்", "அல்லது", "ஆனால்", "இல்", "க்கு", "ல்"},
            "ml": {"അത്", "ഇത്", "എന്ന്", "ഒരു", "കൂടാതെ", "അല്ലെങ്കിൽ", "പക്ഷേ", "ൽ", "ക്ക്", "ന്"},
            "en": {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
        }
        
        lang_function_words = function_words.get(language, function_words["en"])
        function_word_count = sum(1 for token in tokens if token.lower() in lang_function_words)
        features["function_word_ratio"] = function_word_count / len(tokens)
        features["content_word_ratio"] = 1.0 - features["function_word_ratio"]
        
        # Pronoun analysis (important for depression detection)
        pronouns = {
            "ta": {"நான்", "என்", "எனக்கு", "என்னை", "நாம்", "நம்", "நமக்கு"},
            "ml": {"ഞാൻ", "എന്റെ", "എനിക്ക്", "എന്നെ", "നാം", "നമ്മുടെ", "നമുക്ക്"},
            "en": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
        }
        
        lang_pronouns = pronouns.get(language, pronouns["en"])
        pronoun_count = sum(1 for token in tokens if token.lower() in lang_pronouns)
        features["pronoun_ratio"] = pronoun_count / len(tokens)
        
        # First person pronoun analysis (self-reference)
        first_person = {
            "ta": {"நான்", "என்", "எனக்கு", "என்னை"},
            "ml": {"ഞാൻ", "എന്റെ", "എനിക്ക്", "എന്നെ"},
            "en": {"i", "me", "my", "mine", "myself"}
        }
        
        lang_first_person = first_person.get(language, first_person["en"])
        first_person_count = sum(1 for token in tokens if token.lower() in lang_first_person)
        features["first_person_ratio"] = first_person_count / len(tokens)
        
        # Sentiment-based features
        sentiment_result = self.analyze_sentiment(text, tokens, language)
        features.update({
            f"sentiment_{key}": value for key, value in sentiment_result.scores.items()
        })
        features["sentiment_confidence"] = sentiment_result.confidence
        
        # POS-based features (if available)
        if pos_tags and len(pos_tags) == len(tokens):
            pos_counts = Counter(pos_tags)
            total_pos = len(pos_tags)
            
            for pos_tag in ["NOUN", "VERB", "ADJ", "ADV", "PRON"]:
                features[f"{pos_tag.lower()}_ratio"] = pos_counts.get(pos_tag, 0) / total_pos
        
        # Entity-based features (if available)
        if entities:
            features["entity_count"] = len(entities)
            features["entity_density"] = len(entities) / len(tokens)
            
            # Count entity types
            entity_types = {}
            for entity in entities:
                entity_type = getattr(entity, 'label', 'MISC')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            for entity_type in ["PERSON", "MEDICAL", "LOCATION"]:
                features[f"{entity_type.lower()}_entity_count"] = entity_types.get(entity_type, 0)
        
        # Readability features
        features["avg_syllables_per_word"] = self._estimate_syllables(tokens)
        features["complex_word_ratio"] = sum(1 for token in tokens if len(token) > 6) / len(tokens)
        
        # Emotional expression features
        features["repetition_ratio"] = self._calculate_repetition_ratio(tokens)
        features["punctuation_density"] = sum(1 for char in text if char in "!?.,;:") / len(text) if text else 0.0
        
        return features
    
    def _estimate_syllables(self, tokens: List[str]) -> float:
        """Estimate average syllables per word (simplified)."""
        if not tokens:
            return 0.0
        
        total_syllables = 0
        for token in tokens:
            # Simple syllable estimation based on vowels
            vowels = "aeiouAEIOUஆஇஈஉஊஎஏஐஒஓആഇഈഉഊഎഏഐഒഓ"
            syllable_count = max(1, sum(1 for char in token if char in vowels))
            total_syllables += syllable_count
        
        return total_syllables / len(tokens)
    
    def _calculate_repetition_ratio(self, tokens: List[str]) -> float:
        """Calculate ratio of repeated words."""
        if not tokens:
            return 0.0
        
        token_counts = Counter(token.lower() for token in tokens)
        repeated_words = sum(count - 1 for count in token_counts.values() if count > 1)
        
        return repeated_words / len(tokens)


def create_sentiment_analyzer(device: str = "cpu") -> SentimentAnalyzer:
    """Factory function to create sentiment analyzer.
    
    Args:
        device: Device to run on ("cpu", "cuda", "mps")
        
    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer(device=device)


# Example usage and testing
if __name__ == "__main__":
    analyzer = create_sentiment_analyzer()
    
    # Test Tamil text
    text = "நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்"
    tokens = text.split()
    
    sentiment_result = analyzer.analyze_sentiment(text, tokens, "ta")
    print(f"Sentiment: {sentiment_result.overall_sentiment}")
    print(f"Confidence: {sentiment_result.confidence:.3f}")
    print(f"Scores: {sentiment_result.scores}")
    
    # Test linguistic features
    features = analyzer.extract_linguistic_features(text, tokens, language="ta")
    print(f"Features: {len(features)} extracted")
    print(f"Sample features: {dict(list(features.items())[:5])}")