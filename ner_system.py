"""
Named Entity Recognition System for Clinical Context

This module implements NER for clinical transcripts with support for
Dravidian languages (Tamil and Malayalam) and privacy-preserving anonymization.

Reference: Traditional NLP Tasks - Requirement 2
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class NERConfig:
    """Configuration for NER System."""
    model_name: str = "bert-base-multilingual-cased"
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "LOCATION", "ORGANIZATION", "MEDICAL", "MISC", "O"
    ])
    max_length: int = 512
    anonymize_entities: bool = True
    confidence_threshold: float = 0.8
    device: str = "cpu"
    
    # Entity patterns for rule-based fallback
    person_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "ta": ["அவர்", "அவள்", "அவன்", "நான்", "நீ", "நாம்"],
        "ml": ["അവർ", "അവൾ", "അവൻ", "ഞാൻ", "നീ", "നാം"]
    })
    
    medical_patterns: List[str] = field(default_factory=lambda: [
        "depression", "anxiety", "stress", "medication", "therapy", "counseling",
        "மருந்து", "சிகிச்சை", "மருத்துவர்", "ഔഷധം", "ചികിത്സ", "ഡോക്ടർ"
    ])


@dataclass
class Entity:
    """Named entity with metadata."""
    text: str
    label: str
    start_idx: int
    end_idx: int
    confidence: float
    anonymized_form: str = ""
    
    def __post_init__(self):
        """Generate anonymized form if not provided."""
        if not self.anonymized_form:
            self.anonymized_form = self._generate_anonymized_form()
    
    def _generate_anonymized_form(self) -> str:
        """Generate anonymized form based on entity type."""
        if self.label == "PERSON":
            # Use hash-based anonymization for consistency
            hash_obj = hashlib.md5(self.text.encode())
            return f"PERSON_{hash_obj.hexdigest()[:6].upper()}"
        elif self.label == "LOCATION":
            return f"LOCATION_{len(self.text)}"
        elif self.label == "ORGANIZATION":
            return f"ORG_{len(self.text)}"
        elif self.label == "MEDICAL":
            return f"MEDICAL_TERM"
        else:
            return f"{self.label}_{len(self.text)}"


class NERSystem:
    """Named Entity Recognition system for clinical context."""
    
    def __init__(self, config: NERConfig):
        """Initialize NER system with configuration.
        
        Args:
            config: NERConfig instance with model and processing settings
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.model_name,
                num_labels=len(config.entity_types)
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Create label mappings
            self.id2label = {i: label for i, label in enumerate(config.entity_types)}
            self.label2id = {label: i for i, label in enumerate(config.entity_types)}
            
            logger.info(f"Initialized NER system with {config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NER system: {e}")
            # Fallback to rule-based NER
            self._init_fallback_ner()
    
    def _init_fallback_ner(self):
        """Initialize rule-based fallback NER."""
        logger.warning("Using fallback rule-based NER system")
        self.tokenizer = None
        self.model = None
        
        # Compile regex patterns for efficiency
        self.person_regex = re.compile(
            r'\b(?:' + '|'.join(
                self.config.person_patterns.get("ta", []) + 
                self.config.person_patterns.get("ml", [])
            ) + r')\b',
            re.IGNORECASE
        )
        
        self.medical_regex = re.compile(
            r'\b(?:' + '|'.join(self.config.medical_patterns) + r')\b',
            re.IGNORECASE
        )
        
        # Simple location patterns
        self.location_regex = re.compile(
            r'\b(?:hospital|clinic|center|centre|மருத്துவமനை|ആശുപത്രി)\b',
            re.IGNORECASE
        )
    
    def extract_entities(self, tokens: List[str], language: str) -> List[Entity]:
        """Extract named entities from tokens.
        
        Args:
            tokens: List of tokens to process
            language: Language code ("ta" or "ml")
            
        Returns:
            List of Entity objects
        """
        if not tokens:
            return []
        
        if self.model is not None:
            return self._extract_with_model(tokens, language)
        else:
            return self._extract_with_rules(tokens, language)
    
    def _extract_with_model(self, tokens: List[str], language: str) -> List[Entity]:
        """Extract entities using transformer model."""
        # Join tokens for tokenization
        text = " ".join(tokens)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            confidences = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
        
        # Convert predictions to entities
        entities = []
        current_entity = None
        
        for i, (pred, conf, (start, end)) in enumerate(zip(
            predictions[0], confidences[0], offset_mapping
        )):
            if start == end:  # Skip special tokens
                continue
                
            label = self.id2label[pred.item()]
            confidence = conf.item()
            
            if label != "O" and confidence >= self.config.confidence_threshold:
                entity_text = text[start:end]
                
                if current_entity is None or current_entity.label != label:
                    # Start new entity
                    if current_entity is not None:
                        entities.append(current_entity)
                    
                    current_entity = Entity(
                        text=entity_text,
                        label=label,
                        start_idx=start,
                        end_idx=end,
                        confidence=confidence
                    )
                else:
                    # Extend current entity
                    current_entity.text += entity_text
                    current_entity.end_idx = end
                    current_entity.confidence = min(current_entity.confidence, confidence)
            else:
                # End current entity
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities
    
    def _extract_with_rules(self, tokens: List[str], language: str) -> List[Entity]:
        """Extract entities using rule-based approach."""
        text = " ".join(tokens)
        entities = []
        
        # Simple pattern matching for different entity types (without word boundaries for Unicode)
        patterns = {
            "PERSON": [
                r'நான்', r'அவர்', r'அவள்', r'அவன்',
                r'ഞാൻ', r'അവർ', r'അവൾ', r'അവൻ',
                r'Dr\.\s+\w+', r'[A-Z][a-z]+\s+[A-Z][a-z]+'
            ],
            "MEDICAL": [
                r'மருந்து', r'சிகிச்சை', r'மருத்துவர்',
                r'ഔഷധം', r'ചികിത്സ', r'ഡോക്ടർ',
                r'depression', r'anxiety', r'medication', r'therapy'
            ],
            "LOCATION": [
                r'hospital', r'clinic', r'சென்னை', r'Chennai'
            ]
        }
        
        # Find entities for each type
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(),
                        label=entity_type,
                        start_idx=match.start(),
                        end_idx=match.end(),
                        confidence=0.8
                    ))
        
        # Sort entities by start position and remove duplicates
        entities.sort(key=lambda e: e.start_idx)
        
        # Remove overlapping entities (keep the first one)
        filtered_entities = []
        for entity in entities:
            if not any(e.start_idx <= entity.start_idx < e.end_idx for e in filtered_entities):
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def anonymize_text(self, text: str, entities: List[Entity]) -> str:
        """Anonymize text by replacing entities with anonymized forms.
        
        Args:
            text: Original text
            entities: List of entities to anonymize
            
        Returns:
            Anonymized text
        """
        if not self.config.anonymize_entities or not entities:
            return text
        
        # Sort entities by start position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start_idx, reverse=True)
        
        anonymized_text = text
        for entity in sorted_entities:
            anonymized_text = (
                anonymized_text[:entity.start_idx] + 
                entity.anonymized_form + 
                anonymized_text[entity.end_idx:]
            )
        
        return anonymized_text
    
    def extract_ner_features(self, entities: List[Entity]) -> Dict[str, float]:
        """Extract NER-based features for depression analysis.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary of NER features
        """
        if not entities:
            return self._empty_ner_features()
        
        # Count entities by type
        entity_counts = {}
        for entity in entities:
            entity_counts[entity.label] = entity_counts.get(entity.label, 0) + 1
        
        total_entities = len(entities)
        
        # Calculate features
        features = {
            # Basic entity statistics
            "entity_density": total_entities,  # Will be normalized by sentence length later
            "entity_diversity": len(set(e.label for e in entities)),
            "avg_entity_confidence": sum(e.confidence for e in entities) / total_entities,
            
            # Entity type counts
            "person_entity_count": entity_counts.get("PERSON", 0),
            "location_entity_count": entity_counts.get("LOCATION", 0),
            "organization_entity_count": entity_counts.get("ORGANIZATION", 0),
            "medical_entity_count": entity_counts.get("MEDICAL", 0),
            "misc_entity_count": entity_counts.get("MISC", 0),
            
            # Entity type ratios
            "person_entity_ratio": entity_counts.get("PERSON", 0) / total_entities,
            "medical_entity_ratio": entity_counts.get("MEDICAL", 0) / total_entities,
            
            # Depression-specific features
            "self_reference_entities": entity_counts.get("PERSON", 0),  # Proxy for self-reference
            "medical_context_ratio": entity_counts.get("MEDICAL", 0) / total_entities if total_entities > 0 else 0.0,
        }
        
        return features
    
    def _empty_ner_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            "entity_density": 0.0,
            "entity_diversity": 0.0,
            "avg_entity_confidence": 0.0,
            "person_entity_count": 0.0,
            "location_entity_count": 0.0,
            "organization_entity_count": 0.0,
            "medical_entity_count": 0.0,
            "misc_entity_count": 0.0,
            "person_entity_ratio": 0.0,
            "medical_entity_ratio": 0.0,
            "self_reference_entities": 0.0,
            "medical_context_ratio": 0.0,
        }


def create_ner_system(language: str = "ta", device: str = "cpu", 
                     anonymize: bool = True) -> NERSystem:
    """Factory function to create NER system.
    
    Args:
        language: Language code ("ta" or "ml")
        device: Device to run on ("cpu", "cuda", "mps")
        anonymize: Whether to enable anonymization
        
    Returns:
        NERSystem instance
    """
    config = NERConfig(
        anonymize_entities=anonymize,
        device=device
    )
    return NERSystem(config)


if __name__ == "__main__":
    # Example usage
    ner = create_ner_system("ta")
    
    # Test Tamil sentence with entities
    tokens = ["நான்", "மருத்துவரை", "சந்தித்தேன்", "சென்னையில்"]
    entities = ner.extract_entities(tokens, "ta")
    
    print("Extracted entities:")
    for entity in entities:
        print(f"  {entity.text} -> {entity.label} (confidence: {entity.confidence:.2f})")
    
    # Test anonymization
    text = " ".join(tokens)
    anonymized = ner.anonymize_text(text, entities)
    print(f"Original: {text}")
    print(f"Anonymized: {anonymized}")
    
    # Extract features
    features = ner.extract_ner_features(entities)
    print("NER features:", features)