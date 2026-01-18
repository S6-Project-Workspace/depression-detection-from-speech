"""
Unified Linguistic Analyzer

This module combines all traditional NLP tasks (POS tagging, NER, dependency parsing)
into a unified interface for comprehensive linguistic analysis of Dravidian languages.

Reference: Traditional NLP Tasks - Requirement 4
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

from pos_tagger import POSTagger, POSConfig, POSResult
from ner_system import NERSystem, NERConfig, Entity
from dependency_parser import DependencyParser, DependencyConfig, DependencyTree

logger = logging.getLogger(__name__)


@dataclass
class LinguisticConfig:
    """Configuration for unified linguistic analyzer."""
    pos_config: POSConfig = field(default_factory=POSConfig)
    ner_config: NERConfig = field(default_factory=NERConfig)
    dependency_config: DependencyConfig = field(default_factory=DependencyConfig)
    feature_extraction: bool = True
    device: str = "cpu"
    
    def __post_init__(self):
        """Sync device across all configs."""
        self.pos_config.device = self.device
        self.ner_config.device = self.device
        self.dependency_config.device = self.device


@dataclass
class LinguisticAnalysis:
    """Complete linguistic analysis result."""
    text: str
    tokens: List[str]
    pos_result: POSResult
    entities: List[Entity]
    dependency_tree: DependencyTree
    linguistic_features: Dict[str, float]
    language: str
    
    def __post_init__(self):
        """Validate analysis consistency."""
        # Check that all components have the same number of tokens
        assert len(self.tokens) == len(self.pos_result.tokens), \
            "Token count mismatch between tokens and POS result"
        assert len(self.tokens) == len(self.dependency_tree.tokens), \
            "Token count mismatch between tokens and dependency tree"


class LinguisticAnalyzer:
    """Unified linguistic analyzer combining POS, NER, and dependency parsing."""
    
    def __init__(self, config: LinguisticConfig):
        """Initialize linguistic analyzer with configuration.
        
        Args:
            config: LinguisticConfig instance with settings for all components
        """
        self.config = config
        
        # Initialize individual components
        try:
            self.pos_tagger = POSTagger(config.pos_config)
            self.ner_system = NERSystem(config.ner_config)
            self.dependency_parser = DependencyParser(config.dependency_config)
            
            logger.info("Initialized unified linguistic analyzer")
            
        except Exception as e:
            logger.error(f"Failed to initialize linguistic analyzer: {e}")
            raise
    
    def analyze_text(self, text: str, language: str) -> LinguisticAnalysis:
        """Perform complete linguistic analysis on text.
        
        Args:
            text: Input text to analyze
            language: Language code ("ta" or "ml")
            
        Returns:
            LinguisticAnalysis instance with all results
        """
        if not text.strip():
            return self._empty_analysis(text, language)
        
        # Tokenize text (simple whitespace tokenization)
        tokens = text.strip().split()
        
        # Perform individual analyses
        pos_result = self._analyze_pos(tokens, language)
        entities = self._analyze_entities(tokens, language)
        dependency_tree = self._analyze_dependencies(tokens, language)
        
        # Extract linguistic features if enabled
        linguistic_features = {}
        if self.config.feature_extraction:
            linguistic_features = self.extract_linguistic_features(
                pos_result, entities, dependency_tree
            )
        
        return LinguisticAnalysis(
            text=text,
            tokens=tokens,
            pos_result=pos_result,
            entities=entities,
            dependency_tree=dependency_tree,
            linguistic_features=linguistic_features,
            language=language
        )
    
    def _analyze_pos(self, tokens: List[str], language: str) -> POSResult:
        """Perform POS tagging analysis."""
        try:
            tagged = self.pos_tagger.tag_sentence(tokens, language)
            tags = [tag for _, tag in tagged]
            confidences = [0.8] * len(tokens)  # Default confidence
            
            return self.pos_tagger.create_pos_result(
                tokens, tags, confidences, language
            )
        except Exception as e:
            logger.error(f"POS tagging failed: {e}")
            # Return empty result
            return POSResult(
                tokens=tokens,
                tags=["NOUN"] * len(tokens),
                confidence_scores=[0.0] * len(tokens),
                pos_distribution={"NOUN": len(tokens)},
                language=language
            )
    
    def _analyze_entities(self, tokens: List[str], language: str) -> List[Entity]:
        """Perform named entity recognition."""
        try:
            return self.ner_system.extract_entities(tokens, language)
        except Exception as e:
            logger.error(f"NER failed: {e}")
            return []
    
    def _analyze_dependencies(self, tokens: List[str], language: str) -> DependencyTree:
        """Perform dependency parsing."""
        try:
            return self.dependency_parser.parse_sentence(tokens, language)
        except Exception as e:
            logger.error(f"Dependency parsing failed: {e}")
            # Return simple tree with last token as root
            heads = [len(tokens) - 1] * len(tokens)
            heads[-1] = -1  # Last token is root
            relations = ["dep"] * len(tokens)
            relations[-1] = "root"
            confidences = [0.0] * len(tokens)
            
            return DependencyTree(tokens, heads, relations, confidences)
    
    def extract_linguistic_features(self, pos_result: POSResult, entities: List[Entity], 
                                  dependency_tree: DependencyTree) -> Dict[str, float]:
        """Extract comprehensive linguistic feature dictionary.
        
        Args:
            pos_result: POS tagging results
            entities: Named entity recognition results
            dependency_tree: Dependency parsing results
            
        Returns:
            Dictionary of linguistic features
        """
        # Extract features from each component
        pos_features = self.pos_tagger.extract_pos_features(
            list(zip(pos_result.tokens, pos_result.tags))
        )
        ner_features = self.ner_system.extract_ner_features(entities)
        syntactic_features = self.dependency_parser.extract_syntactic_features(dependency_tree)
        
        # Combine all features into a single dictionary
        all_features = {
            **pos_features,
            **ner_features,
            **syntactic_features
        }
        
        return all_features
    
    def extract_linguistic_feature_vector(self, pos_result: POSResult, entities: List[Entity], 
                                        dependency_tree: DependencyTree) -> np.ndarray:
        """Extract comprehensive linguistic feature vector.
        
        Args:
            pos_result: POS tagging results
            entities: Named entity recognition results
            dependency_tree: Dependency parsing results
            
        Returns:
            128-dimensional feature vector
        """
        # Get feature dictionary
        all_features = self.extract_linguistic_features(pos_result, entities, dependency_tree)
        
        # Convert to 128-dimensional vector
        feature_vector = self._features_to_vector(all_features)
        
        return feature_vector
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to 128-dimensional vector.
        
        Args:
            features: Dictionary of linguistic features
            
        Returns:
            128-dimensional numpy array
        """
        # Define the expected 128 features in order
        feature_names = [
            # POS Features (40 dimensions)
            "pos_diversity", "function_word_ratio", "content_word_ratio", "pronoun_ratio",
            "first_person_pronoun_ratio", "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
            "pron_ratio", "adp_ratio", "det_ratio", "punct_ratio", "num_ratio", "cconj_ratio",
            # Additional POS ratios (25 more)
            "aux_ratio", "sconj_ratio", "part_ratio", "intj_ratio", "sym_ratio", "x_ratio",
            "propn_ratio", "compound_ratio", "coordination_ratio", "subordination_pos_ratio",
            "modal_ratio", "tense_ratio", "aspect_ratio", "voice_ratio", "mood_ratio",
            "case_ratio", "number_ratio", "gender_ratio", "person_ratio", "definiteness_ratio",
            "animacy_ratio", "polarity_ratio", "degree_ratio", "verbform_ratio", "clause_ratio",
            
            # NER Features (20 dimensions)
            "entity_density", "entity_diversity", "avg_entity_confidence", "person_entity_count",
            "location_entity_count", "organization_entity_count", "medical_entity_count",
            "misc_entity_count", "person_entity_ratio", "medical_entity_ratio",
            "self_reference_entities", "medical_context_ratio", "entity_length_avg",
            "entity_confidence_std", "named_entity_ratio", "common_entity_ratio",
            "rare_entity_ratio", "entity_overlap_ratio", "entity_span_ratio", "entity_type_entropy",
            
            # Syntactic Features (68 dimensions)
            "tree_depth", "avg_dependency_distance", "syntactic_complexity", "clause_count",
            "total_dependencies", "relation_diversity", "subject_ratio", "object_ratio",
            "modifier_ratio", "subordination_ratio", "tree_validity", "avg_confidence",
            # Additional syntactic features (56 more)
            "root_distance_avg", "left_branching_ratio", "right_branching_ratio", "crossing_edges",
            "projectivity_score", "head_final_ratio", "head_initial_ratio", "center_embedding_depth",
            "dependency_length_variance", "attachment_height_avg", "attachment_height_max",
            "coordination_ratio", "apposition_ratio", "relative_clause_ratio", "complement_ratio",
            "adjunct_ratio", "determiner_ratio", "auxiliary_ratio", "copula_ratio",
            "passive_ratio", "question_ratio", "negation_ratio", "modal_dependency_ratio",
            "temporal_ratio", "spatial_ratio", "causal_ratio", "conditional_ratio",
            "comparative_ratio", "superlative_ratio", "intensifier_ratio", "quantifier_ratio",
            "discourse_marker_ratio", "conjunction_ratio", "preposition_ratio", "particle_ratio",
            "interjection_ratio", "vocative_ratio", "expletive_ratio", "dislocated_ratio",
            "parataxis_ratio", "orphan_ratio", "goeswith_ratio", "reparandum_ratio",
            "fixed_ratio", "flat_ratio", "compound_dependency_ratio", "list_ratio",
            "punct_dependency_ratio", "dep_ratio", "nsubj_pass_ratio", "csubj_pass_ratio",
            "obl_agent_ratio", "obl_tmod_ratio", "nmod_poss_ratio", "nmod_tmod_ratio",
            "acl_relcl_ratio", "advcl_tcl_ratio", "ccomp_ratio", "xcomp_ratio"
        ]
        
        # Create feature vector
        vector = np.zeros(128)
        
        for i, feature_name in enumerate(feature_names):
            if i >= 128:  # Safety check
                break
            
            value = features.get(feature_name, 0.0)
            
            # Ensure value is finite and in reasonable range
            if np.isfinite(value):
                # Clip extreme values
                if "ratio" in feature_name or "diversity" in feature_name:
                    value = np.clip(value, 0.0, 1.0)
                elif "count" in feature_name:
                    value = np.clip(value, 0.0, 100.0)
                else:
                    value = np.clip(value, -10.0, 10.0)
                
                vector[i] = value
        
        return vector
    
    def batch_analyze(self, texts: List[str], language: str) -> List[LinguisticAnalysis]:
        """Analyze multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            language: Language code ("ta" or "ml")
            
        Returns:
            List of LinguisticAnalysis instances
        """
        return [self.analyze_text(text, language) for text in texts]
    
    def _empty_analysis(self, text: str, language: str) -> LinguisticAnalysis:
        """Create empty analysis for empty text."""
        empty_pos = POSResult(
            tokens=[],
            tags=[],
            confidence_scores=[],
            pos_distribution={},
            language=language
        )
        
        empty_tree = DependencyTree([], [], [], [])
        
        return LinguisticAnalysis(
            text=text,
            tokens=[],
            pos_result=empty_pos,
            entities=[],
            dependency_tree=empty_tree,
            linguistic_features={},
            language=language
        )


def create_linguistic_analyzer(language: str = "ta", device: str = "cpu", 
                             feature_extraction: bool = True) -> LinguisticAnalyzer:
    """Factory function to create linguistic analyzer.
    
    Args:
        language: Primary language code ("ta" or "ml")
        device: Device to run on ("cpu", "cuda", "mps")
        feature_extraction: Whether to extract linguistic features
        
    Returns:
        LinguisticAnalyzer instance
    """
    # Create configs with fallback models for testing
    pos_config = POSConfig(model_name="fallback", device=device)
    ner_config = NERConfig(model_name="fallback", device=device)
    dependency_config = DependencyConfig(model_name="fallback", device=device)
    
    config = LinguisticConfig(
        pos_config=pos_config,
        ner_config=ner_config,
        dependency_config=dependency_config,
        feature_extraction=feature_extraction,
        device=device
    )
    
    return LinguisticAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    analyzer = create_linguistic_analyzer("ta")
    
    # Test Tamil sentence
    text = "நான் மருத்துவரை சந்தித்தேன்"
    analysis = analyzer.analyze_text(text, "ta")
    
    print("Linguistic Analysis:")
    print(f"Text: {analysis.text}")
    print(f"Tokens: {analysis.tokens}")
    print(f"POS tags: {analysis.pos_result.tags}")
    print(f"Entities: {[(e.text, e.label) for e in analysis.entities]}")
    print(f"Dependency relations: {analysis.dependency_tree.relations}")
    
    if analysis.linguistic_features:
        feature_vector = analyzer.extract_linguistic_feature_vector(
            analysis.pos_result, analysis.entities, analysis.dependency_tree
        )
        print(f"Feature vector shape: {feature_vector.shape}")
        print(f"Feature vector (first 10): {feature_vector[:10]}")
    
    # Test batch analysis
    texts = ["நான் வருகிறேன்", "அவன் போகிறான்"]
    batch_results = analyzer.batch_analyze(texts, "ta")
    print(f"Batch analysis completed for {len(batch_results)} texts")