"""
Dependency Parser for Syntactic Analysis

This module implements dependency parsing using Universal Dependencies framework
with support for Dravidian languages (Tamil and Malayalam) and SOV word order.

Reference: Traditional NLP Tasks - Requirement 3
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DependencyConfig:
    """Configuration for Dependency Parser."""
    model_name: str = "bert-base-multilingual-cased"
    annotation_scheme: str = "universal_dependencies"
    max_length: int = 512
    return_probabilities: bool = True
    device: str = "cpu"
    
    # Universal Dependencies relation labels
    relation_labels: List[str] = field(default_factory=lambda: [
        "root", "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp",
        "obl", "vocative", "expl", "dislocated", "advcl", "advmod",
        "discourse", "aux", "auxpass", "cop", "mark", "nmod", "appos",
        "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc",
        "fixed", "flat", "compound", "list", "parataxis", "orphan",
        "goeswith", "reparandum", "punct", "dep"
    ])


@dataclass
class DependencyTree:
    """Dependency tree representation."""
    tokens: List[str]
    heads: List[int]  # Head indices for each token (0-based, -1 for root)
    relations: List[str]  # Dependency relation labels
    confidence_scores: List[float]
    
    def __post_init__(self):
        """Validate tree structure."""
        assert len(self.tokens) == len(self.heads) == len(self.relations) == len(self.confidence_scores), \
            "All lists must have the same length"
        
        # Validate head indices
        for i, head in enumerate(self.heads):
            if head != -1:  # Not root
                assert 0 <= head < len(self.tokens), \
                    f"Head index {head} for token {i} is out of bounds"
                assert head != i, \
                    f"Token {i} cannot be its own head"
    
    def get_depth(self) -> int:
        """Calculate maximum depth of dependency tree."""
        def calculate_depth(token_idx: int, visited: set) -> int:
            if token_idx in visited:
                return 0  # Cycle detected, return 0
            
            visited.add(token_idx)
            head = self.heads[token_idx]
            
            if head == -1:  # Root
                return 1
            else:
                return 1 + calculate_depth(head, visited.copy())
        
        max_depth = 0
        for i in range(len(self.tokens)):
            depth = calculate_depth(i, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_complexity_metrics(self) -> Dict[str, float]:
        """Calculate syntactic complexity metrics."""
        if not self.tokens:
            return {"tree_depth": 0.0, "avg_dependency_distance": 0.0, 
                   "syntactic_complexity": 0.0, "clause_count": 0.0}
        
        # Tree depth
        tree_depth = self.get_depth()
        
        # Average dependency distance
        distances = []
        for i, head in enumerate(self.heads):
            if head != -1:
                distances.append(abs(i - head))
        avg_distance = np.mean(distances) if distances else 0.0
        
        # Clause count (approximate by counting certain relations)
        clause_relations = {"csubj", "ccomp", "xcomp", "advcl", "acl"}
        clause_count = sum(1 for rel in self.relations if rel in clause_relations)
        
        # Syntactic complexity (composite measure)
        complexity = (tree_depth * 0.4 + avg_distance * 0.3 + 
                     clause_count * 0.3) if len(self.tokens) > 0 else 0.0
        
        return {
            "tree_depth": float(tree_depth),
            "avg_dependency_distance": float(avg_distance),
            "syntactic_complexity": float(complexity),
            "clause_count": float(clause_count)
        }
    
    def is_valid_tree(self) -> bool:
        """Check if the dependency tree is valid (connected and acyclic)."""
        if not self.tokens:
            return True
        
        # Check for exactly one root
        root_count = sum(1 for head in self.heads if head == -1)
        if root_count != 1:
            return False
        
        # Check for cycles using DFS
        def has_cycle(node: int, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            head = self.heads[node]
            if head != -1:  # Not root
                if head not in visited:
                    if has_cycle(head, visited, rec_stack):
                        return True
                elif head in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for i in range(len(self.tokens)):
            if i not in visited:
                if has_cycle(i, visited, set()):
                    return False
        
        return True


class DependencyParser:
    """Dependency parser for syntactic analysis."""
    
    def __init__(self, config: DependencyConfig):
        """Initialize dependency parser with configuration.
        
        Args:
            config: DependencyConfig instance with model and processing settings
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create relation mappings
            self.id2rel = {i: rel for i, rel in enumerate(config.relation_labels)}
            self.rel2id = {rel: i for i, rel in enumerate(config.relation_labels)}
            
            logger.info(f"Initialized dependency parser with {config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dependency parser: {e}")
            # Fallback to rule-based parser
            self._init_fallback_parser()
    
    def _init_fallback_parser(self):
        """Initialize rule-based fallback parser."""
        logger.warning("Using fallback rule-based dependency parser")
        self.tokenizer = None
        self.model = None
        
        # Simple rules for Dravidian SOV structure
        self.sov_patterns = {
            "ta": {
                "subject_markers": ["நான்", "அவன்", "அவள்", "நீ"],
                "object_markers": ["ஐ", "க்கு", "ல்"],
                "verb_endings": ["கிறேன்", "கிறான்", "கிறாள்", "கிறது"]
            },
            "ml": {
                "subject_markers": ["ഞാൻ", "അവൻ", "അവൾ", "നീ"],
                "object_markers": ["യെ", "ക്ക്", "ൽ"],
                "verb_endings": ["ുന്നു", "ുന്ന", "ിച്ചു"]
            }
        }
    
    def parse_sentence(self, tokens: List[str], language: str) -> DependencyTree:
        """Parse sentence and return dependency tree.
        
        Args:
            tokens: List of tokens to parse
            language: Language code ("ta" or "ml")
            
        Returns:
            DependencyTree instance
        """
        if not tokens:
            return DependencyTree([], [], [], [])
        
        if self.model is not None:
            return self._parse_with_model(tokens, language)
        else:
            return self._parse_with_rules(tokens, language)
    
    def _parse_with_model(self, tokens: List[str], language: str) -> DependencyTree:
        """Parse using transformer model (simplified implementation)."""
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
        
        # Get embeddings (simplified - real parser would have head prediction layers)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0]
        
        # Simplified head prediction (just for demonstration)
        # In a real implementation, this would use proper parsing algorithms
        heads = []
        relations = []
        confidences = []
        
        for i in range(len(tokens)):
            if i == len(tokens) - 1:  # Last token is often the root (verb in SOV)
                heads.append(-1)
                relations.append("root")
            else:
                # Simple heuristic: attach to next token or root
                if i < len(tokens) - 1:
                    heads.append(len(tokens) - 1)  # Attach to root
                    relations.append("nsubj" if i == 0 else "obj")
                else:
                    heads.append(-1)
                    relations.append("root")
            
            confidences.append(0.7)  # Dummy confidence
        
        return DependencyTree(tokens, heads, relations, confidences)
    
    def _parse_with_rules(self, tokens: List[str], language: str) -> DependencyTree:
        """Parse using rule-based approach for SOV languages."""
        if len(tokens) == 1:
            return DependencyTree(tokens, [-1], ["root"], [1.0])
        
        heads = []
        relations = []
        confidences = []
        
        # Simple SOV parsing rules
        patterns = self.sov_patterns.get(language, self.sov_patterns["ta"])
        
        # Find verb (usually at the end in SOV)
        verb_idx = len(tokens) - 1
        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            if any(token.endswith(ending) for ending in patterns["verb_endings"]):
                verb_idx = i
                break
        
        # Assign heads and relations
        for i, token in enumerate(tokens):
            if i == verb_idx:
                # Verb is root
                heads.append(-1)
                relations.append("root")
                confidences.append(0.9)
            elif token in patterns["subject_markers"]:
                # Subject attaches to verb
                heads.append(verb_idx)
                relations.append("nsubj")
                confidences.append(0.8)
            elif any(token.endswith(marker) for marker in patterns["object_markers"]):
                # Object attaches to verb
                heads.append(verb_idx)
                relations.append("obj")
                confidences.append(0.8)
            else:
                # Default: attach to verb
                heads.append(verb_idx)
                relations.append("obl" if i < verb_idx else "advmod")
                confidences.append(0.6)
        
        return DependencyTree(tokens, heads, relations, confidences)
    
    def extract_syntactic_features(self, tree: DependencyTree) -> Dict[str, float]:
        """Extract syntactic features for depression analysis.
        
        Args:
            tree: DependencyTree instance
            
        Returns:
            Dictionary of syntactic features
        """
        if not tree.tokens:
            return self._empty_syntactic_features()
        
        # Get complexity metrics
        complexity_metrics = tree.get_complexity_metrics()
        
        # Count relation types
        relation_counts = {}
        for relation in tree.relations:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        total_relations = len(tree.relations)
        
        # Calculate features
        features = {
            # Basic syntactic metrics
            **complexity_metrics,
            
            # Relation distribution
            "total_dependencies": float(total_relations),
            "relation_diversity": float(len(set(tree.relations))),
            
            # Specific relation ratios (important for depression analysis)
            "subject_ratio": relation_counts.get("nsubj", 0) / total_relations if total_relations > 0 else 0.0,
            "object_ratio": relation_counts.get("obj", 0) / total_relations if total_relations > 0 else 0.0,
            "modifier_ratio": (relation_counts.get("advmod", 0) + relation_counts.get("amod", 0)) / total_relations if total_relations > 0 else 0.0,
            
            # Complexity indicators
            "subordination_ratio": (relation_counts.get("csubj", 0) + relation_counts.get("ccomp", 0) + 
                                  relation_counts.get("advcl", 0)) / total_relations if total_relations > 0 else 0.0,
            
            # Tree validity
            "tree_validity": 1.0 if tree.is_valid_tree() else 0.0,
            "avg_confidence": float(np.mean(tree.confidence_scores)) if tree.confidence_scores else 0.0,
        }
        
        return features
    
    def _empty_syntactic_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            "tree_depth": 0.0,
            "avg_dependency_distance": 0.0,
            "syntactic_complexity": 0.0,
            "clause_count": 0.0,
            "total_dependencies": 0.0,
            "relation_diversity": 0.0,
            "subject_ratio": 0.0,
            "object_ratio": 0.0,
            "modifier_ratio": 0.0,
            "subordination_ratio": 0.0,
            "tree_validity": 0.0,
            "avg_confidence": 0.0,
        }


def create_dependency_parser(language: str = "ta", device: str = "cpu") -> DependencyParser:
    """Factory function to create dependency parser.
    
    Args:
        language: Language code ("ta" or "ml")
        device: Device to run on ("cpu", "cuda", "mps")
        
    Returns:
        DependencyParser instance
    """
    config = DependencyConfig(device=device)
    return DependencyParser(config)


if __name__ == "__main__":
    # Example usage
    parser = create_dependency_parser("ta")
    
    # Test Tamil sentence: "I am coming"
    tokens = ["நான்", "வருகிறேன்"]
    tree = parser.parse_sentence(tokens, "ta")
    
    print("Parsed sentence:")
    for i, (token, head, rel) in enumerate(zip(tree.tokens, tree.heads, tree.relations)):
        head_token = tree.tokens[head] if head != -1 else "ROOT"
        print(f"  {i}: {token} -> {head_token} ({rel})")
    
    # Extract features
    features = parser.extract_syntactic_features(tree)
    print("Syntactic features:", features)
    
    # Check tree validity
    print("Tree is valid:", tree.is_valid_tree())
    print("Tree depth:", tree.get_depth())
    print("Complexity metrics:", tree.get_complexity_metrics())