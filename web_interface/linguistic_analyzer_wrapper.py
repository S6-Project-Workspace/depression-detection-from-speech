"""
Linguistic Analyzer Wrapper for Web Interface

This module provides a wrapper around the existing NLP components
(POS tagger, NER system, dependency parser) with concurrent processing,
error handling, and fallback mechanisms for the web interface.

Reference: NLP Web Interface - Requirements 2.1, 2.2, 2.3, 2.5
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Import existing NLP components
try:
    from pos_tagger import POSTagger, POSConfig, POSResult
    from ner_system import NERSystem, NERConfig, Entity
    from dependency_parser import DependencyParser, DependencyConfig, DependencyTree
    from linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig, LinguisticAnalysis
    HAS_NLP_COMPONENTS = True
except ImportError as e:
    logging.warning(f"NLP components not available: {e}")
    HAS_NLP_COMPONENTS = False
    # Create mock types for type hints
    POSResult = Any
    Entity = Any
    DependencyTree = Any

# Import sentiment analyzer
try:
    from .sentiment_analyzer import SentimentAnalyzer, SentimentResult, create_sentiment_analyzer
    HAS_SENTIMENT_ANALYZER = True
except ImportError as e:
    logging.warning(f"Sentiment analyzer not available: {e}")
    HAS_SENTIMENT_ANALYZER = False
    SentimentResult = Any

logger = logging.getLogger(__name__)


@dataclass
class WebLinguisticConfig:
    """Configuration for web interface linguistic analyzer."""
    device: str = "cpu"
    max_concurrent_tasks: int = 3
    timeout_seconds: float = 30.0
    enable_fallback: bool = True
    cache_results: bool = True
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["ta", "ml", "en"]


@dataclass
class WebAnalysisResult:
    """Web interface analysis result with enhanced metadata."""
    text: str
    tokens: List[str]
    pos_result: Optional[POSResult]
    entities: List[Entity]
    dependency_tree: Optional[DependencyTree]
    sentiment_result: Optional[SentimentResult]
    linguistic_features: Dict[str, float]
    language: str
    processing_time: float
    component_status: Dict[str, str]  # Status of each component
    error_messages: List[str]
    
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return len(self.error_messages) == 0 and (
            self.pos_result is not None or 
            len(self.entities) > 0 or 
            self.dependency_tree is not None or
            self.sentiment_result is not None
        )


class ComponentError(Exception):
    """Exception for component-specific errors."""
    def __init__(self, component: str, message: str, original_error: Exception = None):
        self.component = component
        self.message = message
        self.original_error = original_error
        super().__init__(f"{component}: {message}")


class WebLinguisticAnalyzer:
    """Web interface wrapper for linguistic analysis components."""
    
    def __init__(self, config: WebLinguisticConfig):
        """Initialize the web linguistic analyzer.
        
        Args:
            config: Configuration for the analyzer
        """
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self.cache = {} if config.cache_results else None
        
        # Initialize components
        self.pos_tagger = None
        self.ner_system = None
        self.dependency_parser = None
        self.linguistic_analyzer = None
        self.sentiment_analyzer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize NLP components with error handling."""
        if not HAS_NLP_COMPONENTS:
            logger.warning("NLP components not available, using fallback mode")
            return
        
        try:
            # Initialize POS tagger
            pos_config = POSConfig(device=self.config.device)
            self.pos_tagger = POSTagger(pos_config)
            logger.info("POS tagger initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize POS tagger: {e}")
            if not self.config.enable_fallback:
                raise ComponentError("pos_tagger", "Initialization failed", e)
        
        try:
            # Initialize NER system
            ner_config = NERConfig(device=self.config.device)
            self.ner_system = NERSystem(ner_config)
            logger.info("NER system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NER system: {e}")
            if not self.config.enable_fallback:
                raise ComponentError("ner_system", "Initialization failed", e)
        
        try:
            # Initialize dependency parser
            dep_config = DependencyConfig(device=self.config.device)
            self.dependency_parser = DependencyParser(dep_config)
            logger.info("Dependency parser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dependency parser: {e}")
            if not self.config.enable_fallback:
                raise ComponentError("dependency_parser", "Initialization failed", e)
        
        try:
            # Initialize unified linguistic analyzer
            ling_config = LinguisticConfig(device=self.config.device)
            self.linguistic_analyzer = LinguisticAnalyzer(ling_config)
            logger.info("Linguistic analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize linguistic analyzer: {e}")
            if not self.config.enable_fallback:
                raise ComponentError("linguistic_analyzer", "Initialization failed", e)
        
        try:
            # Initialize sentiment analyzer
            if HAS_SENTIMENT_ANALYZER:
                self.sentiment_analyzer = create_sentiment_analyzer(self.config.device)
                logger.info("Sentiment analyzer initialized successfully")
            else:
                logger.warning("Sentiment analyzer not available")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            if not self.config.enable_fallback:
                raise ComponentError("sentiment_analyzer", "Initialization failed", e)
    
    async def analyze_text(self, text: str, language: str = "ta") -> WebAnalysisResult:
        """Analyze text with concurrent processing and error handling.
        
        Args:
            text: Input text to analyze
            language: Language code ("ta", "ml", "en")
            
        Returns:
            WebAnalysisResult with analysis results and metadata
        """
        start_time = time.time()
        
        # Validate input
        if not text.strip():
            return self._create_empty_result(text, language, "Empty text provided")
        
        if language not in self.config.supported_languages:
            logger.warning(f"Unsupported language: {language}, defaulting to 'ta'")
            language = "ta"
        
        # Check cache
        cache_key = f"{hash(text)}_{language}"
        if self.cache and cache_key in self.cache:
            logger.debug("Returning cached result")
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        # Run concurrent analysis
        result = await self._run_concurrent_analysis(text, tokens, language)
        result.processing_time = time.time() - start_time
        
        # Cache result
        if self.cache:
            self.cache[cache_key] = result
        
        return result
    
    async def _run_concurrent_analysis(self, text: str, tokens: List[str], 
                                     language: str) -> WebAnalysisResult:
        """Run all analysis components concurrently."""
        component_status = {}
        error_messages = []
        
        # Create analysis tasks
        tasks = {}
        
        if self.pos_tagger or self.config.enable_fallback:
            tasks['pos'] = asyncio.get_event_loop().run_in_executor(
                self.executor, self._safe_pos_analysis, tokens, language
            )
        
        if self.ner_system or self.config.enable_fallback:
            tasks['ner'] = asyncio.get_event_loop().run_in_executor(
                self.executor, self._safe_ner_analysis, tokens, language
            )
        
        if self.dependency_parser or self.config.enable_fallback:
            tasks['dep'] = asyncio.get_event_loop().run_in_executor(
                self.executor, self._safe_dependency_analysis, tokens, language
            )
        
        if self.sentiment_analyzer or self.config.enable_fallback:
            tasks['sentiment'] = asyncio.get_event_loop().run_in_executor(
                self.executor, self._safe_sentiment_analysis, text, tokens, language
            )
        
        # Wait for all tasks with timeout
        results = {}
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=self.config.timeout_seconds
            )
            
            for task_name, result in zip(tasks.keys(), completed_tasks):
                if isinstance(result, Exception):
                    component_status[task_name] = "error"
                    error_messages.append(f"{task_name}: {str(result)}")
                    results[task_name] = None
                else:
                    component_status[task_name] = "success"
                    results[task_name] = result
                    
        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout after {self.config.timeout_seconds}s")
            for task_name in tasks.keys():
                component_status[task_name] = "timeout"
            error_messages.append("Analysis timeout")
            results = {task_name: None for task_name in tasks.keys()}
        
        # Extract linguistic features
        linguistic_features = self._extract_combined_features(
            results.get('pos'), results.get('ner'), results.get('dep'), results.get('sentiment')
        )
        
        return WebAnalysisResult(
            text=text,
            tokens=tokens,
            pos_result=results.get('pos'),
            entities=results.get('ner', []),
            dependency_tree=results.get('dep'),
            sentiment_result=results.get('sentiment'),
            linguistic_features=linguistic_features,
            language=language,
            processing_time=0.0,  # Will be set by caller
            component_status=component_status,
            error_messages=error_messages
        )
    
    def _safe_pos_analysis(self, tokens: List[str], language: str) -> Optional[POSResult]:
        """Safely perform POS analysis with error handling."""
        try:
            if self.pos_tagger:
                tagged = self.pos_tagger.tag_sentence(tokens, language)
                tags = [tag for _, tag in tagged]
                confidences = [0.8] * len(tokens)  # Default confidence
                
                return self.pos_tagger.create_pos_result(
                    tokens, tags, confidences, language
                )
            else:
                return self._fallback_pos_analysis(tokens, language)
                
        except Exception as e:
            logger.error(f"POS analysis failed: {e}")
            if self.config.enable_fallback:
                return self._fallback_pos_analysis(tokens, language)
            raise ComponentError("pos_tagger", str(e), e)
    
    def _safe_ner_analysis(self, tokens: List[str], language: str) -> List[Entity]:
        """Safely perform NER analysis with error handling."""
        try:
            if self.ner_system:
                return self.ner_system.extract_entities(tokens, language)
            else:
                return self._fallback_ner_analysis(tokens, language)
                
        except Exception as e:
            logger.error(f"NER analysis failed: {e}")
            if self.config.enable_fallback:
                return self._fallback_ner_analysis(tokens, language)
            raise ComponentError("ner_system", str(e), e)
    
    def _safe_dependency_analysis(self, tokens: List[str], language: str) -> Optional[DependencyTree]:
        """Safely perform dependency analysis with error handling."""
        try:
            if self.dependency_parser:
                return self.dependency_parser.parse_sentence(tokens, language)
            else:
                return self._fallback_dependency_analysis(tokens, language)
                
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            if self.config.enable_fallback:
                return self._fallback_dependency_analysis(tokens, language)
            raise ComponentError("dependency_parser", str(e), e)
    
    def _safe_sentiment_analysis(self, text: str, tokens: List[str], language: str) -> Optional[SentimentResult]:
        """Safely perform sentiment analysis with error handling."""
        try:
            if self.sentiment_analyzer:
                return self.sentiment_analyzer.analyze_sentiment(text, tokens, language)
            else:
                return self._fallback_sentiment_analysis(text, tokens, language)
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            if self.config.enable_fallback:
                return self._fallback_sentiment_analysis(text, tokens, language)
            raise ComponentError("sentiment_analyzer", str(e), e)
    
    def _fallback_pos_analysis(self, tokens: List[str], language: str) -> POSResult:
        """Fallback POS analysis using simple rules."""
        tags = []
        for token in tokens:
            if token in ".,!?;:":
                tags.append("PUNCT")
            elif token.isdigit():
                tags.append("NUM")
            elif token in ["நான்", "அவர்", "நீ", "ഞാൻ", "അവർ", "നീ"]:
                tags.append("PRON")
            else:
                tags.append("NOUN")  # Default
        
        confidences = [0.5] * len(tokens)  # Lower confidence for fallback
        pos_distribution = {tag: tags.count(tag) for tag in set(tags)}
        
        if HAS_NLP_COMPONENTS:
            return POSResult(
                tokens=tokens,
                tags=tags,
                confidence_scores=confidences,
                pos_distribution=pos_distribution,
                language=language
            )
        else:
            # Mock POSResult for testing
            return type('POSResult', (), {
                'tokens': tokens,
                'tags': tags,
                'confidence_scores': confidences,
                'pos_distribution': pos_distribution,
                'language': language
            })()
    
    def _fallback_ner_analysis(self, tokens: List[str], language: str) -> List[Entity]:
        """Fallback NER analysis using simple patterns."""
        entities = []
        
        # Simple pattern matching
        for i, token in enumerate(tokens):
            if token in ["நான்", "அவர்", "ഞാൻ", "അവർ"]:
                if HAS_NLP_COMPONENTS:
                    entities.append(Entity(
                        text=token,
                        label="PERSON",
                        start_idx=i,
                        end_idx=i+1,
                        confidence=0.5
                    ))
                else:
                    # Mock Entity for testing
                    entities.append(type('Entity', (), {
                        'text': token,
                        'label': "PERSON",
                        'start_idx': i,
                        'end_idx': i+1,
                        'confidence': 0.5
                    })())
        
        return entities
    
    def _fallback_dependency_analysis(self, tokens: List[str], language: str) -> DependencyTree:
        """Fallback dependency analysis using simple rules."""
        if not tokens:
            if HAS_NLP_COMPONENTS:
                return DependencyTree([], [], [], [])
            else:
                return type('DependencyTree', (), {
                    'tokens': [],
                    'heads': [],
                    'relations': [],
                    'confidence_scores': []
                })()
        
        # Simple SOV structure: last token is root
        heads = [len(tokens) - 1] * len(tokens)
        heads[-1] = -1  # Root
        
        relations = ["nsubj" if i == 0 else "obj" for i in range(len(tokens))]
        relations[-1] = "root"
        
        confidences = [0.5] * len(tokens)
        
        if HAS_NLP_COMPONENTS:
            return DependencyTree(tokens, heads, relations, confidences)
        else:
            return type('DependencyTree', (), {
                'tokens': tokens,
                'heads': heads,
                'relations': relations,
                'confidence_scores': confidences
            })()
    
    def _fallback_sentiment_analysis(self, text: str, tokens: List[str], language: str) -> SentimentResult:
        """Fallback sentiment analysis using simple rules."""
        # Simple positive/negative word counting
        positive_words = {"good", "great", "happy", "love", "நல்ல", "மகிழ்ச்சி", "നല്ല", "സന്തോഷം"}
        negative_words = {"bad", "sad", "hate", "angry", "கெட்ட", "துக்கம்", "മോശം", "ദുഃഖം"}
        
        positive_count = sum(1 for token in tokens if token.lower() in positive_words)
        negative_count = sum(1 for token in tokens if token.lower() in negative_words)
        
        if positive_count > negative_count:
            overall_sentiment = "positive"
            confidence = 0.6
            scores = {"positive": 0.6, "negative": 0.2, "neutral": 0.2}
        elif negative_count > positive_count:
            overall_sentiment = "negative"
            confidence = 0.6
            scores = {"positive": 0.2, "negative": 0.6, "neutral": 0.2}
        else:
            overall_sentiment = "neutral"
            confidence = 0.5
            scores = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        
        if HAS_SENTIMENT_ANALYZER:
            return SentimentResult(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                scores=scores,
                features={"fallback_sentiment": 1.0}
            )
        else:
            # Mock SentimentResult for testing
            return type('SentimentResult', (), {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'scores': scores,
                'features': {"fallback_sentiment": 1.0}
            })()
    
    def _extract_combined_features(self, pos_result: Optional[POSResult], 
                                 entities: List[Entity], 
                                 dependency_tree: Optional[DependencyTree],
                                 sentiment_result: Optional[SentimentResult]) -> Dict[str, float]:
        """Extract combined linguistic features from all components."""
        features = {}
        
        # POS features
        if pos_result and hasattr(pos_result, 'tokens'):
            if self.pos_tagger and hasattr(self.pos_tagger, 'extract_pos_features'):
                try:
                    tagged_sentence = list(zip(pos_result.tokens, pos_result.tags))
                    pos_features = self.pos_tagger.extract_pos_features(tagged_sentence)
                    features.update(pos_features)
                except Exception as e:
                    logger.error(f"Failed to extract POS features: {e}")
            else:
                # Basic POS features
                features.update({
                    "pos_diversity": len(set(pos_result.tags)) / len(pos_result.tags) if pos_result.tags else 0.0,
                    "noun_ratio": pos_result.tags.count("NOUN") / len(pos_result.tags) if pos_result.tags else 0.0,
                    "verb_ratio": pos_result.tags.count("VERB") / len(pos_result.tags) if pos_result.tags else 0.0,
                })
        
        # NER features
        if entities:
            if self.ner_system and hasattr(self.ner_system, 'extract_ner_features'):
                try:
                    ner_features = self.ner_system.extract_ner_features(entities)
                    features.update(ner_features)
                except Exception as e:
                    logger.error(f"Failed to extract NER features: {e}")
            else:
                # Basic NER features
                features.update({
                    "entity_count": len(entities),
                    "person_entities": sum(1 for e in entities if getattr(e, 'label', '') == "PERSON"),
                    "avg_entity_confidence": np.mean([getattr(e, 'confidence', 0.0) for e in entities]) if entities else 0.0,
                })
        
        # Dependency features
        if dependency_tree and hasattr(dependency_tree, 'tokens'):
            if self.dependency_parser and hasattr(self.dependency_parser, 'extract_syntactic_features'):
                try:
                    dep_features = self.dependency_parser.extract_syntactic_features(dependency_tree)
                    features.update(dep_features)
                except Exception as e:
                    logger.error(f"Failed to extract dependency features: {e}")
            else:
                # Basic dependency features
                features.update({
                    "sentence_length": len(dependency_tree.tokens),
                    "avg_dependency_distance": np.mean([
                        abs(i - head) for i, head in enumerate(dependency_tree.heads) 
                        if head != -1
                    ]) if dependency_tree.heads else 0.0,
                })
        
        # Sentiment features
        if sentiment_result:
            try:
                # Add sentiment scores as features
                for sentiment_type, score in sentiment_result.scores.items():
                    features[f"sentiment_{sentiment_type}"] = score
                
                features["sentiment_confidence"] = sentiment_result.confidence
                
                # Add sentiment-specific features if available
                if hasattr(sentiment_result, 'features') and sentiment_result.features:
                    for feature_name, feature_value in sentiment_result.features.items():
                        features[f"sentiment_feature_{feature_name}"] = feature_value
                
                # Add overall sentiment as categorical feature
                sentiment_mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                features["sentiment_polarity"] = sentiment_mapping.get(sentiment_result.overall_sentiment, 0.0)
                
            except Exception as e:
                logger.error(f"Failed to extract sentiment features: {e}")
        
        # Enhanced linguistic features using sentiment analyzer
        if self.sentiment_analyzer and hasattr(self.sentiment_analyzer, 'extract_linguistic_features'):
            try:
                # Get tokens from any available source
                tokens = []
                if pos_result and hasattr(pos_result, 'tokens'):
                    tokens = pos_result.tokens
                elif dependency_tree and hasattr(dependency_tree, 'tokens'):
                    tokens = dependency_tree.tokens
                
                if tokens:
                    # Extract comprehensive linguistic features
                    pos_tags = pos_result.tags if pos_result and hasattr(pos_result, 'tags') else None
                    
                    enhanced_features = self.sentiment_analyzer.extract_linguistic_features(
                        text=" ".join(tokens),  # Reconstruct text
                        tokens=tokens,
                        pos_tags=pos_tags,
                        entities=entities,
                        language="ta"  # Default language
                    )
                    
                    # Add enhanced features with prefix to avoid conflicts
                    for feature_name, feature_value in enhanced_features.items():
                        if feature_name not in features:  # Avoid overwriting existing features
                            features[f"enhanced_{feature_name}"] = feature_value
                            
            except Exception as e:
                logger.error(f"Failed to extract enhanced linguistic features: {e}")
        
        return features
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for text."""
        # Basic whitespace tokenization with punctuation handling
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def _create_empty_result(self, text: str, language: str, error_msg: str) -> WebAnalysisResult:
        """Create empty result for error cases."""
        return WebAnalysisResult(
            text=text,
            tokens=[],
            pos_result=None,
            entities=[],
            dependency_tree=None,
            sentiment_result=None,
            linguistic_features={},
            language=language,
            processing_time=0.0,
            component_status={},
            error_messages=[error_msg]
        )
    
    async def analyze_batch(self, texts: List[str], language: str = "ta") -> List[WebAnalysisResult]:
        """Analyze multiple texts concurrently.
        
        Args:
            texts: List of texts to analyze
            language: Language code
            
        Returns:
            List of WebAnalysisResult instances
        """
        if not texts:
            return []
        
        # Create analysis tasks
        tasks = [self.analyze_text(text, language) for text in texts]
        
        # Run concurrently with proper error handling
        results = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = self._create_empty_result(
                        texts[i], language, f"Analysis failed: {str(result)}"
                    )
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # Return error results for all texts
            results = [
                self._create_empty_result(text, language, f"Batch analysis failed: {str(e)}")
                for text in texts
            ]
        
        return results
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            "pos_tagger": self.pos_tagger is not None,
            "ner_system": self.ner_system is not None,
            "dependency_parser": self.dependency_parser is not None,
            "linguistic_analyzer": self.linguistic_analyzer is not None,
            "sentiment_analyzer": self.sentiment_analyzer is not None,
            "has_nlp_components": HAS_NLP_COMPONENTS,
            "has_sentiment_analyzer": HAS_SENTIMENT_ANALYZER,
            "fallback_enabled": self.config.enable_fallback
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Analysis cache cleared")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def create_web_linguistic_analyzer(device: str = "cpu", 
                                 max_concurrent_tasks: int = 3,
                                 enable_fallback: bool = True) -> WebLinguisticAnalyzer:
    """Factory function to create web linguistic analyzer.
    
    Args:
        device: Device to run on ("cpu", "cuda", "mps")
        max_concurrent_tasks: Maximum concurrent analysis tasks
        enable_fallback: Whether to enable fallback mechanisms
        
    Returns:
        WebLinguisticAnalyzer instance
    """
    config = WebLinguisticConfig(
        device=device,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_fallback=enable_fallback
    )
    
    return WebLinguisticAnalyzer(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_analyzer():
        """Test the web linguistic analyzer."""
        analyzer = create_web_linguistic_analyzer()
        
        # Test single text analysis
        text = "நான் மகிழ்ச்சியாக இருக்கிறேன்"
        result = await analyzer.analyze_text(text, "ta")
        
        print(f"Analysis successful: {result.is_successful()}")
        print(f"Tokens: {result.tokens}")
        print(f"Component status: {result.component_status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Features: {list(result.linguistic_features.keys())}")
        
        if result.error_messages:
            print(f"Errors: {result.error_messages}")
        
        # Test batch analysis
        texts = ["நான் வருகிறேன்", "அவன் போகிறான்"]
        batch_results = await analyzer.analyze_batch(texts, "ta")
        
        print(f"\nBatch analysis completed: {len(batch_results)} results")
        for i, result in enumerate(batch_results):
            print(f"  Text {i+1}: {result.is_successful()}, {len(result.tokens)} tokens")
        
        # Check component status
        status = analyzer.get_component_status()
        print(f"\nComponent status: {status}")
    
    # Run test
    asyncio.run(test_analyzer())