"""Model loader for all trained models."""
import torch
import pickle
from pathlib import Path
from typing import Dict, Optional
import logging

from config import DepressionDetectionConfig
from text_model import create_muril_model, create_muril_tokenizer
from enhanced_multimodal_model import create_enhanced_multimodal_model
from linguistic_analyzer import create_linguistic_analyzer
from ssl_model import create_ssl_model
from tfidf_classifier import TFIDFClassifier

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage all trained models."""
    
    def __init__(self, checkpoint_dir: str = "outputs/nlp_training_peak_performance/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = DepressionDetectionConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models
        self.tfidf_model: Optional[TFIDFClassifier] = None
        self.text_model: Optional[torch.nn.Module] = None
        self.audio_model: Optional[torch.nn.Module] = None
        self.enhanced_model: Optional[torch.nn.Module] = None
        
        # Tokenizer and analyzer
        self.tokenizer = None
        self.linguistic_analyzer = None
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def load_tfidf_model(self) -> bool:
        """Load TF-IDF baseline model."""
        try:
            model_path = self.checkpoint_dir / "tfidf_baseline.pkl"
            if not model_path.exists():
                logger.warning(f"TF-IDF model not found at {model_path}")
                return False
            
            self.tfidf_model = TFIDFClassifier(config=self.config.tfidf)
            self.tfidf_model.load(str(model_path))
            logger.info("TF-IDF model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {e}")
            return False
    
    def load_text_model(self) -> bool:
        """Load MuRIL text model."""
        try:
            model_path = self.checkpoint_dir / "muril_text_model.pt"
            if not model_path.exists():
                logger.warning(f"Text model not found at {model_path}")
                return False
            
            self.text_model = create_muril_model(self.config.text_model)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.text_model.load_state_dict(checkpoint['model_state_dict'])
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()
            
            self.tokenizer = create_muril_tokenizer(self.config.text_model)
            logger.info("MuRIL text model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            return False
    
    def load_audio_model(self) -> bool:
        """Load pre-trained audio model."""
        try:
            model_path = Path("outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt")
            if not model_path.exists():
                logger.warning(f"Audio model not found at {model_path}")
                return False
            
            self.audio_model = create_ssl_model(self.config)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.audio_model.load_state_dict(checkpoint['model_state_dict'])
            self.audio_model = self.audio_model.to(self.device)
            self.audio_model.eval()
            logger.info("Audio model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading audio model: {e}")
            return False
    
    def load_enhanced_model(self) -> bool:
        """Load enhanced multimodal model with linguistic features."""
        try:
            model_path = self.checkpoint_dir / "best_enhanced_multimodal_model_baseline.pt"
            if not model_path.exists():
                logger.warning(f"Enhanced model not found at {model_path}")
                return False
            
            # Load audio and text models first
            if self.audio_model is None:
                self.load_audio_model()
            if self.text_model is None:
                self.load_text_model()
            
            # Create linguistic analyzer
            self.linguistic_analyzer = create_linguistic_analyzer()
            
            # Create enhanced model
            from enhanced_multimodal_model import EnhancedFusionConfig
            enhanced_config = EnhancedFusionConfig(
                audio_embedding_dim=self.config.fusion.audio_embedding_dim,
                text_embedding_dim=self.config.fusion.text_embedding_dim,
                linguistic_feature_dim=self.config.fusion.linguistic_feature_dim,
                fusion_hidden_dim=self.config.fusion.fusion_hidden_dim,
                fusion_dropout=self.config.fusion.fusion_dropout,
                num_classes=self.config.fusion.num_classes,
                fusion_method=self.config.fusion.fusion_method,
                allow_unimodal_fallback=True,
                allow_bimodal_fallback=True
            )
            
            self.enhanced_model = create_enhanced_multimodal_model(
                audio_model=self.audio_model,
                text_model=self.text_model,
                linguistic_analyzer=self.linguistic_analyzer,
                config=enhanced_config
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.enhanced_model.load_state_dict(checkpoint['model_state_dict'])
            self.enhanced_model = self.enhanced_model.to(self.device)
            self.enhanced_model.eval()
            
            logger.info("Enhanced multimodal model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available models."""
        results = {
            'tfidf': self.load_tfidf_model(),
            'text': self.load_text_model(),
            'audio': self.load_audio_model(),
            'enhanced': self.load_enhanced_model()
        }
        
        logger.info(f"Model loading results: {results}")
        return results
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        available = []
        if self.tfidf_model is not None:
            available.append('tfidf')
        if self.text_model is not None:
            available.append('text')
        if self.audio_model is not None:
            available.append('audio')
        if self.enhanced_model is not None:
            available.append('enhanced')
        return available


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_all_models()
    return _model_loader
