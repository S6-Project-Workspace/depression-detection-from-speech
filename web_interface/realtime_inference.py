"""Real-time inference API for all models."""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from web_interface.model_loader import get_model_loader
from text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class RealtimeInference:
    """Real-time inference using all trained models."""
    
    def __init__(self):
        self.model_loader = get_model_loader()
        self.text_preprocessor = TextPreprocessor()
        self.device = self.model_loader.device
    
    def predict_tfidf(self, text: str) -> Dict:
        """Predict using TF-IDF model."""
        try:
            if self.model_loader.tfidf_model is None:
                return {'error': 'TF-IDF model not loaded'}
            
            # Preprocess text
            preprocessed = self.text_preprocessor.preprocess(text, 'ta')
            preprocessed_str = preprocessed.normalized if hasattr(preprocessed, 'normalized') else str(preprocessed)
            
            # Predict
            prediction = self.model_loader.tfidf_model.predict([preprocessed_str])[0]
            probabilities = self.model_loader.tfidf_model.predict_proba([preprocessed_str])[0]
            
            return {
                'model': 'tfidf',
                'prediction': int(prediction),
                'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'non_depressed': float(probabilities[0]),
                    'depressed': float(probabilities[1])
                }
            }
        except Exception as e:
            logger.error(f"Error in TF-IDF prediction: {e}")
            return {'error': str(e)}
    
    def predict_text(self, text: str) -> Dict:
        """Predict using MuRIL text model."""
        try:
            if self.model_loader.text_model is None or self.model_loader.tokenizer is None:
                return {'error': 'Text model not loaded'}
            
            # Tokenize using the wrapper's tokenize method
            encoded = self.model_loader.tokenizer.tokenize(
                text,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model_loader.text_model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)[0]
                prediction = torch.argmax(logits, dim=1)[0]
            
            return {
                'model': 'text',
                'prediction': int(prediction.cpu()),
                'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
                'confidence': float(probabilities.max().cpu()),
                'probabilities': {
                    'non_depressed': float(probabilities[0].cpu()),
                    'depressed': float(probabilities[1].cpu())
                }
            }
        except Exception as e:
            logger.error(f"Error in text prediction: {e}")
            return {'error': str(e)}
    
    def predict_audio(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Predict using audio model."""
        try:
            if self.model_loader.audio_model is None:
                return {'error': 'Audio model not loaded'}
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                logits, _ = self.model_loader.audio_model(audio_tensor, return_embeddings=True)
                probabilities = torch.softmax(logits, dim=1)[0]
                prediction = torch.argmax(logits, dim=1)[0]
            
            return {
                'model': 'audio',
                'prediction': int(prediction.cpu()),
                'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
                'confidence': float(probabilities.max().cpu()),
                'probabilities': {
                    'non_depressed': float(probabilities[0].cpu()),
                    'depressed': float(probabilities[1].cpu())
                }
            }
        except Exception as e:
            logger.error(f"Error in audio prediction: {e}")
            return {'error': str(e)}
    
    def predict_enhanced(self, audio_array: Optional[np.ndarray], text: Optional[str], 
                        sample_rate: int = 16000) -> Dict:
        """Predict using enhanced multimodal model with linguistic features."""
        try:
            if self.model_loader.enhanced_model is None:
                return {'error': 'Enhanced model not loaded'}
            
            # Determine which modalities are available
            has_audio = audio_array is not None
            has_text = text is not None and text.strip()
            
            if not has_audio and not has_text:
                return {'error': 'At least one modality (audio or text) required'}
            
            # Prepare inputs based on available modalities
            if has_audio and has_text:
                # Full multimodal: audio + text + linguistic
                audio_tensor = torch.from_numpy(audio_array).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                audio_input = audio_tensor.to(self.device)
                
                # Tokenize text
                encoded = self.model_loader.tokenizer.tokenize(
                    text,
                    max_length=512,
                    return_tensors='pt'
                )
                text_input_ids = encoded['input_ids'].to(self.device)
                text_attention_mask = encoded['attention_mask'].to(self.device)
                text_raw = [text]
                
                # Full forward pass
                with torch.no_grad():
                    logits = self.model_loader.enhanced_model(
                        audio_input=audio_input,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        text_raw=text_raw
                    )
                
                modalities_used = {'audio': True, 'text': True, 'linguistic': True}
                
            elif has_text:
                # Text + linguistic only
                encoded = self.model_loader.tokenizer.tokenize(
                    text,
                    max_length=512,
                    return_tensors='pt'
                )
                text_input_ids = encoded['input_ids'].to(self.device)
                text_attention_mask = encoded['attention_mask'].to(self.device)
                text_raw = [text]
                
                # Use text+linguistic forward method
                with torch.no_grad():
                    logits = self.model_loader.enhanced_model.forward_text_linguistic(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        text_raw=text_raw
                    )
                
                modalities_used = {'audio': False, 'text': True, 'linguistic': True}
                
            else:
                # Audio only
                audio_tensor = torch.from_numpy(audio_array).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                audio_input = audio_tensor.to(self.device)
                
                # Use audio-only forward method
                with torch.no_grad():
                    logits = self.model_loader.enhanced_model.forward_audio_only(
                        audio_input=audio_input
                    )
                
                modalities_used = {'audio': True, 'text': False, 'linguistic': False}
            
            # Process logits
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = torch.argmax(logits, dim=1)[0]
            
            return {
                'model': 'enhanced',
                'prediction': int(prediction.cpu()),
                'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
                'confidence': float(probabilities.max().cpu()),
                'probabilities': {
                    'non_depressed': float(probabilities[0].cpu()),
                    'depressed': float(probabilities[1].cpu())
                },
                'modalities_used': modalities_used
            }
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def predict_all(self, audio_array: Optional[np.ndarray], text: Optional[str], 
                   sample_rate: int = 16000) -> Dict:
        """Run prediction with all available models."""
        results = {}
        
        # TF-IDF (text only)
        if text and self.model_loader.tfidf_model is not None:
            results['tfidf'] = self.predict_tfidf(text)
        
        # Text model
        if text and self.model_loader.text_model is not None:
            results['text'] = self.predict_text(text)
        
        # Audio model
        if audio_array is not None and self.model_loader.audio_model is not None:
            results['audio'] = self.predict_audio(audio_array, sample_rate)
        
        # Enhanced multimodal
        if self.model_loader.enhanced_model is not None:
            results['enhanced'] = self.predict_enhanced(audio_array, text, sample_rate)
        
        return results


# Global inference instance
_inference: Optional[RealtimeInference] = None


def get_inference() -> RealtimeInference:
    """Get or create global inference instance."""
    global _inference
    if _inference is None:
        _inference = RealtimeInference()
    return _inference
