"""
Tests for Enhanced Multimodal Trainer

This module tests the enhanced multimodal trainer that supports linguistic features,
progressive training, and feature importance analysis.

Reference: Traditional NLP Tasks - Task 7
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass
from typing import Dict, List, Optional

from enhanced_multimodal_trainer import (
    EnhancedMultimodalTrainer,
    EnhancedTrainingMetrics,
    EnhancedEvaluationResult,
    create_enhanced_multimodal_trainer
)
from enhanced_multimodal_model import EnhancedMultimodalModel, EnhancedFusionConfig
from multimodal_trainer import compute_evaluation_metrics
from config import DepressionDetectionConfig, get_config
from linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig


class TestEnhancedTrainingMetrics:
    """Test enhanced training metrics data structure."""
    
    def test_enhanced_metrics_creation(self):
        """Test creating enhanced training metrics."""
        metrics = EnhancedTrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_f1=0.8,
            val_f1=0.75,
            val_precision=0.76,
            val_recall=0.74,
            val_f1_depressed=0.73,
            val_f1_non_depressed=0.77,
            learning_rate=1e-4,
            best_val_f1=0.75,
            linguistic_feature_importance=0.3,
            audio_feature_importance=0.4,
            text_feature_importance=0.3,
            training_phase="enhanced"
        )
        
        assert metrics.linguistic_feature_importance == 0.3
        assert metrics.audio_feature_importance == 0.4
        assert metrics.text_feature_importance == 0.3
        assert metrics.training_phase == "enhanced"
    
    def test_enhanced_metrics_to_dict(self):
        """Test converting enhanced metrics to dictionary."""
        metrics = EnhancedTrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_f1=0.8,
            val_f1=0.75,
            val_precision=0.76,
            val_recall=0.74,
            val_f1_depressed=0.73,
            val_f1_non_depressed=0.77,
            learning_rate=1e-4,
            best_val_f1=0.75,
            linguistic_feature_importance=0.3,
            training_phase="baseline"
        )
        
        metrics_dict = metrics.to_dict()
        
        # Check enhanced fields
        assert 'linguistic_feature_importance' in metrics_dict
        assert 'training_phase' in metrics_dict
        assert metrics_dict['training_phase'] == "baseline"
        assert metrics_dict['linguistic_feature_importance'] == 0.3
    
    def test_enhanced_metrics_with_confusion_matrix(self):
        """Test enhanced metrics with confusion matrix."""
        cm = np.array([[10, 2], [3, 15]])
        metrics = EnhancedTrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_f1=0.8,
            val_f1=0.75,
            val_precision=0.76,
            val_recall=0.74,
            val_f1_depressed=0.73,
            val_f1_non_depressed=0.77,
            learning_rate=1e-4,
            best_val_f1=0.75,
            confusion_matrix=cm
        )
        
        metrics_dict = metrics.to_dict()
        assert 'confusion_matrix' in metrics_dict
        assert isinstance(metrics_dict['confusion_matrix'], list)
        assert len(metrics_dict['confusion_matrix']) == 2
        assert len(metrics_dict['confusion_matrix'][0]) == 2


class TestEnhancedEvaluationResult:
    """Test enhanced evaluation result data structure."""
    
    def test_enhanced_evaluation_result_creation(self):
        """Test creating enhanced evaluation result."""
        cm = np.array([[10, 2], [3, 15]])
        
        result = EnhancedEvaluationResult(
            macro_f1=0.8,
            f1_depressed=0.75,
            f1_non_depressed=0.85,
            precision_depressed=0.8,
            precision_non_depressed=0.9,
            recall_depressed=0.7,
            recall_non_depressed=0.8,
            macro_precision=0.85,
            macro_recall=0.75,
            confusion_matrix=cm,
            loss=0.3,
            linguistic_feature_importance=0.3,
            audio_feature_importance=0.4,
            text_feature_importance=0.3,
            linguistic_contribution_score=0.05
        )
        
        assert result.linguistic_feature_importance == 0.3
        assert result.audio_feature_importance == 0.4
        assert result.text_feature_importance == 0.3
        assert result.linguistic_contribution_score == 0.05
    
    def test_enhanced_evaluation_result_to_dict(self):
        """Test converting enhanced evaluation result to dictionary."""
        cm = np.array([[10, 2], [3, 15]])
        
        result = EnhancedEvaluationResult(
            macro_f1=0.8,
            f1_depressed=0.75,
            f1_non_depressed=0.85,
            precision_depressed=0.8,
            precision_non_depressed=0.9,
            recall_depressed=0.7,
            recall_non_depressed=0.8,
            macro_precision=0.85,
            macro_recall=0.75,
            confusion_matrix=cm,
            loss=0.3,
            linguistic_feature_importance=0.3,
            audio_feature_importance=0.4,
            text_feature_importance=0.3
        )
        
        result_dict = result.to_dict()
        
        # Check base fields
        assert 'macro_f1' in result_dict
        assert 'confusion_matrix' in result_dict
        
        # Check enhanced fields
        enhanced_keys = [
            'linguistic_feature_importance', 'audio_feature_importance', 'text_feature_importance',
            'trimodal_performance', 'bimodal_performance', 'unimodal_performance',
            'linguistic_contribution_score'
        ]
        for key in enhanced_keys:
            assert key in result_dict, f"Missing enhanced key: {key}"
    
    def test_enhanced_evaluation_result_with_ablation_results(self):
        """Test enhanced evaluation result with ablation study results."""
        cm = np.array([[10, 2], [3, 15]])
        
        trimodal_perf = {'macro_f1': 0.85, 'precision': 0.83, 'recall': 0.87}
        bimodal_perf = {
            'audio_text': {'macro_f1': 0.80, 'precision': 0.78, 'recall': 0.82},
            'audio_linguistic': {'macro_f1': 0.75, 'precision': 0.73, 'recall': 0.77}
        }
        unimodal_perf = {
            'audio': {'macro_f1': 0.70, 'precision': 0.68, 'recall': 0.72},
            'text': {'macro_f1': 0.65, 'precision': 0.63, 'recall': 0.67}
        }
        
        result = EnhancedEvaluationResult(
            macro_f1=0.85,
            f1_depressed=0.83,
            f1_non_depressed=0.87,
            precision_depressed=0.85,
            precision_non_depressed=0.81,
            recall_depressed=0.81,
            recall_non_depressed=0.93,
            macro_precision=0.83,
            macro_recall=0.87,
            confusion_matrix=cm,
            loss=0.25,
            trimodal_performance=trimodal_perf,
            bimodal_performance=bimodal_perf,
            unimodal_performance=unimodal_perf
        )
        
        assert result.trimodal_performance == trimodal_perf
        assert result.bimodal_performance == bimodal_perf
        assert result.unimodal_performance == unimodal_perf
        
        result_dict = result.to_dict()
        assert result_dict['trimodal_performance'] == trimodal_perf
        assert result_dict['bimodal_performance'] == bimodal_perf
        assert result_dict['unimodal_performance'] == unimodal_perf


class TestEnhancedMultimodalTrainer:
    """Test enhanced multimodal trainer functionality."""
    
    @pytest.fixture
    def mock_enhanced_model(self):
        """Create a mock enhanced multimodal model."""
        model = Mock()
        model.forward.return_value = torch.randn(4, 2)  # Batch size 4, 2 classes
        model.forward_bimodal_audio_text.return_value = torch.randn(4, 2)
        model.forward_bimodal_audio_linguistic.return_value = torch.randn(4, 2)
        model.forward_bimodal_text_linguistic.return_value = torch.randn(4, 2)
        model.forward_audio_only.return_value = torch.randn(4, 2)
        model.forward_text_only.return_value = torch.randn(4, 2)
        model.forward_linguistic_only.return_value = torch.randn(4, 2)
        model.get_feature_importance.return_value = {
            'linguistic': 0.3, 'audio': 0.4, 'text': 0.3
        }
        model.set_linguistic_enabled = Mock()
        model.train = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        
        # Mock named_parameters for optimizer creation
        model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
            ('layer1.bias', torch.randn(10, requires_grad=True)),
            ('layer2.weight', torch.randn(2, 10, requires_grad=True))
        ]
        
        return model
    
    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader with enhanced multimodal data."""
        # Create sample batch data
        batch_data = {
            'waveform': torch.randn(4, 16000),  # 4 samples, 1 second audio
            'input_ids': torch.randint(0, 1000, (4, 128)),  # 4 samples, 128 tokens
            'text_attention_mask': torch.ones(4, 128),
            'linguistic_features': torch.randn(4, 128),  # 4 samples, 128 linguistic features
            'text_raw': ['Sample text 1', 'Sample text 2', 'Sample text 3', 'Sample text 4'],
            'label': torch.tensor([0, 1, 0, 1])  # Binary labels
        }
        
        # Create mock loader
        loader = Mock()
        loader.__iter__ = Mock(return_value=iter([batch_data]))
        loader.__len__ = Mock(return_value=10)  # Non-zero length to avoid division by zero
        
        return loader
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return get_config("combined")
    
    def test_enhanced_trainer_initialization(self, mock_enhanced_model, mock_data_loader, config):
        """Test enhanced multimodal trainer initialization."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu',
            progressive_training=True
        )
        
        assert trainer.progressive_training is True
        assert trainer.current_training_phase == "enhanced"
        assert isinstance(trainer.model, Mock)
    
    def test_prepare_batch_with_linguistic_features(self, mock_enhanced_model, mock_data_loader, config):
        """Test batch preparation with linguistic features."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        # Sample batch with all modalities
        batch = {
            'waveform': torch.randn(2, 16000),
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'text_attention_mask': torch.ones(2, 128),
            'linguistic_features': torch.randn(2, 128),
            'text_raw': ['Sample text 1', 'Sample text 2'],
            'label': torch.tensor([0, 1])
        }
        
        prepared = trainer._prepare_batch(batch)
        
        assert 'audio_input' in prepared
        assert 'text_input_ids' in prepared
        assert 'text_attention_mask' in prepared
        assert 'linguistic_features' in prepared
        assert 'text_raw' in prepared
        assert prepared['audio_input'].shape == (2, 16000)
        assert prepared['linguistic_features'].shape == (2, 128)
    
    def test_forward_trimodal(self, mock_enhanced_model, mock_data_loader, config):
        """Test forward pass with all three modalities."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        inputs = {
            'audio_input': torch.randn(2, 16000),
            'text_input_ids': torch.randint(0, 1000, (2, 128)),
            'text_attention_mask': torch.ones(2, 128),
            'linguistic_features': torch.randn(2, 128)
        }
        
        # Set the return value to match expected shape
        mock_enhanced_model.return_value = torch.randn(2, 2)
        
        logits = trainer._forward(inputs)
        
        # Should call the model (which calls __call__)
        mock_enhanced_model.assert_called_once()
        assert logits.shape == (2, 2)
    
    def test_forward_bimodal_fallbacks(self, mock_enhanced_model, mock_data_loader, config):
        """Test forward pass with bimodal fallbacks."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        # Test audio + text
        inputs_audio_text = {
            'audio_input': torch.randn(2, 16000),
            'text_input_ids': torch.randint(0, 1000, (2, 128)),
            'text_attention_mask': torch.ones(2, 128)
        }
        
        logits = trainer._forward(inputs_audio_text)
        mock_enhanced_model.forward_bimodal_audio_text.assert_called_once()
        
        # Test audio + linguistic
        inputs_audio_ling = {
            'audio_input': torch.randn(2, 16000),
            'linguistic_features': torch.randn(2, 128)
        }
        
        logits = trainer._forward(inputs_audio_ling)
        mock_enhanced_model.forward_bimodal_audio_linguistic.assert_called_once()
        
        # Test text + linguistic
        inputs_text_ling = {
            'text_input_ids': torch.randint(0, 1000, (2, 128)),
            'text_attention_mask': torch.ones(2, 128),
            'linguistic_features': torch.randn(2, 128)
        }
        
        logits = trainer._forward(inputs_text_ling)
        mock_enhanced_model.forward_bimodal_text_linguistic.assert_called_once()
    
    def test_forward_unimodal_fallbacks(self, mock_enhanced_model, mock_data_loader, config):
        """Test forward pass with unimodal fallbacks."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        # Test audio only
        inputs_audio = {
            'audio_input': torch.randn(2, 16000)
        }
        
        logits = trainer._forward(inputs_audio)
        mock_enhanced_model.forward_audio_only.assert_called_once()
        
        # Test text only
        inputs_text = {
            'text_input_ids': torch.randint(0, 1000, (2, 128)),
            'text_attention_mask': torch.ones(2, 128)
        }
        
        logits = trainer._forward(inputs_text)
        mock_enhanced_model.forward_text_only.assert_called_once()
        
        # Test linguistic only
        inputs_ling = {
            'linguistic_features': torch.randn(2, 128)
        }
        
        logits = trainer._forward(inputs_ling)
        mock_enhanced_model.forward_linguistic_only.assert_called_once()
    
    def test_forward_no_inputs_raises_error(self, mock_enhanced_model, mock_data_loader, config):
        """Test that forward pass with no inputs raises error."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        inputs = {}  # No inputs
        
        with pytest.raises(ValueError, match="Batch must contain at least one modality"):
            trainer._forward(inputs)
    
    @patch('enhanced_multimodal_trainer.tqdm')
    def test_evaluate_with_ablation(self, mock_tqdm, mock_enhanced_model, mock_data_loader, config):
        """Test evaluation with ablation study."""
        # Setup mock tqdm to return the data loader
        mock_tqdm.return_value = mock_data_loader
        
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        # Mock the criterion
        trainer.criterion = Mock(return_value=torch.tensor(0.5))
        
        result = trainer.evaluate_with_ablation()
        
        assert isinstance(result, EnhancedEvaluationResult)
        assert hasattr(result, 'linguistic_feature_importance')
        assert hasattr(result, 'audio_feature_importance')
        assert hasattr(result, 'text_feature_importance')
        assert hasattr(result, 'linguistic_contribution_score')
    
    def test_validate_returns_enhanced_metrics(self, mock_enhanced_model, mock_data_loader, config):
        """Test that validate returns enhanced metrics."""
        trainer = EnhancedMultimodalTrainer(
            model=mock_enhanced_model,
            config=config,
            train_loader=mock_data_loader,
            val_loader=mock_data_loader,
            device='cpu'
        )
        
        # Mock evaluate method
        mock_eval_result = EnhancedEvaluationResult(
            macro_f1=0.8, f1_depressed=0.75, f1_non_depressed=0.85,
            precision_depressed=0.8, precision_non_depressed=0.9,
            recall_depressed=0.7, recall_non_depressed=0.8,
            macro_precision=0.85, macro_recall=0.75,
            confusion_matrix=np.array([[10, 2], [3, 15]]), loss=0.3,
            linguistic_feature_importance=0.3,
            audio_feature_importance=0.4,
            text_feature_importance=0.3,
            linguistic_contribution_score=0.05
        )
        
        trainer.evaluate = Mock(return_value=mock_eval_result)
        
        metrics = trainer.validate()
        
        # Check enhanced metrics are included
        enhanced_keys = [
            'linguistic_feature_importance', 'audio_feature_importance', 'text_feature_importance',
            'linguistic_contribution_score'
        ]
        for key in enhanced_keys:
            assert key in metrics, f"Missing enhanced metric: {key}"


class TestPropertyBasedTests:
    """Property-based tests for enhanced multimodal trainer."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_length=st.integers(min_value=32, max_value=256),
        audio_length=st.integers(min_value=1000, max_value=32000)
    )
    @settings(deadline=None, max_examples=10)
    def test_property_integration_compatibility(self, batch_size, seq_length, audio_length):
        """
        Feature: traditional-nlp-tasks, Property 6: For any enhanced multimodal model forward pass,
        the linguistic features SHALL integrate seamlessly with existing audio and text embeddings
        without dimension mismatches.
        """
        # Create mock model
        model = Mock()
        model.return_value = torch.randn(batch_size, 2)  # Mock __call__ method
        model.train = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        
        # Mock named_parameters for optimizer creation
        model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
            ('layer1.bias', torch.randn(10, requires_grad=True)),
            ('layer2.weight', torch.randn(2, 10, requires_grad=True))
        ]
        
        # Create mock data loader
        batch_data = {
            'waveform': torch.randn(batch_size, audio_length),
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'text_attention_mask': torch.ones(batch_size, seq_length),
            'linguistic_features': torch.randn(batch_size, 128),  # Fixed 128 dimensions
            'label': torch.randint(0, 2, (batch_size,))
        }
        
        loader = Mock()
        loader.__iter__ = Mock(return_value=iter([batch_data]))
        loader.__len__ = Mock(return_value=10)  # Non-zero length
        
        config = get_config("combined")
        
        trainer = EnhancedMultimodalTrainer(
            model=model,
            config=config,
            train_loader=loader,
            val_loader=loader,
            device='cpu'
        )
        
        # Prepare batch
        prepared = trainer._prepare_batch(batch_data)
        
        # Verify dimensions are compatible
        assert prepared['audio_input'].shape[0] == batch_size
        assert prepared['text_input_ids'].shape[0] == batch_size
        assert prepared['linguistic_features'].shape == (batch_size, 128)
        
        # Forward pass should not raise dimension errors
        logits = trainer._forward(prepared)
        assert logits.shape == (batch_size, 2)


class TestFactoryFunctions:
    """Test factory functions for enhanced multimodal trainer."""
    
    def test_create_enhanced_multimodal_trainer(self):
        """Test factory function for creating enhanced multimodal trainer."""
        # Create mock components
        audio_model = Mock()
        text_model = Mock()
        linguistic_analyzer = Mock(spec=LinguisticAnalyzer)
        
        # Create mock data loaders
        loader = Mock()
        loader.__iter__ = Mock(return_value=iter([]))
        loader.__len__ = Mock(return_value=10)  # Non-zero length
        
        config = get_config("combined")
        
        with patch('enhanced_multimodal_trainer.create_enhanced_multimodal_model') as mock_create_model:
            mock_model = Mock()
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            
            # Mock named_parameters for optimizer creation
            mock_model.named_parameters.return_value = [
                ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
                ('layer1.bias', torch.randn(10, requires_grad=True)),
                ('layer2.weight', torch.randn(2, 10, requires_grad=True))
            ]
            
            mock_create_model.return_value = mock_model
            
            trainer = create_enhanced_multimodal_trainer(
                audio_model=audio_model,
                text_model=text_model,
                linguistic_analyzer=linguistic_analyzer,
                config=config,
                train_loader=loader,
                val_loader=loader,
                device='cpu',
                progressive_training=True
            )
            
            assert isinstance(trainer, EnhancedMultimodalTrainer)
            assert trainer.progressive_training is True
            mock_create_model.assert_called_once()
    
    def test_create_enhanced_multimodal_trainer_with_class_weights(self):
        """Test factory function with class weights."""
        # Create mock components
        audio_model = Mock()
        text_model = Mock()
        linguistic_analyzer = Mock(spec=LinguisticAnalyzer)
        
        # Create mock data loaders
        loader = Mock()
        loader.__iter__ = Mock(return_value=iter([]))
        loader.__len__ = Mock(return_value=10)  # Non-zero length
        
        config = get_config("combined")
        class_weights = torch.tensor([0.4, 0.6])
        
        with patch('enhanced_multimodal_trainer.create_enhanced_multimodal_model') as mock_create_model:
            mock_model = Mock()
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            
            # Mock named_parameters for optimizer creation
            mock_model.named_parameters.return_value = [
                ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
                ('layer1.bias', torch.randn(10, requires_grad=True)),
                ('layer2.weight', torch.randn(2, 10, requires_grad=True))
            ]
            
            mock_create_model.return_value = mock_model
            
            trainer = create_enhanced_multimodal_trainer(
                audio_model=audio_model,
                text_model=text_model,
                linguistic_analyzer=linguistic_analyzer,
                config=config,
                train_loader=loader,
                val_loader=loader,
                device='cpu',
                class_weights=class_weights,
                progressive_training=False
            )
            
            assert isinstance(trainer, EnhancedMultimodalTrainer)
            assert trainer.progressive_training is False
            assert trainer.class_weights is not None


if __name__ == "__main__":
    pytest.main([__file__])