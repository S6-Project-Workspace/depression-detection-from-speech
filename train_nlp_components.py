#!/usr/bin/env python3
"""
Complete NLP Component Training Script

This script trains all NLP components for the depression detection system:
1. Text-only models (TF-IDF baseline + MuRIL transformer)
2. Multimodal fusion models (Audio + Text)
3. Enhanced multimodal with linguistic features

Usage:
    python train_nlp_components.py --mode all
    python train_nlp_components.py --mode text_only
    python train_nlp_components.py --mode multimodal
    python train_nlp_components.py --mode enhanced
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime

# Import configuration
from config import get_config, DepressionDetectionConfig

# Import models
from text_model import create_muril_model
from tfidf_classifier import TFIDFClassifier
from ssl_model import create_ssl_model
from ecapa_model import create_ecapa_model

# Import datasets
from multimodal_dataset import create_multimodal_dataloaders
from dataset import create_dataloaders

# Import trainers
from multimodal_trainer import create_multimodal_trainer, compare_models
from enhanced_multimodal_trainer import create_enhanced_multimodal_trainer
from trainer import ModelTrainer

# Import utilities
from text_preprocessing import TextPreprocessor
from linguistic_analyzer import create_linguistic_analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NLPTrainingPipeline:
    """Complete training pipeline for all NLP components."""
    
    def __init__(
        self,
        config: DepressionDetectionConfig,
        transcript_dir: str = "transcripts",
        output_dir: str = "outputs/nlp_training",
        device: str = "mps"
    ):
        """Initialize training pipeline.
        
        Args:
            config: Depression detection configuration
            transcript_dir: Directory containing transcripts
            output_dir: Directory for saving outputs
            device: Device to train on (cuda/mps/cpu)
        """
        self.config = config
        self.transcript_dir = Path(transcript_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Update config paths to match output directory
        self.config.paths.output_dir = str(self.output_dir)
        self.config.paths.checkpoint_dir = str(self.output_dir / "checkpoints")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor()
        self.linguistic_analyzer = create_linguistic_analyzer()
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized NLP Training Pipeline")
        logger.info(f"  Transcript dir: {self.transcript_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
    
    def check_transcripts(self) -> bool:
        """Check if transcripts are available.
        
        Returns:
            True if transcripts exist, False otherwise
        """
        tamil_depressed = self.transcript_dir / "tamil" / "depressed"
        tamil_non_depressed = self.transcript_dir / "tamil" / "non_depressed"
        malayalam_depressed = self.transcript_dir / "malayalam" / "depressed"
        malayalam_non_depressed = self.transcript_dir / "malayalam" / "non_depressed"
        
        all_exist = all([
            tamil_depressed.exists(),
            tamil_non_depressed.exists(),
            malayalam_depressed.exists(),
            malayalam_non_depressed.exists()
        ])
        
        if all_exist:
            # Count files
            tamil_dep_count = len(list(tamil_depressed.glob("*.json")))
            tamil_non_dep_count = len(list(tamil_non_depressed.glob("*.json")))
            malayalam_dep_count = len(list(malayalam_depressed.glob("*.json")))
            malayalam_non_dep_count = len(list(malayalam_non_depressed.glob("*.json")))
            
            logger.info(f"Transcript counts:")
            logger.info(f"  Tamil Depressed: {tamil_dep_count}")
            logger.info(f"  Tamil Non-Depressed: {tamil_non_dep_count}")
            logger.info(f"  Malayalam Depressed: {malayalam_dep_count}")
            logger.info(f"  Malayalam Non-Depressed: {malayalam_non_dep_count}")
            
            return tamil_dep_count > 0 and malayalam_dep_count > 0
        
        return False
    
    def train_tfidf_baseline(self) -> Dict:
        """Train TF-IDF baseline classifier.
        
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*60)
        logger.info("Training TF-IDF Baseline Classifier")
        logger.info("="*60)
        
        try:
            # Load transcripts
            transcripts_tamil = self._load_transcripts("tamil")
            transcripts_malayalam = self._load_transcripts("malayalam")
            
            # Combine datasets
            all_texts = transcripts_tamil['texts'] + transcripts_malayalam['texts']
            all_labels = transcripts_tamil['labels'] + transcripts_malayalam['labels']
            
            logger.info(f"Loaded {len(all_texts)} transcripts")
            logger.info(f"  Depressed: {sum(all_labels)}")
            logger.info(f"  Non-Depressed: {len(all_labels) - sum(all_labels)}")
            
            # Preprocess texts
            logger.info("Preprocessing texts...")
            preprocessed_texts = []
            for text, lang in zip(all_texts, 
                                 ['ta'] * len(transcripts_tamil['texts']) + 
                                 ['ml'] * len(transcripts_malayalam['texts'])):
                preprocessed = self.text_preprocessor.preprocess(text, lang)
                # Convert PreprocessedText to string
                preprocessed_str = preprocessed.normalized if hasattr(preprocessed, 'normalized') else str(preprocessed)
                preprocessed_texts.append(preprocessed_str)
            
            # Train TF-IDF classifier
            logger.info("Training TF-IDF classifier...")
            tfidf_classifier = TFIDFClassifier(config=self.config.tfidf)
            
            # Split data for training
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                preprocessed_texts, all_labels,
                test_size=0.2,
                random_state=42,
                stratify=all_labels
            )
            
            # Train
            tfidf_classifier.fit(X_train, y_train)
            
            # Evaluate
            train_metrics = tfidf_classifier.evaluate(X_train, y_train)
            val_metrics = tfidf_classifier.evaluate(X_val, y_val)
            
            # Get predictions for detailed metrics
            val_preds = tfidf_classifier.predict(X_val)
            
            from sklearn.metrics import classification_report, f1_score
            val_f1 = val_metrics['macro_f1']
            train_f1 = train_metrics['macro_f1']
            
            logger.info(f"\nTF-IDF Results:")
            logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Train Macro-F1: {train_f1:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val Macro-F1: {val_f1:.4f}")
            
            # Save model
            model_path = self.output_dir / "checkpoints" / "tfidf_baseline.pkl"
            tfidf_classifier.save(str(model_path))
            logger.info(f"Saved TF-IDF model to {model_path}")
            
            results = {
                'train_accuracy': float(train_metrics['accuracy']),
                'train_macro_f1': float(train_f1),
                'val_accuracy': float(val_metrics['accuracy']),
                'val_macro_f1': float(val_f1),
                'model_path': str(model_path)
            }
            
            self.results['tfidf_baseline'] = results
            return results
            
        except Exception as e:
            logger.error(f"Error training TF-IDF baseline: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def train_text_model(self) -> Dict:
        """Train MuRIL text model.
        
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*60)
        logger.info("Training MuRIL Text Model")
        logger.info("="*60)
        
        try:
            # Create text model
            text_model = create_muril_model(self.config.text_model)
            text_model = text_model.to(self.device)
            
            logger.info(f"Created MuRIL model on {self.device}")
            
            # Load transcripts and create dataloaders
            logger.info("Creating dataloaders...")
            train_loader, val_loader = self._create_text_dataloaders()
            
            # Setup optimizer and loss
            # Use lower learning rate for MPS stability
            lr = self.config.training.learning_rate
            if str(self.device) == 'mps':
                lr = lr * 0.1  # Reduce LR by 10x for MPS
                logger.info(f"Using reduced learning rate for MPS: {lr}")
            
            optimizer = torch.optim.AdamW(
                text_model.parameters(),
                lr=lr,
                weight_decay=self.config.training.weight_decay,
                eps=1e-8  # Increase epsilon for numerical stability
            )
            
            criterion = torch.nn.CrossEntropyLoss()
            
            # Gradient clipping value
            max_grad_norm = 1.0
            
            # Training loop
            logger.info("Starting training...")
            best_val_f1 = 0.0
            best_model_state = None
            
            for epoch in range(self.config.training.epochs):
                # Train
                text_model.train()
                train_loss = 0.0
                train_preds = []
                train_labels = []
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['text_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    logits = text_model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                    # Check for nan loss
                    if torch.isnan(loss):
                        logger.warning("NaN loss detected, skipping batch")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                
                train_loss /= len(train_loader)
                train_f1 = f1_score(train_labels, train_preds, average='macro')
                
                # Validate
                text_model.eval()
                val_loss = 0.0
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['text_attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        logits = text_model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_f1 = f1_score(val_labels, val_preds, average='macro')
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config.training.epochs}: "
                    f"Train Loss={train_loss:.4f}, Train F1={train_f1:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}"
                )
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = text_model.state_dict().copy()
            
            # Restore best model
            if best_model_state is not None:
                text_model.load_state_dict(best_model_state)
            
            # Save model
            model_path = self.output_dir / "checkpoints" / "muril_text_model.pt"
            torch.save({
                'model_state_dict': text_model.state_dict(),
                'best_val_f1': best_val_f1
            }, str(model_path))
            logger.info(f"Saved MuRIL model to {model_path}")
            
            results = {
                'best_val_f1': float(best_val_f1),
                'model_path': str(model_path)
            }
            self.results['muril_text'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training MuRIL text model: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def train_multimodal_fusion(self) -> Dict:
        """Train multimodal fusion model (Audio + Text).
        
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*60)
        logger.info("Training Multimodal Fusion Model (Audio + Text)")
        logger.info("="*60)
        
        try:
            # Load pre-trained audio model
            logger.info("Loading pre-trained audio model...")
            audio_model_path = "outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt"
            
            if not os.path.exists(audio_model_path):
                logger.warning(f"Audio model not found at {audio_model_path}")
                logger.warning("Training audio model first...")
                # Train audio model if not exists
                audio_model = create_ssl_model(self.config)
            else:
                audio_model = create_ssl_model(self.config)
                checkpoint = torch.load(audio_model_path, map_location=self.device)
                audio_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded audio model from {audio_model_path}")
            
            audio_model = audio_model.to(self.device)
            
            # Create text model
            logger.info("Creating text model...")
            text_model = create_muril_model(self.config.text_model)
            text_model = text_model.to(self.device)
            
            # Create multimodal dataloaders
            logger.info("Creating multimodal dataloaders...")
            from text_model import create_muril_tokenizer
            tokenizer = create_muril_tokenizer(self.config.text_model)
            
            # Use smaller batch size for multimodal to avoid OOM
            original_batch_size = self.config.training.batch_size
            multimodal_batch_size = min(4, original_batch_size)  # Reduce to 4 or less
            logger.info(f"Using batch size {multimodal_batch_size} for multimodal training (original: {original_batch_size})")
            self.config.training.batch_size = multimodal_batch_size
            
            train_loader, val_loader, _ = create_multimodal_dataloaders(
                config=self.config,
                tokenizer=tokenizer,
                transcript_dir=str(self.transcript_dir),
                fold_idx=0,
                language="combined",
                text_preprocessor=self.text_preprocessor,
                enable_linguistic_features=False
            )
            
            # Restore original batch size
            self.config.training.batch_size = original_batch_size
            
            # Create multimodal trainer
            logger.info("Creating multimodal trainer...")
            
            # Verify actual embedding dimensions from models
            with torch.no_grad():
                dummy_audio = torch.randn(2, 16000).to(self.device)
                _, audio_emb = audio_model(dummy_audio, return_embeddings=True)
                actual_audio_dim = audio_emb.shape[1]
                
                dummy_text_ids = torch.randint(0, 1000, (2, 128)).to(self.device)
                dummy_text_mask = torch.ones(2, 128).to(self.device)
                text_emb = text_model.get_embeddings(dummy_text_ids, dummy_text_mask)
                actual_text_dim = text_emb.shape[1]
            
            logger.info(f"Actual audio embedding dim: {actual_audio_dim}")
            logger.info(f"Actual text embedding dim: {actual_text_dim}")
            logger.info(f"Config audio embedding dim: {self.config.fusion.audio_embedding_dim}")
            logger.info(f"Config text embedding dim: {self.config.fusion.text_embedding_dim}")
            
            # Update config if needed
            if actual_audio_dim != self.config.fusion.audio_embedding_dim:
                logger.warning(f"Updating audio_embedding_dim from {self.config.fusion.audio_embedding_dim} to {actual_audio_dim}")
                self.config.fusion.audio_embedding_dim = actual_audio_dim
            if actual_text_dim != self.config.fusion.text_embedding_dim:
                logger.warning(f"Updating text_embedding_dim from {self.config.fusion.text_embedding_dim} to {actual_text_dim}")
                self.config.fusion.text_embedding_dim = actual_text_dim
            
            trainer = create_multimodal_trainer(
                audio_model=audio_model,
                text_model=text_model,
                config=self.config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device
            )
            
            # Train
            logger.info("Starting multimodal training...")
            results = trainer.train(num_epochs=self.config.training.epochs)
            
            # Save model
            model_path = "multimodal_fusion.pt"
            trainer.save_checkpoint(model_path)
            logger.info(f"Saved multimodal model to {self.output_dir / 'checkpoints' / model_path}")
            
            results['model_path'] = str(model_path)
            self.results['multimodal_fusion'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training multimodal fusion: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def train_enhanced_multimodal(self) -> Dict:
        """Train enhanced multimodal model with linguistic features.
        
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*60)
        logger.info("Training Enhanced Multimodal Model (Audio + Text + Linguistic)")
        logger.info("="*60)
        
        try:
            # Load pre-trained models
            logger.info("Loading pre-trained models...")
            
            # Audio model
            audio_model_path = "outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt"
            audio_model = create_ssl_model(self.config)
            if os.path.exists(audio_model_path):
                checkpoint = torch.load(audio_model_path, map_location=self.device)
                audio_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded audio model from {audio_model_path}")
            audio_model = audio_model.to(self.device)
            
            # Text model
            text_model = create_muril_model(self.config.text_model)
            text_model = text_model.to(self.device)
            
            # Create enhanced multimodal dataloaders with linguistic features
            logger.info("Creating enhanced dataloaders with linguistic features...")
            from text_model import create_muril_tokenizer
            tokenizer = create_muril_tokenizer(self.config.text_model)
            
            # Use smaller batch size for enhanced multimodal to avoid OOM
            original_batch_size = self.config.training.batch_size
            enhanced_batch_size = min(2, original_batch_size)  # Reduce to 2 or less for enhanced
            logger.info(f"Using batch size {enhanced_batch_size} for enhanced multimodal training (original: {original_batch_size})")
            self.config.training.batch_size = enhanced_batch_size
            
            train_loader, val_loader, _ = create_multimodal_dataloaders(
                config=self.config,
                tokenizer=tokenizer,
                transcript_dir=str(self.transcript_dir),
                fold_idx=0,
                language="combined",
                text_preprocessor=self.text_preprocessor,
                linguistic_analyzer=self.linguistic_analyzer,
                enable_linguistic_features=True
            )
            
            # Restore original batch size
            self.config.training.batch_size = original_batch_size
            
            # Create enhanced multimodal trainer
            logger.info("Creating enhanced multimodal trainer...")
            
            # Create EnhancedFusionConfig from FusionConfig
            from enhanced_multimodal_model import EnhancedFusionConfig
            enhanced_fusion_config = EnhancedFusionConfig(
                audio_embedding_dim=self.config.fusion.audio_embedding_dim,
                text_embedding_dim=self.config.fusion.text_embedding_dim,
                linguistic_feature_dim=self.config.fusion.linguistic_feature_dim,
                fusion_hidden_dim=self.config.fusion.fusion_hidden_dim,
                fusion_dropout=self.config.fusion.fusion_dropout,
                num_classes=self.config.fusion.num_classes,
                fusion_method=self.config.fusion.fusion_method,
                allow_unimodal_fallback=self.config.fusion.allow_unimodal_fallback,
                allow_bimodal_fallback=True
            )
            
            # Verify actual embedding dimensions
            with torch.no_grad():
                dummy_audio = torch.randn(2, 16000).to(self.device)
                _, audio_emb = audio_model(dummy_audio, return_embeddings=True)
                actual_audio_dim = audio_emb.shape[1]
                
                dummy_text_ids = torch.randint(0, 1000, (2, 128)).to(self.device)
                dummy_text_mask = torch.ones(2, 128).to(self.device)
                text_emb = text_model.get_embeddings(dummy_text_ids, dummy_text_mask)
                actual_text_dim = text_emb.shape[1]
            
            logger.info(f"Actual audio embedding dim: {actual_audio_dim}")
            logger.info(f"Actual text embedding dim: {actual_text_dim}")
            
            # Update config if needed
            if actual_audio_dim != enhanced_fusion_config.audio_embedding_dim:
                logger.warning(f"Updating audio_embedding_dim from {enhanced_fusion_config.audio_embedding_dim} to {actual_audio_dim}")
                enhanced_fusion_config.audio_embedding_dim = actual_audio_dim
            if actual_text_dim != enhanced_fusion_config.text_embedding_dim:
                logger.warning(f"Updating text_embedding_dim from {enhanced_fusion_config.text_embedding_dim} to {actual_text_dim}")
                enhanced_fusion_config.text_embedding_dim = actual_text_dim
            
            # Create enhanced model directly
            from enhanced_multimodal_model import create_enhanced_multimodal_model
            enhanced_model = create_enhanced_multimodal_model(
                audio_model=audio_model,
                text_model=text_model,
                linguistic_analyzer=self.linguistic_analyzer,
                config=enhanced_fusion_config
            )
            enhanced_model = enhanced_model.to(self.device)
            
            # Create trainer manually
            from enhanced_multimodal_trainer import EnhancedMultimodalTrainer
            trainer = EnhancedMultimodalTrainer(
                model=enhanced_model,
                config=self.config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                progressive_training=True
            )
            
            # Train with progressive strategy
            logger.info("Starting enhanced multimodal training (progressive)...")
            results = trainer.train(num_epochs=self.config.training.epochs)
            
            # Save model
            model_path = "enhanced_multimodal.pt"
            trainer.save_checkpoint(model_path)
            logger.info(f"Saved enhanced multimodal model to {self.output_dir / 'checkpoints' / model_path}")
            
            results['model_path'] = str(model_path)
            self.results['enhanced_multimodal'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training enhanced multimodal: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _load_transcripts(self, language: str) -> Dict[str, List]:
        """Load transcripts for a language.
        
        Args:
            language: 'tamil' or 'malayalam'
            
        Returns:
            Dictionary with texts and labels
        """
        texts = []
        labels = []
        
        # Load depressed transcripts
        depressed_dir = self.transcript_dir / language / "depressed"
        for json_file in depressed_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = data.get('transcript', data.get('text', ''))
                    if text.strip():
                        texts.append(text)
                        labels.append(1)  # Depressed
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        # Load non-depressed transcripts
        non_depressed_dir = self.transcript_dir / language / "non_depressed"
        for json_file in non_depressed_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = data.get('transcript', data.get('text', ''))
                    if text.strip():
                        texts.append(text)
                        labels.append(0)  # Non-depressed
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        return {'texts': texts, 'labels': labels}
    
    def _create_text_dataloaders(self):
        """Create dataloaders for text-only training.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        from torch.utils.data import Dataset, DataLoader, random_split
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.text_model.model_name)
        
        # Load all transcripts
        tamil_data = self._load_transcripts("tamil")
        malayalam_data = self._load_transcripts("malayalam")
        
        all_texts = tamil_data['texts'] + malayalam_data['texts']
        all_labels = tamil_data['labels'] + malayalam_data['labels']
        
        # Create dataset
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'text_attention_mask': encoding['attention_mask'].squeeze(0),
                    'label': torch.tensor(label, dtype=torch.long)
                }
        
        # Create dataset
        dataset = TextDataset(all_texts, all_labels, tokenizer)
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0  # Avoid pickling issues with nested class
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0  # Avoid pickling issues with nested class
        )
        
        return train_loader, val_loader
    
    def save_results(self):
        """Save all training results to JSON."""
        results_file = self.output_dir / "results" / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nSaved results to {results_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        
        for model_name, results in self.results.items():
            logger.info(f"\n{model_name}:")
            for key, value in results.items():
                if key != 'training_history':
                    logger.info(f"  {key}: {value}")
    
    def run_all(self):
        """Run complete training pipeline."""
        logger.info("\n" + "="*60)
        logger.info("Starting Complete NLP Training Pipeline")
        logger.info("="*60)
        
        # Check transcripts
        if not self.check_transcripts():
            logger.error("Transcripts not found! Please run transcript generation first.")
            return
        
        # 1. Train TF-IDF baseline
        logger.info("\n[1/4] Training TF-IDF Baseline...")
        self.train_tfidf_baseline()
        
        # 2. Train MuRIL text model
        logger.info("\n[2/4] Training MuRIL Text Model...")
        self.train_text_model()
        
        # 3. Train multimodal fusion
        logger.info("\n[3/4] Training Multimodal Fusion...")
        self.train_multimodal_fusion()
        
        # 4. Train enhanced multimodal
        logger.info("\n[4/4] Training Enhanced Multimodal...")
        self.train_enhanced_multimodal()
        
        # Save results
        self.save_results()
        
        logger.info("\n" + "="*60)
        logger.info("Complete NLP Training Pipeline Finished!")
        logger.info("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NLP components for depression detection")
    
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'tfidf', 'text', 'multimodal', 'enhanced'],
        help='Training mode'
    )
    
    parser.add_argument(
        '--transcript-dir',
        type=str,
        default='transcripts',
        help='Directory containing transcripts'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/nlp_training',
        help='Output directory for models and results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to train on'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Get configuration
    config = get_config("combined")
    config.device = args.device
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    
    # Create pipeline
    pipeline = NLPTrainingPipeline(
        config=config,
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run training based on mode
    if args.mode == 'all':
        pipeline.run_all()
    elif args.mode == 'tfidf':
        pipeline.train_tfidf_baseline()
        pipeline.save_results()
    elif args.mode == 'text':
        pipeline.train_text_model()
        pipeline.save_results()
    elif args.mode == 'multimodal':
        pipeline.train_multimodal_fusion()
        pipeline.save_results()
    elif args.mode == 'enhanced':
        pipeline.train_enhanced_multimodal()
        pipeline.save_results()


if __name__ == "__main__":
    main()
