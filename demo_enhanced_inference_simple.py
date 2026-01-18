#!/usr/bin/env python3
"""
Enhanced Inference Pipeline Simple Demonstration

This script demonstrates the enhanced inference pipeline capabilities
without loading heavy transformer models to avoid memory issues.

Usage:
    python demo_enhanced_inference_simple.py

Features demonstrated:
- Enhanced prediction result structure
- Linguistic marker extraction and interpretation
- Feature importance analysis
- Clinical report generation
- Mock data simulation for all components

Reference: Traditional NLP Tasks - Task 9 (Enhanced Inference Pipeline)
"""

import os
import sys
import tempfile
import numpy as np
import torch
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from inference import (
    EnhancedDepressionInferencePipeline, 
    EnhancedPredictionResult,
    LinguisticMarkers
)
from linguistic_analyzer import LinguisticAnalysis
from pos_tagger import POSResult
from dependency_parser import DependencyTree
from ner_system import Entity


def create_mock_linguistic_analysis(text: str, risk_level: str = "moderate") -> LinguisticAnalysis:
    """Create a mock linguistic analysis for demonstration."""
    
    # Tokenize text (simple split for demo)
    tokens = text.split()
    
    # Mock POS result
    pos_tags = ["PRON", "ADV", "ADJ", "VERB"] * (len(tokens) // 4 + 1)
    pos_result = POSResult(
        tokens=tokens,
        tags=pos_tags[:len(tokens)],
        confidence_scores=[0.9] * len(tokens),
        pos_distribution={"PRON": 2, "ADV": 1, "ADJ": 1, "VERB": 2},
        language="ta"
    )
    
    # Mock entities
    entities = [
        Entity(text="நான்", label="PERSON", start_idx=0, end_idx=4, confidence=0.95),
        Entity(text="மருத்துவர்", label="MEDICAL", start_idx=10, end_idx=18, confidence=0.88)
    ]
    
    # Mock dependency tree
    dependency_tree = DependencyTree(
        tokens=tokens,
        heads=[-1] + list(range(len(tokens)-1)),  # Simple linear structure
        relations=["root"] + ["dep"] * (len(tokens)-1),
        confidence_scores=[0.9] * len(tokens)
    )
    
    # Mock linguistic features based on risk level
    if risk_level == "low":
        features = {
            'first_person_pronoun_ratio': 0.1,
            'negative_adjective_ratio': 0.0,
            'past_tense_verb_ratio': 0.0,
            'self_reference_entities': 1,
            'medical_entity_count': 0,
            'syntactic_complexity': 2.0,
            'avg_dependency_distance': 1.2
        }
    elif risk_level == "high":
        features = {
            'first_person_pronoun_ratio': 0.4,
            'negative_adjective_ratio': 0.3,
            'past_tense_verb_ratio': 0.2,
            'self_reference_entities': 3,
            'medical_entity_count': 2,
            'syntactic_complexity': 3.5,
            'avg_dependency_distance': 2.1
        }
    else:  # moderate
        features = {
            'first_person_pronoun_ratio': 0.2,
            'negative_adjective_ratio': 0.1,
            'past_tense_verb_ratio': 0.05,
            'self_reference_entities': 2,
            'medical_entity_count': 1,
            'syntactic_complexity': 2.8,
            'avg_dependency_distance': 1.6
        }
    
    return LinguisticAnalysis(
        text=text,
        tokens=tokens,
        pos_result=pos_result,
        entities=entities,
        dependency_tree=dependency_tree,
        linguistic_features=features,
        language="ta"
    )


def create_enhanced_prediction_result(text: str, risk_level: str = "moderate") -> EnhancedPredictionResult:
    """Create an enhanced prediction result for demonstration."""
    
    # Create linguistic analysis
    linguistic_analysis = create_mock_linguistic_analysis(text, risk_level)
    
    # Create linguistic features array
    linguistic_features = np.random.randn(128) * 0.1  # Mock 128-dim features
    
    # Set prediction based on risk level
    if risk_level == "low":
        predicted_label = 0
        probability = 0.25
        ssl_prob = 0.3
        ecapa_prob = 0.2
        text_prob = 0.15
        enhanced_prob = 0.25
        audio_imp = 0.4
        text_imp = 0.35
        ling_imp = 0.25
    elif risk_level == "high":
        predicted_label = 1
        probability = 0.85
        ssl_prob = 0.8
        ecapa_prob = 0.82
        text_prob = 0.75
        enhanced_prob = 0.85
        audio_imp = 0.3
        text_imp = 0.25
        ling_imp = 0.45
    else:  # moderate
        predicted_label = 1
        probability = 0.65
        ssl_prob = 0.6
        ecapa_prob = 0.7
        text_prob = 0.55
        enhanced_prob = 0.65
        audio_imp = 0.35
        text_imp = 0.3
        ling_imp = 0.35
    
    # Feature importance
    feature_importance = {
        'audio_importance': audio_imp,
        'text_importance': text_imp,
        'linguistic_importance': ling_imp
    }
    
    # Extract linguistic markers
    config = get_config("combined")
    pipeline = EnhancedDepressionInferencePipeline(config, inference_mode='enhanced', device='cpu')
    markers = pipeline.extract_linguistic_markers(linguistic_analysis)
    linguistic_markers = markers.to_dict()
    
    return EnhancedPredictionResult(
        file_id=f"demo_{risk_level}_001",
        speaker_id=f"DEMO_SPEAKER_{risk_level.upper()}",
        predicted_label=predicted_label,
        probability=probability,
        ssl_probability=ssl_prob,
        ecapa_probability=ecapa_prob,
        text_probability=text_prob,
        enhanced_probability=enhanced_prob,
        linguistic_analysis=linguistic_analysis,
        linguistic_features=linguistic_features,
        feature_importance=feature_importance,
        linguistic_markers=linguistic_markers
    )


def demonstrate_enhanced_inference():
    """Demonstrate enhanced inference pipeline capabilities."""
    
    print("=" * 80)
    print("ENHANCED INFERENCE PIPELINE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Test samples with different risk levels
    test_samples = [
        {
            'text': "நான் மகிழ்ச்சியாக இருக்கிறேன் நல்ல நாள்",
            'risk_level': 'low',
            'description': 'Low-risk sample (positive sentiment)'
        },
        {
            'text': "நான் சோர்வாக இருக்கிறேன் தூக்கம் வரவில்லை மருத்துவர்",
            'risk_level': 'moderate',
            'description': 'Moderate-risk sample (some symptoms)'
        },
        {
            'text': "நான் சோகமாக இருக்கிறேன் தனிமை மருந்து எடுத்துக்கொள்கிறேன்",
            'risk_level': 'high',
            'description': 'High-risk sample (strong indicators)'
        }
    ]
    
    print("1. Enhanced Inference Pipeline Overview...")
    config = get_config("combined")
    config.device = "cpu"
    
    # Create pipeline without loading heavy models
    pipeline = EnhancedDepressionInferencePipeline(
        config, 
        enhanced_model=None,  # Don't load heavy model
        linguistic_analyzer=None,  # Don't load heavy analyzer
        inference_mode='enhanced',
        device='cpu'
    )
    
    print(f"   ✓ Pipeline mode: {pipeline.inference_mode}")
    print(f"   ✓ Device: {pipeline.device}")
    print(f"   ✓ Enhanced model: {'Loaded' if pipeline.enhanced_model else 'Mock mode'}")
    print(f"   ✓ Linguistic analyzer: {'Loaded' if pipeline.linguistic_analyzer else 'Mock mode'}")
    print()
    
    # Process each sample
    for i, sample in enumerate(test_samples, 1):
        print(f"{i}. Processing {sample['description']}...")
        print(f"   Input text: {sample['text']}")
        print(f"   Expected risk level: {sample['risk_level'].upper()}")
        print()
        
        # Create enhanced prediction result
        result = create_enhanced_prediction_result(sample['text'], sample['risk_level'])
        
        # Show prediction
        prediction_text = "DEPRESSED" if result.predicted_label == 1 else "NON-DEPRESSED"
        print(f"   Prediction: {prediction_text}")
        print(f"   Confidence: {result.probability:.3f}")
        print()
        
        # Show modality breakdown
        print("   Modality Analysis:")
        print(f"     Audio (SSL): {result.ssl_probability:.3f}")
        print(f"     Audio (ECAPA): {result.ecapa_probability:.3f}")
        print(f"     Text: {result.text_probability:.3f}")
        print(f"     Enhanced (Multimodal): {result.enhanced_probability:.3f}")
        print()
        
        # Show feature importance
        print("   Feature Importance:")
        if result.feature_importance:
            for feature, importance in result.feature_importance.items():
                print(f"     {feature.replace('_', ' ').title()}: {importance:.3f}")
        print()
        
        # Show linguistic markers
        print("   Linguistic Markers:")
        if result.linguistic_markers:
            markers = result.linguistic_markers
            print(f"     First-person usage: {markers['first_person_pronoun_ratio']:.3f}")
            print(f"     Negative language: {markers['negative_adjective_ratio']:.3f}")
            print(f"     Past-tense focus: {markers['past_tense_verb_ratio']:.3f}")
            print(f"     Self-references: {markers['self_reference_count']}")
            print(f"     Medical terms: {markers['medical_entity_count']}")
            print(f"     Sentence complexity: {markers['sentence_complexity']:.2f}")
            print(f"     Linguistic risk score: {markers['linguistic_risk_score']:.3f}")
            
            # Risk interpretation
            risk_score = markers['linguistic_risk_score']
            if risk_score > 0.7:
                risk_level = "HIGH"
            elif risk_score > 0.4:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            print(f"     Computed risk level: {risk_level}")
        print()
        
        # Generate clinical report
        print("   Clinical Report Generation...")
        report = pipeline.generate_clinical_report(result)
        
        # Save report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(report)
            report_path = f.name
        
        print(f"     ✓ Report saved: {report_path}")
        
        # Show report preview
        print("     Report preview:")
        lines = report.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            print(f"       {line}")
        if len(lines) > 10:
            print(f"       ... ({len(lines) - 10} more lines)")
        
        print("   " + "-" * 60)
        print()
        
        # Cleanup
        try:
            os.unlink(report_path)
        except:
            pass
    
    # Summary
    print("SUMMARY OF ENHANCED INFERENCE CAPABILITIES:")
    print("=" * 50)
    print("✓ Multimodal fusion (audio + text + linguistic features)")
    print("✓ Traditional NLP task integration:")
    print("  - Part-of-Speech (POS) tagging")
    print("  - Named Entity Recognition (NER)")
    print("  - Dependency parsing")
    print("✓ Clinical linguistic marker extraction:")
    print("  - First-person pronoun usage patterns")
    print("  - Negative language indicators")
    print("  - Temporal focus (past/present tense)")
    print("  - Self-reference frequency")
    print("  - Medical terminology usage")
    print("  - Syntactic complexity measures")
    print("✓ Feature importance analysis for interpretability")
    print("✓ Automated clinical report generation")
    print("✓ Risk level assessment and interpretation")
    print("✓ Support for multiple inference modes (standard/enhanced)")
    print()
    
    print("TECHNICAL ACHIEVEMENTS:")
    print("=" * 30)
    print("• Successfully integrated 3 traditional NLP tasks")
    print("• Created 128-dimensional linguistic feature vectors")
    print("• Implemented clinical interpretation framework")
    print("• Built comprehensive evaluation and reporting system")
    print("• Achieved seamless multimodal integration")
    print("• Provided interpretable AI for clinical applications")
    print()
    
    print("=" * 80)
    print("ENHANCED INFERENCE PIPELINE DEMONSTRATION COMPLETE")
    print("=" * 80)


def demonstrate_feature_importance():
    """Demonstrate feature importance analysis in detail."""
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create samples with different feature importance patterns
    samples = [
        {
            'name': 'Audio-dominant',
            'audio_imp': 0.6, 'text_imp': 0.25, 'ling_imp': 0.15,
            'description': 'Strong vocal indicators (prosody, speech patterns)'
        },
        {
            'name': 'Text-dominant', 
            'audio_imp': 0.2, 'text_imp': 0.6, 'ling_imp': 0.2,
            'description': 'Strong textual content indicators'
        },
        {
            'name': 'Linguistic-dominant',
            'audio_imp': 0.25, 'text_imp': 0.25, 'ling_imp': 0.5,
            'description': 'Strong traditional NLP indicators'
        },
        {
            'name': 'Balanced',
            'audio_imp': 0.33, 'text_imp': 0.33, 'ling_imp': 0.34,
            'description': 'Equal contribution from all modalities'
        }
    ]
    
    for sample in samples:
        print(f"\n{sample['name']} Sample:")
        print(f"Description: {sample['description']}")
        print("Feature Importance:")
        print(f"  Audio: {sample['audio_imp']:.2f}")
        print(f"  Text: {sample['text_imp']:.2f}")
        print(f"  Linguistic: {sample['ling_imp']:.2f}")
        
        # Interpretation
        max_feature = max(
            ('Audio', sample['audio_imp']),
            ('Text', sample['text_imp']),
            ('Linguistic', sample['ling_imp'])
        )
        
        print(f"Primary indicator: {max_feature[0]} ({max_feature[1]:.2f})")
        
        # Clinical interpretation
        if sample['ling_imp'] > 0.4:
            print("Clinical note: Strong linguistic patterns suggest systematic language changes")
        elif sample['audio_imp'] > 0.5:
            print("Clinical note: Vocal characteristics are primary indicators")
        elif sample['text_imp'] > 0.5:
            print("Clinical note: Content analysis reveals key depression markers")
        else:
            print("Clinical note: Multiple modalities contribute to assessment")
        
        print("-" * 40)


if __name__ == "__main__":
    try:
        # Main demonstration
        demonstrate_enhanced_inference()
        
        # Feature importance demonstration
        demonstrate_feature_importance()
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()