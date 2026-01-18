#!/usr/bin/env python3
"""
Enhanced Inference Pipeline Demonstration

This script demonstrates the enhanced inference pipeline with linguistic features
for depression detection. It shows how traditional NLP tasks (POS tagging, NER,
dependency parsing) are integrated with multimodal analysis.

Usage:
    python demo_enhanced_inference.py

Features demonstrated:
- Enhanced multimodal inference with linguistic features
- Linguistic marker extraction for clinical interpretation
- Feature importance analysis for interpretability
- Clinical report generation
- Visualization capabilities (if matplotlib available)

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
    LinguisticMarkers,
    create_enhanced_inference_pipeline
)
from linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig, LinguisticAnalysis
from pos_tagger import POSResult
from dependency_parser import DependencyTree
from ner_system import Entity


def create_demo_linguistic_analysis() -> LinguisticAnalysis:
    """Create a demo linguistic analysis for demonstration."""
    
    # Demo text in Tamil (depression-related content)
    demo_text = "நான் மிகவும் சோர்வாக இருக்கிறேன். எனக்கு தூக்கம் வரவில்லை. மருத்துவரை பார்க்க வேண்டும்."
    tokens = ["நான்", "மிகவும்", "சோர்வாக", "இருக்கிறேன்", "எனக்கு", "தூக்கம்", "வரவில்லை", "மருத்துவரை", "பார்க்க", "வேண்டும்"]
    
    # Mock POS result
    pos_result = POSResult(
        tokens=tokens,
        tags=["PRON", "ADV", "ADJ", "VERB", "PRON", "NOUN", "VERB", "NOUN", "VERB", "AUX"],
        confidence_scores=[0.95, 0.88, 0.92, 0.89, 0.94, 0.87, 0.91, 0.85, 0.88, 0.93],
        pos_distribution={
            "PRON": 2, "ADV": 1, "ADJ": 1, "VERB": 3, 
            "NOUN": 2, "AUX": 1
        },
        language="ta"
    )
    
    # Mock entities (medical and self-reference)
    entities = [
        Entity(text="நான்", label="PERSON", start=0, end=4, confidence=0.95),
        Entity(text="எனக்கு", label="PERSON", start=5, end=10, confidence=0.92),
        Entity(text="மருத்துவரை", label="MEDICAL", start=11, end=20, confidence=0.88)
    ]
    
    # Mock dependency tree
    dependency_tree = DependencyTree(
        tokens=tokens,
        heads=[3, 2, 3, -1, 6, 6, 3, 8, 9, 6],  # Root is "இருக்கிறேன்"
        relations=["nsubj", "advmod", "advmod", "root", "nsubj", "obj", "conj", "obj", "xcomp", "aux"],
        confidence_scores=[0.9, 0.85, 0.88, 0.95, 0.87, 0.82, 0.89, 0.84, 0.86, 0.91]
    )
    
    # Mock linguistic features (depression indicators)
    linguistic_features = {
        # POS-based features
        'first_person_pronoun_ratio': 0.2,  # High first-person usage
        'negative_adjective_ratio': 0.1,    # Some negative language
        'past_tense_verb_ratio': 0.0,       # Present tense focus
        'function_word_ratio': 0.3,
        'pos_diversity': 0.7,
        
        # NER-based features
        'self_reference_entities': 2,       # Multiple self-references
        'medical_entity_count': 1,          # Medical terminology present
        'person_entity_ratio': 0.3,
        
        # Syntactic features
        'syntactic_complexity': 2.8,       # Moderate complexity
        'avg_dependency_distance': 1.6,    # Relatively short dependencies
        'max_dependency_depth': 3,
        'coordination_ratio': 0.1,
        
        # Composite features
        'linguistic_diversity': 0.65,
        'semantic_coherence': 0.75
    }
    
    return LinguisticAnalysis(
        text=demo_text,
        tokens=tokens,
        pos_result=pos_result,
        entities=entities,
        dependency_tree=dependency_tree,
        linguistic_features=linguistic_features,
        language="ta"
    )


def create_demo_prediction_result() -> EnhancedPredictionResult:
    """Create a demo enhanced prediction result."""
    
    # Create linguistic analysis
    linguistic_analysis = create_demo_linguistic_analysis()
    
    # Extract linguistic features as numpy array
    linguistic_features = np.array([
        0.2, 0.1, 0.0, 0.3, 0.7,  # POS features
        2.0, 1.0, 0.3,             # NER features  
        2.8, 1.6, 3.0, 0.1,       # Syntactic features
        0.65, 0.75                 # Composite features
    ] + [0.0] * 114)  # Pad to 128 dimensions
    
    # Feature importance (linguistic features show high contribution)
    feature_importance = {
        'audio_importance': 0.35,
        'text_importance': 0.25,
        'linguistic_importance': 0.40  # Linguistic features are most important
    }
    
    # Linguistic markers for clinical interpretation
    linguistic_markers = {
        'first_person_pronoun_ratio': 0.2,
        'negative_adjective_ratio': 0.1,
        'past_tense_verb_ratio': 0.0,
        'self_reference_count': 2,
        'medical_entity_count': 1,
        'sentence_complexity': 2.8,
        'dependency_distance': 1.6,
        'linguistic_risk_score': 0.68  # Moderate-high risk
    }
    
    return EnhancedPredictionResult(
        file_id="demo_audio_001",
        speaker_id="DEMO_SPEAKER_001",
        predicted_label=1,  # Predicted as depressed
        probability=0.78,   # High confidence
        ssl_probability=0.72,
        ecapa_probability=0.75,
        text_probability=0.65,
        enhanced_probability=0.78,  # Enhanced model shows highest confidence
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
    
    # Create configuration
    print("1. Initializing Enhanced Inference Pipeline...")
    config = get_config("combined")
    config.device = "cpu"  # Use CPU for demo
    
    # Create enhanced pipeline
    pipeline = create_enhanced_inference_pipeline(
        config, 
        inference_mode='enhanced',
        device='cpu'
    )
    
    print(f"   ✓ Pipeline mode: {pipeline.inference_mode}")
    print(f"   ✓ Device: {pipeline.device}")
    print(f"   ✓ Linguistic analyzer: {'Available' if pipeline.linguistic_analyzer else 'Not available'}")
    print()
    
    # Demonstrate linguistic analysis
    print("2. Linguistic Analysis Demonstration...")
    demo_text = "நான் மிகவும் சோர்வாக இருக்கிறேன். எனக்கு தூக்கம் வரவில்லை."
    print(f"   Input text: {demo_text}")
    
    try:
        if pipeline.linguistic_analyzer:
            analysis = pipeline.linguistic_analyzer.analyze_text(demo_text, "ta")
            print(f"   ✓ Tokens: {len(analysis.tokens)}")
            print(f"   ✓ POS tags: {analysis.pos_result.tags}")
            print(f"   ✓ Entities: {[(e.text, e.label) for e in analysis.entities]}")
            print(f"   ✓ Linguistic features: {len(analysis.linguistic_features)} dimensions")
        else:
            print("   ⚠ Linguistic analyzer not available (using mock data)")
    except Exception as e:
        print(f"   ⚠ Linguistic analysis failed: {e}")
        print("   ℹ Using mock data for demonstration")
    
    print()
    
    # Demonstrate enhanced prediction result
    print("3. Enhanced Prediction Result...")
    result = create_demo_prediction_result()
    
    print(f"   File ID: {result.file_id}")
    print(f"   Speaker ID: {result.speaker_id}")
    print(f"   Prediction: {'DEPRESSED' if result.predicted_label == 1 else 'NON-DEPRESSED'}")
    print(f"   Confidence: {result.probability:.3f}")
    print()
    
    print("   Modality Breakdown:")
    print(f"     Audio (SSL): {result.ssl_probability:.3f}")
    print(f"     Audio (ECAPA): {result.ecapa_probability:.3f}")
    print(f"     Text: {result.text_probability:.3f}")
    print(f"     Enhanced (Multimodal): {result.enhanced_probability:.3f}")
    print()
    
    # Demonstrate feature importance
    print("4. Feature Importance Analysis...")
    if result.feature_importance:
        for feature, importance in result.feature_importance.items():
            print(f"   {feature.replace('_', ' ').title()}: {importance:.3f}")
    print()
    
    # Demonstrate linguistic markers
    print("5. Linguistic Markers for Clinical Interpretation...")
    if result.linguistic_markers:
        markers = result.linguistic_markers
        print(f"   First-person pronoun usage: {markers['first_person_pronoun_ratio']:.3f}")
        print(f"   Negative language indicators: {markers['negative_adjective_ratio']:.3f}")
        print(f"   Past-tense focus: {markers['past_tense_verb_ratio']:.3f}")
        print(f"   Self-references: {markers['self_reference_count']}")
        print(f"   Medical terminology: {markers['medical_entity_count']}")
        print(f"   Sentence complexity: {markers['sentence_complexity']:.2f}")
        print(f"   Linguistic risk score: {markers['linguistic_risk_score']:.3f}")
        
        # Risk interpretation
        risk_score = markers['linguistic_risk_score']
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        print(f"   Risk level: {risk_level}")
    print()
    
    # Demonstrate clinical report generation
    print("6. Clinical Report Generation...")
    report = pipeline.generate_clinical_report(result)
    
    # Save report to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(report)
        report_path = f.name
    
    print(f"   ✓ Clinical report generated: {report_path}")
    print("   Report preview:")
    print("   " + "-" * 50)
    
    # Show first few lines of report
    lines = report.split('\n')
    for line in lines[:15]:  # Show first 15 lines
        print(f"   {line}")
    
    if len(lines) > 15:
        print(f"   ... ({len(lines) - 15} more lines)")
    
    print("   " + "-" * 50)
    print()
    
    # Demonstrate visualization (if available)
    print("7. Feature Importance Visualization...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            viz_path = f.name
        
        saved_path = pipeline.visualize_feature_importance(result, viz_path)
        
        if saved_path:
            print(f"   ✓ Visualization saved: {saved_path}")
        else:
            print("   ⚠ Visualization not available (matplotlib not installed)")
    except Exception as e:
        print(f"   ⚠ Visualization failed: {e}")
    
    print()
    
    # Summary
    print("8. Summary...")
    print("   Enhanced inference pipeline successfully demonstrated:")
    print("   ✓ Multimodal fusion (audio + text + linguistic features)")
    print("   ✓ Traditional NLP task integration (POS, NER, dependency parsing)")
    print("   ✓ Clinical linguistic marker extraction")
    print("   ✓ Feature importance analysis for interpretability")
    print("   ✓ Automated clinical report generation")
    print("   ✓ Visualization capabilities")
    print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    # Cleanup
    try:
        os.unlink(report_path)
        if 'viz_path' in locals():
            os.unlink(viz_path)
    except:
        pass


def demonstrate_linguistic_markers():
    """Demonstrate linguistic marker extraction in detail."""
    
    print("\n" + "=" * 60)
    print("LINGUISTIC MARKERS DETAILED DEMONSTRATION")
    print("=" * 60)
    
    # Create pipeline
    config = get_config("combined")
    config.device = "cpu"
    
    pipeline = EnhancedDepressionInferencePipeline(
        config, inference_mode='enhanced', device='cpu'
    )
    
    # Test different text samples
    test_samples = [
        {
            'text': "நான் மகிழ்ச்சியாக இருக்கிறேன். நல்ல நாள்.",
            'expected_risk': 'LOW',
            'description': 'Positive sentiment text'
        },
        {
            'text': "நான் மிகவும் சோர்வாக இருக்கிறேன். எனக்கு தூக்கம் வரவில்லை. மருத்துவரை பார்க்க வேண்டும்.",
            'expected_risk': 'MODERATE-HIGH',
            'description': 'Depression-related symptoms'
        },
        {
            'text': "நான் எப்போதும் சோகமாக இருக்கிறேன். நான் தனிமையில் இருக்கிறேன். மருந்து எடுத்துக்கொள்கிறேன்.",
            'expected_risk': 'HIGH',
            'description': 'Strong depression indicators'
        }
    ]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nSample {i}: {sample['description']}")
        print(f"Text: {sample['text']}")
        
        # Create mock analysis for this sample
        analysis = create_demo_linguistic_analysis()
        analysis.text = sample['text']
        
        # Adjust linguistic features based on sample
        if 'மகிழ்ச்சி' in sample['text']:
            # Positive sample
            analysis.linguistic_features.update({
                'first_person_pronoun_ratio': 0.1,
                'negative_adjective_ratio': 0.0,
                'self_reference_entities': 1,
                'medical_entity_count': 0
            })
        elif 'சோகம' in sample['text'] or 'தனிமை' in sample['text']:
            # High-risk sample
            analysis.linguistic_features.update({
                'first_person_pronoun_ratio': 0.4,
                'negative_adjective_ratio': 0.3,
                'self_reference_entities': 3,
                'medical_entity_count': 2
            })
        
        # Extract markers
        markers = pipeline.extract_linguistic_markers(analysis)
        
        print(f"Linguistic Risk Score: {markers.linguistic_risk_score:.3f}")
        print(f"Expected Risk Level: {sample['expected_risk']}")
        
        # Show key markers
        print("Key Markers:")
        print(f"  - First-person usage: {markers.first_person_pronoun_ratio:.3f}")
        print(f"  - Negative language: {markers.negative_adjective_ratio:.3f}")
        print(f"  - Self-references: {markers.self_reference_count}")
        print(f"  - Medical terms: {markers.medical_entity_count}")
        
        print("-" * 40)


if __name__ == "__main__":
    try:
        # Main demonstration
        demonstrate_enhanced_inference()
        
        # Detailed linguistic markers demonstration
        demonstrate_linguistic_markers()
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()