"""
Pydantic models for NLP analysis results and API data structures.

This module defines the data models used throughout the NLP web interface
for consistent data validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class EntityLabel(str, Enum):
    """Named entity labels."""
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    MEDICAL = "MEDICAL"
    MISC = "MISC"
    O = "O"


class POSTag(BaseModel):
    """Part-of-speech tag with metadata."""
    token: str = Field(..., description="The token text")
    tag: str = Field(..., description="POS tag (Universal Dependencies)")
    position: int = Field(..., ge=0, description="Token position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "token": "running",
                "tag": "VERB",
                "position": 2,
                "confidence": 0.95
            }
        }


class NamedEntity(BaseModel):
    """Named entity with location and metadata."""
    text: str = Field(..., description="Entity text")
    label: EntityLabel = Field(..., description="Entity type")
    start: int = Field(..., ge=0, description="Start character position")
    end: int = Field(..., ge=0, description="End character position")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    @validator('end')
    def end_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('end must be greater than start')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "John Smith",
                "label": "PERSON",
                "start": 0,
                "end": 10,
                "confidence": 0.92
            }
        }


class DependencyRelation(BaseModel):
    """Dependency relation between tokens."""
    head: int = Field(..., ge=-1, description="Head token index (-1 for root)")
    dependent: int = Field(..., ge=0, description="Dependent token index")
    relation: str = Field(..., description="Dependency relation label")
    head_text: str = Field(..., description="Head token text")
    dependent_text: str = Field(..., description="Dependent token text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "head": 1,
                "dependent": 2,
                "relation": "nsubj",
                "head_text": "runs",
                "dependent_text": "John",
                "confidence": 0.88
            }
        }


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    overall_sentiment: str = Field(..., description="Overall sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    scores: Dict[str, float] = Field(..., description="Sentiment scores by category")
    
    @validator('scores')
    def validate_scores(cls, v):
        required_keys = {'positive', 'negative', 'neutral'}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f'scores must contain keys: {required_keys}')
        for score in v.values():
            if not 0.0 <= score <= 1.0:
                raise ValueError('all scores must be between 0.0 and 1.0')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "overall_sentiment": "positive",
                "confidence": 0.85,
                "scores": {
                    "positive": 0.7,
                    "negative": 0.1,
                    "neutral": 0.2
                }
            }
        }


class LinguisticFeatures(BaseModel):
    """Linguistic features extracted from text."""
    sentence_count: int = Field(..., ge=0, description="Number of sentences")
    word_count: int = Field(..., ge=0, description="Number of words")
    avg_sentence_length: float = Field(..., ge=0.0, description="Average sentence length")
    lexical_diversity: float = Field(..., ge=0.0, le=1.0, description="Type-token ratio")
    function_word_ratio: float = Field(..., ge=0.0, le=1.0, description="Function word ratio")
    pos_distribution: Dict[str, float] = Field(..., description="POS tag distribution")
    
    class Config:
        schema_extra = {
            "example": {
                "sentence_count": 3,
                "word_count": 25,
                "avg_sentence_length": 8.33,
                "lexical_diversity": 0.76,
                "function_word_ratio": 0.32,
                "pos_distribution": {
                    "NOUN": 0.28,
                    "VERB": 0.20,
                    "ADJ": 0.12
                }
            }
        }


class AnalysisResult(BaseModel):
    """Complete NLP analysis result for a text segment."""
    text: str = Field(..., description="Original text")
    tokens: List[str] = Field(..., description="Tokenized text")
    pos_tags: List[POSTag] = Field(..., description="POS tagging results")
    named_entities: List[NamedEntity] = Field(..., description="Named entities")
    dependencies: List[DependencyRelation] = Field(..., description="Dependency relations")
    sentiment: SentimentAnalysis = Field(..., description="Sentiment analysis")
    linguistic_features: LinguisticFeatures = Field(..., description="Linguistic features")
    language: str = Field(..., description="Detected language code")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    @validator('pos_tags')
    def validate_pos_tags_length(cls, v, values):
        if 'tokens' in values and len(v) != len(values['tokens']):
            raise ValueError('pos_tags length must match tokens length')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "John runs quickly.",
                "tokens": ["John", "runs", "quickly", "."],
                "pos_tags": [
                    {"token": "John", "tag": "PROPN", "position": 0, "confidence": 0.95}
                ],
                "named_entities": [
                    {"text": "John", "label": "PERSON", "start": 0, "end": 4, "confidence": 0.92}
                ],
                "dependencies": [
                    {"head": 1, "dependent": 0, "relation": "nsubj", "head_text": "runs", "dependent_text": "John", "confidence": 0.88}
                ],
                "sentiment": {
                    "overall_sentiment": "neutral",
                    "confidence": 0.75,
                    "scores": {"positive": 0.3, "negative": 0.2, "neutral": 0.5}
                },
                "linguistic_features": {
                    "sentence_count": 1,
                    "word_count": 4,
                    "avg_sentence_length": 4.0,
                    "lexical_diversity": 1.0,
                    "function_word_ratio": 0.0,
                    "pos_distribution": {"PROPN": 0.25, "VERB": 0.25, "ADV": 0.25, "PUNCT": 0.25}
                },
                "language": "en",
                "processing_time": 0.15,
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class SessionStatus(str, Enum):
    """Session status values."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class SessionData(BaseModel):
    """Session data for managing user interactions."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    audio_file_path: Optional[str] = Field(None, description="Path to uploaded audio file")
    transcription: Optional[str] = Field(None, description="Audio transcription")
    analysis_results: List[AnalysisResult] = Field(default_factory=list, description="Analysis results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "sess_123456789",
                "user_id": "user_001",
                "created_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:05:00",
                "status": "active",
                "audio_file_path": "/uploads/audio_123.wav",
                "transcription": "Hello, this is a test transcription.",
                "analysis_results": [],
                "metadata": {"client_info": "web_browser"}
            }
        }


class AudioQuality(BaseModel):
    """Audio quality metrics."""
    signal_level: float = Field(..., ge=0.0, le=1.0, description="Signal level (0-1)")
    noise_level: float = Field(..., ge=0.0, le=1.0, description="Noise level (0-1)")
    snr: float = Field(..., description="Signal-to-noise ratio in dB")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    warnings: List[str] = Field(default_factory=list, description="Quality warnings")
    
    class Config:
        schema_extra = {
            "example": {
                "signal_level": 0.75,
                "noise_level": 0.15,
                "snr": 15.2,
                "quality_score": 0.85,
                "warnings": ["Background noise detected"]
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Model precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Model recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    training_date: datetime = Field(..., description="Model training date")
    evaluation_data: Dict[str, Any] = Field(..., description="Additional evaluation metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "pos_tagger_v1",
                "version": "1.0.0",
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.91,
                "f1_score": 0.90,
                "training_date": "2024-01-01T00:00:00",
                "evaluation_data": {
                    "test_samples": 1000,
                    "confusion_matrix": [[850, 50], [100, 900]]
                }
            }
        }


# API Request/Response Models

class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    language: Optional[str] = Field(None, description="Language hint (optional)")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_features: bool = Field(default=True, description="Include linguistic features")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text for analysis.",
                "language": "en",
                "include_sentiment": True,
                "include_features": True
            }
        }


class FileUploadResponse(BaseModel):
    """Response model for file uploads."""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_type: str = Field(..., description="File MIME type")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    processing_status: str = Field(default="queued", description="Processing status")
    
    class Config:
        schema_extra = {
            "example": {
                "file_id": "file_123456789",
                "filename": "sample_audio.wav",
                "file_size": 1024000,
                "file_type": "audio/wav",
                "upload_time": "2024-01-01T12:00:00",
                "processing_status": "queued"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid input data provided",
                "details": {"field": "text", "issue": "Text too long"},
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(..., description="Message type")
    session_id: str = Field(..., description="Session identifier")
    data: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "analysis_result",
                "session_id": "sess_123456789",
                "data": {"text": "Hello", "sentiment": "positive"},
                "timestamp": "2024-01-01T12:00:00"
            }
        }