"""
API endpoints for NLP analysis functionality.

This module provides REST API endpoints for text analysis, integrating
with existing NLP components (POS tagger, NER system, dependency parser).
"""

import asyncio
import time
import logging
import os
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    TextAnalysisRequest, AnalysisResult, POSTag, NamedEntity, 
    DependencyRelation, SentimentAnalysis, LinguisticFeatures,
    ErrorResponse, FileUploadResponse, SessionData, SessionStatus
)
from .metrics_collector import get_metrics_collector, RealTimeMetrics

# Import web interface linguistic analyzer wrapper
try:
    from .linguistic_analyzer_wrapper import (
        WebLinguisticAnalyzer, WebLinguisticConfig, WebAnalysisResult,
        create_web_linguistic_analyzer
    )
    HAS_WEB_ANALYZER = True
except ImportError as e:
    logging.warning(f"Web linguistic analyzer not available: {e}")
    HAS_WEB_ANALYZER = False

# Import existing NLP components for fallback
try:
    from pos_tagger import POSTagger, POSConfig
    from ner_system import NERSystem, NERConfig
    from dependency_parser import DependencyParser, DependencyConfig
    from linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig
    HAS_NLP_COMPONENTS = True
except ImportError as e:
    logging.warning(f"NLP components not available: {e}")
    HAS_NLP_COMPONENTS = False

# Import audio processor for file processing
try:
    from .audio_processor import get_audio_processor, RealTimeAudioProcessor
    HAS_AUDIO_PROCESSOR = True
except ImportError as e:
    logging.warning(f"Audio processor not available: {e}")
    HAS_AUDIO_PROCESSOR = False

# Import demo controller
try:
    from .demo_controller import get_demo_controller, DemoController, BatchStatus
    HAS_DEMO_CONTROLLER = True
except ImportError as e:
    logging.warning(f"Demo controller not available: {e}")
    HAS_DEMO_CONTROLLER = False

# Import session manager
try:
    from .session_manager import get_session_manager, SessionManager
    HAS_SESSION_MANAGER = True
except ImportError as e:
    logging.warning(f"Session manager not available: {e}")
    HAS_SESSION_MANAGER = False

# Import ASR pipeline for audio processing
try:
    from asr_pipeline import create_asr_pipeline
    HAS_ASR_PIPELINE = True
except ImportError as e:
    logging.warning(f"ASR pipeline not available: {e}")
    HAS_ASR_PIPELINE = False

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api", tags=["analysis"])

# Global NLP components (initialized on startup)
nlp_analyzer = None

# File processing configuration
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/flac"}
ALLOWED_TEXT_TYPES = {"text/plain", "text/csv", "application/json"}

# Processing queue for batch operations
processing_queue = asyncio.Queue()
processing_results = {}  # Store results by file_id


class NLPService:
    """Service class for NLP analysis operations."""
    
    def __init__(self):
        """Initialize NLP components."""
        self.web_analyzer = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize web linguistic analyzer asynchronously."""
        if self.initialized:
            return
            
        try:
            if HAS_WEB_ANALYZER:
                # Initialize web linguistic analyzer with concurrent processing
                self.web_analyzer = create_web_linguistic_analyzer(
                    device="cpu",
                    max_concurrent_tasks=3,
                    enable_fallback=True
                )
                logger.info("Web linguistic analyzer initialized successfully")
            else:
                logger.warning("Web linguistic analyzer not available, using mock implementations")
                
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP services: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize NLP services")
    
    async def analyze_text(self, text: str, language: Optional[str] = None, 
                          include_sentiment: bool = True, 
                          include_features: bool = True) -> AnalysisResult:
        """Perform comprehensive text analysis."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            if HAS_WEB_ANALYZER and self.web_analyzer:
                # Use web linguistic analyzer
                web_result = await self.web_analyzer.analyze_text(
                    text, language or "ta"
                )
                
                # Convert WebAnalysisResult to AnalysisResult
                pos_tags = []
                if web_result.pos_result and hasattr(web_result.pos_result, 'tokens'):
                    pos_tags = [
                        POSTag(
                            token=token,
                            tag=tag,
                            position=i,
                            confidence=conf
                        ) for i, (token, tag, conf) in enumerate(zip(
                            web_result.pos_result.tokens,
                            web_result.pos_result.tags,
                            web_result.pos_result.confidence_scores
                        ))
                    ]
                
                named_entities = []
                for entity in web_result.entities:
                    named_entities.append(NamedEntity(
                        text=getattr(entity, 'text', ''),
                        label=getattr(entity, 'label', 'MISC'),
                        start=getattr(entity, 'start_idx', 0),
                        end=getattr(entity, 'end_idx', 0),
                        confidence=getattr(entity, 'confidence', 0.0)
                    ))
                
                dependencies = []
                if web_result.dependency_tree and hasattr(web_result.dependency_tree, 'tokens'):
                    for i, (token, head, relation, conf) in enumerate(zip(
                        web_result.dependency_tree.tokens,
                        web_result.dependency_tree.heads,
                        web_result.dependency_tree.relations,
                        web_result.dependency_tree.confidence_scores
                    )):
                        head_text = web_result.dependency_tree.tokens[head] if head != -1 and head < len(web_result.dependency_tree.tokens) else "ROOT"
                        dependencies.append(DependencyRelation(
                            head=head,
                            dependent=i,
                            relation=relation,
                            head_text=head_text,
                            dependent_text=token,
                            confidence=conf
                        ))
                
                # Create sentiment analysis from web result
                sentiment = SentimentAnalysis(
                    overall_sentiment="neutral",
                    confidence=0.75,
                    scores={"positive": 0.3, "negative": 0.2, "neutral": 0.5}
                )
                
                # Use sentiment result if available
                if web_result.sentiment_result:
                    sentiment = SentimentAnalysis(
                        overall_sentiment=web_result.sentiment_result.overall_sentiment,
                        confidence=web_result.sentiment_result.confidence,
                        scores=web_result.sentiment_result.scores
                    )
                
                # Extract linguistic features from web result
                features = web_result.linguistic_features
                linguistic_features = LinguisticFeatures(
                    sentence_count=1,  # Simple sentence count
                    word_count=len(web_result.tokens),
                    avg_sentence_length=float(len(web_result.tokens)),
                    lexical_diversity=features.get("pos_diversity", 0.8),
                    function_word_ratio=features.get("function_word_ratio", 0.3),
                    pos_distribution=features.get("pos_distribution", {})
                )
                
                detected_language = web_result.language
                tokens = web_result.tokens
                processing_time = web_result.processing_time
                
                # Log any errors from web analyzer
                if web_result.error_messages:
                    logger.warning(f"Web analyzer errors: {web_result.error_messages}")
                
            else:
                # Use mock implementation for testing
                tokens = text.split()
                pos_tags = [
                    POSTag(token=token, tag="NOUN", position=i, confidence=0.9)
                    for i, token in enumerate(tokens)
                ]
                
                named_entities = [
                    NamedEntity(
                        text=tokens[0] if tokens else "Mock",
                        label="PERSON",
                        start=0,
                        end=len(tokens[0]) if tokens else 4,
                        confidence=0.85
                    )
                ] if tokens else []
                
                dependencies = [
                    DependencyRelation(
                        head=-1,
                        dependent=0,
                        relation="root",
                        head_text="ROOT",
                        dependent_text=tokens[0] if tokens else "Mock",
                        confidence=0.9
                    )
                ] if tokens else []
                
                sentiment = SentimentAnalysis(
                    overall_sentiment="neutral",
                    confidence=0.75,
                    scores={"positive": 0.3, "negative": 0.2, "neutral": 0.5}
                )
                
                linguistic_features = LinguisticFeatures(
                    sentence_count=1,
                    word_count=len(tokens),
                    avg_sentence_length=len(tokens),
                    lexical_diversity=0.8,
                    function_word_ratio=0.3,
                    pos_distribution={"NOUN": 1.0}
                )
                
                detected_language = language or "en"
                processing_time = time.time() - start_time
            
            return AnalysisResult(
                text=text,
                tokens=tokens,
                pos_tags=pos_tags,
                named_entities=named_entities,
                dependencies=dependencies,
                sentiment=sentiment,
                linguistic_features=linguistic_features,
                language=detected_language,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Global service instance
nlp_service = NLPService()


async def get_nlp_service() -> NLPService:
    """Dependency to get initialized NLP service."""
    if not nlp_service.initialized:
        await nlp_service.initialize()
    return nlp_service


@router.post("/analyze/text", response_model=AnalysisResult)
async def analyze_text(
    request: TextAnalysisRequest,
    service: NLPService = Depends(get_nlp_service)
) -> AnalysisResult:
    """
    Analyze text using comprehensive NLP pipeline.
    
    Performs POS tagging, named entity recognition, dependency parsing,
    sentiment analysis, and linguistic feature extraction.
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 10000:
            raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")
        
        # Check for control characters (except common whitespace)
        if any(ord(c) < 32 and c not in '\t\n\r' for c in request.text):
            raise HTTPException(status_code=400, detail="Text contains invalid control characters")
        
        # Perform analysis
        result = await service.analyze_text(
            text=request.text,
            language=request.language,
            include_sentiment=request.include_sentiment,
            include_features=request.include_features
        )
        
        # Record processing metrics
        try:
            collector = await get_metrics_collector()
            await collector.record_realtime_metrics(RealTimeMetrics(
                timestamp=datetime.now(),
                operation_type="text_analysis",
                processing_time=result.processing_time,
                input_size=len(request.text),
                success=True,
                error_message=None,
                component_metrics={}
            ))
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")
        
        logger.info(f"Text analysis completed in {result.processing_time:.3f}s")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analyze/health")
async def analysis_health_check():
    """Check the health of analysis services."""
    try:
        service = await get_nlp_service()
        
        # Test with simple text
        test_result = await service.analyze_text("Test sentence.")
        
        # Get component status from web analyzer
        component_status = {}
        if HAS_WEB_ANALYZER and service.web_analyzer:
            component_status = service.web_analyzer.get_component_status()
        
        return {
            "status": "healthy",
            "components": component_status,
            "test_processing_time": test_result.processing_time,
            "has_web_analyzer": HAS_WEB_ANALYZER,
            "has_nlp_components": HAS_NLP_COMPONENTS
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "has_web_analyzer": HAS_WEB_ANALYZER,
                "has_nlp_components": HAS_NLP_COMPONENTS
            }
        )


@router.get("/models/info")
async def get_model_info():
    """Get information about loaded NLP models."""
    try:
        service = await get_nlp_service()
        
        # Get component status from web analyzer
        component_status = {}
        if HAS_WEB_ANALYZER and service.web_analyzer:
            component_status = service.web_analyzer.get_component_status()
        
        model_info = {
            "web_analyzer": {
                "available": HAS_WEB_ANALYZER and service.web_analyzer is not None,
                "concurrent_processing": True,
                "fallback_enabled": component_status.get("fallback_enabled", True),
                "supported_languages": ["ta", "ml", "en"]
            },
            "pos_tagger": {
                "available": component_status.get("pos_tagger", False),
                "model_name": "bert-base-multilingual-cased" if HAS_NLP_COMPONENTS else "fallback",
                "languages": ["en", "ta", "ml"] if HAS_NLP_COMPONENTS else ["en"]
            },
            "ner_system": {
                "available": component_status.get("ner_system", False),
                "model_name": "bert-base-multilingual-cased" if HAS_NLP_COMPONENTS else "fallback",
                "entity_types": ["PERSON", "LOCATION", "ORGANIZATION", "MEDICAL", "MISC"]
            },
            "dependency_parser": {
                "available": component_status.get("dependency_parser", False),
                "model_name": "bert-base-multilingual-cased" if HAS_NLP_COMPONENTS else "fallback",
                "annotation_scheme": "universal_dependencies"
            }
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@router.post("/analyze/batch", response_model=List[AnalysisResult])
async def analyze_batch(
    texts: List[str],
    language: Optional[str] = None,
    include_sentiment: bool = True,
    include_features: bool = True,
    service: NLPService = Depends(get_nlp_service)
) -> List[AnalysisResult]:
    """
    Analyze multiple texts in batch.
    
    Processes multiple text inputs concurrently for improved performance.
    """
    try:
        # Validate input
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="Too many texts (max 100)")
        
        for i, text in enumerate(texts):
            if not text.strip():
                raise HTTPException(status_code=400, detail=f"Text {i} cannot be empty")
            if len(text) > 10000:
                raise HTTPException(status_code=400, detail=f"Text {i} too long (max 10000 characters)")
        
        # Use web analyzer batch processing if available
        if HAS_WEB_ANALYZER and service.web_analyzer:
            web_results = await service.web_analyzer.analyze_batch(texts, language or "ta")
            
            # Convert WebAnalysisResult to AnalysisResult
            results = []
            for web_result in web_results:
                # Convert POS tags
                pos_tags = []
                if web_result.pos_result and hasattr(web_result.pos_result, 'tokens'):
                    pos_tags = [
                        POSTag(
                            token=token,
                            tag=tag,
                            position=i,
                            confidence=conf
                        ) for i, (token, tag, conf) in enumerate(zip(
                            web_result.pos_result.tokens,
                            web_result.pos_result.tags,
                            web_result.pos_result.confidence_scores
                        ))
                    ]
                
                # Convert named entities
                named_entities = []
                for entity in web_result.entities:
                    named_entities.append(NamedEntity(
                        text=getattr(entity, 'text', ''),
                        label=getattr(entity, 'label', 'MISC'),
                        start=getattr(entity, 'start_idx', 0),
                        end=getattr(entity, 'end_idx', 0),
                        confidence=getattr(entity, 'confidence', 0.0)
                    ))
                
                # Convert dependencies
                dependencies = []
                if web_result.dependency_tree and hasattr(web_result.dependency_tree, 'tokens'):
                    for i, (token, head, relation, conf) in enumerate(zip(
                        web_result.dependency_tree.tokens,
                        web_result.dependency_tree.heads,
                        web_result.dependency_tree.relations,
                        web_result.dependency_tree.confidence_scores
                    )):
                        head_text = web_result.dependency_tree.tokens[head] if head != -1 and head < len(web_result.dependency_tree.tokens) else "ROOT"
                        dependencies.append(DependencyRelation(
                            head=head,
                            dependent=i,
                            relation=relation,
                            head_text=head_text,
                            dependent_text=token,
                            confidence=conf
                        ))
                
                # Create sentiment and features
                sentiment = SentimentAnalysis(
                    overall_sentiment="neutral",
                    confidence=0.75,
                    scores={"positive": 0.3, "negative": 0.2, "neutral": 0.5}
                )
                
                # Use sentiment result if available
                if web_result.sentiment_result:
                    sentiment = SentimentAnalysis(
                        overall_sentiment=web_result.sentiment_result.overall_sentiment,
                        confidence=web_result.sentiment_result.confidence,
                        scores=web_result.sentiment_result.scores
                    )
                
                features = web_result.linguistic_features
                linguistic_features = LinguisticFeatures(
                    sentence_count=1,
                    word_count=len(web_result.tokens),
                    avg_sentence_length=float(len(web_result.tokens)),
                    lexical_diversity=features.get("pos_diversity", 0.8),
                    function_word_ratio=features.get("function_word_ratio", 0.3),
                    pos_distribution=features.get("pos_distribution", {})
                )
                
                results.append(AnalysisResult(
                    text=web_result.text,
                    tokens=web_result.tokens,
                    pos_tags=pos_tags,
                    named_entities=named_entities,
                    dependencies=dependencies,
                    sentiment=sentiment,
                    linguistic_features=linguistic_features,
                    language=web_result.language,
                    processing_time=web_result.processing_time
                ))
            
        else:
            # Process texts concurrently using individual analysis
            tasks = [
                service.analyze_text(
                    text=text,
                    language=language,
                    include_sentiment=include_sentiment,
                    include_features=include_features
                )
                for text in texts
            ]
            
            results = await asyncio.gather(*tasks)
        
        total_processing_time = sum(result.processing_time for result in results)
        logger.info(f"Batch analysis of {len(texts)} texts completed in {total_processing_time:.3f}s")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")


# File Upload Endpoints

def ensure_upload_dir():
    """Ensure upload directory exists."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_file_type(file: UploadFile, allowed_types: set) -> bool:
    """Validate file type against allowed types."""
    return file.content_type in allowed_types


def generate_file_id() -> str:
    """Generate unique file identifier."""
    return f"file_{uuid.uuid4().hex[:12]}"


async def save_uploaded_file(file: UploadFile, file_id: str) -> str:
    """Save uploaded file and return file path."""
    ensure_upload_dir()
    
    # Create safe filename
    safe_filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            os.remove(file_path) if os.path.exists(file_path) else None
            raise HTTPException(status_code=413, detail="File too large")
        buffer.write(content)
    
    return file_path


async def process_audio_file(file_path: str, file_id: str, language: str = "tamil") -> AnalysisResult:
    """Process audio file through ASR and NLP pipeline."""
    try:
        # Initialize ASR pipeline if available
        if HAS_ASR_PIPELINE:
            asr_pipeline = create_asr_pipeline()
            
            # Transcribe audio
            transcript_result = asr_pipeline.transcribe(file_path, language)
            
            if transcript_result.error:
                raise Exception(f"ASR failed: {transcript_result.error}")
            
            transcription = transcript_result.text
            
        else:
            # Mock transcription for testing
            transcription = f"Mock transcription for file {file_id}"
        
        # Analyze transcribed text
        service = await get_nlp_service()
        analysis_result = await service.analyze_text(
            text=transcription,
            language=language,
            include_sentiment=True,
            include_features=True
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Audio file processing failed for {file_id}: {e}")
        raise


async def process_text_file(file_path: str, file_id: str, language: str = "english") -> List[AnalysisResult]:
    """Process text file through NLP pipeline."""
    try:
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into segments (by lines or sentences)
        if file_path.endswith('.csv'):
            # Handle CSV files - analyze each row
            import csv
            texts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:  # Skip empty rows
                        texts.append(' '.join(row))
        elif file_path.endswith('.json'):
            # Handle JSON files
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                texts = [str(item) for item in data]
            elif isinstance(data, dict):
                texts = [str(value) for value in data.values()]
            else:
                texts = [str(data)]
        else:
            # Handle plain text files - split by lines
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Limit number of texts to process
        if len(texts) > 100:
            texts = texts[:100]
            logger.warning(f"Text file {file_id} contains too many segments, processing first 100")
        
        # Analyze texts in batch
        service = await get_nlp_service()
        results = await service.analyze_batch(
            texts=texts,
            language=language,
            include_sentiment=True,
            include_features=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Text file processing failed for {file_id}: {e}")
        raise


async def background_file_processor():
    """Background task to process files from the queue."""
    while True:
        try:
            # Get next file from queue
            file_info = await processing_queue.get()
            file_id = file_info["file_id"]
            file_path = file_info["file_path"]
            file_type = file_info["file_type"]
            language = file_info.get("language", "english")
            
            logger.info(f"Processing file {file_id} ({file_type})")
            
            # Update status
            processing_results[file_id]["status"] = "processing"
            processing_results[file_id]["started_at"] = datetime.now()
            
            # Process based on file type
            if file_type in ALLOWED_AUDIO_TYPES:
                result = await process_audio_file(file_path, file_id, language)
                processing_results[file_id]["results"] = [result]
            elif file_type in ALLOWED_TEXT_TYPES:
                results = await process_text_file(file_path, file_id, language)
                processing_results[file_id]["results"] = results
            else:
                raise Exception(f"Unsupported file type: {file_type}")
            
            # Update status
            processing_results[file_id]["status"] = "completed"
            processing_results[file_id]["completed_at"] = datetime.now()
            
            logger.info(f"File {file_id} processed successfully")
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            if file_id in processing_results:
                processing_results[file_id]["status"] = "error"
                processing_results[file_id]["error"] = str(e)
                processing_results[file_id]["completed_at"] = datetime.now()
        
        finally:
            processing_queue.task_done()


@router.post("/upload/audio", response_model=FileUploadResponse)
async def upload_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "tamil"
):
    """
    Upload audio file for processing.
    
    Accepts audio files and processes them through ASR and NLP pipeline.
    Requirements: 5.1
    """
    try:
        # Validate file type
        if not validate_file_type(file, ALLOWED_AUDIO_TYPES):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_AUDIO_TYPES)}"
            )
        
        # Generate file ID and save file
        file_id = generate_file_id()
        file_path = await save_uploaded_file(file, file_id)
        
        # Initialize processing result
        processing_results[file_id] = {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_type": file.content_type,
            "language": language,
            "status": "queued",
            "uploaded_at": datetime.now(),
            "results": None,
            "error": None
        }
        
        # Add to processing queue
        await processing_queue.put({
            "file_id": file_id,
            "file_path": file_path,
            "file_type": file.content_type,
            "language": language
        })
        
        # Start background processing if not already running
        background_tasks.add_task(background_file_processor)
        
        logger.info(f"Audio file uploaded: {file.filename} (ID: {file_id})")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_size=os.path.getsize(file_path),
            file_type=file.content_type,
            upload_time=datetime.now(),
            processing_status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio file upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@router.post("/upload/text", response_model=FileUploadResponse)
async def upload_text_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "english"
):
    """
    Upload text file for processing.
    
    Accepts text files and processes them through NLP pipeline.
    Requirements: 5.2
    """
    try:
        # Validate file type
        if not validate_file_type(file, ALLOWED_TEXT_TYPES):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_TEXT_TYPES)}"
            )
        
        # Generate file ID and save file
        file_id = generate_file_id()
        file_path = await save_uploaded_file(file, file_id)
        
        # Initialize processing result
        processing_results[file_id] = {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_type": file.content_type,
            "language": language,
            "status": "queued",
            "uploaded_at": datetime.now(),
            "results": None,
            "error": None
        }
        
        # Add to processing queue
        await processing_queue.put({
            "file_id": file_id,
            "file_path": file_path,
            "file_type": file.content_type,
            "language": language
        })
        
        # Start background processing if not already running
        background_tasks.add_task(background_file_processor)
        
        logger.info(f"Text file uploaded: {file.filename} (ID: {file_id})")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_size=os.path.getsize(file_path),
            file_type=file.content_type,
            upload_time=datetime.now(),
            processing_status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text file upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@router.get("/upload/status/{file_id}")
async def get_file_processing_status(file_id: str):
    """
    Get processing status for uploaded file.
    
    Returns current processing status and results if available.
    """
    if file_id not in processing_results:
        raise HTTPException(status_code=404, detail="File not found")
    
    result = processing_results[file_id]
    
    response = {
        "file_id": file_id,
        "filename": result["filename"],
        "status": result["status"],
        "uploaded_at": result["uploaded_at"],
        "language": result["language"]
    }
    
    if result.get("started_at"):
        response["started_at"] = result["started_at"]
    
    if result.get("completed_at"):
        response["completed_at"] = result["completed_at"]
        
        # Calculate processing time
        if result.get("started_at"):
            processing_time = (result["completed_at"] - result["started_at"]).total_seconds()
            response["processing_time"] = processing_time
    
    if result["status"] == "completed" and result["results"]:
        response["results"] = result["results"]
    
    if result["status"] == "error" and result["error"]:
        response["error"] = result["error"]
    
    return response


@router.get("/upload/results/{file_id}")
async def get_file_results(file_id: str):
    """
    Get processing results for uploaded file.
    
    Returns detailed analysis results if processing is complete.
    """
    if file_id not in processing_results:
        raise HTTPException(status_code=404, detail="File not found")
    
    result = processing_results[file_id]
    
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"File processing not complete. Status: {result['status']}"
        )
    
    if not result["results"]:
        raise HTTPException(status_code=404, detail="No results available")
    
    return {
        "file_id": file_id,
        "filename": result["filename"],
        "language": result["language"],
        "processing_time": (result["completed_at"] - result["started_at"]).total_seconds(),
        "results": result["results"]
    }


# Batch Processing Endpoints

@router.post("/batch/create")
async def create_batch_operation(
    operation_type: str,
    items: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """
    Create a new batch operation for processing multiple items.
    
    Supports batch processing of files, texts, or mixed content.
    Requirements: 5.3, 5.4
    """
    try:
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        # Validate operation type
        valid_operations = ["file_processing", "text_analysis", "audio_transcription"]
        if operation_type not in valid_operations:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation type. Allowed: {', '.join(valid_operations)}"
            )
        
        # Validate items
        if not items:
            raise HTTPException(status_code=400, detail="No items provided")
        
        if len(items) > 100:
            raise HTTPException(status_code=400, detail="Too many items (max 100)")
        
        # Create batch operation
        controller = get_demo_controller()
        batch_id = await controller.create_batch_operation(
            operation_type=operation_type,
            items=items
        )
        
        # Start processing in background
        background_tasks.add_task(controller.start_batch_processing, batch_id)
        
        logger.info(f"Created batch operation {batch_id} with {len(items)} items")
        
        return {
            "batch_id": batch_id,
            "operation_type": operation_type,
            "total_items": len(items),
            "status": "queued",
            "created_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create batch operation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create batch operation")


@router.get("/batch/progress/{batch_id}")
async def get_batch_progress(batch_id: str):
    """
    Get progress information for a batch operation.
    
    Returns current processing status and progress metrics.
    Requirements: 5.3
    """
    try:
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        controller = get_demo_controller()
        progress = controller.get_batch_progress(batch_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "batch_id": batch_id,
            "status": progress.status,
            "total_items": progress.total_items,
            "processed_items": progress.processed_items,
            "successful_items": progress.successful_items,
            "failed_items": progress.failed_items,
            "progress_percentage": progress.progress_percentage,
            "current_item": progress.current_item,
            "started_at": progress.started_at,
            "completed_at": progress.completed_at,
            "processing_time": progress.processing_time,
            "error_message": progress.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch progress: {e}")
        raise HTTPException(status_code=500, detail="Failed to get batch progress")


@router.get("/batch/results/{batch_id}")
async def get_batch_results(batch_id: str):
    """
    Get aggregated results for a completed batch operation.
    
    Returns comprehensive analysis results and statistics.
    Requirements: 5.4
    """
    try:
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        controller = get_demo_controller()
        results = controller.get_batch_results(batch_id)
        
        if not results:
            # Check if batch exists but not completed
            progress = controller.get_batch_progress(batch_id)
            if progress:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch not completed. Status: {progress.status}"
                )
            else:
                raise HTTPException(status_code=404, detail="Batch not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get batch results")


@router.get("/batch/details/{batch_id}")
async def get_batch_details(batch_id: str):
    """
    Get detailed information about a batch operation.
    
    Returns comprehensive batch information including item-level details.
    Requirements: 5.5
    """
    try:
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        controller = get_demo_controller()
        details = controller.get_batch_details(batch_id)
        
        if not details:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get batch details")


@router.post("/batch/cancel/{batch_id}")
async def cancel_batch_operation(batch_id: str):
    """
    Cancel an active batch operation.
    
    Stops processing and marks the batch as cancelled.
    """
    try:
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        controller = get_demo_controller()
        success = await controller.cancel_batch(batch_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Batch not found or not active")
        
        return {
            "batch_id": batch_id,
            "status": "cancelled",
            "cancelled_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel batch")


@router.get("/batch/list")
async def list_batch_operations(
    status: Optional[str] = None,
    limit: int = 50
):
    """
    List batch operations with optional status filtering.
    
    Returns a list of batch operations with basic information.
    """
    try:
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        controller = get_demo_controller()
        
        # Get all batches
        all_batches = []
        
        # Add active batches
        for batch_id, batch_op in controller.active_batches.items():
            all_batches.append({
                "batch_id": batch_id,
                "operation_type": batch_op.operation_type,
                "status": batch_op.progress.status,
                "total_items": batch_op.progress.total_items,
                "processed_items": batch_op.progress.processed_items,
                "progress_percentage": batch_op.progress.progress_percentage,
                "started_at": batch_op.progress.started_at,
                "completed_at": batch_op.progress.completed_at
            })
        
        # Add completed batches
        for batch_id, batch_op in controller.completed_batches.items():
            all_batches.append({
                "batch_id": batch_id,
                "operation_type": batch_op.operation_type,
                "status": batch_op.progress.status,
                "total_items": batch_op.progress.total_items,
                "processed_items": batch_op.progress.processed_items,
                "progress_percentage": batch_op.progress.progress_percentage,
                "started_at": batch_op.progress.started_at,
                "completed_at": batch_op.progress.completed_at
            })
        
        # Filter by status if provided
        if status:
            all_batches = [b for b in all_batches if b["status"] == status]
        
        # Sort by creation time (most recent first)
        all_batches.sort(key=lambda x: x["started_at"] or datetime.min, reverse=True)
        
        # Apply limit
        all_batches = all_batches[:limit]
        
        return {
            "batches": all_batches,
            "total_count": len(all_batches),
            "statistics": controller.get_statistics()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list batches: {e}")
        raise HTTPException(status_code=500, detail="Failed to list batches")


@router.post("/batch/files")
async def create_file_batch(
    files: List[UploadFile] = File(...),
    operation_type: str = "file_processing",
    language: str = "english",
    background_tasks: BackgroundTasks = None
):
    """
    Create a batch operation from multiple uploaded files.
    
    Convenience endpoint for processing multiple files in a single batch.
    Requirements: 5.5
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Too many files (max 50)")
        
        # Process and save all files
        batch_items = []
        for file in files:
            # Validate file type
            if file.content_type in ALLOWED_AUDIO_TYPES:
                file_type = "audio_file"
            elif file.content_type in ALLOWED_TEXT_TYPES:
                file_type = "text_file"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type}"
                )
            
            # Save file
            file_id = generate_file_id()
            file_path = await save_uploaded_file(file, file_id)
            
            # Add to batch items
            batch_items.append({
                "type": file_type,
                "data": file_path,
                "metadata": {
                    "filename": file.filename,
                    "file_id": file_id,
                    "language": language,
                    "content_type": file.content_type
                }
            })
        
        # Create batch operation
        if not HAS_DEMO_CONTROLLER:
            raise HTTPException(status_code=503, detail="Demo controller not available")
        
        controller = get_demo_controller()
        batch_id = await controller.create_batch_operation(
            operation_type=operation_type,
            items=batch_items
        )
        
        # Start processing in background
        if background_tasks:
            background_tasks.add_task(controller.start_batch_processing, batch_id)
        
        logger.info(f"Created file batch {batch_id} with {len(files)} files")
        
        return {
            "batch_id": batch_id,
            "operation_type": operation_type,
            "total_files": len(files),
            "file_types": list(set(item["type"] for item in batch_items)),
            "status": "queued",
            "created_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create file batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to create file batch")


# Session Management Endpoints

@router.get("/sessions")
async def get_sessions_root():
    """
    Get sessions overview.
    
    Returns basic information about session management capabilities.
    """
    try:
        if not HAS_SESSION_MANAGER:
            return {
                "status": "unavailable",
                "message": "Session manager not available",
                "endpoints": {
                    "create": "/api/sessions/create",
                    "history": "/api/sessions/history",
                    "statistics": "/api/sessions/statistics"
                }
            }
        
        session_manager = get_session_manager()
        stats = await session_manager.get_statistics()
        
        return {
            "status": "active",
            "session_manager_available": True,
            "active_sessions": stats.get("active_sessions", 0),
            "total_sessions": stats.get("total_sessions", 0),
            "endpoints": {
                "create": "/api/sessions/create",
                "history": "/api/sessions/history",
                "statistics": "/api/sessions/statistics"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get sessions root: {e}")
        return {
            "status": "error",
            "session_manager_available": False,
            "error": str(e),
            "endpoints": {
                "create": "/api/sessions/create",
                "history": "/api/sessions/history",
                "statistics": "/api/sessions/statistics"
            }
        }


@router.post("/sessions/create")
async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Create a new session.
    
    Creates a unique session with optional user ID and metadata.
    Requirements: 9.1
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        session_id = await session_manager.create_session(
            user_id=user_id,
            metadata=metadata or {}
        )
        
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "status": "active"
        }
        
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get session data by ID.
    
    Returns session information if found and active.
    Requirements: 9.2
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        session = await session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        return session.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")


@router.put("/sessions/{session_id}")
async def update_session(
    session_id: str,
    updates: Dict[str, Any]
):
    """
    Update session data.
    
    Updates specified fields in the session.
    Requirements: 9.2
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        success = await session_manager.update_session(session_id, **updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"session_id": session_id, "updated": True, "updated_at": datetime.now()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.post("/sessions/{session_id}/analysis")
async def add_analysis_to_session(
    session_id: str,
    analysis_result: AnalysisResult
):
    """
    Add analysis result to session.
    
    Stores analysis result in the session for later retrieval.
    Requirements: 9.2
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        success = await session_manager.add_analysis_result(session_id, analysis_result)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "analysis_added": True,
            "timestamp": analysis_result.timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add analysis to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add analysis to session")


@router.post("/sessions/{session_id}/complete")
async def complete_session(session_id: str):
    """
    Mark session as completed.
    
    Finalizes the session and preserves data for history.
    Requirements: 9.2
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        success = await session_manager.complete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "status": "completed",
            "completed_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete session")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its data.
    
    Permanently removes session data.
    Requirements: 9.4
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        success = await session_manager.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "deleted": True,
            "deleted_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.get("/sessions/history")
async def get_session_history(
    user_id: Optional[str] = None,
    limit: int = 100,
    include_completed: bool = True,
    include_expired: bool = False
):
    """
    Get session history.
    
    Returns list of sessions with optional filtering.
    Requirements: 9.3
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        # Validate limit
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        
        session_manager = get_session_manager()
        sessions = await session_manager.get_session_history(
            user_id=user_id,
            limit=limit,
            include_completed=include_completed,
            include_expired=include_expired
        )
        
        # Convert to dict format for JSON response
        session_list = [session.dict() for session in sessions]
        
        return {
            "sessions": session_list,
            "total_count": len(session_list),
            "filters": {
                "user_id": user_id,
                "include_completed": include_completed,
                "include_expired": include_expired
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session history")


@router.get("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = "json"
):
    """
    Export session data in specified format.
    
    Generates downloadable report with analysis summaries.
    Requirements: 9.5
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        # Validate format
        valid_formats = ["json", "csv", "pickle"]
        if format.lower() not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Allowed: {', '.join(valid_formats)}"
            )
        
        session_manager = get_session_manager()
        exported_data = await session_manager.export_session(session_id, format.lower())
        
        if exported_data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Determine content type and filename
        if format.lower() == "json":
            content_type = "application/json"
            filename = f"session_{session_id}.json"
        elif format.lower() == "csv":
            content_type = "text/csv"
            filename = f"session_{session_id}.csv"
        elif format.lower() == "pickle":
            content_type = "application/octet-stream"
            filename = f"session_{session_id}.pkl"
        
        # Return file response
        from fastapi.responses import Response
        return Response(
            content=exported_data if isinstance(exported_data, bytes) else exported_data.encode(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": content_type
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export session")


@router.get("/sessions/statistics")
async def get_session_statistics():
    """
    Get session manager statistics.
    
    Returns comprehensive statistics about session usage.
    Requirements: 9.4
    """
    try:
        if not HAS_SESSION_MANAGER:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = get_session_manager()
        stats = await session_manager.get_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get session statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session statistics")


# Error handlers will be added to the main app, not the router
# These are defined here for reference but applied in main.py

def create_error_handlers():
    """Create error handler functions for the main app."""
    
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions with consistent error format."""
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error",
                message=exc.detail,
                details={"status_code": exc.status_code}
            ).model_dump(mode='json')  # Use model_dump with json mode for proper serialization
        )

    async def general_exception_handler(request, exc):
        """Handle general exceptions with consistent error format."""
        from fastapi.responses import JSONResponse
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred",
                details={"exception_type": type(exc).__name__}
            ).model_dump(mode='json')  # Use model_dump with json mode for proper serialization
        )
    
    return http_exception_handler, general_exception_handler