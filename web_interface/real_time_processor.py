"""
Real-time processing pipeline for audio capture → transcription → analysis → visualization
Implements end-to-end real-time flow with WebSocket streaming
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from .audio_processor import AudioProcessor
from .linguistic_analyzer_wrapper import LinguisticAnalyzerWrapper
from .websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """Orchestrates real-time audio processing pipeline"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.linguistic_analyzer = LinguisticAnalyzerWrapper()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def start_real_time_session(self, websocket, session_id: str) -> Dict[str, Any]:
        """Start a new real-time processing session"""
        
        session_data = {
            'session_id': session_id,
            'websocket': websocket,
            'start_time': datetime.now(),
            'status': 'active',
            'audio_chunks': [],
            'transcription_segments': [],
            'analysis_results': [],
            'total_processed': 0
        }
        
        self.active_sessions[session_id] = session_data
        
        # Send session initialization confirmation
        await self._send_to_client(websocket, {
            'type': 'session_started',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'message': 'Real-time processing session initialized'
        })
        
        logger.info(f"Started real-time session: {session_id}")
        return session_data
    
    async def process_audio_chunk(self, session_id: str, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Process incoming audio chunk through the pipeline"""
        
        if session_id not in self.active_sessions:
            logger.error(f"Session not found: {session_id}")
            return None
            
        session = self.active_sessions[session_id]
        websocket = session['websocket']
        
        try:
            # Step 1: Audio processing and transcription
            transcription_result = await self.audio_processor.process_chunk(audio_data)
            
            if transcription_result and transcription_result.get('text'):
                text = transcription_result['text']
                confidence = transcription_result.get('confidence', 0.0)
                
                # Store transcription segment
                segment = {
                    'text': text,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'chunk_id': len(session['transcription_segments'])
                }
                session['transcription_segments'].append(segment)
                
                # Send transcription update
                await self._send_to_client(websocket, {
                    'type': 'transcription_update',
                    'session_id': session_id,
                    'segment': segment,
                    'total_segments': len(session['transcription_segments'])
                })
                
                # Step 2: Linguistic analysis
                analysis_result = await self._analyze_text_segment(text, session_id)
                
                if analysis_result:
                    # Store analysis result
                    session['analysis_results'].append(analysis_result)
                    
                    # Step 3: Send analysis results for visualization
                    await self._send_to_client(websocket, {
                        'type': 'analysis_update',
                        'session_id': session_id,
                        'analysis': analysis_result,
                        'segment_id': segment['chunk_id']
                    })
                
                session['total_processed'] += 1
                
                # Send processing completion notification
                await self._send_to_client(websocket, {
                    'type': 'processing_complete',
                    'session_id': session_id,
                    'segment_id': segment['chunk_id'],
                    'total_processed': session['total_processed']
                })
                
                return {
                    'transcription': segment,
                    'analysis': analysis_result,
                    'session_stats': {
                        'total_segments': len(session['transcription_segments']),
                        'total_processed': session['total_processed']
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing audio chunk for session {session_id}: {e}")
            await self._send_to_client(websocket, {
                'type': 'processing_error',
                'session_id': session_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
        return None
    
    async def _analyze_text_segment(self, text: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Perform linguistic analysis on text segment"""
        
        try:
            # Run all analyses concurrently for speed
            analysis_tasks = [
                self.linguistic_analyzer.analyze_pos(text),
                self.linguistic_analyzer.analyze_ner(text),
                self.linguistic_analyzer.analyze_dependencies(text),
                self.linguistic_analyzer.analyze_sentiment(text)
            ]
            
            pos_result, ner_result, dep_result, sentiment_result = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )
            
            # Compile results
            analysis = {
                'text': text,
                'timestamp': datetime.now().isoformat(),
                'pos_tags': pos_result if not isinstance(pos_result, Exception) else [],
                'named_entities': ner_result if not isinstance(ner_result, Exception) else [],
                'dependencies': dep_result if not isinstance(dep_result, Exception) else [],
                'sentiment': sentiment_result if not isinstance(sentiment_result, Exception) else {}
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in linguistic analysis: {e}")
            return None
    
    async def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """Finalize real-time processing session"""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
            
        session = self.active_sessions[session_id]
        websocket = session['websocket']
        
        # Calculate session statistics
        end_time = datetime.now()
        duration = (end_time - session['start_time']).total_seconds()
        
        session_summary = {
            'session_id': session_id,
            'start_time': session['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_segments': len(session['transcription_segments']),
            'total_analyses': len(session['analysis_results']),
            'status': 'completed'
        }
        
        # Send session completion notification
        await self._send_to_client(websocket, {
            'type': 'session_completed',
            'session_id': session_id,
            'summary': session_summary,
            'final_results': {
                'transcription_segments': session['transcription_segments'],
                'analysis_results': session['analysis_results']
            }
        })
        
        # Clean up session
        session['status'] = 'completed'
        del self.active_sessions[session_id]
        
        logger.info(f"Finalized session {session_id}: {session_summary}")
        return session_summary
    
    async def _send_to_client(self, websocket, message: Dict[str, Any]):
        """Send message to WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'status': session['status'],
            'start_time': session['start_time'].isoformat(),
            'total_segments': len(session['transcription_segments']),
            'total_analyses': len(session['analysis_results']),
            'total_processed': session['total_processed']
        }
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_sessions.keys())