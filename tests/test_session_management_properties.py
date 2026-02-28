"""
Property-based tests for session management functionality.

This module tests the correctness properties of the session management system
using property-based testing with Hypothesis.

Properties tested:
- Property 39: Session initialization
- Property 40: Result persistence  
- Property 42: Resource cleanup

Reference: NLP Web Interface - Task 10.3
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

# Import session management components
from web_interface.session_manager import SessionManager, get_session_manager, initialize_session_manager
from web_interface.models import SessionData, SessionStatus, AnalysisResult, POSTag, NamedEntity, DependencyRelation, SentimentAnalysis, LinguisticFeatures


# Test data generators
@st.composite
def session_metadata(draw):
    """Generate session metadata."""
    return draw(st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(min_value=0, max_value=1000000),
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.booleans()
        ),
        min_size=0,
        max_size=10
    ))


@st.composite
def user_id_strategy(draw):
    """Generate user IDs."""
    return draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    ))


@st.composite
def analysis_result_strategy(draw):
    """Generate analysis results."""
    text = draw(st.text(min_size=1, max_size=1000))
    tokens = text.split()[:20]  # Limit tokens for performance
    
    pos_tags = [
        POSTag(
            token=token,
            tag=draw(st.sampled_from(["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ"])),
            position=i,
            confidence=draw(st.floats(min_value=0.0, max_value=1.0))
        )
        for i, token in enumerate(tokens)
    ]
    
    named_entities = [
        NamedEntity(
            text=token,
            label=draw(st.sampled_from(["PERSON", "LOCATION", "ORGANIZATION", "MEDICAL", "MISC"])),
            start=i * 5,
            end=(i * 5) + len(token),
            confidence=draw(st.floats(min_value=0.0, max_value=1.0))
        )
        for i, token in enumerate(tokens[:3])  # Limit entities
    ]
    
    dependencies = [
        DependencyRelation(
            head=max(0, i - 1) if i > 0 else -1,
            dependent=i,
            relation=draw(st.sampled_from(["nsubj", "obj", "det", "amod", "root"])),
            head_text=tokens[max(0, i - 1)] if i > 0 else "ROOT",
            dependent_text=token,
            confidence=draw(st.floats(min_value=0.0, max_value=1.0))
        )
        for i, token in enumerate(tokens)
    ]
    
    sentiment = SentimentAnalysis(
        overall_sentiment=draw(st.sampled_from(["positive", "negative", "neutral"])),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        scores={
            "positive": draw(st.floats(min_value=0.0, max_value=1.0)),
            "negative": draw(st.floats(min_value=0.0, max_value=1.0)),
            "neutral": draw(st.floats(min_value=0.0, max_value=1.0))
        }
    )
    
    linguistic_features = LinguisticFeatures(
        sentence_count=draw(st.integers(min_value=1, max_value=10)),
        word_count=len(tokens),
        avg_sentence_length=draw(st.floats(min_value=1.0, max_value=50.0)),
        lexical_diversity=draw(st.floats(min_value=0.0, max_value=1.0)),
        function_word_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
        pos_distribution={"NOUN": 0.3, "VERB": 0.2, "ADJ": 0.1, "OTHER": 0.4}
    )
    
    return AnalysisResult(
        text=text,
        tokens=tokens,
        pos_tags=pos_tags,
        named_entities=named_entities,
        dependencies=dependencies,
        sentiment=sentiment,
        linguistic_features=linguistic_features,
        language=draw(st.sampled_from(["en", "ta", "ml"])),
        processing_time=draw(st.floats(min_value=0.001, max_value=10.0))
    )


class TestSessionManagementProperties:
    """Property-based tests for session management."""
    
    def create_temp_session_manager(self):
        """Create a temporary session manager for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = SessionManager(
            session_timeout_minutes=1,  # Short timeout for testing
            max_sessions=100,
            persistence_path=temp_dir,
            cleanup_interval_minutes=1,
            enable_persistence=True
        )
        return manager, temp_dir
    
    @given(
        user_id=user_id_strategy(),
        metadata=session_metadata()
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_39_session_initialization(self, user_id, metadata):
        """
        Property 39: Session initialization
        For any new session creation, a unique session identifier should be 
        generated and initial state properly initialized.
        
        **Validates: Requirements 9.1**
        """
        async def test_session_initialization():
            manager, temp_dir = self.create_temp_session_manager()
            
            try:
                # Create session
                session_id = await manager.create_session(user_id=user_id, metadata=metadata)
                
                # Verify session ID is unique and properly formatted
                assert session_id is not None
                assert isinstance(session_id, str)
                assert session_id.startswith("sess_")
                assert len(session_id) > 10  # Should have timestamp and random parts
                
                # Verify session was created and initialized properly
                session = await manager.get_session(session_id)
                assert session is not None
                assert session.session_id == session_id
                assert session.user_id == user_id
                assert session.status == SessionStatus.ACTIVE
                assert session.metadata == metadata
                assert isinstance(session.created_at, datetime)
                assert isinstance(session.updated_at, datetime)
                assert session.analysis_results == []
                
                # Verify timestamps are reasonable
                now = datetime.now()
                assert (now - session.created_at).total_seconds() < 10  # Created within last 10 seconds
                assert (now - session.updated_at).total_seconds() < 10  # Updated within last 10 seconds
                
                # Verify session is tracked in manager
                assert session_id in manager.active_sessions
                assert session_id in manager.session_locks
                
            finally:
                await manager.shutdown()
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        asyncio.run(test_session_initialization())
    
    @given(
        user_id=user_id_strategy(),
        metadata=session_metadata(),
        analysis_results=st.lists(analysis_result_strategy(), min_size=1, max_size=5)
    )
    @settings(max_examples=50, deadline=10000)
    def test_property_40_result_persistence(self, user_id, metadata, analysis_results):
        """
        Property 40: Result persistence
        For any analysis operation, results should be saved and accessible 
        through session history.
        
        **Validates: Requirements 9.2**
        """
        async def test_result_persistence():
            manager, temp_dir = self.create_temp_session_manager()
            
            try:
                # Create session
                session_id = await manager.create_session(user_id=user_id, metadata=metadata)
                
                # Add analysis results
                for result in analysis_results:
                    success = await manager.add_analysis_result(session_id, result)
                    assert success is True
                
                # Verify results are persisted in active session
                session = await manager.get_session(session_id)
                assert session is not None
                assert len(session.analysis_results) == len(analysis_results)
                
                # Verify each result is correctly stored
                for i, original_result in enumerate(analysis_results):
                    stored_result = session.analysis_results[i]
                    assert stored_result.text == original_result.text
                    assert stored_result.language == original_result.language
                    assert len(stored_result.tokens) == len(original_result.tokens)
                    assert len(stored_result.pos_tags) == len(original_result.pos_tags)
                    assert len(stored_result.named_entities) == len(original_result.named_entities)
                    assert stored_result.sentiment.overall_sentiment == original_result.sentiment.overall_sentiment
                
                # Complete session to ensure persistence
                await manager.complete_session(session_id)
                
                # Verify session is accessible through history
                history = await manager.get_session_history(user_id=user_id, include_completed=True)
                session_found = False
                for hist_session in history:
                    if hist_session.session_id == session_id:
                        session_found = True
                        assert hist_session.status == SessionStatus.COMPLETED
                        assert len(hist_session.analysis_results) == len(analysis_results)
                        break
                
                assert session_found, "Session not found in history after completion"
                
                # If persistence is enabled, verify data survives manager restart
                if manager.enable_persistence:
                    # Create new manager with same persistence path
                    new_manager = SessionManager(
                        session_timeout_minutes=60,
                        max_sessions=100,
                        persistence_path=manager.persistence_path,
                        enable_persistence=True
                    )
                    
                    try:
                        # Load session from disk
                        loaded_session = await new_manager._load_session_from_disk(session_id)
                        assert loaded_session is not None
                        assert loaded_session.session_id == session_id
                        assert loaded_session.user_id == user_id
                        assert len(loaded_session.analysis_results) == len(analysis_results)
                    finally:
                        await new_manager.shutdown()
                        
            finally:
                await manager.shutdown()
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        asyncio.run(test_result_persistence())
    
    @given(
        user_id=user_id_strategy(),
        metadata=session_metadata(),
        timeout_minutes=st.floats(min_value=0.01, max_value=0.1)  # Very short timeout for testing
    )
    @settings(max_examples=30, deadline=15000)
    def test_property_42_resource_cleanup(self, user_id, metadata, timeout_minutes):
        """
        Property 42: Resource cleanup
        For any inactive session exceeding the configured timeout, resources 
        should be cleaned up automatically.
        
        **Validates: Requirements 9.4**
        """
        async def test_resource_cleanup():
            # Create manager with very short timeout
            temp_dir = tempfile.mkdtemp()
            manager = SessionManager(
                session_timeout_minutes=timeout_minutes,
                max_sessions=100,
                persistence_path=temp_dir,
                cleanup_interval_minutes=timeout_minutes / 2,  # Cleanup more frequently
                enable_persistence=True
            )
            
            try:
                # Create session
                session_id = await manager.create_session(user_id=user_id, metadata=metadata)
                
                # Verify session exists
                session = await manager.get_session(session_id)
                assert session is not None
                assert session_id in manager.active_sessions
                assert session_id in manager.session_locks
                
                # Wait for session to expire (timeout + small buffer)
                wait_time = (timeout_minutes * 60) + 1.0
                await asyncio.sleep(wait_time)
                
                # Trigger cleanup manually to ensure it runs
                await manager._cleanup_expired_sessions()
                
                # Verify session was cleaned up
                expired_session = await manager.get_session(session_id)
                assert expired_session is None, "Session should be None after expiration"
                
                # Verify session removed from active tracking
                assert session_id not in manager.active_sessions, "Session should be removed from active_sessions"
                assert session_id not in manager.session_locks, "Session should be removed from session_locks"
                
                # Verify statistics reflect cleanup
                stats = await manager.get_statistics()
                assert stats["total_expired"] > 0, "Statistics should show expired sessions"
                
                # If persistence enabled, verify session data still exists on disk but marked as expired
                if manager.enable_persistence:
                    persisted_session = await manager._load_session_from_disk(session_id)
                    if persisted_session:  # May be None if cleanup removed it
                        assert persisted_session.status == SessionStatus.TIMEOUT
            
            finally:
                await manager.shutdown()
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        asyncio.run(test_resource_cleanup())
    
    @given(
        num_sessions=st.integers(min_value=1, max_value=10),
        user_ids=st.lists(user_id_strategy(), min_size=1, max_size=5)
    )
    @settings(max_examples=20, deadline=10000)
    def test_concurrent_session_operations(self, num_sessions, user_ids):
        """
        Test concurrent session operations maintain consistency.
        
        Verifies that concurrent session creation, updates, and cleanup
        maintain data integrity and proper resource management.
        """
        async def test_concurrent_operations():
            manager, temp_dir = self.create_temp_session_manager()
            
            try:
                # Create multiple sessions concurrently
                async def create_session_task(i):
                    user_id = user_ids[i % len(user_ids)]
                    metadata = {"session_index": i, "test": "concurrent"}
                    return await manager.create_session(user_id=user_id, metadata=metadata)
                
                # Create sessions concurrently
                session_ids = await asyncio.gather(*[
                    create_session_task(i) for i in range(num_sessions)
                ])
                
                # Verify all sessions were created with unique IDs
                assert len(session_ids) == num_sessions
                assert len(set(session_ids)) == num_sessions  # All unique
                
                # Verify all sessions exist and are properly initialized
                for session_id in session_ids:
                    session = await manager.get_session(session_id)
                    assert session is not None
                    assert session.status == SessionStatus.ACTIVE
                    assert "session_index" in session.metadata
                
                # Perform concurrent updates
                async def update_session_task(session_id, index):
                    return await manager.update_session(
                        session_id, 
                        transcription=f"Test transcription {index}"
                    )
                
                update_results = await asyncio.gather(*[
                    update_session_task(session_id, i) 
                    for i, session_id in enumerate(session_ids)
                ])
                
                # Verify all updates succeeded
                assert all(update_results)
                
                # Verify updates were applied correctly
                for i, session_id in enumerate(session_ids):
                    session = await manager.get_session(session_id)
                    assert session.transcription == f"Test transcription {i}"
                
                # Complete all sessions concurrently
                completion_results = await asyncio.gather(*[
                    manager.complete_session(session_id) 
                    for session_id in session_ids
                ])
                
                # Verify all completions succeeded
                assert all(completion_results)
                
                # Verify final state
                stats = await manager.get_statistics()
                assert stats["total_created"] >= num_sessions
                assert stats["total_completed"] >= num_sessions
                
            finally:
                await manager.shutdown()
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        asyncio.run(test_concurrent_operations())
    
    @given(
        export_format=st.sampled_from(["json", "csv", "pickle"]),
        user_id=user_id_strategy(),
        analysis_results=st.lists(analysis_result_strategy(), min_size=1, max_size=3)
    )
    @settings(max_examples=30, deadline=10000)
    def test_session_export_formats(self, export_format, user_id, analysis_results):
        """
        Test session export functionality across different formats.
        
        Verifies that session data can be exported in various formats
        and contains all necessary information.
        """
        async def test_export_functionality():
            manager, temp_dir = self.create_temp_session_manager()
            
            try:
                # Create session with data
                session_id = await manager.create_session(
                    user_id=user_id,
                    metadata={"export_test": True}
                )
                
                # Add analysis results
                for result in analysis_results:
                    await manager.add_analysis_result(session_id, result)
                
                # Complete session
                await manager.complete_session(session_id)
                
                # Export session
                exported_data = await manager.export_session(session_id, export_format)
                assert exported_data is not None
                
                # Verify export format
                if export_format == "json":
                    assert isinstance(exported_data, str)
                    # Verify it's valid JSON
                    parsed_data = json.loads(exported_data)
                    assert parsed_data["session_id"] == session_id
                    assert parsed_data["user_id"] == user_id
                    assert len(parsed_data["analysis_results"]) == len(analysis_results)
                    
                elif export_format == "csv":
                    assert isinstance(exported_data, str)
                    # Verify CSV structure
                    lines = exported_data.strip().split('\n')
                    assert len(lines) >= 2  # At least header and one data row
                    assert "session_id" in lines[0]  # Header contains session_id
                    
                elif export_format == "pickle":
                    assert isinstance(exported_data, bytes)
                    # Verify pickle can be loaded
                    import pickle
                    unpickled_data = pickle.loads(exported_data)
                    assert isinstance(unpickled_data, dict)
                    assert unpickled_data["session_id"] == session_id
                    
            finally:
                await manager.shutdown()
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        asyncio.run(test_export_functionality())


class SessionManagementStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for session management.
    
    This tests complex interactions and state transitions in the session
    management system using Hypothesis's stateful testing framework.
    """
    
    def __init__(self):
        super().__init__()
        self.temp_dir = None
        self.manager = None
        self.created_sessions = set()
        self.completed_sessions = set()
        self.expired_sessions = set()
    
    @initialize()
    def setup_manager(self):
        """Initialize session manager for testing."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.manager = SessionManager(
            session_timeout_minutes=5,  # 5 minute timeout
            max_sessions=50,
            persistence_path=self.temp_dir,
            cleanup_interval_minutes=1,
            enable_persistence=True
        )
    
    @rule(
        user_id=user_id_strategy(),
        metadata=session_metadata()
    )
    def create_session(self, user_id, metadata):
        """Create a new session."""
        async def _create():
            try:
                session_id = await self.manager.create_session(
                    user_id=user_id,
                    metadata=metadata
                )
                self.created_sessions.add(session_id)
                return session_id
            except RuntimeError:
                # Max sessions reached, this is expected behavior
                pass
        
        asyncio.run(_create())
    
    @rule()
    def complete_random_session(self):
        """Complete a random active session."""
        if not self.created_sessions:
            return
        
        active_sessions = self.created_sessions - self.completed_sessions - self.expired_sessions
        if not active_sessions:
            return
        
        import random
        session_id = random.choice(list(active_sessions))
        
        async def _complete():
            success = await self.manager.complete_session(session_id)
            if success:
                self.completed_sessions.add(session_id)
        
        asyncio.run(_complete())
    
    @rule()
    def trigger_cleanup(self):
        """Trigger session cleanup."""
        async def _cleanup():
            await self.manager._cleanup_expired_sessions()
        
        asyncio.run(_cleanup())
    
    @rule()
    def get_statistics(self):
        """Get manager statistics."""
        async def _stats():
            stats = await self.manager.get_statistics()
            assert isinstance(stats, dict)
            assert "active_sessions" in stats
            assert "total_created" in stats
            assert stats["total_created"] >= len(self.created_sessions)
        
        asyncio.run(_stats())
    
    @invariant()
    def session_consistency(self):
        """Verify session state consistency."""
        if self.manager is None:
            return
        
        # Active sessions should not exceed max
        assert len(self.manager.active_sessions) <= self.manager.max_sessions
        
        # Session locks should match active sessions
        assert len(self.manager.session_locks) == len(self.manager.active_sessions)
        
        # All active session IDs should have corresponding locks
        for session_id in self.manager.active_sessions:
            assert session_id in self.manager.session_locks
    
    def teardown(self):
        """Clean up resources."""
        if self.manager:
            asyncio.run(self.manager.shutdown())
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Stateful test
TestSessionStateMachine = SessionManagementStateMachine.TestCase


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])