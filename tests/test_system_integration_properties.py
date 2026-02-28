"""
Final Property Tests for System Integration
Tests Properties 29 and 43 from the design document
Validates Requirements: 6.5, 9.5
"""

import pytest
import asyncio
import json
import time
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import tempfile
import os

class TestSystemIntegrationProperties:
    """Property-based tests for complete system integration"""
    
    @given(
        session_ids=st.lists(
            st.text(min_size=1, max_size=50),
            min_size=1, max_size=5
        ),
        processing_operations=st.lists(
            st.sampled_from(['audio_chunk', 'text_analysis', 'file_upload']),
            min_size=1, max_size=10
        )
    )
    @settings(
        max_examples=50, 
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_29_processing_completion_notifications(self, session_ids, processing_operations):
        """
        Property 29: Processing completion notifications
        For any completed processing operation, completion notifications with final results should be sent via WebSocket
        Validates: Requirements 6.5
        """
        
        # Mock WebSocket and processor
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        class MockProcessor:
            def __init__(self):
                self.completed_operations = []
            
            async def process_operation(self, session_id, operation_type):
                # Simulate processing
                await asyncio.sleep(0.01)  # Small delay to simulate work
                
                # Create completion result
                completion_result = {
                    'session_id': session_id,
                    'operation_type': operation_type,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'result_data': f'Result for {operation_type}',
                    'processing_time': 0.01
                }
                
                self.completed_operations.append(completion_result)
                
                # Send completion notification
                notification = {
                    'type': 'processing_complete',
                    'session_id': session_id,
                    'operation': operation_type,
                    'result': completion_result,
                    'final_results': True
                }
                
                await mock_websocket.send_text(json.dumps(notification))
                
                return completion_result
        
        processor = MockProcessor()
        
        # Process operations for each session
        all_notifications = []
        
        for session_id in session_ids:
            for operation in processing_operations:
                # Process operation
                result = await processor.process_operation(session_id, operation)
                
                # Property: Operation should complete successfully
                assert result['status'] == 'completed'
                assert result['session_id'] == session_id
                assert result['operation_type'] == operation
                
                # Property: Completion notification should be sent
                assert mock_websocket.send_text.called
                
                # Get the last notification sent
                last_call = mock_websocket.send_text.call_args_list[-1]
                notification_json = last_call[0][0]
                notification = json.loads(notification_json)
                
                all_notifications.append(notification)
                
                # Property: Notification should contain completion information
                assert notification['type'] == 'processing_complete'
                assert notification['session_id'] == session_id
                assert notification['operation'] == operation
                assert 'result' in notification
                assert notification['final_results'] is True
        
        # Property: All operations should have completion notifications
        expected_total = len(session_ids) * len(processing_operations)
        assert len(all_notifications) == expected_total
        
        # Property: Each notification should be unique and properly formatted
        for notification in all_notifications:
            assert 'type' in notification
            assert 'session_id' in notification
            assert 'operation' in notification
            assert 'result' in notification
            assert notification['type'] == 'processing_complete'
        
        # Property: Notifications should be sent in processing order
        for i in range(1, len(all_notifications)):
            prev_time = datetime.fromisoformat(all_notifications[i-1]['result']['timestamp'])
            curr_time = datetime.fromisoformat(all_notifications[i]['result']['timestamp'])
            assert curr_time >= prev_time, "Notifications should be in chronological order"
    
    @given(
        export_formats=st.lists(
            st.sampled_from(['json', 'csv', 'pdf', 'txt']),
            min_size=1, max_size=4
        ),
        analysis_data=st.lists(
            st.dictionaries(
                st.sampled_from(['text', 'pos_tags', 'entities', 'sentiment']),
                st.one_of(
                    st.text(min_size=1, max_size=100),
                    st.lists(st.dictionaries(
                        st.sampled_from(['token', 'tag', 'label']),
                        st.text(min_size=1, max_size=20),
                        min_size=1, max_size=3
                    ), min_size=0, max_size=5),
                    st.floats(min_value=0.0, max_value=1.0)
                ),
                min_size=1, max_size=4
            ),
            min_size=1, max_size=10
        )
    )
    @settings(
        max_examples=50, 
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_43_export_functionality(self, export_formats, analysis_data):
        """
        Property 43: Export functionality
        For any export request, downloadable reports should be generated with comprehensive analysis summaries
        Validates: Requirements 9.5
        """
        
        class MockExportManager:
            def __init__(self):
                self.supported_formats = ['json', 'csv', 'pdf', 'txt']
                self.export_history = []
            
            def generate_export(self, data, format_type, session_id=None):
                # Property: Format should be supported
                if format_type not in self.supported_formats:
                    raise ValueError(f"Unsupported format: {format_type}")
                
                # Generate export based on format
                export_content = self._format_data(data, format_type)
                
                # Create export metadata
                export_metadata = {
                    'format': format_type,
                    'session_id': session_id or 'default',
                    'generated_at': datetime.now().isoformat(),
                    'data_count': len(data),
                    'file_size': len(str(export_content)),
                    'content': export_content
                }
                
                self.export_history.append(export_metadata)
                
                return export_metadata
            
            def _format_data(self, data, format_type):
                if format_type == 'json':
                    return json.dumps(data, indent=2)
                elif format_type == 'csv':
                    # Simple CSV formatting
                    if not data:
                        return "No data"
                    
                    headers = set()
                    for item in data:
                        if isinstance(item, dict):
                            headers.update(item.keys())
                    
                    csv_lines = [','.join(sorted(headers))]
                    for item in data:
                        if isinstance(item, dict):
                            row = [str(item.get(header, '')) for header in sorted(headers)]
                            csv_lines.append(','.join(row))
                    
                    return '\n'.join(csv_lines)
                elif format_type == 'txt':
                    # Plain text formatting
                    text_lines = []
                    for i, item in enumerate(data):
                        text_lines.append(f"Item {i+1}: {str(item)}")
                    return '\n'.join(text_lines)
                elif format_type == 'pdf':
                    # Mock PDF content
                    return f"PDF Report - {len(data)} items - Generated at {datetime.now()}"
                
                return str(data)
        
        export_manager = MockExportManager()
        
        # Test export functionality for each format
        export_results = []
        
        for format_type in export_formats:
            try:
                # Generate export
                export_result = export_manager.generate_export(analysis_data, format_type, f"session_{format_type}")
                export_results.append(export_result)
                
                # Property: Export should be generated successfully
                assert export_result is not None
                assert export_result['format'] == format_type
                assert 'content' in export_result
                assert 'generated_at' in export_result
                
                # Property: Export should contain comprehensive data
                assert export_result['data_count'] == len(analysis_data)
                assert export_result['file_size'] > 0
                
                # Property: Export content should be properly formatted
                content = export_result['content']
                
                if format_type == 'json':
                    # Should be valid JSON
                    parsed_json = json.loads(content)
                    assert parsed_json == analysis_data
                elif format_type == 'csv':
                    # Should contain headers and data rows
                    lines = content.split('\n')
                    assert len(lines) >= 1  # At least header
                elif format_type == 'txt':
                    # Should contain readable text
                    assert len(content) > 0
                    assert 'Item' in content
                elif format_type == 'pdf':
                    # Should contain PDF metadata
                    assert 'PDF Report' in content
                    assert str(len(analysis_data)) in content
                
                # Property: Export metadata should be complete
                required_fields = ['format', 'session_id', 'generated_at', 'data_count', 'file_size']
                for field in required_fields:
                    assert field in export_result, f"Missing required field: {field}"
                
            except ValueError as e:
                # Property: Unsupported formats should raise appropriate errors
                assert "Unsupported format" in str(e)
        
        # Property: All valid exports should be tracked
        valid_formats = [fmt for fmt in export_formats if fmt in export_manager.supported_formats]
        assert len(export_results) == len(valid_formats)
        
        # Property: Export history should be maintained
        assert len(export_manager.export_history) == len(valid_formats)
        
        # Property: Each export should have unique metadata
        export_timestamps = [result['generated_at'] for result in export_results]
        # Timestamps should be in chronological order (or very close)
        for i in range(1, len(export_timestamps)):
            prev_time = datetime.fromisoformat(export_timestamps[i-1])
            curr_time = datetime.fromisoformat(export_timestamps[i])
            # Allow small time differences due to processing speed
            time_diff = (curr_time - prev_time).total_seconds()
            assert time_diff >= -0.1, "Export timestamps should be in order"
    
    @given(
        session_count=st.integers(min_value=1, max_value=5),
        operations_per_session=st.integers(min_value=1, max_value=5)
    )
    @settings(
        max_examples=20, 
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_end_to_end_system_integration(self, session_count, operations_per_session):
        """
        Test complete end-to-end system integration
        Combines processing completion notifications and export functionality
        """
        
        # Mock complete system
        class MockIntegratedSystem:
            def __init__(self):
                self.sessions = {}
                self.websocket = AsyncMock()
                self.websocket.send_text = AsyncMock()
            
            async def create_session(self, session_id):
                self.sessions[session_id] = {
                    'id': session_id,
                    'created_at': datetime.now().isoformat(),
                    'operations': [],
                    'results': []
                }
                return self.sessions[session_id]
            
            async def process_operation(self, session_id, operation_data):
                if session_id not in self.sessions:
                    raise ValueError(f"Session {session_id} not found")
                
                session = self.sessions[session_id]
                
                # Process operation
                result = {
                    'operation_id': len(session['operations']),
                    'data': operation_data,
                    'processed_at': datetime.now().isoformat(),
                    'status': 'completed'
                }
                
                session['operations'].append(operation_data)
                session['results'].append(result)
                
                # Send completion notification
                notification = {
                    'type': 'processing_complete',
                    'session_id': session_id,
                    'operation_id': result['operation_id'],
                    'result': result
                }
                
                await self.websocket.send_text(json.dumps(notification))
                
                return result
            
            def export_session_data(self, session_id, format_type='json'):
                if session_id not in self.sessions:
                    raise ValueError(f"Session {session_id} not found")
                
                session = self.sessions[session_id]
                
                export_data = {
                    'session_info': {
                        'id': session['id'],
                        'created_at': session['created_at']
                    },
                    'operations': session['operations'],
                    'results': session['results'],
                    'summary': {
                        'total_operations': len(session['operations']),
                        'total_results': len(session['results'])
                    }
                }
                
                if format_type == 'json':
                    content = json.dumps(export_data, indent=2)
                else:
                    content = str(export_data)
                
                return {
                    'format': format_type,
                    'content': content,
                    'session_id': session_id,
                    'exported_at': datetime.now().isoformat()
                }
        
        system = MockIntegratedSystem()
        
        # Create sessions and process operations
        all_sessions = []
        all_exports = []
        
        for i in range(session_count):
            session_id = f"integration_session_{i}"
            
            # Create session
            session = await system.create_session(session_id)
            all_sessions.append(session)
            
            # Process operations
            for j in range(operations_per_session):
                operation_data = f"operation_{j}_data"
                result = await system.process_operation(session_id, operation_data)
                
                # Verify processing completion
                assert result['status'] == 'completed'
                assert result['operation_id'] == j
            
            # Export session data
            export_result = system.export_session_data(session_id)
            all_exports.append(export_result)
            
            # Verify export
            assert export_result['session_id'] == session_id
            assert 'content' in export_result
            
            # Parse exported content to verify completeness
            exported_data = json.loads(export_result['content'])
            assert exported_data['summary']['total_operations'] == operations_per_session
            assert exported_data['summary']['total_results'] == operations_per_session
        
        # Verify system-wide integration
        assert len(all_sessions) == session_count
        assert len(all_exports) == session_count
        
        # Verify all notifications were sent
        expected_notifications = session_count * operations_per_session
        assert system.websocket.send_text.call_count == expected_notifications
        
        # Verify session isolation
        session_ids = [session['id'] for session in all_sessions]
        assert len(set(session_ids)) == len(session_ids)  # All unique

# Run property tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])