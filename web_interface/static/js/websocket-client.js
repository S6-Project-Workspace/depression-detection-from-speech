/**
 * WebSocketClient - Real-time communication with backend
 * Implements Requirements: 6.1, 6.2, 6.3
 */
class WebSocketClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds
        this.sessionId = null;
        this.messageQueue = [];
        
        // Event handlers
        this.onConnectionOpen = null;
        this.onConnectionClose = null;
        this.onConnectionError = null;
        this.onTranscriptionUpdate = null;
        this.onAnalysisResult = null;
        this.onProcessingComplete = null;
        
        // Connection state
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, reconnecting
        
        this.init();
    }
    
    /**
     * Initialize WebSocket client
     */
    init() {
        this.generateSessionId();
        this.setupEventHandlers();
    }
    
    /**
     * Generate unique session ID
     */
    generateSessionId() {
        this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Set up default event handlers
     */
    setupEventHandlers() {
        // Default transcription handler
        this.onTranscriptionUpdate = (data) => {
            const transcriptionElement = document.getElementById('transcription-text');
            if (transcriptionElement && data.text) {
                if (data.is_final) {
                    transcriptionElement.innerHTML += `<span class="final-text">${data.text}</span> `;
                } else {
                    // Update or add interim text
                    let interimElement = transcriptionElement.querySelector('.interim-text');
                    if (!interimElement) {
                        interimElement = document.createElement('span');
                        interimElement.className = 'interim-text';
                        transcriptionElement.appendChild(interimElement);
                    }
                    interimElement.textContent = data.text;
                }
                
                // Auto-scroll to bottom
                transcriptionElement.scrollTop = transcriptionElement.scrollHeight;
            }
        };
        
        // Default analysis result handler
        this.onAnalysisResult = (data) => {
            console.log('Analysis result received:', data);
            
            // Trigger custom event for visualization components
            const event = new CustomEvent('analysisResultReceived', {
                detail: data
            });
            document.dispatchEvent(event);
        };
        
        // Default processing complete handler
        this.onProcessingComplete = (data) => {
            console.log('Processing complete:', data);
            
            // Hide loading indicators
            this.hideLoadingIndicator();
            
            // Trigger custom event
            const event = new CustomEvent('processingComplete', {
                detail: data
            });
            document.dispatchEvent(event);
        };
    }
    
    /**
     * Connect to WebSocket server
     * Implements Requirement 6.1: Establish persistent connections
     */
    async connect() {
        if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
            return;
        }
        
        this.connectionState = 'connecting';
        this.showConnectionStatus('Connecting...');
        
        try {
            // Determine WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            const wsUrl = `${protocol}//${host}/ws/audio?session_id=${this.sessionId}`;
            
            console.log('Connecting to WebSocket:', wsUrl);
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = (event) => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.connectionState = 'connected';
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000; // Reset delay
                
                this.showConnectionStatus('Connected', 'success');
                
                // Process queued messages
                this.processMessageQueue();
                
                if (this.onConnectionOpen) {
                    this.onConnectionOpen(event);
                }
            };
            
            this.websocket.onmessage = (event) => {
                this.handleMessage(event);
            };
            
            this.websocket.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                this.connectionState = 'disconnected';
                
                this.showConnectionStatus('Disconnected', 'error');
                
                if (this.onConnectionClose) {
                    this.onConnectionClose(event);
                }
                
                // Attempt reconnection if not intentionally closed
                if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.attemptReconnection();
                }
            };
            
            this.websocket.onerror = (event) => {
                console.error('WebSocket error:', event);
                this.showConnectionStatus('Connection error', 'error');
                
                if (this.onConnectionError) {
                    this.onConnectionError(event);
                }
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.connectionState = 'disconnected';
            this.showConnectionStatus('Connection failed', 'error');
            
            // Provide more specific error information
            let errorMessage = 'Connection failed';
            if (error.name === 'SecurityError') {
                errorMessage = 'Security error - check HTTPS/WSS configuration';
            } else if (error.name === 'NetworkError') {
                errorMessage = 'Network error - check internet connection';
            } else if (error.message) {
                errorMessage = `Connection failed: ${error.message}`;
            }
            
            console.log(`WebSocket error details: ${errorMessage}`);
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.attemptReconnection();
            } else {
                // Show user-friendly error after max attempts
                this.showConnectionStatus(`${errorMessage} - Click retry to try again`, 'error');
            }
        }
    }
    
    /**
     * Attempt reconnection with exponential backoff
     * Implements Requirement 6.3: Automatic reconnection with exponential backoff
     */
    attemptReconnection() {
        if (this.connectionState === 'reconnecting') {
            return;
        }
        
        this.connectionState = 'reconnecting';
        this.reconnectAttempts++;
        
        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
            this.maxReconnectDelay
        );
        
        console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
        this.showConnectionStatus(`Reconnecting in ${Math.ceil(delay/1000)}s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'warning');
        
        setTimeout(() => {
            if (this.reconnectAttempts <= this.maxReconnectAttempts) {
                this.connect();
            } else {
                this.showConnectionStatus('Connection failed - max attempts reached', 'error');
                this.showReconnectionOptions();
            }
        }, delay);
    }
    
    /**
     * Show reconnection options to user
     */
    showReconnectionOptions() {
        const notification = document.createElement('div');
        notification.className = 'reconnection-notification';
        notification.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 2000;
            text-align: center;
            max-width: 400px;
        `;
        
        notification.innerHTML = `
            <h3>Connection Lost</h3>
            <p>Unable to reconnect to the server. Please check your internet connection.</p>
            <div style="margin-top: 1rem;">
                <button id="retry-connection" class="btn btn-primary" style="margin-right: 0.5rem;">Retry Connection</button>
                <button id="refresh-page" class="btn btn-secondary">Refresh Page</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Add event listeners
        document.getElementById('retry-connection').addEventListener('click', () => {
            notification.remove();
            this.reconnectAttempts = 0;
            this.connect();
        });
        
        document.getElementById('refresh-page').addEventListener('click', () => {
            window.location.reload();
        });
    }
    
    /**
     * Handle incoming WebSocket messages
     * Implements Requirement 6.2: Real-time result streaming
     */
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);
            
            switch (data.type) {
                case 'transcription_update':
                    if (this.onTranscriptionUpdate) {
                        this.onTranscriptionUpdate(data);
                    }
                    break;
                    
                case 'analysis_result':
                    if (this.onAnalysisResult) {
                        this.onAnalysisResult(data);
                    }
                    break;
                    
                case 'processing_complete':
                    if (this.onProcessingComplete) {
                        this.onProcessingComplete(data);
                    }
                    break;
                    
                case 'error':
                    this.handleServerError(data);
                    break;
                    
                case 'session_initialized':
                    console.log('Session initialized:', data.session_id);
                    break;
                    
                case 'quality_feedback':
                    this.handleQualityFeedback(data);
                    break;
                    
                case 'connection_established':
                    console.log('✅ Connection established:', data);
                    // Connection is already handled in onopen, just acknowledge
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type);
            }
            
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    /**
     * Handle server error messages
     */
    handleServerError(data) {
        console.error('Server error:', data.message);
        
        const errorNotification = document.createElement('div');
        errorNotification.className = 'error-notification';
        errorNotification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #dc3545;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        `;
        errorNotification.textContent = `Server Error: ${data.message}`;
        
        document.body.appendChild(errorNotification);
        
        setTimeout(() => {
            if (errorNotification.parentElement) {
                errorNotification.remove();
            }
        }, 5000);
    }
    
    /**
     * Handle quality feedback from server
     */
    handleQualityFeedback(data) {
        // Trigger custom event for audio capture interface
        const event = new CustomEvent('qualityFeedback', {
            detail: data
        });
        document.dispatchEvent(event);
    }
    
    /**
     * Send audio chunk to server
     */
    sendAudioChunk(audioData) {
        if (!this.isConnected) {
            console.warn('Cannot send audio chunk - not connected');
            return;
        }
        
        // Convert Blob to ArrayBuffer for transmission
        if (audioData instanceof Blob) {
            const reader = new FileReader();
            reader.onload = () => {
                const message = {
                    type: 'audio_chunk',
                    session_id: this.sessionId,
                    data: Array.from(new Uint8Array(reader.result)),
                    timestamp: Date.now()
                };
                
                this.sendMessage(message);
            };
            reader.readAsArrayBuffer(audioData);
        } else {
            console.error('Invalid audio data type');
        }
    }
    
    /**
     * Send final audio data
     */
    sendFinalAudio(audioBlob) {
        if (!this.isConnected) {
            console.warn('Cannot send final audio - not connected');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = () => {
            const message = {
                type: 'audio_final',
                session_id: this.sessionId,
                data: Array.from(new Uint8Array(reader.result)),
                timestamp: Date.now()
            };
            
            this.sendMessage(message);
        };
        reader.readAsArrayBuffer(audioBlob);
    }
    
    /**
     * Send text for analysis
     */
    sendTextForAnalysis(text) {
        const message = {
            type: 'text_analysis',
            session_id: this.sessionId,
            text: text,
            timestamp: Date.now()
        };
        
        this.sendMessage(message);
    }
    
    /**
     * Send message to server
     */
    sendMessage(message) {
        if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            try {
                this.websocket.send(JSON.stringify(message));
            } catch (error) {
                console.error('Error sending message:', error);
                this.queueMessage(message);
            }
        } else {
            this.queueMessage(message);
        }
    }
    
    /**
     * Queue message for later transmission
     */
    queueMessage(message) {
        this.messageQueue.push(message);
        console.log('Message queued for transmission');
    }
    
    /**
     * Process queued messages
     */
    processMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift();
            this.sendMessage(message);
        }
    }
    
    /**
     * Disconnect from server
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close(1000, 'Client disconnect');
            this.websocket = null;
        }
        
        this.isConnected = false;
        this.connectionState = 'disconnected';
        this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
    }
    
    /**
     * Show connection status to user
     */
    showConnectionStatus(message, type = 'info') {
        // Create or update status indicator
        let statusElement = document.querySelector('.connection-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.className = 'connection-status';
            statusElement.style.cssText = `
                position: fixed;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                z-index: 1000;
                transition: all 0.3s ease;
            `;
            document.body.appendChild(statusElement);
        }
        
        // Update styling based on type
        let backgroundColor, color;
        switch (type) {
            case 'success':
                backgroundColor = '#28a745';
                color = 'white';
                break;
            case 'warning':
                backgroundColor = '#ffc107';
                color = '#212529';
                break;
            case 'error':
                backgroundColor = '#dc3545';
                color = 'white';
                break;
            default:
                backgroundColor = '#6c757d';
                color = 'white';
        }
        
        statusElement.style.backgroundColor = backgroundColor;
        statusElement.style.color = color;
        statusElement.textContent = message;
        
        // Auto-hide success messages
        if (type === 'success') {
            setTimeout(() => {
                if (statusElement.parentElement) {
                    statusElement.style.opacity = '0';
                    setTimeout(() => statusElement.remove(), 300);
                }
            }, 3000);
        }
    }
    
    /**
     * Show loading indicator
     */
    showLoadingIndicator() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('active');
        }
    }
    
    /**
     * Hide loading indicator
     */
    hideLoadingIndicator() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }
    
    /**
     * Get connection state
     */
    getConnectionState() {
        return this.connectionState;
    }
    
    /**
     * Check if connected
     */
    getConnectionStatus() {
        return this.isConnected && this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
    
    /**
     * Get session ID
     */
    getSessionId() {
        return this.sessionId;
    }
    
    /**
     * Set custom event handlers
     */
    setEventHandlers(handlers) {
        if (handlers.onConnectionOpen) this.onConnectionOpen = handlers.onConnectionOpen;
        if (handlers.onConnectionClose) this.onConnectionClose = handlers.onConnectionClose;
        if (handlers.onConnectionError) this.onConnectionError = handlers.onConnectionError;
        if (handlers.onTranscriptionUpdate) this.onTranscriptionUpdate = handlers.onTranscriptionUpdate;
        if (handlers.onAnalysisResult) this.onAnalysisResult = handlers.onAnalysisResult;
        if (handlers.onProcessingComplete) this.onProcessingComplete = handlers.onProcessingComplete;
    }
}

// Add CSS for connection status and notifications (only if not already added)
if (!document.getElementById('websocket-client-styles')) {
    const style = document.createElement('style');
    style.id = 'websocket-client-styles';
    style.textContent = `
        .connection-status {
            animation: slideDown 0.3s ease;
        }
        
        @keyframes slideDown {
            from { transform: translate(-50%, -100%); opacity: 0; }
            to { transform: translate(-50%, 0); opacity: 1; }
        }
        
        .final-text {
            color: #333;
            font-weight: 500;
        }
        
        .interim-text {
            color: #6c757d;
            font-style: italic;
        }
        
        .reconnection-notification {
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
            to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }
    `;
    document.head.appendChild(style);
}

// Export for use in other modules
window.WebSocketClient = WebSocketClient;