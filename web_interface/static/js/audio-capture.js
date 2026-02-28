/**
 * AudioCaptureInterface - Web Audio API integration for real-time audio capture
 * Implements Requirements: 1.1, 8.1, 8.5
 */
class AudioCaptureInterface {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.dataArray = null;
        this.isRecording = false;
        this.audioChunks = [];
        
        // Audio quality monitoring
        this.qualityThresholds = {
            good: 0.7,
            warning: 0.4,
            poor: 0.2
        };
        
        // UI elements
        this.startButton = null;
        this.stopButton = null;
        this.qualityIndicator = null;
        this.qualityBar = null;
        this.audioLevels = null;
        
        // WebSocket connection (will be set by WebSocketClient)
        this.websocket = null;
        
        // Audio processing settings
        this.sampleRate = 16000; // Optimal for speech recognition
        this.bufferSize = 4096;
        this.recordingOptions = {
            mimeType: 'audio/webm;codecs=opus',
            audioBitsPerSecond: 128000
        };
        
        this.init();
    }
    
    /**
     * Initialize the audio capture interface
     */
    init() {
        this.setupUIElements();
        this.setupEventListeners();
        this.createAudioLevelVisualization();
    }
    
    /**
     * Check if getUserMedia is supported with fallbacks
     */
    isGetUserMediaSupported() {
        // Check for modern getUserMedia
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            return true;
        }
        
        // Check for legacy getUserMedia
        if (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Get user media with fallbacks for older browsers
     */
    async getUserMedia(constraints) {
        // Try modern API first
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            return await navigator.mediaDevices.getUserMedia(constraints);
        }
        
        // Fallback to legacy API
        const getUserMedia = navigator.getUserMedia || 
                           navigator.webkitGetUserMedia || 
                           navigator.mozGetUserMedia;
        
        if (getUserMedia) {
            return new Promise((resolve, reject) => {
                getUserMedia.call(navigator, constraints, resolve, reject);
            });
        }
        
        throw new Error('getUserMedia is not supported in this browser');
    }
    
    /**
     * Set up UI element references
     */
    setupUIElements() {
        this.startButton = document.getElementById('start-recording');
        this.stopButton = document.getElementById('stop-recording');
        this.qualityIndicator = document.getElementById('quality-indicator');
        this.qualityBar = this.qualityIndicator?.querySelector('.quality-bar');
        
        // Create audio levels container if it doesn't exist
        const audioControls = document.querySelector('.audio-controls');
        if (audioControls && !document.querySelector('.audio-levels')) {
            const levelsContainer = document.createElement('div');
            levelsContainer.className = 'audio-levels';
            levelsContainer.id = 'audio-levels';
            audioControls.appendChild(levelsContainer);
            this.audioLevels = levelsContainer;
        }
    }
    
    /**
     * Set up event listeners for UI controls
     */
    setupEventListeners() {
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startCapture());
        }
        
        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.stopCapture());
        }
    }
    
    /**
     * Create visual audio level bars
     */
    createAudioLevelVisualization() {
        if (!this.audioLevels) return;
        
        // Create 20 level bars for visualization
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'level-bar';
            bar.style.height = '2px';
            this.audioLevels.appendChild(bar);
        }
    }
    
    /**
     * Start audio capture with quality monitoring
     * Implements Requirement 1.1: Audio capture with minimal latency
     */
    async startCapture() {
        try {
            console.log('Starting audio capture...');
            const startTime = performance.now();
            
            // Check for HTTPS requirement
            if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
                throw new Error('getUserMedia requires HTTPS or localhost');
            }
            
            // Check for getUserMedia support with fallbacks
            if (!this.isGetUserMediaSupported()) {
                throw new Error('getUserMedia is not supported in this browser or requires HTTPS');
            }
            
            // Request microphone access
            const stream = await this.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Check latency (Requirement 1.1: under 100ms)
            const latency = performance.now() - startTime;
            console.log(`Audio capture latency: ${latency.toFixed(2)}ms`);
            
            // Set up audio context and analyser
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.8;
            
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);
            
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            // Set up MediaRecorder for audio streaming
            this.mediaRecorder = new MediaRecorder(stream, this.recordingOptions);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    // Stream audio data via WebSocket if available
                    if (this.websocket && this.websocket.isConnected()) {
                        this.websocket.sendAudioChunk(event.data);
                    }
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.handleRecordingStop();
            };
            
            // Start recording
            this.mediaRecorder.start(100); // 100ms chunks for real-time processing
            this.isRecording = true;
            
            // Update UI
            this.updateUIState(true);
            
            // Start quality monitoring
            this.startQualityMonitoring();
            
            console.log('Audio capture started successfully');
            
        } catch (error) {
            console.error('Error starting audio capture:', error);
            this.handleCaptureError(error);
        }
    }
    
    /**
     * Stop audio capture
     */
    async stopCapture() {
        try {
            console.log('Stopping audio capture...');
            
            if (this.mediaRecorder && this.isRecording) {
                this.mediaRecorder.stop();
            }
            
            if (this.audioContext) {
                await this.audioContext.close();
                this.audioContext = null;
            }
            
            this.isRecording = false;
            this.updateUIState(false);
            this.stopQualityMonitoring();
            
            console.log('Audio capture stopped');
            
        } catch (error) {
            console.error('Error stopping audio capture:', error);
        }
    }
    
    /**
     * Handle recording stop event
     */
    handleRecordingStop() {
        if (this.audioChunks.length > 0) {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            
            // Send final audio data via WebSocket if available
            if (this.websocket && this.websocket.isConnected()) {
                this.websocket.sendFinalAudio(audioBlob);
            }
            
            // Trigger custom event for other components
            const event = new CustomEvent('audioRecordingComplete', {
                detail: { audioBlob, chunks: this.audioChunks.length }
            });
            document.dispatchEvent(event);
        }
        
        this.audioChunks = [];
    }
    
    /**
     * Start real-time audio quality monitoring
     * Implements Requirements 8.1, 8.5: Audio quality monitoring and real-time updates
     */
    startQualityMonitoring() {
        if (!this.analyser) return;
        
        const monitor = () => {
            if (!this.isRecording) return;
            
            // Get frequency data
            this.analyser.getByteFrequencyData(this.dataArray);
            
            // Calculate audio levels and quality metrics
            const audioLevel = this.calculateAudioLevel();
            const signalQuality = this.calculateSignalQuality();
            const noiseLevel = this.calculateNoiseLevel();
            
            // Update visualizations
            this.updateAudioLevelVisualization(audioLevel);
            this.updateQualityIndicator(signalQuality);
            this.updateQualityStatus(signalQuality, noiseLevel);
            
            // Continue monitoring
            requestAnimationFrame(monitor);
        };
        
        monitor();
    }
    
    /**
     * Stop quality monitoring
     */
    stopQualityMonitoring() {
        // Reset visualizations
        this.updateAudioLevelVisualization(0);
        this.updateQualityIndicator(0);
        this.updateQualityStatus(0, 0);
    }
    
    /**
     * Calculate current audio level (0-1)
     */
    calculateAudioLevel() {
        if (!this.dataArray) return 0;
        
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        
        return Math.min(sum / (this.dataArray.length * 255), 1);
    }
    
    /**
     * Calculate signal quality based on frequency analysis
     */
    calculateSignalQuality() {
        if (!this.dataArray) return 0;
        
        // Focus on speech frequency range (300-3400 Hz)
        const speechStart = Math.floor(300 * this.dataArray.length / (this.audioContext.sampleRate / 2));
        const speechEnd = Math.floor(3400 * this.dataArray.length / (this.audioContext.sampleRate / 2));
        
        let speechEnergy = 0;
        let totalEnergy = 0;
        
        for (let i = 0; i < this.dataArray.length; i++) {
            const value = this.dataArray[i];
            totalEnergy += value;
            
            if (i >= speechStart && i <= speechEnd) {
                speechEnergy += value;
            }
        }
        
        if (totalEnergy === 0) return 0;
        
        // Quality is the ratio of speech energy to total energy
        return Math.min(speechEnergy / totalEnergy, 1);
    }
    
    /**
     * Calculate noise level
     */
    calculateNoiseLevel() {
        if (!this.dataArray) return 0;
        
        // Estimate noise from low-frequency components
        const noiseEnd = Math.floor(200 * this.dataArray.length / (this.audioContext.sampleRate / 2));
        
        let noiseEnergy = 0;
        for (let i = 0; i < noiseEnd; i++) {
            noiseEnergy += this.dataArray[i];
        }
        
        return Math.min(noiseEnergy / (noiseEnd * 255), 1);
    }
    
    /**
     * Update audio level visualization bars
     */
    updateAudioLevelVisualization(level) {
        const bars = this.audioLevels?.querySelectorAll('.level-bar');
        if (!bars) return;
        
        const activeBars = Math.floor(level * bars.length);
        
        bars.forEach((bar, index) => {
            const height = index < activeBars ? `${Math.min(40, (index + 1) * 2)}px` : '2px';
            bar.style.height = height;
            
            // Color coding based on level
            bar.className = 'level-bar';
            if (index < activeBars) {
                if (index / bars.length > 0.8) {
                    bar.classList.add('peak');
                } else if (index / bars.length > 0.6) {
                    bar.classList.add('active');
                }
            }
        });
    }
    
    /**
     * Update quality indicator bar
     */
    updateQualityIndicator(quality) {
        if (!this.qualityBar) return;
        
        const percentage = Math.floor(quality * 100);
        this.qualityBar.style.width = `${percentage}%`;
    }
    
    /**
     * Update quality status display
     * Implements Requirement 8.1: Visual feedback about audio quality
     */
    updateQualityStatus(signalQuality, noiseLevel) {
        // Create or update quality status element
        let statusElement = document.querySelector('.quality-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.className = 'quality-status';
            this.qualityIndicator?.parentElement?.appendChild(statusElement);
        }
        
        let status, message, className;
        
        if (signalQuality >= this.qualityThresholds.good && noiseLevel < 0.3) {
            status = 'Good';
            message = 'Excellent audio quality';
            className = 'good';
        } else if (signalQuality >= this.qualityThresholds.warning && noiseLevel < 0.5) {
            status = 'Fair';
            message = 'Acceptable audio quality';
            className = 'warning';
        } else {
            status = 'Poor';
            message = 'Audio quality issues detected';
            className = 'poor';
        }
        
        statusElement.className = `quality-status ${className}`;
        statusElement.innerHTML = `
            <div class="quality-icon"></div>
            <span>${status}: ${message}</span>
        `;
        
        // Show quality warnings if needed
        if (signalQuality < this.qualityThresholds.warning) {
            this.showQualityWarning('Low signal quality detected. Please speak closer to the microphone.');
        } else if (noiseLevel > 0.5) {
            this.showQualityWarning('High background noise detected. Please reduce ambient noise.');
        }
    }
    
    /**
     * Show quality warning message
     * Implements Requirement 8.2: Warnings about potential transcription issues
     */
    showQualityWarning(message) {
        // Throttle warnings to avoid spam
        if (this.lastWarningTime && Date.now() - this.lastWarningTime < 5000) {
            return;
        }
        
        this.lastWarningTime = Date.now();
        
        // Create warning notification
        const warning = document.createElement('div');
        warning.className = 'quality-warning';
        warning.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ffc107;
            color: #212529;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
            animation: slideIn 0.3s ease;
        `;
        warning.textContent = message;
        
        document.body.appendChild(warning);
        
        // Auto-remove after 4 seconds
        setTimeout(() => {
            if (warning.parentElement) {
                warning.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => warning.remove(), 300);
            }
        }, 4000);
    }
    
    /**
     * Update UI state based on recording status
     */
    updateUIState(isRecording) {
        if (this.startButton) {
            this.startButton.disabled = isRecording;
            this.startButton.textContent = isRecording ? 'Recording...' : 'Start Recording';
        }
        
        if (this.stopButton) {
            this.stopButton.disabled = !isRecording;
        }
        
        // Update transcription area visual state
        const transcriptionText = document.getElementById('transcription-text');
        if (transcriptionText) {
            if (isRecording) {
                transcriptionText.classList.add('active');
                transcriptionText.textContent = 'Listening...';
            } else {
                transcriptionText.classList.remove('active');
            }
        }
    }
    
    /**
     * Handle capture errors
     */
    handleCaptureError(error) {
        console.error('Audio capture error:', error);
        
        let message = 'Failed to start audio capture. ';
        let suggestions = [];
        
        if (error.message.includes('getUserMedia is not supported')) {
            message = 'Audio recording is not supported in this browser. ';
            suggestions = [
                'Try using Chrome, Firefox, or Safari',
                'Make sure you\'re using HTTPS (not HTTP)',
                'Check if your browser supports WebRTC'
            ];
        } else if (error.message.includes('requires HTTPS')) {
            message = 'Audio recording requires a secure connection. ';
            suggestions = [
                'Access the site using HTTPS instead of HTTP',
                'Use localhost for development',
                'Contact your administrator about SSL certificates'
            ];
        } else if (error.name === 'NotAllowedError') {
            message = 'Microphone access was denied. ';
            suggestions = [
                'Click the microphone icon in your browser\'s address bar',
                'Allow microphone access when prompted',
                'Check your browser\'s privacy settings'
            ];
        } else if (error.name === 'NotFoundError') {
            message = 'No microphone found. ';
            suggestions = [
                'Connect a microphone to your device',
                'Check your system\'s audio settings',
                'Make sure your microphone is not being used by another application'
            ];
        } else if (error.name === 'NotSupportedError') {
            message = 'Your browser does not support audio recording. ';
            suggestions = [
                'Update your browser to the latest version',
                'Try using Chrome, Firefox, or Safari',
                'Enable WebRTC in your browser settings'
            ];
        } else {
            message = 'An unexpected error occurred. ';
            suggestions = [
                'Check your microphone connection',
                'Refresh the page and try again',
                'Try using a different browser'
            ];
        }
        
        this.showError(message, suggestions);
        this.updateUIState(false);
    }
    
    /**
     * Show error message to user
     */
    showError(message, suggestions = []) {
        const error = document.createElement('div');
        error.className = 'error-notification';
        error.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #dc3545;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 350px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
        let html = `<div style="font-weight: 600; margin-bottom: 0.5rem;">🎤 Audio Capture Error</div>`;
        html += `<div style="margin-bottom: 0.5rem;">${message}</div>`;
        
        if (suggestions.length > 0) {
            html += `<div style="font-size: 0.9em; opacity: 0.9;">`;
            html += `<div style="margin-bottom: 0.25rem;">Try these solutions:</div>`;
            html += `<ul style="margin: 0; padding-left: 1rem;">`;
            suggestions.forEach(suggestion => {
                html += `<li style="margin-bottom: 0.25rem;">${suggestion}</li>`;
            });
            html += `</ul></div>`;
        }
        
        html += `<button onclick="this.parentElement.remove()" style="
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            cursor: pointer;
            margin-top: 0.5rem;
            font-size: 0.8rem;
        ">Dismiss</button>`;
        
        error.innerHTML = html;
        
        document.body.appendChild(error);
        
        setTimeout(() => {
            if (error.parentElement) {
                error.remove();
            }
        }, 5000);
    }
    
    /**
     * Set WebSocket connection for audio streaming
     */
    setWebSocket(websocket) {
        this.websocket = websocket;
    }
    
    /**
     * Get current recording state
     */
    isCurrentlyRecording() {
        return this.isRecording;
    }
    
    /**
     * Get audio context for external use
     */
    getAudioContext() {
        return this.audioContext;
    }
    
    /**
     * Get current audio quality metrics
     */
    getQualityMetrics() {
        if (!this.isRecording || !this.dataArray) {
            return { level: 0, quality: 0, noise: 0 };
        }
        
        return {
            level: this.calculateAudioLevel(),
            quality: this.calculateSignalQuality(),
            noise: this.calculateNoiseLevel()
        };
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Export for use in other modules
window.AudioCaptureInterface = AudioCaptureInterface;