/**
 * Audio Quality Monitor
 * Implements real-time quality indicators, noise detection, and troubleshooting guidance
 * Requirements: 8.2, 8.3, 8.4
 */

class AudioQualityMonitor {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.isMonitoring = false;
        
        // Quality thresholds
        this.thresholds = {
            minSignalLevel: -40, // dB
            maxNoiseLevel: -60,  // dB
            minSNR: 10,          // Signal-to-noise ratio
            optimalRange: [-20, -10] // Optimal signal range in dB
        };
        
        // Quality indicators
        this.qualityIndicators = {
            signalLevel: 0,
            noiseLevel: 0,
            snr: 0,
            quality: 'unknown', // poor, fair, good, excellent
            warnings: []
        };
        
        this.callbacks = {
            onQualityUpdate: null,
            onWarning: null,
            onTroubleshooting: null
        };
        
        this.setupUI();
    }
    
    setupUI() {
        // Create quality monitoring UI elements
        const qualityContainer = document.createElement('div');
        qualityContainer.className = 'audio-quality-monitor';
        qualityContainer.innerHTML = `
            <div class="quality-header">
                <h4>Audio Quality</h4>
                <div class="quality-status" id="quality-status">
                    <span class="status-dot" id="quality-dot"></span>
                    <span class="status-text" id="quality-text">Not monitoring</span>
                </div>
            </div>
            
            <div class="quality-meters">
                <div class="meter-group">
                    <label>Signal Level</label>
                    <div class="level-meter">
                        <div class="meter-fill" id="signal-meter"></div>
                        <span class="meter-value" id="signal-value">-∞ dB</span>
                    </div>
                </div>
                
                <div class="meter-group">
                    <label>Noise Level</label>
                    <div class="level-meter">
                        <div class="meter-fill" id="noise-meter"></div>
                        <span class="meter-value" id="noise-value">-∞ dB</span>
                    </div>
                </div>
                
                <div class="meter-group">
                    <label>Signal-to-Noise Ratio</label>
                    <div class="level-meter">
                        <div class="meter-fill" id="snr-meter"></div>
                        <span class="meter-value" id="snr-value">0 dB</span>
                    </div>
                </div>
            </div>
            
            <div class="quality-warnings" id="quality-warnings" style="display: none;">
                <h5>⚠️ Quality Issues</h5>
                <ul id="warning-list"></ul>
            </div>
            
            <div class="troubleshooting-panel" id="troubleshooting-panel" style="display: none;">
                <h5>🔧 Troubleshooting</h5>
                <div class="diagnostic-info">
                    <div class="status-item">
                        <span class="status-label">Microphone Access:</span>
                        <span class="status-indicator" id="mic-access-status">Unknown</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Audio Input:</span>
                        <span class="status-indicator" id="audio-input-status">Unknown</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Background Noise:</span>
                        <span class="status-indicator" id="noise-status">Unknown</span>
                    </div>
                </div>
                <div class="suggestions" id="suggestions"></div>
            </div>
        `;
        
        // Add to audio controls if they exist
        const audioControls = document.querySelector('.audio-controls');
        if (audioControls) {
            audioControls.appendChild(qualityContainer);
        }
    }
    
    async startMonitoring(stream) {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            
            // Configure analyser
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;
            
            // Connect audio nodes
            this.microphone.connect(this.analyser);
            
            this.isMonitoring = true;
            this.updateQualityStatus('monitoring', 'Monitoring audio quality...');
            
            // Start monitoring loop
            this.monitoringLoop();
            
            // Update troubleshooting status
            this.updateTroubleshootingStatus('mic-access-status', 'good', 'Connected');
            this.updateTroubleshootingStatus('audio-input-status', 'good', 'Active');
            
        } catch (error) {
            console.error('Error starting audio quality monitoring:', error);
            this.updateQualityStatus('error', 'Monitoring failed');
            this.showTroubleshooting('microphone_error', error.message);
        }
    }
    
    stopMonitoring() {
        this.isMonitoring = false;
        
        if (this.microphone) {
            this.microphone.disconnect();
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        this.updateQualityStatus('stopped', 'Monitoring stopped');
        this.clearQualityMeters();
    }
    
    monitoringLoop() {
        if (!this.isMonitoring) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const timeDataArray = new Float32Array(bufferLength);
        
        // Get frequency and time domain data
        this.analyser.getByteFrequencyData(dataArray);
        this.analyser.getFloatTimeDomainData(timeDataArray);
        
        // Calculate audio metrics
        const metrics = this.calculateAudioMetrics(dataArray, timeDataArray);
        
        // Update quality indicators
        this.updateQualityIndicators(metrics);
        
        // Update UI
        this.updateQualityUI(metrics);
        
        // Check for warnings
        this.checkQualityWarnings(metrics);
        
        // Continue monitoring
        requestAnimationFrame(() => this.monitoringLoop());
    }
    
    calculateAudioMetrics(frequencyData, timeData) {
        // Calculate RMS (Root Mean Square) for signal level
        let rms = 0;
        for (let i = 0; i < timeData.length; i++) {
            rms += timeData[i] * timeData[i];
        }
        rms = Math.sqrt(rms / timeData.length);
        const signalLevel = 20 * Math.log10(rms + 1e-10); // Convert to dB
        
        // Estimate noise floor (lowest 10% of frequency bins)
        const sortedFreqs = Array.from(frequencyData).sort((a, b) => a - b);
        const noiseFloorSamples = sortedFreqs.slice(0, Math.floor(sortedFreqs.length * 0.1));
        const noiseFloor = noiseFloorSamples.reduce((sum, val) => sum + val, 0) / noiseFloorSamples.length;
        const noiseLevel = 20 * Math.log10((noiseFloor / 255) + 1e-10);
        
        // Calculate SNR
        const snr = signalLevel - noiseLevel;
        
        // Determine overall quality
        let quality = 'poor';
        if (snr > 20 && signalLevel > -30) quality = 'excellent';
        else if (snr > 15 && signalLevel > -35) quality = 'good';
        else if (snr > 10 && signalLevel > -40) quality = 'fair';
        
        return {
            signalLevel: Math.max(signalLevel, -80), // Clamp to reasonable range
            noiseLevel: Math.max(noiseLevel, -80),
            snr: Math.max(snr, 0),
            quality,
            rms
        };
    }
    
    updateQualityIndicators(metrics) {
        this.qualityIndicators = {
            ...this.qualityIndicators,
            ...metrics,
            warnings: this.generateWarnings(metrics)
        };
        
        // Trigger callback if set
        if (this.callbacks.onQualityUpdate) {
            this.callbacks.onQualityUpdate(this.qualityIndicators);
        }
    }
    
    updateQualityUI(metrics) {
        // Update status
        const statusColors = {
            excellent: '#4CAF50',
            good: '#8BC34A',
            fair: '#FF9800',
            poor: '#F44336'
        };
        
        this.updateQualityStatus(metrics.quality, `Quality: ${metrics.quality.toUpperCase()}`);
        
        const qualityDot = document.getElementById('quality-dot');
        if (qualityDot) {
            qualityDot.style.backgroundColor = statusColors[metrics.quality];
        }
        
        // Update meters
        this.updateMeter('signal-meter', 'signal-value', metrics.signalLevel, -80, 0, 'dB');
        this.updateMeter('noise-meter', 'noise-value', metrics.noiseLevel, -80, -20, 'dB');
        this.updateMeter('snr-meter', 'snr-value', metrics.snr, 0, 30, 'dB');
    }
    
    updateMeter(meterId, valueId, value, min, max, unit) {
        const meter = document.getElementById(meterId);
        const valueEl = document.getElementById(valueId);
        
        if (meter && valueEl) {
            const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
            meter.style.width = `${percentage}%`;
            
            // Color coding
            let color = '#4CAF50'; // Green
            if (percentage < 30) color = '#F44336'; // Red
            else if (percentage < 60) color = '#FF9800'; // Orange
            
            meter.style.backgroundColor = color;
            valueEl.textContent = `${value.toFixed(1)} ${unit}`;
        }
    }
    
    generateWarnings(metrics) {
        const warnings = [];
        
        if (metrics.signalLevel < this.thresholds.minSignalLevel) {
            warnings.push({
                type: 'low_signal',
                message: 'Signal level too low - speak closer to microphone',
                severity: 'warning'
            });
        }
        
        if (metrics.noiseLevel > this.thresholds.maxNoiseLevel) {
            warnings.push({
                type: 'high_noise',
                message: 'High background noise detected - find quieter environment',
                severity: 'warning'
            });
        }
        
        if (metrics.snr < this.thresholds.minSNR) {
            warnings.push({
                type: 'poor_snr',
                message: 'Poor signal-to-noise ratio - reduce background noise',
                severity: 'error'
            });
        }
        
        if (metrics.signalLevel > -5) {
            warnings.push({
                type: 'clipping',
                message: 'Audio may be clipping - reduce microphone gain',
                severity: 'error'
            });
        }
        
        return warnings;
    }
    
    checkQualityWarnings(metrics) {
        const warnings = this.generateWarnings(metrics);
        
        if (warnings.length > 0) {
            this.showWarnings(warnings);
            
            // Trigger warning callback
            if (this.callbacks.onWarning) {
                this.callbacks.onWarning(warnings);
            }
            
            // Show troubleshooting for severe issues
            const hasErrors = warnings.some(w => w.severity === 'error');
            if (hasErrors) {
                this.showTroubleshooting('quality_issues', warnings);
            }
        } else {
            this.hideWarnings();
        }
    }
    
    showWarnings(warnings) {
        const warningsContainer = document.getElementById('quality-warnings');
        const warningList = document.getElementById('warning-list');
        
        if (warningsContainer && warningList) {
            warningList.innerHTML = warnings.map(warning => 
                `<li class="warning-item ${warning.severity}">${warning.message}</li>`
            ).join('');
            
            warningsContainer.style.display = 'block';
        }
    }
    
    hideWarnings() {
        const warningsContainer = document.getElementById('quality-warnings');
        if (warningsContainer) {
            warningsContainer.style.display = 'none';
        }
    }
    
    showTroubleshooting(issueType, details) {
        const troubleshootingPanel = document.getElementById('troubleshooting-panel');
        const suggestionsEl = document.getElementById('suggestions');
        
        if (!troubleshootingPanel || !suggestionsEl) return;
        
        const suggestions = this.getTroubleshootingSuggestions(issueType, details);
        
        suggestionsEl.innerHTML = `
            <h6>Suggested Solutions:</h6>
            <ul>
                ${suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
            </ul>
        `;
        
        troubleshootingPanel.style.display = 'block';
        
        // Update noise status based on current metrics
        if (this.qualityIndicators.noiseLevel > this.thresholds.maxNoiseLevel) {
            this.updateTroubleshootingStatus('noise-status', 'warning', 'High');
        } else {
            this.updateTroubleshootingStatus('noise-status', 'good', 'Low');
        }
        
        // Trigger troubleshooting callback
        if (this.callbacks.onTroubleshooting) {
            this.callbacks.onTroubleshooting(issueType, suggestions);
        }
    }
    
    getTroubleshootingSuggestions(issueType, details) {
        const suggestions = {
            microphone_error: [
                'Check microphone permissions in browser settings',
                'Ensure microphone is properly connected',
                'Try refreshing the page and allowing microphone access',
                'Check if another application is using the microphone'
            ],
            quality_issues: [
                'Move to a quieter environment',
                'Speak directly into the microphone',
                'Adjust microphone position (6-12 inches from mouth)',
                'Check microphone settings in system preferences',
                'Use a headset microphone for better quality'
            ],
            low_signal: [
                'Speak louder or move closer to microphone',
                'Check microphone input level in system settings',
                'Ensure microphone is not muted',
                'Try a different microphone if available'
            ],
            high_noise: [
                'Close windows and doors to reduce outside noise',
                'Turn off fans, air conditioning, or other noise sources',
                'Use noise-canceling microphone if available',
                'Move away from computer fans or hard drives'
            ]
        };
        
        return suggestions[issueType] || ['Contact support for assistance'];
    }
    
    updateTroubleshootingStatus(elementId, status, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
            element.className = `status-indicator ${status}`;
        }
    }
    
    updateQualityStatus(status, text) {
        const statusText = document.getElementById('quality-text');
        if (statusText) {
            statusText.textContent = text;
        }
    }
    
    clearQualityMeters() {
        ['signal-meter', 'noise-meter', 'snr-meter'].forEach(meterId => {
            const meter = document.getElementById(meterId);
            if (meter) {
                meter.style.width = '0%';
            }
        });
        
        ['signal-value', 'noise-value', 'snr-value'].forEach(valueId => {
            const valueEl = document.getElementById(valueId);
            if (valueEl) {
                valueEl.textContent = valueId.includes('snr') ? '0 dB' : '-∞ dB';
            }
        });
    }
    
    // Public API methods
    setCallback(type, callback) {
        if (this.callbacks.hasOwnProperty(type)) {
            this.callbacks[type] = callback;
        }
    }
    
    getQualityMetrics() {
        return { ...this.qualityIndicators };
    }
    
    isGoodQuality() {
        return ['good', 'excellent'].includes(this.qualityIndicators.quality);
    }
}

// Export for use in other modules
window.AudioQualityMonitor = AudioQualityMonitor;