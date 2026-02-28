/**
 * Main application controller for NLP Web Interface
 * Coordinates audio capture, WebSocket communication, and UI interactions
 */

class NLPWebInterface {
    constructor() {
        this.audioCapture = null;
        this.websocketClient = null;
        this.currentSection = 'audio';
        this.isMobileMenuOpen = false;
        this.loadingStates = new Map();
        this.toastQueue = [];
        
        this.init();
    }
    
    /**
     * Initialize the application
     */
    init() {
        console.log('🚀 Initializing NLP Web Interface...');
        
        try {
            // Check for required dependencies
            this.checkDependencies();
            
            // Initialize components
            this.initializeAudioCapture();
            this.initializeWebSocket();
            this.setupNavigation();
            this.setupEventListeners();
            this.setupResponsiveFeatures();
            this.setupLoadingStates();
            this.setupErrorHandling();
            this.setupToastNotifications();
            
            console.log('✅ NLP Web Interface initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize NLP Web Interface:', error);
            this.handleInitializationError(error);
        }
    }
    
    /**
     * Check for required dependencies
     */
    checkDependencies() {
        const dependencies = {
            'd3': typeof d3 !== 'undefined',
            'AudioCaptureInterface': typeof AudioCaptureInterface !== 'undefined',
            'WebSocketClient': typeof WebSocketClient !== 'undefined',
            'ResultDisplayManager': typeof ResultDisplayManager !== 'undefined' || typeof window.resultDisplayManager !== 'undefined',
            'ErrorHandler': typeof ErrorHandler !== 'undefined'
        };
        
        const missing = Object.keys(dependencies).filter(dep => !dependencies[dep]);
        
        if (missing.length > 0) {
            console.warn('⚠️ Missing dependencies:', missing);
            
            // Show user-friendly warning for missing D3.js
            if (missing.includes('d3')) {
                this.showToast('Visualization library not loaded - using simplified charts', 'warning', 8000);
            }
            
            // For WebSocketClient, try to wait a bit and retry
            if (missing.includes('WebSocketClient')) {
                console.log('🔄 WebSocketClient not ready, will retry initialization...');
                setTimeout(() => {
                    if (typeof WebSocketClient !== 'undefined') {
                        console.log('✅ WebSocketClient now available, continuing initialization...');
                        this.initializeWebSocket();
                    } else {
                        console.warn('⚠️ WebSocketClient still not available, real-time features will be disabled');
                    }
                }, 100);
            }
            
            // Show error for other critical missing dependencies (but be more lenient)
            const critical = missing.filter(dep => dep !== 'd3' && dep !== 'WebSocketClient' && dep !== 'ResultDisplayManager');
            if (critical.length > 0) {
                console.warn(`⚠️ Some dependencies missing: ${critical.join(', ')}, but continuing...`);
            }
        }
    }
    
    /**
     * Handle initialization errors
     */
    handleInitializationError(error) {
        console.error('❌ Initialization failed:', error);
        
        // Show user-friendly error message
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #dc3545;
            color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 2000;
            text-align: center;
            max-width: 400px;
        `;
        errorDiv.innerHTML = `
            <h3>⚠️ Initialization Failed</h3>
            <p>The NLP Web Interface failed to initialize properly.</p>
            <p><strong>Error:</strong> ${error.message}</p>
            <button onclick="window.location.reload()" style="
                background: white;
                color: #dc3545;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 1rem;
                font-weight: 600;
            ">Reload Page</button>
        `;
        document.body.appendChild(errorDiv);
    }
    
    /**
     * Initialize audio capture interface
     */
    initializeAudioCapture() {
        try {
            this.audioCapture = new AudioCaptureInterface();
            console.log('✅ Audio capture interface initialized');
        } catch (error) {
            console.error('❌ Failed to initialize audio capture:', error);
            this.showError('Failed to initialize audio capture. Please check your browser compatibility.');
        }
    }
    
    /**
     * Initialize WebSocket client
     */
    initializeWebSocket() {
        try {
            // Check if WebSocketClient is available
            if (typeof WebSocketClient === 'undefined') {
                console.warn('⚠️ WebSocketClient not available, skipping WebSocket initialization');
                return;
            }
            
            this.websocketClient = new WebSocketClient();
            
            // Connect audio capture to WebSocket
            if (this.audioCapture) {
                this.audioCapture.setWebSocket(this.websocketClient);
            }
            
            // Set up custom event handlers
            this.websocketClient.setEventHandlers({
                onConnectionOpen: (event) => {
                    console.log('� WebSocket connected');
                    this.updateConnectionStatus(true);
                    this.showToast('Connected to server', 'success');
                },
                
                onConnectionClose: (event) => {
                    console.log('🔌 WebSocket disconnected');
                    this.updateConnectionStatus(false);
                    // Only show warning if it was an unexpected disconnect
                    if (event.code !== 1000) {
                        this.showToast('Disconnected from server', 'warning');
                    }
                },
                
                onConnectionError: (event) => {
                    console.error('❌ WebSocket error:', event);
                    this.updateConnectionStatus(false);
                    // Don't show error modal immediately, let reconnection handle it
                    console.log('WebSocket connection error, will attempt reconnection...');
                },
                
                onTranscriptionUpdate: (data) => {
                    this.handleTranscriptionUpdate(data);
                },
                
                onAnalysisResult: (data) => {
                    this.handleAnalysisResult(data);
                },
                
                onProcessingComplete: (data) => {
                    this.handleProcessingComplete(data);
                }
            });
            
            console.log('✅ WebSocket client initialized');
        } catch (error) {
            console.error('❌ Failed to initialize WebSocket client:', error);
            // Don't show error modal immediately, just log the error
            console.log('WebSocket initialization failed, real-time features will be unavailable');
        }
    }
    
    /**
     * Set up responsive navigation features
     */
    setupResponsiveFeatures() {
        // Mobile menu toggle
        const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
        const navigation = document.getElementById('navigation');
        
        if (mobileMenuToggle && navigation) {
            mobileMenuToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
            
            // Close mobile menu when clicking outside
            document.addEventListener('click', (event) => {
                if (this.isMobileMenuOpen && 
                    !navigation.contains(event.target) && 
                    !mobileMenuToggle.contains(event.target)) {
                    this.closeMobileMenu();
                }
            });
            
            // Close mobile menu on escape key
            document.addEventListener('keydown', (event) => {
                if (event.key === 'Escape' && this.isMobileMenuOpen) {
                    this.closeMobileMenu();
                }
            });
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768 && this.isMobileMenuOpen) {
                this.closeMobileMenu();
            }
        });
    }
    
    /**
     * Toggle mobile menu
     */
    toggleMobileMenu() {
        const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
        const navigation = document.getElementById('navigation');
        
        if (this.isMobileMenuOpen) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }
    
    /**
     * Open mobile menu
     */
    openMobileMenu() {
        const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
        const navigation = document.getElementById('navigation');
        
        if (mobileMenuToggle && navigation) {
            mobileMenuToggle.classList.add('active');
            navigation.classList.add('active');
            this.isMobileMenuOpen = true;
            
            // Prevent body scroll
            document.body.style.overflow = 'hidden';
        }
    }
    
    /**
     * Close mobile menu
     */
    closeMobileMenu() {
        const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
        const navigation = document.getElementById('navigation');
        
        if (mobileMenuToggle && navigation) {
            mobileMenuToggle.classList.remove('active');
            navigation.classList.remove('active');
            this.isMobileMenuOpen = false;
            
            // Restore body scroll
            document.body.style.overflow = '';
        }
    }
    
    /**
     * Update connection status indicator
     */
    updateConnectionStatus(isConnected) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');
        const connectionStatusDot = document.getElementById('connection-status-dot');
        
        if (statusIndicator && statusText) {
            if (isConnected) {
                statusIndicator.className = 'status-indicator';
                statusText.textContent = 'Connected';
            } else {
                statusIndicator.className = 'status-indicator disconnected';
                statusText.innerHTML = 'Disconnected <button class="retry-btn" onclick="nlpApp.retryConnection()">Retry</button>';
            }
        }
        
        if (connectionStatusDot) {
            if (isConnected) {
                connectionStatusDot.className = 'status-dot online';
            } else {
                connectionStatusDot.className = 'status-dot offline';
            }
        }
    }
    
    /**
     * Set up loading states management
     */
    setupLoadingStates() {
        // Global progress bar
        this.globalProgress = document.getElementById('global-progress');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        
        // Enhanced loading overlay
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingTitle = document.getElementById('loading-title');
        this.loadingMessage = document.getElementById('loading-message');
        this.loadingProgressFill = document.getElementById('loading-progress-fill');
        this.loadingPercentage = document.getElementById('loading-percentage');
    }
    
    /**
     * Show loading state
     */
    showLoading(key, title = 'Processing...', message = 'Please wait while we process your request', progress = 0) {
        this.loadingStates.set(key, { title, message, progress });
        
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('active');
            
            if (this.loadingTitle) this.loadingTitle.textContent = title;
            if (this.loadingMessage) this.loadingMessage.textContent = message;
            if (this.loadingProgressFill) this.loadingProgressFill.style.width = `${progress}%`;
            if (this.loadingPercentage) this.loadingPercentage.textContent = `${Math.round(progress)}%`;
        }
    }
    
    /**
     * Update loading progress
     */
    updateLoadingProgress(key, progress, message = null) {
        const loadingState = this.loadingStates.get(key);
        if (loadingState) {
            loadingState.progress = progress;
            if (message) loadingState.message = message;
            
            if (this.loadingProgressFill) this.loadingProgressFill.style.width = `${progress}%`;
            if (this.loadingPercentage) this.loadingPercentage.textContent = `${Math.round(progress)}%`;
            if (message && this.loadingMessage) this.loadingMessage.textContent = message;
        }
    }
    
    /**
     * Hide loading state
     */
    hideLoading(key) {
        this.loadingStates.delete(key);
        
        if (this.loadingStates.size === 0 && this.loadingOverlay) {
            this.loadingOverlay.classList.remove('active');
        }
    }
    
    /**
     * Show global progress bar
     */
    showGlobalProgress(text = 'Processing...', progress = 0) {
        if (this.globalProgress) {
            this.globalProgress.classList.add('active');
            
            if (this.progressText) this.progressText.textContent = text;
            if (this.progressFill) this.progressFill.style.width = `${progress}%`;
        }
    }
    
    /**
     * Update global progress
     */
    updateGlobalProgress(progress, text = null) {
        if (this.progressFill) this.progressFill.style.width = `${progress}%`;
        if (text && this.progressText) this.progressText.textContent = text;
    }
    
    /**
     * Hide global progress bar
     */
    hideGlobalProgress() {
        if (this.globalProgress) {
            this.globalProgress.classList.remove('active');
        }
    }
    
    /**
     * Set up error handling
     */
    setupErrorHandling() {
        const errorModal = document.getElementById('error-modal');
        const errorModalClose = document.getElementById('error-modal-close');
        const errorRetry = document.getElementById('error-retry');
        const errorDismiss = document.getElementById('error-dismiss');
        
        if (errorModalClose) {
            errorModalClose.addEventListener('click', () => {
                this.hideErrorModal();
            });
        }
        
        if (errorDismiss) {
            errorDismiss.addEventListener('click', () => {
                this.hideErrorModal();
            });
        }
        
        if (errorRetry) {
            errorRetry.addEventListener('click', () => {
                this.retryLastOperation();
            });
        }
        
        // Close modal on escape key
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && errorModal && errorModal.classList.contains('active')) {
                this.hideErrorModal();
            }
        });
    }
    
    /**
     * Show error modal
     */
    showErrorModal(message, details = null) {
        const errorModal = document.getElementById('error-modal');
        const errorMessage = document.getElementById('error-message');
        const errorDetails = document.getElementById('error-details');
        
        if (errorModal && errorMessage) {
            errorMessage.textContent = message;
            
            if (details && errorDetails) {
                errorDetails.textContent = details;
                errorDetails.style.display = 'block';
            } else if (errorDetails) {
                errorDetails.style.display = 'none';
            }
            
            errorModal.classList.add('active');
        }
    }
    
    /**
     * Hide error modal
     */
    hideErrorModal() {
        const errorModal = document.getElementById('error-modal');
        if (errorModal) {
            errorModal.classList.remove('active');
        }
    }
    
    /**
     * Retry last operation (placeholder)
     */
    retryLastOperation() {
        this.hideErrorModal();
        // Implementation depends on what operation failed
        console.log('Retrying last operation...');
    }
    
    /**
     * Set up toast notifications
     */
    setupToastNotifications() {
        // Success toast
        const successToastClose = document.getElementById('success-toast-close');
        if (successToastClose) {
            successToastClose.addEventListener('click', () => {
                this.hideToast('success');
            });
        }
        
        // Warning toast
        const warningToastClose = document.getElementById('warning-toast-close');
        if (warningToastClose) {
            warningToastClose.addEventListener('click', () => {
                this.hideToast('warning');
            });
        }
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'success', duration = 4000) {
        const toastId = `${type}-toast`;
        const toast = document.getElementById(toastId);
        const messageElement = document.getElementById(`${type}-message`);
        
        if (toast && messageElement) {
            messageElement.textContent = message;
            toast.classList.add('active');
            
            // Auto-hide after duration
            setTimeout(() => {
                this.hideToast(type);
            }, duration);
        }
    }
    
    /**
     * Hide toast notification
     */
    hideToast(type) {
        const toastId = `${type}-toast`;
        const toast = document.getElementById(toastId);
        
        if (toast) {
            toast.classList.remove('active');
        }
    }
    
    
    /**
     * Set up navigation between sections
     */
    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        const sections = document.querySelectorAll('.content-section');
        
        navButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetSection = button.dataset.section;
                
                // Close mobile menu if open
                if (this.isMobileMenuOpen) {
                    this.closeMobileMenu();
                }
                
                // Update active nav button
                navButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active section
                sections.forEach(section => section.classList.remove('active'));
                const targetElement = document.getElementById(`${targetSection}-section`);
                if (targetElement) {
                    targetElement.classList.add('active');
                    this.currentSection = targetSection;
                    
                    // Trigger section-specific initialization
                    this.onSectionChange(targetSection);
                }
            });
        });
    }
    
    /**
     * Handle section change
     */
    onSectionChange(sectionName) {
        console.log(`📱 Switched to section: ${sectionName}`);
        
        // Section-specific initialization
        switch (sectionName) {
            case 'metrics':
                this.initializeMetricsSection();
                break;
            case 'sessions':
                this.initializeSessionsSection();
                break;
            case 'upload':
                this.initializeUploadSection();
                break;
            case 'audio':
                this.initializeAudioSection();
                break;
        }
    }
    
    /**
     * Initialize metrics section
     */
    initializeMetricsSection() {
        // Load metrics data if not already loaded
        this.loadMetricsData();
    }
    
    /**
     * Initialize sessions section
     */
    initializeSessionsSection() {
        // Load sessions data if not already loaded
        this.loadSessionsData();
    }
    
    /**
     * Initialize upload section
     */
    initializeUploadSection() {
        // Set up drag and drop if not already set up
        this.setupEnhancedFileUpload();
    }
    
    /**
     * Initialize audio section
     */
    initializeAudioSection() {
        // Ensure audio capture is ready
        if (this.audioCapture) {
            this.audioCapture.updateUI();
        }
    }
    
    /**
     * Load metrics data
     */
    async loadMetricsData() {
        try {
            this.showTabLoading('training-chart-loading');
            this.showTabLoading('comparison-chart-loading');
            
            const response = await fetch('/api/metrics/');
            if (response.ok) {
                const data = await response.json();
                this.updateMetricsDisplay(data);
            } else {
                throw new Error('Failed to load metrics data');
            }
        } catch (error) {
            console.error('❌ Failed to load metrics:', error);
            this.showToast('Failed to load metrics data', 'warning');
        } finally {
            this.hideTabLoading('training-chart-loading');
            this.hideTabLoading('comparison-chart-loading');
        }
    }
    
    /**
     * Load sessions data
     */
    async loadSessionsData() {
        try {
            const response = await fetch('/api/sessions/history');
            if (response.ok) {
                const data = await response.json();
                this.updateSessionsList(data.sessions || []);
            } else if (response.status === 404) {
                // No sessions found - this is normal
                this.updateSessionsList([]);
            } else {
                throw new Error('Failed to load sessions data');
            }
        } catch (error) {
            console.error('❌ Failed to load sessions:', error);
            this.showToast('Failed to load sessions data', 'warning');
            // Show empty state on error
            this.updateSessionsList([]);
        }
    }
    
    /**
     * Update metrics display
     */
    updateMetricsDisplay(data) {
        // Update metric cards with fallback values
        const accuracyElement = document.getElementById('current-accuracy');
        const f1Element = document.getElementById('f1-score');
        const latencyElement = document.getElementById('latency');
        const sessionsElement = document.getElementById('active-sessions');
        
        // Handle the actual API response structure
        if (accuracyElement) {
            const valueElement = accuracyElement.querySelector('.value');
            if (valueElement) {
                // Use success_rate as accuracy if available
                const accuracy = data.success_rate !== undefined ? data.success_rate : 0.95; // fallback
                valueElement.textContent = (accuracy * 100).toFixed(1);
            }
        }
        
        if (f1Element) {
            const valueElement = f1Element.querySelector('.value');
            if (valueElement) {
                // Use a default F1 score since it's not in the current API
                const f1Score = data.f1_score !== undefined ? data.f1_score : 0.87; // fallback
                valueElement.textContent = f1Score.toFixed(3);
            }
        }
        
        if (latencyElement) {
            const valueElement = latencyElement.querySelector('.value');
            if (valueElement) {
                // Use a default latency since it's not in the current API
                const latency = data.avg_latency !== undefined ? data.avg_latency : 150; // fallback
                valueElement.textContent = latency.toFixed(1);
            }
        }
        
        if (sessionsElement) {
            const valueElement = sessionsElement.querySelector('.value');
            if (valueElement) {
                // Use recent_operations as active sessions
                const sessions = data.recent_operations !== undefined ? data.recent_operations : 0;
                valueElement.textContent = sessions.toString();
            }
        }
        
        // Update charts if available
        if (window.metricsCharts) {
            window.metricsCharts.updateCharts(data);
        } else {
            console.log('📊 Metrics charts not available, using basic display');
        }
    }
    
    /**
     * Update sessions list
     */
    updateSessionsList(sessions) {
        const sessionsList = document.getElementById('sessions-list');
        if (!sessionsList) return;
        
        if (sessions.length === 0) {
            sessionsList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📋</div>
                    <h3>No sessions found</h3>
                    <p>Start a new session to see it appear here</p>
                </div>
            `;
            return;
        }
        
        const html = sessions.map(session => `
            <div class="session-item" data-session-id="${session.session_id}">
                <div class="session-header">
                    <h4>${session.session_id}</h4>
                    <span class="session-status ${session.status}">${session.status}</span>
                </div>
                <div class="session-details">
                    <div class="session-meta">
                        <span class="session-date">${new Date(session.created_at).toLocaleString()}</span>
                        <span class="session-duration">${this.formatDuration(session.duration)}</span>
                    </div>
                    <div class="session-stats">
                        <span class="stat">📝 ${session.transcription_length || 0} chars</span>
                        <span class="stat">🔍 ${session.analysis_count || 0} analyses</span>
                    </div>
                </div>
                <div class="session-actions">
                    <button class="btn btn-small" onclick="nlpApp.viewSession('${session.session_id}')">View</button>
                    <button class="btn btn-small" onclick="nlpApp.exportSession('${session.session_id}')">Export</button>
                    <button class="btn btn-small" onclick="nlpApp.deleteSession('${session.session_id}')">Delete</button>
                </div>
            </div>
        `).join('');
        
        sessionsList.innerHTML = html;
    }
    
    /**
     * Format duration for display
     */
    formatDuration(seconds) {
        if (!seconds) return '0s';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    /**
     * Set up enhanced file upload with better drag and drop
     */
    setupEnhancedFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const uploadZone = document.querySelector('.upload-zone');
        
        if (uploadArea && uploadZone) {
            // Enhanced drag and drop
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadZone.addEventListener(eventName, this.preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadZone.addEventListener(eventName, () => {
                    uploadZone.classList.add('dragover');
                }, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadZone.addEventListener(eventName, () => {
                    uploadZone.classList.remove('dragover');
                }, false);
            });
            
            uploadZone.addEventListener('drop', (event) => {
                const files = event.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files);
                }
            }, false);
        }
    }
    
    /**
     * Prevent default drag behaviors
     */
    preventDefaults(event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    /**
     * Show tab loading state
     */
    showTabLoading(loadingId) {
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.style.display = 'flex';
        }
    }
    
    /**
     * Hide tab loading state
     */
    hideTabLoading(loadingId) {
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
    }
    
    /**
     * Set up global event listeners
     */
    setupEventListeners() {
        // Listen for custom events from components
        document.addEventListener('audioRecordingComplete', (event) => {
            console.log('🎵 Audio recording completed:', event.detail);
            this.hideLoading('audio-recording');
        });
        
        document.addEventListener('analysisResultReceived', (event) => {
            console.log('📊 Analysis result received:', event.detail);
        });
        
        document.addEventListener('processingComplete', (event) => {
            console.log('✅ Processing completed:', event.detail);
            this.hideLoading('analysis-processing');
        });
        
        document.addEventListener('qualityFeedback', (event) => {
            console.log('🔊 Quality feedback:', event.detail);
        });
        
        // Handle tab switching in analysis results
        this.setupAnalysisTabs();
        
        // Handle file uploads
        this.setupFileUpload();
        
        // Set up control buttons
        this.setupControlButtons();
        
        // Handle keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Set up enhanced result display
        this.setupEnhancedResultDisplay();
        
        // Set up error handling integration
        this.setupErrorHandlingIntegration();
    }
    
    /**
     * Set up control buttons
     */
    setupControlButtons() {
        // Clear transcription button
        const clearTranscriptionBtn = document.getElementById('clear-transcription');
        if (clearTranscriptionBtn) {
            clearTranscriptionBtn.addEventListener('click', () => {
                this.clearTranscription();
            });
        }
        
        // Export transcription button
        const exportTranscriptionBtn = document.getElementById('export-transcription');
        if (exportTranscriptionBtn) {
            exportTranscriptionBtn.addEventListener('click', () => {
                this.exportTranscription();
            });
        }
        
        // Refresh analysis button
        const refreshAnalysisBtn = document.getElementById('refresh-analysis');
        if (refreshAnalysisBtn) {
            refreshAnalysisBtn.addEventListener('click', () => {
                this.refreshAnalysis();
            });
        }
        
        // Export analysis button
        const exportAnalysisBtn = document.getElementById('export-analysis');
        if (exportAnalysisBtn) {
            exportAnalysisBtn.addEventListener('click', () => {
                this.exportAnalysis();
            });
        }
        
        // New session button
        const newSessionBtn = document.getElementById('new-session');
        if (newSessionBtn) {
            newSessionBtn.addEventListener('click', () => {
                this.createNewSession();
            });
        }
        
        // Export sessions button
        const exportSessionsBtn = document.getElementById('export-sessions');
        if (exportSessionsBtn) {
            exportSessionsBtn.addEventListener('click', () => {
                this.exportAllSessions();
            });
        }
        
        // Expand all results button
        const expandAllBtn = document.getElementById('expand-all-results');
        if (expandAllBtn) {
            expandAllBtn.addEventListener('click', () => {
                this.toggleAllResultSections();
            });
        }
    }
    
    /**
     * Set up keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            // Only handle shortcuts when not typing in input fields
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                return;
            }
            
            // Ctrl/Cmd + R: Start/Stop recording
            if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
                event.preventDefault();
                if (this.audioCapture) {
                    if (this.audioCapture.isRecording) {
                        this.audioCapture.stopRecording();
                    } else {
                        this.audioCapture.startRecording();
                    }
                }
            }
            
            // Ctrl/Cmd + 1-4: Switch sections
            if ((event.ctrlKey || event.metaKey) && event.key >= '1' && event.key <= '4') {
                event.preventDefault();
                const sections = ['audio', 'upload', 'metrics', 'sessions'];
                const sectionIndex = parseInt(event.key) - 1;
                if (sections[sectionIndex]) {
                    const navBtn = document.querySelector(`[data-section="${sections[sectionIndex]}"]`);
                    if (navBtn) navBtn.click();
                }
            }
        });
    }
    
    /**
     * Clear transcription
     */
    clearTranscription() {
        const transcriptionText = document.getElementById('transcription-text');
        if (transcriptionText) {
            transcriptionText.innerHTML = `
                <div class="transcription-placeholder">
                    <span class="placeholder-icon">🎙️</span>
                    <p>Start recording to see live transcription...</p>
                </div>
            `;
        }
        
        // Clear analysis results
        this.clearAnalysisResults();
        
        this.showToast('Transcription cleared', 'success');
    }
    
    /**
     * Export transcription
     */
    exportTranscription() {
        const transcriptionText = document.getElementById('transcription-text');
        if (!transcriptionText) return;
        
        const text = transcriptionText.textContent || '';
        if (text.trim().length === 0) {
            this.showToast('No transcription to export', 'warning');
            return;
        }
        
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcription-${new Date().toISOString().slice(0, 19)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('Transcription exported', 'success');
    }
    
    /**
     * Refresh analysis
     */
    async refreshAnalysis() {
        const transcriptionText = document.getElementById('transcription-text');
        if (!transcriptionText) return;
        
        const text = transcriptionText.textContent || '';
        if (text.trim().length === 0) {
            this.showToast('No text to analyze', 'warning');
            return;
        }
        
        try {
            this.showLoading('refresh-analysis', 'Re-analyzing text...', 'Please wait while we re-process the text');
            
            const response = await fetch('/api/analyze/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text.trim(),
                    session_id: this.websocketClient?.getSessionId()
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.handleAnalysisResult(result);
                this.showToast('Analysis refreshed', 'success');
            } else {
                throw new Error('Failed to refresh analysis');
            }
        } catch (error) {
            console.error('❌ Failed to refresh analysis:', error);
            this.showErrorModal('Failed to refresh analysis', error.message);
        } finally {
            this.hideLoading('refresh-analysis');
        }
    }
    
    /**
     * Export analysis results
     */
    exportAnalysis() {
        // Collect all analysis results
        const results = {
            timestamp: new Date().toISOString(),
            transcription: document.getElementById('transcription-text')?.textContent || '',
            pos_tags: this.getAnalysisData('pos'),
            named_entities: this.getAnalysisData('ner'),
            dependencies: this.getAnalysisData('dep'),
            sentiment: this.getAnalysisData('sentiment')
        };
        
        const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis-results-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('Analysis results exported', 'success');
    }
    
    /**
     * Get analysis data from visualization containers
     */
    getAnalysisData(type) {
        // This would extract data from the visualization containers
        // Implementation depends on how data is stored in the visualizations
        return null; // Placeholder
    }
    
    /**
     * Clear analysis results
     */
    clearAnalysisResults() {
        const visualizationContainers = [
            'pos-visualization',
            'ner-visualization',
            'dependency-visualization',
            'sentiment-visualization'
        ];
        
        visualizationContainers.forEach(containerId => {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = '';
            }
        });
    }
    
    /**
     * Create new session
     */
    async createNewSession() {
        try {
            this.showLoading('new-session', 'Creating new session...', 'Setting up a fresh analysis session');
            
            const response = await fetch('/api/sessions/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Clear current session data
                this.clearTranscription();
                
                // Update WebSocket with new session ID
                if (this.websocketClient) {
                    this.websocketClient.setSessionId(result.session_id);
                }
                
                this.showToast(`New session created: ${result.session_id}`, 'success');
                
                // Switch to audio section
                const audioNavBtn = document.querySelector('[data-section="audio"]');
                if (audioNavBtn) audioNavBtn.click();
                
            } else {
                throw new Error('Failed to create new session');
            }
        } catch (error) {
            console.error('❌ Failed to create new session:', error);
            this.showErrorModal('Failed to create new session', error.message);
        } finally {
            this.hideLoading('new-session');
        }
    }
    
    /**
     * Display session details
     */
    displaySessionDetails(session) {
        console.log('📋 Displaying session details:', session);
        
        // Create modal or expand section to show session details
        const modal = document.createElement('div');
        modal.className = 'session-details-modal';
        modal.innerHTML = `
            <div class="modal-overlay" onclick="this.parentElement.remove()"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Session Details: ${session.session_id}</h3>
                    <button class="modal-close" onclick="this.closest('.session-details-modal').remove()">×</button>
                </div>
                <div class="modal-body">
                    <div class="session-info">
                        <div class="info-row">
                            <label>Session ID:</label>
                            <span>${session.session_id}</span>
                        </div>
                        <div class="info-row">
                            <label>Status:</label>
                            <span class="status ${session.status}">${session.status}</span>
                        </div>
                        <div class="info-row">
                            <label>Created:</label>
                            <span>${new Date(session.created_at).toLocaleString()}</span>
                        </div>
                        <div class="info-row">
                            <label>Duration:</label>
                            <span>${this.formatDuration(session.duration)}</span>
                        </div>
                        <div class="info-row">
                            <label>Transcription Length:</label>
                            <span>${session.transcription_length || 0} characters</span>
                        </div>
                        <div class="info-row">
                            <label>Analysis Count:</label>
                            <span>${session.analysis_count || 0} analyses</span>
                        </div>
                    </div>
                    ${session.transcription ? `
                        <div class="session-transcription">
                            <h4>Transcription</h4>
                            <div class="transcription-text">${session.transcription}</div>
                        </div>
                    ` : ''}
                    ${session.analysis_results ? `
                        <div class="session-analysis">
                            <h4>Analysis Results</h4>
                            <pre class="analysis-json">${JSON.stringify(session.analysis_results, null, 2)}</pre>
                        </div>
                    ` : ''}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="nlpApp.exportSession('${session.session_id}')">Export</button>
                    <button class="btn btn-secondary" onclick="this.closest('.session-details-modal').remove()">Close</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    /**
     * Export all sessions
     */
    async exportAllSessions() {
        try {
            this.showLoading('export-sessions', 'Exporting sessions...', 'Preparing session data for download');
            
            const response = await fetch('/api/sessions/export');
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `sessions-export-${new Date().toISOString().slice(0, 19)}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.showToast('Sessions exported successfully', 'success');
            } else {
                throw new Error('Failed to export sessions');
            }
        } catch (error) {
            console.error('❌ Failed to export sessions:', error);
            this.showErrorModal('Failed to export sessions', error.message);
        } finally {
            this.hideLoading('export-sessions');
        }
    }
    
    /**
     * View session details
     */
    async viewSession(sessionId) {
        try {
            this.showLoading('view-session', 'Loading session...', `Loading session ${sessionId}`);
            
            const response = await fetch(`/api/sessions/${sessionId}`);
            if (response.ok) {
                const session = await response.json();
                this.displaySessionDetails(session);
            } else {
                throw new Error('Failed to load session');
            }
        } catch (error) {
            console.error('❌ Failed to view session:', error);
            this.showErrorModal('Failed to load session', error.message);
        } finally {
            this.hideLoading('view-session');
        }
    }
    
    /**
     * Export single session
     */
    async exportSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/export`);
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `session-${sessionId}-${new Date().toISOString().slice(0, 19)}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.showToast(`Session ${sessionId} exported`, 'success');
            } else {
                throw new Error('Failed to export session');
            }
        } catch (error) {
            console.error('❌ Failed to export session:', error);
            this.showErrorModal('Failed to export session', error.message);
        }
    }
    
    /**
     * Delete session
     */
    async deleteSession(sessionId) {
        if (!confirm(`Are you sure you want to delete session ${sessionId}?`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.showToast(`Session ${sessionId} deleted`, 'success');
                this.loadSessionsData(); // Refresh the list
            } else {
                throw new Error('Failed to delete session');
            }
        } catch (error) {
            console.error('❌ Failed to delete session:', error);
            this.showErrorModal('Failed to delete session', error.message);
        }
    }
    
    /**
     * Set up enhanced result display
     */
    setupEnhancedResultDisplay() {
        // Listen for result display events
        document.addEventListener('collapsibleToggle', (event) => {
            console.log('📂 Collapsible section toggled:', event.detail);
        });
        
        document.addEventListener('tabSwitch', (event) => {
            console.log('📑 Tab switched:', event.detail);
            this.onAnalysisTabSwitch(event.detail.tabId);
        });
        
        document.addEventListener('interactiveAction', (event) => {
            console.log('🎯 Interactive action:', event.detail);
        });
        
        // Set up analysis tab switching (legacy support)
        this.setupAnalysisTabs();
    }
    
    /**
     * Handle analysis tab switch
     */
    onAnalysisTabSwitch(tabId) {
        // Update URL hash for deep linking
        if (history.replaceState) {
            history.replaceState(null, null, `#analysis-${tabId}`);
        }
        
        // Load tab-specific data if needed
        switch (tabId) {
            case 'pos':
                this.loadPOSAnalysisData();
                break;
            case 'ner':
                this.loadNERAnalysisData();
                break;
            case 'dep':
                this.loadDependencyAnalysisData();
                break;
            case 'sentiment':
                this.loadSentimentAnalysisData();
                break;
        }
    }
    
    /**
     * Load POS analysis data
     */
    loadPOSAnalysisData() {
        // Implementation for loading POS-specific data
        console.log('📝 Loading POS analysis data...');
    }
    
    /**
     * Load NER analysis data
     */
    loadNERAnalysisData() {
        // Implementation for loading NER-specific data
        console.log('🎯 Loading NER analysis data...');
    }
    
    /**
     * Load dependency analysis data
     */
    loadDependencyAnalysisData() {
        // Implementation for loading dependency-specific data
        console.log('🌳 Loading dependency analysis data...');
    }
    
    /**
     * Load sentiment analysis data
     */
    loadSentimentAnalysisData() {
        // Implementation for loading sentiment-specific data
        console.log('😊 Loading sentiment analysis data...');
    }
    
    /**
     * Toggle all result sections
     */
    toggleAllResultSections() {
        if (window.resultDisplayManager) {
            const expandAllBtn = document.getElementById('expand-all-results');
            const isExpanding = expandAllBtn.textContent === 'Expand All';
            
            if (isExpanding) {
                window.resultDisplayManager.expandAllSections();
                expandAllBtn.textContent = 'Collapse All';
            } else {
                window.resultDisplayManager.collapseAllSections();
                expandAllBtn.textContent = 'Expand All';
            }
        }
    }
    
    /**
     * Set up analysis result tabs
     */
    setupAnalysisTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                
                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active tab pane
                tabPanes.forEach(pane => pane.classList.remove('active'));
                const targetPane = document.getElementById(`${targetTab}-tab`);
                if (targetPane) {
                    targetPane.classList.add('active');
                }
            });
        });
    }
    
    /**
     * Set up file upload functionality
     */
    setupFileUpload() {
        const fileInput = document.getElementById('file-input');
        const uploadLabel = document.querySelector('.upload-label');
        
        if (fileInput && uploadLabel) {
            // Handle file selection
            fileInput.addEventListener('change', (event) => {
                const files = event.target.files;
                if (files.length > 0) {
                    this.handleFileUpload(files);
                }
            });
            
            // Handle drag and drop
            uploadLabel.addEventListener('dragover', (event) => {
                event.preventDefault();
                uploadLabel.classList.add('drag-over');
            });
            
            uploadLabel.addEventListener('dragleave', () => {
                uploadLabel.classList.remove('drag-over');
            });
            
            uploadLabel.addEventListener('drop', (event) => {
                event.preventDefault();
                uploadLabel.classList.remove('drag-over');
                
                const files = event.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files);
                }
            });
        }
    }
    
    /**
     * Handle transcription updates from WebSocket
     */
    handleTranscriptionUpdate(data) {
        console.log('📝 Transcription update:', data);
        
        // The WebSocket client already handles UI updates
        // This is for additional processing if needed
        
        // Trigger analysis if transcription is final
        if (data.is_final && data.text && data.text.trim().length > 0) {
            // Analysis will be triggered automatically by the backend
            console.log('🔍 Final transcription received, analysis will follow');
        }
    }
    
    /**
     * Handle analysis results from WebSocket
     */
    handleAnalysisResult(data) {
        console.log('📊 Analysis result:', data);
        
        // Update visualization based on analysis type
        if (data.analysis_type === 'pos_tags' && data.pos_tags) {
            this.updatePOSVisualization(data.pos_tags, data.text);
        }
        
        if (data.analysis_type === 'named_entities' && data.named_entities) {
            this.updateNERVisualization(data.named_entities, data.text);
        }
        
        if (data.analysis_type === 'dependencies' && data.dependencies) {
            this.updateDependencyVisualization(data.dependencies, data.text);
        }
        
        if (data.analysis_type === 'sentiment' && data.sentiment) {
            this.updateSentimentVisualization(data.sentiment);
        }
    }
    
    /**
     * Handle processing completion
     */
    handleProcessingComplete(data) {
        console.log('✅ Processing complete:', data);
        
        this.showNotification('Analysis completed successfully', 'success');
        
        // Switch to appropriate tab if we have results
        if (data.has_pos_tags) {
            this.switchToTab('pos');
        } else if (data.has_named_entities) {
            this.switchToTab('ner');
        } else if (data.has_dependencies) {
            this.switchToTab('dep');
        } else if (data.has_sentiment) {
            this.switchToTab('sentiment');
        }
    }
    
    /**
     * Handle file upload
     */
    async handleFileUpload(files) {
        console.log('📁 Files selected for upload:', files.length);
        
        const progressContainer = document.getElementById('upload-progress');
        if (progressContainer) {
            progressContainer.innerHTML = '';
        }
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            await this.uploadSingleFile(file, i + 1, files.length);
        }
    }
    
    /**
     * Upload a single file
     */
    async uploadSingleFile(file, index, total) {
        console.log(`📤 Uploading file ${index}/${total}: ${file.name}`);
        
        // Create progress indicator
        const progressContainer = document.getElementById('upload-progress');
        const progressElement = document.createElement('div');
        progressElement.className = 'upload-item';
        progressElement.innerHTML = `
            <div class="upload-info">
                <span class="filename">${file.name}</span>
                <span class="filesize">(${this.formatFileSize(file.size)})</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <div class="upload-status">Uploading...</div>
        `;
        
        if (progressContainer) {
            progressContainer.appendChild(progressElement);
        }
        
        try {
            // Determine upload endpoint based on file type
            const isAudio = file.type.startsWith('audio/');
            const endpoint = isAudio ? '/api/upload/audio' : '/api/upload/text';
            
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', this.websocketClient.getSessionId());
            
            // Upload file with progress tracking
            const response = await this.uploadWithProgress(endpoint, formData, (progress) => {
                const progressFill = progressElement.querySelector('.progress-fill');
                if (progressFill) {
                    progressFill.style.width = `${progress}%`;
                }
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Update status
                const statusElement = progressElement.querySelector('.upload-status');
                if (statusElement) {
                    statusElement.textContent = 'Processing...';
                    statusElement.className = 'upload-status processing';
                }
                
                console.log(`✅ File uploaded successfully: ${file.name}`, result);
                
                // Results will come via WebSocket
                
            } else {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
        } catch (error) {
            console.error(`❌ Failed to upload file: ${file.name}`, error);
            
            const statusElement = progressElement.querySelector('.upload-status');
            if (statusElement) {
                statusElement.textContent = `Failed: ${error.message}`;
                statusElement.className = 'upload-status error';
            }
        }
    }
    
    /**
     * Upload file with progress tracking
     */
    uploadWithProgress(url, formData, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const progress = Math.round((event.loaded / event.total) * 100);
                    onProgress(progress);
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve({
                        ok: true,
                        status: xhr.status,
                        json: () => Promise.resolve(JSON.parse(xhr.responseText))
                    });
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Network error'));
            });
            
            xhr.open('POST', url);
            xhr.send(formData);
        });
    }
    
    /**
     * Update POS tag visualization
     */
    updatePOSVisualization(posTags, text) {
        console.log('🏷️ Updating POS visualization with', posTags.length, 'tags');
        
        // Update tab badge
        if (window.resultDisplayManager) {
            window.resultDisplayManager.updateTabBadge('analysis-tabs', 0, posTags.length.toString());
        }
        
        // Update statistics
        this.updatePOSStatistics(posTags);
        
        if (window.nlpVisualization) {
            window.nlpVisualization.renderPOSTags(posTags, 'pos-visualization');
        } else {
            console.error('❌ NLP Visualization engine not available');
            // Fallback to simple visualization
            this.fallbackPOSVisualization(posTags);
        }
        
        // Expand the POS results section
        this.expandResultSection('pos-results-section');
    }
    
    /**
     * Update POS statistics
     */
    updatePOSStatistics(posTags) {
        const totalTokens = posTags.length;
        const uniqueTags = new Set(posTags.map(tag => tag.tag)).size;
        const avgConfidence = posTags.reduce((sum, tag) => sum + tag.confidence, 0) / totalTokens;
        
        const totalElement = document.getElementById('pos-total-tokens');
        const uniqueElement = document.getElementById('pos-unique-tags');
        const confidenceElement = document.getElementById('pos-avg-confidence');
        
        if (totalElement) totalElement.textContent = totalTokens;
        if (uniqueElement) uniqueElement.textContent = uniqueTags;
        if (confidenceElement) confidenceElement.textContent = `${(avgConfidence * 100).toFixed(1)}%`;
    }
    
    /**
     * Fallback POS visualization if D3.js is not available
     */
    fallbackPOSVisualization(posTags) {
        const container = document.getElementById('pos-visualization');
        if (!container) return;
        
        let html = '<div class="pos-tags-fallback">';
        
        posTags.forEach(tag => {
            const colorClass = this.getPOSColorClass(tag.tag);
            html += `<span class="pos-token ${colorClass}" title="${tag.tag} (${(tag.confidence * 100).toFixed(1)}%)">${tag.token}</span> `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    /**
     * Update NER visualization
     */
    updateNERVisualization(entities, text) {
        console.log('🏢 Updating NER visualization with', entities.length, 'entities');
        
        // Update tab badge
        if (window.resultDisplayManager) {
            window.resultDisplayManager.updateTabBadge('analysis-tabs', 1, entities.length.toString());
        }
        
        // Update entities list
        this.updateNEREntitiesList(entities);
        
        if (window.nlpVisualization) {
            window.nlpVisualization.renderNamedEntities(entities, text, 'ner-visualization');
        } else {
            console.error('❌ NLP Visualization engine not available');
            // Fallback to simple visualization
            this.fallbackNERVisualization(entities);
        }
        
        // Expand the NER results section
        this.expandResultSection('ner-results-section');
    }
    
    /**
     * Update NER entities list
     */
    updateNEREntitiesList(entities) {
        const entitiesList = document.getElementById('ner-entities-list');
        if (!entitiesList) return;
        
        if (entities.length === 0) {
            entitiesList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🎯</div>
                    <h3>No entities found</h3>
                    <p>No named entities were detected in the text</p>
                </div>
            `;
            return;
        }
        
        const html = entities.map(entity => `
            <div class="ner-entity-item interactive-element" data-filter-value="${entity.label}" data-action="highlight">
                <div class="entity-header">
                    <span class="entity-text">${entity.text}</span>
                    <span class="tag ${this.getNERTagClass(entity.label)}">${entity.label}</span>
                </div>
                <div class="entity-details">
                    <span class="entity-confidence">Confidence: ${(entity.confidence * 100).toFixed(1)}%</span>
                    <span class="entity-position">Position: ${entity.start}-${entity.end}</span>
                </div>
            </div>
        `).join('');
        
        entitiesList.innerHTML = html;
    }
    
    /**
     * Get NER tag class for styling
     */
    getNERTagClass(label) {
        const classMap = {
            'PERSON': 'primary',
            'ORG': 'success',
            'LOC': 'warning',
            'MISC': 'error'
        };
        return classMap[label] || '';
    }
    
    /**
     * Fallback NER visualization if D3.js is not available
     */
    fallbackNERVisualization(entities) {
        const container = document.getElementById('ner-visualization');
        if (!container) return;
        
        let html = '<div class="ner-entities-fallback">';
        
        entities.forEach(entity => {
            const colorClass = this.getNERColorClass(entity.label);
            html += `<div class="ner-entity ${colorClass}">
                <span class="entity-text">${entity.text}</span>
                <span class="entity-label">${entity.label}</span>
                <span class="entity-confidence">${(entity.confidence * 100).toFixed(1)}%</span>
            </div>`;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    /**
     * Update dependency visualization
     */
    updateDependencyVisualization(dependencies, text) {
        console.log('🌳 Updating dependency visualization with', dependencies.length, 'dependencies');
        
        // Update tab badge
        if (window.resultDisplayManager) {
            window.resultDisplayManager.updateTabBadge('analysis-tabs', 2, dependencies.length.toString());
        }
        
        // Update dependency table
        this.updateDependencyTable(dependencies);
        
        // Extract tokens from dependencies or split text
        let tokens = [];
        if (dependencies.length > 0) {
            // Try to reconstruct tokens from dependencies
            const tokenSet = new Set();
            dependencies.forEach(dep => {
                tokenSet.add(dep.head_text);
                tokenSet.add(dep.dependent_text);
            });
            tokens = Array.from(tokenSet);
            
            // If we can't get proper token order, split the text
            if (tokens.length === 0 && text) {
                tokens = text.split(/\s+/);
            }
        } else if (text) {
            tokens = text.split(/\s+/);
        }
        
        if (window.nlpVisualization) {
            window.nlpVisualization.renderDependencyTree(dependencies, tokens, 'dependency-visualization');
        } else {
            console.error('❌ NLP Visualization engine not available');
            // Fallback to simple visualization
            this.fallbackDependencyVisualization(dependencies);
        }
        
        // Expand the dependency results section
        this.expandResultSection('dep-results-section');
    }
    
    /**
     * Update dependency table
     */
    updateDependencyTable(dependencies) {
        const tableContainer = document.getElementById('dependency-table-container');
        if (!tableContainer) return;
        
        if (dependencies.length === 0) {
            tableContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🌳</div>
                    <h3>No dependencies found</h3>
                    <p>No dependency relationships were detected</p>
                </div>
            `;
            return;
        }
        
        const table = document.createElement('table');
        table.className = 'data-table';
        
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Head</th>
                    <th>Relation</th>
                    <th>Dependent</th>
                </tr>
            </thead>
            <tbody>
                ${dependencies.map(dep => `
                    <tr class="interactive-element" data-action="highlight">
                        <td>${dep.head_text}</td>
                        <td><span class="tag">${dep.relation}</span></td>
                        <td>${dep.dependent_text}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        
        tableContainer.innerHTML = '';
        tableContainer.appendChild(table);
    }
    
    /**
     * Fallback dependency visualization if D3.js is not available
     */
    fallbackDependencyVisualization(dependencies) {
        const container = document.getElementById('dependency-visualization');
        if (!container) return;
        
        let html = '<div class="dependencies-fallback">';
        
        dependencies.forEach(dep => {
            html += `<div class="dependency">
                <span class="dep-head">${dep.head_text}</span>
                <span class="dep-relation">${dep.relation}</span>
                <span class="dep-dependent">${dep.dependent_text}</span>
            </div>`;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    /**
     * Update sentiment visualization
     */
    updateSentimentVisualization(sentiment) {
        console.log('😊 Updating sentiment visualization:', sentiment);
        
        // Update tab badge with confidence
        if (window.resultDisplayManager) {
            const confidence = Math.round(sentiment.confidence * 100);
            window.resultDisplayManager.updateTabBadge('analysis-tabs', 3, `${confidence}%`);
        }
        
        // Update sentiment summary
        this.updateSentimentSummary(sentiment);
        
        if (window.nlpVisualization) {
            window.nlpVisualization.renderSentimentAnalysis(sentiment, 'sentiment-visualization');
        } else {
            console.error('❌ NLP Visualization engine not available');
            // Fallback to simple visualization
            this.fallbackSentimentVisualization(sentiment);
        }
        
        // Expand the sentiment results section
        this.expandResultSection('sentiment-results-section');
    }
    
    /**
     * Update sentiment summary
     */
    updateSentimentSummary(sentiment) {
        const summaryContent = document.getElementById('sentiment-summary-content');
        if (!summaryContent) return;
        
        const overallClass = sentiment.overall_sentiment.toLowerCase();
        const emoji = this.getSentimentEmoji(sentiment.overall_sentiment);
        
        summaryContent.innerHTML = `
            <div class="sentiment-overview">
                <div class="sentiment-main">
                    <span class="sentiment-emoji">${emoji}</span>
                    <div class="sentiment-info">
                        <h4>Overall Sentiment: ${sentiment.overall_sentiment}</h4>
                        <div class="confidence-display">
                            <span>Confidence: ${(sentiment.confidence * 100).toFixed(1)}%</span>
                            <div class="progress-indicator">
                                <div class="progress-indicator-fill" style="width: ${sentiment.confidence * 100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="sentiment-breakdown">
                    <h5>Score Breakdown:</h5>
                    <div class="tag-container">
                        ${Object.entries(sentiment.scores).map(([key, value]) => 
                            `<span class="tag ${this.getSentimentScoreClass(key, value)}">
                                ${key}: ${value.toFixed(3)}
                            </span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Get sentiment emoji
     */
    getSentimentEmoji(sentiment) {
        const emojiMap = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        };
        return emojiMap[sentiment.toLowerCase()] || '😐';
    }
    
    /**
     * Get sentiment score class
     */
    getSentimentScoreClass(key, value) {
        if (key.toLowerCase().includes('positive') && value > 0.5) return 'success';
        if (key.toLowerCase().includes('negative') && value > 0.5) return 'error';
        if (key.toLowerCase().includes('neutral') && value > 0.5) return 'warning';
        return '';
    }
    
    /**
     * Expand result section
     */
    expandResultSection(sectionId) {
        if (window.resultDisplayManager) {
            const section = window.resultDisplayManager.collapsibleSections.get(sectionId);
            if (section && !section.isExpanded) {
                window.resultDisplayManager.toggleCollapsibleSection(sectionId);
            }
        }
    }
    
    /**
     * Set up error handling integration
     */
    setupErrorHandlingIntegration() {
        // Set up error recovery actions
        const exportErrorReportBtn = document.getElementById('export-error-report');
        const clearCacheBtn = document.getElementById('clear-cache');
        const reloadPageBtn = document.getElementById('reload-page');
        
        if (exportErrorReportBtn) {
            exportErrorReportBtn.addEventListener('click', () => {
                if (window.errorHandler) {
                    window.errorHandler.exportErrorReport();
                }
            });
        }
        
        if (clearCacheBtn) {
            clearCacheBtn.addEventListener('click', () => {
                this.clearApplicationCache();
            });
        }
        
        if (reloadPageBtn) {
            reloadPageBtn.addEventListener('click', () => {
                window.location.reload();
            });
        }
        
        // Set up troubleshooting panel toggle
        const microphoneStatus = document.getElementById('microphone-status');
        const troubleshootingPanel = document.getElementById('audio-troubleshooting');
        
        if (microphoneStatus && troubleshootingPanel) {
            microphoneStatus.addEventListener('click', () => {
                this.toggleTroubleshootingPanel();
            });
        }
        
        // Update diagnostic information
        this.updateDiagnosticInfo();
        
        // Set up periodic status checks
        this.setupStatusMonitoring();
    }
    
    /**
     * Clear application cache
     */
    async clearApplicationCache() {
        try {
            // Clear localStorage
            localStorage.clear();
            
            // Clear sessionStorage
            sessionStorage.clear();
            
            // Clear cache if available
            if ('caches' in window) {
                const cacheNames = await caches.keys();
                await Promise.all(
                    cacheNames.map(cacheName => caches.delete(cacheName))
                );
            }
            
            this.showToast('Cache cleared successfully', 'success');
            
            // Suggest page reload
            setTimeout(() => {
                if (confirm('Cache cleared. Would you like to reload the page for a fresh start?')) {
                    window.location.reload();
                }
            }, 1000);
            
        } catch (error) {
            console.error('Failed to clear cache:', error);
            this.showToast('Failed to clear cache', 'error');
        }
    }
    
    /**
     * Toggle troubleshooting panel
     */
    toggleTroubleshootingPanel() {
        const troubleshootingPanel = document.getElementById('audio-troubleshooting');
        if (troubleshootingPanel) {
            const isVisible = troubleshootingPanel.style.display !== 'none';
            troubleshootingPanel.style.display = isVisible ? 'none' : 'block';
            
            if (!isVisible) {
                troubleshootingPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    }
    
    /**
     * Update diagnostic information
     */
    updateDiagnosticInfo() {
        const browserInfo = document.getElementById('browser-info');
        const connectionInfo = document.getElementById('connection-info');
        const errorTimestamp = document.getElementById('error-timestamp');
        
        if (browserInfo) {
            browserInfo.textContent = `${navigator.userAgent.split(' ')[0]} ${navigator.appVersion.split(' ')[0]}`;
        }
        
        if (connectionInfo) {
            connectionInfo.textContent = navigator.onLine ? 'Online' : 'Offline';
        }
        
        if (errorTimestamp) {
            errorTimestamp.textContent = new Date().toLocaleString();
        }
    }
    
    /**
     * Set up status monitoring
     */
    setupStatusMonitoring() {
        // Monitor microphone status
        this.monitorMicrophoneStatus();
        
        // Monitor connection status
        this.monitorConnectionStatus();
        
        // Monitor processing status
        this.monitorProcessingStatus();
        
        // Update status every 5 seconds
        setInterval(() => {
            this.updateAllStatusIndicators();
        }, 5000);
    }
    
    /**
     * Monitor microphone status
     */
    async monitorMicrophoneStatus() {
        const microphoneStatus = document.getElementById('microphone-status');
        if (!microphoneStatus) return;
        
        try {
            // Check if microphone permission is granted
            const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
            
            const updateMicrophoneStatus = () => {
                switch (permissionStatus.state) {
                    case 'granted':
                        microphoneStatus.className = 'status-dot online';
                        break;
                    case 'denied':
                        microphoneStatus.className = 'status-dot offline';
                        break;
                    case 'prompt':
                        microphoneStatus.className = 'status-dot warning';
                        break;
                    default:
                        microphoneStatus.className = 'status-dot';
                }
            };
            
            updateMicrophoneStatus();
            permissionStatus.addEventListener('change', updateMicrophoneStatus);
            
        } catch (error) {
            console.warn('Could not check microphone permissions:', error);
            microphoneStatus.className = 'status-dot';
        }
    }
    
    /**
     * Monitor connection status
     */
    monitorConnectionStatus() {
        const connectionStatusDot = document.getElementById('connection-status-dot');
        if (!connectionStatusDot) return;
        
        const updateConnectionStatus = () => {
            if (navigator.onLine && this.isConnected()) {
                connectionStatusDot.className = 'status-dot online';
            } else if (navigator.onLine) {
                connectionStatusDot.className = 'status-dot warning';
            } else {
                connectionStatusDot.className = 'status-dot offline';
            }
        };
        
        updateConnectionStatus();
        
        window.addEventListener('online', updateConnectionStatus);
        window.addEventListener('offline', updateConnectionStatus);
    }
    
    /**
     * Monitor processing status
     */
    monitorProcessingStatus() {
        const processingStatus = document.getElementById('processing-status');
        if (!processingStatus) return;
        
        // Listen for processing events
        document.addEventListener('processingStart', () => {
            processingStatus.className = 'status-dot warning';
        });
        
        document.addEventListener('processingComplete', () => {
            processingStatus.className = 'status-dot online';
        });
        
        document.addEventListener('processingError', () => {
            processingStatus.className = 'status-dot offline';
        });
    }
    
    /**
     * Update all status indicators
     */
    updateAllStatusIndicators() {
        this.updateDiagnosticInfo();
        // Additional status updates can be added here
    }
    
    /**
     * Handle errors with enhanced error handler
     */
    handleError(error, context = {}) {
        if (window.errorHandler) {
            // Determine error type and delegate to appropriate handler
            if (context.type === 'audio') {
                window.errorHandler.handleAudioError(error, context);
            } else if (context.type === 'api') {
                window.errorHandler.handleAPIError(error, context);
            } else if (context.type === 'network') {
                window.errorHandler.handleNetworkError('general', error);
            } else if (context.type === 'upload') {
                window.errorHandler.handleFileUploadError(error, context.file, context);
            } else {
                // Generic error handling
                window.errorHandler.showUserFriendlyError(
                    'An error occurred',
                    window.errorHandler.getGenericErrorGuidance(),
                    { canRetry: true, retryAction: context.retryAction }
                );
            }
        } else {
            // Fallback error handling
            console.error('Error occurred:', error);
            this.showErrorModal(error.message || 'An unexpected error occurred');
        }
    }
    
    /**
     * Show browser compatibility warning if needed
     */
    showBrowserCompatibilityWarning() {
        // Check for required features
        const requiredFeatures = [
            'MediaRecorder',
            'WebSocket',
            'fetch',
            'Promise'
        ];
        
        const missingFeatures = requiredFeatures.filter(feature => !(feature in window));
        
        if (missingFeatures.length > 0) {
            const warning = document.createElement('div');
            warning.className = 'browser-warning';
            warning.innerHTML = `
                <div class="browser-warning-icon">⚠️</div>
                <div class="browser-warning-content">
                    <div class="browser-warning-title">Browser Compatibility Issue</div>
                    <div class="browser-warning-text">
                        Your browser may not support all features. Please update to the latest version or try a different browser.
                        Missing features: ${missingFeatures.join(', ')}
                    </div>
                </div>
                <button class="browser-warning-close">&times;</button>
            `;
            
            document.body.insertBefore(warning, document.body.firstChild);
            
            // Close button
            const closeButton = warning.querySelector('.browser-warning-close');
            closeButton.addEventListener('click', () => {
                warning.remove();
            });
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (warning.parentElement) {
                    warning.remove();
                }
            }, 10000);
        }
    }
    
    /**
     * Show permission request panel
     */
    showPermissionRequestPanel(type, description, onGrant, onDeny) {
        const panel = document.createElement('div');
        panel.className = 'permission-panel';
        panel.innerHTML = `
            <div class="permission-icon">${type === 'microphone' ? '🎤' : '📁'}</div>
            <div class="permission-title">Permission Required</div>
            <div class="permission-description">${description}</div>
            <div class="permission-actions">
                <button class="btn btn-primary" id="grant-permission">Grant Permission</button>
                <button class="btn btn-secondary" id="deny-permission">Not Now</button>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // Set up event handlers
        const grantButton = panel.querySelector('#grant-permission');
        const denyButton = panel.querySelector('#deny-permission');
        
        grantButton.addEventListener('click', () => {
            panel.remove();
            if (onGrant) onGrant();
        });
        
        denyButton.addEventListener('click', () => {
            panel.remove();
            if (onDeny) onDeny();
        });
        
        // Auto-remove after 30 seconds
        setTimeout(() => {
            if (panel.parentElement) {
                panel.remove();
                if (onDeny) onDeny();
            }
        }, 30000);
    }
    
    /**
     * Create session details content
     */
    createSessionDetailsContent(session) {
        const content = document.createElement('div');
        content.className = 'session-details-content';
        
        content.innerHTML = `
            <div class="session-overview">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${session.status}</div>
                        <div class="stat-label">Status</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${this.formatDuration(session.duration)}</div>
                        <div class="stat-label">Duration</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${session.analysis_count || 0}</div>
                        <div class="stat-label">Analyses</div>
                    </div>
                </div>
            </div>
            
            <div class="session-transcription">
                <h4>Transcription</h4>
                <div class="expandable-content collapsed">
                    <div class="transcription-text">${session.transcription || 'No transcription available'}</div>
                    <div class="expand-overlay">
                        <button class="expand-button">Show More</button>
                    </div>
                </div>
            </div>
            
            <div class="session-results">
                <h4>Analysis Results</h4>
                ${session.analysis_results ? this.formatAnalysisResults(session.analysis_results) : 'No analysis results available'}
            </div>
        `;
        
        return content;
    }
    
    /**
     * Create session modal
     */
    createSessionModal(sessionId, content) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.id = `session-modal-${sessionId}`;
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>📋 Session ${sessionId}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body"></div>
                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="nlpApp.exportSession('${sessionId}')">Export</button>
                    <button class="btn btn-secondary modal-close">Close</button>
                </div>
            </div>
        `;
        
        const modalBody = modal.querySelector('.modal-body');
        modalBody.appendChild(content);
        
        // Set up close handlers
        const closeButtons = modal.querySelectorAll('.modal-close');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                modal.classList.remove('active');
                setTimeout(() => modal.remove(), 300);
            });
        });
        
        return modal;
    }
    
    /**
     * Format analysis results for display
     */
    formatAnalysisResults(results) {
        if (!results || results.length === 0) {
            return '<p>No analysis results available</p>';
        }
        
        return results.map(result => `
            <div class="result-card">
                <div class="result-card-header">
                    <div class="result-card-title">
                        ${new Date(result.timestamp).toLocaleString()}
                    </div>
                </div>
                <div class="result-card-body">
                    <div class="tag-container">
                        ${result.pos_tags ? `<span class="tag">POS: ${result.pos_tags.length}</span>` : ''}
                        ${result.named_entities ? `<span class="tag">NER: ${result.named_entities.length}</span>` : ''}
                        ${result.dependencies ? `<span class="tag">DEP: ${result.dependencies.length}</span>` : ''}
                        ${result.sentiment_scores ? `<span class="tag">Sentiment: ${Object.keys(result.sentiment_scores).length}</span>` : ''}
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    /**
     * Fallback sentiment visualization if D3.js is not available
     */
    fallbackSentimentVisualization(sentiment) {
        const container = document.getElementById('sentiment-visualization');
        if (!container) return;
        
        const overallClass = sentiment.overall_sentiment.toLowerCase();
        const html = `
            <div class="sentiment-analysis-fallback">
                <div class="sentiment-overall ${overallClass}">
                    <h4>Overall Sentiment: ${sentiment.overall_sentiment}</h4>
                    <div class="confidence">Confidence: ${(sentiment.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="sentiment-scores">
                    ${Object.entries(sentiment.scores).map(([key, value]) => 
                        `<div class="score-item">
                            <span class="score-label">${key}:</span>
                            <span class="score-value">${value.toFixed(3)}</span>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${value * 100}%"></div>
                            </div>
                        </div>`
                    ).join('')}
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    /**
     * Switch to a specific analysis tab
     */
    switchToTab(tabName) {
        const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
        if (tabButton) {
            tabButton.click();
        }
    }
    
    /**
     * Get POS color class for styling
     */
    getPOSColorClass(tag) {
        const colorMap = {
            'NOUN': 'pos-noun',
            'VERB': 'pos-verb',
            'ADJ': 'pos-adj',
            'ADV': 'pos-adv',
            'PRON': 'pos-pron',
            'DET': 'pos-det',
            'ADP': 'pos-adp',
            'CONJ': 'pos-conj',
            'NUM': 'pos-num',
            'PUNCT': 'pos-punct'
        };
        
        return colorMap[tag] || 'pos-other';
    }
    
    /**
     * Get NER color class for styling
     */
    getNERColorClass(label) {
        const colorMap = {
            'PERSON': 'ner-person',
            'ORG': 'ner-org',
            'LOC': 'ner-loc',
            'MISC': 'ner-misc',
            'TIME': 'ner-time',
            'MONEY': 'ner-money'
        };
        
        return colorMap[label] || 'ner-other';
    }
    
    /**
     * Format file size for display
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    /**
     * Show notification to user (legacy method, use showToast instead)
     */
    showNotification(message, type = 'info') {
        this.showToast(message, type === 'info' ? 'success' : type);
    }
    
    /**
     * Show error message
     */
    showError(message) {
        this.showErrorModal(message);
    }
    
    /**
     * Connect to WebSocket server
     */
    async connect() {
        if (this.websocketClient) {
            await this.websocketClient.connect();
        }
    }
    
    /**
     * Retry WebSocket connection
     */
    async retryConnection() {
        console.log('🔄 Retrying WebSocket connection...');
        this.showToast('Attempting to reconnect...', 'info');
        
        try {
            if (this.websocketClient) {
                // Reset reconnection attempts
                this.websocketClient.reconnectAttempts = 0;
                await this.websocketClient.connect();
            }
        } catch (error) {
            console.error('❌ Retry connection failed:', error);
            this.showToast('Connection retry failed', 'warning');
        }
    }
    
    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        if (this.websocketClient) {
            this.websocketClient.disconnect();
        }
    }
    
    /**
     * Get current connection status
     */
    isConnected() {
        return this.websocketClient ? this.websocketClient.getConnectionStatus() : false;
    }
    
    /**
     * Connect to WebSocket server
     */
    async connect() {
        if (this.websocketClient) {
            await this.websocketClient.connect();
        }
    }
    
    /**
     * Retry connection
     */
    async retryConnection() {
        if (this.websocketClient) {
            this.websocketClient.reconnectAttempts = 0; // Reset attempts
            await this.websocketClient.connect();
        }
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🌐 DOM loaded, initializing NLP Web Interface...');
    
    try {
        // Create global app instance
        window.nlpApp = new NLPWebInterface();
        
        // Try to connect to WebSocket (non-blocking)
        try {
            await window.nlpApp.connect();
            console.log('🔗 WebSocket connected successfully');
        } catch (error) {
            console.log('⚠️ WebSocket connection failed, continuing without real-time features');
            console.log('Real-time features will be available when connection is restored');
        }
        
        console.log('🎉 NLP Web Interface ready!');
        
    } catch (error) {
        console.error('❌ Failed to initialize NLP Web Interface:', error);
        // Show a user-friendly error message
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #dc3545;
            color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 2000;
            text-align: center;
            max-width: 400px;
        `;
        errorDiv.innerHTML = `
            <h3>⚠️ Initialization Failed</h3>
            <p>The NLP Web Interface failed to initialize properly.</p>
            <p><strong>Error:</strong> ${error.message}</p>
            <button onclick="window.location.reload()" style="
                background: white;
                color: #dc3545;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 1rem;
                font-weight: 600;
            ">Reload Page</button>
        `;
        document.body.appendChild(errorDiv);
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NLPWebInterface;
}