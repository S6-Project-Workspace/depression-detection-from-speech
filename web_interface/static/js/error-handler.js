/**
 * Error Handling and User Feedback Module
 * Provides comprehensive error handling, user guidance, and troubleshooting features
 */

class ErrorHandler {
    constructor() {
        this.errorHistory = [];
        this.retryQueue = [];
        this.userGuidance = new Map();
        this.troubleshootingSteps = new Map();
        
        this.init();
    }
    
    /**
     * Initialize error handling system
     */
    init() {
        console.log('🛡️ Initializing Error Handler...');
        
        this.setupGlobalErrorHandling();
        this.setupUserGuidance();
        this.setupTroubleshootingSteps();
        this.setupErrorRecovery();
        this.setupNetworkErrorHandling();
        
        console.log('✅ Error Handler initialized');
    }
    
    /**
     * Set up global error handling
     */
    setupGlobalErrorHandling() {
        // Handle unhandled JavaScript errors
        window.addEventListener('error', (event) => {
            this.handleJavaScriptError(event.error, event.filename, event.lineno, event.colno);
        });
        
        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.handlePromiseRejection(event.reason);
            event.preventDefault(); // Prevent console error
        });
        
        // Handle network errors
        window.addEventListener('offline', () => {
            this.handleNetworkError('offline');
        });
        
        window.addEventListener('online', () => {
            this.handleNetworkRecovery();
        });
    }
    
    /**
     * Handle JavaScript errors
     */
    handleJavaScriptError(error, filename, lineno, colno) {
        const errorInfo = {
            type: 'javascript',
            message: error?.message || 'Unknown JavaScript error',
            filename: filename || 'unknown',
            line: lineno || 0,
            column: colno || 0,
            stack: error?.stack || '',
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent
        };
        
        this.logError(errorInfo);
        this.showUserFriendlyError('A technical error occurred', this.getJavaScriptErrorGuidance(error));
    }
    
    /**
     * Handle promise rejections
     */
    handlePromiseRejection(reason) {
        const errorInfo = {
            type: 'promise_rejection',
            message: reason?.message || reason || 'Unhandled promise rejection',
            stack: reason?.stack || '',
            timestamp: new Date().toISOString()
        };
        
        this.logError(errorInfo);
        
        // Determine if this is a network error, API error, etc.
        if (this.isNetworkError(reason)) {
            this.handleNetworkError('promise_rejection', reason);
        } else if (this.isAPIError(reason)) {
            this.handleAPIError(reason);
        } else {
            this.showUserFriendlyError('An unexpected error occurred', this.getGenericErrorGuidance());
        }
    }
    
    /**
     * Handle API errors
     */
    handleAPIError(error, context = {}) {
        const errorInfo = {
            type: 'api_error',
            message: error?.message || 'API request failed',
            status: error?.status || 0,
            statusText: error?.statusText || '',
            url: error?.url || context.url || '',
            method: context.method || 'GET',
            timestamp: new Date().toISOString()
        };
        
        this.logError(errorInfo);
        
        // Provide specific guidance based on status code
        let guidance = this.getAPIErrorGuidance(errorInfo.status);
        let title = this.getAPIErrorTitle(errorInfo.status);
        
        this.showUserFriendlyError(title, guidance, {
            canRetry: this.canRetryAPIError(errorInfo.status),
            retryAction: () => this.retryLastAPICall(context)
        });
    }
    
    /**
     * Handle network errors
     */
    handleNetworkError(type, error = null) {
        const errorInfo = {
            type: 'network_error',
            subtype: type,
            message: error?.message || 'Network connection issue',
            isOnline: navigator.onLine,
            timestamp: new Date().toISOString()
        };
        
        this.logError(errorInfo);
        
        if (type === 'offline') {
            this.showNetworkOfflineMessage();
        } else {
            this.showUserFriendlyError(
                'Connection Problem',
                this.getNetworkErrorGuidance(),
                {
                    canRetry: true,
                    retryAction: () => this.retryNetworkOperation()
                }
            );
        }
    }
    
    /**
     * Handle network recovery
     */
    handleNetworkRecovery() {
        this.hideNetworkOfflineMessage();
        this.showToast('Connection restored', 'success');
        
        // Retry queued operations
        this.processRetryQueue();
    }
    
    /**
     * Handle audio/microphone errors
     */
    handleAudioError(error, context = {}) {
        const errorInfo = {
            type: 'audio_error',
            message: error?.message || 'Audio processing error',
            name: error?.name || '',
            constraint: error?.constraint || '',
            timestamp: new Date().toISOString(),
            context
        };
        
        this.logError(errorInfo);
        
        let title, guidance;
        
        if (error?.name === 'NotAllowedError') {
            title = 'Microphone Access Denied';
            guidance = this.getMicrophonePermissionGuidance();
        } else if (error?.name === 'NotFoundError') {
            title = 'No Microphone Found';
            guidance = this.getMicrophoneNotFoundGuidance();
        } else if (error?.name === 'NotReadableError') {
            title = 'Microphone Not Available';
            guidance = this.getMicrophoneNotReadableGuidance();
        } else {
            title = 'Audio Processing Error';
            guidance = this.getGenericAudioErrorGuidance();
        }
        
        this.showUserFriendlyError(title, guidance, {
            canRetry: true,
            retryAction: () => this.retryAudioOperation(context)
        });
    }
    
    /**
     * Handle file upload errors
     */
    handleFileUploadError(error, file, context = {}) {
        const errorInfo = {
            type: 'file_upload_error',
            message: error?.message || 'File upload failed',
            fileName: file?.name || 'unknown',
            fileSize: file?.size || 0,
            fileType: file?.type || 'unknown',
            timestamp: new Date().toISOString(),
            context
        };
        
        this.logError(errorInfo);
        
        let title, guidance;
        
        if (error?.message?.includes('size')) {
            title = 'File Too Large';
            guidance = this.getFileSizeErrorGuidance(file);
        } else if (error?.message?.includes('type')) {
            title = 'Unsupported File Type';
            guidance = this.getFileTypeErrorGuidance(file);
        } else if (error?.message?.includes('network')) {
            title = 'Upload Failed';
            guidance = this.getUploadNetworkErrorGuidance();
        } else {
            title = 'Upload Error';
            guidance = this.getGenericUploadErrorGuidance();
        }
        
        this.showUserFriendlyError(title, guidance, {
            canRetry: true,
            retryAction: () => this.retryFileUpload(file, context)
        });
    }
    
    /**
     * Set up user guidance messages
     */
    setupUserGuidance() {
        this.userGuidance.set('microphone_permission', {
            title: 'Enable Microphone Access',
            steps: [
                'Click the microphone icon in your browser\'s address bar',
                'Select "Allow" to grant microphone permission',
                'Refresh the page if needed',
                'Try recording again'
            ],
            browserSpecific: {
                chrome: 'Look for the microphone icon next to the star in the address bar',
                firefox: 'Click the microphone icon in the address bar',
                safari: 'Go to Safari > Preferences > Websites > Microphone',
                edge: 'Click the microphone icon in the address bar'
            }
        });
        
        this.userGuidance.set('microphone_not_found', {
            title: 'Connect a Microphone',
            steps: [
                'Check that your microphone is properly connected',
                'Try unplugging and reconnecting your microphone',
                'Check your system\'s audio settings',
                'Restart your browser if the issue persists'
            ]
        });
        
        this.userGuidance.set('network_error', {
            title: 'Check Your Connection',
            steps: [
                'Verify your internet connection is working',
                'Try refreshing the page',
                'Check if other websites are loading',
                'Contact your network administrator if the problem persists'
            ]
        });
        
        this.userGuidance.set('api_error_500', {
            title: 'Server Error',
            steps: [
                'The server is experiencing issues',
                'Please try again in a few minutes',
                'If the problem persists, contact support',
                'Your data has been saved locally when possible'
            ]
        });
    }
    
    /**
     * Set up troubleshooting steps
     */
    setupTroubleshootingSteps() {
        this.troubleshootingSteps.set('audio_quality_poor', [
            'Move closer to your microphone',
            'Reduce background noise',
            'Check microphone settings in your system',
            'Try using a different microphone',
            'Close other applications using the microphone'
        ]);
        
        this.troubleshootingSteps.set('transcription_inaccurate', [
            'Speak more clearly and slowly',
            'Ensure good audio quality',
            'Reduce background noise',
            'Check your microphone positioning',
            'Try recording shorter segments'
        ]);
        
        this.troubleshootingSteps.set('analysis_slow', [
            'Check your internet connection speed',
            'Try analyzing shorter text segments',
            'Close other browser tabs to free up memory',
            'Wait for current analysis to complete before starting new ones'
        ]);
    }
    
    /**
     * Set up error recovery mechanisms
     */
    setupErrorRecovery() {
        // Auto-retry mechanism for transient errors
        this.autoRetryConfig = {
            maxRetries: 3,
            retryDelay: 1000, // Start with 1 second
            backoffMultiplier: 2,
            retryableErrors: ['network_error', 'api_error_502', 'api_error_503', 'api_error_504']
        };
    }
    
    /**
     * Set up network error handling
     */
    setupNetworkErrorHandling() {
        // Monitor network connectivity
        window.addEventListener('online', () => {
            console.log('🌐 Network connection restored');
            this.hideOfflineMessage();
            this.handleNetworkRecovery();
        });
        
        window.addEventListener('offline', () => {
            console.log('📡 Network connection lost');
            this.showOfflineMessage();
        });
        
        // Set up fetch interceptor for network errors
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            try {
                const response = await originalFetch(...args);
                if (!response.ok) {
                    this.handleAPIError(response, args[0]);
                }
                return response;
            } catch (error) {
                this.handleNetworkError(error, args[0]);
                throw error;
            }
        };
    }
    
    /**
     * Show user-friendly error message
     */
    showUserFriendlyError(title, guidance, options = {}) {
        const errorModal = document.getElementById('error-modal');
        const errorMessage = document.getElementById('error-message');
        const errorDetails = document.getElementById('error-details');
        const retryButton = document.getElementById('error-retry');
        
        if (!errorModal || !errorMessage) {
            console.error('Error modal elements not found');
            return;
        }
        
        // Set title and message
        const modalTitle = errorModal.querySelector('.modal-header h3');
        if (modalTitle) modalTitle.textContent = `⚠️ ${title}`;
        
        errorMessage.textContent = guidance.message || guidance;
        
        // Set up guidance details
        if (guidance.steps) {
            const stepsHtml = `
                <div class="error-guidance">
                    <h4>How to fix this:</h4>
                    <ol>
                        ${guidance.steps.map(step => `<li>${step}</li>`).join('')}
                    </ol>
                </div>
            `;
            errorDetails.innerHTML = stepsHtml;
            errorDetails.style.display = 'block';
        } else {
            errorDetails.style.display = 'none';
        }
        
        // Set up retry button
        if (retryButton) {
            if (options.canRetry && options.retryAction) {
                retryButton.style.display = 'inline-block';
                retryButton.onclick = () => {
                    this.hideErrorModal();
                    options.retryAction();
                };
            } else {
                retryButton.style.display = 'none';
            }
        }
        
        // Show modal
        errorModal.classList.add('active');
        
        // Auto-hide after delay for non-critical errors
        if (options.autoHide) {
            setTimeout(() => {
                this.hideErrorModal();
            }, options.autoHideDelay || 5000);
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
     * Show network offline message
     */
    showNetworkOfflineMessage() {
        let offlineMessage = document.getElementById('offline-message');
        
        if (!offlineMessage) {
            offlineMessage = document.createElement('div');
            offlineMessage.id = 'offline-message';
            offlineMessage.className = 'offline-banner';
            offlineMessage.innerHTML = `
                <div class="offline-content">
                    <span class="offline-icon">📡</span>
                    <span class="offline-text">You're currently offline. Some features may not work.</span>
                    <button class="btn btn-small" onclick="location.reload()">Retry</button>
                </div>
            `;
            document.body.appendChild(offlineMessage);
        }
        
        offlineMessage.classList.add('active');
    }
    
    /**
     * Hide network offline message
     */
    hideNetworkOfflineMessage() {
        const offlineMessage = document.getElementById('offline-message');
        if (offlineMessage) {
            offlineMessage.classList.remove('active');
        }
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 4000) {
        // Use the main app's toast system if available
        if (window.nlpApp && window.nlpApp.showToast) {
            window.nlpApp.showToast(message, type, duration);
            return;
        }
        
        // Fallback toast implementation
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${this.getToastIcon(type)}</span>
                <span class="toast-message">${message}</span>
            </div>
            <button class="toast-close">&times;</button>
        `;
        
        document.body.appendChild(toast);
        toast.classList.add('active');
        
        // Auto-hide
        setTimeout(() => {
            toast.classList.remove('active');
            setTimeout(() => toast.remove(), 300);
        }, duration);
        
        // Close button
        const closeButton = toast.querySelector('.toast-close');
        closeButton.addEventListener('click', () => {
            toast.classList.remove('active');
            setTimeout(() => toast.remove(), 300);
        });
    }
    
    /**
     * Get toast icon for type
     */
    getToastIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }
    
    /**
     * Log error for debugging and analytics
     */
    logError(errorInfo) {
        this.errorHistory.push(errorInfo);
        
        // Keep only last 50 errors to prevent memory issues
        if (this.errorHistory.length > 50) {
            this.errorHistory.shift();
        }
        
        // Log to console in development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            console.error('🚨 Error logged:', errorInfo);
        }
        
        // Send to analytics/monitoring service in production
        this.sendErrorToAnalytics(errorInfo);
    }
    
    /**
     * Send error to analytics service
     */
    sendErrorToAnalytics(errorInfo) {
        // Implementation would send error data to monitoring service
        // This is a placeholder for production error tracking
        if (window.gtag) {
            window.gtag('event', 'exception', {
                description: errorInfo.message,
                fatal: errorInfo.type === 'javascript'
            });
        }
    }
    
    /**
     * Get JavaScript error guidance
     */
    getJavaScriptErrorGuidance(error) {
        return {
            message: 'A technical error occurred while processing your request.',
            steps: [
                'Try refreshing the page',
                'Clear your browser cache and cookies',
                'Disable browser extensions temporarily',
                'Try using a different browser',
                'Contact support if the problem persists'
            ]
        };
    }
    
    /**
     * Get generic error guidance
     */
    getGenericErrorGuidance() {
        return {
            message: 'Something went wrong. Please try again.',
            steps: [
                'Refresh the page and try again',
                'Check your internet connection',
                'Clear your browser cache',
                'Contact support if the issue continues'
            ]
        };
    }
    
    /**
     * Get API error guidance based on status code
     */
    getAPIErrorGuidance(status) {
        const guidanceMap = {
            400: {
                message: 'The request was invalid. Please check your input and try again.',
                steps: [
                    'Verify that all required fields are filled',
                    'Check that file formats are supported',
                    'Ensure text is not too long',
                    'Try with different input data'
                ]
            },
            401: {
                message: 'Authentication failed. Please refresh the page.',
                steps: [
                    'Refresh the page to renew your session',
                    'Clear browser cookies and try again',
                    'Contact support if the problem persists'
                ]
            },
            403: {
                message: 'Access denied. You don\'t have permission for this action.',
                steps: [
                    'Check that you have the necessary permissions',
                    'Contact your administrator',
                    'Try logging out and back in'
                ]
            },
            404: {
                message: 'The requested resource was not found.',
                steps: [
                    'Check that the URL is correct',
                    'The resource may have been moved or deleted',
                    'Contact support for assistance'
                ]
            },
            429: {
                message: 'Too many requests. Please wait before trying again.',
                steps: [
                    'Wait a few minutes before retrying',
                    'Reduce the frequency of your requests',
                    'Contact support if you need higher limits'
                ]
            },
            500: {
                message: 'Server error. Our team has been notified.',
                steps: [
                    'Try again in a few minutes',
                    'The issue is on our end, not yours',
                    'Contact support if the problem persists',
                    'Your data has been preserved when possible'
                ]
            },
            502: {
                message: 'Service temporarily unavailable.',
                steps: [
                    'The service is temporarily down',
                    'Please try again in a few minutes',
                    'Check our status page for updates'
                ]
            },
            503: {
                message: 'Service overloaded. Please try again later.',
                steps: [
                    'The service is experiencing high traffic',
                    'Please wait and try again',
                    'Try during off-peak hours'
                ]
            }
        };
        
        return guidanceMap[status] || this.getGenericErrorGuidance();
    }
    
    /**
     * Get API error title based on status code
     */
    getAPIErrorTitle(status) {
        const titleMap = {
            400: 'Invalid Request',
            401: 'Authentication Required',
            403: 'Access Denied',
            404: 'Not Found',
            429: 'Rate Limited',
            500: 'Server Error',
            502: 'Service Unavailable',
            503: 'Service Overloaded'
        };
        
        return titleMap[status] || 'Request Failed';
    }
    
    /**
     * Get network error guidance
     */
    getNetworkErrorGuidance() {
        return this.userGuidance.get('network_error');
    }
    
    /**
     * Get microphone permission guidance
     */
    getMicrophonePermissionGuidance() {
        const guidance = this.userGuidance.get('microphone_permission');
        const browser = this.detectBrowser();
        
        if (guidance.browserSpecific[browser]) {
            guidance.steps.unshift(guidance.browserSpecific[browser]);
        }
        
        return guidance;
    }
    
    /**
     * Get microphone not found guidance
     */
    getMicrophoneNotFoundGuidance() {
        return this.userGuidance.get('microphone_not_found');
    }
    
    /**
     * Get microphone not readable guidance
     */
    getMicrophoneNotReadableGuidance() {
        return {
            message: 'Your microphone is being used by another application.',
            steps: [
                'Close other applications that might be using the microphone',
                'Check your system\'s audio settings',
                'Restart your browser',
                'Try using a different microphone'
            ]
        };
    }
    
    /**
     * Get generic audio error guidance
     */
    getGenericAudioErrorGuidance() {
        return {
            message: 'There was a problem with audio processing.',
            steps: [
                'Check your microphone connection',
                'Verify microphone permissions',
                'Try refreshing the page',
                'Contact support if the issue persists'
            ]
        };
    }
    
    /**
     * Get file size error guidance
     */
    getFileSizeErrorGuidance(file) {
        return {
            message: `The file "${file.name}" is too large. Maximum size is 10MB.`,
            steps: [
                'Choose a smaller file',
                'Compress your audio file',
                'Split large files into smaller segments',
                'Contact support for higher limits'
            ]
        };
    }
    
    /**
     * Get file type error guidance
     */
    getFileTypeErrorGuidance(file) {
        return {
            message: `The file type "${file.type}" is not supported.`,
            steps: [
                'Use supported formats: WAV, MP3, or TXT',
                'Convert your file to a supported format',
                'Check that the file is not corrupted',
                'Contact support for additional format support'
            ]
        };
    }
    
    /**
     * Get upload network error guidance
     */
    getUploadNetworkErrorGuidance() {
        return {
            message: 'The file upload failed due to a network error.',
            steps: [
                'Check your internet connection',
                'Try uploading a smaller file',
                'Wait and try again',
                'Contact support if uploads consistently fail'
            ]
        };
    }
    
    /**
     * Get generic upload error guidance
     */
    getGenericUploadErrorGuidance() {
        return {
            message: 'The file upload failed.',
            steps: [
                'Try uploading the file again',
                'Check that the file is not corrupted',
                'Ensure you have sufficient permissions',
                'Contact support if the problem persists'
            ]
        };
    }
    
    /**
     * Detect browser for specific guidance
     */
    detectBrowser() {
        const userAgent = navigator.userAgent.toLowerCase();
        
        if (userAgent.includes('chrome')) return 'chrome';
        if (userAgent.includes('firefox')) return 'firefox';
        if (userAgent.includes('safari')) return 'safari';
        if (userAgent.includes('edge')) return 'edge';
        
        return 'unknown';
    }
    
    /**
     * Check if error is network-related
     */
    isNetworkError(error) {
        if (!error) return false;
        
        const networkErrorMessages = [
            'network error',
            'fetch failed',
            'connection refused',
            'timeout',
            'no internet',
            'offline'
        ];
        
        const message = (error.message || error.toString()).toLowerCase();
        return networkErrorMessages.some(msg => message.includes(msg));
    }
    
    /**
     * Check if error is API-related
     */
    isAPIError(error) {
        return error?.status || error?.response || error?.url;
    }
    
    /**
     * Check if API error can be retried
     */
    canRetryAPIError(status) {
        const retryableStatuses = [408, 429, 500, 502, 503, 504];
        return retryableStatuses.includes(status);
    }
    
    /**
     * Add operation to retry queue
     */
    addToRetryQueue(operation) {
        this.retryQueue.push({
            operation,
            timestamp: Date.now(),
            attempts: 0
        });
    }
    
    /**
     * Process retry queue
     */
    processRetryQueue() {
        const now = Date.now();
        const retryableOperations = this.retryQueue.filter(item => 
            item.attempts < this.autoRetryConfig.maxRetries &&
            (now - item.timestamp) > this.getRetryDelay(item.attempts)
        );
        
        retryableOperations.forEach(item => {
            item.attempts++;
            item.timestamp = now;
            
            try {
                item.operation();
            } catch (error) {
                console.error('Retry operation failed:', error);
            }
        });
        
        // Remove operations that have exceeded max retries
        this.retryQueue = this.retryQueue.filter(item => 
            item.attempts < this.autoRetryConfig.maxRetries
        );
    }
    
    /**
     * Get retry delay with exponential backoff
     */
    getRetryDelay(attempts) {
        return this.autoRetryConfig.retryDelay * Math.pow(this.autoRetryConfig.backoffMultiplier, attempts);
    }
    
    /**
     * Retry last API call
     */
    retryLastAPICall(context) {
        console.log('🔄 Retrying API call:', context);
        // Implementation would retry the specific API call
        this.showToast('Retrying...', 'info');
    }
    
    /**
     * Retry network operation
     */
    retryNetworkOperation() {
        console.log('🔄 Retrying network operation...');
        // Implementation would retry the last network operation
        this.showToast('Checking connection...', 'info');
    }
    
    /**
     * Retry audio operation
     */
    retryAudioOperation(context) {
        console.log('🔄 Retrying audio operation:', context);
        // Implementation would retry audio capture/processing
        this.showToast('Retrying audio access...', 'info');
    }
    
    /**
     * Retry file upload
     */
    retryFileUpload(file, context) {
        console.log('🔄 Retrying file upload:', file.name);
        // Implementation would retry the file upload
        this.showToast(`Retrying upload of ${file.name}...`, 'info');
    }
    
    /**
     * Get error history for debugging
     */
    getErrorHistory() {
        return this.errorHistory;
    }
    
    /**
     * Clear error history
     */
    clearErrorHistory() {
        this.errorHistory = [];
    }
    
    /**
     * Export error report
     */
    exportErrorReport() {
        const report = {
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href,
            errors: this.errorHistory,
            systemInfo: {
                language: navigator.language,
                platform: navigator.platform,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine
            }
        };
        
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `error-report-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('Error report exported', 'success');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.errorHandler = new ErrorHandler();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ErrorHandler;
}