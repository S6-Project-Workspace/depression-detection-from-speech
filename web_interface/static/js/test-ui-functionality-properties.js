/**
 * Property-Based Tests for UI Functionality
 * Tests Properties 30, 31, and 33 from the design document
 * Validates Requirements: 7.1, 7.3, 7.5
 */

/**
 * Mock fast-check for property-based testing (simplified version)
 */
class UIPropertyTester {
    constructor() {
        this.testCount = 0;
        this.passCount = 0;
        this.failCount = 0;
        this.failures = [];
    }
    
    /**
     * Generate random float in range
     */
    static float(min = 0, max = 1) {
        return Math.random() * (max - min) + min;
    }
    
    /**
     * Generate random integer in range
     */
    static integer(min = 0, max = 100) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
    
    /**
     * Generate random boolean
     */
    static boolean() {
        return Math.random() < 0.5;
    }
    
    /**
     * Generate random string
     */
    static string(minLength = 1, maxLength = 50) {
        const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ';
        const length = this.integer(minLength, maxLength);
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }
    
    /**
     * Generate random array
     */
    static array(generator, minSize = 0, maxSize = 10) {
        const size = this.integer(minSize, maxSize);
        const result = [];
        for (let i = 0; i < size; i++) {
            result.push(generator());
        }
        return result;
    }
    
    /**
     * Generate random choice from array
     */
    static choice(choices) {
        return choices[Math.floor(Math.random() * choices.length)];
    }
    
    /**
     * Run property test
     */
    async property(name, generator, testFn, iterations = 100) {
        console.log(`\n🧪 Testing Property: ${name}`);
        console.log(`Running ${iterations} iterations...`);
        
        let localPassCount = 0;
        let localFailCount = 0;
        
        for (let i = 0; i < iterations; i++) {
            this.testCount++;
            
            try {
                const input = generator();
                const result = await testFn(input, i);
                
                if (result === false) {
                    throw new Error(`Property violated with input: ${JSON.stringify(input)}`);
                }
                
                this.passCount++;
                localPassCount++;
                
                // Show progress every 20 iterations
                if ((i + 1) % 20 === 0) {
                    console.log(`  ✓ ${i + 1}/${iterations} iterations passed`);
                }
                
            } catch (error) {
                this.failCount++;
                localFailCount++;
                this.failures.push({
                    property: name,
                    iteration: i + 1,
                    error: error.message,
                    input: generator()
                });
                
                console.error(`  ❌ Iteration ${i + 1} failed: ${error.message}`);
                
                // Continue testing but limit failures shown
                if (localFailCount >= 5) {
                    console.log(`  ... stopping after 5 failures for this property`);
                    break;
                }
            }
        }
        
        const passed = localFailCount === 0;
        console.log(`${passed ? '✅' : '❌'} Property "${name}": ${passed ? 'PASSED' : 'FAILED'} (${localPassCount}/${localPassCount + localFailCount})`);
        
        return passed;
    }
    
    /**
     * Get test summary
     */
    getSummary() {
        return {
            total: this.testCount,
            passed: this.passCount,
            failed: this.failCount,
            failures: this.failures
        };
    }
}

/**
 * UI Functionality Property Tests
 */
class UIFunctionalityPropertyTests {
    constructor() {
        this.tester = new UIPropertyTester();
        this.testContainer = null;
        this.originalViewport = null;
    }
    
    /**
     * Setup test environment
     */
    async setup() {
        console.log('🔧 Setting up UI functionality test environment...');
        
        // Store original viewport
        this.originalViewport = document.querySelector('meta[name="viewport"]');
        
        // Create test container
        this.createTestContainer();
        
        // Ensure main app is available
        if (!window.nlpApp) {
            console.warn('⚠️ Main NLP app not available, creating mock...');
            this.createMockApp();
        }
        
        console.log('✅ Test environment setup complete');
    }
    
    /**
     * Create test container in DOM
     */
    createTestContainer() {
        // Remove existing test container
        const existing = document.getElementById('ui-test-container');
        if (existing) {
            existing.remove();
        }
        
        // Create new test container
        this.testContainer = document.createElement('div');
        this.testContainer.id = 'ui-test-container';
        this.testContainer.style.cssText = `
            position: fixed;
            top: -9999px;
            left: -9999px;
            width: 100vw;
            height: 100vh;
            visibility: hidden;
            z-index: -1;
        `;
        
        // Add test UI elements
        this.testContainer.innerHTML = `
            <!-- Test responsive elements -->
            <div class="test-responsive-container">
                <nav class="navigation" id="test-navigation">
                    <div class="nav-items">
                        <button class="nav-btn" data-section="audio">Audio</button>
                        <button class="nav-btn" data-section="upload">Upload</button>
                        <button class="nav-btn" data-section="metrics">Metrics</button>
                    </div>
                    <button class="mobile-menu-toggle" id="test-mobile-menu-toggle">☰</button>
                </nav>
                
                <main class="main-content">
                    <section class="content-section active" id="test-audio-section">
                        <div class="audio-controls">
                            <button id="test-start-recording">Start Recording</button>
                        </div>
                    </section>
                </main>
            </div>
            
            <!-- Test loading elements -->
            <div class="test-loading-container">
                <div class="loading-overlay" id="test-loading-overlay">
                    <div class="loading-content">
                        <div class="loading-title" id="test-loading-title">Processing...</div>
                        <div class="loading-message" id="test-loading-message">Please wait</div>
                        <div class="loading-progress">
                            <div class="loading-progress-fill" id="test-loading-progress-fill"></div>
                        </div>
                        <div class="loading-percentage" id="test-loading-percentage">0%</div>
                    </div>
                </div>
                
                <div class="global-progress" id="test-global-progress">
                    <div class="progress-fill" id="test-progress-fill"></div>
                    <div class="progress-text" id="test-progress-text">Loading...</div>
                </div>
                
                <div class="tab-loading" id="test-tab-loading">
                    <div class="loading-spinner"></div>
                    <span>Loading data...</span>
                </div>
            </div>
            
            <!-- Test error elements -->
            <div class="test-error-container">
                <div class="error-modal" id="test-error-modal">
                    <div class="error-modal-content">
                        <div class="error-modal-header">
                            <h3>Error</h3>
                            <button class="error-modal-close" id="test-error-modal-close">&times;</button>
                        </div>
                        <div class="error-modal-body">
                            <div class="error-message" id="test-error-message">An error occurred</div>
                            <div class="error-details" id="test-error-details" style="display: none;"></div>
                        </div>
                        <div class="error-modal-footer">
                            <button class="btn btn-primary" id="test-error-retry">Retry</button>
                            <button class="btn btn-secondary" id="test-error-dismiss">Dismiss</button>
                        </div>
                    </div>
                </div>
                
                <div class="toast success-toast" id="test-success-toast">
                    <div class="toast-content">
                        <span class="toast-icon">✅</span>
                        <span class="toast-message" id="test-success-message">Success</span>
                        <button class="toast-close" id="test-success-toast-close">&times;</button>
                    </div>
                </div>
                
                <div class="toast warning-toast" id="test-warning-toast">
                    <div class="toast-content">
                        <span class="toast-icon">⚠️</span>
                        <span class="toast-message" id="test-warning-message">Warning</span>
                        <button class="toast-close" id="test-warning-toast-close">&times;</button>
                    </div>
                </div>
                
                <div class="troubleshooting-panel" id="test-troubleshooting-panel" style="display: none;">
                    <h4>Troubleshooting</h4>
                    <div class="diagnostic-info">
                        <div class="status-item">
                            <span class="status-label">Microphone:</span>
                            <span class="status-dot" id="test-microphone-status"></span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Connection:</span>
                            <span class="status-dot" id="test-connection-status"></span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.testContainer);
    }
    
    /**
     * Create mock app for testing
     */
    createMockApp() {
        window.nlpApp = {
            showLoading: (key, title, message, progress) => {
                const overlay = document.getElementById('test-loading-overlay');
                const titleEl = document.getElementById('test-loading-title');
                const messageEl = document.getElementById('test-loading-message');
                const progressEl = document.getElementById('test-loading-progress-fill');
                const percentageEl = document.getElementById('test-loading-percentage');
                
                if (overlay) overlay.classList.add('active');
                if (titleEl) titleEl.textContent = title || 'Processing...';
                if (messageEl) messageEl.textContent = message || 'Please wait';
                if (progressEl) progressEl.style.width = `${progress || 0}%`;
                if (percentageEl) percentageEl.textContent = `${Math.round(progress || 0)}%`;
            },
            
            hideLoading: (key) => {
                const overlay = document.getElementById('test-loading-overlay');
                if (overlay) overlay.classList.remove('active');
            },
            
            showGlobalProgress: (text, progress) => {
                const globalProgress = document.getElementById('test-global-progress');
                const progressFill = document.getElementById('test-progress-fill');
                const progressText = document.getElementById('test-progress-text');
                
                if (globalProgress) globalProgress.classList.add('active');
                if (progressFill) progressFill.style.width = `${progress || 0}%`;
                if (progressText) progressText.textContent = text || 'Loading...';
            },
            
            hideGlobalProgress: () => {
                const globalProgress = document.getElementById('test-global-progress');
                if (globalProgress) globalProgress.classList.remove('active');
            },
            
            showErrorModal: (message, details) => {
                const modal = document.getElementById('test-error-modal');
                const messageEl = document.getElementById('test-error-message');
                const detailsEl = document.getElementById('test-error-details');
                
                if (modal) modal.classList.add('active');
                if (messageEl) messageEl.textContent = message || 'An error occurred';
                if (detailsEl) {
                    if (details) {
                        detailsEl.textContent = details;
                        detailsEl.style.display = 'block';
                    } else {
                        detailsEl.style.display = 'none';
                    }
                }
            },
            
            hideErrorModal: () => {
                const modal = document.getElementById('test-error-modal');
                if (modal) modal.classList.remove('active');
            },
            
            showToast: (message, type) => {
                const toastId = type === 'warning' ? 'test-warning-toast' : 'test-success-toast';
                const messageId = type === 'warning' ? 'test-warning-message' : 'test-success-message';
                
                const toast = document.getElementById(toastId);
                const messageEl = document.getElementById(messageId);
                
                if (toast) toast.classList.add('active');
                if (messageEl) messageEl.textContent = message || 'Notification';
            },
            
            hideToast: (type) => {
                const toastId = type === 'warning' ? 'test-warning-toast' : 'test-success-toast';
                const toast = document.getElementById(toastId);
                if (toast) toast.classList.remove('active');
            }
        };
        
        // Mock error handler
        window.errorHandler = {
            showUserFriendlyError: (title, message, options) => {
                window.nlpApp.showErrorModal(title, message);
            },
            
            handleAudioError: (error, context) => {
                window.nlpApp.showErrorModal('Audio Error', error.message);
            },
            
            handleAPIError: (error, context) => {
                window.nlpApp.showErrorModal('API Error', error.message);
            },
            
            handleNetworkError: (type, error) => {
                window.nlpApp.showErrorModal('Network Error', error.message);
            }
        };
    }
    
    /**
     * Simulate viewport resize
     */
    simulateViewportResize(width, height) {
        // Create or update viewport meta tag
        let viewport = document.querySelector('meta[name="viewport"]');
        if (!viewport) {
            viewport = document.createElement('meta');
            viewport.name = 'viewport';
            document.head.appendChild(viewport);
        }
        
        viewport.content = `width=${width}, height=${height}, initial-scale=1.0`;
        
        // Trigger resize event
        const resizeEvent = new Event('resize');
        window.dispatchEvent(resizeEvent);
        
        // Update window dimensions (for testing)
        Object.defineProperty(window, 'innerWidth', {
            writable: true,
            configurable: true,
            value: width
        });
        
        Object.defineProperty(window, 'innerHeight', {
            writable: true,
            configurable: true,
            value: height
        });
        
        // Allow time for CSS media queries to apply
        return new Promise(resolve => setTimeout(resolve, 50));
    }
    
    /**
     * Property 30: Responsive design
     * For any screen size within the supported range (320px-2560px width), 
     * the interface should display appropriately
     * Validates: Requirements 7.1
     */
    async testResponsiveDesignProperty() {
        const generator = () => ({
            width: UIPropertyTester.integer(320, 2560),
            height: UIPropertyTester.integer(480, 1440),
            orientation: UIPropertyTester.choice(['portrait', 'landscape'])
        });
        
        const testFn = async (input, iteration) => {
            const { width, height, orientation } = input;
            
            // Adjust dimensions based on orientation
            const actualWidth = orientation === 'portrait' && width > height ? height : width;
            const actualHeight = orientation === 'portrait' && width > height ? width : height;
            
            // Simulate viewport resize
            await this.simulateViewportResize(actualWidth, actualHeight);
            
            // Test navigation responsiveness
            const navigation = document.getElementById('test-navigation');
            const mobileToggle = document.getElementById('test-mobile-menu-toggle');
            
            if (!navigation) {
                throw new Error('Navigation element not found');
            }
            
            // Property: Mobile menu toggle should be visible on small screens
            if (actualWidth <= 768) {
                if (mobileToggle) {
                    const toggleStyle = window.getComputedStyle(mobileToggle);
                    if (toggleStyle.display === 'none') {
                        throw new Error(`Mobile menu toggle should be visible at width ${actualWidth}px`);
                    }
                }
            } else {
                // Property: Mobile menu toggle should be hidden on large screens
                if (mobileToggle) {
                    const toggleStyle = window.getComputedStyle(mobileToggle);
                    if (toggleStyle.display !== 'none') {
                        throw new Error(`Mobile menu toggle should be hidden at width ${actualWidth}px`);
                    }
                }
            }
            
            // Property: Navigation should not overflow container
            const navRect = navigation.getBoundingClientRect();
            if (navRect.width > actualWidth + 10) { // Allow small tolerance
                throw new Error(`Navigation width ${navRect.width}px exceeds viewport width ${actualWidth}px`);
            }
            
            // Property: Content should be accessible (not cut off)
            const mainContent = this.testContainer.querySelector('.main-content');
            if (mainContent) {
                const contentRect = mainContent.getBoundingClientRect();
                if (contentRect.width > actualWidth + 10) {
                    throw new Error(`Content width ${contentRect.width}px exceeds viewport width ${actualWidth}px`);
                }
            }
            
            // Property: Text should remain readable (minimum font size)
            const textElements = this.testContainer.querySelectorAll('button, span, div');
            for (const element of textElements) {
                if (element.textContent.trim()) {
                    const style = window.getComputedStyle(element);
                    const fontSize = parseFloat(style.fontSize);
                    
                    if (fontSize < 12) { // Minimum readable font size
                        throw new Error(`Font size ${fontSize}px too small for element: ${element.textContent.slice(0, 20)}`);
                    }
                }
            }
            
            // Property: Interactive elements should be touch-friendly on small screens
            if (actualWidth <= 768) {
                const buttons = this.testContainer.querySelectorAll('button');
                for (const button of buttons) {
                    const buttonRect = button.getBoundingClientRect();
                    const minTouchSize = 44; // iOS/Android recommendation
                    
                    if (buttonRect.width > 0 && buttonRect.height > 0) {
                        if (buttonRect.width < minTouchSize || buttonRect.height < minTouchSize) {
                            throw new Error(`Button too small for touch: ${buttonRect.width}x${buttonRect.height}px (min: ${minTouchSize}px)`);
                        }
                    }
                }
            }
            
            return true;
        };
        
        return await this.tester.property(
            'Responsive design (Property 30)',
            generator,
            testFn,
            100
        );
    }
    
    /**
     * Property 31: Loading state management
     * For any active processing operation, appropriate loading indicators should be displayed
     * Validates: Requirements 7.3
     */
    async testLoadingStateManagementProperty() {
        const generator = () => ({
            operationType: UIPropertyTester.choice(['audio-recording', 'analysis-processing', 'file-upload', 'session-creation']),
            loadingTitle: UIPropertyTester.string(5, 50),
            loadingMessage: UIPropertyTester.string(10, 100),
            progress: UIPropertyTester.float(0, 100),
            duration: UIPropertyTester.integer(100, 2000),
            showGlobalProgress: UIPropertyTester.boolean(),
            showTabLoading: UIPropertyTester.boolean()
        });
        
        const testFn = async (input, iteration) => {
            const { operationType, loadingTitle, loadingMessage, progress, duration, showGlobalProgress, showTabLoading } = input;
            
            // Test loading overlay
            window.nlpApp.showLoading(operationType, loadingTitle, loadingMessage, progress);
            
            // Property: Loading overlay should be visible
            const loadingOverlay = document.getElementById('test-loading-overlay');
            if (!loadingOverlay) {
                throw new Error('Loading overlay element not found');
            }
            
            if (!loadingOverlay.classList.contains('active')) {
                throw new Error('Loading overlay should be active when loading is shown');
            }
            
            // Property: Loading title should be displayed correctly
            const titleElement = document.getElementById('test-loading-title');
            if (titleElement && titleElement.textContent !== loadingTitle) {
                throw new Error(`Loading title mismatch: expected "${loadingTitle}", got "${titleElement.textContent}"`);
            }
            
            // Property: Loading message should be displayed correctly
            const messageElement = document.getElementById('test-loading-message');
            if (messageElement && messageElement.textContent !== loadingMessage) {
                throw new Error(`Loading message mismatch: expected "${loadingMessage}", got "${messageElement.textContent}"`);
            }
            
            // Property: Progress should be displayed correctly
            const progressElement = document.getElementById('test-loading-progress-fill');
            const percentageElement = document.getElementById('test-loading-percentage');
            
            if (progressElement) {
                const expectedWidth = `${progress}%`;
                if (progressElement.style.width !== expectedWidth) {
                    throw new Error(`Progress width mismatch: expected "${expectedWidth}", got "${progressElement.style.width}"`);
                }
            }
            
            if (percentageElement) {
                const expectedPercentage = `${Math.round(progress)}%`;
                if (percentageElement.textContent !== expectedPercentage) {
                    throw new Error(`Percentage text mismatch: expected "${expectedPercentage}", got "${percentageElement.textContent}"`);
                }
            }
            
            // Test global progress if enabled
            if (showGlobalProgress) {
                const progressText = `Processing ${operationType}...`;
                window.nlpApp.showGlobalProgress(progressText, progress);
                
                const globalProgress = document.getElementById('test-global-progress');
                if (globalProgress && !globalProgress.classList.contains('active')) {
                    throw new Error('Global progress should be active when shown');
                }
                
                const globalProgressFill = document.getElementById('test-progress-fill');
                if (globalProgressFill && globalProgressFill.style.width !== `${progress}%`) {
                    throw new Error(`Global progress width mismatch: expected "${progress}%", got "${globalProgressFill.style.width}"`);
                }
            }
            
            // Simulate operation completion
            await new Promise(resolve => setTimeout(resolve, Math.min(duration, 100)));
            
            // Property: Loading should be hideable
            window.nlpApp.hideLoading(operationType);
            
            if (loadingOverlay.classList.contains('active')) {
                throw new Error('Loading overlay should not be active after hiding');
            }
            
            if (showGlobalProgress) {
                window.nlpApp.hideGlobalProgress();
                const globalProgress = document.getElementById('test-global-progress');
                if (globalProgress && globalProgress.classList.contains('active')) {
                    throw new Error('Global progress should not be active after hiding');
                }
            }
            
            // Property: Multiple loading states should be manageable
            if (iteration % 10 === 0) { // Test occasionally
                // Show multiple loading states
                window.nlpApp.showLoading('operation1', 'Op 1', 'Message 1', 25);
                window.nlpApp.showLoading('operation2', 'Op 2', 'Message 2', 75);
                
                // Both should be tracked, but UI should show the latest
                const currentTitle = document.getElementById('test-loading-title');
                if (currentTitle && currentTitle.textContent !== 'Op 2') {
                    throw new Error('Latest loading operation should be displayed');
                }
                
                // Hide one operation
                window.nlpApp.hideLoading('operation1');
                
                // Loading should still be visible (operation2 still active)
                if (!loadingOverlay.classList.contains('active')) {
                    throw new Error('Loading should remain visible when other operations are still active');
                }
                
                // Hide remaining operation
                window.nlpApp.hideLoading('operation2');
                
                // Now loading should be hidden
                if (loadingOverlay.classList.contains('active')) {
                    throw new Error('Loading should be hidden when all operations complete');
                }
            }
            
            return true;
        };
        
        return await this.tester.property(
            'Loading state management (Property 31)',
            generator,
            testFn,
            100
        );
    }
    
    /**
     * Property 33: Error handling and recovery
     * For any system error, helpful error messages and recovery options should be provided to the user
     * Validates: Requirements 7.5
     */
    async testErrorHandlingAndRecoveryProperty() {
        const generator = () => ({
            errorType: UIPropertyTester.choice(['audio', 'api', 'network', 'upload', 'generic']),
            errorMessage: UIPropertyTester.string(10, 100),
            errorDetails: UIPropertyTester.boolean() ? UIPropertyTester.string(20, 200) : null,
            hasRetryOption: UIPropertyTester.boolean(),
            toastType: UIPropertyTester.choice(['success', 'warning']),
            toastMessage: UIPropertyTester.string(5, 50),
            showTroubleshooting: UIPropertyTester.boolean()
        });
        
        const testFn = async (input, iteration) => {
            const { errorType, errorMessage, errorDetails, hasRetryOption, toastType, toastMessage, showTroubleshooting } = input;
            
            // Test error modal display
            window.nlpApp.showErrorModal(errorMessage, errorDetails);
            
            // Property: Error modal should be visible
            const errorModal = document.getElementById('test-error-modal');
            if (!errorModal) {
                throw new Error('Error modal element not found');
            }
            
            if (!errorModal.classList.contains('active')) {
                throw new Error('Error modal should be active when error is shown');
            }
            
            // Property: Error message should be displayed correctly
            const messageElement = document.getElementById('test-error-message');
            if (!messageElement) {
                throw new Error('Error message element not found');
            }
            
            if (messageElement.textContent !== errorMessage) {
                throw new Error(`Error message mismatch: expected "${errorMessage}", got "${messageElement.textContent}"`);
            }
            
            // Property: Error details should be handled correctly
            const detailsElement = document.getElementById('test-error-details');
            if (detailsElement) {
                if (errorDetails) {
                    if (detailsElement.style.display === 'none') {
                        throw new Error('Error details should be visible when provided');
                    }
                    if (detailsElement.textContent !== errorDetails) {
                        throw new Error(`Error details mismatch: expected "${errorDetails}", got "${detailsElement.textContent}"`);
                    }
                } else {
                    if (detailsElement.style.display !== 'none') {
                        throw new Error('Error details should be hidden when not provided');
                    }
                }
            }
            
            // Property: Recovery options should be available
            const retryButton = document.getElementById('test-error-retry');
            const dismissButton = document.getElementById('test-error-dismiss');
            
            if (!retryButton || !dismissButton) {
                throw new Error('Error modal should have retry and dismiss buttons');
            }
            
            // Property: Buttons should be functional
            const retryStyle = window.getComputedStyle(retryButton);
            const dismissStyle = window.getComputedStyle(dismissButton);
            
            if (retryStyle.display === 'none' && dismissStyle.display === 'none') {
                throw new Error('At least one recovery option should be visible');
            }
            
            // Test modal dismissal
            window.nlpApp.hideErrorModal();
            
            if (errorModal.classList.contains('active')) {
                throw new Error('Error modal should not be active after hiding');
            }
            
            // Test toast notifications
            window.nlpApp.showToast(toastMessage, toastType);
            
            const toastId = toastType === 'warning' ? 'test-warning-toast' : 'test-success-toast';
            const toast = document.getElementById(toastId);
            
            if (!toast) {
                throw new Error(`Toast element not found: ${toastId}`);
            }
            
            if (!toast.classList.contains('active')) {
                throw new Error(`Toast should be active when shown: ${toastType}`);
            }
            
            // Property: Toast message should be correct
            const toastMessageId = toastType === 'warning' ? 'test-warning-message' : 'test-success-message';
            const toastMessageElement = document.getElementById(toastMessageId);
            
            if (toastMessageElement && toastMessageElement.textContent !== toastMessage) {
                throw new Error(`Toast message mismatch: expected "${toastMessage}", got "${toastMessageElement.textContent}"`);
            }
            
            // Test toast dismissal
            window.nlpApp.hideToast(toastType);
            
            if (toast.classList.contains('active')) {
                throw new Error('Toast should not be active after hiding');
            }
            
            // Test troubleshooting panel (occasionally)
            if (showTroubleshooting && iteration % 20 === 0) {
                const troubleshootingPanel = document.getElementById('test-troubleshooting-panel');
                if (troubleshootingPanel) {
                    // Show troubleshooting panel
                    troubleshootingPanel.style.display = 'block';
                    
                    // Property: Status indicators should be present
                    const microphoneStatus = document.getElementById('test-microphone-status');
                    const connectionStatus = document.getElementById('test-connection-status');
                    
                    if (!microphoneStatus || !connectionStatus) {
                        throw new Error('Troubleshooting panel should have status indicators');
                    }
                    
                    // Property: Status indicators should have appropriate classes
                    const validStatusClasses = ['online', 'offline', 'warning'];
                    const micClass = Array.from(microphoneStatus.classList).find(cls => validStatusClasses.includes(cls));
                    const connClass = Array.from(connectionStatus.classList).find(cls => validStatusClasses.includes(cls));
                    
                    if (!micClass) {
                        throw new Error('Microphone status should have a valid status class');
                    }
                    
                    if (!connClass) {
                        throw new Error('Connection status should have a valid status class');
                    }
                    
                    // Hide troubleshooting panel
                    troubleshootingPanel.style.display = 'none';
                }
            }
            
            // Property: Error handling should be consistent across error types
            if (iteration % 15 === 0) { // Test occasionally
                // Test different error types
                const testErrors = [
                    { type: 'audio', message: 'Microphone access denied' },
                    { type: 'network', message: 'Connection failed' },
                    { type: 'api', message: 'Server error' }
                ];
                
                for (const testError of testErrors) {
                    if (window.errorHandler) {
                        // Test specific error handlers
                        const mockError = new Error(testError.message);
                        
                        switch (testError.type) {
                            case 'audio':
                                window.errorHandler.handleAudioError(mockError, {});
                                break;
                            case 'network':
                                window.errorHandler.handleNetworkError('general', mockError);
                                break;
                            case 'api':
                                window.errorHandler.handleAPIError(mockError, {});
                                break;
                        }
                        
                        // Property: Error should be displayed
                        if (!errorModal.classList.contains('active')) {
                            throw new Error(`Error handler for ${testError.type} should show error modal`);
                        }
                        
                        // Hide error for next test
                        window.nlpApp.hideErrorModal();
                    }
                }
            }
            
            return true;
        };
        
        return await this.tester.property(
            'Error handling and recovery (Property 33)',
            generator,
            testFn,
            100
        );
    }
    
    /**
     * Cleanup test environment
     */
    cleanup() {
        // Remove test container
        if (this.testContainer) {
            this.testContainer.remove();
        }
        
        // Restore original viewport
        if (this.originalViewport) {
            const currentViewport = document.querySelector('meta[name="viewport"]');
            if (currentViewport && currentViewport !== this.originalViewport) {
                currentViewport.remove();
                document.head.appendChild(this.originalViewport);
            }
        }
        
        // Reset window dimensions
        delete window.innerWidth;
        delete window.innerHeight;
        
        console.log('🧹 Test environment cleaned up');
    }
    
    /**
     * Run all UI functionality property tests
     */
    async runAllTests() {
        console.log('🚀 Starting UI Functionality Property Tests');
        console.log('=============================================');
        
        const startTime = Date.now();
        
        try {
            await this.setup();
            
            const results = [];
            
            // Run Property 30: Responsive design
            results.push(await this.testResponsiveDesignProperty());
            
            // Run Property 31: Loading state management
            results.push(await this.testLoadingStateManagementProperty());
            
            // Run Property 33: Error handling and recovery
            results.push(await this.testErrorHandlingAndRecoveryProperty());
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            
            // Calculate overall results
            const summary = this.tester.getSummary();
            const allPassed = results.every(result => result === true);
            
            const testSummary = {
                allPassed,
                totalTests: summary.total,
                totalPassed: summary.passed,
                totalFailed: summary.failed,
                duration,
                results,
                failures: summary.failures,
                timestamp: new Date().toISOString()
            };
            
            // Log summary
            console.log('\n📊 UI FUNCTIONALITY PROPERTY TESTS SUMMARY');
            console.log('===========================================');
            console.log(`Overall Result: ${allPassed ? '✅ PASSED' : '❌ FAILED'}`);
            console.log(`Total Tests: ${summary.passed}/${summary.total} passed`);
            console.log(`Duration: ${duration}ms`);
            console.log('\nProperty Results:');
            
            const propertyNames = [
                'Responsive design (Property 30)',
                'Loading state management (Property 31)',
                'Error handling and recovery (Property 33)'
            ];
            
            results.forEach((result, index) => {
                const status = result ? '✅' : '❌';
                console.log(`${status} ${propertyNames[index]}: ${result ? 'PASSED' : 'FAILED'}`);
            });
            
            if (summary.failures.length > 0) {
                console.log('\n❌ First few failures:');
                summary.failures.slice(0, 3).forEach((failure, index) => {
                    console.log(`${index + 1}. ${failure.property} (iteration ${failure.iteration})`);
                    console.log(`   Error: ${failure.error}`);
                });
            }
            
            return testSummary;
            
        } finally {
            this.cleanup();
        }
    }
    
    /**
     * Get test results
     */
    getResults() {
        return this.tester.getSummary();
    }
}

// Auto-run tests if this script is loaded directly
if (typeof window !== 'undefined' && window.location.pathname.includes('test-runner.html')) {
    document.addEventListener('DOMContentLoaded', async () => {
        console.log('🧪 Auto-running UI functionality property tests...');
        
        // Wait for main app to load
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const tester = new UIFunctionalityPropertyTests();
        const results = await tester.runAllTests();
        
        // Display results in the page
        const resultsContainer = document.getElementById('test-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <h2>UI Functionality Property Tests Results</h2>
                <div class="test-summary ${results.allPassed ? 'passed' : 'failed'}">
                    <h3>${results.allPassed ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED'}</h3>
                    <p>Total: ${results.totalPassed}/${results.totalTests} tests passed</p>
                    <p>Duration: ${results.duration}ms</p>
                    <p>Timestamp: ${results.timestamp}</p>
                </div>
                <div class="test-details">
                    <h4>Property Results:</h4>
                    <ul>
                        <li>${results.results[0] ? '✅' : '❌'} Property 30: Responsive design</li>
                        <li>${results.results[1] ? '✅' : '❌'} Property 31: Loading state management</li>
                        <li>${results.results[2] ? '✅' : '❌'} Property 33: Error handling and recovery</li>
                    </ul>
                    ${results.failures.length > 0 ? 
                        `<details>
                            <summary>Failures (${results.failures.length})</summary>
                            <pre>${JSON.stringify(results.failures.slice(0, 5), null, 2)}</pre>
                        </details>` : ''
                    }
                </div>
            `;
        }
    });
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIFunctionalityPropertyTests;
}

// Make available globally
window.UIFunctionalityPropertyTests = UIFunctionalityPropertyTests;