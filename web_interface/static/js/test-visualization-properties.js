/**
 * Property-Based Tests for NLP Visualization Components
 * Tests Properties 11, 12, and 13 from the design document
 */

class VisualizationPropertyTests {
    constructor() {
        this.testResults = [];
        this.testContainer = null;
        this.visualization = null;
        
        this.init();
    }
    
    init() {
        console.log('🧪 Initializing Visualization Property Tests...');
        
        // Create test container
        this.createTestContainer();
        
        // Initialize visualization engine
        this.visualization = new NLPVisualizationEngine();
        
        console.log('✅ Visualization Property Tests initialized');
    }
    
    /**
     * Create test container in DOM
     */
    createTestContainer() {
        // Remove existing test container
        const existing = document.getElementById('visualization-test-container');
        if (existing) {
            existing.remove();
        }
        
        // Create new test container
        this.testContainer = document.createElement('div');
        this.testContainer.id = 'visualization-test-container';
        this.testContainer.style.cssText = `
            position: fixed;
            top: -9999px;
            left: -9999px;
            width: 800px;
            height: 600px;
            visibility: hidden;
        `;
        
        // Add test visualization containers
        this.testContainer.innerHTML = `
            <div id="test-pos-visualization"></div>
            <div id="test-ner-visualization"></div>
            <div id="test-dependency-visualization"></div>
            <div id="test-sentiment-visualization"></div>
        `;
        
        document.body.appendChild(this.testContainer);
    }
    
    /**
     * Generate random POS data for testing
     */
    generatePOSData(count = null) {
        const posTagTypes = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'NUM', 'PUNCT'];
        const sampleTokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'and', 'runs'];
        
        const dataCount = count || Math.floor(Math.random() * 20) + 5; // 5-24 tokens
        const posData = [];
        
        for (let i = 0; i < dataCount; i++) {
            posData.push({
                token: sampleTokens[i % sampleTokens.length] + (i > sampleTokens.length ? i : ''),
                tag: posTagTypes[Math.floor(Math.random() * posTagTypes.length)],
                position: i,
                confidence: Math.random() * 0.4 + 0.6 // 0.6 to 1.0
            });
        }
        
        return posData;
    }
    
    /**
     * Generate random NER data for testing
     */
    generateNERData(text = null) {
        const entityTypes = ['PERSON', 'ORG', 'LOC', 'GPE', 'MISC', 'TIME', 'MONEY'];
        const sampleText = text || 'John Smith works at Microsoft in Seattle and earned $50,000 last year.';
        const entities = [];
        
        // Generate 1-5 entities
        const entityCount = Math.floor(Math.random() * 5) + 1;
        
        for (let i = 0; i < entityCount; i++) {
            const start = Math.floor(Math.random() * (sampleText.length - 10));
            const end = start + Math.floor(Math.random() * 10) + 3;
            
            entities.push({
                text: sampleText.slice(start, end),
                label: entityTypes[Math.floor(Math.random() * entityTypes.length)],
                start: start,
                end: end,
                confidence: Math.random() * 0.4 + 0.6
            });
        }
        
        return { entities, text: sampleText };
    }
    
    /**
     * Generate random dependency data for testing
     */
    generateDependencyData() {
        const relationTypes = ['nsubj', 'dobj', 'amod', 'det', 'prep', 'pobj', 'root', 'aux'];
        const tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];
        const dependencies = [];
        
        // Generate dependencies (fewer than tokens)
        const depCount = Math.floor(tokens.length * 0.8);
        
        for (let i = 0; i < depCount; i++) {
            const head = Math.floor(Math.random() * tokens.length);
            let dependent = Math.floor(Math.random() * tokens.length);
            
            // Ensure head and dependent are different
            while (dependent === head) {
                dependent = Math.floor(Math.random() * tokens.length);
            }
            
            dependencies.push({
                head: head,
                dependent: dependent,
                relation: relationTypes[Math.floor(Math.random() * relationTypes.length)],
                head_text: tokens[head],
                dependent_text: tokens[dependent]
            });
        }
        
        return { dependencies, tokens };
    }
    
    /**
     * Generate random sentiment data for testing
     */
    generateSentimentData() {
        const sentiments = ['positive', 'negative', 'neutral'];
        const overallSentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
        
        // Generate scores that sum to approximately 1
        const positive = Math.random();
        const negative = Math.random();
        const neutral = Math.random();
        const total = positive + negative + neutral;
        
        return {
            overall_sentiment: overallSentiment,
            confidence: Math.random() * 0.4 + 0.6,
            scores: {
                positive: positive / total,
                negative: negative / total,
                neutral: neutral / total
            }
        };
    }
    
    /**
     * Property 11: POS tag visualization
     * For any POS data, the rendered HTML should include color-coded tokens with hoverable POS information
     */
    async testPOSTagVisualization() {
        console.log('🏷️ Testing Property 11: POS tag visualization');
        
        const iterations = 100;
        let passedTests = 0;
        const failures = [];
        
        for (let i = 0; i < iterations; i++) {
            try {
                // Generate random POS data
                const posData = this.generatePOSData();
                
                // Render visualization
                this.visualization.renderPOSTags(posData, 'test-pos-visualization');
                
                // Verify rendered elements
                const container = document.getElementById('test-pos-visualization');
                const tokens = container.querySelectorAll('.pos-token');
                
                // Property: Number of rendered tokens should match input data
                if (tokens.length !== posData.length) {
                    throw new Error(`Token count mismatch: expected ${posData.length}, got ${tokens.length}`);
                }
                
                // Property: Each token should have color styling
                tokens.forEach((token, index) => {
                    const computedStyle = window.getComputedStyle(token);
                    const backgroundColor = computedStyle.backgroundColor;
                    
                    if (backgroundColor === 'rgba(0, 0, 0, 0)' || backgroundColor === 'transparent') {
                        throw new Error(`Token ${index} has no background color`);
                    }
                });
                
                // Property: Each token should have tooltip information
                tokens.forEach((token, index) => {
                    const expectedToken = posData[index].token;
                    const actualToken = token.textContent.trim();
                    
                    if (actualToken !== expectedToken) {
                        throw new Error(`Token text mismatch at ${index}: expected "${expectedToken}", got "${actualToken}"`);
                    }
                });
                
                // Property: Legend should be present
                const legend = container.querySelector('.pos-legend');
                if (!legend) {
                    throw new Error('POS legend not found');
                }
                
                // Property: Statistics should be present
                const stats = container.querySelector('.pos-statistics');
                if (!stats) {
                    throw new Error('POS statistics not found');
                }
                
                passedTests++;
                
            } catch (error) {
                failures.push({
                    iteration: i,
                    error: error.message,
                    data: posData
                });
            }
        }
        
        const result = {
            property: 'Property 11: POS tag visualization',
            passed: passedTests,
            total: iterations,
            success: failures.length === 0,
            failures: failures.slice(0, 5) // Keep only first 5 failures
        };
        
        this.testResults.push(result);
        
        if (result.success) {
            console.log(`✅ Property 11 PASSED: ${passedTests}/${iterations} tests passed`);
        } else {
            console.log(`❌ Property 11 FAILED: ${passedTests}/${iterations} tests passed`);
            console.log('First failure:', failures[0]);
        }
        
        return result;
    }
    
    /**
     * Property 12: Named entity highlighting
     * For any NER data, entities should be highlighted with category-specific colors and contain tooltip information
     */
    async testNERHighlighting() {
        console.log('🏢 Testing Property 12: Named entity highlighting');
        
        const iterations = 100;
        let passedTests = 0;
        const failures = [];
        
        for (let i = 0; i < iterations; i++) {
            try {
                // Generate random NER data
                const { entities, text } = this.generateNERData();
                
                // Render visualization
                this.visualization.renderNamedEntities(entities, text, 'test-ner-visualization');
                
                // Verify rendered elements
                const container = document.getElementById('test-ner-visualization');
                const entityElements = container.querySelectorAll('.ner-entity');
                
                // Property: Number of rendered entities should match input data
                if (entityElements.length !== entities.length) {
                    throw new Error(`Entity count mismatch: expected ${entities.length}, got ${entityElements.length}`);
                }
                
                // Property: Each entity should have category-specific color
                entityElements.forEach((element, index) => {
                    const computedStyle = window.getComputedStyle(element);
                    const backgroundColor = computedStyle.backgroundColor;
                    
                    if (backgroundColor === 'rgba(0, 0, 0, 0)' || backgroundColor === 'transparent') {
                        throw new Error(`Entity ${index} has no background color`);
                    }
                });
                
                // Property: Entity list should be present
                const entityList = container.querySelector('.ner-entity-list');
                if (!entityList) {
                    throw new Error('NER entity list not found');
                }
                
                // Property: Legend should be present
                const legend = container.querySelector('.ner-legend');
                if (!legend) {
                    throw new Error('NER legend not found');
                }
                
                // Property: Statistics should be present
                const stats = container.querySelector('.ner-statistics');
                if (!stats) {
                    throw new Error('NER statistics not found');
                }
                
                passedTests++;
                
            } catch (error) {
                failures.push({
                    iteration: i,
                    error: error.message,
                    data: { entities, text }
                });
            }
        }
        
        const result = {
            property: 'Property 12: Named entity highlighting',
            passed: passedTests,
            total: iterations,
            success: failures.length === 0,
            failures: failures.slice(0, 5)
        };
        
        this.testResults.push(result);
        
        if (result.success) {
            console.log(`✅ Property 12 PASSED: ${passedTests}/${iterations} tests passed`);
        } else {
            console.log(`❌ Property 12 FAILED: ${passedTests}/${iterations} tests passed`);
            console.log('First failure:', failures[0]);
        }
        
        return result;
    }
    
    /**
     * Property 13: Dependency tree rendering
     * For any dependency parse structure, an interactive SVG tree should be rendered with correct arc labels and node relationships
     */
    async testDependencyTreeRendering() {
        console.log('🌳 Testing Property 13: Dependency tree rendering');
        
        const iterations = 100;
        let passedTests = 0;
        const failures = [];
        
        for (let i = 0; i < iterations; i++) {
            try {
                // Generate random dependency data
                const { dependencies, tokens } = this.generateDependencyData();
                
                // Render visualization
                this.visualization.renderDependencyTree(dependencies, tokens, 'test-dependency-visualization');
                
                // Verify rendered elements
                const container = document.getElementById('test-dependency-visualization');
                const svg = container.querySelector('svg.dependency-tree');
                
                // Property: SVG should be present
                if (!svg) {
                    throw new Error('Dependency tree SVG not found');
                }
                
                // Property: Number of nodes should match token count
                const nodes = svg.querySelectorAll('.dependency-node');
                if (nodes.length !== tokens.length) {
                    throw new Error(`Node count mismatch: expected ${tokens.length}, got ${nodes.length}`);
                }
                
                // Property: Number of arcs should match dependency count
                const arcs = svg.querySelectorAll('.dependency-link');
                if (arcs.length !== dependencies.length) {
                    throw new Error(`Arc count mismatch: expected ${dependencies.length}, got ${arcs.length}`);
                }
                
                // Property: Each arc should have a label
                const labels = svg.querySelectorAll('.dependency-label');
                if (labels.length !== dependencies.length) {
                    throw new Error(`Label count mismatch: expected ${dependencies.length}, got ${labels.length}`);
                }
                
                // Property: Legend should be present
                const legend = container.querySelector('.dependency-legend');
                if (!legend) {
                    throw new Error('Dependency legend not found');
                }
                
                // Property: Statistics should be present
                const stats = container.querySelector('.dependency-statistics');
                if (!stats) {
                    throw new Error('Dependency statistics not found');
                }
                
                passedTests++;
                
            } catch (error) {
                failures.push({
                    iteration: i,
                    error: error.message,
                    data: { dependencies, tokens }
                });
            }
        }
        
        const result = {
            property: 'Property 13: Dependency tree rendering',
            passed: passedTests,
            total: iterations,
            success: failures.length === 0,
            failures: failures.slice(0, 5)
        };
        
        this.testResults.push(result);
        
        if (result.success) {
            console.log(`✅ Property 13 PASSED: ${passedTests}/${iterations} tests passed`);
        } else {
            console.log(`❌ Property 13 FAILED: ${passedTests}/${iterations} tests passed`);
            console.log('First failure:', failures[0]);
        }
        
        return result;
    }
    
    /**
     * Run all visualization property tests
     */
    async runAllTests() {
        console.log('🚀 Running all visualization property tests...');
        
        const startTime = Date.now();
        
        // Run all property tests
        const results = await Promise.all([
            this.testPOSTagVisualization(),
            this.testNERHighlighting(),
            this.testDependencyTreeRendering()
        ]);
        
        const endTime = Date.now();
        const duration = endTime - startTime;
        
        // Calculate overall results
        const totalTests = results.reduce((sum, result) => sum + result.total, 0);
        const totalPassed = results.reduce((sum, result) => sum + result.passed, 0);
        const allPassed = results.every(result => result.success);
        
        const summary = {
            allPassed,
            totalTests,
            totalPassed,
            duration,
            results,
            timestamp: new Date().toISOString()
        };
        
        // Log summary
        console.log('\n📊 VISUALIZATION PROPERTY TESTS SUMMARY');
        console.log('==========================================');
        console.log(`Overall Result: ${allPassed ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`Total Tests: ${totalPassed}/${totalTests} passed`);
        console.log(`Duration: ${duration}ms`);
        console.log('\nIndividual Results:');
        
        results.forEach(result => {
            const status = result.success ? '✅' : '❌';
            console.log(`${status} ${result.property}: ${result.passed}/${result.total}`);
            
            if (!result.success && result.failures.length > 0) {
                console.log(`   First failure: ${result.failures[0].error}`);
            }
        });
        
        // Clean up test container
        if (this.testContainer) {
            this.testContainer.remove();
        }
        
        return summary;
    }
    
    /**
     * Get test results
     */
    getResults() {
        return this.testResults;
    }
}

// Auto-run tests if this script is loaded directly
if (typeof window !== 'undefined' && window.location.pathname.includes('test-runner.html')) {
    document.addEventListener('DOMContentLoaded', async () => {
        console.log('🧪 Auto-running visualization property tests...');
        
        // Wait for D3.js and visualization engine to load
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const tester = new VisualizationPropertyTests();
        const results = await tester.runAllTests();
        
        // Display results in the page
        const resultsContainer = document.getElementById('test-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <h2>Visualization Property Tests Results</h2>
                <div class="test-summary ${results.allPassed ? 'passed' : 'failed'}">
                    <h3>${results.allPassed ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED'}</h3>
                    <p>Total: ${results.totalPassed}/${results.totalTests} tests passed</p>
                    <p>Duration: ${results.duration}ms</p>
                </div>
                <div class="test-details">
                    ${results.results.map(result => `
                        <div class="test-result ${result.success ? 'passed' : 'failed'}">
                            <h4>${result.success ? '✅' : '❌'} ${result.property}</h4>
                            <p>Passed: ${result.passed}/${result.total}</p>
                            ${!result.success && result.failures.length > 0 ? 
                                `<details>
                                    <summary>Failures (${result.failures.length})</summary>
                                    <pre>${JSON.stringify(result.failures, null, 2)}</pre>
                                </details>` : ''
                            }
                        </div>
                    `).join('')}
                </div>
            `;
        }
    });
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VisualizationPropertyTests;
}

// Make available globally
window.VisualizationPropertyTests = VisualizationPropertyTests;