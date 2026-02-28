/**
 * NLP Visualization Engine using D3.js
 * Provides interactive visualizations for POS tags, NER, dependencies, and sentiment
 */

class NLPVisualizationEngine {
    constructor() {
        // Check if D3.js is available
        this.d3Available = typeof d3 !== 'undefined';
        
        if (!this.d3Available) {
            console.warn('⚠️ D3.js not available - falling back to simple visualizations');
        }
        
        this.posColors = {
            'NOUN': '#2E86AB',      // Blue
            'VERB': '#A23B72',      // Purple
            'ADJ': '#F18F01',       // Orange
            'ADV': '#C73E1D',       // Red
            'PRON': '#6A994E',      // Green
            'DET': '#577590',       // Blue-gray
            'ADP': '#F2CC8F',       // Light orange
            'CONJ': '#81B29A',      // Light green
            'NUM': '#E07A5F',       // Coral
            'PUNCT': '#3D5A80',     // Dark blue
            'PROPN': '#264653',     // Dark green
            'PART': '#E9C46A',      // Yellow
            'INTJ': '#F4A261',      // Light orange
            'SYM': '#E76F51',       // Red-orange
            'X': '#6C757D'          // Gray for unknown
        };
        
        this.nerColors = {
            'PERSON': '#FF6B6B',    // Red
            'ORG': '#4ECDC4',       // Teal
            'LOC': '#45B7D1',       // Blue
            'GPE': '#96CEB4',       // Light green
            'MISC': '#FFEAA7',      // Yellow
            'TIME': '#DDA0DD',      // Plum
            'DATE': '#98D8C8',      // Mint
            'MONEY': '#F7DC6F',     // Gold
            'PERCENT': '#BB8FCE',   // Light purple
            'FACILITY': '#85C1E9',  // Light blue
            'PRODUCT': '#F8C471',   // Light orange
            'EVENT': '#82E0AA',     // Light green
            'WORK_OF_ART': '#F1948A', // Light red
            'LAW': '#AED6F1',       // Very light blue
            'LANGUAGE': '#D7BDE2'   // Very light purple
        };
        
        this.dependencyColors = {
            'nsubj': '#FF6B6B',     // Subject - Red
            'dobj': '#4ECDC4',      // Direct object - Teal
            'iobj': '#45B7D1',      // Indirect object - Blue
            'nmod': '#96CEB4',      // Nominal modifier - Light green
            'amod': '#FFEAA7',      // Adjectival modifier - Yellow
            'advmod': '#DDA0DD',    // Adverbial modifier - Plum
            'det': '#98D8C8',       // Determiner - Mint
            'prep': '#F7DC6F',      // Preposition - Gold
            'pobj': '#BB8FCE',      // Object of preposition - Light purple
            'conj': '#85C1E9',      // Conjunct - Light blue
            'cc': '#F8C471',        // Coordinating conjunction - Light orange
            'aux': '#82E0AA',       // Auxiliary - Light green
            'cop': '#F1948A',       // Copula - Light red
            'root': '#2E86AB'       // Root - Dark blue
        };
        
        this.init();
    }
    
    init() {
        console.log('🎨 Initializing NLP Visualization Engine...');
        this.setupTooltip();
        console.log('✅ NLP Visualization Engine initialized');
    }
    
    /**
     * Set up shared tooltip for all visualizations
     */
    setupTooltip() {
        if (!this.d3Available) {
            console.log('📊 D3.js not available - skipping tooltip setup');
            return;
        }
        
        // Remove existing tooltip if any
        d3.select('#nlp-tooltip').remove();
        
        this.tooltip = d3.select('body')
            .append('div')
            .attr('id', 'nlp-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.9)')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '5px')
            .style('font-size', '12px')
            .style('font-family', 'Arial, sans-serif')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('z-index', 1000)
            .style('max-width', '300px')
            .style('box-shadow', '0 2px 10px rgba(0,0,0,0.3)');
    }
    
    /**
     * Show tooltip with content
     */
    showTooltip(content, event) {
        this.tooltip
            .style('opacity', 1)
            .html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }
    
    /**
     * Hide tooltip
     */
    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }
    
    /**
     * Render POS tags visualization with D3.js
     * @param {Array} posData - Array of {token, tag, position, confidence} objects
     * @param {string} containerId - ID of container element
     */
    renderPOSTags(posData, containerId = 'pos-visualization') {
        console.log('🏷️ Rendering POS tags visualization:', posData.length, 'tokens');
        
        if (!this.d3Available) {
            console.log('📊 D3.js not available - using fallback visualization');
            this.fallbackPOSVisualization(posData, containerId);
            return;
        }
        
        const container = d3.select(`#${containerId}`);
        if (container.empty()) {
            console.error('❌ POS visualization container not found:', containerId);
            return;
        }
        
        // Clear previous content
        container.selectAll('*').remove();
        
        // Create main container
        const mainDiv = container
            .append('div')
            .attr('class', 'pos-visualization-container')
            .style('padding', '20px')
            .style('font-family', 'Arial, sans-serif');
        
        // Add title
        mainDiv
            .append('h4')
            .text('Part-of-Speech Tags')
            .style('margin-bottom', '15px')
            .style('color', '#333');
        
        // Create tokens container
        const tokensContainer = mainDiv
            .append('div')
            .attr('class', 'pos-tokens')
            .style('line-height', '2.5')
            .style('font-size', '16px');
        
        // Create token elements
        const tokens = tokensContainer
            .selectAll('.pos-token')
            .data(posData)
            .enter()
            .append('span')
            .attr('class', 'pos-token')
            .style('display', 'inline-block')
            .style('margin', '2px 4px')
            .style('padding', '6px 12px')
            .style('border-radius', '20px')
            .style('cursor', 'pointer')
            .style('transition', 'all 0.3s ease')
            .style('font-weight', '500')
            .style('border', '2px solid transparent')
            .style('background-color', d => this.posColors[d.tag] || this.posColors['X'])
            .style('color', 'white')
            .text(d => d.token);
        
        // Add hover effects and tooltips
        tokens
            .on('mouseover', (event, d) => {
                // Highlight token
                d3.select(event.target)
                    .style('transform', 'scale(1.1)')
                    .style('border-color', '#333')
                    .style('box-shadow', '0 4px 15px rgba(0,0,0,0.3)');
                
                // Show tooltip
                const tooltipContent = `
                    <strong>Token:</strong> ${d.token}<br>
                    <strong>POS Tag:</strong> ${d.tag}<br>
                    <strong>Confidence:</strong> ${(d.confidence * 100).toFixed(1)}%<br>
                    <strong>Position:</strong> ${d.position}
                `;
                this.showTooltip(tooltipContent, event);
            })
            .on('mouseout', (event) => {
                // Remove highlight
                d3.select(event.target)
                    .style('transform', 'scale(1)')
                    .style('border-color', 'transparent')
                    .style('box-shadow', 'none');
                
                // Hide tooltip
                this.hideTooltip();
            })
            .on('click', (event, d) => {
                // Emit custom event for token click
                const customEvent = new CustomEvent('posTokenClick', {
                    detail: { token: d.token, tag: d.tag, position: d.position, confidence: d.confidence }
                });
                document.dispatchEvent(customEvent);
                
                console.log('🖱️ POS token clicked:', d);
            });
        
        // Add legend
        this.addPOSLegend(mainDiv, posData);
        
        // Add statistics
        this.addPOSStatistics(mainDiv, posData);
        
        console.log('✅ POS tags visualization rendered successfully');
    }
    
    /**
     * Add POS tags legend
     */
    addPOSLegend(container, posData) {
        // Get unique tags from data
        const uniqueTags = [...new Set(posData.map(d => d.tag))];
        
        const legendContainer = container
            .append('div')
            .attr('class', 'pos-legend')
            .style('margin-top', '20px')
            .style('padding', '15px')
            .style('background', '#f8f9fa')
            .style('border-radius', '8px');
        
        legendContainer
            .append('h5')
            .text('POS Tags Legend')
            .style('margin-bottom', '10px')
            .style('color', '#495057');
        
        const legendItems = legendContainer
            .append('div')
            .style('display', 'flex')
            .style('flex-wrap', 'wrap')
            .style('gap', '8px');
        
        uniqueTags.forEach(tag => {
            const count = posData.filter(d => d.tag === tag).length;
            
            const legendItem = legendItems
                .append('div')
                .style('display', 'flex')
                .style('align-items', 'center')
                .style('gap', '5px')
                .style('padding', '4px 8px')
                .style('background', 'white')
                .style('border-radius', '15px')
                .style('font-size', '12px')
                .style('border', '1px solid #dee2e6');
            
            legendItem
                .append('div')
                .style('width', '12px')
                .style('height', '12px')
                .style('border-radius', '50%')
                .style('background-color', this.posColors[tag] || this.posColors['X']);
            
            legendItem
                .append('span')
                .text(`${tag} (${count})`);
        });
    }
    
    /**
     * Add POS statistics
     */
    addPOSStatistics(container, posData) {
        const statsContainer = container
            .append('div')
            .attr('class', 'pos-statistics')
            .style('margin-top', '15px')
            .style('padding', '15px')
            .style('background', '#e3f2fd')
            .style('border-radius', '8px');
        
        statsContainer
            .append('h5')
            .text('Statistics')
            .style('margin-bottom', '10px')
            .style('color', '#1565c0');
        
        const stats = statsContainer
            .append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fit, minmax(150px, 1fr))')
            .style('gap', '10px')
            .style('font-size', '14px');
        
        // Total tokens
        stats
            .append('div')
            .html(`<strong>Total Tokens:</strong> ${posData.length}`);
        
        // Unique POS tags
        const uniqueTagsCount = new Set(posData.map(d => d.tag)).size;
        stats
            .append('div')
            .html(`<strong>Unique POS Tags:</strong> ${uniqueTagsCount}`);
        
        // Average confidence
        const avgConfidence = posData.reduce((sum, d) => sum + d.confidence, 0) / posData.length;
        stats
            .append('div')
            .html(`<strong>Avg Confidence:</strong> ${(avgConfidence * 100).toFixed(1)}%`);
        
        // Most common tag
        const tagCounts = {};
        posData.forEach(d => {
            tagCounts[d.tag] = (tagCounts[d.tag] || 0) + 1;
        });
        const mostCommonTag = Object.keys(tagCounts).reduce((a, b) => 
            tagCounts[a] > tagCounts[b] ? a : b
        );
        stats
            .append('div')
            .html(`<strong>Most Common:</strong> ${mostCommonTag} (${tagCounts[mostCommonTag]})`);
    }
    
    /**
     * Fallback POS visualization when D3.js is not available
     */
    fallbackPOSVisualization(posData, containerId = 'pos-visualization') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('❌ POS visualization container not found:', containerId);
            return;
        }
        
        // Clear previous content
        container.innerHTML = '';
        
        // Create main container
        const mainDiv = document.createElement('div');
        mainDiv.className = 'pos-visualization-container';
        mainDiv.style.cssText = `
            padding: 20px;
            font-family: Arial, sans-serif;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        `;
        
        // Add title
        const title = document.createElement('h4');
        title.textContent = 'Part-of-Speech Tags';
        title.style.cssText = `
            margin-bottom: 15px;
            color: #333;
        `;
        mainDiv.appendChild(title);
        
        // Create tokens container
        const tokensContainer = document.createElement('div');
        tokensContainer.className = 'pos-tokens';
        tokensContainer.style.cssText = `
            line-height: 2.5;
            font-size: 16px;
            margin-bottom: 20px;
        `;
        
        // Create token elements
        posData.forEach(d => {
            const token = document.createElement('span');
            token.className = 'pos-token';
            token.style.cssText = `
                display: inline-block;
                margin: 2px 4px;
                padding: 6px 12px;
                border-radius: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 500;
                border: 2px solid transparent;
                background-color: ${this.posColors[d.tag] || this.posColors['X']};
                color: white;
            `;
            token.textContent = d.token;
            token.title = `${d.token} (${d.tag}) - ${(d.confidence * 100).toFixed(1)}%`;
            
            // Add hover effects
            token.addEventListener('mouseenter', () => {
                token.style.transform = 'scale(1.1)';
                token.style.borderColor = '#333';
                token.style.boxShadow = '0 4px 15px rgba(0,0,0,0.3)';
            });
            
            token.addEventListener('mouseleave', () => {
                token.style.transform = 'scale(1)';
                token.style.borderColor = 'transparent';
                token.style.boxShadow = 'none';
            });
            
            tokensContainer.appendChild(token);
        });
        
        mainDiv.appendChild(tokensContainer);
        
        // Add simple legend
        this.addSimplePOSLegend(mainDiv, posData);
        
        // Add simple statistics
        this.addSimplePOSStatistics(mainDiv, posData);
        
        container.appendChild(mainDiv);
        
        console.log('✅ Fallback POS tags visualization rendered successfully');
    }
    
    /**
     * Add simple POS legend
     */
    addSimplePOSLegend(container, posData) {
        const uniqueTags = [...new Set(posData.map(d => d.tag))];
        
        const legendContainer = document.createElement('div');
        legendContainer.className = 'pos-legend';
        legendContainer.style.cssText = `
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        `;
        
        const legendTitle = document.createElement('h5');
        legendTitle.textContent = 'POS Tags Legend';
        legendTitle.style.cssText = `
            margin-bottom: 10px;
            color: #495057;
        `;
        legendContainer.appendChild(legendTitle);
        
        const legendItems = document.createElement('div');
        legendItems.style.cssText = `
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        `;
        
        uniqueTags.forEach(tag => {
            const count = posData.filter(d => d.tag === tag).length;
            
            const legendItem = document.createElement('div');
            legendItem.style.cssText = `
                display: flex;
                align-items: center;
                gap: 5px;
                padding: 4px 8px;
                background: white;
                border-radius: 15px;
                font-size: 12px;
                border: 1px solid #dee2e6;
            `;
            
            const colorDot = document.createElement('div');
            colorDot.style.cssText = `
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: ${this.posColors[tag] || this.posColors['X']};
            `;
            
            const labelSpan = document.createElement('span');
            labelSpan.textContent = `${tag} (${count})`;
            
            legendItem.appendChild(colorDot);
            legendItem.appendChild(labelSpan);
            legendItems.appendChild(legendItem);
        });
        
        legendContainer.appendChild(legendItems);
        container.appendChild(legendContainer);
    }
    
    /**
     * Add simple POS statistics
     */
    addSimplePOSStatistics(container, posData) {
        const statsContainer = document.createElement('div');
        statsContainer.className = 'pos-statistics';
        statsContainer.style.cssText = `
            margin-top: 15px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 8px;
            border: 1px solid #bbdefb;
        `;
        
        const statsTitle = document.createElement('h5');
        statsTitle.textContent = 'Statistics';
        statsTitle.style.cssText = `
            margin-bottom: 10px;
            color: #1565c0;
        `;
        statsContainer.appendChild(statsTitle);
        
        const stats = document.createElement('div');
        stats.style.cssText = `
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            font-size: 14px;
        `;
        
        // Total tokens
        const totalStat = document.createElement('div');
        totalStat.innerHTML = `<strong>Total Tokens:</strong> ${posData.length}`;
        stats.appendChild(totalStat);
        
        // Unique POS tags
        const uniqueTagsCount = new Set(posData.map(d => d.tag)).size;
        const uniqueStat = document.createElement('div');
        uniqueStat.innerHTML = `<strong>Unique POS Tags:</strong> ${uniqueTagsCount}`;
        stats.appendChild(uniqueStat);
        
        // Average confidence
        const avgConfidence = posData.reduce((sum, d) => sum + d.confidence, 0) / posData.length;
        const confidenceStat = document.createElement('div');
        confidenceStat.innerHTML = `<strong>Avg Confidence:</strong> ${(avgConfidence * 100).toFixed(1)}%`;
        stats.appendChild(confidenceStat);
        
        // Most common tag
        const tagCounts = {};
        posData.forEach(d => {
            tagCounts[d.tag] = (tagCounts[d.tag] || 0) + 1;
        });
        const mostCommonTag = Object.keys(tagCounts).reduce((a, b) => 
            tagCounts[a] > tagCounts[b] ? a : b
        );
        const commonStat = document.createElement('div');
        commonStat.innerHTML = `<strong>Most Common:</strong> ${mostCommonTag} (${tagCounts[mostCommonTag]})`;
        stats.appendChild(commonStat);
        
        statsContainer.appendChild(stats);
        container.appendChild(statsContainer);
    }
    getPOSColor(tag) {
        return this.posColors[tag] || this.posColors['X'];
    }
    
    /**
     * Clear POS visualization
     */
    clearPOSVisualization(containerId = 'pos-visualization') {
        const container = d3.select(`#${containerId}`);
        container.selectAll('*').remove();
    }
    
    /**
     * Render Named Entity Recognition visualization with D3.js
     * @param {Array} nerData - Array of {text, label, start, end, confidence} objects
     * @param {string} originalText - Original text to highlight entities in
     * @param {string} containerId - ID of container element
     */
    renderNamedEntities(nerData, originalText, containerId = 'ner-visualization') {
        console.log('🏢 Rendering NER visualization:', nerData.length, 'entities');
        
        const container = d3.select(`#${containerId}`);
        if (container.empty()) {
            console.error('❌ NER visualization container not found:', containerId);
            return;
        }
        
        // Clear previous content
        container.selectAll('*').remove();
        
        // Create main container
        const mainDiv = container
            .append('div')
            .attr('class', 'ner-visualization-container')
            .style('font-family', 'Arial, sans-serif');
        
        // Add title
        mainDiv
            .append('h4')
            .text('Named Entity Recognition')
            .style('margin-bottom', '15px')
            .style('color', '#333');
        
        // Create highlighted text
        this.createHighlightedText(mainDiv, nerData, originalText);
        
        // Add entity list
        this.addNEREntityList(mainDiv, nerData);
        
        // Add NER legend
        this.addNERLegend(mainDiv, nerData);
        
        // Add NER statistics
        this.addNERStatistics(mainDiv, nerData);
        
        console.log('✅ NER visualization rendered successfully');
    }
    
    /**
     * Create highlighted text with entity annotations
     */
    createHighlightedText(container, nerData, originalText) {
        const textContainer = container
            .append('div')
            .attr('class', 'ner-text-container')
            .style('margin-bottom', '20px')
            .style('padding', '15px')
            .style('background', '#f8f9fa')
            .style('border-radius', '8px')
            .style('border', '1px solid #dee2e6');
        
        textContainer
            .append('h5')
            .text('Text with Entity Highlighting')
            .style('margin-bottom', '10px')
            .style('color', '#495057');
        
        const textDiv = textContainer
            .append('div')
            .attr('class', 'ner-text')
            .style('line-height', '2.2')
            .style('font-size', '16px')
            .style('background', 'white')
            .style('padding', '15px')
            .style('border-radius', '4px');
        
        // Sort entities by start position to avoid overlaps
        const sortedEntities = [...nerData].sort((a, b) => a.start - b.start);
        
        // Create text segments with entity highlighting
        let currentPos = 0;
        let textSegments = [];
        
        sortedEntities.forEach(entity => {
            // Add text before entity
            if (entity.start > currentPos) {
                textSegments.push({
                    type: 'text',
                    content: originalText.slice(currentPos, entity.start)
                });
            }
            
            // Add entity
            textSegments.push({
                type: 'entity',
                content: entity.text,
                entity: entity
            });
            
            currentPos = entity.end;
        });
        
        // Add remaining text
        if (currentPos < originalText.length) {
            textSegments.push({
                type: 'text',
                content: originalText.slice(currentPos)
            });
        }
        
        // Render text segments
        textSegments.forEach(segment => {
            if (segment.type === 'text') {
                textDiv.append('span').text(segment.content);
            } else {
                const entitySpan = textDiv
                    .append('span')
                    .attr('class', 'ner-entity')
                    .style('background-color', this.nerColors[segment.entity.label] || '#6C757D')
                    .style('color', 'white')
                    .style('padding', '4px 8px')
                    .style('border-radius', '4px')
                    .style('margin', '0 2px')
                    .style('cursor', 'pointer')
                    .style('transition', 'all 0.3s ease')
                    .style('border', '2px solid transparent')
                    .style('font-weight', '500')
                    .text(segment.content);
                
                // Add hover effects and tooltips
                entitySpan
                    .on('mouseover', (event) => {
                        // Highlight entity
                        d3.select(event.target)
                            .style('transform', 'scale(1.05)')
                            .style('border-color', '#333')
                            .style('box-shadow', '0 4px 15px rgba(0,0,0,0.3)');
                        
                        // Show tooltip
                        const tooltipContent = `
                            <strong>Entity:</strong> ${segment.entity.text}<br>
                            <strong>Label:</strong> ${segment.entity.label}<br>
                            <strong>Confidence:</strong> ${(segment.entity.confidence * 100).toFixed(1)}%<br>
                            <strong>Position:</strong> ${segment.entity.start}-${segment.entity.end}
                        `;
                        this.showTooltip(tooltipContent, event);
                    })
                    .on('mouseout', (event) => {
                        // Remove highlight
                        d3.select(event.target)
                            .style('transform', 'scale(1)')
                            .style('border-color', 'transparent')
                            .style('box-shadow', 'none');
                        
                        // Hide tooltip
                        this.hideTooltip();
                    })
                    .on('click', (event) => {
                        // Emit custom event for entity click
                        const customEvent = new CustomEvent('nerEntityClick', {
                            detail: segment.entity
                        });
                        document.dispatchEvent(customEvent);
                        
                        console.log('🖱️ NER entity clicked:', segment.entity);
                    });
            }
        });
    }
    
    /**
     * Add entity list with details
     */
    addNEREntityList(container, nerData) {
        if (nerData.length === 0) return;
        
        const listContainer = container
            .append('div')
            .attr('class', 'ner-entity-list')
            .style('margin-bottom', '20px')
            .style('padding', '15px')
            .style('background', '#fff3cd')
            .style('border-radius', '8px')
            .style('border', '1px solid #ffeaa7');
        
        listContainer
            .append('h5')
            .text('Detected Entities')
            .style('margin-bottom', '10px')
            .style('color', '#856404');
        
        const entitiesGrid = listContainer
            .append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fill, minmax(250px, 1fr))')
            .style('gap', '10px');
        
        nerData.forEach(entity => {
            const entityCard = entitiesGrid
                .append('div')
                .style('background', 'white')
                .style('padding', '10px')
                .style('border-radius', '6px')
                .style('border-left', `4px solid ${this.nerColors[entity.label] || '#6C757D'}`)
                .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
                .style('cursor', 'pointer')
                .style('transition', 'transform 0.2s ease');
            
            entityCard
                .on('mouseover', function() {
                    d3.select(this).style('transform', 'translateY(-2px)');
                })
                .on('mouseout', function() {
                    d3.select(this).style('transform', 'translateY(0)');
                })
                .on('click', () => {
                    const customEvent = new CustomEvent('nerEntityClick', { detail: entity });
                    document.dispatchEvent(customEvent);
                });
            
            entityCard
                .append('div')
                .style('font-weight', '600')
                .style('color', '#333')
                .style('margin-bottom', '4px')
                .text(entity.text);
            
            entityCard
                .append('div')
                .style('font-size', '12px')
                .style('color', '#666')
                .style('margin-bottom', '2px')
                .text(`${entity.label} • ${(entity.confidence * 100).toFixed(1)}%`);
            
            entityCard
                .append('div')
                .style('font-size', '11px')
                .style('color', '#999')
                .text(`Position: ${entity.start}-${entity.end}`);
        });
    }
    
    /**
     * Add NER legend
     */
    addNERLegend(container, nerData) {
        const uniqueLabels = [...new Set(nerData.map(d => d.label))];
        
        const legendContainer = container
            .append('div')
            .attr('class', 'ner-legend')
            .style('margin-bottom', '20px')
            .style('padding', '15px')
            .style('background', '#f8f9fa')
            .style('border-radius', '8px')
            .style('border', '1px solid #dee2e6');
        
        legendContainer
            .append('h5')
            .text('Entity Types Legend')
            .style('margin-bottom', '10px')
            .style('color', '#495057');
        
        const legendItems = legendContainer
            .append('div')
            .style('display', 'flex')
            .style('flex-wrap', 'wrap')
            .style('gap', '8px');
        
        uniqueLabels.forEach(label => {
            const count = nerData.filter(d => d.label === label).length;
            
            const legendItem = legendItems
                .append('div')
                .style('display', 'flex')
                .style('align-items', 'center')
                .style('gap', '5px')
                .style('padding', '6px 12px')
                .style('background', 'white')
                .style('border-radius', '20px')
                .style('font-size', '12px')
                .style('border', '1px solid #dee2e6')
                .style('font-weight', '500');
            
            legendItem
                .append('div')
                .style('width', '12px')
                .style('height', '12px')
                .style('border-radius', '50%')
                .style('background-color', this.nerColors[label] || '#6C757D');
            
            legendItem
                .append('span')
                .text(`${label} (${count})`);
        });
    }
    
    /**
     * Add NER statistics
     */
    addNERStatistics(container, nerData) {
        const statsContainer = container
            .append('div')
            .attr('class', 'ner-statistics')
            .style('padding', '15px')
            .style('background', '#e8f5e8')
            .style('border-radius', '8px')
            .style('border', '1px solid #c3e6c3');
        
        statsContainer
            .append('h5')
            .text('Entity Statistics')
            .style('margin-bottom', '10px')
            .style('color', '#155724');
        
        const stats = statsContainer
            .append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fit, minmax(150px, 1fr))')
            .style('gap', '10px')
            .style('font-size', '14px');
        
        // Total entities
        stats
            .append('div')
            .html(`<strong>Total Entities:</strong> ${nerData.length}`);
        
        // Unique entity types
        const uniqueTypes = new Set(nerData.map(d => d.label)).size;
        stats
            .append('div')
            .html(`<strong>Entity Types:</strong> ${uniqueTypes}`);
        
        // Average confidence
        if (nerData.length > 0) {
            const avgConfidence = nerData.reduce((sum, d) => sum + d.confidence, 0) / nerData.length;
            stats
                .append('div')
                .html(`<strong>Avg Confidence:</strong> ${(avgConfidence * 100).toFixed(1)}%`);
        }
        
        // Most common entity type
        if (nerData.length > 0) {
            const typeCounts = {};
            nerData.forEach(d => {
                typeCounts[d.label] = (typeCounts[d.label] || 0) + 1;
            });
            const mostCommonType = Object.keys(typeCounts).reduce((a, b) => 
                typeCounts[a] > typeCounts[b] ? a : b
            );
            stats
                .append('div')
                .html(`<strong>Most Common:</strong> ${mostCommonType} (${typeCounts[mostCommonType]})`);
        }
    }
    
    /**
     * Get color for NER label
     */
    getNERColor(label) {
        return this.nerColors[label] || '#6C757D';
    }
    
    /**
     * Clear NER visualization
     */
    clearNERVisualization(containerId = 'ner-visualization') {
        const container = d3.select(`#${containerId}`);
        container.selectAll('*').remove();
    }
    
    /**
     * Render Dependency Tree visualization with D3.js
     * @param {Array} dependencies - Array of {head, dependent, relation, head_text, dependent_text} objects
     * @param {Array} tokens - Array of token strings
     * @param {string} containerId - ID of container element
     */
    renderDependencyTree(dependencies, tokens, containerId = 'dependency-visualization') {
        console.log('🌳 Rendering dependency tree visualization:', dependencies.length, 'dependencies');
        
        const container = d3.select(`#${containerId}`);
        if (container.empty()) {
            console.error('❌ Dependency visualization container not found:', containerId);
            return;
        }
        
        // Clear previous content
        container.selectAll('*').remove();
        
        // Create main container
        const mainDiv = container
            .append('div')
            .attr('class', 'dependency-visualization-container')
            .style('font-family', 'Arial, sans-serif');
        
        // Add title
        mainDiv
            .append('h4')
            .text('Dependency Parse Tree')
            .style('margin-bottom', '15px')
            .style('color', '#333');
        
        // Create SVG container
        const svgContainer = mainDiv
            .append('div')
            .style('width', '100%')
            .style('overflow-x', 'auto')
            .style('background', '#fafafa')
            .style('border', '1px solid #dee2e6')
            .style('border-radius', '8px')
            .style('padding', '20px');
        
        // Calculate dimensions
        const tokenCount = tokens.length;
        const width = Math.max(800, tokenCount * 80);
        const height = 400;
        
        const svg = svgContainer
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('class', 'dependency-tree');
        
        // Define arrow marker
        svg.append('defs')
            .append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#666');
        
        // Create node positions
        const nodePositions = tokens.map((token, i) => ({
            id: i,
            token: token,
            x: (width / (tokenCount + 1)) * (i + 1),
            y: height - 80
        }));
        
        // Create dependency arcs
        const arcs = dependencies.map(dep => {
            const headPos = nodePositions[dep.head];
            const depPos = nodePositions[dep.dependent];
            
            if (!headPos || !depPos) return null;
            
            return {
                ...dep,
                headPos,
                depPos,
                color: this.dependencyColors[dep.relation] || '#666'
            };
        }).filter(arc => arc !== null);
        
        // Draw dependency arcs
        const arcGroup = svg.append('g').attr('class', 'dependency-arcs');
        
        arcs.forEach(arc => {
            const x1 = arc.headPos.x;
            const y1 = arc.headPos.y - 20;
            const x2 = arc.depPos.x;
            const y2 = arc.depPos.y - 20;
            
            // Calculate arc height based on distance
            const distance = Math.abs(x2 - x1);
            const arcHeight = Math.min(150, Math.max(40, distance * 0.3));
            
            // Create curved path
            const midX = (x1 + x2) / 2;
            const midY = Math.min(y1, y2) - arcHeight;
            
            const path = `M ${x1} ${y1} Q ${midX} ${midY} ${x2} ${y2}`;
            
            // Draw arc
            const arcPath = arcGroup
                .append('path')
                .attr('d', path)
                .attr('class', 'dependency-link')
                .style('stroke', arc.color)
                .style('fill', 'none')
                .style('stroke-width', '2px')
                .style('marker-end', 'url(#arrowhead)')
                .style('cursor', 'pointer');
            
            // Add hover effects
            arcPath
                .on('mouseover', (event) => {
                    d3.select(event.target)
                        .style('stroke-width', '3px')
                        .style('stroke', '#333');
                    
                    const tooltipContent = `
                        <strong>Relation:</strong> ${arc.relation}<br>
                        <strong>Head:</strong> ${arc.head_text}<br>
                        <strong>Dependent:</strong> ${arc.dependent_text}
                    `;
                    this.showTooltip(tooltipContent, event);
                })
                .on('mouseout', (event) => {
                    d3.select(event.target)
                        .style('stroke-width', '2px')
                        .style('stroke', arc.color);
                    
                    this.hideTooltip();
                })
                .on('click', () => {
                    const customEvent = new CustomEvent('dependencyClick', { detail: arc });
                    document.dispatchEvent(customEvent);
                    console.log('🖱️ Dependency clicked:', arc);
                });
            
            // Add relation label
            arcGroup
                .append('text')
                .attr('x', midX)
                .attr('y', midY - 5)
                .attr('class', 'dependency-label')
                .style('text-anchor', 'middle')
                .style('font-size', '11px')
                .style('font-weight', '600')
                .style('fill', arc.color)
                .style('background', 'white')
                .style('padding', '2px 4px')
                .style('border-radius', '3px')
                .style('cursor', 'pointer')
                .text(arc.relation)
                .on('mouseover', (event) => {
                    const tooltipContent = `
                        <strong>Relation:</strong> ${arc.relation}<br>
                        <strong>Head:</strong> ${arc.head_text}<br>
                        <strong>Dependent:</strong> ${arc.dependent_text}
                    `;
                    this.showTooltip(tooltipContent, event);
                })
                .on('mouseout', () => {
                    this.hideTooltip();
                });
        });
        
        // Draw nodes (tokens)
        const nodeGroup = svg.append('g').attr('class', 'dependency-nodes');
        
        nodePositions.forEach((node, i) => {
            const nodeG = nodeGroup
                .append('g')
                .attr('class', 'dependency-node')
                .attr('transform', `translate(${node.x}, ${node.y})`)
                .style('cursor', 'pointer');
            
            // Node circle
            nodeG
                .append('circle')
                .attr('r', 20)
                .style('fill', '#fff')
                .style('stroke', '#333')
                .style('stroke-width', '2px');
            
            // Node label (token)
            nodeG
                .append('text')
                .attr('y', 5)
                .style('text-anchor', 'middle')
                .style('font-size', '12px')
                .style('font-weight', '600')
                .style('fill', '#333')
                .text(node.token.length > 8 ? node.token.substring(0, 8) + '...' : node.token);
            
            // Position index
            nodeG
                .append('text')
                .attr('y', 35)
                .style('text-anchor', 'middle')
                .style('font-size', '10px')
                .style('fill', '#666')
                .text(i);
            
            // Add hover effects
            nodeG
                .on('mouseover', (event) => {
                    nodeG.select('circle')
                        .style('fill', '#e3f2fd')
                        .style('stroke', '#1976d2')
                        .style('stroke-width', '3px');
                    
                    const tooltipContent = `
                        <strong>Token:</strong> ${node.token}<br>
                        <strong>Position:</strong> ${i}<br>
                        <strong>Dependencies:</strong> ${dependencies.filter(d => d.head === i || d.dependent === i).length}
                    `;
                    this.showTooltip(tooltipContent, event);
                })
                .on('mouseout', () => {
                    nodeG.select('circle')
                        .style('fill', '#fff')
                        .style('stroke', '#333')
                        .style('stroke-width', '2px');
                    
                    this.hideTooltip();
                })
                .on('click', () => {
                    const customEvent = new CustomEvent('dependencyNodeClick', { 
                        detail: { token: node.token, position: i } 
                    });
                    document.dispatchEvent(customEvent);
                    console.log('🖱️ Dependency node clicked:', node);
                });
        });
        
        // Add dependency legend
        this.addDependencyLegend(mainDiv, dependencies);
        
        // Add dependency statistics
        this.addDependencyStatistics(mainDiv, dependencies, tokens);
        
        console.log('✅ Dependency tree visualization rendered successfully');
    }
    
    /**
     * Add dependency legend
     */
    addDependencyLegend(container, dependencies) {
        const uniqueRelations = [...new Set(dependencies.map(d => d.relation))];
        
        const legendContainer = container
            .append('div')
            .attr('class', 'dependency-legend')
            .style('margin-top', '20px')
            .style('padding', '15px')
            .style('background', '#f8f9fa')
            .style('border-radius', '8px')
            .style('border', '1px solid #dee2e6');
        
        legendContainer
            .append('h5')
            .text('Dependency Relations Legend')
            .style('margin-bottom', '10px')
            .style('color', '#495057');
        
        const legendItems = legendContainer
            .append('div')
            .style('display', 'flex')
            .style('flex-wrap', 'wrap')
            .style('gap', '8px');
        
        uniqueRelations.forEach(relation => {
            const count = dependencies.filter(d => d.relation === relation).length;
            
            const legendItem = legendItems
                .append('div')
                .style('display', 'flex')
                .style('align-items', 'center')
                .style('gap', '5px')
                .style('padding', '6px 12px')
                .style('background', 'white')
                .style('border-radius', '20px')
                .style('font-size', '12px')
                .style('border', '1px solid #dee2e6')
                .style('font-weight', '500');
            
            legendItem
                .append('div')
                .style('width', '12px')
                .style('height', '12px')
                .style('border-radius', '50%')
                .style('background-color', this.dependencyColors[relation] || '#666');
            
            legendItem
                .append('span')
                .text(`${relation} (${count})`);
        });
    }
    
    /**
     * Add dependency statistics
     */
    addDependencyStatistics(container, dependencies, tokens) {
        const statsContainer = container
            .append('div')
            .attr('class', 'dependency-statistics')
            .style('margin-top', '15px')
            .style('padding', '15px')
            .style('background', '#fff3cd')
            .style('border-radius', '8px')
            .style('border', '1px solid #ffeaa7');
        
        statsContainer
            .append('h5')
            .text('Dependency Statistics')
            .style('margin-bottom', '10px')
            .style('color', '#856404');
        
        const stats = statsContainer
            .append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fit, minmax(150px, 1fr))')
            .style('gap', '10px')
            .style('font-size', '14px');
        
        // Total dependencies
        stats
            .append('div')
            .html(`<strong>Total Dependencies:</strong> ${dependencies.length}`);
        
        // Total tokens
        stats
            .append('div')
            .html(`<strong>Total Tokens:</strong> ${tokens.length}`);
        
        // Unique relations
        const uniqueRelations = new Set(dependencies.map(d => d.relation)).size;
        stats
            .append('div')
            .html(`<strong>Unique Relations:</strong> ${uniqueRelations}`);
        
        // Most common relation
        if (dependencies.length > 0) {
            const relationCounts = {};
            dependencies.forEach(d => {
                relationCounts[d.relation] = (relationCounts[d.relation] || 0) + 1;
            });
            const mostCommonRelation = Object.keys(relationCounts).reduce((a, b) => 
                relationCounts[a] > relationCounts[b] ? a : b
            );
            stats
                .append('div')
                .html(`<strong>Most Common:</strong> ${mostCommonRelation} (${relationCounts[mostCommonRelation]})`);
        }
    }
    
    /**
     * Get color for dependency relation
     */
    getDependencyColor(relation) {
        return this.dependencyColors[relation] || '#666';
    }
    
    /**
     * Clear dependency visualization
     */
    clearDependencyVisualization(containerId = 'dependency-visualization') {
        const container = d3.select(`#${containerId}`);
        container.selectAll('*').remove();
    }
    
    /**
     * Render Sentiment Analysis visualization with D3.js
     * @param {Object} sentimentData - {overall_sentiment, confidence, scores} object
     * @param {string} containerId - ID of container element
     */
    renderSentimentAnalysis(sentimentData, containerId = 'sentiment-visualization') {
        console.log('😊 Rendering sentiment analysis visualization:', sentimentData);
        
        const container = d3.select(`#${containerId}`);
        if (container.empty()) {
            console.error('❌ Sentiment visualization container not found:', containerId);
            return;
        }
        
        // Clear previous content
        container.selectAll('*').remove();
        
        // Create main container
        const mainDiv = container
            .append('div')
            .attr('class', 'sentiment-visualization-container')
            .style('font-family', 'Arial, sans-serif');
        
        // Add title
        mainDiv
            .append('h4')
            .text('Sentiment Analysis')
            .style('margin-bottom', '15px')
            .style('color', '#333');
        
        // Create overall sentiment display
        this.createOverallSentiment(mainDiv, sentimentData);
        
        // Create sentiment gauge
        this.createSentimentGauge(mainDiv, sentimentData);
        
        // Create detailed scores
        this.createSentimentScores(mainDiv, sentimentData);
        
        // Add sentiment statistics
        this.addSentimentStatistics(mainDiv, sentimentData);
        
        console.log('✅ Sentiment analysis visualization rendered successfully');
    }
    
    /**
     * Create overall sentiment display
     */
    createOverallSentiment(container, sentimentData) {
        const overallContainer = container
            .append('div')
            .attr('class', 'sentiment-overall')
            .style('text-align', 'center')
            .style('padding', '20px')
            .style('margin-bottom', '20px')
            .style('background', this.getSentimentBackgroundColor(sentimentData.overall_sentiment))
            .style('border-radius', '12px')
            .style('border', '2px solid ' + this.getSentimentBorderColor(sentimentData.overall_sentiment));
        
        // Sentiment emoji
        const emoji = this.getSentimentEmoji(sentimentData.overall_sentiment);
        overallContainer
            .append('div')
            .style('font-size', '48px')
            .style('margin-bottom', '10px')
            .text(emoji);
        
        // Overall sentiment text
        overallContainer
            .append('h3')
            .style('margin-bottom', '8px')
            .style('color', this.getSentimentTextColor(sentimentData.overall_sentiment))
            .text(`Overall Sentiment: ${sentimentData.overall_sentiment}`);
        
        // Confidence
        overallContainer
            .append('div')
            .style('font-size', '16px')
            .style('color', '#666')
            .text(`Confidence: ${(sentimentData.confidence * 100).toFixed(1)}%`);
    }
    
    /**
     * Create sentiment gauge visualization
     */
    createSentimentGauge(container, sentimentData) {
        const gaugeContainer = container
            .append('div')
            .attr('class', 'sentiment-gauge-container')
            .style('margin-bottom', '20px')
            .style('padding', '20px')
            .style('background', '#f8f9fa')
            .style('border-radius', '8px')
            .style('border', '1px solid #dee2e6');
        
        gaugeContainer
            .append('h5')
            .text('Sentiment Gauge')
            .style('text-align', 'center')
            .style('margin-bottom', '15px')
            .style('color', '#495057');
        
        // Create SVG for gauge
        const gaugeWidth = 300;
        const gaugeHeight = 200;
        
        const gaugeSvg = gaugeContainer
            .append('div')
            .style('display', 'flex')
            .style('justify-content', 'center')
            .append('svg')
            .attr('width', gaugeWidth)
            .attr('height', gaugeHeight)
            .attr('class', 'sentiment-gauge');
        
        const centerX = gaugeWidth / 2;
        const centerY = gaugeHeight - 40;
        const radius = 80;
        
        // Create gauge background arc
        const arc = d3.arc()
            .innerRadius(radius - 20)
            .outerRadius(radius)
            .startAngle(-Math.PI)
            .endAngle(0);
        
        gaugeSvg
            .append('path')
            .attr('d', arc)
            .attr('transform', `translate(${centerX}, ${centerY})`)
            .style('fill', '#e9ecef');
        
        // Create sentiment value arc
        const sentimentValue = this.getSentimentValue(sentimentData);
        const sentimentAngle = -Math.PI + (sentimentValue + 1) * Math.PI / 2; // Map -1 to 1 to -π to 0
        
        const valueArc = d3.arc()
            .innerRadius(radius - 20)
            .outerRadius(radius)
            .startAngle(-Math.PI)
            .endAngle(sentimentAngle);
        
        gaugeSvg
            .append('path')
            .attr('d', valueArc)
            .attr('transform', `translate(${centerX}, ${centerY})`)
            .style('fill', this.getSentimentColor(sentimentData.overall_sentiment))
            .style('transition', 'all 0.5s ease');
        
        // Add gauge labels
        const labels = [
            { text: 'Negative', angle: -Math.PI, color: '#dc3545' },
            { text: 'Neutral', angle: -Math.PI / 2, color: '#6c757d' },
            { text: 'Positive', angle: 0, color: '#28a745' }
        ];
        
        labels.forEach(label => {
            const labelX = centerX + (radius + 30) * Math.cos(label.angle);
            const labelY = centerY + (radius + 30) * Math.sin(label.angle);
            
            gaugeSvg
                .append('text')
                .attr('x', labelX)
                .attr('y', labelY)
                .style('text-anchor', 'middle')
                .style('font-size', '12px')
                .style('font-weight', '600')
                .style('fill', label.color)
                .text(label.text);
        });
        
        // Add needle
        const needleAngle = -Math.PI + (sentimentValue + 1) * Math.PI / 2;
        const needleLength = radius - 10;
        const needleX = centerX + needleLength * Math.cos(needleAngle);
        const needleY = centerY + needleLength * Math.sin(needleAngle);
        
        gaugeSvg
            .append('line')
            .attr('x1', centerX)
            .attr('y1', centerY)
            .attr('x2', needleX)
            .attr('y2', needleY)
            .style('stroke', '#333')
            .style('stroke-width', '3px')
            .style('stroke-linecap', 'round');
        
        // Add center circle
        gaugeSvg
            .append('circle')
            .attr('cx', centerX)
            .attr('cy', centerY)
            .attr('r', 8)
            .style('fill', '#333');
        
        // Add value text
        gaugeSvg
            .append('text')
            .attr('x', centerX)
            .attr('y', centerY + 30)
            .style('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', '700')
            .style('fill', '#333')
            .text(sentimentValue.toFixed(2));
    }
    
    /**
     * Create detailed sentiment scores
     */
    createSentimentScores(container, sentimentData) {
        const scoresContainer = container
            .append('div')
            .attr('class', 'sentiment-scores-container')
            .style('margin-bottom', '20px')
            .style('padding', '20px')
            .style('background', '#fff')
            .style('border-radius', '8px')
            .style('border', '1px solid #dee2e6');
        
        scoresContainer
            .append('h5')
            .text('Detailed Sentiment Scores')
            .style('margin-bottom', '15px')
            .style('color', '#495057');
        
        const scoresGrid = scoresContainer
            .append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fit, minmax(200px, 1fr))')
            .style('gap', '15px');
        
        Object.entries(sentimentData.scores).forEach(([sentiment, score]) => {
            const scoreItem = scoresGrid
                .append('div')
                .attr('class', 'sentiment-score-item')
                .style('background', '#f8f9fa')
                .style('padding', '15px')
                .style('border-radius', '8px')
                .style('border-left', `4px solid ${this.getSentimentColor(sentiment)}`);
            
            scoreItem
                .append('div')
                .attr('class', 'sentiment-score-label')
                .style('font-weight', '600')
                .style('color', '#495057')
                .style('margin-bottom', '8px')
                .style('text-transform', 'capitalize')
                .text(sentiment);
            
            scoreItem
                .append('div')
                .attr('class', 'sentiment-score-value')
                .style('font-size', '1.5rem')
                .style('font-weight', '700')
                .style('color', this.getSentimentColor(sentiment))
                .style('margin-bottom', '8px')
                .text(score.toFixed(3));
            
            // Add progress bar
            const progressBar = scoreItem
                .append('div')
                .attr('class', 'sentiment-bar')
                .style('width', '100%')
                .style('height', '8px')
                .style('background', '#e9ecef')
                .style('border-radius', '4px')
                .style('overflow', 'hidden');
            
            progressBar
                .append('div')
                .attr('class', 'sentiment-bar-fill')
                .style('height', '100%')
                .style('background', this.getSentimentColor(sentiment))
                .style('border-radius', '4px')
                .style('width', '0%')
                .style('transition', 'width 0.8s ease')
                .transition()
                .duration(800)
                .style('width', `${score * 100}%`);
            
            // Add percentage text
            scoreItem
                .append('div')
                .style('font-size', '12px')
                .style('color', '#666')
                .style('margin-top', '4px')
                .text(`${(score * 100).toFixed(1)}%`);
        });
    }
    
    /**
     * Add sentiment statistics
     */
    addSentimentStatistics(container, sentimentData) {
        const statsContainer = container
            .append('div')
            .attr('class', 'sentiment-statistics')
            .style('padding', '15px')
            .style('background', '#e3f2fd')
            .style('border-radius', '8px')
            .style('border', '1px solid #bbdefb');
        
        statsContainer
            .append('h5')
            .text('Analysis Summary')
            .style('margin-bottom', '10px')
            .style('color', '#1565c0');
        
        const stats = statsContainer
            .append('div')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fit, minmax(200px, 1fr))')
            .style('gap', '10px')
            .style('font-size', '14px');
        
        // Overall sentiment
        stats
            .append('div')
            .html(`<strong>Overall Sentiment:</strong> ${sentimentData.overall_sentiment}`);
        
        // Confidence level
        const confidenceLevel = sentimentData.confidence > 0.8 ? 'High' : 
                               sentimentData.confidence > 0.6 ? 'Medium' : 'Low';
        stats
            .append('div')
            .html(`<strong>Confidence Level:</strong> ${confidenceLevel}`);
        
        // Dominant emotion
        const dominantEmotion = Object.keys(sentimentData.scores).reduce((a, b) => 
            sentimentData.scores[a] > sentimentData.scores[b] ? a : b
        );
        stats
            .append('div')
            .html(`<strong>Dominant Emotion:</strong> ${dominantEmotion}`);
        
        // Score range
        const scores = Object.values(sentimentData.scores);
        const scoreRange = (Math.max(...scores) - Math.min(...scores)).toFixed(3);
        stats
            .append('div')
            .html(`<strong>Score Range:</strong> ${scoreRange}`);
    }
    
    /**
     * Get sentiment value for gauge (-1 to 1)
     */
    getSentimentValue(sentimentData) {
        const scores = sentimentData.scores;
        if (scores.positive && scores.negative) {
            return scores.positive - scores.negative;
        } else if (scores.pos && scores.neg) {
            return scores.pos - scores.neg;
        } else {
            // Fallback based on overall sentiment
            switch (sentimentData.overall_sentiment.toLowerCase()) {
                case 'positive': return 0.7;
                case 'negative': return -0.7;
                case 'neutral': return 0;
                default: return 0;
            }
        }
    }
    
    /**
     * Get sentiment color
     */
    getSentimentColor(sentiment) {
        const colors = {
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d',
            'pos': '#28a745',
            'neg': '#dc3545',
            'neu': '#6c757d'
        };
        return colors[sentiment.toLowerCase()] || '#6c757d';
    }
    
    /**
     * Get sentiment background color
     */
    getSentimentBackgroundColor(sentiment) {
        const colors = {
            'positive': '#d4edda',
            'negative': '#f8d7da',
            'neutral': '#e2e3e5'
        };
        return colors[sentiment.toLowerCase()] || '#e2e3e5';
    }
    
    /**
     * Get sentiment border color
     */
    getSentimentBorderColor(sentiment) {
        const colors = {
            'positive': '#c3e6cb',
            'negative': '#f5c6cb',
            'neutral': '#d6d8db'
        };
        return colors[sentiment.toLowerCase()] || '#d6d8db';
    }
    
    /**
     * Get sentiment text color
     */
    getSentimentTextColor(sentiment) {
        const colors = {
            'positive': '#155724',
            'negative': '#721c24',
            'neutral': '#383d41'
        };
        return colors[sentiment.toLowerCase()] || '#383d41';
    }
    
    /**
     * Get sentiment emoji
     */
    getSentimentEmoji(sentiment) {
        const emojis = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        };
        return emojis[sentiment.toLowerCase()] || '😐';
    }
    
    /**
     * Clear sentiment visualization
     */
    clearSentimentVisualization(containerId = 'sentiment-visualization') {
        const container = d3.select(`#${containerId}`);
        container.selectAll('*').remove();
    }
}

// Create global instance
window.nlpVisualization = new NLPVisualizationEngine();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NLPVisualizationEngine;
}