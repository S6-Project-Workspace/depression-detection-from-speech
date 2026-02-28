/**
 * Result Display and Organization Module
 * Handles collapsible sections, enhanced tabs, and interactive result display
 */

class ResultDisplayManager {
    constructor() {
        this.collapsibleSections = new Map();
        this.enhancedTabs = new Map();
        this.expandableContent = new Map();
        
        this.init();
    }
    
    /**
     * Initialize result display features
     */
    init() {
        console.log('🎨 Initializing Result Display Manager...');
        
        this.setupCollapsibleSections();
        this.setupEnhancedTabs();
        this.setupExpandableContent();
        this.setupInteractiveElements();
        this.setupSearchAndFilter();
        
        console.log('✅ Result Display Manager initialized');
    }
    
    /**
     * Set up collapsible sections
     */
    setupCollapsibleSections() {
        const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
        
        collapsibleHeaders.forEach(header => {
            const section = header.closest('.collapsible-section');
            const content = section?.querySelector('.collapsible-content');
            const toggle = header.querySelector('.collapsible-toggle');
            
            if (section && content && toggle) {
                const sectionId = section.id || `collapsible-${Date.now()}-${Math.random()}`;
                section.id = sectionId;
                
                this.collapsibleSections.set(sectionId, {
                    header,
                    content,
                    toggle,
                    isExpanded: content.classList.contains('expanded')
                });
                
                header.addEventListener('click', () => {
                    this.toggleCollapsibleSection(sectionId);
                });
                
                // Set initial state
                this.updateCollapsibleState(sectionId);
            }
        });
    }
    
    /**
     * Toggle collapsible section
     */
    toggleCollapsibleSection(sectionId) {
        const section = this.collapsibleSections.get(sectionId);
        if (!section) return;
        
        section.isExpanded = !section.isExpanded;
        this.updateCollapsibleState(sectionId);
        
        // Dispatch custom event
        document.dispatchEvent(new CustomEvent('collapsibleToggle', {
            detail: { sectionId, isExpanded: section.isExpanded }
        }));
    }
    
    /**
     * Update collapsible section state
     */
    updateCollapsibleState(sectionId) {
        const section = this.collapsibleSections.get(sectionId);
        if (!section) return;
        
        const { header, content, toggle, isExpanded } = section;
        
        if (isExpanded) {
            header.classList.add('active');
            content.classList.add('expanded');
            toggle.classList.add('expanded');
            toggle.innerHTML = '▲';
        } else {
            header.classList.remove('active');
            content.classList.remove('expanded');
            toggle.classList.remove('expanded');
            toggle.innerHTML = '▼';
        }
    }
    
    /**
     * Set up enhanced tabs
     */
    setupEnhancedTabs() {
        const tabContainers = document.querySelectorAll('.enhanced-tabs');
        
        tabContainers.forEach(container => {
            const tabButtons = container.querySelectorAll('.enhanced-tab-btn');
            const tabPanes = container.querySelectorAll('.enhanced-tab-pane');
            
            if (tabButtons.length === 0 || tabPanes.length === 0) return;
            
            const containerId = container.id || `tabs-${Date.now()}-${Math.random()}`;
            container.id = containerId;
            
            this.enhancedTabs.set(containerId, {
                container,
                buttons: Array.from(tabButtons),
                panes: Array.from(tabPanes),
                activeTab: this.getActiveTabIndex(tabButtons)
            });
            
            tabButtons.forEach((button, index) => {
                button.addEventListener('click', () => {
                    this.switchTab(containerId, index);
                });
            });
            
            // Set initial state
            this.updateTabState(containerId);
        });
    }
    
    /**
     * Get active tab index
     */
    getActiveTabIndex(tabButtons) {
        for (let i = 0; i < tabButtons.length; i++) {
            if (tabButtons[i].classList.contains('active')) {
                return i;
            }
        }
        return 0; // Default to first tab
    }
    
    /**
     * Switch to specific tab
     */
    switchTab(containerId, tabIndex) {
        const tabGroup = this.enhancedTabs.get(containerId);
        if (!tabGroup || tabIndex < 0 || tabIndex >= tabGroup.buttons.length) return;
        
        tabGroup.activeTab = tabIndex;
        this.updateTabState(containerId);
        
        // Dispatch custom event
        document.dispatchEvent(new CustomEvent('tabSwitch', {
            detail: { containerId, tabIndex, tabId: tabGroup.buttons[tabIndex].dataset.tab }
        }));
    }
    
    /**
     * Update tab state
     */
    updateTabState(containerId) {
        const tabGroup = this.enhancedTabs.get(containerId);
        if (!tabGroup) return;
        
        const { buttons, panes, activeTab } = tabGroup;
        
        // Update buttons
        buttons.forEach((button, index) => {
            if (index === activeTab) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
        
        // Update panes
        panes.forEach((pane, index) => {
            if (index === activeTab) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
    }
    
    /**
     * Set up expandable content
     */
    setupExpandableContent() {
        const expandableElements = document.querySelectorAll('.expandable-content');
        
        expandableElements.forEach(element => {
            const expandButton = element.querySelector('.expand-button');
            if (!expandButton) return;
            
            const elementId = element.id || `expandable-${Date.now()}-${Math.random()}`;
            element.id = elementId;
            
            this.expandableContent.set(elementId, {
                element,
                button: expandButton,
                isExpanded: element.classList.contains('expanded')
            });
            
            expandButton.addEventListener('click', () => {
                this.toggleExpandableContent(elementId);
            });
            
            // Set initial button text
            this.updateExpandableState(elementId);
        });
    }
    
    /**
     * Toggle expandable content
     */
    toggleExpandableContent(elementId) {
        const expandable = this.expandableContent.get(elementId);
        if (!expandable) return;
        
        expandable.isExpanded = !expandable.isExpanded;
        this.updateExpandableState(elementId);
        
        // Dispatch custom event
        document.dispatchEvent(new CustomEvent('contentExpand', {
            detail: { elementId, isExpanded: expandable.isExpanded }
        }));
    }
    
    /**
     * Update expandable content state
     */
    updateExpandableState(elementId) {
        const expandable = this.expandableContent.get(elementId);
        if (!expandable) return;
        
        const { element, button, isExpanded } = expandable;
        
        if (isExpanded) {
            element.classList.add('expanded');
            element.classList.remove('collapsed');
            button.textContent = 'Show Less';
            
            // Hide the overlay
            const overlay = element.querySelector('.expand-overlay');
            if (overlay) overlay.style.display = 'none';
        } else {
            element.classList.remove('expanded');
            element.classList.add('collapsed');
            button.textContent = 'Show More';
            
            // Show the overlay
            const overlay = element.querySelector('.expand-overlay');
            if (overlay) overlay.style.display = 'flex';
        }
    }
    
    /**
     * Set up interactive elements
     */
    setupInteractiveElements() {
        const interactiveElements = document.querySelectorAll('.interactive-element');
        
        interactiveElements.forEach(element => {
            // Add click handler if data-action is specified
            const action = element.dataset.action;
            if (action) {
                element.addEventListener('click', () => {
                    this.handleInteractiveAction(action, element);
                });
            }
            
            // Add hover effects
            element.addEventListener('mouseenter', () => {
                element.style.transform = 'scale(1.02)';
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.transform = 'scale(1)';
            });
        });
    }
    
    /**
     * Handle interactive element actions
     */
    handleInteractiveAction(action, element) {
        const data = element.dataset;
        
        switch (action) {
            case 'highlight':
                this.highlightElement(element);
                break;
            case 'copy':
                this.copyElementContent(element);
                break;
            case 'expand':
                this.expandElement(element);
                break;
            case 'filter':
                this.filterByElement(element);
                break;
            default:
                console.log(`🎯 Interactive action: ${action}`, data);
        }
        
        // Dispatch custom event
        document.dispatchEvent(new CustomEvent('interactiveAction', {
            detail: { action, element, data }
        }));
    }
    
    /**
     * Highlight element
     */
    highlightElement(element) {
        element.style.background = 'var(--primary-light)';
        element.style.transform = 'scale(1.05)';
        
        setTimeout(() => {
            element.style.background = '';
            element.style.transform = '';
        }, 1000);
    }
    
    /**
     * Copy element content to clipboard
     */
    async copyElementContent(element) {
        const text = element.textContent || element.innerText || '';
        
        try {
            await navigator.clipboard.writeText(text);
            this.showCopyFeedback(element);
        } catch (error) {
            console.error('Failed to copy text:', error);
            // Fallback for older browsers
            this.fallbackCopyText(text);
        }
    }
    
    /**
     * Show copy feedback
     */
    showCopyFeedback(element) {
        const originalText = element.textContent;
        element.textContent = 'Copied!';
        element.style.color = 'var(--success-color)';
        
        setTimeout(() => {
            element.textContent = originalText;
            element.style.color = '';
        }, 1000);
    }
    
    /**
     * Fallback copy method for older browsers
     */
    fallbackCopyText(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            console.log('Text copied using fallback method');
        } catch (error) {
            console.error('Fallback copy failed:', error);
        }
        
        document.body.removeChild(textArea);
    }
    
    /**
     * Set up search and filter functionality
     */
    setupSearchAndFilter() {
        const searchInputs = document.querySelectorAll('.search-input');
        const filterDropdowns = document.querySelectorAll('.filter-dropdown');
        
        // Set up search inputs
        searchInputs.forEach(input => {
            let searchTimeout;
            
            input.addEventListener('input', (event) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.handleSearch(event.target.value, input);
                }, 300); // Debounce search
            });
        });
        
        // Set up filter dropdowns
        filterDropdowns.forEach(dropdown => {
            const button = dropdown.querySelector('.filter-dropdown-btn');
            const content = dropdown.querySelector('.filter-dropdown-content');
            const options = dropdown.querySelectorAll('.filter-option');
            
            if (button && content) {
                button.addEventListener('click', (event) => {
                    event.stopPropagation();
                    this.toggleFilterDropdown(dropdown);
                });
                
                options.forEach(option => {
                    option.addEventListener('click', () => {
                        this.selectFilterOption(dropdown, option);
                    });
                });
            }
        });
        
        // Close dropdowns when clicking outside
        document.addEventListener('click', () => {
            filterDropdowns.forEach(dropdown => {
                dropdown.classList.remove('active');
            });
        });
    }
    
    /**
     * Handle search functionality
     */
    handleSearch(query, input) {
        const targetSelector = input.dataset.target;
        if (!targetSelector) return;
        
        const targetElements = document.querySelectorAll(targetSelector);
        const searchQuery = query.toLowerCase().trim();
        
        targetElements.forEach(element => {
            const text = element.textContent.toLowerCase();
            const shouldShow = searchQuery === '' || text.includes(searchQuery);
            
            if (shouldShow) {
                element.style.display = '';
                element.classList.remove('search-hidden');
            } else {
                element.style.display = 'none';
                element.classList.add('search-hidden');
            }
        });
        
        // Dispatch search event
        document.dispatchEvent(new CustomEvent('searchPerformed', {
            detail: { query: searchQuery, target: targetSelector, input }
        }));
    }
    
    /**
     * Toggle filter dropdown
     */
    toggleFilterDropdown(dropdown) {
        const isActive = dropdown.classList.contains('active');
        
        // Close all other dropdowns
        document.querySelectorAll('.filter-dropdown').forEach(d => {
            d.classList.remove('active');
        });
        
        // Toggle current dropdown
        if (!isActive) {
            dropdown.classList.add('active');
        }
    }
    
    /**
     * Select filter option
     */
    selectFilterOption(dropdown, option) {
        const options = dropdown.querySelectorAll('.filter-option');
        const button = dropdown.querySelector('.filter-dropdown-btn');
        const value = option.dataset.value;
        
        // Update active option
        options.forEach(opt => opt.classList.remove('active'));
        option.classList.add('active');
        
        // Update button text
        if (button) {
            button.textContent = option.textContent;
        }
        
        // Close dropdown
        dropdown.classList.remove('active');
        
        // Apply filter
        this.applyFilter(dropdown, value);
        
        // Dispatch filter event
        document.dispatchEvent(new CustomEvent('filterApplied', {
            detail: { dropdown, value, option }
        }));
    }
    
    /**
     * Apply filter
     */
    applyFilter(dropdown, value) {
        const targetSelector = dropdown.dataset.target;
        if (!targetSelector) return;
        
        const targetElements = document.querySelectorAll(targetSelector);
        
        targetElements.forEach(element => {
            const elementValue = element.dataset.filterValue || '';
            const shouldShow = value === 'all' || elementValue === value;
            
            if (shouldShow) {
                element.style.display = '';
                element.classList.remove('filter-hidden');
            } else {
                element.style.display = 'none';
                element.classList.add('filter-hidden');
            }
        });
    }
    
    /**
     * Create result card
     */
    createResultCard(title, content, actions = []) {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        const header = document.createElement('div');
        header.className = 'result-card-header';
        
        const titleElement = document.createElement('div');
        titleElement.className = 'result-card-title';
        titleElement.textContent = title;
        
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'result-card-actions';
        
        actions.forEach(action => {
            const button = document.createElement('button');
            button.className = 'btn btn-small';
            button.textContent = action.label;
            button.addEventListener('click', action.handler);
            actionsContainer.appendChild(button);
        });
        
        header.appendChild(titleElement);
        header.appendChild(actionsContainer);
        
        const body = document.createElement('div');
        body.className = 'result-card-body';
        
        if (typeof content === 'string') {
            body.innerHTML = content;
        } else {
            body.appendChild(content);
        }
        
        card.appendChild(header);
        card.appendChild(body);
        
        return card;
    }
    
    /**
     * Create collapsible section
     */
    createCollapsibleSection(title, content, isExpanded = false) {
        const section = document.createElement('div');
        section.className = 'collapsible-section';
        
        const header = document.createElement('div');
        header.className = 'collapsible-header';
        
        const titleElement = document.createElement('div');
        titleElement.className = 'collapsible-title';
        titleElement.innerHTML = `<span class="collapsible-icon">📋</span>${title}`;
        
        const toggle = document.createElement('button');
        toggle.className = 'collapsible-toggle';
        toggle.innerHTML = isExpanded ? '▲' : '▼';
        
        header.appendChild(titleElement);
        header.appendChild(toggle);
        
        const contentContainer = document.createElement('div');
        contentContainer.className = `collapsible-content ${isExpanded ? 'expanded' : ''}`;
        
        const body = document.createElement('div');
        body.className = 'collapsible-body';
        
        if (typeof content === 'string') {
            body.innerHTML = content;
        } else {
            body.appendChild(content);
        }
        
        contentContainer.appendChild(body);
        
        section.appendChild(header);
        section.appendChild(contentContainer);
        
        // Add to DOM first, then initialize
        document.body.appendChild(section);
        this.setupCollapsibleSections();
        document.body.removeChild(section);
        
        return section;
    }
    
    /**
     * Create enhanced tabs
     */
    createEnhancedTabs(tabs) {
        const container = document.createElement('div');
        container.className = 'enhanced-tabs';
        
        const nav = document.createElement('div');
        nav.className = 'enhanced-tab-nav';
        
        const content = document.createElement('div');
        content.className = 'enhanced-tab-content';
        
        tabs.forEach((tab, index) => {
            // Create tab button
            const button = document.createElement('button');
            button.className = `enhanced-tab-btn ${index === 0 ? 'active' : ''}`;
            button.dataset.tab = tab.id;
            
            button.innerHTML = `
                <span class="tab-icon">${tab.icon || '📄'}</span>
                <span class="tab-text">${tab.label}</span>
                ${tab.badge ? `<span class="enhanced-tab-badge">${tab.badge}</span>` : ''}
            `;
            
            nav.appendChild(button);
            
            // Create tab pane
            const pane = document.createElement('div');
            pane.className = `enhanced-tab-pane ${index === 0 ? 'active' : ''}`;
            pane.id = `${tab.id}-pane`;
            
            if (typeof tab.content === 'string') {
                pane.innerHTML = tab.content;
            } else {
                pane.appendChild(tab.content);
            }
            
            content.appendChild(pane);
        });
        
        container.appendChild(nav);
        container.appendChild(content);
        
        // Add to DOM first, then initialize
        document.body.appendChild(container);
        this.setupEnhancedTabs();
        document.body.removeChild(container);
        
        return container;
    }
    
    /**
     * Update tab badge
     */
    updateTabBadge(containerId, tabIndex, badgeText) {
        const tabGroup = this.enhancedTabs.get(containerId);
        if (!tabGroup || !tabGroup.buttons[tabIndex]) return;
        
        const button = tabGroup.buttons[tabIndex];
        let badge = button.querySelector('.enhanced-tab-badge');
        
        if (!badge) {
            badge = document.createElement('span');
            badge.className = 'enhanced-tab-badge';
            button.appendChild(badge);
        }
        
        badge.textContent = badgeText;
        
        if (badgeText === '' || badgeText === '0') {
            badge.style.display = 'none';
        } else {
            badge.style.display = 'inline-block';
        }
    }
    
    /**
     * Get current active tab
     */
    getActiveTab(containerId) {
        const tabGroup = this.enhancedTabs.get(containerId);
        return tabGroup ? tabGroup.activeTab : -1;
    }
    
    /**
     * Expand all collapsible sections
     */
    expandAllSections() {
        this.collapsibleSections.forEach((section, sectionId) => {
            if (!section.isExpanded) {
                this.toggleCollapsibleSection(sectionId);
            }
        });
    }
    
    /**
     * Collapse all collapsible sections
     */
    collapseAllSections() {
        this.collapsibleSections.forEach((section, sectionId) => {
            if (section.isExpanded) {
                this.toggleCollapsibleSection(sectionId);
            }
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.resultDisplayManager = new ResultDisplayManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResultDisplayManager;
}