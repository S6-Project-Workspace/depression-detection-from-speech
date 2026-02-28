/**
 * Interactive Metrics Exploration Components
 * 
 * This module provides interactive features for detailed metric exploration,
 * including drill-down capabilities, tooltips, and detailed breakdowns.
 * 
 * Implements Requirement 4.5 for interactive metric exploration.
 */

class InteractiveMetricsExplorer {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.detailModal = null;
        this.currentMetricData = null;
        this.tooltipTimeout = null;
        
        this.initializeModal();
        this.setupInteractiveFeatures();
    }
    
    initializeModal() {
        // Create modal for detailed metric exploration
        const modalHTML = `
            <div class="modal fade" id="metricDetailModal" tabindex="-1" aria-labelledby="metricDetailModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="metricDetailModalLabel">Metric Details</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="metric-detail-content">
                                <!-- Content will be populated dynamically -->
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" id="export-metric-data">
                                <i class="fas fa-download me-2"></i>Export Data
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document if it doesn't exist
        if (!document.getElementById('metricDetailModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }
        
        this.detailModal = new bootstrap.Modal(document.getElementById('metricDetailModal'));
        
        // Setup export functionality
        document.getElementById('export-metric-data').addEventListener('click', () => {
            this.exportCurrentMetricData();
        });
    }
    
    setupInteractiveFeatures() {
        // Add click handlers to metric cards for drill-down
        this.setupMetricCardInteractions();
        
        // Add hover tooltips for charts
        this.setupChartTooltips();
        
        // Add context menus for charts
        this.setupChartContextMenus();
        
        // Add keyboard shortcuts
        this.setupKeyboardShortcuts();
    }
    
    setupMetricCardInteractions() {
        // Make metric cards clickable for detailed exploration
        const metricCards = document.querySelectorAll('.metric-card');
        
        metricCards.forEach(card => {
            card.style.cursor = 'pointer';
            card.addEventListener('click', (e) => {
                const cardType = this.getMetricCardType(card);
                this.showMetricDetails(cardType);
            });
            
            // Add hover effects
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-2px)';
                card.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = 'none';
            });
        });
    }
    
    getMetricCardType(card) {
        // Determine metric type based on card content
        if (card.querySelector('#total-models')) return 'total-models';
        if (card.querySelector('#avg-f1-score')) return 'avg-f1-score';
        if (card.querySelector('#best-model-name')) return 'best-model';
        if (card.querySelector('#success-rate')) return 'success-rate';
        return 'unknown';
    }
    
    async showMetricDetails(metricType) {
        try {
            let detailContent = '';
            
            switch (metricType) {
                case 'total-models':
                    detailContent = await this.generateModelListDetails();
                    break;
                case 'avg-f1-score':
                    detailContent = await this.generateF1ScoreDetails();
                    break;
                case 'best-model':
                    detailContent = await this.generateBestModelDetails();
                    break;
                case 'success-rate':
                    detailContent = await this.generateSuccessRateDetails();
                    break;
                default:
                    detailContent = '<p>No detailed information available.</p>';
            }
            
            document.getElementById('metric-detail-content').innerHTML = detailContent;
            document.getElementById('metricDetailModalLabel').textContent = 
                this.getMetricTitle(metricType);
            
            this.detailModal.show();
            
        } catch (error) {
            console.error('Failed to show metric details:', error);
            this.showErrorInModal('Failed to load metric details');
        }
    }
    
    getMetricTitle(metricType) {
        const titles = {
            'total-models': 'Model Inventory Details',
            'avg-f1-score': 'F1 Score Analysis',
            'best-model': 'Best Model Performance',
            'success-rate': 'Processing Success Rate Analysis'
        };
        return titles[metricType] || 'Metric Details';
    }
    
    async generateModelListDetails() {
        try {
            const response = await fetch('/api/metrics/models?limit=50');
            const models = await response.json();
            
            let html = `
                <div class="row">
                    <div class="col-md-8">
                        <h6>All Models</h6>
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Model Name</th>
                                        <th>Version</th>
                                        <th>F1 Score</th>
                                        <th>Accuracy</th>
                                        <th>Training Date</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
            `;
            
            models.forEach(model => {
                html += `
                    <tr>
                        <td><strong>${model.model_name}</strong></td>
                        <td><span class="badge bg-secondary">${model.version}</span></td>
                        <td>${model.f1_score.toFixed(3)}</td>
                        <td>${model.accuracy.toFixed(3)}</td>
                        <td>${new Date(model.training_date).toLocaleDateString()}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary" onclick="interactiveExplorer.showTrainingHistory('${model.model_name}', '${model.version}')">
                                <i class="fas fa-chart-line"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-info" onclick="interactiveExplorer.showModelComparison(['${model.model_name}'])">
                                <i class="fas fa-balance-scale"></i>
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <h6>Model Statistics</h6>
                        <div class="card">
                            <div class="card-body">
                                <p><strong>Total Models:</strong> ${models.length}</p>
                                <p><strong>Avg F1 Score:</strong> ${(models.reduce((sum, m) => sum + m.f1_score, 0) / models.length).toFixed(3)}</p>
                                <p><strong>Best F1 Score:</strong> ${Math.max(...models.map(m => m.f1_score)).toFixed(3)}</p>
                                <p><strong>Worst F1 Score:</strong> ${Math.min(...models.map(m => m.f1_score)).toFixed(3)}</p>
                            </div>
                        </div>
                        
                        <h6 class="mt-3">Model Distribution</h6>
                        <canvas id="model-distribution-chart" width="300" height="200"></canvas>
                    </div>
                </div>
            `;
            
            // Create distribution chart after modal is shown
            setTimeout(() => {
                this.createModelDistributionChart(models);
            }, 500);
            
            return html;
            
        } catch (error) {
            console.error('Failed to generate model list details:', error);
            return '<p class="text-danger">Failed to load model details.</p>';
        }
    }
    
    createModelDistributionChart(models) {
        const ctx = document.getElementById('model-distribution-chart');
        if (!ctx) return;
        
        // Group models by F1 score ranges
        const ranges = {
            'Excellent (>0.9)': 0,
            'Good (0.8-0.9)': 0,
            'Fair (0.7-0.8)': 0,
            'Poor (<0.7)': 0
        };
        
        models.forEach(model => {
            if (model.f1_score > 0.9) ranges['Excellent (>0.9)']++;
            else if (model.f1_score > 0.8) ranges['Good (0.8-0.9)']++;
            else if (model.f1_score > 0.7) ranges['Fair (0.7-0.8)']++;
            else ranges['Poor (<0.7)']++;
        });
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(ranges),
                datasets: [{
                    data: Object.values(ranges),
                    backgroundColor: [
                        '#28a745',
                        '#17a2b8',
                        '#ffc107',
                        '#dc3545'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    async generateF1ScoreDetails() {
        try {
            const response = await fetch('/api/metrics/performance-history?days=30');
            const data = await response.json();
            
            const performanceData = data.performance_data || [];
            
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>F1 Score Distribution</h6>
                        <canvas id="f1-distribution-chart" width="400" height="300"></canvas>
                    </div>
                    <div class="col-md-6">
                        <h6>F1 Score Trends (30 days)</h6>
                        <canvas id="f1-trends-chart" width="400" height="300"></canvas>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h6>Detailed F1 Score Analysis</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                        <th>Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Average F1 Score</td>
                                        <td>${data.summary.avg_f1.toFixed(3)}</td>
                                        <td>Mean F1 score across all models</td>
                                    </tr>
                                    <tr>
                                        <td>Best F1 Score</td>
                                        <td>${data.summary.best_f1.toFixed(3)}</td>
                                        <td>Highest F1 score achieved</td>
                                    </tr>
                                    <tr>
                                        <td>Latest F1 Score</td>
                                        <td>${data.summary.latest_f1.toFixed(3)}</td>
                                        <td>Most recent F1 score</td>
                                    </tr>
                                    <tr>
                                        <td>Total Evaluations</td>
                                        <td>${data.summary.total_evaluations}</td>
                                        <td>Number of model evaluations</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
            
            // Create charts after modal is shown
            setTimeout(() => {
                this.createF1DistributionChart(performanceData);
                this.createF1TrendsChart(performanceData);
            }, 500);
            
            return html;
            
        } catch (error) {
            console.error('Failed to generate F1 score details:', error);
            return '<p class="text-danger">Failed to load F1 score details.</p>';
        }
    }
    
    createF1DistributionChart(performanceData) {
        const ctx = document.getElementById('f1-distribution-chart');
        if (!ctx) return;
        
        const f1Scores = performanceData.map(p => p.f1_score);
        
        new Chart(ctx, {
            type: 'histogram',
            data: {
                datasets: [{
                    label: 'F1 Score Distribution',
                    data: f1Scores,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'F1 Score'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                }
            }
        });
    }
    
    createF1TrendsChart(performanceData) {
        const ctx = document.getElementById('f1-trends-chart');
        if (!ctx) return;
        
        const sortedData = performanceData.sort((a, b) => 
            new Date(a.evaluation_date) - new Date(b.evaluation_date)
        );
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: sortedData.map(p => new Date(p.evaluation_date).toLocaleDateString()),
                datasets: [{
                    label: 'F1 Score',
                    data: sortedData.map(p => p.f1_score),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'F1 Score'
                        }
                    }
                }
            }
        });
    }
    
    async generateBestModelDetails() {
        try {
            const response = await fetch('/api/metrics/models?limit=1');
            const models = await response.json();
            
            if (models.length === 0) {
                return '<p>No model data available.</p>';
            }
            
            const bestModel = models[0];
            const evalData = bestModel.evaluation_data;
            
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Model Information</h6>
                        <table class="table table-borderless">
                            <tr><td><strong>Name:</strong></td><td>${bestModel.model_name}</td></tr>
                            <tr><td><strong>Version:</strong></td><td>${bestModel.version}</td></tr>
                            <tr><td><strong>Training Date:</strong></td><td>${new Date(bestModel.training_date).toLocaleDateString()}</td></tr>
                            <tr><td><strong>F1 Score:</strong></td><td>${bestModel.f1_score.toFixed(3)}</td></tr>
                            <tr><td><strong>Accuracy:</strong></td><td>${bestModel.accuracy.toFixed(3)}</td></tr>
                            <tr><td><strong>Precision:</strong></td><td>${bestModel.precision.toFixed(3)}</td></tr>
                            <tr><td><strong>Recall:</strong></td><td>${bestModel.recall.toFixed(3)}</td></tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h6>Performance Breakdown</h6>
                        <canvas id="best-model-breakdown-chart" width="300" height="300"></canvas>
                    </div>
                </div>
            `;
            
            if (evalData.confusion_matrix) {
                html += `
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <h6>Confusion Matrix</h6>
                            <table class="table table-bordered text-center">
                                <thead>
                                    <tr>
                                        <th></th>
                                        <th>Predicted Non-Depressed</th>
                                        <th>Predicted Depressed</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <th>Actual Non-Depressed</th>
                                        <td class="bg-success text-white">${evalData.confusion_matrix[0][0]}</td>
                                        <td class="bg-danger text-white">${evalData.confusion_matrix[0][1]}</td>
                                    </tr>
                                    <tr>
                                        <th>Actual Depressed</th>
                                        <td class="bg-danger text-white">${evalData.confusion_matrix[1][0]}</td>
                                        <td class="bg-success text-white">${evalData.confusion_matrix[1][1]}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="col-md-6">
                            <h6>Training Information</h6>
                            <p><strong>Training Epochs:</strong> ${evalData.training_epochs || 'N/A'}</p>
                            <p><strong>Best Epoch:</strong> ${evalData.best_epoch || 'N/A'}</p>
                            <p><strong>Training Time:</strong> ${(evalData.training_time_minutes || 0).toFixed(1)} minutes</p>
                            <p><strong>Dataset Size:</strong> ${evalData.dataset_size || 'N/A'} samples</p>
                            <p><strong>Test Samples:</strong> ${evalData.test_samples || 'N/A'}</p>
                        </div>
                    </div>
                `;
            }
            
            // Create performance breakdown chart
            setTimeout(() => {
                this.createPerformanceBreakdownChart(bestModel);
            }, 500);
            
            return html;
            
        } catch (error) {
            console.error('Failed to generate best model details:', error);
            return '<p class="text-danger">Failed to load best model details.</p>';
        }
    }
    
    createPerformanceBreakdownChart(model) {
        const ctx = document.getElementById('best-model-breakdown-chart');
        if (!ctx) return;
        
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['F1 Score', 'Accuracy', 'Precision', 'Recall'],
                datasets: [{
                    label: model.model_name,
                    data: [
                        model.f1_score,
                        model.accuracy,
                        model.precision,
                        model.recall
                    ],
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
    }
    
    async generateSuccessRateDetails() {
        try {
            const response = await fetch('/api/metrics/realtime-summary?hours=24');
            const data = await response.json();
            
            const overall = data.overall;
            const byType = data.by_operation_type;
            const hourly = data.hourly_breakdown;
            
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Success Rate by Operation Type</h6>
                        <canvas id="success-by-type-chart" width="400" height="300"></canvas>
                    </div>
                    <div class="col-md-6">
                        <h6>Hourly Success Rate</h6>
                        <canvas id="hourly-success-chart" width="400" height="300"></canvas>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h6>Detailed Statistics (24 hours)</h6>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5 class="card-title">${overall.total_operations}</h5>
                                        <p class="card-text">Total Operations</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5 class="card-title">${overall.successful_operations}</h5>
                                        <p class="card-text">Successful</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5 class="card-title">${overall.failed_operations}</h5>
                                        <p class="card-text">Failed</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5 class="card-title">${(overall.success_rate * 100).toFixed(1)}%</h5>
                                        <p class="card-text">Success Rate</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Create charts after modal is shown
            setTimeout(() => {
                this.createSuccessByTypeChart(byType);
                this.createHourlySuccessChart(hourly);
            }, 500);
            
            return html;
            
        } catch (error) {
            console.error('Failed to generate success rate details:', error);
            return '<p class="text-danger">Failed to load success rate details.</p>';
        }
    }
    
    createSuccessByTypeChart(byType) {
        const ctx = document.getElementById('success-by-type-chart');
        if (!ctx) return;
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: byType.map(t => t.operation_type),
                datasets: [{
                    label: 'Success Rate',
                    data: byType.map(t => t.success_rate * 100),
                    backgroundColor: 'rgba(40, 167, 69, 0.6)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Success Rate (%)'
                        }
                    }
                }
            }
        });
    }
    
    createHourlySuccessChart(hourly) {
        const ctx = document.getElementById('hourly-success-chart');
        if (!ctx) return;
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: hourly.map(h => new Date(h.hour).getHours() + ':00'),
                datasets: [{
                    label: 'Operations',
                    data: hourly.map(h => h.operations),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Operations Count'
                        }
                    }
                }
            }
        });
    }
    
    setupChartTooltips() {
        // Enhanced tooltips for all charts
        Chart.defaults.plugins.tooltip.callbacks.title = function(context) {
            return context[0].label || '';
        };
        
        Chart.defaults.plugins.tooltip.callbacks.afterBody = function(context) {
            // Add additional context information
            return ['Click for detailed analysis'];
        };
    }
    
    setupChartContextMenus() {
        // Add right-click context menus to charts
        document.addEventListener('contextmenu', (e) => {
            const canvas = e.target.closest('canvas');
            if (canvas && canvas.id.includes('chart')) {
                e.preventDefault();
                this.showChartContextMenu(e, canvas);
            }
        });
    }
    
    showChartContextMenu(event, canvas) {
        // Create context menu for chart interactions
        const menu = document.createElement('div');
        menu.className = 'chart-context-menu';
        menu.style.cssText = `
            position: fixed;
            top: ${event.clientY}px;
            left: ${event.clientX}px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 1000;
            padding: 8px 0;
        `;
        
        const menuItems = [
            { text: 'Export as PNG', action: () => this.exportChart(canvas, 'png') },
            { text: 'Export as SVG', action: () => this.exportChart(canvas, 'svg') },
            { text: 'View Data Table', action: () => this.showChartDataTable(canvas) },
            { text: 'Refresh Data', action: () => this.refreshChartData(canvas) }
        ];
        
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.textContent = item.text;
            menuItem.style.cssText = `
                padding: 8px 16px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
            `;
            menuItem.addEventListener('click', () => {
                item.action();
                document.body.removeChild(menu);
            });
            menu.appendChild(menuItem);
        });
        
        document.body.appendChild(menu);
        
        // Remove menu when clicking elsewhere
        setTimeout(() => {
            document.addEventListener('click', () => {
                if (document.body.contains(menu)) {
                    document.body.removeChild(menu);
                }
            }, { once: true });
        }, 100);
    }
    
    exportChart(canvas, format) {
        const link = document.createElement('a');
        link.download = `chart-${Date.now()}.${format}`;
        link.href = canvas.toDataURL(`image/${format}`);
        link.click();
    }
    
    showChartDataTable(canvas) {
        // Extract chart data and show in table format
        const chart = Chart.getChart(canvas);
        if (!chart) return;
        
        const data = chart.data;
        let tableHTML = `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Label</th>
        `;
        
        data.datasets.forEach(dataset => {
            tableHTML += `<th>${dataset.label}</th>`;
        });
        
        tableHTML += '</tr></thead><tbody>';
        
        data.labels.forEach((label, index) => {
            tableHTML += `<tr><td>${label}</td>`;
            data.datasets.forEach(dataset => {
                tableHTML += `<td>${dataset.data[index]}</td>`;
            });
            tableHTML += '</tr>';
        });
        
        tableHTML += '</tbody></table></div>';
        
        // Show in modal
        document.getElementById('metric-detail-content').innerHTML = tableHTML;
        document.getElementById('metricDetailModalLabel').textContent = 'Chart Data';
        this.detailModal.show();
    }
    
    refreshChartData(canvas) {
        // Refresh the specific chart's data
        const chart = Chart.getChart(canvas);
        if (chart && this.dashboard) {
            this.dashboard.loadDashboardData();
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + R: Refresh dashboard
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                if (this.dashboard) {
                    this.dashboard.loadDashboardData();
                }
            }
            
            // Ctrl/Cmd + E: Export current view
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                this.exportCurrentView();
            }
            
            // Escape: Close modal
            if (e.key === 'Escape' && this.detailModal) {
                this.detailModal.hide();
            }
        });
    }
    
    exportCurrentView() {
        // Export current dashboard view as PDF or image
        const dashboardElement = document.querySelector('.container-fluid');
        if (dashboardElement) {
            // Use html2canvas or similar library to export
            console.log('Export functionality would be implemented here');
        }
    }
    
    exportCurrentMetricData() {
        if (!this.currentMetricData) {
            console.warn('No metric data to export');
            return;
        }
        
        const dataStr = JSON.stringify(this.currentMetricData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `metric-data-${Date.now()}.json`;
        link.click();
    }
    
    showErrorInModal(message) {
        document.getElementById('metric-detail-content').innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
    
    // Public methods for external access
    async showTrainingHistory(modelName, version) {
        if (this.dashboard) {
            await this.dashboard.loadTrainingHistory(modelName, version);
        }
    }
    
    async showModelComparison(modelNames) {
        if (this.dashboard) {
            await this.dashboard.compareModels(modelNames);
        }
    }
}

// Global instance
let interactiveExplorer = null;

// Initialize when dashboard is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for dashboard to be initialized
    setTimeout(() => {
        if (window.metricsDashboard) {
            interactiveExplorer = new InteractiveMetricsExplorer(window.metricsDashboard);
        }
    }, 1000);
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InteractiveMetricsExplorer;
}