/**
 * Metrics Dashboard Visualization Components
 * 
 * This module implements interactive dashboard components for displaying
 * model performance metrics, training curves, and real-time statistics.
 * 
 * Uses Chart.js for data visualization and provides real-time updates.
 * Implements Requirements 4.2, 4.3, 4.4 for dashboard visualization.
 */

class MetricsDashboard {
    constructor() {
        this.charts = {};
        this.updateInterval = null;
        this.isInitialized = false;
        
        // Chart.js default configuration
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.color = '#333';
        
        // Color palette for consistent styling
        this.colors = {
            primary: '#3498db',
            secondary: '#2ecc71',
            warning: '#f39c12',
            danger: '#e74c3c',
            info: '#9b59b6',
            success: '#27ae60',
            muted: '#95a5a6'
        };
        
        this.gradientColors = [
            'rgba(52, 152, 219, 0.8)',
            'rgba(46, 204, 113, 0.8)',
            'rgba(243, 156, 18, 0.8)',
            'rgba(231, 76, 60, 0.8)',
            'rgba(155, 89, 182, 0.8)',
            'rgba(39, 174, 96, 0.8)'
        ];
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        try {
            // Initialize dashboard components
            await this.loadDashboardData();
            this.setupEventListeners();
            this.startRealTimeUpdates();
            
            this.isInitialized = true;
            console.log('Metrics dashboard initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize metrics dashboard:', error);
            this.showError('Failed to initialize dashboard');
        }
    }
    
    async loadDashboardData() {
        try {
            const response = await fetch('/api/metrics/dashboard-data');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Update overview cards
            this.updateOverviewCards(data.overview);
            
            // Create/update charts
            this.createModelPerformanceChart(data.latest_models);
            this.createPerformanceTrendsChart(data.performance_trends);
            this.createRealTimeMetricsChart(data.realtime_metrics);
            
            // Update last refresh time
            this.updateLastRefreshTime();
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            throw error;
        }
    }
    
    updateOverviewCards(overview) {
        // Update total models
        const totalModelsElement = document.getElementById('total-models');
        if (totalModelsElement) {
            totalModelsElement.textContent = overview.total_models || 0;
        }
        
        // Update average F1 score
        const avgF1Element = document.getElementById('avg-f1-score');
        if (avgF1Element) {
            avgF1Element.textContent = (overview.avg_f1_score || 0).toFixed(3);
        }
        
        // Update best model info
        const bestModelElement = document.getElementById('best-model-name');
        const bestModelScoreElement = document.getElementById('best-model-score');
        
        if (bestModelElement && overview.best_model.name) {
            bestModelElement.textContent = overview.best_model.name;
        }
        
        if (bestModelScoreElement && overview.best_model.f1_score) {
            bestModelScoreElement.textContent = overview.best_model.f1_score.toFixed(3);
        }
        
        // Update processing stats
        const processingStatsElement = document.getElementById('processing-stats');
        if (processingStatsElement) {
            // This will be updated by real-time metrics
        }
    }
    
    createModelPerformanceChart(models) {
        const ctx = document.getElementById('model-performance-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.modelPerformance) {
            this.charts.modelPerformance.destroy();
        }
        
        const modelNames = models.map(m => m.model_name);
        const f1Scores = models.map(m => m.f1_score);
        const accuracyScores = models.map(m => m.accuracy);
        const precisionScores = models.map(m => m.precision);
        const recallScores = models.map(m => m.recall);
        
        this.charts.modelPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [
                    {
                        label: 'F1 Score',
                        data: f1Scores,
                        backgroundColor: this.colors.primary,
                        borderColor: this.colors.primary,
                        borderWidth: 1
                    },
                    {
                        label: 'Accuracy',
                        data: accuracyScores,
                        backgroundColor: this.colors.secondary,
                        borderColor: this.colors.secondary,
                        borderWidth: 1
                    },
                    {
                        label: 'Precision',
                        data: precisionScores,
                        backgroundColor: this.colors.warning,
                        borderColor: this.colors.warning,
                        borderWidth: 1
                    },
                    {
                        label: 'Recall',
                        data: recallScores,
                        backgroundColor: this.colors.info,
                        borderColor: this.colors.info,
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Performance Comparison'
                    },
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Score'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(2);
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Models'
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    createPerformanceTrendsChart(trendsData) {
        const ctx = document.getElementById('performance-trends-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.performanceTrends) {
            this.charts.performanceTrends.destroy();
        }
        
        const datasets = [];
        let colorIndex = 0;
        
        // Create dataset for each model
        Object.keys(trendsData).forEach(modelName => {
            const modelData = trendsData[modelName];
            const dates = modelData.map(d => new Date(d.date));
            const f1Scores = modelData.map(d => d.f1_score);
            
            datasets.push({
                label: modelName,
                data: f1Scores,
                borderColor: this.gradientColors[colorIndex % this.gradientColors.length],
                backgroundColor: this.gradientColors[colorIndex % this.gradientColors.length].replace('0.8', '0.2'),
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6
            });
            
            colorIndex++;
        });
        
        // Get all unique dates and sort them
        const allDates = new Set();
        Object.values(trendsData).forEach(modelData => {
            modelData.forEach(d => allDates.add(d.date));
        });
        const sortedDates = Array.from(allDates).sort();
        
        this.charts.performanceTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: sortedDates.map(date => new Date(date).toLocaleDateString()),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Trends Over Time'
                    },
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'F1 Score'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(2);
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    createRealTimeMetricsChart(realtimeData) {
        const ctx = document.getElementById('realtime-metrics-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.realtimeMetrics) {
            this.charts.realtimeMetrics.destroy();
        }
        
        // Update real-time stats display
        this.updateRealTimeStats(realtimeData);
        
        // Create hourly breakdown chart
        const hourlyData = realtimeData.hourly_breakdown || [];
        const hours = hourlyData.map(h => new Date(h.hour).getHours() + ':00');
        const operations = hourlyData.map(h => h.operations);
        const avgTimes = hourlyData.map(h => h.avg_processing_time);
        
        this.charts.realtimeMetrics = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours,
                datasets: [
                    {
                        label: 'Operations per Hour',
                        data: operations,
                        borderColor: this.colors.primary,
                        backgroundColor: this.colors.primary + '20',
                        borderWidth: 2,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Avg Processing Time (s)',
                        data: avgTimes,
                        borderColor: this.colors.warning,
                        backgroundColor: this.colors.warning + '20',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Real-Time Processing Metrics (24h)'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Operations Count'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Processing Time (s)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hour'
                        }
                    }
                }
            }
        });
    }
    
    updateRealTimeStats(realtimeData) {
        const overall = realtimeData.overall || {};
        const indicators = realtimeData.performance_indicators || {};
        
        // Update success rate
        const successRateElement = document.getElementById('success-rate');
        if (successRateElement) {
            const rate = (overall.success_rate || 0) * 100;
            successRateElement.textContent = rate.toFixed(1) + '%';
            
            // Color coding based on success rate
            successRateElement.className = rate >= 95 ? 'text-success' : 
                                         rate >= 90 ? 'text-warning' : 'text-danger';
        }
        
        // Update average processing time
        const avgTimeElement = document.getElementById('avg-processing-time');
        if (avgTimeElement) {
            const time = overall.avg_processing_time || 0;
            avgTimeElement.textContent = time.toFixed(3) + 's';
            
            // Color coding based on processing time
            avgTimeElement.className = time <= 0.1 ? 'text-success' : 
                                     time <= 0.5 ? 'text-warning' : 'text-danger';
        }
        
        // Update total operations
        const totalOpsElement = document.getElementById('total-operations');
        if (totalOpsElement) {
            totalOpsElement.textContent = overall.total_operations || 0;
        }
        
        // Update throughput
        const throughputElement = document.getElementById('throughput');
        if (throughputElement) {
            const throughput = indicators.avg_throughput_per_hour || 0;
            throughputElement.textContent = throughput.toFixed(1) + '/h';
        }
    }
    
    async loadTrainingHistory(modelName, version = null) {
        try {
            let url = `/api/metrics/training-history/${modelName}`;
            if (version) {
                url += `?version=${version}`;
            }
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.createTrainingCurvesChart(data);
            
        } catch (error) {
            console.error('Failed to load training history:', error);
            this.showError('Failed to load training history');
        }
    }
    
    createTrainingCurvesChart(trainingData) {
        const ctx = document.getElementById('training-curves-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.trainingCurves) {
            this.charts.trainingCurves.destroy();
        }
        
        const curves = trainingData.training_curves;
        
        this.charts.trainingCurves = new Chart(ctx, {
            type: 'line',
            data: {
                labels: curves.epochs,
                datasets: [
                    {
                        label: 'Training Loss',
                        data: curves.train_loss,
                        borderColor: this.colors.danger,
                        backgroundColor: this.colors.danger + '20',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Validation Loss',
                        data: curves.val_loss,
                        borderColor: this.colors.warning,
                        backgroundColor: this.colors.warning + '20',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Training F1',
                        data: curves.train_f1,
                        borderColor: this.colors.primary,
                        backgroundColor: this.colors.primary + '20',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Validation F1',
                        data: curves.val_f1,
                        borderColor: this.colors.secondary,
                        backgroundColor: this.colors.secondary + '20',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Training Curves - ${trainingData.model_name}`
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'F1 Score'
                        },
                        grid: {
                            drawOnChartArea: false
                        },
                        max: 1.0
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                }
            }
        });
        
        // Highlight best epoch
        const bestEpoch = trainingData.best_epoch;
        if (bestEpoch && this.charts.trainingCurves) {
            // Add annotation for best epoch (would need Chart.js annotation plugin)
            console.log(`Best epoch: ${bestEpoch.epoch} with F1: ${bestEpoch.val_f1}`);
        }
    }
    
    async compareModels(modelNames, metric = 'f1_score') {
        try {
            const params = new URLSearchParams();
            modelNames.forEach(name => params.append('model_names', name));
            params.append('metric', metric);
            
            const response = await fetch(`/api/metrics/comparison?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.createModelComparisonChart(data);
            
        } catch (error) {
            console.error('Failed to compare models:', error);
            this.showError('Failed to compare models');
        }
    }
    
    createModelComparisonChart(comparisonData) {
        const ctx = document.getElementById('model-comparison-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.modelComparison) {
            this.charts.modelComparison.destroy();
        }
        
        const models = comparisonData.comparison_data;
        const metric = comparisonData.comparison_metric;
        
        const modelNames = models.map(m => m.model_name);
        const scores = models.map(m => m[metric]);
        const colors = models.map((_, i) => this.gradientColors[i % this.gradientColors.length]);
        
        this.charts.modelComparison = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: metric.replace('_', ' ').toUpperCase(),
                    data: scores,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1.0')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Model Comparison - ${metric.replace('_', ' ').toUpperCase()}`
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const model = models[context.dataIndex];
                                return [
                                    `${metric}: ${context.parsed.x.toFixed(3)}`,
                                    `Rank: #${model.rank}`,
                                    `Version: ${model.version}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    }
                }
            }
        });
    }
    
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refresh-dashboard');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadDashboardData());
        }
        
        // Model selection for training curves
        const modelSelect = document.getElementById('training-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                if (e.target.value) {
                    this.loadTrainingHistory(e.target.value);
                }
            });
        }
        
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startRealTimeUpdates();
                } else {
                    this.stopRealTimeUpdates();
                }
            });
        }
    }
    
    startRealTimeUpdates() {
        // Update every 30 seconds
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            this.loadDashboardData();
        }, 30000);
    }
    
    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    updateLastRefreshTime() {
        const element = document.getElementById('last-refresh-time');
        if (element) {
            element.textContent = new Date().toLocaleTimeString();
        }
    }
    
    showError(message) {
        const errorContainer = document.getElementById('dashboard-errors');
        if (errorContainer) {
            errorContainer.innerHTML = `
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error:</strong> ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }
    }
    
    destroy() {
        // Clean up charts and intervals
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.isInitialized = false;
    }
}

// Global dashboard instance
let metricsDashboard = null;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    metricsDashboard = new MetricsDashboard();
    metricsDashboard.initialize();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetricsDashboard;
}