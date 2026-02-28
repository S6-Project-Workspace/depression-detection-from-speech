"""
API endpoints for model metrics and dashboard functionality.

This module provides REST API endpoints for accessing model performance metrics,
training history, and real-time processing statistics.

Implements Requirements 4.1, 4.2, 4.3, 4.4 for metrics dashboard functionality.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from .models import ModelMetrics
from .metrics_collector import (
    MetricsCollector, get_metrics_collector, TrainingMetrics,
    ModelPerformanceSnapshot, RealTimeMetrics
)

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/")
async def get_metrics_root(
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """
    Get basic metrics overview.
    
    Returns a summary of available metrics and system status.
    """
    try:
        # Ensure collector is initialized
        await collector.initialize()
        
        # Get basic system metrics with fallback
        try:
            realtime_summary = await collector.get_realtime_metrics_summary(hours=1)
            recent_operations = realtime_summary["overall"]["total_operations"]
            success_rate = realtime_summary["overall"]["success_rate"]
        except Exception as e:
            logger.warning(f"Could not get realtime metrics: {e}")
            recent_operations = 0
            success_rate = 1.0
        
        try:
            latest_models = await collector.get_latest_model_metrics(limit=3)
            model_count = len(latest_models)
        except Exception as e:
            logger.warning(f"Could not get model metrics: {e}")
            model_count = 0
        
        return {
            "status": "active",
            "metrics_available": True,
            "last_updated": datetime.now().isoformat(),
            "recent_operations": recent_operations,
            "success_rate": success_rate,
            "latest_models": model_count,
            "endpoints": {
                "models": "/api/metrics/models",
                "dashboard": "/api/metrics/dashboard-data",
                "realtime": "/api/metrics/realtime-summary"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get metrics root: {e}")
        return {
            "status": "error",
            "metrics_available": False,
            "error": str(e),
            "last_updated": datetime.now().isoformat(),
            "recent_operations": 0,
            "success_rate": 0.0,
            "latest_models": 0,
            "endpoints": {
                "models": "/api/metrics/models",
                "dashboard": "/api/metrics/dashboard-data",
                "realtime": "/api/metrics/realtime-summary"
            }
        }


@router.get("/models", response_model=List[ModelMetrics])
async def get_model_metrics(
    limit: int = Query(default=10, ge=1, le=100),
    model_name: Optional[str] = Query(default=None),
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> List[ModelMetrics]:
    """
    Get latest model performance metrics.
    
    Returns model performance data for dashboard visualization.
    Implements Requirement 4.1 for model performance display.
    """
    try:
        if model_name:
            # Filter by specific model
            all_metrics = await collector.get_latest_model_metrics(limit=100)
            filtered_metrics = [m for m in all_metrics if m.model_name == model_name]
            return filtered_metrics[:limit]
        else:
            return await collector.get_latest_model_metrics(limit=limit)
            
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model metrics")


@router.get("/training-history/{model_name}")
async def get_training_history(
    model_name: str,
    version: Optional[str] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=5000),
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """
    Get training history for a specific model.
    
    Returns training curves data for visualization.
    Implements Requirement 4.2 for training curve display.
    """
    try:
        history = await collector.get_training_history(
            model_name=model_name,
            version=version,
            limit=limit
        )
        
        if not history:
            raise HTTPException(status_code=404, detail=f"No training history found for model: {model_name}")
        
        # Format data for Chart.js
        epochs = [h.epoch for h in history]
        train_losses = [h.train_loss for h in history]
        val_losses = [h.val_loss for h in history]
        train_f1_scores = [h.train_f1 for h in history]
        val_f1_scores = [h.val_f1 for h in history]
        learning_rates = [h.learning_rate for h in history]
        timestamps = [h.timestamp.isoformat() for h in history]
        
        return {
            "model_name": model_name,
            "version": version,
            "total_epochs": len(history),
            "training_curves": {
                "epochs": epochs,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_f1": train_f1_scores,
                "val_f1": val_f1_scores,
                "learning_rate": learning_rates,
                "timestamps": timestamps
            },
            "best_epoch": {
                "epoch": max(history, key=lambda h: h.val_f1).epoch,
                "val_f1": max(h.val_f1 for h in history),
                "val_loss": min(h.val_loss for h in history)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training history for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training history")


@router.get("/performance-history")
async def get_performance_history(
    model_name: Optional[str] = Query(default=None),
    days: int = Query(default=30, ge=1, le=365),
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """
    Get model performance history over time.
    
    Returns historical performance data for trend analysis.
    Implements Requirement 4.2 for performance tracking over time.
    """
    try:
        history = await collector.get_model_performance_history(
            model_name=model_name,
            days=days
        )
        
        if not history:
            return {
                "model_name": model_name,
                "days": days,
                "performance_data": [],
                "summary": {
                    "total_evaluations": 0,
                    "avg_f1": 0.0,
                    "best_f1": 0.0,
                    "latest_f1": 0.0
                }
            }
        
        # Format data for visualization
        performance_data = []
        for perf in history:
            performance_data.append({
                "model_name": perf.model_name,
                "version": perf.version,
                "evaluation_date": perf.evaluation_date.isoformat(),
                "f1_score": perf.f1_score,
                "macro_f1": perf.macro_f1,
                "accuracy": perf.accuracy,
                "precision": perf.precision,
                "recall": perf.recall,
                "f1_depressed": perf.f1_depressed,
                "f1_non_depressed": perf.f1_non_depressed,
                "training_epochs": perf.training_epochs,
                "best_epoch": perf.best_epoch,
                "training_time_minutes": perf.training_time_minutes
            })
        
        # Calculate summary statistics
        f1_scores = [p.f1_score for p in history]
        summary = {
            "total_evaluations": len(history),
            "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "best_f1": max(f1_scores) if f1_scores else 0.0,
            "latest_f1": f1_scores[0] if f1_scores else 0.0  # history is ordered by date DESC
        }
        
        return {
            "model_name": model_name,
            "days": days,
            "performance_data": performance_data,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance history")


@router.get("/comparison")
async def compare_models(
    model_names: List[str] = Query(..., description="List of model names to compare"),
    metric: str = Query(default="f1_score", description="Metric to compare"),
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """
    Compare performance metrics across multiple models.
    
    Returns comparison data for model evaluation.
    Implements Requirement 4.3 for model comparison visualization.
    """
    try:
        if len(model_names) > 10:
            raise HTTPException(status_code=400, detail="Too many models for comparison (max 10)")
        
        valid_metrics = [
            "f1_score", "macro_f1", "accuracy", "precision", "recall",
            "f1_depressed", "f1_non_depressed"
        ]
        
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid metric. Must be one of: {valid_metrics}"
            )
        
        comparison_data = []
        
        for model_name in model_names:
            # Get latest performance for each model
            history = await collector.get_model_performance_history(
                model_name=model_name,
                days=365  # Look back up to a year
            )
            
            if history:
                latest = history[0]  # Most recent performance
                comparison_data.append({
                    "model_name": model_name,
                    "version": latest.version,
                    "evaluation_date": latest.evaluation_date.isoformat(),
                    "f1_score": latest.f1_score,
                    "macro_f1": latest.macro_f1,
                    "accuracy": latest.accuracy,
                    "precision": latest.precision,
                    "recall": latest.recall,
                    "f1_depressed": latest.f1_depressed,
                    "f1_non_depressed": latest.f1_non_depressed,
                    "precision_depressed": latest.precision_depressed,
                    "precision_non_depressed": latest.precision_non_depressed,
                    "recall_depressed": latest.recall_depressed,
                    "recall_non_depressed": latest.recall_non_depressed,
                    "training_epochs": latest.training_epochs,
                    "training_time_minutes": latest.training_time_minutes,
                    "dataset_size": latest.dataset_size,
                    "test_samples": latest.test_samples
                })
            else:
                # Model not found, add placeholder
                comparison_data.append({
                    "model_name": model_name,
                    "version": "N/A",
                    "evaluation_date": None,
                    "f1_score": 0.0,
                    "macro_f1": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_depressed": 0.0,
                    "f1_non_depressed": 0.0,
                    "precision_depressed": 0.0,
                    "precision_non_depressed": 0.0,
                    "recall_depressed": 0.0,
                    "recall_non_depressed": 0.0,
                    "training_epochs": 0,
                    "training_time_minutes": 0.0,
                    "dataset_size": 0,
                    "test_samples": 0
                })
        
        # Sort by the comparison metric
        comparison_data.sort(key=lambda x: x.get(metric, 0.0), reverse=True)
        
        # Calculate ranking and differences
        best_score = comparison_data[0].get(metric, 0.0) if comparison_data else 0.0
        
        for i, data in enumerate(comparison_data):
            data["rank"] = i + 1
            data["score_difference"] = data.get(metric, 0.0) - best_score
            data["relative_performance"] = (
                data.get(metric, 0.0) / best_score if best_score > 0 else 0.0
            )
        
        return {
            "comparison_metric": metric,
            "model_count": len(model_names),
            "comparison_data": comparison_data,
            "best_model": comparison_data[0]["model_name"] if comparison_data else None,
            "best_score": best_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare models")


@router.get("/realtime-summary")
async def get_realtime_summary(
    hours: int = Query(default=24, ge=1, le=168),  # Max 1 week
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """
    Get real-time processing metrics summary.
    
    Returns processing performance statistics for monitoring.
    Implements Requirement 4.4 for real-time metric updates.
    """
    try:
        summary = await collector.get_realtime_metrics_summary(hours=hours)
        
        # Add additional computed metrics
        overall = summary["overall"]
        if overall["total_operations"] > 0:
            summary["performance_indicators"] = {
                "avg_throughput_per_hour": overall["total_operations"] / hours,
                "error_rate": overall["failed_operations"] / overall["total_operations"],
                "performance_score": min(1.0, 1.0 / max(overall["avg_processing_time"], 0.001)),
                "reliability_score": overall["success_rate"]
            }
        else:
            summary["performance_indicators"] = {
                "avg_throughput_per_hour": 0.0,
                "error_rate": 0.0,
                "performance_score": 0.0,
                "reliability_score": 0.0
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get real-time summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve real-time metrics")


@router.post("/record-processing")
async def record_processing_metrics(
    operation_type: str,
    processing_time: float,
    input_size: int,
    success: bool = True,
    error_message: Optional[str] = None,
    component_metrics: Optional[Dict[str, float]] = None,
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, str]:
    """
    Record real-time processing metrics.
    
    Allows components to report their processing metrics for monitoring.
    Implements Requirement 4.4 for real-time metric collection.
    """
    try:
        metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            operation_type=operation_type,
            processing_time=processing_time,
            input_size=input_size,
            success=success,
            error_message=error_message,
            component_metrics=component_metrics or {}
        )
        
        await collector.record_realtime_metrics(metrics)
        
        return {"status": "recorded", "timestamp": metrics.timestamp.isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to record processing metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to record metrics")


@router.get("/dashboard-data")
async def get_dashboard_data(
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, Any]:
    """
    Get comprehensive dashboard data in a single request.
    
    Returns all data needed for the metrics dashboard.
    Implements Requirements 4.1, 4.2, 4.4 for dashboard display.
    """
    try:
        # Get latest model metrics
        latest_models = await collector.get_latest_model_metrics(limit=5)
        
        # Get real-time summary
        realtime_summary = await collector.get_realtime_metrics_summary(hours=24)
        
        # Get performance history for all models
        performance_history = await collector.get_model_performance_history(days=7)
        
        # Group performance history by model
        models_performance = {}
        for perf in performance_history:
            if perf.model_name not in models_performance:
                models_performance[perf.model_name] = []
            models_performance[perf.model_name].append({
                "date": perf.evaluation_date.isoformat(),
                "f1_score": perf.f1_score,
                "accuracy": perf.accuracy,
                "precision": perf.precision,
                "recall": perf.recall
            })
        
        # Calculate overall statistics
        if latest_models:
            avg_f1 = sum(m.f1_score for m in latest_models) / len(latest_models)
            best_model = max(latest_models, key=lambda m: m.f1_score)
        else:
            avg_f1 = 0.0
            best_model = None
        
        return {
            "overview": {
                "total_models": len(latest_models),
                "avg_f1_score": avg_f1,
                "best_model": {
                    "name": best_model.model_name if best_model else None,
                    "f1_score": best_model.f1_score if best_model else 0.0,
                    "version": best_model.version if best_model else None
                },
                "last_updated": datetime.now().isoformat()
            },
            "latest_models": [
                {
                    "model_name": m.model_name,
                    "version": m.version,
                    "f1_score": m.f1_score,
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "training_date": m.training_date.isoformat()
                }
                for m in latest_models
            ],
            "realtime_metrics": realtime_summary,
            "performance_trends": models_performance
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@router.delete("/cleanup")
async def cleanup_old_metrics(
    days_to_keep: int = Query(default=90, ge=7, le=365),
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> Dict[str, str]:
    """
    Clean up old metrics data.
    
    Removes old metrics data to manage database size.
    """
    try:
        await collector.cleanup_old_data(days_to_keep=days_to_keep)
        return {
            "status": "completed",
            "message": f"Cleaned up metrics data older than {days_to_keep} days"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup metrics data")