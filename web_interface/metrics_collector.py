"""
Metrics data collection system for model performance tracking.

This module implements comprehensive metrics collection for the NLP web interface,
gathering performance data from training runs, model evaluations, and real-time
analysis operations.

Implements Requirements 4.1, 4.2 for metrics collection and historical data management.
"""

import json
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import aiosqlite

from .models import ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_f1: float
    val_f1: float
    val_precision: float
    val_recall: float
    learning_rate: float
    timestamp: datetime
    model_name: str
    version: str
    additional_metrics: Dict[str, Any]


@dataclass
class ModelPerformanceSnapshot:
    """Complete performance snapshot for a model."""
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    macro_f1: float
    f1_depressed: float
    f1_non_depressed: float
    precision_depressed: float
    precision_non_depressed: float
    recall_depressed: float
    recall_non_depressed: float
    confusion_matrix: List[List[int]]
    training_date: datetime
    evaluation_date: datetime
    dataset_size: int
    test_samples: int
    training_epochs: int
    best_epoch: int
    training_time_minutes: float
    evaluation_data: Dict[str, Any]


@dataclass
class RealTimeMetrics:
    """Real-time processing metrics."""
    timestamp: datetime
    operation_type: str  # 'text_analysis', 'audio_processing', 'batch_analysis'
    processing_time: float
    input_size: int  # characters for text, samples for audio
    success: bool
    error_message: Optional[str]
    component_metrics: Dict[str, float]  # per-component timing


class MetricsCollector:
    """Centralized metrics collection and storage system."""
    
    def __init__(self, db_path: str = "web_interface/metrics.db"):
        """Initialize metrics collector with SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        
    async def initialize(self):
        """Initialize database schema."""
        if self._initialized:
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            # Training metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL NOT NULL,
                    val_loss REAL NOT NULL,
                    train_f1 REAL NOT NULL,
                    val_f1 REAL NOT NULL,
                    val_precision REAL NOT NULL,
                    val_recall REAL NOT NULL,
                    learning_rate REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    additional_metrics TEXT
                )
            """)
            
            # Create indexes separately
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_metrics_model 
                ON training_metrics(model_name, version, timestamp)
            """)
            
            # Model performance snapshots table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    macro_f1 REAL NOT NULL,
                    f1_depressed REAL NOT NULL,
                    f1_non_depressed REAL NOT NULL,
                    precision_depressed REAL NOT NULL,
                    precision_non_depressed REAL NOT NULL,
                    recall_depressed REAL NOT NULL,
                    recall_non_depressed REAL NOT NULL,
                    confusion_matrix TEXT NOT NULL,
                    training_date DATETIME NOT NULL,
                    evaluation_date DATETIME NOT NULL,
                    dataset_size INTEGER NOT NULL,
                    test_samples INTEGER NOT NULL,
                    training_epochs INTEGER NOT NULL,
                    best_epoch INTEGER NOT NULL,
                    training_time_minutes REAL NOT NULL,
                    evaluation_data TEXT
                )
            """)
            
            # Create indexes separately
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_performance_model 
                ON model_performance(model_name, version, evaluation_date)
            """)
            
            # Real-time processing metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS realtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    operation_type TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    input_size INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    component_metrics TEXT
                )
            """)
            
            # Create indexes separately
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_realtime_metrics_timestamp 
                ON realtime_metrics(timestamp, operation_type)
            """)
            
            # Model comparison cache table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_id TEXT NOT NULL UNIQUE,
                    model_names TEXT NOT NULL,
                    comparison_data TEXT NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """)
            
            # Create indexes separately
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_comparisons_id 
                ON model_comparisons(comparison_id, created_at)
            """)
            
            await db.commit()
            
        self._initialized = True
        logger.info(f"Metrics collector initialized with database: {self.db_path}")
    
    async def record_training_metrics(self, metrics: TrainingMetrics):
        """Record training metrics for a single epoch."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO training_metrics (
                    model_name, version, epoch, train_loss, val_loss,
                    train_f1, val_f1, val_precision, val_recall,
                    learning_rate, timestamp, additional_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.model_name, metrics.version, metrics.epoch,
                metrics.train_loss, metrics.val_loss, metrics.train_f1,
                metrics.val_f1, metrics.val_precision, metrics.val_recall,
                metrics.learning_rate, metrics.timestamp,
                json.dumps(metrics.additional_metrics)
            ))
            await db.commit()
    
    async def record_model_performance(self, performance: ModelPerformanceSnapshot):
        """Record complete model performance snapshot."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO model_performance (
                    model_name, version, accuracy, precision, recall, f1_score,
                    macro_f1, f1_depressed, f1_non_depressed,
                    precision_depressed, precision_non_depressed,
                    recall_depressed, recall_non_depressed,
                    confusion_matrix, training_date, evaluation_date,
                    dataset_size, test_samples, training_epochs, best_epoch,
                    training_time_minutes, evaluation_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance.model_name, performance.version,
                performance.accuracy, performance.precision, performance.recall,
                performance.f1_score, performance.macro_f1,
                performance.f1_depressed, performance.f1_non_depressed,
                performance.precision_depressed, performance.precision_non_depressed,
                performance.recall_depressed, performance.recall_non_depressed,
                json.dumps(performance.confusion_matrix),
                performance.training_date, performance.evaluation_date,
                performance.dataset_size, performance.test_samples,
                performance.training_epochs, performance.best_epoch,
                performance.training_time_minutes,
                json.dumps(performance.evaluation_data)
            ))
            await db.commit()
    
    async def record_realtime_metrics(self, metrics: RealTimeMetrics):
        """Record real-time processing metrics."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO realtime_metrics (
                    timestamp, operation_type, processing_time, input_size,
                    success, error_message, component_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.operation_type, metrics.processing_time,
                metrics.input_size, metrics.success, metrics.error_message,
                json.dumps(metrics.component_metrics)
            ))
            await db.commit()
    
    async def get_training_history(self, model_name: str, version: Optional[str] = None,
                                 limit: int = 1000) -> List[TrainingMetrics]:
        """Get training history for a model."""
        await self.initialize()
        
        query = """
            SELECT * FROM training_metrics 
            WHERE model_name = ?
        """
        params = [model_name]
        
        if version:
            query += " AND version = ?"
            params.append(version)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    TrainingMetrics(
                        epoch=row['epoch'],
                        train_loss=row['train_loss'],
                        val_loss=row['val_loss'],
                        train_f1=row['train_f1'],
                        val_f1=row['val_f1'],
                        val_precision=row['val_precision'],
                        val_recall=row['val_recall'],
                        learning_rate=row['learning_rate'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        model_name=row['model_name'],
                        version=row['version'],
                        additional_metrics=json.loads(row['additional_metrics'] or '{}')
                    )
                    for row in rows
                ]
    
    async def get_model_performance_history(self, model_name: Optional[str] = None,
                                          days: int = 30) -> List[ModelPerformanceSnapshot]:
        """Get model performance history."""
        await self.initialize()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT * FROM model_performance 
            WHERE evaluation_date >= ?
        """
        params = [cutoff_date]
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
            
        query += " ORDER BY evaluation_date DESC"
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    ModelPerformanceSnapshot(
                        model_name=row['model_name'],
                        version=row['version'],
                        accuracy=row['accuracy'],
                        precision=row['precision'],
                        recall=row['recall'],
                        f1_score=row['f1_score'],
                        macro_f1=row['macro_f1'],
                        f1_depressed=row['f1_depressed'],
                        f1_non_depressed=row['f1_non_depressed'],
                        precision_depressed=row['precision_depressed'],
                        precision_non_depressed=row['precision_non_depressed'],
                        recall_depressed=row['recall_depressed'],
                        recall_non_depressed=row['recall_non_depressed'],
                        confusion_matrix=json.loads(row['confusion_matrix']),
                        training_date=datetime.fromisoformat(row['training_date']),
                        evaluation_date=datetime.fromisoformat(row['evaluation_date']),
                        dataset_size=row['dataset_size'],
                        test_samples=row['test_samples'],
                        training_epochs=row['training_epochs'],
                        best_epoch=row['best_epoch'],
                        training_time_minutes=row['training_time_minutes'],
                        evaluation_data=json.loads(row['evaluation_data'] or '{}')
                    )
                    for row in rows
                ]
    
    async def get_realtime_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of real-time processing metrics."""
        await self.initialize()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Overall statistics
            async with db.execute("""
                SELECT 
                    COUNT(*) as total_operations,
                    AVG(processing_time) as avg_processing_time,
                    MIN(processing_time) as min_processing_time,
                    MAX(processing_time) as max_processing_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_operations,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_operations
                FROM realtime_metrics 
                WHERE timestamp >= ?
            """, (cutoff_time,)) as cursor:
                overall_stats = await cursor.fetchone()
            
            # Per-operation type statistics
            async with db.execute("""
                SELECT 
                    operation_type,
                    COUNT(*) as count,
                    AVG(processing_time) as avg_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                FROM realtime_metrics 
                WHERE timestamp >= ?
                GROUP BY operation_type
            """, (cutoff_time,)) as cursor:
                operation_stats = await cursor.fetchall()
            
            # Hourly breakdown
            async with db.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    COUNT(*) as operations,
                    AVG(processing_time) as avg_time
                FROM realtime_metrics 
                WHERE timestamp >= ?
                GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
                ORDER BY hour
            """, (cutoff_time,)) as cursor:
                hourly_stats = await cursor.fetchall()
        
        return {
            "time_period_hours": hours,
            "overall": {
                "total_operations": overall_stats[0] or 0,
                "avg_processing_time": overall_stats[1] or 0.0,
                "min_processing_time": overall_stats[2] or 0.0,
                "max_processing_time": overall_stats[3] or 0.0,
                "successful_operations": overall_stats[4] or 0,
                "failed_operations": overall_stats[5] or 0,
                "success_rate": (overall_stats[4] or 0) / max(overall_stats[0] or 1, 1)
            },
            "by_operation_type": [
                {
                    "operation_type": row[0],
                    "count": row[1],
                    "avg_processing_time": row[2],
                    "success_count": row[3],
                    "success_rate": row[3] / max(row[1], 1)
                }
                for row in operation_stats
            ],
            "hourly_breakdown": [
                {
                    "hour": row[0],
                    "operations": row[1],
                    "avg_processing_time": row[2]
                }
                for row in hourly_stats
            ]
        }
    
    async def get_latest_model_metrics(self, limit: int = 10) -> List[ModelMetrics]:
        """Get latest model performance metrics for API."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM model_performance 
                ORDER BY evaluation_date DESC 
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    ModelMetrics(
                        model_name=row['model_name'],
                        version=row['version'],
                        accuracy=row['accuracy'],
                        precision=row['precision'],
                        recall=row['recall'],
                        f1_score=row['f1_score'],
                        training_date=datetime.fromisoformat(row['training_date']),
                        evaluation_data={
                            "macro_f1": row['macro_f1'],
                            "f1_depressed": row['f1_depressed'],
                            "f1_non_depressed": row['f1_non_depressed'],
                            "precision_depressed": row['precision_depressed'],
                            "precision_non_depressed": row['precision_non_depressed'],
                            "recall_depressed": row['recall_depressed'],
                            "recall_non_depressed": row['recall_non_depressed'],
                            "confusion_matrix": json.loads(row['confusion_matrix']),
                            "dataset_size": row['dataset_size'],
                            "test_samples": row['test_samples'],
                            "training_epochs": row['training_epochs'],
                            "best_epoch": row['best_epoch'],
                            "training_time_minutes": row['training_time_minutes'],
                            **json.loads(row['evaluation_data'] or '{}')
                        }
                    )
                    for row in rows
                ]
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old metrics data to manage database size."""
        await self.initialize()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Clean up old real-time metrics (keep more recent training/performance data)
            await db.execute("""
                DELETE FROM realtime_metrics 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            # Clean up old training metrics (but keep at least the latest run for each model)
            await db.execute("""
                DELETE FROM training_metrics 
                WHERE timestamp < ? 
                AND id NOT IN (
                    SELECT id FROM training_metrics t1
                    WHERE t1.model_name = training_metrics.model_name
                    AND t1.version = training_metrics.version
                    ORDER BY t1.timestamp DESC
                    LIMIT 100
                )
            """, (cutoff_date,))
            
            await db.commit()
            
        logger.info(f"Cleaned up metrics data older than {days_to_keep} days")


# Global metrics collector instance
_metrics_collector = None


async def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        await _metrics_collector.initialize()
    return _metrics_collector


def create_training_metrics_from_multimodal(
    multimodal_metrics: Dict[str, Any],
    model_name: str,
    version: str
) -> TrainingMetrics:
    """Convert multimodal trainer metrics to TrainingMetrics format."""
    return TrainingMetrics(
        epoch=multimodal_metrics.get('epoch', 0),
        train_loss=multimodal_metrics.get('train_loss', 0.0),
        val_loss=multimodal_metrics.get('val_loss', 0.0),
        train_f1=multimodal_metrics.get('train_f1', 0.0),
        val_f1=multimodal_metrics.get('val_f1', 0.0),
        val_precision=multimodal_metrics.get('val_precision', 0.0),
        val_recall=multimodal_metrics.get('val_recall', 0.0),
        learning_rate=multimodal_metrics.get('learning_rate', 0.0),
        timestamp=datetime.now(),
        model_name=model_name,
        version=version,
        additional_metrics={
            k: v for k, v in multimodal_metrics.items()
            if k not in ['epoch', 'train_loss', 'val_loss', 'train_f1', 
                        'val_f1', 'val_precision', 'val_recall', 'learning_rate']
        }
    )


def create_performance_snapshot_from_evaluation(
    eval_result: Dict[str, Any],
    model_name: str,
    version: str,
    training_info: Dict[str, Any]
) -> ModelPerformanceSnapshot:
    """Convert evaluation result to ModelPerformanceSnapshot."""
    return ModelPerformanceSnapshot(
        model_name=model_name,
        version=version,
        accuracy=eval_result.get('accuracy', eval_result.get('macro_f1', 0.0)),
        precision=eval_result.get('macro_precision', 0.0),
        recall=eval_result.get('macro_recall', 0.0),
        f1_score=eval_result.get('macro_f1', 0.0),
        macro_f1=eval_result.get('macro_f1', 0.0),
        f1_depressed=eval_result.get('f1_depressed', 0.0),
        f1_non_depressed=eval_result.get('f1_non_depressed', 0.0),
        precision_depressed=eval_result.get('precision_depressed', 0.0),
        precision_non_depressed=eval_result.get('precision_non_depressed', 0.0),
        recall_depressed=eval_result.get('recall_depressed', 0.0),
        recall_non_depressed=eval_result.get('recall_non_depressed', 0.0),
        confusion_matrix=eval_result.get('confusion_matrix', [[0, 0], [0, 0]]),
        training_date=training_info.get('training_date', datetime.now()),
        evaluation_date=datetime.now(),
        dataset_size=training_info.get('dataset_size', 0),
        test_samples=training_info.get('test_samples', 0),
        training_epochs=training_info.get('training_epochs', 0),
        best_epoch=training_info.get('best_epoch', 0),
        training_time_minutes=training_info.get('training_time_minutes', 0.0),
        evaluation_data=eval_result
    )


if __name__ == "__main__":
    # Test metrics collector
    import asyncio
    
    async def test_metrics_collector():
        print("Testing MetricsCollector...")
        
        collector = MetricsCollector("test_metrics.db")
        await collector.initialize()
        
        # Test training metrics
        training_metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_f1=0.8,
            val_f1=0.85,
            val_precision=0.82,
            val_recall=0.88,
            learning_rate=0.001,
            timestamp=datetime.now(),
            model_name="test_model",
            version="1.0.0",
            additional_metrics={"batch_size": 32}
        )
        
        await collector.record_training_metrics(training_metrics)
        print("✓ Training metrics recorded")
        
        # Test performance snapshot
        performance = ModelPerformanceSnapshot(
            model_name="test_model",
            version="1.0.0",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            macro_f1=0.85,
            f1_depressed=0.83,
            f1_non_depressed=0.87,
            precision_depressed=0.80,
            precision_non_depressed=0.84,
            recall_depressed=0.86,
            recall_non_depressed=0.90,
            confusion_matrix=[[450, 50], [40, 460]],
            training_date=datetime.now(),
            evaluation_date=datetime.now(),
            dataset_size=1000,
            test_samples=200,
            training_epochs=10,
            best_epoch=8,
            training_time_minutes=120.5,
            evaluation_data={"additional_info": "test"}
        )
        
        await collector.record_model_performance(performance)
        print("✓ Performance snapshot recorded")
        
        # Test real-time metrics
        realtime_metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            operation_type="text_analysis",
            processing_time=0.15,
            input_size=100,
            success=True,
            error_message=None,
            component_metrics={"pos_tagging": 0.05, "ner": 0.03, "parsing": 0.07}
        )
        
        await collector.record_realtime_metrics(realtime_metrics)
        print("✓ Real-time metrics recorded")
        
        # Test retrieval
        history = await collector.get_training_history("test_model")
        print(f"✓ Retrieved {len(history)} training records")
        
        performance_history = await collector.get_model_performance_history()
        print(f"✓ Retrieved {len(performance_history)} performance records")
        
        summary = await collector.get_realtime_metrics_summary(hours=1)
        print(f"✓ Real-time summary: {summary['overall']['total_operations']} operations")
        
        latest_metrics = await collector.get_latest_model_metrics()
        print(f"✓ Retrieved {len(latest_metrics)} latest model metrics")
        
        print("All tests passed!")
    
    asyncio.run(test_metrics_collector())