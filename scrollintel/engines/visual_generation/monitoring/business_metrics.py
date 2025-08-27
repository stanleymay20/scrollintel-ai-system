"""
Business Metrics Tracking for Visual Generation System
Tracks generation volume, costs, quality scores, and user engagement
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import aiosqlite
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    THUMBNAIL = "thumbnail"

class GenerationStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationRecord:
    generation_id: str
    user_id: str
    model_name: str
    content_type: ContentType
    status: GenerationStatus
    prompt: str
    parameters: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime]
    generation_time_seconds: Optional[float]
    cost_usd: float
    quality_score: Optional[float]
    file_size_bytes: Optional[int]
    error_message: Optional[str]

@dataclass
class UsageMetrics:
    timestamp: datetime
    total_generations: int
    successful_generations: int
    failed_generations: int
    total_cost_usd: float
    average_generation_time: float
    average_quality_score: float
    unique_users: int
    content_type_breakdown: Dict[str, int]
    model_usage_breakdown: Dict[str, int]

@dataclass
class UserEngagementMetrics:
    user_id: str
    total_generations: int
    successful_generations: int
    total_cost_usd: float
    average_quality_score: float
    favorite_model: str
    favorite_content_type: str
    first_generation: datetime
    last_generation: datetime
    generation_frequency_per_day: float

class BusinessMetricsCollector:
    """Collects and analyzes business metrics for visual generation"""
    
    def __init__(self, db_path: str = "visual_generation_metrics.db"):
        self.db_path = db_path
        self.generation_records: Dict[str, GenerationRecord] = {}
        self.real_time_metrics = deque(maxlen=1000)  # Keep last 1000 metrics
        
        # Initialize database
        asyncio.create_task(self._initialize_database())
    
    async def _initialize_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS generation_records (
                        generation_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        prompt TEXT,
                        parameters TEXT,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        generation_time_seconds REAL,
                        cost_usd REAL NOT NULL,
                        quality_score REAL,
                        file_size_bytes INTEGER,
                        error_message TEXT
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS daily_metrics (
                        date TEXT PRIMARY KEY,
                        total_generations INTEGER,
                        successful_generations INTEGER,
                        failed_generations INTEGER,
                        total_cost_usd REAL,
                        average_generation_time REAL,
                        average_quality_score REAL,
                        unique_users INTEGER,
                        content_type_breakdown TEXT,
                        model_usage_breakdown TEXT
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS user_metrics (
                        user_id TEXT,
                        date TEXT,
                        generations_count INTEGER,
                        total_cost_usd REAL,
                        average_quality_score REAL,
                        PRIMARY KEY (user_id, date)
                    )
                """)
                
                await db.commit()
                
            logger.info("Business metrics database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {str(e)}")
            raise
    
    async def record_generation_start(self, generation_id: str, user_id: str, 
                                    model_name: str, content_type: ContentType,
                                    prompt: str, parameters: Dict[str, Any]) -> GenerationRecord:
        """Record the start of a generation request"""
        try:
            record = GenerationRecord(
                generation_id=generation_id,
                user_id=user_id,
                model_name=model_name,
                content_type=content_type,
                status=GenerationStatus.PENDING,
                prompt=prompt,
                parameters=parameters,
                started_at=datetime.utcnow(),
                completed_at=None,
                generation_time_seconds=None,
                cost_usd=0.0,  # Will be calculated on completion
                quality_score=None,
                file_size_bytes=None,
                error_message=None
            )
            
            self.generation_records[generation_id] = record
            
            # Persist to database
            await self._persist_generation_record(record)
            
            logger.info(f"Generation started: {generation_id}")
            return record
            
        except Exception as e:
            logger.error(f"Failed to record generation start: {str(e)}")
            raise
    
    async def record_generation_completion(self, generation_id: str, 
                                         status: GenerationStatus,
                                         cost_usd: float,
                                         quality_score: Optional[float] = None,
                                         file_size_bytes: Optional[int] = None,
                                         error_message: Optional[str] = None):
        """Record the completion of a generation request"""
        try:
            if generation_id not in self.generation_records:
                logger.error(f"Generation record not found: {generation_id}")
                return
            
            record = self.generation_records[generation_id]
            record.status = status
            record.completed_at = datetime.utcnow()
            record.generation_time_seconds = (
                record.completed_at - record.started_at
            ).total_seconds()
            record.cost_usd = cost_usd
            record.quality_score = quality_score
            record.file_size_bytes = file_size_bytes
            record.error_message = error_message
            
            # Update database
            await self._persist_generation_record(record)
            
            # Update real-time metrics
            await self._update_real_time_metrics(record)
            
            logger.info(f"Generation completed: {generation_id}, Status: {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to record generation completion: {str(e)}")
    
    async def _persist_generation_record(self, record: GenerationRecord):
        """Persist generation record to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO generation_records 
                    (generation_id, user_id, model_name, content_type, status, prompt, 
                     parameters, started_at, completed_at, generation_time_seconds, 
                     cost_usd, quality_score, file_size_bytes, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.generation_id,
                    record.user_id,
                    record.model_name,
                    record.content_type.value,
                    record.status.value,
                    record.prompt,
                    json.dumps(record.parameters),
                    record.started_at.isoformat(),
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.generation_time_seconds,
                    record.cost_usd,
                    record.quality_score,
                    record.file_size_bytes,
                    record.error_message
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to persist generation record: {str(e)}")
    
    async def _update_real_time_metrics(self, record: GenerationRecord):
        """Update real-time metrics with completed generation"""
        try:
            # Add to real-time metrics queue
            self.real_time_metrics.append({
                'timestamp': record.completed_at,
                'status': record.status.value,
                'model': record.model_name,
                'content_type': record.content_type.value,
                'generation_time': record.generation_time_seconds,
                'cost': record.cost_usd,
                'quality_score': record.quality_score,
                'user_id': record.user_id
            })
            
        except Exception as e:
            logger.error(f"Failed to update real-time metrics: {str(e)}")
    
    async def get_usage_metrics(self, start_date: datetime, end_date: datetime) -> UsageMetrics:
        """Get usage metrics for specified date range"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_generations,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_generations,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_generations,
                        SUM(cost_usd) as total_cost_usd,
                        AVG(CASE WHEN generation_time_seconds IS NOT NULL THEN generation_time_seconds END) as avg_generation_time,
                        AVG(CASE WHEN quality_score IS NOT NULL THEN quality_score END) as avg_quality_score,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                
                row = await cursor.fetchone()
                
                # Get content type breakdown
                cursor = await db.execute("""
                    SELECT content_type, COUNT(*) 
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ?
                    GROUP BY content_type
                """, (start_date.isoformat(), end_date.isoformat()))
                
                content_type_breakdown = dict(await cursor.fetchall())
                
                # Get model usage breakdown
                cursor = await db.execute("""
                    SELECT model_name, COUNT(*) 
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ?
                    GROUP BY model_name
                """, (start_date.isoformat(), end_date.isoformat()))
                
                model_usage_breakdown = dict(await cursor.fetchall())
                
                return UsageMetrics(
                    timestamp=datetime.utcnow(),
                    total_generations=row[0] or 0,
                    successful_generations=row[1] or 0,
                    failed_generations=row[2] or 0,
                    total_cost_usd=row[3] or 0.0,
                    average_generation_time=row[4] or 0.0,
                    average_quality_score=row[5] or 0.0,
                    unique_users=row[6] or 0,
                    content_type_breakdown=content_type_breakdown,
                    model_usage_breakdown=model_usage_breakdown
                )
                
        except Exception as e:
            logger.error(f"Failed to get usage metrics: {str(e)}")
            raise
    
    async def get_user_engagement_metrics(self, user_id: str) -> Optional[UserEngagementMetrics]:
        """Get engagement metrics for specific user"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_generations,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_generations,
                        SUM(cost_usd) as total_cost_usd,
                        AVG(CASE WHEN quality_score IS NOT NULL THEN quality_score END) as avg_quality_score,
                        MIN(started_at) as first_generation,
                        MAX(started_at) as last_generation
                    FROM generation_records 
                    WHERE user_id = ?
                """, (user_id,))
                
                row = await cursor.fetchone()
                
                if not row or row[0] == 0:
                    return None
                
                # Get favorite model
                cursor = await db.execute("""
                    SELECT model_name, COUNT(*) as usage_count
                    FROM generation_records 
                    WHERE user_id = ?
                    GROUP BY model_name
                    ORDER BY usage_count DESC
                    LIMIT 1
                """, (user_id,))
                
                favorite_model_row = await cursor.fetchone()
                favorite_model = favorite_model_row[0] if favorite_model_row else "unknown"
                
                # Get favorite content type
                cursor = await db.execute("""
                    SELECT content_type, COUNT(*) as usage_count
                    FROM generation_records 
                    WHERE user_id = ?
                    GROUP BY content_type
                    ORDER BY usage_count DESC
                    LIMIT 1
                """, (user_id,))
                
                favorite_content_type_row = await cursor.fetchone()
                favorite_content_type = favorite_content_type_row[0] if favorite_content_type_row else "unknown"
                
                # Calculate generation frequency
                first_gen = datetime.fromisoformat(row[4])
                last_gen = datetime.fromisoformat(row[5])
                days_active = max(1, (last_gen - first_gen).days)
                generation_frequency = row[0] / days_active
                
                return UserEngagementMetrics(
                    user_id=user_id,
                    total_generations=row[0],
                    successful_generations=row[1] or 0,
                    total_cost_usd=row[2] or 0.0,
                    average_quality_score=row[3] or 0.0,
                    favorite_model=favorite_model,
                    favorite_content_type=favorite_content_type,
                    first_generation=first_gen,
                    last_generation=last_gen,
                    generation_frequency_per_day=generation_frequency
                )
                
        except Exception as e:
            logger.error(f"Failed to get user engagement metrics: {str(e)}")
            return None
    
    async def get_real_time_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get real-time metrics for the last N minutes"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            
            recent_metrics = [
                m for m in self.real_time_metrics
                if m['timestamp'] and m['timestamp'] > cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    "time_period_minutes": minutes,
                    "total_generations": 0,
                    "successful_generations": 0,
                    "failed_generations": 0,
                    "average_generation_time": 0.0,
                    "average_cost": 0.0,
                    "average_quality_score": 0.0,
                    "generations_per_minute": 0.0
                }
            
            successful = [m for m in recent_metrics if m['status'] == 'completed']
            failed = [m for m in recent_metrics if m['status'] == 'failed']
            
            # Calculate averages
            avg_generation_time = 0.0
            avg_cost = 0.0
            avg_quality_score = 0.0
            
            if successful:
                generation_times = [m['generation_time'] for m in successful if m['generation_time']]
                costs = [m['cost'] for m in successful if m['cost']]
                quality_scores = [m['quality_score'] for m in successful if m['quality_score']]
                
                avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0.0
                avg_cost = sum(costs) / len(costs) if costs else 0.0
                avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                "time_period_minutes": minutes,
                "total_generations": len(recent_metrics),
                "successful_generations": len(successful),
                "failed_generations": len(failed),
                "average_generation_time": avg_generation_time,
                "average_cost": avg_cost,
                "average_quality_score": avg_quality_score,
                "generations_per_minute": len(recent_metrics) / minutes,
                "success_rate": len(successful) / len(recent_metrics) if recent_metrics else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {str(e)}")
            return {}
    
    async def get_cost_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total costs by model
                cursor = await db.execute("""
                    SELECT model_name, SUM(cost_usd) as total_cost, COUNT(*) as generation_count
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? AND status = 'completed'
                    GROUP BY model_name
                    ORDER BY total_cost DESC
                """, (start_date.isoformat(), end_date.isoformat()))
                
                model_costs = await cursor.fetchall()
                
                # Total costs by content type
                cursor = await db.execute("""
                    SELECT content_type, SUM(cost_usd) as total_cost, COUNT(*) as generation_count
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? AND status = 'completed'
                    GROUP BY content_type
                    ORDER BY total_cost DESC
                """, (start_date.isoformat(), end_date.isoformat()))
                
                content_type_costs = await cursor.fetchall()
                
                # Daily cost trends
                cursor = await db.execute("""
                    SELECT DATE(started_at) as date, SUM(cost_usd) as daily_cost
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? AND status = 'completed'
                    GROUP BY DATE(started_at)
                    ORDER BY date
                """, (start_date.isoformat(), end_date.isoformat()))
                
                daily_costs = await cursor.fetchall()
                
                # Top spending users
                cursor = await db.execute("""
                    SELECT user_id, SUM(cost_usd) as total_cost, COUNT(*) as generation_count
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? AND status = 'completed'
                    GROUP BY user_id
                    ORDER BY total_cost DESC
                    LIMIT 10
                """, (start_date.isoformat(), end_date.isoformat()))
                
                top_users = await cursor.fetchall()
                
                return {
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "model_costs": [
                        {"model": row[0], "total_cost": row[1], "generation_count": row[2]}
                        for row in model_costs
                    ],
                    "content_type_costs": [
                        {"content_type": row[0], "total_cost": row[1], "generation_count": row[2]}
                        for row in content_type_costs
                    ],
                    "daily_costs": [
                        {"date": row[0], "cost": row[1]}
                        for row in daily_costs
                    ],
                    "top_users": [
                        {"user_id": row[0], "total_cost": row[1], "generation_count": row[2]}
                        for row in top_users
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get cost analysis: {str(e)}")
            return {}
    
    async def get_quality_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get quality score analysis"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Quality scores by model
                cursor = await db.execute("""
                    SELECT 
                        model_name,
                        AVG(quality_score) as avg_quality,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality,
                        COUNT(*) as generation_count
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? 
                    AND status = 'completed' 
                    AND quality_score IS NOT NULL
                    GROUP BY model_name
                """, (start_date.isoformat(), end_date.isoformat()))
                
                model_quality = await cursor.fetchall()
                
                # Quality scores by content type
                cursor = await db.execute("""
                    SELECT 
                        content_type,
                        AVG(quality_score) as avg_quality,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality,
                        COUNT(*) as generation_count
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? 
                    AND status = 'completed' 
                    AND quality_score IS NOT NULL
                    GROUP BY content_type
                """, (start_date.isoformat(), end_date.isoformat()))
                
                content_type_quality = await cursor.fetchall()
                
                # Quality trends over time
                cursor = await db.execute("""
                    SELECT 
                        DATE(started_at) as date,
                        AVG(quality_score) as avg_quality
                    FROM generation_records 
                    WHERE started_at BETWEEN ? AND ? 
                    AND status = 'completed' 
                    AND quality_score IS NOT NULL
                    GROUP BY DATE(started_at)
                    ORDER BY date
                """, (start_date.isoformat(), end_date.isoformat()))
                
                quality_trends = await cursor.fetchall()
                
                return {
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "model_quality": [
                        {
                            "model": row[0],
                            "avg_quality": row[1],
                            "min_quality": row[2],
                            "max_quality": row[3],
                            "generation_count": row[4]
                        }
                        for row in model_quality
                    ],
                    "content_type_quality": [
                        {
                            "content_type": row[0],
                            "avg_quality": row[1],
                            "min_quality": row[2],
                            "max_quality": row[3],
                            "generation_count": row[4]
                        }
                        for row in content_type_quality
                    ],
                    "quality_trends": [
                        {"date": row[0], "avg_quality": row[1]}
                        for row in quality_trends
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get quality analysis: {str(e)}")
            return {}

# Global metrics collector
business_metrics = BusinessMetricsCollector()

async def initialize_business_metrics():
    """Initialize business metrics collection"""
    try:
        await business_metrics._initialize_database()
        logger.info("Business metrics initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize business metrics: {str(e)}")
        return False