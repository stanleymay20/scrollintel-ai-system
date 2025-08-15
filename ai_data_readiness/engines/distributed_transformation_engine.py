"""
Distributed Transformation Engine for AI Data Readiness Platform

This module provides distributed data transformation capabilities with
intelligent pipeline orchestration and resource optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import Future
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import time
import json

from .distributed_processor import DistributedDataProcessor, ProcessingTask, ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformationStep:
    """Represents a single transformation step in a pipeline"""
    name: str
    function: Callable[[pd.DataFrame], pd.DataFrame]
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    parallel: bool = True
    priority: int = 1


@dataclass
class TransformationPipeline:
    """Represents a complete transformation pipeline"""
    name: str
    steps: List[TransformationStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class TransformationRegistry:
    """Registry for transformation functions and pipelines"""
    
    def __init__(self):
        self._transformations: Dict[str, Callable] = {}
        self._pipelines: Dict[str, TransformationPipeline] = {}
    
    def register_transformation(self, name: str, func: Callable):
        """Register a transformation function"""
        self._transformations[name] = func
        logger.info(f"Registered transformation: {name}")
    
    def register_pipeline(self, pipeline: TransformationPipeline):
        """Register a transformation pipeline"""
        self._pipelines[pipeline.name] = pipeline
        logger.info(f"Registered pipeline: {pipeline.name}")
    
    def get_transformation(self, name: str) -> Optional[Callable]:
        """Get a registered transformation function"""
        return self._transformations.get(name)
    
    def get_pipeline(self, name: str) -> Optional[TransformationPipeline]:
        """Get a registered pipeline"""
        return self._pipelines.get(name)
    
    def list_transformations(self) -> List[str]:
        """List all registered transformations"""
        return list(self._transformations.keys())
    
    def list_pipelines(self) -> List[str]:
        """List all registered pipelines"""
        return list(self._pipelines.keys())


class DistributedTransformationEngine:
    """
    Distributed transformation engine with intelligent pipeline orchestration
    
    Features:
    - Pipeline dependency resolution
    - Parallel execution of independent steps
    - Resource-aware scheduling
    - Fault tolerance and retry logic
    - Performance optimization
    """
    
    def __init__(self, 
                 processor: Optional[DistributedDataProcessor] = None,
                 config: Optional[ProcessingConfig] = None):
        self.processor = processor or DistributedDataProcessor(config)
        self.registry = TransformationRegistry()
        self._register_builtin_transformations()
        
        # Pipeline execution state
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        logger.info("DistributedTransformationEngine initialized")
    
    def _register_builtin_transformations(self):
        """Register built-in transformation functions"""
        
        def normalize_numeric(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
            """Normalize numeric columns to 0-1 range"""
            result = df.copy()
            numeric_cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                if col in result.columns:
                    min_val = result[col].min()
                    max_val = result[col].max()
                    if max_val > min_val:
                        result[col] = (result[col] - min_val) / (max_val - min_val)
            
            return result
        
        def standardize_numeric(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
            """Standardize numeric columns (z-score normalization)"""
            result = df.copy()
            numeric_cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                if col in result.columns:
                    mean_val = result[col].mean()
                    std_val = result[col].std()
                    if std_val > 0:
                        result[col] = (result[col] - mean_val) / std_val
            
            return result
        
        def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
            """Handle missing values with various strategies"""
            result = df.copy()
            
            if strategy == 'mean':
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
                
                categorical_cols = result.select_dtypes(include=['object']).columns
                result[categorical_cols] = result[categorical_cols].fillna(result[categorical_cols].mode().iloc[0] if not result[categorical_cols].mode().empty else 'Unknown')
            
            elif strategy == 'median':
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
            
            elif strategy == 'drop':
                result = result.dropna()
            
            return result
        
        # Register built-in transformations
        self.registry.register_transformation('normalize_numeric', normalize_numeric)
        self.registry.register_transformation('standardize_numeric', standardize_numeric)
        self.registry.register_transformation('handle_missing_values', handle_missing_values)
    
    def create_pipeline(self, name: str, steps: List[Dict[str, Any]]) -> TransformationPipeline:
        """
        Create a transformation pipeline from step definitions
        
        Args:
            name: Pipeline name
            steps: List of step definitions with 'name', 'function', 'parameters', etc.
        
        Returns:
            TransformationPipeline object
        """
        pipeline_steps = []
        
        for step_def in steps:
            func_name = step_def.get('function')
            func = self.registry.get_transformation(func_name)
            
            if not func:
                raise ValueError(f"Transformation function '{func_name}' not found in registry")
            
            step = TransformationStep(
                name=step_def['name'],
                function=func,
                parameters=step_def.get('parameters', {}),
                dependencies=step_def.get('dependencies', []),
                parallel=step_def.get('parallel', True),
                priority=step_def.get('priority', 1)
            )
            pipeline_steps.append(step)
        
        pipeline = TransformationPipeline(name=name, steps=pipeline_steps)
        self.registry.register_pipeline(pipeline)
        
        return pipeline