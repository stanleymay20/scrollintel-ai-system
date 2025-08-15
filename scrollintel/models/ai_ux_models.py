"""
AI UX Optimization Data Models

This module defines the data models for AI-powered user experience optimization,
including failure predictions, user behavior analysis, personalized degradation
strategies, and interface optimizations.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

Base = declarative_base()

class FailurePredictionModel(Base):
    """Model for storing failure predictions"""
    __tablename__ = 'failure_predictions'
    
    id = Column(Integer, primary_key=True)
    prediction_type = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    time_to_failure = Column(Integer)  # minutes
    contributing_factors = Column(JSON)
    recommended_actions = Column(JSON)
    system_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    actual_failure_occurred = Column(Boolean)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'prediction_type': self.prediction_type,
            'probability': self.probability,
            'confidence': self.confidence,
            'time_to_failure': self.time_to_failure,
            'contributing_factors': self.contributing_factors,
            'recommended_actions': self.recommended_actions,
            'system_metrics': self.system_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'actual_failure_occurred': self.actual_failure_occurred
        }

class UserBehaviorAnalysisModel(Base):
    """Model for storing user behavior analysis"""
    __tablename__ = 'user_behavior_analysis'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    behavior_pattern = Column(String(50), nullable=False)
    engagement_score = Column(Float, nullable=False)
    frustration_indicators = Column(JSON)
    preferred_features = Column(JSON)
    usage_patterns = Column(JSON)
    assistance_needs = Column(JSON)
    interaction_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'behavior_pattern': self.behavior_pattern,
            'engagement_score': self.engagement_score,
            'frustration_indicators': self.frustration_indicators,
            'preferred_features': self.preferred_features,
            'usage_patterns': self.usage_patterns,
            'assistance_needs': self.assistance_needs,
            'interaction_data': self.interaction_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class PersonalizedDegradationModel(Base):
    """Model for storing personalized degradation strategies"""
    __tablename__ = 'personalized_degradation'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    feature_priorities = Column(JSON)
    acceptable_delays = Column(JSON)
    fallback_preferences = Column(JSON)
    communication_style = Column(String(50))
    system_conditions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    applied_at = Column(DateTime)
    effectiveness_score = Column(Float)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'strategy': self.strategy,
            'feature_priorities': self.feature_priorities,
            'acceptable_delays': self.acceptable_delays,
            'fallback_preferences': self.fallback_preferences,
            'communication_style': self.communication_style,
            'system_conditions': self.system_conditions,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'effectiveness_score': self.effectiveness_score
        }

class InterfaceOptimizationModel(Base):
    """Model for storing interface optimizations"""
    __tablename__ = 'interface_optimization'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    layout_preferences = Column(JSON)
    interaction_patterns = Column(JSON)
    performance_requirements = Column(JSON)
    accessibility_needs = Column(JSON)
    optimization_suggestions = Column(JSON)
    current_interface = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    applied_at = Column(DateTime)
    user_satisfaction_score = Column(Float)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'layout_preferences': self.layout_preferences,
            'interaction_patterns': self.interaction_patterns,
            'performance_requirements': self.performance_requirements,
            'accessibility_needs': self.accessibility_needs,
            'optimization_suggestions': self.optimization_suggestions,
            'current_interface': self.current_interface,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'user_satisfaction_score': self.user_satisfaction_score
        }

class UserInteractionModel(Base):
    """Model for storing user interaction data"""
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100), nullable=False)
    action_type = Column(String(100), nullable=False)
    feature_used = Column(String(100))
    page_visited = Column(String(200))
    duration = Column(Float)  # seconds
    success = Column(Boolean)
    error_encountered = Column(String(500))
    help_requested = Column(Boolean, default=False)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'action_type': self.action_type,
            'feature_used': self.feature_used,
            'page_visited': self.page_visited,
            'duration': self.duration,
            'success': self.success,
            'error_encountered': self.error_encountered,
            'help_requested': self.help_requested,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class SystemMetricsModel(Base):
    """Model for storing system metrics for failure prediction"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_latency = Column(Float)
    error_rate = Column(Float)
    response_time = Column(Float)
    active_users = Column(Integer)
    request_rate = Column(Float)
    system_load = Column(Float)
    additional_metrics = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_latency': self.network_latency,
            'error_rate': self.error_rate,
            'response_time': self.response_time,
            'active_users': self.active_users,
            'request_rate': self.request_rate,
            'system_load': self.system_load,
            'additional_metrics': self.additional_metrics,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class AIUXModelPerformanceModel(Base):
    """Model for tracking AI UX model performance"""
    __tablename__ = 'ai_ux_model_performance'
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(100), nullable=False)  # failure_predictor, behavior_analyzer, etc.
    model_version = Column(String(50))
    accuracy_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    training_duration = Column(Float)  # seconds
    performance_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'accuracy_score': self.accuracy_score,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score,
            'f1_score': self.f1_score,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'training_duration': self.training_duration,
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserFeedbackModel(Base):
    """Model for storing user feedback on AI UX optimizations"""
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    optimization_id = Column(Integer)  # Reference to optimization applied
    optimization_type = Column(String(100))  # degradation, interface, etc.
    satisfaction_score = Column(Float)  # 1-5 scale
    feedback_text = Column(Text)
    improvement_suggestions = Column(JSON)
    would_recommend = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'optimization_id': self.optimization_id,
            'optimization_type': self.optimization_type,
            'satisfaction_score': self.satisfaction_score,
            'feedback_text': self.feedback_text,
            'improvement_suggestions': self.improvement_suggestions,
            'would_recommend': self.would_recommend,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Helper functions for data processing

def create_user_interaction_from_dict(data: Dict[str, Any]) -> UserInteractionModel:
    """Create UserInteractionModel from dictionary"""
    return UserInteractionModel(
        user_id=data.get('user_id'),
        session_id=data.get('session_id'),
        action_type=data.get('action_type'),
        feature_used=data.get('feature_used'),
        page_visited=data.get('page_visited'),
        duration=data.get('duration'),
        success=data.get('success', True),
        error_encountered=data.get('error_encountered'),
        help_requested=data.get('help_requested', False),
        metadata=data.get('metadata', {})
    )

def create_system_metrics_from_dict(data: Dict[str, Any]) -> SystemMetricsModel:
    """Create SystemMetricsModel from dictionary"""
    return SystemMetricsModel(
        cpu_usage=data.get('cpu_usage'),
        memory_usage=data.get('memory_usage'),
        disk_usage=data.get('disk_usage'),
        network_latency=data.get('network_latency'),
        error_rate=data.get('error_rate'),
        response_time=data.get('response_time'),
        active_users=data.get('active_users'),
        request_rate=data.get('request_rate'),
        system_load=data.get('system_load'),
        additional_metrics=data.get('additional_metrics', {})
    )

def aggregate_user_interactions(interactions: List[UserInteractionModel]) -> Dict[str, Any]:
    """Aggregate user interactions for behavior analysis"""
    if not interactions:
        return {}
    
    total_duration = sum(i.duration or 0 for i in interactions)
    successful_actions = sum(1 for i in interactions if i.success)
    errors = sum(1 for i in interactions if i.error_encountered)
    help_requests = sum(1 for i in interactions if i.help_requested)
    unique_features = len(set(i.feature_used for i in interactions if i.feature_used))
    unique_pages = len(set(i.page_visited for i in interactions if i.page_visited))
    
    return {
        'session_duration': total_duration / 60,  # Convert to minutes
        'total_actions': len(interactions),
        'success_rate': successful_actions / len(interactions) if interactions else 0,
        'error_rate': errors / len(interactions) if interactions else 0,
        'help_request_rate': help_requests / len(interactions) if interactions else 0,
        'feature_diversity': unique_features,
        'page_diversity': unique_pages,
        'clicks_per_minute': len(interactions) / (total_duration / 60) if total_duration > 0 else 0,
        'features_used_list': [i.feature_used for i in interactions if i.feature_used],
        'pages_visited': unique_pages,
        'errors_encountered': errors,
        'help_requests': help_requests,
        'advanced_features_used': sum(1 for i in interactions if i.feature_used and 'advanced' in (i.feature_used or '').lower())
    }