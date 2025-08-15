"""
Design Iteration Engine for Autonomous Innovation Lab

This module provides iterative design improvement and optimization capabilities
for the ScrollIntel autonomous innovation lab system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np

from ..models.prototype_models import (
    Prototype, PrototypeStatus, QualityMetrics, ValidationResult
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IterationType(Enum):
    """Types of design iterations"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    USABILITY_IMPROVEMENT = "usability_improvement"
    FUNCTIONALITY_ENHANCEMENT = "functionality_enhancement"
    ARCHITECTURE_REFINEMENT = "architecture_refinement"
    SECURITY_HARDENING = "security_hardening"
    SCALABILITY_IMPROVEMENT = "scalability_improvement"
    CODE_QUALITY_IMPROVEMENT = "code_quality_improvement"


class FeedbackType(Enum):
    """Types of feedback for design iteration"""
    USER_FEEDBACK = "user_feedback"
    AUTOMATED_TESTING = "automated_testing"
    PERFORMANCE_METRICS = "performance_metrics"
    SECURITY_ANALYSIS = "security_analysis"
    CODE_REVIEW = "code_review"
    USABILITY_TESTING = "usability_testing"
    LOAD_TESTING = "load_testing"


class ConvergenceStatus(Enum):
    """Status of design convergence"""
    DIVERGING = "diverging"
    CONVERGING = "converging"
    CONVERGED = "converged"
    OSCILLATING = "oscillating"
    STAGNANT = "stagnant"


@dataclass
class DesignFeedback:
    """Feedback for design iteration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prototype_id: str = ""
    feedback_type: FeedbackType = FeedbackType.USER_FEEDBACK
    source: str = ""  # Source of feedback (user, system, test, etc.)
    content: str = ""
    severity: float = 0.5  # 0-1 scale, 1 being critical
    priority: float = 0.5  # 0-1 scale, 1 being highest priority
    category: str = ""  # Category of feedback (performance, usability, etc.)
    actionable_items: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    addressed: bool = False


@dataclass
class DesignIteration:
    """Single design iteration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prototype_id: str = ""
    iteration_number: int = 0
    iteration_type: IterationType = IterationType.PERFORMANCE_OPTIMIZATION
    objectives: List[str] = field(default_factory=list)
    changes_made: List[str] = field(default_factory=list)
    feedback_addressed: List[str] = field(default_factory=list)
    
    # Metrics before and after iteration
    pre_iteration_metrics: Optional[QualityMetrics] = None
    post_iteration_metrics: Optional[QualityMetrics] = None
    
    # Iteration results
    improvement_score: float = 0.0  # Overall improvement score
    success_indicators: Dict[str, float] = field(default_factory=dict)
    issues_resolved: List[str] = field(default_factory=list)
    new_issues_introduced: List[str] = field(default_factory=list)
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    
    # Status
    status: str = "planned"  # planned, in_progress, completed, failed


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking design convergence"""
    prototype_id: str = ""
    total_iterations: int = 0
    convergence_status: ConvergenceStatus = ConvergenceStatus.DIVERGING
    convergence_score: float = 0.0  # 0-1 scale, 1 being fully converged
    stability_score: float = 0.0  # How stable the design is
    improvement_velocity: float = 0.0  # Rate of improvement
    diminishing_returns_threshold: float = 0.05  # Threshold for diminishing returns
    
    # Quality metrics over time
    quality_history: List[float] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    usability_history: List[float] = field(default_factory=list)
    
    # Convergence indicators
    last_significant_improvement: Optional[datetime] = None
    consecutive_minor_improvements: int = 0
    oscillation_count: int = 0
    stagnation_count: int = 0
    
    # Predictions
    estimated_iterations_to_convergence: int = 0
    predicted_final_quality_score: float = 0.0


class FeedbackAnalyzer:
    """Analyzes feedback to identify improvement opportunities"""
    
    def __init__(self):
        self.feedback_patterns = self._load_feedback_patterns()
        self.priority_weights = self._load_priority_weights()
    
    def _load_feedback_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for feedback analysis"""
        return {
            "performance": {
                "keywords": ["slow", "lag", "timeout", "performance", "speed", "response time"],
                "severity_multiplier": 1.2,
                "iteration_type": IterationType.PERFORMANCE_OPTIMIZATION
            },
            "usability": {
                "keywords": ["confusing", "difficult", "user-friendly", "intuitive", "ux", "ui"],
                "severity_multiplier": 1.0,
                "iteration_type": IterationType.USABILITY_IMPROVEMENT
            },
            "functionality": {
                "keywords": ["feature", "function", "capability", "missing", "broken", "bug"],
                "severity_multiplier": 1.1,
                "iteration_type": IterationType.FUNCTIONALITY_ENHANCEMENT
            },
            "security": {
                "keywords": ["security", "vulnerability", "exploit", "authentication", "authorization"],
                "severity_multiplier": 1.5,
                "iteration_type": IterationType.SECURITY_HARDENING
            },
            "scalability": {
                "keywords": ["scale", "load", "capacity", "concurrent", "throughput"],
                "severity_multiplier": 1.3,
                "iteration_type": IterationType.SCALABILITY_IMPROVEMENT
            }
        }
    
    def _load_priority_weights(self) -> Dict[FeedbackType, float]:
        """Load priority weights for different feedback types"""
        return {
            FeedbackType.SECURITY_ANALYSIS: 1.5,
            FeedbackType.PERFORMANCE_METRICS: 1.3,
            FeedbackType.USER_FEEDBACK: 1.2,
            FeedbackType.AUTOMATED_TESTING: 1.1,
            FeedbackType.CODE_REVIEW: 1.0,
            FeedbackType.USABILITY_TESTING: 0.9,
            FeedbackType.LOAD_TESTING: 0.8
        }
    
    async def analyze_feedback(self, feedback_list: List[DesignFeedback]) -> Dict[str, Any]:
        """Analyze feedback to identify improvement opportunities"""
        try:
            analysis_results = {
                "total_feedback_items": len(feedback_list),
                "feedback_by_type": {},
                "feedback_by_category": {},
                "priority_items": [],
                "recommended_iterations": [],
                "overall_sentiment": 0.0,
                "critical_issues": [],
                "improvement_opportunities": []
            }
            
            # Categorize feedback
            for feedback in feedback_list:
                # Count by type
                feedback_type = feedback.feedback_type.value
                analysis_results["feedback_by_type"][feedback_type] = \
                    analysis_results["feedback_by_type"].get(feedback_type, 0) + 1
                
                # Analyze content for categories
                category = self._categorize_feedback_content(feedback.content)
                feedback.category = category
                analysis_results["feedback_by_category"][category] = \
                    analysis_results["feedback_by_category"].get(category, 0) + 1
                
                # Identify high priority items
                if feedback.severity > 0.7 or feedback.priority > 0.8:
                    analysis_results["priority_items"].append({
                        "id": feedback.id,
                        "content": feedback.content,
                        "severity": feedback.severity,
                        "priority": feedback.priority,
                        "category": category
                    })
                
                # Identify critical issues
                if feedback.severity > 0.8:
                    analysis_results["critical_issues"].append({
                        "id": feedback.id,
                        "content": feedback.content,
                        "severity": feedback.severity,
                        "category": category
                    })
            
            # Generate recommended iterations
            analysis_results["recommended_iterations"] = \
                await self._generate_iteration_recommendations(feedback_list)
            
            # Calculate overall sentiment
            if feedback_list:
                sentiment_scores = [1.0 - feedback.severity for feedback in feedback_list]
                analysis_results["overall_sentiment"] = sum(sentiment_scores) / len(sentiment_scores)
            
            # Identify improvement opportunities
            analysis_results["improvement_opportunities"] = \
                self._identify_improvement_opportunities(feedback_list)
            
            logger.info(f"Analyzed {len(feedback_list)} feedback items")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing feedback: {str(e)}")
            return {"error": str(e)}
    
    def _categorize_feedback_content(self, content: str) -> str:
        """Categorize feedback content based on keywords"""
        content_lower = content.lower()
        
        for category, pattern in self.feedback_patterns.items():
            if any(keyword in content_lower for keyword in pattern["keywords"]):
                return category
        
        return "general"
    
    async def _generate_iteration_recommendations(self, feedback_list: List[DesignFeedback]) -> List[Dict[str, Any]]:
        """Generate iteration recommendations based on feedback"""
        recommendations = []
        category_counts = {}
        
        # Count feedback by category
        for feedback in feedback_list:
            category = feedback.category
            if category not in category_counts:
                category_counts[category] = {"count": 0, "total_severity": 0.0, "items": []}
            
            category_counts[category]["count"] += 1
            category_counts[category]["total_severity"] += feedback.severity
            category_counts[category]["items"].append(feedback)
        
        # Generate recommendations for each significant category
        for category, data in category_counts.items():
            if data["count"] >= 2 or data["total_severity"] > 1.0:  # Significant feedback
                avg_severity = data["total_severity"] / data["count"]
                
                pattern = self.feedback_patterns.get(category, {})
                iteration_type = pattern.get("iteration_type", IterationType.PERFORMANCE_OPTIMIZATION)
                
                recommendation = {
                    "iteration_type": iteration_type.value,
                    "category": category,
                    "priority": avg_severity,
                    "feedback_count": data["count"],
                    "estimated_effort": self._estimate_iteration_effort(category, data["count"]),
                    "expected_improvement": self._estimate_expected_improvement(category, avg_severity),
                    "feedback_items": [item.id for item in data["items"]]
                }
                
                recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations
    
    def _estimate_iteration_effort(self, category: str, feedback_count: int) -> str:
        """Estimate effort required for iteration"""
        base_effort = {
            "performance": 3,
            "usability": 2,
            "functionality": 4,
            "security": 5,
            "scalability": 4,
            "general": 2
        }
        
        effort_hours = base_effort.get(category, 2) * (1 + feedback_count * 0.2)
        
        if effort_hours <= 4:
            return "low"
        elif effort_hours <= 8:
            return "medium"
        else:
            return "high"
    
    def _estimate_expected_improvement(self, category: str, severity: float) -> float:
        """Estimate expected improvement from addressing feedback"""
        improvement_potential = {
            "performance": 0.8,
            "usability": 0.7,
            "functionality": 0.9,
            "security": 0.6,
            "scalability": 0.8,
            "general": 0.5
        }
        
        base_improvement = improvement_potential.get(category, 0.5)
        return min(base_improvement * severity, 1.0)
    
    def _identify_improvement_opportunities(self, feedback_list: List[DesignFeedback]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities"""
        opportunities = []
        
        # Group feedback by actionable items
        actionable_groups = {}
        for feedback in feedback_list:
            for item in feedback.actionable_items:
                if item not in actionable_groups:
                    actionable_groups[item] = []
                actionable_groups[item].append(feedback)
        
        # Create opportunities from grouped actionable items
        for item, feedback_group in actionable_groups.items():
            if len(feedback_group) >= 2:  # Multiple feedback items mention this
                avg_severity = sum(f.severity for f in feedback_group) / len(feedback_group)
                
                opportunity = {
                    "description": item,
                    "supporting_feedback_count": len(feedback_group),
                    "average_severity": avg_severity,
                    "potential_impact": avg_severity * len(feedback_group),
                    "feedback_ids": [f.id for f in feedback_group]
                }
                
                opportunities.append(opportunity)
        
        # Sort by potential impact
        opportunities.sort(key=lambda x: x["potential_impact"], reverse=True)
        
        return opportunities


class IterationPlanner:
    """Plans and executes design iterations"""
    
    def __init__(self):
        self.iteration_strategies = self._load_iteration_strategies()
    
    def _load_iteration_strategies(self) -> Dict[IterationType, Dict[str, Any]]:
        """Load strategies for different iteration types"""
        return {
            IterationType.PERFORMANCE_OPTIMIZATION: {
                "focus_areas": ["response_time", "throughput", "resource_usage"],
                "techniques": ["caching", "optimization", "profiling", "load_balancing"],
                "success_metrics": ["response_time_improvement", "throughput_increase"],
                "typical_duration": 4  # hours
            },
            IterationType.USABILITY_IMPROVEMENT: {
                "focus_areas": ["user_interface", "user_experience", "accessibility"],
                "techniques": ["ui_redesign", "workflow_optimization", "accessibility_fixes"],
                "success_metrics": ["usability_score", "user_satisfaction"],
                "typical_duration": 6
            },
            IterationType.FUNCTIONALITY_ENHANCEMENT: {
                "focus_areas": ["feature_completeness", "functionality", "capabilities"],
                "techniques": ["feature_addition", "bug_fixes", "capability_enhancement"],
                "success_metrics": ["feature_coverage", "functionality_score"],
                "typical_duration": 8
            },
            IterationType.SECURITY_HARDENING: {
                "focus_areas": ["authentication", "authorization", "data_protection"],
                "techniques": ["security_patches", "encryption", "access_control"],
                "success_metrics": ["security_score", "vulnerability_count"],
                "typical_duration": 6
            },
            IterationType.SCALABILITY_IMPROVEMENT: {
                "focus_areas": ["horizontal_scaling", "vertical_scaling", "load_handling"],
                "techniques": ["architecture_changes", "database_optimization", "caching"],
                "success_metrics": ["concurrent_users", "throughput_scaling"],
                "typical_duration": 10
            }
        }
    
    async def plan_iteration(self, prototype: Prototype, feedback_analysis: Dict[str, Any], 
                           iteration_type: IterationType) -> DesignIteration:
        """Plan a design iteration based on feedback analysis"""
        try:
            iteration = DesignIteration(
                prototype_id=prototype.id,
                iteration_number=self._get_next_iteration_number(prototype.id),
                iteration_type=iteration_type,
                pre_iteration_metrics=prototype.quality_metrics
            )
            
            # Set objectives based on feedback analysis
            iteration.objectives = self._generate_objectives(feedback_analysis, iteration_type)
            
            # Plan changes based on iteration type and feedback
            iteration.changes_made = self._plan_changes(feedback_analysis, iteration_type)
            
            # Identify feedback items to address
            iteration.feedback_addressed = self._select_feedback_to_address(
                feedback_analysis, iteration_type
            )
            
            logger.info(f"Planned iteration {iteration.iteration_number} for prototype {prototype.id}")
            return iteration
            
        except Exception as e:
            logger.error(f"Error planning iteration: {str(e)}")
            raise
    
    def _get_next_iteration_number(self, prototype_id: str) -> int:
        """Get the next iteration number for a prototype"""
        # In a real implementation, this would query the database
        # For now, return a simple incremental number
        return 1
    
    def _generate_objectives(self, feedback_analysis: Dict[str, Any], 
                           iteration_type: IterationType) -> List[str]:
        """Generate objectives for the iteration"""
        strategy = self.iteration_strategies.get(iteration_type, {})
        focus_areas = strategy.get("focus_areas", [])
        
        objectives = []
        
        # Add objectives based on critical issues
        for issue in feedback_analysis.get("critical_issues", []):
            objectives.append(f"Address critical issue: {issue['content'][:100]}")
        
        # Add objectives based on focus areas
        for area in focus_areas:
            objectives.append(f"Improve {area.replace('_', ' ')}")
        
        # Add objectives based on improvement opportunities
        for opportunity in feedback_analysis.get("improvement_opportunities", [])[:3]:
            objectives.append(f"Implement: {opportunity['description']}")
        
        return objectives
    
    def _plan_changes(self, feedback_analysis: Dict[str, Any], 
                     iteration_type: IterationType) -> List[str]:
        """Plan specific changes for the iteration"""
        strategy = self.iteration_strategies.get(iteration_type, {})
        techniques = strategy.get("techniques", [])
        
        changes = []
        
        # Plan changes based on recommended iterations
        for recommendation in feedback_analysis.get("recommended_iterations", []):
            if recommendation["iteration_type"] == iteration_type.value:
                changes.append(f"Apply {recommendation['category']} improvements")
        
        # Add technique-based changes
        for technique in techniques[:2]:  # Limit to top 2 techniques
            changes.append(f"Implement {technique.replace('_', ' ')}")
        
        return changes
    
    def _select_feedback_to_address(self, feedback_analysis: Dict[str, Any], 
                                  iteration_type: IterationType) -> List[str]:
        """Select feedback items to address in this iteration"""
        feedback_ids = []
        
        # Address critical issues first
        for issue in feedback_analysis.get("critical_issues", []):
            feedback_ids.append(issue["id"])
        
        # Address high priority items
        for item in feedback_analysis.get("priority_items", [])[:5]:
            if item["id"] not in feedback_ids:
                feedback_ids.append(item["id"])
        
        return feedback_ids
    
    async def execute_iteration(self, iteration: DesignIteration, 
                              prototype: Prototype) -> DesignIteration:
        """Execute a design iteration"""
        try:
            iteration.status = "in_progress"
            iteration.start_time = datetime.utcnow()
            
            logger.info(f"Executing iteration {iteration.iteration_number} for prototype {prototype.id}")
            
            # Simulate iteration execution
            await self._simulate_iteration_execution(iteration, prototype)
            
            # Update prototype with iteration results
            await self._apply_iteration_results(iteration, prototype)
            
            # Calculate improvement metrics
            iteration.improvement_score = self._calculate_improvement_score(iteration)
            
            # Mark iteration as completed
            iteration.status = "completed"
            iteration.end_time = datetime.utcnow()
            iteration.duration_minutes = (
                iteration.end_time - iteration.start_time
            ).total_seconds() / 60
            
            logger.info(f"Completed iteration {iteration.iteration_number} with improvement score: {iteration.improvement_score:.2f}")
            return iteration
            
        except Exception as e:
            logger.error(f"Error executing iteration: {str(e)}")
            iteration.status = "failed"
            raise
    
    async def _simulate_iteration_execution(self, iteration: DesignIteration, 
                                          prototype: Prototype):
        """Simulate the execution of iteration changes"""
        # Simulate work being done
        await asyncio.sleep(0.2)
        
        # Simulate addressing issues
        for objective in iteration.objectives:
            logger.info(f"Working on objective: {objective}")
            await asyncio.sleep(0.1)
        
        # Simulate implementing changes
        for change in iteration.changes_made:
            logger.info(f"Implementing change: {change}")
            await asyncio.sleep(0.1)
    
    async def _apply_iteration_results(self, iteration: DesignIteration, 
                                     prototype: Prototype):
        """Apply iteration results to the prototype"""
        # Create improved quality metrics
        if prototype.quality_metrics:
            improved_metrics = QualityMetrics(
                code_coverage=min(prototype.quality_metrics.code_coverage + 0.05, 1.0),
                performance_score=min(prototype.quality_metrics.performance_score + 0.1, 1.0),
                usability_score=min(prototype.quality_metrics.usability_score + 0.08, 1.0),
                reliability_score=min(prototype.quality_metrics.reliability_score + 0.06, 1.0),
                security_score=min(prototype.quality_metrics.security_score + 0.07, 1.0),
                maintainability_score=min(prototype.quality_metrics.maintainability_score + 0.04, 1.0),
                scalability_score=min(prototype.quality_metrics.scalability_score + 0.09, 1.0)
            )
        else:
            improved_metrics = QualityMetrics(
                code_coverage=0.75,
                performance_score=0.8,
                usability_score=0.78,
                reliability_score=0.82,
                security_score=0.77,
                maintainability_score=0.74,
                scalability_score=0.79
            )
        
        iteration.post_iteration_metrics = improved_metrics
        prototype.quality_metrics = improved_metrics
        
        # Update prototype status if significantly improved
        if iteration.improvement_score > 0.1:
            if prototype.status == PrototypeStatus.FUNCTIONAL:
                prototype.status = PrototypeStatus.OPTIMIZED
    
    def _calculate_improvement_score(self, iteration: DesignIteration) -> float:
        """Calculate overall improvement score for the iteration"""
        if not iteration.pre_iteration_metrics or not iteration.post_iteration_metrics:
            return 0.0
        
        pre = iteration.pre_iteration_metrics
        post = iteration.post_iteration_metrics
        
        # Calculate improvements in each metric
        improvements = [
            post.code_coverage - pre.code_coverage,
            post.performance_score - pre.performance_score,
            post.usability_score - pre.usability_score,
            post.reliability_score - pre.reliability_score,
            post.security_score - pre.security_score,
            post.maintainability_score - pre.maintainability_score,
            post.scalability_score - pre.scalability_score
        ]
        
        # Calculate weighted average improvement
        weights = [0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.10]
        weighted_improvement = sum(imp * weight for imp, weight in zip(improvements, weights))
        
        return max(weighted_improvement, 0.0)  # Ensure non-negative


class ConvergenceTracker:
    """Tracks design convergence and optimization progress"""
    
    def __init__(self):
        self.convergence_history: Dict[str, ConvergenceMetrics] = {}
    
    async def track_convergence(self, prototype_id: str, 
                              iteration: DesignIteration) -> ConvergenceMetrics:
        """Track convergence metrics for a prototype"""
        try:
            # Get or create convergence metrics
            if prototype_id not in self.convergence_history:
                self.convergence_history[prototype_id] = ConvergenceMetrics(
                    prototype_id=prototype_id
                )
            
            metrics = self.convergence_history[prototype_id]
            
            # Update iteration count
            metrics.total_iterations += 1
            
            # Update quality history
            if iteration.post_iteration_metrics:
                overall_quality = self._calculate_overall_quality(iteration.post_iteration_metrics)
                metrics.quality_history.append(overall_quality)
                metrics.performance_history.append(iteration.post_iteration_metrics.performance_score)
                metrics.usability_history.append(iteration.post_iteration_metrics.usability_score)
            
            # Calculate convergence status
            metrics.convergence_status = self._determine_convergence_status(metrics)
            
            # Calculate convergence score
            metrics.convergence_score = self._calculate_convergence_score(metrics)
            
            # Calculate stability score
            metrics.stability_score = self._calculate_stability_score(metrics)
            
            # Calculate improvement velocity
            metrics.improvement_velocity = self._calculate_improvement_velocity(metrics)
            
            # Update convergence indicators
            self._update_convergence_indicators(metrics, iteration)
            
            # Make predictions
            metrics.estimated_iterations_to_convergence = self._predict_iterations_to_convergence(metrics)
            metrics.predicted_final_quality_score = self._predict_final_quality_score(metrics)
            
            logger.info(f"Updated convergence metrics for prototype {prototype_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking convergence: {str(e)}")
            raise
    
    def _calculate_overall_quality(self, quality_metrics: QualityMetrics) -> float:
        """Calculate overall quality score from quality metrics"""
        weights = [0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.10]
        scores = [
            quality_metrics.code_coverage,
            quality_metrics.performance_score,
            quality_metrics.usability_score,
            quality_metrics.reliability_score,
            quality_metrics.security_score,
            quality_metrics.maintainability_score,
            quality_metrics.scalability_score
        ]
        
        return sum(score * weight for score, weight in zip(scores, weights))
    
    def _determine_convergence_status(self, metrics: ConvergenceMetrics) -> ConvergenceStatus:
        """Determine the convergence status based on quality history"""
        if len(metrics.quality_history) < 3:
            return ConvergenceStatus.DIVERGING
        
        recent_scores = metrics.quality_history[-5:]  # Last 5 iterations
        
        # Check for convergence (small improvements)
        if len(recent_scores) >= 3:
            recent_improvements = [
                recent_scores[i] - recent_scores[i-1] 
                for i in range(1, len(recent_scores))
            ]
            
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            if abs(avg_improvement) < metrics.diminishing_returns_threshold:
                if all(abs(imp) < metrics.diminishing_returns_threshold for imp in recent_improvements):
                    return ConvergenceStatus.CONVERGED
                else:
                    return ConvergenceStatus.OSCILLATING
            elif avg_improvement > 0:
                return ConvergenceStatus.CONVERGING
            else:
                return ConvergenceStatus.DIVERGING
        
        return ConvergenceStatus.DIVERGING
    
    def _calculate_convergence_score(self, metrics: ConvergenceMetrics) -> float:
        """Calculate convergence score (0-1, 1 being fully converged)"""
        if len(metrics.quality_history) < 2:
            return 0.0
        
        # Calculate stability of recent improvements
        recent_scores = metrics.quality_history[-5:]
        if len(recent_scores) < 2:
            return 0.0
        
        # Calculate variance in recent improvements
        improvements = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
        if not improvements:
            return 0.0
        
        variance = np.var(improvements) if len(improvements) > 1 else abs(improvements[0])
        
        # Lower variance indicates higher convergence
        convergence_score = max(0.0, 1.0 - (variance * 10))  # Scale variance
        
        return min(convergence_score, 1.0)
    
    def _calculate_stability_score(self, metrics: ConvergenceMetrics) -> float:
        """Calculate stability score based on quality consistency"""
        if len(metrics.quality_history) < 3:
            return 0.0
        
        recent_scores = metrics.quality_history[-5:]
        
        # Calculate coefficient of variation (std dev / mean)
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / mean_score
        
        # Lower coefficient of variation indicates higher stability
        stability_score = max(0.0, 1.0 - cv)
        
        return min(stability_score, 1.0)
    
    def _calculate_improvement_velocity(self, metrics: ConvergenceMetrics) -> float:
        """Calculate rate of improvement"""
        if len(metrics.quality_history) < 2:
            return 0.0
        
        # Calculate improvement over last few iterations
        recent_scores = metrics.quality_history[-3:]
        if len(recent_scores) < 2:
            return 0.0
        
        total_improvement = recent_scores[-1] - recent_scores[0]
        iterations = len(recent_scores) - 1
        
        return total_improvement / iterations if iterations > 0 else 0.0
    
    def _update_convergence_indicators(self, metrics: ConvergenceMetrics, 
                                     iteration: DesignIteration):
        """Update convergence indicators"""
        # Check for significant improvement
        if iteration.improvement_score > 0.05:  # Significant improvement threshold
            metrics.last_significant_improvement = datetime.utcnow()
            metrics.consecutive_minor_improvements = 0
        else:
            metrics.consecutive_minor_improvements += 1
        
        # Update oscillation count
        if len(metrics.quality_history) >= 3:
            last_three = metrics.quality_history[-3:]
            if (last_three[1] > last_three[0] and last_three[2] < last_three[1]) or \
               (last_three[1] < last_three[0] and last_three[2] > last_three[1]):
                metrics.oscillation_count += 1
        
        # Update stagnation count
        if iteration.improvement_score < 0.01:  # Very small improvement
            metrics.stagnation_count += 1
        else:
            metrics.stagnation_count = 0
    
    def _predict_iterations_to_convergence(self, metrics: ConvergenceMetrics) -> int:
        """Predict how many more iterations until convergence"""
        if metrics.convergence_status == ConvergenceStatus.CONVERGED:
            return 0
        
        if len(metrics.quality_history) < 3:
            return 10  # Default estimate
        
        # Simple linear extrapolation based on improvement velocity
        if metrics.improvement_velocity > 0:
            current_quality = metrics.quality_history[-1]
            target_quality = 0.95  # Target quality score
            remaining_improvement = target_quality - current_quality
            
            if remaining_improvement <= 0:
                return 0
            
            estimated_iterations = int(remaining_improvement / metrics.improvement_velocity)
            return max(1, min(estimated_iterations, 20))  # Cap between 1 and 20
        
        return 15  # Default estimate for slow/no improvement
    
    def _predict_final_quality_score(self, metrics: ConvergenceMetrics) -> float:
        """Predict final quality score based on current trends"""
        if len(metrics.quality_history) < 2:
            return 0.8  # Default prediction
        
        current_quality = metrics.quality_history[-1]
        
        # If converged, current quality is likely final
        if metrics.convergence_status == ConvergenceStatus.CONVERGED:
            return current_quality
        
        # Extrapolate based on improvement velocity
        if metrics.improvement_velocity > 0:
            # Assume diminishing returns
            remaining_potential = (1.0 - current_quality) * 0.8  # 80% of remaining potential
            predicted_final = current_quality + remaining_potential
            return min(predicted_final, 0.98)  # Cap at 98%
        
        return current_quality  # No improvement expected
    
    async def get_convergence_status(self, prototype_id: str) -> Optional[ConvergenceMetrics]:
        """Get current convergence status for a prototype"""
        return self.convergence_history.get(prototype_id)
    
    async def get_convergence_report(self, prototype_id: str) -> Dict[str, Any]:
        """Generate a comprehensive convergence report"""
        metrics = self.convergence_history.get(prototype_id)
        if not metrics:
            return {"error": "No convergence data available"}
        
        return {
            "prototype_id": prototype_id,
            "total_iterations": metrics.total_iterations,
            "convergence_status": metrics.convergence_status.value,
            "convergence_score": metrics.convergence_score,
            "stability_score": metrics.stability_score,
            "improvement_velocity": metrics.improvement_velocity,
            "quality_trend": {
                "current": metrics.quality_history[-1] if metrics.quality_history else 0,
                "peak": max(metrics.quality_history) if metrics.quality_history else 0,
                "average": np.mean(metrics.quality_history) if metrics.quality_history else 0,
                "history": metrics.quality_history[-10:]  # Last 10 iterations
            },
            "predictions": {
                "estimated_iterations_to_convergence": metrics.estimated_iterations_to_convergence,
                "predicted_final_quality_score": metrics.predicted_final_quality_score
            },
            "indicators": {
                "last_significant_improvement": metrics.last_significant_improvement,
                "consecutive_minor_improvements": metrics.consecutive_minor_improvements,
                "oscillation_count": metrics.oscillation_count,
                "stagnation_count": metrics.stagnation_count
            }
        }


class DesignIterationEngine:
    """Main design iteration engine"""
    
    def __init__(self):
        self.feedback_analyzer = FeedbackAnalyzer()
        self.iteration_planner = IterationPlanner()
        self.convergence_tracker = ConvergenceTracker()
        self.active_iterations: Dict[str, List[DesignIteration]] = {}
        self.feedback_store: Dict[str, List[DesignFeedback]] = {}
    
    async def add_feedback(self, prototype_id: str, feedback: DesignFeedback) -> bool:
        """Add feedback for a prototype"""
        try:
            if prototype_id not in self.feedback_store:
                self.feedback_store[prototype_id] = []
            
            feedback.prototype_id = prototype_id
            self.feedback_store[prototype_id].append(feedback)
            
            logger.info(f"Added feedback for prototype {prototype_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False
    
    async def create_feedback_from_text(self, prototype_id: str, feedback_text: str, 
                                      feedback_type: FeedbackType = FeedbackType.USER_FEEDBACK,
                                      source: str = "user") -> DesignFeedback:
        """Create feedback from text input"""
        # Simple sentiment analysis to determine severity
        negative_words = ["slow", "bad", "broken", "confusing", "difficult", "error", "bug"]
        positive_words = ["good", "fast", "easy", "clear", "excellent", "great"]
        
        text_lower = feedback_text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Calculate severity (higher for more negative feedback)
        severity = 0.5 + (negative_count - positive_count) * 0.1
        severity = max(0.0, min(severity, 1.0))
        
        # Extract actionable items (simple keyword extraction)
        actionable_items = []
        if "improve" in text_lower:
            actionable_items.append("General improvement needed")
        if "fix" in text_lower:
            actionable_items.append("Bug fix required")
        if "add" in text_lower:
            actionable_items.append("Feature addition requested")
        
        feedback = DesignFeedback(
            prototype_id=prototype_id,
            feedback_type=feedback_type,
            source=source,
            content=feedback_text,
            severity=severity,
            priority=severity,  # Use severity as initial priority
            actionable_items=actionable_items
        )
        
        return feedback
    
    async def analyze_prototype_feedback(self, prototype_id: str) -> Dict[str, Any]:
        """Analyze all feedback for a prototype"""
        feedback_list = self.feedback_store.get(prototype_id, [])
        return await self.feedback_analyzer.analyze_feedback(feedback_list)
    
    async def plan_next_iteration(self, prototype: Prototype) -> Optional[DesignIteration]:
        """Plan the next iteration for a prototype"""
        try:
            # Analyze feedback
            feedback_analysis = await self.analyze_prototype_feedback(prototype.id)
            
            if not feedback_analysis.get("recommended_iterations"):
                logger.info(f"No iterations recommended for prototype {prototype.id}")
                return None
            
            # Get the highest priority recommendation
            top_recommendation = feedback_analysis["recommended_iterations"][0]
            iteration_type = IterationType(top_recommendation["iteration_type"])
            
            # Plan the iteration
            iteration = await self.iteration_planner.plan_iteration(
                prototype, feedback_analysis, iteration_type
            )
            
            # Store the iteration
            if prototype.id not in self.active_iterations:
                self.active_iterations[prototype.id] = []
            self.active_iterations[prototype.id].append(iteration)
            
            logger.info(f"Planned iteration {iteration.iteration_number} for prototype {prototype.id}")
            return iteration
            
        except Exception as e:
            logger.error(f"Error planning iteration: {str(e)}")
            return None
    
    async def execute_iteration(self, prototype: Prototype, 
                              iteration: DesignIteration) -> DesignIteration:
        """Execute a design iteration"""
        try:
            # Execute the iteration
            completed_iteration = await self.iteration_planner.execute_iteration(iteration, prototype)
            
            # Track convergence
            convergence_metrics = await self.convergence_tracker.track_convergence(
                prototype.id, completed_iteration
            )
            
            # Mark addressed feedback as addressed
            await self._mark_feedback_addressed(prototype.id, completed_iteration.feedback_addressed)
            
            logger.info(f"Executed iteration {completed_iteration.iteration_number} for prototype {prototype.id}")
            return completed_iteration
            
        except Exception as e:
            logger.error(f"Error executing iteration: {str(e)}")
            raise
    
    async def _mark_feedback_addressed(self, prototype_id: str, feedback_ids: List[str]):
        """Mark feedback items as addressed"""
        feedback_list = self.feedback_store.get(prototype_id, [])
        for feedback in feedback_list:
            if feedback.id in feedback_ids:
                feedback.addressed = True
    
    async def run_iteration_cycle(self, prototype: Prototype) -> List[DesignIteration]:
        """Run a complete iteration cycle for a prototype"""
        try:
            iterations = []
            max_iterations = 5  # Limit iterations per cycle
            
            for i in range(max_iterations):
                # Plan next iteration
                iteration = await self.plan_next_iteration(prototype)
                
                if not iteration:
                    logger.info(f"No more iterations needed for prototype {prototype.id}")
                    break
                
                # Execute iteration
                completed_iteration = await self.execute_iteration(prototype, iteration)
                iterations.append(completed_iteration)
                
                # Check convergence
                convergence_metrics = await self.convergence_tracker.get_convergence_status(prototype.id)
                if convergence_metrics and convergence_metrics.convergence_status == ConvergenceStatus.CONVERGED:
                    logger.info(f"Prototype {prototype.id} has converged after {i+1} iterations")
                    break
                
                # Check for diminishing returns
                if completed_iteration.improvement_score < 0.02:  # Very small improvement
                    logger.info(f"Diminishing returns detected for prototype {prototype.id}")
                    break
            
            logger.info(f"Completed iteration cycle for prototype {prototype.id} with {len(iterations)} iterations")
            return iterations
            
        except Exception as e:
            logger.error(f"Error running iteration cycle: {str(e)}")
            return []
    
    async def get_iteration_history(self, prototype_id: str) -> List[DesignIteration]:
        """Get iteration history for a prototype"""
        return self.active_iterations.get(prototype_id, [])
    
    async def get_feedback_summary(self, prototype_id: str) -> Dict[str, Any]:
        """Get feedback summary for a prototype"""
        feedback_list = self.feedback_store.get(prototype_id, [])
        
        total_feedback = len(feedback_list)
        addressed_feedback = len([f for f in feedback_list if f.addressed])
        pending_feedback = total_feedback - addressed_feedback
        
        avg_severity = sum(f.severity for f in feedback_list) / total_feedback if total_feedback > 0 else 0
        
        feedback_by_type = {}
        for feedback in feedback_list:
            feedback_type = feedback.feedback_type.value
            feedback_by_type[feedback_type] = feedback_by_type.get(feedback_type, 0) + 1
        
        return {
            "total_feedback": total_feedback,
            "addressed_feedback": addressed_feedback,
            "pending_feedback": pending_feedback,
            "average_severity": avg_severity,
            "feedback_by_type": feedback_by_type,
            "recent_feedback": [
                {
                    "id": f.id,
                    "content": f.content[:100],
                    "severity": f.severity,
                    "addressed": f.addressed,
                    "timestamp": f.timestamp
                }
                for f in sorted(feedback_list, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    async def generate_iteration_report(self, prototype_id: str) -> Dict[str, Any]:
        """Generate comprehensive iteration report"""
        try:
            iterations = await self.get_iteration_history(prototype_id)
            feedback_summary = await self.get_feedback_summary(prototype_id)
            convergence_report = await self.convergence_tracker.get_convergence_report(prototype_id)
            
            # Calculate iteration statistics
            total_iterations = len(iterations)
            successful_iterations = len([i for i in iterations if i.status == "completed"])
            avg_improvement = sum(i.improvement_score for i in iterations) / total_iterations if total_iterations > 0 else 0
            total_duration = sum(i.duration_minutes for i in iterations)
            
            return {
                "prototype_id": prototype_id,
                "iteration_statistics": {
                    "total_iterations": total_iterations,
                    "successful_iterations": successful_iterations,
                    "success_rate": successful_iterations / total_iterations if total_iterations > 0 else 0,
                    "average_improvement_score": avg_improvement,
                    "total_duration_minutes": total_duration
                },
                "feedback_summary": feedback_summary,
                "convergence_report": convergence_report,
                "recent_iterations": [
                    {
                        "iteration_number": i.iteration_number,
                        "iteration_type": i.iteration_type.value,
                        "improvement_score": i.improvement_score,
                        "duration_minutes": i.duration_minutes,
                        "status": i.status,
                        "objectives_count": len(i.objectives)
                    }
                    for i in iterations[-5:]  # Last 5 iterations
                ],
                "recommendations": self._generate_iteration_recommendations(iterations, convergence_report)
            }
            
        except Exception as e:
            logger.error(f"Error generating iteration report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_iteration_recommendations(self, iterations: List[DesignIteration], 
                                          convergence_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on iteration history"""
        recommendations = []
        
        if not iterations:
            recommendations.append("Start collecting feedback to enable design iterations")
            return recommendations
        
        # Check convergence status
        convergence_status = convergence_report.get("convergence_status", "diverging")
        
        if convergence_status == "converged":
            recommendations.append("Design has converged - consider moving to production")
        elif convergence_status == "oscillating":
            recommendations.append("Design is oscillating - consider stabilizing changes")
        elif convergence_status == "stagnant":
            recommendations.append("Design improvements have stagnated - consider new approaches")
        
        # Check improvement trends
        recent_improvements = [i.improvement_score for i in iterations[-3:]]
        if recent_improvements and all(score < 0.05 for score in recent_improvements):
            recommendations.append("Recent improvements are minimal - consider major redesign")
        
        # Check iteration types
        iteration_types = [i.iteration_type.value for i in iterations]
        most_common_type = max(set(iteration_types), key=iteration_types.count) if iteration_types else None
        
        if most_common_type:
            recommendations.append(f"Focus has been on {most_common_type.replace('_', ' ')} - consider diversifying iteration types")
        
        return recommendations