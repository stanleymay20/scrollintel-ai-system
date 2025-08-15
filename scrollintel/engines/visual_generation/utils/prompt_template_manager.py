"""
Prompt Template Manager for visual generation.
Handles storage, retrieval, and management of successful prompt patterns and templates.
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from scrollintel.models.database_utils import get_sync_db
from scrollintel.models.prompt_enhancement_models import (
    VisualPromptTemplate, VisualPromptPattern, VisualPromptVariation, 
    VisualABTestExperiment, VisualABTestResult, VisualPromptCategory,
    VisualPromptUsageLog, VisualPromptOptimizationSuggestion
)

class PromptTemplateManager:
    """Manages prompt templates, patterns, and A/B testing."""
    
    def __init__(self, db_session=None):
        if db_session:
            self.db = db_session
        else:
            # Use simple SQLite connection for demo
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            database_url = os.getenv("DATABASE_URL", "sqlite:///./scrollintel.db")
            engine = create_engine(database_url)
            SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
            self.db = SessionLocal()
    
    def get_template_by_id(self, template_id: int) -> Optional[VisualPromptTemplate]:
        """Get a specific template by ID."""
        return self.db.query(VisualPromptTemplate).filter(
            VisualPromptTemplate.id == template_id,
            VisualPromptTemplate.is_active == True
        ).first()
    
    def get_templates_by_category(self, category_name: str, limit: int = 10) -> List[VisualPromptTemplate]:
        """Get templates by category name."""
        return self.db.query(VisualPromptTemplate).join(VisualPromptCategory).filter(
            VisualPromptCategory.name == category_name,
            VisualPromptTemplate.is_active == True,
            VisualPromptCategory.is_active == True
        ).order_by(desc(VisualPromptTemplate.success_rate)).limit(limit).all()
    
    def search_templates(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[VisualPromptTemplate]:
        """Search templates by name, description, or tags."""
        search_filter = or_(
            VisualPromptTemplate.name.ilike(f"%{query}%"),
            VisualPromptTemplate.description.ilike(f"%{query}%"),
            VisualPromptTemplate.template.ilike(f"%{query}%")
        )
        
        query_obj = self.db.query(VisualPromptTemplate).filter(
            search_filter,
            VisualPromptTemplate.is_active == True
        )
        
        if category:
            query_obj = query_obj.join(VisualPromptCategory).filter(
                VisualPromptCategory.name == category,
                VisualPromptCategory.is_active == True
            )
        
        return query_obj.order_by(desc(VisualPromptTemplate.success_rate)).limit(limit).all()
    
    def get_top_templates(self, limit: int = 10) -> List[VisualPromptTemplate]:
        """Get top-performing templates by success rate."""
        return self.db.query(VisualPromptTemplate).filter(
            VisualPromptTemplate.is_active == True,
            VisualPromptTemplate.usage_count > 5  # Only templates with sufficient usage
        ).order_by(desc(VisualPromptTemplate.success_rate)).limit(limit).all()
    
    def create_template(self, name: str, template: str, description: str, 
                       category_id: int, parameters: List[str], 
                       created_by: str, tags: Optional[List[str]] = None) -> VisualPromptTemplate:
        """Create a new prompt template."""
        new_template = VisualPromptTemplate(
            name=name,
            template=template,
            description=description,
            category_id=category_id,
            parameters=parameters,
            tags=tags or [],
            created_by=created_by
        )
        
        self.db.add(new_template)
        self.db.commit()
        self.db.refresh(new_template)
        return new_template
    
    def update_template_performance(self, template_id: int, quality_score: float, 
                                  success: bool, generation_time: float) -> None:
        """Update template performance metrics."""
        template = self.get_template_by_id(template_id)
        if not template:
            return
        
        # Update usage count
        template.usage_count += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        if success:
            template.success_rate = template.success_rate * (1 - alpha) + alpha
        else:
            template.success_rate = template.success_rate * (1 - alpha)
        
        # Update average quality score
        if template.average_quality_score == 0:
            template.average_quality_score = quality_score
        else:
            template.average_quality_score = (
                template.average_quality_score * 0.9 + quality_score * 0.1
            )
        
        template.updated_at = datetime.utcnow()
        self.db.commit()
    
    def get_successful_patterns(self, pattern_type: Optional[str] = None, 
                              context: Optional[str] = None, limit: int = 20) -> List[VisualPromptPattern]:
        """Get successful prompt patterns."""
        query = self.db.query(VisualPromptPattern).filter(
            VisualPromptPattern.is_active == True,
            VisualPromptPattern.success_rate > 0.7
        )
        
        if pattern_type:
            query = query.filter(VisualPromptPattern.pattern_type == pattern_type)
        
        if context:
            query = query.filter(VisualPromptPattern.context.ilike(f"%{context}%"))
        
        return query.order_by(desc(VisualPromptPattern.effectiveness_score)).limit(limit).all()
    
    def create_pattern(self, pattern_text: str, pattern_type: str, 
                      context: str, effectiveness_score: float) -> VisualPromptPattern:
        """Create a new successful pattern."""
        new_pattern = VisualPromptPattern(
            pattern_text=pattern_text,
            pattern_type=pattern_type,
            context=context,
            effectiveness_score=effectiveness_score
        )
        
        self.db.add(new_pattern)
        self.db.commit()
        self.db.refresh(new_pattern)
        return new_pattern
    
    def generate_template_variations(self, template_id: int, num_variations: int = 3) -> List[VisualPromptVariation]:
        """Generate variations of a template for A/B testing."""
        template = self.get_template_by_id(template_id)
        if not template:
            return []
        
        # Get successful patterns that could enhance this template
        patterns = self.get_successful_patterns(limit=10)
        
        variations = []
        for i in range(num_variations):
            # Create variation by adding successful patterns
            selected_patterns = random.sample(patterns, min(2, len(patterns)))
            
            variation_text = template.template
            for pattern in selected_patterns:
                if pattern.pattern_text not in variation_text:
                    variation_text += f", {pattern.pattern_text}"
            
            variation = VisualPromptVariation(
                template_id=template_id,
                variation_name=f"Variation {i+1}",
                variation_text=variation_text,
                description=f"Enhanced with {len(selected_patterns)} successful patterns"
            )
            
            self.db.add(variation)
            variations.append(variation)
        
        self.db.commit()
        return variations
    
    def create_ab_test(self, name: str, description: str, template_id: int, 
                      created_by: str, target_sample_size: int = 100) -> VisualABTestExperiment:
        """Create a new A/B test experiment."""
        # Generate variations if they don't exist
        existing_variations = self.db.query(VisualPromptVariation).filter(
            VisualPromptVariation.template_id == template_id,
            VisualPromptVariation.is_active == True
        ).count()
        
        if existing_variations < 2:
            self.generate_template_variations(template_id, 3)
        
        experiment = VisualABTestExperiment(
            name=name,
            description=description,
            template_id=template_id,
            target_sample_size=target_sample_size,
            created_by=created_by
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def get_active_ab_tests(self, limit: int = 10) -> List[VisualABTestExperiment]:
        """Get active A/B test experiments."""
        return self.db.query(VisualABTestExperiment).filter(
            VisualABTestExperiment.status == "active"
        ).order_by(desc(VisualABTestExperiment.created_at)).limit(limit).all()
    
    def record_ab_test_result(self, experiment_id: int, variation_id: int, 
                            user_id: str, quality_score: float, user_rating: int,
                            generation_time: float, success: bool, 
                            feedback: Optional[str] = None) -> VisualABTestResult:
        """Record a result from an A/B test."""
        result = VisualABTestResult(
            experiment_id=experiment_id,
            variation_id=variation_id,
            user_id=user_id,
            quality_score=quality_score,
            user_rating=user_rating,
            generation_time=generation_time,
            success=success,
            feedback=feedback
        )
        
        self.db.add(result)
        
        # Update experiment sample size
        experiment = self.db.query(VisualABTestExperiment).filter(
            VisualABTestExperiment.id == experiment_id
        ).first()
        
        if experiment:
            experiment.current_sample_size += 1
            
            # Check if experiment is complete
            if experiment.current_sample_size >= experiment.target_sample_size:
                self.analyze_ab_test_results(experiment_id)
        
        self.db.commit()
        self.db.refresh(result)
        return result
    
    def analyze_ab_test_results(self, experiment_id: int) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner."""
        experiment = self.db.query(VisualABTestExperiment).filter(
            VisualABTestExperiment.id == experiment_id
        ).first()
        
        if not experiment:
            return {}
        
        # Get results grouped by variation
        results = self.db.query(
            VisualABTestResult.variation_id,
            func.count(VisualABTestResult.id).label('count'),
            func.avg(VisualABTestResult.quality_score).label('avg_quality'),
            func.avg(VisualABTestResult.user_rating).label('avg_rating'),
            func.sum(func.cast(VisualABTestResult.success, func.INTEGER)).label('success_count')
        ).filter(
            VisualABTestResult.experiment_id == experiment_id
        ).group_by(VisualABTestResult.variation_id).all()
        
        if not results:
            return {}
        
        # Find the best performing variation
        best_variation = max(results, key=lambda x: (x.avg_quality + x.avg_rating) / 2)
        
        # Calculate statistical significance (simplified)
        total_results = sum(r.count for r in results)
        best_success_rate = best_variation.success_count / best_variation.count
        
        # Update experiment
        experiment.status = "completed"
        experiment.end_date = datetime.utcnow()
        experiment.winning_variation_id = best_variation.variation_id
        experiment.statistical_significance = best_success_rate
        
        # Update winning variation performance
        winning_variation = self.db.query(VisualPromptVariation).filter(
            VisualPromptVariation.id == best_variation.variation_id
        ).first()
        
        if winning_variation:
            winning_variation.success_rate = best_success_rate
            winning_variation.average_quality_score = best_variation.avg_quality
            winning_variation.usage_count += best_variation.count
        
        self.db.commit()
        
        return {
            'experiment_id': experiment_id,
            'winning_variation_id': best_variation.variation_id,
            'success_rate': best_success_rate,
            'avg_quality': best_variation.avg_quality,
            'avg_rating': best_variation.avg_rating,
            'sample_size': best_variation.count,
            'statistical_significance': best_success_rate
        }
    
    def get_template_for_ab_test(self, experiment_id: int, user_id: str) -> Optional[Tuple[VisualPromptVariation, bool]]:
        """Get a template variation for A/B testing."""
        experiment = self.db.query(VisualABTestExperiment).filter(
            VisualABTestExperiment.id == experiment_id,
            VisualABTestExperiment.status == "active"
        ).first()
        
        if not experiment:
            return None
        
        # Check if user has already participated
        existing_result = self.db.query(VisualABTestResult).filter(
            VisualABTestResult.experiment_id == experiment_id,
            VisualABTestResult.user_id == user_id
        ).first()
        
        if existing_result:
            # Return the same variation they used before
            variation = self.db.query(VisualPromptVariation).filter(
                VisualPromptVariation.id == existing_result.variation_id
            ).first()
            return (variation, False) if variation else None
        
        # Get all variations for this experiment
        variations = self.db.query(VisualPromptVariation).filter(
            VisualPromptVariation.template_id == experiment.template_id,
            VisualPromptVariation.is_active == True
        ).all()
        
        if not variations:
            return None
        
        # Randomly assign a variation (simple randomization)
        selected_variation = random.choice(variations)
        return (selected_variation, True)
    
    def log_prompt_usage(self, template_id: Optional[int], user_id: str, 
                        prompt_text: str, parameters_used: Dict[str, Any],
                        quality_score: float, user_rating: Optional[int],
                        generation_time: float, success: bool, model_used: str) -> None:
        """Log prompt usage for analytics."""
        usage_log = VisualPromptUsageLog(
            template_id=template_id,
            user_id=user_id,
            prompt_text=prompt_text,
            parameters_used=parameters_used,
            quality_score=quality_score,
            user_rating=user_rating,
            generation_time=generation_time,
            success=success,
            model_used=model_used
        )
        
        self.db.add(usage_log)
        self.db.commit()
    
    def get_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage analytics for the specified period."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Total usage
        total_usage = self.db.query(VisualPromptUsageLog).filter(
            VisualPromptUsageLog.created_at >= start_date
        ).count()
        
        # Success rate
        successful_usage = self.db.query(VisualPromptUsageLog).filter(
            VisualPromptUsageLog.created_at >= start_date,
            VisualPromptUsageLog.success == True
        ).count()
        
        success_rate = successful_usage / total_usage if total_usage > 0 else 0
        
        # Average quality score
        avg_quality = self.db.query(func.avg(VisualPromptUsageLog.quality_score)).filter(
            VisualPromptUsageLog.created_at >= start_date,
            VisualPromptUsageLog.quality_score.isnot(None)
        ).scalar() or 0
        
        # Top templates
        top_templates = self.db.query(
            VisualPromptTemplate.name,
            func.count(VisualPromptUsageLog.id).label('usage_count')
        ).join(VisualPromptUsageLog).filter(
            VisualPromptUsageLog.created_at >= start_date
        ).group_by(VisualPromptTemplate.id, VisualPromptTemplate.name).order_by(
            desc('usage_count')
        ).limit(10).all()
        
        return {
            'period_days': days,
            'total_usage': total_usage,
            'success_rate': success_rate,
            'average_quality_score': avg_quality,
            'top_templates': [{'name': t.name, 'usage_count': t.usage_count} for t in top_templates]
        }
    
    def suggest_prompt_optimization(self, original_prompt: str, 
                                  suggestion_type: str = "quality_improvement") -> Optional[str]:
        """Generate AI-powered prompt optimization suggestions."""
        # Get successful patterns that could improve this prompt
        patterns = self.get_successful_patterns(limit=5)
        
        if not patterns:
            return None
        
        # Simple optimization: add successful patterns that aren't already present
        optimized_prompt = original_prompt
        added_patterns = []
        
        for pattern in patterns:
            if pattern.pattern_text.lower() not in original_prompt.lower():
                optimized_prompt += f", {pattern.pattern_text}"
                added_patterns.append(pattern.pattern_text)
                
                # Limit to 2-3 additional patterns to avoid over-optimization
                if len(added_patterns) >= 2:
                    break
        
        if added_patterns:
            # Log the suggestion
            suggestion = VisualPromptOptimizationSuggestion(
                original_prompt=original_prompt,
                suggested_prompt=optimized_prompt,
                suggestion_type=suggestion_type,
                confidence_score=0.8,  # Simple confidence score
                reasoning=f"Added successful patterns: {', '.join(added_patterns)}"
            )
            
            self.db.add(suggestion)
            self.db.commit()
            
            return optimized_prompt
        
        return None
    
    def get_categories(self) -> List[VisualPromptCategory]:
        """Get all active prompt categories."""
        return self.db.query(VisualPromptCategory).filter(
            VisualPromptCategory.is_active == True
        ).order_by(VisualPromptCategory.name).all()
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()