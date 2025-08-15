"""
Conversion Funnel Analysis and Optimization
Tracks user journey through conversion funnels and identifies optimization opportunities
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class FunnelStep:
    step_id: str
    name: str
    description: str
    event_criteria: Dict[str, Any]
    order: int
    is_required: bool = True

@dataclass
class FunnelAnalysis:
    funnel_id: str
    analysis_id: str
    period_start: datetime
    period_end: datetime
    total_users: int
    step_conversions: Dict[str, int]
    step_rates: Dict[str, float]
    drop_off_analysis: Dict[str, Any]
    optimization_suggestions: List[str]
    created_at: datetime

@dataclass
class UserJourney:
    user_id: str
    session_id: str
    funnel_id: str
    steps_completed: List[str]
    current_step: Optional[str]
    completion_time: Optional[datetime]
    drop_off_step: Optional[str]
    journey_data: Dict[str, Any]

class ConversionFunnelAnalyzer:
    """Analyzes conversion funnels and provides optimization insights"""
    
    def __init__(self):
        self.funnels: Dict[str, List[FunnelStep]] = {}
        self.user_journeys: Dict[str, UserJourney] = {}
        self.funnel_analyses: List[FunnelAnalysis] = []
        
        # Initialize default funnels
        self._create_default_funnels()
    
    def _create_default_funnels(self):
        """Create default conversion funnels"""
        # User Onboarding Funnel
        onboarding_steps = [
            FunnelStep("landing", "Landing Page Visit", "User visits landing page", 
                      {"event_name": "page_view", "page_url": "/"}, 1),
            FunnelStep("signup", "Sign Up", "User creates account", 
                      {"event_name": "user_signup"}, 2),
            FunnelStep("onboarding", "Complete Onboarding", "User completes onboarding", 
                      {"event_name": "onboarding_completed"}, 3),
            FunnelStep("first_action", "First Action", "User performs first meaningful action", 
                      {"event_name": "first_analysis"}, 4),
            FunnelStep("activation", "User Activation", "User becomes activated", 
                      {"event_name": "user_activated"}, 5)
        ]
        self.funnels["user_onboarding"] = onboarding_steps
        
        # Subscription Funnel
        subscription_steps = [
            FunnelStep("trial_start", "Trial Started", "User starts trial", 
                      {"event_name": "trial_started"}, 1),
            FunnelStep("feature_usage", "Feature Usage", "User uses key features", 
                      {"event_name": "feature_used"}, 2),
            FunnelStep("upgrade_intent", "Upgrade Intent", "User shows upgrade intent", 
                      {"event_name": "pricing_viewed"}, 3),
            FunnelStep("payment_info", "Payment Info", "User enters payment information", 
                      {"event_name": "payment_info_entered"}, 4),
            FunnelStep("subscription", "Subscription", "User completes subscription", 
                      {"event_name": "subscription_created"}, 5)
        ]
        self.funnels["subscription"] = subscription_steps
        
        # Feature Adoption Funnel
        feature_steps = [
            FunnelStep("feature_discovery", "Feature Discovery", "User discovers feature", 
                      {"event_name": "feature_viewed"}, 1),
            FunnelStep("feature_trial", "Feature Trial", "User tries feature", 
                      {"event_name": "feature_tried"}, 2),
            FunnelStep("feature_success", "Feature Success", "User successfully uses feature", 
                      {"event_name": "feature_success"}, 3),
            FunnelStep("feature_adoption", "Feature Adoption", "User adopts feature regularly", 
                      {"event_name": "feature_adopted"}, 4)
        ]
        self.funnels["feature_adoption"] = feature_steps
    
    async def create_custom_funnel(self,
                                  funnel_id: str,
                                  steps: List[Dict[str, Any]]) -> str:
        """Create a custom conversion funnel"""
        try:
            funnel_steps = []
            for i, step_data in enumerate(steps):
                step = FunnelStep(
                    step_id=step_data["step_id"],
                    name=step_data["name"],
                    description=step_data.get("description", ""),
                    event_criteria=step_data["event_criteria"],
                    order=i + 1,
                    is_required=step_data.get("is_required", True)
                )
                funnel_steps.append(step)
            
            self.funnels[funnel_id] = funnel_steps
            logger.info(f"Created custom funnel: {funnel_id} with {len(steps)} steps")
            return funnel_id
            
        except Exception as e:
            logger.error(f"Error creating custom funnel: {str(e)}")
            raise
    
    async def track_user_journey(self,
                                user_id: str,
                                session_id: str,
                                event_name: str,
                                event_properties: Dict[str, Any],
                                page_url: str = "") -> Dict[str, Any]:
        """Track user progress through funnels"""
        try:
            journey_updates = {}
            
            # Check all funnels for matching events
            for funnel_id, steps in self.funnels.items():
                journey_key = f"{user_id}_{funnel_id}"
                
                # Initialize journey if not exists
                if journey_key not in self.user_journeys:
                    self.user_journeys[journey_key] = UserJourney(
                        user_id=user_id,
                        session_id=session_id,
                        funnel_id=funnel_id,
                        steps_completed=[],
                        current_step=None,
                        completion_time=None,
                        drop_off_step=None,
                        journey_data={}
                    )
                
                journey = self.user_journeys[journey_key]
                
                # Check if event matches any funnel step
                for step in steps:
                    if self._event_matches_criteria(event_name, event_properties, page_url, step.event_criteria):
                        if step.step_id not in journey.steps_completed:
                            journey.steps_completed.append(step.step_id)
                            journey.current_step = step.step_id
                            journey.journey_data[step.step_id] = {
                                "completed_at": datetime.utcnow().isoformat(),
                                "event_properties": event_properties
                            }
                            
                            # Check if funnel is complete
                            if len(journey.steps_completed) == len(steps):
                                journey.completion_time = datetime.utcnow()
                            
                            journey_updates[funnel_id] = {
                                "step_completed": step.step_id,
                                "step_name": step.name,
                                "total_steps": len(steps),
                                "completed_steps": len(journey.steps_completed),
                                "is_complete": journey.completion_time is not None
                            }
                            
                            logger.info(f"User {user_id} completed step {step.step_id} in funnel {funnel_id}")
            
            return journey_updates
            
        except Exception as e:
            logger.error(f"Error tracking user journey: {str(e)}")
            raise
    
    def _event_matches_criteria(self,
                               event_name: str,
                               event_properties: Dict[str, Any],
                               page_url: str,
                               criteria: Dict[str, Any]) -> bool:
        """Check if event matches funnel step criteria"""
        # Check event name
        if "event_name" in criteria and event_name != criteria["event_name"]:
            return False
        
        # Check page URL
        if "page_url" in criteria and page_url != criteria["page_url"]:
            return False
        
        # Check event properties
        if "properties" in criteria:
            for key, value in criteria["properties"].items():
                if key not in event_properties or event_properties[key] != value:
                    return False
        
        return True
    
    async def analyze_funnel_performance(self,
                                       funnel_id: str,
                                       days: int = 30) -> FunnelAnalysis:
        """Analyze funnel performance and identify optimization opportunities"""
        try:
            if funnel_id not in self.funnels:
                raise ValueError(f"Funnel {funnel_id} not found")
            
            period_end = datetime.utcnow()
            period_start = period_end - timedelta(days=days)
            
            # Get relevant user journeys
            relevant_journeys = [
                journey for journey in self.user_journeys.values()
                if (journey.funnel_id == funnel_id and 
                    any(datetime.fromisoformat(step_data["completed_at"]) >= period_start
                        for step_data in journey.journey_data.values()))
            ]
            
            if not relevant_journeys:
                return self._create_empty_analysis(funnel_id, period_start, period_end)
            
            # Calculate step conversions
            steps = self.funnels[funnel_id]
            step_conversions = {}
            step_rates = {}
            
            total_users = len(set(j.user_id for j in relevant_journeys))
            
            for step in steps:
                users_completed = len([
                    j for j in relevant_journeys
                    if step.step_id in j.steps_completed
                ])
                step_conversions[step.step_id] = users_completed
                step_rates[step.step_id] = (users_completed / total_users * 100) if total_users > 0 else 0
            
            # Analyze drop-offs
            drop_off_analysis = await self._analyze_drop_offs(relevant_journeys, steps)
            
            # Generate optimization suggestions
            optimization_suggestions = await self._generate_optimization_suggestions(
                step_rates, drop_off_analysis, steps
            )
            
            analysis = FunnelAnalysis(
                funnel_id=funnel_id,
                analysis_id=str(uuid.uuid4()),
                period_start=period_start,
                period_end=period_end,
                total_users=total_users,
                step_conversions=step_conversions,
                step_rates=step_rates,
                drop_off_analysis=drop_off_analysis,
                optimization_suggestions=optimization_suggestions,
                created_at=datetime.utcnow()
            )
            
            self.funnel_analyses.append(analysis)
            logger.info(f"Completed funnel analysis for {funnel_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing funnel performance: {str(e)}")
            raise
    
    def _create_empty_analysis(self, funnel_id: str, period_start: datetime, period_end: datetime) -> FunnelAnalysis:
        """Create empty analysis when no data available"""
        return FunnelAnalysis(
            funnel_id=funnel_id,
            analysis_id=str(uuid.uuid4()),
            period_start=period_start,
            period_end=period_end,
            total_users=0,
            step_conversions={},
            step_rates={},
            drop_off_analysis={"message": "No data available for analysis period"},
            optimization_suggestions=["Increase traffic to start funnel analysis"],
            created_at=datetime.utcnow()
        )
    
    async def _analyze_drop_offs(self,
                                journeys: List[UserJourney],
                                steps: List[FunnelStep]) -> Dict[str, Any]:
        """Analyze where users drop off in the funnel"""
        drop_off_points = {}
        
        for i, step in enumerate(steps[:-1]):  # Exclude last step
            next_step = steps[i + 1]
            
            users_at_step = len([j for j in journeys if step.step_id in j.steps_completed])
            users_at_next = len([j for j in journeys if next_step.step_id in j.steps_completed])
            
            if users_at_step > 0:
                drop_off_rate = ((users_at_step - users_at_next) / users_at_step * 100)
                drop_off_points[f"{step.step_id}_to_{next_step.step_id}"] = {
                    "from_step": step.name,
                    "to_step": next_step.name,
                    "users_at_start": users_at_step,
                    "users_continued": users_at_next,
                    "drop_off_rate": drop_off_rate,
                    "drop_off_count": users_at_step - users_at_next
                }
        
        # Find biggest drop-off point
        biggest_drop_off = None
        max_drop_rate = 0
        
        for transition, data in drop_off_points.items():
            if data["drop_off_rate"] > max_drop_rate:
                max_drop_rate = data["drop_off_rate"]
                biggest_drop_off = transition
        
        return {
            "drop_off_points": drop_off_points,
            "biggest_drop_off": biggest_drop_off,
            "max_drop_off_rate": max_drop_rate,
            "total_drop_offs": sum(data["drop_off_count"] for data in drop_off_points.values())
        }
    
    async def _generate_optimization_suggestions(self,
                                               step_rates: Dict[str, float],
                                               drop_off_analysis: Dict[str, Any],
                                               steps: List[FunnelStep]) -> List[str]:
        """Generate optimization suggestions based on funnel analysis"""
        suggestions = []
        
        # Low conversion rate suggestions
        for step_id, rate in step_rates.items():
            step_name = next(s.name for s in steps if s.step_id == step_id)
            if rate < 20:
                suggestions.append(f"Very low conversion at '{step_name}' ({rate:.1f}%) - consider simplifying or improving UX")
            elif rate < 50:
                suggestions.append(f"Low conversion at '{step_name}' ({rate:.1f}%) - analyze user behavior and optimize")
        
        # Drop-off specific suggestions
        if "biggest_drop_off" in drop_off_analysis and drop_off_analysis["biggest_drop_off"]:
            biggest_drop = drop_off_analysis["drop_off_points"][drop_off_analysis["biggest_drop_off"]]
            suggestions.append(
                f"Major drop-off between '{biggest_drop['from_step']}' and '{biggest_drop['to_step']}' "
                f"({biggest_drop['drop_off_rate']:.1f}%) - investigate user experience issues"
            )
        
        # General suggestions
        if len(step_rates) > 0:
            avg_rate = sum(step_rates.values()) / len(step_rates)
            if avg_rate < 30:
                suggestions.append("Overall funnel performance is low - consider A/B testing different approaches")
            elif avg_rate > 70:
                suggestions.append("Good funnel performance - consider expanding to capture more users")
        
        return suggestions if suggestions else ["Funnel performance looks good - continue monitoring"]
    
    async def get_funnel_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of all funnel performance"""
        try:
            summary = {}
            
            for funnel_id in self.funnels.keys():
                analysis = await self.analyze_funnel_performance(funnel_id, days)
                
                # Calculate overall funnel conversion rate
                if analysis.step_rates:
                    first_step_rate = list(analysis.step_rates.values())[0]
                    last_step_rate = list(analysis.step_rates.values())[-1]
                    overall_rate = (last_step_rate / first_step_rate * 100) if first_step_rate > 0 else 0
                else:
                    overall_rate = 0
                
                summary[funnel_id] = {
                    "total_users": analysis.total_users,
                    "overall_conversion_rate": overall_rate,
                    "step_count": len(self.funnels[funnel_id]),
                    "biggest_drop_off_rate": analysis.drop_off_analysis.get("max_drop_off_rate", 0),
                    "optimization_suggestions_count": len(analysis.optimization_suggestions)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting funnel summary: {str(e)}")
            raise

# Global funnel analyzer instance
funnel_analyzer = ConversionFunnelAnalyzer()