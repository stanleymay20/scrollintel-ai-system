"""
Intentionality engine for goal-directed behavior in AGI cognitive architecture.
Implements intention formation, goal management, and purposeful action planning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid

from ..models.consciousness_models import (
    Goal, IntentionalState, CognitiveContext, ConsciousnessState,
    Thought, Experience
)


logger = logging.getLogger(__name__)


class IntentionalityEngine:
    """Engine for managing intentional states and goal-directed behavior"""
    
    def __init__(self):
        self.goal_registry: Dict[str, Goal] = {}
        self.intention_history: List[IntentionalState] = []
        self.goal_achievement_history: List[Dict[str, Any]] = []
        self.intention_strength_threshold = 0.6
        self.max_concurrent_goals = 10
        
    async def form_intention(self, description: str, context: CognitiveContext, 
                           priority: float = 0.5) -> Goal:
        """Form a new intention/goal"""
        logger.info(f"Forming intention: {description}")
        
        goal = Goal(
            description=description,
            priority=self._calculate_contextual_priority(priority, context),
            status="forming",
            context={
                "formation_context": context.situation,
                "complexity": context.complexity_level,
                "urgency": context.time_pressure,
                "resources": context.available_resources,
                "constraints": context.constraints
            }
        )
        
        # Decompose into sub-goals if complex
        if context.complexity_level > 0.7:
            sub_goals = await self._decompose_goal(goal, context)
            goal.sub_goals = [sg.id for sg in sub_goals]
            
            # Register sub-goals
            for sub_goal in sub_goals:
                self.goal_registry[sub_goal.id] = sub_goal
        
        # Register main goal
        self.goal_registry[goal.id] = goal
        goal.status = "active"
        
        return goal
    
    async def generate_intentional_state(self, primary_goal: Goal, 
                                       consciousness_state: ConsciousnessState) -> IntentionalState:
        """Generate comprehensive intentional state"""
        logger.info(f"Generating intentional state for goal: {primary_goal.description}")
        
        intentional_state = IntentionalState()
        intentional_state.primary_goal = primary_goal
        
        # Select active goals based on priority and context
        active_goals = await self._select_active_goals(primary_goal, consciousness_state)
        intentional_state.active_goals = active_goals
        
        # Build goal hierarchy
        hierarchy = await self._build_goal_hierarchy(active_goals)
        intentional_state.goal_hierarchy = hierarchy
        
        # Calculate intention strength
        intention_strength = await self._calculate_intention_strength(
            primary_goal, active_goals, consciousness_state
        )
        intentional_state.intention_strength = intention_strength
        
        # Determine focus direction
        focus_direction = await self._determine_focus_direction(
            primary_goal, consciousness_state
        )
        intentional_state.focus_direction = focus_direction
        
        # Calculate commitment level
        commitment_level = await self._calculate_commitment_level(
            primary_goal, active_goals, consciousness_state
        )
        intentional_state.commitment_level = commitment_level
        
        # Store in history
        self.intention_history.append(intentional_state)
        
        return intentional_state
    
    async def update_goal_progress(self, goal_id: str, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update progress on a specific goal"""
        if goal_id not in self.goal_registry:
            raise ValueError(f"Goal {goal_id} not found in registry")
        
        goal = self.goal_registry[goal_id]
        logger.info(f"Updating progress for goal: {goal.description}")
        
        # Update goal context with progress
        goal.context.update({
            "progress_update": progress_data,
            "last_updated": datetime.now(),
            "progress_score": progress_data.get("completion_percentage", 0.0)
        })
        
        # Check if goal is completed
        if progress_data.get("completion_percentage", 0.0) >= 1.0:
            goal.status = "completed"
            await self._handle_goal_completion(goal, progress_data)
        
        # Update sub-goals if applicable
        if goal.sub_goals:
            await self._update_sub_goal_progress(goal, progress_data)
        
        return {
            "goal_id": goal_id,
            "updated_status": goal.status,
            "progress_score": goal.context.get("progress_score", 0.0),
            "next_actions": await self._suggest_next_actions(goal)
        }
    
    async def resolve_goal_conflicts(self, conflicting_goals: List[Goal]) -> Dict[str, Any]:
        """Resolve conflicts between competing goals"""
        logger.info(f"Resolving conflicts between {len(conflicting_goals)} goals")
        
        resolution_strategy = await self._determine_resolution_strategy(conflicting_goals)
        
        if resolution_strategy == "prioritize":
            resolved_goals = await self._prioritize_goals(conflicting_goals)
        elif resolution_strategy == "merge":
            resolved_goals = await self._merge_goals(conflicting_goals)
        elif resolution_strategy == "sequence":
            resolved_goals = await self._sequence_goals(conflicting_goals)
        else:
            resolved_goals = conflicting_goals  # No resolution needed
        
        return {
            "strategy_used": resolution_strategy,
            "original_goals": [g.id for g in conflicting_goals],
            "resolved_goals": [g.id for g in resolved_goals],
            "resolution_rationale": await self._generate_resolution_rationale(
                resolution_strategy, conflicting_goals, resolved_goals
            )
        }
    
    async def adapt_intentions_to_context(self, new_context: CognitiveContext,
                                        current_intentions: IntentionalState) -> IntentionalState:
        """Adapt intentions based on changing context"""
        logger.info("Adapting intentions to new context")
        
        adapted_state = IntentionalState()
        
        # Re-evaluate primary goal relevance
        if current_intentions.primary_goal:
            adapted_primary = await self._adapt_goal_to_context(
                current_intentions.primary_goal, new_context
            )
            adapted_state.primary_goal = adapted_primary
        
        # Re-evaluate active goals
        adapted_active_goals = []
        for goal in current_intentions.active_goals:
            adapted_goal = await self._adapt_goal_to_context(goal, new_context)
            if adapted_goal.priority > 0.3:  # Keep goals above threshold
                adapted_active_goals.append(adapted_goal)
        
        adapted_state.active_goals = adapted_active_goals
        
        # Rebuild hierarchy with adapted goals
        adapted_state.goal_hierarchy = await self._build_goal_hierarchy(adapted_active_goals)
        
        # Recalculate intention metrics
        adapted_state.intention_strength = await self._calculate_contextual_intention_strength(
            adapted_state.primary_goal, new_context
        )
        
        adapted_state.focus_direction = await self._determine_contextual_focus(
            adapted_state.primary_goal, new_context
        )
        
        adapted_state.commitment_level = await self._calculate_contextual_commitment(
            adapted_state.primary_goal, adapted_active_goals, new_context
        )
        
        return adapted_state
    
    async def _decompose_goal(self, goal: Goal, context: CognitiveContext) -> List[Goal]:
        """Decompose complex goal into sub-goals"""
        sub_goals = []
        
        # Simple decomposition based on goal description keywords
        if "implement" in goal.description.lower():
            sub_goals.extend([
                Goal(description=f"Design {goal.description}", priority=goal.priority * 0.9),
                Goal(description=f"Develop {goal.description}", priority=goal.priority * 0.8),
                Goal(description=f"Test {goal.description}", priority=goal.priority * 0.7),
                Goal(description=f"Deploy {goal.description}", priority=goal.priority * 0.6)
            ])
        elif "analyze" in goal.description.lower():
            sub_goals.extend([
                Goal(description=f"Gather data for {goal.description}", priority=goal.priority * 0.9),
                Goal(description=f"Process data for {goal.description}", priority=goal.priority * 0.8),
                Goal(description=f"Generate insights for {goal.description}", priority=goal.priority * 0.7)
            ])
        else:
            # Generic decomposition
            sub_goals.extend([
                Goal(description=f"Plan {goal.description}", priority=goal.priority * 0.9),
                Goal(description=f"Execute {goal.description}", priority=goal.priority * 0.8),
                Goal(description=f"Validate {goal.description}", priority=goal.priority * 0.7)
            ])
        
        return sub_goals
    
    async def _select_active_goals(self, primary_goal: Goal, 
                                 consciousness_state: ConsciousnessState) -> List[Goal]:
        """Select active goals based on priority and context"""
        all_goals = list(self.goal_registry.values())
        
        # Filter active goals
        active_goals = [g for g in all_goals if g.status == "active"]
        
        # Sort by priority
        active_goals.sort(key=lambda g: g.priority, reverse=True)
        
        # Include primary goal if not already in list
        if primary_goal not in active_goals:
            active_goals.insert(0, primary_goal)
        
        # Limit to max concurrent goals
        return active_goals[:self.max_concurrent_goals]
    
    async def _build_goal_hierarchy(self, goals: List[Goal]) -> Dict[str, List[str]]:
        """Build hierarchical structure of goals"""
        hierarchy = {}
        
        for goal in goals:
            hierarchy[goal.id] = goal.sub_goals
            
            # Add parent-child relationships
            for sub_goal_id in goal.sub_goals:
                if sub_goal_id in self.goal_registry:
                    sub_goal = self.goal_registry[sub_goal_id]
                    if "parent_goals" not in sub_goal.context:
                        sub_goal.context["parent_goals"] = []
                    if goal.id not in sub_goal.context["parent_goals"]:
                        sub_goal.context["parent_goals"].append(goal.id)
        
        return hierarchy
    
    async def _calculate_intention_strength(self, primary_goal: Goal, active_goals: List[Goal],
                                          consciousness_state: ConsciousnessState) -> float:
        """Calculate overall intention strength"""
        factors = []
        
        # Primary goal priority
        factors.append(primary_goal.priority)
        
        # Goal alignment
        if active_goals:
            avg_priority = sum(g.priority for g in active_goals) / len(active_goals)
            factors.append(avg_priority)
        
        # Consciousness coherence
        factors.append(consciousness_state.consciousness_coherence)
        
        # Commitment consistency
        if self.intention_history:
            recent_intentions = [i for i in self.intention_history[-5:]]
            if recent_intentions:
                avg_commitment = sum(i.commitment_level for i in recent_intentions) / len(recent_intentions)
                factors.append(avg_commitment)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    async def _determine_focus_direction(self, primary_goal: Goal, 
                                       consciousness_state: ConsciousnessState) -> str:
        """Determine the direction of cognitive focus"""
        if primary_goal.priority > 0.8:
            return "high_priority_execution"
        elif primary_goal.priority > 0.6:
            return "strategic_planning"
        elif "analyze" in primary_goal.description.lower():
            return "analytical_investigation"
        elif "create" in primary_goal.description.lower():
            return "creative_exploration"
        else:
            return "balanced_approach"
    
    async def _calculate_commitment_level(self, primary_goal: Goal, active_goals: List[Goal],
                                        consciousness_state: ConsciousnessState) -> float:
        """Calculate commitment level to goals"""
        if not active_goals:
            return 1.0
        
        # Base commitment on primary goal priority
        base_commitment = primary_goal.priority
        
        # Adjust for goal conflicts
        total_priority = sum(g.priority for g in active_goals)
        if total_priority > 0:
            priority_ratio = primary_goal.priority / total_priority
            commitment_adjustment = min(1.0, priority_ratio * 2)
        else:
            commitment_adjustment = 0.5
        
        # Factor in consciousness coherence
        coherence_factor = consciousness_state.consciousness_coherence
        
        return min(1.0, (base_commitment + commitment_adjustment + coherence_factor) / 3)
    
    def _calculate_contextual_priority(self, base_priority: float, context: CognitiveContext) -> float:
        """Calculate priority adjusted for context"""
        adjusted_priority = base_priority
        
        # Increase priority for urgent contexts
        if context.time_pressure > 0.7:
            adjusted_priority *= 1.2
        
        # Increase priority for complex contexts
        if context.complexity_level > 0.8:
            adjusted_priority *= 1.1
        
        # Decrease priority if resources are limited
        if len(context.available_resources) < 3:
            adjusted_priority *= 0.9
        
        return min(1.0, adjusted_priority)
    
    async def _handle_goal_completion(self, goal: Goal, completion_data: Dict[str, Any]) -> None:
        """Handle goal completion"""
        logger.info(f"Goal completed: {goal.description}")
        
        # Record achievement
        achievement_record = {
            "goal_id": goal.id,
            "description": goal.description,
            "completion_time": datetime.now(),
            "completion_data": completion_data,
            "priority": goal.priority,
            "sub_goals_completed": len(goal.sub_goals)
        }
        
        self.goal_achievement_history.append(achievement_record)
        
        # Update parent goals if this was a sub-goal
        if "parent_goals" in goal.context:
            for parent_id in goal.context["parent_goals"]:
                if parent_id in self.goal_registry:
                    await self._update_parent_goal_progress(parent_id, goal.id)
    
    async def _update_sub_goal_progress(self, parent_goal: Goal, progress_data: Dict[str, Any]) -> None:
        """Update progress of sub-goals"""
        for sub_goal_id in parent_goal.sub_goals:
            if sub_goal_id in self.goal_registry:
                sub_goal = self.goal_registry[sub_goal_id]
                # Propagate relevant progress data to sub-goals
                sub_progress = {
                    "parent_progress": progress_data.get("completion_percentage", 0.0),
                    "inherited_context": progress_data.get("context", {})
                }
                sub_goal.context.update(sub_progress)
    
    async def _update_parent_goal_progress(self, parent_id: str, completed_sub_goal_id: str) -> None:
        """Update parent goal when sub-goal is completed"""
        if parent_id not in self.goal_registry:
            return
        
        parent_goal = self.goal_registry[parent_id]
        
        # Count completed sub-goals
        completed_sub_goals = 0
        for sub_goal_id in parent_goal.sub_goals:
            if sub_goal_id in self.goal_registry:
                sub_goal = self.goal_registry[sub_goal_id]
                if sub_goal.status == "completed":
                    completed_sub_goals += 1
        
        # Update parent progress
        if parent_goal.sub_goals:
            completion_percentage = completed_sub_goals / len(parent_goal.sub_goals)
            parent_goal.context["sub_goal_completion"] = completion_percentage
            
            # Mark parent as completed if all sub-goals are done
            if completion_percentage >= 1.0:
                parent_goal.status = "completed"
                await self._handle_goal_completion(parent_goal, {
                    "completion_percentage": 1.0,
                    "completion_method": "sub_goal_completion"
                })
    
    async def _suggest_next_actions(self, goal: Goal) -> List[str]:
        """Suggest next actions for a goal"""
        actions = []
        
        progress_score = goal.context.get("progress_score", 0.0)
        
        if progress_score < 0.3:
            actions.extend([
                "Define detailed implementation plan",
                "Identify required resources",
                "Set intermediate milestones"
            ])
        elif progress_score < 0.7:
            actions.extend([
                "Execute planned activities",
                "Monitor progress regularly",
                "Adjust approach if needed"
            ])
        else:
            actions.extend([
                "Complete final tasks",
                "Validate outcomes",
                "Prepare for goal closure"
            ])
        
        # Add sub-goal specific actions
        if goal.sub_goals:
            incomplete_sub_goals = [
                self.goal_registry[sg_id] for sg_id in goal.sub_goals
                if sg_id in self.goal_registry and self.goal_registry[sg_id].status != "completed"
            ]
            
            if incomplete_sub_goals:
                actions.append(f"Focus on {len(incomplete_sub_goals)} remaining sub-goals")
        
        return actions
    
    async def _determine_resolution_strategy(self, conflicting_goals: List[Goal]) -> str:
        """Determine strategy for resolving goal conflicts"""
        if len(conflicting_goals) <= 2:
            return "prioritize"
        
        # Check if goals can be merged
        descriptions = [g.description.lower() for g in conflicting_goals]
        if any(desc in other_desc for desc in descriptions for other_desc in descriptions if desc != other_desc):
            return "merge"
        
        # Check priority spread
        priorities = [g.priority for g in conflicting_goals]
        priority_spread = max(priorities) - min(priorities)
        
        if priority_spread > 0.5:
            return "prioritize"
        else:
            return "sequence"
    
    async def _prioritize_goals(self, goals: List[Goal]) -> List[Goal]:
        """Prioritize goals by importance"""
        return sorted(goals, key=lambda g: g.priority, reverse=True)
    
    async def _merge_goals(self, goals: List[Goal]) -> List[Goal]:
        """Merge similar goals into composite goals"""
        if len(goals) < 2:
            return goals
        
        # Simple merge: combine descriptions and take highest priority
        merged_description = " and ".join(g.description for g in goals)
        merged_priority = max(g.priority for g in goals)
        
        merged_goal = Goal(
            description=f"Combined: {merged_description}",
            priority=merged_priority,
            sub_goals=[g.id for g in goals],
            context={
                "merged_from": [g.id for g in goals],
                "merge_timestamp": datetime.now()
            }
        )
        
        return [merged_goal]
    
    async def _sequence_goals(self, goals: List[Goal]) -> List[Goal]:
        """Sequence goals in optimal order"""
        # Sort by priority and add sequence information
        sequenced_goals = sorted(goals, key=lambda g: g.priority, reverse=True)
        
        for i, goal in enumerate(sequenced_goals):
            goal.context["sequence_order"] = i
            goal.context["sequenced"] = True
        
        return sequenced_goals
    
    async def _generate_resolution_rationale(self, strategy: str, original_goals: List[Goal],
                                           resolved_goals: List[Goal]) -> str:
        """Generate rationale for conflict resolution"""
        rationales = {
            "prioritize": f"Prioritized {len(original_goals)} goals by importance and urgency",
            "merge": f"Merged {len(original_goals)} similar goals into {len(resolved_goals)} composite goals",
            "sequence": f"Sequenced {len(original_goals)} goals for optimal execution order"
        }
        
        return rationales.get(strategy, "Applied default resolution strategy")
    
    async def _adapt_goal_to_context(self, goal: Goal, new_context: CognitiveContext) -> Goal:
        """Adapt a goal to new context"""
        adapted_goal = Goal(
            id=goal.id,
            description=goal.description,
            priority=self._calculate_contextual_priority(goal.priority, new_context),
            status=goal.status,
            sub_goals=goal.sub_goals,
            context=goal.context.copy(),
            created_at=goal.created_at
        )
        
        # Update context with adaptation information
        adapted_goal.context.update({
            "adapted_to_context": new_context.situation,
            "adaptation_timestamp": datetime.now(),
            "original_priority": goal.priority,
            "context_complexity": new_context.complexity_level,
            "context_urgency": new_context.time_pressure
        })
        
        return adapted_goal
    
    async def _calculate_contextual_intention_strength(self, goal: Optional[Goal], 
                                                     context: CognitiveContext) -> float:
        """Calculate intention strength for specific context"""
        if not goal:
            return 0.0
        
        base_strength = goal.priority
        
        # Adjust for context factors
        context_multiplier = 1.0
        
        if context.time_pressure > 0.7:
            context_multiplier *= 1.2  # Urgency increases intention
        
        if context.complexity_level > 0.8:
            context_multiplier *= 0.9  # High complexity may reduce focus
        
        if len(context.available_resources) > 5:
            context_multiplier *= 1.1  # More resources increase confidence
        
        return min(1.0, base_strength * context_multiplier)
    
    async def _determine_contextual_focus(self, goal: Optional[Goal], 
                                        context: CognitiveContext) -> str:
        """Determine focus direction based on goal and context"""
        if not goal:
            return "exploratory"
        
        if context.time_pressure > 0.8:
            return "urgent_execution"
        elif context.complexity_level > 0.8:
            return "analytical_deep_dive"
        elif goal.priority > 0.8:
            return "high_priority_focus"
        else:
            return "balanced_approach"
    
    async def _calculate_contextual_commitment(self, primary_goal: Optional[Goal], 
                                             active_goals: List[Goal],
                                             context: CognitiveContext) -> float:
        """Calculate commitment level considering context"""
        if not primary_goal:
            return 0.0
        
        base_commitment = primary_goal.priority
        
        # Context adjustments
        if context.time_pressure > 0.7:
            base_commitment *= 1.1  # Urgency increases commitment
        
        if len(context.constraints) > 3:
            base_commitment *= 0.9  # Many constraints may reduce commitment
        
        if len(active_goals) > 5:
            base_commitment *= 0.8  # Too many goals may dilute commitment
        
        return min(1.0, base_commitment)