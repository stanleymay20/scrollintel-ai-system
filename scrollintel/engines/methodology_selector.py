"""
Experimental methodology selection and optimization system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta
from dataclasses import dataclass

from ..models.experimental_design_models import (
    MethodologyType, ExperimentType, MethodologyRecommendation,
    ResourceRequirement
)


@dataclass
class MethodologyProfile:
    """Profile of an experimental methodology."""
    methodology: MethodologyType
    experiment_types: List[ExperimentType]
    strengths: List[str]
    weaknesses: List[str]
    suitable_domains: List[str]
    resource_intensity: str  # low, medium, high
    complexity_level: str  # low, medium, high
    typical_duration: timedelta
    accuracy_potential: float
    reliability_potential: float
    cost_factor: float


class MethodologySelector:
    """
    System for selecting and optimizing experimental methodologies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.methodology_profiles = self._initialize_methodology_profiles()
        self.domain_preferences = self._initialize_domain_preferences()
        self.optimization_rules = self._initialize_optimization_rules()
    
    def select_optimal_methodology(
        self,
        research_context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> MethodologyRecommendation:
        """
        Select the optimal experimental methodology.
        
        Args:
            research_context: Context including domain, complexity, objectives
            constraints: Resource, time, or other constraints
            preferences: User preferences for methodology characteristics
            
        Returns:
            Optimal methodology recommendation
        """
        try:
            # Analyze research context
            context_analysis = self._analyze_research_context(research_context)
            
            # Generate candidate methodologies
            candidates = self._generate_methodology_candidates(
                context_analysis, constraints
            )
            
            # Score and rank candidates
            scored_candidates = self._score_methodologies(
                candidates, context_analysis, constraints, preferences
            )
            
            # Select optimal methodology
            optimal = self._select_optimal(scored_candidates)
            
            # Create detailed recommendation
            recommendation = self._create_methodology_recommendation(
                optimal, context_analysis, constraints
            )
            
            self.logger.info(
                f"Selected methodology: {recommendation.methodology.value} "
                f"with score: {recommendation.suitability_score:.3f}"
            )
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error selecting methodology: {str(e)}")
            raise
    
    def compare_methodologies(
        self,
        research_context: Dict[str, Any],
        methodologies: List[MethodologyType],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[MethodologyRecommendation]:
        """
        Compare multiple methodologies for the same research context.
        
        Args:
            research_context: Research context
            methodologies: List of methodologies to compare
            constraints: Optional constraints
            
        Returns:
            List of methodology recommendations for comparison
        """
        try:
            context_analysis = self._analyze_research_context(research_context)
            recommendations = []
            
            for methodology in methodologies:
                # Find suitable experiment types for this methodology
                suitable_types = self._find_suitable_experiment_types(
                    methodology, context_analysis
                )
                
                for exp_type in suitable_types:
                    # Calculate suitability score
                    score = self._calculate_methodology_score(
                        methodology, exp_type, context_analysis, constraints
                    )
                    
                    if score > 0.2:  # Only include viable options
                        recommendation = self._create_methodology_recommendation(
                            {
                                'methodology': methodology,
                                'experiment_type': exp_type,
                                'score': score
                            },
                            context_analysis,
                            constraints
                        )
                        recommendations.append(recommendation)
            
            # Sort by suitability score
            recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            self.logger.info(f"Compared {len(methodologies)} methodologies")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error comparing methodologies: {str(e)}")
            raise
    
    def optimize_methodology_selection(
        self,
        research_context: Dict[str, Any],
        optimization_criteria: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> MethodologyRecommendation:
        """
        Optimize methodology selection for specific criteria.
        
        Args:
            research_context: Research context
            optimization_criteria: Criteria to optimize (cost, time, accuracy, etc.)
            constraints: Optional constraints
            
        Returns:
            Optimized methodology recommendation
        """
        try:
            context_analysis = self._analyze_research_context(research_context)
            
            # Generate all viable methodologies
            candidates = self._generate_methodology_candidates(
                context_analysis, constraints
            )
            
            # Apply optimization criteria
            optimized_candidates = []
            for candidate in candidates:
                optimized_score = self._apply_optimization_criteria(
                    candidate, optimization_criteria, context_analysis
                )
                
                if optimized_score > 0.3:
                    candidate['optimized_score'] = optimized_score
                    optimized_candidates.append(candidate)
            
            # Select best optimized candidate
            if not optimized_candidates:
                raise ValueError("No viable methodologies found for optimization criteria")
            
            optimal = max(optimized_candidates, key=lambda x: x['optimized_score'])
            
            # Create recommendation with optimization details
            recommendation = self._create_optimized_recommendation(
                optimal, optimization_criteria, context_analysis, constraints
            )
            
            self.logger.info(
                f"Optimized methodology selection for criteria: {optimization_criteria}"
            )
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error optimizing methodology selection: {str(e)}")
            raise
    
    def _initialize_methodology_profiles(self) -> Dict[MethodologyType, MethodologyProfile]:
        """Initialize methodology profiles database."""
        profiles = {}
        
        # Quantitative methodology
        profiles[MethodologyType.QUANTITATIVE] = MethodologyProfile(
            methodology=MethodologyType.QUANTITATIVE,
            experiment_types=[
                ExperimentType.CONTROLLED, ExperimentType.RANDOMIZED,
                ExperimentType.FACTORIAL, ExperimentType.COMPARATIVE
            ],
            strengths=[
                "High statistical power",
                "Objective measurements",
                "Generalizable results",
                "Reproducible findings",
                "Large sample analysis"
            ],
            weaknesses=[
                "Limited contextual depth",
                "May miss nuanced effects",
                "Requires large samples",
                "Less flexible design"
            ],
            suitable_domains=[
                "engineering", "physics", "chemistry", "medicine",
                "psychology", "economics"
            ],
            resource_intensity="medium",
            complexity_level="medium",
            typical_duration=timedelta(weeks=8),
            accuracy_potential=0.85,
            reliability_potential=0.90,
            cost_factor=1.0
        )
        
        # Qualitative methodology
        profiles[MethodologyType.QUALITATIVE] = MethodologyProfile(
            methodology=MethodologyType.QUALITATIVE,
            experiment_types=[
                ExperimentType.OBSERVATIONAL, ExperimentType.LONGITUDINAL
            ],
            strengths=[
                "Rich contextual insights",
                "Flexible design",
                "Deep understanding",
                "Captures complexity",
                "Exploratory potential"
            ],
            weaknesses=[
                "Limited generalizability",
                "Subjective interpretation",
                "Time intensive analysis",
                "Smaller sample sizes"
            ],
            suitable_domains=[
                "social sciences", "anthropology", "user research",
                "organizational studies", "education"
            ],
            resource_intensity="high",
            complexity_level="high",
            typical_duration=timedelta(weeks=12),
            accuracy_potential=0.75,
            reliability_potential=0.70,
            cost_factor=1.3
        )
        
        # Mixed methods
        profiles[MethodologyType.MIXED_METHODS] = MethodologyProfile(
            methodology=MethodologyType.MIXED_METHODS,
            experiment_types=[
                ExperimentType.CONTROLLED, ExperimentType.LONGITUDINAL,
                ExperimentType.COMPARATIVE
            ],
            strengths=[
                "Comprehensive understanding",
                "Triangulation of results",
                "Balanced approach",
                "Addresses multiple questions",
                "Robust findings"
            ],
            weaknesses=[
                "Complex design",
                "Resource intensive",
                "Longer duration",
                "Integration challenges"
            ],
            suitable_domains=[
                "health sciences", "education", "business",
                "policy research", "evaluation studies"
            ],
            resource_intensity="high",
            complexity_level="high",
            typical_duration=timedelta(weeks=16),
            accuracy_potential=0.88,
            reliability_potential=0.85,
            cost_factor=1.5
        )
        
        # Computational methodology
        profiles[MethodologyType.COMPUTATIONAL] = MethodologyProfile(
            methodology=MethodologyType.COMPUTATIONAL,
            experiment_types=[
                ExperimentType.CONTROLLED, ExperimentType.FACTORIAL,
                ExperimentType.COMPARATIVE
            ],
            strengths=[
                "Precise control",
                "Scalable experiments",
                "Reproducible results",
                "Cost effective",
                "Rapid iteration"
            ],
            weaknesses=[
                "Limited real-world validity",
                "Model assumptions",
                "Computational constraints",
                "May oversimplify"
            ],
            suitable_domains=[
                "computer science", "artificial intelligence",
                "systems engineering", "operations research"
            ],
            resource_intensity="low",
            complexity_level="medium",
            typical_duration=timedelta(weeks=4),
            accuracy_potential=0.92,
            reliability_potential=0.95,
            cost_factor=0.6
        )
        
        # Simulation methodology
        profiles[MethodologyType.SIMULATION] = MethodologyProfile(
            methodology=MethodologyType.SIMULATION,
            experiment_types=[
                ExperimentType.CONTROLLED, ExperimentType.FACTORIAL
            ],
            strengths=[
                "Safe experimentation",
                "Cost effective",
                "Controllable conditions",
                "Rapid prototyping",
                "Risk-free testing"
            ],
            weaknesses=[
                "Model validity concerns",
                "Abstraction limitations",
                "Verification challenges",
                "Real-world gap"
            ],
            suitable_domains=[
                "engineering", "physics", "economics",
                "logistics", "manufacturing"
            ],
            resource_intensity="low",
            complexity_level="medium",
            typical_duration=timedelta(weeks=6),
            accuracy_potential=0.80,
            reliability_potential=0.88,
            cost_factor=0.7
        )
        
        return profiles
    
    def _initialize_domain_preferences(self) -> Dict[str, Dict[str, float]]:
        """Initialize domain-specific methodology preferences."""
        return {
            'engineering': {
                'quantitative': 0.9,
                'computational': 0.95,
                'simulation': 0.85,
                'mixed_methods': 0.7,
                'qualitative': 0.3
            },
            'social_sciences': {
                'qualitative': 0.95,
                'mixed_methods': 0.9,
                'quantitative': 0.6,
                'computational': 0.3,
                'simulation': 0.2
            },
            'computer_science': {
                'computational': 0.95,
                'simulation': 0.9,
                'quantitative': 0.8,
                'mixed_methods': 0.5,
                'qualitative': 0.3
            },
            'medicine': {
                'quantitative': 0.95,
                'mixed_methods': 0.8,
                'qualitative': 0.6,
                'computational': 0.7,
                'simulation': 0.5
            }
        }
    
    def _initialize_optimization_rules(self) -> Dict[str, Dict[str, float]]:
        """Initialize optimization rules for different criteria."""
        return {
            'cost': {
                'computational': 1.0,
                'simulation': 0.9,
                'quantitative': 0.7,
                'qualitative': 0.4,
                'mixed_methods': 0.3
            },
            'time': {
                'computational': 1.0,
                'simulation': 0.8,
                'quantitative': 0.6,
                'mixed_methods': 0.4,
                'qualitative': 0.3
            },
            'accuracy': {
                'computational': 1.0,
                'mixed_methods': 0.9,
                'quantitative': 0.85,
                'simulation': 0.8,
                'qualitative': 0.7
            },
            'reliability': {
                'computational': 1.0,
                'quantitative': 0.9,
                'simulation': 0.85,
                'mixed_methods': 0.8,
                'qualitative': 0.6
            }
        }
    
    def _analyze_research_context(self, research_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research context for methodology selection."""
        analysis = {
            'domain': research_context.get('domain', 'general'),
            'complexity': research_context.get('complexity', 'medium'),
            'sample_size': research_context.get('sample_size', 100),
            'objectives': research_context.get('objectives', ['hypothesis_testing']),
            'data_types': research_context.get('data_types', ['quantitative']),
            'timeline': research_context.get('timeline', timedelta(weeks=8)),
            'budget': research_context.get('budget', 10000.0)
        }
        return analysis
    
    def _generate_methodology_candidates(
        self,
        context_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate candidate methodologies based on context."""
        candidates = []
        
        for methodology_type, profile in self.methodology_profiles.items():
            # Check domain suitability - be more flexible with domain matching
            domain_match = (
                context_analysis['domain'] in profile.suitable_domains or
                any(domain in context_analysis['domain'] for domain in profile.suitable_domains) or
                any(context_analysis['domain'] in domain for domain in profile.suitable_domains)
            )
            
            if domain_match:
                for exp_type in profile.experiment_types:
                    # Check constraints
                    if self._meets_constraints(profile, constraints):
                        candidate = {
                            'methodology': methodology_type,
                            'experiment_type': exp_type,
                            'profile': profile
                        }
                        candidates.append(candidate)
        
        # If no candidates found, add all methodologies as fallback
        if not candidates:
            for methodology_type, profile in self.methodology_profiles.items():
                for exp_type in profile.experiment_types:
                    if self._meets_constraints(profile, constraints):
                        candidate = {
                            'methodology': methodology_type,
                            'experiment_type': exp_type,
                            'profile': profile
                        }
                        candidates.append(candidate)
        
        return candidates
    
    def _score_methodologies(
        self,
        candidates: List[Dict[str, Any]],
        context_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]],
        preferences: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score methodology candidates."""
        scored_candidates = []
        
        for candidate in candidates:
            score = self._calculate_methodology_score(
                candidate['methodology'],
                candidate['experiment_type'],
                context_analysis,
                constraints,
                preferences
            )
            
            candidate['score'] = score
            scored_candidates.append(candidate)
        
        return scored_candidates
    
    def _calculate_methodology_score(
        self,
        methodology: MethodologyType,
        experiment_type: ExperimentType,
        context_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate suitability score for methodology."""
        profile = self.methodology_profiles[methodology]
        score = 0.0
        
        # Domain suitability (40% weight)
        domain_score = 0.5  # Default score
        
        # Check exact domain match first
        if context_analysis['domain'] in profile.suitable_domains:
            domain_preferences = self.domain_preferences.get(
                context_analysis['domain'], {}
            )
            domain_score = domain_preferences.get(methodology.value, 0.7)
        else:
            # Check if we have specific preferences for this domain
            domain_preferences = self.domain_preferences.get(
                context_analysis['domain'], {}
            )
            if domain_preferences:
                domain_score = domain_preferences.get(methodology.value, 0.3)
            else:
                # Check partial domain matches
                for suitable_domain in profile.suitable_domains:
                    if (suitable_domain in context_analysis['domain'] or 
                        context_analysis['domain'] in suitable_domain):
                        domain_preferences = self.domain_preferences.get(
                            suitable_domain, {}
                        )
                        domain_score = max(domain_score, 
                                         domain_preferences.get(methodology.value, 0.5))
        
        score += 0.4 * domain_score
        
        # Resource constraints (20% weight)
        resource_score = self._calculate_resource_score(profile, constraints)
        score += 0.2 * resource_score
        
        # Accuracy and reliability (25% weight)
        quality_score = (profile.accuracy_potential + profile.reliability_potential) / 2
        score += 0.25 * quality_score
        
        # User preferences (15% weight)
        preference_score = self._calculate_preference_score(
            methodology, preferences
        )
        score += 0.15 * preference_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_resource_score(
        self,
        profile: MethodologyProfile,
        constraints: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate resource constraint score."""
        if not constraints:
            return 0.8  # Default good score
        
        score = 1.0
        
        # Budget constraint
        if 'budget' in constraints:
            budget_ratio = constraints['budget'] / (10000.0 * profile.cost_factor)
            if budget_ratio < 0.5:
                score *= 0.3
            elif budget_ratio < 1.0:
                score *= 0.7
        
        # Time constraint
        if 'timeline' in constraints:
            time_ratio = constraints['timeline'] / profile.typical_duration
            if time_ratio < 0.5:
                score *= 0.2
            elif time_ratio < 1.0:
                score *= 0.6
        
        return score
    
    def _calculate_preference_score(
        self,
        methodology: MethodologyType,
        preferences: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate user preference score."""
        if not preferences:
            return 0.5  # Neutral score
        
        methodology_preferences = preferences.get('methodologies', {})
        return methodology_preferences.get(methodology.value, 0.5)
    
    def _meets_constraints(
        self,
        profile: MethodologyProfile,
        constraints: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if methodology meets hard constraints."""
        if not constraints:
            return True
        
        # Hard budget constraint
        if 'max_budget' in constraints:
            estimated_cost = 10000.0 * profile.cost_factor
            if estimated_cost > constraints['max_budget']:
                return False
        
        # Hard time constraint
        if 'max_timeline' in constraints:
            if profile.typical_duration > constraints['max_timeline']:
                return False
        
        return True
    
    def _select_optimal(self, scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select optimal methodology from scored candidates."""
        if not scored_candidates:
            raise ValueError("No viable methodology candidates found")
        
        return max(scored_candidates, key=lambda x: x['score'])
    
    def _create_methodology_recommendation(
        self,
        optimal: Dict[str, Any],
        context_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> MethodologyRecommendation:
        """Create detailed methodology recommendation."""
        methodology = optimal['methodology']
        experiment_type = optimal['experiment_type']
        profile = self.methodology_profiles[methodology]
        
        # Estimate resources
        resources = self._estimate_methodology_resources(
            profile, context_analysis, constraints
        )
        
        recommendation = MethodologyRecommendation(
            methodology=methodology,
            experiment_type=experiment_type,
            suitability_score=optimal['score'],
            advantages=profile.strengths,
            disadvantages=profile.weaknesses,
            resource_requirements=resources,
            estimated_duration=profile.typical_duration,
            confidence_level=profile.reliability_potential
        )
        
        return recommendation
    
    def _estimate_methodology_resources(
        self,
        profile: MethodologyProfile,
        context_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> List[ResourceRequirement]:
        """Estimate resource requirements for methodology."""
        resources = []
        
        # Personnel requirements
        if profile.resource_intensity == "high":
            personnel_count = 3
        elif profile.resource_intensity == "medium":
            personnel_count = 2
        else:
            personnel_count = 1
        
        personnel = ResourceRequirement(
            resource_type="personnel",
            resource_name="Research Staff",
            quantity_needed=personnel_count,
            duration_needed=profile.typical_duration,
            cost_estimate=personnel_count * 2000.0
        )
        resources.append(personnel)
        
        # Equipment requirements
        if profile.methodology in [MethodologyType.COMPUTATIONAL, MethodologyType.SIMULATION]:
            equipment = ResourceRequirement(
                resource_type="computational",
                resource_name="Computing Resources",
                quantity_needed=1,
                duration_needed=profile.typical_duration,
                cost_estimate=500.0 * profile.cost_factor
            )
        else:
            equipment = ResourceRequirement(
                resource_type="equipment",
                resource_name="Research Equipment",
                quantity_needed=1,
                duration_needed=profile.typical_duration,
                cost_estimate=1000.0 * profile.cost_factor
            )
        resources.append(equipment)
        
        return resources
    
    def _find_suitable_experiment_types(
        self,
        methodology: MethodologyType,
        context_analysis: Dict[str, Any]
    ) -> List[ExperimentType]:
        """Find suitable experiment types for methodology."""
        profile = self.methodology_profiles[methodology]
        return profile.experiment_types
    
    def _apply_optimization_criteria(
        self,
        candidate: Dict[str, Any],
        optimization_criteria: List[str],
        context_analysis: Dict[str, Any]
    ) -> float:
        """Apply optimization criteria to candidate."""
        methodology = candidate['methodology']
        base_score = candidate.get('score', 0.5)
        
        optimization_score = 0.0
        for criterion in optimization_criteria:
            if criterion in self.optimization_rules:
                criterion_score = self.optimization_rules[criterion].get(
                    methodology.value, 0.5
                )
                optimization_score += criterion_score
        
        # Average optimization score
        if optimization_criteria:
            optimization_score /= len(optimization_criteria)
        else:
            optimization_score = 0.5
        
        # Combine base score with optimization score
        final_score = 0.6 * base_score + 0.4 * optimization_score
        return final_score
    
    def _create_optimized_recommendation(
        self,
        optimal: Dict[str, Any],
        optimization_criteria: List[str],
        context_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> MethodologyRecommendation:
        """Create optimized methodology recommendation."""
        # Add the optimized score as the regular score for the recommendation creation
        optimal['score'] = optimal['optimized_score']
        
        recommendation = self._create_methodology_recommendation(
            optimal, context_analysis, constraints
        )
        
        # Add optimization advantages
        optimization_advantages = []
        for criterion in optimization_criteria:
            optimization_advantages.append(f"Optimized for {criterion}")
        
        recommendation.advantages.extend(optimization_advantages)
        
        return recommendation