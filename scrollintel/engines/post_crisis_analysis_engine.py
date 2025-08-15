"""
Post-Crisis Analysis Engine

Comprehensive crisis response analysis and evaluation system.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

from ..models.post_crisis_analysis_models import (
    PostCrisisAnalysis, AnalysisType, LessonLearned, LessonCategory,
    ImprovementRecommendation, RecommendationPriority, CrisisMetric,
    AnalysisReport
)
from ..models.crisis_detection_models import CrisisModel


class PostCrisisAnalysisEngine:
    """Engine for comprehensive post-crisis analysis and evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_templates = self._initialize_analysis_templates()
        self.metric_calculators = self._initialize_metric_calculators()
    
    def conduct_comprehensive_analysis(
        self, 
        crisis: CrisisModel,
        response_data: Dict[str, Any],
        analyst_id: str
    ) -> PostCrisisAnalysis:
        """Conduct comprehensive crisis response analysis"""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Calculate performance metrics
            response_metrics = self._calculate_response_metrics(crisis, response_data)
            overall_score = self._calculate_overall_performance_score(response_metrics)
            
            # Identify strengths and weaknesses
            strengths = self._identify_strengths(crisis, response_data, response_metrics)
            weaknesses = self._identify_weaknesses(crisis, response_data, response_metrics)
            
            # Extract lessons learned
            lessons = self._extract_lessons_learned(crisis, response_data, strengths, weaknesses)
            
            # Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(lessons, weaknesses)
            
            # Assess impacts
            stakeholder_impact = self._assess_stakeholder_impact(crisis, response_data)
            business_impact = self._assess_business_impact(crisis, response_data)
            reputation_impact = self._assess_reputation_impact(crisis, response_data)
            
            analysis = PostCrisisAnalysis(
                id=analysis_id,
                crisis_id=crisis.id,
                analysis_type=AnalysisType.RESPONSE_EFFECTIVENESS,
                analyst_id=analyst_id,
                analysis_date=datetime.now(),
                crisis_summary=self._generate_crisis_summary(crisis),
                crisis_duration=self._calculate_crisis_duration(crisis),
                crisis_severity=crisis.severity_level.value,
                response_metrics=response_metrics,
                overall_performance_score=overall_score,
                strengths_identified=strengths,
                weaknesses_identified=weaknesses,
                lessons_learned=lessons,
                improvement_recommendations=recommendations,
                stakeholder_impact=stakeholder_impact,
                business_impact=business_impact,
                reputation_impact=reputation_impact,
                analysis_methodology="Comprehensive Multi-Factor Analysis",
                data_sources=self._identify_data_sources(response_data),
                confidence_level=self._calculate_confidence_level(response_data),
                review_status="pending_review"
            )
            
            self.logger.info(f"Completed comprehensive analysis for crisis {crisis.id}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error conducting crisis analysis: {str(e)}")
            raise
    
    def identify_lessons_learned(
        self, 
        crisis: CrisisModel,
        response_data: Dict[str, Any]
    ) -> List[LessonLearned]:
        """Identify and document lessons learned from crisis response"""
        try:
            lessons = []
            
            # Analyze decision-making patterns
            decision_lessons = self._analyze_decision_patterns(crisis, response_data)
            lessons.extend(decision_lessons)
            
            # Analyze communication effectiveness
            communication_lessons = self._analyze_communication_patterns(crisis, response_data)
            lessons.extend(communication_lessons)
            
            # Analyze resource utilization
            resource_lessons = self._analyze_resource_utilization(crisis, response_data)
            lessons.extend(resource_lessons)
            
            # Analyze team coordination
            team_lessons = self._analyze_team_coordination(crisis, response_data)
            lessons.extend(team_lessons)
            
            # Validate and prioritize lessons
            validated_lessons = self._validate_lessons(lessons)
            
            self.logger.info(f"Identified {len(validated_lessons)} lessons learned")
            return validated_lessons
            
        except Exception as e:
            self.logger.error(f"Error identifying lessons learned: {str(e)}")
            raise
    
    def generate_improvement_recommendations(
        self, 
        lessons_learned: List[LessonLearned]
    ) -> List[ImprovementRecommendation]:
        """Generate actionable improvement recommendations"""
        try:
            recommendations = []
            
            for lesson in lessons_learned:
                # Generate specific recommendations for each lesson
                lesson_recommendations = self._generate_lesson_recommendations(lesson)
                recommendations.extend(lesson_recommendations)
            
            # Prioritize recommendations
            prioritized_recommendations = self._prioritize_recommendations(recommendations)
            
            # Add implementation details
            detailed_recommendations = self._add_implementation_details(prioritized_recommendations)
            
            self.logger.info(f"Generated {len(detailed_recommendations)} improvement recommendations")
            return detailed_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def generate_analysis_report(
        self, 
        analysis: PostCrisisAnalysis,
        report_format: str = "comprehensive"
    ) -> AnalysisReport:
        """Generate formatted analysis report"""
        try:
            report_title = f"Post-Crisis Analysis Report - {analysis.crisis_id}"
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(analysis)
            
            # Generate detailed findings
            detailed_findings = self._generate_detailed_findings(analysis)
            
            # Generate recommendations summary
            recommendations_summary = self._generate_recommendations_summary(analysis)
            
            # Generate appendices
            appendices = self._generate_appendices(analysis)
            
            report = AnalysisReport(
                analysis_id=analysis.id,
                report_title=report_title,
                executive_summary=executive_summary,
                detailed_findings=detailed_findings,
                recommendations_summary=recommendations_summary,
                appendices=appendices,
                generated_date=datetime.now(),
                report_format=report_format
            )
            
            self.logger.info(f"Generated analysis report for {analysis.id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {str(e)}")
            raise
    
    def _calculate_response_metrics(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any]
    ) -> List[CrisisMetric]:
        """Calculate crisis response performance metrics"""
        metrics = []
        
        # Response time metric
        if 'response_time' in response_data:
            response_time_metric = CrisisMetric(
                metric_name="Response Time",
                target_value=30.0,  # 30 minutes target
                actual_value=response_data['response_time'],
                variance=response_data['response_time'] - 30.0,
                performance_score=max(0, 100 - (response_data['response_time'] - 30) * 2),
                measurement_unit="minutes"
            )
            metrics.append(response_time_metric)
        
        # Communication effectiveness metric
        if 'communication_score' in response_data:
            comm_metric = CrisisMetric(
                metric_name="Communication Effectiveness",
                target_value=90.0,
                actual_value=response_data['communication_score'],
                variance=response_data['communication_score'] - 90.0,
                performance_score=response_data['communication_score'],
                measurement_unit="percentage"
            )
            metrics.append(comm_metric)
        
        # Resource utilization metric
        if 'resource_efficiency' in response_data:
            resource_metric = CrisisMetric(
                metric_name="Resource Utilization Efficiency",
                target_value=85.0,
                actual_value=response_data['resource_efficiency'],
                variance=response_data['resource_efficiency'] - 85.0,
                performance_score=response_data['resource_efficiency'],
                measurement_unit="percentage"
            )
            metrics.append(resource_metric)
        
        return metrics
    
    def _calculate_overall_performance_score(self, metrics: List[CrisisMetric]) -> float:
        """Calculate overall crisis response performance score"""
        if not metrics:
            return 0.0
        
        total_score = sum(metric.performance_score for metric in metrics)
        return total_score / len(metrics)
    
    def _identify_strengths(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any], 
        metrics: List[CrisisMetric]
    ) -> List[str]:
        """Identify strengths in crisis response"""
        strengths = []
        
        # Check high-performing metrics
        for metric in metrics:
            if metric.performance_score >= 80:
                strengths.append(f"Excellent {metric.metric_name.lower()}")
        
        # Check specific response aspects
        if response_data.get('stakeholder_satisfaction', 0) >= 80:
            strengths.append("High stakeholder satisfaction maintained")
        
        if response_data.get('team_coordination_score', 0) >= 85:
            strengths.append("Effective team coordination and collaboration")
        
        return strengths
    
    def _identify_weaknesses(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any], 
        metrics: List[CrisisMetric]
    ) -> List[str]:
        """Identify weaknesses in crisis response"""
        weaknesses = []
        
        # Check low-performing metrics
        for metric in metrics:
            if metric.performance_score < 60:
                weaknesses.append(f"Poor {metric.metric_name.lower()}")
        
        # Check specific response issues
        if response_data.get('communication_delays', 0) > 0:
            weaknesses.append("Communication delays impacted response effectiveness")
        
        if response_data.get('resource_shortages', 0) > 0:
            weaknesses.append("Resource shortages hindered optimal response")
        
        return weaknesses
    
    def _extract_lessons_learned(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any], 
        strengths: List[str], 
        weaknesses: List[str]
    ) -> List[LessonLearned]:
        """Extract specific lessons learned from crisis response"""
        lessons = []
        
        # Generate lessons from weaknesses
        for weakness in weaknesses:
            lesson = LessonLearned(
                id=str(uuid.uuid4()),
                crisis_id=crisis.id,
                category=LessonCategory.PROCESS_IMPROVEMENT,
                title=f"Improvement needed: {weakness}",
                description=f"Analysis revealed {weakness} during crisis response",
                root_cause="To be determined through detailed investigation",
                impact_assessment="Medium impact on response effectiveness",
                evidence=[weakness],
                identified_by="Post-Crisis Analysis Engine",
                identification_date=datetime.now(),
                validation_status="pending"
            )
            lessons.append(lesson)
        
        return lessons
    
    def _generate_improvement_recommendations(
        self, 
        lessons: List[LessonLearned], 
        weaknesses: List[str]
    ) -> List[ImprovementRecommendation]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        for lesson in lessons:
            recommendation = ImprovementRecommendation(
                id=str(uuid.uuid4()),
                lesson_id=lesson.id,
                title=f"Address: {lesson.title}",
                description=f"Implement improvements to address {lesson.description}",
                priority=RecommendationPriority.HIGH,
                implementation_effort="Medium",
                expected_impact="Improved crisis response effectiveness",
                success_metrics=["Response time improvement", "Stakeholder satisfaction"],
                responsible_team="Crisis Response Team",
                target_completion=datetime.now() + timedelta(days=90),
                implementation_status="not_started"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _assess_stakeholder_impact(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess impact on stakeholders"""
        return {
            "customers": response_data.get('customer_impact', "Medium"),
            "employees": response_data.get('employee_impact', "Low"),
            "investors": response_data.get('investor_impact', "Medium"),
            "partners": response_data.get('partner_impact', "Low"),
            "overall_satisfaction": response_data.get('stakeholder_satisfaction', 75)
        }
    
    def _assess_business_impact(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess business impact of crisis"""
        return {
            "revenue_impact": response_data.get('revenue_impact', 0),
            "operational_disruption": response_data.get('operational_disruption', "Medium"),
            "recovery_time": response_data.get('recovery_time', 0),
            "cost_of_response": response_data.get('response_cost', 0)
        }
    
    def _assess_reputation_impact(
        self, 
        crisis: CrisisModel, 
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess reputation impact of crisis"""
        return {
            "media_sentiment": response_data.get('media_sentiment', "Neutral"),
            "social_media_impact": response_data.get('social_impact', "Low"),
            "brand_perception_change": response_data.get('brand_impact', 0),
            "recovery_timeline": response_data.get('reputation_recovery_time', 30)
        }
    
    def _initialize_analysis_templates(self) -> Dict[str, Any]:
        """Initialize analysis templates"""
        return {
            "comprehensive": "Full multi-factor analysis template",
            "focused": "Targeted analysis template",
            "rapid": "Quick assessment template"
        }
    
    def _initialize_metric_calculators(self) -> Dict[str, Any]:
        """Initialize metric calculation functions"""
        return {
            "response_time": self._calculate_response_time,
            "communication_effectiveness": self._calculate_communication_effectiveness,
            "resource_efficiency": self._calculate_resource_efficiency
        }
    
    def _calculate_response_time(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> float:
        """Calculate crisis response time"""
        return response_data.get('response_time', 0)
    
    def _calculate_communication_effectiveness(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> float:
        """Calculate communication effectiveness score"""
        return response_data.get('communication_score', 0)
    
    def _calculate_resource_efficiency(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> float:
        """Calculate resource utilization efficiency"""
        return response_data.get('resource_efficiency', 0)
    
    def _generate_crisis_summary(self, crisis: CrisisModel) -> str:
        """Generate crisis summary"""
        return f"Crisis {crisis.id}: {crisis.crisis_type.value} - {crisis.severity_level.value}"
    
    def _calculate_crisis_duration(self, crisis: CrisisModel) -> float:
        """Calculate crisis duration in hours"""
        if crisis.resolution_time and crisis.start_time:
            duration = crisis.resolution_time - crisis.start_time
            return duration.total_seconds() / 3600
        return 0.0
    
    def _identify_data_sources(self, response_data: Dict[str, Any]) -> List[str]:
        """Identify data sources used in analysis"""
        return ["Crisis logs", "Response metrics", "Stakeholder feedback", "Performance data"]
    
    def _calculate_confidence_level(self, response_data: Dict[str, Any]) -> float:
        """Calculate analysis confidence level"""
        data_completeness = len(response_data) / 10  # Assume 10 key data points
        return min(100, data_completeness * 100)
    
    def _analyze_decision_patterns(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> List[LessonLearned]:
        """Analyze decision-making patterns"""
        return []  # Placeholder for decision pattern analysis
    
    def _analyze_communication_patterns(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> List[LessonLearned]:
        """Analyze communication patterns"""
        return []  # Placeholder for communication pattern analysis
    
    def _analyze_resource_utilization(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> List[LessonLearned]:
        """Analyze resource utilization patterns"""
        return []  # Placeholder for resource utilization analysis
    
    def _analyze_team_coordination(self, crisis: CrisisModel, response_data: Dict[str, Any]) -> List[LessonLearned]:
        """Analyze team coordination patterns"""
        return []  # Placeholder for team coordination analysis
    
    def _validate_lessons(self, lessons: List[LessonLearned]) -> List[LessonLearned]:
        """Validate and filter lessons learned"""
        return lessons  # Placeholder for lesson validation
    
    def _generate_lesson_recommendations(self, lesson: LessonLearned) -> List[ImprovementRecommendation]:
        """Generate recommendations for specific lesson"""
        return []  # Placeholder for lesson-specific recommendations
    
    def _prioritize_recommendations(self, recommendations: List[ImprovementRecommendation]) -> List[ImprovementRecommendation]:
        """Prioritize improvement recommendations"""
        return sorted(recommendations, key=lambda x: x.priority.value)
    
    def _add_implementation_details(self, recommendations: List[ImprovementRecommendation]) -> List[ImprovementRecommendation]:
        """Add implementation details to recommendations"""
        return recommendations  # Placeholder for implementation details
    
    def _generate_executive_summary(self, analysis: PostCrisisAnalysis) -> str:
        """Generate executive summary"""
        return f"Crisis {analysis.crisis_id} analysis completed with overall score {analysis.overall_performance_score:.1f}%"
    
    def _generate_detailed_findings(self, analysis: PostCrisisAnalysis) -> str:
        """Generate detailed findings section"""
        return f"Detailed analysis of {len(analysis.response_metrics)} key metrics with {len(analysis.lessons_learned)} lessons identified"
    
    def _generate_recommendations_summary(self, analysis: PostCrisisAnalysis) -> str:
        """Generate recommendations summary"""
        return f"{len(analysis.improvement_recommendations)} improvement recommendations generated"
    
    def _generate_appendices(self, analysis: PostCrisisAnalysis) -> List[str]:
        """Generate report appendices"""
        return ["Detailed metrics", "Raw data", "Methodology notes"]