#!/usr/bin/env python3
"""
ScrollIntel Launch Report Generator
Comprehensive launch analysis and reporting system
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LaunchMetric:
    name: str
    target: float
    actual: float
    unit: str
    status: str
    variance_percentage: float

class LaunchReportGenerator:
    """Generates comprehensive launch reports and analysis"""
    
    def __init__(self):
        self.launch_date = datetime(2025, 8, 22)
        self.report_date = datetime.now()
        self.success_criteria = self._load_success_criteria()
        
    def _load_success_criteria(self) -> Dict[str, Any]:
        """Load success criteria from configuration"""
        return {
            "technical_success_metrics": {
                "system_uptime": {"target": 99.9, "unit": "%"},
                "response_time": {"target": 2.0, "unit": "seconds"},
                "file_processing": {"target": 30.0, "unit": "seconds"},
                "concurrent_users": {"target": 100, "unit": "users"},
                "error_rate": {"target": 0.1, "unit": "%"}
            },
            "user_experience_metrics": {
                "onboarding_completion_rate": {"target": 80.0, "unit": "%"},
                "user_activation_rate": {"target": 60.0, "unit": "%"},
                "user_satisfaction_score": {"target": 4.5, "unit": "/5"},
                "support_ticket_resolution_time": {"target": 24.0, "unit": "hours"},
                "feature_adoption_rate": {"target": 50.0, "unit": "%"}
            },
            "business_metrics": {
                "launch_day_signups": {"target": 100, "unit": "signups"},
                "week_1_paying_customers": {"target": 10, "unit": "customers"},
                "month_1_revenue": {"target": 1000.0, "unit": "USD"},
                "customer_acquisition_cost": {"target": 100.0, "unit": "USD"},
                "monthly_recurring_revenue_growth": {"target": 20.0, "unit": "%"}
            }
        }
    
    def generate_comprehensive_launch_report(self) -> Dict[str, Any]:
        """Generate comprehensive launch report"""
        logger.info("📊 Generating comprehensive launch report...")
        
        # Collect all metrics data
        actual_metrics = self._collect_actual_metrics()
        
        # Calculate performance against targets
        performance_analysis = self._analyze_performance(actual_metrics)
        
        # Generate insights and recommendations
        insights = self._generate_insights(performance_analysis)
        
        # Create visualizations
        visualizations = self._create_visualizations(performance_analysis)
        
        # Compile comprehensive report
        report = {
            "report_metadata": {
                "generated_at": self.report_date.isoformat(),
                "launch_date": self.launch_date.isoformat(),
                "days_since_launch": (self.report_date - self.launch_date).days,
                "report_type": "Comprehensive Launch Analysis",
                "version": "1.0"
            },
            "executive_summary": self._create_executive_summary(performance_analysis),
            "performance_analysis": performance_analysis,
            "detailed_metrics": self._create_detailed_metrics_analysis(actual_metrics),
            "user_feedback_analysis": self._analyze_user_feedback(),
            "competitive_analysis": self._create_competitive_analysis(),
            "financial_performance": self._analyze_financial_performance(actual_metrics),
            "technical_performance": self._analyze_technical_performance(actual_metrics),
            "user_experience_analysis": self._analyze_user_experience(actual_metrics),
            "market_response": self._analyze_market_response(),
            "lessons_learned": self._capture_lessons_learned(),
            "recommendations": insights["recommendations"],
            "next_steps": self._define_next_steps(performance_analysis),
            "risk_assessment": self._assess_risks(performance_analysis),
            "success_factors": self._identify_success_factors(performance_analysis),
            "improvement_opportunities": insights["improvement_opportunities"],
            "visualizations": visualizations
        }
        
        # Save report
        self._save_report(report)
        
        logger.info("✅ Comprehensive launch report generated successfully")
        return report
    
    def _collect_actual_metrics(self) -> Dict[str, float]:
        """Collect actual metrics data (mock data for demonstration)"""
        # In production, this would collect real metrics from various sources
        return {
            # Technical Metrics
            "system_uptime": 99.95,
            "response_time": 1.2,
            "file_processing": 25.0,
            "concurrent_users": 150,
            "error_rate": 0.05,
            
            # User Experience Metrics
            "onboarding_completion_rate": 85.0,
            "user_activation_rate": 65.0,
            "user_satisfaction_score": 4.6,
            "support_ticket_resolution_time": 18.0,
            "feature_adoption_rate": 55.0,
            
            # Business Metrics
            "launch_day_signups": 125,
            "week_1_paying_customers": 15,
            "month_1_revenue": 1250.0,
            "customer_acquisition_cost": 85.0,
            "monthly_recurring_revenue_growth": 25.0
        }
    
    def _analyze_performance(self, actual_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance against targets"""
        performance_analysis = {
            "overall_performance": {},
            "category_performance": {},
            "individual_metrics": {}
        }
        
        total_metrics = 0
        successful_metrics = 0
        
        for category, metrics in self.success_criteria.items():
            category_successful = 0
            category_total = len(metrics)
            category_metrics = []
            
            for metric_name, config in metrics.items():
                if metric_name in actual_metrics:
                    actual_value = actual_metrics[metric_name]
                    target_value = config["target"]
                    
                    # Calculate variance
                    if target_value != 0:
                        variance = ((actual_value - target_value) / target_value) * 100
                    else:
                        variance = 0
                    
                    # Determine status
                    if metric_name in ["response_time", "file_processing", "support_ticket_resolution_time", 
                                     "customer_acquisition_cost", "error_rate"]:
                        # Lower is better
                        status = "success" if actual_value <= target_value else "needs_improvement"
                    else:
                        # Higher is better
                        status = "success" if actual_value >= target_value else "needs_improvement"
                    
                    if status == "success":
                        successful_metrics += 1
                        category_successful += 1
                    
                    total_metrics += 1
                    
                    metric_analysis = LaunchMetric(
                        name=metric_name,
                        target=target_value,
                        actual=actual_value,
                        unit=config["unit"],
                        status=status,
                        variance_percentage=variance
                    )
                    
                    category_metrics.append(metric_analysis)
                    performance_analysis["individual_metrics"][metric_name] = {
                        "target": target_value,
                        "actual": actual_value,
                        "unit": config["unit"],
                        "status": status,
                        "variance_percentage": variance
                    }
            
            # Category performance
            category_success_rate = (category_successful / category_total) * 100
            performance_analysis["category_performance"][category] = {
                "success_rate": category_success_rate,
                "successful_metrics": category_successful,
                "total_metrics": category_total,
                "metrics": category_metrics
            }
        
        # Overall performance
        overall_success_rate = (successful_metrics / total_metrics) * 100
        performance_analysis["overall_performance"] = {
            "success_rate": overall_success_rate,
            "successful_metrics": successful_metrics,
            "total_metrics": total_metrics,
            "status": self._determine_overall_status(overall_success_rate)
        }
        
        return performance_analysis
    
    def _determine_overall_status(self, success_rate: float) -> str:
        """Determine overall launch status"""
        if success_rate >= 90:
            return "Exceptional Success"
        elif success_rate >= 80:
            return "Strong Success"
        elif success_rate >= 70:
            return "Moderate Success"
        elif success_rate >= 60:
            return "Mixed Results"
        else:
            return "Needs Significant Improvement"
    
    def _create_executive_summary(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary"""
        overall_perf = performance_analysis["overall_performance"]
        
        return {
            "launch_status": overall_perf["status"],
            "overall_success_rate": overall_perf["success_rate"],
            "key_achievements": [
                f"Achieved {overall_perf['successful_metrics']} out of {overall_perf['total_metrics']} success metrics",
                "System maintained 99.95% uptime during launch",
                "User satisfaction score exceeded target at 4.6/5",
                "Launch day signups exceeded target by 25%",
                "Customer acquisition cost 15% below target"
            ],
            "areas_for_improvement": [
                "Feature adoption rate needs optimization",
                "Support response time optimization required",
                "User onboarding flow refinement needed"
            ],
            "financial_highlights": {
                "revenue_performance": "125% of target",
                "customer_acquisition": "150% of target",
                "cost_efficiency": "15% better than target"
            },
            "technical_highlights": {
                "system_reliability": "Exceeded uptime target",
                "performance": "40% faster than target response time",
                "scalability": "Handled 50% more concurrent users than target"
            },
            "user_experience_highlights": {
                "satisfaction": "Exceeded target satisfaction score",
                "onboarding": "5% above target completion rate",
                "activation": "8% above target activation rate"
            }
        }
    
    def _create_detailed_metrics_analysis(self, actual_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create detailed analysis of each metric"""
        detailed_analysis = {}
        
        for category, metrics in self.success_criteria.items():
            category_analysis = {}
            
            for metric_name, config in metrics.items():
                if metric_name in actual_metrics:
                    actual_value = actual_metrics[metric_name]
                    target_value = config["target"]
                    
                    analysis = {
                        "metric_name": metric_name.replace("_", " ").title(),
                        "target": target_value,
                        "actual": actual_value,
                        "unit": config["unit"],
                        "performance": "Above Target" if actual_value >= target_value else "Below Target",
                        "variance": actual_value - target_value,
                        "variance_percentage": ((actual_value - target_value) / target_value) * 100 if target_value != 0 else 0,
                        "trend_analysis": self._analyze_metric_trend(metric_name),
                        "contributing_factors": self._identify_contributing_factors(metric_name, actual_value, target_value),
                        "recommendations": self._generate_metric_recommendations(metric_name, actual_value, target_value)
                    }
                    
                    category_analysis[metric_name] = analysis
            
            detailed_analysis[category] = category_analysis
        
        return detailed_analysis
    
    def _analyze_user_feedback(self) -> Dict[str, Any]:
        """Analyze user feedback and sentiment"""
        return {
            "feedback_summary": {
                "total_feedback_received": 150,
                "positive_feedback": 120,
                "neutral_feedback": 20,
                "negative_feedback": 10,
                "sentiment_score": 4.2
            },
            "common_themes": {
                "positive": [
                    "Easy to use interface",
                    "Powerful AI capabilities",
                    "Excellent customer support",
                    "Fast response times",
                    "Comprehensive features"
                ],
                "negative": [
                    "Onboarding could be simpler",
                    "Some features hard to find",
                    "Mobile experience needs improvement",
                    "Pricing clarity needed"
                ]
            },
            "feature_feedback": {
                "ai_cto_agent": {"rating": 4.7, "usage": "95%"},
                "code_analysis": {"rating": 4.5, "usage": "80%"},
                "system_monitoring": {"rating": 4.3, "usage": "70%"},
                "team_analytics": {"rating": 4.1, "usage": "60%"}
            },
            "support_feedback": {
                "response_time_satisfaction": 4.6,
                "resolution_quality": 4.4,
                "support_team_helpfulness": 4.8
            }
        }
    
    def _create_competitive_analysis(self) -> Dict[str, Any]:
        """Create competitive analysis"""
        return {
            "market_position": {
                "category": "AI-Powered Technical Leadership",
                "market_size": "$2.1B",
                "our_market_share": "0.01%",
                "growth_potential": "High"
            },
            "competitive_advantages": [
                "First-to-market AI CTO solution",
                "Comprehensive technical leadership capabilities",
                "Cost-effective compared to human CTOs",
                "24/7 availability and consistency",
                "Continuous learning and improvement"
            ],
            "competitor_comparison": {
                "traditional_cto_services": {
                    "cost_advantage": "90% lower cost",
                    "availability_advantage": "24/7 vs business hours",
                    "consistency_advantage": "No human variability"
                },
                "ai_development_tools": {
                    "scope_advantage": "Holistic technical leadership vs point solutions",
                    "strategic_advantage": "Strategic decision making vs tactical tools"
                }
            },
            "market_response": {
                "industry_interest": "High",
                "media_coverage": "Positive",
                "investor_interest": "Strong",
                "customer_demand": "Growing"
            }
        }
    
    def _analyze_financial_performance(self, actual_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze financial performance"""
        return {
            "revenue_analysis": {
                "month_1_revenue": actual_metrics.get("month_1_revenue", 0),
                "target_revenue": self.success_criteria["business_metrics"]["month_1_revenue"]["target"],
                "revenue_performance": "125% of target",
                "revenue_sources": {
                    "subscription_revenue": 80,
                    "usage_based_revenue": 15,
                    "professional_services": 5
                }
            },
            "customer_metrics": {
                "total_signups": actual_metrics.get("launch_day_signups", 0),
                "paying_customers": actual_metrics.get("week_1_paying_customers", 0),
                "conversion_rate": 12.0,
                "average_revenue_per_user": 83.33,
                "customer_lifetime_value": 2500.0
            },
            "cost_analysis": {
                "customer_acquisition_cost": actual_metrics.get("customer_acquisition_cost", 0),
                "target_cac": self.success_criteria["business_metrics"]["customer_acquisition_cost"]["target"],
                "cac_performance": "15% below target",
                "cost_breakdown": {
                    "marketing_costs": 60,
                    "sales_costs": 25,
                    "operational_costs": 15
                }
            },
            "profitability_outlook": {
                "gross_margin": 85,
                "contribution_margin": 70,
                "break_even_timeline": "Month 6",
                "profitability_timeline": "Month 12"
            }
        }
    
    def _analyze_technical_performance(self, actual_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze technical performance"""
        return {
            "system_reliability": {
                "uptime": actual_metrics.get("system_uptime", 0),
                "target_uptime": self.success_criteria["technical_success_metrics"]["system_uptime"]["target"],
                "uptime_performance": "Exceeded target",
                "incident_count": 2,
                "mean_time_to_recovery": 15
            },
            "performance_metrics": {
                "response_time": actual_metrics.get("response_time", 0),
                "target_response_time": self.success_criteria["technical_success_metrics"]["response_time"]["target"],
                "performance_improvement": "40% faster than target",
                "throughput": 1000,
                "concurrent_users_handled": actual_metrics.get("concurrent_users", 0)
            },
            "scalability_analysis": {
                "peak_concurrent_users": 200,
                "system_capacity": 500,
                "auto_scaling_events": 15,
                "resource_utilization": 60
            },
            "security_assessment": {
                "security_incidents": 0,
                "vulnerability_scan_results": "Clean",
                "compliance_status": "Fully Compliant",
                "security_score": 95
            }
        }
    
    def _analyze_user_experience(self, actual_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze user experience metrics"""
        return {
            "onboarding_analysis": {
                "completion_rate": actual_metrics.get("onboarding_completion_rate", 0),
                "target_completion_rate": self.success_criteria["user_experience_metrics"]["onboarding_completion_rate"]["target"],
                "average_completion_time": 18,
                "drop_off_points": ["Step 3: Code Upload", "Step 5: Feature Exploration"],
                "improvement_opportunities": ["Simplify code upload", "Add more guidance"]
            },
            "user_activation": {
                "activation_rate": actual_metrics.get("user_activation_rate", 0),
                "target_activation_rate": self.success_criteria["user_experience_metrics"]["user_activation_rate"]["target"],
                "time_to_activation": 2.5,
                "activation_drivers": ["First AI analysis", "Successful code review", "Dashboard usage"]
            },
            "user_satisfaction": {
                "satisfaction_score": actual_metrics.get("user_satisfaction_score", 0),
                "target_satisfaction": self.success_criteria["user_experience_metrics"]["user_satisfaction_score"]["target"],
                "nps_score": 52,
                "satisfaction_drivers": ["AI quality", "Response time", "Feature completeness"]
            },
            "feature_adoption": {
                "adoption_rate": actual_metrics.get("feature_adoption_rate", 0),
                "target_adoption_rate": self.success_criteria["user_experience_metrics"]["feature_adoption_rate"]["target"],
                "most_used_features": ["AI CTO Chat", "Code Analysis", "System Monitoring"],
                "least_used_features": ["Team Analytics", "Advanced Reporting"]
            }
        }
    
    def _analyze_market_response(self) -> Dict[str, Any]:
        """Analyze market response to launch"""
        return {
            "media_coverage": {
                "total_mentions": 25,
                "positive_coverage": 20,
                "neutral_coverage": 4,
                "negative_coverage": 1,
                "reach": 500000,
                "sentiment_score": 4.1
            },
            "social_media_response": {
                "total_engagement": 5000,
                "shares": 1200,
                "comments": 800,
                "likes": 3000,
                "sentiment": "Positive",
                "viral_content": ["Launch announcement", "AI CTO demo video"]
            },
            "industry_response": {
                "analyst_coverage": "Positive",
                "peer_recognition": "High",
                "partnership_inquiries": 15,
                "investor_interest": "Strong"
            },
            "customer_response": {
                "inbound_inquiries": 300,
                "demo_requests": 150,
                "trial_signups": 125,
                "word_of_mouth_referrals": 25
            }
        }
    
    def _capture_lessons_learned(self) -> Dict[str, Any]:
        """Capture lessons learned from launch"""
        return {
            "what_worked_well": [
                "Pre-launch testing and validation was thorough",
                "Marketing messaging resonated with target audience",
                "Technical infrastructure handled launch traffic well",
                "Customer support team was well-prepared",
                "AI CTO capabilities exceeded user expectations"
            ],
            "what_could_be_improved": [
                "Onboarding flow could be more streamlined",
                "Feature discovery needs improvement",
                "Mobile experience requires optimization",
                "Pricing communication could be clearer",
                "Advanced features need better documentation"
            ],
            "unexpected_challenges": [
                "Higher than expected support volume initially",
                "Some users struggled with advanced features",
                "Mobile usage was higher than anticipated",
                "Enterprise customers had specific compliance questions"
            ],
            "unexpected_successes": [
                "Viral social media response exceeded expectations",
                "Word-of-mouth referrals were stronger than projected",
                "User satisfaction scores were higher than target",
                "System performance exceeded all benchmarks"
            ],
            "key_insights": [
                "Users value simplicity over feature richness initially",
                "AI quality is the primary driver of satisfaction",
                "Support responsiveness is critical for early adoption",
                "Clear value demonstration drives conversion",
                "Community building accelerates growth"
            ]
        }
    
    def _define_next_steps(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Define next steps based on launch performance"""
        return {
            "immediate_actions": [
                "Optimize onboarding flow based on user feedback",
                "Improve feature discovery and guidance",
                "Enhance mobile user experience",
                "Expand customer support capacity",
                "Implement advanced analytics tracking"
            ],
            "short_term_goals": [
                "Achieve 90% onboarding completion rate",
                "Increase feature adoption to 70%",
                "Reduce customer acquisition cost by 20%",
                "Launch enterprise features",
                "Expand to 3 new market segments"
            ],
            "medium_term_objectives": [
                "Reach 1000 paying customers",
                "Achieve $50K monthly recurring revenue",
                "Launch API and integration platform",
                "Expand to international markets",
                "Build partner ecosystem"
            ],
            "long_term_vision": [
                "Become the leading AI CTO platform",
                "Achieve $10M annual recurring revenue",
                "Expand to full AI executive suite",
                "IPO readiness preparation",
                "Global market leadership"
            ]
        }
    
    def _assess_risks(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks based on launch performance"""
        return {
            "technical_risks": [
                {"risk": "System scalability limits", "probability": "Medium", "impact": "High"},
                {"risk": "AI model performance degradation", "probability": "Low", "impact": "High"},
                {"risk": "Security vulnerabilities", "probability": "Low", "impact": "Critical"}
            ],
            "business_risks": [
                {"risk": "Competitive response", "probability": "High", "impact": "Medium"},
                {"risk": "Market saturation", "probability": "Low", "impact": "High"},
                {"risk": "Customer churn", "probability": "Medium", "impact": "Medium"}
            ],
            "operational_risks": [
                {"risk": "Support capacity constraints", "probability": "Medium", "impact": "Medium"},
                {"risk": "Team scaling challenges", "probability": "Medium", "impact": "Medium"},
                {"risk": "Quality control issues", "probability": "Low", "impact": "High"}
            ],
            "mitigation_strategies": [
                "Implement proactive monitoring and alerting",
                "Develop comprehensive disaster recovery plans",
                "Build redundancy into critical systems",
                "Establish clear escalation procedures",
                "Maintain strong customer communication"
            ]
        }
    
    def _identify_success_factors(self, performance_analysis: Dict[str, Any]) -> List[str]:
        """Identify key success factors"""
        return [
            "Strong product-market fit validation",
            "Excellent technical execution and reliability",
            "Effective marketing and positioning strategy",
            "Responsive customer support and engagement",
            "Continuous improvement based on user feedback",
            "Clear value proposition and pricing strategy",
            "Strong team execution and coordination",
            "Comprehensive testing and quality assurance",
            "Effective launch timing and market conditions",
            "Strong community building and word-of-mouth"
        ]
    
    def _generate_insights(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and recommendations"""
        return {
            "key_insights": [
                "Technical performance exceeded expectations across all metrics",
                "User satisfaction is high but onboarding needs optimization",
                "Business metrics show strong early traction",
                "Market response is overwhelmingly positive",
                "Cost efficiency is better than projected"
            ],
            "recommendations": [
                "Prioritize onboarding flow optimization",
                "Invest in mobile experience improvement",
                "Expand customer success team capacity",
                "Accelerate feature development roadmap",
                "Increase marketing investment in successful channels"
            ],
            "improvement_opportunities": [
                "Feature adoption rate optimization",
                "Advanced user workflow simplification",
                "Enterprise feature development",
                "International market expansion",
                "API and integration platform development"
            ]
        }
    
    def _analyze_metric_trend(self, metric_name: str) -> str:
        """Analyze trend for specific metric"""
        # Mock trend analysis - in production, this would analyze historical data
        trends = {
            "system_uptime": "Stable and improving",
            "response_time": "Consistently fast",
            "user_satisfaction_score": "Increasing",
            "monthly_recurring_revenue_growth": "Strong upward trend"
        }
        return trends.get(metric_name, "Stable")
    
    def _identify_contributing_factors(self, metric_name: str, actual: float, target: float) -> List[str]:
        """Identify factors contributing to metric performance"""
        # Mock contributing factors - in production, this would be based on actual analysis
        if actual >= target:
            return ["Strong technical execution", "Effective user onboarding", "Positive market response"]
        else:
            return ["User experience friction", "Feature complexity", "Market education needed"]
    
    def _generate_metric_recommendations(self, metric_name: str, actual: float, target: float) -> List[str]:
        """Generate recommendations for specific metric"""
        if actual >= target:
            return ["Continue current approach", "Monitor for consistency", "Consider raising targets"]
        else:
            return ["Analyze root causes", "Implement improvement plan", "Increase monitoring frequency"]
    
    def _create_visualizations(self, performance_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create visualizations for the report"""
        visualizations = {}
        
        try:
            # Create performance comparison chart
            self._create_performance_chart(performance_analysis)
            visualizations["performance_chart"] = "performance_comparison.png"
            
            # Create success rate by category chart
            self._create_category_chart(performance_analysis)
            visualizations["category_chart"] = "category_performance.png"
            
            # Create trend analysis chart
            self._create_trend_chart()
            visualizations["trend_chart"] = "metrics_trend.png"
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
        
        return visualizations
    
    def _create_performance_chart(self, performance_analysis: Dict[str, Any]):
        """Create performance comparison chart"""
        metrics = []
        targets = []
        actuals = []
        
        for metric_name, data in performance_analysis["individual_metrics"].items():
            metrics.append(metric_name.replace("_", " ").title())
            targets.append(data["target"])
            actuals.append(data["actual"])
        
        plt.figure(figsize=(12, 8))
        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], targets, width, label='Target', alpha=0.7)
        plt.bar([i + width/2 for i in x], actuals, width, label='Actual', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Launch Metrics: Target vs Actual Performance')
        plt.xticks(x, metrics, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_category_chart(self, performance_analysis: Dict[str, Any]):
        """Create category performance chart"""
        categories = []
        success_rates = []
        
        for category, data in performance_analysis["category_performance"].items():
            categories.append(category.replace("_", " ").title())
            success_rates.append(data["success_rate"])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, success_rates, color=['green' if rate >= 80 else 'orange' if rate >= 60 else 'red' for rate in success_rates])
        
        plt.xlabel('Metric Categories')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Metric Category')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('category_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trend_chart(self):
        """Create trend analysis chart"""
        # Mock trend data - in production, this would use actual historical data
        days = list(range(1, 8))  # First week
        signups = [15, 25, 30, 20, 35, 40, 45]
        revenue = [150, 300, 450, 600, 850, 1100, 1250]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Signups trend
        ax1.plot(days, signups, marker='o', linewidth=2, color='blue')
        ax1.set_xlabel('Days Since Launch')
        ax1.set_ylabel('Daily Signups')
        ax1.set_title('Daily Signups Trend')
        ax1.grid(True, alpha=0.3)
        
        # Revenue trend
        ax2.plot(days, revenue, marker='s', linewidth=2, color='green')
        ax2.set_xlabel('Days Since Launch')
        ax2.set_ylabel('Cumulative Revenue ($)')
        ax2.set_title('Cumulative Revenue Trend')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('metrics_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_report(self, report: Dict[str, Any]):
        """Save the comprehensive report"""
        # Create reports directory
        os.makedirs("launch_reports", exist_ok=True)
        
        # Save JSON report
        timestamp = self.report_date.strftime('%Y%m%d_%H%M%S')
        json_filename = f"launch_reports/comprehensive_launch_report_{timestamp}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save executive summary as separate file
        exec_summary_filename = f"launch_reports/executive_summary_{timestamp}.json"
        with open(exec_summary_filename, 'w') as f:
            json.dump(report["executive_summary"], f, indent=2, default=str)
        
        logger.info(f"📄 Report saved to {json_filename}")
        logger.info(f"📋 Executive summary saved to {exec_summary_filename}")

def main():
    """Main execution function"""
    generator = LaunchReportGenerator()
    
    print("📊 ScrollIntel Launch Report Generator")
    print("=" * 50)
    print("Generating comprehensive launch analysis report...")
    print("=" * 50)
    
    # Generate comprehensive report
    report = generator.generate_comprehensive_launch_report()
    
    # Print executive summary
    print("\n🎯 EXECUTIVE SUMMARY")
    print("-" * 30)
    exec_summary = report["executive_summary"]
    print(f"Launch Status: {exec_summary['launch_status']}")
    print(f"Overall Success Rate: {exec_summary['overall_success_rate']:.1f}%")
    
    print("\n✅ Key Achievements:")
    for achievement in exec_summary["key_achievements"][:3]:
        print(f"  • {achievement}")
    
    print("\n🔧 Areas for Improvement:")
    for improvement in exec_summary["areas_for_improvement"][:3]:
        print(f"  • {improvement}")
    
    print("\n💰 Financial Highlights:")
    fin_highlights = exec_summary["financial_highlights"]
    print(f"  • Revenue Performance: {fin_highlights['revenue_performance']}")
    print(f"  • Customer Acquisition: {fin_highlights['customer_acquisition']}")
    print(f"  • Cost Efficiency: {fin_highlights['cost_efficiency']}")
    
    print("\n📈 Next Steps:")
    next_steps = report["next_steps"]["immediate_actions"][:3]
    for step in next_steps:
        print(f"  • {step}")
    
    print("\n" + "=" * 50)
    print("✅ Comprehensive launch report generated successfully!")
    print("📄 Check launch_reports/ directory for detailed analysis")
    print("=" * 50)

if __name__ == "__main__":
    main()