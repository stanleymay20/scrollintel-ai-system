"""
Code Review Engine for Automated Code Generation System

This module provides automated code review and improvement suggestions
by integrating validation, security, performance, and compliance analysis.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .code_validator import CodeValidator, ValidationResult
from .security_scanner import SecurityScanner, SecurityReport
from .performance_analyzer import PerformanceAnalyzer, PerformanceReport
from .compliance_checker import ComplianceChecker, ComplianceReport

logger = logging.getLogger(__name__)

class ReviewPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ReviewIssue:
    priority: ReviewPriority
    category: str
    source: str  # validator, security, performance, compliance
    title: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: str = ""
    fix_suggestion: Optional[str] = None
    impact: Optional[str] = None

@dataclass
class CodeQualityMetrics:
    overall_score: float  # 0-100 overall quality score
    validation_score: float
    security_score: float  # Inverted risk score (100 - risk)
    performance_score: float
    compliance_score: float
    maintainability_score: float
    readability_score: float

@dataclass
class ReviewSummary:
    total_issues: int
    issues_by_priority: Dict[str, int]
    issues_by_category: Dict[str, int]
    issues_by_source: Dict[str, int]
    estimated_fix_time: str
    deployment_readiness: str

@dataclass
class CodeReviewReport:
    issues: List[ReviewIssue]
    metrics: CodeQualityMetrics
    summary: ReviewSummary
    recommendations: List[str]
    improvement_plan: List[str]
    metadata: Dict[str, Any]

class CodeReviewEngine:
    """
    Comprehensive automated code review engine that integrates multiple analysis tools
    """
    
    def __init__(self):
        self.validator = CodeValidator()
        self.security_scanner = SecurityScanner()
        self.performance_analyzer = PerformanceAnalyzer()
        self.compliance_checker = ComplianceChecker()
        
        # Priority mapping from different tools
        self.priority_mapping = {
            # Validation
            'error': ReviewPriority.CRITICAL,
            'warning': ReviewPriority.MEDIUM,
            'info': ReviewPriority.LOW,
            
            # Security
            'critical': ReviewPriority.CRITICAL,
            'high': ReviewPriority.HIGH,
            'medium': ReviewPriority.MEDIUM,
            'low': ReviewPriority.LOW,
            
            # Performance
            'critical': ReviewPriority.CRITICAL,
            'high': ReviewPriority.HIGH,
            'medium': ReviewPriority.MEDIUM,
            'low': ReviewPriority.LOW,
            
            # Compliance
            'critical': ReviewPriority.CRITICAL,
            'high': ReviewPriority.HIGH,
            'medium': ReviewPriority.MEDIUM,
            'low': ReviewPriority.LOW
        }
    
    def review_code(self, code: str, language: str, 
                   context: Optional[Dict] = None,
                   standards: Optional[List[str]] = None) -> CodeReviewReport:
        """
        Perform comprehensive automated code review
        
        Args:
            code: Source code to review
            language: Programming language
            context: Additional context (project type, requirements, etc.)
            standards: Coding standards to apply
            
        Returns:
            CodeReviewReport with comprehensive analysis and recommendations
        """
        try:
            logger.info(f"Starting code review for {language} code ({len(code)} characters)")
            
            # Run all analysis tools
            validation_result = self.validator.validate_code(code, language, context)
            security_report = self.security_scanner.scan_code(code, language, context)
            performance_report = self.performance_analyzer.analyze_performance(code, language, context)
            compliance_report = self.compliance_checker.check_compliance(code, language, standards, context)
            
            # Consolidate issues from all sources
            issues = self._consolidate_issues(
                validation_result, security_report, performance_report, compliance_report
            )
            
            # Calculate comprehensive quality metrics
            metrics = self._calculate_quality_metrics(
                validation_result, security_report, performance_report, compliance_report, code
            )
            
            # Generate summary
            summary = self._generate_summary(issues)
            
            # Generate recommendations and improvement plan
            recommendations = self._generate_recommendations(issues, metrics)
            improvement_plan = self._generate_improvement_plan(issues, metrics)
            
            return CodeReviewReport(
                issues=issues,
                metrics=metrics,
                summary=summary,
                recommendations=recommendations,
                improvement_plan=improvement_plan,
                metadata={
                    'language': language,
                    'code_length': len(code),
                    'lines_of_code': len(code.split('\n')),
                    'analysis_tools': ['validator', 'security', 'performance', 'compliance'],
                    'standards_applied': standards or []
                }
            )
            
        except Exception as e:
            logger.error(f"Code review failed: {str(e)}")
            return self._create_error_report(str(e))
    
    def _consolidate_issues(self, validation_result: ValidationResult,
                           security_report: SecurityReport,
                           performance_report: PerformanceReport,
                           compliance_report: ComplianceReport) -> List[ReviewIssue]:
        """Consolidate issues from all analysis tools"""
        issues = []
        
        # Process validation issues
        for issue in validation_result.issues:
            priority = self.priority_mapping.get(issue.severity.value, ReviewPriority.MEDIUM)
            issues.append(ReviewIssue(
                priority=priority,
                category="validation",
                source="validator",
                title=issue.message,
                description=issue.message,
                line_number=issue.line_number,
                recommendation=issue.suggestion or "Review and fix validation issue",
                impact="Code may not compile or run correctly"
            ))
        
        # Process security vulnerabilities
        for vuln in security_report.vulnerabilities:
            priority = self.priority_mapping.get(vuln.severity.value, ReviewPriority.MEDIUM)
            issues.append(ReviewIssue(
                priority=priority,
                category="security",
                source="security",
                title=vuln.title,
                description=vuln.description,
                line_number=vuln.line_number,
                code_snippet=vuln.code_snippet,
                recommendation=vuln.recommendation or "Address security vulnerability",
                impact="Potential security risk"
            ))
        
        # Process performance issues
        for perf_issue in performance_report.issues:
            priority = self.priority_mapping.get(perf_issue.impact.value, ReviewPriority.MEDIUM)
            issues.append(ReviewIssue(
                priority=priority,
                category="performance",
                source="performance",
                title=perf_issue.title,
                description=perf_issue.description,
                line_number=perf_issue.line_number,
                code_snippet=perf_issue.code_snippet,
                recommendation=perf_issue.recommendation or "Optimize for better performance",
                impact=perf_issue.estimated_impact or "Performance degradation"
            ))
        
        # Process compliance violations
        for violation in compliance_report.violations:
            priority = self.priority_mapping.get(violation.severity.value, ReviewPriority.MEDIUM)
            issues.append(ReviewIssue(
                priority=priority,
                category="compliance",
                source="compliance",
                title=violation.title,
                description=violation.description,
                line_number=violation.line_number,
                code_snippet=violation.code_snippet,
                recommendation=violation.recommendation or "Fix compliance violation",
                impact="Code style and maintainability"
            ))
        
        # Sort issues by priority and line number
        priority_order = {
            ReviewPriority.CRITICAL: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3,
            ReviewPriority.INFO: 4
        }
        
        issues.sort(key=lambda x: (priority_order[x.priority], x.line_number or 0))
        
        return issues
    
    def _calculate_quality_metrics(self, validation_result: ValidationResult,
                                  security_report: SecurityReport,
                                  performance_report: PerformanceReport,
                                  compliance_report: ComplianceReport,
                                  code: str) -> CodeQualityMetrics:
        """Calculate comprehensive code quality metrics"""
        
        # Individual scores from each tool
        validation_score = validation_result.score
        security_score = max(0, 100 - security_report.risk_score)  # Invert risk score
        performance_score = performance_report.metrics.overall_score
        compliance_score = compliance_report.metrics.compliance_score
        
        # Calculate maintainability score
        maintainability_score = self._calculate_maintainability_score(code, validation_result, compliance_report)
        
        # Calculate readability score
        readability_score = self._calculate_readability_score(code, compliance_report)
        
        # Calculate overall score (weighted average)
        weights = {
            'validation': 0.25,
            'security': 0.25,
            'performance': 0.20,
            'compliance': 0.15,
            'maintainability': 0.10,
            'readability': 0.05
        }
        
        overall_score = (
            validation_score * weights['validation'] +
            security_score * weights['security'] +
            performance_score * weights['performance'] +
            compliance_score * weights['compliance'] +
            maintainability_score * weights['maintainability'] +
            readability_score * weights['readability']
        )
        
        return CodeQualityMetrics(
            overall_score=overall_score,
            validation_score=validation_score,
            security_score=security_score,
            performance_score=performance_score,
            compliance_score=compliance_score,
            maintainability_score=maintainability_score,
            readability_score=readability_score
        )
    
    def _calculate_maintainability_score(self, code: str, validation_result: ValidationResult,
                                       compliance_report: ComplianceReport) -> float:
        """Calculate maintainability score based on code characteristics"""
        base_score = 100.0
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Deduct for code complexity
        if len(non_empty_lines) > 500:
            base_score -= 20
        elif len(non_empty_lines) > 200:
            base_score -= 10
        
        # Deduct for validation issues (complexity indicators)
        complexity_issues = [issue for issue in validation_result.issues 
                           if 'complexity' in issue.message.lower()]
        base_score -= len(complexity_issues) * 5
        
        # Deduct for documentation issues
        doc_violations = [v for v in compliance_report.violations 
                         if v.category == 'documentation']
        base_score -= len(doc_violations) * 3
        
        return max(0.0, base_score)
    
    def _calculate_readability_score(self, code: str, compliance_report: ComplianceReport) -> float:
        """Calculate readability score based on formatting and style"""
        base_score = 100.0
        
        # Deduct for formatting violations
        formatting_violations = [v for v in compliance_report.violations 
                               if v.category == 'formatting']
        base_score -= len(formatting_violations) * 2
        
        # Deduct for naming violations
        naming_violations = [v for v in compliance_report.violations 
                           if v.category == 'naming']
        base_score -= len(naming_violations) * 3
        
        # Check for comments (positive indicator)
        lines = code.split('\n')
        comment_lines = [line for line in lines if line.strip().startswith('#') or 
                        line.strip().startswith('//') or line.strip().startswith('/*')]
        comment_ratio = len(comment_lines) / max(1, len(lines))
        
        if comment_ratio > 0.1:  # Good comment ratio
            base_score += 5
        elif comment_ratio < 0.02:  # Very few comments
            base_score -= 10
        
        return max(0.0, base_score)
    
    def _generate_summary(self, issues: List[ReviewIssue]) -> ReviewSummary:
        """Generate review summary statistics"""
        total_issues = len(issues)
        
        # Count by priority
        priority_counts = {}
        for issue in issues:
            priority = issue.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Count by category
        category_counts = {}
        for issue in issues:
            category = issue.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by source
        source_counts = {}
        for issue in issues:
            source = issue.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Estimate fix time
        estimated_fix_time = self._estimate_fix_time(issues)
        
        # Determine deployment readiness
        deployment_readiness = self._assess_deployment_readiness(issues)
        
        return ReviewSummary(
            total_issues=total_issues,
            issues_by_priority=priority_counts,
            issues_by_category=category_counts,
            issues_by_source=source_counts,
            estimated_fix_time=estimated_fix_time,
            deployment_readiness=deployment_readiness
        )
    
    def _estimate_fix_time(self, issues: List[ReviewIssue]) -> str:
        """Estimate time required to fix all issues"""
        time_estimates = {
            ReviewPriority.CRITICAL: 60,  # minutes
            ReviewPriority.HIGH: 30,
            ReviewPriority.MEDIUM: 15,
            ReviewPriority.LOW: 5,
            ReviewPriority.INFO: 2
        }
        
        total_minutes = sum(time_estimates.get(issue.priority, 10) for issue in issues)
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        elif total_minutes < 480:  # 8 hours
            hours = total_minutes / 60
            return f"{hours:.1f} hours"
        else:
            days = total_minutes / 480  # 8-hour work day
            return f"{days:.1f} days"
    
    def _assess_deployment_readiness(self, issues: List[ReviewIssue]) -> str:
        """Assess if code is ready for deployment"""
        critical_issues = [i for i in issues if i.priority == ReviewPriority.CRITICAL]
        high_issues = [i for i in issues if i.priority == ReviewPriority.HIGH]
        security_issues = [i for i in issues if i.category == 'security' and 
                          i.priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]]
        
        if critical_issues:
            return "NOT READY - Critical issues must be fixed"
        elif len(security_issues) > 0:
            return "NOT READY - Security vulnerabilities present"
        elif len(high_issues) > 5:
            return "REVIEW REQUIRED - Multiple high-priority issues"
        elif len(high_issues) > 0:
            return "CAUTION - Some high-priority issues present"
        else:
            return "READY - No blocking issues found"
    
    def _generate_recommendations(self, issues: List[ReviewIssue], 
                                metrics: CodeQualityMetrics) -> List[str]:
        """Generate high-level recommendations"""
        recommendations = []
        
        # Priority-based recommendations
        critical_count = len([i for i in issues if i.priority == ReviewPriority.CRITICAL])
        high_count = len([i for i in issues if i.priority == ReviewPriority.HIGH])
        
        if critical_count > 0:
            recommendations.append(f"URGENT: Fix {critical_count} critical issues before deployment")
        
        if high_count > 0:
            recommendations.append(f"Address {high_count} high-priority issues")
        
        # Category-specific recommendations
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for category, count in top_categories:
            if count > 2:
                if category == "security":
                    recommendations.append("Conduct security review and implement fixes")
                elif category == "performance":
                    recommendations.append("Optimize performance-critical sections")
                elif category == "compliance":
                    recommendations.append("Improve code style and standards compliance")
                elif category == "validation":
                    recommendations.append("Fix syntax and logic errors")
        
        # Metric-based recommendations
        if metrics.overall_score < 60:
            recommendations.append("Consider significant refactoring for better code quality")
        elif metrics.overall_score < 80:
            recommendations.append("Address quality issues for production readiness")
        
        if metrics.security_score < 70:
            recommendations.append("Implement additional security measures")
        
        if metrics.performance_score < 70:
            recommendations.append("Profile and optimize performance bottlenecks")
        
        if metrics.maintainability_score < 70:
            recommendations.append("Refactor for better maintainability")
        
        return recommendations
    
    def _generate_improvement_plan(self, issues: List[ReviewIssue], 
                                 metrics: CodeQualityMetrics) -> List[str]:
        """Generate step-by-step improvement plan"""
        plan = []
        
        # Phase 1: Critical fixes
        critical_issues = [i for i in issues if i.priority == ReviewPriority.CRITICAL]
        if critical_issues:
            plan.append("Phase 1: Fix Critical Issues")
            for issue in critical_issues[:5]:  # Top 5 critical issues
                plan.append(f"  - {issue.title} (Line {issue.line_number or 'N/A'})")
        
        # Phase 2: Security fixes
        security_issues = [i for i in issues if i.category == 'security' and 
                          i.priority in [ReviewPriority.HIGH, ReviewPriority.MEDIUM]]
        if security_issues:
            plan.append("Phase 2: Address Security Vulnerabilities")
            for issue in security_issues[:3]:  # Top 3 security issues
                plan.append(f"  - {issue.title}")
        
        # Phase 3: Performance optimization
        perf_issues = [i for i in issues if i.category == 'performance' and 
                      i.priority in [ReviewPriority.HIGH, ReviewPriority.MEDIUM]]
        if perf_issues:
            plan.append("Phase 3: Performance Optimization")
            for issue in perf_issues[:3]:  # Top 3 performance issues
                plan.append(f"  - {issue.title}")
        
        # Phase 4: Code quality improvements
        quality_issues = [i for i in issues if i.category in ['compliance', 'validation'] and 
                         i.priority == ReviewPriority.MEDIUM]
        if quality_issues:
            plan.append("Phase 4: Code Quality Improvements")
            plan.append("  - Address formatting and style issues")
            plan.append("  - Add missing documentation")
            plan.append("  - Improve naming conventions")
        
        # Phase 5: Final polish
        if len(issues) > len(critical_issues) + len(security_issues) + len(perf_issues):
            plan.append("Phase 5: Final Polish")
            plan.append("  - Fix remaining low-priority issues")
            plan.append("  - Add comprehensive tests")
            plan.append("  - Final code review")
        
        return plan
    
    def _create_error_report(self, error_message: str) -> CodeReviewReport:
        """Create error report when review fails"""
        return CodeReviewReport(
            issues=[ReviewIssue(
                priority=ReviewPriority.CRITICAL,
                category="review_error",
                source="system",
                title="Code Review Failed",
                description=f"Unable to complete code review: {error_message}",
                recommendation="Manual code review required"
            )],
            metrics=CodeQualityMetrics(0, 0, 0, 0, 0, 0, 0),
            summary=ReviewSummary(1, {'critical': 1}, {'review_error': 1}, {'system': 1}, 
                                "Unknown", "NOT READY - Review failed"),
            recommendations=["Manual code review required due to system error"],
            improvement_plan=["Investigate and fix review system error", "Perform manual code review"],
            metadata={'error': error_message}
        )
    
    def generate_review_report_json(self, report: CodeReviewReport) -> str:
        """Generate JSON report for code review results"""
        # Convert dataclasses to dictionaries for JSON serialization
        report_dict = asdict(report)
        
        # Convert enums to strings
        for issue in report_dict['issues']:
            issue['priority'] = issue['priority']
        
        return json.dumps(report_dict, indent=2, default=str)
    
    def generate_review_report_html(self, report: CodeReviewReport) -> str:
        """Generate HTML report for code review results"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Review Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .issues {{ margin: 20px 0; }}
                .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .critical {{ border-left-color: #d32f2f; }}
                .high {{ border-left-color: #f57c00; }}
                .medium {{ border-left-color: #fbc02d; }}
                .low {{ border-left-color: #388e3c; }}
                .recommendations {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Code Review Report</h1>
                <p><strong>Overall Score:</strong> {report.metrics.overall_score:.1f}/100</p>
                <p><strong>Deployment Status:</strong> {report.summary.deployment_readiness}</p>
                <p><strong>Estimated Fix Time:</strong> {report.summary.estimated_fix_time}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Validation</h3>
                    <p>{report.metrics.validation_score:.1f}/100</p>
                </div>
                <div class="metric">
                    <h3>Security</h3>
                    <p>{report.metrics.security_score:.1f}/100</p>
                </div>
                <div class="metric">
                    <h3>Performance</h3>
                    <p>{report.metrics.performance_score:.1f}/100</p>
                </div>
                <div class="metric">
                    <h3>Compliance</h3>
                    <p>{report.metrics.compliance_score:.1f}/100</p>
                </div>
            </div>
            
            <div class="issues">
                <h2>Issues Found ({report.summary.total_issues})</h2>
        """
        
        for issue in report.issues[:20]:  # Show top 20 issues
            priority_class = issue.priority.value
            html += f"""
                <div class="issue {priority_class}">
                    <h4>{issue.title} ({issue.priority.value.upper()})</h4>
                    <p>{issue.description}</p>
                    {f'<p><strong>Line:</strong> {issue.line_number}</p>' if issue.line_number else ''}
                    <p><strong>Recommendation:</strong> {issue.recommendation}</p>
                </div>
            """
        
        html += f"""
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
            
            <div class="recommendations">
                <h2>Improvement Plan</h2>
                <ol>
        """
        
        for step in report.improvement_plan:
            html += f"<li>{step}</li>"
        
        html += """
                </ol>
            </div>
        </body>
        </html>
        """
        
        return html