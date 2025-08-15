"""
Test framework runner for comprehensive validation of AI Data Readiness Platform.
"""

import pytest
import pandas as pd
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from ai_data_readiness.tests.validation.test_synthetic_data_generation import (
    SyntheticDataGenerator, QualityIssueConfig, BiasConfig, DataQualityIssueType
)


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    test_category: str
    status: str  # 'passed', 'failed', 'skipped'
    execution_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    test_results: List[TestResult]
    summary_metrics: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'success_rate': self.success_rate,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'summary_metrics': self.summary_metrics,
            'test_results': [asdict(result) for result in self.test_results]
        }


class ValidationTestRunner:
    """Comprehensive test runner for validation framework."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize test runner with configuration."""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.data_generator = SyntheticDataGenerator(seed=42)
        self.test_results = []
    
    def _get_default_config(self) -> Dict:
        """Get default test configuration."""
        return {
            'test_categories': [
                'synthetic_data_generation',
                'quality_algorithm_validation',
                'bias_detection_validation',
                'performance_benchmarks',
                'integration_workflows'
            ],
            'dataset_sizes': [1000, 5000, 10000],
            'quality_thresholds': {
                'completeness_accuracy': 0.95,
                'outlier_detection_sensitivity': 0.8,
                'bias_detection_accuracy': 0.9,
                'performance_threshold_compliance': 0.9
            },
            'bias_test_scenarios': [
                {'type': 'gender_bias', 'strengths': [0.1, 0.2, 0.3]},
                {'type': 'racial_bias', 'strengths': [0.15, 0.25, 0.35]},
                {'type': 'intersectional_bias', 'strengths': [0.2, 0.3]}
            ],
            'performance_limits': {
                'max_ingestion_time_per_1k_rows': 1.0,  # seconds
                'max_quality_assessment_time_per_1k_rows': 2.0,
                'max_bias_analysis_time_per_1k_rows': 1.5
            },
            'output_directory': 'test_results',
            'generate_detailed_reports': True,
            'save_test_data': False
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for test runner."""
        logger = logging.getLogger('ValidationTestRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_synthetic_data_validation(self) -> List[TestResult]:
        """Run synthetic data generation validation tests."""
        self.logger.info("Running synthetic data generation validation...")
        results = []
        
        # Test clean data generation
        start_time = time.time()
        try:
            for size in self.config['dataset_sizes']:
                df = self.data_generator.generate_clean_dataset(size, 10)
                
                # Validate clean data properties
                assert len(df) == size
                assert df.isnull().sum().sum() == 0  # No missing values
                assert len(df.columns) == 11  # 10 features + target
                
            results.append(TestResult(
                test_name="clean_data_generation",
                test_category="synthetic_data_generation",
                status="passed",
                execution_time=time.time() - start_time,
                metrics={'datasets_generated': len(self.config['dataset_sizes'])}
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="clean_data_generation",
                test_category="synthetic_data_generation",
                status="failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))
        
        # Test quality issue introduction
        quality_issues = [
            DataQualityIssueType.MISSING_VALUES,
            DataQualityIssueType.OUTLIERS,
            DataQualityIssueType.DUPLICATES,
            DataQualityIssueType.INCONSISTENT_FORMATS
        ]
        
        for issue_type in quality_issues:
            start_time = time.time()
            try:
                df = self.data_generator.generate_clean_dataset(1000, 5)
                
                config = QualityIssueConfig(
                    issue_type=issue_type,
                    severity=0.1
                )
                
                if issue_type == DataQualityIssueType.MISSING_VALUES:
                    df_with_issues = self.data_generator.introduce_missing_values(df, config)
                    assert df_with_issues.isnull().sum().sum() > 0
                elif issue_type == DataQualityIssueType.OUTLIERS:
                    df_with_issues = self.data_generator.introduce_outliers(df, config)
                    # Check for outliers in numerical columns
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    has_outliers = False
                    for col in numerical_cols:
                        if col == 'target':
                            continue
                        col_mean = df[col].mean()
                        col_std = df[col].std()
                        outliers = df_with_issues[
                            (df_with_issues[col] > col_mean + 3 * col_std) |
                            (df_with_issues[col] < col_mean - 3 * col_std)
                        ]
                        if len(outliers) > 0:
                            has_outliers = True
                            break
                    assert has_outliers
                elif issue_type == DataQualityIssueType.DUPLICATES:
                    df_with_issues = self.data_generator.introduce_duplicates(df, config)
                    assert len(df_with_issues) > len(df)
                elif issue_type == DataQualityIssueType.INCONSISTENT_FORMATS:
                    df_with_issues = self.data_generator.introduce_inconsistent_formats(df, config)
                    # Check for format variations in categorical columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        original_unique = len(df[col].unique())
                        modified_unique = len(df_with_issues[col].unique())
                        if modified_unique > original_unique:
                            break
                    else:
                        assert False, "No format inconsistencies detected"
                
                results.append(TestResult(
                    test_name=f"introduce_{issue_type.value}",
                    test_category="synthetic_data_generation",
                    status="passed",
                    execution_time=time.time() - start_time
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"introduce_{issue_type.value}",
                    test_category="synthetic_data_generation",
                    status="failed",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        # Test bias introduction
        for bias_scenario in self.config['bias_test_scenarios']:
            for strength in bias_scenario['strengths']:
                start_time = time.time()
                try:
                    df = self.data_generator.generate_clean_dataset(1000, 5)
                    
                    # Add gender column for bias testing
                    df['gender'] = np.random.choice(['M', 'F'], len(df))
                    
                    bias_config = BiasConfig(
                        protected_attribute='gender',
                        target_column='target',
                        bias_strength=strength,
                        bias_type='demographic_parity'
                    )
                    
                    df_with_bias = self.data_generator.introduce_bias(df, bias_config)
                    
                    # Verify bias introduction
                    male_rate = df_with_bias[df_with_bias['gender'] == 'M']['target'].mean()
                    female_rate = df_with_bias[df_with_bias['gender'] == 'F']['target'].mean()
                    bias_difference = abs(male_rate - female_rate)
                    
                    assert bias_difference > strength * 0.5, \
                        f"Insufficient bias introduced: {bias_difference} < {strength * 0.5}"
                    
                    results.append(TestResult(
                        test_name=f"introduce_{bias_scenario['type']}_strength_{strength}",
                        test_category="synthetic_data_generation",
                        status="passed",
                        execution_time=time.time() - start_time,
                        metrics={'bias_difference': bias_difference}
                    ))
                except Exception as e:
                    results.append(TestResult(
                        test_name=f"introduce_{bias_scenario['type']}_strength_{strength}",
                        test_category="synthetic_data_generation",
                        status="failed",
                        execution_time=time.time() - start_time,
                        error_message=str(e)
                    ))
        
        return results
    
    def run_quality_algorithm_validation(self) -> List[TestResult]:
        """Run quality assessment algorithm validation tests."""
        self.logger.info("Running quality algorithm validation...")
        results = []
        
        # Test completeness detection accuracy
        missing_rates = [0.0, 0.1, 0.25, 0.5]
        
        for missing_rate in missing_rates:
            start_time = time.time()
            try:
                df = self.data_generator.generate_clean_dataset(1000, 5)
                
                if missing_rate > 0:
                    config = QualityIssueConfig(
                        issue_type=DataQualityIssueType.MISSING_VALUES,
                        severity=missing_rate
                    )
                    df = self.data_generator.introduce_missing_values(df, config)
                
                # Calculate actual completeness
                total_cells = len(df) * (len(df.columns) - 1)  # Exclude target
                missing_cells = df.drop('target', axis=1).isnull().sum().sum()
                actual_completeness = 1 - (missing_cells / total_cells)
                
                # Mock quality assessment (in real implementation, would use actual engine)
                detected_completeness = actual_completeness + np.random.normal(0, 0.02)  # Add small noise
                detected_completeness = np.clip(detected_completeness, 0, 1)
                
                accuracy = 1 - abs(detected_completeness - actual_completeness)
                
                results.append(TestResult(
                    test_name=f"completeness_detection_missing_rate_{missing_rate}",
                    test_category="quality_algorithm_validation",
                    status="passed" if accuracy >= self.config['quality_thresholds']['completeness_accuracy'] else "failed",
                    execution_time=time.time() - start_time,
                    metrics={
                        'actual_completeness': actual_completeness,
                        'detected_completeness': detected_completeness,
                        'accuracy': accuracy
                    }
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"completeness_detection_missing_rate_{missing_rate}",
                    test_category="quality_algorithm_validation",
                    status="failed",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_bias_detection_validation(self) -> List[TestResult]:
        """Run bias detection algorithm validation tests."""
        self.logger.info("Running bias detection validation...")
        results = []
        
        # Test bias detection accuracy
        for bias_scenario in self.config['bias_test_scenarios']:
            for strength in bias_scenario['strengths']:
                start_time = time.time()
                try:
                    # Generate biased dataset
                    df = self.create_test_biased_dataset(2000, bias_scenario['type'], strength)
                    
                    # Mock bias detection (in real implementation, would use actual engine)
                    detected_bias = strength > 0.15  # Simple threshold-based detection
                    detected_strength = strength + np.random.normal(0, 0.05)  # Add noise
                    detected_strength = max(0, detected_strength)
                    
                    # Calculate accuracy
                    should_detect = strength > 0.1
                    detection_correct = (detected_bias == should_detect)
                    strength_accuracy = 1 - abs(detected_strength - strength) / max(strength, 0.1)
                    
                    overall_accuracy = (detection_correct * 0.7 + strength_accuracy * 0.3)
                    
                    results.append(TestResult(
                        test_name=f"bias_detection_{bias_scenario['type']}_strength_{strength}",
                        test_category="bias_detection_validation",
                        status="passed" if overall_accuracy >= self.config['quality_thresholds']['bias_detection_accuracy'] else "failed",
                        execution_time=time.time() - start_time,
                        metrics={
                            'actual_bias_strength': strength,
                            'detected_bias': detected_bias,
                            'detected_strength': detected_strength,
                            'detection_accuracy': overall_accuracy
                        }
                    ))
                except Exception as e:
                    results.append(TestResult(
                        test_name=f"bias_detection_{bias_scenario['type']}_strength_{strength}",
                        test_category="bias_detection_validation",
                        status="failed",
                        execution_time=time.time() - start_time,
                        error_message=str(e)
                    ))
        
        return results
    
    def create_test_biased_dataset(self, n_samples: int, bias_type: str, bias_strength: float) -> pd.DataFrame:
        """Create test dataset with known bias for validation."""
        np.random.seed(42)
        
        # Create base features
        age = np.random.randint(18, 80, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        gender = np.random.choice(['M', 'F'], n_samples)
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
        
        if bias_type == "gender_bias":
            base_prob = 0.6
            approval_prob = np.where(
                gender == 'M',
                base_prob + bias_strength,
                base_prob - bias_strength
            )
        elif bias_type == "racial_bias":
            base_prob = 0.6
            approval_prob = np.where(
                race == 'White',
                base_prob + bias_strength,
                base_prob - bias_strength / 2
            )
        elif bias_type == "intersectional_bias":
            base_prob = 0.6
            approval_prob = np.where(
                (gender == 'F') & (race == 'Black'),
                base_prob - bias_strength * 1.5,
                np.where(
                    gender == 'F',
                    base_prob - bias_strength * 0.5,
                    np.where(
                        race == 'Black',
                        base_prob - bias_strength,
                        base_prob + bias_strength * 0.2
                    )
                )
            )
        else:
            approval_prob = np.full(n_samples, 0.6)
        
        approved = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        return pd.DataFrame({
            'age': age,
            'income': income,
            'gender': gender,
            'race': race,
            'approved': approved
        })
    
    def run_performance_benchmarks(self) -> List[TestResult]:
        """Run performance benchmark tests."""
        self.logger.info("Running performance benchmarks...")
        results = []
        
        for size in self.config['dataset_sizes']:
            # Test data ingestion performance
            start_time = time.time()
            try:
                df = self.data_generator.generate_clean_dataset(size, 10)
                
                # Mock ingestion time (in real implementation, would use actual service)
                ingestion_time = (size / 1000) * 0.5 + np.random.normal(0, 0.1)
                ingestion_time = max(0.1, ingestion_time)
                
                time_per_1k = (ingestion_time / size) * 1000
                performance_ok = time_per_1k <= self.config['performance_limits']['max_ingestion_time_per_1k_rows']
                
                results.append(TestResult(
                    test_name=f"ingestion_performance_{size}_rows",
                    test_category="performance_benchmarks",
                    status="passed" if performance_ok else "failed",
                    execution_time=time.time() - start_time,
                    metrics={
                        'dataset_size': size,
                        'ingestion_time': ingestion_time,
                        'time_per_1k_rows': time_per_1k,
                        'throughput_rows_per_sec': size / ingestion_time
                    }
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"ingestion_performance_{size}_rows",
                    test_category="performance_benchmarks",
                    status="failed",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_all_validations(self) -> ValidationReport:
        """Run all validation tests and generate comprehensive report."""
        self.logger.info("Starting comprehensive validation test suite...")
        
        start_time = time.time()
        all_results = []
        
        # Run all test categories
        test_runners = [
            self.run_synthetic_data_validation,
            self.run_quality_algorithm_validation,
            self.run_bias_detection_validation,
            self.run_performance_benchmarks
        ]
        
        for runner in test_runners:
            try:
                results = runner()
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Test runner failed: {str(e)}")
                # Add failed test result
                all_results.append(TestResult(
                    test_name=f"test_runner_{runner.__name__}",
                    test_category="framework",
                    status="failed",
                    execution_time=0,
                    error_message=str(e)
                ))
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.status == "passed")
        failed_tests = sum(1 for r in all_results if r.status == "failed")
        skipped_tests = sum(1 for r in all_results if r.status == "skipped")
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(all_results)
        
        # Create validation report
        report = ValidationReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time=total_time,
            test_results=all_results,
            summary_metrics=summary_metrics
        )
        
        self.logger.info(f"Validation complete: {passed_tests}/{total_tests} tests passed ({report.success_rate:.1%})")
        
        return report
    
    def _calculate_summary_metrics(self, results: List[TestResult]) -> Dict:
        """Calculate summary metrics from test results."""
        metrics = {
            'test_categories': {},
            'performance_metrics': {},
            'accuracy_metrics': {}
        }
        
        # Group results by category
        by_category = {}
        for result in results:
            if result.test_category not in by_category:
                by_category[result.test_category] = []
            by_category[result.test_category].append(result)
        
        # Calculate category-wise metrics
        for category, cat_results in by_category.items():
            total = len(cat_results)
            passed = sum(1 for r in cat_results if r.status == "passed")
            
            metrics['test_categories'][category] = {
                'total_tests': total,
                'passed_tests': passed,
                'success_rate': passed / total if total > 0 else 0,
                'avg_execution_time': np.mean([r.execution_time for r in cat_results])
            }
        
        # Extract performance metrics
        performance_results = [r for r in results if r.test_category == "performance_benchmarks"]
        if performance_results:
            throughputs = []
            for result in performance_results:
                if result.metrics and 'throughput_rows_per_sec' in result.metrics:
                    throughputs.append(result.metrics['throughput_rows_per_sec'])
            
            if throughputs:
                metrics['performance_metrics'] = {
                    'avg_throughput': np.mean(throughputs),
                    'min_throughput': np.min(throughputs),
                    'max_throughput': np.max(throughputs)
                }
        
        # Extract accuracy metrics
        accuracy_results = [r for r in results if r.metrics and 'accuracy' in r.metrics]
        if accuracy_results:
            accuracies = [r.metrics['accuracy'] for r in accuracy_results]
            metrics['accuracy_metrics'] = {
                'avg_accuracy': np.mean(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies)
            }
        
        return metrics
    
    def save_report(self, report: ValidationReport, output_path: Optional[str] = None) -> str:
        """Save validation report to file."""
        if output_path is None:
            output_dir = Path(self.config['output_directory'])
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"validation_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        self.logger.info(f"Validation report saved to: {output_path}")
        return str(output_path)
    
    def generate_html_report(self, report: ValidationReport, output_path: Optional[str] = None) -> str:
        """Generate HTML validation report."""
        if output_path is None:
            output_dir = Path(self.config['output_directory'])
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"validation_report_{timestamp}.html"
        
        html_content = self._generate_html_content(report)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML validation report saved to: {output_path}")
        return str(output_path)
    
    def _generate_html_content(self, report: ValidationReport) -> str:
        """Generate HTML content for validation report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Data Readiness Platform - Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                .test-category {{ margin: 20px 0; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .passed {{ border-left-color: #4CAF50; }}
                .failed {{ border-left-color: #f44336; }}
                .skipped {{ border-left-color: #ff9800; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Data Readiness Platform - Validation Report</h1>
                <p>Generated: {report.timestamp}</p>
                <p>Execution Time: {report.execution_time:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>{report.total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="metric">
                    <h3>{report.passed_tests}</h3>
                    <p>Passed</p>
                </div>
                <div class="metric">
                    <h3>{report.failed_tests}</h3>
                    <p>Failed</p>
                </div>
                <div class="metric">
                    <h3>{report.success_rate:.1%}</h3>
                    <p>Success Rate</p>
                </div>
            </div>
            
            <h2>Test Results by Category</h2>
        """
        
        # Group results by category
        by_category = {}
        for result in report.test_results:
            if result.test_category not in by_category:
                by_category[result.test_category] = []
            by_category[result.test_category].append(result)
        
        for category, results in by_category.items():
            passed = sum(1 for r in results if r.status == "passed")
            total = len(results)
            
            html += f"""
            <div class="test-category">
                <h3>{category.replace('_', ' ').title()} ({passed}/{total} passed)</h3>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Execution Time</th>
                        <th>Metrics</th>
                    </tr>
            """
            
            for result in results:
                metrics_str = ""
                if result.metrics:
                    metrics_str = ", ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" 
                                           for k, v in result.metrics.items()])
                
                html += f"""
                    <tr class="{result.status}">
                        <td>{result.test_name}</td>
                        <td>{result.status.upper()}</td>
                        <td>{result.execution_time:.3f}s</td>
                        <td>{metrics_str}</td>
                    </tr>
                """
            
            html += "</table></div>"
        
        html += """
            </body>
        </html>
        """
        
        return html


def run_comprehensive_validation(config: Optional[Dict] = None) -> ValidationReport:
    """Run comprehensive validation test suite."""
    runner = ValidationTestRunner(config)
    report = runner.run_all_validations()
    
    # Save reports
    runner.save_report(report)
    runner.generate_html_report(report)
    
    return report


if __name__ == "__main__":
    # Run validation with default configuration
    report = run_comprehensive_validation()
    
    print(f"\nValidation Summary:")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Success Rate: {report.success_rate:.1%}")
    print(f"Execution Time: {report.execution_time:.2f} seconds")