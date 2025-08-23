# Agent Steering System Integration Tests

This directory contains comprehensive integration tests for the Agent Steering System, designed to validate enterprise-grade functionality, performance, and security across all system components.

## Overview

The integration test suite validates the complete Agent Steering System implementation according to the requirements specified in task 16 of the implementation plan:

- **Enterprise Connectors**: Tests for SAP, Salesforce, Snowflake, Oracle, and other enterprise system integrations
- **End-to-End Workflows**: Complete business scenarios from data ingestion to decision making
- **Performance Testing**: Enterprise-scale load testing and performance validation
- **Security Penetration**: Comprehensive security testing and vulnerability assessment

## Test Structure

```
tests/integration/
├── README.md                           # This file
├── conftest.py                         # Test configuration and fixtures
├── integration_test_config.json       # Test suite configuration
├── test_runner.py                      # Python test runner
├── test_integration_runner.py         # Comprehensive test orchestrator
├── test_enterprise_connectors.py      # Enterprise connector tests
├── test_end_to_end_workflows.py       # Business workflow tests
├── test_performance.py                # Performance and scalability tests
├── test_security_penetration.py       # Security penetration tests
├── test_agent_interactions.py         # Multi-agent coordination tests
├── test_data_pipelines.py             # Data pipeline integration tests
├── test_ci_cd_pipeline.py             # CI/CD pipeline validation
└── test_smoke_tests.py                # Basic smoke tests
```

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-xdist`, `pytest-timeout`
- Docker (optional, for containerized services)
- Access to test databases and services

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Install additional test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout pytest-html pytest-json-report
```

### Running Tests

#### Run All Tests (Recommended)

```bash
# Linux/macOS
./scripts/run-integration-tests.sh

# Windows
scripts\run-integration-tests.bat
```

#### Run Specific Test Suites

```bash
# Run enterprise connector tests only
python -m pytest tests/integration/test_enterprise_connectors.py -v

# Run performance tests only
python -m pytest tests/integration/test_performance.py -v

# Run security tests only
python -m pytest tests/integration/test_security_penetration.py -v
```

#### Run with Custom Configuration

```bash
# Use custom configuration file
./scripts/run-integration-tests.sh -c custom_config.json

# Run specific suite with custom output directory
./scripts/run-integration-tests.sh -s performance -o /tmp/test_results
```

## Test Suites

### 1. Enterprise Connectors (`test_enterprise_connectors.py`)

Tests integration with enterprise systems:

- **SAP Integration**: Connection, data extraction, real-time streaming
- **Salesforce CRM**: SOQL queries, bulk operations, change tracking
- **Snowflake Data Lake**: SQL queries, data loading, streaming
- **Multi-Connector Pipelines**: Coordinated data processing across systems
- **Error Handling**: Connection failures, retry mechanisms, data corruption
- **Performance**: Large dataset processing, concurrent operations
- **Security**: SSL/TLS validation, authentication, data masking

**Key Test Scenarios:**
```python
# SAP connector with real-world data volumes
async def test_sap_connector_integration()

# Salesforce bulk data extraction
async def test_salesforce_connector_integration()

# Multi-system data pipeline coordination
async def test_multi_connector_data_pipeline()

# Performance under enterprise loads
async def test_performance_benchmarks()
```

### 2. End-to-End Workflows (`test_end_to_end_workflows.py`)

Tests complete business scenarios:

- **Customer Churn Prediction**: ML analysis workflow with business validation
- **Fraud Detection**: Real-time transaction processing and alerting
- **Supply Chain Optimization**: Multi-system data analysis and recommendations
- **Market Intelligence**: Comprehensive competitive analysis workflow
- **Multi-Agent Coordination**: Complex agent collaboration scenarios
- **Error Recovery**: Workflow resilience and failure handling

**Key Test Scenarios:**
```python
# Complete customer churn prediction workflow
async def test_customer_churn_prediction_workflow()

# Real-time fraud detection with sub-second response
async def test_fraud_detection_workflow()

# Multi-agent coordination for complex analysis
async def test_multi_agent_coordination()

# Workflow error recovery and resilience
async def test_workflow_error_recovery()
```

### 3. Performance Testing (`test_performance.py`)

Tests system performance under enterprise loads:

- **Concurrent User Scaling**: 10 to 1,000+ concurrent users
- **Data Volume Processing**: 1K to 1M+ record datasets
- **Complex Workflow Performance**: Multi-agent enterprise scenarios
- **Memory Optimization**: Efficient resource utilization
- **Real-Time Processing**: High-throughput event processing
- **Database Performance**: Concurrent query optimization
- **Stress Testing**: Extreme load conditions and recovery

**Key Test Scenarios:**
```python
# Concurrent user scaling validation
async def test_concurrent_user_scaling()

# Large dataset processing performance
async def test_data_volume_scaling()

# Real-time event processing throughput
async def test_real_time_processing_performance()

# System behavior under extreme load
async def test_extreme_concurrent_load()
```

### 4. Security Penetration (`test_security_penetration.py`)

Comprehensive security testing:

- **Authentication Security**: Brute force protection, session management, JWT validation
- **Data Protection**: Encryption, masking, SQL injection prevention
- **Network Security**: DDoS protection, SSL/TLS validation, API security
- **Threat Detection**: Anomaly detection, malware scanning, intrusion detection
- **Incident Response**: Automated security response and escalation

**Key Test Scenarios:**
```python
# Brute force attack protection
async def test_brute_force_protection()

# SQL injection prevention
async def test_sql_injection_protection()

# DDoS attack mitigation
async def test_ddos_protection()

# Automated incident response
async def test_incident_response()
```

## Configuration

### Test Configuration File (`integration_test_config.json`)

The test suite uses a comprehensive configuration file that defines:

- **Test Suites**: Individual test configurations with timeouts and requirements
- **Reporting**: Output formats, coverage thresholds, and report generation
- **Environment**: Setup/teardown commands and environment variables
- **Performance**: Load scenarios, metrics, and resource limits
- **Security**: Penetration test configurations and compliance checks
- **Notifications**: Alert configurations for test results

### Key Configuration Sections

```json
{
  "test_suites": [
    {
      "name": "enterprise_connectors",
      "timeout": 600,
      "required": true,
      "parallel": false
    }
  ],
  "thresholds": {
    "success_rate": 0.95,
    "performance_threshold": 1800,
    "security_score_threshold": 0.90
  },
  "performance": {
    "concurrent_users": [10, 50, 100, 500, 1000],
    "data_volumes": [1000, 10000, 100000, 1000000]
  }
}
```

## Test Data

### Generated Test Data

The test suite automatically generates realistic test data:

- **Customers**: 10,000 customer records with realistic attributes
- **Transactions**: 100,000 transaction records with temporal patterns
- **Agents**: 50 agent configurations with various capabilities
- **Business Scenarios**: Realistic business contexts and objectives

### Mock Services

For testing enterprise connectors without actual systems:

- **SAP Mock Service**: Simulates SAP RFC calls and table data
- **Salesforce Mock API**: Simulates SOQL queries and bulk operations
- **Snowflake Mock Warehouse**: Simulates SQL queries and data loading

## Reporting

### Test Reports

The integration test suite generates comprehensive reports:

- **JSON Report**: Machine-readable test results and metrics
- **HTML Report**: Human-readable test results with visualizations
- **JUnit XML**: CI/CD integration compatible format
- **Coverage Report**: Code coverage analysis with HTML visualization

### Report Locations

```
test_results/integration/
├── summary.json                    # Overall test summary
├── enterprise_connectors/
│   ├── report.html                # HTML test report
│   ├── report.json                # JSON test results
│   ├── coverage/                  # Coverage analysis
│   └── junit.xml                  # JUnit XML format
├── performance/
│   ├── benchmark.json             # Performance benchmarks
│   └── load_test_results.json     # Load testing metrics
└── security_penetration/
    ├── vulnerability_report.json  # Security assessment
    └── compliance_report.html     # Compliance validation
```

### Metrics and Thresholds

The test suite validates against enterprise-grade thresholds:

- **Success Rate**: ≥95% test pass rate
- **Performance**: ≤30s response time for complex workflows
- **Security Score**: ≥90% security assessment score
- **Coverage**: ≥80% code coverage
- **Throughput**: ≥100 requests/second for simple operations

## CI/CD Integration

### GitHub Actions Integration

```yaml
name: Integration Tests
on: [push, pull_request]
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: ./scripts/run-integration-tests.sh
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: integration-test-results
          path: test_results/
```

### Jenkins Integration

```groovy
pipeline {
    agent any
    stages {
        stage('Integration Tests') {
            steps {
                sh './scripts/run-integration-tests.sh'
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test_results/integration',
                        reportFiles: '*/report.html',
                        reportName: 'Integration Test Report'
                    ])
                    junit 'test_results/integration/*/junit.xml'
                }
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Test Timeouts**
   ```bash
   # Increase timeout for specific tests
   pytest tests/integration/test_performance.py --timeout=1200
   ```

2. **Memory Issues with Large Datasets**
   ```bash
   # Run tests with memory profiling
   pytest tests/integration/test_performance.py --memray
   ```

3. **Database Connection Issues**
   ```bash
   # Check database connectivity
   python -c "from scrollintel.core.database import test_connection; test_connection()"
   ```

4. **Missing Dependencies**
   ```bash
   # Install all test dependencies
   pip install -r tests/integration/requirements.txt
   ```

### Debug Mode

Run tests in debug mode for detailed output:

```bash
# Enable debug logging
export SCROLLINTEL_LOG_LEVEL=DEBUG
pytest tests/integration/ -v -s --log-cli-level=DEBUG
```

### Performance Debugging

Profile test performance:

```bash
# Run with performance profiling
pytest tests/integration/test_performance.py --profile --profile-svg
```

## Contributing

### Adding New Tests

1. **Create Test File**: Follow naming convention `test_<component>.py`
2. **Add Configuration**: Update `integration_test_config.json`
3. **Update Test Runner**: Add to test suite list in runner scripts
4. **Documentation**: Update this README with test descriptions

### Test Guidelines

- **Real Data**: Use realistic test data, avoid hardcoded values
- **Enterprise Scale**: Test with enterprise-appropriate data volumes
- **Error Scenarios**: Include comprehensive error handling tests
- **Performance**: Validate performance requirements for all tests
- **Security**: Include security validation in all components
- **Documentation**: Document test scenarios and expected outcomes

### Code Quality

- **Type Hints**: Use type hints for all test functions
- **Async/Await**: Use async patterns for I/O operations
- **Mocking**: Mock external services appropriately
- **Assertions**: Use descriptive assertion messages
- **Cleanup**: Ensure proper test cleanup and resource management

## Support

For issues with integration tests:

1. **Check Logs**: Review test logs in `test_results/integration/`
2. **Validate Environment**: Ensure all prerequisites are installed
3. **Run Individual Tests**: Isolate failing tests for debugging
4. **Check Configuration**: Validate test configuration settings
5. **Contact Team**: Reach out to the development team for assistance

## License

This test suite is part of the ScrollIntel Agent Steering System and is subject to the same licensing terms as the main project.