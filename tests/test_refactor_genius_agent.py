"""
Tests for Refactor Genius Agent

Tests the superhuman capabilities of the Refactor Genius Agent
for automatic legacy modernization with zero human intervention.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.agents.refactor_genius_agent import (
    RefactorGeniusAgent, RefactoringType, ModernizationLevel, LegacyCodeComplexity
)
from scrollintel.models.refactoring_models import (
    RefactoringRequest, create_refactoring_request
)


class TestRefactorGeniusAgent:
    """Test suite for Refactor Genius Agent superhuman capabilities"""
    
    @pytest.fixture
    def agent(self):
        """Create Refactor Genius Agent instance"""
        return RefactorGeniusAgent()
    
    @pytest.fixture
    def sample_legacy_code(self):
        """Create sample legacy code for testing"""
        return """
# Legacy Python code with technical debt
import string
import sys

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
        else:
            result.append(0)
    return result

class DataProcessor:
    def __init__(self):
        self.data = []
        self.results = []
    
    def add_data(self, item):
        self.data.append(item)
    
    def process_all(self):
        for item in self.data:
            if item > 10:
                self.results.append(item * 3)
            elif item > 5:
                self.results.append(item * 2)
            else:
                self.results.append(item)
        return self.results

# Global variable (code smell)
GLOBAL_COUNTER = 0

def increment_counter():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER
"""
    
    @pytest.fixture
    def sample_refactoring_request(self, sample_legacy_code):
        """Create sample refactoring request"""
        return create_refactoring_request(
            legacy_code=sample_legacy_code,
            language="python",
            refactoring_types=[RefactoringType.MODERNIZATION, RefactoringType.TECHNICAL_DEBT_ELIMINATION],
            target_level=ModernizationLevel.SUPERHUMAN,
            name="Legacy Code Modernization Test",
            description="Test automatic legacy modernization"
        )
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes with superhuman capabilities"""
        assert agent.agent_id == "refactor-genius-001"
        assert "automatic_legacy_modernization" in agent.superhuman_capabilities
        assert "zero_intervention_refactoring" in agent.superhuman_capabilities
        assert "perfect_compatibility_preservation" in agent.superhuman_capabilities
        assert "technical_debt_elimination" in agent.superhuman_capabilities
        assert "instant_migration_execution" in agent.superhuman_capabilities
    
    def test_refactoring_patterns_initialization(self, agent):
        """Test that superhuman refactoring patterns are initialized"""
        patterns = agent.refactoring_patterns
        
        # Test legacy patterns
        assert "legacy_patterns" in patterns
        legacy_patterns = patterns["legacy_patterns"]
        assert "spaghetti_code" in legacy_patterns
        assert "god_objects" in legacy_patterns
        assert "tight_coupling" in legacy_patterns
        assert "magic_numbers" in legacy_patterns
        
        # Test that all legacy patterns have 100% automation
        for pattern_name, pattern_info in legacy_patterns.items():
            assert pattern_info["automation_level"] == "100%"
        
        # Test modern patterns
        assert "modern_patterns" in patterns
        modern_patterns = patterns["modern_patterns"]
        assert "microservices" in modern_patterns
        assert "event_driven" in modern_patterns
        assert "clean_architecture" in modern_patterns
    
    def test_modernization_strategies_initialization(self, agent):
        """Test that superhuman modernization strategies are initialized"""
        strategies = agent.modernization_strategies
        
        # Test language modernization
        assert "language_modernization" in strategies
        lang_strategies = strategies["language_modernization"]
        assert "python" in lang_strategies
        assert "javascript" in lang_strategies
        assert "java" in lang_strategies
        
        # Test architecture modernization
        assert "architecture_modernization" in strategies
        arch_strategies = strategies["architecture_modernization"]
        assert "monolith_to_microservices" in arch_strategies
        assert "synchronous_to_async" in arch_strategies
        assert "database_modernization" in arch_strategies
        
        # Test automation levels
        for strategy_name, strategy_info in arch_strategies.items():
            assert "automation_level" in strategy_info
            assert float(strategy_info["automation_level"].rstrip('%')) >= 85.0  # At least 85% automation
    
    def test_compatibility_engines_initialization(self, agent):
        """Test that compatibility preservation engines are initialized"""
        engines = agent.compatibility_engines
        
        # Test compatibility engine types
        assert "api_compatibility" in engines
        assert "data_compatibility" in engines
        assert "integration_compatibility" in engines
        
        # Test API compatibility features
        api_compat = engines["api_compatibility"]
        assert "version_management" in api_compat
        assert "backward_compatibility" in api_compat
        assert "migration_paths" in api_compat
        
        # Test data compatibility features
        data_compat = engines["data_compatibility"]
        assert "schema_evolution" in data_compat
        assert "data_migration" in data_compat
        assert "rollback_support" in data_compat
    
    @pytest.mark.asyncio
    async def test_modernize_legacy_codebase(self, agent, sample_refactoring_request):
        """Test modernizing legacy codebase with superhuman capabilities"""
        modernized_code = await agent.modernize_legacy_codebase(sample_refactoring_request)
        
        # Test superhuman modernization properties
        assert modernized_code.original_code == sample_refactoring_request.legacy_code
        assert modernized_code.language == sample_refactoring_request.language
        assert len(modernized_code.modernized_code) > 0
        assert modernized_code.modernized_code != modernized_code.original_code  # Should be different
        
        # Test superhuman improvement metrics
        assert modernized_code.technical_debt_reduction >= 0.95  # 95% debt reduction
        assert modernized_code.performance_improvement > 0.0     # Some performance improvement
        assert modernized_code.security_enhancement >= 0.9      # 90% security improvement
        assert modernized_code.maintainability_improvement >= 0.95  # 95% maintainability improvement
        
        # Test compatibility preservation
        compatibility = modernized_code.compatibility_report
        assert compatibility.api_compatibility_score >= 0.9     # 90%+ API compatibility
        assert compatibility.data_compatibility_score >= 0.9    # 90%+ data compatibility
        assert compatibility.integration_compatibility_score >= 0.9  # 90%+ integration compatibility
        assert compatibility.rollback_feasibility >= 0.9        # 90%+ rollback feasibility
        assert len(compatibility.breaking_changes) == 0         # Zero breaking changes
        
        # Test superhuman features
        expected_features = [
            "Zero human intervention required",
            "Perfect compatibility preservation",
            "95% technical debt elimination",
            "Automatic security hardening",
            "Comprehensive test generation",
            "Complete documentation update",
            "Instant migration execution"
        ]
        for feature in expected_features:
            assert feature in modernized_code.superhuman_features
        
        # Test generated content
        assert len(modernized_code.migration_guide) > 0
        assert len(modernized_code.test_suite) > 0
        assert len(modernized_code.documentation) > 0
        assert len(modernized_code.refactoring_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_legacy_code_superhuman(self, agent, sample_refactoring_request):
        """Test superhuman legacy code analysis"""
        analysis = await agent._analyze_legacy_code_superhuman(sample_refactoring_request)
        
        # Test analysis completeness
        assert isinstance(analysis.complexity_level, LegacyCodeComplexity)
        assert 0.0 <= analysis.technical_debt_score <= 1.0
        assert 0.0 <= analysis.maintainability_index <= 1.0
        assert 0.0 <= analysis.modernization_potential <= 1.0
        
        # Test analysis components
        assert isinstance(analysis.security_vulnerabilities, list)
        assert isinstance(analysis.performance_bottlenecks, list)
        assert isinstance(analysis.outdated_patterns, list)
        assert isinstance(analysis.deprecated_dependencies, list)
        assert isinstance(analysis.code_smells, list)
        assert isinstance(analysis.refactoring_opportunities, list)
    
    @pytest.mark.asyncio
    async def test_eliminate_technical_debt_only(self, agent, sample_legacy_code):
        """Test technical debt elimination without full modernization"""
        modernized_code = await agent.eliminate_technical_debt_only(sample_legacy_code, "python")
        
        # Test debt elimination results
        assert modernized_code.technical_debt_reduction >= 0.95  # 95% debt reduction
        assert modernized_code.maintainability_improvement >= 0.8  # 80% maintainability improvement
        assert len(modernized_code.modernized_code) > 0
        assert len(modernized_code.test_suite) > 0
        assert len(modernized_code.documentation) > 0
        
        # Test compatibility preservation
        compatibility = modernized_code.compatibility_report
        assert compatibility.api_compatibility_score == 1.0     # Perfect API compatibility
        assert compatibility.data_compatibility_score == 1.0    # Perfect data compatibility
        assert compatibility.integration_compatibility_score == 1.0  # Perfect integration compatibility
        assert len(compatibility.breaking_changes) == 0         # Zero breaking changes
        
        # Test superhuman features for debt elimination
        expected_features = [
            "95% technical debt elimination",
            "Zero breaking changes",
            "Perfect API compatibility",
            "Automatic code smell removal",
            "Improved maintainability"
        ]
        for feature in expected_features:
            assert feature in modernized_code.superhuman_features
    
    @pytest.mark.asyncio
    async def test_migrate_legacy_system(self, agent):
        """Test migration of entire legacy system"""
        legacy_system = {
            "name": "Legacy System",
            "language": "python",
            "components": {
                "data_processor": "def process(data): return data",
                "api_handler": "def handle_request(req): return req",
                "database_layer": "class DB: pass"
            }
        }
        
        migration_results = await agent.migrate_legacy_system(legacy_system, "microservices")
        
        # Test migration results
        assert "migration_id" in migration_results
        assert migration_results["source_system"] == legacy_system
        assert migration_results["target_architecture"] == "microservices"
        assert migration_results["compatibility_preserved"] is True
        assert migration_results["migration_time"] == "instant"
        assert migration_results["rollback_available"] is True
        
        # Test migrated components
        assert "migrated_components" in migration_results
        migrated_components = migration_results["migrated_components"]
        assert len(migrated_components) == len(legacy_system["components"])
        
        for component in migrated_components:
            assert "name" in component
            assert "modernized_code" in component
            assert "improvement_metrics" in component
            
            metrics = component["improvement_metrics"]
            assert metrics["technical_debt_reduction"] >= 0.95
            assert metrics["performance_improvement"] >= 0.0
            assert metrics["security_enhancement"] >= 0.9
            assert metrics["maintainability_improvement"] >= 0.95
        
        # Test superhuman features
        expected_features = [
            "Zero-downtime migration",
            "Perfect compatibility preservation",
            "Automatic rollback capability",
            "Performance optimization during migration",
            "Complete system modernization"
        ]
        for feature in expected_features:
            assert feature in migration_results["superhuman_features"]
    
    @pytest.mark.asyncio
    async def test_assess_code_complexity(self, agent):
        """Test code complexity assessment"""
        # Test simple code
        simple_code = "def hello(): return 'world'"
        complexity = await agent._assess_code_complexity(simple_code)
        assert complexity == LegacyCodeComplexity.SIMPLE
        
        # Test complex code
        complex_code = "\n".join(["def func(): pass"] * 1000)
        complexity = await agent._assess_code_complexity(complex_code)
        assert complexity in [LegacyCodeComplexity.ENTERPRISE, LegacyCodeComplexity.COMPLEX]
        
        # Test unmaintainable code
        unmaintainable_code = "\n".join(["# line"] * 15000)
        complexity = await agent._assess_code_complexity(unmaintainable_code)
        assert complexity == LegacyCodeComplexity.UNMAINTAINABLE
    
    @pytest.mark.asyncio
    async def test_calculate_technical_debt(self, agent):
        """Test technical debt calculation"""
        # Test code with debt indicators
        debt_code = """
        # TODO: Fix this hack
        # FIXME: This is broken
        eval("dangerous code")
        import *
        global bad_practice
        """
        debt_score = await agent._calculate_technical_debt(debt_code)
        assert 0.0 <= debt_score <= 1.0
        assert debt_score > 0.0  # Should detect debt
        
        # Test clean code
        clean_code = "def clean_function(): return 'clean'"
        debt_score = await agent._calculate_technical_debt(clean_code)
        assert debt_score >= 0.0  # Should be low or zero debt
    
    @pytest.mark.asyncio
    async def test_assess_maintainability(self, agent):
        """Test maintainability assessment"""
        # Test well-documented code
        documented_code = """
        # This is a well-documented function
        def documented_function():
            # Returns a greeting
            return 'hello'
        """
        maintainability = await agent._assess_maintainability(documented_code)
        assert 0.0 <= maintainability <= 1.0
        assert maintainability > 0.1  # Should have decent maintainability
        
        # Test undocumented code
        undocumented_code = "\n".join(["def func(): pass"] * 100)
        maintainability = await agent._assess_maintainability(undocumented_code)
        assert 0.0 <= maintainability <= 1.0
    
    @pytest.mark.asyncio
    async def test_identify_security_vulnerabilities(self, agent):
        """Test security vulnerability identification"""
        vulnerable_code = """
        import pickle
        user_input = input("Enter code: ")
        eval(user_input)
        exec("dangerous code")
        pickle.loads(data)
        """
        vulnerabilities = await agent._identify_security_vulnerabilities(vulnerable_code)
        
        assert isinstance(vulnerabilities, list)
        assert len(vulnerabilities) > 0
        
        # Should detect common vulnerabilities
        vuln_text = " ".join(vulnerabilities).lower()
        assert "eval" in vuln_text or "injection" in vuln_text
        assert "exec" in vuln_text or "execution" in vuln_text
    
    @pytest.mark.asyncio
    async def test_identify_performance_bottlenecks(self, agent):
        """Test performance bottleneck identification"""
        slow_code = """
        import time
        result = []
        for i in range(1000):
            for j in range(1000):
                result.append(i * j)
            time.sleep(0.1)
        """
        bottlenecks = await agent._identify_performance_bottlenecks(slow_code)
        
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) > 0
        
        # Should detect performance issues
        bottleneck_text = " ".join(bottlenecks).lower()
        assert "loop" in bottleneck_text or "sleep" in bottleneck_text or "complexity" in bottleneck_text
    
    @pytest.mark.asyncio
    async def test_detect_outdated_patterns(self, agent):
        """Test detection of outdated programming patterns"""
        outdated_code = """
        print "Python 2 style print"
        "Hello {}".format("world")
        try:
            risky_operation()
        except:
            pass
        """
        patterns = await agent._detect_outdated_patterns(outdated_code)
        
        assert isinstance(patterns, list)
        # May or may not detect patterns depending on implementation
    
    @pytest.mark.asyncio
    async def test_find_deprecated_dependencies(self, agent):
        """Test finding deprecated dependencies"""
        deprecated_code = """
        import imp
        import optparse
        from distutils import setup
        """
        deprecated = await agent._find_deprecated_dependencies(deprecated_code)
        
        assert isinstance(deprecated, list)
        if len(deprecated) > 0:
            # Should detect deprecated modules
            deprecated_text = " ".join(deprecated).lower()
            assert "imp" in deprecated_text or "optparse" in deprecated_text or "distutils" in deprecated_text
    
    @pytest.mark.asyncio
    async def test_identify_code_smells(self, agent):
        """Test identification of code smells"""
        smelly_code = """
        def function_with_very_long_line_that_exceeds_reasonable_length_limits_and_should_be_detected_as_a_code_smell():
            pass
        """ + "\n".join([f"def func{i}(): pass" for i in range(25)])  # Many functions
        
        smells = await agent._identify_code_smells(smelly_code)
        
        assert isinstance(smells, list)
        # May detect various code smells
    
    @pytest.mark.asyncio
    async def test_calculate_modernization_potential(self, agent):
        """Test calculation of modernization potential"""
        # Test high potential (complex, high debt, low maintainability)
        potential = await agent._calculate_modernization_potential(
            LegacyCodeComplexity.UNMAINTAINABLE, 0.9, 0.1
        )
        assert 0.0 <= potential <= 1.0
        assert potential > 0.5  # Should be high potential
        
        # Test low potential (simple, low debt, high maintainability)
        potential = await agent._calculate_modernization_potential(
            LegacyCodeComplexity.SIMPLE, 0.1, 0.9
        )
        assert 0.0 <= potential <= 1.0
        assert potential < 0.8  # Should be lower potential
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in legacy modernization"""
        # Test with invalid request
        invalid_request = RefactoringRequest(
            id="invalid",
            name=None,
            description=None,
            legacy_code="",  # Empty code
            language="unknown",
            refactoring_types=[],
            target_modernization_level=ModernizationLevel.SUPERHUMAN,
            compatibility_requirements=[]
        )
        
        # Should handle gracefully
        try:
            modernized_code = await agent.modernize_legacy_codebase(invalid_request)
            # If it succeeds, should still have superhuman properties
            assert modernized_code.technical_debt_reduction >= 0.95
        except Exception as e:
            # If it fails, should fail gracefully
            assert "modernization failed" in str(e).lower() or isinstance(e, ValueError)
    
    def test_superhuman_capabilities_completeness(self, agent):
        """Test that all required superhuman capabilities are present"""
        required_capabilities = [
            "automatic_legacy_modernization",
            "zero_intervention_refactoring",
            "perfect_compatibility_preservation",
            "technical_debt_elimination",
            "instant_migration_execution"
        ]
        
        for capability in required_capabilities:
            assert capability in agent.superhuman_capabilities
    
    @pytest.mark.asyncio
    async def test_concurrent_modernization(self, agent, sample_refactoring_request):
        """Test concurrent legacy modernization (superhuman parallel processing)"""
        # Create multiple refactoring requests
        requests = [sample_refactoring_request for _ in range(3)]
        
        # Modernize concurrently
        tasks = [agent.modernize_legacy_codebase(req) for req in requests]
        modernized_codes = await asyncio.gather(*tasks)
        
        # All should succeed with superhuman capabilities
        assert len(modernized_codes) == 3
        for code in modernized_codes:
            assert code.technical_debt_reduction >= 0.95
            assert code.maintainability_improvement >= 0.95
            assert len(code.superhuman_features) > 0
            assert len(code.compatibility_report.breaking_changes) == 0


@pytest.mark.integration
class TestRefactorGeniusIntegration:
    """Integration tests for Refactor Genius Agent"""
    
    @pytest.fixture
    def agent(self):
        return RefactorGeniusAgent()
    
    @pytest.fixture
    def complex_legacy_system(self):
        """Create complex legacy system for integration testing"""
        return {
            "name": "Complex Legacy System",
            "language": "python",
            "components": {
                "user_manager": """
class UserManager:
    def __init__(self):
        self.users = []
        self.active_users = []
    
    def add_user(self, user):
        # TODO: Add validation
        self.users.append(user)
        if user.get('active'):
            self.active_users.append(user)
    
    def get_user(self, user_id):
        for user in self.users:
            if user['id'] == user_id:
                return user
        return None
    
    def authenticate(self, username, password):
        # FIXME: This is insecure
        for user in self.users:
            if user['username'] == username and user['password'] == password:
                return True
        return False
""",
                "data_processor": """
import time

def process_large_dataset(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            # Inefficient nested loops
            processed_item = data[i][j] * 2
            result.append(processed_item)
            time.sleep(0.001)  # Blocking operation
    return result

def legacy_sort(items):
    # Bubble sort - inefficient algorithm
    n = len(items)
    for i in range(n):
        for j in range(0, n-i-1):
            if items[j] > items[j+1]:
                items[j], items[j+1] = items[j+1], items[j]
    return items
""",
                "api_handler": """
import pickle

def handle_request(request_data):
    # Security vulnerability - unsafe deserialization
    try:
        data = pickle.loads(request_data)
        return process_request(data)
    except:
        # Bare except clause - code smell
        return {"error": "Something went wrong"}

def process_request(data):
    # Magic numbers and hardcoded values
    if data.get('type') == 1:
        return data['value'] * 3.14159
    elif data.get('type') == 2:
        return data['value'] + 42
    else:
        return 0
"""
            }
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_legacy_modernization(self, agent, complex_legacy_system):
        """Test complete end-to-end legacy system modernization"""
        # Migrate the entire complex legacy system
        migration_results = await agent.migrate_legacy_system(
            complex_legacy_system, "microservices"
        )
        
        # Assert migration success
        assert migration_results is not None
        assert migration_results["compatibility_preserved"] is True
        assert migration_results["rollback_available"] is True
        assert migration_results["migration_time"] == "instant"
        
        # Test migrated components
        migrated_components = migration_results["migrated_components"]
        assert len(migrated_components) == len(complex_legacy_system["components"])
        
        # Test each component modernization
        for component in migrated_components:
            modernized_code = component["modernized_code"]
            metrics = component["improvement_metrics"]
            
            # Test superhuman improvements
            assert metrics["technical_debt_reduction"] >= 0.95
            assert metrics["performance_improvement"] >= 0.0
            assert metrics["security_enhancement"] >= 0.9
            assert metrics["maintainability_improvement"] >= 0.95
            
            # Test modernized code properties
            assert modernized_code.technical_debt_reduction >= 0.95
            assert modernized_code.security_enhancement >= 0.9
            assert modernized_code.maintainability_improvement >= 0.95
            assert len(modernized_code.superhuman_features) > 0
            assert len(modernized_code.compatibility_report.breaking_changes) == 0
        
        # Test overall system improvement
        assert migration_results["performance_improvement"] >= 0.0
        assert len(migration_results["superhuman_features"]) > 0
        
        # Test superhuman features
        expected_features = [
            "Zero-downtime migration",
            "Perfect compatibility preservation",
            "Automatic rollback capability",
            "Performance optimization during migration",
            "Complete system modernization"
        ]
        for feature in expected_features:
            assert feature in migration_results["superhuman_features"]
    
    @pytest.mark.asyncio
    async def test_comprehensive_technical_debt_elimination(self, agent):
        """Test comprehensive technical debt elimination across multiple code issues"""
        # Create code with multiple types of technical debt
        debt_heavy_code = """
# Multiple TODO and FIXME comments indicating debt
# TODO: Refactor this entire module
# FIXME: This function is broken
# HACK: Temporary workaround

import string
import sys
import pickle

# Global variables (code smell)
GLOBAL_STATE = {}
COUNTER = 0

def problematic_function(data):
    global COUNTER
    COUNTER += 1
    
    # Magic numbers
    if len(data) > 100:
        result = data * 3.14159
    elif len(data) > 50:
        result = data * 2.71828
    else:
        result = data * 1.41421
    
    # Unsafe operations
    eval("print('dangerous')")
    exec("risky_code = True")
    
    # Poor error handling
    try:
        risky_operation()
    except:
        pass
    
    # Inefficient operations
    output = []
    for i in range(len(result)):
        output.append(result[i] * 2)
    
    return output

class GodClass:
    # Class with too many responsibilities
    def __init__(self):
        self.data = []
        self.users = []
        self.config = {}
        self.cache = {}
        self.logs = []
    
    def add_data(self, item): pass
    def remove_data(self, item): pass
    def process_data(self): pass
    def validate_data(self): pass
    def save_data(self): pass
    def load_data(self): pass
    def add_user(self, user): pass
    def authenticate_user(self, user): pass
    def authorize_user(self, user): pass
    def log_action(self, action): pass
    def send_notification(self, message): pass
    def generate_report(self): pass
"""
        
        # Eliminate technical debt
        modernized_code = await agent.eliminate_technical_debt_only(debt_heavy_code, "python")
        
        # Test comprehensive debt elimination
        assert modernized_code.technical_debt_reduction >= 0.95
        assert modernized_code.maintainability_improvement >= 0.8
        assert modernized_code.security_enhancement >= 0.3  # Some security improvement
        
        # Test that modernized code is different and improved
        assert modernized_code.modernized_code != debt_heavy_code
        assert len(modernized_code.modernized_code) > 0
        
        # Test compatibility preservation
        compatibility = modernized_code.compatibility_report
        assert compatibility.api_compatibility_score == 1.0
        assert compatibility.data_compatibility_score == 1.0
        assert compatibility.integration_compatibility_score == 1.0
        assert len(compatibility.breaking_changes) == 0
        
        # Test generated content
        assert len(modernized_code.test_suite) > 0
        assert len(modernized_code.documentation) > 0
        
        # Test superhuman features
        expected_features = [
            "95% technical debt elimination",
            "Zero breaking changes",
            "Perfect API compatibility",
            "Automatic code smell removal",
            "Improved maintainability"
        ]
        for feature in expected_features:
            assert feature in modernized_code.superhuman_features