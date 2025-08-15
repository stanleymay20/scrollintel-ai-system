"""
Tests for protocol generation system.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.protocol_generator import (
    ProtocolGenerator, ProtocolTemplate, ProtocolOptimization, ProtocolValidation
)
from scrollintel.models.experimental_design_models import (
    ExperimentalProtocol, MethodologyType, ExperimentType,
    ExperimentalVariable, ExperimentalCondition, Hypothesis
)


class TestProtocolGenerator:
    """Test cases for ProtocolGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ProtocolGenerator()
        
        # Create sample experimental components
        self.variables = [
            ExperimentalVariable(
                name="temperature",
                variable_type="independent",
                data_type="continuous",
                measurement_unit="celsius"
            ),
            ExperimentalVariable(
                name="reaction_rate",
                variable_type="dependent",
                data_type="continuous",
                measurement_unit="mol/s"
            )
        ]
        
        self.conditions = [
            ExperimentalCondition(
                condition_id="control",
                name="Control Condition",
                variables={"temperature": 25},
                sample_size=30
            ),
            ExperimentalCondition(
                condition_id="treatment",
                name="Treatment Condition",
                variables={"temperature": 50},
                sample_size=30
            )
        ]
        
        self.hypotheses = [
            Hypothesis(
                hypothesis_id="h1",
                statement="Higher temperature increases reaction rate",
                null_hypothesis="Temperature has no effect on reaction rate",
                alternative_hypothesis="Temperature significantly affects reaction rate",
                variables_involved=["temperature", "reaction_rate"]
            )
        ]
    
    def test_generate_protocol_quantitative(self):
        """Test protocol generation for quantitative methodology."""
        experiment_context = {
            'title': 'Temperature Effect Study',
            'objective': 'Investigate temperature effects on reaction rate',
            'domain': 'chemistry',
            'complexity': 'medium'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        assert isinstance(protocol, ExperimentalProtocol)
        assert protocol.protocol_id
        assert protocol.title == 'Protocol for Temperature Effect Study'
        assert protocol.objective == 'Investigate temperature effects on reaction rate'
        assert protocol.methodology == MethodologyType.QUANTITATIVE
        assert len(protocol.procedures) > 0
        assert len(protocol.materials_required) > 0
        assert len(protocol.safety_considerations) > 0
        assert len(protocol.quality_controls) > 0
        assert len(protocol.data_collection_methods) > 0
        assert protocol.analysis_plan
        assert protocol.estimated_duration > timedelta(0)
    
    def test_generate_protocol_qualitative(self):
        """Test protocol generation for qualitative methodology."""
        experiment_context = {
            'title': 'User Experience Study',
            'objective': 'Understand user interactions',
            'domain': 'psychology',
            'complexity': 'high'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUALITATIVE,
            experiment_type=ExperimentType.OBSERVATIONAL,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        assert isinstance(protocol, ExperimentalProtocol)
        assert protocol.methodology == MethodologyType.QUALITATIVE
        assert 'observation' in protocol.analysis_plan.lower() or 'thematic' in protocol.analysis_plan.lower()
        assert any('interview' in method.lower() for method in protocol.data_collection_methods)
    
    def test_generate_protocol_computational(self):
        """Test protocol generation for computational methodology."""
        experiment_context = {
            'title': 'Algorithm Performance Study',
            'objective': 'Compare algorithm efficiency',
            'domain': 'computer_science',
            'complexity': 'medium'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.COMPUTATIONAL,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        assert isinstance(protocol, ExperimentalProtocol)
        assert protocol.methodology == MethodologyType.COMPUTATIONAL
        assert any('simulation' in procedure.lower() or 'computational' in procedure.lower() 
                  for procedure in protocol.procedures)
        assert 'statistical analysis' in protocol.analysis_plan.lower()
    
    def test_optimize_protocol_efficiency(self):
        """Test protocol optimization for efficiency."""
        # First generate a protocol
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test optimization',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Optimize for efficiency
        optimization = self.generator.optimize_protocol(
            protocol=protocol,
            optimization_goals=['efficiency']
        )
        
        assert isinstance(optimization, ProtocolOptimization)
        assert optimization.original_protocol.protocol_id == protocol.protocol_id
        assert optimization.optimized_protocol.protocol_id != protocol.protocol_id
        assert len(optimization.optimization_strategies) > 0
        assert len(optimization.improvements) > 0
        assert 0 <= optimization.confidence_score <= 1
        
        # Check that duration was reduced
        assert (optimization.optimized_protocol.estimated_duration < 
                optimization.original_protocol.estimated_duration)
    
    def test_optimize_protocol_accuracy(self):
        """Test protocol optimization for accuracy."""
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test optimization',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        original_qc_count = len(protocol.quality_controls)
        
        # Optimize for accuracy
        optimization = self.generator.optimize_protocol(
            protocol=protocol,
            optimization_goals=['accuracy']
        )
        
        assert isinstance(optimization, ProtocolOptimization)
        assert any('validation' in strategy.lower() or 'quality' in strategy.lower() 
                  for strategy in optimization.optimization_strategies)
        
        # Check that quality controls were added
        assert (len(optimization.optimized_protocol.quality_controls) >= 
                original_qc_count)
    
    def test_optimize_protocol_multiple_goals(self):
        """Test protocol optimization with multiple goals."""
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test optimization',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Optimize for multiple goals
        optimization = self.generator.optimize_protocol(
            protocol=protocol,
            optimization_goals=['efficiency', 'accuracy', 'cost']
        )
        
        assert isinstance(optimization, ProtocolOptimization)
        assert len(optimization.optimization_strategies) > 0
        assert len(optimization.improvements) > 0
        assert len(optimization.trade_offs) > 0  # Multiple goals should have trade-offs
    
    def test_standardize_protocol(self):
        """Test protocol standardization."""
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test standardization',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Standardize protocol
        standardized = self.generator.standardize_protocol(protocol)
        
        assert isinstance(standardized, ExperimentalProtocol)
        assert standardized.protocol_id != protocol.protocol_id
        assert 'standardized' in standardized.protocol_id
        assert len(standardized.procedures) >= len(protocol.procedures)
        assert len(standardized.safety_considerations) >= len(protocol.safety_considerations)
        assert len(standardized.quality_controls) >= len(protocol.quality_controls)
    
    def test_standardize_protocol_specific_standards(self):
        """Test protocol standardization with specific standards."""
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test standardization',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Standardize with specific standards
        standardized = self.generator.standardize_protocol(
            protocol=protocol,
            standards=['procedure_clarity', 'safety_compliance']
        )
        
        assert isinstance(standardized, ExperimentalProtocol)
        # Check that standardization was applied (either timing info or other enhancements)
        assert (any('step' in procedure.lower() and 'time' in procedure.lower() 
                   for procedure in standardized.procedures) or
                len(standardized.procedures) >= len(protocol.procedures) or
                len(standardized.safety_considerations) > len(protocol.safety_considerations))
    
    def test_validate_protocol_valid(self):
        """Test protocol validation for valid protocol."""
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test validation',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Validate protocol
        validation = self.generator.validate_protocol(protocol)
        
        assert isinstance(validation, ProtocolValidation)
        assert validation.protocol_id == protocol.protocol_id
        assert len(validation.validation_checks) > 0
        assert len(validation.passed_checks) > 0
        assert 0 <= validation.overall_score <= 1
        assert isinstance(validation.is_valid, bool)
    
    def test_validate_protocol_incomplete(self):
        """Test protocol validation for incomplete protocol."""
        # Create incomplete protocol
        incomplete_protocol = ExperimentalProtocol(
            protocol_id="incomplete",
            title="",  # Missing title
            objective="",  # Missing objective
            methodology=MethodologyType.QUANTITATIVE,
            procedures=["Step 1"],  # Too few procedures
            materials_required=[],
            safety_considerations=[],
            quality_controls=[],  # Missing quality controls
            data_collection_methods=[],
            analysis_plan="",
            estimated_duration=timedelta(days=1)
        )
        
        validation = self.generator.validate_protocol(incomplete_protocol)
        
        assert isinstance(validation, ProtocolValidation)
        assert len(validation.failed_checks) > 0
        assert len(validation.recommendations) > 0
        assert validation.overall_score < 0.8
        assert not validation.is_valid
    
    def test_validate_protocol_specific_criteria(self):
        """Test protocol validation with specific criteria."""
        experiment_context = {
            'title': 'Test Study',
            'objective': 'Test validation',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Validate with specific criteria
        validation = self.generator.validate_protocol(
            protocol=protocol,
            validation_criteria=['completeness', 'clarity']
        )
        
        assert isinstance(validation, ProtocolValidation)
        assert len(validation.validation_checks) == 2
        assert 'completeness' in validation.validation_checks
        assert 'clarity' in validation.validation_checks
    
    def test_protocol_templates_initialization(self):
        """Test that protocol templates are properly initialized."""
        templates = self.generator.protocol_templates
        
        assert len(templates) > 0
        
        for template_id, template in templates.items():
            assert isinstance(template, ProtocolTemplate)
            assert template.template_id == template_id
            assert template.name
            assert template.methodology in MethodologyType
            assert len(template.experiment_types) > 0
            assert len(template.procedure_steps) > 0
            assert len(template.required_materials) > 0
            assert len(template.safety_requirements) > 0
            assert len(template.quality_controls) > 0
            assert len(template.data_collection_methods) > 0
            assert len(template.analysis_approaches) > 0
            assert template.typical_duration > timedelta(0)
            assert template.complexity_level in ['low', 'medium', 'high']
    
    def test_domain_specific_customization(self):
        """Test domain-specific protocol customization."""
        domains = ['chemistry', 'biology', 'engineering', 'psychology']
        
        for domain in domains:
            experiment_context = {
                'title': f'{domain.title()} Study',
                'objective': f'Test {domain} protocols',
                'domain': domain
            }
            
            protocol = self.generator.generate_protocol(
                experiment_context=experiment_context,
                methodology=MethodologyType.QUANTITATIVE,
                experiment_type=ExperimentType.CONTROLLED,
                variables=self.variables,
                conditions=self.conditions,
                hypotheses=self.hypotheses
            )
            
            assert isinstance(protocol, ExperimentalProtocol)
            
            # Check domain-specific elements
            if domain == 'chemistry':
                assert any('chemical' in material.lower() or 'reagent' in material.lower() 
                          for material in protocol.materials_required)
                assert any('chemical' in safety.lower() or 'waste' in safety.lower() 
                          for safety in protocol.safety_considerations)
            
            elif domain == 'biology':
                assert any('biosafety' in safety.lower() or 'contamination' in safety.lower() 
                          for safety in protocol.safety_considerations)
            
            elif domain == 'engineering':
                assert any('equipment' in safety.lower() or 'electrical' in safety.lower() 
                          for safety in protocol.safety_considerations)
    
    def test_variable_integration(self):
        """Test integration of variables into protocol."""
        # Test with different variable types
        variables = [
            ExperimentalVariable(
                name="pressure",
                variable_type="independent",
                data_type="continuous",
                measurement_unit="Pa"
            ),
            ExperimentalVariable(
                name="flow_rate",
                variable_type="dependent",
                data_type="continuous",
                measurement_unit="L/min"
            ),
            ExperimentalVariable(
                name="material_type",
                variable_type="control",
                data_type="categorical",
                measurement_unit=None
            )
        ]
        
        experiment_context = {
            'title': 'Variable Integration Test',
            'objective': 'Test variable integration',
            'domain': 'engineering'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=variables,
            conditions=self.conditions,
            hypotheses=self.hypotheses
        )
        
        # Check that variables are integrated
        assert any('pressure' in procedure.lower() for procedure in protocol.procedures)
        assert any('flow_rate' in method.lower() for method in protocol.data_collection_methods)
        assert any('pressure' in material.lower() for material in protocol.materials_required)
    
    def test_condition_integration(self):
        """Test integration of conditions into protocol."""
        # Test with multiple conditions
        conditions = [
            ExperimentalCondition(
                condition_id="low_temp",
                name="Low Temperature",
                variables={"temperature": 20},
                sample_size=25
            ),
            ExperimentalCondition(
                condition_id="med_temp",
                name="Medium Temperature",
                variables={"temperature": 40},
                sample_size=25
            ),
            ExperimentalCondition(
                condition_id="high_temp",
                name="High Temperature",
                variables={"temperature": 60},
                sample_size=25
            )
        ]
        
        experiment_context = {
            'title': 'Condition Integration Test',
            'objective': 'Test condition integration',
            'domain': 'chemistry'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.FACTORIAL,
            variables=self.variables,
            conditions=conditions,
            hypotheses=self.hypotheses
        )
        
        # Check that conditions are integrated
        assert any('3 experimental conditions' in procedure or 'condition' in procedure.lower() 
                  for procedure in protocol.procedures)
        assert any('condition' in material.lower() for material in protocol.materials_required)
        assert any('randomization' in control.lower() or 'condition' in control.lower() 
                  for control in protocol.quality_controls)
    
    def test_hypothesis_integration(self):
        """Test integration of hypotheses into protocol."""
        # Test with multiple hypotheses
        hypotheses = [
            Hypothesis(
                hypothesis_id="h1",
                statement="Temperature positively affects reaction rate",
                null_hypothesis="Temperature has no effect",
                alternative_hypothesis="Temperature has significant effect",
                variables_involved=["temperature", "reaction_rate"]
            ),
            Hypothesis(
                hypothesis_id="h2",
                statement="Pressure moderates temperature effect",
                null_hypothesis="Pressure has no moderating effect",
                alternative_hypothesis="Pressure significantly moderates effect",
                variables_involved=["temperature", "pressure", "reaction_rate"]
            )
        ]
        
        experiment_context = {
            'title': 'Hypothesis Integration Test',
            'objective': 'Test hypothesis integration',
            'domain': 'chemistry'
        }
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.FACTORIAL,
            variables=self.variables,
            conditions=self.conditions,
            hypotheses=hypotheses
        )
        
        # Check that hypotheses are integrated into analysis plan
        assert 'temperature positively affects' in protocol.analysis_plan.lower() or \
               'test hypothesis' in protocol.analysis_plan.lower()
        assert 'pressure moderates' in protocol.analysis_plan.lower() or \
               len([h for h in hypotheses if h.statement.lower() in protocol.analysis_plan.lower()]) > 0


class TestProtocolOptimization:
    """Test cases for protocol optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ProtocolGenerator()
    
    def test_optimization_trade_offs(self):
        """Test identification of optimization trade-offs."""
        # Create a protocol
        experiment_context = {
            'title': 'Trade-off Test',
            'objective': 'Test trade-offs',
            'domain': 'engineering'
        }
        
        variables = [
            ExperimentalVariable(
                name="input",
                variable_type="independent",
                data_type="continuous"
            )
        ]
        
        conditions = [
            ExperimentalCondition(
                condition_id="test",
                name="Test Condition",
                variables={"input": 1},
                sample_size=10
            )
        ]
        
        hypotheses = [
            Hypothesis(
                hypothesis_id="h1",
                statement="Test hypothesis",
                null_hypothesis="No effect",
                alternative_hypothesis="Significant effect",
                variables_involved=["input"]
            )
        ]
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=variables,
            conditions=conditions,
            hypotheses=hypotheses
        )
        
        # Test conflicting optimization goals
        optimization = self.generator.optimize_protocol(
            protocol=protocol,
            optimization_goals=['efficiency', 'accuracy']
        )
        
        assert len(optimization.trade_offs) > 0
        assert any('accuracy' in trade_off.lower() and 'duration' in trade_off.lower() 
                  for trade_off in optimization.trade_offs)
    
    def test_optimization_confidence_calculation(self):
        """Test optimization confidence calculation."""
        experiment_context = {
            'title': 'Confidence Test',
            'objective': 'Test confidence',
            'domain': 'engineering'
        }
        
        variables = [
            ExperimentalVariable(
                name="test_var",
                variable_type="independent",
                data_type="continuous"
            )
        ]
        
        conditions = [
            ExperimentalCondition(
                condition_id="test",
                name="Test",
                variables={"test_var": 1},
                sample_size=10
            )
        ]
        
        hypotheses = [
            Hypothesis(
                hypothesis_id="h1",
                statement="Test",
                null_hypothesis="No effect",
                alternative_hypothesis="Effect",
                variables_involved=["test_var"]
            )
        ]
        
        protocol = self.generator.generate_protocol(
            experiment_context=experiment_context,
            methodology=MethodologyType.QUANTITATIVE,
            experiment_type=ExperimentType.CONTROLLED,
            variables=variables,
            conditions=conditions,
            hypotheses=hypotheses
        )
        
        # Test single goal optimization (should have higher confidence)
        single_goal_opt = self.generator.optimize_protocol(
            protocol=protocol,
            optimization_goals=['efficiency']
        )
        
        # Test multiple goal optimization (should have lower confidence)
        multi_goal_opt = self.generator.optimize_protocol(
            protocol=protocol,
            optimization_goals=['efficiency', 'accuracy', 'cost', 'reliability']
        )
        
        # Single goal should have higher confidence than multiple conflicting goals
        assert single_goal_opt.confidence_score >= multi_goal_opt.confidence_score


if __name__ == "__main__":
    pytest.main([__file__])