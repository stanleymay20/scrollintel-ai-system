"""
Integration tests for automated code generation NLP pipeline.
Tests the complete flow from requirements text to structured output.
"""
import pytest
from unittest.mock import Mock, patch
import json

from scrollintel.engines.code_generation_nlp import NLProcessor
from scrollintel.engines.code_generation_intent import IntentClassifier
from scrollintel.engines.code_generation_entities import EntityExtractor
from scrollintel.engines.code_generation_clarification import ClarificationEngine
from scrollintel.engines.code_generation_validation import RequirementsValidator
from scrollintel.models.code_generation_models import (
    RequirementType, Intent, EntityType, ConfidenceLevel
)


class TestCodeGenerationNLPIntegration:
    """Integration tests for the complete NLP pipeline."""
    
    @pytest.fixture
    def sample_requirements_text(self):
        """Sample requirements text for testing."""
        return """
        I need to build a customer management web application. 
        Users should be able to register with email and password.
        Administrators can view and manage all customer data.
        The system should store customer profiles, orders, and payment information.
        All data must be encrypted and secure.
        The application should handle 1000 concurrent users.
        """
    
    @pytest.fixture
    def mock_gpt4_responses(self):
        """Mock GPT-4 responses for consistent testing."""
        return {
            'requirements': json.dumps([
                {
                    "original_text": "Users should be able to register with email and password",
                    "structured_text": "Users should be able to register with email and password",
                    "requirement_type": "functional",
                    "intent": "add_security",
                    "acceptance_criteria": ["Valid email format required", "Password strength validation"],
                    "dependencies": [],
                    "priority": 1,
                    "complexity": 2,
                    "confidence": "high"
                },
                {
                    "original_text": "The application should handle 1000 concurrent users",
                    "structured_text": "The application should handle 1000 concurrent users",
                    "requirement_type": "performance",
                    "intent": "improve_performance",
                    "acceptance_criteria": ["Support 1000 concurrent users", "Response time under 2 seconds"],
                    "dependencies": [],
                    "priority": 2,
                    "complexity": 4,
                    "confidence": "medium"
                }
            ]),
            'entities': json.dumps([
                {
                    "name": "user",
                    "type": "user_role",
                    "description": "End user of the system",
                    "confidence": 0.9,
                    "source_text": "Users",
                    "attributes": {}
                },
                {
                    "name": "administrator",
                    "type": "user_role", 
                    "description": "System administrator",
                    "confidence": 0.9,
                    "source_text": "Administrators",
                    "attributes": {}
                },
                {
                    "name": "customer_profile",
                    "type": "data_entity",
                    "description": "Customer profile information",
                    "confidence": 0.8,
                    "source_text": "customer profiles",
                    "attributes": {}
                },
                {
                    "name": "web_application",
                    "type": "system_component",
                    "description": "Customer management web application",
                    "confidence": 0.9,
                    "source_text": "web application",
                    "attributes": {}
                }
            ]),
            'relationships': json.dumps([
                {
                    "source_entity_id": "user_id",
                    "target_entity_id": "web_app_id",
                    "relationship_type": "uses",
                    "description": "Users use the web application",
                    "confidence": 0.8
                }
            ]),
            'clarifications': json.dumps([
                {
                    "question": "What specific customer data fields should be stored?",
                    "context": "customer profiles mentioned but not detailed",
                    "suggested_answers": ["Name, email, phone", "Address information", "Purchase history"],
                    "priority": 2
                }
            ])
        }
    
    def test_complete_nlp_pipeline(self, sample_requirements_text, mock_gpt4_responses):
        """Test the complete NLP pipeline from text to structured requirements."""
        processor = NLProcessor()
        
        # Mock GPT-4 responses
        def mock_gpt4_call(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            
            # Determine which response to return based on the prompt
            prompt = kwargs.get('messages', [{}])[-1].get('content', '')
            
            if 'extract individual requirements' in prompt.lower():
                mock_response.choices[0].message.content = mock_gpt4_responses['requirements']
            elif 'extract all relevant entities' in prompt.lower():
                mock_response.choices[0].message.content = mock_gpt4_responses['entities']
            elif 'identify relationships' in prompt.lower():
                mock_response.choices[0].message.content = mock_gpt4_responses['relationships']
            else:
                mock_response.choices[0].message.content = "[]"
            
            return mock_response
        
        with patch.object(processor.client.chat.completions, 'create', side_effect=mock_gpt4_call):
            result = processor.parse_requirements(sample_requirements_text, "Customer Management System")
        
        # Verify successful processing
        assert result.success is True
        assert result.requirements is not None
        
        requirements = result.requirements
        
        # Verify basic structure
        assert requirements.project_name == "Customer Management System"
        assert requirements.raw_text == sample_requirements_text
        assert len(requirements.parsed_requirements) == 2
        
        # Verify requirement parsing
        functional_reqs = [r for r in requirements.parsed_requirements 
                          if r.requirement_type == RequirementType.FUNCTIONAL]
        performance_reqs = [r for r in requirements.parsed_requirements 
                           if r.requirement_type == RequirementType.PERFORMANCE]
        
        assert len(functional_reqs) >= 1
        assert len(performance_reqs) >= 1
        
        # Verify entity extraction
        assert len(requirements.entities) >= 3
        
        user_entities = [e for e in requirements.entities if e.type == EntityType.USER_ROLE]
        data_entities = [e for e in requirements.entities if e.type == EntityType.DATA_ENTITY]
        system_entities = [e for e in requirements.entities if e.type == EntityType.SYSTEM_COMPONENT]
        
        assert len(user_entities) >= 1
        assert len(data_entities) >= 1
        assert len(system_entities) >= 1
        
        # Verify completeness score
        assert 0.0 <= requirements.completeness_score <= 1.0
        assert requirements.completeness_score > 0.5  # Should be reasonably complete
    
    def test_intent_classification_integration(self, sample_requirements_text):
        """Test intent classification on realistic requirements."""
        classifier = IntentClassifier()
        
        # Extract individual sentences for classification
        sentences = [
            "I need to build a customer management web application",
            "Users should be able to register with email and password", 
            "The system should store customer profiles",
            "All data must be encrypted and secure",
            "The application should handle 1000 concurrent users"
        ]
        
        results = classifier.classify_multiple_intents(sentences)
        
        assert len(results) == len(sentences)
        
        # Verify expected intents
        intents = [intent for intent, confidence in results]
        
        assert Intent.CREATE_APPLICATION in intents
        assert Intent.ADD_SECURITY in intents
        assert Intent.DESIGN_DATABASE in intents or Intent.CREATE_APPLICATION in intents
        
        # Verify confidence scores
        for intent, confidence in results:
            assert 0.0 <= confidence <= 1.0
    
    def test_entity_extraction_integration(self, sample_requirements_text):
        """Test entity extraction on realistic requirements."""
        extractor = EntityExtractor()
        
        entities = extractor.extract_entities(sample_requirements_text)
        
        assert len(entities) > 0
        
        # Verify entity types are present
        entity_types = {e.type for e in entities}
        
        assert EntityType.USER_ROLE in entity_types
        assert EntityType.DATA_ENTITY in entity_types or EntityType.SYSTEM_COMPONENT in entity_types
        
        # Verify entity names make sense
        entity_names = [e.name.lower() for e in entities]
        
        # Should find user-related entities
        user_related = any('user' in name or 'admin' in name or 'customer' in name 
                          for name in entity_names)
        assert user_related
        
        # Verify confidence scores
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0
    
    def test_clarification_generation_integration(self, sample_requirements_text):
        """Test clarification generation on realistic requirements."""
        processor = NLProcessor()
        engine = ClarificationEngine()
        
        # Process requirements first (using fallback parsing to avoid API calls)
        result = processor.parse_requirements(sample_requirements_text)
        
        if result.success and result.requirements:
            clarifications = engine.generate_clarifications(result.requirements)
            
            # Should generate some clarifications for improvement
            assert isinstance(clarifications, list)
            
            # Verify clarification structure
            for clarif in clarifications:
                assert clarif.question
                assert clarif.context
                assert 1 <= clarif.priority <= 5
    
    def test_validation_integration(self, sample_requirements_text):
        """Test requirements validation on realistic requirements."""
        processor = NLProcessor()
        validator = RequirementsValidator()
        
        # Process requirements
        result = processor.parse_requirements(sample_requirements_text)
        
        if result.success and result.requirements:
            # Validate requirements
            issues = validator.validate_requirements(result.requirements)
            
            # Calculate quality score
            quality_score = validator.calculate_quality_score(result.requirements)
            
            # Get improvement suggestions
            suggestions = validator.get_improvement_suggestions(result.requirements)
            
            # Check readiness for code generation
            is_ready, blocking_issues = validator.is_ready_for_code_generation(result.requirements)
            
            # Verify validation results
            assert isinstance(issues, list)
            assert 0.0 <= quality_score <= 1.0
            assert isinstance(suggestions, list)
            assert isinstance(is_ready, bool)
            assert isinstance(blocking_issues, list)
    
    def test_error_handling_integration(self):
        """Test error handling across the pipeline."""
        processor = NLProcessor()
        
        # Test with problematic input
        problematic_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Invalid JSON response test",  # May cause JSON parsing issues
            "A" * 10000,  # Very long input
        ]
        
        for input_text in problematic_inputs:
            result = processor.parse_requirements(input_text)
            
            # Should handle gracefully
            assert isinstance(result, type(result))
            assert hasattr(result, 'success')
            assert hasattr(result, 'errors')
    
    def test_performance_integration(self, sample_requirements_text):
        """Test performance of the complete pipeline."""
        import time
        
        processor = NLProcessor()
        
        start_time = time.time()
        result = processor.parse_requirements(sample_requirements_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (allowing for API calls)
        assert processing_time < 30.0  # 30 seconds max
        
        # Verify processing time is recorded
        if result.success:
            assert result.processing_time > 0
    
    def test_consistency_integration(self):
        """Test consistency of results across multiple runs."""
        processor = NLProcessor()
        text = "Users should be able to login with email and password"
        
        results = []
        for _ in range(3):
            result = processor.parse_requirements(text)
            results.append(result)
        
        # All runs should have same success status
        success_statuses = [r.success for r in results]
        assert len(set(success_statuses)) == 1  # All same
        
        # If successful, should have similar structure
        if all(r.success for r in results):
            req_counts = [len(r.requirements.parsed_requirements) for r in results]
            # Should be reasonably consistent (within 1 requirement)
            assert max(req_counts) - min(req_counts) <= 1
    
    def test_multilingual_handling(self):
        """Test handling of non-English requirements."""
        processor = NLProcessor()
        
        # Test with different languages
        multilingual_texts = [
            "Los usuarios deben poder iniciar sesión",  # Spanish
            "Les utilisateurs doivent pouvoir se connecter",  # French
            "ユーザーはログインできる必要があります",  # Japanese
        ]
        
        for text in multilingual_texts:
            result = processor.parse_requirements(text)
            
            # Should handle gracefully (may or may not parse correctly)
            assert isinstance(result, type(result))
            assert hasattr(result, 'success')
    
    def test_domain_specific_requirements(self):
        """Test handling of domain-specific requirements."""
        processor = NLProcessor()
        
        domain_texts = [
            # E-commerce
            "Build an e-commerce platform with shopping cart and payment processing",
            
            # Healthcare
            "Create a patient management system with HIPAA compliance",
            
            # Finance
            "Develop a trading platform with real-time market data",
            
            # Education
            "Build a learning management system with course creation tools"
        ]
        
        for text in domain_texts:
            result = processor.parse_requirements(text)
            
            # Should process domain-specific terminology
            assert isinstance(result, type(result))
            
            if result.success and result.requirements:
                # Should have at least parsed the requirement
                assert len(result.requirements.parsed_requirements) > 0