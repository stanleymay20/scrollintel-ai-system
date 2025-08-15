"""
Demo script for automated code generation NLP foundation.
Demonstrates the natural language processing capabilities for requirements analysis.
"""
import asyncio
from scrollintel.engines.code_generation_nlp import NLProcessor
from scrollintel.engines.code_generation_intent import IntentClassifier
from scrollintel.engines.code_generation_entities import EntityExtractor
from scrollintel.engines.code_generation_clarification import ClarificationEngine
from scrollintel.engines.code_generation_validation import RequirementsValidator


def print_separator(title: str):
    """Print a formatted separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def demo_nlp_foundation():
    """Demonstrate the NLP foundation capabilities."""
    print("🚀 Automated Code Generation - NLP Foundation Demo")
    print("This demo showcases the natural language processing capabilities")
    print("for analyzing and structuring software requirements.")
    
    # Sample requirements text
    requirements_text = """
    I need to build a customer management web application for my business.
    Users should be able to register with email and password authentication.
    Administrators can view and manage all customer data and generate reports.
    The system should store customer profiles, orders, and payment information securely.
    All sensitive data must be encrypted and the system should be GDPR compliant.
    The application should handle at least 1000 concurrent users with response times under 2 seconds.
    Integration with Stripe payment gateway and SendGrid email service is required.
    """
    
    print(f"\n📝 Sample Requirements:")
    print(requirements_text.strip())
    
    # Initialize components
    processor = NLProcessor()
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    clarification_engine = ClarificationEngine()
    validator = RequirementsValidator()
    
    print_separator("1. Requirements Processing")
    
    # Process requirements
    print("🔄 Processing requirements with NLP engine...")
    result = processor.parse_requirements(requirements_text, "Customer Management System")
    
    if not result.success:
        print("❌ Processing failed:")
        for error in result.errors:
            print(f"  - {error}")
        return
    
    requirements = result.requirements
    print(f"✅ Successfully processed requirements in {result.processing_time:.2f}s")
    print(f"📊 Completeness Score: {requirements.completeness_score:.1%}")
    print(f"📋 Found {len(requirements.parsed_requirements)} requirements")
    print(f"🏷️  Found {len(requirements.entities)} entities")
    print(f"🔗 Found {len(requirements.relationships)} relationships")
    
    print_separator("2. Parsed Requirements")
    
    for i, req in enumerate(requirements.parsed_requirements, 1):
        print(f"\n📌 Requirement {i}:")
        print(f"   Type: {req.requirement_type.value}")
        print(f"   Intent: {req.intent.value}")
        print(f"   Priority: {req.priority}/5")
        print(f"   Complexity: {req.complexity}/5")
        print(f"   Confidence: {req.confidence.value}")
        print(f"   Text: {req.structured_text}")
        
        if req.acceptance_criteria:
            print(f"   Acceptance Criteria:")
            for criterion in req.acceptance_criteria:
                print(f"     - {criterion}")
    
    print_separator("3. Intent Classification")
    
    # Demonstrate intent classification
    individual_requirements = [req.structured_text for req in requirements.parsed_requirements]
    intent_results = intent_classifier.classify_multiple_intents(individual_requirements)
    
    print("🎯 Intent Analysis:")
    for req_text, (intent, confidence) in zip(individual_requirements, intent_results):
        print(f"   Intent: {intent.value} (confidence: {confidence:.2f})")
        print(f"   Requirement: {req_text[:80]}...")
        print()
    
    # Get intent distribution
    distribution = intent_classifier.get_intent_distribution(individual_requirements)
    print("📊 Intent Distribution:")
    for intent, count in distribution.items():
        if count > 0:
            print(f"   {intent.value}: {count}")
    
    # Suggest architecture patterns
    intents = [intent for intent, _ in intent_results]
    patterns = intent_classifier.suggest_architecture_patterns(intents)
    print(f"\n🏗️  Suggested Architecture Patterns:")
    for pattern in patterns:
        print(f"   - {pattern}")
    
    print_separator("4. Entity Extraction")
    
    print("🏷️  Extracted Entities:")
    
    # Group entities by domain
    grouped_entities = entity_extractor.group_entities_by_domain(requirements.entities)
    
    for domain, entities in grouped_entities.items():
        if entities:
            print(f"\n   {domain}:")
            for entity in entities:
                print(f"     - {entity.name} ({entity.type.value}) [confidence: {entity.confidence:.2f}]")
                if entity.description:
                    print(f"       Description: {entity.description}")
    
    # Validate entities
    entity_warnings = entity_extractor.validate_entities(requirements.entities)
    if entity_warnings:
        print(f"\n⚠️  Entity Validation Warnings:")
        for warning in entity_warnings:
            print(f"   - {warning}")
    
    print_separator("5. Clarification Questions")
    
    # Generate clarifications
    clarifications = clarification_engine.generate_clarifications(requirements)
    
    if clarifications:
        print("❓ Clarification Questions (to improve requirements):")
        for i, clarif in enumerate(clarifications[:5], 1):  # Show top 5
            print(f"\n   {i}. {clarif.question}")
            print(f"      Priority: {clarif.priority}/5")
            print(f"      Context: {clarif.context[:100]}...")
            if clarif.suggested_answers:
                print(f"      Suggested answers:")
                for answer in clarif.suggested_answers:
                    print(f"        - {answer}")
    else:
        print("✅ No clarifications needed - requirements are clear!")
    
    print_separator("6. Requirements Validation")
    
    # Validate requirements
    issues = validator.validate_requirements(requirements)
    quality_score = validator.calculate_quality_score(requirements)
    
    print(f"📊 Quality Score: {quality_score:.1%}")
    
    if issues:
        print(f"\n🔍 Validation Issues ({len(issues)} found):")
        
        critical_issues = [i for i in issues if i.severity.value == "critical"]
        warning_issues = [i for i in issues if i.severity.value == "warning"]
        info_issues = [i for i in issues if i.severity.value == "info"]
        
        if critical_issues:
            print(f"\n   🚨 Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"     - {issue.message}")
        
        if warning_issues:
            print(f"\n   ⚠️  Warnings ({len(warning_issues)}):")
            for issue in warning_issues[:3]:  # Show top 3
                print(f"     - {issue.message}")
        
        if info_issues:
            print(f"\n   ℹ️  Info ({len(info_issues)}):")
            for issue in info_issues[:3]:  # Show top 3
                print(f"     - {issue.message}")
    else:
        print("✅ No validation issues found!")
    
    # Check readiness for code generation
    is_ready, blocking_issues = validator.is_ready_for_code_generation(requirements)
    
    print(f"\n🚀 Ready for Code Generation: {'✅ Yes' if is_ready else '❌ No'}")
    
    if blocking_issues:
        print("   Blocking Issues:")
        for issue in blocking_issues:
            print(f"     - {issue}")
    
    # Get improvement suggestions
    suggestions = validator.get_improvement_suggestions(requirements)
    if suggestions:
        print(f"\n💡 Improvement Suggestions:")
        for suggestion in suggestions[:5]:  # Show top 5
            print(f"   - {suggestion}")
    
    print_separator("7. Summary")
    
    print("📈 Processing Summary:")
    print(f"   ✅ Requirements processed: {len(requirements.parsed_requirements)}")
    print(f"   ✅ Entities extracted: {len(requirements.entities)}")
    print(f"   ✅ Relationships found: {len(requirements.relationships)}")
    print(f"   ✅ Clarifications generated: {len(clarifications)}")
    print(f"   ✅ Quality score: {quality_score:.1%}")
    print(f"   ✅ Ready for code generation: {'Yes' if is_ready else 'No'}")
    print(f"   ✅ Processing time: {result.processing_time:.2f}s")
    print(f"   ✅ Tokens used: {result.tokens_used}")
    
    print(f"\n🎉 NLP Foundation Demo Complete!")
    print("The system successfully analyzed the requirements and extracted structured information")
    print("that can be used for automated code generation.")


if __name__ == "__main__":
    demo_nlp_foundation()