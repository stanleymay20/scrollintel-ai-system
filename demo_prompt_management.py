#!/usr/bin/env python3
"""
Demo script for the Advanced Prompt Management System.
"""
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.prompt_manager import PromptManager, SearchQuery, PromptChanges
from scrollintel.core.prompt_import_export import PromptImportExport
from scrollintel.models.prompt_models import PromptVariable


def demo_prompt_management():
    """Demonstrate the prompt management functionality."""
    print("🚀 Advanced Prompt Management System Demo")
    print("=" * 50)
    
    try:
        # Create prompt manager
        print("\n📋 Creating PromptManager...")
        prompt_manager = PromptManager()
        
        # Create a sample prompt template
        print("\n✨ Creating sample prompt template...")
        variables = [
            {"name": "user_name", "type": "string", "required": True, "description": "User's name"},
            {"name": "topic", "type": "string", "required": True, "description": "Topic to discuss"},
            {"name": "tone", "type": "string", "required": False, "default": "friendly", "description": "Tone of response"}
        ]
        
        prompt_id = prompt_manager.create_prompt(
            name="Personalized Greeting",
            content="Hello {{user_name}}! I'd be happy to help you with {{topic}}. Let me provide a {{tone}} response.",
            category="greeting",
            created_by="demo_user",
            tags=["greeting", "personalized", "demo"],
            variables=variables,
            description="A personalized greeting template with customizable tone"
        )
        
        print(f"✅ Created prompt template with ID: {prompt_id}")
        
        # Test variable substitution
        print("\n🔄 Testing variable substitution...")
        content = "Hello {{user_name}}! I'd be happy to help you with {{topic}}. Let me provide a {{tone}} response."
        substituted = prompt_manager.substitute_variables(content, {
            "user_name": "Alice",
            "topic": "machine learning",
            "tone": "detailed"
        })
        print(f"Original: {content}")
        print(f"Substituted: {substituted}")
        
        # Test variable validation
        print("\n✅ Testing variable validation...")
        errors = prompt_manager.validate_prompt_variables(content, variables)
        if errors:
            print(f"Validation errors: {errors}")
        else:
            print("✅ All variables are valid!")
        
        # Update the prompt
        print("\n📝 Updating prompt template...")
        changes = PromptChanges(
            content="Hi {{user_name}}! I'm excited to help you learn about {{topic}}. Here's a {{tone}} explanation.",
            tags=["greeting", "personalized", "demo", "updated"],
            changes_description="Updated greeting to be more enthusiastic"
        )
        
        new_version = prompt_manager.update_prompt(prompt_id, changes, "demo_user")
        print(f"✅ Created new version: {new_version.version}")
        
        # Search for prompts
        print("\n🔍 Searching for prompts...")
        query = SearchQuery(text="greeting", tags=["demo"], limit=10)
        results = prompt_manager.search_prompts(query)
        print(f"Found {len(results)} prompts matching the search criteria")
        
        for prompt in results:
            print(f"  - {prompt.name} (Category: {prompt.category})")
        
        # Get prompt history
        print("\n📚 Getting prompt history...")
        history = prompt_manager.get_prompt_history(prompt_id)
        print(f"Found {len(history)} versions:")
        for version in history:
            print(f"  - Version {version.version}: {version.changes}")
        
        # Test import/export
        print("\n📤 Testing import/export functionality...")
        import_export = PromptImportExport(prompt_manager)
        
        # Export as JSON
        json_data = import_export.export_prompts([prompt_id], "json")
        print(f"✅ Exported prompt as JSON ({len(json_data)} characters)")
        
        # Export as CSV
        csv_data = import_export.export_prompts([prompt_id], "csv")
        print(f"✅ Exported prompt as CSV ({len(csv_data)} characters)")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ Prompt template creation with variables")
        print("✅ Variable substitution and validation")
        print("✅ Version control and history tracking")
        print("✅ Search and filtering capabilities")
        print("✅ Import/export functionality")
        print("✅ Categorization and tagging system")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = demo_prompt_management()
    sys.exit(0 if success else 1)