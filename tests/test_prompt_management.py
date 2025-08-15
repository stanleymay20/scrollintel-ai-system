"""
Unit tests for the Advanced Prompt Management System.
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from scrollintel.core.prompt_manager import PromptManager, SearchQuery, PromptChanges
from scrollintel.core.prompt_import_export import PromptImportExport
from scrollintel.models.prompt_models import AdvancedPromptTemplate, AdvancedPromptVersion, AdvancedPromptCategory, AdvancedPromptTag, PromptVariable


class TestPromptVariable:
    """Test PromptVariable class."""
    
    def test_create_variable(self):
        """Test creating a prompt variable."""
        var = PromptVariable("user_name", "string", "John", "User's name", True)
        assert var.name == "user_name"
        assert var.type == "string"
        assert var.default == "John"
        assert var.description == "User's name"
        assert var.required == True
    
    def test_to_dict(self):
        """Test converting variable to dictionary."""
        var = PromptVariable("age", "number", "25", "User's age", False)
        result = var.to_dict()
        expected = {
            "name": "age",
            "type": "number",
            "default": "25",
            "description": "User's age",
            "required": False
        }
        assert result == expected
    
    def test_from_dict(self):
        """Test creating variable from dictionary."""
        data = {
            "name": "email",
            "type": "string",
            "description": "User's email",
            "required": True
        }
        var = PromptVariable.from_dict(data)
        assert var.name == "email"
        assert var.type == "string"
        assert var.description == "User's email"
        assert var.required == True
        assert var.default is None


class TestPromptManager:
    """Test PromptManager class."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def prompt_manager(self, mock_db_session):
        """Create PromptManager instance with mocked database."""
        return PromptManager(mock_db_session)
    
    def test_create_prompt(self, prompt_manager, mock_db_session):
        """Test creating a new prompt template."""
        # Mock the database operations
        mock_prompt = Mock()
        mock_prompt.id = "test-prompt-id"
        mock_db_session.add.return_value = None
        mock_db_session.flush.return_value = None
        mock_db_session.commit.return_value = None
        
        # Create prompt
        variables = [{"name": "user_name", "type": "string", "required": True}]
        prompt_id = prompt_manager.create_prompt(
            name="Test Prompt",
            content="Hello {{user_name}}!",
            category="greeting",
            created_by="test_user",
            tags=["greeting", "personalized"],
            variables=variables,
            description="A test greeting prompt"
        )
        
        # Verify database operations
        assert mock_db_session.add.call_count == 2  # Prompt + Version
        mock_db_session.flush.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_update_prompt(self, prompt_manager, mock_db_session):
        """Test updating an existing prompt template."""
        # Mock existing prompt
        existing_prompt = Mock()
        existing_prompt.id = "test-prompt-id"
        existing_prompt.name = "Old Name"
        existing_prompt.content = "Old content"
        existing_prompt.category = "old_category"
        existing_prompt.tags = ["old_tag"]
        existing_prompt.variables = []
        
        # Mock latest version
        latest_version = Mock()
        latest_version.version = "1.0.0"
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_prompt
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = latest_version
        
        # Update prompt
        changes = PromptChanges(
            name="New Name",
            content="New content {{user_name}}",
            category="new_category",
            tags=["new_tag"],
            changes_description="Updated for testing"
        )
        
        result = prompt_manager.update_prompt("test-prompt-id", changes, "test_user")
        
        # Verify updates
        assert existing_prompt.name == "New Name"
        assert existing_prompt.content == "New content {{user_name}}"
        assert existing_prompt.category == "new_category"
        assert existing_prompt.tags == ["new_tag"]
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()
    
    def test_search_prompts_text(self, prompt_manager, mock_db_session):
        """Test searching prompts by text."""
        # Mock query chain
        mock_query = Mock()
        mock_db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        # Search prompts
        query = SearchQuery(text="greeting", limit=10)
        results = prompt_manager.search_prompts(query)
        
        # Verify query was built correctly
        mock_db_session.query.assert_called_with(AdvancedPromptTemplate)
        assert mock_query.filter.call_count >= 2  # Active filter + text filter
        mock_query.order_by.assert_called()
        mock_query.offset.assert_called_with(0)
        mock_query.limit.assert_called_with(10)
        mock_query.all.assert_called()
    
    def test_substitute_variables(self, prompt_manager):
        """Test variable substitution in prompt content."""
        content = "Hello {{user_name}}, your age is {{age}} years old."
        variables = {"user_name": "John", "age": 25}
        
        result = prompt_manager.substitute_variables(content, variables)
        expected = "Hello John, your age is 25 years old."
        assert result == expected
    
    def test_validate_prompt_variables(self, prompt_manager):
        """Test prompt variable validation."""
        content = "Hello {{user_name}}, welcome to {{platform}}!"
        variables = [
            {"name": "user_name", "required": True},
            {"name": "platform", "required": True},
            {"name": "unused_var", "required": True}
        ]
        
        errors = prompt_manager.validate_prompt_variables(content, variables)
        assert len(errors) == 1
        assert "unused_var" in errors[0]
    
    def test_increment_version(self, prompt_manager):
        """Test version number incrementing."""
        assert prompt_manager._increment_version("1.0.0") == "1.0.1"
        assert prompt_manager._increment_version("1.2.5") == "1.2.6"
        assert prompt_manager._increment_version("invalid") == "1.0.0"
    
    def test_get_prompt_history(self, prompt_manager, mock_db_session):
        """Test getting prompt version history."""
        # Mock query chain
        mock_query = Mock()
        mock_db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        
        # Get history
        history = prompt_manager.get_prompt_history("test-prompt-id")
        
        # Verify query
        mock_db_session.query.assert_called_with(AdvancedPromptVersion)
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called()
        mock_query.all.assert_called()
    
    def test_delete_prompt(self, prompt_manager, mock_db_session):
        """Test soft deleting a prompt."""
        # Mock existing prompt
        existing_prompt = Mock()
        existing_prompt.is_active = True
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_prompt
        
        # Delete prompt
        result = prompt_manager.delete_prompt("test-prompt-id")
        
        # Verify soft delete
        assert result == True
        assert existing_prompt.is_active == False
        mock_db_session.commit.assert_called()
    
    def test_create_category(self, prompt_manager, mock_db_session):
        """Test creating a prompt category."""
        mock_category = Mock()
        mock_category.id = "category-id"
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        
        # Create category
        category_id = prompt_manager.create_category(
            name="Test Category",
            description="A test category",
            parent_id="parent-id"
        )
        
        # Verify database operations
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()
    
    def test_create_tag(self, prompt_manager, mock_db_session):
        """Test creating a prompt tag."""
        mock_tag = Mock()
        mock_tag.id = "tag-id"
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        
        # Create tag
        tag_id = prompt_manager.create_tag(
            name="test-tag",
            description="A test tag",
            color="#FF0000"
        )
        
        # Verify database operations
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()


class TestPromptImportExport:
    """Test PromptImportExport class."""
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Mock PromptManager."""
        return Mock(spec=PromptManager)
    
    @pytest.fixture
    def import_export(self, mock_prompt_manager):
        """Create PromptImportExport instance."""
        return PromptImportExport(mock_prompt_manager)
    
    def test_export_json(self, import_export, mock_prompt_manager):
        """Test exporting prompts as JSON."""
        # Mock prompt data
        mock_prompt = Mock()
        mock_prompt.id = "test-id"
        mock_prompt.name = "Test Prompt"
        mock_prompt.content = "Hello {{user}}!"
        mock_prompt.category = "greeting"
        mock_prompt.tags = ["test"]
        mock_prompt.variables = []
        mock_prompt.description = "Test description"
        mock_prompt.created_by = "test_user"
        mock_prompt.created_at = datetime.now()
        mock_prompt.updated_at = datetime.now()
        
        mock_prompt_manager.get_prompt.return_value = mock_prompt
        mock_prompt_manager.get_prompt_history.return_value = []
        
        # Export prompts
        result = import_export.export_prompts(["test-id"], "json")
        
        # Verify JSON structure
        assert isinstance(result, str)
        data = json.loads(result)
        assert "version" in data
        assert "exported_at" in data
        assert "prompts" in data
        assert len(data["prompts"]) == 1
        assert data["prompts"][0]["name"] == "Test Prompt"
    
    def test_import_json(self, import_export, mock_prompt_manager):
        """Test importing prompts from JSON."""
        # Prepare test data
        test_data = {
            "version": "1.0",
            "prompts": [
                {
                    "name": "Imported Prompt",
                    "content": "Hello {{user}}!",
                    "category": "greeting",
                    "tags": ["imported"],
                    "variables": [],
                    "description": "An imported prompt"
                }
            ]
        }
        
        mock_prompt_manager.get_prompt.return_value = None  # No existing prompt
        mock_prompt_manager.create_prompt.return_value = "new-prompt-id"
        
        # Import prompts
        result = import_export.import_prompts(json.dumps(test_data), "json", "test_user")
        
        # Verify import results
        assert result["imported"] == 1
        assert result["updated"] == 0
        assert result["skipped"] == 0
        assert len(result["errors"]) == 0
        mock_prompt_manager.create_prompt.assert_called_once()
    
    def test_export_csv(self, import_export, mock_prompt_manager):
        """Test exporting prompts as CSV."""
        # Mock prompt data
        mock_prompt = Mock()
        mock_prompt.id = "test-id"
        mock_prompt.name = "Test Prompt"
        mock_prompt.content = "Hello!"
        mock_prompt.category = "greeting"
        mock_prompt.tags = ["test"]
        mock_prompt.variables = []
        mock_prompt.description = "Test"
        mock_prompt.created_by = "test_user"
        mock_prompt.created_at = datetime.now()
        mock_prompt.updated_at = datetime.now()
        
        mock_prompt_manager.get_prompt.return_value = mock_prompt
        mock_prompt_manager.get_prompt_history.return_value = []
        
        # Export as CSV
        result = import_export.export_prompts(["test-id"], "csv")
        
        # Verify CSV structure
        assert isinstance(result, str)
        lines = result.strip().split('\n')
        assert len(lines) >= 2  # Header + data
        assert "id,name,content" in lines[0]
    
    def test_format_prompt_as_markdown(self, import_export):
        """Test formatting prompt as Markdown."""
        prompt_data = {
            "name": "Test Prompt",
            "description": "A test prompt",
            "category": "test",
            "tags": ["test", "example"],
            "variables": [
                {
                    "name": "user_name",
                    "type": "string",
                    "required": True,
                    "description": "User's name"
                }
            ],
            "content": "Hello {{user_name}}!",
            "created_by": "test_user",
            "created_at": "2023-01-01T00:00:00"
        }
        
        result = import_export._format_prompt_as_markdown(prompt_data)
        
        # Verify Markdown structure
        assert "# Test Prompt" in result
        assert "**Description:** A test prompt" in result
        assert "**Category:** test" in result
        assert "**Tags:** `test`, `example`" in result
        assert "## Variables" in result
        assert "- **user_name**" in result
        assert "## Content" in result
        assert "Hello {{user_name}}!" in result
        assert "*Created by: test_user*" in result
    
    def test_import_with_overwrite(self, import_export, mock_prompt_manager):
        """Test importing with overwrite option."""
        # Mock existing prompt
        existing_prompt = Mock()
        existing_prompt.id = "existing-id"
        mock_prompt_manager.get_prompt.return_value = existing_prompt
        mock_prompt_manager.update_prompt.return_value = Mock()
        
        test_data = {
            "prompts": [
                {
                    "id": "existing-id",
                    "name": "Updated Prompt",
                    "content": "Updated content",
                    "category": "updated"
                }
            ]
        }
        
        # Import with overwrite
        result = import_export.import_prompts(json.dumps(test_data), "json", "test_user", overwrite=True)
        
        # Verify update was called
        assert result["updated"] == 1
        assert result["imported"] == 0
        mock_prompt_manager.update_prompt.assert_called_once()
    
    def test_import_without_overwrite_skips_existing(self, import_export, mock_prompt_manager):
        """Test importing without overwrite skips existing prompts."""
        # Mock existing prompt
        existing_prompt = Mock()
        existing_prompt.id = "existing-id"
        mock_prompt_manager.get_prompt.return_value = existing_prompt
        
        test_data = {
            "prompts": [
                {
                    "id": "existing-id",
                    "name": "Existing Prompt",
                    "content": "Existing content",
                    "category": "existing"
                }
            ]
        }
        
        # Import without overwrite
        result = import_export.import_prompts(json.dumps(test_data), "json", "test_user", overwrite=False)
        
        # Verify prompt was skipped
        assert result["skipped"] == 1
        assert result["imported"] == 0
        assert result["updated"] == 0
        mock_prompt_manager.create_prompt.assert_not_called()
        mock_prompt_manager.update_prompt.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])