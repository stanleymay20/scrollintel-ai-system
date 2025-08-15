"""
Prompt Management System - Core functionality for managing prompt templates.
"""
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from ..models.prompt_models import AdvancedPromptTemplate, AdvancedPromptVersion, AdvancedPromptCategory, AdvancedPromptTag, PromptVariable
from ..models.database_utils import db_manager


class SearchQuery:
    """Search query parameters for prompt templates."""
    
    def __init__(self, text: Optional[str] = None, category: Optional[str] = None,
                 tags: Optional[List[str]] = None, created_by: Optional[str] = None,
                 date_from: Optional[datetime] = None, date_to: Optional[datetime] = None,
                 limit: int = 50, offset: int = 0):
        self.text = text
        self.category = category
        self.tags = tags or []
        self.created_by = created_by
        self.date_from = date_from
        self.date_to = date_to
        self.limit = limit
        self.offset = offset


class PromptChanges:
    """Represents changes to be made to a prompt template."""
    
    def __init__(self, name: Optional[str] = None, content: Optional[str] = None,
                 category: Optional[str] = None, tags: Optional[List[str]] = None,
                 variables: Optional[List[Dict[str, Any]]] = None,
                 description: Optional[str] = None, changes_description: Optional[str] = None):
        self.name = name
        self.content = content
        self.category = category
        self.tags = tags
        self.variables = variables
        self.description = description
        self.changes_description = changes_description


class PromptManager:
    """Core prompt management functionality."""
    
    def __init__(self, db_session: Optional[Session] = None):
        if db_session:
            self.db = db_session
            self._owns_session = False
        else:
            self.db = db_manager.SessionLocal()
            self._owns_session = True
    
    def __del__(self):
        """Clean up database session if we own it."""
        if hasattr(self, '_owns_session') and self._owns_session and hasattr(self, 'db'):
            self.db.close()
    
    def create_prompt(self, name: str, content: str, category: str, 
                     created_by: str, tags: Optional[List[str]] = None,
                     variables: Optional[List[Dict[str, Any]]] = None,
                     description: Optional[str] = None) -> str:
        """Create a new prompt template."""
        # Validate variables
        validated_variables = []
        if variables:
            for var_data in variables:
                var = PromptVariable.from_dict(var_data)
                validated_variables.append(var.to_dict())
        
        # Create prompt template
        prompt = AdvancedPromptTemplate(
            name=name,
            content=content,
            category=category,
            tags=tags or [],
            variables=validated_variables,
            description=description,
            created_by=created_by
        )
        
        self.db.add(prompt)
        self.db.flush()  # Get the ID
        
        # Create initial version
        initial_version = AdvancedPromptVersion(
            prompt_id=prompt.id,
            version="1.0.0",
            content=content,
            variables=validated_variables,
            tags=tags or [],
            changes="Initial version",
            created_by=created_by
        )
        
        self.db.add(initial_version)
        
        # Update tag usage counts
        self._update_tag_usage(tags or [])
        
        self.db.commit()
        return prompt.id
    
    def update_prompt(self, prompt_id: str, changes: PromptChanges, 
                     updated_by: str) -> AdvancedPromptVersion:
        """Update an existing prompt template and create a new version."""
        prompt = self.db.query(AdvancedPromptTemplate).filter(AdvancedPromptTemplate.id == prompt_id).first()
        if not prompt:
            raise ValueError(f"Prompt with ID {prompt_id} not found")
        
        # Get current version number and increment
        latest_version = self.db.query(AdvancedPromptVersion)\
            .filter(AdvancedPromptVersion.prompt_id == prompt_id)\
            .order_by(desc(AdvancedPromptVersion.created_at))\
            .first()
        
        new_version_num = self._increment_version(latest_version.version if latest_version else "1.0.0")
        
        # Update prompt template
        if changes.name is not None:
            prompt.name = changes.name
        if changes.content is not None:
            prompt.content = changes.content
        if changes.category is not None:
            prompt.category = changes.category
        if changes.tags is not None:
            prompt.tags = changes.tags
        if changes.variables is not None:
            # Validate variables
            validated_variables = []
            for var_data in changes.variables:
                var = PromptVariable.from_dict(var_data)
                validated_variables.append(var.to_dict())
            prompt.variables = validated_variables
        if changes.description is not None:
            prompt.description = changes.description
        
        prompt.updated_at = datetime.utcnow()
        
        # Create new version
        new_version = AdvancedPromptVersion(
            prompt_id=prompt_id,
            version=new_version_num,
            content=prompt.content,
            variables=prompt.variables,
            tags=prompt.tags,
            changes=changes.changes_description or "Updated prompt",
            created_by=updated_by
        )
        
        self.db.add(new_version)
        
        # Update tag usage counts
        if changes.tags is not None:
            self._update_tag_usage(changes.tags)
        
        self.db.commit()
        return new_version
    
    def get_prompt(self, prompt_id: str) -> Optional[AdvancedPromptTemplate]:
        """Get a prompt template by ID."""
        return self.db.query(AdvancedPromptTemplate).filter(AdvancedPromptTemplate.id == prompt_id).first()
    
    def search_prompts(self, query: SearchQuery) -> List[AdvancedPromptTemplate]:
        """Search prompt templates based on query parameters."""
        q = self.db.query(AdvancedPromptTemplate).filter(AdvancedPromptTemplate.is_active == True)
        
        # Text search in name, content, and description
        if query.text:
            text_filter = or_(
                AdvancedPromptTemplate.name.ilike(f"%{query.text}%"),
                AdvancedPromptTemplate.content.ilike(f"%{query.text}%"),
                AdvancedPromptTemplate.description.ilike(f"%{query.text}%")
            )
            q = q.filter(text_filter)
        
        # Category filter
        if query.category:
            q = q.filter(AdvancedPromptTemplate.category == query.category)
        
        # Tags filter (contains any of the specified tags)
        if query.tags:
            for tag in query.tags:
                q = q.filter(AdvancedPromptTemplate.tags.contains([tag]))
        
        # Created by filter
        if query.created_by:
            q = q.filter(AdvancedPromptTemplate.created_by == query.created_by)
        
        # Date range filter
        if query.date_from:
            q = q.filter(AdvancedPromptTemplate.created_at >= query.date_from)
        if query.date_to:
            q = q.filter(AdvancedPromptTemplate.created_at <= query.date_to)
        
        # Apply pagination and ordering
        q = q.order_by(desc(AdvancedPromptTemplate.updated_at))
        q = q.offset(query.offset).limit(query.limit)
        
        return q.all()
    
    def get_prompt_history(self, prompt_id: str) -> List[AdvancedPromptVersion]:
        """Get version history for a prompt template."""
        return self.db.query(AdvancedPromptVersion)\
            .filter(AdvancedPromptVersion.prompt_id == prompt_id)\
            .order_by(desc(AdvancedPromptVersion.created_at))\
            .all()
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Soft delete a prompt template."""
        prompt = self.db.query(AdvancedPromptTemplate).filter(AdvancedPromptTemplate.id == prompt_id).first()
        if not prompt:
            return False
        
        prompt.is_active = False
        prompt.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    def get_categories(self) -> List[AdvancedPromptCategory]:
        """Get all prompt categories."""
        return self.db.query(AdvancedPromptCategory).order_by(AdvancedPromptCategory.name).all()
    
    def create_category(self, name: str, description: Optional[str] = None,
                       parent_id: Optional[str] = None) -> str:
        """Create a new prompt category."""
        category = AdvancedPromptCategory(
            name=name,
            description=description,
            parent_id=parent_id
        )
        self.db.add(category)
        self.db.commit()
        return category.id
    
    def get_tags(self) -> List[AdvancedPromptTag]:
        """Get all prompt tags ordered by usage count."""
        return self.db.query(AdvancedPromptTag)\
            .order_by(desc(AdvancedPromptTag.usage_count), AdvancedPromptTag.name)\
            .all()
    
    def create_tag(self, name: str, description: Optional[str] = None,
                  color: Optional[str] = None) -> str:
        """Create a new prompt tag."""
        tag = AdvancedPromptTag(
            name=name,
            description=description,
            color=color
        )
        self.db.add(tag)
        self.db.commit()
        return tag.id
    
    def substitute_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in prompt content."""
        # Simple variable substitution using {{variable_name}} syntax
        result = content
        for var_name, var_value in variables.items():
            pattern = r'\{\{\s*' + re.escape(var_name) + r'\s*\}\}'
            result = re.sub(pattern, str(var_value), result)
        return result
    
    def validate_prompt_variables(self, content: str, variables: List[Dict[str, Any]]) -> List[str]:
        """Validate that all required variables are defined and used."""
        errors = []
        
        # Extract variable names from content
        content_vars = set(re.findall(r'\{\{\s*(\w+)\s*\}\}', content))
        
        # Check defined variables
        defined_vars = {var['name'] for var in variables}
        required_vars = {var['name'] for var in variables if var.get('required', True)}
        
        # Check for undefined variables in content
        undefined_vars = content_vars - defined_vars
        if undefined_vars:
            errors.append(f"Undefined variables in content: {', '.join(undefined_vars)}")
        
        # Check for unused required variables
        unused_required = required_vars - content_vars
        if unused_required:
            errors.append(f"Required variables not used in content: {', '.join(unused_required)}")
        
        return errors
    
    def _increment_version(self, current_version: str) -> str:
        """Increment version number (semantic versioning)."""
        try:
            parts = current_version.split('.')
            if len(parts) != 3:
                return "1.0.0"
            
            major, minor, patch = map(int, parts)
            return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            return "1.0.0"
    
    def _update_tag_usage(self, tags: List[str]):
        """Update usage count for tags."""
        for tag_name in tags:
            tag = self.db.query(AdvancedPromptTag).filter(AdvancedPromptTag.name == tag_name).first()
            if tag:
                tag.usage_count += 1
            else:
                # Create new tag if it doesn't exist
                new_tag = AdvancedPromptTag(name=tag_name, usage_count=1)
                self.db.add(new_tag)