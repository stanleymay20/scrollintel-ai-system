"""
Import/Export functionality for prompt templates.
"""
import json
import csv
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from io import StringIO
import zipfile
import tempfile
import os

from .prompt_manager import PromptManager
from ..models.prompt_models import AdvancedPromptTemplate, PromptVariable


class PromptImportExport:
    """Handles import and export of prompt templates in various formats."""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def export_prompts(self, prompt_ids: List[str], format: str = "json") -> Union[str, bytes]:
        """Export prompt templates in specified format."""
        prompts_data = []
        
        for prompt_id in prompt_ids:
            prompt = self.prompt_manager.get_prompt(prompt_id)
            if prompt:
                # Get version history
                versions = self.prompt_manager.get_prompt_history(prompt_id)
                
                prompt_data = {
                    "id": prompt.id,
                    "name": prompt.name,
                    "content": prompt.content,
                    "category": prompt.category,
                    "tags": prompt.tags,
                    "variables": prompt.variables,
                    "description": prompt.description,
                    "created_by": prompt.created_by,
                    "created_at": prompt.created_at.isoformat() if prompt.created_at else None,
                    "updated_at": prompt.updated_at.isoformat() if prompt.updated_at else None,
                    "versions": [version.to_dict() for version in versions]
                }
                prompts_data.append(prompt_data)
        
        if format.lower() == "json":
            return self._export_json(prompts_data)
        elif format.lower() == "yaml":
            return self._export_yaml(prompts_data)
        elif format.lower() == "csv":
            return self._export_csv(prompts_data)
        elif format.lower() == "zip":
            return self._export_zip(prompts_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_prompts(self, data: Union[str, bytes], format: str = "json", 
                      created_by: str = "system", overwrite: bool = False) -> Dict[str, Any]:
        """Import prompt templates from various formats."""
        if format.lower() == "json":
            prompts_data = self._import_json(data)
        elif format.lower() == "yaml":
            prompts_data = self._import_yaml(data)
        elif format.lower() == "csv":
            prompts_data = self._import_csv(data)
        elif format.lower() == "zip":
            prompts_data = self._import_zip(data)
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        return self._process_import_data(prompts_data, created_by, overwrite)
    
    def export_template_library(self, category: Optional[str] = None) -> bytes:
        """Export entire template library as a ZIP archive."""
        # Get all prompts or filtered by category
        from .prompt_manager import SearchQuery
        query = SearchQuery(category=category, limit=1000)
        prompts = self.prompt_manager.search_prompts(query)
        
        prompt_ids = [prompt.id for prompt in prompts]
        return self.export_prompts(prompt_ids, "zip")
    
    def _export_json(self, prompts_data: List[Dict[str, Any]]) -> str:
        """Export prompts as JSON."""
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "prompts": prompts_data
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _export_yaml(self, prompts_data: List[Dict[str, Any]]) -> str:
        """Export prompts as YAML."""
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "prompts": prompts_data
        }
        return yaml.dump(export_data, default_flow_style=False, allow_unicode=True)
    
    def _export_csv(self, prompts_data: List[Dict[str, Any]]) -> str:
        """Export prompts as CSV (flattened structure)."""
        output = StringIO()
        if not prompts_data:
            return ""
        
        fieldnames = ["id", "name", "content", "category", "tags", "variables", 
                     "description", "created_by", "created_at", "updated_at"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for prompt in prompts_data:
            row = {
                "id": prompt["id"],
                "name": prompt["name"],
                "content": prompt["content"],
                "category": prompt["category"],
                "tags": json.dumps(prompt["tags"]),
                "variables": json.dumps(prompt["variables"]),
                "description": prompt["description"],
                "created_by": prompt["created_by"],
                "created_at": prompt["created_at"],
                "updated_at": prompt["updated_at"]
            }
            writer.writerow(row)
        
        return output.getvalue()
    
    def _export_zip(self, prompts_data: List[Dict[str, Any]]) -> bytes:
        """Export prompts as ZIP archive with multiple formats."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add JSON export
                json_data = self._export_json(prompts_data)
                zip_file.writestr("prompts.json", json_data)
                
                # Add YAML export
                yaml_data = self._export_yaml(prompts_data)
                zip_file.writestr("prompts.yaml", yaml_data)
                
                # Add CSV export
                csv_data = self._export_csv(prompts_data)
                zip_file.writestr("prompts.csv", csv_data)
                
                # Add individual prompt files
                for prompt in prompts_data:
                    prompt_filename = f"prompts/{prompt['name'].replace('/', '_')}.md"
                    prompt_content = self._format_prompt_as_markdown(prompt)
                    zip_file.writestr(prompt_filename, prompt_content)
                
                # Add metadata
                metadata = {
                    "export_info": {
                        "version": "1.0",
                        "exported_at": datetime.utcnow().isoformat(),
                        "total_prompts": len(prompts_data),
                        "categories": list(set(p["category"] for p in prompts_data)),
                        "tags": list(set(tag for p in prompts_data for tag in p["tags"]))
                    }
                }
                zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            tmp_file.seek(0)
            return tmp_file.read()
    
    def _import_json(self, data: str) -> List[Dict[str, Any]]:
        """Import prompts from JSON."""
        parsed_data = json.loads(data)
        if isinstance(parsed_data, dict) and "prompts" in parsed_data:
            return parsed_data["prompts"]
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            raise ValueError("Invalid JSON format for prompt import")
    
    def _import_yaml(self, data: str) -> List[Dict[str, Any]]:
        """Import prompts from YAML."""
        parsed_data = yaml.safe_load(data)
        if isinstance(parsed_data, dict) and "prompts" in parsed_data:
            return parsed_data["prompts"]
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            raise ValueError("Invalid YAML format for prompt import")
    
    def _import_csv(self, data: str) -> List[Dict[str, Any]]:
        """Import prompts from CSV."""
        reader = csv.DictReader(StringIO(data))
        prompts_data = []
        
        for row in reader:
            prompt_data = {
                "id": row.get("id"),
                "name": row["name"],
                "content": row["content"],
                "category": row["category"],
                "tags": json.loads(row.get("tags", "[]")),
                "variables": json.loads(row.get("variables", "[]")),
                "description": row.get("description"),
                "created_by": row.get("created_by", "imported"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }
            prompts_data.append(prompt_data)
        
        return prompts_data
    
    def _import_zip(self, data: bytes) -> List[Dict[str, Any]]:
        """Import prompts from ZIP archive."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(data)
            tmp_file.flush()
            
            with zipfile.ZipFile(tmp_file.name, 'r') as zip_file:
                # Try to find prompts.json first
                if "prompts.json" in zip_file.namelist():
                    json_data = zip_file.read("prompts.json").decode('utf-8')
                    return self._import_json(json_data)
                elif "prompts.yaml" in zip_file.namelist():
                    yaml_data = zip_file.read("prompts.yaml").decode('utf-8')
                    return self._import_yaml(yaml_data)
                else:
                    raise ValueError("No supported prompt data found in ZIP archive")
    
    def _process_import_data(self, prompts_data: List[Dict[str, Any]], 
                           created_by: str, overwrite: bool) -> Dict[str, Any]:
        """Process imported prompt data and create/update prompts."""
        results = {
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "errors": []
        }
        
        for prompt_data in prompts_data:
            try:
                # Check if prompt already exists
                existing_prompt = None
                if prompt_data.get("id"):
                    existing_prompt = self.prompt_manager.get_prompt(prompt_data["id"])
                
                if existing_prompt and not overwrite:
                    results["skipped"] += 1
                    continue
                
                if existing_prompt and overwrite:
                    # Update existing prompt
                    from .prompt_manager import PromptChanges
                    changes = PromptChanges(
                        name=prompt_data.get("name"),
                        content=prompt_data.get("content"),
                        category=prompt_data.get("category"),
                        tags=prompt_data.get("tags"),
                        variables=prompt_data.get("variables"),
                        description=prompt_data.get("description"),
                        changes_description="Imported update"
                    )
                    self.prompt_manager.update_prompt(existing_prompt.id, changes, created_by)
                    results["updated"] += 1
                else:
                    # Create new prompt
                    self.prompt_manager.create_prompt(
                        name=prompt_data["name"],
                        content=prompt_data["content"],
                        category=prompt_data.get("category", "imported"),
                        created_by=created_by,
                        tags=prompt_data.get("tags"),
                        variables=prompt_data.get("variables"),
                        description=prompt_data.get("description")
                    )
                    results["imported"] += 1
                    
            except Exception as e:
                results["errors"].append(f"Error processing prompt '{prompt_data.get('name', 'unknown')}': {str(e)}")
        
        return results
    
    def _format_prompt_as_markdown(self, prompt_data: Dict[str, Any]) -> str:
        """Format a prompt as Markdown for individual file export."""
        md_content = f"# {prompt_data['name']}\n\n"
        
        if prompt_data.get("description"):
            md_content += f"**Description:** {prompt_data['description']}\n\n"
        
        md_content += f"**Category:** {prompt_data['category']}\n\n"
        
        if prompt_data.get("tags"):
            tags_str = ", ".join(f"`{tag}`" for tag in prompt_data["tags"])
            md_content += f"**Tags:** {tags_str}\n\n"
        
        if prompt_data.get("variables"):
            md_content += "## Variables\n\n"
            for var in prompt_data["variables"]:
                md_content += f"- **{var['name']}** ({var.get('type', 'string')})"
                if var.get('required', True):
                    md_content += " *required*"
                if var.get('description'):
                    md_content += f": {var['description']}"
                if var.get('default'):
                    md_content += f" (default: `{var['default']}`)"
                md_content += "\n"
            md_content += "\n"
        
        md_content += "## Content\n\n"
        md_content += f"```\n{prompt_data['content']}\n```\n\n"
        
        md_content += "---\n"
        md_content += f"*Created by: {prompt_data['created_by']}*\n"
        if prompt_data.get("created_at"):
            md_content += f"*Created at: {prompt_data['created_at']}*\n"
        
        return md_content