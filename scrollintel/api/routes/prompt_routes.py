"""
API routes for the Advanced Prompt Management System.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import io

from ...models.database_utils import get_db
from ...core.prompt_manager import PromptManager, SearchQuery, PromptChanges
from ...core.prompt_import_export import PromptImportExport
from ...models.prompt_models import AdvancedPromptTemplate, AdvancedPromptVersion


# Pydantic models for API
class PromptVariableSchema(BaseModel):
    name: str
    type: str = "string"
    default: Optional[str] = None
    description: Optional[str] = None
    required: bool = True


class PromptTemplateCreate(BaseModel):
    name: str
    content: str
    category: str
    tags: Optional[List[str]] = []
    variables: Optional[List[PromptVariableSchema]] = []
    description: Optional[str] = None


class PromptTemplateUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    variables: Optional[List[PromptVariableSchema]] = None
    description: Optional[str] = None
    changes_description: Optional[str] = None


class PromptTemplateResponse(BaseModel):
    id: str
    name: str
    content: str
    category: str
    tags: List[str]
    variables: List[Dict[str, Any]]
    description: Optional[str]
    is_active: bool
    created_by: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PromptVersionResponse(BaseModel):
    id: str
    prompt_id: str
    version: str
    content: str
    changes: Optional[str]
    variables: List[Dict[str, Any]]
    tags: List[str]
    created_by: str
    created_at: datetime

    class Config:
        from_attributes = True


class PromptSearchRequest(BaseModel):
    text: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50
    offset: int = 0


class PromptCategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None


class PromptTagCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = None


class VariableSubstitutionRequest(BaseModel):
    content: str
    variables: Dict[str, Any]


class ImportResult(BaseModel):
    imported: int
    updated: int
    skipped: int
    errors: List[str]


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


def get_prompt_manager(db: Session = Depends(get_db)) -> PromptManager:
    """Get PromptManager instance."""
    return PromptManager(db)


def get_import_export(prompt_manager: PromptManager = Depends(get_prompt_manager)) -> PromptImportExport:
    """Get PromptImportExport instance."""
    return PromptImportExport(prompt_manager)


@router.post("/", response_model=Dict[str, str])
async def create_prompt(
    prompt: PromptTemplateCreate,
    current_user: str = "system",  # TODO: Get from auth
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Create a new prompt template."""
    try:
        variables_dict = [var.dict() for var in prompt.variables] if prompt.variables else []
        
        prompt_id = prompt_manager.create_prompt(
            name=prompt.name,
            content=prompt.content,
            category=prompt.category,
            created_by=current_user,
            tags=prompt.tags,
            variables=variables_dict,
            description=prompt.description
        )
        
        return {"id": prompt_id, "message": "Prompt created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{prompt_id}", response_model=PromptTemplateResponse)
async def get_prompt(
    prompt_id: str,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get a prompt template by ID."""
    prompt = prompt_manager.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return PromptTemplateResponse.from_orm(prompt)


@router.put("/{prompt_id}", response_model=PromptVersionResponse)
async def update_prompt(
    prompt_id: str,
    updates: PromptTemplateUpdate,
    current_user: str = "system",  # TODO: Get from auth
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Update a prompt template."""
    try:
        variables_dict = None
        if updates.variables is not None:
            variables_dict = [var.dict() for var in updates.variables]
        
        changes = PromptChanges(
            name=updates.name,
            content=updates.content,
            category=updates.category,
            tags=updates.tags,
            variables=variables_dict,
            description=updates.description,
            changes_description=updates.changes_description
        )
        
        new_version = prompt_manager.update_prompt(prompt_id, changes, current_user)
        return PromptVersionResponse.from_orm(new_version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{prompt_id}")
async def delete_prompt(
    prompt_id: str,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Delete (deactivate) a prompt template."""
    success = prompt_manager.delete_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {"message": "Prompt deleted successfully"}


@router.post("/search", response_model=List[PromptTemplateResponse])
async def search_prompts(
    search_request: PromptSearchRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Search prompt templates."""
    query = SearchQuery(
        text=search_request.text,
        category=search_request.category,
        tags=search_request.tags,
        created_by=search_request.created_by,
        date_from=search_request.date_from,
        date_to=search_request.date_to,
        limit=search_request.limit,
        offset=search_request.offset
    )
    
    prompts = prompt_manager.search_prompts(query)
    return [PromptTemplateResponse.from_orm(prompt) for prompt in prompts]


@router.get("/{prompt_id}/history", response_model=List[PromptVersionResponse])
async def get_prompt_history(
    prompt_id: str,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get version history for a prompt template."""
    versions = prompt_manager.get_prompt_history(prompt_id)
    return [PromptVersionResponse.from_orm(version) for version in versions]


@router.post("/substitute", response_model=Dict[str, str])
async def substitute_variables(
    request: VariableSubstitutionRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Substitute variables in prompt content."""
    result = prompt_manager.substitute_variables(request.content, request.variables)
    return {"result": result}


@router.post("/{prompt_id}/validate", response_model=Dict[str, Any])
async def validate_prompt(
    prompt_id: str,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Validate a prompt template's variables."""
    prompt = prompt_manager.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    errors = prompt_manager.validate_prompt_variables(prompt.content, prompt.variables)
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# Category management
@router.get("/categories/", response_model=List[Dict[str, Any]])
async def get_categories(
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get all prompt categories."""
    categories = prompt_manager.get_categories()
    return [category.to_dict() for category in categories]


@router.post("/categories/", response_model=Dict[str, str])
async def create_category(
    category: PromptCategoryCreate,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Create a new prompt category."""
    category_id = prompt_manager.create_category(
        name=category.name,
        description=category.description,
        parent_id=category.parent_id
    )
    return {"id": category_id, "message": "Category created successfully"}


# Tag management
@router.get("/tags/", response_model=List[Dict[str, Any]])
async def get_tags(
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Get all prompt tags."""
    tags = prompt_manager.get_tags()
    return [tag.to_dict() for tag in tags]


@router.post("/tags/", response_model=Dict[str, str])
async def create_tag(
    tag: PromptTagCreate,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Create a new prompt tag."""
    tag_id = prompt_manager.create_tag(
        name=tag.name,
        description=tag.description,
        color=tag.color
    )
    return {"id": tag_id, "message": "Tag created successfully"}


# Import/Export functionality
@router.post("/export", response_class=StreamingResponse)
async def export_prompts(
    prompt_ids: List[str],
    format: str = "json",
    import_export: PromptImportExport = Depends(get_import_export)
):
    """Export prompt templates."""
    try:
        data = import_export.export_prompts(prompt_ids, format)
        
        if format.lower() == "zip":
            media_type = "application/zip"
            filename = "prompts.zip"
        elif format.lower() == "csv":
            media_type = "text/csv"
            filename = "prompts.csv"
        elif format.lower() == "yaml":
            media_type = "text/yaml"
            filename = "prompts.yaml"
        else:
            media_type = "application/json"
            filename = "prompts.json"
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return StreamingResponse(
            io.BytesIO(data),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import", response_model=ImportResult)
async def import_prompts(
    file: UploadFile = File(...),
    format: str = Form("json"),
    overwrite: bool = Form(False),
    current_user: str = "system",  # TODO: Get from auth
    import_export: PromptImportExport = Depends(get_import_export)
):
    """Import prompt templates from file."""
    try:
        content = await file.read()
        
        if format.lower() in ["json", "yaml", "csv"]:
            content = content.decode('utf-8')
        
        result = import_export.import_prompts(content, format, current_user, overwrite)
        return ImportResult(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/export/library")
async def export_template_library(
    category: Optional[str] = None,
    import_export: PromptImportExport = Depends(get_import_export)
):
    """Export entire template library as ZIP."""
    try:
        data = import_export.export_template_library(category)
        
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=template_library.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))