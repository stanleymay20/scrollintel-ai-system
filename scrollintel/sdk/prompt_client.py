"""
ScrollIntel Prompt Management Client - Python SDK for programmatic access.
"""
import json
import time
from typing import List, Optional, Dict, Any, Union, BinaryIO
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    ScrollIntelSDKError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
    ServerError,
    NetworkError,
    TimeoutError
)
from .models import (
    PromptTemplate,
    PromptVersion,
    PromptVariable,
    SearchQuery,
    APIResponse,
    PaginatedResponse,
    PromptUsageMetrics,
    BatchOperation,
    WebhookConfig
)


class PromptClient:
    """Client for interacting with the ScrollIntel Prompt Management API."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_version: str = "v1",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.3
    ):
        """
        Initialize the PromptClient.
        
        Args:
            base_url: Base URL of the ScrollIntel API
            api_key: API key for authentication
            api_version: API version to use (default: "v1")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.timeout = timeout
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"ScrollIntel-SDK-Python/1.0.0"
        })
    
    def _get_url(self, endpoint: str) -> str:
        """Get full URL for an endpoint."""
        return f"{self.base_url}/api/{self.api_version}/prompts{endpoint}"
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = {"message": response.text}
        
        if response.status_code == 200:
            return data
        elif response.status_code == 401:
            raise AuthenticationError(data.get("message", "Authentication failed"))
        elif response.status_code == 403:
            raise AuthorizationError(data.get("message", "Access denied"))
        elif response.status_code == 404:
            raise NotFoundError(data.get("message", "Resource not found"))
        elif response.status_code == 400:
            raise ValidationError(data.get("message", "Validation error"), data.get("errors", []))
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(data.get("message", "Rate limit exceeded"), retry_after)
        elif response.status_code >= 500:
            raise ServerError(data.get("message", "Server error"), response.status_code)
        else:
            raise ScrollIntelSDKError(
                data.get("message", f"HTTP {response.status_code}"),
                response.status_code,
                data
            )
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to the API."""
        url = self._get_url(endpoint)
        
        try:
            if files:
                # For file uploads, don't set Content-Type header
                headers = {k: v for k, v in self.session.headers.items() if k != "Content-Type"}
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data if not files else None,
                    params=params,
                    files=files,
                    timeout=self.timeout,
                    headers=headers if files else None
                )
            else:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
            
            return self._handle_response(response)
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Network error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise ScrollIntelSDKError(f"Request error: {str(e)}")
    
    # Prompt CRUD operations
    
    def create_prompt(
        self,
        name: str,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        variables: Optional[List[PromptVariable]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a new prompt template.
        
        Args:
            name: Name of the prompt
            content: Prompt content
            category: Category of the prompt
            tags: List of tags
            variables: List of prompt variables
            description: Description of the prompt
            
        Returns:
            ID of the created prompt
        """
        data = {
            "name": name,
            "content": content,
            "category": category,
            "tags": tags or [],
            "variables": [var.to_dict() for var in (variables or [])],
            "description": description
        }
        
        response = self._make_request("POST", "/", data=data)
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to create prompt")
        
        return api_response.data["id"]
    
    def get_prompt(self, prompt_id: str) -> PromptTemplate:
        """
        Get a prompt template by ID.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            PromptTemplate object
        """
        response = self._make_request("GET", f"/{prompt_id}")
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to get prompt")
        
        return PromptTemplate.from_dict(api_response.data)
    
    def update_prompt(
        self,
        prompt_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        variables: Optional[List[PromptVariable]] = None,
        description: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> PromptVersion:
        """
        Update a prompt template.
        
        Args:
            prompt_id: ID of the prompt to update
            name: New name (optional)
            content: New content (optional)
            category: New category (optional)
            tags: New tags (optional)
            variables: New variables (optional)
            description: New description (optional)
            changes_description: Description of changes made
            
        Returns:
            New PromptVersion object
        """
        data = {}
        if name is not None:
            data["name"] = name
        if content is not None:
            data["content"] = content
        if category is not None:
            data["category"] = category
        if tags is not None:
            data["tags"] = tags
        if variables is not None:
            data["variables"] = [var.to_dict() for var in variables]
        if description is not None:
            data["description"] = description
        if changes_description is not None:
            data["changes_description"] = changes_description
        
        response = self._make_request("PUT", f"/{prompt_id}", data=data)
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to update prompt")
        
        return PromptVersion.from_dict(api_response.data)
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """
        Delete (deactivate) a prompt template.
        
        Args:
            prompt_id: ID of the prompt to delete
            
        Returns:
            True if successful
        """
        response = self._make_request("DELETE", f"/{prompt_id}")
        api_response = APIResponse.from_dict(response)
        
        return api_response.success
    
    def list_prompts(
        self,
        page: int = 1,
        page_size: int = 50,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> PaginatedResponse:
        """
        List prompts with pagination and filtering.
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            PaginatedResponse with PromptTemplate objects
        """
        params = {
            "page": page,
            "page_size": page_size
        }
        if category:
            params["category"] = category
        if tags:
            params["tags"] = ",".join(tags)
        
        response = self._make_request("GET", "/", params=params)
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to list prompts")
        
        paginated_data = PaginatedResponse.from_dict(api_response.data)
        # Convert items to PromptTemplate objects
        paginated_data.items = [PromptTemplate.from_dict(item) for item in paginated_data.items]
        
        return paginated_data
    
    def search_prompts(self, query: SearchQuery) -> PaginatedResponse:
        """
        Search prompt templates.
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            PaginatedResponse with PromptTemplate objects
        """
        response = self._make_request("POST", "/search", data=query.to_dict())
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to search prompts")
        
        paginated_data = PaginatedResponse.from_dict(api_response.data)
        # Convert items to PromptTemplate objects
        paginated_data.items = [PromptTemplate.from_dict(item) for item in paginated_data.items]
        
        return paginated_data
    
    def get_prompt_history(self, prompt_id: str) -> List[PromptVersion]:
        """
        Get version history for a prompt template.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            List of PromptVersion objects
        """
        response = self._make_request("GET", f"/{prompt_id}/history")
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to get prompt history")
        
        return [PromptVersion.from_dict(version) for version in api_response.data]
    
    def get_prompt_metrics(self, prompt_id: str) -> PromptUsageMetrics:
        """
        Get usage metrics for a prompt template.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            PromptUsageMetrics object
        """
        response = self._make_request("GET", f"/{prompt_id}/metrics")
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to get prompt metrics")
        
        return PromptUsageMetrics.from_dict(api_response.data)
    
    # Batch operations
    
    def batch_operations(self, operations: List[BatchOperation]) -> Dict[str, Any]:
        """
        Perform batch operations on multiple prompts.
        
        Args:
            operations: List of BatchOperation objects
            
        Returns:
            Dictionary with results and errors
        """
        data = [op.to_dict() for op in operations]
        
        response = self._make_request("POST", "/batch", data=data)
        api_response = APIResponse.from_dict(response)
        
        return api_response.data
    
    # Utility methods
    
    def substitute_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in prompt content.
        
        Args:
            content: Prompt content with variables
            variables: Dictionary of variable values
            
        Returns:
            Content with substituted variables
        """
        data = {
            "content": content,
            "variables": variables
        }
        
        # Use the original API endpoint for this
        url = f"{self.base_url}/api/prompts/substitute"
        response = self.session.post(url, json=data, timeout=self.timeout)
        result = self._handle_response(response)
        
        return result["result"]
    
    def validate_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Validate a prompt template's variables.
        
        Args:
            prompt_id: ID of the prompt to validate
            
        Returns:
            Validation result with errors if any
        """
        # Use the original API endpoint for this
        url = f"{self.base_url}/api/prompts/{prompt_id}/validate"
        response = self.session.post(url, timeout=self.timeout)
        return self._handle_response(response)
    
    # Import/Export functionality
    
    def export_prompts(
        self,
        prompt_ids: List[str],
        format: str = "json"
    ) -> bytes:
        """
        Export prompt templates.
        
        Args:
            prompt_ids: List of prompt IDs to export
            format: Export format ("json", "yaml", "csv", "zip")
            
        Returns:
            Exported data as bytes
        """
        # Use the original API endpoint for this
        url = f"{self.base_url}/api/prompts/export"
        data = prompt_ids
        params = {"format": format}
        
        response = self.session.post(url, json=data, params=params, timeout=self.timeout)
        
        if response.status_code != 200:
            self._handle_response(response)
        
        return response.content
    
    def import_prompts(
        self,
        file_data: Union[str, bytes, BinaryIO],
        format: str = "json",
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Import prompt templates from file data.
        
        Args:
            file_data: File data to import (string, bytes, or file-like object)
            format: Import format ("json", "yaml", "csv")
            overwrite: Whether to overwrite existing prompts
            
        Returns:
            Import result with counts and errors
        """
        # Use the original API endpoint for this
        url = f"{self.base_url}/api/prompts/import"
        
        if hasattr(file_data, 'read'):
            # File-like object
            files = {"file": file_data}
        elif isinstance(file_data, bytes):
            files = {"file": ("prompts", file_data)}
        else:
            # String data
            files = {"file": ("prompts", file_data.encode('utf-8'))}
        
        data = {
            "format": format,
            "overwrite": overwrite
        }
        
        response = self.session.post(url, files=files, data=data, timeout=self.timeout)
        return self._handle_response(response)
    
    # Context manager support
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
    
    def close(self):
        """Close the client session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    # Webhook management
    
    def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ) -> str:
        """
        Register a webhook endpoint.
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Secret for signature verification
            
        Returns:
            Webhook ID
        """
        data = {
            "url": url,
            "events": events,
            "secret": secret,
            "active": True
        }
        
        response = self._make_request("POST", "/webhooks", data=data)
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to register webhook")
        
        return api_response.data["webhook_id"]
    
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """
        List all webhook endpoints.
        
        Returns:
            List of webhook configurations
        """
        response = self._make_request("GET", "/webhooks")
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to list webhooks")
        
        return api_response.data
    
    def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        Update a webhook endpoint.
        
        Args:
            webhook_id: ID of the webhook to update
            url: New webhook URL
            events: New list of events
            secret: New secret
            active: Whether webhook is active
            
        Returns:
            True if successful
        """
        data = {}
        if url is not None:
            data["url"] = url
        if events is not None:
            data["events"] = events
        if secret is not None:
            data["secret"] = secret
        if active is not None:
            data["active"] = active
        
        response = self._make_request("PUT", f"/webhooks/{webhook_id}", data=data)
        api_response = APIResponse.from_dict(response)
        
        return api_response.success
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook endpoint.
        
        Args:
            webhook_id: ID of the webhook to delete
            
        Returns:
            True if successful
        """
        response = self._make_request("DELETE", f"/webhooks/{webhook_id}")
        api_response = APIResponse.from_dict(response)
        
        return api_response.success
    
    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """
        Test a webhook endpoint.
        
        Args:
            webhook_id: ID of the webhook to test
            
        Returns:
            Test result
        """
        response = self._make_request("POST", f"/webhooks/{webhook_id}/test")
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to test webhook")
        
        return api_response.data
    
    # Usage analytics
    
    def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage summary for the current user.
        
        Args:
            start_date: Start date for the summary
            end_date: End date for the summary
            
        Returns:
            Usage summary data
        """
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        response = self._make_request("GET", "/usage/summary", params=params)
        api_response = APIResponse.from_dict(response)
        
        if not api_response.success:
            raise ScrollIntelSDKError(api_response.message or "Failed to get usage summary")
        
        return api_response.data
    
    # Health check
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        url = f"{self.base_url}/health"
        response = self.session.get(url, timeout=self.timeout)
        return self._handle_response(response)