"""
API Documentation Generator

This module generates comprehensive API documentation including OpenAPI/Swagger specs,
interactive documentation, and developer guides.
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
from jinja2 import Environment, BaseLoader, Template

from ..models.api_generation_models import (
    APISpec, Endpoint, Parameter, Response, HTTPMethod,
    SecurityScheme, APIVersion, GeneratedAPICode
)


class APIDocumentationGenerator:
    """Generates comprehensive API documentation."""
    
    def __init__(self):
        self.jinja_env = Environment(loader=BaseLoader())
    
    def generate_openapi_specification(self, api_spec: APISpec) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": api_spec.name,
                "description": api_spec.description,
                "version": api_spec.version,
                "contact": {
                    "name": "API Support",
                    "email": "support@example.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": api_spec.base_url,
                    "description": "Production server"
                },
                {
                    "url": api_spec.base_url.replace("api", "api-staging"),
                    "description": "Staging server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "responses": {},
                "parameters": {},
                "examples": {},
                "requestBodies": {},
                "headers": {},
                "securitySchemes": {},
                "links": {},
                "callbacks": {}
            },
            "security": [],
            "tags": [],
            "externalDocs": {
                "description": "Find more info here",
                "url": "https://example.com/docs"
            }
        }
        
        # Add security schemes
        for scheme in api_spec.security_schemes:
            openapi_spec["components"]["securitySchemes"][scheme.name] = self._security_scheme_to_openapi(scheme)
        
        # Add global security if any schemes exist
        if api_spec.security_schemes:
            openapi_spec["security"] = [
                {scheme.name: []} for scheme in api_spec.security_schemes
            ]
        
        # Generate tags from endpoints
        tags = self._extract_tags(api_spec)
        openapi_spec["tags"] = [
            {"name": tag, "description": f"Operations related to {tag}"}
            for tag in tags
        ]
        
        # Add paths
        for endpoint in api_spec.endpoints:
            self._add_endpoint_to_openapi(endpoint, openapi_spec)
        
        # Add common schemas
        self._add_common_schemas(openapi_spec)
        
        # Add common responses
        self._add_common_responses(openapi_spec)
        
        return openapi_spec
    
    def generate_swagger_ui_html(self, api_spec: APISpec, openapi_spec: Dict[str, Any]) -> str:
        """Generate Swagger UI HTML page."""
        template = Template('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ api_spec.name }} - API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        body {
            margin:0;
            background: #fafafa;
        }
        .swagger-ui .topbar {
            background-color: #2c3e50;
        }
        .swagger-ui .topbar .download-url-wrapper .select-label {
            color: #ffffff;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                spec: {{ openapi_spec | tojson }},
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {
                    console.log("Swagger UI loaded successfully");
                },
                onFailure: function(data) {
                    console.error("Failed to load Swagger UI", data);
                }
            });
        };
    </script>
</body>
</html>''')
        
        return template.render(api_spec=api_spec, openapi_spec=openapi_spec)
    
    def generate_redoc_html(self, api_spec: APISpec, openapi_spec: Dict[str, Any]) -> str:
        """Generate ReDoc HTML page."""
        template = Template('''<!DOCTYPE html>
<html>
<head>
    <title>{{ api_spec.name }} - API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
</head>
<body>
    <redoc spec-url="data:application/json;base64,{{ openapi_spec_b64 }}"></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
</body>
</html>''')
        
        import base64
        openapi_json = json.dumps(openapi_spec)
        openapi_b64 = base64.b64encode(openapi_json.encode()).decode()
        
        return template.render(api_spec=api_spec, openapi_spec_b64=openapi_b64)
    
    def generate_postman_collection(self, api_spec: APISpec) -> Dict[str, Any]:
        """Generate Postman collection for API testing."""
        collection = {
            "info": {
                "name": api_spec.name,
                "description": api_spec.description,
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                "_postman_id": api_spec.id,
                "version": {
                    "major": int(api_spec.version.split('.')[0]),
                    "minor": int(api_spec.version.split('.')[1]),
                    "patch": int(api_spec.version.split('.')[2])
                }
            },
            "item": [],
            "auth": None,
            "event": [],
            "variable": [
                {
                    "key": "baseUrl",
                    "value": api_spec.base_url,
                    "type": "string"
                }
            ]
        }
        
        # Add authentication if present
        if api_spec.security_schemes:
            auth_scheme = api_spec.security_schemes[0]  # Use first scheme
            if auth_scheme.type == "http" and auth_scheme.scheme == "bearer":
                collection["auth"] = {
                    "type": "bearer",
                    "bearer": [
                        {
                            "key": "token",
                            "value": "{{authToken}}",
                            "type": "string"
                        }
                    ]
                }
                collection["variable"].append({
                    "key": "authToken",
                    "value": "your_auth_token_here",
                    "type": "string"
                })
        
        # Group endpoints by tags
        tag_groups = {}
        for endpoint in api_spec.endpoints:
            for tag in endpoint.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(endpoint)
        
        # Create folders for each tag
        for tag, endpoints in tag_groups.items():
            folder = {
                "name": tag,
                "item": [],
                "description": f"Operations related to {tag}"
            }
            
            for endpoint in endpoints:
                request_item = self._endpoint_to_postman_request(endpoint, api_spec)
                folder["item"].append(request_item)
            
            collection["item"].append(folder)
        
        return collection
    
    def generate_api_client_code(
        self, 
        api_spec: APISpec, 
        language: str = "python"
    ) -> Dict[str, str]:
        """Generate API client code in various languages."""
        if language == "python":
            return self._generate_python_client(api_spec)
        elif language == "javascript":
            return self._generate_javascript_client(api_spec)
        elif language == "typescript":
            return self._generate_typescript_client(api_spec)
        else:
            raise ValueError(f"Unsupported client language: {language}")
    
    def generate_developer_guide(self, api_spec: APISpec) -> str:
        """Generate comprehensive developer guide."""
        template = Template('''# {{ api_spec.name }} Developer Guide

{{ api_spec.description }}

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [API Reference](#api-reference)
6. [SDKs and Libraries](#sdks-and-libraries)
7. [Examples](#examples)
8. [Changelog](#changelog)
9. [Support](#support)

## Getting Started

### Base URL
```
{{ api_spec.base_url }}
```

### API Version
Current version: `{{ api_spec.version }}`

### Content Type
All requests should use `application/json` content type unless otherwise specified.

### Quick Start Example

```bash
curl -X GET "{{ api_spec.base_url }}/health" \\
  -H "Accept: application/json"
```

## Authentication

{% if api_spec.security_schemes %}
This API uses the following authentication methods:

{% for scheme in api_spec.security_schemes %}
### {{ scheme.name }}

**Type:** {{ scheme.type }}
{% if scheme.description %}
**Description:** {{ scheme.description }}
{% endif %}

{% if scheme.type == "http" and scheme.scheme == "bearer" %}
Include the Bearer token in the Authorization header:

```bash
curl -X GET "{{ api_spec.base_url }}/protected-endpoint" \\
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

```python
import requests

headers = {
    "Authorization": "Bearer YOUR_TOKEN_HERE",
    "Content-Type": "application/json"
}

response = requests.get("{{ api_spec.base_url }}/protected-endpoint", headers=headers)
```
{% endif %}

{% endfor %}
{% else %}
This API does not require authentication.
{% endif %}

## Rate Limiting

To ensure fair usage, this API implements rate limiting:

- **Rate Limit:** 1000 requests per hour per API key
- **Burst Limit:** 100 requests per minute

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Request limit per hour
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets (Unix timestamp)

When rate limit is exceeded, the API returns a `429 Too Many Requests` status code.

## Error Handling

The API uses standard HTTP status codes and returns error details in JSON format:

```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": {
    "field": "email",
    "code": "invalid_format"
  }
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 201  | Created |
| 400  | Bad Request - Invalid input |
| 401  | Unauthorized - Authentication required |
| 403  | Forbidden - Insufficient permissions |
| 404  | Not Found - Resource doesn't exist |
| 422  | Unprocessable Entity - Validation error |
| 429  | Too Many Requests - Rate limit exceeded |
| 500  | Internal Server Error |

## API Reference

{% for tag in unique_tags %}
### {{ tag }}

{% for endpoint in api_spec.endpoints %}
{% if tag in endpoint.tags %}
#### {{ endpoint.method.value }} {{ endpoint.path }}

{{ endpoint.description }}

**Parameters:**

{% if endpoint.parameters %}
| Name | Type | Required | Description |
|------|------|----------|-------------|
{% for param in endpoint.parameters %}
| `{{ param.name }}` | {{ param.type }} | {{ "Yes" if param.required else "No" }} | {{ param.description or "No description" }} |
{% endfor %}
{% else %}
No parameters required.
{% endif %}

{% if endpoint.request_body %}
**Request Body:**

```json
{
  "example": "request body"
}
```
{% endif %}

**Responses:**

{% for response in endpoint.responses %}
**{{ response.status_code }}** - {{ response.description }}

{% if response.schema %}
```json
{
  "example": "response"
}
```
{% endif %}

{% endfor %}

**Example Request:**

```bash
curl -X {{ endpoint.method.value }} "{{ api_spec.base_url }}{{ endpoint.path }}" \\
{% if api_spec.security_schemes %}
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
{% endif %}
  -H "Content-Type: application/json"
{% if endpoint.request_body %}
  -d '{
    "example": "data"
  }'
{% endif %}
```

```python
import requests

{% if api_spec.security_schemes %}
headers = {
    "Authorization": "Bearer YOUR_TOKEN_HERE",
    "Content-Type": "application/json"
}
{% else %}
headers = {"Content-Type": "application/json"}
{% endif %}

{% if endpoint.method.value == "GET" %}
response = requests.get("{{ api_spec.base_url }}{{ endpoint.path }}", headers=headers)
{% elif endpoint.method.value == "POST" %}
data = {"example": "data"}
response = requests.post("{{ api_spec.base_url }}{{ endpoint.path }}", json=data, headers=headers)
{% elif endpoint.method.value == "PUT" %}
data = {"example": "data"}
response = requests.put("{{ api_spec.base_url }}{{ endpoint.path }}", json=data, headers=headers)
{% elif endpoint.method.value == "DELETE" %}
response = requests.delete("{{ api_spec.base_url }}{{ endpoint.path }}", headers=headers)
{% endif %}

print(response.json())
```

---

{% endif %}
{% endfor %}
{% endfor %}

## SDKs and Libraries

### Python SDK

```bash
pip install {{ api_spec.name.lower().replace(' ', '-') }}-sdk
```

```python
from {{ api_spec.name.lower().replace(' ', '_') }}_sdk import Client

client = Client(api_key="your_api_key")
result = client.get_data()
```

### JavaScript SDK

```bash
npm install {{ api_spec.name.lower().replace(' ', '-') }}-sdk
```

```javascript
const { Client } = require('{{ api_spec.name.lower().replace(' ', '-') }}-sdk');

const client = new Client({ apiKey: 'your_api_key' });
const result = await client.getData();
```

## Examples

### Complete Workflow Example

```python
import requests
import json

# Configuration
BASE_URL = "{{ api_spec.base_url }}"
{% if api_spec.security_schemes %}
API_KEY = "your_api_key_here"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
{% else %}
HEADERS = {"Content-Type": "application/json"}
{% endif %}

# Example workflow
def example_workflow():
    # Step 1: Get list of items
    response = requests.get(f"{BASE_URL}/items", headers=HEADERS)
    items = response.json()
    print(f"Found {len(items)} items")
    
    # Step 2: Create a new item
    new_item = {
        "name": "Example Item",
        "description": "This is an example"
    }
    response = requests.post(f"{BASE_URL}/items", json=new_item, headers=HEADERS)
    created_item = response.json()
    print(f"Created item with ID: {created_item['id']}")
    
    # Step 3: Update the item
    update_data = {"description": "Updated description"}
    response = requests.put(
        f"{BASE_URL}/items/{created_item['id']}", 
        json=update_data, 
        headers=HEADERS
    )
    updated_item = response.json()
    print(f"Updated item: {updated_item}")
    
    # Step 4: Delete the item
    response = requests.delete(f"{BASE_URL}/items/{created_item['id']}", headers=HEADERS)
    print(f"Deleted item, status: {response.status_code}")

if __name__ == "__main__":
    example_workflow()
```

## Changelog

{% if api_spec.versions %}
{% for version in api_spec.versions %}
### Version {{ version.version }} - {{ version.release_date.strftime('%Y-%m-%d') }}

{% if version.breaking_changes %}
**Breaking Changes:**
{% for change in version.breaking_changes %}
- {{ change }}
{% endfor %}
{% endif %}

{% if version.changelog %}
**Changes:**
{% for change in version.changelog %}
- {{ change }}
{% endfor %}
{% endif %}

{% if version.deprecated %}
⚠️ **This version is deprecated.** {% if version.sunset_date %}It will be sunset on {{ version.sunset_date.strftime('%Y-%m-%d') }}.{% endif %}
{% endif %}

{% endfor %}
{% else %}
### Version {{ api_spec.version }}
- Initial release
{% endif %}

## Support

### Getting Help

- **Documentation:** [API Documentation]({{ api_spec.base_url }}/docs)
- **Email:** support@example.com
- **GitHub Issues:** [Report Issues](https://github.com/example/api/issues)

### Status Page

Check the current status of the API at: [Status Page](https://status.example.com)

### Community

- **Discord:** [Join our Discord](https://discord.gg/example)
- **Stack Overflow:** Tag your questions with `{{ api_spec.name.lower().replace(' ', '-') }}-api`

---

*This documentation was auto-generated on {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}*
''')
        
        return template.render(
            api_spec=api_spec,
            unique_tags=self._extract_tags(api_spec),
            datetime=datetime
        )
    
    def _security_scheme_to_openapi(self, scheme: SecurityScheme) -> Dict[str, Any]:
        """Convert security scheme to OpenAPI format."""
        openapi_scheme = {
            "type": scheme.type
        }
        
        if scheme.description:
            openapi_scheme["description"] = scheme.description
        
        if scheme.type == "http":
            openapi_scheme["scheme"] = scheme.scheme
            if scheme.bearer_format:
                openapi_scheme["bearerFormat"] = scheme.bearer_format
        elif scheme.type == "apiKey":
            openapi_scheme["in"] = scheme.in_location
            openapi_scheme["name"] = scheme.name
        elif scheme.type == "oauth2":
            openapi_scheme["flows"] = scheme.flows or {}
        
        return openapi_scheme
    
    def _extract_tags(self, api_spec: APISpec) -> List[str]:
        """Extract unique tags from API specification."""
        tags = set()
        for endpoint in api_spec.endpoints:
            tags.update(endpoint.tags)
        return sorted(list(tags))
    
    def _add_endpoint_to_openapi(self, endpoint: Endpoint, openapi_spec: Dict[str, Any]):
        """Add endpoint to OpenAPI specification."""
        path = endpoint.path
        method = endpoint.method.value.lower()
        
        if path not in openapi_spec["paths"]:
            openapi_spec["paths"][path] = {}
        
        operation = {
            "operationId": endpoint.name,
            "summary": endpoint.summary,
            "description": endpoint.description,
            "tags": endpoint.tags,
            "parameters": [],
            "responses": {}
        }
        
        # Add parameters
        for param in endpoint.parameters:
            param_location = "path" if f"{{{param.name}}}" in path else "query"
            param_spec = {
                "name": param.name,
                "in": param_location,
                "required": param.required or param_location == "path",
                "schema": {"type": param.type}
            }
            
            if param.description:
                param_spec["description"] = param.description
            
            if param.default_value is not None:
                param_spec["schema"]["default"] = param.default_value
            
            # Add validation rules
            for rule in param.validation_rules:
                if rule["rule"] == "min_length":
                    param_spec["schema"]["minLength"] = rule["value"]
                elif rule["rule"] == "max_length":
                    param_spec["schema"]["maxLength"] = rule["value"]
                elif rule["rule"] == "pattern":
                    param_spec["schema"]["pattern"] = rule["value"]
                elif rule["rule"] == "min_value":
                    param_spec["schema"]["minimum"] = rule["value"]
                elif rule["rule"] == "max_value":
                    param_spec["schema"]["maximum"] = rule["value"]
            
            operation["parameters"].append(param_spec)
        
        # Add request body
        if endpoint.request_body:
            operation["requestBody"] = endpoint.request_body
        
        # Add responses
        for response in endpoint.responses:
            response_spec = {
                "description": response.description
            }
            
            if response.schema:
                response_spec["content"] = {
                    "application/json": {
                        "schema": response.schema
                    }
                }
            
            if response.headers:
                response_spec["headers"] = {}
                for header_name, header_desc in response.headers.items():
                    response_spec["headers"][header_name] = {
                        "description": header_desc,
                        "schema": {"type": "string"}
                    }
            
            operation["responses"][str(response.status_code)] = response_spec
        
        # Add security requirements
        if endpoint.security_requirements:
            operation["security"] = [
                {req: []} for req in endpoint.security_requirements
            ]
        
        # Mark as deprecated if needed
        if endpoint.deprecated:
            operation["deprecated"] = True
        
        openapi_spec["paths"][path][method] = operation
    
    def _add_common_schemas(self, openapi_spec: Dict[str, Any]):
        """Add common schemas to OpenAPI specification."""
        common_schemas = {
            "Error": {
                "type": "object",
                "required": ["error", "message"],
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error code"
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable error message"
                    },
                    "details": {
                        "type": "object",
                        "description": "Additional error details"
                    }
                }
            },
            "PaginationMeta": {
                "type": "object",
                "properties": {
                    "total": {
                        "type": "integer",
                        "description": "Total number of items"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Current page number"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Items per page"
                    },
                    "pages": {
                        "type": "integer",
                        "description": "Total number of pages"
                    }
                }
            }
        }
        
        openapi_spec["components"]["schemas"].update(common_schemas)
    
    def _add_common_responses(self, openapi_spec: Dict[str, Any]):
        """Add common responses to OpenAPI specification."""
        common_responses = {
            "BadRequest": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Unauthorized": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Forbidden": {
                "description": "Forbidden",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "InternalServerError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            }
        }
        
        openapi_spec["components"]["responses"].update(common_responses)
    
    def _endpoint_to_postman_request(self, endpoint: Endpoint, api_spec: APISpec) -> Dict[str, Any]:
        """Convert endpoint to Postman request format."""
        request_item = {
            "name": endpoint.summary or endpoint.name,
            "request": {
                "method": endpoint.method.value,
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json",
                        "type": "text"
                    }
                ],
                "url": {
                    "raw": f"{{{{baseUrl}}}}{endpoint.path}",
                    "host": ["{{baseUrl}}"],
                    "path": endpoint.path.strip("/").split("/"),
                    "query": []
                },
                "description": endpoint.description
            },
            "response": []
        }
        
        # Add query parameters
        for param in endpoint.parameters:
            if f"{{{param.name}}}" not in endpoint.path:  # Not a path parameter
                request_item["request"]["url"]["query"].append({
                    "key": param.name,
                    "value": str(param.default_value) if param.default_value else "",
                    "description": param.description,
                    "disabled": not param.required
                })
        
        # Add request body
        if endpoint.request_body:
            request_item["request"]["body"] = {
                "mode": "raw",
                "raw": json.dumps({"example": "data"}, indent=2),
                "options": {
                    "raw": {
                        "language": "json"
                    }
                }
            }
        
        # Add example responses
        for response in endpoint.responses:
            example_response = {
                "name": f"{response.status_code} - {response.description}",
                "originalRequest": request_item["request"].copy(),
                "status": response.description,
                "code": response.status_code,
                "_postman_previewlanguage": "json",
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "cookie": [],
                "body": json.dumps({"example": "response"}, indent=2) if response.schema else ""
            }
            request_item["response"].append(example_response)
        
        return request_item
    
    def _generate_python_client(self, api_spec: APISpec) -> Dict[str, str]:
        """Generate Python API client."""
        client_code = f'''"""
Python SDK for {api_spec.name}
Auto-generated API client
"""

import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json


@dataclass
class APIResponse:
    """API response wrapper."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    success: bool
    
    @property
    def json(self) -> Any:
        """Get response data as JSON."""
        return self.data


class {api_spec.name.replace(' ', '')}Client:
    """Client for {api_spec.name} API."""
    
    def __init__(self, base_url: str = "{api_spec.base_url}", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({{
            "Content-Type": "application/json",
            "User-Agent": f"{api_spec.name.replace(' ', '')}-Python-SDK/{api_spec.version}"
        }})
        
        # Set authentication
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {{api_key}}"
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """Make HTTP request."""
        url = f"{{self.base_url}}{{endpoint}}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                **kwargs
            )
            
            # Parse response
            try:
                response_data = response.json()
            except ValueError:
                response_data = response.text
            
            return APIResponse(
                status_code=response.status_code,
                data=response_data,
                headers=dict(response.headers),
                success=response.status_code < 400
            )
            
        except requests.RequestException as e:
            raise Exception(f"Request failed: {{str(e)}}")
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make GET request."""
        return self._request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make POST request."""
        return self._request("POST", endpoint, data=data)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make PUT request."""
        return self._request("PUT", endpoint, data=data)
    
    def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make PATCH request."""
        return self._request("PATCH", endpoint, data=data)
    
    def delete(self, endpoint: str) -> APIResponse:
        """Make DELETE request."""
        return self._request("DELETE", endpoint)
'''
        
        # Add method for each endpoint
        for endpoint in api_spec.endpoints:
            method_name = endpoint.name.replace("-", "_")
            client_code += f'''
    def {method_name}(self'''
            
            # Add parameters
            for param in endpoint.parameters:
                if f"{{{param.name}}}" in endpoint.path:
                    client_code += f", {param.name}: str"
                else:
                    param_type = "str" if param.type == "string" else param.type
                    if not param.required:
                        client_code += f", {param.name}: Optional[{param_type}] = None"
                    else:
                        client_code += f", {param.name}: {param_type}"
            
            # Add request body parameter
            if endpoint.request_body:
                client_code += ", data: Dict[str, Any]"
            
            client_code += ") -> APIResponse:\n"
            client_code += f'        """{endpoint.description}"""\n'
            
            # Build endpoint path
            path = endpoint.path
            for param in endpoint.parameters:
                if f"{{{param.name}}}" in path:
                    path = path.replace(f"{{{param.name}}}", f"{{{param.name}}}")
            
            client_code += f'        endpoint = f"{path}"\n'
            
            # Build query parameters
            query_params = [p for p in endpoint.parameters if f"{{{p.name}}}" not in endpoint.path]
            if query_params:
                client_code += "        params = {}\n"
                for param in query_params:
                    client_code += f'        if {param.name} is not None:\n'
                    client_code += f'            params["{param.name}"] = {param.name}\n'
                client_code += f'        return self.{endpoint.method.value.lower()}(endpoint, params=params'
            else:
                client_code += f'        return self.{endpoint.method.value.lower()}(endpoint'
            
            # Add data parameter
            if endpoint.request_body:
                client_code += ", data=data"
            
            client_code += ")\n"
        
        return {"client.py": client_code}
    
    def _generate_javascript_client(self, api_spec: APISpec) -> Dict[str, str]:
        """Generate JavaScript API client."""
        # Implementation for JavaScript client
        return {"client.js": "// JavaScript client implementation"}
    
    def _generate_typescript_client(self, api_spec: APISpec) -> Dict[str, str]:
        """Generate TypeScript API client."""
        # Implementation for TypeScript client
        return {"client.ts": "// TypeScript client implementation"}


def generate_complete_documentation(
    api_spec: APISpec, 
    output_formats: List[str] = None
) -> Dict[str, Any]:
    """Generate complete API documentation in multiple formats."""
    if output_formats is None:
        output_formats = ["openapi", "swagger", "redoc", "postman", "guide"]
    
    generator = APIDocumentationGenerator()
    documentation = {}
    
    if "openapi" in output_formats:
        openapi_spec = generator.generate_openapi_specification(api_spec)
        documentation["openapi.json"] = json.dumps(openapi_spec, indent=2)
        documentation["openapi.yaml"] = yaml.dump(openapi_spec, default_flow_style=False)
    
    if "swagger" in output_formats:
        openapi_spec = generator.generate_openapi_specification(api_spec)
        documentation["swagger-ui.html"] = generator.generate_swagger_ui_html(api_spec, openapi_spec)
    
    if "redoc" in output_formats:
        openapi_spec = generator.generate_openapi_specification(api_spec)
        documentation["redoc.html"] = generator.generate_redoc_html(api_spec, openapi_spec)
    
    if "postman" in output_formats:
        postman_collection = generator.generate_postman_collection(api_spec)
        documentation["postman-collection.json"] = json.dumps(postman_collection, indent=2)
    
    if "guide" in output_formats:
        documentation["developer-guide.md"] = generator.generate_developer_guide(api_spec)
    
    return documentation