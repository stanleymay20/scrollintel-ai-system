# ScrollIntel Prompt Management API Documentation

## Overview

The ScrollIntel Prompt Management API provides comprehensive functionality for managing prompt templates, including version control, A/B testing, optimization, and real-time notifications through webhooks.

## Base URL

```
https://api.scrollintel.com/api/v1/prompts
```

## Authentication

All API requests require authentication using an API key:

```http
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **User Rate Limit**: 100 requests per minute (burst: 150)
- **IP Rate Limit**: 60 requests per minute
- **Global Rate Limit**: 1000 requests per minute
- **Concurrent Requests**: 10 per user

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## API Versioning

The API uses URL-based versioning. Current version is `v1`. Backward compatibility is maintained for at least one major version.

## Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0",
  "request_id": "req_123456789",
  "rate_limit": {
    "user_requests_per_minute": {
      "current": 5,
      "limit": 100,
      "window_seconds": 60,
      "type": "requests_per_minute",
      "reset_time": 1640995200
    }
  }
}
```

## Error Handling

Error responses include detailed information:

```json
{
  "success": false,
  "message": "Validation error",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789",
  "errors": [
    {
      "field": "name",
      "message": "Name is required"
    }
  ]
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (validation error)
- `401` - Unauthorized (invalid API key)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

## Endpoints

### Prompt Management

#### Create Prompt

```http
POST /api/v1/prompts/
```

Create a new prompt template.

**Request Body:**

```json
{
  "name": "Customer Support Greeting",
  "content": "Hello {{customer_name}}, how can I help you today?",
  "category": "customer_support",
  "tags": ["greeting", "support"],
  "variables": [
    {
      "name": "customer_name",
      "type": "string",
      "required": true,
      "description": "Customer's name"
    }
  ],
  "description": "Standard greeting for customer support"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "prompt_abc123"
  },
  "message": "Prompt created successfully"
}
```

#### Get Prompt

```http
GET /api/v1/prompts/{prompt_id}
```

Retrieve a prompt template by ID.

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "prompt_abc123",
    "name": "Customer Support Greeting",
    "content": "Hello {{customer_name}}, how can I help you today?",
    "category": "customer_support",
    "tags": ["greeting", "support"],
    "variables": [
      {
        "name": "customer_name",
        "type": "string",
        "required": true,
        "description": "Customer's name"
      }
    ],
    "description": "Standard greeting for customer support",
    "is_active": true,
    "created_by": "user_123",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

#### Update Prompt

```http
PUT /api/v1/prompts/{prompt_id}
```

Update a prompt template and create a new version.

**Request Body:**

```json
{
  "name": "Updated Customer Support Greeting",
  "content": "Hello {{customer_name}}, welcome back! How can I assist you?",
  "changes_description": "Added welcome back message"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "version_def456",
    "prompt_id": "prompt_abc123",
    "version": "1.0.1",
    "content": "Hello {{customer_name}}, welcome back! How can I assist you?",
    "changes": "Added welcome back message",
    "created_by": "user_123",
    "created_at": "2024-01-01T12:30:00Z"
  }
}
```

#### Delete Prompt

```http
DELETE /api/v1/prompts/{prompt_id}
```

Soft delete (deactivate) a prompt template.

**Response:**

```json
{
  "success": true,
  "message": "Prompt deleted successfully"
}
```

#### Search Prompts

```http
POST /api/v1/prompts/search
```

Search prompt templates with filters and pagination.

**Request Body:**

```json
{
  "text": "customer support",
  "category": "customer_support",
  "tags": ["greeting"],
  "created_by": "user_123",
  "date_from": "2024-01-01T00:00:00Z",
  "date_to": "2024-01-31T23:59:59Z",
  "limit": 20,
  "offset": 0
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "prompt_abc123",
        "name": "Customer Support Greeting",
        "content": "Hello {{customer_name}}...",
        "category": "customer_support",
        "tags": ["greeting", "support"],
        "created_at": "2024-01-01T12:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "has_next": false,
    "has_previous": false
  }
}
```

#### List Prompts

```http
GET /api/v1/prompts/?page=1&page_size=50&category=customer_support&tags=greeting,support
```

List prompts with pagination and filtering.

#### Get Prompt History

```http
GET /api/v1/prompts/{prompt_id}/history
```

Get version history for a prompt template.

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "version_def456",
      "prompt_id": "prompt_abc123",
      "version": "1.0.1",
      "content": "Updated content...",
      "changes": "Added welcome back message",
      "created_by": "user_123",
      "created_at": "2024-01-01T12:30:00Z"
    },
    {
      "id": "version_ghi789",
      "prompt_id": "prompt_abc123",
      "version": "1.0.0",
      "content": "Original content...",
      "changes": "Initial version",
      "created_by": "user_123",
      "created_at": "2024-01-01T12:00:00Z"
    }
  ]
}
```

#### Get Prompt Metrics

```http
GET /api/v1/prompts/{prompt_id}/metrics
```

Get usage metrics for a prompt template.

**Response:**

```json
{
  "success": true,
  "data": {
    "prompt_id": "prompt_abc123",
    "total_uses": 1250,
    "unique_users": 45,
    "avg_response_time": 0.25,
    "success_rate": 0.98,
    "last_used": "2024-01-01T11:45:00Z"
  }
}
```

#### Batch Operations

```http
POST /api/v1/prompts/batch
```

Perform batch operations on multiple prompts.

**Request Body:**

```json
[
  {
    "type": "update",
    "prompt_id": "prompt_abc123",
    "changes": {
      "name": "Updated Name",
      "changes_description": "Batch update"
    }
  },
  {
    "type": "delete",
    "prompt_id": "prompt_def456"
  }
]
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "operation": 0,
        "type": "update",
        "prompt_id": "prompt_abc123",
        "version": "1.0.2"
      },
      {
        "operation": 1,
        "type": "delete",
        "prompt_id": "prompt_def456",
        "success": true
      }
    ],
    "errors": []
  },
  "message": "Batch operation completed. 2 successful, 0 errors"
}
```

### Webhook Management

#### Register Webhook

```http
POST /api/v1/prompts/webhooks
```

Register a new webhook endpoint.

**Request Body:**

```json
{
  "url": "https://your-app.com/webhooks/prompts",
  "events": ["prompt.created", "prompt.updated", "prompt.deleted"],
  "secret": "your-webhook-secret",
  "active": true
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "webhook_id": "webhook_xyz789"
  },
  "message": "Webhook registered successfully"
}
```

#### List Webhooks

```http
GET /api/v1/prompts/webhooks
```

List all webhook endpoints for the user.

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "webhook_xyz789",
      "name": "webhook_1640995200",
      "url": "https://your-app.com/webhooks/prompts",
      "events": ["prompt.created", "prompt.updated"],
      "active": true,
      "created_at": "2024-01-01T12:00:00Z",
      "last_success": "2024-01-01T11:45:00Z",
      "failure_count": 0
    }
  ]
}
```

#### Update Webhook

```http
PUT /api/v1/prompts/webhooks/{webhook_id}
```

Update a webhook endpoint.

**Request Body:**

```json
{
  "url": "https://your-app.com/new-webhooks/prompts",
  "events": ["prompt.created"],
  "active": false
}
```

#### Delete Webhook

```http
DELETE /api/v1/prompts/webhooks/{webhook_id}
```

Delete a webhook endpoint.

#### Test Webhook

```http
POST /api/v1/prompts/webhooks/{webhook_id}/test
```

Test a webhook endpoint with a test event.

**Response:**

```json
{
  "success": true,
  "data": {
    "success": true,
    "status": "delivered",
    "response_status": 200,
    "delivery_id": "delivery_123"
  }
}
```

### Usage Analytics

#### Get Usage Summary

```http
GET /api/v1/prompts/usage/summary?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z
```

Get usage summary for the current user.

**Response:**

```json
{
  "success": true,
  "data": {
    "total_requests": 5420,
    "total_tokens": 125000,
    "total_errors": 23,
    "avg_response_time": 0.35,
    "error_rate": 0.004,
    "endpoints": {
      "/api/v1/prompts/": 3200,
      "/api/v1/prompts/search": 1800,
      "/api/v1/prompts/{id}": 420
    },
    "status_codes": {
      "200": 5397,
      "400": 15,
      "404": 8
    },
    "hourly_distribution": {
      "2024-01-01 09:00": 45,
      "2024-01-01 10:00": 67,
      "2024-01-01 11:00": 89
    },
    "first_request": "2024-01-01T08:30:00Z",
    "last_request": "2024-01-31T17:45:00Z"
  }
}
```

## Webhook Events

When you register webhooks, you'll receive HTTP POST requests for subscribed events:

### Event Types

- `prompt.created` - New prompt template created
- `prompt.updated` - Prompt template updated
- `prompt.deleted` - Prompt template deleted
- `prompt.version.created` - New version created

### Webhook Payload

```json
{
  "event": {
    "id": "event_123456789",
    "event_type": "prompt.created",
    "resource_type": "prompt",
    "resource_id": "prompt_abc123",
    "action": "create",
    "timestamp": "2024-01-01T12:00:00Z",
    "user_id": "user_123",
    "data": {
      "name": "Customer Support Greeting",
      "category": "customer_support"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "delivery_id": "delivery_def456"
}
```

### Webhook Security

Webhooks include a signature header for verification:

```http
X-ScrollIntel-Signature: sha256=abc123def456...
```

To verify the signature:

```python
import hmac
import hashlib

def verify_webhook_signature(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(
        f"sha256={expected_signature}",
        signature
    )
```

## Python SDK

### Installation

```bash
pip install scrollintel-sdk
```

### Quick Start

```python
from scrollintel.sdk import PromptClient
from scrollintel.sdk.models import PromptVariable, SearchQuery

# Initialize client
client = PromptClient(
    base_url="https://api.scrollintel.com",
    api_key="your-api-key"
)

# Create a prompt
variables = [
    PromptVariable(
        name="customer_name",
        type="string",
        required=True,
        description="Customer's name"
    )
]

prompt_id = client.create_prompt(
    name="Customer Greeting",
    content="Hello {{customer_name}}, how can I help?",
    category="support",
    tags=["greeting"],
    variables=variables
)

# Get a prompt
prompt = client.get_prompt(prompt_id)
print(f"Prompt: {prompt.name}")

# Search prompts
query = SearchQuery(
    text="customer",
    category="support",
    limit=10
)

results = client.search_prompts(query)
print(f"Found {results.total} prompts")

# Register webhook
webhook_id = client.register_webhook(
    url="https://your-app.com/webhook",
    events=["prompt.created", "prompt.updated"],
    secret="your-secret"
)

# Get usage summary
summary = client.get_usage_summary()
print(f"Total requests: {summary['total_requests']}")
```

### Error Handling

```python
from scrollintel.sdk.exceptions import (
    RateLimitError, ValidationError, NotFoundError
)

try:
    prompt = client.get_prompt("invalid-id")
except NotFoundError:
    print("Prompt not found")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ValidationError as e:
    print(f"Validation error: {e.errors}")
```

### Context Manager

```python
with PromptClient(base_url="...", api_key="...") as client:
    prompt = client.get_prompt("prompt-id")
    # Client automatically closed when exiting context
```

## Best Practices

### Rate Limiting

- Implement exponential backoff for rate limit errors
- Cache frequently accessed prompts
- Use batch operations when possible

### Webhooks

- Always verify webhook signatures
- Implement idempotency for webhook handlers
- Use HTTPS endpoints only
- Handle webhook failures gracefully

### Error Handling

- Always check the `success` field in responses
- Implement proper retry logic for transient errors
- Log request IDs for debugging

### Security

- Keep API keys secure and rotate regularly
- Use webhook secrets for signature verification
- Implement proper access controls

## Support

For API support, contact: api-support@scrollintel.com

## Changelog

### v1.0.0 (2024-01-01)
- Initial API release
- Prompt CRUD operations
- Search and filtering
- Webhook support
- Rate limiting
- Usage analytics