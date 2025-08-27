"""
Comprehensive API documentation for Advanced Analytics Dashboard System.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def create_api_documentation() -> Dict[str, Any]:
    """Create comprehensive OpenAPI documentation."""
    
    return {
        "openapi": "3.0.2",
        "info": {
            "title": "Advanced Analytics Dashboard API",
            "description": """
# Advanced Analytics Dashboard System API

A comprehensive REST and GraphQL API for managing executive analytics dashboards, ROI tracking, 
business intelligence consolidation, and automated insight generation.

## Features

- **Executive Dashboards**: Role-specific dashboards with real-time updates
- **ROI Tracking**: Comprehensive ROI calculation and financial impact analysis
- **AI Insights**: Automated insight generation with natural language explanations
- **Predictive Analytics**: Forecasting and predictive modeling for business metrics
- **Multi-source Integration**: Connect to ERP, CRM, BI tools, and cloud platforms
- **Real-time Updates**: WebSocket connections for live dashboard updates
- **Webhook System**: External integrations and event notifications
- **GraphQL Support**: Flexible data querying capabilities

## Authentication

The API supports two authentication methods:

### JWT Bearer Token
```
Authorization: Bearer <jwt_token>
```

### API Key
```
X-API-Key: <api_key>
```

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Standard endpoints**: 60 requests per minute per user
- **Resource-intensive endpoints**: 10-30 requests per minute per user
- **Webhook endpoints**: 5-10 requests per minute per user

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Rate Limit Exceeded
- `500`: Internal Server Error

Error responses include detailed error information:

```json
{
    "detail": "Error description",
    "error_code": "SPECIFIC_ERROR_CODE",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## Webhooks

The system supports outgoing webhooks for real-time event notifications:

### Event Types
- `dashboard.created`: New dashboard created
- `dashboard.updated`: Dashboard configuration updated
- `dashboard.deleted`: Dashboard deleted
- `metric.updated`: Dashboard metrics updated
- `insight.generated`: New AI insight generated
- `roi.calculated`: ROI analysis completed
- `forecast.created`: New forecast generated
- `data_source.connected`: Data source connected
- `data_source.disconnected`: Data source disconnected
- `alert.triggered`: Alert threshold breached

### Webhook Security
All webhook payloads are signed with HMAC-SHA256:

```
X-Webhook-Signature: sha256=<signature>
```

Verify signatures using your webhook secret:

```python
import hmac
import hashlib

def verify_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## GraphQL

The API includes a GraphQL endpoint at `/graphql` for flexible data querying:

### Query Examples

```graphql
# Get dashboards with metrics
query {
  dashboards(filter: {role: "CTO"}) {
    id
    name
    metrics {
      name
      value
      unit
    }
  }
}

# Get ROI analysis with breakdown
query {
  roiAnalyses(filter: {minRoi: 15.0}) {
    projectId
    roiPercentage
    costBreakdown {
      category
      amount
    }
  }
}
```

### Subscriptions

Real-time updates via GraphQL subscriptions:

```graphql
subscription {
  dashboardUpdates(dashboardId: "dash_123") {
    id
    name
    updatedAt
  }
}
```

## SDK and Examples

### Python SDK

```python
from scrollintel_sdk import AnalyticsClient

client = AnalyticsClient(api_key="your_api_key")

# Create dashboard
dashboard = client.dashboards.create(
    name="Executive Dashboard",
    role="CTO",
    template_id="executive_template"
)

# Add metrics
client.dashboards.add_metrics(dashboard.id, [
    {
        "name": "Technology ROI",
        "category": "financial",
        "value": 18.5,
        "unit": "percentage"
    }
])

# Generate insights
insights = client.insights.generate(
    data_sources=["erp_system", "crm_system"],
    analysis_type="comprehensive"
)
```

### JavaScript SDK

```javascript
import { AnalyticsClient } from '@scrollintel/analytics-sdk';

const client = new AnalyticsClient({
    apiKey: 'your_api_key',
    baseUrl: 'https://api.scrollintel.com'
});

// Create dashboard
const dashboard = await client.dashboards.create({
    name: 'Executive Dashboard',
    role: 'CTO',
    templateId: 'executive_template'
});

// Subscribe to real-time updates
client.dashboards.subscribe(dashboard.id, (update) => {
    console.log('Dashboard updated:', update);
});
```

## Best Practices

### Performance
- Use pagination for large datasets (`limit` and `offset` parameters)
- Cache frequently accessed data
- Use GraphQL for complex queries to reduce over-fetching
- Implement client-side rate limiting

### Security
- Store API keys securely (environment variables, key management systems)
- Verify webhook signatures
- Use HTTPS for all API calls
- Implement proper error handling

### Monitoring
- Monitor API usage and rate limits
- Set up webhook delivery monitoring
- Track dashboard performance metrics
- Implement proper logging

## Support

For API support and questions:
- Documentation: https://docs.scrollintel.com/api
- Support: support@scrollintel.com
- Status Page: https://status.scrollintel.com
            """,
            "version": "1.0.0",
            "contact": {
                "name": "ScrollIntel API Support",
                "email": "api-support@scrollintel.com",
                "url": "https://docs.scrollintel.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "https://api.scrollintel.com",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.scrollintel.com",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token authentication"
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key authentication"
                }
            },
            "schemas": {
                "Dashboard": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique dashboard identifier"},
                        "name": {"type": "string", "description": "Dashboard name"},
                        "role": {"type": "string", "description": "Executive role (CTO, CFO, CEO, etc.)"},
                        "type": {"type": "string", "description": "Dashboard type"},
                        "owner_id": {"type": "string", "description": "Owner user ID"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "updated_at": {"type": "string", "format": "date-time"},
                        "widget_count": {"type": "integer", "description": "Number of widgets"}
                    },
                    "required": ["id", "name", "role", "type", "owner_id"]
                },
                "Metric": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string", "description": "Metric name"},
                        "category": {"type": "string", "description": "Metric category"},
                        "value": {"type": "number", "description": "Metric value"},
                        "unit": {"type": "string", "description": "Unit of measurement"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "source": {"type": "string", "description": "Data source"}
                    },
                    "required": ["name", "category", "value", "unit", "source"]
                },
                "ROIAnalysis": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "roi_percentage": {"type": "number", "description": "ROI as percentage"},
                        "total_investment": {"type": "number", "description": "Total investment amount"},
                        "total_benefits": {"type": "number", "description": "Total benefits amount"},
                        "payback_period": {"type": "integer", "description": "Payback period in months"},
                        "npv": {"type": "number", "description": "Net Present Value"},
                        "irr": {"type": "number", "description": "Internal Rate of Return"},
                        "analysis_date": {"type": "string", "format": "date-time"}
                    },
                    "required": ["project_id", "roi_percentage", "total_investment", "total_benefits"]
                },
                "Insight": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string", "description": "Insight title"},
                        "description": {"type": "string", "description": "Detailed description"},
                        "type": {"type": "string", "description": "Insight type"},
                        "significance": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "created_at": {"type": "string", "format": "date-time"}
                    },
                    "required": ["title", "description", "type", "significance", "confidence"]
                },
                "Forecast": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "predictions": {"type": "array", "items": {"type": "object"}},
                        "confidence_intervals": {"type": "array", "items": {"type": "object"}},
                        "model_accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                        "created_at": {"type": "string", "format": "date-time"}
                    },
                    "required": ["metric_name", "predictions", "model_accuracy"]
                },
                "WebhookEndpoint": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "url": {"type": "string", "format": "uri"},
                        "events": {"type": "array", "items": {"type": "string"}},
                        "active": {"type": "boolean"},
                        "retry_count": {"type": "integer", "minimum": 1, "maximum": 10},
                        "timeout": {"type": "integer", "minimum": 5, "maximum": 300},
                        "created_at": {"type": "string", "format": "date-time"}
                    },
                    "required": ["url", "events"]
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "description": "Error description"},
                        "error_code": {"type": "string", "description": "Specific error code"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    },
                    "required": ["detail"]
                }
            },
            "responses": {
                "UnauthorizedError": {
                    "description": "Authentication required",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    }
                },
                "ForbiddenError": {
                    "description": "Insufficient permissions",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    }
                },
                "NotFoundError": {
                    "description": "Resource not found",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    }
                },
                "RateLimitError": {
                    "description": "Rate limit exceeded",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    },
                    "headers": {
                        "X-RateLimit-Limit": {
                            "description": "Maximum requests allowed",
                            "schema": {"type": "integer"}
                        },
                        "X-RateLimit-Remaining": {
                            "description": "Remaining requests in current window",
                            "schema": {"type": "integer"}
                        },
                        "X-RateLimit-Reset": {
                            "description": "Unix timestamp when limit resets",
                            "schema": {"type": "integer"}
                        },
                        "Retry-After": {
                            "description": "Seconds to wait before retrying",
                            "schema": {"type": "integer"}
                        }
                    }
                }
            },
            "examples": {
                "DashboardExample": {
                    "summary": "Executive Dashboard Example",
                    "value": {
                        "id": "dash_cto_001",
                        "name": "CTO Executive Dashboard",
                        "role": "CTO",
                        "type": "executive",
                        "owner_id": "user_123",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T12:00:00Z",
                        "widget_count": 8
                    }
                },
                "MetricExample": {
                    "summary": "Technology ROI Metric",
                    "value": {
                        "id": "metric_roi_001",
                        "name": "Technology ROI",
                        "category": "financial",
                        "value": 18.5,
                        "unit": "percentage",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "source": "erp_system"
                    }
                },
                "WebhookExample": {
                    "summary": "Dashboard Webhook",
                    "value": {
                        "id": "wh_001",
                        "url": "https://your-app.com/webhooks/dashboard",
                        "events": ["dashboard.created", "dashboard.updated", "metric.updated"],
                        "active": True,
                        "retry_count": 3,
                        "timeout": 30,
                        "created_at": "2024-01-01T00:00:00Z"
                    }
                }
            }
        },
        "security": [
            {"BearerAuth": []},
            {"ApiKeyAuth": []}
        ],
        "tags": [
            {
                "name": "dashboards",
                "description": "Dashboard management operations"
            },
            {
                "name": "metrics",
                "description": "Metrics and KPI operations"
            },
            {
                "name": "roi",
                "description": "ROI calculation and analysis"
            },
            {
                "name": "insights",
                "description": "AI insight generation"
            },
            {
                "name": "predictions",
                "description": "Predictive analytics and forecasting"
            },
            {
                "name": "data-sources",
                "description": "Data source integration"
            },
            {
                "name": "webhooks",
                "description": "Webhook management and events"
            },
            {
                "name": "templates",
                "description": "Dashboard template operations"
            }
        ]
    }


def customize_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Customize OpenAPI schema with additional documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    # Get base schema
    openapi_schema = get_openapi(
        title="Advanced Analytics Dashboard API",
        version="1.0.0",
        description="Comprehensive API for executive analytics and business intelligence",
        routes=app.routes,
    )
    
    # Add custom documentation
    custom_docs = create_api_documentation()
    
    # Merge schemas
    openapi_schema.update({
        "info": custom_docs["info"],
        "servers": custom_docs["servers"],
        "components": {
            **openapi_schema.get("components", {}),
            **custom_docs["components"]
        },
        "security": custom_docs["security"],
        "tags": custom_docs["tags"]
    })
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# API Examples and Code Snippets
API_EXAMPLES = {
    "python": {
        "create_dashboard": '''
import requests

# Create a new dashboard
response = requests.post(
    "https://api.scrollintel.com/api/v1/analytics/dashboards",
    headers={
        "Authorization": "Bearer your_jwt_token",
        "Content-Type": "application/json"
    },
    json={
        "name": "Executive Dashboard",
        "role": "CTO",
        "template_id": "executive_template",
        "config": {
            "theme": "dark",
            "auto_refresh": True,
            "refresh_interval": 300
        }
    }
)

dashboard = response.json()
print(f"Created dashboard: {dashboard['id']}")
        ''',
        "add_metrics": '''
# Add metrics to dashboard
metrics = [
    {
        "name": "Technology ROI",
        "category": "financial",
        "value": 18.5,
        "unit": "percentage",
        "source": "erp_system"
    },
    {
        "name": "System Uptime",
        "category": "operational",
        "value": 99.9,
        "unit": "percentage",
        "source": "monitoring_system"
    }
]

response = requests.post(
    f"https://api.scrollintel.com/api/v1/analytics/dashboards/{dashboard_id}/metrics",
    headers={"Authorization": "Bearer your_jwt_token"},
    json={"metrics": metrics}
)
        ''',
        "webhook_setup": '''
# Set up webhook endpoint
webhook_config = {
    "url": "https://your-app.com/webhooks/analytics",
    "events": [
        "dashboard.created",
        "dashboard.updated",
        "metric.updated",
        "insight.generated"
    ],
    "secret": "your_webhook_secret",
    "retry_count": 3,
    "timeout": 30
}

response = requests.post(
    "https://api.scrollintel.com/api/webhooks/endpoints",
    headers={"Authorization": "Bearer your_jwt_token"},
    json=webhook_config
)

webhook = response.json()
print(f"Webhook created: {webhook['id']}")
        '''
    },
    "javascript": {
        "create_dashboard": '''
// Create a new dashboard
const response = await fetch('https://api.scrollintel.com/api/v1/analytics/dashboards', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer your_jwt_token',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        name: 'Executive Dashboard',
        role: 'CTO',
        template_id: 'executive_template',
        config: {
            theme: 'dark',
            auto_refresh: true,
            refresh_interval: 300
        }
    })
});

const dashboard = await response.json();
console.log(`Created dashboard: ${dashboard.id}`);
        ''',
        "graphql_query": '''
// GraphQL query example
const query = `
    query GetDashboardData($dashboardId: String!) {
        dashboard(id: $dashboardId) {
            id
            name
            metrics {
                name
                value
                unit
                timestamp
            }
            widgets {
                id
                type
                title
                data
            }
        }
    }
`;

const response = await fetch('https://api.scrollintel.com/graphql', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer your_jwt_token',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        query,
        variables: { dashboardId: 'dash_123' }
    })
});

const data = await response.json();
console.log(data.data.dashboard);
        '''
    },
    "curl": {
        "create_dashboard": '''
# Create dashboard with curl
curl -X POST https://api.scrollintel.com/api/v1/analytics/dashboards \\
  -H "Authorization: Bearer your_jwt_token" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Executive Dashboard",
    "role": "CTO",
    "template_id": "executive_template"
  }'
        ''',
        "webhook_verification": '''
# Verify webhook signature
curl -X POST https://api.scrollintel.com/api/webhooks/verify-signature \\
  -H "Content-Type: application/json" \\
  -d '{
    "payload": "{\\"event_type\\":\\"dashboard.created\\"}",
    "signature": "sha256=abc123...",
    "secret": "your_webhook_secret"
  }'
        '''
    }
}