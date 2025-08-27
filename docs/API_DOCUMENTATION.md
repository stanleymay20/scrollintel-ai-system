# Advanced Analytics Dashboard API Documentation

## Overview

The Advanced Analytics Dashboard API provides comprehensive access to executive analytics, ROI tracking, business intelligence, and automated insight generation. This RESTful API with GraphQL support enables developers to integrate powerful analytics capabilities into their applications.

## Base URL

```
Production: https://api.scrollintel.com/v1
Staging: https://staging-api.scrollintel.com/v1
Development: http://localhost:8000/api/v1
```

## Authentication

The API uses API key authentication. Include your API key in the `Authorization` header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Getting an API Key

1. Sign up for a ScrollIntel account
2. Navigate to Settings > API Keys
3. Click "Generate New API Key"
4. Copy and securely store your API key

### API Key Tiers

| Tier | Rate Limit | Features |
|------|------------|----------|
| **Free** | 100 requests/minute | Basic dashboards, insights |
| **Premium** | 1,000 requests/minute | Advanced analytics, ROI tracking |
| **Enterprise** | 10,000 requests/minute | Full feature access, webhooks |

## Rate Limiting

API requests are rate-limited based on your subscription tier. Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

When rate limits are exceeded, the API returns a `429 Too Many Requests` status with retry information.

## Response Format

All API responses follow a consistent JSON format:

### Success Response
```json
{
  "data": { ... },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid dashboard configuration",
    "details": {
      "field": "name",
      "issue": "Name cannot be empty"
    }
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## Endpoints

### Dashboards

#### List Dashboards
```http
GET /dashboards
```

**Parameters:**
- `type` (optional): Filter by dashboard type (`EXECUTIVE`, `DEPARTMENT`, `PROJECT`, `CUSTOM`)
- `owner` (optional): Filter by owner ID
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 20, max: 100)
- `sort` (optional): Sort field (`name`, `created_at`, `updated_at`)
- `order` (optional): Sort order (`asc`, `desc`)

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "https://api.scrollintel.com/v1/dashboards?type=EXECUTIVE&limit=10"
```

**Example Response:**
```json
{
  "data": {
    "dashboards": [
      {
        "id": "dash_123456789",
        "name": "Executive Overview",
        "type": "EXECUTIVE",
        "owner": "user_123456789",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T15:45:00Z",
        "widget_count": 8,
        "is_shared": true,
        "last_viewed": "2024-01-15T16:20:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 25,
      "pages": 3
    }
  }
}
```

#### Get Dashboard
```http
GET /dashboards/{dashboard_id}
```

**Example Response:**
```json
{
  "data": {
    "id": "dash_123456789",
    "name": "Executive Overview",
    "type": "EXECUTIVE",
    "owner": "user_123456789",
    "config": {
      "theme": "dark",
      "auto_refresh": 300,
      "timezone": "UTC"
    },
    "widgets": [
      {
        "id": "widget_123456789",
        "type": "CHART",
        "title": "Revenue Trend",
        "position": {"x": 0, "y": 0, "width": 6, "height": 4},
        "config": {
          "chart_type": "line",
          "data_source": "revenue_data",
          "time_range": "30d"
        }
      }
    ],
    "permissions": [
      {
        "user_id": "user_987654321",
        "role": "VIEWER",
        "granted_at": "2024-01-15T10:30:00Z"
      }
    ]
  }
}
```

#### Create Dashboard
```http
POST /dashboards
```

**Request Body:**
```json
{
  "name": "New Dashboard",
  "type": "EXECUTIVE",
  "description": "Executive performance dashboard",
  "config": {
    "theme": "light",
    "auto_refresh": 300
  },
  "template_id": "template_123456789"
}
```

#### Update Dashboard
```http
PUT /dashboards/{dashboard_id}
```

#### Delete Dashboard
```http
DELETE /dashboards/{dashboard_id}
```

### Widgets

#### Add Widget to Dashboard
```http
POST /dashboards/{dashboard_id}/widgets
```

**Request Body:**
```json
{
  "type": "CHART",
  "title": "Sales Performance",
  "position": {"x": 0, "y": 0, "width": 6, "height": 4},
  "config": {
    "chart_type": "bar",
    "data_source": "sales_data",
    "metrics": ["revenue", "units_sold"],
    "time_range": "7d"
  }
}
```

#### Update Widget
```http
PUT /dashboards/{dashboard_id}/widgets/{widget_id}
```

#### Remove Widget
```http
DELETE /dashboards/{dashboard_id}/widgets/{widget_id}
```

### ROI Analysis

#### List ROI Analyses
```http
GET /roi/analyses
```

**Parameters:**
- `project_id` (optional): Filter by project ID
- `date_range` (optional): Date range filter (`7d`, `30d`, `90d`, `1y`, or custom `YYYY-MM-DD/YYYY-MM-DD`)
- `min_roi` (optional): Minimum ROI percentage
- `page`, `limit`, `sort`, `order`: Pagination and sorting

**Example Response:**
```json
{
  "data": {
    "analyses": [
      {
        "id": "roi_123456789",
        "project_id": "proj_123456789",
        "project_name": "AI Implementation",
        "total_investment": 500000,
        "total_benefits": 750000,
        "roi_percentage": 50.0,
        "payback_period": 18,
        "npv": 250000,
        "irr": 0.25,
        "analysis_date": "2024-01-15T10:30:00Z",
        "status": "COMPLETED"
      }
    ]
  }
}
```

#### Create ROI Analysis
```http
POST /roi/analyses
```

**Request Body:**
```json
{
  "project_id": "proj_123456789",
  "project_name": "AI Implementation",
  "timeframe_months": 24,
  "costs": {
    "initial_investment": 300000,
    "operational_costs": 200000,
    "training_costs": 50000
  },
  "benefits": {
    "cost_savings": 400000,
    "revenue_increase": 350000,
    "efficiency_gains": 100000
  }
}
```

#### Get ROI Analysis
```http
GET /roi/analyses/{analysis_id}
```

### Insights

#### List Insights
```http
GET /insights
```

**Parameters:**
- `type` (optional): Filter by insight type (`TREND`, `ANOMALY`, `CORRELATION`, `PREDICTION`, `RECOMMENDATION`)
- `significance` (optional): Minimum significance score (0.0-1.0)
- `dashboard_id` (optional): Filter by dashboard
- `date_range` (optional): Date range filter

**Example Response:**
```json
{
  "data": {
    "insights": [
      {
        "id": "insight_123456789",
        "type": "ANOMALY",
        "title": "Unusual Revenue Spike Detected",
        "description": "Revenue increased by 45% compared to the same period last month, significantly above the normal variance of 8%.",
        "significance": 0.92,
        "confidence": 0.87,
        "dashboard_id": "dash_123456789",
        "widget_id": "widget_123456789",
        "created_at": "2024-01-15T10:30:00Z",
        "recommendations": [
          "Investigate the cause of the revenue spike",
          "Analyze customer acquisition channels",
          "Review marketing campaign performance"
        ],
        "business_impact": {
          "category": "REVENUE",
          "magnitude": 0.45,
          "affected_metrics": ["monthly_revenue", "customer_acquisition"]
        }
      }
    ]
  }
}
```

#### Get Insight
```http
GET /insights/{insight_id}
```

### Predictive Analytics

#### List Forecasts
```http
GET /forecasts
```

**Example Response:**
```json
{
  "data": {
    "forecasts": [
      {
        "id": "forecast_123456789",
        "metric": "monthly_revenue",
        "horizon": 12,
        "model": "ARIMA",
        "accuracy": 0.89,
        "confidence": 0.85,
        "generated_at": "2024-01-15T10:30:00Z",
        "predictions": [
          {
            "period": "2024-02",
            "value": 125000,
            "lower_bound": 115000,
            "upper_bound": 135000,
            "confidence": 0.85
          }
        ]
      }
    ]
  }
}
```

#### Create Forecast
```http
POST /forecasts
```

**Request Body:**
```json
{
  "metric": "monthly_revenue",
  "horizon": 12,
  "model": "ARIMA",
  "data_source": "revenue_data",
  "historical_periods": 24
}
```

### Data Sources

#### List Data Sources
```http
GET /data-sources
```

**Example Response:**
```json
{
  "data": {
    "sources": [
      {
        "id": "source_123456789",
        "name": "Salesforce CRM",
        "type": "CRM",
        "status": "CONNECTED",
        "last_sync": "2024-01-15T10:30:00Z",
        "record_count": 15420,
        "sync_frequency": "hourly",
        "data_quality": 0.95
      }
    ]
  }
}
```

#### Create Data Source
```http
POST /data-sources
```

#### Test Data Source Connection
```http
POST /data-sources/{source_id}/test
```

### Templates

#### List Dashboard Templates
```http
GET /templates
```

**Parameters:**
- `category` (optional): Filter by category (`EXECUTIVE`, `FINANCE`, `SALES`, `MARKETING`, `OPERATIONS`)
- `industry` (optional): Filter by industry
- `popularity` (optional): Sort by popularity

**Example Response:**
```json
{
  "data": {
    "templates": [
      {
        "id": "template_123456789",
        "name": "Executive KPI Dashboard",
        "description": "Comprehensive executive dashboard with key performance indicators",
        "category": "EXECUTIVE",
        "industry": "Technology",
        "popularity": 95,
        "rating": 4.8,
        "widget_count": 12,
        "preview_url": "https://cdn.scrollintel.com/templates/preview/template_123456789.png"
      }
    ]
  }
}
```

### Reports

#### Generate Report
```http
POST /reports/generate
```

**Request Body:**
```json
{
  "title": "Monthly Executive Report",
  "type": "EXECUTIVE_SUMMARY",
  "format": "PDF",
  "dashboard_id": "dash_123456789",
  "date_range": "30d",
  "sections": ["overview", "kpis", "insights", "recommendations"]
}
```

#### Schedule Report
```http
POST /reports/schedule
```

**Request Body:**
```json
{
  "name": "Weekly Executive Report",
  "report_config": {
    "title": "Weekly Executive Report",
    "type": "EXECUTIVE_SUMMARY",
    "format": "PDF",
    "dashboard_id": "dash_123456789"
  },
  "schedule": {
    "frequency": "WEEKLY",
    "day_of_week": 1,
    "time": "09:00",
    "timezone": "UTC"
  },
  "delivery": {
    "method": "EMAIL",
    "recipients": ["ceo@company.com", "cfo@company.com"]
  }
}
```

### Webhooks

#### Register Webhook
```http
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/scrollintel",
  "events": [
    "dashboard.created",
    "insight.generated",
    "alert.triggered"
  ],
  "secret": "your-webhook-secret",
  "headers": {
    "X-Custom-Header": "value"
  }
}
```

#### List Webhooks
```http
GET /webhooks
```

#### Test Webhook
```http
POST /webhooks/{webhook_id}/test
```

## GraphQL API

The API also supports GraphQL for flexible data querying.

### GraphQL Endpoint
```
POST /graphql
```

### GraphQL Playground
```
GET /graphql/playground
```

### Example GraphQL Query
```graphql
query GetDashboardWithInsights($dashboardId: ID!) {
  dashboard(id: $dashboardId) {
    id
    name
    type
    widgets {
      id
      title
      type
      config
    }
    metrics {
      totalViews
      uniqueUsers
      avgSessionDuration
    }
  }
  
  insights(dashboardId: $dashboardId, limit: 5) {
    id
    type
    title
    significance
    confidence
    recommendations
  }
}
```

### Example GraphQL Mutation
```graphql
mutation CreateDashboard($input: DashboardInput!) {
  createDashboard(input: $input) {
    id
    name
    type
    createdAt
  }
}
```

## WebSocket API

Real-time updates are available via WebSocket connections.

### Dashboard Updates
```
wss://api.scrollintel.com/ws/dashboard/{dashboard_id}
```

### Insights Stream
```
wss://api.scrollintel.com/ws/insights
```

### Alerts Stream
```
wss://api.scrollintel.com/ws/alerts
```

### Example WebSocket Usage
```javascript
const ws = new WebSocket('wss://api.scrollintel.com/ws/dashboard/dash_123456789');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Dashboard update:', update);
};
```

## Error Codes

| Code | Description |
|------|-------------|
| `400` | Bad Request - Invalid request format |
| `401` | Unauthorized - Invalid or missing API key |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource not found |
| `422` | Unprocessable Entity - Validation error |
| `429` | Too Many Requests - Rate limit exceeded |
| `500` | Internal Server Error - Server error |

## SDKs and Libraries

### JavaScript/Node.js
```bash
npm install @scrollintel/analytics-api
```

```javascript
import { ScrollIntelAPI } from '@scrollintel/analytics-api';

const api = new ScrollIntelAPI('YOUR_API_KEY');

// List dashboards
const dashboards = await api.dashboards.list();

// Create dashboard
const dashboard = await api.dashboards.create({
  name: 'My Dashboard',
  type: 'EXECUTIVE'
});
```

### Python
```bash
pip install scrollintel-analytics
```

```python
from scrollintel import AnalyticsAPI

api = AnalyticsAPI('YOUR_API_KEY')

# List dashboards
dashboards = api.dashboards.list()

# Create dashboard
dashboard = api.dashboards.create(
    name='My Dashboard',
    type='EXECUTIVE'
)
```

### cURL Examples

#### Create Dashboard
```bash
curl -X POST https://api.scrollintel.com/v1/dashboards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Executive Dashboard",
    "type": "EXECUTIVE",
    "description": "Main executive overview"
  }'
```

#### Get Insights
```bash
curl -X GET "https://api.scrollintel.com/v1/insights?type=ANOMALY&limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Best Practices

### 1. API Key Security
- Never expose API keys in client-side code
- Use environment variables to store API keys
- Rotate API keys regularly
- Use different keys for different environments

### 2. Rate Limiting
- Implement exponential backoff for retries
- Cache responses when appropriate
- Use webhooks instead of polling for real-time updates

### 3. Error Handling
- Always check response status codes
- Implement proper error handling for all API calls
- Log errors for debugging

### 4. Performance
- Use pagination for large datasets
- Implement request timeouts
- Use GraphQL for complex queries to reduce over-fetching

### 5. Webhooks
- Verify webhook signatures for security
- Implement idempotency for webhook handlers
- Use HTTPS endpoints for webhook URLs

## Support

- **Documentation**: https://docs.scrollintel.com
- **API Status**: https://status.scrollintel.com
- **Support Email**: api-support@scrollintel.com
- **Community Forum**: https://community.scrollintel.com

## Changelog

### v1.0.0 (2024-01-15)
- Initial API release
- Dashboard management endpoints
- ROI analysis functionality
- Insight generation
- Predictive analytics
- GraphQL support
- WebSocket real-time updates
- Webhook system