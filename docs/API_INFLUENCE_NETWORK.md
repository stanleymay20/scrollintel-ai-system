# Global Influence Network API Documentation

## Overview

The Global Influence Network API provides comprehensive endpoints for orchestrating worldwide influence campaigns, managing relationship networks, and coordinating strategic partnerships. This API enables superhuman influence capabilities that surpass individual human CTO reach.

## Base URL

```
https://api.scrollintel.com/api/global-influence
```

## Authentication

All API requests require authentication using JWT Bearer tokens.

### Getting an Access Token

```bash
POST /auth/token
Content-Type: application/json

{
  "user_id": "your_user_id",
  "permissions": ["influence_network_user"]
}
```

### Using the Token

```bash
Authorization: Bearer <your_jwt_token>
```

## Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|--------|
| Campaign Creation | 10 requests | 1 hour |
| Network Sync | 5 requests | 5 minutes |
| Analytics | 100 requests | 1 hour |
| Target Management | 50 requests | 1 hour |
| Default | 1000 requests | 1 hour |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Campaign Management

### Create Influence Campaign

Create and orchestrate a new global influence campaign.

```http
POST /campaigns/create
```

**Request Body:**
```json
{
  "objective": "Establish AI technology leadership in healthcare",
  "target_outcomes": [
    "Gain recognition as healthcare AI thought leader",
    "Build partnerships with 50+ medical institutions",
    "Influence healthcare AI policy in US, EU, and Asia"
  ],
  "timeline_days": 180,
  "priority": "critical",
  "target_domains": ["healthcare", "technology", "policy"],
  "scope": "global",
  "constraints": {
    "budget_limit": 2000000,
    "geographic_focus": ["US", "EU", "Asia"],
    "compliance_requirements": ["HIPAA", "GDPR", "FDA"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Influence campaign created and orchestrated successfully",
  "data": {
    "campaign_id": "campaign_20250822_143052",
    "orchestration_plan": {
      "phases": [
        {
          "name": "Strategic Analysis",
          "duration_days": 30,
          "activities": ["Market research", "Stakeholder mapping"]
        }
      ]
    },
    "execution_status": {
      "status": "planning",
      "next_phase": "Strategic Analysis"
    },
    "estimated_timeline": "180 days",
    "success_probability": 0.85
  }
}
```

### List Campaigns

Retrieve list of influence campaigns with optional filtering.

```http
GET /campaigns?status=active&priority=high&limit=20&offset=0
```

**Query Parameters:**
- `status` (optional): Filter by campaign status (planning, active, paused, completed, cancelled)
- `priority` (optional): Filter by priority (critical, high, medium, low)
- `limit` (optional): Maximum campaigns to return (default: 50)
- `offset` (optional): Number of campaigns to skip (default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "campaigns": [
      {
        "campaign_id": "campaign_20250822_143052",
        "objective": "Establish AI technology leadership in healthcare",
        "status": "active",
        "priority": "critical",
        "created_at": "2025-08-22T14:30:52Z"
      }
    ],
    "total_count": 15,
    "limit": 20,
    "offset": 0
  }
}
```

### Get Campaign Details

Retrieve detailed information about a specific campaign.

```http
GET /campaigns/{campaign_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "campaign_id": "campaign_20250822_143052",
    "objective": "Establish AI technology leadership in healthcare",
    "target_outcomes": ["..."],
    "status": "active",
    "priority": "critical",
    "orchestration_plan": {
      "phases": ["..."],
      "relationship_strategy": {"..."},
      "influence_strategy": {"..."}
    },
    "progress": {
      "completion_percentage": 35,
      "current_phase": "Network Building",
      "milestones_achieved": 8
    }
  }
}
```

### Update Campaign Status

Update the status of an influence campaign.

```http
PUT /campaigns/{campaign_id}/status
```

**Request Body:**
```json
{
  "new_status": "paused",
  "reason": "Awaiting regulatory approval"
}
```

## Network Management

### Get Network Status

Retrieve comprehensive status of the global influence network.

```http
GET /network/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "active_campaigns": 12,
    "network_health": {
      "status": "healthy",
      "score": 0.89
    },
    "influence_metrics": {
      "total_influence_score": 0.82,
      "network_reach": 15000
    },
    "relationship_status": {
      "active_relationships": 450,
      "relationship_health": 0.85
    },
    "partnership_status": {
      "active_partnerships": 25,
      "partnership_value": 50000000
    },
    "last_updated": "2025-08-22T14:30:52Z"
  }
}
```

### Synchronize Network Data

Trigger synchronization of influence network data across all systems.

```http
POST /network/sync
```

**Response:**
```json
{
  "success": true,
  "message": "Network synchronization started",
  "data": {
    "sync_initiated": true,
    "timestamp": "2025-08-22T14:30:52Z"
  }
}
```

### Get Sync Status

Retrieve current synchronization status.

```http
GET /network/sync/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "last_sync": "2025-08-22T14:25:00Z",
    "relationship_sync_status": "success",
    "influence_sync_status": "success",
    "partnership_sync_status": "success",
    "network_sync_status": "success",
    "sync_errors": [],
    "records_synced": {
      "relationships": 450,
      "influence_targets": 1200,
      "partnerships": 25
    },
    "sync_health_score": 0.95
  }
}
```

## Analytics and Metrics

### Get Campaign Metrics

Retrieve analytics and metrics for a specific campaign.

```http
GET /analytics/campaigns/{campaign_id}/metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "campaign_id": "campaign_20250822_143052",
    "network_reach": 5000,
    "influence_score": 0.78,
    "relationship_quality": 0.82,
    "narrative_adoption": 0.65,
    "media_coverage": 45,
    "sentiment_score": 0.73,
    "partnership_conversions": 8,
    "roi": 3.2,
    "success_rate": 0.85,
    "measurement_date": "2025-08-22T14:30:52Z"
  }
}
```

### Get Network Performance

Retrieve overall network performance analytics.

```http
GET /analytics/network/performance
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_campaigns": 25,
    "active_campaigns": 12,
    "network_health_score": 0.89,
    "average_campaign_success_rate": 0.82,
    "total_influence_reach": 50000,
    "partnership_conversion_rate": 0.28,
    "roi_average": 4.1,
    "measurement_period": "30_days",
    "last_updated": "2025-08-22T14:30:52Z"
  }
}
```

## Influence Target Management

### Add Influence Target

Add a new influence target to the network.

```http
POST /targets/add
```

**Request Body:**
```json
{
  "name": "Dr. Sarah Johnson",
  "title": "Chief Medical Officer",
  "organization": "Global Health Systems",
  "stakeholder_type": "executive",
  "influence_score": 0.85,
  "contact_info": {
    "email": "sarah.johnson@globalhealthsys.com",
    "linkedin": "linkedin.com/in/sarahjohnsonmd",
    "phone": "+1-555-0123"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Influence target added successfully",
  "data": {
    "id": "target_12345",
    "name": "Dr. Sarah Johnson",
    "title": "Chief Medical Officer",
    "organization": "Global Health Systems",
    "stakeholder_type": "executive",
    "influence_score": 0.85
  }
}
```

### Search Influence Targets

Search for influence targets in the network.

```http
GET /targets/search?query=healthcare&stakeholder_type=executive&min_influence_score=0.7&limit=20
```

**Query Parameters:**
- `query`: Search query (name, title, organization)
- `stakeholder_type` (optional): Filter by stakeholder type
- `organization` (optional): Filter by organization
- `min_influence_score` (optional): Minimum influence score
- `limit` (optional): Maximum results to return (default: 20)

**Response:**
```json
{
  "success": true,
  "data": {
    "targets": [
      {
        "id": "target_12345",
        "name": "Dr. Sarah Johnson",
        "title": "Chief Medical Officer",
        "organization": "Global Health Systems",
        "stakeholder_type": "executive",
        "influence_score": 0.85,
        "relationship_status": "active"
      }
    ],
    "query": "healthcare",
    "total_results": 1,
    "limit": 20
  }
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "success": false,
  "error": {
    "code": "CAMPAIGN_NOT_FOUND",
    "message": "Campaign with ID 'invalid_id' not found",
    "details": {
      "campaign_id": "invalid_id",
      "timestamp": "2025-08-22T14:30:52Z"
    }
  }
}
```

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `INVALID_TOKEN` | JWT token is invalid or expired |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `CAMPAIGN_NOT_FOUND` | Campaign ID not found |
| `INVALID_PARAMETERS` | Request parameters are invalid |
| `SYNC_IN_PROGRESS` | Network sync already in progress |
| `ORCHESTRATION_FAILED` | Campaign orchestration failed |

## Webhooks

### Campaign Status Updates

Receive notifications when campaign status changes.

**Webhook URL Configuration:**
```http
POST /webhooks/configure
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/influence-network",
  "events": ["campaign.status_changed", "campaign.milestone_reached"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload Example:**
```json
{
  "event": "campaign.status_changed",
  "timestamp": "2025-08-22T14:30:52Z",
  "data": {
    "campaign_id": "campaign_20250822_143052",
    "old_status": "active",
    "new_status": "completed",
    "completion_rate": 1.0
  }
}
```

## SDKs and Libraries

### Python SDK

```python
from scrollintel_sdk import InfluenceNetworkClient

client = InfluenceNetworkClient(
    api_key="your_api_key",
    base_url="https://api.scrollintel.com"
)

# Create campaign
campaign = client.campaigns.create(
    objective="Establish AI leadership",
    target_outcomes=["Build partnerships", "Influence policy"],
    timeline_days=180
)

# Get network status
status = client.network.get_status()
```

### JavaScript SDK

```javascript
import { InfluenceNetworkClient } from '@scrollintel/sdk';

const client = new InfluenceNetworkClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.scrollintel.com'
});

// Create campaign
const campaign = await client.campaigns.create({
  objective: 'Establish AI leadership',
  targetOutcomes: ['Build partnerships', 'Influence policy'],
  timelineDays: 180
});

// Get analytics
const metrics = await client.analytics.getCampaignMetrics(campaign.id);
```

## Health Check

### System Health

Check the health status of the influence network system.

```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2025-08-22T14:30:52Z",
    "version": "1.0.0",
    "components": {
      "orchestrator": "healthy",
      "relationship_engine": "healthy",
      "influence_engine": "healthy",
      "partnership_engine": "healthy"
    },
    "active_campaigns": 12,
    "system_uptime": "99.9%"
  }
}
```

## Best Practices

### Campaign Design

1. **Clear Objectives**: Define specific, measurable campaign objectives
2. **Realistic Timelines**: Allow adequate time for relationship building
3. **Stakeholder Mapping**: Identify all relevant stakeholders early
4. **Compliance Awareness**: Consider regulatory requirements from the start

### API Usage

1. **Authentication**: Always use secure token storage
2. **Rate Limiting**: Implement exponential backoff for rate-limited requests
3. **Error Handling**: Handle all error scenarios gracefully
4. **Monitoring**: Monitor API usage and campaign performance

### Security

1. **Token Management**: Rotate tokens regularly
2. **Permissions**: Use least-privilege access principles
3. **Audit Logging**: Monitor all API access and changes
4. **Data Protection**: Encrypt sensitive influence data

## Support

For API support and questions:
- Documentation: https://docs.scrollintel.com/influence-network
- Support Email: api-support@scrollintel.com
- Status Page: https://status.scrollintel.com
- GitHub Issues: https://github.com/scrollintel/api-issues