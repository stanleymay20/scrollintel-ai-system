"""
Knowledge base seeder for ScrollIntel Launch MVP.
Creates comprehensive help documentation and FAQ entries.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from scrollintel.models.support_models import KnowledgeBaseArticle, FAQ

def seed_knowledge_base(db: Session):
    """Seed the knowledge base with comprehensive help content."""
    
    # Clear existing content
    db.query(KnowledgeBaseArticle).delete()
    db.query(FAQ).delete()
    
    # Knowledge Base Articles
    articles = [
        {
            "title": "Getting Started with ScrollIntel",
            "slug": "getting-started",
            "content": """
# Getting Started with ScrollIntel

Welcome to ScrollIntel, your AI-powered CTO replacement platform! This guide will help you get up and running quickly.

## What is ScrollIntel?

ScrollIntel is an advanced AI platform that provides expert-level technical leadership and data analysis capabilities. Our AI agents can replace human CTOs, data scientists, ML engineers, and other technical roles.

## Key Features

- **AI CTO Agent**: Strategic technical leadership and decision-making
- **Data Scientist Agent**: Automated data analysis and insights
- **ML Engineer Agent**: Model building and deployment
- **Business Intelligence Agent**: Interactive dashboards and reporting
- **API Integration**: Programmatic access to all capabilities

## Quick Start Steps

1. **Sign Up**: Create your ScrollIntel account
2. **Upload Data**: Upload your first dataset (CSV, Excel, or JSON)
3. **Choose an Agent**: Select the AI agent that matches your needs
4. **Get Insights**: Ask questions and receive expert-level analysis
5. **Export Results**: Download reports, charts, and recommendations

## Your First Analysis

1. Navigate to the dashboard
2. Click "Upload Data" and select a file
3. Choose the "Data Scientist" agent
4. Ask: "What insights can you find in this data?"
5. Review the automated analysis and visualizations

## Need Help?

- Browse our FAQ section
- Check out video tutorials
- Contact support for personalized assistance
- Join our community forum

Ready to transform your business with AI? Let's get started!
            """,
            "summary": "Complete guide to getting started with ScrollIntel's AI-powered platform",
            "category": "getting_started",
            "tags": ["beginner", "setup", "overview"]
        },
        {
            "title": "Data Upload and File Formats",
            "slug": "data-upload-formats",
            "content": """
# Data Upload and File Formats

ScrollIntel supports multiple data formats and provides flexible upload options for your datasets.

## Supported File Formats

### CSV Files
- Standard comma-separated values
- UTF-8 encoding recommended
- Maximum file size: 100MB
- Headers in first row preferred

### Excel Files (.xlsx, .xls)
- Multiple worksheets supported
- Automatic sheet detection
- Preserves data types and formatting
- Maximum file size: 100MB

### JSON Files
- Structured JSON data
- Nested objects supported
- Array of objects format preferred
- Maximum file size: 100MB

### SQL Files
- Database dump files
- CREATE and INSERT statements
- Multiple table support
- Automatic schema detection

## Upload Methods

### Drag and Drop
1. Navigate to the data upload area
2. Drag your file from your computer
3. Drop it in the designated zone
4. Wait for upload confirmation

### File Browser
1. Click "Choose File" button
2. Browse and select your file
3. Click "Open" to start upload
4. Monitor upload progress

### API Upload
```python
import requests

files = {'file': open('data.csv', 'rb')}
response = requests.post('https://api.scrollintel.com/upload', files=files)
```

## Data Quality Tips

- **Clean Headers**: Use descriptive column names without special characters
- **Consistent Formats**: Ensure date and number formats are consistent
- **Handle Missing Values**: Use empty cells or standard null indicators
- **Remove Duplicates**: Clean duplicate rows before upload
- **Validate Data Types**: Ensure numeric columns contain only numbers

## File Processing

After upload, ScrollIntel automatically:
- Detects data types and formats
- Identifies missing values and outliers
- Generates data quality reports
- Creates initial visualizations
- Suggests analysis approaches

## Troubleshooting Upload Issues

### File Too Large
- Split large files into smaller chunks
- Use data sampling for initial analysis
- Contact support for enterprise solutions

### Format Not Recognized
- Check file extension matches content
- Ensure proper encoding (UTF-8)
- Validate file structure and format

### Upload Fails
- Check internet connection
- Try a different browser
- Clear browser cache and cookies
- Contact support if issues persist

## Security and Privacy

- All uploads are encrypted in transit
- Data is stored securely in the cloud
- Access controls protect your information
- Data retention policies apply
- GDPR and compliance standards met

Need help with a specific file format? Contact our support team!
            """,
            "summary": "Complete guide to uploading data and supported file formats",
            "category": "data_upload",
            "tags": ["upload", "formats", "csv", "excel", "json"]
        },
        {
            "title": "Understanding AI Agents",
            "slug": "ai-agents-overview",
            "content": """
# Understanding ScrollIntel AI Agents

ScrollIntel's AI agents are specialized artificial intelligence systems designed to replace human experts in various technical roles.

## Available AI Agents

### CTO Agent
**Role**: Chief Technology Officer
**Capabilities**:
- Strategic technology planning
- Architecture recommendations
- Technology stack decisions
- Risk assessment and mitigation
- Team structure and hiring guidance
- Budget planning and resource allocation

**Best For**: High-level strategic decisions, technology roadmaps, executive reporting

### Data Scientist Agent
**Role**: Senior Data Scientist
**Capabilities**:
- Exploratory data analysis
- Statistical modeling
- Pattern recognition
- Predictive analytics
- Data visualization
- Hypothesis testing

**Best For**: Data exploration, statistical analysis, research questions

### ML Engineer Agent
**Role**: Machine Learning Engineer
**Capabilities**:
- Model development and training
- Feature engineering
- Model deployment and monitoring
- Performance optimization
- MLOps pipeline creation
- A/B testing frameworks

**Best For**: Building and deploying machine learning models

### Business Intelligence Agent
**Role**: BI Analyst
**Capabilities**:
- Dashboard creation
- KPI tracking and monitoring
- Business metrics analysis
- Report generation
- Data storytelling
- Executive summaries

**Best For**: Business reporting, dashboards, executive presentations

### AI Engineer Agent
**Role**: AI/ML Infrastructure Engineer
**Capabilities**:
- AI system architecture
- Model serving infrastructure
- Scalability optimization
- Integration planning
- Performance monitoring
- Technical documentation

**Best For**: AI infrastructure, system integration, technical implementation

## How to Choose the Right Agent

### For Strategic Questions
- Use **CTO Agent** for technology strategy, architecture decisions, and executive-level planning

### For Data Analysis
- Use **Data Scientist Agent** for exploring data, finding patterns, and statistical analysis

### For Model Building
- Use **ML Engineer Agent** for creating predictive models and machine learning solutions

### For Business Reporting
- Use **BI Agent** for dashboards, reports, and business intelligence

### For Technical Implementation
- Use **AI Engineer Agent** for system architecture and technical integration

## Agent Interaction Tips

### Be Specific
- Provide clear, detailed questions
- Include relevant context and constraints
- Specify desired output format

### Provide Context
- Share business objectives
- Explain data sources and limitations
- Mention any existing systems or processes

### Iterate and Refine
- Start with broad questions, then drill down
- Ask follow-up questions for clarification
- Request alternative approaches or solutions

## Example Interactions

### CTO Agent
"What technology stack would you recommend for a fintech startup handling 10,000 transactions per day?"

### Data Scientist Agent
"Analyze this customer data and identify the key factors that predict churn"

### ML Engineer Agent
"Build a recommendation system for our e-commerce platform using this purchase history data"

### BI Agent
"Create a executive dashboard showing our key business metrics and trends"

### AI Engineer Agent
"Design the architecture for deploying our ML models at scale with 99.9% uptime"

## Agent Capabilities Matrix

| Capability | CTO | Data Scientist | ML Engineer | BI Agent | AI Engineer |
|------------|-----|----------------|-------------|----------|-------------|
| Strategy Planning | ✅ | ❌ | ❌ | ❌ | ❌ |
| Data Analysis | ❌ | ✅ | ✅ | ✅ | ❌ |
| Model Building | ❌ | ✅ | ✅ | ❌ | ❌ |
| Visualization | ❌ | ✅ | ❌ | ✅ | ❌ |
| Architecture | ✅ | ❌ | ❌ | ❌ | ✅ |
| Deployment | ❌ | ❌ | ✅ | ❌ | ✅ |

## Getting the Best Results

1. **Choose the right agent** for your specific need
2. **Provide comprehensive context** about your situation
3. **Ask specific questions** rather than vague requests
4. **Iterate based on responses** to refine your approach
5. **Combine agents** for complex multi-faceted projects

Ready to work with our AI agents? Start by uploading your data and selecting the most appropriate agent for your needs!
            """,
            "summary": "Comprehensive guide to ScrollIntel's AI agents and their capabilities",
            "category": "ai_agents",
            "tags": ["agents", "cto", "data-scientist", "ml-engineer", "bi"]
        },
        {
            "title": "API Documentation and Integration",
            "slug": "api-documentation",
            "content": """
# ScrollIntel API Documentation

The ScrollIntel API provides programmatic access to all platform capabilities, allowing you to integrate AI-powered analysis into your applications.

## Authentication

All API requests require authentication using API keys.

### Getting Your API Key
1. Navigate to Settings > API Keys
2. Click "Generate New Key"
3. Copy and securely store your key
4. Use the key in the Authorization header

### Authentication Header
```
Authorization: Bearer YOUR_API_KEY
```

## Base URL
```
https://api.scrollintel.com/v1
```

## Core Endpoints

### Data Upload
Upload datasets for analysis.

```http
POST /data/upload
Content-Type: multipart/form-data

{
  "file": [binary file data],
  "name": "dataset_name",
  "description": "Dataset description"
}
```

**Response:**
```json
{
  "dataset_id": "uuid",
  "name": "dataset_name",
  "status": "processing",
  "file_size": 1024000,
  "rows": 10000,
  "columns": 25
}
```

### Agent Interaction
Interact with AI agents for analysis.

```http
POST /agents/{agent_type}/analyze
Content-Type: application/json

{
  "dataset_id": "uuid",
  "query": "What insights can you find in this data?",
  "context": "E-commerce customer data for churn analysis"
}
```

**Response:**
```json
{
  "analysis_id": "uuid",
  "agent_type": "data_scientist",
  "status": "completed",
  "insights": [
    {
      "type": "finding",
      "title": "Customer Churn Pattern",
      "description": "High-value customers show 23% churn rate...",
      "confidence": 0.87
    }
  ],
  "visualizations": [
    {
      "type": "chart",
      "url": "https://api.scrollintel.com/charts/uuid.png"
    }
  ]
}
```

### Results Export
Export analysis results in various formats.

```http
GET /analysis/{analysis_id}/export?format=pdf
```

**Supported Formats:**
- `pdf` - PDF report
- `excel` - Excel workbook
- `json` - JSON data
- `csv` - CSV data

## Agent Types

- `cto` - CTO Agent
- `data_scientist` - Data Scientist Agent
- `ml_engineer` - ML Engineer Agent
- `bi_analyst` - Business Intelligence Agent
- `ai_engineer` - AI Engineer Agent

## Rate Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour
- **Enterprise**: Custom limits

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

**Error Response Format:**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is missing required parameters",
    "details": "Missing 'dataset_id' parameter"
  }
}
```

## SDK Libraries

### Python SDK
```bash
pip install scrollintel-sdk
```

```python
from scrollintel import ScrollIntelClient

client = ScrollIntelClient(api_key="your_api_key")

# Upload data
dataset = client.upload_data("data.csv")

# Analyze with Data Scientist agent
analysis = client.data_scientist.analyze(
    dataset_id=dataset.id,
    query="Find patterns in customer behavior"
)

# Export results
report = client.export_analysis(analysis.id, format="pdf")
```

### JavaScript SDK
```bash
npm install scrollintel-js
```

```javascript
import ScrollIntel from 'scrollintel-js';

const client = new ScrollIntel({ apiKey: 'your_api_key' });

// Upload data
const dataset = await client.uploadData('data.csv');

// Analyze with CTO agent
const analysis = await client.cto.analyze({
  datasetId: dataset.id,
  query: 'What technology architecture do you recommend?'
});

// Get results
const results = await client.getAnalysis(analysis.id);
```

## Webhooks

Subscribe to events for real-time updates.

### Webhook Events
- `analysis.completed` - Analysis finished
- `data.processed` - Data upload processed
- `export.ready` - Export file ready

### Webhook Payload
```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "analysis_id": "uuid",
    "status": "completed",
    "agent_type": "data_scientist"
  }
}
```

## Best Practices

1. **Cache Results**: Store analysis results to avoid repeated API calls
2. **Handle Rate Limits**: Implement exponential backoff for rate-limited requests
3. **Secure API Keys**: Never expose API keys in client-side code
4. **Monitor Usage**: Track API usage to optimize costs
5. **Error Handling**: Implement robust error handling for all requests

## Support

- **Documentation**: https://docs.scrollintel.com
- **Support Email**: api-support@scrollintel.com
- **Status Page**: https://status.scrollintel.com
- **Community**: https://community.scrollintel.com

Ready to integrate ScrollIntel into your application? Get your API key and start building!
            """,
            "summary": "Complete API documentation for integrating ScrollIntel into your applications",
            "category": "api_documentation",
            "tags": ["api", "integration", "sdk", "authentication"]
        },
        {
            "title": "Billing and Subscription Plans",
            "slug": "billing-subscription",
            "content": """
# Billing and Subscription Plans

ScrollIntel offers flexible pricing plans to meet the needs of individuals, teams, and enterprises.

## Subscription Plans

### Free Tier
**Price**: $0/month
**Features**:
- 5 data uploads per month
- 100 AI agent interactions
- Basic visualizations
- Email support
- 1GB data storage

**Best For**: Personal projects, learning, small datasets

### Professional Plan
**Price**: $49/month
**Features**:
- Unlimited data uploads
- 1,000 AI agent interactions
- Advanced visualizations
- Priority email support
- 10GB data storage
- API access (100 requests/hour)
- Export to PDF/Excel

**Best For**: Small businesses, consultants, regular users

### Team Plan
**Price**: $149/month
**Features**:
- Everything in Professional
- 5,000 AI agent interactions
- Team collaboration features
- Phone support
- 50GB data storage
- API access (500 requests/hour)
- Custom branding
- Advanced analytics

**Best For**: Growing teams, departments, collaborative projects

### Enterprise Plan
**Price**: Custom pricing
**Features**:
- Everything in Team
- Unlimited AI agent interactions
- Dedicated account manager
- 24/7 phone support
- Unlimited data storage
- Custom API limits
- On-premise deployment option
- Custom integrations
- SLA guarantees

**Best For**: Large organizations, mission-critical applications

## Usage-Based Billing

Some features are billed based on usage:

### AI Agent Interactions
- **Free**: 100 interactions/month included
- **Overage**: $0.10 per additional interaction

### Data Storage
- **Free**: 1GB included
- **Additional**: $0.50 per GB/month

### API Requests
- **Free**: 100 requests/hour
- **Additional**: $0.01 per request

## Payment Methods

- **Credit Cards**: Visa, MasterCard, American Express
- **PayPal**: Available for all plans
- **Bank Transfer**: Available for Enterprise plans
- **Purchase Orders**: Available for Enterprise plans

## Billing Cycle

- **Monthly**: Billed on the same day each month
- **Annual**: 20% discount when paying annually
- **Enterprise**: Custom billing terms available

## Managing Your Subscription

### Upgrade/Downgrade
1. Go to Settings > Billing
2. Click "Change Plan"
3. Select your new plan
4. Confirm changes

**Note**: Upgrades take effect immediately. Downgrades take effect at the next billing cycle.

### Cancel Subscription
1. Go to Settings > Billing
2. Click "Cancel Subscription"
3. Confirm cancellation
4. Access continues until end of billing period

### View Usage
Monitor your current usage:
- AI agent interactions used/remaining
- Data storage used/available
- API requests this month
- Billing history and invoices

## Enterprise Features

### Custom Deployment
- On-premise installation
- Private cloud deployment
- Hybrid cloud solutions
- Custom security configurations

### Advanced Support
- Dedicated account manager
- 24/7 phone support
- Custom training sessions
- Implementation assistance

### Custom Integrations
- API customizations
- Third-party integrations
- Custom data connectors
- Workflow automation

## Frequently Asked Questions

### Can I change plans anytime?
Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately for upgrades and at the next billing cycle for downgrades.

### What happens if I exceed my limits?
For usage-based features, you'll be charged overage fees. For hard limits, you'll need to upgrade your plan to continue using the service.

### Do you offer refunds?
We offer a 30-day money-back guarantee for new subscriptions. Contact support for refund requests.

### Can I get a custom plan?
Yes, we offer custom Enterprise plans tailored to your specific needs. Contact our sales team for details.

### How secure is my billing information?
All payment information is processed securely using industry-standard encryption. We never store credit card details on our servers.

## Getting Started

1. **Choose Your Plan**: Select the plan that best fits your needs
2. **Sign Up**: Create your ScrollIntel account
3. **Add Payment**: Securely add your payment method
4. **Start Using**: Begin uploading data and interacting with AI agents

## Need Help?

- **Billing Questions**: billing@scrollintel.com
- **Sales Inquiries**: sales@scrollintel.com
- **Technical Support**: support@scrollintel.com
- **Phone Support**: Available for Team and Enterprise plans

Ready to get started? Choose your plan and unlock the power of AI-driven insights!
            """,
            "summary": "Complete guide to ScrollIntel's pricing plans and billing information",
            "category": "billing",
            "tags": ["pricing", "billing", "subscription", "plans"]
        }
    ]
    
    # Create articles
    for article_data in articles:
        article = KnowledgeBaseArticle(**article_data)
        db.add(article)
    
    # FAQ Entries
    faqs = [
        {
            "question": "What is ScrollIntel and how does it work?",
            "answer": "ScrollIntel is an AI-powered platform that replaces human technical experts with specialized AI agents. Upload your data, choose an AI agent (CTO, Data Scientist, ML Engineer, etc.), and get expert-level analysis and recommendations instantly.",
            "category": "general",
            "is_featured": True,
            "order_index": 1
        },
        {
            "question": "What file formats can I upload?",
            "answer": "ScrollIntel supports CSV, Excel (.xlsx, .xls), JSON, and SQL files up to 100MB in size. We automatically detect data types and formats for seamless processing.",
            "category": "data_upload",
            "is_featured": True,
            "order_index": 2
        },
        {
            "question": "How do I choose the right AI agent?",
            "answer": "Choose based on your needs: CTO Agent for strategy, Data Scientist for analysis, ML Engineer for models, BI Agent for dashboards, and AI Engineer for technical architecture. Each agent specializes in different aspects of technical work.",
            "category": "ai_agents",
            "is_featured": True,
            "order_index": 3
        },
        {
            "question": "Is my data secure and private?",
            "answer": "Yes, we use enterprise-grade security with encryption in transit and at rest. Your data is never shared with third parties, and we comply with GDPR, SOC 2, and other privacy standards.",
            "category": "security",
            "is_featured": True,
            "order_index": 4
        },
        {
            "question": "Can I export my analysis results?",
            "answer": "Yes, you can export results in multiple formats including PDF reports, Excel workbooks, CSV data, and JSON. All visualizations and insights are included in the exports.",
            "category": "export",
            "is_featured": True,
            "order_index": 5
        },
        {
            "question": "Do you offer an API for integration?",
            "answer": "Yes, we provide a comprehensive REST API with SDKs for Python and JavaScript. You can upload data, interact with agents, and export results programmatically.",
            "category": "api",
            "is_featured": False,
            "order_index": 6
        },
        {
            "question": "What are the usage limits for each plan?",
            "answer": "Free: 100 interactions/month, Professional: 1,000 interactions/month, Team: 5,000 interactions/month, Enterprise: unlimited. Storage and API limits also vary by plan.",
            "category": "billing",
            "is_featured": False,
            "order_index": 7
        },
        {
            "question": "Can I collaborate with my team?",
            "answer": "Yes, Team and Enterprise plans include collaboration features like shared workspaces, team member management, and role-based access controls.",
            "category": "collaboration",
            "is_featured": False,
            "order_index": 8
        },
        {
            "question": "How accurate are the AI agent recommendations?",
            "answer": "Our AI agents are trained on vast datasets and provide expert-level accuracy. However, we recommend reviewing all recommendations in the context of your specific business needs and constraints.",
            "category": "ai_agents",
            "is_featured": False,
            "order_index": 9
        },
        {
            "question": "What support options are available?",
            "answer": "We offer email support for all plans, priority email for Professional+, phone support for Team+, and dedicated account managers for Enterprise. We also have comprehensive documentation and video tutorials.",
            "category": "support",
            "is_featured": False,
            "order_index": 10
        },
        {
            "question": "Can I cancel my subscription anytime?",
            "answer": "Yes, you can cancel anytime. Your access continues until the end of your current billing period. We also offer a 30-day money-back guarantee for new subscriptions.",
            "category": "billing",
            "is_featured": False,
            "order_index": 11
        },
        {
            "question": "Do you offer custom enterprise solutions?",
            "answer": "Yes, we offer custom Enterprise plans with on-premise deployment, custom integrations, dedicated support, and tailored features to meet your organization's specific needs.",
            "category": "enterprise",
            "is_featured": False,
            "order_index": 12
        },
        {
            "question": "How do I get started with ScrollIntel?",
            "answer": "Simply sign up for an account, upload your first dataset, choose an AI agent, and start asking questions. Our onboarding tutorial will guide you through the process step-by-step.",
            "category": "getting_started",
            "is_featured": False,
            "order_index": 13
        },
        {
            "question": "What happens if I exceed my plan limits?",
            "answer": "For usage-based features, you'll be charged overage fees at standard rates. For hard limits, you'll need to upgrade your plan to continue using the service.",
            "category": "billing",
            "is_featured": False,
            "order_index": 14
        },
        {
            "question": "Can I integrate ScrollIntel with my existing tools?",
            "answer": "Yes, we offer API integration and can work with your team to create custom integrations with your existing business intelligence, data warehouse, and analytics tools.",
            "category": "integration",
            "is_featured": False,
            "order_index": 15
        }
    ]
    
    # Create FAQs
    for faq_data in faqs:
        faq = FAQ(**faq_data)
        db.add(faq)
    
    # Commit all changes
    db.commit()
    print(f"Seeded {len(articles)} knowledge base articles and {len(faqs)} FAQ entries")

if __name__ == "__main__":
    from scrollintel.models.database import get_db
    
    db = next(get_db())
    seed_knowledge_base(db)
    db.close()