"""
Cloud Cost Collector for automated cost collection from cloud platforms and tools.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ..core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CloudCostData:
    """Structured cloud cost data."""
    service_name: str
    cost: float
    currency: str
    billing_period: str
    usage_start_date: datetime
    usage_end_date: datetime
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    region: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    usage_quantity: Optional[float] = None
    usage_unit: Optional[str] = None
    rate: Optional[float] = None


class AWSCostCollector:
    """AWS Cost Explorer integration for automated cost collection."""
    
    def __init__(self, access_key: str, secret_key: str, region: str = "us-east-1"):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS cost collection. Install with: pip install boto3")
        
        self.client = boto3.client(
            'ce',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    
    def get_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "MONTHLY",
        group_by: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, str]] = None
    ) -> List[CloudCostData]:
        """Get AWS costs from Cost Explorer API."""
        try:
            # Prepare time period
            time_period = {
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            }
            
            # Prepare group by
            group_by_params = []
            if group_by:
                for group in group_by:
                    group_by_params.append({'Type': 'DIMENSION', 'Key': group})
            else:
                group_by_params = [{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            
            # Prepare filter
            filter_params = None
            if filter_tags:
                tag_filters = []
                for key, value in filter_tags.items():
                    tag_filters.append({
                        'Key': key,
                        'Values': [value],
                        'MatchOptions': ['EQUALS']
                    })
                filter_params = {
                    'Tags': {
                        'Key': 'user:Project',
                        'Values': list(filter_tags.values()),
                        'MatchOptions': ['EQUALS']
                    }
                }
            
            # Make API call
            response = self.client.get_cost_and_usage(
                TimePeriod=time_period,
                Granularity=granularity,
                Metrics=['BlendedCost', 'UsageQuantity'],
                GroupBy=group_by_params,
                Filter=filter_params
            )
            
            cost_data = []
            
            for result in response.get('ResultsByTime', []):
                period_start = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d')
                period_end = datetime.strptime(result['TimePeriod']['End'], '%Y-%m-%d')
                
                for group in result.get('Groups', []):
                    service_name = group['Keys'][0] if group['Keys'] else 'Unknown'
                    cost_amount = float(group['Metrics']['BlendedCost']['Amount'])
                    usage_quantity = float(group['Metrics']['UsageQuantity']['Amount'])
                    
                    if cost_amount > 0:  # Only include non-zero costs
                        cost_data.append(CloudCostData(
                            service_name=service_name,
                            cost=cost_amount,
                            currency=group['Metrics']['BlendedCost']['Unit'],
                            billing_period=result['TimePeriod']['Start'][:7],  # YYYY-MM format
                            usage_start_date=period_start,
                            usage_end_date=period_end,
                            usage_quantity=usage_quantity,
                            usage_unit=group['Metrics']['UsageQuantity']['Unit']
                        ))
            
            logger.info(f"Collected {len(cost_data)} AWS cost items")
            return cost_data
            
        except Exception as e:
            logger.error(f"Error collecting AWS costs: {str(e)}")
            raise
    
    def get_resource_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        resource_tags: Dict[str, str]
    ) -> List[CloudCostData]:
        """Get costs for specific resources by tags."""
        try:
            # Use resource-level cost allocation
            return self.get_costs(
                start_date=start_date,
                end_date=end_date,
                group_by=['SERVICE', 'USAGE_TYPE'],
                filter_tags=resource_tags
            )
        except Exception as e:
            logger.error(f"Error collecting AWS resource costs: {str(e)}")
            raise


class AzureCostCollector:
    """Azure Cost Management integration for automated cost collection."""
    
    def __init__(self, subscription_id: str, client_id: str, client_secret: str, tenant_id: str):
        self.subscription_id = subscription_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.access_token = None
    
    def _get_access_token(self) -> str:
        """Get Azure access token for API calls."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required for Azure cost collection. Install with: pip install requests")
            
        try:
            token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"
            
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'resource': 'https://management.azure.com/'
            }
            
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            return self.access_token
            
        except Exception as e:
            logger.error(f"Error getting Azure access token: {str(e)}")
            raise
    
    def get_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        resource_group: Optional[str] = None
    ) -> List[CloudCostData]:
        """Get Azure costs from Cost Management API."""
        try:
            if not self.access_token:
                self._get_access_token()
            
            # Prepare API endpoint
            scope = f"/subscriptions/{self.subscription_id}"
            if resource_group:
                scope += f"/resourceGroups/{resource_group}"
            
            url = f"https://management.azure.com{scope}/providers/Microsoft.CostManagement/query"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            # Prepare query
            query = {
                "type": "ActualCost",
                "timeframe": "Custom",
                "timePeriod": {
                    "from": start_date.strftime('%Y-%m-%dT00:00:00Z'),
                    "to": end_date.strftime('%Y-%m-%dT23:59:59Z')
                },
                "dataset": {
                    "granularity": "Monthly",
                    "aggregation": {
                        "totalCost": {
                            "name": "PreTaxCost",
                            "function": "Sum"
                        }
                    },
                    "grouping": [
                        {
                            "type": "Dimension",
                            "name": "ServiceName"
                        }
                    ]
                }
            }
            
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests is required for Azure cost collection. Install with: pip install requests")
                
            response = requests.post(url, headers=headers, json=query)
            response.raise_for_status()
            
            data = response.json()
            cost_data = []
            
            for row in data.get('properties', {}).get('rows', []):
                if len(row) >= 3:
                    cost_amount = float(row[0])
                    service_name = row[2]
                    billing_period = row[1]  # Date in YYYYMMDD format
                    
                    if cost_amount > 0:
                        # Convert billing period to datetime
                        period_date = datetime.strptime(str(billing_period), '%Y%m%d')
                        
                        cost_data.append(CloudCostData(
                            service_name=service_name,
                            cost=cost_amount,
                            currency='USD',  # Azure typically returns USD
                            billing_period=period_date.strftime('%Y-%m'),
                            usage_start_date=period_date,
                            usage_end_date=period_date + timedelta(days=30)
                        ))
            
            logger.info(f"Collected {len(cost_data)} Azure cost items")
            return cost_data
            
        except Exception as e:
            logger.error(f"Error collecting Azure costs: {str(e)}")
            raise


class GCPCostCollector:
    """Google Cloud Platform cost collection integration."""
    
    def __init__(self, project_id: str, service_account_key: str):
        self.project_id = project_id
        self.service_account_key = service_account_key
    
    def get_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        services: Optional[List[str]] = None
    ) -> List[CloudCostData]:
        """Get GCP costs from Cloud Billing API."""
        try:
            # This would require google-cloud-billing library
            # For now, return empty list as placeholder
            logger.warning("GCP cost collection not fully implemented")
            return []
            
        except Exception as e:
            logger.error(f"Error collecting GCP costs: {str(e)}")
            raise


class ToolCostCollector:
    """Collector for various tool and service costs."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def collect_github_costs(
        self,
        organization: str,
        api_token: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CloudCostData]:
        """Collect GitHub usage and billing information."""
        try:
            headers = {
                'Authorization': f'token {api_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Get billing information
            billing_url = f"https://api.github.com/orgs/{organization}/settings/billing/actions"
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests is required for GitHub cost collection. Install with: pip install requests")
                
            response = requests.get(billing_url, headers=headers)
            response.raise_for_status()
            
            billing_data = response.json()
            
            cost_data = []
            
            # GitHub Actions minutes
            if 'total_minutes_used' in billing_data:
                minutes_used = billing_data['total_minutes_used']
                # Estimate cost based on GitHub pricing
                estimated_cost = minutes_used * 0.008  # $0.008 per minute for private repos
                
                cost_data.append(CloudCostData(
                    service_name="GitHub Actions",
                    cost=estimated_cost,
                    currency="USD",
                    billing_period=start_date.strftime('%Y-%m'),
                    usage_start_date=start_date,
                    usage_end_date=end_date,
                    usage_quantity=minutes_used,
                    usage_unit="minutes"
                ))
            
            logger.info(f"Collected GitHub cost data for {organization}")
            return cost_data
            
        except Exception as e:
            logger.error(f"Error collecting GitHub costs: {str(e)}")
            return []
    
    def collect_datadog_costs(
        self,
        api_key: str,
        app_key: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CloudCostData]:
        """Collect Datadog usage and billing information."""
        try:
            headers = {
                'DD-API-KEY': api_key,
                'DD-APPLICATION-KEY': app_key
            }
            
            # Get usage data
            start_hr = int(start_date.timestamp())
            end_hr = int(end_date.timestamp())
            
            url = f"https://api.datadoghq.com/api/v1/usage/hosts"
            params = {
                'start_hr': start_hr,
                'end_hr': end_hr
            }
            
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests is required for Datadog cost collection. Install with: pip install requests")
                
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            usage_data = response.json()
            cost_data = []
            
            # Estimate costs based on host usage
            for usage in usage_data.get('usage', []):
                host_count = usage.get('host_count', 0)
                # Estimate cost at $15 per host per month
                estimated_cost = host_count * 15
                
                usage_date = datetime.fromtimestamp(usage.get('hour', 0))
                
                cost_data.append(CloudCostData(
                    service_name="Datadog Monitoring",
                    cost=estimated_cost,
                    currency="USD",
                    billing_period=usage_date.strftime('%Y-%m'),
                    usage_start_date=usage_date,
                    usage_end_date=usage_date + timedelta(hours=1),
                    usage_quantity=host_count,
                    usage_unit="hosts"
                ))
            
            logger.info(f"Collected Datadog cost data")
            return cost_data
            
        except Exception as e:
            logger.error(f"Error collecting Datadog costs: {str(e)}")
            return []


class CloudConnectorManager:
    """Manager for all cloud cost collection connectors."""
    
    def __init__(self):
        self.settings = get_settings()
        self.collectors = {}
    
    def register_aws_collector(self, access_key: str, secret_key: str, region: str = "us-east-1"):
        """Register AWS cost collector."""
        self.collectors['aws'] = AWSCostCollector(access_key, secret_key, region)
    
    def register_azure_collector(self, subscription_id: str, client_id: str, client_secret: str, tenant_id: str):
        """Register Azure cost collector."""
        self.collectors['azure'] = AzureCostCollector(subscription_id, client_id, client_secret, tenant_id)
    
    def register_gcp_collector(self, project_id: str, service_account_key: str):
        """Register GCP cost collector."""
        self.collectors['gcp'] = GCPCostCollector(project_id, service_account_key)
    
    def get_costs(
        self,
        provider: str,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Get costs from specified provider."""
        try:
            if provider not in self.collectors:
                raise ValueError(f"No collector registered for provider: {provider}")
            
            collector = self.collectors[provider]
            
            if provider == 'aws':
                cost_data = collector.get_costs(start_date, end_date, filter_tags=tags)
            elif provider == 'azure':
                cost_data = collector.get_costs(start_date, end_date)
            elif provider == 'gcp':
                cost_data = collector.get_costs(start_date, end_date)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Convert to dict format
            return [
                {
                    'service_name': item.service_name,
                    'cost': item.cost,
                    'currency': item.currency,
                    'billing_period': item.billing_period,
                    'usage_start_date': item.usage_start_date,
                    'usage_end_date': item.usage_end_date,
                    'resource_id': item.resource_id,
                    'resource_name': item.resource_name,
                    'region': item.region,
                    'tags': item.tags,
                    'usage_quantity': item.usage_quantity,
                    'usage_unit': item.usage_unit,
                    'rate': item.rate
                }
                for item in cost_data
            ]
            
        except Exception as e:
            logger.error(f"Error getting costs from {provider}: {str(e)}")
            raise