"""
API Billing Service for ScrollIntel Launch MVP.
Handles usage-based billing and integration with payment systems.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..models.api_key_models import APIKey, APIUsage, APIQuota
from ..models.database import User
from ..engines.usage_analytics_engine import UsageAnalyticsEngine


class APIBillingService:
    """
    Service for calculating and managing API usage billing.
    """
    
    # Pricing tiers (USD)
    PRICING_TIERS = {
        'free': {
            'requests_per_month': 1000,
            'cost_per_request': 0.0,
            'data_transfer_gb_included': 1.0,
            'cost_per_gb': 0.0,
            'compute_seconds_included': 60.0,
            'cost_per_compute_second': 0.0
        },
        'starter': {
            'requests_per_month': 10000,
            'cost_per_request': 0.001,  # $0.001 per request
            'data_transfer_gb_included': 10.0,
            'cost_per_gb': 0.10,  # $0.10 per GB
            'compute_seconds_included': 600.0,
            'cost_per_compute_second': 0.01  # $0.01 per second
        },
        'professional': {
            'requests_per_month': 100000,
            'cost_per_request': 0.0008,  # $0.0008 per request
            'data_transfer_gb_included': 100.0,
            'cost_per_gb': 0.08,  # $0.08 per GB
            'compute_seconds_included': 6000.0,
            'cost_per_compute_second': 0.008  # $0.008 per second
        },
        'enterprise': {
            'requests_per_month': None,  # Unlimited
            'cost_per_request': 0.0005,  # $0.0005 per request
            'data_transfer_gb_included': 1000.0,
            'cost_per_gb': 0.05,  # $0.05 per GB
            'compute_seconds_included': 60000.0,
            'cost_per_compute_second': 0.005  # $0.005 per second
        }
    }
    
    def __init__(self, db: Session):
        self.db = db
        self.analytics = UsageAnalyticsEngine(db)
    
    def calculate_usage_cost(
        self,
        api_key_id: str,
        period_start: datetime,
        period_end: datetime,
        pricing_tier: str = 'starter'
    ) -> Dict[str, Any]:
        """
        Calculate usage cost for an API key in a specific period.
        """
        if pricing_tier not in self.PRICING_TIERS:
            raise ValueError(f"Invalid pricing tier: {pricing_tier}")
        
        tier_config = self.PRICING_TIERS[pricing_tier]
        
        # Get usage data
        usage_query = self.db.query(APIUsage).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= period_start,
                APIUsage.timestamp <= period_end
            )
        )
        
        # Calculate metrics
        total_requests = usage_query.count()
        
        # Data transfer calculation
        data_transfer_bytes = self.db.query(
            func.sum(
                func.coalesce(APIUsage.request_size_bytes, 0) +
                func.coalesce(APIUsage.response_size_bytes, 0)
            )
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= period_start,
                APIUsage.timestamp <= period_end
            )
        ).scalar() or 0
        
        data_transfer_gb = data_transfer_bytes / (1024 ** 3)
        
        # Compute time calculation
        total_compute_seconds = self.db.query(
            func.sum(APIUsage.response_time_ms)
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= period_start,
                APIUsage.timestamp <= period_end
            )
        ).scalar() or 0
        
        total_compute_seconds = total_compute_seconds / 1000.0  # Convert ms to seconds
        
        # Calculate costs
        costs = self._calculate_tier_costs(
            total_requests,
            data_transfer_gb,
            total_compute_seconds,
            tier_config
        )
        
        return {
            'api_key_id': api_key_id,
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'pricing_tier': pricing_tier,
            'usage': {
                'requests': total_requests,
                'data_transfer_gb': round(data_transfer_gb, 4),
                'compute_seconds': round(total_compute_seconds, 2)
            },
            'costs': costs,
            'total_cost_usd': costs['total']
        }
    
    def calculate_monthly_bill(
        self,
        user_id: str,
        year: int,
        month: int,
        pricing_tier: str = 'starter'
    ) -> Dict[str, Any]:
        """
        Calculate monthly bill for a user across all API keys.
        """
        # Get period boundaries
        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)
        
        # Get user's API keys
        api_keys = self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).all()
        
        api_key_costs = []
        total_cost = Decimal('0.00')
        
        for api_key in api_keys:
            cost_data = self.calculate_usage_cost(
                str(api_key.id),
                period_start,
                period_end,
                pricing_tier
            )
            
            cost_data['api_key_name'] = api_key.name
            api_key_costs.append(cost_data)
            total_cost += Decimal(str(cost_data['total_cost_usd']))
        
        # Get user info
        user = self.db.query(User).filter(User.id == user_id).first()
        
        return {
            'bill_metadata': {
                'user_id': user_id,
                'user_email': user.email if user else None,
                'billing_period': f"{year}-{month:02d}",
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat(),
                'pricing_tier': pricing_tier,
                'generated_at': datetime.utcnow().isoformat()
            },
            'api_key_costs': api_key_costs,
            'summary': {
                'total_api_keys': len(api_keys),
                'total_requests': sum(cost['usage']['requests'] for cost in api_key_costs),
                'total_data_transfer_gb': sum(cost['usage']['data_transfer_gb'] for cost in api_key_costs),
                'total_compute_seconds': sum(cost['usage']['compute_seconds'] for cost in api_key_costs),
                'total_cost_usd': float(total_cost)
            }
        }
    
    def update_quota_costs(self, api_key_id: str, pricing_tier: str = 'starter'):
        """
        Update quota record with calculated costs.
        """
        # Get current month quota
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        quota = self.db.query(APIQuota).filter(
            and_(
                APIQuota.api_key_id == api_key_id,
                APIQuota.period_start == period_start
            )
        ).first()
        
        if not quota:
            return
        
        # Calculate costs based on current usage
        tier_config = self.PRICING_TIERS[pricing_tier]
        data_transfer_gb = quota.data_transfer_bytes / (1024 ** 3)
        
        costs = self._calculate_tier_costs(
            quota.requests_count,
            data_transfer_gb,
            quota.compute_time_seconds,
            tier_config
        )
        
        # Update quota with cost
        quota.cost_usd = costs['total']
        self.db.commit()
    
    def get_pricing_estimate(
        self,
        requests_per_month: int,
        data_transfer_gb_per_month: float,
        compute_seconds_per_month: float,
        pricing_tier: str = 'starter'
    ) -> Dict[str, Any]:
        """
        Get pricing estimate for projected usage.
        """
        if pricing_tier not in self.PRICING_TIERS:
            raise ValueError(f"Invalid pricing tier: {pricing_tier}")
        
        tier_config = self.PRICING_TIERS[pricing_tier]
        
        costs = self._calculate_tier_costs(
            requests_per_month,
            data_transfer_gb_per_month,
            compute_seconds_per_month,
            tier_config
        )
        
        return {
            'pricing_tier': pricing_tier,
            'projected_usage': {
                'requests_per_month': requests_per_month,
                'data_transfer_gb_per_month': data_transfer_gb_per_month,
                'compute_seconds_per_month': compute_seconds_per_month
            },
            'estimated_costs': costs,
            'monthly_estimate_usd': costs['total']
        }
    
    def get_all_pricing_tiers(self) -> Dict[str, Any]:
        """
        Get information about all available pricing tiers.
        """
        return {
            'pricing_tiers': self.PRICING_TIERS,
            'currency': 'USD',
            'billing_period': 'monthly'
        }
    
    def _calculate_tier_costs(
        self,
        requests: int,
        data_transfer_gb: float,
        compute_seconds: float,
        tier_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate costs based on tier configuration.
        """
        # Request costs
        if tier_config['requests_per_month'] is None:
            # Unlimited tier
            billable_requests = requests
        else:
            billable_requests = max(0, requests - tier_config['requests_per_month'])
        
        request_cost = billable_requests * tier_config['cost_per_request']
        
        # Data transfer costs
        billable_data_gb = max(0, data_transfer_gb - tier_config['data_transfer_gb_included'])
        data_transfer_cost = billable_data_gb * tier_config['cost_per_gb']
        
        # Compute costs
        billable_compute_seconds = max(0, compute_seconds - tier_config['compute_seconds_included'])
        compute_cost = billable_compute_seconds * tier_config['cost_per_compute_second']
        
        total_cost = request_cost + data_transfer_cost + compute_cost
        
        return {
            'requests': round(request_cost, 4),
            'data_transfer': round(data_transfer_cost, 4),
            'compute': round(compute_cost, 4),
            'total': round(total_cost, 4),
            'breakdown': {
                'billable_requests': billable_requests,
                'billable_data_gb': round(billable_data_gb, 4),
                'billable_compute_seconds': round(billable_compute_seconds, 2)
            }
        }
    
    def create_invoice(
        self,
        user_id: str,
        year: int,
        month: int,
        pricing_tier: str = 'starter'
    ) -> Dict[str, Any]:
        """
        Create a detailed invoice for a billing period.
        """
        bill = self.calculate_monthly_bill(user_id, year, month, pricing_tier)
        
        # Add invoice-specific information
        invoice_number = f"INV-{user_id[:8]}-{year}{month:02d}"
        due_date = datetime(year, month, 1) + timedelta(days=30)
        
        invoice = {
            'invoice_number': invoice_number,
            'invoice_date': datetime.utcnow().isoformat(),
            'due_date': due_date.isoformat(),
            'status': 'pending',
            **bill
        }
        
        return invoice