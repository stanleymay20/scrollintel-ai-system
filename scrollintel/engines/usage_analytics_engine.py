"""
Usage Analytics Engine for ScrollIntel Launch MVP.
Provides comprehensive analytics and reporting for API usage.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
import pandas as pd
from collections import defaultdict

from ..models.api_key_models import APIKey, APIUsage, APIQuota, RateLimitRecord
from ..models.database import User


class UsageAnalyticsEngine:
    """
    Engine for generating usage analytics and reports.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_overview(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive usage overview for a user.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get user's API keys
        api_keys = self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).all()
        
        if not api_keys:
            return {
                'total_api_keys': 0,
                'active_api_keys': 0,
                'total_requests': 0,
                'successful_requests': 0,
                'error_rate': 0.0,
                'average_response_time': 0.0,
                'data_transfer_gb': 0.0,
                'top_endpoints': [],
                'daily_usage': [],
                'error_breakdown': []
            }
        
        api_key_ids = [str(key.id) for key in api_keys]
        
        # Basic statistics
        total_api_keys = len(api_keys)
        active_api_keys = len([key for key in api_keys if key.is_active])
        
        # Usage statistics
        usage_query = self.db.query(APIUsage).filter(
            and_(
                APIUsage.user_id == user_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        )
        
        total_requests = usage_query.count()
        successful_requests = usage_query.filter(
            APIUsage.status_code < 400
        ).count()
        
        error_rate = (total_requests - successful_requests) / max(total_requests, 1) * 100
        
        # Average response time
        avg_response_time = self.db.query(
            func.avg(APIUsage.response_time_ms)
        ).filter(
            and_(
                APIUsage.user_id == user_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).scalar() or 0.0
        
        # Data transfer
        data_transfer_bytes = self.db.query(
            func.sum(APIUsage.request_size_bytes + APIUsage.response_size_bytes)
        ).filter(
            and_(
                APIUsage.user_id == user_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date,
                APIUsage.request_size_bytes.isnot(None),
                APIUsage.response_size_bytes.isnot(None)
            )
        ).scalar() or 0
        
        data_transfer_gb = data_transfer_bytes / (1024 ** 3)
        
        # Top endpoints
        top_endpoints = self.db.query(
            APIUsage.endpoint,
            func.count(APIUsage.id).label('count'),
            func.avg(APIUsage.response_time_ms).label('avg_response_time')
        ).filter(
            and_(
                APIUsage.user_id == user_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).group_by(APIUsage.endpoint).order_by(
            desc('count')
        ).limit(10).all()
        
        # Daily usage
        daily_usage = self.db.query(
            func.date(APIUsage.timestamp).label('date'),
            func.count(APIUsage.id).label('requests'),
            func.sum(func.case([(APIUsage.status_code < 400, 1)], else_=0)).label('successful'),
            func.avg(APIUsage.response_time_ms).label('avg_response_time')
        ).filter(
            and_(
                APIUsage.user_id == user_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).group_by(func.date(APIUsage.timestamp)).order_by('date').all()
        
        # Error breakdown
        error_breakdown = self.db.query(
            APIUsage.status_code,
            func.count(APIUsage.id).label('count')
        ).filter(
            and_(
                APIUsage.user_id == user_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date,
                APIUsage.status_code >= 400
            )
        ).group_by(APIUsage.status_code).order_by(desc('count')).all()
        
        return {
            'total_api_keys': total_api_keys,
            'active_api_keys': active_api_keys,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'error_rate': round(error_rate, 2),
            'average_response_time': round(avg_response_time, 2),
            'data_transfer_gb': round(data_transfer_gb, 4),
            'top_endpoints': [
                {
                    'endpoint': endpoint,
                    'count': count,
                    'avg_response_time': round(avg_response_time, 2)
                }
                for endpoint, count, avg_response_time in top_endpoints
            ],
            'daily_usage': [
                {
                    'date': date.isoformat(),
                    'requests': requests,
                    'successful': successful,
                    'error_rate': round((requests - successful) / max(requests, 1) * 100, 2),
                    'avg_response_time': round(avg_response_time, 2)
                }
                for date, requests, successful, avg_response_time in daily_usage
            ],
            'error_breakdown': [
                {
                    'status_code': status_code,
                    'count': count,
                    'percentage': round(count / max(total_requests, 1) * 100, 2)
                }
                for status_code, count in error_breakdown
            ]
        }
    
    def get_api_key_analytics(
        self,
        api_key_id: str,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific API key.
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Verify API key belongs to user
        api_key = self.db.query(APIKey).filter(
            and_(
                APIKey.id == api_key_id,
                APIKey.user_id == user_id
            )
        ).first()
        
        if not api_key:
            return {}
        
        # Usage statistics
        usage_query = self.db.query(APIUsage).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        )
        
        total_requests = usage_query.count()
        successful_requests = usage_query.filter(
            APIUsage.status_code < 400
        ).count()
        
        # Performance metrics
        performance_stats = self.db.query(
            func.avg(APIUsage.response_time_ms).label('avg_response_time'),
            func.min(APIUsage.response_time_ms).label('min_response_time'),
            func.max(APIUsage.response_time_ms).label('max_response_time'),
            func.percentile_cont(0.5).within_group(APIUsage.response_time_ms).label('median_response_time'),
            func.percentile_cont(0.95).within_group(APIUsage.response_time_ms).label('p95_response_time')
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).first()
        
        # Hourly usage pattern
        hourly_usage = self.db.query(
            func.extract('hour', APIUsage.timestamp).label('hour'),
            func.count(APIUsage.id).label('requests'),
            func.avg(APIUsage.response_time_ms).label('avg_response_time')
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).group_by(func.extract('hour', APIUsage.timestamp)).order_by('hour').all()
        
        # Geographic distribution (by IP)
        geographic_usage = self.db.query(
            APIUsage.ip_address,
            func.count(APIUsage.id).label('requests')
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date,
                APIUsage.ip_address.isnot(None)
            )
        ).group_by(APIUsage.ip_address).order_by(desc('requests')).limit(20).all()
        
        # User agent analysis
        user_agent_stats = self.db.query(
            APIUsage.user_agent,
            func.count(APIUsage.id).label('requests')
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date,
                APIUsage.user_agent.isnot(None)
            )
        ).group_by(APIUsage.user_agent).order_by(desc('requests')).limit(10).all()
        
        return {
            'api_key_info': {
                'id': str(api_key.id),
                'name': api_key.name,
                'created_at': api_key.created_at.isoformat(),
                'last_used': api_key.last_used.isoformat() if api_key.last_used else None
            },
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'error_rate': round((total_requests - successful_requests) / max(total_requests, 1) * 100, 2),
                'avg_response_time': round(performance_stats.avg_response_time or 0, 2),
                'min_response_time': round(performance_stats.min_response_time or 0, 2),
                'max_response_time': round(performance_stats.max_response_time or 0, 2),
                'median_response_time': round(performance_stats.median_response_time or 0, 2),
                'p95_response_time': round(performance_stats.p95_response_time or 0, 2)
            },
            'hourly_pattern': [
                {
                    'hour': int(hour),
                    'requests': requests,
                    'avg_response_time': round(avg_response_time, 2)
                }
                for hour, requests, avg_response_time in hourly_usage
            ],
            'geographic_distribution': [
                {
                    'ip_address': ip_address,
                    'requests': requests
                }
                for ip_address, requests in geographic_usage
            ],
            'user_agents': [
                {
                    'user_agent': user_agent[:100] + '...' if len(user_agent) > 100 else user_agent,
                    'requests': requests
                }
                for user_agent, requests in user_agent_stats
            ]
        }
    
    def get_quota_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get current quota status for all user's API keys.
        """
        # Get current month period
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get user's API keys
        api_keys = self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).all()
        
        quota_status = []
        
        for api_key in api_keys:
            # Get quota for current period
            quota = self.db.query(APIQuota).filter(
                and_(
                    APIQuota.api_key_id == api_key.id,
                    APIQuota.period_start == period_start
                )
            ).first()
            
            if quota:
                quota_info = {
                    'api_key_id': str(api_key.id),
                    'api_key_name': api_key.name,
                    'period_start': quota.period_start.isoformat(),
                    'period_end': quota.period_end.isoformat(),
                    'requests': {
                        'used': quota.requests_count,
                        'limit': quota.requests_limit,
                        'percentage': quota.get_usage_percentage('requests')
                    },
                    'data_transfer': {
                        'used_bytes': quota.data_transfer_bytes,
                        'limit_bytes': quota.data_transfer_limit_bytes,
                        'percentage': quota.get_usage_percentage('data_transfer')
                    },
                    'compute_time': {
                        'used_seconds': quota.compute_time_seconds,
                        'limit_seconds': quota.compute_time_limit_seconds,
                        'percentage': quota.get_usage_percentage('compute_time')
                    },
                    'cost': {
                        'used_usd': quota.cost_usd,
                        'limit_usd': quota.cost_limit_usd,
                        'percentage': quota.get_usage_percentage('cost')
                    },
                    'is_exceeded': quota.is_exceeded,
                    'exceeded_at': quota.exceeded_at.isoformat() if quota.exceeded_at else None
                }
            else:
                quota_info = {
                    'api_key_id': str(api_key.id),
                    'api_key_name': api_key.name,
                    'period_start': period_start.isoformat(),
                    'period_end': None,
                    'requests': {'used': 0, 'limit': api_key.quota_requests_per_month, 'percentage': 0.0},
                    'data_transfer': {'used_bytes': 0, 'limit_bytes': None, 'percentage': 0.0},
                    'compute_time': {'used_seconds': 0.0, 'limit_seconds': None, 'percentage': 0.0},
                    'cost': {'used_usd': 0.0, 'limit_usd': None, 'percentage': 0.0},
                    'is_exceeded': False,
                    'exceeded_at': None
                }
            
            quota_status.append(quota_info)
        
        return {
            'user_id': user_id,
            'current_period': period_start.isoformat(),
            'api_keys': quota_status
        }
    
    def generate_usage_report(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive usage report.
        """
        # Get user overview
        days = (end_date - start_date).days
        overview = self.get_user_overview(user_id, days)
        
        # Get detailed analytics for each API key
        api_keys = self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).all()
        
        api_key_details = []
        for api_key in api_keys:
            analytics = self.get_api_key_analytics(
                str(api_key.id), user_id, start_date, end_date
            )
            if analytics:
                api_key_details.append(analytics)
        
        # Get quota status
        quota_status = self.get_quota_status(user_id)
        
        report = {
            'report_metadata': {
                'user_id': user_id,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'generated_at': datetime.utcnow().isoformat(),
                'period_days': days
            },
            'overview': overview,
            'api_key_details': api_key_details,
            'quota_status': quota_status
        }
        
        return report