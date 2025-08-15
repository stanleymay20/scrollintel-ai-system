"""Usage tracking system for comprehensive data access and modification analytics."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from ..models.governance_models import (
    UsageMetrics, AuditEvent, AuditEventType, User
)
from ..models.governance_database import (
    UsageMetricsModel, AuditEventModel, UserModel, DataCatalogEntryModel
)
from ..models.database import get_db_session
from ..core.exceptions import AIDataReadinessError


logger = logging.getLogger(__name__)


class UsageTrackerError(AIDataReadinessError):
    """Exception raised for usage tracker errors."""
    pass


class UsageTracker:
    """Comprehensive usage tracking and analytics system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def track_data_access(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str = "read",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track data access event."""
        try:
            # This would typically be called by the audit logger
            # but we can also track additional usage-specific metrics
            
            with get_db_session() as session:
                # Update real-time usage counters
                self._update_realtime_counters(
                    session, resource_id, resource_type, user_id, action
                )
                
                # Track session-based metrics
                if session_id:
                    self._track_session_metrics(
                        session, session_id, user_id, resource_id, resource_type, action
                    )
                
                session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error tracking data access: {str(e)}")
            return False
    
    def get_usage_analytics(
        self,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        aggregation_level: str = "daily"  # hourly, daily, weekly, monthly
    ) -> Dict[str, Any]:
        """Get comprehensive usage analytics."""
        try:
            with get_db_session() as session:
                # Build base query
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.event_type.in_([
                        AuditEventType.DATA_ACCESS.value,
                        AuditEventType.DATA_MODIFICATION.value
                    ])
                )
                
                # Apply filters
                if resource_id:
                    query = query.filter(AuditEventModel.resource_id == resource_id)
                
                if resource_type:
                    query = query.filter(AuditEventModel.resource_type == resource_type)
                
                if user_id:
                    query = query.filter(AuditEventModel.user_id == user_id)
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                events = query.all()
                
                # Calculate analytics
                analytics = {
                    'summary': self._calculate_usage_summary(events),
                    'trends': self._calculate_usage_trends(events, aggregation_level),
                    'user_analytics': self._calculate_user_analytics(events),
                    'resource_analytics': self._calculate_resource_analytics(events),
                    'access_patterns': self._calculate_access_patterns(events),
                    'performance_metrics': self._calculate_performance_metrics(events)
                }
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"Error getting usage analytics: {str(e)}")
            raise UsageTrackerError(f"Failed to get usage analytics: {str(e)}")
    
    def get_user_activity_report(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Get detailed user activity report."""
        try:
            with get_db_session() as session:
                # Get user information
                user = session.query(UserModel).filter(UserModel.id == user_id).first()
                if not user:
                    raise UsageTrackerError(f"User {user_id} not found")
                
                # Build query for user events
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.user_id == user_id
                )
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                events = query.order_by(desc(AuditEventModel.timestamp)).all()
                
                # Calculate user-specific metrics
                report = {
                    'user_info': {
                        'id': user_id,
                        'username': user.username,
                        'email': user.email,
                        'department': user.department,
                        'role': user.role
                    },
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'activity_summary': self._calculate_user_activity_summary(events),
                    'resource_usage': self._calculate_user_resource_usage(events),
                    'behavioral_patterns': self._calculate_user_behavioral_patterns(events),
                    'compliance_metrics': self._calculate_user_compliance_metrics(events),
                    'risk_indicators': self._calculate_user_risk_indicators(events)
                }
                
                if include_details:
                    report['detailed_events'] = [
                        {
                            'timestamp': event.timestamp,
                            'event_type': event.event_type,
                            'action': event.action,
                            'resource_id': event.resource_id,
                            'resource_type': event.resource_type,
                            'success': event.success,
                            'details': event.details
                        }
                        for event in events[:1000]  # Limit to recent 1000 events
                    ]
                
                return report
                
        except Exception as e:
            self.logger.error(f"Error getting user activity report: {str(e)}")
            raise UsageTrackerError(f"Failed to get user activity report: {str(e)}")
    
    def get_resource_usage_report(
        self,
        resource_id: str,
        resource_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_user_breakdown: bool = True
    ) -> Dict[str, Any]:
        """Get detailed resource usage report."""
        try:
            with get_db_session() as session:
                # Get resource information
                resource = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == resource_id
                ).first()
                
                # Build query for resource events
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.resource_id == resource_id,
                    AuditEventModel.resource_type == resource_type
                )
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                events = query.all()
                
                report = {
                    'resource_info': {
                        'id': resource_id,
                        'type': resource_type,
                        'name': resource.name if resource else 'Unknown',
                        'classification': resource.classification.value if resource else 'Unknown',
                        'owner': resource.owner if resource else 'Unknown'
                    },
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'usage_summary': self._calculate_resource_usage_summary(events),
                    'access_patterns': self._calculate_resource_access_patterns(events),
                    'performance_metrics': self._calculate_resource_performance_metrics(events),
                    'security_metrics': self._calculate_resource_security_metrics(events)
                }
                
                if include_user_breakdown:
                    report['user_breakdown'] = self._calculate_resource_user_breakdown(events)
                
                return report
                
        except Exception as e:
            self.logger.error(f"Error getting resource usage report: {str(e)}")
            raise UsageTrackerError(f"Failed to get resource usage report: {str(e)}")
    
    def get_system_usage_overview(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get system-wide usage overview."""
        try:
            with get_db_session() as session:
                # Build base query
                query = session.query(AuditEventModel)
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                events = query.all()
                
                # Get additional metrics
                total_users = session.query(UserModel).filter(UserModel.is_active == True).count()
                total_resources = session.query(DataCatalogEntryModel).count()
                
                overview = {
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'system_metrics': {
                        'total_users': total_users,
                        'total_resources': total_resources,
                        'total_events': len(events)
                    },
                    'activity_summary': self._calculate_system_activity_summary(events),
                    'top_users': self._calculate_top_users(events),
                    'top_resources': self._calculate_top_resources(events),
                    'security_overview': self._calculate_system_security_overview(events),
                    'compliance_overview': self._calculate_system_compliance_overview(events)
                }
                
                return overview
                
        except Exception as e:
            self.logger.error(f"Error getting system usage overview: {str(e)}")
            raise UsageTrackerError(f"Failed to get system usage overview: {str(e)}")
    
    def generate_usage_trends(
        self,
        metric_type: str = "access_count",  # access_count, unique_users, data_volume
        resource_type: Optional[str] = None,
        time_period: str = "30d",  # 7d, 30d, 90d, 1y
        granularity: str = "daily"  # hourly, daily, weekly
    ) -> Dict[str, Any]:
        """Generate usage trends for visualization."""
        try:
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            elif time_period == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)
            
            with get_db_session() as session:
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.timestamp >= start_date,
                    AuditEventModel.timestamp <= end_date
                )
                
                if resource_type:
                    query = query.filter(AuditEventModel.resource_type == resource_type)
                
                events = query.all()
                
                # Generate trend data based on granularity
                trends = self._generate_trend_data(events, start_date, end_date, granularity, metric_type)
                
                return {
                    'metric_type': metric_type,
                    'time_period': time_period,
                    'granularity': granularity,
                    'resource_type': resource_type,
                    'trends': trends,
                    'summary': {
                        'total_data_points': len(trends),
                        'trend_direction': self._calculate_trend_direction(trends),
                        'peak_value': max(trends, key=lambda x: x['value']) if trends else None,
                        'average_value': sum(t['value'] for t in trends) / len(trends) if trends else 0
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error generating usage trends: {str(e)}")
            raise UsageTrackerError(f"Failed to generate usage trends: {str(e)}")
    
    def _update_realtime_counters(
        self,
        session: Session,
        resource_id: str,
        resource_type: str,
        user_id: str,
        action: str
    ) -> None:
        """Update real-time usage counters."""
        # This would update in-memory counters or cache for real-time metrics
        # For now, we'll just log the update
        self.logger.debug(f"Updated real-time counter: {resource_type}:{resource_id} by {user_id}")
    
    def _track_session_metrics(
        self,
        session: Session,
        session_id: str,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str
    ) -> None:
        """Track session-based metrics."""
        # This would track session duration, resources accessed per session, etc.
        self.logger.debug(f"Tracked session metric: {session_id}")
    
    def _calculate_usage_summary(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate usage summary statistics."""
        total_events = len(events)
        unique_users = len(set(event.user_id for event in events))
        unique_resources = len(set(f"{event.resource_type}:{event.resource_id}" 
                                 for event in events if event.resource_id))
        
        # Group by event type
        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type] += 1
        
        # Group by action
        actions = defaultdict(int)
        for event in events:
            actions[event.action] += 1
        
        return {
            'total_events': total_events,
            'unique_users': unique_users,
            'unique_resources': unique_resources,
            'event_types': dict(event_types),
            'actions': dict(actions),
            'success_rate': sum(1 for event in events if event.success) / total_events if total_events > 0 else 0
        }
    
    def _calculate_usage_trends(
        self, 
        events: List[AuditEventModel], 
        aggregation_level: str
    ) -> List[Dict[str, Any]]:
        """Calculate usage trends over time."""
        if not events:
            return []
        
        # Group events by time period
        time_groups = defaultdict(int)
        
        for event in events:
            if aggregation_level == "hourly":
                time_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            elif aggregation_level == "daily":
                time_key = event.timestamp.strftime("%Y-%m-%d")
            elif aggregation_level == "weekly":
                # Get Monday of the week
                monday = event.timestamp - timedelta(days=event.timestamp.weekday())
                time_key = monday.strftime("%Y-%m-%d")
            elif aggregation_level == "monthly":
                time_key = event.timestamp.strftime("%Y-%m")
            else:
                time_key = event.timestamp.strftime("%Y-%m-%d")
            
            time_groups[time_key] += 1
        
        # Convert to list of trend points
        trends = [
            {'timestamp': time_key, 'count': count}
            for time_key, count in sorted(time_groups.items())
        ]
        
        return trends
    
    def _calculate_user_analytics(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate user-specific analytics."""
        user_stats = defaultdict(lambda: {'events': 0, 'resources': set(), 'actions': set()})
        
        for event in events:
            user_stats[event.user_id]['events'] += 1
            if event.resource_id:
                user_stats[event.user_id]['resources'].add(f"{event.resource_type}:{event.resource_id}")
            user_stats[event.user_id]['actions'].add(event.action)
        
        # Convert to serializable format
        user_analytics = {}
        for user_id, stats in user_stats.items():
            user_analytics[user_id] = {
                'total_events': stats['events'],
                'unique_resources': len(stats['resources']),
                'unique_actions': len(stats['actions'])
            }
        
        return user_analytics
    
    def _calculate_resource_analytics(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate resource-specific analytics."""
        resource_stats = defaultdict(lambda: {'events': 0, 'users': set(), 'actions': set()})
        
        for event in events:
            if event.resource_id:
                resource_key = f"{event.resource_type}:{event.resource_id}"
                resource_stats[resource_key]['events'] += 1
                resource_stats[resource_key]['users'].add(event.user_id)
                resource_stats[resource_key]['actions'].add(event.action)
        
        # Convert to serializable format
        resource_analytics = {}
        for resource_key, stats in resource_stats.items():
            resource_analytics[resource_key] = {
                'total_events': stats['events'],
                'unique_users': len(stats['users']),
                'unique_actions': len(stats['actions'])
            }
        
        return resource_analytics
    
    def _calculate_access_patterns(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate access patterns."""
        hourly_pattern = defaultdict(int)
        daily_pattern = defaultdict(int)
        
        for event in events:
            hourly_pattern[event.timestamp.hour] += 1
            daily_pattern[event.timestamp.strftime("%A")] += 1
        
        return {
            'hourly_distribution': dict(hourly_pattern),
            'daily_distribution': dict(daily_pattern)
        }
    
    def _calculate_performance_metrics(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        # This would include response times, throughput, etc.
        # For now, return basic metrics
        return {
            'total_events': len(events),
            'events_per_hour': len(events) / 24 if events else 0,  # Assuming 24-hour period
            'success_rate': sum(1 for event in events if event.success) / len(events) if events else 0
        }
    
    def _calculate_user_activity_summary(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate user activity summary."""
        return self._calculate_usage_summary(events)
    
    def _calculate_user_resource_usage(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate user resource usage patterns."""
        resource_usage = defaultdict(int)
        
        for event in events:
            if event.resource_id:
                resource_key = f"{event.resource_type}:{event.resource_id}"
                resource_usage[resource_key] += 1
        
        # Sort by usage count
        sorted_usage = sorted(resource_usage.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_resources_accessed': len(resource_usage),
            'most_accessed_resources': sorted_usage[:10],
            'resource_distribution': dict(resource_usage)
        }
    
    def _calculate_user_behavioral_patterns(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate user behavioral patterns."""
        return self._calculate_access_patterns(events)
    
    def _calculate_user_compliance_metrics(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate user compliance metrics."""
        compliance_events = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_CHECK.value]
        
        return {
            'compliance_checks': len(compliance_events),
            'compliance_violations': sum(1 for event in compliance_events if not event.success)
        }
    
    def _calculate_user_risk_indicators(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate user risk indicators."""
        failed_events = [e for e in events if not e.success]
        unusual_hours = [e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22]
        
        return {
            'failed_attempts': len(failed_events),
            'unusual_hour_access': len(unusual_hours),
            'risk_score': min(100, (len(failed_events) * 10) + (len(unusual_hours) * 5))
        }
    
    def _calculate_resource_usage_summary(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate resource usage summary."""
        return self._calculate_usage_summary(events)
    
    def _calculate_resource_access_patterns(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate resource access patterns."""
        return self._calculate_access_patterns(events)
    
    def _calculate_resource_performance_metrics(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate resource performance metrics."""
        return self._calculate_performance_metrics(events)
    
    def _calculate_resource_security_metrics(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate resource security metrics."""
        failed_access = [e for e in events if not e.success and e.event_type == AuditEventType.DATA_ACCESS.value]
        
        return {
            'failed_access_attempts': len(failed_access),
            'security_incidents': len([e for e in failed_access if 'unauthorized' in str(e.details).lower()])
        }
    
    def _calculate_resource_user_breakdown(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate resource user breakdown."""
        user_breakdown = defaultdict(lambda: {'access_count': 0, 'actions': set()})
        
        for event in events:
            user_breakdown[event.user_id]['access_count'] += 1
            user_breakdown[event.user_id]['actions'].add(event.action)
        
        # Convert to serializable format
        breakdown = {}
        for user_id, stats in user_breakdown.items():
            breakdown[user_id] = {
                'access_count': stats['access_count'],
                'unique_actions': len(stats['actions'])
            }
        
        return breakdown
    
    def _calculate_system_activity_summary(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate system activity summary."""
        return self._calculate_usage_summary(events)
    
    def _calculate_top_users(self, events: List[AuditEventModel]) -> List[Dict[str, Any]]:
        """Calculate top users by activity."""
        user_counts = defaultdict(int)
        
        for event in events:
            user_counts[event.user_id] += 1
        
        sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'user_id': user_id, 'event_count': count}
            for user_id, count in sorted_users[:10]
        ]
    
    def _calculate_top_resources(self, events: List[AuditEventModel]) -> List[Dict[str, Any]]:
        """Calculate top resources by access."""
        resource_counts = defaultdict(int)
        
        for event in events:
            if event.resource_id:
                resource_key = f"{event.resource_type}:{event.resource_id}"
                resource_counts[resource_key] += 1
        
        sorted_resources = sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'resource': resource, 'access_count': count}
            for resource, count in sorted_resources[:10]
        ]
    
    def _calculate_system_security_overview(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate system security overview."""
        failed_events = [e for e in events if not e.success]
        
        return {
            'total_security_events': len(failed_events),
            'security_incident_rate': len(failed_events) / len(events) if events else 0
        }
    
    def _calculate_system_compliance_overview(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate system compliance overview."""
        compliance_events = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_CHECK.value]
        
        return {
            'compliance_checks': len(compliance_events),
            'compliance_violations': sum(1 for event in compliance_events if not event.success)
        }
    
    def _generate_trend_data(
        self,
        events: List[AuditEventModel],
        start_date: datetime,
        end_date: datetime,
        granularity: str,
        metric_type: str
    ) -> List[Dict[str, Any]]:
        """Generate trend data for visualization."""
        # Group events by time period
        time_groups = defaultdict(lambda: {'events': [], 'users': set()})
        
        for event in events:
            if granularity == "hourly":
                time_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            elif granularity == "daily":
                time_key = event.timestamp.strftime("%Y-%m-%d")
            elif granularity == "weekly":
                monday = event.timestamp - timedelta(days=event.timestamp.weekday())
                time_key = monday.strftime("%Y-%m-%d")
            else:
                time_key = event.timestamp.strftime("%Y-%m-%d")
            
            time_groups[time_key]['events'].append(event)
            time_groups[time_key]['users'].add(event.user_id)
        
        # Calculate metric values
        trends = []
        for time_key, data in sorted(time_groups.items()):
            if metric_type == "access_count":
                value = len(data['events'])
            elif metric_type == "unique_users":
                value = len(data['users'])
            elif metric_type == "data_volume":
                # This would require additional data about data volume
                value = len(data['events'])  # Placeholder
            else:
                value = len(data['events'])
            
            trends.append({
                'timestamp': time_key,
                'value': value
            })
        
        return trends
    
    def _calculate_trend_direction(self, trends: List[Dict[str, Any]]) -> str:
        """Calculate overall trend direction."""
        if len(trends) < 2:
            return "stable"
        
        first_half = trends[:len(trends)//2]
        second_half = trends[len(trends)//2:]
        
        first_avg = sum(t['value'] for t in first_half) / len(first_half)
        second_avg = sum(t['value'] for t in second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"