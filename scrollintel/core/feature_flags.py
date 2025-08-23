"""
Feature Flags System for Agent Steering System
Enables gradual rollout and canary deployments with real-time configuration updates.
"""

import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RolloutStrategy(Enum):
    """Rollout strategy types"""
    PERCENTAGE = "percentage"
    USER_GROUPS = "user_groups"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"

@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    enabled: bool
    rollout_percentage: int = 0
    target_groups: List[str] = None
    geographic_regions: List[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    description: str = ""
    created_by: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.target_groups is None:
            self.target_groups = []
        if self.geographic_regions is None:
            self.geographic_regions = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class FeatureFlagManager:
    """Manages feature flags for gradual rollout and canary deployments"""
    
    def __init__(self, redis_url: str = None):
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        self.cache_ttl = 300  # 5 minutes
        self.flags_cache = {}
        
    def create_flag(self, flag: FeatureFlag) -> bool:
        """Create a new feature flag"""
        try:
            flag_data = asdict(flag)
            # Convert datetime objects to ISO strings
            for key, value in flag_data.items():
                if isinstance(value, datetime):
                    flag_data[key] = value.isoformat()
            
            if self.redis_client:
                self.redis_client.hset(
                    "feature_flags", 
                    flag.name, 
                    json.dumps(flag_data)
                )
            else:
                self.flags_cache[flag.name] = flag_data
            
            logger.info(f"Created feature flag: {flag.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create feature flag {flag.name}: {str(e)}")
            return False
    
    def update_flag(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing feature flag"""
        try:
            flag_data = self.get_flag_data(name)
            if not flag_data:
                logger.error(f"Feature flag {name} not found")
                return False
            
            # Update fields
            flag_data.update(updates)
            flag_data["updated_at"] = datetime.now().isoformat()
            
            if self.redis_client:
                self.redis_client.hset(
                    "feature_flags", 
                    name, 
                    json.dumps(flag_data)
                )
            else:
                self.flags_cache[name] = flag_data
            
            logger.info(f"Updated feature flag: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update feature flag {name}: {str(e)}")
            return False
    
    def get_flag_data(self, name: str) -> Optional[Dict[str, Any]]:
        """Get feature flag data"""
        try:
            if self.redis_client:
                flag_json = self.redis_client.hget("feature_flags", name)
                if flag_json:
                    return json.loads(flag_json)
            else:
                return self.flags_cache.get(name)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get feature flag {name}: {str(e)}")
            return None
    
    def is_enabled(self, flag_name: str, user_id: str = None, 
                   user_groups: List[str] = None, 
                   geographic_region: str = None) -> bool:
        """Check if a feature flag is enabled for the given context"""
        try:
            flag_data = self.get_flag_data(flag_name)
            if not flag_data:
                logger.warning(f"Feature flag {flag_name} not found, defaulting to disabled")
                return False
            
            # Check if flag is globally disabled
            if not flag_data.get("enabled", False):
                return False
            
            # Check time-based rollout
            if not self._is_time_enabled(flag_data):
                return False
            
            # Check percentage rollout
            if not self._is_percentage_enabled(flag_data, user_id):
                return False
            
            # Check user group targeting
            if not self._is_user_group_enabled(flag_data, user_groups):
                return False
            
            # Check geographic targeting
            if not self._is_geographic_enabled(flag_data, geographic_region):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking feature flag {flag_name}: {str(e)}")
            return False
    
    def _is_time_enabled(self, flag_data: Dict[str, Any]) -> bool:
        """Check if flag is enabled based on time constraints"""
        start_time_str = flag_data.get("start_time")
        end_time_str = flag_data.get("end_time")
        
        if not start_time_str and not end_time_str:
            return True
        
        now = datetime.now()
        
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            if now < start_time:
                return False
        
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str)
            if now > end_time:
                return False
        
        return True
    
    def _is_percentage_enabled(self, flag_data: Dict[str, Any], user_id: str = None) -> bool:
        """Check if flag is enabled based on percentage rollout"""
        rollout_percentage = flag_data.get("rollout_percentage", 0)
        
        if rollout_percentage >= 100:
            return True
        
        if rollout_percentage <= 0:
            return False
        
        if not user_id:
            # If no user ID provided, use random sampling
            import random
            return random.randint(1, 100) <= rollout_percentage
        
        # Use consistent hashing based on user ID
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < rollout_percentage
    
    def _is_user_group_enabled(self, flag_data: Dict[str, Any], user_groups: List[str] = None) -> bool:
        """Check if flag is enabled based on user group targeting"""
        target_groups = flag_data.get("target_groups", [])
        
        if not target_groups:
            return True  # No targeting means enabled for all
        
        if not user_groups:
            return False  # User has no groups but flag targets specific groups
        
        # Check if user belongs to any target group
        return bool(set(user_groups) & set(target_groups))
    
    def _is_geographic_enabled(self, flag_data: Dict[str, Any], geographic_region: str = None) -> bool:
        """Check if flag is enabled based on geographic targeting"""
        target_regions = flag_data.get("geographic_regions", [])
        
        if not target_regions:
            return True  # No targeting means enabled for all regions
        
        if not geographic_region:
            return False  # No region provided but flag targets specific regions
        
        return geographic_region in target_regions
    
    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags"""
        try:
            if self.redis_client:
                flags = {}
                flag_data = self.redis_client.hgetall("feature_flags")
                for name, data in flag_data.items():
                    flags[name.decode()] = json.loads(data.decode())
                return flags
            else:
                return self.flags_cache.copy()
                
        except Exception as e:
            logger.error(f"Failed to get all feature flags: {str(e)}")
            return {}
    
    def delete_flag(self, name: str) -> bool:
        """Delete a feature flag"""
        try:
            if self.redis_client:
                self.redis_client.hdel("feature_flags", name)
            else:
                self.flags_cache.pop(name, None)
            
            logger.info(f"Deleted feature flag: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete feature flag {name}: {str(e)}")
            return False
    
    def gradual_rollout(self, flag_name: str, target_percentage: int, 
                       increment: int = 10, interval_minutes: int = 5) -> bool:
        """Perform gradual rollout of a feature flag"""
        try:
            flag_data = self.get_flag_data(flag_name)
            if not flag_data:
                logger.error(f"Feature flag {flag_name} not found")
                return False
            
            current_percentage = flag_data.get("rollout_percentage", 0)
            
            if current_percentage >= target_percentage:
                logger.info(f"Feature flag {flag_name} already at target percentage {target_percentage}%")
                return True
            
            # Calculate next percentage
            next_percentage = min(current_percentage + increment, target_percentage)
            
            # Update flag
            updates = {
                "rollout_percentage": next_percentage,
                "updated_at": datetime.now().isoformat()
            }
            
            if self.update_flag(flag_name, updates):
                logger.info(f"Rolled out {flag_name} to {next_percentage}%")
                
                # Schedule next rollout if not at target
                if next_percentage < target_percentage:
                    # In a real implementation, you would use a task scheduler like Celery
                    logger.info(f"Next rollout for {flag_name} scheduled in {interval_minutes} minutes")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to perform gradual rollout for {flag_name}: {str(e)}")
            return False
    
    def emergency_disable(self, flag_name: str, reason: str = "") -> bool:
        """Emergency disable a feature flag"""
        try:
            updates = {
                "enabled": False,
                "rollout_percentage": 0,
                "emergency_disabled": True,
                "emergency_reason": reason,
                "emergency_disabled_at": datetime.now().isoformat()
            }
            
            if self.update_flag(flag_name, updates):
                logger.warning(f"Emergency disabled feature flag {flag_name}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to emergency disable {flag_name}: {str(e)}")
            return False
    
    def get_flag_metrics(self, flag_name: str) -> Dict[str, Any]:
        """Get metrics for a feature flag"""
        try:
            flag_data = self.get_flag_data(flag_name)
            if not flag_data:
                return {}
            
            # In a real implementation, you would collect actual usage metrics
            metrics = {
                "flag_name": flag_name,
                "enabled": flag_data.get("enabled", False),
                "rollout_percentage": flag_data.get("rollout_percentage", 0),
                "target_groups": flag_data.get("target_groups", []),
                "created_at": flag_data.get("created_at"),
                "updated_at": flag_data.get("updated_at"),
                "estimated_users_affected": self._estimate_users_affected(flag_data),
                "rollout_status": self._get_rollout_status(flag_data)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics for {flag_name}: {str(e)}")
            return {}
    
    def _estimate_users_affected(self, flag_data: Dict[str, Any]) -> int:
        """Estimate number of users affected by flag"""
        # This is a simplified estimation
        # In a real implementation, you would use actual user data
        base_users = 10000  # Assume 10k total users
        rollout_percentage = flag_data.get("rollout_percentage", 0)
        
        return int(base_users * (rollout_percentage / 100))
    
    def _get_rollout_status(self, flag_data: Dict[str, Any]) -> str:
        """Get rollout status description"""
        if not flag_data.get("enabled", False):
            return "disabled"
        
        rollout_percentage = flag_data.get("rollout_percentage", 0)
        
        if rollout_percentage == 0:
            return "not_started"
        elif rollout_percentage < 100:
            return "in_progress"
        else:
            return "complete"

# Global feature flag manager instance
feature_flags = FeatureFlagManager()

def init_feature_flags(redis_url: str = None):
    """Initialize the global feature flag manager"""
    global feature_flags
    feature_flags = FeatureFlagManager(redis_url)

def is_feature_enabled(flag_name: str, user_id: str = None, 
                      user_groups: List[str] = None, 
                      geographic_region: str = None) -> bool:
    """Check if a feature is enabled (convenience function)"""
    return feature_flags.is_enabled(flag_name, user_id, user_groups, geographic_region)

def create_agent_steering_flags():
    """Create default feature flags for Agent Steering System"""
    flags = [
        FeatureFlag(
            name="agent_steering_system",
            enabled=True,
            rollout_percentage=0,
            description="Main Agent Steering System feature",
            created_by="deployment_system"
        ),
        FeatureFlag(
            name="orchestration_engine",
            enabled=True,
            rollout_percentage=0,
            description="Agent orchestration engine",
            created_by="deployment_system"
        ),
        FeatureFlag(
            name="intelligence_engine",
            enabled=True,
            rollout_percentage=0,
            description="Business intelligence engine",
            created_by="deployment_system"
        ),
        FeatureFlag(
            name="real_time_monitoring",
            enabled=True,
            rollout_percentage=100,
            description="Real-time monitoring and alerting",
            created_by="deployment_system"
        ),
        FeatureFlag(
            name="advanced_analytics",
            enabled=False,
            rollout_percentage=0,
            target_groups=["beta_users", "enterprise_customers"],
            description="Advanced analytics features",
            created_by="deployment_system"
        )
    ]
    
    for flag in flags:
        feature_flags.create_flag(flag)
    
    logger.info("Created default Agent Steering System feature flags")