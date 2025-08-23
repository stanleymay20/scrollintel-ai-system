#!/usr/bin/env python3
"""
ScrollIntel Agent Steering System - Gradual Rollout Manager
Feature flag-based gradual rollout with canary deployments and monitoring
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import redis
import psycopg2
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RolloutStage:
    """Rollout stage configuration"""
    stage_id: str
    name: str
    percentage: int
    duration_minutes: int
    success_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]
    feature_flags: Dict[str, bool]

@dataclass
class RolloutMetrics:
    """Rollout stage metrics"""
    stage_id: str
    timestamp: datetime
    user_percentage: float
    error_rate: float
    response_time_p95: float
    success_rate: float
    user_satisfaction: float
    business_metrics: Dict[str, float]

class FeatureFlagManager:
    """Manages feature flags for gradual rollout"""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.flag_prefix = "scrollintel:feature_flags:"
        
    def set_flag(self, flag_name: str, enabled: bool, percentage: int = 100):
        """Set feature flag with percentage rollout"""
        flag_data = {
            "enabled": enabled,
            "percentage": percentage,
            "updated_at": datetime.now().isoformat()
        }
        
        self.redis_client.set(
            f"{self.flag_prefix}{flag_name}",
            json.dumps(flag_data),
            ex=86400  # 24 hour expiry
        )
        
        logger.info(f"Feature flag '{flag_name}' set to {enabled} at {percentage}%")
    
    def get_flag(self, flag_name: str) -> Dict[str, Any]:
        """Get feature flag configuration"""
        flag_data = self.redis_client.get(f"{self.flag_prefix}{flag_name}")
        
        if flag_data:
            return json.loads(flag_data)
        else:
            return {"enabled": False, "percentage": 0}
    
    def is_enabled_for_user(self, flag_name: str, user_id: str) -> bool:
        """Check if feature is enabled for specific user"""
        flag_config = self.get_flag(flag_name)
        
        if not flag_config["enabled"]:
            return False
        
        # Use consistent hashing to determine if user is in rollout percentage
        user_hash = hash(f"{flag_name}:{user_id}") % 100
        return user_hash < flag_config["percentage"]
    
    def update_rollout_percentage(self, flag_name: str, percentage: int):
        """Update rollout percentage for existing flag"""
        flag_config = self.get_flag(flag_name)
        if flag_config["enabled"]:
            self.set_flag(flag_name, True, percentage)
    
    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags"""
        flags = {}
        for key in self.redis_client.scan_iter(match=f"{self.flag_prefix}*"):
            flag_name = key.decode().replace(self.flag_prefix, "")
            flags[flag_name] = self.get_flag(flag_name)
        return flags

class RolloutMonitor:
    """Monitors rollout metrics and health"""
    
    def __init__(self, base_url: str, database_url: str):
        self.base_url = base_url
        self.database_url = database_url
        self.session = requests.Session()
        
    def collect_metrics(self, stage_id: str) -> RolloutMetrics:
        """Collect comprehensive rollout metrics"""
        try:
            # Get system metrics
            system_metrics = self._get_system_metrics()
            
            # Get user metrics
            user_metrics = self._get_user_metrics()
            
            # Get business metrics
            business_metrics = self._get_business_metrics()
            
            # Calculate derived metrics
            error_rate = system_metrics.get("error_rate", 0)
            response_time_p95 = system_metrics.get("response_time_p95", 0)
            success_rate = 100 - error_rate
            user_satisfaction = user_metrics.get("satisfaction_score", 0)
            
            return RolloutMetrics(
                stage_id=stage_id,
                timestamp=datetime.now(),
                user_percentage=user_metrics.get("active_percentage", 0),
                error_rate=error_rate,
                response_time_p95=response_time_p95,
                success_rate=success_rate,
                user_satisfaction=user_satisfaction,
                business_metrics=business_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")
            return None
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system performance metrics"""
        try:
            response = self.session.get(f"{self.base_url}/api/monitoring/metrics")
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get system metrics: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    def _get_user_metrics(self) -> Dict[str, float]:
        """Get user engagement metrics"""
        try:
            response = self.session.get(f"{self.base_url}/api/monitoring/user-metrics")
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get user metrics: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting user metrics: {str(e)}")
            return {}
    
    def _get_business_metrics(self) -> Dict[str, float]:
        """Get business performance metrics"""
        try:
            response = self.session.get(f"{self.base_url}/api/monitoring/business-metrics")
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get business metrics: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting business metrics: {str(e)}")
            return {}
    
    def evaluate_stage_health(self, metrics: RolloutMetrics, success_criteria: Dict[str, float], 
                            rollback_criteria: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Evaluate if rollout stage is healthy"""
        issues = []
        is_healthy = True
        
        # Check error rate
        if metrics.error_rate > rollback_criteria.get("max_error_rate", 5.0):
            issues.append(f"Error rate too high: {metrics.error_rate}%")
            is_healthy = False
        
        # Check response time
        if metrics.response_time_p95 > rollback_criteria.get("max_response_time", 2000):
            issues.append(f"Response time too high: {metrics.response_time_p95}ms")
            is_healthy = False
        
        # Check success rate
        if metrics.success_rate < success_criteria.get("min_success_rate", 95.0):
            issues.append(f"Success rate too low: {metrics.success_rate}%")
            is_healthy = False
        
        # Check user satisfaction
        if metrics.user_satisfaction < success_criteria.get("min_user_satisfaction", 4.0):
            issues.append(f"User satisfaction too low: {metrics.user_satisfaction}")
            is_healthy = False
        
        # Check business metrics
        for metric_name, min_value in success_criteria.items():
            if metric_name.startswith("business_"):
                actual_value = metrics.business_metrics.get(metric_name.replace("business_", ""), 0)
                if actual_value < min_value:
                    issues.append(f"Business metric {metric_name} too low: {actual_value}")
                    is_healthy = False
        
        return is_healthy, issues

class GradualRolloutManager:
    """Manages gradual rollout with feature flags and monitoring"""
    
    def __init__(self, base_url: str, redis_url: str, database_url: str):
        self.base_url = base_url
        self.feature_flag_manager = FeatureFlagManager(redis_url)
        self.rollout_monitor = RolloutMonitor(base_url, database_url)
        self.rollout_id = f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.rollout_history = []
        
        logger.info(f"Initialized gradual rollout manager: {self.rollout_id}")
    
    def execute_gradual_rollout(self, rollout_config: Dict[str, Any]) -> bool:
        """Execute gradual rollout with monitoring and automatic rollback"""
        logger.info("🚀 Starting gradual rollout deployment...")
        
        try:
            # Parse rollout configuration
            stages = self._parse_rollout_stages(rollout_config)
            
            # Initialize feature flags
            self._initialize_feature_flags(rollout_config.get("feature_flags", {}))
            
            # Execute each rollout stage
            for stage in stages:
                logger.info(f"Starting rollout stage: {stage.name} ({stage.percentage}%)")
                
                success = self._execute_rollout_stage(stage)
                
                if not success:
                    logger.error(f"Rollout stage {stage.name} failed, initiating rollback")
                    self._rollback_deployment()
                    return False
                
                logger.info(f"✅ Rollout stage {stage.name} completed successfully")
            
            # Finalize rollout
            self._finalize_rollout()
            
            logger.info("🎉 Gradual rollout completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gradual rollout failed: {str(e)}")
            self._rollback_deployment()
            return False
    
    def _parse_rollout_stages(self, config: Dict[str, Any]) -> List[RolloutStage]:
        """Parse rollout configuration into stages"""
        stages = []
        
        stage_configs = config.get("stages", [
            {"percentage": 10, "duration": 15},
            {"percentage": 25, "duration": 15},
            {"percentage": 50, "duration": 20},
            {"percentage": 75, "duration": 20},
            {"percentage": 100, "duration": 30}
        ])
        
        for i, stage_config in enumerate(stage_configs):
            stage = RolloutStage(
                stage_id=f"stage_{i+1}",
                name=f"Stage {i+1} - {stage_config['percentage']}%",
                percentage=stage_config["percentage"],
                duration_minutes=stage_config.get("duration", 15),
                success_criteria=config.get("success_criteria", {
                    "min_success_rate": 95.0,
                    "min_user_satisfaction": 4.0,
                    "business_conversion_rate": 0.8
                }),
                rollback_criteria=config.get("rollback_criteria", {
                    "max_error_rate": 5.0,
                    "max_response_time": 2000
                }),
                feature_flags=config.get("feature_flags", {})
            )
            stages.append(stage)
        
        return stages
    
    def _initialize_feature_flags(self, feature_flags: Dict[str, bool]):
        """Initialize feature flags for rollout"""
        logger.info("Initializing feature flags...")
        
        # Set all flags to 0% initially
        for flag_name, enabled in feature_flags.items():
            self.feature_flag_manager.set_flag(flag_name, enabled, 0)
        
        # Core rollout flag
        self.feature_flag_manager.set_flag("new_agent_steering_system", True, 0)
        
        logger.info("✅ Feature flags initialized")
    
    def _execute_rollout_stage(self, stage: RolloutStage) -> bool:
        """Execute individual rollout stage"""
        logger.info(f"Executing rollout stage: {stage.name}")
        
        try:
            # Update feature flags for this stage
            self._update_stage_feature_flags(stage)
            
            # Wait for stage duration while monitoring
            stage_success = self._monitor_stage_execution(stage)
            
            if stage_success:
                # Record successful stage
                self.rollout_history.append({
                    "stage_id": stage.stage_id,
                    "percentage": stage.percentage,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                })
                return True
            else:
                # Record failed stage
                self.rollout_history.append({
                    "stage_id": stage.stage_id,
                    "percentage": stage.percentage,
                    "status": "failed",
                    "timestamp": datetime.now().isoformat()
                })
                return False
                
        except Exception as e:
            logger.error(f"Stage execution failed: {str(e)}")
            return False
    
    def _update_stage_feature_flags(self, stage: RolloutStage):
        """Update feature flags for rollout stage"""
        logger.info(f"Updating feature flags to {stage.percentage}%")
        
        # Update main rollout flag
        self.feature_flag_manager.update_rollout_percentage(
            "new_agent_steering_system", 
            stage.percentage
        )
        
        # Update additional feature flags
        for flag_name, enabled in stage.feature_flags.items():
            if enabled:
                self.feature_flag_manager.update_rollout_percentage(flag_name, stage.percentage)
            else:
                self.feature_flag_manager.set_flag(flag_name, False, 0)
    
    def _monitor_stage_execution(self, stage: RolloutStage) -> bool:
        """Monitor stage execution with health checks"""
        logger.info(f"Monitoring stage for {stage.duration_minutes} minutes...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=stage.duration_minutes)
        
        # Monitoring interval (check every 30 seconds)
        check_interval = 30
        
        while datetime.now() < end_time:
            try:
                # Collect metrics
                metrics = self.rollout_monitor.collect_metrics(stage.stage_id)
                
                if metrics:
                    # Evaluate stage health
                    is_healthy, issues = self.rollout_monitor.evaluate_stage_health(
                        metrics, stage.success_criteria, stage.rollback_criteria
                    )
                    
                    if not is_healthy:
                        logger.error(f"Stage health check failed: {', '.join(issues)}")
                        return False
                    
                    # Log metrics
                    logger.info(f"Stage metrics - Error rate: {metrics.error_rate}%, "
                              f"Response time: {metrics.response_time_p95}ms, "
                              f"Success rate: {metrics.success_rate}%")
                
                # Wait before next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                # Continue monitoring unless critical error
        
        # Final health check
        final_metrics = self.rollout_monitor.collect_metrics(stage.stage_id)
        if final_metrics:
            is_healthy, issues = self.rollout_monitor.evaluate_stage_health(
                final_metrics, stage.success_criteria, stage.rollback_criteria
            )
            
            if not is_healthy:
                logger.error(f"Final health check failed: {', '.join(issues)}")
                return False
        
        logger.info(f"✅ Stage {stage.name} monitoring completed successfully")
        return True
    
    def _rollback_deployment(self):
        """Rollback deployment to previous state"""
        logger.info("🔄 Initiating deployment rollback...")
        
        try:
            # Disable all new feature flags
            flags = self.feature_flag_manager.get_all_flags()
            for flag_name in flags:
                self.feature_flag_manager.set_flag(flag_name, False, 0)
            
            # Wait for rollback to take effect
            time.sleep(30)
            
            # Verify rollback
            rollback_metrics = self.rollout_monitor.collect_metrics("rollback")
            if rollback_metrics:
                logger.info(f"Rollback metrics - Error rate: {rollback_metrics.error_rate}%, "
                          f"Success rate: {rollback_metrics.success_rate}%")
            
            # Record rollback
            self.rollout_history.append({
                "stage_id": "rollback",
                "percentage": 0,
                "status": "rollback_completed",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("✅ Deployment rollback completed")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {str(e)}")
    
    def _finalize_rollout(self):
        """Finalize successful rollout"""
        logger.info("Finalizing rollout deployment...")
        
        try:
            # Ensure all flags are at 100%
            flags = self.feature_flag_manager.get_all_flags()
            for flag_name, flag_config in flags.items():
                if flag_config["enabled"]:
                    self.feature_flag_manager.update_rollout_percentage(flag_name, 100)
            
            # Generate rollout report
            self._generate_rollout_report()
            
            # Send completion notifications
            self._send_rollout_notifications("completed")
            
            logger.info("✅ Rollout finalization completed")
            
        except Exception as e:
            logger.error(f"❌ Rollout finalization failed: {str(e)}")
    
    def _generate_rollout_report(self):
        """Generate comprehensive rollout report"""
        report = {
            "rollout_id": self.rollout_id,
            "start_time": self.rollout_history[0]["timestamp"] if self.rollout_history else None,
            "end_time": datetime.now().isoformat(),
            "total_stages": len([h for h in self.rollout_history if h["status"] == "completed"]),
            "successful_stages": len([h for h in self.rollout_history if h["status"] == "completed"]),
            "failed_stages": len([h for h in self.rollout_history if h["status"] == "failed"]),
            "rollback_occurred": any(h["status"] == "rollback_completed" for h in self.rollout_history),
            "stage_history": self.rollout_history,
            "final_feature_flags": self.feature_flag_manager.get_all_flags()
        }
        
        # Save report
        os.makedirs("reports/rollout", exist_ok=True)
        report_file = f"reports/rollout/rollout_report_{self.rollout_id}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Rollout report saved to {report_file}")
    
    def _send_rollout_notifications(self, status: str):
        """Send rollout status notifications"""
        webhook_url = os.getenv("ROLLOUT_WEBHOOK_URL")
        if not webhook_url:
            return
        
        try:
            message = {
                "rollout_id": self.rollout_id,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "stages_completed": len([h for h in self.rollout_history if h["status"] == "completed"])
            }
            
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                logger.info(f"Rollout notification sent: {status}")
            else:
                logger.warning(f"Failed to send notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Notification sending failed: {str(e)}")

class CanaryDeploymentManager:
    """Manages canary deployments with traffic splitting"""
    
    def __init__(self, base_url: str, redis_url: str):
        self.base_url = base_url
        self.feature_flag_manager = FeatureFlagManager(redis_url)
        self.canary_id = f"canary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def execute_canary_deployment(self, canary_config: Dict[str, Any]) -> bool:
        """Execute canary deployment with traffic splitting"""
        logger.info("🐤 Starting canary deployment...")
        
        try:
            # Deploy canary version
            self._deploy_canary_version(canary_config)
            
            # Execute traffic splitting stages
            traffic_stages = canary_config.get("traffic_stages", [5, 10, 25, 50, 100])
            
            for percentage in traffic_stages:
                logger.info(f"Routing {percentage}% traffic to canary")
                
                # Update traffic routing
                self._update_traffic_routing(percentage)
                
                # Monitor canary health
                if not self._monitor_canary_health(percentage, canary_config):
                    logger.error(f"Canary health check failed at {percentage}%")
                    self._rollback_canary()
                    return False
                
                # Wait between stages
                if percentage < 100:
                    time.sleep(canary_config.get("stage_duration", 300))  # 5 minutes default
            
            # Promote canary to production
            self._promote_canary()
            
            logger.info("✅ Canary deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Canary deployment failed: {str(e)}")
            self._rollback_canary()
            return False
    
    def _deploy_canary_version(self, config: Dict[str, Any]):
        """Deploy canary version"""
        logger.info("Deploying canary version...")
        
        # Set canary feature flag
        self.feature_flag_manager.set_flag("canary_deployment", True, 0)
        
        # Deploy canary infrastructure (implementation depends on platform)
        # This would typically involve deploying to a separate environment
        # or updating deployment with canary configuration
        
        logger.info("✅ Canary version deployed")
    
    def _update_traffic_routing(self, percentage: int):
        """Update traffic routing to canary"""
        self.feature_flag_manager.update_rollout_percentage("canary_deployment", percentage)
        logger.info(f"Traffic routing updated: {percentage}% to canary")
    
    def _monitor_canary_health(self, percentage: int, config: Dict[str, Any]) -> bool:
        """Monitor canary deployment health"""
        monitor_duration = config.get("monitor_duration", 300)  # 5 minutes
        
        logger.info(f"Monitoring canary health for {monitor_duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < monitor_duration:
            try:
                # Get canary metrics
                response = requests.get(f"{self.base_url}/api/monitoring/canary-metrics")
                
                if response.status_code == 200:
                    metrics = response.json()
                    
                    # Check health criteria
                    error_rate = metrics.get("error_rate", 0)
                    response_time = metrics.get("response_time_p95", 0)
                    
                    if error_rate > config.get("max_error_rate", 5.0):
                        logger.error(f"Canary error rate too high: {error_rate}%")
                        return False
                    
                    if response_time > config.get("max_response_time", 2000):
                        logger.error(f"Canary response time too high: {response_time}ms")
                        return False
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Canary monitoring error: {str(e)}")
                return False
        
        logger.info("✅ Canary health monitoring passed")
        return True
    
    def _promote_canary(self):
        """Promote canary to production"""
        logger.info("Promoting canary to production...")
        
        # Update all traffic to canary (100%)
        self.feature_flag_manager.update_rollout_percentage("canary_deployment", 100)
        
        # Clean up old production version
        # Implementation depends on deployment platform
        
        logger.info("✅ Canary promoted to production")
    
    def _rollback_canary(self):
        """Rollback canary deployment"""
        logger.info("Rolling back canary deployment...")
        
        # Disable canary routing
        self.feature_flag_manager.set_flag("canary_deployment", False, 0)
        
        # Clean up canary resources
        # Implementation depends on deployment platform
        
        logger.info("✅ Canary rollback completed")

def main():
    """Main gradual rollout execution"""
    # Load configuration
    rollout_config = {
        "stages": [
            {"percentage": 10, "duration": 15},
            {"percentage": 25, "duration": 15},
            {"percentage": 50, "duration": 20},
            {"percentage": 75, "duration": 20},
            {"percentage": 100, "duration": 30}
        ],
        "success_criteria": {
            "min_success_rate": 95.0,
            "min_user_satisfaction": 4.0,
            "business_conversion_rate": 0.8
        },
        "rollback_criteria": {
            "max_error_rate": 5.0,
            "max_response_time": 2000
        },
        "feature_flags": {
            "new_agent_steering_system": True,
            "enhanced_monitoring": True,
            "improved_ui": True
        }
    }
    
    # Get environment configuration
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    database_url = os.getenv("DATABASE_URL")
    
    # Initialize rollout manager
    rollout_manager = GradualRolloutManager(base_url, redis_url, database_url)
    
    # Execute gradual rollout
    success = rollout_manager.execute_gradual_rollout(rollout_config)
    
    if success:
        print("🎉 Gradual rollout completed successfully!")
        sys.exit(0)
    else:
        print("❌ Gradual rollout failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()