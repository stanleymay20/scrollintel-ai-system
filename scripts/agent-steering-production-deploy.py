#!/usr/bin/env python3
"""
Agent Steering System Production Deployment Script
Deploys the enterprise-grade Agent Steering System to production with full monitoring,
user acceptance testing, gradual rollout, and comprehensive support documentation.
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict,Tuple
sycopg2
import redis
from dataclasses imps

# Configure logging
logging.basicC
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s -
    h[
 log'),
        logging.StreamHandler()

)
logger = logging.getLog)

@dataclass
class DeploymentConfig:
    """Production deployment confi"
    environment: str = "production"
    enable_blue_green: bool = True
    enable_canary: bool = True
    enable_feature_flags: bool = True
    rollback_on_failure: bool = True
    health_check_timeout: int = 300
    user_acceptance_test: bool 
    gradual_rollout_percentage: int = 0
rue
    backupTrue

class ProductionDeployer:
    """Enterpr
    
    def __init__(self, onfig):
        self.config = config
        self.deployment_id = f"dep"
        self.backup_dir = Patd}")
 {
            "deployment_id": self.depl
            "status": "initializing",
    
            "components": {},
            "health_checks": {},
            "user_acceptance": {},
            "rollout_progress": 0
        }
        
    def ool:
        """Execute complete p"
        try:
            d}")
            
            ation
            self._update_status("pre_deployment_n")
            if not self._pre_deployment_validatio
                raise Exception("Pre-deployment validation f
            
            # Phstem
            if self.config.backup_before_deploy:
                self._update_status("backup")
                if not self._create_system_backup():
                    raise Ex
            
            # Phase 3: Deploy infrastructure
            self._update_status("infrastructure_de
            if not self._deploy_infrastructure():
                raise Except")
            
            # Phase 4: Deploy Agent Steering S
            self._update_status("component_deployme)
            if not self._deploy_agent_steering_components():
                raise Exception("Component deployment failed")
            
            # Phase oring
            self._update_status("healt)
            if not self._comprehensive_health_check):
                raise Exception("Health checks failed")
            
            # Phng
            if self.config.user_acceptance_test:
                self._update_status("user_acceptance_t
                if not self._run_user_acceptance_tests():
                    raise Exed")
            
            # Phase 7: Gradual rollout with feature flags
            if self.cony:
            )
                if not self._e:
                    raise Exception("Gradual rollout failed")
            
            # Phase 8: Full production acti
            self._update")
    n():
                raise Exception("Production aed")
            
            # Phase 9: Post-deployment monitoring setup
        )
            if not
                raise Exception("Monitoring setup failed")
            
            # Phase 10: Generate support documentation
            self._update_status("documentation_generation")
            self._generate_support_documentation()
            
            self._update_status("completed")
         fully!")
        rn True
            
        
            logger.error(f"❌ Deployment failed: {str(e)}")
            if self.config.rollback_on_failure:
                self._rollback_deployment()
        
    
    def _pre_deployment_validation(self) -> bool:
        """Validate system readiness for production deploym""
        logg
        
    
            "environment_variables": self._validate(),
            "database_connectivity": self._validate_d(),
            "redis_connec
            "external_services": self._validate_external_s
            "security_configuration": self._validatetion(),
            "resource_availability": self._validate_resourcety(),
         
        }
        
        tions
        
        failed_validations = [k for k, v in validations.items() if not v]
        if failed_valida
            
            return lse
     
        logger.info("✅ Pre-deployment validation pa")
        return True
    
    def _validate_environme
        """V"
        required_vars = [
            "DATABASE_URL", "REDIS,
            KEY",
            "MONITORING_API_KEY", "BA
        ]
        
        missing_vars = [var fo
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        return True
    
    def _valool:
        """Validate database connectivity and schema"""
        try:
            conn
            cursor = co
            
            # Check if agent sst
            cursor.execute("""
                SELECT t
    '
            """)
            
            tables =()
            required_tables = ['agent_registry', 'agent_t
            existing_tables = [table[0] for 
            
         les]
        s:
                logger.error(f"Missing databa")
                return False
            
            conn.close()
            return rue
       
        except Exception as e:
            logger.error(f"Database validation fail}")
            
    
    def _validate_redis_conn
        """Validate Redis connectivity"
        try:
            r = r_URL"))
            r.ping()
            return True
        except E e:
            logger.error(f"Redis validation failed: {str(e)}")
            return False
    
    def _validate_external_ser:
        """Validate external service connectivity"""
        services = {
    ",
            "Monitoring Service": os.getenv("MONITORquery")
        }
        
        for service_name, endpoint in services.items():
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code not in [200, 401]:  # 401 is OK for auth-required services
         
        
            except Exception as e:
        
                return Fal
        
        return True
    
    def _validate_securit) -> bool:
    ""
        # Check JWT secret strength
        jwt_secret = os.getenv("JWT_SECRET_KEY",)
        if len(jwt_se2:
         weak")
            return False
        
        # Check SSL configur
        ssl_cert_path = os.getenv("SSL_CERT_PATH")
        if sh):
            logger.error(f"S}")
            return False
        
        return True
    
    def _validate_resourcebool:
        """Validate system resource ay"""
        import psutil
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
    
            logger.warning(f"High CPU usage:)
        
        # Chry
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            logger.warning
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 85:
            logg")
            return False
        
        return T
    
    def _run
        """Run comprehensive t"
        try:
            # Run agent ts
    n([
                "python", "-m", "pytest", 
                "tests/test_agent_steering_integration.py
            
                "tests/test_intelligence_engine_integration.py",
                "-v", "--tb=short","
            ], capture_output=True, text=True, timeout=300)
            
            if r
                logger.rr}")
            alse
            
            logger.info("✅ Test suite passed")
            return True
        
        except subprocess.TimeoutExpired:
            logger.error("Test suite timed out")
            return False
        as e:
            )
            return False
    
    def _create_system_backu bool:
        """Creat
        logger.info("💾 Creating .")
        
        try:
            self
            
            # Backup database
            db_backup_path ="
            subpess.run([
                "pg_dump", os.get), 
                "-f", str(db_backup_path)
            ], check=True)
            
            # Backup configuration files
            config_back"
            )
            
            config_files = [".env.production", "docker-compose.prod]
            for config_ffiles:
    
                    subprocess.run([
                        "cp", config_file, str(config_backuh)
                    ], check=True)
         
            odels
            if os.path.exists("uploads"):
                subprocess.run([
            "
                ], check=True)
            
            if os.path.existls"):
                subprocess.run([
                    "tar",odels/"
            ck=True)
            
            # Create backup manifest
            backup_manifest{
                "backup_id": self.deployment_id,
                "timestamp": datetime.now().isoformat(),
                "files": list(self.backup_dir.glob("*")
             
            }
            
            with open(self.backup_dir / "manifest.j f:
                =str)
            
            logger.infor}")
            urn True
            
        except Exception as e:
            logger.error
    rn False
    
    def _deploy_infrastructure(self) -> bool:
        """Deploy production infrastructure"""
        ")
        
        try:
            # Deploy with Docker Compose
            subprocess.run([
                "docker-compose", "-f", "docker-compose.prod.yml", 
                "up", "-d", "--build"
            ], check=True)
            
            art
            time.sleep(30)
            
            # Verify infrastruct
            services = ["scrollintel-api"x"]
            for service in services:
                result = subprocess.run([
                    "docker-co
                 ce
                ], capture_output=True, texe)
                
                if not :
            ")
                    return Fal
            
            logger.info(ly")
    
            
        except Exception as e:
            logger.error(f"❌ Infrastructure deployment faile)}")
        se
    
    def _deploy_agent_steering_componentl:
        """Deploy Agent Steering System c"
        logger.info("🤖 De
        
        components = {
            "orchestration_(),
            "intelligence_engine": see(),
            "agent_registry": self._deploy_gistry(),
            "work(),
            
            "security_framework": self._deprk()
        }
        
        self.deployment_status["components"]["agent_stents
        
        failed_components = [k for k,
        if failed_components:
            logger.error(f"❌ Component deplots}")
            return False
        
        logger.in
        return True
    
    def _deploy_orchestration_engine(self) ol:
        """Deploy orchestration engine"""
        try:
            # Run database migrations fortration
            subprocess.run([
                "python", "scripts/migrate-, 
                "--component","
            ], chrue)
            
            # Inngine
            subprocess.run([
             
                "from scrollin "
                "engine = RealtimeOrchestrationEngine(); "
                "engine.
    rue)
            
            return True
        except Exception as e:
        )
            
    
    def _deploy_intellige:
        """Deploy intelligence engine"""
        try:
            # Initialize intelligence engine
            subprocess.run([
                "p",
                "from scrollintel.engines.intel
                "engine = IntelligenceE
             "
            heck=True)
            
            return True
        except Exception as e:
            )}")
            return Fals
    
    def _deploy_agent_registry bool:
        """Deploy agent registry"""
        try:
     registry
            subprocess.run([
                "python", "-c",
                "from scrollintel.core.agent_registry import  "
        "
            "
            ], check=True)
            
            return True
        except Ee:
            logger.error(f"Age)
            return False
    
    def _deploy_l:
        """Deploy communication framework"""
        try:
            # Initialize sec
            subp[
                "python", "-c",
                "from scrollintel.core.secure_c"
                "comm = Secu
                )"
            ], check=True)
            
            ue
        except Exception as e:
            logger.error(f"Communication framework deployment faile)}")
            return False
    
    def _deploy_monitoring_system(self) -> bool:
        """Deploy monitoring system"""
        try:
        
            ess.run([
                "docker-compose", "-f
                "up", "-d"
            ], check=True)
            
            return True
        exce:
            logger.error(f"Monitoring sy)}")
            return False
    
    def _deploy_security_fol:
        """Dork"""
        try:
            amework
            subprocess.run([
                "python", "-c",
                "from scwork; "
    work(); "
                "security.initialize_pro()"
            ], check=True)
            
        True
        exce as e:
            logger.error(f"Security )}")
            return False
    
    def _comprehensive_hea
        """Rs"""
        logger.info("🏥 Running comprehensive..")
        
        heals = {
            "api_health,
            
            "redis_health": self._check_redis_heal,
            "agent_health": self._check_agent_health(),
            "orchestrati
    ),
            "security_health": self._checlth(),
            "performance_health": self._check_p)
        }
        
        self.deploymks
        
        failed_checks = [k for k, v in health_checks.items() if not v]
        if failed_checks:
         
        False
        
        logger.info("✅ All health checks passed")
        return True
    
    def _check_api_ool:
    
        try:
            response = requests.get("http://localhost0)
            return respon
        except Excetion:
        se
    
    def _check_dool:
        """Check database health"""
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()
            cursor.eLECT 1")
            conn.close()
            return True
        except Eion:
            return False
    
    def _check_redis_heabool:
        """C"""
        try:
            r = redi"))
    r.ping()
            return True
        except Exception:
            return False
    
    def _che
        """Check agent system health"
        try:
            response = req
            return response.ste == 200
        except Exception:
            return False
    
    def _che
        """Check orchestration engine health"""
        try:
            response = requests.get("heout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_intelbool:
        """Check intelligence engine h"""
        try:
            response = reque
            retu= 200
        except Exception:
            return False
    
    def _check_s bool:
        """Check securi
        try:
            response = requestut=10)
            return response.status_code == 200
        except Exception
    
    
    def _check_performance_health(self) -> bool:
        """C
        try:
            # Test response time
            start_time = time.time()
            response = requests.get("
            response_time = time.time() - s
            
            me < 2.0
        except Exception:
            return False
    
    def _run_user_acceptan
        """Ru"""
        logg.")
        
        test_scenarios = {
            "agent_orchestra(),
            "bus,
            "real_time_processing": s
            "security_comrio(),
            "user_interface": self._test_usio()
        }
        
        self.scenarios
        
        failed_tests = [k for k, v inot v]
        if failed_tests:
            logger.error(f"❌ Usts}")
            return False
        
        logg passed")
        return True
    
    def _test_agent_orchestr:
        """Test ""
        try:
            sk
            task_data = {
                "title": "Market Analysis Report",
                "descrip
    ,
                "requirements": {
                    "capabilities": ["data_analy
            "]
                }
            }
            
            response = requests.post(
                "http://locaasks",
                ,
                timeout=30
            )
            
            if response.status_code != 201:
                return False
            
            task_id = rsk_id"]
            
            # Monitor task execution
            for _ in range(30):  # Wait up to 5 minutes
                status_rts.get(
    ",
                    timeout=10
                )
                
        200:
            us"]
                    if status == "co
                        return True
                    elif sta
                
                
                time.sleep(10)
            
            retuFalse
            
        except Exception as e:
            logger.error(f"A
            retue
    
    def _test_business_ol:
        """T""
        try:
            # Test intelligence engine with business query
            query_data = {
    s?",
                "context": "quarterly_review",
                "data_sources": ["sales", "customer_fee"]
            }
            
            post(
                "http://localhost:8000/api/intery",
                json=query
                timeout=30
            )
            
            return response.status_code n()
            
        exces e:
            logger.error(f"Busine")
            return False
    
    def _tesool:
        """Test real-ti"
        try:
            # Test real-time dg
            stream_data = {
                "stream_",
     [
                    {"timestamp": datetim
                    {"timestamp": datetime.now().i50}
                ]
           }
            
            response = requests.post(
                "http://localhost:800m",
                json=stream_data,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Ee:
            logger.error(f"Real-time processing te)}")
            return False
    
    def _test_security_complia:
        """Test security and compliance features"""
        try:
    on
            auth_response = requests.post(
                "http://localhost:8000/api/auth/validate"
                headers={"Authorization": f"Bearer {os.geten
        
            
            
            # Test audit logging
            audit_response = requests.get(
                "http://localhost:8000/api/audit/recent",
                timeout=10
            )
            
            return auth_
            
        except Exception as e:
            logger.error(f"Security compli")
            return False
    
    def _testbool:
        """T"
        try:
            # Test frontend accessibility
            response = requeeout=10)
            retu == 200
            
        except Exceptio:
            
            return False
    
    def _execute_gradual
    """
        logger.info("🐤 Executing gradual rollo)
        
        rollout_stages = [10, 25, 50, 75, 100]
        
        for es:
            logger.info(f"Rolling out to {stage_percentage}% of users...")
            
            # Update feature flags
            if not self._update_feature_flags(stage_percentage):
                
                return False
            
            # Monitor metrics fo stage
            if not sage):
                logger.error(f"Rollout monitoring failtage}%")
                return False
            
            # Wait b
            if stage_percentage < 100:
                logger.info(f"Waiting 5 minutes before next rollout stage..")
                time.sleep(300)  # 5 minutes
            
            self.depentage
        
        logger.info("✅ Gradual rollout completed successfully")
        return True
    
    def _update_l:
        """Update feature flags for gradual rollout"""
        try:
            feature_flags = {
            
                    "enabled":
                    "rollout_percentage": percentage,
                    "tars"]
         }
            }
            
            
                "http://localhost:8000/api/featu,
                json=feat
                timeout=10
            )
            
            retur
            
        exce
            logger.error(f"Feature flag update failed: {str(e)}")
            return False
    
    def _monitor_rollou:
        """M
        try:
            # Monitor for 2 minutes
            for _ in ran
    
                metrics_response = requests.get(
                    "http://localhost:8000/api/metrent",
                    timeout=5
                )
             
                if metrics_response.status_c
                    metrics = metrics_response.json()
                    error_rate = metrics.get(
                    response_time = metrics.get("avg_respo", 0)
              
                    # Rollback if egh
                    if error_rate > 5.0:  # 5% error rate
                rate}%")
                        return False
                    
                    # Rollback if response time is too high
                    if response_time > 2000:  #d
                        logger.e")
                     False
                
                time.sleep(10)
            
            return True
            
        exce as e:
            logger.error(f"Rol
            return False
    
    l:
        """Activate full production mode"""
        logger.info("🎯 Activating full prod)
        
        try:
            # Enable allres
            
                "mode": "pr
                "auto_scaling": True,
                "load_balancing": True,
                "caching": True,
            e,
                "security": "enterprise",
                "backup": "continuous"
            }
            
            response = re(
                "http://localhost:8000/api/system/config",
                json=production_config,
            
            )
            
            if response.status= 200:
                return False
            
    
            status_response = requests.get(
                "http://localhost:8000/api/system/status"
                timeout=10
            )
            
            if status_response.statu00:
                status = status_responn()
         n"
        
            return False
            
        except Exception as e:
            logger.error(f"Production activation failed: {str(e)}")
            return False
    
    def _setup_prodbool:
    
        logger.info("📊 Setting up production moni")
        
        try:
        lerts
            ules = {
                "high_error_rate"{
                    "condition": "error_rate > 1%",
                    "duratio5m",
                "
                },
                "high_response_time": {
                    "conditi 1s",
                ,
                    "severity": "warning"
                },
                "agent_failu: {
                ",
                    "duration": "1m",
                    "severity": "critical"
                }
            }
            
            # Setup Grafana dashboards
            dashboard_config
                ": {
                    "panels": ["agent_performance", "orchestration_metpact"],
                    "refresh": "30s"
                },
            ": {
                    "panels": 
                    "refresh": "1m"
                }
     }
            
            # Configure alerting channels
            alerting_config = {
        "),
            "),
                "pagerduty": os.getenv(
            }
            
            # Apply monitoring configuration
            monitoring_response = requests.post(
              
            son={
                    "alerts": alert_rules,
                    "dashboards": dashboard_config,
                    "alerting": nfig
                },
                timeout=10
            )
            
            retur1]
            
        except En as e:
            logger.error(f"Monitoring setup faile
            return False
    
    def _generate_support_docu
        """Generate comprehensive support documentation"""
        logger.info("📚 
    
        docs_dir = Path("docs/production")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        ry
        depl
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "status": self.deployment_status,
            "configurat": {
            ent,
                "features_enabd": {
                    "blue_green_deployment": self.config.enn,
                    "can
    
                    "monitoring": self.config.monitorig_enabled
                }
            },
        oints": {
            ",
                "frontend": "http://localhost:3000",
                "monitoring": "http://localhost:
            st:9090"
            },
            "support_contacts": {
                "technical_lead": "tech-lead@companyom",
                "devops_team": "devops@company.com",
                "emergency": "emergency@company.com"
            }
        }
        
        with open(docs_dir / "deployment_s as f:
            json.dump(deployment_summary, f, indent=2)
        
        # Gel runbook
        runbook_content = f"""# Agent Steering System Production Runbook

## Deployment Inon
- **Deployment ID**: {self.deployment_id}
- **Deployment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
t}

## System Architecture
The Agent Steering System is deployed with ts:
- Orchestration Engine: Coordinates multiple AI agents
- Intelligence Engine: Provides business decision-making capabilits
ies
- Communicatioon
- Monitoring System: Real-time performance an

## Health Check Endpoints
- **API Health**: GET /health
led
- **Agent Health**:lth
- **Orchestration Health**: GET /api/orche
e/health

## Monitoring and Alerting

- **Grafana**: http001
- **Alert Manager**: http://localhost:9093

## Common Operations

### Restart Servics
```bash
docker-compose -f docker-compose.prod.yml restart
```


```bash
docker-compose -f dock_name]
```

### Scale Services
```bash
docker-compo=3
```

### Database Opes
```bash
# Backup database
pg_dump $DAT

# Restore database
psql $DATABASE_URL < bacile.sql
```

## Troubleshooting

### Highrror Rate
1. Check app logs
2. Verify database connectivity
3. Check external service status
4. Review recent deployments

### High Response Time
1. Check system resources (CPU, memory)
2. Review database query performance
3. Check cach
4. Verify nennectivity

### Agent Failures
1. Check agent registryus
2. Review or
3. Verify agent communication ls
4. Check resource allocation

## Ees

### Rollback Deployment
```bash
python snt_id}
```

### Emergency Shutdown
```bash
docker-compose -f docker-compose.prod.yml down
```

### Contact Information
- **Technical Lead**: tech-lead@company.com
- **DevOps Teom
- **Emergenc

## Performance Baselines
- **Response Time**: < 1 second (95)
- **Error Rate**: < 1%
- **Agent Success Rate**: > 95%
- **Uptime**: > 99.9%
"""
        
        with open(docs_dir / "operational_ras f:
            f.wrntent)
        
        logger.info(f"✅
    
    def _rollback_deployment(s
        """Rollback deployment to previous state"""
        logger.info("🔄 ..")
        
        try:
            # Restore database from backup
            if (self.backup_dir / "database_backup.sql").exits():
        un([
            L"),
                    "-f", str(sel")
                ], check=True)
            
            # Restore configures
            config_backup_path = self.backup_dir / "config"
            if config_backup_path.exists():
                for config_file "):
                    subprocess.run([
                        "cp", "
                =True)
            
            # Restart servicuration
            subprocess.run([
                "docker-co",
            n"
            ], check=True)
            
            subprocess.run([
                "docker-co,
            -d"
            ], check=True)
            
            ly")
            
        except Exception as e:
            )
    
    def _update_status(s: str):
        """Us"""
        self.deployment_statusus
        self.deployment_status["last_updated"] = )
        
    file
        status_file = Path(f"logs/deployment_status_{self.deployment_id}.json")
        with open(status_file, "w") as f:
            json.dump(self.deployment_status,dent=2)

def main():
    """Main deployment function"""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_=True)
    
    # Loaration
    conf
        environment=os.getenv("DEPLOYMENT_ENV", "prodtion"),
        enable_blue_green=os.getenv("ENABLE_BLUE_GREEN", "ue",

        ena",
        rollback_on_failure=os.gete
        user_acceptance_test=o,
        monitoring_enabled=os.get
    )
    
    # Create and run deployer
    deployer = ProductionDeployer(config)
    success = deployer.deploy()
    
    if success:
     ")
    
        print("🚀 AGENT STEERING SYSTEM PRODUCTION DEP
        print("="*80)
    d}")
        print(f}")
        print("Features Enabled:")
        print(f"  •}")
        p)
        print(f"  • Feature Flags: {config.enable_feature_flags}")
        print(f"  •
")
        print("  • API: ht")
        prin()
    ma":"__main___ == 
if __name_it(1)
ex sys.      iled!")
 yment fa deploonm productiering Systeent Ste"❌ Agr( logger.erro        else:
  t(0)
 .exi       sys
 "="*80)int(
        prations")tificng not up alertiSe"  5. t(rin        pules")
backup schedigure ("  4. Confnt    priion")
     producto point toNS tte Dda Upprint("  3.       ting")
  tesnceer acceptact final us 2. Condu" nt(        pri)
trics"nce and mermasystem perfo1. Monitor int("         prteps:")
 nNext Snt("\   pri    t:9090")
 calhosttp://loics: h"  • Metrprint(  ")
      t:3001p://localhos: httringtot("  • Monirin       p
 t:3000")p://localhos httnd:• Fronte"  int(