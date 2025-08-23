"""
Integration Orchestrator
Coordinates all enterprise integration components for seamless operation
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time

from .auto_discovery_engine import AutoDiscoveryEngine, IntegrationProfile
from .etl_recommendation_engine import ETLRecommendationEngine, ETLRecommendation
from .connector_registry import ConnectorRegistry, ConnectorInstance
from .data_quality_engine import DataQualityEngine, QualityReport
from .streaming_engine import HighPerformanceStreamingEngine, StreamEvent
from .visual_integration_builder import VisualIntegrationBuilder, IntegrationWorkflow

logger = logging.getLogger(__name__)

@dataclass
class IntegrationProject:
    """Complete integration project"""
    project_id: str
    name: str
    description: str
    source_systems: List[Dict[str, Any]]
    target_systems: List[Dict[str, Any]]
    integration_profiles: List[IntegrationProfile]
    etl_recommendations: List[ETLRecommendation]
    quality_reports: List[QualityReport]
    workflows: List[IntegrationWorkflow]
    connector_instances: List[ConnectorInstance]
    status: str
    progress_percentage: float
    created_at: datetime
    updated_at: datetime

class IntegrationOrchestrator:
    """
    Master orchestrator for enterprise integration excellence
    Coordinates auto-discovery, ETL recommendations, connectors, quality, streaming, and visual builder
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize all components
        self.auto_discovery = AutoDiscoveryEngine()
        self.etl_engine = ETLRecommendationEngine()
        self.connector_registry = ConnectorRegistry()
        self.quality_engine = DataQualityEngine()
        self.streaming_engine = HighPerformanceStreamingEngine(config.get('streaming', {}))
        self.visual_builder = VisualIntegrationBuilder()
        
        # Project management
        self.projects = {}
        self.active_integrations = {}
        
        logger.info("Integration Orchestrator initialized with all components")
    
    async def start(self):
        """Start the integration orchestrator"""
        try:
            # Start streaming engine
            await self.streaming_engine.start()
            
            logger.info("Integration Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Integration Orchestrator: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the integration orchestrator"""
        try:
            # Stop streaming engine
            await self.streaming_engine.stop()
            
            logger.info("Integration Orchestrator stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Integration Orchestrator: {str(e)}")
    
    async def create_integration_project(
        self,
        name: str,
        description: str,
        source_systems: List[Dict[str, Any]],
        target_systems: List[Dict[str, Any]]
    ) -> IntegrationProject:
        """Create a comprehensive integration project"""
        
        project = IntegrationProject(
            project_id=self._generate_project_id(),
            name=name,
            description=description,
            source_systems=source_systems,
            target_systems=target_systems,
            integration_profiles=[],
            etl_recommendations=[],
            quality_reports=[],
            workflows=[],
            connector_instances=[],
            status='created',
            progress_percentage=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.projects[project.project_id] = project
        
        logger.info(f"Created integration project: {name} ({project.project_id})")
        return project  
  
    async def execute_full_integration_workflow(
        self,
        project_id: str,
        auto_execute: bool = True
    ) -> Dict[str, Any]:
        """Execute complete integration workflow from discovery to deployment"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        execution_results = {
            'project_id': project_id,
            'start_time': datetime.now(),
            'phases': {},
            'overall_status': 'running'
        }
        
        try:
            logger.info(f"Starting full integration workflow for project {project.name}")
            
            # Phase 1: Auto-Discovery
            logger.info("Phase 1: Auto-Discovery")
            discovery_results = await self._execute_discovery_phase(project)
            execution_results['phases']['discovery'] = discovery_results
            project.progress_percentage = 20.0
            
            # Phase 2: ETL Recommendations
            logger.info("Phase 2: ETL Recommendations")
            etl_results = await self._execute_etl_recommendation_phase(project)
            execution_results['phases']['etl_recommendations'] = etl_results
            project.progress_percentage = 40.0
            
            # Phase 3: Data Quality Assessment
            logger.info("Phase 3: Data Quality Assessment")
            quality_results = await self._execute_quality_assessment_phase(project)
            execution_results['phases']['quality_assessment'] = quality_results
            project.progress_percentage = 60.0
            
            # Phase 4: Connector Setup
            logger.info("Phase 4: Connector Setup")
            connector_results = await self._execute_connector_setup_phase(project)
            execution_results['phases']['connector_setup'] = connector_results
            project.progress_percentage = 80.0
            
            # Phase 5: Workflow Generation and Deployment
            logger.info("Phase 5: Workflow Generation")
            workflow_results = await self._execute_workflow_generation_phase(project, auto_execute)
            execution_results['phases']['workflow_generation'] = workflow_results
            project.progress_percentage = 100.0
            
            # Update project status
            project.status = 'completed'
            project.updated_at = datetime.now()
            execution_results['overall_status'] = 'completed'
            execution_results['end_time'] = datetime.now()
            
            logger.info(f"Integration workflow completed successfully for project {project.name}")
            
        except Exception as e:
            logger.error(f"Integration workflow failed: {str(e)}")
            project.status = 'failed'
            execution_results['overall_status'] = 'failed'
            execution_results['error'] = str(e)
            execution_results['end_time'] = datetime.now()
            raise
        
        return execution_results
    
    async def _execute_discovery_phase(self, project: IntegrationProject) -> Dict[str, Any]:
        """Execute auto-discovery phase"""
        
        discovery_results = {
            'discovered_sources': [],
            'integration_profiles': [],
            'total_entities': 0,
            'total_relationships': 0,
            'discovery_time_seconds': 0
        }
        
        start_time = time.time()
        
        try:
            # Discover each source system
            for source_system in project.source_systems:
                logger.info(f"Discovering source system: {source_system.get('name', 'Unknown')}")
                
                profile = await self.auto_discovery.discover_data_source(
                    connection_params=source_system['connection_params'],
                    source_type=source_system.get('type', 'database')
                )
                
                project.integration_profiles.append(profile)
                discovery_results['integration_profiles'].append(asdict(profile))
                discovery_results['total_entities'] += len(profile.entities)
                discovery_results['total_relationships'] += len(profile.relationships)
                
                discovered_source = {
                    'source_id': profile.source_id,
                    'name': source_system.get('name', 'Unknown'),
                    'type': source_system.get('type', 'database'),
                    'entities_count': len(profile.entities),
                    'relationships_count': len(profile.relationships),
                    'quality_score': profile.quality_metrics.get('overall_score', 0.0),
                    'complexity': profile.integration_complexity,
                    'estimated_time_hours': profile.estimated_integration_time
                }
                discovery_results['discovered_sources'].append(discovered_source)
            
            discovery_results['discovery_time_seconds'] = time.time() - start_time
            
            logger.info(f"Discovery phase completed: {discovery_results['total_entities']} entities, {discovery_results['tota