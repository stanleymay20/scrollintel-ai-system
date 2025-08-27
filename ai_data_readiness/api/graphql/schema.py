"""Modern GraphQL schema definition using Strawberry GraphQL."""

import strawberry
from typing import List, Optional, AsyncGenerator
from datetime import datetime
import asyncio

from .types import (
    Dataset, QualityReport, BiasReport, AIReadinessScore,
    FeatureRecommendations, ComplianceReport, LineageInfo,
    DriftReport, ProcessingJob
)
from .resolvers import (
    DatasetResolver, QualityResolver, BiasResolver,
    FeatureResolver, ComplianceResolver, LineageResolver,
    DriftResolver, ProcessingResolver
)


@strawberry.type
class Query:
    """GraphQL Query root."""
    
    # Dataset queries
    dataset: Optional[Dataset] = strawberry.field(resolver=DatasetResolver.get_dataset)
    datasets: List[Dataset] = strawberry.field(resolver=DatasetResolver.list_datasets)
    search_datasets: List[Dataset] = strawberry.field(resolver=DatasetResolver.search_datasets)
    
    # Quality queries
    quality_report: Optional[QualityReport] = strawberry.field(resolver=QualityResolver.get_quality_report)
    ai_readiness: Optional[AIReadinessScore] = strawberry.field(resolver=QualityResolver.get_ai_readiness)
    quality_trends: List[QualityReport] = strawberry.field(resolver=QualityResolver.get_quality_trends)
    
    # Bias queries
    bias_report: Optional[BiasReport] = strawberry.field(resolver=BiasResolver.get_bias_report)
    bias_comparison: List[BiasReport] = strawberry.field(resolver=BiasResolver.compare_bias_reports)
    
    # Feature queries
    feature_recommendations: Optional[FeatureRecommendations] = strawberry.field(
        resolver=FeatureResolver.get_feature_recommendations
    )
    feature_impact_analysis: List[dict] = strawberry.field(resolver=FeatureResolver.analyze_feature_impact)
    
    # Compliance queries
    compliance_report: Optional[ComplianceReport] = strawberry.field(
        resolver=ComplianceResolver.get_compliance_report
    )
    compliance_summary: dict = strawberry.field(resolver=ComplianceResolver.get_compliance_summary)
    
    # Lineage queries
    lineage: Optional[LineageInfo] = strawberry.field(resolver=LineageResolver.get_lineage)
    lineage_graph: dict = strawberry.field(resolver=LineageResolver.get_lineage_graph)
    impact_analysis: List[str] = strawberry.field(resolver=LineageResolver.analyze_impact)
    
    # Drift queries
    drift_report: Optional[DriftReport] = strawberry.field(resolver=DriftResolver.get_drift_report)
    drift_trends: List[DriftReport] = strawberry.field(resolver=DriftResolver.get_drift_trends)
    
    # Processing queries
    processing_job: Optional[ProcessingJob] = strawberry.field(resolver=ProcessingResolver.get_job)
    processing_jobs: List[ProcessingJob] = strawberry.field(resolver=ProcessingResolver.list_jobs)
    job_queue_status: dict = strawberry.field(resolver=ProcessingResolver.get_queue_status)
    
    # Analytics queries
    system_metrics: dict = strawberry.field(resolver=DatasetResolver.get_system_metrics)
    usage_analytics: dict = strawberry.field(resolver=DatasetResolver.get_usage_analytics)
    
    # Complex relationship queries
    dataset_analysis: Optional[dict] = strawberry.field(resolver=DatasetResolver.get_dataset_analysis)
    cross_dataset_comparison: dict = strawberry.field(resolver=DatasetResolver.compare_datasets)
    pipeline_health: dict = strawberry.field(resolver=ProcessingResolver.get_pipeline_health)
    compliance_dashboard: dict = strawberry.field(resolver=ComplianceResolver.get_compliance_dashboard)


@strawberry.type
class Mutation:
    """GraphQL Mutation root."""
    
    # Dataset mutations
    create_dataset: Dataset = strawberry.field(resolver=DatasetResolver.create_dataset)
    update_dataset: Dataset = strawberry.field(resolver=DatasetResolver.update_dataset)
    delete_dataset: bool = strawberry.field(resolver=DatasetResolver.delete_dataset)
    
    # Quality mutations
    assess_quality: QualityReport = strawberry.field(resolver=QualityResolver.assess_quality)
    
    # Bias mutations
    analyze_bias: BiasReport = strawberry.field(resolver=BiasResolver.analyze_bias)
    
    # Feature mutations
    generate_features: FeatureRecommendations = strawberry.field(
        resolver=FeatureResolver.generate_recommendations
    )
    
    # Compliance mutations
    check_compliance: ComplianceReport = strawberry.field(resolver=ComplianceResolver.check_compliance)
    
    # Drift mutations
    setup_drift_monitoring: DriftReport = strawberry.field(resolver=DriftResolver.setup_monitoring)
    
    # Processing mutations
    create_processing_job: ProcessingJob = strawberry.field(resolver=ProcessingResolver.create_job)
    cancel_processing_job: bool = strawberry.field(resolver=ProcessingResolver.cancel_job)


@strawberry.type
class Subscription:
    """Modern GraphQL Subscription root for real-time updates."""
    
    @strawberry.subscription
    async def dataset_updates(self, dataset_id: Optional[str] = None) -> AsyncGenerator[Dataset, None]:
        """Subscribe to dataset updates with modern async generator pattern."""
        async for update in DatasetResolver.subscribe_dataset_updates(dataset_id):
            yield update
    
    @strawberry.subscription
    async def quality_updates(self, dataset_id: str) -> AsyncGenerator[QualityReport, None]:
        """Subscribe to quality report updates."""
        async for update in QualityResolver.subscribe_quality_updates(dataset_id):
            yield update
    
    @strawberry.subscription
    async def job_status_updates(self, job_id: str) -> AsyncGenerator[ProcessingJob, None]:
        """Subscribe to processing job status updates."""
        async for update in ProcessingResolver.subscribe_job_updates(job_id):
            yield update
    
    @strawberry.subscription
    async def drift_alerts(self, dataset_id: str) -> AsyncGenerator[DriftReport, None]:
        """Subscribe to drift detection alerts."""
        async for alert in DriftResolver.subscribe_drift_alerts(dataset_id):
            yield alert
    
    @strawberry.subscription
    async def system_alerts(self) -> AsyncGenerator[strawberry.scalars.JSON, None]:
        """Subscribe to system-wide alerts."""
        async for alert in DatasetResolver.subscribe_system_alerts():
            yield alert
    
    @strawberry.subscription
    async def pipeline_status_updates(self, pipeline_id: str) -> AsyncGenerator[strawberry.scalars.JSON, None]:
        """Subscribe to pipeline status updates."""
        async for update in ProcessingResolver.subscribe_pipeline_status(pipeline_id):
            yield update
    
    @strawberry.subscription
    async def compliance_alerts(self, dataset_id: str) -> AsyncGenerator[ComplianceReport, None]:
        """Subscribe to compliance alerts."""
        async for alert in ComplianceResolver.subscribe_compliance_alerts(dataset_id):
            yield alert


# Create the modern GraphQL schema with enhanced features
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    # Enable modern features
    extensions=[
        # Add query complexity analysis
        strawberry.extensions.QueryDepthLimiter(max_depth=10),
        # Add performance tracing
        strawberry.extensions.Tracing(),
    ]
)