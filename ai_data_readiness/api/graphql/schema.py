"""GraphQL schema definition."""

import strawberry
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass

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
    """GraphQL Subscription root for real-time updates."""
    
    # Dataset subscriptions
    dataset_updates: Dataset = strawberry.field(resolver=DatasetResolver.subscribe_dataset_updates)
    
    # Quality subscriptions
    quality_updates: QualityReport = strawberry.field(resolver=QualityResolver.subscribe_quality_updates)
    
    # Processing subscriptions
    job_status_updates: ProcessingJob = strawberry.field(resolver=ProcessingResolver.subscribe_job_updates)
    
    # Drift subscriptions
    drift_alerts: DriftReport = strawberry.field(resolver=DriftResolver.subscribe_drift_alerts)
    
    # System subscriptions
    system_alerts: dict = strawberry.field(resolver=DatasetResolver.subscribe_system_alerts)
    
    # Enhanced subscriptions for real-time monitoring
    pipeline_status_updates: dict = strawberry.field(resolver=ProcessingResolver.subscribe_pipeline_status)
    compliance_alerts: ComplianceReport = strawberry.field(resolver=ComplianceResolver.subscribe_compliance_alerts)
    bias_monitoring_updates: BiasReport = strawberry.field(resolver=BiasResolver.subscribe_bias_monitoring)
    feature_performance_updates: dict = strawberry.field(resolver=FeatureResolver.subscribe_feature_performance)


# Create the GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)