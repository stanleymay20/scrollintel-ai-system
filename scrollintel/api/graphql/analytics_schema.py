"""
GraphQL Schema for Advanced Analytics Dashboard API

This module defines the GraphQL schema for flexible data querying
across all dashboard and analytics functionality.
"""

from graphql import build_schema
from typing import Dict, List, Optional, Any
from datetime import datetime

# GraphQL Schema Definition
analytics_schema_definition = """
    scalar DateTime
    scalar JSON

    # Dashboard Types
    type Dashboard {
        id: ID!
        name: String!
        type: DashboardType!
        owner: String!
        config: JSON!
        widgets: [Widget!]!
        permissions: [Permission!]!
        createdAt: DateTime!
        updatedAt: DateTime!
        isActive: Boolean!
        metrics: DashboardMetrics
    }

    type Widget {
        id: ID!
        type: WidgetType!
        title: String!
        config: JSON!
        position: WidgetPosition!
        data: JSON
        lastUpdated: DateTime!
    }

    type WidgetPosition {
        x: Int!
        y: Int!
        width: Int!
        height: Int!
    }

    type Permission {
        userId: String!
        role: PermissionRole!
        grantedAt: DateTime!
        grantedBy: String!
    }

    type DashboardMetrics {
        totalViews: Int!
        uniqueUsers: Int!
        avgSessionDuration: Float!
        lastAccessed: DateTime
        popularWidgets: [String!]!
    }

    # ROI Types
    type ROIAnalysis {
        id: ID!
        projectId: String!
        projectName: String!
        totalInvestment: Float!
        totalBenefits: Float!
        roiPercentage: Float!
        paybackPeriod: Int!
        npv: Float!
        irr: Float!
        analysisDate: DateTime!
        breakdown: ROIBreakdown!
        trends: [ROITrend!]!
    }

    type ROIBreakdown {
        directCosts: Float!
        indirectCosts: Float!
        operationalSavings: Float!
        productivityGains: Float!
        revenueIncrease: Float!
        riskMitigation: Float!
    }

    type ROITrend {
        period: String!
        investment: Float!
        benefits: Float!
        cumulativeROI: Float!
    }

    # Insight Types
    type Insight {
        id: ID!
        type: InsightType!
        title: String!
        description: String!
        significance: Float!
        confidence: Float!
        recommendations: [String!]!
        createdAt: DateTime!
        dataPoints: [DataPoint!]!
        visualizations: [Visualization!]!
        actionItems: [ActionItem!]!
        businessImpact: BusinessImpact!
    }

    type DataPoint {
        metric: String!
        value: Float!
        unit: String!
        timestamp: DateTime!
        source: String!
        context: JSON
    }

    type Visualization {
        type: VisualizationType!
        title: String!
        config: JSON!
        data: JSON!
    }

    type ActionItem {
        id: ID!
        title: String!
        description: String!
        priority: Priority!
        estimatedImpact: Float!
        estimatedEffort: Int!
        assignee: String
        dueDate: DateTime
        status: ActionItemStatus!
    }

    type BusinessImpact {
        category: ImpactCategory!
        magnitude: Float!
        timeframe: String!
        affectedMetrics: [String!]!
        riskLevel: RiskLevel!
    }

    # Predictive Analytics Types
    type Forecast {
        id: ID!
        metric: String!
        horizon: Int!
        predictions: [Prediction!]!
        confidence: Float!
        model: String!
        accuracy: Float!
        generatedAt: DateTime!
        scenarios: [Scenario!]!
    }

    type Prediction {
        timestamp: DateTime!
        value: Float!
        lowerBound: Float!
        upperBound: Float!
        confidence: Float!
    }

    type Scenario {
        id: ID!
        name: String!
        description: String!
        assumptions: JSON!
        predictions: [Prediction!]!
        impact: ScenarioImpact!
    }

    type ScenarioImpact {
        revenueChange: Float!
        costChange: Float!
        riskChange: Float!
        timeToRealization: Int!
    }

    # Data Integration Types
    type DataSource {
        id: ID!
        name: String!
        type: DataSourceType!
        status: ConnectionStatus!
        config: JSON!
        lastSync: DateTime
        metrics: DataSourceMetrics!
        schema: DataSchema!
    }

    type DataSourceMetrics {
        recordCount: Int!
        syncFrequency: String!
        errorRate: Float!
        avgSyncTime: Float!
        dataQuality: Float!
    }

    type DataSchema {
        tables: [TableSchema!]!
        relationships: [Relationship!]!
        lastUpdated: DateTime!
    }

    type TableSchema {
        name: String!
        columns: [ColumnSchema!]!
        recordCount: Int!
    }

    type ColumnSchema {
        name: String!
        type: String!
        nullable: Boolean!
        unique: Boolean!
        description: String
    }

    type Relationship {
        fromTable: String!
        toTable: String!
        type: RelationshipType!
        columns: [String!]!
    }

    # Template Types
    type DashboardTemplate {
        id: ID!
        name: String!
        description: String!
        category: TemplateCategory!
        industry: String
        config: JSON!
        widgets: [TemplateWidget!]!
        popularity: Int!
        rating: Float!
        createdBy: String!
        createdAt: DateTime!
        isPublic: Boolean!
    }

    type TemplateWidget {
        type: WidgetType!
        title: String!
        config: JSON!
        position: WidgetPosition!
        dataSources: [String!]!
    }

    # Analytics Types
    type AnalyticsReport {
        id: ID!
        title: String!
        type: ReportType!
        format: ReportFormat!
        generatedAt: DateTime!
        fileSize: Int!
        downloadUrl: String!
        metadata: JSON!
        schedule: ReportSchedule
    }

    type ReportSchedule {
        id: ID!
        frequency: ScheduleFrequency!
        nextRun: DateTime!
        lastRun: DateTime
        recipients: [String!]!
        isActive: Boolean!
    }

    # Enums
    enum DashboardType {
        EXECUTIVE
        DEPARTMENT
        PROJECT
        CUSTOM
    }

    enum WidgetType {
        CHART
        TABLE
        METRIC
        MAP
        TEXT
        IMAGE
        IFRAME
    }

    enum PermissionRole {
        OWNER
        EDITOR
        VIEWER
        COMMENTER
    }

    enum InsightType {
        TREND
        ANOMALY
        CORRELATION
        PREDICTION
        RECOMMENDATION
        ALERT
    }

    enum VisualizationType {
        LINE_CHART
        BAR_CHART
        PIE_CHART
        SCATTER_PLOT
        HEATMAP
        GAUGE
        TABLE
        MAP
    }

    enum Priority {
        LOW
        MEDIUM
        HIGH
        CRITICAL
    }

    enum ActionItemStatus {
        PENDING
        IN_PROGRESS
        COMPLETED
        CANCELLED
    }

    enum ImpactCategory {
        REVENUE
        COST
        EFFICIENCY
        QUALITY
        RISK
        COMPLIANCE
    }

    enum RiskLevel {
        LOW
        MEDIUM
        HIGH
        CRITICAL
    }

    enum DataSourceType {
        ERP
        CRM
        BI_TOOL
        CLOUD_PLATFORM
        DATABASE
        API
        FILE
    }

    enum ConnectionStatus {
        CONNECTED
        DISCONNECTED
        ERROR
        SYNCING
    }

    enum RelationshipType {
        ONE_TO_ONE
        ONE_TO_MANY
        MANY_TO_MANY
    }

    enum TemplateCategory {
        EXECUTIVE
        FINANCE
        SALES
        MARKETING
        OPERATIONS
        HR
        IT
        CUSTOM
    }

    enum ReportType {
        EXECUTIVE_SUMMARY
        FINANCIAL_ANALYSIS
        PERFORMANCE_DASHBOARD
        TREND_ANALYSIS
        CUSTOM
    }

    enum ReportFormat {
        PDF
        EXCEL
        CSV
        JSON
        HTML
    }

    enum ScheduleFrequency {
        DAILY
        WEEKLY
        MONTHLY
        QUARTERLY
        YEARLY
        CUSTOM
    }

    # Input Types
    input DashboardInput {
        name: String!
        type: DashboardType!
        config: JSON
        templateId: ID
    }

    input WidgetInput {
        type: WidgetType!
        title: String!
        config: JSON!
        position: WidgetPositionInput!
    }

    input WidgetPositionInput {
        x: Int!
        y: Int!
        width: Int!
        height: Int!
    }

    input ROIAnalysisInput {
        projectId: String!
        projectName: String!
        costs: JSON!
        benefits: JSON!
        timeframe: Int!
    }

    input ForecastInput {
        metric: String!
        horizon: Int!
        dataPoints: [DataPointInput!]!
        model: String
    }

    input DataPointInput {
        metric: String!
        value: Float!
        timestamp: DateTime!
        source: String!
    }

    input DataSourceInput {
        name: String!
        type: DataSourceType!
        config: JSON!
    }

    input ReportInput {
        title: String!
        type: ReportType!
        format: ReportFormat!
        config: JSON!
        scheduleConfig: ScheduleInput
    }

    input ScheduleInput {
        frequency: ScheduleFrequency!
        recipients: [String!]!
        startDate: DateTime!
        endDate: DateTime
    }

    input FilterInput {
        field: String!
        operator: String!
        value: JSON!
    }

    input SortInput {
        field: String!
        direction: String!
    }

    input PaginationInput {
        page: Int!
        limit: Int!
    }

    # Query Type
    type Query {
        # Dashboard Queries
        dashboard(id: ID!): Dashboard
        dashboards(
            type: DashboardType
            owner: String
            pagination: PaginationInput
            sort: SortInput
        ): [Dashboard!]!
        
        dashboardMetrics(id: ID!, timeRange: String): DashboardMetrics
        
        # ROI Queries
        roiAnalysis(id: ID!): ROIAnalysis
        roiAnalyses(
            projectId: String
            dateRange: String
            pagination: PaginationInput
        ): [ROIAnalysis!]!
        
        roiTrends(projectId: String!, timeRange: String!): [ROITrend!]!
        roiComparison(projectIds: [String!]!): [ROIAnalysis!]!
        
        # Insight Queries
        insight(id: ID!): Insight
        insights(
            type: InsightType
            significance: Float
            dateRange: String
            pagination: PaginationInput
            sort: SortInput
        ): [Insight!]!
        
        insightTrends(timeRange: String!): [Insight!]!
        
        # Predictive Analytics Queries
        forecast(id: ID!): Forecast
        forecasts(
            metric: String
            horizon: Int
            pagination: PaginationInput
        ): [Forecast!]!
        
        scenarioAnalysis(forecastId: ID!, scenarios: [JSON!]!): [Scenario!]!
        
        # Data Integration Queries
        dataSource(id: ID!): DataSource
        dataSources(
            type: DataSourceType
            status: ConnectionStatus
            pagination: PaginationInput
        ): [DataSource!]!
        
        dataQuality(sourceId: ID!, timeRange: String): JSON
        
        # Template Queries
        dashboardTemplate(id: ID!): DashboardTemplate
        dashboardTemplates(
            category: TemplateCategory
            industry: String
            pagination: PaginationInput
            sort: SortInput
        ): [DashboardTemplate!]!
        
        # Analytics Queries
        analyticsReport(id: ID!): AnalyticsReport
        analyticsReports(
            type: ReportType
            dateRange: String
            pagination: PaginationInput
        ): [AnalyticsReport!]!
        
        # Search and Discovery
        search(query: String!, types: [String!], limit: Int): JSON
        recommendations(userId: String!, context: JSON): JSON
    }

    # Mutation Type
    type Mutation {
        # Dashboard Mutations
        createDashboard(input: DashboardInput!): Dashboard!
        updateDashboard(id: ID!, input: DashboardInput!): Dashboard!
        deleteDashboard(id: ID!): Boolean!
        
        addWidget(dashboardId: ID!, widget: WidgetInput!): Widget!
        updateWidget(dashboardId: ID!, widgetId: ID!, widget: WidgetInput!): Widget!
        removeWidget(dashboardId: ID!, widgetId: ID!): Boolean!
        
        shareDashboard(id: ID!, userId: String!, role: PermissionRole!): Boolean!
        
        # ROI Mutations
        createROIAnalysis(input: ROIAnalysisInput!): ROIAnalysis!
        updateROIAnalysis(id: ID!, input: ROIAnalysisInput!): ROIAnalysis!
        deleteROIAnalysis(id: ID!): Boolean!
        
        # Forecast Mutations
        createForecast(input: ForecastInput!): Forecast!
        updateForecast(id: ID!, input: ForecastInput!): Forecast!
        deleteForecast(id: ID!): Boolean!
        
        # Data Source Mutations
        createDataSource(input: DataSourceInput!): DataSource!
        updateDataSource(id: ID!, input: DataSourceInput!): DataSource!
        deleteDataSource(id: ID!): Boolean!
        
        syncDataSource(id: ID!): Boolean!
        testDataSourceConnection(id: ID!): Boolean!
        
        # Report Mutations
        generateReport(input: ReportInput!): AnalyticsReport!
        scheduleReport(input: ReportInput!): AnalyticsReport!
        cancelScheduledReport(id: ID!): Boolean!
        
        # System Mutations
        refreshInsights(dashboardId: ID): Boolean!
        recalculateMetrics(dashboardId: ID!): Boolean!
        optimizePerformance: Boolean!
    }

    # Subscription Type
    type Subscription {
        # Real-time Dashboard Updates
        dashboardUpdated(id: ID!): Dashboard!
        widgetUpdated(dashboardId: ID!, widgetId: ID!): Widget!
        
        # Real-time Insights
        newInsight(dashboardId: ID): Insight!
        insightUpdated(id: ID!): Insight!
        
        # Real-time Alerts
        alert(severity: String): JSON!
        
        # Real-time Metrics
        metricsUpdated(dashboardId: ID!): JSON!
        
        # Data Source Updates
        dataSourceStatusChanged(id: ID!): DataSource!
        syncCompleted(sourceId: ID!): JSON!
    }

    # Schema
    schema {
        query: Query
        mutation: Mutation
        subscription: Subscription
    }
"""

# Build the schema
analytics_schema = build_schema(analytics_schema_definition)