"""
ROI Analysis and Cost Tracking data models for the Advanced Analytics Dashboard System.
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from enum import Enum
import uuid

from .database import Base


class CostType(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    OPERATIONAL = "operational"
    INFRASTRUCTURE = "infrastructure"
    PERSONNEL = "personnel"
    LICENSING = "licensing"
    CLOUD_SERVICES = "cloud_services"
    TOOLS = "tools"


class BenefitType(Enum):
    COST_SAVINGS = "cost_savings"
    PRODUCTIVITY_GAIN = "productivity_gain"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    REVENUE_INCREASE = "revenue_increase"
    RISK_REDUCTION = "risk_reduction"
    TIME_SAVINGS = "time_savings"
    QUALITY_IMPROVEMENT = "quality_improvement"


class ProjectStatus(Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class ROIAnalysis(Base):
    """ROI Analysis model for comprehensive ROI tracking and calculation."""
    __tablename__ = "roi_analyses"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, nullable=False, unique=True)
    project_name = Column(String(255), nullable=False)
    project_description = Column(Text)
    project_status = Column(String(50), nullable=False)  # ProjectStatus enum
    
    # Investment tracking
    total_investment = Column(Float, nullable=False, default=0.0)
    initial_investment = Column(Float, nullable=False, default=0.0)
    ongoing_costs = Column(Float, nullable=False, default=0.0)
    
    # Benefit tracking
    total_benefits = Column(Float, nullable=False, default=0.0)
    realized_benefits = Column(Float, nullable=False, default=0.0)
    projected_benefits = Column(Float, nullable=False, default=0.0)
    
    # ROI calculations
    roi_percentage = Column(Float)  # (Benefits - Investment) / Investment * 100
    net_present_value = Column(Float)  # NPV calculation
    internal_rate_of_return = Column(Float)  # IRR calculation
    payback_period_months = Column(Integer)  # Time to recover investment
    break_even_date = Column(DateTime)
    
    # Time tracking
    project_start_date = Column(DateTime, nullable=False)
    project_end_date = Column(DateTime)
    analysis_period_start = Column(DateTime, nullable=False)
    analysis_period_end = Column(DateTime, nullable=False)
    
    # Metadata
    analysis_methodology = Column(JSON)  # Details about calculation methods
    assumptions = Column(JSON)  # Key assumptions made in calculations
    risk_factors = Column(JSON)  # Identified risks and their impact
    confidence_level = Column(Float, default=0.8)  # Confidence in ROI calculation
    
    # Audit fields
    created_by = Column(String, nullable=False)
    last_updated_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cost_items = relationship("CostTracking", back_populates="roi_analysis", cascade="all, delete-orphan")
    benefit_items = relationship("BenefitTracking", back_populates="roi_analysis", cascade="all, delete-orphan")
    roi_reports = relationship("ROIReport", back_populates="roi_analysis", cascade="all, delete-orphan")


class CostTracking(Base):
    """Cost Tracking model for detailed cost breakdown and monitoring."""
    __tablename__ = "cost_tracking"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    roi_analysis_id = Column(String, ForeignKey("roi_analyses.id"), nullable=False)
    
    # Cost details
    cost_category = Column(String(100), nullable=False)  # CostType enum
    cost_subcategory = Column(String(100))
    description = Column(Text, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    
    # Time tracking
    cost_date = Column(DateTime, nullable=False)
    billing_period_start = Column(DateTime)
    billing_period_end = Column(DateTime)
    is_recurring = Column(Boolean, default=False)
    recurrence_frequency = Column(String(50))  # monthly, quarterly, annually
    
    # Source tracking
    data_source = Column(String(255))  # Where the cost data came from
    vendor = Column(String(255))
    invoice_number = Column(String(100))
    cost_center = Column(String(100))
    
    # Automation flags
    is_automated_collection = Column(Boolean, default=False)
    collection_method = Column(String(100))  # api, manual, import
    last_verified = Column(DateTime)
    verification_status = Column(String(50), default="pending")
    
    # Metadata
    tags = Column(JSON, default=list)
    additional_metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    roi_analysis = relationship("ROIAnalysis", back_populates="cost_items")


class BenefitTracking(Base):
    """Benefit Tracking model for measuring and tracking project benefits."""
    __tablename__ = "benefit_tracking"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    roi_analysis_id = Column(String, ForeignKey("roi_analyses.id"), nullable=False)
    
    # Benefit details
    benefit_category = Column(String(100), nullable=False)  # BenefitType enum
    benefit_subcategory = Column(String(100))
    description = Column(Text, nullable=False)
    quantified_value = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    
    # Measurement details
    measurement_method = Column(String(255), nullable=False)
    baseline_value = Column(Float)
    current_value = Column(Float)
    target_value = Column(Float)
    measurement_unit = Column(String(50))
    
    # Time tracking
    benefit_date = Column(DateTime, nullable=False)
    measurement_period_start = Column(DateTime)
    measurement_period_end = Column(DateTime)
    is_recurring = Column(Boolean, default=False)
    
    # Realization tracking
    is_realized = Column(Boolean, default=False)
    realization_percentage = Column(Float, default=0.0)
    projected_realization_date = Column(DateTime)
    actual_realization_date = Column(DateTime)
    
    # Quality metrics
    confidence_level = Column(Float, default=0.8)
    data_quality_score = Column(Float)
    verification_status = Column(String(50), default="pending")
    last_verified = Column(DateTime)
    
    # Automation flags
    is_automated_measurement = Column(Boolean, default=False)
    measurement_source = Column(String(255))
    collection_method = Column(String(100))
    
    # Metadata
    tags = Column(JSON, default=list)
    additional_metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    roi_analysis = relationship("ROIAnalysis", back_populates="benefit_items")


class ROIReport(Base):
    """ROI Report model for storing generated ROI reports and visualizations."""
    __tablename__ = "roi_reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    roi_analysis_id = Column(String, ForeignKey("roi_analyses.id"), nullable=False)
    
    # Report details
    report_name = Column(String(255), nullable=False)
    report_type = Column(String(50), nullable=False)  # summary, detailed, executive, comparison
    report_format = Column(String(50), nullable=False)  # json, pdf, html, excel
    
    # Report content
    report_data = Column(JSON, nullable=False)  # Structured report data
    visualizations = Column(JSON, default=list)  # Chart and graph configurations
    executive_summary = Column(Text)
    key_findings = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    
    # Report metadata
    report_period_start = Column(DateTime, nullable=False)
    report_period_end = Column(DateTime, nullable=False)
    generated_by = Column(String, nullable=False)
    report_version = Column(String(20), default="1.0")
    
    # Distribution
    recipients = Column(JSON, default=list)
    distribution_method = Column(String(50))  # email, dashboard, api
    is_scheduled = Column(Boolean, default=False)
    schedule_frequency = Column(String(50))  # daily, weekly, monthly, quarterly
    next_generation_date = Column(DateTime)
    
    # Status
    generation_status = Column(String(50), default="completed")
    file_path = Column(String(500))  # Path to generated report file
    file_size_bytes = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    roi_analysis = relationship("ROIAnalysis", back_populates="roi_reports")


class CloudCostCollection(Base):
    """Cloud Cost Collection model for automated cost collection from cloud platforms."""
    __tablename__ = "cloud_cost_collection"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Cloud provider details
    provider = Column(String(50), nullable=False)  # aws, azure, gcp
    account_id = Column(String(255), nullable=False)
    account_name = Column(String(255))
    region = Column(String(100))
    
    # Service details
    service_name = Column(String(255), nullable=False)
    service_category = Column(String(100))
    resource_id = Column(String(500))
    resource_name = Column(String(255))
    
    # Cost details
    cost_amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    billing_period = Column(String(50), nullable=False)  # YYYY-MM format
    usage_start_date = Column(DateTime, nullable=False)
    usage_end_date = Column(DateTime, nullable=False)
    
    # Usage metrics
    usage_quantity = Column(Float)
    usage_unit = Column(String(50))
    rate = Column(Float)
    
    # Collection metadata
    collected_at = Column(DateTime, default=datetime.utcnow)
    collection_method = Column(String(100), nullable=False)  # api, billing_export, manual
    data_source_url = Column(String(500))
    raw_data = Column(JSON)  # Original data from provider
    
    # Processing status
    processing_status = Column(String(50), default="collected")
    processed_at = Column(DateTime)
    assigned_to_project = Column(String)  # Project ID if assigned
    cost_allocation_percentage = Column(Float, default=100.0)
    
    # Tags and metadata
    resource_tags = Column(JSON, default=dict)
    cost_tags = Column(JSON, default=dict)
    additional_metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProductivityMetric(Base):
    """Productivity Metric model for tracking productivity and efficiency gains."""
    __tablename__ = "productivity_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Metric identification
    metric_name = Column(String(255), nullable=False)
    metric_category = Column(String(100), nullable=False)  # development, operations, analysis, etc.
    metric_type = Column(String(50), nullable=False)  # time_saved, tasks_automated, etc.
    
    # Measurement details
    baseline_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    improvement_percentage = Column(Float)
    measurement_unit = Column(String(50), nullable=False)
    
    # Time tracking
    measurement_date = Column(DateTime, nullable=False)
    measurement_period_start = Column(DateTime, nullable=False)
    measurement_period_end = Column(DateTime, nullable=False)
    
    # Context
    team_or_department = Column(String(255))
    process_or_task = Column(String(255), nullable=False)
    tool_or_system = Column(String(255))
    project_id = Column(String)  # Link to ROI analysis
    
    # Measurement methodology
    measurement_method = Column(String(255), nullable=False)
    data_source = Column(String(255))
    sample_size = Column(Integer)
    confidence_level = Column(Float, default=0.8)
    
    # Automation and collection
    is_automated_collection = Column(Boolean, default=False)
    collection_frequency = Column(String(50))  # daily, weekly, monthly
    last_collected = Column(DateTime)
    
    # Metadata
    tags = Column(JSON, default=list)
    additional_context = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EfficiencyGainMetric(Base):
    """Efficiency Gain Metric model for measuring specific efficiency improvements."""
    __tablename__ = "efficiency_gain_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, nullable=False)
    
    # Efficiency measurement
    process_name = Column(String(255), nullable=False)
    efficiency_category = Column(String(100), nullable=False)  # automation, optimization, elimination
    
    # Before/after metrics
    time_before_hours = Column(Float, nullable=False)
    time_after_hours = Column(Float, nullable=False)
    time_saved_hours = Column(Float, nullable=False)
    time_saved_percentage = Column(Float, nullable=False)
    
    # Cost impact
    hourly_rate = Column(Float, default=50.0)  # Default hourly rate for calculations
    cost_savings_per_period = Column(Float, nullable=False)
    frequency_per_month = Column(Float, default=1.0)  # How often process occurs per month
    monthly_savings = Column(Float, nullable=False)
    annual_savings = Column(Float, nullable=False)
    
    # Quality metrics
    error_rate_before = Column(Float, default=0.0)
    error_rate_after = Column(Float, default=0.0)
    quality_improvement_percentage = Column(Float, default=0.0)
    
    # Measurement details
    measurement_date = Column(DateTime, default=datetime.utcnow)
    measurement_method = Column(String(255), nullable=False)
    data_collection_period_days = Column(Integer, default=30)
    sample_size = Column(Integer, default=1)
    
    # Validation
    is_validated = Column(Boolean, default=False)
    validated_by = Column(String(255))
    validation_date = Column(DateTime)
    validation_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)