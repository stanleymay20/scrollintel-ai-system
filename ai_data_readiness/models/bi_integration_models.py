"""
Data models for BI and Analytics tool integration
"""

from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, Optional

from .base_models import Base


class BITool(Base):
    """Model for registered BI and analytics tools"""
    __tablename__ = 'bi_tools'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    tool_type = Column(String(50), nullable=False)  # tableau, powerbi, looker, generic
    server_url = Column(String(500), nullable=False)
    workspace_id = Column(String(100))  # workspace/project ID in the BI tool
    credentials = Column(JSON)  # Encrypted credentials
    metadata = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    data_sources = relationship("BIDataSource", back_populates="bi_tool")
    reports = relationship("BIReport", back_populates="bi_tool")
    export_jobs = relationship("DataExportJob", back_populates="bi_tool")


class BIDataSource(Base):
    """Model for data sources in BI tools"""
    __tablename__ = 'bi_data_sources'
    
    id = Column(String(50), primary_key=True)
    bi_tool_id = Column(String(50), ForeignKey('bi_tools.id'), nullable=False)
    source_id = Column(String(100), nullable=False)  # ID in the BI tool
    source_name = Column(String(200), nullable=False)
    connection_type = Column(String(50), nullable=False)
    dataset_id = Column(String(50))  # Associated dataset from our platform
    last_refresh = Column(DateTime)
    refresh_status = Column(String(50), default='unknown')  # success, failed, in_progress
    row_count = Column(Integer)
    column_count = Column(Integer)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bi_tool = relationship("BITool", back_populates="data_sources")
    refresh_history = relationship("DataSourceRefresh", back_populates="data_source")


class BIReport(Base):
    """Model for reports/dashboards in BI tools"""
    __tablename__ = 'bi_reports'
    
    id = Column(String(50), primary_key=True)
    bi_tool_id = Column(String(50), ForeignKey('bi_tools.id'), nullable=False)
    report_id = Column(String(100), nullable=False)  # ID in the BI tool
    report_name = Column(String(200), nullable=False)
    report_type = Column(String(50))  # dashboard, report, workbook
    data_source_ids = Column(JSON)  # List of associated data source IDs
    report_url = Column(String(500))
    is_published = Column(Boolean, default=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bi_tool = relationship("BITool", back_populates="reports")
    distributions = relationship("ReportDistribution", back_populates="report")


class DataExportJob(Base):
    """Model for tracking data export jobs"""
    __tablename__ = 'data_export_jobs'
    
    id = Column(String(50), primary_key=True)
    bi_tool_id = Column(String(50), ForeignKey('bi_tools.id'))
    dataset_id = Column(String(50), nullable=False)
    export_format = Column(String(20), nullable=False)  # csv, json, parquet, excel
    export_config = Column(JSON)  # Export configuration details
    file_path = Column(String(500))
    file_size_bytes = Column(Integer)
    row_count = Column(Integer)
    status = Column(String(50), nullable=False)  # pending, in_progress, completed, failed
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    bi_tool = relationship("BITool", back_populates="export_jobs")


class ReportDistribution(Base):
    """Model for report distribution configurations"""
    __tablename__ = 'report_distributions'
    
    id = Column(String(50), primary_key=True)
    report_id = Column(String(50), ForeignKey('bi_reports.id'), nullable=False)
    distribution_name = Column(String(200), nullable=False)
    recipients = Column(JSON, nullable=False)  # List of email addresses
    distribution_schedule = Column(String(100))  # cron expression or description
    format = Column(String(20))  # pdf, excel, image
    is_active = Column(Boolean, default=True)
    last_sent = Column(DateTime)
    next_scheduled = Column(DateTime)
    send_count = Column(Integer, default=0)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    report = relationship("BIReport", back_populates="distributions")
    distribution_history = relationship("DistributionHistory", back_populates="distribution")


class DistributionHistory(Base):
    """Model for tracking report distribution history"""
    __tablename__ = 'distribution_history'
    
    id = Column(String(50), primary_key=True)
    distribution_id = Column(String(50), ForeignKey('report_distributions.id'), nullable=False)
    sent_at = Column(DateTime, nullable=False)
    recipients_sent = Column(JSON)  # List of recipients who received the report
    status = Column(String(50), nullable=False)  # sent, failed, partial
    error_message = Column(Text)
    file_size_bytes = Column(Integer)
    metadata = Column(JSON)
    
    # Relationships
    distribution = relationship("ReportDistribution", back_populates="distribution_history")


class DataSourceRefresh(Base):
    """Model for tracking data source refresh history"""
    __tablename__ = 'data_source_refreshes'
    
    id = Column(String(50), primary_key=True)
    data_source_id = Column(String(50), ForeignKey('bi_data_sources.id'), nullable=False)
    refresh_type = Column(String(50), nullable=False)  # manual, scheduled, triggered
    status = Column(String(50), nullable=False)  # success, failed, in_progress
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    rows_processed = Column(Integer)
    error_message = Column(Text)
    metadata = Column(JSON)
    
    # Relationships
    data_source = relationship("BIDataSource", back_populates="refresh_history")


class BIIntegrationSync(Base):
    """Model for tracking synchronization with BI tools"""
    __tablename__ = 'bi_integration_syncs'
    
    id = Column(String(50), primary_key=True)
    bi_tool_id = Column(String(50), ForeignKey('bi_tools.id'), nullable=False)
    sync_type = Column(String(50), nullable=False)  # data_sources, reports, full
    sync_status = Column(String(50), nullable=False)  # success, failed, in_progress
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    items_processed = Column(Integer, default=0)
    items_created = Column(Integer, default=0)
    items_updated = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    error_message = Column(Text)
    sync_details = Column(JSON)
    
    # Relationships
    bi_tool = relationship("BITool")


class DataExportTemplate(Base):
    """Model for data export templates"""
    __tablename__ = 'data_export_templates'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    export_format = Column(String(20), nullable=False)
    export_config = Column(JSON, nullable=False)  # Template configuration
    filters = Column(JSON)  # Default filters to apply
    is_default = Column(Boolean, default=False)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BIIntegrationEvent(Base):
    """Model for tracking BI integration events"""
    __tablename__ = 'bi_integration_events'
    
    id = Column(String(50), primary_key=True)
    bi_tool_id = Column(String(50), ForeignKey('bi_tools.id'))
    event_type = Column(String(50), nullable=False)  # connection, export, sync, error
    event_data = Column(JSON)
    status = Column(String(50), nullable=False)  # success, failed, warning
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    bi_tool = relationship("BITool")