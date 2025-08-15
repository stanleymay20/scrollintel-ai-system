"""
Dashboard and Widget data models for the Advanced Analytics Dashboard System.
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from enum import Enum
import uuid

from .database import Base


class DashboardType(Enum):
    EXECUTIVE = "executive"
    DEPARTMENT = "department"
    CUSTOM = "custom"
    TEMPLATE = "template"


class ExecutiveRole(Enum):
    CTO = "cto"
    CFO = "cfo"
    CEO = "ceo"
    VP_ENGINEERING = "vp_engineering"
    VP_DATA = "vp_data"
    DEPARTMENT_HEAD = "department_head"


class WidgetType(Enum):
    CHART = "chart"
    METRIC = "metric"
    TABLE = "table"
    MAP = "map"
    TEXT = "text"
    KPI = "kpi"


class AnalyticsDashboard(Base):
    __tablename__ = "analytics_dashboards"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # DashboardType enum
    owner_id = Column(String, nullable=False)
    role = Column(String(50))  # ExecutiveRole enum for executive dashboards
    description = Column(Text)
    config = Column(JSON)  # Dashboard configuration and layout
    is_template = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    widgets = relationship("Widget", back_populates="dashboard", cascade="all, delete-orphan")
    permissions = relationship("DashboardPermission", back_populates="dashboard", cascade="all, delete-orphan")


class Widget(Base):
    __tablename__ = "widgets"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dashboard_id = Column(String, ForeignKey("analytics_dashboards.id"), nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # WidgetType enum
    position_x = Column(Integer, default=0)
    position_y = Column(Integer, default=0)
    width = Column(Integer, default=4)
    height = Column(Integer, default=3)
    config = Column(JSON)  # Widget-specific configuration
    data_source = Column(String(255))  # Data source identifier
    query = Column(Text)  # Query or data retrieval specification
    refresh_interval = Column(Integer, default=300)  # Refresh interval in seconds
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dashboard = relationship("AnalyticsDashboard", back_populates="widgets")


class DashboardPermission(Base):
    __tablename__ = "dashboard_permissions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dashboard_id = Column(String, ForeignKey("analytics_dashboards.id"), nullable=False)
    user_id = Column(String, nullable=False)
    permission_type = Column(String(50), nullable=False)  # view, edit, admin
    granted_by = Column(String, nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    dashboard = relationship("AnalyticsDashboard", back_populates="permissions")


class DashboardTemplate(Base):
    __tablename__ = "dashboard_templates"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    category = Column(String(100))  # Industry or use case category
    role = Column(String(50))  # Target executive role
    description = Column(Text)
    template_config = Column(JSON)  # Template configuration
    preview_image = Column(String(500))  # URL to preview image
    is_public = Column(Boolean, default=False)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BusinessMetric(Base):
    __tablename__ = "business_metrics"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    category = Column(String(100))  # Metric category (financial, operational, etc.)
    value = Column(Float, nullable=False)
    unit = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(255))  # Data source
    context = Column(JSON)  # Additional context data
    dashboard_id = Column(String, ForeignKey("analytics_dashboards.id"))
    widget_id = Column(String, ForeignKey("widgets.id"))
    
    # Relationships
    dashboard = relationship("AnalyticsDashboard")
    widget = relationship("Widget")