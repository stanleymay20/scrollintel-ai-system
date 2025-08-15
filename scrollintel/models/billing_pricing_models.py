"""
Billing and pricing models for visual content generation system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class PricingTier(Enum):
    """Pricing tiers for different user types."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class BillingCycle(Enum):
    """Billing cycle options."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    PAY_AS_YOU_GO = "pay_as_you_go"


class PaymentStatus(Enum):
    """Payment status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class InvoiceStatus(Enum):
    """Invoice status options."""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


@dataclass
class PricingRule:
    """Individual pricing rule for a service or resource."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    resource_type: str = ""  # gpu_seconds, api_calls, etc.
    
    # Pricing structure
    base_price: float = 0.0
    unit_price: float = 0.0
    currency: str = "USD"
    
    # Tier-specific pricing
    tier_multipliers: Dict[PricingTier, float] = field(default_factory=dict)
    
    # Volume discounts
    volume_tiers: List[Dict[str, Union[float, int]]] = field(default_factory=list)
    
    # Time-based pricing
    peak_hours_multiplier: float = 1.0
    off_peak_hours_multiplier: float = 0.8
    peak_hours: List[int] = field(default_factory=lambda: list(range(9, 17)))  # 9 AM - 5 PM
    
    # Validity
    effective_from: datetime = field(default_factory=datetime.utcnow)
    effective_until: Optional[datetime] = None
    active: bool = True


@dataclass
class SubscriptionPlan:
    """Subscription plan definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tier: PricingTier = PricingTier.BASIC
    
    # Pricing
    monthly_price: float = 0.0
    yearly_price: float = 0.0
    currency: str = "USD"
    
    # Included quotas
    included_image_generations: int = 0
    included_video_generations: int = 0
    included_enhancement_operations: int = 0
    included_gpu_seconds: float = 0.0
    included_storage_gb: float = 0.0
    included_bandwidth_gb: float = 0.0
    
    # Overage pricing
    overage_pricing: Dict[str, float] = field(default_factory=dict)
    
    # Features
    features: List[str] = field(default_factory=list)
    max_concurrent_generations: int = 1
    priority_processing: bool = False
    advanced_models_access: bool = False
    api_access: bool = False
    
    # Limits
    max_resolution: tuple = (1024, 1024)
    max_video_duration: float = 30.0
    max_batch_size: int = 1
    
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserSubscription:
    """User's active subscription."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    plan_id: str = ""
    plan_name: str = ""
    tier: PricingTier = PricingTier.FREE
    
    # Billing details
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    price: float = 0.0
    currency: str = "USD"
    
    # Subscription period
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    next_billing_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    
    # Usage tracking
    current_period_usage: Dict[str, float] = field(default_factory=dict)
    remaining_quotas: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: str = "active"  # active, cancelled, suspended, expired
    auto_renew: bool = True
    
    # Payment
    payment_method_id: Optional[str] = None
    last_payment_date: Optional[datetime] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InvoiceLineItem:
    """Individual line item on an invoice."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    total_price: float = 0.0
    currency: str = "USD"
    
    # Metadata
    resource_type: Optional[str] = None
    generation_ids: List[str] = field(default_factory=list)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


@dataclass
class Invoice:
    """Billing invoice."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    invoice_number: str = ""
    user_id: str = ""
    subscription_id: Optional[str] = None
    
    # Invoice details
    issue_date: datetime = field(default_factory=datetime.utcnow)
    due_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    
    # Line items
    line_items: List[InvoiceLineItem] = field(default_factory=list)
    
    # Totals
    subtotal: float = 0.0
    tax_rate: float = 0.0
    tax_amount: float = 0.0
    discount_amount: float = 0.0
    total_amount: float = 0.0
    currency: str = "USD"
    
    # Status
    status: InvoiceStatus = InvoiceStatus.DRAFT
    payment_status: PaymentStatus = PaymentStatus.PENDING
    
    # Payment details
    payment_method: Optional[str] = None
    payment_date: Optional[datetime] = None
    payment_reference: Optional[str] = None
    
    # Metadata
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PaymentMethod:
    """User payment method."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    
    # Payment method details
    type: str = "credit_card"  # credit_card, bank_account, paypal, etc.
    provider: str = "stripe"   # stripe, paypal, etc.
    provider_id: str = ""      # External provider's ID
    
    # Card details (if applicable)
    last_four: Optional[str] = None
    brand: Optional[str] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    
    # Status
    is_default: bool = False
    is_verified: bool = False
    active: bool = True
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Payment:
    """Payment record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    invoice_id: Optional[str] = None
    subscription_id: Optional[str] = None
    payment_method_id: str = ""
    
    # Payment details
    amount: float = 0.0
    currency: str = "USD"
    description: str = ""
    
    # External references
    provider: str = "stripe"
    provider_payment_id: str = ""
    provider_charge_id: Optional[str] = None
    
    # Status
    status: PaymentStatus = PaymentStatus.PENDING
    failure_reason: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimizationAlert:
    """Cost optimization alert for billing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    alert_type: str = ""
    
    # Alert details
    title: str = ""
    message: str = ""
    severity: str = "medium"  # low, medium, high, critical
    
    # Cost information
    current_cost: float = 0.0
    projected_cost: float = 0.0
    potential_savings: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Status
    acknowledged: bool = False
    dismissed: bool = False
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None


@dataclass
class BillingReport:
    """Billing report for analytics."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None  # None for system-wide reports
    report_type: str = "monthly"   # monthly, quarterly, yearly, custom
    
    # Report period
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    
    # Revenue metrics
    total_revenue: float = 0.0
    subscription_revenue: float = 0.0
    usage_revenue: float = 0.0
    
    # Usage metrics
    total_generations: int = 0
    total_gpu_seconds: float = 0.0
    total_storage_gb: float = 0.0
    
    # Customer metrics
    active_subscriptions: int = 0
    new_subscriptions: int = 0
    cancelled_subscriptions: int = 0
    churn_rate: float = 0.0
    
    # Cost metrics
    average_revenue_per_user: float = 0.0
    cost_per_generation: float = 0.0
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaxConfiguration:
    """Tax configuration for different regions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    country_code: str = ""
    state_code: Optional[str] = None
    
    # Tax details
    tax_name: str = ""
    tax_rate: float = 0.0
    tax_type: str = "percentage"  # percentage, fixed
    
    # Applicability
    applies_to_subscriptions: bool = True
    applies_to_usage: bool = True
    
    effective_from: datetime = field(default_factory=datetime.utcnow)
    effective_until: Optional[datetime] = None
    active: bool = True