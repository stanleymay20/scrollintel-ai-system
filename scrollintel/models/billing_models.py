"""
Billing and Subscription Database Models

Comprehensive models for subscription management, billing, payments,
and usage tracking with full audit trail and compliance features.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
from uuid import uuid4
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum as SQLEnum, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, DECIMAL
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
import enum

from .database import Base


class SubscriptionTier(enum.Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    SOVEREIGN = "sovereign"


class BillingCycle(enum.Enum):
    """Billing cycle enumeration."""
    MONTHLY = "monthly"
    YEARLY = "yearly"
    QUARTERLY = "quarterly"


class SubscriptionStatus(enum.Enum):
    """Subscription status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    SUSPENDED = "suspended"
    TRIALING = "trialing"


class PaymentStatus(enum.Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class TransactionType(enum.Enum):
    """Transaction type enumeration."""
    SUBSCRIPTION = "subscription"
    USAGE = "usage"
    RECHARGE = "recharge"
    REWARD = "reward"
    REFUND = "refund"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"


class PaymentMethod(enum.Enum):
    """Payment method enumeration."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    STRIPE = "stripe"
    SCROLLCOIN = "scrollcoin"
    INVOICE = "invoice"


class UsageMetricType(enum.Enum):
    """Usage metric type enumeration."""
    API_CALLS = "api_calls"
    MODEL_INFERENCE = "model_inference"
    TRAINING_JOBS = "training_jobs"
    DATA_PROCESSING = "data_processing"
    STORAGE_GB = "storage_gb"
    COMPUTE_HOURS = "compute_hours"
    SCROLLCOINS = "scrollcoins"


class Subscription(Base):
    """User subscription model."""
    
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Subscription details
    tier = Column(SQLEnum(SubscriptionTier), nullable=False, default=SubscriptionTier.FREE)
    status = Column(SQLEnum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.ACTIVE)
    billing_cycle = Column(SQLEnum(BillingCycle), nullable=False, default=BillingCycle.MONTHLY)
    
    # Pricing
    base_price = Column(DECIMAL(10, 2), nullable=False, default=0.00)
    currency = Column(String(3), nullable=False, default="USD")
    
    # Billing dates
    current_period_start = Column(DateTime, nullable=False, default=datetime.utcnow)
    current_period_end = Column(DateTime, nullable=False)
    next_billing_date = Column(DateTime, nullable=True)
    trial_end = Column(DateTime, nullable=True)
    
    # Stripe integration
    stripe_subscription_id = Column(String(255), nullable=True, unique=True)
    stripe_customer_id = Column(String(255), nullable=True)
    stripe_price_id = Column(String(255), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at = Column(DateTime, nullable=True)
    cancelled_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="subscriptions")
    cancelled_by_user = relationship("User", foreign_keys=[cancelled_by])
    payments = relationship("Payment", back_populates="subscription", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="subscription", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="subscription", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_subscription_user_id", "user_id"),
        Index("idx_subscription_status", "status"),
        Index("idx_subscription_tier", "tier"),
        Index("idx_subscription_stripe_id", "stripe_subscription_id"),
        Index("idx_subscription_next_billing", "next_billing_date"),
    )
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status == SubscriptionStatus.ACTIVE
    
    @hybrid_property
    def is_trial(self) -> bool:
        """Check if subscription is in trial period."""
        return (
            self.status == SubscriptionStatus.TRIALING or
            (self.trial_end and self.trial_end > datetime.utcnow())
        )
    
    @hybrid_property
    def days_until_renewal(self) -> Optional[int]:
        """Calculate days until next renewal."""
        if not self.next_billing_date:
            return None
        delta = self.next_billing_date - datetime.utcnow()
        return max(0, delta.days)
    
    def calculate_next_billing_date(self) -> datetime:
        """Calculate next billing date based on cycle."""
        if self.billing_cycle == BillingCycle.MONTHLY:
            return self.current_period_end + timedelta(days=30)
        elif self.billing_cycle == BillingCycle.QUARTERLY:
            return self.current_period_end + timedelta(days=90)
        elif self.billing_cycle == BillingCycle.YEARLY:
            return self.current_period_end + timedelta(days=365)
        return self.current_period_end
    
    def get_tier_limits(self) -> Dict[str, Any]:
        """Get usage limits for current tier."""
        tier_limits = {
            SubscriptionTier.FREE: {
                "api_calls": 1000,
                "training_jobs": 1,
                "storage_gb": 1,
                "scrollcoins": 100,
                "features": ["basic_analysis", "simple_visualizations"]
            },
            SubscriptionTier.STARTER: {
                "api_calls": 10000,
                "training_jobs": 10,
                "storage_gb": 10,
                "scrollcoins": 1000,
                "features": ["advanced_analysis", "ml_training", "basic_explanations"]
            },
            SubscriptionTier.PROFESSIONAL: {
                "api_calls": 100000,
                "training_jobs": 50,
                "storage_gb": 100,
                "scrollcoins": 5000,
                "features": ["full_ml_suite", "advanced_explanations", "multimodal_ai"]
            },
            SubscriptionTier.ENTERPRISE: {
                "api_calls": 1000000,
                "training_jobs": 200,
                "storage_gb": 1000,
                "scrollcoins": 20000,
                "features": ["enterprise_features", "priority_support", "custom_models"]
            },
            SubscriptionTier.SOVEREIGN: {
                "api_calls": -1,  # Unlimited
                "training_jobs": -1,
                "storage_gb": -1,
                "scrollcoins": 100000,
                "features": ["full_sovereignty", "white_label", "on_premise"]
            }
        }
        return tier_limits.get(self.tier, tier_limits[SubscriptionTier.FREE])


class ScrollCoinWallet(Base):
    """ScrollCoin wallet for internal token system."""
    
    __tablename__ = "scrollcoin_wallets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Balance
    balance = Column(DECIMAL(15, 2), nullable=False, default=0.00)
    reserved_balance = Column(DECIMAL(15, 2), nullable=False, default=0.00)  # For pending transactions
    
    # Metadata
    wallet_address = Column(String(255), nullable=True, unique=True)
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_transaction_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="scrollcoin_wallet")
    transactions = relationship("ScrollCoinTransaction", back_populates="wallet", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("balance >= 0", name="check_positive_balance"),
        CheckConstraint("reserved_balance >= 0", name="check_positive_reserved"),
        Index("idx_wallet_user_id", "user_id"),
        Index("idx_wallet_balance", "balance"),
    )
    
    @hybrid_property
    def available_balance(self) -> Decimal:
        """Calculate available balance (total - reserved)."""
        return self.balance - self.reserved_balance
    
    def can_spend(self, amount: Decimal) -> bool:
        """Check if wallet has sufficient available balance."""
        return self.available_balance >= amount
    
    def reserve_amount(self, amount: Decimal) -> bool:
        """Reserve amount for pending transaction."""
        if self.can_spend(amount):
            self.reserved_balance += amount
            return True
        return False
    
    def release_reservation(self, amount: Decimal):
        """Release reserved amount."""
        self.reserved_balance = max(0, self.reserved_balance - amount)
    
    def debit(self, amount: Decimal) -> bool:
        """Debit amount from wallet."""
        if self.balance >= amount:
            self.balance -= amount
            self.last_transaction_at = datetime.utcnow()
            return True
        return False
    
    def credit(self, amount: Decimal):
        """Credit amount to wallet."""
        self.balance += amount
        self.last_transaction_at = datetime.utcnow()


class ScrollCoinTransaction(Base):
    """ScrollCoin transaction history."""
    
    __tablename__ = "scrollcoin_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    wallet_id = Column(UUID(as_uuid=True), ForeignKey("scrollcoin_wallets.id"), nullable=False)
    
    # Transaction details
    transaction_type = Column(SQLEnum(TransactionType), nullable=False)
    amount = Column(DECIMAL(15, 2), nullable=False)
    balance_before = Column(DECIMAL(15, 2), nullable=False)
    balance_after = Column(DECIMAL(15, 2), nullable=False)
    
    # Reference information
    reference_id = Column(String(255), nullable=True)  # External reference
    reference_type = Column(String(100), nullable=True)  # Type of reference
    description = Column(Text, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Relationships
    wallet = relationship("ScrollCoinWallet", back_populates="transactions")
    created_by_user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("idx_scrollcoin_tx_wallet", "wallet_id"),
        Index("idx_scrollcoin_tx_type", "transaction_type"),
        Index("idx_scrollcoin_tx_created", "created_at"),
        Index("idx_scrollcoin_tx_reference", "reference_id", "reference_type"),
    )


class Payment(Base):
    """Payment records for subscriptions and purchases."""
    
    __tablename__ = "payments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=True)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id"), nullable=True)
    
    # Payment details
    amount = Column(DECIMAL(10, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(PaymentStatus), nullable=False, default=PaymentStatus.PENDING)
    payment_method = Column(SQLEnum(PaymentMethod), nullable=False)
    
    # External payment processor details
    stripe_payment_intent_id = Column(String(255), nullable=True)
    stripe_charge_id = Column(String(255), nullable=True)
    external_transaction_id = Column(String(255), nullable=True)
    
    # Payment metadata
    description = Column(Text, nullable=True)
    failure_reason = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription", back_populates="payments")
    invoice = relationship("Invoice", back_populates="payments")
    refunds = relationship("PaymentRefund", back_populates="payment", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_payment_user_id", "user_id"),
        Index("idx_payment_subscription_id", "subscription_id"),
        Index("idx_payment_status", "status"),
        Index("idx_payment_stripe_intent", "stripe_payment_intent_id"),
        Index("idx_payment_created", "created_at"),
    )
    
    @hybrid_property
    def is_successful(self) -> bool:
        """Check if payment was successful."""
        return self.status == PaymentStatus.SUCCEEDED
    
    @hybrid_property
    def refunded_amount(self) -> Decimal:
        """Calculate total refunded amount."""
        return sum(refund.amount for refund in self.refunds if refund.status == PaymentStatus.SUCCEEDED)
    
    @hybrid_property
    def net_amount(self) -> Decimal:
        """Calculate net amount after refunds."""
        return self.amount - self.refunded_amount


class PaymentRefund(Base):
    """Payment refund records."""
    
    __tablename__ = "payment_refunds"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    payment_id = Column(UUID(as_uuid=True), ForeignKey("payments.id"), nullable=False)
    
    # Refund details
    amount = Column(DECIMAL(10, 2), nullable=False)
    reason = Column(String(255), nullable=True)
    status = Column(SQLEnum(PaymentStatus), nullable=False, default=PaymentStatus.PENDING)
    
    # External processor details
    stripe_refund_id = Column(String(255), nullable=True)
    external_refund_id = Column(String(255), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Relationships
    payment = relationship("Payment", back_populates="refunds")
    created_by_user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("idx_refund_payment_id", "payment_id"),
        Index("idx_refund_status", "status"),
        Index("idx_refund_created", "created_at"),
    )


class Invoice(Base):
    """Invoice records for billing periods."""
    
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=True)
    
    # Invoice details
    invoice_number = Column(String(100), nullable=False, unique=True)
    status = Column(String(50), nullable=False, default="draft")  # draft, sent, paid, overdue, cancelled
    
    # Amounts
    subtotal = Column(DECIMAL(10, 2), nullable=False, default=0.00)
    tax_amount = Column(DECIMAL(10, 2), nullable=False, default=0.00)
    discount_amount = Column(DECIMAL(10, 2), nullable=False, default=0.00)
    total_amount = Column(DECIMAL(10, 2), nullable=False, default=0.00)
    currency = Column(String(3), nullable=False, default="USD")
    
    # Billing period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Due dates
    issued_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    due_date = Column(DateTime, nullable=False)
    paid_at = Column(DateTime, nullable=True)
    
    # External integration
    stripe_invoice_id = Column(String(255), nullable=True)
    
    # Metadata
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="invoices")
    subscription = relationship("Subscription", back_populates="invoices")
    line_items = relationship("InvoiceLineItem", back_populates="invoice", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="invoice", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_invoice_user_id", "user_id"),
        Index("idx_invoice_subscription_id", "subscription_id"),
        Index("idx_invoice_number", "invoice_number"),
        Index("idx_invoice_status", "status"),
        Index("idx_invoice_due_date", "due_date"),
    )
    
    @hybrid_property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        return self.status != "paid" and self.due_date < datetime.utcnow()
    
    @hybrid_property
    def days_overdue(self) -> int:
        """Calculate days overdue."""
        if not self.is_overdue:
            return 0
        return (datetime.utcnow() - self.due_date).days


class InvoiceLineItem(Base):
    """Individual line items on invoices."""
    
    __tablename__ = "invoice_line_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id"), nullable=False)
    
    # Line item details
    description = Column(String(500), nullable=False)
    quantity = Column(DECIMAL(10, 2), nullable=False, default=1.00)
    unit_price = Column(DECIMAL(10, 2), nullable=False)
    total_price = Column(DECIMAL(10, 2), nullable=False)
    
    # Categorization
    item_type = Column(String(100), nullable=True)  # subscription, usage, addon, etc.
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    invoice = relationship("Invoice", back_populates="line_items")
    
    # Indexes
    __table_args__ = (
        Index("idx_line_item_invoice_id", "invoice_id"),
        Index("idx_line_item_type", "item_type"),
    )


class UsageRecord(Base):
    """Usage tracking for billing and analytics."""
    
    __tablename__ = "usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=True)
    
    # Usage details
    metric_type = Column(SQLEnum(UsageMetricType), nullable=False)
    quantity = Column(DECIMAL(15, 2), nullable=False)
    unit_cost = Column(DECIMAL(10, 4), nullable=True)  # Cost per unit
    total_cost = Column(DECIMAL(10, 2), nullable=True)  # Total cost for this usage
    
    # Time tracking
    usage_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    billing_period_start = Column(DateTime, nullable=True)
    billing_period_end = Column(DateTime, nullable=True)
    
    # Reference information
    resource_id = Column(String(255), nullable=True)  # ID of resource used
    resource_type = Column(String(100), nullable=True)  # Type of resource
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="usage_records")
    subscription = relationship("Subscription", back_populates="usage_records")
    
    # Indexes
    __table_args__ = (
        Index("idx_usage_user_id", "user_id"),
        Index("idx_usage_subscription_id", "subscription_id"),
        Index("idx_usage_metric_type", "metric_type"),
        Index("idx_usage_date", "usage_date"),
        Index("idx_usage_billing_period", "billing_period_start", "billing_period_end"),
    )


class BillingAlert(Base):
    """Billing alerts and notifications."""
    
    __tablename__ = "billing_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=True)
    
    # Alert details
    alert_type = Column(String(100), nullable=False)  # usage_limit, payment_failed, trial_ending, etc.
    severity = Column(String(50), nullable=False, default="info")  # info, warning, critical
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Status
    is_read = Column(Boolean, nullable=False, default=False)
    is_dismissed = Column(Boolean, nullable=False, default=False)
    
    # Actions
    action_required = Column(Boolean, nullable=False, default=False)
    action_url = Column(String(500), nullable=True)
    action_text = Column(String(100), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)
    dismissed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="billing_alerts")
    subscription = relationship("Subscription")
    
    # Indexes
    __table_args__ = (
        Index("idx_alert_user_id", "user_id"),
        Index("idx_alert_type", "alert_type"),
        Index("idx_alert_severity", "severity"),
        Index("idx_alert_unread", "user_id", "is_read"),
        Index("idx_alert_created", "created_at"),
    )


class PaymentMethod(Base):
    """Stored payment methods for users."""
    
    __tablename__ = "payment_methods"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Payment method details
    type = Column(String(50), nullable=False)  # card, bank_account, paypal, etc.
    is_default = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Card details (encrypted/tokenized)
    last_four = Column(String(4), nullable=True)
    brand = Column(String(50), nullable=True)  # visa, mastercard, etc.
    exp_month = Column(Integer, nullable=True)
    exp_year = Column(Integer, nullable=True)
    
    # External processor details
    stripe_payment_method_id = Column(String(255), nullable=True)
    
    # Metadata
    nickname = Column(String(100), nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="payment_methods")
    
    # Indexes
    __table_args__ = (
        Index("idx_payment_method_user_id", "user_id"),
        Index("idx_payment_method_default", "user_id", "is_default"),
        Index("idx_payment_method_stripe", "stripe_payment_method_id"),
    )
    
    @validates('is_default')
    def validate_default(self, key, value):
        """Ensure only one default payment method per user."""
        if value and self.user_id:
            # This would need to be handled in the application logic
            # to ensure only one default per user
            pass
        return value


# Add relationships to User model (this would be added to the existing User model)
"""
Add these relationships to the existing User model in database.py:

# Billing relationships
subscriptions = relationship("Subscription", foreign_keys="Subscription.user_id", back_populates="user", cascade="all, delete-orphan")
scrollcoin_wallet = relationship("ScrollCoinWallet", back_populates="user", uselist=False, cascade="all, delete-orphan")
payments = relationship("Payment", back_populates="user", cascade="all, delete-orphan")
invoices = relationship("Invoice", back_populates="user", cascade="all, delete-orphan")
usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")
billing_alerts = relationship("BillingAlert", back_populates="user", cascade="all, delete-orphan")
payment_methods = relationship("PaymentMethod", back_populates="user", cascade="all, delete-orphan")
"""