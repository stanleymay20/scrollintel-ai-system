"""Add comprehensive billing system

Revision ID: billing_system_20250814_001300
Revises: 
Create Date: 2025-08-14T00:13:00.358501

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'billing_system_20250814_001300'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create billing system tables."""
    
    # Create subscription tiers enum
    subscription_tier_enum = postgresql.ENUM(
        'free', 'starter', 'professional', 'enterprise', 'sovereign',
        name='subscriptiontier'
    )
    subscription_tier_enum.create(op.get_bind())
    
    # Create billing cycle enum
    billing_cycle_enum = postgresql.ENUM(
        'monthly', 'yearly', 'quarterly',
        name='billingcycle'
    )
    billing_cycle_enum.create(op.get_bind())
    
    # Create subscription status enum
    subscription_status_enum = postgresql.ENUM(
        'active', 'inactive', 'cancelled', 'past_due', 'suspended', 'trialing',
        name='subscriptionstatus'
    )
    subscription_status_enum.create(op.get_bind())
    
    # Create payment status enum
    payment_status_enum = postgresql.ENUM(
        'pending', 'processing', 'succeeded', 'failed', 'cancelled', 
        'refunded', 'partially_refunded',
        name='paymentstatus'
    )
    payment_status_enum.create(op.get_bind())
    
    # Create transaction type enum
    transaction_type_enum = postgresql.ENUM(
        'subscription', 'usage', 'recharge', 'reward', 'refund', 'transfer', 'adjustment',
        name='transactiontype'
    )
    transaction_type_enum.create(op.get_bind())
    
    # Create payment method enum
    payment_method_enum = postgresql.ENUM(
        'credit_card', 'debit_card', 'bank_transfer', 'paypal', 'stripe', 'scrollcoin', 'invoice',
        name='paymentmethod'
    )
    payment_method_enum.create(op.get_bind())
    
    # Create usage metric type enum
    usage_metric_type_enum = postgresql.ENUM(
        'api_calls', 'model_inference', 'training_jobs', 'data_processing', 
        'storage_gb', 'compute_hours', 'scrollcoins',
        name='usagemetrictype'
    )
    usage_metric_type_enum.create(op.get_bind())
    
    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('tier', subscription_tier_enum, nullable=False, default='free'),
        sa.Column('status', subscription_status_enum, nullable=False, default='active'),
        sa.Column('billing_cycle', billing_cycle_enum, nullable=False, default='monthly'),
        sa.Column('base_price', sa.DECIMAL(10, 2), nullable=False, default=0.00),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('current_period_start', sa.DateTime, nullable=False),
        sa.Column('current_period_end', sa.DateTime, nullable=False),
        sa.Column('next_billing_date', sa.DateTime, nullable=True),
        sa.Column('trial_end', sa.DateTime, nullable=True),
        sa.Column('stripe_subscription_id', sa.String(255), nullable=True, unique=True),
        sa.Column('stripe_customer_id', sa.String(255), nullable=True),
        sa.Column('stripe_price_id', sa.String(255), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('cancelled_at', sa.DateTime, nullable=True),
        sa.Column('cancelled_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
    )
    
    # Create indexes for subscriptions
    op.create_index('idx_subscription_user_id', 'subscriptions', ['user_id'])
    op.create_index('idx_subscription_status', 'subscriptions', ['status'])
    op.create_index('idx_subscription_tier', 'subscriptions', ['tier'])
    op.create_index('idx_subscription_stripe_id', 'subscriptions', ['stripe_subscription_id'])
    op.create_index('idx_subscription_next_billing', 'subscriptions', ['next_billing_date'])
    
    # Create ScrollCoin wallets table
    op.create_table(
        'scrollcoin_wallets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False, unique=True),
        sa.Column('balance', sa.DECIMAL(15, 2), nullable=False, default=0.00),
        sa.Column('reserved_balance', sa.DECIMAL(15, 2), nullable=False, default=0.00),
        sa.Column('wallet_address', sa.String(255), nullable=True, unique=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('last_transaction_at', sa.DateTime, nullable=True),
        sa.CheckConstraint('balance >= 0', name='check_positive_balance'),
        sa.CheckConstraint('reserved_balance >= 0', name='check_positive_reserved'),
    )
    
    # Create indexes for wallets
    op.create_index('idx_wallet_user_id', 'scrollcoin_wallets', ['user_id'])
    op.create_index('idx_wallet_balance', 'scrollcoin_wallets', ['balance'])
    
    # Create ScrollCoin transactions table
    op.create_table(
        'scrollcoin_transactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('wallet_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('scrollcoin_wallets.id'), nullable=False),
        sa.Column('transaction_type', transaction_type_enum, nullable=False),
        sa.Column('amount', sa.DECIMAL(15, 2), nullable=False),
        sa.Column('balance_before', sa.DECIMAL(15, 2), nullable=False),
        sa.Column('balance_after', sa.DECIMAL(15, 2), nullable=False),
        sa.Column('reference_id', sa.String(255), nullable=True),
        sa.Column('reference_type', sa.String(100), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
    )
    
    # Create indexes for ScrollCoin transactions
    op.create_index('idx_scrollcoin_tx_wallet', 'scrollcoin_transactions', ['wallet_id'])
    op.create_index('idx_scrollcoin_tx_type', 'scrollcoin_transactions', ['transaction_type'])
    op.create_index('idx_scrollcoin_tx_created', 'scrollcoin_transactions', ['created_at'])
    op.create_index('idx_scrollcoin_tx_reference', 'scrollcoin_transactions', ['reference_id', 'reference_type'])
    
    # Create payments table
    op.create_table(
        'payments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('subscription_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('subscriptions.id'), nullable=True),
        sa.Column('invoice_id', postgresql.UUID(as_uuid=True), nullable=True),  # Will add FK later
        sa.Column('amount', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('status', payment_status_enum, nullable=False, default='pending'),
        sa.Column('payment_method', payment_method_enum, nullable=False),
        sa.Column('stripe_payment_intent_id', sa.String(255), nullable=True),
        sa.Column('stripe_charge_id', sa.String(255), nullable=True),
        sa.Column('external_transaction_id', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('failure_reason', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('processed_at', sa.DateTime, nullable=True),
        sa.Column('failed_at', sa.DateTime, nullable=True),
    )
    
    # Create indexes for payments
    op.create_index('idx_payment_user_id', 'payments', ['user_id'])
    op.create_index('idx_payment_subscription_id', 'payments', ['subscription_id'])
    op.create_index('idx_payment_status', 'payments', ['status'])
    op.create_index('idx_payment_stripe_intent', 'payments', ['stripe_payment_intent_id'])
    op.create_index('idx_payment_created', 'payments', ['created_at'])
    
    # Create payment refunds table
    op.create_table(
        'payment_refunds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('payment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('payments.id'), nullable=False),
        sa.Column('amount', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('reason', sa.String(255), nullable=True),
        sa.Column('status', payment_status_enum, nullable=False, default='pending'),
        sa.Column('stripe_refund_id', sa.String(255), nullable=True),
        sa.Column('external_refund_id', sa.String(255), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('processed_at', sa.DateTime, nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
    )
    
    # Create indexes for refunds
    op.create_index('idx_refund_payment_id', 'payment_refunds', ['payment_id'])
    op.create_index('idx_refund_status', 'payment_refunds', ['status'])
    op.create_index('idx_refund_created', 'payment_refunds', ['created_at'])
    
    # Create invoices table
    op.create_table(
        'invoices',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('subscription_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('subscriptions.id'), nullable=True),
        sa.Column('invoice_number', sa.String(100), nullable=False, unique=True),
        sa.Column('status', sa.String(50), nullable=False, default='draft'),
        sa.Column('subtotal', sa.DECIMAL(10, 2), nullable=False, default=0.00),
        sa.Column('tax_amount', sa.DECIMAL(10, 2), nullable=False, default=0.00),
        sa.Column('discount_amount', sa.DECIMAL(10, 2), nullable=False, default=0.00),
        sa.Column('total_amount', sa.DECIMAL(10, 2), nullable=False, default=0.00),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('period_start', sa.DateTime, nullable=False),
        sa.Column('period_end', sa.DateTime, nullable=False),
        sa.Column('issued_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('due_date', sa.DateTime, nullable=False),
        sa.Column('paid_at', sa.DateTime, nullable=True),
        sa.Column('stripe_invoice_id', sa.String(255), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )
    
    # Create indexes for invoices
    op.create_index('idx_invoice_user_id', 'invoices', ['user_id'])
    op.create_index('idx_invoice_subscription_id', 'invoices', ['subscription_id'])
    op.create_index('idx_invoice_number', 'invoices', ['invoice_number'])
    op.create_index('idx_invoice_status', 'invoices', ['status'])
    op.create_index('idx_invoice_due_date', 'invoices', ['due_date'])
    
    # Add foreign key constraint to payments table
    op.create_foreign_key('fk_payment_invoice', 'payments', 'invoices', ['invoice_id'], ['id'])
    
    # Create invoice line items table
    op.create_table(
        'invoice_line_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('invoice_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('invoices.id'), nullable=False),
        sa.Column('description', sa.String(500), nullable=False),
        sa.Column('quantity', sa.DECIMAL(10, 2), nullable=False, default=1.00),
        sa.Column('unit_price', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('total_price', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('item_type', sa.String(100), nullable=True),
        sa.Column('period_start', sa.DateTime, nullable=True),
        sa.Column('period_end', sa.DateTime, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
    )
    
    # Create indexes for line items
    op.create_index('idx_line_item_invoice_id', 'invoice_line_items', ['invoice_id'])
    op.create_index('idx_line_item_type', 'invoice_line_items', ['item_type'])
    
    # Create usage records table
    op.create_table(
        'usage_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('subscription_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('subscriptions.id'), nullable=True),
        sa.Column('metric_type', usage_metric_type_enum, nullable=False),
        sa.Column('quantity', sa.DECIMAL(15, 2), nullable=False),
        sa.Column('unit_cost', sa.DECIMAL(10, 4), nullable=True),
        sa.Column('total_cost', sa.DECIMAL(10, 2), nullable=True),
        sa.Column('usage_date', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('billing_period_start', sa.DateTime, nullable=True),
        sa.Column('billing_period_end', sa.DateTime, nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('resource_type', sa.String(100), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )
    
    # Create indexes for usage records
    op.create_index('idx_usage_user_id', 'usage_records', ['user_id'])
    op.create_index('idx_usage_subscription_id', 'usage_records', ['subscription_id'])
    op.create_index('idx_usage_metric_type', 'usage_records', ['metric_type'])
    op.create_index('idx_usage_date', 'usage_records', ['usage_date'])
    op.create_index('idx_usage_billing_period', 'usage_records', ['billing_period_start', 'billing_period_end'])
    
    # Create billing alerts table
    op.create_table(
        'billing_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('subscription_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('subscriptions.id'), nullable=True),
        sa.Column('alert_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(50), nullable=False, default='info'),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('is_read', sa.Boolean, nullable=False, default=False),
        sa.Column('is_dismissed', sa.Boolean, nullable=False, default=False),
        sa.Column('action_required', sa.Boolean, nullable=False, default=False),
        sa.Column('action_url', sa.String(500), nullable=True),
        sa.Column('action_text', sa.String(100), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('read_at', sa.DateTime, nullable=True),
        sa.Column('dismissed_at', sa.DateTime, nullable=True),
    )
    
    # Create indexes for alerts
    op.create_index('idx_alert_user_id', 'billing_alerts', ['user_id'])
    op.create_index('idx_alert_type', 'billing_alerts', ['alert_type'])
    op.create_index('idx_alert_severity', 'billing_alerts', ['severity'])
    op.create_index('idx_alert_unread', 'billing_alerts', ['user_id', 'is_read'])
    op.create_index('idx_alert_created', 'billing_alerts', ['created_at'])
    
    # Create payment methods table
    op.create_table(
        'payment_methods',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('is_default', sa.Boolean, nullable=False, default=False),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('last_four', sa.String(4), nullable=True),
        sa.Column('brand', sa.String(50), nullable=True),
        sa.Column('exp_month', sa.Integer, nullable=True),
        sa.Column('exp_year', sa.Integer, nullable=True),
        sa.Column('stripe_payment_method_id', sa.String(255), nullable=True),
        sa.Column('nickname', sa.String(100), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )
    
    # Create indexes for payment methods
    op.create_index('idx_payment_method_user_id', 'payment_methods', ['user_id'])
    op.create_index('idx_payment_method_default', 'payment_methods', ['user_id', 'is_default'])
    op.create_index('idx_payment_method_stripe', 'payment_methods', ['stripe_payment_method_id'])


def downgrade():
    """Drop billing system tables."""
    
    # Drop tables in reverse order
    op.drop_table('payment_methods')
    op.drop_table('billing_alerts')
    op.drop_table('usage_records')
    op.drop_table('invoice_line_items')
    op.drop_table('invoices')
    op.drop_table('payment_refunds')
    op.drop_table('payments')
    op.drop_table('scrollcoin_transactions')
    op.drop_table('scrollcoin_wallets')
    op.drop_table('subscriptions')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS usagemetrictype')
    op.execute('DROP TYPE IF EXISTS paymentmethod')
    op.execute('DROP TYPE IF EXISTS transactiontype')
    op.execute('DROP TYPE IF EXISTS paymentstatus')
    op.execute('DROP TYPE IF EXISTS subscriptionstatus')
    op.execute('DROP TYPE IF EXISTS billingcycle')
    op.execute('DROP TYPE IF EXISTS subscriptiontier')
