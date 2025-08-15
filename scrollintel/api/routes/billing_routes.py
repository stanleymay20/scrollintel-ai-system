"""
Billing and Subscription API Routes

Comprehensive REST API endpoints for subscription management, billing,
payments, and usage tracking with full Stripe integration.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import stripe
import os
import json

from scrollintel.models.database_utils import get_db
from scrollintel.models.billing_models import (
    Subscription, ScrollCoinWallet, Payment, Invoice, UsageRecord,
    BillingAlert, PaymentMethodModel,
    SubscriptionTier, BillingCycle, SubscriptionStatus, PaymentStatus,
    TransactionType, PaymentMethod, UsageMetricType, ScrollCoinTransaction
)
from scrollintel.engines.billing_engine import ScrollBillingEngine
from scrollintel.security.auth import get_current_user
from scrollintel.core.error_handling import handle_api_error

logger = logging.getLogger(__name__)
security = HTTPBearer()
router = APIRouter(prefix="/api/billing", tags=["billing"])

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
billing_engine = ScrollBillingEngine()


# Pydantic models for request/response
from pydantic import BaseModel, Field

class SubscriptionCreateRequest(BaseModel):
    tier: str = Field(..., description="Subscription tier")
    billing_cycle: str = Field(default="monthly", description="Billing cycle")
    payment_method_id: Optional[str] = Field(None, description="Stripe payment method ID")
    trial_days: Optional[int] = Field(None, description="Trial period in days")

class SubscriptionUpdateRequest(BaseModel):
    tier: Optional[str] = None
    billing_cycle: Optional[str] = None
    payment_method_id: Optional[str] = None

class PaymentMethodRequest(BaseModel):
    stripe_payment_method_id: str
    is_default: bool = False
    nickname: Optional[str] = None

class ScrollCoinRechargeRequest(BaseModel):
    amount: Decimal = Field(..., description="Amount in USD to charge")
    payment_method_id: str = Field(..., description="Payment method to use")

class UsageTrackingRequest(BaseModel):
    metric_type: str
    quantity: Decimal
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.get("/subscription")
async def get_current_subscription(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get current user's subscription details."""
    try:
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.ACTIVE
        ).first()
        
        if not subscription:
            # Create free tier subscription if none exists
            subscription = Subscription(
                user_id=current_user.id,
                tier=SubscriptionTier.FREE,
                status=SubscriptionStatus.ACTIVE,
                billing_cycle=BillingCycle.MONTHLY,
                base_price=Decimal('0.00'),
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
            db.add(subscription)
            db.commit()
            db.refresh(subscription)
        
        # Get usage statistics
        usage_stats = await _get_usage_statistics(current_user.id, db)
        
        # Get tier limits
        tier_limits = subscription.get_tier_limits()
        
        return {
            "subscription": {
                "id": str(subscription.id),
                "tier": subscription.tier.value,
                "status": subscription.status.value,
                "billing_cycle": subscription.billing_cycle.value,
                "base_price": float(subscription.base_price),
                "currency": subscription.currency,
                "current_period_start": subscription.current_period_start.isoformat(),
                "current_period_end": subscription.current_period_end.isoformat(),
                "next_billing_date": subscription.next_billing_date.isoformat() if subscription.next_billing_date else None,
                "trial_end": subscription.trial_end.isoformat() if subscription.trial_end else None,
                "is_trial": subscription.is_trial,
                "days_until_renewal": subscription.days_until_renewal
            },
            "tier_limits": tier_limits,
            "usage_stats": usage_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve subscription")


@router.post("/subscription")
async def create_subscription(
    request: SubscriptionCreateRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Create or upgrade subscription."""
    try:
        # Validate tier
        try:
            tier = SubscriptionTier(request.tier.lower())
            billing_cycle = BillingCycle(request.billing_cycle.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tier or billing cycle")
        
        # Check for existing subscription
        existing_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.ACTIVE
        ).first()
        
        # Get pricing
        tier_pricing = _get_tier_pricing(tier, billing_cycle)
        
        # Create Stripe subscription if not free tier
        stripe_subscription_id = None
        stripe_customer_id = None
        
        if tier != SubscriptionTier.FREE and request.payment_method_id:
            stripe_result = await _create_stripe_subscription(
                current_user, tier, billing_cycle, request.payment_method_id, request.trial_days
            )
            stripe_subscription_id = stripe_result["subscription_id"]
            stripe_customer_id = stripe_result["customer_id"]
        
        # Calculate dates
        now = datetime.utcnow()
        trial_end = now + timedelta(days=request.trial_days) if request.trial_days else None
        period_end = now + timedelta(days=30 if billing_cycle == BillingCycle.MONTHLY else 365)
        
        if existing_subscription:
            # Update existing subscription
            existing_subscription.tier = tier
            existing_subscription.billing_cycle = billing_cycle
            existing_subscription.base_price = tier_pricing["price"]
            existing_subscription.stripe_subscription_id = stripe_subscription_id
            existing_subscription.stripe_customer_id = stripe_customer_id
            existing_subscription.trial_end = trial_end
            existing_subscription.current_period_end = period_end
            existing_subscription.next_billing_date = existing_subscription.calculate_next_billing_date()
            existing_subscription.updated_at = now
            
            subscription = existing_subscription
        else:
            # Create new subscription
            subscription = Subscription(
                user_id=current_user.id,
                tier=tier,
                status=SubscriptionStatus.TRIALING if trial_end else SubscriptionStatus.ACTIVE,
                billing_cycle=billing_cycle,
                base_price=tier_pricing["price"],
                current_period_start=now,
                current_period_end=period_end,
                next_billing_date=period_end,
                trial_end=trial_end,
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id
            )
            db.add(subscription)
        
        db.commit()
        db.refresh(subscription)
        
        # Initialize ScrollCoin wallet if needed
        if tier != SubscriptionTier.FREE:
            await _ensure_scrollcoin_wallet(current_user.id, tier_pricing["scrollcoins_included"], db)
        
        # Send confirmation email
        background_tasks.add_task(_send_subscription_confirmation, current_user.email, subscription)
        
        return {
            "success": True,
            "message": f"Subscription {'updated' if existing_subscription else 'created'} successfully",
            "subscription_id": str(subscription.id),
            "tier": tier.value,
            "trial_end": trial_end.isoformat() if trial_end else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create subscription")


@router.put("/subscription")
async def update_subscription(
    request: SubscriptionUpdateRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Update existing subscription."""
    try:
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.ACTIVE
        ).first()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        # Update fields
        if request.tier:
            try:
                new_tier = SubscriptionTier(request.tier.lower())
                old_tier = subscription.tier
                
                # Handle tier change
                if new_tier != old_tier:
                    tier_pricing = _get_tier_pricing(new_tier, subscription.billing_cycle)
                    subscription.tier = new_tier
                    subscription.base_price = tier_pricing["price"]
                    
                    # Update Stripe subscription if needed
                    if subscription.stripe_subscription_id:
                        await _update_stripe_subscription(subscription.stripe_subscription_id, new_tier)
                    
                    # Create billing alert for tier change
                    await _create_billing_alert(
                        current_user.id,
                        "tier_change",
                        f"Subscription upgraded from {old_tier.value} to {new_tier.value}",
                        f"Your subscription has been upgraded. New features are now available.",
                        db
                    )
                    
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid tier")
        
        if request.billing_cycle:
            try:
                new_cycle = BillingCycle(request.billing_cycle.lower())
                subscription.billing_cycle = new_cycle
                subscription.next_billing_date = subscription.calculate_next_billing_date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid billing cycle")
        
        if request.payment_method_id and subscription.stripe_customer_id:
            # Update default payment method in Stripe
            stripe.Customer.modify(
                subscription.stripe_customer_id,
                invoice_settings={"default_payment_method": request.payment_method_id}
            )
        
        subscription.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "success": True,
            "message": "Subscription updated successfully",
            "subscription_id": str(subscription.id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update subscription")


@router.delete("/subscription")
async def cancel_subscription(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Cancel current subscription."""
    try:
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.ACTIVE
        ).first()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        # Cancel Stripe subscription
        if subscription.stripe_subscription_id:
            stripe.Subscription.modify(
                subscription.stripe_subscription_id,
                cancel_at_period_end=True
            )
        
        # Update subscription status
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.cancelled_at = datetime.utcnow()
        subscription.cancelled_by = current_user.id
        subscription.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "success": True,
            "message": "Subscription cancelled successfully",
            "cancellation_date": subscription.cancelled_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")


@router.get("/wallet")
async def get_scrollcoin_wallet(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get ScrollCoin wallet details."""
    try:
        wallet = db.query(ScrollCoinWallet).filter(
            ScrollCoinWallet.user_id == current_user.id
        ).first()
        
        if not wallet:
            # Create wallet if it doesn't exist
            wallet = ScrollCoinWallet(user_id=current_user.id)
            db.add(wallet)
            db.commit()
            db.refresh(wallet)
        
        # Get recent transactions
        from scrollintel.models.billing_models import ScrollCoinTransaction
        recent_transactions = db.query(ScrollCoinTransaction).filter(
            ScrollCoinTransaction.wallet_id == wallet.id
        ).order_by(ScrollCoinTransaction.created_at.desc()).limit(10).all()
        
        return {
            "wallet": {
                "id": str(wallet.id),
                "balance": float(wallet.balance),
                "reserved_balance": float(wallet.reserved_balance),
                "available_balance": float(wallet.available_balance),
                "last_transaction_at": wallet.last_transaction_at.isoformat() if wallet.last_transaction_at else None,
                "created_at": wallet.created_at.isoformat()
            },
            "recent_transactions": [
                {
                    "id": str(tx.id),
                    "type": tx.transaction_type.value,
                    "amount": float(tx.amount),
                    "balance_after": float(tx.balance_after),
                    "description": tx.description,
                    "created_at": tx.created_at.isoformat()
                }
                for tx in recent_transactions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting wallet: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve wallet")


@router.post("/wallet/recharge")
async def recharge_scrollcoin_wallet(
    request: ScrollCoinRechargeRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Recharge ScrollCoin wallet with fiat payment."""
    try:
        # Process payment through billing engine
        result = await billing_engine.process({
            "operation": "recharge_wallet",
            "user_id": str(current_user.id),
            "amount": float(request.amount),
            "payment_method": request.payment_method_id
        })
        
        if result.get("success"):
            return {
                "success": True,
                "message": "Wallet recharged successfully",
                "transaction_id": result["transaction_id"],
                "amount_charged": result["amount_charged"],
                "scrollcoins_added": result["scrollcoins_added"],
                "new_balance": result["new_balance"]
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Recharge failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recharging wallet: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to recharge wallet")


@router.post("/usage/track")
async def track_usage(
    request: UsageTrackingRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Track usage for billing purposes."""
    try:
        # Get current subscription
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.ACTIVE
        ).first()
        
        # Create usage record
        usage_record = UsageRecord(
            user_id=current_user.id,
            subscription_id=subscription.id if subscription else None,
            metric_type=request.metric_type,
            quantity=request.quantity,
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            metadata=request.metadata or {}
        )
        
        db.add(usage_record)
        db.commit()
        
        # Check usage limits and create alerts if needed
        if subscription:
            await _check_usage_limits(current_user.id, subscription, db)
        
        return {
            "success": True,
            "message": "Usage tracked successfully",
            "usage_id": str(usage_record.id)
        }
        
    except Exception as e:
        logger.error(f"Error tracking usage: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to track usage")


@router.get("/payment-methods")
async def get_payment_methods(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get user's saved payment methods."""
    try:
        payment_methods = db.query(PaymentMethodModel).filter(
            PaymentMethodModel.user_id == current_user.id,
            PaymentMethodModel.is_active == True
        ).all()
        
        return {
            "payment_methods": [
                {
                    "id": str(pm.id),
                    "type": pm.type,
                    "last_four": pm.last_four,
                    "brand": pm.brand,
                    "exp_month": pm.exp_month,
                    "exp_year": pm.exp_year,
                    "is_default": pm.is_default,
                    "nickname": pm.nickname,
                    "created_at": pm.created_at.isoformat()
                }
                for pm in payment_methods
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting payment methods: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve payment methods")


@router.post("/payment-methods")
async def add_payment_method(
    request: PaymentMethodRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Add new payment method."""
    try:
        # Get payment method details from Stripe
        stripe_pm = stripe.PaymentMethod.retrieve(request.stripe_payment_method_id)
        
        # Create payment method record
        payment_method = PaymentMethodModel(
            user_id=current_user.id,
            type=stripe_pm.type,
            last_four=stripe_pm.card.last4 if stripe_pm.card else None,
            brand=stripe_pm.card.brand if stripe_pm.card else None,
            exp_month=stripe_pm.card.exp_month if stripe_pm.card else None,
            exp_year=stripe_pm.card.exp_year if stripe_pm.card else None,
            is_default=request.is_default,
            nickname=request.nickname,
            stripe_payment_method_id=request.stripe_payment_method_id
        )
        
        # If this is set as default, unset other defaults
        if request.is_default:
            db.query(PaymentMethodModel).filter(
                PaymentMethodModel.user_id == current_user.id,
                PaymentMethodModel.is_default == True
            ).update({"is_default": False})
        
        db.add(payment_method)
        db.commit()
        db.refresh(payment_method)
        
        return {
            "success": True,
            "message": "Payment method added successfully",
            "payment_method_id": str(payment_method.id)
        }
        
    except Exception as e:
        logger.error(f"Error adding payment method: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add payment method")


@router.get("/invoices")
async def get_invoices(
    limit: int = 10,
    offset: int = 0,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get user's billing invoices."""
    try:
        invoices = db.query(Invoice).filter(
            Invoice.user_id == current_user.id
        ).order_by(Invoice.created_at.desc()).offset(offset).limit(limit).all()
        
        return {
            "invoices": [
                {
                    "id": str(invoice.id),
                    "invoice_number": invoice.invoice_number,
                    "status": invoice.status,
                    "total_amount": float(invoice.total_amount),
                    "currency": invoice.currency,
                    "period_start": invoice.period_start.isoformat(),
                    "period_end": invoice.period_end.isoformat(),
                    "issued_at": invoice.issued_at.isoformat(),
                    "due_date": invoice.due_date.isoformat(),
                    "paid_at": invoice.paid_at.isoformat() if invoice.paid_at else None,
                    "is_overdue": invoice.is_overdue
                }
                for invoice in invoices
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting invoices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve invoices")


@router.get("/alerts")
async def get_billing_alerts(
    unread_only: bool = False,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get billing alerts for user."""
    try:
        query = db.query(BillingAlert).filter(BillingAlert.user_id == current_user.id)
        
        if unread_only:
            query = query.filter(BillingAlert.is_read == False)
        
        alerts = query.order_by(BillingAlert.created_at.desc()).limit(20).all()
        
        return {
            "alerts": [
                {
                    "id": str(alert.id),
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "is_read": alert.is_read,
                    "action_required": alert.action_required,
                    "action_url": alert.action_url,
                    "action_text": alert.action_text,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """Handle Stripe webhooks."""
    try:
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')
        endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        
        # Verify webhook signature
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        
        # Handle different event types
        if event['type'] == 'invoice.payment_succeeded':
            background_tasks.add_task(_handle_payment_succeeded, event['data']['object'])
        elif event['type'] == 'invoice.payment_failed':
            background_tasks.add_task(_handle_payment_failed, event['data']['object'])
        elif event['type'] == 'customer.subscription.updated':
            background_tasks.add_task(_handle_subscription_updated, event['data']['object'])
        elif event['type'] == 'customer.subscription.deleted':
            background_tasks.add_task(_handle_subscription_deleted, event['data']['object'])
        
        return {"success": True}
        
    except ValueError as e:
        logger.error(f"Invalid payload: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")


# Helper functions
def _get_tier_pricing(tier: SubscriptionTier, cycle: BillingCycle) -> Dict[str, Any]:
    """Get pricing for subscription tier and cycle."""
    pricing = {
        SubscriptionTier.FREE: {"monthly": 0, "yearly": 0, "scrollcoins_included": 100},
        SubscriptionTier.STARTER: {"monthly": 29, "yearly": 290, "scrollcoins_included": 1000},
        SubscriptionTier.PROFESSIONAL: {"monthly": 99, "yearly": 990, "scrollcoins_included": 5000},
        SubscriptionTier.ENTERPRISE: {"monthly": 299, "yearly": 2990, "scrollcoins_included": 20000},
        SubscriptionTier.SOVEREIGN: {"monthly": 999, "yearly": 9990, "scrollcoins_included": 100000}
    }
    
    tier_pricing = pricing[tier]
    price_key = "yearly" if cycle == BillingCycle.YEARLY else "monthly"
    
    return {
        "price": Decimal(str(tier_pricing[price_key])),
        "scrollcoins_included": tier_pricing["scrollcoins_included"]
    }


async def _create_stripe_subscription(
    user, tier: SubscriptionTier, cycle: BillingCycle, 
    payment_method_id: str, trial_days: Optional[int]
) -> Dict[str, str]:
    """Create Stripe subscription."""
    try:
        # Create or get customer
        customer = stripe.Customer.create(
            email=user.email,
            name=user.full_name,
            payment_method=payment_method_id,
            invoice_settings={"default_payment_method": payment_method_id}
        )
        
        # Create subscription
        subscription_params = {
            "customer": customer.id,
            "items": [{"price": _get_stripe_price_id(tier, cycle)}],
            "payment_behavior": "default_incomplete",
            "expand": ["latest_invoice.payment_intent"]
        }
        
        if trial_days:
            subscription_params["trial_period_days"] = trial_days
        
        subscription = stripe.Subscription.create(**subscription_params)
        
        return {
            "subscription_id": subscription.id,
            "customer_id": customer.id,
            "client_secret": subscription.latest_invoice.payment_intent.client_secret
        }
        
    except Exception as e:
        logger.error(f"Error creating Stripe subscription: {str(e)}")
        raise


def _get_stripe_price_id(tier: SubscriptionTier, cycle: BillingCycle) -> str:
    """Get Stripe price ID for tier and cycle."""
    # This would map to actual Stripe price IDs in production
    price_map = {
        (SubscriptionTier.STARTER, BillingCycle.MONTHLY): "price_starter_monthly",
        (SubscriptionTier.STARTER, BillingCycle.YEARLY): "price_starter_yearly",
        (SubscriptionTier.PROFESSIONAL, BillingCycle.MONTHLY): "price_pro_monthly",
        (SubscriptionTier.PROFESSIONAL, BillingCycle.YEARLY): "price_pro_yearly",
        (SubscriptionTier.ENTERPRISE, BillingCycle.MONTHLY): "price_enterprise_monthly",
        (SubscriptionTier.ENTERPRISE, BillingCycle.YEARLY): "price_enterprise_yearly",
        (SubscriptionTier.SOVEREIGN, BillingCycle.MONTHLY): "price_sovereign_monthly",
        (SubscriptionTier.SOVEREIGN, BillingCycle.YEARLY): "price_sovereign_yearly"
    }
    return price_map.get((tier, cycle), "price_default")


async def _ensure_scrollcoin_wallet(user_id: UUID, initial_balance: int, db: Session):
    """Ensure user has ScrollCoin wallet with initial balance."""
    wallet = db.query(ScrollCoinWallet).filter(ScrollCoinWallet.user_id == user_id).first()
    
    if not wallet:
        wallet = ScrollCoinWallet(
            user_id=user_id,
            balance=Decimal(str(initial_balance))
        )
        db.add(wallet)
        db.commit()


async def _get_usage_statistics(user_id: UUID, db: Session) -> Dict[str, Any]:
    """Get usage statistics for user."""
    # This would calculate actual usage statistics
    return {
        "current_period": {
            "api_calls": 150,
            "training_jobs": 2,
            "storage_gb": 0.5,
            "scrollcoins_used": 50
        },
        "limits": {
            "api_calls": 1000,
            "training_jobs": 10,
            "storage_gb": 10,
            "scrollcoins": 1000
        }
    }


async def _check_usage_limits(user_id: UUID, subscription: Subscription, db: Session):
    """Check usage limits and create alerts if needed."""
    # Implementation would check actual usage against limits
    pass


async def _create_billing_alert(
    user_id: UUID, alert_type: str, title: str, message: str, db: Session
):
    """Create billing alert for user."""
    alert = BillingAlert(
        user_id=user_id,
        alert_type=alert_type,
        title=title,
        message=message,
        severity="info"
    )
    db.add(alert)
    db.commit()


async def _send_subscription_confirmation(email: str, subscription: Subscription):
    """Send subscription confirmation email."""
    # Implementation would send actual email
    logger.info(f"Sending subscription confirmation to {email}")


async def _update_stripe_subscription(subscription_id: str, new_tier: SubscriptionTier):
    """Update Stripe subscription tier."""
    # Implementation would update Stripe subscription
    pass


# Webhook handlers
async def _handle_payment_succeeded(invoice_data: Dict[str, Any]):
    """Handle successful payment webhook."""
    logger.info(f"Payment succeeded for invoice: {invoice_data['id']}")


async def _handle_payment_failed(invoice_data: Dict[str, Any]):
    """Handle failed payment webhook."""
    logger.info(f"Payment failed for invoice: {invoice_data['id']}")


async def _handle_subscription_updated(subscription_data: Dict[str, Any]):
    """Handle subscription update webhook."""
    logger.info(f"Subscription updated: {subscription_data['id']}")


async def _handle_subscription_deleted(subscription_data: Dict[str, Any]):
    """Handle subscription deletion webhook."""
    logger.info(f"Subscription deleted: {subscription_data['id']}")