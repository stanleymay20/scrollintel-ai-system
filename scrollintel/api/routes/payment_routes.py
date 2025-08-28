"""
Payment API routes for ScrollIntel
"""
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

from ...core.payment_service import PaymentService
from ...models.payment_models import PaymentStatus, PaymentProvider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments", tags=["payments"])
payment_service = PaymentService()

# Pydantic Models
class PaymentIntentRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Payment amount")
    currency: str = Field(default="USD", description="Currency code")
    description: Optional[str] = Field(None, description="Payment description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SubscriptionRequest(BaseModel):
    plan_id: str = Field(..., description="Subscription plan ID")
    payment_method_id: Optional[str] = Field(None, description="Payment method ID")
    trial_days: Optional[int] = Field(None, description="Trial period in days")

class PayPalPaymentRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Payment amount")
    currency: str = Field(default="USD", description="Currency code")
    description: str = Field(default="ScrollIntel Payment", description="Payment description")

class PayPalExecuteRequest(BaseModel):
    payment_id: str = Field(..., description="PayPal payment ID")
    payer_id: str = Field(..., description="PayPal payer ID")

class NotificationRequest(BaseModel):
    phone_number: str = Field(..., description="Phone number for SMS")
    message: str = Field(..., description="SMS message content")

# Stripe Routes
@router.post("/stripe/create-payment-intent")
async def create_stripe_payment_intent(
    request: PaymentIntentRequest,
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Create a Stripe payment intent"""
    try:
        result = await payment_service.create_stripe_payment_intent(
            amount=int(request.amount * 100),  # Convert to cents
            currency=request.currency.lower(),
            metadata=request.metadata
        )
        
        if result["success"]:
            return {
                "success": True,
                "client_secret": result["client_secret"],
                "payment_intent_id": result["payment_intent_id"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Payment intent creation failed: {e}")
        raise HTTPException(status_code=500, detail="Payment processing failed")

@router.post("/stripe/create-subscription")
async def create_stripe_subscription(
    request: SubscriptionRequest,
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Create a Stripe subscription"""
    try:
        # In production, you'd get the customer_id from the authenticated user
        customer_id = "cus_placeholder"  # This should come from user data
        
        result = await payment_service.create_stripe_subscription(
            customer_id=customer_id,
            price_id=request.plan_id,
            trial_period_days=request.trial_days
        )
        
        if result["success"]:
            return {
                "success": True,
                "subscription_id": result["subscription_id"],
                "client_secret": result["client_secret"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Subscription creation failed: {e}")
        raise HTTPException(status_code=500, detail="Subscription creation failed")

@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle Stripe webhook events"""
    try:
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature")
        
        if not sig_header:
            raise HTTPException(status_code=400, detail="Missing signature header")
        
        result = await payment_service.handle_stripe_webhook(
            payload.decode("utf-8"), 
            sig_header
        )
        
        if result["success"]:
            return {"received": True, "processed": result.get("processed")}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

# PayPal Routes
@router.post("/paypal/create-payment")
async def create_paypal_payment(
    request: PayPalPaymentRequest,
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Create a PayPal payment"""
    try:
        result = await payment_service.create_paypal_payment(
            amount=request.amount,
            currency=request.currency,
            description=request.description
        )
        
        if result["success"]:
            return {
                "success": True,
                "payment_id": result["payment_id"],
                "approval_url": result["approval_url"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"PayPal payment creation failed: {e}")
        raise HTTPException(status_code=500, detail="PayPal payment creation failed")

@router.post("/paypal/execute-payment")
async def execute_paypal_payment(
    request: PayPalExecuteRequest,
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Execute a PayPal payment"""
    try:
        result = await payment_service.execute_paypal_payment(
            payment_id=request.payment_id,
            payer_id=request.payer_id
        )
        
        if result["success"]:
            return {
                "success": True,
                "payment_id": result["payment_id"],
                "state": result["state"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"PayPal payment execution failed: {e}")
        raise HTTPException(status_code=500, detail="PayPal payment execution failed")

# Twilio Routes
@router.post("/notifications/sms")
async def send_payment_sms(
    request: NotificationRequest,
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Send payment notification via SMS"""
    try:
        result = await payment_service.send_payment_notification(
            phone_number=request.phone_number,
            message=request.message
        )
        
        if result["success"]:
            return {
                "success": True,
                "message_sid": result["message_sid"],
                "status": result["status"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"SMS sending failed: {e}")
        raise HTTPException(status_code=500, detail="SMS sending failed")

# Payment Management Routes
@router.get("/history")
async def get_payment_history(
    current_user: Dict = Depends(lambda: {"id": "user123"}),  # Placeholder auth
    limit: int = 10,
    offset: int = 0
):
    """Get user payment history"""
    try:
        # In production, this would query the database for user payments
        return {
            "success": True,
            "payments": [
                {
                    "id": "pay_123",
                    "amount": 29.99,
                    "currency": "USD",
                    "status": "completed",
                    "provider": "stripe",
                    "created_at": "2024-01-15T10:30:00Z",
                    "description": "ScrollIntel Pro Subscription"
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Payment history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve payment history")

@router.get("/subscription/status")
async def get_subscription_status(
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Get current subscription status"""
    try:
        # In production, this would query the database for user subscription
        return {
            "success": True,
            "subscription": {
                "id": "sub_123",
                "status": "active",
                "plan": "pro",
                "current_period_end": "2024-02-15T10:30:00Z",
                "cancel_at_period_end": False
            }
        }
    except Exception as e:
        logger.error(f"Subscription status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve subscription status")

@router.post("/subscription/cancel")
async def cancel_subscription(
    current_user: Dict = Depends(lambda: {"id": "user123"})  # Placeholder auth
):
    """Cancel user subscription"""
    try:
        # In production, this would cancel the subscription via Stripe/PayPal
        return {
            "success": True,
            "message": "Subscription cancelled successfully",
            "cancelled_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Subscription cancellation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")

# Health Check
@router.get("/health")
async def payment_health_check():
    """Payment system health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "stripe": "connected",
            "paypal": "connected", 
            "twilio": "connected"
        }
    }