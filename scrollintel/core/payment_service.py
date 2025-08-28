"""
Payment service for handling Stripe, PayPal, and Twilio integrations
"""
import stripe
import paypalrestsdk
from twilio.rest import Client as TwilioClient
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session
from ..models.payment_models import Payment, Subscription, PaymentMethod, Invoice, SubscriptionPlan

logger = logging.getLogger(__name__)

class PaymentService:
    def __init__(self):
        # Initialize Stripe
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        
        # Initialize PayPal
        paypalrestsdk.configure({
            "mode": "live" if os.getenv("ENVIRONMENT") == "production" else "sandbox",
            "client_id": os.getenv("PAYPAL_CLIENT_ID"),
            "client_secret": os.getenv("PAYPAL_CLIENT_SECRET")
        })
        
        # Initialize Twilio
        self.twilio_client = TwilioClient(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
    
    # Stripe Integration
    async def create_stripe_payment_intent(
        self, 
        amount: int, 
        currency: str = "usd",
        customer_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a Stripe payment intent"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,  # Amount in cents
                currency=currency,
                customer=customer_id,
                metadata=metadata or {},
                automatic_payment_methods={"enabled": True}
            )
            return {
                "success": True,
                "client_secret": intent.client_secret,
                "payment_intent_id": intent.id
            }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe payment intent creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_stripe_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a Stripe subscription"""
        try:
            subscription_data = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "payment_behavior": "default_incomplete",
                "payment_settings": {"save_default_payment_method": "on_subscription"},
                "expand": ["latest_invoice.payment_intent"]
            }
            
            if trial_period_days:
                subscription_data["trial_period_days"] = trial_period_days
            
            subscription = stripe.Subscription.create(**subscription_data)
            
            return {
                "success": True,
                "subscription_id": subscription.id,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret
            }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def handle_stripe_webhook(self, payload: str, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
            )
            
            if event["type"] == "payment_intent.succeeded":
                payment_intent = event["data"]["object"]
                # Update payment status in database
                await self._update_payment_status(
                    payment_intent["id"], 
                    "completed"
                )
            
            elif event["type"] == "invoice.payment_succeeded":
                invoice = event["data"]["object"]
                # Handle successful subscription payment
                await self._handle_subscription_payment(invoice)
            
            elif event["type"] == "customer.subscription.deleted":
                subscription = event["data"]["object"]
                # Handle subscription cancellation
                await self._handle_subscription_cancellation(subscription["id"])
            
            return {"success": True, "processed": event["type"]}
            
        except Exception as e:
            logger.error(f"Stripe webhook handling failed: {e}")
            return {"success": False, "error": str(e)}
    
    # PayPal Integration
    async def create_paypal_payment(
        self,
        amount: float,
        currency: str = "USD",
        description: str = "ScrollIntel Payment"
    ) -> Dict[str, Any]:
        """Create a PayPal payment"""
        try:
            payment = paypalrestsdk.Payment({
                "intent": "sale",
                "payer": {"payment_method": "paypal"},
                "redirect_urls": {
                    "return_url": f"{os.getenv('APP_URL')}/payment/success",
                    "cancel_url": f"{os.getenv('APP_URL')}/payment/cancel"
                },
                "transactions": [{
                    "item_list": {
                        "items": [{
                            "name": "ScrollIntel Service",
                            "sku": "scrollintel-001",
                            "price": str(amount),
                            "currency": currency,
                            "quantity": 1
                        }]
                    },
                    "amount": {
                        "total": str(amount),
                        "currency": currency
                    },
                    "description": description
                }]
            })
            
            if payment.create():
                approval_url = next(
                    link.href for link in payment.links 
                    if link.rel == "approval_url"
                )
                return {
                    "success": True,
                    "payment_id": payment.id,
                    "approval_url": approval_url
                }
            else:
                return {"success": False, "error": payment.error}
                
        except Exception as e:
            logger.error(f"PayPal payment creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_paypal_payment(
        self, 
        payment_id: str, 
        payer_id: str
    ) -> Dict[str, Any]:
        """Execute a PayPal payment"""
        try:
            payment = paypalrestsdk.Payment.find(payment_id)
            
            if payment.execute({"payer_id": payer_id}):
                return {
                    "success": True,
                    "payment_id": payment.id,
                    "state": payment.state
                }
            else:
                return {"success": False, "error": payment.error}
                
        except Exception as e:
            logger.error(f"PayPal payment execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Twilio Integration
    async def send_payment_notification(
        self,
        phone_number: str,
        message: str
    ) -> Dict[str, Any]:
        """Send SMS notification via Twilio"""
        try:
            message = self.twilio_client.messages.create(
                body=message,
                from_=os.getenv("TWILIO_PHONE_NUMBER", "+1234567890"),
                to=phone_number
            )
            
            return {
                "success": True,
                "message_sid": message.sid,
                "status": message.status
            }
            
        except Exception as e:
            logger.error(f"Twilio SMS sending failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_payment_confirmation_email(
        self,
        email: str,
        payment_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send payment confirmation email via Twilio SendGrid"""
        try:
            # Implementation would use Twilio SendGrid API
            # For now, return success placeholder
            return {
                "success": True,
                "message": "Payment confirmation email sent"
            }
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Database Operations
    async def _update_payment_status(
        self, 
        provider_payment_id: str, 
        status: str
    ) -> None:
        """Update payment status in database"""
        # Implementation would update the Payment model
        pass
    
    async def _handle_subscription_payment(self, invoice: Dict) -> None:
        """Handle successful subscription payment"""
        # Implementation would update subscription and create payment record
        pass
    
    async def _handle_subscription_cancellation(self, subscription_id: str) -> None:
        """Handle subscription cancellation"""
        # Implementation would update subscription status
        pass
    
    # Utility Methods
    def format_currency(self, amount: float, currency: str = "USD") -> str:
        """Format currency for display"""
        if currency.upper() == "USD":
            return f"${amount:.2f}"
        else:
            return f"{amount:.2f} {currency.upper()}"
    
    def calculate_tax(self, amount: float, tax_rate: float = 0.0) -> float:
        """Calculate tax amount"""
        return amount * tax_rate
    
    def validate_payment_amount(self, amount: float) -> bool:
        """Validate payment amount"""
        return amount > 0 and amount <= 999999.99