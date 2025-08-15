"""
Comprehensive Billing and Subscription Management Engine

This engine handles all billing operations including:
- Stripe payment processing
- Subscription management
- Usage-based billing
- Invoice generation
- ScrollCoin wallet management
"""

import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from decimal import Decimal
from enum import Enum
import json

# Stripe integration
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None

from .base_engine import BaseEngine, EngineStatus, EngineCapability
from scrollintel.models.database_utils import get_db
from scrollintel.models.billing_models import (
    Subscription, ScrollCoinWallet, Payment, Invoice, UsageRecord,
    BillingAlert, PaymentMethod as PaymentMethodModel,
    SubscriptionTier, BillingCycle, SubscriptionStatus, PaymentStatus,
    TransactionType, PaymentMethod, UsageMetricType, ScrollCoinTransaction
)

logger = logging.getLogger(__name__)


class BillingType(str, Enum):
    """Types of billing."""
    FIAT = "fiat"
    SCROLLCOIN = "scrollcoin"
    HYBRID = "hybrid"


class ScrollCoinAction(str, Enum):
    """Actions that consume ScrollCoins."""
    MODEL_INFERENCE = "model_inference"
    TRAINING_JOB = "training_job"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    REPORT_GENERATION = "report_generation"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    EXPLANATION_GENERATION = "explanation_generation"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    FEDERATED_TRAINING = "federated_training"
    DRIFT_MONITORING = "drift_monitoring"
    API_CALL = "api_call"


class ScrollBillingEngine(BaseEngine):
    """Advanced dual billing system with ScrollCoin and Stripe integration."""
    
    def __init__(self):
        super().__init__(
            engine_id="scroll-billing-engine",
            name="ScrollBilling Engine",
            capabilities=[
                EngineCapability.SECURE_STORAGE,
                EngineCapability.DATA_ANALYSIS
            ]
        )
        
        # Stripe configuration
        if STRIPE_AVAILABLE:
            stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
            self.stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        # ScrollCoin configuration
        self.scrollcoin_rates = self._initialize_scrollcoin_rates()
        self.subscription_plans = self._initialize_subscription_plans()
        
        # Billing state
        self.active_subscriptions = {}
        self.wallet_cache = {}
        self.usage_tracking = {}
    
    def _initialize_scrollcoin_rates(self) -> Dict[ScrollCoinAction, Decimal]:
        """Initialize ScrollCoin consumption rates for different actions."""
        return {
            ScrollCoinAction.MODEL_INFERENCE: Decimal("0.1"),
            ScrollCoinAction.TRAINING_JOB: Decimal("10.0"),
            ScrollCoinAction.DATA_ANALYSIS: Decimal("1.0"),
            ScrollCoinAction.VISUALIZATION: Decimal("0.5"),
            ScrollCoinAction.REPORT_GENERATION: Decimal("2.0"),
            ScrollCoinAction.PROMPT_OPTIMIZATION: Decimal("1.5"),
            ScrollCoinAction.EXPLANATION_GENERATION: Decimal("0.8"),
            ScrollCoinAction.MULTIMODAL_PROCESSING: Decimal("2.5"),
            ScrollCoinAction.FEDERATED_TRAINING: Decimal("15.0"),
            ScrollCoinAction.DRIFT_MONITORING: Decimal("0.3"),
            ScrollCoinAction.API_CALL: Decimal("0.05")
        }
    
    def _initialize_subscription_plans(self) -> Dict[SubscriptionTier, Dict[str, Any]]:
        """Initialize subscription plans with pricing and limits."""
        return {
            SubscriptionTier.FREE: {
                "price_monthly": 0,
                "price_yearly": 0,
                "scrollcoins_included": 100,
                "api_calls_limit": 1000,
                "training_jobs_limit": 1,
                "storage_limit_gb": 1,
                "features": ["basic_analysis", "simple_visualizations"]
            },
            SubscriptionTier.STARTER: {
                "price_monthly": 29,
                "price_yearly": 290,
                "scrollcoins_included": 1000,
                "api_calls_limit": 10000,
                "training_jobs_limit": 10,
                "storage_limit_gb": 10,
                "features": ["advanced_analysis", "ml_training", "basic_explanations"]
            },
            SubscriptionTier.PROFESSIONAL: {
                "price_monthly": 99,
                "price_yearly": 990,
                "scrollcoins_included": 5000,
                "api_calls_limit": 100000,
                "training_jobs_limit": 50,
                "storage_limit_gb": 100,
                "features": ["full_ml_suite", "advanced_explanations", "multimodal_ai", "federated_learning"]
            },
            SubscriptionTier.ENTERPRISE: {
                "price_monthly": 299,
                "price_yearly": 2990,
                "scrollcoins_included": 20000,
                "api_calls_limit": 1000000,
                "training_jobs_limit": 200,
                "storage_limit_gb": 1000,
                "features": ["enterprise_features", "priority_support", "custom_models", "compliance_tools"]
            },
            SubscriptionTier.SOVEREIGN: {
                "price_monthly": 999,
                "price_yearly": 9990,
                "scrollcoins_included": 100000,
                "api_calls_limit": -1,  # Unlimited
                "training_jobs_limit": -1,  # Unlimited
                "storage_limit_gb": -1,  # Unlimited
                "features": ["full_sovereignty", "white_label", "on_premise", "dedicated_support"]
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the billing engine."""
        try:
            # Verify Stripe configuration
            if STRIPE_AVAILABLE and stripe.api_key:
                # Test Stripe connection
                await asyncio.to_thread(stripe.Account.retrieve)
                logger.info("Stripe integration initialized successfully")
            else:
                logger.warning("Stripe not available or not configured")
            
            # Initialize database tables
            await self._ensure_billing_tables()
            
            self.status = EngineStatus.READY
            logger.info("ScrollBillingEngine initialized successfully")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Failed to initialize ScrollBillingEngine: {e}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process billing operations."""
        params = parameters or {}
        operation = params.get("operation")
        user_id = params.get("user_id")
        
        if operation == "create_subscription":
            return await self._create_subscription(user_id, params)
        elif operation == "charge_scrollcoins":
            return await self._charge_scrollcoins(user_id, params)
        elif operation == "recharge_wallet":
            return await self._recharge_wallet(user_id, params)
        elif operation == "get_wallet_balance":
            return await self._get_wallet_balance(user_id)
        elif operation == "process_stripe_webhook":
            return await self._process_stripe_webhook(input_data, params)
        elif operation == "generate_invoice":
            return await self._generate_invoice(user_id, params)
        elif operation == "usage_analytics":
            return await self._get_usage_analytics(user_id, params)
        else:
            raise ValueError(f"Unknown billing operation: {operation}")
    
    async def _create_subscription(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new subscription."""
        tier = SubscriptionTier(params.get("tier", SubscriptionTier.STARTER))
        billing_type = BillingType(params.get("billing_type", BillingType.FIAT))
        billing_cycle = params.get("billing_cycle", "monthly")
        
        plan = self.subscription_plans[tier]
        
        if billing_type == BillingType.FIAT and STRIPE_AVAILABLE:
            # Create Stripe subscription
            stripe_result = await self._create_stripe_subscription(user_id, tier, billing_cycle)
            subscription_id = stripe_result["subscription_id"]
        else:
            # Create ScrollCoin subscription
            subscription_id = f"scroll-sub-{uuid4()}"
        
        # Create subscription record
        subscription = {
            "id": subscription_id,
            "user_id": user_id,
            "tier": tier.value,
            "billing_type": billing_type.value,
            "billing_cycle": billing_cycle,
            "status": "active",
            "created_at": datetime.utcnow(),
            "next_billing_date": datetime.utcnow() + timedelta(days=30 if billing_cycle == "monthly" else 365)
        }
        
        # Store subscription
        self.active_subscriptions[subscription_id] = subscription
        
        # Initialize ScrollCoin wallet if needed
        if billing_type in [BillingType.SCROLLCOIN, BillingType.HYBRID]:
            await self._initialize_wallet(user_id, plan["scrollcoins_included"])
        
        return {
            "subscription_id": subscription_id,
            "tier": tier.value,
            "billing_type": billing_type.value,
            "status": "active",
            "features": plan["features"],
            "scrollcoins_included": plan["scrollcoins_included"]
        }
    
    async def _charge_scrollcoins(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Charge ScrollCoins for an action."""
        action = ScrollCoinAction(params.get("action"))
        quantity = params.get("quantity", 1)
        metadata = params.get("metadata", {})
        
        # Calculate cost
        base_cost = self.scrollcoin_rates[action]
        total_cost = base_cost * Decimal(str(quantity))
        
        # Get wallet balance
        wallet = await self._get_wallet(user_id)
        if wallet["balance"] < total_cost:
            return {
                "success": False,
                "error": "insufficient_balance",
                "required": float(total_cost),
                "available": float(wallet["balance"])
            }
        
        # Deduct ScrollCoins
        new_balance = wallet["balance"] - total_cost
        await self._update_wallet_balance(user_id, new_balance)
        
        # Record transaction
        transaction = {
            "id": f"tx-{uuid4()}",
            "user_id": user_id,
            "type": TransactionType.USAGE.value,
            "action": action.value,
            "amount": float(total_cost),
            "quantity": quantity,
            "metadata": metadata,
            "timestamp": datetime.utcnow()
        }
        
        await self._record_transaction(transaction)
        
        # Track usage
        await self._track_usage(user_id, action, quantity, total_cost)
        
        return {
            "success": True,
            "transaction_id": transaction["id"],
            "cost": float(total_cost),
            "remaining_balance": float(new_balance),
            "action": action.value
        }
    
    async def _recharge_wallet(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recharge ScrollCoin wallet."""
        amount = Decimal(str(params.get("amount", 0)))
        payment_method = params.get("payment_method", "stripe")
        
        if payment_method == "stripe" and STRIPE_AVAILABLE:
            # Process Stripe payment
            stripe_result = await self._process_stripe_payment(user_id, amount, params)
            if not stripe_result["success"]:
                return stripe_result
        
        # Calculate ScrollCoins (1 USD = 100 ScrollCoins)
        scrollcoins = amount * 100
        
        # Update wallet
        wallet = await self._get_wallet(user_id)
        new_balance = wallet["balance"] + scrollcoins
        await self._update_wallet_balance(user_id, new_balance)
        
        # Record transaction
        transaction = {
            "id": f"tx-{uuid4()}",
            "user_id": user_id,
            "type": TransactionType.RECHARGE.value,
            "amount": float(amount),
            "scrollcoins": float(scrollcoins),
            "payment_method": payment_method,
            "timestamp": datetime.utcnow()
        }
        
        await self._record_transaction(transaction)
        
        return {
            "success": True,
            "transaction_id": transaction["id"],
            "amount_charged": float(amount),
            "scrollcoins_added": float(scrollcoins),
            "new_balance": float(new_balance)
        }
    
    async def _get_wallet_balance(self, user_id: str) -> Dict[str, Any]:
        """Get wallet balance and details."""
        wallet = await self._get_wallet(user_id)
        
        # Get recent transactions
        transactions = await self._get_recent_transactions(user_id, limit=10)
        
        # Calculate usage statistics
        usage_stats = await self._calculate_usage_stats(user_id)
        
        return {
            "balance": float(wallet["balance"]),
            "currency": "ScrollCoins",
            "last_updated": wallet["last_updated"].isoformat(),
            "recent_transactions": transactions,
            "usage_stats": usage_stats,
            "wallet_id": wallet["id"]
        }
    
    async def _generate_invoice(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate invoice for billing period."""
        period_start = datetime.fromisoformat(params.get("period_start"))
        period_end = datetime.fromisoformat(params.get("period_end"))
        
        # Get subscription details
        subscription = await self._get_user_subscription(user_id)
        
        # Get usage during period
        usage = await self._get_usage_for_period(user_id, period_start, period_end)
        
        # Calculate costs
        subscription_cost = self._calculate_subscription_cost(subscription)
        usage_cost = self._calculate_usage_cost(usage)
        total_cost = subscription_cost + usage_cost
        
        # Generate invoice
        invoice = {
            "id": f"inv-{uuid4()}",
            "user_id": user_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "subscription_cost": float(subscription_cost),
            "usage_cost": float(usage_cost),
            "total_cost": float(total_cost),
            "currency": "USD",
            "status": "generated",
            "generated_at": datetime.utcnow().isoformat(),
            "line_items": self._generate_line_items(subscription, usage)
        }
        
        return invoice
    
    async def _get_usage_analytics(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed usage analytics."""
        period_days = params.get("period_days", 30)
        period_start = datetime.utcnow() - timedelta(days=period_days)
        
        # Get usage data
        usage_data = await self._get_usage_for_period(user_id, period_start, datetime.utcnow())
        
        # Calculate analytics
        analytics = {
            "period_days": period_days,
            "total_actions": sum(usage_data.values()),
            "total_cost": float(sum(self.scrollcoin_rates[ScrollCoinAction(action)] * count 
                                 for action, count in usage_data.items())),
            "action_breakdown": {action: count for action, count in usage_data.items()},
            "cost_breakdown": {action: float(self.scrollcoin_rates[ScrollCoinAction(action)] * count)
                             for action, count in usage_data.items()},
            "daily_usage": await self._get_daily_usage_trend(user_id, period_start),
            "top_actions": sorted(usage_data.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return analytics
    
    async def cleanup(self) -> None:
        """Clean up billing engine resources."""
        self.active_subscriptions.clear()
        self.wallet_cache.clear()
        self.usage_tracking.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get billing engine status."""
        return {
            "engine_id": self.engine_id,
            "status": self.status.value,
            "stripe_available": STRIPE_AVAILABLE,
            "active_subscriptions": len(self.active_subscriptions),
            "cached_wallets": len(self.wallet_cache),
            "supported_tiers": [tier.value for tier in SubscriptionTier],
            "supported_actions": [action.value for action in ScrollCoinAction],
            "healthy": self.status == EngineStatus.READY
        }
    
    # Helper methods
    async def _get_wallet(self, user_id: str) -> Dict[str, Any]:
        """Get or create user wallet."""
        if user_id in self.wallet_cache:
            return self.wallet_cache[user_id]
        
        # Mock wallet for now - in production, this would query the database
        wallet = {
            "id": f"wallet-{user_id}",
            "user_id": user_id,
            "balance": Decimal("1000.0"),  # Default balance
            "last_updated": datetime.utcnow()
        }
        
        self.wallet_cache[user_id] = wallet
        return wallet
    
    async def _update_wallet_balance(self, user_id: str, new_balance: Decimal):
        """Update wallet balance."""
        if user_id in self.wallet_cache:
            self.wallet_cache[user_id]["balance"] = new_balance
            self.wallet_cache[user_id]["last_updated"] = datetime.utcnow()
    
    async def _record_transaction(self, transaction: Dict[str, Any]):
        """Record a billing transaction."""
        # In production, this would save to database
        logger.info(f"Transaction recorded: {transaction['id']}")
    
    async def _track_usage(self, user_id: str, action: ScrollCoinAction, quantity: int, cost: Decimal):
        """Track usage for analytics."""
        if user_id not in self.usage_tracking:
            self.usage_tracking[user_id] = {}
        
        if action.value not in self.usage_tracking[user_id]:
            self.usage_tracking[user_id][action.value] = {"count": 0, "cost": Decimal("0")}
        
        self.usage_tracking[user_id][action.value]["count"] += quantity
        self.usage_tracking[user_id][action.value]["cost"] += cost
    
    async def _ensure_billing_tables(self):
        """Ensure billing database tables exist."""
        # In production, this would create/migrate database tables
        pass
    
    async def _create_stripe_subscription(self, user_id: str, tier: SubscriptionTier, billing_cycle: str) -> Dict[str, Any]:
        """Create Stripe subscription."""
        # Mock Stripe subscription creation
        return {
            "subscription_id": f"stripe-sub-{uuid4()}",
            "status": "active"
        }
    
    async def _process_stripe_payment(self, user_id: str, amount: Decimal, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process Stripe payment."""
        # Mock Stripe payment processing
        return {
            "success": True,
            "payment_intent_id": f"pi-{uuid4()}"
        }