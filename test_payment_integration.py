"""
Test payment integration for ScrollIntel
"""
import asyncio
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.payment_service import PaymentService

async def test_payment_service():
    """Test the payment service functionality"""
    print("🧪 Testing ScrollIntel Payment Integration")
    print("=" * 50)
    
    # Initialize payment service
    payment_service = PaymentService()
    
    # Test 1: Stripe Payment Intent
    print("\n1. Testing Stripe Payment Intent Creation...")
    try:
        result = await payment_service.create_stripe_payment_intent(
            amount=2999,  # $29.99 in cents
            currency="usd",
            metadata={"product": "ScrollIntel Pro Subscription"}
        )
        
        if result["success"]:
            print("✅ Stripe payment intent created successfully")
            print(f"   Payment Intent ID: {result.get('payment_intent_id', 'N/A')}")
        else:
            print(f"❌ Stripe payment intent failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Stripe test failed: {e}")
    
    # Test 2: PayPal Payment
    print("\n2. Testing PayPal Payment Creation...")
    try:
        result = await payment_service.create_paypal_payment(
            amount=29.99,
            currency="USD",
            description="ScrollIntel Pro Subscription"
        )
        
        if result["success"]:
            print("✅ PayPal payment created successfully")
            print(f"   Payment ID: {result.get('payment_id', 'N/A')}")
            print(f"   Approval URL: {result.get('approval_url', 'N/A')}")
        else:
            print(f"❌ PayPal payment failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ PayPal test failed: {e}")
    
    # Test 3: Twilio SMS (using a test number)
    print("\n3. Testing Twilio SMS Notification...")
    try:
        result = await payment_service.send_payment_notification(
            phone_number="+1234567890",  # Test number
            message="Your ScrollIntel payment of $29.99 has been processed successfully!"
        )
        
        if result["success"]:
            print("✅ Twilio SMS sent successfully")
            print(f"   Message SID: {result.get('message_sid', 'N/A')}")
        else:
            print(f"❌ Twilio SMS failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Twilio test failed: {e}")
    
    # Test 4: Utility Functions
    print("\n4. Testing Utility Functions...")
    try:
        # Test currency formatting
        formatted = payment_service.format_currency(29.99, "USD")
        print(f"✅ Currency formatting: {formatted}")
        
        # Test tax calculation
        tax = payment_service.calculate_tax(29.99, 0.08)
        print(f"✅ Tax calculation: ${tax:.2f}")
        
        # Test amount validation
        valid = payment_service.validate_payment_amount(29.99)
        print(f"✅ Amount validation: {valid}")
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Payment integration testing completed!")
    print("\n📋 Next Steps:")
    print("1. Install payment libraries: pip install stripe twilio paypalrestsdk")
    print("2. Test payment flows in development environment")
    print("3. Set up webhook endpoints for payment notifications")
    print("4. Configure payment method UI components")
    print("5. Implement subscription billing logic")

def test_environment_variables():
    """Test if all required environment variables are set"""
    print("\n🔍 Checking Environment Variables...")
    print("-" * 30)
    
    required_vars = [
        "STRIPE_SECRET_KEY",
        "STRIPE_PUBLISHABLE_KEY", 
        "STRIPE_WEBHOOK_SECRET",
        "PAYPAL_CLIENT_ID",
        "PAYPAL_CLIENT_SECRET",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN"
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing variables: {', '.join(missing_vars)}")
        return False
    else:
        print("\n✅ All payment environment variables are configured!")
        return True

def main():
    """Main test function"""
    print("🚀 ScrollIntel Payment System Test Suite")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test environment variables first
    env_ok = test_environment_variables()
    
    if env_ok:
        # Run async payment tests
        asyncio.run(test_payment_service())
    else:
        print("\n❌ Cannot run payment tests without proper environment configuration")
        print("Please check your .env.production file and ensure all payment credentials are set")

if __name__ == "__main__":
    main()