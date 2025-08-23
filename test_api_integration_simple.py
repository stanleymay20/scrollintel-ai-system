"""
Simple test for API Integration Framework
Tests basic functionality without external dependencies
"""

import asyncio
from scrollintel.connectors.api_connectors import (
    APIConnectorFactory, APIType, AuthType, AuthConfig, 
    RateLimitConfig, APIEndpoint, RateLimiter
)


def test_rate_limiter():
    """Test rate limiter basic functionality"""
    config = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_limit=10
    )
    limiter = RateLimiter(config)
    
    assert limiter.config.requests_per_minute == 60
    assert limiter.tokens == 10
    print("✓ Rate limiter initialization works")


def test_connector_factory():
    """Test API connector factory"""
    auth_config = AuthConfig(auth_type=AuthType.NONE)
    
    # Test REST connector creation
    rest_connector = APIConnectorFactory.create_connector(
        APIType.REST, "https://api.example.com", auth_config
    )
    assert rest_connector.base_url == "https://api.example.com"
    print("✓ REST connector factory works")
    
    # Test GraphQL connector creation
    graphql_connector = APIConnectorFactory.create_connector(
        APIType.GRAPHQL, "https://api.example.com/graphql", auth_config
    )
    assert graphql_connector.base_url == "https://api.example.com/graphql"
    print("✓ GraphQL connector factory works")
    
    # Test SOAP connector creation
    soap_connector = APIConnectorFactory.create_connector(
        APIType.SOAP, "https://api.example.com/soap", auth_config
    )
    assert soap_connector.base_url == "https://api.example.com/soap"
    print("✓ SOAP connector factory works")


def test_auth_config():
    """Test authentication configuration"""
    # API Key auth
    api_key_auth = AuthConfig(
        auth_type=AuthType.API_KEY,
        credentials={"api_key": "test_key", "key_name": "X-API-Key"}
    )
    assert api_key_auth.auth_type == AuthType.API_KEY
    assert api_key_auth.credentials["api_key"] == "test_key"
    print("✓ API Key authentication config works")
    
    # Bearer token auth
    bearer_auth = AuthConfig(
        auth_type=AuthType.BEARER,
        credentials={"token": "bearer_token"}
    )
    assert bearer_auth.auth_type == AuthType.BEARER
    print("✓ Bearer token authentication config works")


def test_api_endpoint():
    """Test API endpoint configuration"""
    endpoint = APIEndpoint(
        url="/users",
        method="GET",
        headers={"Accept": "application/json"},
        params={"limit": 10},
        timeout=30
    )
    
    assert endpoint.url == "/users"
    assert endpoint.method == "GET"
    assert endpoint.headers["Accept"] == "application/json"
    assert endpoint.params["limit"] == 10
    assert endpoint.timeout == 30
    print("✓ API endpoint configuration works")


async def test_rate_limiter_async():
    """Test rate limiter async functionality"""
    config = RateLimitConfig(
        requests_per_minute=60,
        burst_limit=3
    )
    limiter = RateLimiter(config)
    
    # Should be able to acquire tokens up to burst limit
    for i in range(3):
        result = await limiter.acquire()
        assert result == True
    
    # Should fail after burst limit
    result = await limiter.acquire()
    assert result == False
    
    print("✓ Rate limiter async functionality works")


def test_soap_envelope_building():
    """Test SOAP envelope construction"""
    from scrollintel.connectors.api_connectors import SOAPAPIConnector
    
    auth_config = AuthConfig(auth_type=AuthType.NONE)
    connector = SOAPAPIConnector("https://api.example.com/soap", auth_config)
    
    envelope = connector._build_soap_envelope("GetUser", {"userId": "123"})
    
    assert "GetUser" in envelope
    assert "<userId>123</userId>" in envelope
    assert "soap:Envelope" in envelope
    print("✓ SOAP envelope building works")


async def main():
    """Run all simple tests"""
    print("🧪 Running API Integration Simple Tests")
    print("=" * 40)
    
    try:
        # Synchronous tests
        test_rate_limiter()
        test_connector_factory()
        test_auth_config()
        test_api_endpoint()
        test_soap_envelope_building()
        
        # Asynchronous tests
        await test_rate_limiter_async()
        
        print("\n" + "=" * 40)
        print("✅ All API Integration Tests Passed!")
        print("\nTested Components:")
        print("  ✓ Rate Limiter")
        print("  ✓ Connector Factory")
        print("  ✓ Authentication Configuration")
        print("  ✓ API Endpoint Configuration")
        print("  ✓ SOAP Envelope Building")
        print("  ✓ Async Rate Limiting")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())