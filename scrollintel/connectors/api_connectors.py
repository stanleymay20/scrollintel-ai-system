"""
API Integration Framework for Enterprise Connectivity
Supports REST, GraphQL, and SOAP APIs with authentication and rate limiting
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
import base64
import hashlib
import hmac


class APIType(Enum):
    REST = "rest"
    GRAPHQL = "graphql"
    SOAP = "soap"


class AuthType(Enum):
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    OAUTH1 = "oauth1"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    DIGEST = "digest"


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    backoff_factor: float = 1.5
    max_retries: int = 3


@dataclass
class AuthConfig:
    auth_type: AuthType
    credentials: Dict[str, Any] = field(default_factory=dict)
    token_url: Optional[str] = None
    refresh_url: Optional[str] = None
    scopes: List[str] = field(default_factory=list)


@dataclass
class APIEndpoint:
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30


class RateLimiter:
    """Token bucket rate limiter with burst support"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_limit
        self.last_update = time.time()
        self.request_times = []
    
    async def acquire(self) -> bool:
        """Acquire a token for making a request"""
        now = time.time()
        
        # Remove old request times (older than 1 hour)
        cutoff_time = now - 3600
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check hourly limit
        if len(self.request_times) >= self.config.requests_per_hour:
            return False
        
        # Check per-minute limit
        minute_cutoff = now - 60
        recent_requests = [t for t in self.request_times if t > minute_cutoff]
        if len(recent_requests) >= self.config.requests_per_minute:
            return False
        
        # Update token bucket
        time_passed = now - self.last_update
        self.tokens = min(
            self.config.burst_limit,
            self.tokens + time_passed * (self.config.requests_per_minute / 60.0)
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            self.request_times.append(now)
            return True
        
        return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / (self.config.requests_per_minute / 60.0)


class BaseAPIConnector(ABC):
    """Base class for all API connectors"""
    
    def __init__(self, base_url: str, auth_config: AuthConfig, 
                 rate_limit_config: Optional[RateLimitConfig] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_config = auth_config
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.session = None
        self._auth_token = None
        self._token_expires_at = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def _authenticate(self):
        """Implement authentication logic"""
        pass
    
    @abstractmethod
    async def make_request(self, endpoint: APIEndpoint, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request with rate limiting and retry logic"""
        pass
    
    async def _wait_for_rate_limit(self):
        """Wait if rate limit is exceeded"""
        while not await self.rate_limiter.acquire():
            wait_time = self.rate_limiter.get_wait_time()
            await asyncio.sleep(wait_time)
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {}
        
        if self.auth_config.auth_type == AuthType.BEARER and self._auth_token:
            headers['Authorization'] = f'Bearer {self._auth_token}'
        elif self.auth_config.auth_type == AuthType.API_KEY:
            key_name = self.auth_config.credentials.get('key_name', 'X-API-Key')
            api_key = self.auth_config.credentials.get('api_key')
            if api_key:
                headers[key_name] = api_key
        
        return headers


class RESTAPIConnector(BaseAPIConnector):
    """REST API connector with full HTTP method support"""
    
    async def _authenticate(self):
        """Handle REST API authentication"""
        if self.auth_config.auth_type == AuthType.OAUTH2:
            await self._oauth2_authenticate()
        elif self.auth_config.auth_type == AuthType.BEARER:
            # Token might be provided directly
            self._auth_token = self.auth_config.credentials.get('token')
    
    async def _oauth2_authenticate(self):
        """OAuth2 authentication flow"""
        if not self.auth_config.token_url:
            raise ValueError("OAuth2 requires token_url")
        
        credentials = self.auth_config.credentials
        data = {
            'grant_type': 'client_credentials',
            'client_id': credentials.get('client_id'),
            'client_secret': credentials.get('client_secret'),
        }
        
        if self.auth_config.scopes:
            data['scope'] = ' '.join(self.auth_config.scopes)
        
        async with self.session.post(self.auth_config.token_url, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                self._auth_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            else:
                raise Exception(f"OAuth2 authentication failed: {response.status}")
    
    async def make_request(self, endpoint: APIEndpoint, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make REST API request"""
        await self._wait_for_rate_limit()
        
        # Check if token needs refresh
        if (self._token_expires_at and 
            datetime.now() > self._token_expires_at - timedelta(minutes=5)):
            await self._oauth2_authenticate()
        
        url = urljoin(self.base_url + '/', endpoint.url.lstrip('/'))
        headers = {**endpoint.headers, **self._get_auth_headers()}
        
        # Handle different HTTP methods
        method = endpoint.method.upper()
        kwargs = {
            'headers': headers,
            'params': endpoint.params,
            'timeout': aiohttp.ClientTimeout(total=endpoint.timeout)
        }
        
        if data and method in ['POST', 'PUT', 'PATCH']:
            if headers.get('Content-Type', '').startswith('application/json'):
                kwargs['json'] = data
            else:
                kwargs['data'] = data
        
        for attempt in range(self.rate_limiter.config.max_retries + 1):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 429:  # Rate limited
                        if attempt < self.rate_limiter.config.max_retries:
                            wait_time = (self.rate_limiter.config.backoff_factor ** attempt)
                            await asyncio.sleep(wait_time)
                            continue
                    
                    response.raise_for_status()
                    
                    # Handle different content types
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        return await response.json()
                    else:
                        text_content = await response.text()
                        return {'content': text_content, 'content_type': content_type}
                        
            except aiohttp.ClientError as e:
                if attempt == self.rate_limiter.config.max_retries:
                    raise Exception(f"Request failed after {attempt + 1} attempts: {str(e)}")
                
                wait_time = (self.rate_limiter.config.backoff_factor ** attempt)
                await asyncio.sleep(wait_time)
        
        raise Exception("Request failed after all retry attempts")


class GraphQLAPIConnector(BaseAPIConnector):
    """GraphQL API connector with query and mutation support"""
    
    def __init__(self, base_url: str, auth_config: AuthConfig, 
                 rate_limit_config: Optional[RateLimitConfig] = None):
        super().__init__(base_url, auth_config, rate_limit_config)
        self.endpoint_url = urljoin(base_url.rstrip('/') + '/', 'graphql')
    
    async def _authenticate(self):
        """Handle GraphQL API authentication (similar to REST)"""
        if self.auth_config.auth_type == AuthType.OAUTH2:
            await self._oauth2_authenticate()
        elif self.auth_config.auth_type == AuthType.BEARER:
            self._auth_token = self.auth_config.credentials.get('token')
    
    async def _oauth2_authenticate(self):
        """OAuth2 authentication for GraphQL"""
        # Similar to REST implementation
        if not self.auth_config.token_url:
            raise ValueError("OAuth2 requires token_url")
        
        credentials = self.auth_config.credentials
        data = {
            'grant_type': 'client_credentials',
            'client_id': credentials.get('client_id'),
            'client_secret': credentials.get('client_secret'),
        }
        
        async with self.session.post(self.auth_config.token_url, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                self._auth_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
    
    async def make_request(self, endpoint: APIEndpoint, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GraphQL request (query or mutation)"""
        await self._wait_for_rate_limit()
        
        headers = {
            'Content-Type': 'application/json',
            **endpoint.headers,
            **self._get_auth_headers()
        }
        
        # GraphQL request format
        graphql_data = {
            'query': data.get('query') if data else '',
            'variables': data.get('variables', {}) if data else {}
        }
        
        for attempt in range(self.rate_limiter.config.max_retries + 1):
            try:
                async with self.session.post(
                    self.endpoint_url,
                    json=graphql_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    
                    if response.status == 429:
                        if attempt < self.rate_limiter.config.max_retries:
                            wait_time = (self.rate_limiter.config.backoff_factor ** attempt)
                            await asyncio.sleep(wait_time)
                            continue
                    
                    response.raise_for_status()
                    result = await response.json()
                    
                    # Check for GraphQL errors
                    if 'errors' in result:
                        raise Exception(f"GraphQL errors: {result['errors']}")
                    
                    return result.get('data', {})
                    
            except aiohttp.ClientError as e:
                if attempt == self.rate_limiter.config.max_retries:
                    raise Exception(f"GraphQL request failed: {str(e)}")
                
                wait_time = (self.rate_limiter.config.backoff_factor ** attempt)
                await asyncio.sleep(wait_time)
        
        raise Exception("GraphQL request failed after all retry attempts")
    
    async def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query"""
        endpoint = APIEndpoint(url='', method='POST')
        data = {'query': query, 'variables': variables or {}}
        return await self.make_request(endpoint, data)
    
    async def execute_mutation(self, mutation: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL mutation"""
        endpoint = APIEndpoint(url='', method='POST')
        data = {'query': mutation, 'variables': variables or {}}
        return await self.make_request(endpoint, data)


class SOAPAPIConnector(BaseAPIConnector):
    """SOAP API connector with WSDL support"""
    
    def __init__(self, base_url: str, auth_config: AuthConfig, 
                 wsdl_url: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        super().__init__(base_url, auth_config, rate_limit_config)
        self.wsdl_url = wsdl_url
        self.soap_action_header = 'SOAPAction'
    
    async def _authenticate(self):
        """Handle SOAP API authentication"""
        if self.auth_config.auth_type == AuthType.BASIC:
            # Basic auth will be handled in headers
            pass
        elif self.auth_config.auth_type == AuthType.BEARER:
            self._auth_token = self.auth_config.credentials.get('token')
    
    async def make_request(self, endpoint: APIEndpoint, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make SOAP request"""
        await self._wait_for_rate_limit()
        
        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            **endpoint.headers,
            **self._get_auth_headers()
        }
        
        # Add SOAP action if provided
        if 'soap_action' in endpoint.params:
            headers[self.soap_action_header] = endpoint.params['soap_action']
        
        # Handle basic auth
        auth = None
        if self.auth_config.auth_type == AuthType.BASIC:
            username = self.auth_config.credentials.get('username')
            password = self.auth_config.credentials.get('password')
            if username and password:
                auth = aiohttp.BasicAuth(username, password)
        
        soap_body = data.get('soap_body') if data else ''
        
        for attempt in range(self.rate_limiter.config.max_retries + 1):
            try:
                async with self.session.post(
                    self.base_url,
                    data=soap_body,
                    headers=headers,
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    
                    if response.status == 429:
                        if attempt < self.rate_limiter.config.max_retries:
                            wait_time = (self.rate_limiter.config.backoff_factor ** attempt)
                            await asyncio.sleep(wait_time)
                            continue
                    
                    response.raise_for_status()
                    soap_response = await response.text()
                    
                    return {
                        'soap_response': soap_response,
                        'status_code': response.status,
                        'headers': dict(response.headers)
                    }
                    
            except aiohttp.ClientError as e:
                if attempt == self.rate_limiter.config.max_retries:
                    raise Exception(f"SOAP request failed: {str(e)}")
                
                wait_time = (self.rate_limiter.config.backoff_factor ** attempt)
                await asyncio.sleep(wait_time)
        
        raise Exception("SOAP request failed after all retry attempts")
    
    async def call_soap_method(self, method_name: str, parameters: Dict[str, Any], 
                              soap_action: Optional[str] = None) -> Dict[str, Any]:
        """Call a SOAP method with parameters"""
        # Build SOAP envelope
        soap_body = self._build_soap_envelope(method_name, parameters)
        
        endpoint = APIEndpoint(
            url='',
            method='POST',
            params={'soap_action': soap_action} if soap_action else {}
        )
        
        data = {'soap_body': soap_body}
        return await self.make_request(endpoint, data)
    
    def _build_soap_envelope(self, method_name: str, parameters: Dict[str, Any]) -> str:
        """Build SOAP envelope for method call"""
        # Basic SOAP envelope template
        envelope = f'''<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        <{method_name}>
'''
        
        # Add parameters
        for key, value in parameters.items():
            envelope += f'            <{key}>{value}</{key}>\n'
        
        envelope += f'''        </{method_name}>
    </soap:Body>
</soap:Envelope>'''
        
        return envelope


class APIConnectorFactory:
    """Factory for creating API connectors"""
    
    @staticmethod
    def create_connector(api_type: APIType, base_url: str, auth_config: AuthConfig,
                        rate_limit_config: Optional[RateLimitConfig] = None,
                        **kwargs) -> BaseAPIConnector:
        """Create appropriate API connector based on type"""
        
        if api_type == APIType.REST:
            return RESTAPIConnector(base_url, auth_config, rate_limit_config)
        elif api_type == APIType.GRAPHQL:
            return GraphQLAPIConnector(base_url, auth_config, rate_limit_config)
        elif api_type == APIType.SOAP:
            wsdl_url = kwargs.get('wsdl_url')
            return SOAPAPIConnector(base_url, auth_config, wsdl_url, rate_limit_config)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")


# Webhook support for real-time data updates
class WebhookManager:
    """Manage webhook endpoints for real-time API updates"""
    
    def __init__(self):
        self.webhooks = {}
        self.handlers = {}
    
    def register_webhook(self, webhook_id: str, endpoint_url: str, 
                        secret: Optional[str] = None):
        """Register a webhook endpoint"""
        self.webhooks[webhook_id] = {
            'url': endpoint_url,
            'secret': secret,
            'created_at': datetime.now()
        }
    
    def register_handler(self, webhook_id: str, handler_func):
        """Register a handler function for webhook events"""
        self.handlers[webhook_id] = handler_func
    
    async def process_webhook(self, webhook_id: str, payload: Dict[str, Any], 
                            headers: Dict[str, str]) -> bool:
        """Process incoming webhook payload"""
        if webhook_id not in self.webhooks:
            return False
        
        webhook_config = self.webhooks[webhook_id]
        
        # Verify webhook signature if secret is configured
        if webhook_config.get('secret'):
            if not self._verify_signature(payload, headers, webhook_config['secret']):
                return False
        
        # Call registered handler
        if webhook_id in self.handlers:
            try:
                await self.handlers[webhook_id](payload, headers)
                return True
            except Exception as e:
                print(f"Webhook handler error: {e}")
                return False
        
        return True
    
    def _verify_signature(self, payload: Dict[str, Any], headers: Dict[str, str], 
                         secret: str) -> bool:
        """Verify webhook signature (basic implementation)"""
        # This is a simplified signature verification
        # In production, implement proper HMAC verification based on the API provider
        signature_header = headers.get('X-Signature') or headers.get('X-Hub-Signature')
        return signature_header is not None


# API Schema Discovery
class APISchemaDiscovery:
    """Discover and document API schemas"""
    
    def __init__(self, connector: BaseAPIConnector):
        self.connector = connector
    
    async def discover_rest_schema(self, openapi_url: Optional[str] = None) -> Dict[str, Any]:
        """Discover REST API schema from OpenAPI/Swagger"""
        if openapi_url:
            endpoint = APIEndpoint(url=openapi_url)
            try:
                schema = await self.connector.make_request(endpoint)
                return self._parse_openapi_schema(schema)
            except Exception as e:
                return {'error': f'Failed to fetch OpenAPI schema: {e}'}
        
        return {'error': 'OpenAPI URL not provided'}
    
    async def discover_graphql_schema(self) -> Dict[str, Any]:
        """Discover GraphQL schema using introspection"""
        if not isinstance(self.connector, GraphQLAPIConnector):
            return {'error': 'Not a GraphQL connector'}
        
        introspection_query = '''
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    description
                    fields {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
            }
        }
        '''
        
        try:
            result = await self.connector.execute_query(introspection_query)
            return self._parse_graphql_schema(result)
        except Exception as e:
            return {'error': f'GraphQL introspection failed: {e}'}
    
    def _parse_openapi_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAPI schema into simplified format"""
        parsed = {
            'api_type': 'REST',
            'version': schema.get('openapi', schema.get('swagger', 'unknown')),
            'title': schema.get('info', {}).get('title', 'Unknown API'),
            'endpoints': []
        }
        
        paths = schema.get('paths', {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    parsed['endpoints'].append({
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'parameters': details.get('parameters', [])
                    })
        
        return parsed
    
    def _parse_graphql_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GraphQL introspection result"""
        parsed = {
            'api_type': 'GraphQL',
            'types': [],
            'queries': [],
            'mutations': []
        }
        
        types = schema.get('__schema', {}).get('types', [])
        for type_info in types:
            if not type_info['name'].startswith('__'):  # Skip introspection types
                parsed['types'].append({
                    'name': type_info['name'],
                    'kind': type_info['kind'],
                    'description': type_info.get('description', ''),
                    'fields': [f['name'] for f in type_info.get('fields', [])]
                })
        
        return parsed