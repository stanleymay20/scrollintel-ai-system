"""
API Integration Framework for Enterprise Connectivity
Supports REST, GraphQL, and SOAP integrations with authentication and rate limiting.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from requests_oauthlib import OAuth1, OAuth2Session


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
class APICredentials:
    auth_type: AuthType
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: Optional[str] = "X-API-Key"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    oauth_token: Optional[str] = None
    oauth_token_secret: Optional[str] = None


@dataclass
class RateLimitConfig:
    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    requests_per_hour: float = 36000.0
    burst_limit: int = 50
    retry_after_seconds: int = 60


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on_status: List[int] = None
    
    def __post_init__(self):
        if self.retry_on_status is None:
            self.retry_on_status = [429, 500, 502, 503, 504]


@dataclass
class APIEndpoint:
    url: str
    method: str = "GET"
    headers: Dict[str, str] = None
    timeout: int = 30
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class RateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token for making a request"""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Add tokens based on rate
            tokens_to_add = time_passed * self.config.requests_per_second
            self.tokens = min(self.config.burst_limit, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.config.requests_per_second


class BaseAPIConnector(ABC):
    """Base class for API connectors"""
    
    def __init__(
        self,
        base_url: str,
        credentials: APICredentials,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.credentials = credentials
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.retry_config = retry_config or RetryConfig()
        self.session = None
    
    def _get_auth(self) -> Optional[Any]:
        """Get authentication object for requests"""
        if self.credentials.auth_type == AuthType.BASIC:
            return HTTPBasicAuth(self.credentials.username, self.credentials.password)
        elif self.credentials.auth_type == AuthType.DIGEST:
            return HTTPDigestAuth(self.credentials.username, self.credentials.password)
        elif self.credentials.auth_type == AuthType.OAUTH1:
            return OAuth1(
                self.credentials.client_id,
                client_secret=self.credentials.client_secret,
                resource_owner_key=self.credentials.oauth_token,
                resource_owner_secret=self.credentials.oauth_token_secret
            )
        return None
    
    def _get_headers(self, additional_headers: Dict[str, str] = None) -> Dict[str, str]:
        """Get headers with authentication"""
        headers = {"Content-Type": "application/json"}
        
        if self.credentials.auth_type == AuthType.BEARER:
            headers["Authorization"] = f"Bearer {self.credentials.token}"
        elif self.credentials.auth_type == AuthType.API_KEY:
            headers[self.credentials.api_key_header] = self.credentials.api_key
        
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response:
        """Make HTTP request with retry logic"""
        for attempt in range(self.retry_config.max_retries + 1):
            # Rate limiting
            if not await self.rate_limiter.acquire():
                wait_time = self.rate_limiter.get_wait_time()
                await asyncio.sleep(wait_time)
                continue
            
            try:
                response = requests.request(method, url, **kwargs)
                
                if response.status_code not in self.retry_config.retry_on_status:
                    return response
                
                if attempt < self.retry_config.max_retries:
                    wait_time = self.retry_config.backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    return response
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.retry_config.max_retries:
                    wait_time = self.retry_config.backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        
        raise Exception("Max retries exceeded")
    
    @abstractmethod
    async def execute_request(self, endpoint: APIEndpoint, data: Any = None) -> Dict[str, Any]:
        """Execute API request"""
        pass
    
    @abstractmethod
    async def discover_schema(self) -> Dict[str, Any]:
        """Discover API schema/documentation"""
        pass


class RESTConnector(BaseAPIConnector):
    """REST API connector with full HTTP method support"""
    
    async def execute_request(self, endpoint: APIEndpoint, data: Any = None) -> Dict[str, Any]:
        """Execute REST API request"""
        url = urljoin(self.base_url + '/', endpoint.url.lstrip('/'))
        headers = self._get_headers(endpoint.headers)
        auth = self._get_auth()
        
        kwargs = {
            'headers': headers,
            'auth': auth,
            'timeout': endpoint.timeout
        }
        
        if data and endpoint.method.upper() in ['POST', 'PUT', 'PATCH']:
            kwargs['json'] = data
        elif data and endpoint.method.upper() == 'GET':
            kwargs['params'] = data
        
        response = await self._make_request_with_retry(
            endpoint.method.upper(),
            url,
            **kwargs
        )
        
        try:
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': response.json() if response.content else None,
                'success': 200 <= response.status_code < 300
            }
        except json.JSONDecodeError:
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': response.text,
                'success': 200 <= response.status_code < 300
            }
    
    async def discover_schema(self) -> Dict[str, Any]:
        """Discover REST API schema via OpenAPI/Swagger"""
        common_schema_paths = [
            '/swagger.json',
            '/api/swagger.json',
            '/openapi.json',
            '/api/openapi.json',
            '/docs/swagger.json',
            '/api-docs'
        ]
        
        for path in common_schema_paths:
            try:
                endpoint = APIEndpoint(url=path, method="GET")
                result = await self.execute_request(endpoint)
                if result['success'] and result['data']:
                    return {
                        'type': 'openapi',
                        'schema': result['data'],
                        'discovered_at': path
                    }
            except Exception:
                continue
        
        return {'type': 'unknown', 'schema': None}


class GraphQLConnector(BaseAPIConnector):
    """GraphQL API connector with introspection support"""
    
    def __init__(self, *args, graphql_endpoint: str = "/graphql", **kwargs):
        super().__init__(*args, **kwargs)
        self.graphql_endpoint = graphql_endpoint
    
    async def execute_request(self, endpoint: APIEndpoint, data: Any = None) -> Dict[str, Any]:
        """Execute GraphQL query/mutation"""
        url = urljoin(self.base_url + '/', self.graphql_endpoint.lstrip('/'))
        headers = self._get_headers(endpoint.headers)
        auth = self._get_auth()
        
        # GraphQL requests are always POST
        graphql_data = {
            'query': data.get('query') if isinstance(data, dict) else str(data),
            'variables': data.get('variables', {}) if isinstance(data, dict) else {}
        }
        
        response = await self._make_request_with_retry(
            'POST',
            url,
            json=graphql_data,
            headers=headers,
            auth=auth,
            timeout=endpoint.timeout
        )
        
        try:
            result = response.json()
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': result.get('data'),
                'errors': result.get('errors'),
                'success': 200 <= response.status_code < 300 and not result.get('errors')
            }
        except json.JSONDecodeError:
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': None,
                'errors': [{'message': 'Invalid JSON response'}],
                'success': False
            }
    
    async def discover_schema(self) -> Dict[str, Any]:
        """Discover GraphQL schema via introspection"""
        introspection_query = """
        query IntrospectionQuery {
            __schema {
                queryType { name }
                mutationType { name }
                subscriptionType { name }
                types {
                    ...FullType
                }
            }
        }
        
        fragment FullType on __Type {
            kind
            name
            description
            fields(includeDeprecated: true) {
                name
                description
                args {
                    ...InputValue
                }
                type {
                    ...TypeRef
                }
                isDeprecated
                deprecationReason
            }
            inputFields {
                ...InputValue
            }
            interfaces {
                ...TypeRef
            }
            enumValues(includeDeprecated: true) {
                name
                description
                isDeprecated
                deprecationReason
            }
            possibleTypes {
                ...TypeRef
            }
        }
        
        fragment InputValue on __InputValue {
            name
            description
            type { ...TypeRef }
            defaultValue
        }
        
        fragment TypeRef on __Type {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                    ofType {
                                        kind
                                        name
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        try:
            endpoint = APIEndpoint(url=self.graphql_endpoint, method="POST")
            result = await self.execute_request(endpoint, {'query': introspection_query})
            
            if result['success'] and result['data']:
                return {
                    'type': 'graphql',
                    'schema': result['data']['__schema'],
                    'introspection': True
                }
        except Exception as e:
            return {
                'type': 'graphql',
                'schema': None,
                'error': str(e)
            }
        
        return {'type': 'graphql', 'schema': None}


class SOAPConnector(BaseAPIConnector):
    """SOAP API connector with WSDL support"""
    
    def __init__(self, *args, wsdl_url: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wsdl_url = wsdl_url
        self.soap_action_header = "SOAPAction"
    
    async def execute_request(self, endpoint: APIEndpoint, data: Any = None) -> Dict[str, Any]:
        """Execute SOAP request"""
        url = urljoin(self.base_url + '/', endpoint.url.lstrip('/'))
        headers = self._get_headers(endpoint.headers)
        headers["Content-Type"] = "text/xml; charset=utf-8"
        
        # Add SOAPAction header if provided
        if "soap_action" in endpoint.headers:
            headers[self.soap_action_header] = endpoint.headers["soap_action"]
        
        auth = self._get_auth()
        
        # Convert data to SOAP envelope if it's not already XML
        if isinstance(data, dict):
            soap_body = self._dict_to_soap_body(data)
        else:
            soap_body = str(data) if data else ""
        
        soap_envelope = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        {soap_body}
    </soap:Body>
</soap:Envelope>"""
        
        response = await self._make_request_with_retry(
            'POST',
            url,
            data=soap_envelope,
            headers=headers,
            auth=auth,
            timeout=endpoint.timeout
        )
        
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'data': response.text,
            'success': 200 <= response.status_code < 300
        }
    
    def _dict_to_soap_body(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to simple SOAP body XML"""
        def dict_to_xml(d, root_name="request"):
            xml = f"<{root_name}>"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml += dict_to_xml(value, key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml += dict_to_xml(item, key)
                        else:
                            xml += f"<{key}>{item}</{key}>"
                else:
                    xml += f"<{key}>{value}</{key}>"
            xml += f"</{root_name}>"
            return xml
        
        return dict_to_xml(data)
    
    async def discover_schema(self) -> Dict[str, Any]:
        """Discover SOAP schema via WSDL"""
        if not self.wsdl_url:
            # Try common WSDL locations
            wsdl_paths = ['?wsdl', '/wsdl', '?WSDL']
            for path in wsdl_paths:
                try:
                    wsdl_url = self.base_url + path
                    endpoint = APIEndpoint(url=path, method="GET")
                    result = await self.execute_request(endpoint)
                    if result['success'] and 'wsdl' in result['data'].lower():
                        return {
                            'type': 'soap',
                            'wsdl': result['data'],
                            'wsdl_url': wsdl_url
                        }
                except Exception:
                    continue
        else:
            try:
                response = requests.get(self.wsdl_url, timeout=30)
                if response.status_code == 200:
                    return {
                        'type': 'soap',
                        'wsdl': response.text,
                        'wsdl_url': self.wsdl_url
                    }
            except Exception:
                pass
        
        return {'type': 'soap', 'wsdl': None}


class APIConnector:
    """Main API connector factory and manager"""
    
    @staticmethod
    def create_connector(
        api_type: APIType,
        base_url: str,
        credentials: APICredentials,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> BaseAPIConnector:
        """Create appropriate API connector based on type"""
        
        if api_type == APIType.REST:
            return RESTConnector(base_url, credentials, rate_limit_config, retry_config)
        elif api_type == APIType.GRAPHQL:
            return GraphQLConnector(
                base_url, credentials, rate_limit_config, retry_config, **kwargs
            )
        elif api_type == APIType.SOAP:
            return SOAPConnector(
                base_url, credentials, rate_limit_config, retry_config, **kwargs
            )
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    @staticmethod
    def create_rest_connector(
        base_url: str,
        credentials: APICredentials,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> RESTConnector:
        """Create REST API connector"""
        return RESTConnector(base_url, credentials, rate_limit_config, retry_config)
    
    @staticmethod
    def create_graphql_connector(
        base_url: str,
        credentials: APICredentials,
        graphql_endpoint: str = "/graphql",
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> GraphQLConnector:
        """Create GraphQL API connector"""
        return GraphQLConnector(
            base_url, credentials, rate_limit_config, retry_config,
            graphql_endpoint=graphql_endpoint
        )
    
    @staticmethod
    def create_soap_connector(
        base_url: str,
        credentials: APICredentials,
        wsdl_url: Optional[str] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> SOAPConnector:
        """Create SOAP API connector"""
        return SOAPConnector(
            base_url, credentials, rate_limit_config, retry_config,
            wsdl_url=wsdl_url
        )