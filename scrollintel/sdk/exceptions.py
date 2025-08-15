"""
SDK Exception classes for ScrollIntel Python SDK.
"""


class ScrollIntelSDKError(Exception):
    """Base exception for ScrollIntel SDK errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class AuthenticationError(ScrollIntelSDKError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationError(ScrollIntelSDKError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403)


class RateLimitError(ScrollIntelSDKError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(ScrollIntelSDKError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str = "Validation error", errors: list = None):
        super().__init__(message, status_code=400)
        self.errors = errors or []


class NotFoundError(ScrollIntelSDKError):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class ServerError(ScrollIntelSDKError):
    """Raised when server returns an error."""
    
    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, status_code=status_code)


class NetworkError(ScrollIntelSDKError):
    """Raised when network request fails."""
    
    def __init__(self, message: str = "Network error"):
        super().__init__(message)


class TimeoutError(ScrollIntelSDKError):
    """Raised when request times out."""
    
    def __init__(self, message: str = "Request timeout"):
        super().__init__(message)