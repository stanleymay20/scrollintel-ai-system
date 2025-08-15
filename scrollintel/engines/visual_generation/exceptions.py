"""
Custom exceptions for visual generation engines.
"""


class VisualGenerationError(Exception):
    """Base exception for visual generation errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "VISUAL_GENERATION_ERROR"
        self.details = details or {}


class PromptError(VisualGenerationError):
    """Errors related to prompt processing."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "PROMPT_ERROR", details)


class PromptAnalysisError(VisualGenerationError):
    """Errors related to prompt analysis."""
    
    def __init__(self, message: str, analysis_stage: str = None, details: dict = None):
        details = details or {}
        if analysis_stage:
            details["analysis_stage"] = analysis_stage
        super().__init__(message, "PROMPT_ANALYSIS_ERROR", details)


class PromptEnhancementError(VisualGenerationError):
    """Errors related to prompt enhancement."""
    
    def __init__(self, message: str, enhancement_stage: str = None, details: dict = None):
        details = details or {}
        if enhancement_stage:
            details["enhancement_stage"] = enhancement_stage
        super().__init__(message, "PROMPT_ENHANCEMENT_ERROR", details)


class ModelError(VisualGenerationError):
    """Errors related to model execution."""
    
    def __init__(self, message: str, model_name: str = None, details: dict = None):
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, "MODEL_ERROR", details)


class ResourceError(VisualGenerationError):
    """Errors related to resource availability."""
    
    def __init__(self, message: str, resource_type: str = None, details: dict = None):
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        super().__init__(message, "RESOURCE_ERROR", details)


class SafetyError(VisualGenerationError):
    """Errors related to content safety."""
    
    def __init__(self, message: str, violation_type: str = None, details: dict = None):
        details = details or {}
        if violation_type:
            details["violation_type"] = violation_type
        super().__init__(message, "SAFETY_ERROR", details)


class QualityError(VisualGenerationError):
    """Errors related to content quality."""
    
    def __init__(self, message: str, quality_issue: str = None, details: dict = None):
        details = details or {}
        if quality_issue:
            details["quality_issue"] = quality_issue
        super().__init__(message, "QUALITY_ERROR", details)


class ConfigurationError(VisualGenerationError):
    """Errors related to configuration."""
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, "CONFIGURATION_ERROR", details)


class APIError(VisualGenerationError):
    """Errors related to external API calls."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None, details: dict = None):
        details = details or {}
        if api_name:
            details["api_name"] = api_name
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, "API_ERROR", details)


class TimeoutError(VisualGenerationError):
    """Errors related to operation timeouts."""
    
    def __init__(self, message: str, timeout_duration: float = None, details: dict = None):
        details = details or {}
        if timeout_duration:
            details["timeout_duration"] = timeout_duration
        super().__init__(message, "TIMEOUT_ERROR", details)


class ValidationError(VisualGenerationError):
    """Errors related to input validation."""
    
    def __init__(self, message: str, field_name: str = None, details: dict = None):
        details = details or {}
        if field_name:
            details["field_name"] = field_name
        super().__init__(message, "VALIDATION_ERROR", details)


class StorageError(VisualGenerationError):
    """Errors related to content storage."""
    
    def __init__(self, message: str, storage_path: str = None, details: dict = None):
        details = details or {}
        if storage_path:
            details["storage_path"] = storage_path
        super().__init__(message, "STORAGE_ERROR", details)


class RateLimitError(VisualGenerationError):
    """Errors related to API rate limiting."""
    
    def __init__(self, message: str, retry_after: int = None, details: dict = None):
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class InvalidRequestError(VisualGenerationError):
    """Errors related to invalid requests."""
    
    def __init__(self, message: str, request_field: str = None, details: dict = None):
        details = details or {}
        if request_field:
            details["request_field"] = request_field
        super().__init__(message, "INVALID_REQUEST_ERROR", details)


class APIConnectionError(VisualGenerationError):
    """Errors related to API connection failures."""
    
    def __init__(self, message: str, endpoint: str = None, details: dict = None):
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, "API_CONNECTION_ERROR", details)


class PerformanceError(VisualGenerationError):
    """Errors related to performance issues."""
    
    def __init__(self, message: str, performance_metric: str = None, details: dict = None):
        details = details or {}
        if performance_metric:
            details["performance_metric"] = performance_metric
        super().__init__(message, "PERFORMANCE_ERROR", details)


class BatchProcessingError(VisualGenerationError):
    """Errors related to batch processing operations."""
    
    def __init__(self, message: str, batch_id: str = None, operation: str = None, details: dict = None):
        details = details or {}
        if batch_id:
            details["batch_id"] = batch_id
        if operation:
            details["operation"] = operation
        super().__init__(message, "BATCH_PROCESSING_ERROR", details)