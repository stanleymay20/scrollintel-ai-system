"""Custom exceptions for AI Data Readiness Platform."""


class AIDataReadinessError(Exception):
    """Base exception for AI Data Readiness Platform."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DataIngestionError(AIDataReadinessError):
    """Errors during data ingestion process."""
    pass


class SchemaValidationError(DataIngestionError):
    """Schema validation failures."""
    pass


class DataFormatError(DataIngestionError):
    """Invalid data format errors."""
    pass


class ConnectionError(DataIngestionError):
    """Data source connection errors."""
    pass


class MetadataExtractionError(AIDataReadinessError):
    """Errors during metadata extraction."""
    pass


class QualityAssessmentError(AIDataReadinessError):
    """Errors during quality assessment."""
    pass


class InsufficientDataError(QualityAssessmentError):
    """Insufficient data for analysis."""
    pass


class BiasDetectionError(AIDataReadinessError):
    """Errors during bias analysis."""
    pass


class FeatureEngineeringError(AIDataReadinessError):
    """Errors during feature engineering."""
    pass


class TransformationError(FeatureEngineeringError):
    """Feature transformation failures."""
    pass


class DriftMonitoringError(AIDataReadinessError):
    """Errors during drift monitoring."""
    pass


class ComplianceValidationError(AIDataReadinessError):
    """Compliance validation errors."""
    pass


class ComplianceAnalysisError(AIDataReadinessError):
    """Errors during compliance analysis."""
    pass


class ProcessingError(AIDataReadinessError):
    """General processing errors."""
    pass


class ResourceExhaustionError(ProcessingError):
    """Resource exhaustion errors."""
    pass


class ConfigurationError(AIDataReadinessError):
    """Configuration-related errors."""
    pass


class DatabaseError(AIDataReadinessError):
    """Database operation errors."""
    pass


class APIError(AIDataReadinessError):
    """API-related errors."""
    pass


class AuthenticationError(APIError):
    """Authentication failures."""
    pass


class AuthorizationError(APIError):
    """Authorization failures."""
    pass


class ValidationError(AIDataReadinessError):
    """Data validation errors."""
    pass


class TimeoutError(AIDataReadinessError):
    """Operation timeout errors."""
    pass