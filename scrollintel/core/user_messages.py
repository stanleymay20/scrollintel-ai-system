"""
User-friendly error messages and actionable guidance.
Provides clear, helpful messages for different error scenarios.
"""

from typing import Dict, Any, Optional, List
from enum import Enum

from .error_handling import ErrorCategory, ErrorSeverity


class UserMessageType(Enum):
    """Types of user messages."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class UserMessageGenerator:
    """Generates user-friendly messages with actionable guidance."""
    
    def __init__(self):
        self.error_messages = self._initialize_error_messages()
        self.recovery_actions = self._initialize_recovery_actions()
        self.technical_explanations = self._initialize_technical_explanations()
    
    def generate_user_message(
        self,
        error_category: ErrorCategory,
        error_severity: ErrorSeverity,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive user message with guidance."""
        
        # Get base message
        base_message = self._get_base_message(error_category, component)
        
        # Get recovery actions
        recovery_actions = self._get_recovery_actions(error_category, error_severity)
        
        # Get technical explanation if needed
        technical_explanation = self._get_technical_explanation(error_category, component)
        
        # Determine message type
        message_type = self._determine_message_type(error_severity)
        
        return {
            "type": message_type.value,
            "title": self._get_message_title(error_category, component),
            "message": base_message,
            "recovery_actions": recovery_actions,
            "technical_explanation": technical_explanation,
            "severity": error_severity.value,
            "category": error_category.value,
            "component": component,
            "operation": operation,
            "context": context or {}
        }
    
    def _initialize_error_messages(self) -> Dict[ErrorCategory, Dict[str, str]]:
        """Initialize error messages for different categories and components."""
        return {
            ErrorCategory.AGENT: {
                "scroll_cto": "The AI CTO is temporarily unavailable due to high demand. We're automatically scaling resources to restore service. Your request will be processed as soon as capacity is available.",
                "scroll_data_scientist": "The Data Scientist AI is currently offline for maintenance. Your data analysis request has been queued and will be processed automatically when service resumes.",
                "scroll_ml_engineer": "The ML Engineer AI is experiencing high load. Model training requests are being processed with backup systems, which may take slightly longer than usual.",
                "scroll_ai_engineer": "The AI Engineer is temporarily unavailable. Basic AI features are still available through our backup systems, but advanced features may be limited.",
                "scroll_analyst": "The Business Analyst AI is temporarily down for optimization. Report generation is available through our simplified engine with core functionality.",
                "scroll_bi": "The BI Dashboard AI is offline for updates. You can still view existing dashboards, but new dashboard creation is temporarily unavailable.",
                "default": "The AI agent you're trying to use is temporarily unavailable. Our system is automatically working to restore service. Please try again in a few moments."
            },
            ErrorCategory.ENGINE: {
                "automodel": "The AutoML engine is experiencing high load. Your model training request has been queued and will be processed automatically. You'll receive a notification when complete.",
                "scroll_qa": "The Q&A engine is temporarily offline. Basic question answering is available through our backup system, though responses may be simpler than usual.",
                "scroll_viz": "The visualization engine is being optimized. Basic charts are still available, but advanced interactive visualizations are temporarily limited.",
                "scroll_forecast": "The forecasting engine is under maintenance. Historical data analysis is still available, but new predictions are temporarily unavailable.",
                "vault": "The secure vault system is experiencing connectivity issues. Your data is safe, but retrieval may be slower than usual.",
                "default": "A core processing engine is temporarily unavailable. Basic functionality remains available through our backup systems."
            },
            ErrorCategory.SECURITY: {
                "authentication": "Your session has expired or your credentials are invalid. Please log in again.",
                "authorization": "You don't have permission to access this resource. Contact your administrator if you believe this is an error.",
                "audit": "Security logging is experiencing issues. Some actions may be temporarily restricted.",
                "default": "A security check failed. Please verify your credentials and try again."
            },
            ErrorCategory.DATA: {
                "file_upload": "There's an issue with your uploaded file. Please check the file format and try again.",
                "database": "We're experiencing database connectivity issues. Your data may not be fully up to date.",
                "validation": "The data you provided doesn't meet our requirements. Please check the format and try again.",
                "default": "There's an issue with your data. Please verify the format and content."
            },
            ErrorCategory.EXTERNAL_SERVICE: {
                "openai": "OpenAI's services are temporarily unavailable. AI features may be limited.",
                "anthropic": "Anthropic's Claude AI is currently offline. Some AI capabilities are affected.",
                "pinecone": "Vector database services are experiencing issues. Search functionality may be limited.",
                "supabase": "Database services are temporarily unavailable. Some features may not work properly.",
                "default": "An external service we depend on is temporarily unavailable. We're working to restore full functionality."
            },
            ErrorCategory.NETWORK: {
                "timeout": "The request took too long to complete. This might be due to network issues or high server load.",
                "connection": "We're having trouble connecting to our servers. Please check your internet connection.",
                "default": "Network connectivity issues are affecting the service. Please try again in a moment."
            },
            ErrorCategory.RESOURCE: {
                "memory": "The system is running low on memory. Please try again in a few minutes.",
                "cpu": "The system is under heavy load. Processing may be slower than usual.",
                "storage": "Storage capacity is limited. Some features may be temporarily restricted.",
                "default": "System resources are temporarily limited. Please try again shortly."
            },
            ErrorCategory.VALIDATION: {
                "input": "Please check your input and make sure all required fields are filled correctly.",
                "format": "The format of your data is not supported. Please use one of the supported formats.",
                "size": "Your file or data is too large. Please reduce the size and try again.",
                "default": "Please review your input and correct any errors before trying again."
            }
        }
    
    def _initialize_recovery_actions(self) -> Dict[ErrorCategory, List[str]]:
        """Initialize recovery actions for different error categories."""
        return {
            ErrorCategory.AGENT: [
                "Wait 2-3 minutes and try again (service usually recovers automatically)",
                "Try a different AI agent for similar functionality",
                "Check our status page at /status for real-time updates",
                "Use the 'Retry with Backup' button if available",
                "Contact support if the issue persists beyond 10 minutes"
            ],
            ErrorCategory.ENGINE: [
                "Click 'Retry' - the system often recovers within 1-2 minutes",
                "Try processing smaller datasets or simpler queries",
                "Use the 'Basic Mode' option if available",
                "Check if your request can be split into smaller parts",
                "Monitor the progress bar - queued requests are processed automatically"
            ],
            ErrorCategory.SECURITY: [
                "Log out and log back in",
                "Clear your browser cache and cookies",
                "Contact your administrator for permission issues",
                "Use a different browser or device"
            ],
            ErrorCategory.DATA: [
                "Use our file validator tool to check your data format",
                "Download our sample template to ensure correct structure",
                "Try uploading files smaller than 100MB",
                "Remove special characters and ensure UTF-8 encoding",
                "Check that all required columns are present and named correctly"
            ],
            ErrorCategory.EXTERNAL_SERVICE: [
                "Wait a few minutes and try again",
                "Check if alternative features are available",
                "Monitor our status page for service updates",
                "Try using cached or offline features if available"
            ],
            ErrorCategory.NETWORK: [
                "Check your internet connection",
                "Try refreshing the page",
                "Wait a moment and retry",
                "Try using a different network if possible"
            ],
            ErrorCategory.RESOURCE: [
                "Wait a few minutes for resources to free up",
                "Try processing smaller amounts of data",
                "Use simpler operations if available",
                "Try again during off-peak hours"
            ],
            ErrorCategory.VALIDATION: [
                "Review the error details and correct your input",
                "Check the format requirements",
                "Ensure all required fields are completed",
                "Try using the example format provided"
            ]
        }
    
    def _initialize_technical_explanations(self) -> Dict[ErrorCategory, str]:
        """Initialize technical explanations for different error categories."""
        return {
            ErrorCategory.AGENT: "AI agents are microservices that handle specific tasks. When an agent is unavailable, it's usually due to high load, maintenance, or dependency issues.",
            ErrorCategory.ENGINE: "Processing engines handle data transformation and analysis. Engine failures can be caused by resource constraints, data complexity, or service dependencies.",
            ErrorCategory.SECURITY: "Security errors occur when authentication or authorization checks fail. This protects your data and ensures proper access control.",
            ErrorCategory.DATA: "Data errors happen when the system cannot process your input due to format, size, or content issues. Data validation helps ensure system stability.",
            ErrorCategory.EXTERNAL_SERVICE: "We integrate with external AI and data services. When these services are unavailable, some features may be limited until they recover.",
            ErrorCategory.NETWORK: "Network errors occur when communication between your device and our servers is interrupted or slow. This can be due to connectivity issues.",
            ErrorCategory.RESOURCE: "Resource errors happen when the system is under heavy load or running low on computational resources like memory or CPU.",
            ErrorCategory.VALIDATION: "Validation errors occur when your input doesn't meet the required format or constraints. This helps prevent system errors and ensures data quality."
        }
    
    def _get_base_message(self, category: ErrorCategory, component: str) -> str:
        """Get base error message for category and component."""
        category_messages = self.error_messages.get(category, {})
        return category_messages.get(component, category_messages.get("default", "An error occurred."))
    
    def _get_recovery_actions(self, category: ErrorCategory, severity: ErrorSeverity) -> List[str]:
        """Get recovery actions based on category and severity."""
        actions = self.recovery_actions.get(category, [])
        
        # Add severity-specific actions
        if severity == ErrorSeverity.CRITICAL:
            actions = ["Contact support immediately"] + actions
        elif severity == ErrorSeverity.HIGH:
            actions = ["Try alternative features if available"] + actions
        
        return actions[:4]  # Limit to 4 actions for better UX
    
    def _get_technical_explanation(self, category: ErrorCategory, component: str) -> str:
        """Get technical explanation for the error."""
        return self.technical_explanations.get(category, "A technical issue occurred.")
    
    def _determine_message_type(self, severity: ErrorSeverity) -> UserMessageType:
        """Determine message type based on severity."""
        if severity == ErrorSeverity.CRITICAL:
            return UserMessageType.ERROR
        elif severity == ErrorSeverity.HIGH:
            return UserMessageType.ERROR
        elif severity == ErrorSeverity.MEDIUM:
            return UserMessageType.WARNING
        else:
            return UserMessageType.INFO
    
    def _get_message_title(self, category: ErrorCategory, component: str) -> str:
        """Get appropriate title for the error message."""
        titles = {
            ErrorCategory.AGENT: f"AI Agent Unavailable",
            ErrorCategory.ENGINE: f"Processing Engine Issue",
            ErrorCategory.SECURITY: f"Security Check Failed",
            ErrorCategory.DATA: f"Data Issue",
            ErrorCategory.EXTERNAL_SERVICE: f"Service Temporarily Unavailable",
            ErrorCategory.NETWORK: f"Connection Issue",
            ErrorCategory.RESOURCE: f"System Resources Limited",
            ErrorCategory.VALIDATION: f"Input Validation Error"
        }
        
        return titles.get(category, "System Issue")
    
    def generate_success_message(
        self,
        operation: str,
        component: str,
        fallback_used: bool = False,
        degraded: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate success message with any caveats."""
        
        if fallback_used:
            message = f"Your request was completed using backup systems. All functionality is available."
            title = "Request Completed (Backup Systems)"
        elif degraded:
            message = f"Your request was completed with limited functionality. Some advanced features may not be available."
            title = "Request Completed (Limited Mode)"
        else:
            message = f"Your request was completed successfully."
            title = "Request Completed"
        
        return {
            "type": UserMessageType.SUCCESS.value,
            "title": title,
            "message": message,
            "operation": operation,
            "component": component,
            "fallback_used": fallback_used,
            "degraded": degraded,
            "context": context or {}
        }
    
    def generate_maintenance_message(
        self,
        component: str,
        estimated_duration: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate maintenance message."""
        
        message = f"The {component} is currently undergoing maintenance."
        if estimated_duration:
            message += f" Expected completion: {estimated_duration}."
        message += " We apologize for any inconvenience."
        
        return {
            "type": UserMessageType.INFO.value,
            "title": "Scheduled Maintenance",
            "message": message,
            "component": component,
            "estimated_duration": estimated_duration,
            "recovery_actions": [
                "Check back after the maintenance window",
                "Use alternative features if available",
                "Subscribe to status updates for notifications"
            ]
        }


# Global message generator instance
user_message_generator = UserMessageGenerator()


def get_user_friendly_error(
    error_category: ErrorCategory,
    error_severity: ErrorSeverity,
    component: str,
    operation: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get user-friendly error message with guidance."""
    return user_message_generator.generate_user_message(
        error_category, error_severity, component, operation, context
    )


def get_success_message(
    operation: str,
    component: str,
    fallback_used: bool = False,
    degraded: bool = False,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get success message with any caveats."""
    return user_message_generator.generate_success_message(
        operation, component, fallback_used, degraded, context
    )


def get_maintenance_message(
    component: str,
    estimated_duration: Optional[str] = None
) -> Dict[str, Any]:
    """Get maintenance message."""
    return user_message_generator.generate_maintenance_message(
        component, estimated_duration
    )