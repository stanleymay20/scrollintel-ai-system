"""
API Versioning and Backward Compatibility Manager

This module handles API versioning, backward compatibility checks,
and migration between API versions.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import difflib

from ..models.api_generation_models import (
    APISpec, Endpoint, Parameter, Response, HTTPMethod,
    APIVersion, GeneratedAPICode
)


class ChangeType(Enum):
    """Types of API changes."""
    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"
    DEPRECATED = "deprecated"
    ADDITION = "addition"


@dataclass
class APIChange:
    """Represents a change between API versions."""
    change_type: ChangeType
    component: str  # endpoint, parameter, response, etc.
    component_name: str
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    migration_notes: Optional[str] = None


@dataclass
class CompatibilityReport:
    """Report on API compatibility between versions."""
    old_version: str
    new_version: str
    is_backward_compatible: bool
    breaking_changes: List[APIChange] = field(default_factory=list)
    non_breaking_changes: List[APIChange] = field(default_factory=list)
    additions: List[APIChange] = field(default_factory=list)
    deprecations: List[APIChange] = field(default_factory=list)
    migration_guide: str = ""


class APIVersioningManager:
    """Manages API versioning and backward compatibility."""
    
    def __init__(self):
        self.breaking_change_rules = {
            "endpoint_removed": "Endpoint removal is a breaking change",
            "endpoint_method_changed": "Changing HTTP method is a breaking change",
            "required_parameter_added": "Adding required parameter is a breaking change",
            "parameter_removed": "Removing parameter is a breaking change",
            "parameter_type_changed": "Changing parameter type is a breaking change",
            "response_schema_changed": "Changing response schema is a breaking change",
            "response_status_removed": "Removing response status is a breaking change"
        }
    
    def compare_api_versions(
        self, 
        old_spec: APISpec, 
        new_spec: APISpec
    ) -> CompatibilityReport:
        """Compare two API specifications and generate compatibility report."""
        report = CompatibilityReport(
            old_version=old_spec.version,
            new_version=new_spec.version,
            is_backward_compatible=True
        )
        
        # Compare endpoints
        endpoint_changes = self._compare_endpoints(old_spec.endpoints, new_spec.endpoints)
        
        # Categorize changes
        for change in endpoint_changes:
            if change.change_type == ChangeType.BREAKING:
                report.breaking_changes.append(change)
                report.is_backward_compatible = False
            elif change.change_type == ChangeType.NON_BREAKING:
                report.non_breaking_changes.append(change)
            elif change.change_type == ChangeType.ADDITION:
                report.additions.append(change)
            elif change.change_type == ChangeType.DEPRECATED:
                report.deprecations.append(change)
        
        # Compare security schemes
        security_changes = self._compare_security_schemes(
            old_spec.security_schemes, 
            new_spec.security_schemes
        )
        
        for change in security_changes:
            if change.change_type == ChangeType.BREAKING:
                report.breaking_changes.append(change)
                report.is_backward_compatible = False
            else:
                report.non_breaking_changes.append(change)
        
        # Generate migration guide
        report.migration_guide = self._generate_migration_guide(report)
        
        return report
    
    def _compare_endpoints(
        self, 
        old_endpoints: List[Endpoint], 
        new_endpoints: List[Endpoint]
    ) -> List[APIChange]:
        """Compare endpoints between versions."""
        changes = []
        
        # Create lookup maps
        old_endpoint_map = {
            f"{ep.method.value}:{ep.path}": ep for ep in old_endpoints
        }
        new_endpoint_map = {
            f"{ep.method.value}:{ep.path}": ep for ep in new_endpoints
        }
        
        # Find removed endpoints
        for key, old_endpoint in old_endpoint_map.items():
            if key not in new_endpoint_map:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="endpoint",
                    component_name=f"{old_endpoint.method.value} {old_endpoint.path}",
                    description="Endpoint removed",
                    old_value=old_endpoint.path,
                    migration_notes=f"Endpoint {old_endpoint.path} has been removed. "
                                  f"Check for alternative endpoints or contact support."
                ))
        
        # Find added endpoints
        for key, new_endpoint in new_endpoint_map.items():
            if key not in old_endpoint_map:
                changes.append(APIChange(
                    change_type=ChangeType.ADDITION,
                    component="endpoint",
                    component_name=f"{new_endpoint.method.value} {new_endpoint.path}",
                    description="New endpoint added",
                    new_value=new_endpoint.path
                ))
        
        # Compare existing endpoints
        for key in old_endpoint_map.keys() & new_endpoint_map.keys():
            old_endpoint = old_endpoint_map[key]
            new_endpoint = new_endpoint_map[key]
            
            endpoint_changes = self._compare_single_endpoint(old_endpoint, new_endpoint)
            changes.extend(endpoint_changes)
        
        return changes
    
    def _compare_single_endpoint(
        self, 
        old_endpoint: Endpoint, 
        new_endpoint: Endpoint
    ) -> List[APIChange]:
        """Compare a single endpoint between versions."""
        changes = []
        endpoint_key = f"{old_endpoint.method.value} {old_endpoint.path}"
        
        # Check if endpoint is deprecated
        if not old_endpoint.deprecated and new_endpoint.deprecated:
            changes.append(APIChange(
                change_type=ChangeType.DEPRECATED,
                component="endpoint",
                component_name=endpoint_key,
                description="Endpoint deprecated",
                migration_notes="This endpoint is deprecated and may be removed in future versions."
            ))
        
        # Compare parameters
        param_changes = self._compare_parameters(
            old_endpoint.parameters, 
            new_endpoint.parameters,
            endpoint_key
        )
        changes.extend(param_changes)
        
        # Compare responses
        response_changes = self._compare_responses(
            old_endpoint.responses,
            new_endpoint.responses,
            endpoint_key
        )
        changes.extend(response_changes)
        
        # Compare request body
        if old_endpoint.request_body != new_endpoint.request_body:
            if old_endpoint.request_body and not new_endpoint.request_body:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="request_body",
                    component_name=endpoint_key,
                    description="Request body removed",
                    old_value=old_endpoint.request_body,
                    migration_notes="Request body is no longer accepted for this endpoint."
                ))
            elif not old_endpoint.request_body and new_endpoint.request_body:
                # Check if request body is required
                is_required = new_endpoint.request_body.get("required", False)
                change_type = ChangeType.BREAKING if is_required else ChangeType.NON_BREAKING
                changes.append(APIChange(
                    change_type=change_type,
                    component="request_body",
                    component_name=endpoint_key,
                    description="Request body added",
                    new_value=new_endpoint.request_body,
                    migration_notes="New request body schema available." if not is_required 
                                  else "Request body is now required for this endpoint."
                ))
        
        return changes
    
    def _compare_parameters(
        self, 
        old_params: List[Parameter], 
        new_params: List[Parameter],
        endpoint_key: str
    ) -> List[APIChange]:
        """Compare parameters between endpoint versions."""
        changes = []
        
        # Create lookup maps
        old_param_map = {param.name: param for param in old_params}
        new_param_map = {param.name: param for param in new_params}
        
        # Find removed parameters
        for param_name, old_param in old_param_map.items():
            if param_name not in new_param_map:
                change_type = ChangeType.BREAKING if old_param.required else ChangeType.NON_BREAKING
                changes.append(APIChange(
                    change_type=change_type,
                    component="parameter",
                    component_name=f"{endpoint_key} - {param_name}",
                    description=f"Parameter '{param_name}' removed",
                    old_value=old_param.type,
                    migration_notes=f"Parameter '{param_name}' is no longer supported."
                ))
        
        # Find added parameters
        for param_name, new_param in new_param_map.items():
            if param_name not in old_param_map:
                change_type = ChangeType.BREAKING if new_param.required else ChangeType.ADDITION
                changes.append(APIChange(
                    change_type=change_type,
                    component="parameter",
                    component_name=f"{endpoint_key} - {param_name}",
                    description=f"Parameter '{param_name}' added",
                    new_value=new_param.type,
                    migration_notes=f"New parameter '{param_name}' available." if not new_param.required
                                  else f"Parameter '{param_name}' is now required."
                ))
        
        # Compare existing parameters
        for param_name in old_param_map.keys() & new_param_map.keys():
            old_param = old_param_map[param_name]
            new_param = new_param_map[param_name]
            
            # Check type changes
            if old_param.type != new_param.type:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="parameter",
                    component_name=f"{endpoint_key} - {param_name}",
                    description=f"Parameter '{param_name}' type changed",
                    old_value=old_param.type,
                    new_value=new_param.type,
                    migration_notes=f"Parameter '{param_name}' type changed from {old_param.type} to {new_param.type}."
                ))
            
            # Check required status changes
            if not old_param.required and new_param.required:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="parameter",
                    component_name=f"{endpoint_key} - {param_name}",
                    description=f"Parameter '{param_name}' is now required",
                    migration_notes=f"Parameter '{param_name}' is now required and must be provided."
                ))
            elif old_param.required and not new_param.required:
                changes.append(APIChange(
                    change_type=ChangeType.NON_BREAKING,
                    component="parameter",
                    component_name=f"{endpoint_key} - {param_name}",
                    description=f"Parameter '{param_name}' is now optional",
                    migration_notes=f"Parameter '{param_name}' is now optional."
                ))
        
        return changes
    
    def _compare_responses(
        self, 
        old_responses: List[Response], 
        new_responses: List[Response],
        endpoint_key: str
    ) -> List[APIChange]:
        """Compare responses between endpoint versions."""
        changes = []
        
        # Create lookup maps
        old_response_map = {resp.status_code: resp for resp in old_responses}
        new_response_map = {resp.status_code: resp for resp in new_responses}
        
        # Find removed response codes
        for status_code, old_response in old_response_map.items():
            if status_code not in new_response_map:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="response",
                    component_name=f"{endpoint_key} - {status_code}",
                    description=f"Response status {status_code} removed",
                    old_value=old_response.description,
                    migration_notes=f"Response status {status_code} is no longer returned."
                ))
        
        # Find added response codes
        for status_code, new_response in new_response_map.items():
            if status_code not in old_response_map:
                changes.append(APIChange(
                    change_type=ChangeType.ADDITION,
                    component="response",
                    component_name=f"{endpoint_key} - {status_code}",
                    description=f"Response status {status_code} added",
                    new_value=new_response.description
                ))
        
        # Compare existing responses
        for status_code in old_response_map.keys() & new_response_map.keys():
            old_response = old_response_map[status_code]
            new_response = new_response_map[status_code]
            
            # Compare schemas
            if old_response.schema != new_response.schema:
                # This is a complex comparison - for now, mark as potentially breaking
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="response_schema",
                    component_name=f"{endpoint_key} - {status_code}",
                    description=f"Response schema changed for status {status_code}",
                    old_value=old_response.schema,
                    new_value=new_response.schema,
                    migration_notes=f"Response schema for status {status_code} has changed. "
                                  f"Please update your client code accordingly."
                ))
        
        return changes
    
    def _compare_security_schemes(
        self, 
        old_schemes: List, 
        new_schemes: List
    ) -> List[APIChange]:
        """Compare security schemes between versions."""
        changes = []
        
        # Create lookup maps
        old_scheme_map = {scheme.name: scheme for scheme in old_schemes}
        new_scheme_map = {scheme.name: scheme for scheme in new_schemes}
        
        # Find removed schemes
        for scheme_name, old_scheme in old_scheme_map.items():
            if scheme_name not in new_scheme_map:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    component="security_scheme",
                    component_name=scheme_name,
                    description=f"Security scheme '{scheme_name}' removed",
                    old_value=old_scheme.type,
                    migration_notes=f"Security scheme '{scheme_name}' is no longer supported."
                ))
        
        # Find added schemes
        for scheme_name, new_scheme in new_scheme_map.items():
            if scheme_name not in old_scheme_map:
                changes.append(APIChange(
                    change_type=ChangeType.ADDITION,
                    component="security_scheme",
                    component_name=scheme_name,
                    description=f"Security scheme '{scheme_name}' added",
                    new_value=new_scheme.type
                ))
        
        return changes
    
    def _generate_migration_guide(self, report: CompatibilityReport) -> str:
        """Generate migration guide based on compatibility report."""
        guide_parts = [
            f"# Migration Guide: {report.old_version} → {report.new_version}",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if report.is_backward_compatible:
            guide_parts.extend([
                "## ✅ Backward Compatible",
                "",
                "This version is backward compatible with the previous version. "
                "No immediate action is required, but we recommend reviewing the changes below.",
                ""
            ])
        else:
            guide_parts.extend([
                "## ⚠️ Breaking Changes",
                "",
                "This version contains breaking changes. Please review and update your code accordingly.",
                ""
            ])
        
        # Breaking changes section
        if report.breaking_changes:
            guide_parts.extend([
                "## Breaking Changes",
                "",
                "The following changes require immediate attention:",
                ""
            ])
            
            for change in report.breaking_changes:
                guide_parts.extend([
                    f"### {change.component_name}",
                    f"**Change:** {change.description}",
                    ""
                ])
                
                if change.old_value and change.new_value:
                    guide_parts.extend([
                        "**Before:**",
                        f"```",
                        str(change.old_value),
                        "```",
                        "",
                        "**After:**",
                        f"```",
                        str(change.new_value),
                        "```",
                        ""
                    ])
                
                if change.migration_notes:
                    guide_parts.extend([
                        "**Migration Notes:**",
                        change.migration_notes,
                        ""
                    ])
                
                guide_parts.append("---")
                guide_parts.append("")
        
        # Deprecations section
        if report.deprecations:
            guide_parts.extend([
                "## Deprecated Features",
                "",
                "The following features are deprecated and will be removed in future versions:",
                ""
            ])
            
            for change in report.deprecations:
                guide_parts.extend([
                    f"- **{change.component_name}**: {change.description}",
                    f"  {change.migration_notes or 'No migration notes available.'}",
                    ""
                ])
        
        # New features section
        if report.additions:
            guide_parts.extend([
                "## New Features",
                "",
                "The following new features are available:",
                ""
            ])
            
            for change in report.additions:
                guide_parts.extend([
                    f"- **{change.component_name}**: {change.description}",
                    ""
                ])
        
        # Non-breaking changes section
        if report.non_breaking_changes:
            guide_parts.extend([
                "## Other Changes",
                "",
                "The following non-breaking changes have been made:",
                ""
            ])
            
            for change in report.non_breaking_changes:
                guide_parts.extend([
                    f"- **{change.component_name}**: {change.description}",
                    ""
                ])
        
        # Testing recommendations
        guide_parts.extend([
            "## Testing Recommendations",
            "",
            "1. **Update your test suite** to cover the new API version",
            "2. **Test all affected endpoints** in your staging environment",
            "3. **Verify error handling** for changed response formats",
            "4. **Update API client libraries** to the latest version",
            "5. **Monitor API usage** after deployment for any issues",
            ""
        ])
        
        # Support section
        guide_parts.extend([
            "## Support",
            "",
            "If you need help with the migration:",
            "",
            "- Check the API documentation for detailed examples",
            "- Contact our support team at support@example.com",
            "- Join our developer community for assistance",
            ""
        ])
        
        return "\n".join(guide_parts)
    
    def create_versioned_api(
        self, 
        base_spec: APISpec, 
        version: str,
        changes: List[APIChange] = None
    ) -> APISpec:
        """Create a new versioned API specification."""
        # Clone the base specification
        new_spec = APISpec(
            name=base_spec.name,
            description=base_spec.description,
            version=version,
            api_type=base_spec.api_type,
            base_url=base_spec.base_url.replace(
                f"/v{base_spec.version.split('.')[0]}", 
                f"/v{version.split('.')[0]}"
            )
        )
        
        # Copy endpoints
        for endpoint in base_spec.endpoints:
            new_endpoint = Endpoint(
                path=endpoint.path,
                method=endpoint.method,
                name=endpoint.name,
                description=endpoint.description,
                summary=endpoint.summary,
                tags=endpoint.tags.copy(),
                parameters=endpoint.parameters.copy(),
                request_body=endpoint.request_body,
                responses=endpoint.responses.copy(),
                security_requirements=endpoint.security_requirements.copy(),
                version=version
            )
            new_spec.add_endpoint(new_endpoint)
        
        # Copy other components
        new_spec.security_schemes = base_spec.security_schemes.copy()
        new_spec.global_parameters = base_spec.global_parameters.copy()
        new_spec.global_headers = base_spec.global_headers.copy()
        
        # Apply changes if provided
        if changes:
            self._apply_changes_to_spec(new_spec, changes)
        
        # Add version info
        api_version = APIVersion(
            version=version,
            release_date=datetime.now(),
            breaking_changes=[c.description for c in (changes or []) if c.change_type == ChangeType.BREAKING],
            changelog=[c.description for c in (changes or [])]
        )
        new_spec.add_version(api_version)
        
        return new_spec
    
    def _apply_changes_to_spec(self, spec: APISpec, changes: List[APIChange]):
        """Apply changes to an API specification."""
        for change in changes:
            if change.component == "endpoint" and change.change_type == ChangeType.BREAKING:
                # Remove endpoint if it's being removed
                if "removed" in change.description.lower():
                    spec.endpoints = [
                        ep for ep in spec.endpoints 
                        if f"{ep.method.value} {ep.path}" != change.component_name
                    ]
    
    def generate_version_compatibility_matrix(
        self, 
        api_specs: List[APISpec]
    ) -> Dict[str, Dict[str, bool]]:
        """Generate compatibility matrix between API versions."""
        matrix = {}
        
        # Sort specs by version
        sorted_specs = sorted(api_specs, key=lambda x: x.version)
        
        for i, spec1 in enumerate(sorted_specs):
            matrix[spec1.version] = {}
            
            for j, spec2 in enumerate(sorted_specs):
                if i == j:
                    matrix[spec1.version][spec2.version] = True
                elif i < j:  # Only check forward compatibility
                    report = self.compare_api_versions(spec1, spec2)
                    matrix[spec1.version][spec2.version] = report.is_backward_compatible
                else:
                    matrix[spec1.version][spec2.version] = False
        
        return matrix
    
    def suggest_version_number(
        self, 
        current_version: str, 
        changes: List[APIChange]
    ) -> str:
        """Suggest next version number based on changes."""
        major, minor, patch = map(int, current_version.split('.'))
        
        # Check for breaking changes
        has_breaking_changes = any(
            change.change_type == ChangeType.BREAKING for change in changes
        )
        
        # Check for new features
        has_new_features = any(
            change.change_type == ChangeType.ADDITION for change in changes
        )
        
        if has_breaking_changes:
            # Major version bump
            return f"{major + 1}.0.0"
        elif has_new_features:
            # Minor version bump
            return f"{major}.{minor + 1}.0"
        else:
            # Patch version bump
            return f"{major}.{minor}.{patch + 1}"
    
    def generate_deprecation_timeline(
        self, 
        api_spec: APISpec, 
        deprecation_period_months: int = 6
    ) -> Dict[str, datetime]:
        """Generate deprecation timeline for API components."""
        timeline = {}
        current_date = datetime.now()
        
        # Find deprecated endpoints
        for endpoint in api_spec.endpoints:
            if endpoint.deprecated:
                endpoint_key = f"{endpoint.method.value} {endpoint.path}"
                timeline[endpoint_key] = current_date + timedelta(days=30 * deprecation_period_months)
        
        return timeline


def create_api_version_with_changes(
    base_spec: APISpec,
    version: str,
    endpoint_changes: Dict[str, Any] = None,
    parameter_changes: Dict[str, Any] = None,
    response_changes: Dict[str, Any] = None
) -> APISpec:
    """Helper function to create API version with specific changes."""
    manager = APIVersioningManager()
    
    changes = []
    
    # Process endpoint changes
    if endpoint_changes:
        for endpoint_key, change_info in endpoint_changes.items():
            changes.append(APIChange(
                change_type=ChangeType(change_info.get("type", "non_breaking")),
                component="endpoint",
                component_name=endpoint_key,
                description=change_info.get("description", ""),
                old_value=change_info.get("old_value"),
                new_value=change_info.get("new_value"),
                migration_notes=change_info.get("migration_notes")
            ))
    
    return manager.create_versioned_api(base_spec, version, changes)