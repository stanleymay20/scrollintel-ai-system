"""
Integration Tests for Vendor and Supply Chain Security System
Tests the complete vendor security management workflow
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

from security.vendor_supply_chain.vendor_security_assessor import (
    VendorSecurityAssessor, VendorProfile, RiskLevel, AssessmentStatus
)
from security.vendor_supply_chain.vulnerability_scanner import (
    ThirdPartySoftwareScanner, VulnerabilitySeverity, ScanType
)
from security.vendor_supply_chain.sbom_manager import (
    SBOMManager, SBOMFormat, ComponentType
)
from security.vendor_supply_chain.vendor_access_monitor import (
    VendorAccessMonitor, AccessType, AccessStatus
)
from security.vendor_supply_chain.incident_tracker import (
    VendorIncidentTracker, IncidentSeverity, IncidentCategory
)
from security.vendor_supply_chain.contract_templates import (
    SecurityContractTemplateManager, ContractType, ComplianceFramework
)

class TestVendorSupplyChainIntegration:
    """Integration tests for vendor supply chain security"""
    
    @pytest.fixture
    def vendor_assessor(self):
        """Create vendor security assessor instance"""
        return VendorSecurityAssessor()
    
    @pytest.fixture
    def vulnerability_scanner(self):
        """Create vulnerability scanner instance"""
        return ThirdPartySoftwareScanner()
    
    @pytest.fixture
    def sbom_manager(self):
        """Create SBOM manager instance"""
        return SBOMManager()
    
    @pytest.fixture
    def access_monitor(self):
        """Create vendor access monitor instance"""
        return VendorAccessMonitor()
    
    @pytest.fixture
    def incident_tracker(self):
        """Create incident tracker instance"""
        return VendorIncidentTracker()
    
    @pytest.fixture
    def contract_manager(self):
        """Create contract template manager instance"""
        return SecurityContractTemplateManager()
    
    @pytest.fixture
    def sample_vendor_profile(self):
        """Create sample vendor profile"""
        return VendorProfile(
            vendor_id="VENDOR-001",
            name="Test Security Vendor",
            contact_email="security@testvendor.com",
            business_type="cloud_provider",
            services_provided=["data_processing", "authentication"],
            data_access_level="confidential",
            compliance_certifications=["SOC2_TYPE_II", "ISO27001"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def sample_software_file(self):
        """Create sample software file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.zip', delete=False) as f:
            f.write("Sample software package content")
            return f.name
    
    @pytest.mark.asyncio
    async def test_complete_vendor_onboarding_workflow(
        self, vendor_assessor, contract_manager, sample_vendor_profile
    ):
        """Test complete vendor onboarding workflow"""
        
        # Step 1: Generate contract requirements
        cloud_templates = contract_manager.get_templates_by_type(ContractType.CLOUD_SERVICE)
        assert len(cloud_templates) > 0
        
        template = cloud_templates[0]
        contract_language = contract_manager.generate_contract_language(
            template_id=template.template_id,
            vendor_name=sample_vendor_profile.name,
            service_description="Cloud data processing services"
        )
        
        assert "template_id" in contract_language
        assert "contract_sections" in contract_language
        assert sample_vendor_profile.name in contract_language["vendor_name"]
        
        # Step 2: Validate vendor compliance
        vendor_capabilities = {
            "dp001": True,  # Data encryption at rest
            "dp002": True,  # Data encryption in transit
            "ac001": True,  # Multi-factor authentication
            "ir001": True   # Incident response plan
        }
        
        compliance_results = contract_manager.validate_vendor_compliance(
            template_id=template.template_id,
            vendor_certifications=sample_vendor_profile.compliance_certifications,
            vendor_capabilities=vendor_capabilities
        )
        
        assert "overall_compliance" in compliance_results
        assert "requirement_compliance" in compliance_results
        
        # Step 3: Conduct security assessment
        assessment = await vendor_assessor.assess_vendor(sample_vendor_profile)
        
        assert assessment.vendor_id == sample_vendor_profile.vendor_id
        assert assessment.status == AssessmentStatus.COMPLETED
        assert isinstance(assessment.risk_score, float)
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(assessment.findings) > 0
        assert len(assessment.recommendations) > 0
        
        # Step 4: Generate assessment report
        report = await vendor_assessor.generate_assessment_report(assessment)
        
        assert "assessment_summary" in report
        assert "findings" in report
        assert "recommendations" in report
        assert report["assessment_summary"]["vendor_id"] == sample_vendor_profile.vendor_id
    
    @pytest.mark.asyncio
    async def test_software_security_analysis_workflow(
        self, vulnerability_scanner, sbom_manager, sample_software_file
    ):
        """Test complete software security analysis workflow"""
        
        vendor_id = "VENDOR-002"
        software_name = "TestApp"
        software_version = "1.0.0"
        
        # Step 1: Vulnerability scanning
        scan_result = await vulnerability_scanner.scan_software_package(
            vendor_id=vendor_id,
            software_path=sample_software_file,
            software_name=software_name,
            software_version=software_version
        )
        
        assert scan_result.vendor_id == vendor_id
        assert scan_result.software_name == software_name
        assert scan_result.software_version == software_version
        assert isinstance(scan_result.overall_risk_score, float)
        assert scan_result.overall_risk_score >= 0.0
        assert scan_result.scanned_files >= 0
        
        # Step 2: Generate scan report
        scan_report = await vulnerability_scanner.generate_scan_report(scan_result)
        
        assert "scan_summary" in scan_report
        assert "vulnerabilities" in scan_report
        assert "backdoor_indicators" in scan_report
        assert "recommendations" in scan_report
        
        # Step 3: Generate SBOM
        sbom = await sbom_manager.generate_sbom(
            vendor_id=vendor_id,
            software_path=sample_software_file,
            software_name=software_name,
            software_version=software_version,
            sbom_format=SBOMFormat.SPDX_JSON
        )
        
        assert sbom.vendor_id == vendor_id
        assert sbom.software_name == software_name
        assert sbom.version == software_version
        assert sbom.format == SBOMFormat.SPDX_JSON
        assert len(sbom.components) > 0
        
        # Step 4: Export SBOM
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            export_success = await sbom_manager.export_sbom(
                sbom_id=sbom.sbom_id,
                format=SBOMFormat.SPDX_JSON,
                output_path=temp_file.name
            )
        
        assert export_success
        
        # Step 5: Get SBOM analytics
        analytics = await sbom_manager.get_sbom_analytics(sbom.sbom_id)
        
        assert "sbom_id" in analytics
        assert "summary" in analytics
        assert "distributions" in analytics
        assert "risk_assessment" in analytics
        
        # Cleanup
        Path(sample_software_file).unlink(missing_ok=True)
        Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_vendor_access_management_workflow(self, access_monitor):
        """Test complete vendor access management workflow"""
        
        vendor_id = "VENDOR-003"
        requester_email = "analyst@company.com"
        
        # Step 1: Request access
        access_request = await access_monitor.request_access(
            vendor_id=vendor_id,
            requester_email=requester_email,
            access_type=AccessType.READ_WRITE,
            resources=["database", "config"],
            business_justification="Security assessment required",
            duration_hours=4,
            emergency=False
        )
        
        assert access_request.vendor_id == vendor_id
        assert access_request.requester_email == requester_email
        assert access_request.access_type == AccessType.READ_WRITE
        assert access_request.status in [AccessStatus.PENDING_APPROVAL, AccessStatus.ACTIVE]
        
        # Step 2: Approve access (if required)
        if access_request.approver_required:
            approval_success = await access_monitor.approve_access_request(
                request_id=access_request.request_id,
                approver="security_manager@company.com"
            )
            assert approval_success
        
        # Step 3: Get active grants
        active_grants = access_monitor.get_active_grants(vendor_id)
        assert len(active_grants) > 0
        
        grant = active_grants[0]
        assert grant.vendor_id == vendor_id
        assert grant.user_email == requester_email
        assert grant.status == AccessStatus.ACTIVE
        
        # Step 4: Log access activity
        activity = await access_monitor.log_access_activity(
            token=grant.access_token,
            action="read_config",
            resource="database",
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0",
            success=True
        )
        
        assert activity.grant_id == grant.grant_id
        assert activity.vendor_id == vendor_id
        assert activity.action == "read_config"
        assert activity.success
        assert isinstance(activity.risk_score, float)
        
        # Step 5: Generate access report
        report = await access_monitor.generate_access_report(vendor_id, 7)
        
        assert "vendor_id" in report
        assert "access_summary" in report
        assert "risk_analysis" in report
        assert "recommendations" in report
        
        # Step 6: Revoke access
        revoke_success = await access_monitor.revoke_access(
            grant_id=grant.grant_id,
            reason="Assessment completed"
        )
        assert revoke_success
    
    @pytest.mark.asyncio
    async def test_incident_management_workflow(self, incident_tracker):
        """Test complete incident management workflow"""
        
        vendor_id = "VENDOR-004"
        vendor_name = "Test Incident Vendor"
        
        # Step 1: Create incident
        incident = await incident_tracker.create_incident(
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            title="Unauthorized Access Detected",
            description="Suspicious login activity detected from unusual location",
            category=IncidentCategory.UNAUTHORIZED_ACCESS,
            severity=IncidentSeverity.HIGH,
            reporter="security_analyst@company.com",
            affected_systems=["authentication_service", "user_database"],
            affected_data_types=["user_credentials", "personal_data"]
        )
        
        assert incident.vendor_id == vendor_id
        assert incident.vendor_name == vendor_name
        assert incident.category == IncidentCategory.UNAUTHORIZED_ACCESS
        assert incident.severity == IncidentSeverity.HIGH
        assert len(incident.actions) > 0  # Should have initial actions
        assert len(incident.timeline) > 0  # Should have creation entry
        
        # Step 2: Assign incident
        assign_success = await incident_tracker.assign_incident(
            incident_id=incident.incident_id,
            assignee="senior_analyst@company.com",
            assigner="security_manager@company.com"
        )
        assert assign_success
        
        # Step 3: Add evidence
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as evidence_file:
            evidence_file.write("Sample log evidence")
            evidence_file_path = evidence_file.name
        
        evidence = await incident_tracker.add_evidence(
            incident_id=incident.incident_id,
            evidence_type="log",
            description="Authentication logs showing suspicious activity",
            file_path=evidence_file_path,
            collected_by="security_analyst@company.com"
        )
        
        assert evidence is not None
        assert evidence.incident_id == incident.incident_id
        assert evidence.evidence_type == "log"
        assert evidence.file_path == evidence_file_path
        
        # Step 4: Complete an action
        if incident.actions:
            action = incident.actions[0]
            complete_success = await incident_tracker.complete_action(
                incident_id=incident.incident_id,
                action_id=action.action_id,
                results="Unauthorized access contained and credentials reset",
                completed_by="senior_analyst@company.com"
            )
            assert complete_success
        
        # Step 5: Update incident status
        status_update_success = await incident_tracker.update_incident_status(
            incident_id=incident.incident_id,
            new_status=incident_tracker.IncidentStatus.RESOLVED,
            updater="senior_analyst@company.com",
            notes="Incident resolved - access revoked and systems secured"
        )
        assert status_update_success
        
        # Step 6: Generate incident report
        report = await incident_tracker.generate_incident_report(incident.incident_id)
        
        assert "incident_summary" in report
        assert "metrics" in report
        assert "impact_assessment" in report
        assert "response_details" in report
        assert "compliance_analysis" in report
        
        # Step 7: Generate vendor incident summary
        summary = await incident_tracker.generate_vendor_incident_summary(vendor_id, 30)
        
        assert "vendor_id" in summary
        assert "incident_summary" in summary
        assert "risk_assessment" in summary
        assert "recommendations" in summary
        
        # Cleanup
        Path(evidence_file_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_contract_template_workflow(self, contract_manager):
        """Test contract template management workflow"""
        
        # Step 1: Get available templates
        templates = contract_manager.get_templates_by_type(ContractType.SAAS_SUBSCRIPTION)
        assert len(templates) > 0
        
        template = templates[0]
        
        # Step 2: Generate contract language
        contract_language = contract_manager.generate_contract_language(
            template_id=template.template_id,
            vendor_name="SaaS Test Vendor",
            service_description="Customer relationship management software"
        )
        
        assert "template_id" in contract_language
        assert "vendor_name" in contract_language
        assert "contract_sections" in contract_language
        assert contract_language["vendor_name"] == "SaaS Test Vendor"
        
        # Step 3: Search security requirements
        data_protection_reqs = contract_manager.search_requirements(
            category=contract_manager.SecurityRequirementCategory.DATA_PROTECTION,
            mandatory_only=True
        )
        
        assert len(data_protection_reqs) > 0
        for req in data_protection_reqs:
            assert req.mandatory
            assert req.category.value == "data_protection"
        
        # Step 4: Create custom template
        custom_template = contract_manager.create_custom_template(
            name="Custom Integration Template",
            contract_type=ContractType.CONSULTING_SERVICE,
            requirement_ids=["DP001", "DP002", "AC001", "IR001"],
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.GDPR]
        )
        
        assert custom_template.name == "Custom Integration Template"
        assert custom_template.contract_type == ContractType.CONSULTING_SERVICE
        assert len(custom_template.security_requirements) == 4
        assert ComplianceFramework.SOC2_TYPE_II in custom_template.compliance_requirements
        
        # Step 5: Validate vendor compliance
        vendor_capabilities = {
            "dp001": True,
            "dp002": True,
            "ac001": False,  # Missing MFA
            "ir001": True
        }
        
        compliance_results = contract_manager.validate_vendor_compliance(
            template_id=custom_template.template_id,
            vendor_certifications=["SOC2_TYPE_II"],
            vendor_capabilities=vendor_capabilities
        )
        
        assert "overall_compliance" in compliance_results
        assert "requirement_compliance" in compliance_results
        assert "missing_certifications" in compliance_results
        assert "recommendations" in compliance_results
        
        # Should not be fully compliant due to missing MFA
        assert not compliance_results["overall_compliance"]
        
        # Step 6: Generate requirements matrix
        matrix = contract_manager.generate_requirement_matrix(custom_template.template_id)
        
        assert "template_id" in matrix
        assert "requirements_by_category" in matrix
        assert "compliance_mapping" in matrix
        assert "sla_summary" in matrix
    
    @pytest.mark.asyncio
    async def test_end_to_end_vendor_lifecycle(
        self, vendor_assessor, vulnerability_scanner, sbom_manager,
        access_monitor, incident_tracker, contract_manager,
        sample_vendor_profile, sample_software_file
    ):
        """Test complete end-to-end vendor lifecycle management"""
        
        vendor_id = sample_vendor_profile.vendor_id
        
        # Phase 1: Contract and Onboarding
        templates = contract_manager.get_templates_by_type(ContractType.CLOUD_SERVICE)
        template = templates[0]
        
        contract_language = contract_manager.generate_contract_language(
            template_id=template.template_id,
            vendor_name=sample_vendor_profile.name,
            service_description="Cloud security services"
        )
        assert "contract_sections" in contract_language
        
        # Phase 2: Security Assessment
        assessment = await vendor_assessor.assess_vendor(sample_vendor_profile)
        assert assessment.status == AssessmentStatus.COMPLETED
        
        # Phase 3: Software Analysis
        scan_result = await vulnerability_scanner.scan_software_package(
            vendor_id=vendor_id,
            software_path=sample_software_file,
            software_name="VendorApp",
            software_version="2.1.0"
        )
        assert scan_result.vendor_id == vendor_id
        
        sbom = await sbom_manager.generate_sbom(
            vendor_id=vendor_id,
            software_path=sample_software_file,
            software_name="VendorApp",
            software_version="2.1.0"
        )
        assert sbom.vendor_id == vendor_id
        
        # Phase 4: Access Management
        access_request = await access_monitor.request_access(
            vendor_id=vendor_id,
            requester_email="vendor@testvendor.com",
            access_type=AccessType.READ_ONLY,
            resources=["monitoring_data"],
            business_justification="Routine maintenance access",
            duration_hours=2
        )
        assert access_request.vendor_id == vendor_id
        
        # Phase 5: Incident Handling (simulate incident)
        incident = await incident_tracker.create_incident(
            vendor_id=vendor_id,
            vendor_name=sample_vendor_profile.name,
            title="Minor Configuration Issue",
            description="Non-critical configuration drift detected",
            category=IncidentCategory.SERVICE_DISRUPTION,
            severity=IncidentSeverity.LOW,
            reporter="monitoring_system@company.com"
        )
        assert incident.vendor_id == vendor_id
        
        # Phase 6: Reporting and Analytics
        assessment_report = await vendor_assessor.generate_assessment_report(assessment)
        scan_report = await vulnerability_scanner.generate_scan_report(scan_result)
        sbom_analytics = await sbom_manager.get_sbom_analytics(sbom.sbom_id)
        access_report = await access_monitor.generate_access_report(vendor_id, 7)
        incident_summary = await incident_tracker.generate_vendor_incident_summary(vendor_id, 30)
        
        # Verify all reports generated successfully
        assert "assessment_summary" in assessment_report
        assert "scan_summary" in scan_report
        assert "summary" in sbom_analytics
        assert "access_summary" in access_report
        assert "incident_summary" in incident_summary
        
        # Phase 7: Compliance Validation
        vendor_capabilities = {req.requirement_id.lower(): True for req in template.security_requirements}
        compliance_results = contract_manager.validate_vendor_compliance(
            template_id=template.template_id,
            vendor_certifications=sample_vendor_profile.compliance_certifications,
            vendor_capabilities=vendor_capabilities
        )
        
        assert "overall_compliance" in compliance_results
        
        # Cleanup
        Path(sample_software_file).unlink(missing_ok=True)
    
    def test_system_integration_health_check(
        self, vendor_assessor, vulnerability_scanner, sbom_manager,
        access_monitor, incident_tracker, contract_manager
    ):
        """Test that all system components are properly integrated"""
        
        # Verify all components are initialized
        assert vendor_assessor is not None
        assert vulnerability_scanner is not None
        assert sbom_manager is not None
        assert access_monitor is not None
        assert incident_tracker is not None
        assert contract_manager is not None
        
        # Verify configuration loading
        assert vendor_assessor.config is not None
        assert vulnerability_scanner.config is not None
        assert sbom_manager.config is not None
        assert access_monitor.config is not None
        assert incident_tracker.config is not None
        assert contract_manager.config is not None
        
        # Verify component libraries/databases are loaded
        assert len(vendor_assessor.assessment_templates) > 0
        assert len(vulnerability_scanner.backdoor_patterns) > 0
        assert len(sbom_manager.requirement_library) > 0
        assert len(access_monitor.anomaly_detector) > 0
        assert len(incident_tracker.incident_workflows) > 0
        assert len(contract_manager.requirement_library) > 0
        
        print("✅ All vendor supply chain security components are properly integrated")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])