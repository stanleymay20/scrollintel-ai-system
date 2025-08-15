"""
Tests for legal compliance features.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from scrollintel.core.legal_compliance_manager import LegalComplianceManager
from scrollintel.models.legal_models import (
    LegalDocument, UserConsent, DataExportRequest, ComplianceAudit,
    CookieSettings, PrivacySettings
)

class TestLegalComplianceManager:
    """Test legal compliance manager functionality."""
    
    @pytest.fixture
    def compliance_manager(self):
        """Create compliance manager instance."""
        return LegalComplianceManager()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('scrollintel.core.legal_compliance_manager.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session
            yield mock_session
    
    @pytest.mark.asyncio
    async def test_get_legal_document(self, compliance_manager, mock_db_session):
        """Test getting legal document."""
        # Mock document
        mock_document = MagicMock()
        mock_document.id = 1
        mock_document.document_type = "terms_of_service"
        mock_document.version = "1.0"
        mock_document.title = "Terms of Service"
        mock_document.content = "<h1>Terms</h1>"
        mock_document.effective_date = datetime.utcnow()
        mock_document.document_metadata = {}
        
        mock_db_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.first.return_value = mock_document
        
        result = await compliance_manager.get_legal_document("terms_of_service")
        
        assert result is not None
        assert result["document_type"] == "terms_of_service"
        assert result["version"] == "1.0"
        assert result["title"] == "Terms of Service"
    
    @pytest.mark.asyncio
    async def test_create_legal_document(self, compliance_manager, mock_db_session):
        """Test creating legal document."""
        mock_document = MagicMock()
        mock_document.id = 1
        mock_document.document_type = "privacy_policy"
        mock_document.version = "1.0"
        mock_document.title = "Privacy Policy"
        mock_document.effective_date = datetime.utcnow()
        
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None
        
        with patch.object(compliance_manager, '_log_compliance_audit') as mock_log:
            result = await compliance_manager.create_legal_document(
                document_type="privacy_policy",
                title="Privacy Policy",
                content="<h1>Privacy</h1>",
                version="1.0",
                effective_date=datetime.utcnow()
            )
            
            assert result["status"] == "created"
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_record_user_consent(self, compliance_manager, mock_db_session):
        """Test recording user consent."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        mock_consent = MagicMock()
        mock_consent.id = 1
        mock_consent.consent_date = datetime.utcnow()
        
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None
        
        with patch.object(compliance_manager, '_log_compliance_audit') as mock_log:
            result = await compliance_manager.record_user_consent(
                user_id="user123",
                consent_type="cookies_analytics",
                consent_given=True,
                ip_address="127.0.0.1"
            )
            
            assert result["status"] == "recorded"
            assert result["consent_type"] == "cookies_analytics"
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_consents(self, compliance_manager, mock_db_session):
        """Test getting user consents."""
        mock_consent = MagicMock()
        mock_consent.id = 1
        mock_consent.consent_type = "cookies_analytics"
        mock_consent.consent_given = True
        mock_consent.consent_date = datetime.utcnow()
        mock_consent.document_version = "1.0"
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_consent]
        
        result = await compliance_manager.get_user_consents("user123")
        
        assert len(result) == 1
        assert result[0]["consent_type"] == "cookies_analytics"
        assert result[0]["consent_given"] is True
    
    @pytest.mark.asyncio
    async def test_request_data_export(self, compliance_manager, mock_db_session):
        """Test requesting data export."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        mock_request = MagicMock()
        mock_request.id = 1
        
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None
        
        with patch.object(compliance_manager, '_log_compliance_audit') as mock_log:
            with patch('asyncio.create_task') as mock_task:
                result = await compliance_manager.request_data_export(
                    user_id="user123",
                    request_type="export"
                )
                
                assert result["status"] == "pending"
                assert result["request_type"] == "export"
                assert "verification_token" in result
                mock_log.assert_called_once()
                mock_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_export_user_data(self, compliance_manager):
        """Test exporting user data."""
        with patch.object(compliance_manager, '_get_user_profile_data') as mock_profile:
            with patch.object(compliance_manager, 'get_user_consents') as mock_consents:
                with patch.object(compliance_manager, '_get_user_activity_logs') as mock_logs:
                    with patch.object(compliance_manager, '_get_user_files') as mock_files:
                        mock_profile.return_value = {"user_id": "user123"}
                        mock_consents.return_value = []
                        mock_logs.return_value = []
                        mock_files.return_value = []
                        
                        result = await compliance_manager._export_user_data("user123")
                        
                        assert result.exists()
                        assert result.suffix == '.zip'
                        
                        # Clean up
                        if result.exists():
                            os.remove(result)
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, compliance_manager, mock_db_session):
        """Test generating compliance report."""
        mock_db_session.query.return_value.filter.return_value.count.return_value = 10
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        result = await compliance_manager.generate_compliance_report(start_date, end_date)
        
        assert result.report_type == "gdpr_compliance"
        assert result.period_start == start_date
        assert result.period_end == end_date
        assert isinstance(result.consent_stats, dict)
        assert result.compliance_score >= 0

class TestLegalRoutes:
    """Test legal compliance API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from scrollintel.api.main import app
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return {"user_id": "user123", "email": "test@example.com"}
    
    def test_get_terms_of_service(self, client):
        """Test getting terms of service."""
        with patch('scrollintel.api.routes.legal_routes.compliance_manager') as mock_manager:
            mock_manager.get_legal_document.return_value = {
                "id": 1,
                "document_type": "terms_of_service",
                "version": "1.0",
                "title": "Terms of Service",
                "content": "<h1>Terms</h1>",
                "effective_date": datetime.utcnow(),
                "is_active": True,
                "document_metadata": {}
            }
            
            response = client.get("/api/legal/terms-of-service")
            assert response.status_code == 200
            data = response.json()
            assert data["document_type"] == "terms_of_service"
    
    def test_get_privacy_policy(self, client):
        """Test getting privacy policy."""
        with patch('scrollintel.api.routes.legal_routes.compliance_manager') as mock_manager:
            mock_manager.get_legal_document.return_value = {
                "id": 1,
                "document_type": "privacy_policy",
                "version": "1.0",
                "title": "Privacy Policy",
                "content": "<h1>Privacy</h1>",
                "effective_date": datetime.utcnow(),
                "is_active": True,
                "document_metadata": {}
            }
            
            response = client.get("/api/legal/privacy-policy")
            assert response.status_code == 200
            data = response.json()
            assert data["document_type"] == "privacy_policy"
    
    def test_record_consent(self, client, mock_user):
        """Test recording user consent."""
        with patch('scrollintel.api.routes.legal_routes.get_current_user') as mock_auth:
            with patch('scrollintel.api.routes.legal_routes.compliance_manager') as mock_manager:
                mock_auth.return_value = mock_user
                mock_manager.record_user_consent.return_value = {
                    "id": 1,
                    "consent_type": "cookies_analytics",
                    "consent_given": True,
                    "consent_date": datetime.utcnow(),
                    "status": "recorded"
                }
                
                response = client.post("/api/legal/consent", json={
                    "consent_type": "cookies_analytics",
                    "consent_given": True
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["consent_type"] == "cookies_analytics"
                assert data["consent_given"] is True
    
    def test_set_cookie_consent(self, client, mock_user):
        """Test setting cookie consent."""
        with patch('scrollintel.api.routes.legal_routes.get_current_user') as mock_auth:
            with patch('scrollintel.api.routes.legal_routes.compliance_manager') as mock_manager:
                mock_auth.return_value = mock_user
                mock_manager.record_user_consent.return_value = {
                    "id": 1,
                    "status": "recorded"
                }
                
                response = client.post("/api/legal/cookie-consent", json={
                    "necessary": True,
                    "analytics": True,
                    "marketing": False,
                    "preferences": True
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
    
    def test_request_data_export(self, client, mock_user):
        """Test requesting data export."""
        with patch('scrollintel.api.routes.legal_routes.get_current_user') as mock_auth:
            with patch('scrollintel.api.routes.legal_routes.compliance_manager') as mock_manager:
                mock_auth.return_value = mock_user
                mock_manager.request_data_export.return_value = {
                    "id": 1,
                    "request_type": "export",
                    "status": "pending",
                    "verification_token": "abc123",
                    "message": "Request submitted successfully"
                }
                
                response = client.post("/api/legal/data-export", json={
                    "request_type": "export",
                    "verification_email": "test@example.com"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["request_type"] == "export"
                assert data["status"] == "pending"

class TestCookieSettings:
    """Test cookie settings functionality."""
    
    def test_cookie_settings_validation(self):
        """Test cookie settings validation."""
        settings = CookieSettings(
            necessary=True,
            analytics=True,
            marketing=False,
            preferences=True
        )
        
        assert settings.necessary is True
        assert settings.analytics is True
        assert settings.marketing is False
        assert settings.preferences is True
    
    def test_privacy_settings_validation(self):
        """Test privacy settings validation."""
        settings = PrivacySettings(
            data_processing_consent=True,
            marketing_emails=False,
            analytics_tracking=True,
            third_party_sharing=False,
            data_retention_period="2_years"
        )
        
        assert settings.data_processing_consent is True
        assert settings.marketing_emails is False
        assert settings.analytics_tracking is True
        assert settings.third_party_sharing is False
        assert settings.data_retention_period == "2_years"

class TestFrontendComponents:
    """Test frontend component functionality."""
    
    def test_cookie_consent_banner_props(self):
        """Test cookie consent banner component props."""
        # This would be a more comprehensive test in a real frontend testing environment
        # For now, we'll just verify the component structure exists
        
        cookie_banner_path = "frontend/src/components/legal/cookie-consent-banner.tsx"
        assert os.path.exists(cookie_banner_path)
        
        with open(cookie_banner_path, 'r') as f:
            content = f.read()
            assert "CookieConsentBanner" in content
            assert "CookieSettings" in content
            assert "onAcceptAll" in content
            assert "onRejectAll" in content
            assert "onCustomize" in content
    
    def test_privacy_settings_component(self):
        """Test privacy settings component."""
        privacy_settings_path = "frontend/src/components/legal/privacy-settings.tsx"
        assert os.path.exists(privacy_settings_path)
        
        with open(privacy_settings_path, 'r') as f:
            content = f.read()
            assert "PrivacySettingsPanel" in content
            assert "DataExportRequest" in content
            assert "requestDataExport" in content
            assert "requestDataDeletion" in content

if __name__ == "__main__":
    pytest.main([__file__])