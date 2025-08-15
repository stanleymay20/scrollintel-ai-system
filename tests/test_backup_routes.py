"""
Tests for Backup API Routes
Test backup REST API endpoints
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from scrollintel.api.main import app
from scrollintel.core.backup_system import backup_system
from scrollintel.core.backup_scheduler import backup_scheduler

client = TestClient(app)

class TestBackupRoutes:
    """Test backup API routes"""
    
    @pytest.fixture
    def mock_admin_user(self):
        """Mock admin user for authentication"""
        return {"id": 1, "email": "admin@test.com", "role": "admin"}
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test_token"}
    
    def test_list_backups(self, auth_headers):
        """Test listing backups endpoint"""
        mock_backups = [
            {
                "filename": "backup_1.sql",
                "timestamp": "20240101_120000",
                "type": "database",
                "size": 1024
            },
            {
                "filename": "backup_2.tar.gz",
                "timestamp": "20240101_130000",
                "type": "files",
                "size": 2048
            }
        ]
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            with patch.object(backup_system, 'list_backups', return_value=mock_backups):
                response = client.get("/api/v1/backup/list", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["backups"]) == 2
        assert data["total_count"] == 2
    
    def test_trigger_backup(self, auth_headers):
        """Test triggering manual backup"""
        request_data = {"backup_type": "database"}
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            response = client.post(
                "/api/v1/backup/trigger",
                json=request_data,
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["backup_type"] == "database"
        assert "triggered_at" in data
    
    def test_verify_backup(self, auth_headers):
        """Test backup verification endpoint"""
        mock_backup = {
            "filename": "test_backup.sql",
            "timestamp": "20240101_120000",
            "checksum": "test_checksum"
        }
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            with patch.object(backup_system, 'list_backups', return_value=[mock_backup]):
                with patch.object(backup_system, 'verify_backup_integrity', return_value=True):
                    response = client.post(
                        "/api/v1/backup/verify/test_backup.sql",
                        headers=auth_headers
                    )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_valid"] is True
        assert data["backup_id"] == "test_backup.sql"
    
    def test_verify_backup_not_found(self, auth_headers):
        """Test backup verification with non-existent backup"""
        with patch('scrollintel.security.auth.get_current_admin_user'):
            with patch.object(backup_system, 'list_backups', return_value=[]):
                response = client.post(
                    "/api/v1/backup/verify/nonexistent.sql",
                    headers=auth_headers
                )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_restore_database(self, auth_headers):
        """Test database restore endpoint"""
        mock_backup = {
            "filename": "test_backup.sql",
            "timestamp": "20240101_120000",
            "type": "database"
        }
        
        request_data = {"backup_id": "test_backup.sql"}
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            with patch.object(backup_system, 'list_backups', return_value=[mock_backup]):
                response = client.post(
                    "/api/v1/backup/restore/database",
                    json=request_data,
                    headers=auth_headers
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["backup_id"] == "test_backup.sql"
        assert "initiated_at" in data
    
    def test_restore_files(self, auth_headers):
        """Test file restore endpoint"""
        mock_backup = {
            "filename": "test_files.tar.gz",
            "timestamp": "20240101_120000",
            "type": "files"
        }
        
        request_data = {
            "backup_id": "test_files.tar.gz",
            "restore_path": "/tmp/restore"
        }
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            with patch.object(backup_system, 'list_backups', return_value=[mock_backup]):
                response = client.post(
                    "/api/v1/backup/restore/files",
                    json=request_data,
                    headers=auth_headers
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["backup_id"] == "test_files.tar.gz"
        assert data["restore_path"] == "/tmp/restore"
    
    def test_point_in_time_recovery(self, auth_headers):
        """Test point-in-time recovery endpoint"""
        target_time = datetime.now().isoformat()
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            response = client.post(
                f"/api/v1/backup/point-in-time-recovery?target_time={target_time}",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["target_time"] == target_time
    
    def test_point_in_time_recovery_invalid_time(self, auth_headers):
        """Test point-in-time recovery with invalid time format"""
        invalid_time = "invalid-time-format"
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            response = client.post(
                f"/api/v1/backup/point-in-time-recovery?target_time={invalid_time}",
                headers=auth_headers
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "invalid datetime format" in data["detail"].lower()
    
    def test_backup_status(self, auth_headers):
        """Test backup status endpoint"""
        mock_backups = [
            {
                "filename": "backup_1.sql",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "type": "database",
                "size": 1024
            }
        ]
        
        mock_next_backups = {
            "daily_database_backup": datetime.now().isoformat()
        }
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            with patch.object(backup_system, 'list_backups', return_value=mock_backups):
                with patch.object(backup_scheduler, 'get_next_backup_times', return_value=mock_next_backups):
                    with patch.object(backup_scheduler, 'is_running', True):
                        response = client.get("/api/v1/backup/status", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_backups"] == 1
        assert data["scheduler_running"] is True
        assert "next_scheduled_backups" in data
    
    def test_cleanup_old_backups(self, auth_headers):
        """Test backup cleanup endpoint"""
        with patch('scrollintel.security.auth.get_current_admin_user'):
            response = client.delete(
                "/api/v1/backup/cleanup?retention_days=30",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["retention_days"] == 30
        assert "initiated_at" in data
    
    def test_backup_health_check(self):
        """Test backup health check endpoint (no auth required)"""
        mock_backups = [
            {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        ]
        
        with patch.object(backup_system, 'backup_dir') as mock_dir:
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True
            
            with patch.object(backup_system, 'list_backups', return_value=mock_backups):
                with patch.object(backup_scheduler, 'is_running', True):
                    response = client.get("/api/v1/backup/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["healthy"] is True
        assert data["backup_directory_exists"] is True
        assert data["scheduler_running"] is True
    
    def test_backup_health_check_unhealthy(self):
        """Test backup health check when system is unhealthy"""
        with patch.object(backup_system, 'backup_dir') as mock_dir:
            mock_dir.exists.return_value = False
            
            with patch.object(backup_system, 'list_backups', side_effect=Exception("Test error")):
                response = client.get("/api/v1/backup/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["healthy"] is False
        assert "error" in data
    
    def test_unauthorized_access(self):
        """Test unauthorized access to protected endpoints"""
        response = client.get("/api/v1/backup/list")
        assert response.status_code == 401
        
        response = client.post("/api/v1/backup/trigger", json={"backup_type": "database"})
        assert response.status_code == 401
    
    def test_invalid_backup_type(self, auth_headers):
        """Test triggering backup with invalid type"""
        request_data = {"backup_type": "invalid_type"}
        
        with patch('scrollintel.security.auth.get_current_admin_user'):
            response = client.post(
                "/api/v1/backup/trigger",
                json=request_data,
                headers=auth_headers
            )
        
        # Should still accept the request but the background task will handle validation
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])