"""
Backup System Integration Tests
Tests the complete backup and recovery workflow
"""

import pytest
import asyncio
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock

from scrollintel.core.backup_system import backup_system
from scrollintel.core.backup_scheduler import backup_scheduler

class TestBackupIntegration:
    """Integration tests for backup system"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix="backup_test_"))
        backup_system.backup_dir = temp_dir
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_data_dir(self):
        """Create test data directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_data_"))
        
        # Create test files
        (temp_dir / "file1.txt").write_text("Test content 1")
        (temp_dir / "file2.txt").write_text("Test content 2")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("Test content 3")
        
        yield temp_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_backup_workflow(self, temp_backup_dir, test_data_dir):
        """Test complete backup workflow"""
        # 1. Create file backup
        backup_metadata = await backup_system.create_file_backup([str(test_data_dir)])
        
        assert backup_metadata is not None
        assert backup_metadata["type"] == "files"
        assert backup_metadata["status"] == "completed"
        assert "checksum" in backup_metadata
        
        # 2. Verify backup integrity
        is_valid = await backup_system.verify_backup_integrity(backup_metadata)
        assert is_valid is True
        
        # 3. List backups
        backups = await backup_system.list_backups()
        assert len(backups) >= 1
        assert any(b["filename"] == backup_metadata["filename"] for b in backups)
        
        # 4. Test restore
        restore_dir = Path(tempfile.mkdtemp(prefix="restore_test_"))
        try:
            success = await backup_system.restore_files(backup_metadata, str(restore_dir))
            assert success is True
            
            # Verify restored files
            restored_test_dir = restore_dir / test_data_dir.name
            assert restored_test_dir.exists()
            assert (restored_test_dir / "file1.txt").read_text() == "Test content 1"
            
        finally:
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
    
    @pytest.mark.asyncio
    async def test_backup_metadata_persistence(self, temp_backup_dir, test_data_dir):
        """Test backup metadata persistence"""
        # Create backup
        backup_metadata = await backup_system.create_file_backup([str(test_data_dir)])
        
        # Check metadata file exists
        backup_path = Path(backup_metadata["path"])
        metadata_path = backup_path.with_suffix('.json')
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["filename"] == backup_metadata["filename"]
        assert saved_metadata["checksum"] == backup_metadata["checksum"]
        assert saved_metadata["type"] == backup_metadata["type"]
    
    @pytest.mark.asyncio
    async def test_backup_cleanup(self, temp_backup_dir, test_data_dir):
        """Test backup cleanup functionality"""
        # Create multiple backups with different timestamps
        backup1 = await backup_system.create_file_backup([str(test_data_dir)])
        
        # Simulate old backup by modifying timestamp
        old_timestamp = "20200101_120000"  # Very old timestamp
        backup1["timestamp"] = old_timestamp
        
        # Update metadata file
        backup_path = Path(backup1["path"])
        metadata_path = backup_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(backup1, f)
        
        # Run cleanup with 1 day retention
        cleaned_count = await backup_system.cleanup_old_backups(1)
        
        # Should clean up the old backup
        assert cleaned_count >= 1
        assert not backup_path.exists()
        assert not metadata_path.exists()
    
    @pytest.mark.asyncio
    async def test_scheduler_manual_trigger(self, temp_backup_dir):
        """Test manual backup trigger through scheduler"""
        # Create test directory
        test_dir = Path("test_scheduler_data")
        test_dir.mkdir(exist_ok=True)
        (test_dir / "test.txt").write_text("scheduler test")
        
        try:
            # Trigger manual backup
            result = await backup_scheduler.trigger_manual_backup("files")
            
            assert isinstance(result, dict)
            if "files" in result:
                assert result["files"]["status"] == "completed" or "filename" in result["files"]
        
        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_backup_verification_failure(self, temp_backup_dir, test_data_dir):
        """Test backup verification with corrupted backup"""
        # Create backup
        backup_metadata = await backup_system.create_file_backup([str(test_data_dir)])
        
        # Corrupt the backup file
        backup_path = Path(backup_metadata["path"])
        with open(backup_path, 'ab') as f:
            f.write(b"corrupted data")
        
        # Verification should fail
        is_valid = await backup_system.verify_backup_integrity(backup_metadata)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_cloud_replication_mock(self, temp_backup_dir, test_data_dir):
        """Test cloud replication with mocked S3"""
        # Create backup
        backup_metadata = await backup_system.create_file_backup([str(test_data_dir)])
        
        # Mock S3 client
        mock_s3_client = Mock()
        backup_system.s3_client = mock_s3_client
        backup_system.settings.backup_s3_bucket = "test-bucket"
        
        # Test replication
        result = await backup_system.replicate_to_cloud(backup_metadata)
        
        assert result["cloud_replicated"] is True
        assert result["s3_bucket"] == "test-bucket"
        assert "replicated_at" in result
        
        # Verify S3 upload was called
        assert mock_s3_client.upload_file.call_count == 2  # backup + metadata
    
    @pytest.mark.asyncio
    async def test_point_in_time_recovery_logic(self, temp_backup_dir, test_data_dir):
        """Test point-in-time recovery logic"""
        # Create backup
        backup_metadata = await backup_system.create_file_backup([str(test_data_dir)])
        backup_metadata["type"] = "database"  # Simulate database backup
        
        # Mock list_backups to return our backup
        with patch.object(backup_system, 'list_backups', return_value=[backup_metadata]):
            with patch.object(backup_system, 'restore_database', return_value=True):
                # Test recovery to a time after backup
                backup_time = datetime.fromisoformat(backup_metadata["timestamp"].replace("_", "T"))
                target_time = backup_time.replace(minute=backup_time.minute + 30)
                
                success = await backup_system.point_in_time_recovery(target_time)
                assert success is True
    
    def test_backup_system_health_check(self, temp_backup_dir):
        """Test backup system health check"""
        # Check backup directory
        assert temp_backup_dir.exists()
        assert temp_backup_dir.is_dir()
        
        # Check if we can write to backup directory
        test_file = temp_backup_dir / "health_check.txt"
        test_file.write_text("health check")
        assert test_file.exists()
        test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_backup_error_handling(self, temp_backup_dir):
        """Test backup error handling"""
        # Test with non-existent directory
        try:
            await backup_system.create_file_backup(["/non/existent/directory"])
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "failed" in str(e).lower() or "error" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_backups(self, temp_backup_dir, test_data_dir):
        """Test concurrent backup operations"""
        # Create multiple backup tasks
        tasks = []
        for i in range(3):
            task = backup_system.create_file_backup([str(test_data_dir)])
            tasks.append(task)
        
        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some should succeed
        successful_backups = [r for r in results if isinstance(r, dict) and r.get("status") == "completed"]
        assert len(successful_backups) > 0

if __name__ == "__main__":
    pytest.main([__file__])