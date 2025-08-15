"""
Tests for Backup System
Comprehensive tests for backup and recovery functionality
"""

import pytest
import asyncio
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.backup_system import BackupSystem, backup_system
from scrollintel.core.backup_scheduler import BackupScheduler, backup_scheduler

class TestBackupSystem:
    """Test backup system functionality"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_backup_system(self, temp_backup_dir):
        """Create backup system with temporary directory"""
        backup_sys = BackupSystem()
        backup_sys.backup_dir = temp_backup_dir
        return backup_sys
    
    @pytest.mark.asyncio
    async def test_database_backup_creation(self, mock_backup_system):
        """Test database backup creation"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful pg_dump
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process
            
            # Create a dummy backup file
            backup_file = mock_backup_system.backup_dir / "test_backup.sql"
            backup_file.write_text("-- Test backup content")
            
            with patch.object(mock_backup_system, '_calculate_file_checksum', return_value='test_checksum'):
                metadata = await mock_backup_system.create_database_backup()
            
            assert metadata['type'] == 'database'
            assert metadata['status'] == 'completed'
            assert metadata['checksum'] == 'test_checksum'
            assert 'timestamp' in metadata
    
    @pytest.mark.asyncio
    async def test_file_backup_creation(self, mock_backup_system):
        """Test file backup creation"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful tar command
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process
            
            # Create a dummy backup file
            backup_file = mock_backup_system.backup_dir / "test_files_backup.tar.gz"
            backup_file.write_bytes(b'test archive content')
            
            with patch.object(mock_backup_system, '_calculate_file_checksum', return_value='test_checksum'):
                metadata = await mock_backup_system.create_file_backup(['test_dir'])
            
            assert metadata['type'] == 'files'
            assert metadata['status'] == 'completed'
            assert metadata['directories'] == ['test_dir']
            assert metadata['checksum'] == 'test_checksum'
    
    @pytest.mark.asyncio
    async def test_backup_integrity_verification(self, mock_backup_system):
        """Test backup integrity verification"""
        # Create test backup file
        backup_file = mock_backup_system.backup_dir / "test_backup.sql"
        backup_file.write_text("test content")
        
        # Create metadata with correct checksum
        with patch.object(mock_backup_system, '_calculate_file_checksum', return_value='correct_checksum'):
            correct_checksum = await mock_backup_system._calculate_file_checksum(backup_file)
        
        metadata = {
            "path": str(backup_file),
            "checksum": correct_checksum
        }
        
        # Test with correct checksum
        with patch.object(mock_backup_system, '_calculate_file_checksum', return_value=correct_checksum):
            is_valid = await mock_backup_system.verify_backup_integrity(metadata)
            assert is_valid is True
        
        # Test with incorrect checksum
        with patch.object(mock_backup_system, '_calculate_file_checksum', return_value='wrong_checksum'):
            is_valid = await mock_backup_system.verify_backup_integrity(metadata)
            assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_cloud_replication(self, mock_backup_system):
        """Test cloud backup replication"""
        # Create test backup file
        backup_file = mock_backup_system.backup_dir / "test_backup.sql"
        backup_file.write_text("test content")
        
        # Create metadata file
        metadata_file = backup_file.with_suffix('.json')
        metadata = {
            "filename": "test_backup.sql",
            "path": str(backup_file)
        }
        metadata_file.write_text(json.dumps(metadata))
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_backup_system.s3_client = mock_s3_client
        mock_backup_system.settings.backup_s3_bucket = "test-bucket"
        
        result = await mock_backup_system.replicate_to_cloud(metadata)
        
        assert result["cloud_replicated"] is True
        assert result["s3_bucket"] == "test-bucket"
        assert "replicated_at" in result
        
        # Verify S3 upload calls
        assert mock_s3_client.upload_file.call_count == 2  # backup file + metadata
    
    @pytest.mark.asyncio
    async def test_database_restore(self, mock_backup_system):
        """Test database restore"""
        # Create test backup file
        backup_file = mock_backup_system.backup_dir / "test_backup.sql"
        backup_file.write_text("-- Test restore content")
        
        metadata = {
            "path": str(backup_file),
            "checksum": "test_checksum"
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful psql command
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process
            
            with patch.object(mock_backup_system, 'verify_backup_integrity', return_value=True):
                success = await mock_backup_system.restore_database(metadata)
            
            assert success is True
            mock_subprocess.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_restore(self, mock_backup_system):
        """Test file restore"""
        # Create test backup file
        backup_file = mock_backup_system.backup_dir / "test_files_backup.tar.gz"
        backup_file.write_bytes(b'test archive')
        
        metadata = {
            "path": str(backup_file),
            "checksum": "test_checksum"
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful tar command
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process
            
            with patch.object(mock_backup_system, 'verify_backup_integrity', return_value=True):
                success = await mock_backup_system.restore_files(metadata, "/tmp/restore")
            
            assert success is True
            mock_subprocess.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_point_in_time_recovery(self, mock_backup_system):
        """Test point-in-time recovery"""
        target_time = datetime.now() - timedelta(hours=1)
        
        # Mock backup list with suitable backup
        backup_metadata = {
            "timestamp": (target_time - timedelta(minutes=30)).strftime("%Y%m%d_%H%M%S"),
            "type": "database",
            "path": "test_backup.sql",
            "checksum": "test_checksum"
        }
        
        with patch.object(mock_backup_system, 'list_backups', return_value=[backup_metadata]):
            with patch.object(mock_backup_system, 'restore_database', return_value=True):
                success = await mock_backup_system.point_in_time_recovery(target_time)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_backup_listing(self, mock_backup_system):
        """Test backup listing"""
        # Create test metadata files
        for i in range(3):
            metadata = {
                "filename": f"backup_{i}.sql",
                "timestamp": f"20240101_0{i}0000",
                "type": "database"
            }
            metadata_file = mock_backup_system.backup_dir / f"backup_{i}.json"
            metadata_file.write_text(json.dumps(metadata))
        
        backups = await mock_backup_system.list_backups()
        
        assert len(backups) == 3
        # Should be sorted by timestamp (newest first)
        assert backups[0]["timestamp"] > backups[1]["timestamp"]
    
    @pytest.mark.asyncio
    async def test_backup_cleanup(self, mock_backup_system):
        """Test old backup cleanup"""
        # Create old and new backup files
        old_time = datetime.now() - timedelta(days=35)
        new_time = datetime.now() - timedelta(days=5)
        
        # Old backup
        old_backup = mock_backup_system.backup_dir / "old_backup.sql"
        old_backup.write_text("old backup")
        old_metadata = {
            "filename": "old_backup.sql",
            "path": str(old_backup),
            "timestamp": old_time.strftime("%Y%m%d_%H%M%S")
        }
        old_metadata_file = old_backup.with_suffix('.json')
        old_metadata_file.write_text(json.dumps(old_metadata))
        
        # New backup
        new_backup = mock_backup_system.backup_dir / "new_backup.sql"
        new_backup.write_text("new backup")
        new_metadata = {
            "filename": "new_backup.sql",
            "path": str(new_backup),
            "timestamp": new_time.strftime("%Y%m%d_%H%M%S")
        }
        new_metadata_file = new_backup.with_suffix('.json')
        new_metadata_file.write_text(json.dumps(new_metadata))
        
        # Run cleanup with 30 day retention
        cleaned_count = await mock_backup_system.cleanup_old_backups(30)
        
        assert cleaned_count == 1
        assert not old_backup.exists()
        assert not old_metadata_file.exists()
        assert new_backup.exists()
        assert new_metadata_file.exists()

class TestBackupScheduler:
    """Test backup scheduler functionality"""
    
    @pytest.fixture
    def mock_scheduler(self):
        """Create backup scheduler for testing"""
        scheduler = BackupScheduler()
        return scheduler
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, mock_scheduler):
        """Test scheduler start and stop"""
        assert not mock_scheduler.is_running
        
        await mock_scheduler.start()
        assert mock_scheduler.is_running
        
        await mock_scheduler.stop()
        assert not mock_scheduler.is_running
    
    @pytest.mark.asyncio
    async def test_manual_backup_trigger(self, mock_scheduler):
        """Test manual backup trigger"""
        with patch.object(backup_system, 'create_database_backup') as mock_db_backup:
            with patch.object(backup_system, 'verify_backup_integrity', return_value=True):
                with patch.object(backup_system, 'replicate_to_cloud') as mock_replicate:
                    mock_db_backup.return_value = {"type": "database", "checksum": "test"}
                    mock_replicate.return_value = {"cloud_replicated": True}
                    
                    result = await mock_scheduler.trigger_manual_backup("database")
                    
                    assert "database" in result
                    mock_db_backup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daily_database_backup_job(self, mock_scheduler):
        """Test daily database backup job"""
        with patch.object(backup_system, 'create_database_backup') as mock_backup:
            with patch.object(backup_system, 'verify_backup_integrity', return_value=True):
                with patch.object(backup_system, 'replicate_to_cloud') as mock_replicate:
                    mock_backup.return_value = {"type": "database", "checksum": "test"}
                    mock_replicate.return_value = {"cloud_replicated": True}
                    
                    await mock_scheduler.run_daily_database_backup()
                    
                    mock_backup.assert_called_once()
                    mock_replicate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_backup_verification_job(self, mock_scheduler):
        """Test backup verification job"""
        # Mock recent backups
        recent_backup = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "filename": "test_backup.sql"
        }
        
        with patch.object(backup_system, 'list_backups', return_value=[recent_backup]):
            with patch.object(backup_system, 'verify_backup_integrity', return_value=True):
                await mock_scheduler.run_backup_verification()
                
                # Should not raise any exceptions
    
    def test_next_backup_times(self, mock_scheduler):
        """Test getting next backup times"""
        # Mock scheduler with jobs
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.next_run_time = datetime.now()
        
        mock_scheduler.scheduler = Mock()
        mock_scheduler.scheduler.get_jobs.return_value = [mock_job]
        
        times = mock_scheduler.get_next_backup_times()
        
        assert "test_job" in times
        assert isinstance(times["test_job"], str)

if __name__ == "__main__":
    pytest.main([__file__])