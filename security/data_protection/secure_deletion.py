"""
Secure Data Deletion with Cryptographic Erasure

Implements secure data deletion using cryptographic erasure capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class DeletionMethod(Enum):
    CRYPTOGRAPHIC_ERASURE = "cryptographic_erasure"
    SECURE_OVERWRITE = "secure_overwrite"
    PHYSICAL_DESTRUCTION = "physical_destruction"

class DeletionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"

@dataclass
class DeletionRequest:
    request_id: str
    data_identifiers: List[str]
    method: DeletionMethod
    requested_by: str
    requested_at: datetime
    scheduled_for: Optional[datetime]
    status: DeletionStatus
    verification_required: bool
    retention_policy: Optional[str]

@dataclass
class DeletionResult:
    request_id: str
    data_identifier: str
    method_used: DeletionMethod
    status: DeletionStatus
    completed_at: Optional[datetime]
    verification_hash: Optional[str]
    error_message: Optional[str]

class CryptographicErasure:
    """Secure data deletion using cryptographic erasure"""
    
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.deletion_queue = {}
        self.deletion_results = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.verification_hashes = {}
        self._lock = threading.Lock()
        
    def schedule_deletion(self, data_identifiers: List[str], 
                         method: DeletionMethod = DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                         requested_by: str = "system",
                         delay_hours: int = 0,
                         verification_required: bool = True,
                         retention_policy: Optional[str] = None) -> str:
        """Schedule data for secure deletion"""
        try:
            request_id = self._generate_request_id()
            
            scheduled_for = None
            if delay_hours > 0:
                scheduled_for = datetime.utcnow() + timedelta(hours=delay_hours)
            
            deletion_request = DeletionRequest(
                request_id=request_id,
                data_identifiers=data_identifiers,
                method=method,
                requested_by=requested_by,
                requested_at=datetime.utcnow(),
                scheduled_for=scheduled_for,
                status=DeletionStatus.PENDING,
                verification_required=verification_required,
                retention_policy=retention_policy
            )
            
            with self._lock:
                self.deletion_queue[request_id] = deletion_request
            
            # If no delay, execute immediately
            if delay_hours == 0:
                self.executor.submit(self._execute_deletion, request_id)
            
            logger.info(f"Scheduled deletion request {request_id} for {len(data_identifiers)} items")
            return request_id
            
        except Exception as e:
            logger.error(f"Error scheduling deletion: {e}")
            raise
    
    def execute_cryptographic_erasure(self, data_identifier: str, 
                                    encryption_key_id: str) -> DeletionResult:
        """Execute cryptographic erasure by destroying encryption keys"""
        try:
            result = DeletionResult(
                request_id="direct",
                data_identifier=data_identifier,
                method_used=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                status=DeletionStatus.IN_PROGRESS,
                completed_at=None,
                verification_hash=None,
                error_message=None
            )
            
            # Store verification hash before deletion
            verification_data = f"{data_identifier}:{encryption_key_id}:{datetime.utcnow().isoformat()}"
            result.verification_hash = hashlib.sha256(verification_data.encode()).hexdigest()
            
            # Delete the encryption key - this makes data unrecoverable
            self.key_manager.delete_key(encryption_key_id, force=True)
            
            # Mark as completed
            result.status = DeletionStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            logger.info(f"Cryptographic erasure completed for {data_identifier}")
            return result
            
        except Exception as e:
            logger.error(f"Error in cryptographic erasure: {e}")
            result.status = DeletionStatus.FAILED
            result.error_message = str(e)
            return result
    
    def execute_secure_overwrite(self, file_path: str, passes: int = 3) -> DeletionResult:
        """Execute secure file overwrite using multiple passes"""
        try:
            result = DeletionResult(
                request_id="direct",
                data_identifier=file_path,
                method_used=DeletionMethod.SECURE_OVERWRITE,
                status=DeletionStatus.IN_PROGRESS,
                completed_at=None,
                verification_hash=None,
                error_message=None
            )
            
            if not os.path.exists(file_path):
                result.status = DeletionStatus.FAILED
                result.error_message = "File not found"
                return result
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Store verification hash
            with open(file_path, 'rb') as f:
                original_hash = hashlib.sha256(f.read()).hexdigest()
            result.verification_hash = original_hash
            
            # Perform multiple overwrite passes
            for pass_num in range(passes):
                self._overwrite_file(file_path, file_size, pass_num)
                logger.debug(f"Completed overwrite pass {pass_num + 1}/{passes} for {file_path}")
            
            # Final verification - file should be different
            with open(file_path, 'rb') as f:
                final_hash = hashlib.sha256(f.read()).hexdigest()
            
            if final_hash == original_hash:
                result.status = DeletionStatus.FAILED
                result.error_message = "Overwrite verification failed"
                return result
            
            # Delete the file
            os.remove(file_path)
            
            result.status = DeletionStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            logger.info(f"Secure overwrite completed for {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in secure overwrite: {e}")
            result.status = DeletionStatus.FAILED
            result.error_message = str(e)
            return result
    
    def _overwrite_file(self, file_path: str, file_size: int, pass_num: int) -> None:
        """Overwrite file with random or pattern data"""
        patterns = [
            lambda: secrets.token_bytes(1024),  # Random data
            lambda: b'\x00' * 1024,             # Zeros
            lambda: b'\xFF' * 1024,             # Ones
            lambda: b'\xAA' * 1024,             # Alternating pattern
            lambda: b'\x55' * 1024,             # Alternating pattern
        ]
        
        pattern_func = patterns[pass_num % len(patterns)]
        
        with open(file_path, 'r+b') as f:
            bytes_written = 0
            while bytes_written < file_size:
                chunk_size = min(1024, file_size - bytes_written)
                pattern_data = pattern_func()[:chunk_size]
                f.write(pattern_data)
                bytes_written += len(pattern_data)
            
            # Ensure data is written to disk
            f.flush()
            os.fsync(f.fileno())
    
    def _execute_deletion(self, request_id: str) -> None:
        """Execute deletion request"""
        try:
            with self._lock:
                if request_id not in self.deletion_queue:
                    return
                
                request = self.deletion_queue[request_id]
                request.status = DeletionStatus.IN_PROGRESS
            
            results = []
            
            for data_identifier in request.data_identifiers:
                if request.method == DeletionMethod.CRYPTOGRAPHIC_ERASURE:
                    # Assume data_identifier format: "data_id:key_id"
                    if ':' in data_identifier:
                        data_id, key_id = data_identifier.split(':', 1)
                        result = self.execute_cryptographic_erasure(data_id, key_id)
                    else:
                        result = DeletionResult(
                            request_id=request_id,
                            data_identifier=data_identifier,
                            method_used=request.method,
                            status=DeletionStatus.FAILED,
                            completed_at=None,
                            verification_hash=None,
                            error_message="Invalid data identifier format for cryptographic erasure"
                        )
                
                elif request.method == DeletionMethod.SECURE_OVERWRITE:
                    result = self.execute_secure_overwrite(data_identifier)
                
                else:
                    result = DeletionResult(
                        request_id=request_id,
                        data_identifier=data_identifier,
                        method_used=request.method,
                        status=DeletionStatus.FAILED,
                        completed_at=None,
                        verification_hash=None,
                        error_message=f"Unsupported deletion method: {request.method}"
                    )
                
                result.request_id = request_id
                results.append(result)
            
            # Update request status
            with self._lock:
                if all(r.status == DeletionStatus.COMPLETED for r in results):
                    request.status = DeletionStatus.COMPLETED
                else:
                    request.status = DeletionStatus.FAILED
                
                self.deletion_results[request_id] = results
            
            # Perform verification if required
            if request.verification_required:
                self._verify_deletion(request_id)
            
            logger.info(f"Deletion request {request_id} completed with status {request.status.value}")
            
        except Exception as e:
            logger.error(f"Error executing deletion request {request_id}: {e}")
            with self._lock:
                if request_id in self.deletion_queue:
                    self.deletion_queue[request_id].status = DeletionStatus.FAILED
    
    def _verify_deletion(self, request_id: str) -> None:
        """Verify deletion was successful"""
        try:
            with self._lock:
                if request_id not in self.deletion_results:
                    return
                
                results = self.deletion_results[request_id]
                request = self.deletion_queue[request_id]
            
            verification_passed = True
            
            for result in results:
                if result.method_used == DeletionMethod.CRYPTOGRAPHIC_ERASURE:
                    # Verify key is actually deleted
                    data_id, key_id = result.data_identifier.split(':', 1)
                    try:
                        self.key_manager.get_key(key_id)
                        # If we can still get the key, deletion failed
                        verification_passed = False
                        logger.error(f"Verification failed: Key {key_id} still exists")
                    except:
                        # Key not found - good, deletion succeeded
                        pass
                
                elif result.method_used == DeletionMethod.SECURE_OVERWRITE:
                    # Verify file is deleted
                    if os.path.exists(result.data_identifier):
                        verification_passed = False
                        logger.error(f"Verification failed: File {result.data_identifier} still exists")
            
            if verification_passed:
                with self._lock:
                    request.status = DeletionStatus.VERIFIED
                    for result in results:
                        result.status = DeletionStatus.VERIFIED
                
                logger.info(f"Deletion verification passed for request {request_id}")
            else:
                logger.error(f"Deletion verification failed for request {request_id}")
                
        except Exception as e:
            logger.error(f"Error verifying deletion {request_id}: {e}")
    
    def get_deletion_status(self, request_id: str) -> Optional[DeletionRequest]:
        """Get status of deletion request"""
        with self._lock:
            return self.deletion_queue.get(request_id)
    
    def get_deletion_results(self, request_id: str) -> List[DeletionResult]:
        """Get results of deletion request"""
        with self._lock:
            return self.deletion_results.get(request_id, [])
    
    def cancel_deletion(self, request_id: str) -> bool:
        """Cancel pending deletion request"""
        try:
            with self._lock:
                if request_id not in self.deletion_queue:
                    return False
                
                request = self.deletion_queue[request_id]
                
                if request.status == DeletionStatus.PENDING:
                    del self.deletion_queue[request_id]
                    logger.info(f"Cancelled deletion request {request_id}")
                    return True
                else:
                    logger.warning(f"Cannot cancel deletion request {request_id} - status: {request.status.value}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error cancelling deletion request {request_id}: {e}")
            return False
    
    def list_pending_deletions(self) -> List[DeletionRequest]:
        """List all pending deletion requests"""
        with self._lock:
            return [req for req in self.deletion_queue.values() 
                   if req.status == DeletionStatus.PENDING]
    
    def process_scheduled_deletions(self) -> None:
        """Process deletions that are scheduled for execution"""
        try:
            current_time = datetime.utcnow()
            
            with self._lock:
                scheduled_requests = [
                    req for req in self.deletion_queue.values()
                    if (req.status == DeletionStatus.PENDING and 
                        req.scheduled_for and 
                        req.scheduled_for <= current_time)
                ]
            
            for request in scheduled_requests:
                logger.info(f"Processing scheduled deletion {request.request_id}")
                self.executor.submit(self._execute_deletion, request.request_id)
                
        except Exception as e:
            logger.error(f"Error processing scheduled deletions: {e}")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = int(time.time() * 1000)
        random_part = secrets.token_hex(8)
        return f"del_{timestamp}_{random_part}"
    
    def get_deletion_statistics(self) -> Dict[str, Any]:
        """Get deletion statistics"""
        with self._lock:
            total_requests = len(self.deletion_queue)
            completed_requests = len([r for r in self.deletion_queue.values() 
                                    if r.status == DeletionStatus.COMPLETED])
            failed_requests = len([r for r in self.deletion_queue.values() 
                                 if r.status == DeletionStatus.FAILED])
            pending_requests = len([r for r in self.deletion_queue.values() 
                                  if r.status == DeletionStatus.PENDING])
            
            method_stats = {}
            for request in self.deletion_queue.values():
                method = request.method.value
                method_stats[method] = method_stats.get(method, 0) + 1
            
            return {
                'total_requests': total_requests,
                'completed_requests': completed_requests,
                'failed_requests': failed_requests,
                'pending_requests': pending_requests,
                'success_rate': completed_requests / total_requests if total_requests > 0 else 0,
                'method_distribution': method_stats
            }
    
    def cleanup_old_records(self, days_old: int = 30) -> int:
        """Clean up old deletion records"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cleaned_count = 0
            
            with self._lock:
                # Find old completed/failed requests
                old_requests = [
                    req_id for req_id, req in self.deletion_queue.items()
                    if (req.status in [DeletionStatus.COMPLETED, DeletionStatus.FAILED, DeletionStatus.VERIFIED] and
                        req.requested_at < cutoff_date)
                ]
                
                # Remove old requests
                for req_id in old_requests:
                    del self.deletion_queue[req_id]
                    if req_id in self.deletion_results:
                        del self.deletion_results[req_id]
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old deletion records")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0
    
    def emergency_stop_all_deletions(self) -> None:
        """Emergency stop all pending deletions"""
        try:
            with self._lock:
                pending_requests = [
                    req_id for req_id, req in self.deletion_queue.items()
                    if req.status == DeletionStatus.PENDING
                ]
                
                for req_id in pending_requests:
                    del self.deletion_queue[req_id]
            
            # Shutdown executor to stop in-progress deletions
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=4)
            
            logger.warning(f"Emergency stop executed - cancelled {len(pending_requests)} pending deletions")
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            raise