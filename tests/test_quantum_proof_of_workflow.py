"""
Comprehensive Tests for Quantum-Secured Proof-of-Workflow System

This test suite validates:
- Post-quantum cryptographic signing
- Zero-knowledge proofs for privacy-preserving attestations
- Quantum-resistant hash chaining with lattice-based cryptography
- Homomorphic encryption for computation on encrypted attestations
- Blockchain-based immutable storage with smart contract validation
- Quantum key distribution for ultra-secure attestation transmission
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.engines.quantum_proof_engine import (
    QuantumProofOfWorkflowEngine,
    QuantumProofAttestation,
    ZeroKnowledgeProof,
    QuantumKeyPair
)
from scrollintel.models.quantum_models import (
    QuantumAttestationModel,
    ZeroKnowledgeProofModel,
    QuantumKeyPairModel,
    CreateAttestationRequest,
    VerifyAttestationRequest
)

class TestQuantumProofOfWorkflowEngine:
    """Test suite for the quantum proof-of-workflow engine"""
    
    @pytest.fixture
    def quantum_engine(self):
        """Create a quantum engine instance for testing"""
        return QuantumProofOfWorkflowEngine()
    
    @pytest.fixture
    def sample_workflow_data(self):
        """Sample workflow data for testing"""
        return {
            "agent_id": "test_agent_001",
            "user_id": "test_user_001",
            "workflow_type": "data_analysis",
            "inputs": {
                "dataset": "customer_data.csv",
                "analysis_type": "predictive_modeling"
            },
            "outputs": {
                "model_accuracy": 0.95,
                "predictions": ["high_value", "medium_value", "low_value"]
            },
            "execution_time": 45.2,
            "resources_used": {
                "cpu_hours": 2.5,
                "memory_gb": 8.0,
                "gpu_hours": 1.2
            }
        }
    
    def test_quantum_keypair_generation(self, quantum_engine):
        """Test post-quantum cryptographic key pair generation"""
        # Test key pair properties
        assert quantum_engine.quantum_keys is not None
        assert quantum_engine.quantum_keys.public_key is not None
        assert quantum_engine.quantum_keys.private_key is not None
        assert quantum_engine.quantum_keys.algorithm == "lattice_based_rsa4096"
        assert quantum_engine.quantum_keys.key_size == 4096
        
        # Test key format (should be base64 encoded)
        import base64
        try:
            base64.b64decode(quantum_engine.quantum_keys.public_key)
            base64.b64decode(quantum_engine.quantum_keys.private_key)
        except Exception:
            pytest.fail("Keys should be valid base64 encoded strings")
    
    def test_zero_knowledge_proof_creation_and_verification(self, quantum_engine):
        """Test zero-knowledge proof creation and verification"""
        secret = "my_secret_workflow_data"
        public_statement = "workflow_completed_successfully"
        
        # Create zero-knowledge proof
        zk_proof = quantum_engine.create_zero_knowledge_proof(secret, public_statement)
        
        # Validate proof structure
        assert isinstance(zk_proof, ZeroKnowledgeProof)
        assert zk_proof.commitment is not None
        assert zk_proof.challenge is not None
        assert zk_proof.response is not None
        assert zk_proof.public_parameters is not None
        assert zk_proof.proof_type == "schnorr"
        
        # Verify the proof
        is_valid = quantum_engine.verify_zero_knowledge_proof(zk_proof, public_statement)
        assert is_valid is True
        
        # Test with wrong public statement (should fail)
        is_valid_wrong = quantum_engine.verify_zero_knowledge_proof(zk_proof, "wrong_statement")
        assert is_valid_wrong is False
    
    def test_quantum_resistant_hash_chaining(self, quantum_engine):
        """Test quantum-resistant hash chaining with lattice-based cryptography"""
        # Test initial hash
        data1 = "first_workflow_data"
        hash1 = quantum_engine.quantum_resistant_hash(data1)
        
        assert hash1 is not None
        assert len(hash1) == 128  # SHA3-512 produces 128 hex characters
        assert hash1 in quantum_engine.hash_chain
        
        # Test chained hash
        data2 = "second_workflow_data"
        hash2 = quantum_engine.quantum_resistant_hash(data2, hash1)
        
        assert hash2 is not None
        assert hash2 != hash1
        assert hash2 in quantum_engine.hash_chain
        assert len(quantum_engine.hash_chain) == 2
        
        # Test hash chain integrity
        assert quantum_engine.hash_chain[0] == hash1
        assert quantum_engine.hash_chain[1] == hash2
    
    def test_homomorphic_encryption(self, quantum_engine, sample_workflow_data):
        """Test homomorphic encryption for computation on encrypted attestations"""
        # Encrypt workflow data
        encrypted_data = quantum_engine.homomorphic_encrypt(sample_workflow_data)
        
        assert encrypted_data is not None
        assert isinstance(encrypted_data, str)
        
        # Decode and validate structure
        import base64
        try:
            decoded_data = base64.b64decode(encrypted_data)
            encrypted_package = json.loads(decoded_data.decode())
            
            assert "encrypted_data" in encrypted_package
            assert "key_hash" in encrypted_package
            assert "algorithm" in encrypted_package
            assert "timestamp" in encrypted_package
            assert encrypted_package["algorithm"] == "additive_homomorphic"
            
        except Exception as e:
            pytest.fail(f"Homomorphic encryption should produce valid encrypted package: {e}")
    
    def test_quantum_signing_and_verification(self, quantum_engine):
        """Test post-quantum cryptographic signing and verification"""
        test_data = "important_workflow_attestation_data"
        
        # Sign data
        signature = quantum_engine.quantum_sign(test_data)
        
        assert signature is not None
        assert isinstance(signature, str)
        
        # Verify signature with correct data
        is_valid = quantum_engine.verify_quantum_signature(test_data, signature)
        assert is_valid is True
        
        # Verify signature with incorrect data (should fail)
        is_valid_wrong = quantum_engine.verify_quantum_signature("wrong_data", signature)
        assert is_valid_wrong is False
        
        # Test with different public key (should fail)
        other_engine = QuantumProofOfWorkflowEngine()
        is_valid_other_key = quantum_engine.verify_quantum_signature(
            test_data, signature, other_engine.quantum_keys.public_key
        )
        assert is_valid_other_key is False
    
    def test_blockchain_entry_creation(self, quantum_engine, sample_workflow_data):
        """Test blockchain-based immutable storage"""
        # Create blockchain entry
        blockchain_hash = quantum_engine.create_blockchain_entry(sample_workflow_data)
        
        assert blockchain_hash is not None
        assert isinstance(blockchain_hash, str)
        assert len(blockchain_hash) == 64  # SHA3-256 produces 64 hex characters
        
        # Test multiple entries create different hashes
        blockchain_hash2 = quantum_engine.create_blockchain_entry({"different": "data"})
        assert blockchain_hash2 != blockchain_hash
    
    def test_quantum_key_distribution(self, quantum_engine):
        """Test quantum key distribution for ultra-secure attestation transmission"""
        recipient_id = "test_recipient_001"
        
        # Perform quantum key distribution
        qkd_result = quantum_engine.quantum_key_distribution(recipient_id)
        
        assert qkd_result is not None
        assert "recipient_id" in qkd_result
        assert "shared_secret_hash" in qkd_result
        assert "key_length" in qkd_result
        assert "protocol" in qkd_result
        assert "timestamp" in qkd_result
        
        assert qkd_result["recipient_id"] == recipient_id
        assert qkd_result["key_length"] == 256
        assert qkd_result["protocol"] == "bb84_simulation"
        
        # Test different recipients get different keys
        qkd_result2 = quantum_engine.quantum_key_distribution("test_recipient_002")
        assert qkd_result2["shared_secret_hash"] != qkd_result["shared_secret_hash"]
    
    def test_complete_quantum_attestation_workflow(self, quantum_engine, sample_workflow_data):
        """Test complete quantum attestation creation and verification workflow"""
        workflow_id = "test_workflow_001"
        secret_data = "confidential_workflow_secret"
        
        # Create quantum attestation
        attestation = quantum_engine.create_quantum_attestation(
            workflow_id=workflow_id,
            workflow_data=sample_workflow_data,
            secret_data=secret_data
        )
        
        # Validate attestation structure
        assert isinstance(attestation, QuantumProofAttestation)
        assert attestation.id is not None
        assert attestation.workflow_id == workflow_id
        assert attestation.timestamp is not None
        assert attestation.hash_chain is not None
        assert attestation.zk_proof is not None
        assert attestation.quantum_signature is not None
        assert attestation.homomorphic_data is not None
        assert attestation.blockchain_hash is not None
        assert attestation.verification_metadata is not None
        
        # Verify the attestation
        public_statement = f"workflow_{workflow_id}_completed"
        is_valid = quantum_engine.verify_quantum_attestation(attestation, public_statement)
        assert is_valid is True
        
        # Test verification with wrong statement (should fail)
        is_valid_wrong = quantum_engine.verify_quantum_attestation(attestation, "wrong_statement")
        assert is_valid_wrong is False
    
    def test_attestation_stats(self, quantum_engine, sample_workflow_data):
        """Test attestation statistics collection"""
        # Get initial stats
        initial_stats = quantum_engine.get_attestation_stats()
        initial_count = initial_stats["total_attestations"]
        
        # Create some attestations
        for i in range(3):
            quantum_engine.create_quantum_attestation(
                workflow_id=f"test_workflow_{i}",
                workflow_data=sample_workflow_data
            )
        
        # Get updated stats
        updated_stats = quantum_engine.get_attestation_stats()
        
        assert updated_stats["total_attestations"] == initial_count + 3
        assert updated_stats["quantum_key_algorithm"] == "lattice_based_rsa4096"
        assert updated_stats["key_size"] == 4096
        assert updated_stats["hash_chain_length"] == initial_count + 3
        assert updated_stats["zk_proof_system"] == "schnorr_sigma"
        assert updated_stats["last_attestation"] is not None
    
    def test_performance_benchmarks(self, quantum_engine, sample_workflow_data):
        """Test performance benchmarks for quantum operations"""
        import time
        
        # Benchmark attestation creation
        start_time = time.time()
        attestation = quantum_engine.create_quantum_attestation(
            workflow_id="performance_test",
            workflow_data=sample_workflow_data
        )
        creation_time = time.time() - start_time
        
        # Should complete within reasonable time (< 1 second for test)
        assert creation_time < 1.0
        
        # Benchmark attestation verification
        start_time = time.time()
        is_valid = quantum_engine.verify_quantum_attestation(
            attestation,
            "workflow_performance_test_completed"
        )
        verification_time = time.time() - start_time
        
        # Should complete within reasonable time (< 0.5 seconds for test)
        assert verification_time < 0.5
        assert is_valid is True
    
    def test_error_handling(self, quantum_engine):
        """Test error handling in quantum operations"""
        # Test invalid zero-knowledge proof verification
        invalid_proof = ZeroKnowledgeProof(
            commitment="invalid",
            challenge="invalid",
            response="invalid",
            public_parameters={},
            proof_type="schnorr"
        )
        
        is_valid = quantum_engine.verify_zero_knowledge_proof(invalid_proof, "test")
        assert is_valid is False
        
        # Test quantum signature verification with invalid signature
        is_valid_sig = quantum_engine.verify_quantum_signature("test", "invalid_signature")
        assert is_valid_sig is False
    
    def test_security_properties(self, quantum_engine, sample_workflow_data):
        """Test security properties of quantum attestations"""
        # Create two attestations with same data
        attestation1 = quantum_engine.create_quantum_attestation(
            workflow_id="security_test_1",
            workflow_data=sample_workflow_data
        )
        
        attestation2 = quantum_engine.create_quantum_attestation(
            workflow_id="security_test_2",
            workflow_data=sample_workflow_data
        )
        
        # Attestations should be different even with same input data
        assert attestation1.id != attestation2.id
        assert attestation1.hash_chain != attestation2.hash_chain
        assert attestation1.quantum_signature != attestation2.quantum_signature
        assert attestation1.blockchain_hash != attestation2.blockchain_hash
        
        # Each should verify independently
        assert quantum_engine.verify_quantum_attestation(
            attestation1, "workflow_security_test_1_completed"
        ) is True
        assert quantum_engine.verify_quantum_attestation(
            attestation2, "workflow_security_test_2_completed"
        ) is True

class TestQuantumProofAPI:
    """Test suite for quantum proof API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from scrollintel.api.routes.quantum_proof_routes import router
        
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test_token"}
    
    def test_create_attestation_endpoint(self, client, auth_headers):
        """Test attestation creation endpoint"""
        request_data = {
            "workflow_id": "api_test_workflow",
            "workflow_data": {
                "agent": "test_agent",
                "result": "success"
            },
            "secret_data": "test_secret",
            "compliance_tags": ["gdpr", "hipaa"],
            "retention_policy": "7_years"
        }
        
        response = client.post(
            "/api/v1/quantum-proof/attestations",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["workflow_id"] == "api_test_workflow"
        assert "quantum_signature" in data
        assert "hash_chain_entry" in data
    
    def test_quantum_key_distribution_endpoint(self, client, auth_headers):
        """Test quantum key distribution endpoint"""
        request_data = {
            "recipient_id": "test_recipient",
            "key_length": 256,
            "protocol": "bb84",
            "expires_in_hours": 24
        }
        
        response = client.post(
            "/api/v1/quantum-proof/quantum-keys/distribute",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["recipient_id"] == "test_recipient"
        assert "shared_secret_hash" in data
        assert data["protocol"] == "bb84"
    
    def test_zero_knowledge_proof_endpoints(self, client, auth_headers):
        """Test zero-knowledge proof creation and verification endpoints"""
        # Create ZK proof
        create_response = client.post(
            "/api/v1/quantum-proof/zero-knowledge/prove",
            params={
                "secret": "test_secret",
                "public_statement": "test_statement"
            },
            headers=auth_headers
        )
        
        assert create_response.status_code == 200
        proof_data = create_response.json()
        assert "commitment" in proof_data
        assert "challenge" in proof_data
        assert "response" in proof_data
        
        # Verify ZK proof
        verify_request = {
            "public_statement": "test_statement",
            "commitment": proof_data["commitment"],
            "challenge": proof_data["challenge"],
            "response": proof_data["response"],
            "public_parameters": proof_data["public_parameters"]
        }
        
        verify_response = client.post(
            f"/api/v1/quantum-proof/zero-knowledge/{proof_data['id']}/verify",
            json=verify_request,
            headers=auth_headers
        )
        
        assert verify_response.status_code == 200
        verify_data = verify_response.json()
        assert verify_data["valid"] is True
    
    def test_stats_endpoint(self, client, auth_headers):
        """Test attestation statistics endpoint"""
        response = client.get(
            "/api/v1/quantum-proof/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_attestations" in data
        assert "quantum_key_pairs" in data
        assert "hash_chain_length" in data
    
    def test_health_check_endpoint(self, client):
        """Test quantum system health check endpoint"""
        response = client.get("/api/v1/quantum-proof/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "quantum_engine" in data
        assert "key_algorithm" in data
    
    def test_test_workflow_endpoint(self, client, auth_headers):
        """Test the test workflow endpoint"""
        test_data = {
            "test_workflow": True,
            "data": "sample_data"
        }
        
        response = client.post(
            "/api/v1/quantum-proof/test/create-and-verify",
            json=test_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["created"] is True
        assert data["verified"] is True
        assert "attestation_id" in data
        assert "quantum_signature" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])