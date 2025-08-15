#!/usr/bin/env python3
"""
ScrollIntel-G6 Demo Script
Demonstrates the unbeatable AI system with proof-of-workflow, council deliberation,
cost-aware routing, chaos testing, transparency ledger, and verifier marketplace.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ScrollIntel-G6 components
try:
    from scrollintel.core.proof_of_workflow import (
        create_workflow_attestation, 
        verify_workflow_attestation,
        export_public_verifier_data
    )
    from scrollintel.core.council_of_models import council_deliberation
    from scrollintel.core.cost_aware_router import route_request, get_cache_stats
    from scrollintel.core.chaos_sanctum import run_chaos_experiment, get_chaos_status
    from scrollintel.core.transparency_ledger import (
        add_model_update, 
        add_eval_score, 
        get_public_changelog,
        verify_ledger_integrity
    )
    from scrollintel.core.marketplace_verifiers import (
        run_verification_suite,
        get_marketplace_stats,
        submit_bounty,
        Severity
    )
    
    COMPONENTS_AVAILABLE = True
    logger.info("✅ All ScrollIntel-G6 components loaded successfully")
    
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logger.error(f"❌ Failed to import ScrollIntel-G6 components: {e}")
    logger.info("Running in simulation mode...")


async def demo_proof_of_workflow():
    """Demonstrate Proof-of-Workflow attestation system."""
    print("\n" + "="*60)
    print("🔐 PROOF-OF-WORKFLOW ATTESTATION DEMO")
    print("="*60)
    
    if not COMPONENTS_AVAILABLE:
        print("⚠️  Running in simulation mode")
        return
    
    try:
        # Create a workflow attestation
        print("Creating workflow attestation...")
        attestation = create_workflow_attestation(
            action_type="ai_code_generation",
            agent_id="scroll_cto_agent",
            user_id="demo_user",
            prompt="Generate a Python function to calculate fibonacci numbers",
            tools_used=["code_generator", "syntax_validator"],
            datasets_used=["python_examples"],
            model_version="scroll-core-m-1.0",
            verifier_evidence={
                "syntax_valid": True,
                "performance_score": 0.95,
                "security_scan": "passed"
            },
            content="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        )
        
        print(f"✅ Attestation created: {attestation.id}")
        print(f"   Timestamp: {attestation.timestamp}")
        print(f"   Content hash: {attestation.content_hash[:16]}...")
        print(f"   Signature: {attestation.signature[:32]}..." if attestation.signature else "No signature")
        
        # Verify the attestation
        print("\nVerifying attestation...")
        is_valid = verify_workflow_attestation(attestation)
        print(f"✅ Attestation verification: {'VALID' if is_valid else 'INVALID'}")
        
        # Export public verifier data
        print("\nExporting public verifier data...")
        public_data = export_public_verifier_data()
        print(f"✅ Public key available: {len(public_data['public_key'])} characters")
        print(f"✅ Attestations in chain: {len(public_data['attestations'])}")
        print(f"✅ Chain integrity: {'VALID' if public_data['chain_valid'] else 'INVALID'}")
        
    except Exception as e:
        print(f"❌ Error in PoWf demo: {e}")


async def demo_council_of_models():
    """Demonstrate Council of Models deliberation."""
    print("\n" + "="*60)
    print("🏛️  COUNCIL OF MODELS DELIBERATION DEMO")
    print("="*60)
    
    if not COMPONENTS_AVAILABLE:
        print("⚠️  Running in simulation mode")
        return
    
    try:
        # High-risk task requiring council deliberation
        task = """
        Design a secure authentication system for a financial application.
        Consider multi-factor authentication, session management, and regulatory compliance.
        The system must handle 1M+ users with 99.9% uptime requirements.