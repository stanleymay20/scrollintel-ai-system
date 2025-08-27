"""
Demo script for ScrollIntel FederatedEngine
Demonstrates distributed learning capabilities with PySyft integration,
TensorFlow Federated support, differential privacy, secure aggregation,
and edge device simulation.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_federated_engine():
    """Comprehensive demo of federated learning capabilities"""
    
    print("🚀 ScrollIntel FederatedEngine Demo")
    print("=" * 50)
    
    try:
        # Import the federated engine
        from scrollintel.engines.federated_engine import (
            get_federated_engine, 
            initialize_federated_engine,
            EdgeDeviceType,
            PrivacyLevel
        )
        
        # Initialize the engine
        print("\n1. Initializing FederatedEngine...")
        engine = await initialize_federated_engine()
        print("✅ FederatedEngine initialized successfully")
        
        # Check framework support
        print("\n2. Checking framework support...")
        from scrollintel.engines.federated_engine import HAS_TORCH, HAS_TENSORFLOW, HAS_SYFT, HAS_TFF
        
        frameworks = {
            "PyTorch": HAS_TORCH,
            "TensorFlow": HAS_TENSORFLOW,
            "PySyft": HAS_SYFT,
            "TensorFlow Federated": HAS_TFF
        }
        
        for framework, available in frameworks.items():
            status = "✅ Available" if available else "❌ Not available"
            print(f"   {framework}: {status}")
        
        # Demo 1: Edge Device Simulation
        print("\n3. Creating simulated edge devices...")
        
        device_configs = [
            {
                "device_name": "Mobile_Device_1",
                "device_type": EdgeDeviceType.MOBILE,
                "data_size": 500,
                "compute_power": 0.5,
                "bandwidth": 5.0,
                "battery_level": 85.0,
                "privacy_level": PrivacyLevel.HIGH
            },
            {
                "device_name": "Desktop_Workstation",
                "device_type": EdgeDeviceType.DESKTOP,
                "data_size": 2000,
                "compute_power": 2.0,
                "bandwidth": 50.0,
                "privacy_level": PrivacyLevel.MEDIUM
            },
            {
                "device_name": "IoT_Sensor_Hub",
                "device_type": EdgeDeviceType.IOT,
                "data_size": 100,
                "compute_power": 0.2,
                "bandwidth": 1.0,
                "privacy_level": PrivacyLevel.MAXIMUM
            },
            {
                "device_name": "Cloud_Server",
                "device_type": EdgeDeviceType.CLOUD,
                "data_size": 5000,
                "compute_power": 10.0,
                "bandwidth": 1000.0,
                "privacy_level": PrivacyLevel.LOW
            }
        ]
        
        device_ids = []
        for config in device_configs:
            device_id = await engine.add_edge_device(config)
            device_ids.append(device_id)
            print(f"   ✅ Created {config['device_name']} (ID: {device_id[:8]}...)")
        
        # Demo 2: Differential Privacy Engine
        print("\n4. Testing Differential Privacy Engine...")
        
        # Test gradient clipping and noise addition
        test_gradients = np.random.normal(0, 1, (100, 10))
        print(f"   Original gradients shape: {test_gradients.shape}")
        print(f"   Original gradients norm: {np.linalg.norm(test_gradients):.4f}")
        
        # Clip gradients
        clipped_gradients = engine.privacy_engine.clip_gradients(test_gradients, clip_norm=1.0)
        print(f"   Clipped gradients norm: {np.linalg.norm(clipped_gradients):.4f}")
        
        # Add differential privacy noise
        noisy_gradients = engine.privacy_engine.add_noise(clipped_gradients)
        print(f"   Noisy gradients norm: {np.linalg.norm(noisy_gradients):.4f}")
        print(f"   Privacy budget used: {engine.privacy_engine.privacy_budget_used:.4f}")
        
        # Demo 3: Secure Aggregation Protocol
        print("\n5. Testing Secure Aggregation Protocol...")
        
        # Generate secret shares
        secret_value = 42.0
        num_parties = len(device_ids)
        shares = engine.secure_aggregation.generate_secret_shares(secret_value, num_parties)
        print(f"   Generated {len(shares)} secret shares for value {secret_value}")
        
        # Reconstruct secret
        reconstructed = engine.secure_aggregation.reconstruct_secret(shares[:3])  # Use minimum threshold
        print(f"   Reconstructed value: {reconstructed:.4f}")
        print(f"   Reconstruction error: {abs(secret_value - reconstructed):.6f}")
        
        # Demo 4: Create Federated Learning Task
        print("\n6. Creating federated learning task...")
        
        task_config = {
            "task_name": "MNIST_Classification_Demo",
            "model_architecture": {
                "framework": "pytorch",
                "input_size": 784,
                "hidden_size": 128,
                "output_size": 10,
                "num_classes": 10
            },
            "training_config": {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.01
            },
            "privacy_config": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "sensitivity": 1.0,
                "mechanism": "gaussian"
            },
            "participating_devices": device_ids,
            "target_rounds": 5,
            "convergence_threshold": 0.01
        }
        
        task_id = await engine.create_federated_task(task_config)
        print(f"   ✅ Created task: {task_config['task_name']} (ID: {task_id[:8]}...)")
        
        # Demo 5: Run Custom Federated Training
        print("\n7. Starting federated training (custom implementation)...")
        
        training_success = await engine.start_federated_training(task_id, "custom")
        
        if training_success:
            print("   ✅ Federated training completed successfully")
            
            # Get final task status
            final_status = await engine.get_task_status(task_id)
            if final_status:
                print(f"   📊 Rounds completed: {final_status['rounds_completed']}")
                print(f"   📊 Final status: {final_status['status']}")
        else:
            print("   ❌ Federated training failed")
        
        # Demo 6: Device Status and Monitoring
        print("\n8. Checking device status after training...")
        
        for i, device_id in enumerate(device_ids):
            device_status = engine.device_simulator.get_device_status(device_id)
            if device_status:
                print(f"   Device {i+1}: {device_status['device_name']}")
                print(f"      Status: {device_status['status']}")
                print(f"      Model Version: {device_status['model_version']}")
                if device_status['battery_level'] is not None:
                    print(f"      Battery: {device_status['battery_level']:.1f}%")
        
        # Demo 7: Simulate Device Failures
        print("\n9. Simulating device failures...")
        
        # Simulate network failure on mobile device
        mobile_device_id = device_ids[0]
        engine.device_simulator.simulate_device_failure(mobile_device_id, "network")
        print(f"   📱 Simulated network failure on mobile device")
        
        # Simulate battery failure on mobile device
        engine.device_simulator.simulate_device_failure(mobile_device_id, "battery")
        print(f"   🔋 Simulated battery failure on mobile device")
        
        # Check updated status
        updated_status = engine.device_simulator.get_device_status(mobile_device_id)
        if updated_status:
            print(f"   Updated status: {updated_status['status']}")
            print(f"   Battery level: {updated_status['battery_level']}")
        
        # Demo 8: PySyft Integration (if available)
        if HAS_SYFT:
            print("\n10. Testing PySyft integration...")
            
            # Create virtual workers
            worker_ids = ["worker_1", "worker_2", "worker_3"]
            for worker_id in worker_ids:
                worker = engine.pysyft_integration.create_virtual_worker(worker_id)
                if worker:
                    print(f"   ✅ Created PySyft virtual worker: {worker_id}")
            
            # Test PySyft training
            pysyft_task_config = task_config.copy()
            pysyft_task_config["task_name"] = "PySyft_Demo_Task"
            pysyft_task_config["participating_devices"] = worker_ids
            
            pysyft_task_id = await engine.create_federated_task(pysyft_task_config)
            print(f"   📝 Created PySyft task: {pysyft_task_id[:8]}...")
            
            if HAS_TORCH:
                pysyft_success = await engine.start_federated_training(pysyft_task_id, "pysyft")
                if pysyft_success:
                    print("   ✅ PySyft training completed")
                else:
                    print("   ⚠️ PySyft training encountered issues")
        else:
            print("\n10. PySyft not available - skipping PySyft demo")
        
        # Demo 9: TensorFlow Federated Integration (if available)
        if HAS_TFF:
            print("\n11. Testing TensorFlow Federated integration...")
            
            # Prepare sample data for TFF
            client_data = {}
            for i, device_id in enumerate(device_ids[:2]):  # Use first 2 devices
                # Generate sample MNIST-like data
                features = np.random.rand(100, 28, 28).astype(np.float32)
                labels = np.random.randint(0, 10, 100).astype(np.int32)
                client_data[device_id] = (features.reshape(-1, 784), labels)
            
            # Create TFF model
            model_fn = engine.tff_integration.create_keras_model((784,), 10)
            if model_fn:
                print("   ✅ Created TFF Keras model")
                
                # Create federated data
                federated_data = engine.tff_integration.create_federated_data(client_data)
                if federated_data:
                    print("   ✅ Created TFF federated data")
                    
                    # Build and run federated process
                    process = engine.tff_integration.build_federated_averaging_process()
                    if process:
                        print("   ✅ Built TFF federated averaging process")
                        
                        # Run a few rounds of training
                        results = engine.tff_integration.run_federated_training(3)
                        if results:
                            print("   ✅ TFF training completed")
                            print(f"   📊 Training rounds: {len(results['training_results'])}")
        else:
            print("\n11. TensorFlow Federated not available - skipping TFF demo")
        
        # Demo 10: Federation Status and Cleanup
        print("\n12. Getting federation status...")
        
        federation_status = await engine.get_federation_status()
        print(f"   Engine Status: {federation_status['engine_status']}")
        print(f"   Total Tasks: {federation_status['total_tasks']}")
        print(f"   Total Devices: {federation_status['total_devices']}")
        print(f"   Online Devices: {federation_status['online_devices']}")
        print(f"   Privacy Budget Remaining: {federation_status['privacy_budget_remaining']:.4f}")
        
        # Clean up completed tasks
        print("\n13. Cleaning up completed tasks...")
        cleaned_count = await engine.cleanup_completed_tasks(0)  # Clean all completed tasks
        print(f"   🧹 Cleaned up {cleaned_count} completed tasks")
        
        # Demo 11: Privacy Budget Management
        print("\n14. Privacy budget management...")
        
        initial_budget = engine.privacy_engine.get_remaining_budget()
        print(f"   Initial remaining budget: {initial_budget:.4f}")
        
        # Consume some privacy budget
        test_data = np.random.normal(0, 1, (10, 5))
        noisy_data = engine.privacy_engine.add_noise(test_data)
        
        remaining_budget = engine.privacy_engine.get_remaining_budget()
        print(f"   Remaining budget after noise addition: {remaining_budget:.4f}")
        print(f"   Budget consumed: {initial_budget - remaining_budget:.4f}")
        
        # Reset privacy budget
        engine.privacy_engine.reset_budget()
        reset_budget = engine.privacy_engine.get_remaining_budget()
        print(f"   Budget after reset: {reset_budget:.4f}")
        
        print("\n🎉 FederatedEngine demo completed successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required dependencies are installed:")
        print("  - torch (optional, for PyTorch support)")
        print("  - tensorflow (optional, for TensorFlow support)")
        print("  - syft (optional, for PySyft support)")
        print("  - tensorflow-federated (optional, for TFF support)")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.exception("Demo failed")
        return False

async def demo_api_integration():
    """Demo API integration with federated engine"""
    
    print("\n🌐 API Integration Demo")
    print("=" * 30)
    
    try:
        import httpx
        
        # This would normally connect to a running FastAPI server
        # For demo purposes, we'll simulate the API calls
        
        print("📡 Simulating API calls...")
        
        # Simulate device registration
        device_config = {
            "device_name": "API_Test_Device",
            "device_type": "desktop",
            "data_size": 1000,
            "compute_power": 1.5,
            "bandwidth": 25.0,
            "privacy_level": "medium"
        }
        
        print(f"   POST /api/federated/devices")
        print(f"   Request: {json.dumps(device_config, indent=2)}")
        print("   Response: {'device_id': 'abc123...', 'status': 'created'}")
        
        # Simulate task creation
        task_config = {
            "task_name": "API_Demo_Task",
            "model_architecture": {
                "input_size": 784,
                "hidden_size": 64,
                "output_size": 10
            },
            "target_rounds": 3
        }
        
        print(f"\n   POST /api/federated/tasks")
        print(f"   Request: {json.dumps(task_config, indent=2)}")
        print("   Response: {'task_id': 'def456...', 'status': 'created'}")
        
        # Simulate training start
        training_request = {
            "task_id": "def456...",
            "framework": "pytorch"
        }
        
        print(f"\n   POST /api/federated/tasks/def456.../start")
        print(f"   Request: {json.dumps(training_request, indent=2)}")
        print("   Response: {'task_id': 'def456...', 'status': 'training_started'}")
        
        # Simulate status check
        print(f"\n   GET /api/federated/status")
        print("   Response: {")
        print("     'engine_status': 'ready',")
        print("     'total_tasks': 1,")
        print("     'total_devices': 1,")
        print("     'online_devices': 1")
        print("   }")
        
        print("\n✅ API integration demo completed")
        
    except ImportError:
        print("❌ httpx not available for API demo")
    except Exception as e:
        print(f"❌ API demo failed: {e}")

def main():
    """Main demo function"""
    
    print("🔬 ScrollIntel FederatedEngine Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases:")
    print("• PySyft integration for federated learning")
    print("• TensorFlow Federated (TFF) support")
    print("• Differential privacy mechanisms")
    print("• Secure aggregation protocols")
    print("• Edge device simulation and coordination")
    print("• Integration tests for federated workflows")
    print("=" * 60)
    
    # Run the main demo
    success = asyncio.run(demo_federated_engine())
    
    if success:
        # Run API demo
        asyncio.run(demo_api_integration())
        
        print("\n🏆 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Install optional dependencies for full functionality:")
        print("   pip install torch tensorflow syft tensorflow-federated")
        print("2. Start the FastAPI server to test API endpoints")
        print("3. Integrate with real edge devices for production use")
        print("4. Configure privacy parameters for your use case")
        print("5. Set up monitoring and logging for production deployment")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())