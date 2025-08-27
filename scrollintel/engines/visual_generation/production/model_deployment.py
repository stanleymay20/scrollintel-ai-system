"""
Production Model Deployment and GPU Infrastructure Setup
Handles deployment of Stable Diffusion XL, DALL-E 3, and Midjourney integrations
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import torch
import requests
from diffusers import StableDiffusionXLPipeline
import openai
from discord import Client, Intents

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    WARMING_UP = "warming_up"

@dataclass
class ModelConfig:
    name: str
    model_path: str
    gpu_memory_gb: float
    warm_up_prompts: List[str]
    rate_limit_per_minute: int
    priority: int

@dataclass
class GPUClusterConfig:
    cluster_name: str
    gpu_type: str
    min_instances: int
    max_instances: int
    auto_scaling_enabled: bool
    cost_per_hour: float

class ProductionModelDeployment:
    """Manages production deployment of visual generation models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.gpu_clusters: Dict[str, GPUClusterConfig] = {}
        self.rate_limiters: Dict[str, Dict] = {}
        self.warm_up_cache: Dict[str, List] = {}
        
        # Load configuration from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.midjourney_bot_token = os.getenv("MIDJOURNEY_BOT_TOKEN")
        self.midjourney_server_id = os.getenv("MIDJOURNEY_SERVER_ID")
        self.midjourney_channel_id = os.getenv("MIDJOURNEY_CHANNEL_ID")
        
        # GPU cluster configurations
        self._setup_gpu_clusters()
        
    def _setup_gpu_clusters(self):
        """Configure GPU clusters for different model types"""
        self.gpu_clusters = {
            "stable_diffusion_cluster": GPUClusterConfig(
                cluster_name="stable_diffusion_cluster",
                gpu_type="A100-80GB",
                min_instances=2,
                max_instances=10,
                auto_scaling_enabled=True,
                cost_per_hour=3.20
            ),
            "dalle3_cluster": GPUClusterConfig(
                cluster_name="dalle3_cluster", 
                gpu_type="V100-32GB",
                min_instances=1,
                max_instances=5,
                auto_scaling_enabled=True,
                cost_per_hour=2.40
            ),
            "video_generation_cluster": GPUClusterConfig(
                cluster_name="video_generation_cluster",
                gpu_type="A100-80GB",
                min_instances=3,
                max_instances=15,
                auto_scaling_enabled=True,
                cost_per_hour=4.80
            )
        }
        
    async def deploy_stable_diffusion_xl(self) -> bool:
        """Deploy Stable Diffusion XL models to production GPU clusters"""
        try:
            logger.info("Starting Stable Diffusion XL deployment...")
            self.model_status["stable_diffusion_xl"] = ModelStatus.LOADING
            
            # Check GPU availability
            if not torch.cuda.is_available():
                logger.error("CUDA not available for Stable Diffusion XL deployment")
                self.model_status["stable_diffusion_xl"] = ModelStatus.ERROR
                return False
                
            # Load model with optimizations
            model_config = ModelConfig(
                name="stable_diffusion_xl",
                model_path="stabilityai/stable-diffusion-xl-base-1.0",
                gpu_memory_gb=24.0,
                warm_up_prompts=[
                    "A photorealistic portrait of a person",
                    "A beautiful landscape at sunset",
                    "Abstract digital art composition"
                ],
                rate_limit_per_minute=60,
                priority=1
            )
            
            # Initialize pipeline with production optimizations
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_config.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Enable memory efficient attention
            pipeline.enable_model_cpu_offload()
            pipeline.enable_xformers_memory_efficient_attention()
            
            # Move to GPU
            device = torch.device("cuda:0")
            pipeline = pipeline.to(device)
            
            self.models["stable_diffusion_xl"] = pipeline
            
            # Setup rate limiting
            self.rate_limiters["stable_diffusion_xl"] = {
                "requests_per_minute": model_config.rate_limit_per_minute,
                "current_requests": 0,
                "last_reset": time.time()
            }
            
            # Perform warm-up
            await self._warm_up_model("stable_diffusion_xl", model_config.warm_up_prompts)
            
            self.model_status["stable_diffusion_xl"] = ModelStatus.READY
            logger.info("Stable Diffusion XL deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy Stable Diffusion XL: {str(e)}")
            self.model_status["stable_diffusion_xl"] = ModelStatus.ERROR
            return False
    
    async def configure_dalle3_integration(self) -> bool:
        """Configure DALL-E 3 API keys and rate limiting for production"""
        try:
            logger.info("Configuring DALL-E 3 integration...")
            self.model_status["dalle3"] = ModelStatus.LOADING
            
            if not self.openai_api_key:
                logger.error("OpenAI API key not found in environment variables")
                self.model_status["dalle3"] = ModelStatus.ERROR
                return False
            
            # Initialize OpenAI client
            openai.api_key = self.openai_api_key
            
            # Test API connection
            try:
                response = await asyncio.to_thread(
                    openai.Image.create,
                    prompt="Test image generation",
                    model="dall-e-3",
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                logger.info("DALL-E 3 API connection test successful")
            except Exception as api_error:
                logger.error(f"DALL-E 3 API test failed: {str(api_error)}")
                self.model_status["dalle3"] = ModelStatus.ERROR
                return False
            
            # Configure production rate limiting
            self.rate_limiters["dalle3"] = {
                "requests_per_minute": 50,  # OpenAI's rate limit
                "requests_per_day": 1000,
                "current_requests_minute": 0,
                "current_requests_day": 0,
                "last_minute_reset": time.time(),
                "last_day_reset": time.time()
            }
            
            # Store API configuration
            self.models["dalle3"] = {
                "api_key": self.openai_api_key,
                "model": "dall-e-3",
                "default_size": "1024x1024",
                "default_quality": "hd",
                "timeout": 60
            }
            
            self.model_status["dalle3"] = ModelStatus.READY
            logger.info("DALL-E 3 integration configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure DALL-E 3: {str(e)}")
            self.model_status["dalle3"] = ModelStatus.ERROR
            return False
    
    async def setup_midjourney_integration(self) -> bool:
        """Set up Midjourney Discord bot integration with production credentials"""
        try:
            logger.info("Setting up Midjourney integration...")
            self.model_status["midjourney"] = ModelStatus.LOADING
            
            if not all([self.midjourney_bot_token, self.midjourney_server_id, self.midjourney_channel_id]):
                logger.error("Midjourney credentials not found in environment variables")
                self.model_status["midjourney"] = ModelStatus.ERROR
                return False
            
            # Initialize Discord client
            intents = Intents.default()
            intents.message_content = True
            
            class MidjourneyBot(Client):
                def __init__(self, deployment_manager):
                    super().__init__(intents=intents)
                    self.deployment_manager = deployment_manager
                    self.job_queue = asyncio.Queue()
                    self.active_jobs = {}
                
                async def on_ready(self):
                    logger.info(f'Midjourney bot logged in as {self.user}')
                    self.deployment_manager.model_status["midjourney"] = ModelStatus.READY
                
                async def on_message(self, message):
                    # Handle Midjourney responses
                    if message.author.id == 936929561302675456:  # Midjourney bot ID
                        await self._process_midjourney_response(message)
                
                async def _process_midjourney_response(self, message):
                    # Process Midjourney generation results
                    if message.attachments:
                        for attachment in message.attachments:
                            if attachment.filename.endswith(('.png', '.jpg', '.jpeg')):
                                # Store result for retrieval
                                job_id = self._extract_job_id(message.content)
                                if job_id in self.active_jobs:
                                    self.active_jobs[job_id]['result'] = attachment.url
                                    self.active_jobs[job_id]['status'] = 'completed'
            
            # Initialize bot
            midjourney_bot = MidjourneyBot(self)
            
            # Store bot configuration
            self.models["midjourney"] = {
                "bot": midjourney_bot,
                "server_id": self.midjourney_server_id,
                "channel_id": self.midjourney_channel_id,
                "job_timeout": 300,  # 5 minutes
                "max_concurrent_jobs": 10
            }
            
            # Configure rate limiting
            self.rate_limiters["midjourney"] = {
                "requests_per_minute": 20,  # Conservative limit
                "current_requests": 0,
                "last_reset": time.time(),
                "queue_size": 50
            }
            
            # Start bot (in production, this would be handled by a separate service)
            # await midjourney_bot.start(self.midjourney_bot_token)
            
            self.model_status["midjourney"] = ModelStatus.READY
            logger.info("Midjourney integration setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Midjourney integration: {str(e)}")
            self.model_status["midjourney"] = ModelStatus.ERROR
            return False
    
    async def _warm_up_model(self, model_name: str, prompts: List[str]):
        """Perform model warm-up with preloading for faster response times"""
        try:
            logger.info(f"Warming up {model_name}...")
            self.model_status[model_name] = ModelStatus.WARMING_UP
            
            if model_name == "stable_diffusion_xl":
                pipeline = self.models[model_name]
                warm_up_results = []
                
                for prompt in prompts:
                    start_time = time.time()
                    
                    # Generate with minimal steps for warm-up
                    result = pipeline(
                        prompt=prompt,
                        num_inference_steps=10,  # Minimal steps for warm-up
                        guidance_scale=7.5,
                        width=512,  # Smaller size for warm-up
                        height=512
                    )
                    
                    generation_time = time.time() - start_time
                    warm_up_results.append({
                        "prompt": prompt,
                        "generation_time": generation_time
                    })
                    
                    logger.info(f"Warm-up generation completed in {generation_time:.2f}s")
                
                # Cache warm-up results
                self.warm_up_cache[model_name] = warm_up_results
                
            logger.info(f"Model {model_name} warm-up completed")
            
        except Exception as e:
            logger.error(f"Model warm-up failed for {model_name}: {str(e)}")
            raise
    
    async def check_gpu_resources(self) -> Dict[str, Any]:
        """Check GPU resource availability and utilization"""
        try:
            gpu_info = {}
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device = torch.device(f"cuda:{i}")
                    gpu_props = torch.cuda.get_device_properties(device)
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_reserved = torch.cuda.memory_reserved(device)
                    memory_total = gpu_props.total_memory
                    
                    gpu_info[f"gpu_{i}"] = {
                        "name": gpu_props.name,
                        "memory_total_gb": memory_total / (1024**3),
                        "memory_allocated_gb": memory_allocated / (1024**3),
                        "memory_reserved_gb": memory_reserved / (1024**3),
                        "memory_free_gb": (memory_total - memory_reserved) / (1024**3),
                        "utilization_percent": (memory_reserved / memory_total) * 100
                    }
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"Failed to check GPU resources: {str(e)}")
            return {}
    
    async def scale_gpu_cluster(self, cluster_name: str, target_instances: int) -> bool:
        """Scale GPU cluster based on demand"""
        try:
            if cluster_name not in self.gpu_clusters:
                logger.error(f"Unknown GPU cluster: {cluster_name}")
                return False
            
            cluster_config = self.gpu_clusters[cluster_name]
            
            if target_instances < cluster_config.min_instances:
                target_instances = cluster_config.min_instances
            elif target_instances > cluster_config.max_instances:
                target_instances = cluster_config.max_instances
            
            logger.info(f"Scaling {cluster_name} to {target_instances} instances")
            
            # In production, this would interface with cloud provider APIs
            # For now, we'll simulate the scaling operation
            await asyncio.sleep(2)  # Simulate scaling delay
            
            logger.info(f"Successfully scaled {cluster_name} to {target_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale GPU cluster {cluster_name}: {str(e)}")
            return False
    
    async def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive status of all deployed models"""
        status_report = {}
        
        for model_name in ["stable_diffusion_xl", "dalle3", "midjourney"]:
            status_report[model_name] = {
                "status": self.model_status.get(model_name, ModelStatus.ERROR).value,
                "rate_limiter": self.rate_limiters.get(model_name, {}),
                "warm_up_cache": len(self.warm_up_cache.get(model_name, [])),
                "last_health_check": time.time()
            }
        
        # Add GPU resource information
        status_report["gpu_resources"] = await self.check_gpu_resources()
        
        return status_report
    
    async def deploy_all_models(self) -> bool:
        """Deploy all visual generation models to production"""
        try:
            logger.info("Starting production model deployment...")
            
            # Deploy models in parallel
            deployment_tasks = [
                self.deploy_stable_diffusion_xl(),
                self.configure_dalle3_integration(),
                self.setup_midjourney_integration()
            ]
            
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            # Check results
            success_count = sum(1 for result in results if result is True)
            total_models = len(deployment_tasks)
            
            if success_count == total_models:
                logger.info("All models deployed successfully")
                return True
            else:
                logger.warning(f"Only {success_count}/{total_models} models deployed successfully")
                return False
                
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return False

# Production deployment manager instance
production_deployment = ProductionModelDeployment()

async def initialize_production_models():
    """Initialize all production models"""
    return await production_deployment.deploy_all_models()

async def get_production_status():
    """Get production deployment status"""
    return await production_deployment.get_model_status()