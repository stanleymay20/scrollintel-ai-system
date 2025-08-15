"""
Tests for Advanced 2D-to-3D conversion engine.
Tests conversion accuracy and 3D quality validation.
"""

import pytest
import numpy as np
import torch
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from scrollintel.engines.visual_generation.models.depth_estimation import (
    Advanced3DDepthEstimator,
    SubPixelDepthEstimator,
    GeometricReconstructor,
    TemporalDepthConsistencyEngine,
    RealisticParallaxGenerator,
    DepthEstimationResult,
    GeometryReconstructionResult,
    ParallaxGenerationResult
)


class TestAdvanced3DDepthEstimator:
    """Test suite for Advanced3DDepthEstimator."""
    
    @pytest.fixture
    def depth_estimator(self):
        """Create depth estimator instance for testing."""
        config = {
            "precision_level": "sub_pixel",
            "accuracy_target": 0.99
        }
        return Advanced3DDepthEstimator(config)
    
    @pytest.fixture
    def sample_2d_image(self):
        """Create sample 2D image for testing."""
        return np.random.rand(512, 512, 3).astype(np.float32)
    
    @pytest.fixture
    def sample_video_sequence(self):
        """Create sample video sequence for testing."""
        return np.random.rand(10, 512, 512, 3).astype(np.float32)
    
    @pytest.mark.asyncio
    async def test_convert_2d_to_ultra_3d_image(self, depth_estimator, sample_2d_image):
        """Test 2D-to-3D conversion for single image."""
        result = await depth_estimator.convert_2d_to_ultra_3d(sample_2d_image)
        
        assert isinstance(result, DepthEstimationResult)
        assert result.depth_maps is not None
        assert result.confidence_maps is not None
        assert result.geometry is not None
        assert result.conversion_accuracy >= 0.99
        assert result.processing_time > 0
        assert "input_type" in result.metadata
    
    @pytest.mark.asyncio
    async def test_convert_2d_to_ultra_3d_video(self, depth_estimator, sample_video_sequence):
        """Test 2D-to-3D conversion for video sequence."""
        result = await depth_estimator.convert_2d_to_ultra_3d(sample_video_sequence)
        
        assert isinstance(result, DepthEstimationResult)
        assert result.depth_maps is not None
        assert result.parallax_data is not None
        assert result.conversion_accuracy >= 0.99
        assert result.metadata["input_type"] == "ndarray"
    
    @pytest.mark.asyncio
    async def test_multi_scale_depth_estimation(self, depth_estimator, sample_2d_image):
        """Test multi-scale depth estimation with sub-pixel precision."""
        depth_maps = await depth_estimator._estimate_multi_scale_depth(
            sample_2d_image,
            precision_level="sub_pixel",
            accuracy_target=0.99
        )
        
        assert isinstance(depth_maps, np.ndarray)
        assert depth_maps.shape[-2:] == sample_2d_image.shape[:2]
        assert depth_maps.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_sub_pixel_refinement(self, depth_estimator):
        """Test sub-pixel refinement process."""
        # Create sample depth map
        depth_map = torch.randn(1, 1, 256, 256)
        
        refined_depth = await depth_estimator._apply_sub_pixel_refinement(
            depth_map, 
            scale=1.0
        )
        
        assert refined_depth.shape == depth_map.shape
        assert isinstance(refined_depth, torch.Tensor)
    
    @pytest.mark.asyncio
    async def test_edge_preserving_smoothing(self, depth_estimator):
        """Test edge-preserving smoothing functionality."""
        depth_map = torch.randn(1, 1, 128, 128)
        
        smoothed = await depth_estimator._edge_preserving_smoothing(depth_map)
        
        assert smoothed.shape == depth_map.shape
        assert isinstance(smoothed, torch.Tensor)
    
    def test_gaussian_kernel_creation(self, depth_estimator):
        """Test Gaussian kernel creation."""
        kernel = depth_estimator._create_gaussian_kernel(5, 1.0)
        
        assert kernel.shape == (5, 5)
        assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
    
    @pytest.mark.asyncio
    async def test_multi_scale_fusion(self, depth_estimator):
        """Test multi-scale prediction fusion."""
        # Create sample predictions at different scales
        predictions = [
            torch.randn(1, 1, 128, 128),
            torch.randn(1, 1, 64, 64),
            torch.randn(1, 1, 32, 32),
            torch.randn(1, 1, 16, 16)
        ]
        scales = [1.0, 0.5, 0.25, 0.125]
        
        fused = await depth_estimator._fuse_multi_scale_predictions(
            predictions, 
            scales, 
            accuracy_target=0.99
        )
        
        assert fused.shape == predictions[0].shape
        assert isinstance(fused, torch.Tensor)
    
    def test_input_tensor_preparation(self, depth_estimator, sample_2d_image):
        """Test input tensor preparation."""
        tensor = depth_estimator._prepare_input_tensor(sample_2d_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) == 4  # Batch dimension added
        assert tensor.device == depth_estimator.device
    
    def test_video_sequence_detection(self, depth_estimator, sample_video_sequence):
        """Test video sequence detection."""
        is_video = depth_estimator._is_video_sequence(sample_video_sequence)
        assert is_video is True
        
        # Test with single image
        single_image = np.random.rand(512, 512, 3)
        is_video = depth_estimator._is_video_sequence(single_image)
        assert is_video is False
    
    def test_confidence_map_generation(self, depth_estimator):
        """Test confidence map generation."""
        depth_maps = np.random.rand(256, 256).astype(np.float32)
        
        confidence_maps = depth_estimator._generate_confidence_maps(depth_maps)
        
        assert confidence_maps.shape == depth_maps.shape
        assert np.all(confidence_maps >= 0.0)
        assert np.all(confidence_maps <= 1.0)
        assert confidence_maps.dtype == np.float32


class TestSubPixelDepthEstimator:
    """Test suite for SubPixelDepthEstimator neural network."""
    
    @pytest.fixture
    def depth_model(self):
        """Create depth estimation model for testing."""
        return SubPixelDepthEstimator(
            precision_level="sub_pixel",
            accuracy_target=0.99
        )
    
    def test_model_initialization(self, depth_model):
        """Test model initialization."""
        assert depth_model.precision_level == "sub_pixel"
        assert depth_model.accuracy_target == 0.99
        assert hasattr(depth_model, 'encoder')
        assert hasattr(depth_model, 'decoder')
    
    def test_forward_pass(self, depth_model):
        """Test forward pass through the model."""
        input_tensor = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            output = depth_model(input_tensor)
        
        assert output.shape == (2, 1, 256, 256)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)
    
    def test_model_parameters(self, depth_model):
        """Test model has trainable parameters."""
        params = list(depth_model.parameters())
        assert len(params) > 0
        
        total_params = sum(p.numel() for p in params if p.requires_grad)
        assert total_params > 0


class TestGeometricReconstructor:
    """Test suite for GeometricReconstructor."""
    
    @pytest.fixture
    def reconstructor(self):
        """Create geometric reconstructor for testing."""
        return GeometricReconstructor()
    
    @pytest.fixture
    def sample_depth_maps(self):
        """Create sample depth maps for testing."""
        return np.random.rand(64, 64).astype(np.float32)
    
    @pytest.mark.asyncio
    async def test_build_3d_geometry(self, reconstructor, sample_depth_maps):
        """Test 3D geometry reconstruction."""
        result = await reconstructor.build_3d_geometry(
            sample_depth_maps,
            mesh_quality="ultra_high",
            edge_preservation=True
        )
        
        assert isinstance(result, GeometryReconstructionResult)
        assert result.vertices is not None
        assert result.faces is not None
        assert result.normals is not None
        assert result.texture_coords is not None
        assert result.mesh_quality == "ultra_high"
        assert result.edge_preservation is True
        assert result.reconstruction_accuracy >= 0.99
    
    @pytest.mark.asyncio
    async def test_vertex_generation(self, reconstructor, sample_depth_maps):
        """Test vertex generation from depth maps."""
        vertices = await reconstructor._generate_vertices_from_depth(sample_depth_maps)
        
        assert isinstance(vertices, np.ndarray)
        assert vertices.shape[1] == 3  # X, Y, Z coordinates
        assert vertices.dtype == np.float32
        assert len(vertices) == sample_depth_maps.size
    
    @pytest.mark.asyncio
    async def test_face_generation(self, reconstructor):
        """Test optimal face generation."""
        # Create sample vertices for a 4x4 grid
        vertices = np.random.rand(16, 3).astype(np.float32)
        
        faces = await reconstructor._generate_optimal_faces(vertices, "ultra_high")
        
        assert isinstance(faces, np.ndarray)
        assert faces.shape[1] == 3  # Triangular faces
        assert faces.dtype == np.int32
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))
    
    @pytest.mark.asyncio
    async def test_normal_computation(self, reconstructor):
        """Test vertex normal computation."""
        # Create simple triangle vertices
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2],
            [1, 3, 2]
        ], dtype=np.int32)
        
        normals = await reconstructor._compute_vertex_normals(vertices, faces)
        
        assert normals.shape == vertices.shape
        assert normals.dtype == np.float32
        
        # Check that normals are normalized
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    @pytest.mark.asyncio
    async def test_texture_coordinate_generation(self, reconstructor):
        """Test texture coordinate generation."""
        vertices = np.random.rand(100, 3).astype(np.float32)
        
        tex_coords = await reconstructor._generate_texture_coordinates(vertices)
        
        assert tex_coords.shape == (100, 2)
        assert tex_coords.dtype == np.float32
        assert np.all(tex_coords >= 0.0)
        assert np.all(tex_coords <= 1.0)
    
    @pytest.mark.asyncio
    async def test_edge_preservation(self, reconstructor):
        """Test sharp edge preservation."""
        vertices = np.random.rand(50, 3).astype(np.float32)
        faces = np.random.randint(0, 50, (30, 3)).astype(np.int32)
        normals = np.random.rand(50, 3).astype(np.float32)
        
        preserved_vertices, preserved_faces, preserved_normals = await reconstructor._preserve_sharp_edges(
            vertices, faces, normals
        )
        
        assert preserved_vertices.shape == vertices.shape
        assert preserved_faces.shape == faces.shape
        assert preserved_normals.shape == normals.shape


class TestTemporalDepthConsistencyEngine:
    """Test suite for TemporalDepthConsistencyEngine."""
    
    @pytest.fixture
    def consistency_engine(self):
        """Create temporal consistency engine for testing."""
        return TemporalDepthConsistencyEngine()
    
    @pytest.fixture
    def geometry_sequence(self):
        """Create sample geometry sequence for testing."""
        sequence = []
        for i in range(5):
            geometry = GeometryReconstructionResult(
                vertices=np.random.rand(100, 3).astype(np.float32),
                faces=np.random.randint(0, 100, (50, 3)).astype(np.int32),
                normals=np.random.rand(100, 3).astype(np.float32),
                texture_coords=np.random.rand(100, 2).astype(np.float32),
                reconstruction_accuracy=0.99
            )
            sequence.append(geometry)
        return sequence
    
    @pytest.mark.asyncio
    async def test_ensure_depth_consistency(self, consistency_engine, geometry_sequence):
        """Test temporal depth consistency enforcement."""
        consistent_sequence = await consistency_engine.ensure_depth_consistency(
            geometry_sequence
        )
        
        assert len(consistent_sequence) == len(geometry_sequence)
        assert all(isinstance(frame, GeometryReconstructionResult) for frame in consistent_sequence)
        
        # First frame should be unchanged
        assert np.array_equal(
            consistent_sequence[0].vertices, 
            geometry_sequence[0].vertices
        )
    
    @pytest.mark.asyncio
    async def test_temporal_smoothing(self, consistency_engine):
        """Test temporal smoothing between frames."""
        # Create two similar frames
        frame1 = GeometryReconstructionResult(
            vertices=np.ones((10, 3), dtype=np.float32),
            faces=np.zeros((5, 3), dtype=np.int32),
            normals=np.ones((10, 3), dtype=np.float32),
            texture_coords=np.ones((10, 2), dtype=np.float32),
            reconstruction_accuracy=0.99
        )
        
        frame2 = GeometryReconstructionResult(
            vertices=np.zeros((10, 3), dtype=np.float32),
            faces=np.zeros((5, 3), dtype=np.int32),
            normals=np.zeros((10, 3), dtype=np.float32),
            texture_coords=np.zeros((10, 2), dtype=np.float32),
            reconstruction_accuracy=0.99
        )
        
        smoothed = await consistency_engine._apply_temporal_smoothing(frame2, frame1)
        
        assert isinstance(smoothed, GeometryReconstructionResult)
        # Smoothed vertices should be between original values
        assert np.all(smoothed.vertices >= 0.0)
        assert np.all(smoothed.vertices <= 1.0)
    
    @pytest.mark.asyncio
    async def test_single_frame_consistency(self, consistency_engine):
        """Test consistency with single frame."""
        single_frame = [GeometryReconstructionResult(
            vertices=np.random.rand(10, 3).astype(np.float32),
            faces=np.random.randint(0, 10, (5, 3)).astype(np.int32),
            normals=np.random.rand(10, 3).astype(np.float32),
            texture_coords=np.random.rand(10, 2).astype(np.float32),
            reconstruction_accuracy=0.99
        )]
        
        result = await consistency_engine.ensure_depth_consistency(single_frame)
        
        assert len(result) == 1
        assert np.array_equal(result[0].vertices, single_frame[0].vertices)


class TestRealisticParallaxGenerator:
    """Test suite for RealisticParallaxGenerator."""
    
    @pytest.fixture
    def parallax_generator(self):
        """Create parallax generator for testing."""
        return RealisticParallaxGenerator()
    
    @pytest.fixture
    def sample_geometry(self):
        """Create sample geometry for testing."""
        return GeometryReconstructionResult(
            vertices=np.random.rand(64, 3).astype(np.float32),
            faces=np.random.randint(0, 64, (30, 3)).astype(np.int32),
            normals=np.random.rand(64, 3).astype(np.float32),
            texture_coords=np.random.rand(64, 2).astype(np.float32),
            reconstruction_accuracy=0.99
        )
    
    @pytest.mark.asyncio
    async def test_create_realistic_parallax(self, parallax_generator, sample_geometry):
        """Test realistic parallax generation."""
        result = await parallax_generator.create_realistic_parallax(
            sample_geometry,
            camera_movement_realism=0.99
        )
        
        assert isinstance(result, ParallaxGenerationResult)
        assert result.parallax_maps is not None
        assert result.camera_movement_data is not None
        assert result.realism_score >= 0.9
        assert result.movement_accuracy == 0.99
    
    @pytest.mark.asyncio
    async def test_parallax_map_generation(self, parallax_generator, sample_geometry):
        """Test parallax map generation."""
        parallax_maps = await parallax_generator._generate_parallax_maps(sample_geometry)
        
        assert isinstance(parallax_maps, np.ndarray)
        assert parallax_maps.shape[0] == 2  # X and Y parallax
        assert parallax_maps.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_camera_movement_generation(self, parallax_generator, sample_geometry):
        """Test camera movement parameter generation."""
        movement_data = await parallax_generator._generate_camera_movement(
            sample_geometry,
            realism_target=0.99
        )
        
        assert isinstance(movement_data, dict)
        assert "translation" in movement_data
        assert "rotation" in movement_data
        assert "focal_length" in movement_data
        assert "aperture" in movement_data
        assert "realism_factors" in movement_data
        
        # Check translation parameters
        translation = movement_data["translation"]
        assert "x" in translation
        assert "y" in translation
        assert "z" in translation
        
        # Check rotation parameters
        rotation = movement_data["rotation"]
        assert "pitch" in rotation
        assert "yaw" in rotation
        assert "roll" in rotation
    
    @pytest.mark.asyncio
    async def test_parallax_realism_calculation(self, parallax_generator):
        """Test parallax realism score calculation."""
        parallax_maps = np.random.rand(2, 32, 32).astype(np.float32)
        camera_data = {
            "translation": {"x": 0.1, "y": 0.05, "z": 0.2},
            "rotation": {"pitch": 2, "yaw": 5, "roll": 1}
        }
        
        realism_score = await parallax_generator._calculate_parallax_realism(
            parallax_maps,
            camera_data
        )
        
        assert isinstance(realism_score, float)
        assert 0.0 <= realism_score <= 0.99


class TestIntegration:
    """Integration tests for the complete 2D-to-3D conversion pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_conversion_pipeline(self):
        """Test complete 2D-to-3D conversion pipeline."""
        # Create depth estimator
        estimator = Advanced3DDepthEstimator()
        
        # Create sample input
        sample_input = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Run complete conversion
        result = await estimator.convert_2d_to_ultra_3d(sample_input)
        
        # Verify complete result
        assert isinstance(result, DepthEstimationResult)
        assert result.depth_maps is not None
        assert result.confidence_maps is not None
        assert result.geometry is not None
        assert result.conversion_accuracy >= 0.99
        assert result.processing_time > 0
        
        # Verify geometry quality
        geometry = result.geometry
        assert isinstance(geometry, GeometryReconstructionResult)
        assert geometry.vertices is not None
        assert geometry.faces is not None
        assert geometry.normals is not None
        assert geometry.texture_coords is not None
    
    @pytest.mark.asyncio
    async def test_video_sequence_conversion(self):
        """Test conversion of video sequence with temporal consistency."""
        estimator = Advanced3DDepthEstimator()
        
        # Create video sequence
        video_sequence = np.random.rand(5, 128, 128, 3).astype(np.float32)
        
        # Convert video sequence
        result = await estimator.convert_2d_to_ultra_3d(video_sequence)
        
        # Verify video-specific features
        assert result.parallax_data is not None
        assert isinstance(result.parallax_data, ParallaxGenerationResult)
        assert result.conversion_accuracy >= 0.99
    
    @pytest.mark.asyncio
    async def test_accuracy_requirements(self):
        """Test that accuracy requirements are met."""
        estimator = Advanced3DDepthEstimator()
        sample_input = np.random.rand(128, 128, 3).astype(np.float32)
        
        result = await estimator.convert_2d_to_ultra_3d(sample_input)
        
        # Verify accuracy requirements from task specification
        assert result.conversion_accuracy >= 0.99  # 99% accuracy requirement
        
        # Verify geometry reconstruction accuracy
        geometry = result.geometry
        assert geometry.reconstruction_accuracy >= 0.99
        
        # Verify parallax accuracy if present
        if result.parallax_data:
            parallax_result = result.parallax_data
            if isinstance(parallax_result, dict) and "movement_accuracy" in parallax_result:
                assert parallax_result["movement_accuracy"] >= 0.99


if __name__ == "__main__":
    pytest.main([__file__])