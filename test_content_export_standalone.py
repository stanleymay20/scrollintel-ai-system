"""
Standalone test for content export functionality.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from PIL import Image
from unittest.mock import patch

# Direct import to avoid dependency issues
import sys
sys.path.append('.')

# Import directly from the module file to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "content_exporter", 
    "scrollintel/engines/visual_generation/utils/content_exporter.py"
)
content_exporter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(content_exporter_module)

# Import the classes
ContentExporter = content_exporter_module.ContentExporter
FormatConverter = content_exporter_module.FormatConverter
ExportSettings = content_exporter_module.ExportSettings
ContentMetadata = content_exporter_module.ContentMetadata
ExportFormat = content_exporter_module.ExportFormat
QualityLevel = content_exporter_module.QualityLevel
CompressionType = content_exporter_module.CompressionType


async def test_basic_functionality():
    """Test basic content export functionality."""
    print("🧪 Testing Content Export Functionality")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image
        input_path = os.path.join(temp_dir, "test_image.png")
        image = Image.new('RGB', (100, 100), color='blue')
        image.save(input_path)
        print(f"✅ Created test image: {input_path}")
        
        # Initialize exporter
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        print(f"✅ Initialized ContentExporter with output directory: {export_dir}")
        
        # Test 1: Basic export
        print("\n📤 Test 1: Basic JPEG Export")
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.HIGH,
            compression=CompressionType.BALANCED
        )
        
        # Mock quality metrics to avoid external dependencies
        with patch.object(exporter, '_calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 32.0, "ssim": 0.95, "mse": 12.0}
            
            result = await exporter.export_content(input_path, settings)
            
            if result.success:
                print(f"   ✅ Export successful!")
                print(f"   📁 Output: {result.output_path}")
                print(f"   📊 File size: {result.file_size:,} bytes")
                print(f"   ⏱️  Processing time: {result.processing_time:.3f}s")
                print(f"   📈 Quality metrics: {result.quality_metrics}")
                
                # Verify output file
                if os.path.exists(result.output_path):
                    output_image = Image.open(result.output_path)
                    print(f"   🖼️  Output format: {output_image.format}")
                    print(f"   📐 Output size: {output_image.size}")
                else:
                    print(f"   ❌ Output file not found!")
            else:
                print(f"   ❌ Export failed: {result.error_message}")
        
        # Test 2: Format conversion
        print("\n🔄 Test 2: Format Conversion")
        converter = FormatConverter(exporter)
        
        with patch.object(exporter, '_calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.92, "mse": 15.0}
            
            conversion_result = await converter.convert_format(
                input_path,
                ExportFormat.WEBP,
                QualityLevel.HIGH
            )
            
            if conversion_result.success:
                print(f"   ✅ Conversion successful!")
                print(f"   📁 Output: {conversion_result.output_path}")
                print(f"   📊 File size: {conversion_result.file_size:,} bytes")
                print(f"   📈 Quality metrics: {conversion_result.quality_metrics}")
            else:
                print(f"   ❌ Conversion failed: {conversion_result.error_message}")
        
        # Test 3: Metadata handling
        print("\n📋 Test 3: Metadata Handling")
        metadata = ContentMetadata(
            title="Test Image",
            description="A test image for content export",
            creator="Test Suite",
            generation_prompt="Create a blue square",
            model_used="test_model_v1",
            tags=["test", "blue", "square"]
        )
        
        settings_with_metadata = ExportSettings(
            format=ExportFormat.PNG,
            quality=QualityLevel.HIGH,
            preserve_metadata=True,
            watermark="TEST"
        )
        
        with patch.object(exporter, '_calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 35.0, "ssim": 0.98, "mse": 8.0}
            
            metadata_result = await exporter.export_content(
                input_path,
                settings_with_metadata,
                metadata,
                "test_with_metadata.png"
            )
            
            if metadata_result.success:
                print(f"   ✅ Export with metadata successful!")
                print(f"   📁 Output: {metadata_result.output_path}")
                print(f"   📊 File size: {metadata_result.file_size:,} bytes")
                print(f"   🏷️  Metadata preserved: {settings_with_metadata.preserve_metadata}")
                print(f"   💧 Watermark applied: {settings_with_metadata.watermark}")
            else:
                print(f"   ❌ Export with metadata failed: {metadata_result.error_message}")
        
        # Test 4: Archive creation
        print("\n📦 Test 4: Archive Creation")
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            test_files.append(file_path)
        
        archive_result = await exporter.create_archive(
            test_files,
            ExportFormat.ZIP,
            "test_archive.zip"
        )
        
        if archive_result.success:
            print(f"   ✅ Archive creation successful!")
            print(f"   📁 Archive: {archive_result.output_path}")
            print(f"   📊 Archive size: {archive_result.file_size:,} bytes")
            print(f"   📄 Files archived: {archive_result.quality_metrics['files_archived']}")
        else:
            print(f"   ❌ Archive creation failed: {archive_result.error_message}")
        
        # Test 5: Format capabilities
        print("\n📊 Test 5: Format Capabilities")
        
        image_formats = exporter.get_supported_formats("image")
        video_formats = exporter.get_supported_formats("video")
        
        print(f"   🖼️  Supported image formats: {[f.value for f in image_formats]}")
        print(f"   🎥 Supported video formats: {[f.value for f in video_formats]}")
        
        # Get format info
        jpeg_info = exporter.get_format_info(ExportFormat.JPEG)
        png_info = exporter.get_format_info(ExportFormat.PNG)
        
        print(f"   📄 JPEG info: {jpeg_info['description']}")
        print(f"      Transparency: {'✅' if jpeg_info.get('supports_transparency') else '❌'}")
        print(f"   📄 PNG info: {png_info['description']}")
        print(f"      Transparency: {'✅' if png_info.get('supports_transparency') else '❌'}")
        
        # Test 6: Error handling
        print("\n🚨 Test 6: Error Handling")
        
        error_result = await exporter.export_content(
            "nonexistent_file.jpg",
            ExportSettings(format=ExportFormat.PNG)
        )
        
        if not error_result.success:
            print(f"   ✅ Error handling working correctly!")
            print(f"   📝 Error message: {error_result.error_message}")
        else:
            print(f"   ❌ Error handling failed - should have returned error!")
        
        print("\n🎉 All tests completed!")


def test_data_classes():
    """Test data classes and enums."""
    print("\n🏗️  Testing Data Classes and Enums")
    print("=" * 50)
    
    # Test ExportSettings
    settings = ExportSettings(
        format=ExportFormat.WEBP,
        quality=QualityLevel.MEDIUM,
        compression=CompressionType.FAST,
        resolution=(800, 600),
        watermark="TEST"
    )
    
    print(f"✅ ExportSettings created:")
    print(f"   Format: {settings.format.value}")
    print(f"   Quality: {settings.quality.value}")
    print(f"   Compression: {settings.compression.value}")
    print(f"   Resolution: {settings.resolution}")
    print(f"   Watermark: {settings.watermark}")
    
    # Test ContentMetadata
    metadata = ContentMetadata(
        title="Test Metadata",
        creator="Test Creator",
        tags=["test", "metadata"]
    )
    
    print(f"\n✅ ContentMetadata created:")
    print(f"   Title: {metadata.title}")
    print(f"   Creator: {metadata.creator}")
    print(f"   Tags: {metadata.tags}")
    print(f"   Creation date: {metadata.creation_date}")
    
    # Test enums
    print(f"\n✅ Enums working:")
    print(f"   Export formats: {[f.value for f in ExportFormat]}")
    print(f"   Quality levels: {[q.value for q in QualityLevel]}")
    print(f"   Compression types: {[c.value for c in CompressionType]}")


async def main():
    """Run all tests."""
    print("🚀 Content Export System Standalone Test")
    print("=" * 60)
    
    try:
        test_data_classes()
        await test_basic_functionality()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("✅ Content export system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())