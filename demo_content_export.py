"""
Demo script for content export and format conversion functionality.

This script demonstrates the capabilities of the ContentExporter system,
including format conversion, quality settings, batch processing, and metadata handling.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime

from scrollintel.engines.visual_generation.utils.content_exporter import (
    ContentExporter,
    FormatConverter,
    ExportSettings,
    ContentMetadata,
    ExportFormat,
    QualityLevel,
    CompressionType
)


def create_sample_images(temp_dir: str) -> list:
    """Create sample images for demonstration."""
    sample_images = []
    
    # Create a simple colored image
    simple_path = os.path.join(temp_dir, "simple_red.png")
    simple_image = Image.new('RGB', (200, 200), color='red')
    simple_image.save(simple_path)
    sample_images.append(simple_path)
    
    # Create a gradient image
    gradient_path = os.path.join(temp_dir, "gradient.png")
    width, height = 300, 200
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(width):
        gradient[:, i, 0] = int(255 * i / width)  # Red gradient
        gradient[:, i, 1] = int(255 * (1 - i / width))  # Green gradient
        gradient[:, i, 2] = 128  # Constant blue
    
    gradient_image = Image.fromarray(gradient)
    gradient_image.save(gradient_path)
    sample_images.append(gradient_path)
    
    # Create an image with transparency
    transparent_path = os.path.join(temp_dir, "transparent.png")
    transparent_image = Image.new('RGBA', (150, 150), (0, 255, 0, 128))  # Semi-transparent green
    transparent_image.save(transparent_path)
    sample_images.append(transparent_path)
    
    # Create a complex pattern image
    pattern_path = os.path.join(temp_dir, "pattern.png")
    pattern = np.zeros((200, 200, 3), dtype=np.uint8)
    
    for i in range(200):
        for j in range(200):
            pattern[i, j, 0] = (i + j) % 256
            pattern[i, j, 1] = (i * j) % 256
            pattern[i, j, 2] = (i ^ j) % 256
    
    pattern_image = Image.fromarray(pattern)
    pattern_image.save(pattern_path)
    sample_images.append(pattern_path)
    
    return sample_images


async def demo_basic_export():
    """Demonstrate basic export functionality."""
    print("🎨 Demo: Basic Content Export")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample images
        sample_images = create_sample_images(temp_dir)
        
        # Initialize exporter
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        # Basic export with different formats
        formats_to_test = [
            (ExportFormat.JPEG, "JPEG with high quality"),
            (ExportFormat.PNG, "PNG with lossless compression"),
            (ExportFormat.WEBP, "WebP with balanced compression")
        ]
        
        for format_type, description in formats_to_test:
            print(f"\n📤 Exporting to {description}")
            
            settings = ExportSettings(
                format=format_type,
                quality=QualityLevel.HIGH,
                compression=CompressionType.BALANCED
            )
            
            result = await exporter.export_content(
                sample_images[0],  # Use the simple red image
                settings
            )
            
            if result.success:
                print(f"✅ Export successful!")
                print(f"   Output: {result.output_path}")
                print(f"   File size: {result.file_size:,} bytes")
                print(f"   Processing time: {result.processing_time:.2f}s")
                if result.quality_metrics:
                    print(f"   Quality metrics: {result.quality_metrics}")
            else:
                print(f"❌ Export failed: {result.error_message}")


async def demo_quality_comparison():
    """Demonstrate quality level comparison."""
    print("\n🔍 Demo: Quality Level Comparison")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_images = create_sample_images(temp_dir)
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        quality_levels = [
            (QualityLevel.LOW, "Low Quality"),
            (QualityLevel.MEDIUM, "Medium Quality"),
            (QualityLevel.HIGH, "High Quality"),
            (QualityLevel.LOSSLESS, "Lossless Quality")
        ]
        
        print(f"📊 Comparing quality levels for JPEG export:")
        
        for quality, description in quality_levels:
            settings = ExportSettings(
                format=ExportFormat.JPEG,
                quality=quality,
                compression=CompressionType.BALANCED
            )
            
            result = await exporter.export_content(
                sample_images[1],  # Use the gradient image
                settings,
                output_filename=f"gradient_{quality.value}.jpg"
            )
            
            if result.success:
                print(f"   {description:15} | Size: {result.file_size:6,} bytes | Time: {result.processing_time:.2f}s")
            else:
                print(f"   {description:15} | ❌ Failed: {result.error_message}")


async def demo_batch_export():
    """Demonstrate batch export functionality."""
    print("\n📦 Demo: Batch Export")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_images = create_sample_images(temp_dir)
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        print(f"🚀 Batch exporting {len(sample_images)} images to WebP format...")
        
        settings = ExportSettings(
            format=ExportFormat.WEBP,
            quality=QualityLevel.HIGH,
            compression=CompressionType.BALANCED
        )
        
        start_time = asyncio.get_event_loop().time()
        results = await exporter.batch_export(
            sample_images,
            settings,
            max_concurrent=2
        )
        total_time = asyncio.get_event_loop().time() - start_time
        
        successful_exports = [r for r in results if r.success]
        failed_exports = [r for r in results if not r.success]
        
        print(f"✅ Batch export completed in {total_time:.2f}s")
        print(f"   Successful: {len(successful_exports)}")
        print(f"   Failed: {len(failed_exports)}")
        
        if successful_exports:
            total_size = sum(r.file_size for r in successful_exports)
            avg_processing_time = sum(r.processing_time for r in successful_exports) / len(successful_exports)
            print(f"   Total output size: {total_size:,} bytes")
            print(f"   Average processing time: {avg_processing_time:.2f}s")


async def demo_format_conversion():
    """Demonstrate format conversion capabilities."""
    print("\n🔄 Demo: Format Conversion")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_images = create_sample_images(temp_dir)
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        converter = FormatConverter(exporter)
        
        conversions = [
            (sample_images[0], ExportFormat.WEBP, "PNG → WebP"),
            (sample_images[1], ExportFormat.JPEG, "PNG → JPEG"),
            (sample_images[2], ExportFormat.PNG, "RGBA PNG → RGB PNG"),
        ]
        
        for input_path, target_format, description in conversions:
            print(f"\n🔄 Converting: {description}")
            
            original_size = os.path.getsize(input_path)
            
            result = await converter.convert_format(
                input_path,
                target_format,
                QualityLevel.HIGH,
                preserve_metadata=True
            )
            
            if result.success:
                compression_ratio = original_size / result.file_size
                print(f"   ✅ Conversion successful!")
                print(f"   Original size: {original_size:,} bytes")
                print(f"   New size: {result.file_size:,} bytes")
                print(f"   Compression ratio: {compression_ratio:.2f}x")
                print(f"   Processing time: {result.processing_time:.2f}s")
            else:
                print(f"   ❌ Conversion failed: {result.error_message}")


async def demo_metadata_handling():
    """Demonstrate metadata handling."""
    print("\n📋 Demo: Metadata Handling")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_images = create_sample_images(temp_dir)
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        # Create rich metadata
        metadata = ContentMetadata(
            title="Demo Image with Metadata",
            description="This is a demonstration image with comprehensive metadata",
            creator="ScrollIntel Content Export Demo",
            generation_prompt="Create a beautiful gradient image for testing",
            model_used="demo_model_v1.0",
            tags=["demo", "gradient", "test", "metadata"],
            copyright="© 2024 ScrollIntel Demo",
            custom_fields={
                "demo_version": "1.0",
                "export_purpose": "demonstration",
                "quality_target": "high"
            }
        )
        
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.HIGH,
            preserve_metadata=True,
            watermark="DEMO"
        )
        
        print("📝 Exporting image with comprehensive metadata...")
        
        result = await exporter.export_content(
            sample_images[1],  # Use gradient image
            settings,
            metadata,
            "demo_with_metadata.jpg"
        )
        
        if result.success:
            print(f"✅ Export with metadata successful!")
            print(f"   Output: {result.output_path}")
            print(f"   File size: {result.file_size:,} bytes")
            print(f"   Metadata preserved: {settings.preserve_metadata}")
            print(f"   Watermark applied: {settings.watermark}")
        else:
            print(f"❌ Export failed: {result.error_message}")


async def demo_web_optimization():
    """Demonstrate web optimization."""
    print("\n🌐 Demo: Web Optimization")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a large image for web optimization
        large_image_path = os.path.join(temp_dir, "large_image.png")
        large_image = Image.new('RGB', (2400, 1800), color='blue')
        large_image.save(large_image_path)
        
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        converter = FormatConverter(exporter)
        
        original_size = os.path.getsize(large_image_path)
        
        print(f"🖼️  Original image: 2400x1800 pixels, {original_size:,} bytes")
        print("🔧 Optimizing for web delivery...")
        
        result = await converter.optimize_for_web(large_image_path)
        
        if result.success:
            # Check output image dimensions
            output_image = Image.open(result.output_path)
            compression_ratio = original_size / result.file_size
            
            print(f"✅ Web optimization successful!")
            print(f"   Output format: {result.format.value.upper()}")
            print(f"   Output dimensions: {output_image.size[0]}x{output_image.size[1]} pixels")
            print(f"   Original size: {original_size:,} bytes")
            print(f"   Optimized size: {result.file_size:,} bytes")
            print(f"   Compression ratio: {compression_ratio:.2f}x")
            print(f"   Size reduction: {((original_size - result.file_size) / original_size * 100):.1f}%")
        else:
            print(f"❌ Web optimization failed: {result.error_message}")


async def demo_archive_creation():
    """Demonstrate archive creation."""
    print("\n📁 Demo: Archive Creation")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_images = create_sample_images(temp_dir)
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        # Create ZIP archive
        print("📦 Creating ZIP archive of sample images...")
        
        zip_result = await exporter.create_archive(
            sample_images,
            ExportFormat.ZIP,
            "sample_images.zip"
        )
        
        if zip_result.success:
            print(f"✅ ZIP archive created successfully!")
            print(f"   Archive: {zip_result.output_path}")
            print(f"   Archive size: {zip_result.file_size:,} bytes")
            print(f"   Files archived: {zip_result.quality_metrics['files_archived']}")
        else:
            print(f"❌ ZIP archive creation failed: {zip_result.error_message}")
        
        # Create TAR archive
        print("\n📦 Creating TAR archive of sample images...")
        
        tar_result = await exporter.create_archive(
            sample_images,
            ExportFormat.TAR,
            "sample_images.tar.gz"
        )
        
        if tar_result.success:
            print(f"✅ TAR archive created successfully!")
            print(f"   Archive: {tar_result.output_path}")
            print(f"   Archive size: {tar_result.file_size:,} bytes")
            print(f"   Files archived: {tar_result.quality_metrics['files_archived']}")
        else:
            print(f"❌ TAR archive creation failed: {tar_result.error_message}")


async def demo_format_capabilities():
    """Demonstrate format capabilities inquiry."""
    print("\n📊 Demo: Format Capabilities")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        content_types = ["image", "video"]
        
        for content_type in content_types:
            print(f"\n📋 Supported formats for {content_type} content:")
            formats = exporter.get_supported_formats(content_type)
            
            for format_type in formats:
                info = exporter.get_format_info(format_type)
                print(f"   {format_type.value.upper():6} | {info.get('description', 'No description')}")
                
                if 'supports_transparency' in info:
                    transparency = "✅" if info['supports_transparency'] else "❌"
                    print(f"          | Transparency: {transparency}")
                
                if 'supports_animation' in info:
                    animation = "✅" if info['supports_animation'] else "❌"
                    print(f"          | Animation: {animation}")


async def main():
    """Run all demonstrations."""
    print("🚀 ScrollIntel Content Export System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive content export and")
    print("format conversion capabilities of the ScrollIntel system.")
    print("=" * 60)
    
    try:
        await demo_basic_export()
        await demo_quality_comparison()
        await demo_batch_export()
        await demo_format_conversion()
        await demo_metadata_handling()
        await demo_web_optimization()
        await demo_archive_creation()
        await demo_format_capabilities()
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("The content export system demonstrates:")
        print("✅ Multiple format support (JPEG, PNG, WebP, MP4, etc.)")
        print("✅ Quality level control and compression options")
        print("✅ Batch processing capabilities")
        print("✅ Metadata preservation and embedding")
        print("✅ Web optimization features")
        print("✅ Archive creation (ZIP, TAR)")
        print("✅ Format conversion accuracy")
        print("✅ Error handling and recovery")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())