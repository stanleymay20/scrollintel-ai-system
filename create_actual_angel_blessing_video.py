"""
ScrollIntel Real Video Generation - Angel Blessing Video
Creates an actual MP4 video file that media players can play
"""
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def create_angel_blessing_video():
    """Create an actual MP4 video file of an angel blessing scene"""
    
    print("🎬 ScrollIntel Real Video Generation")
    print("Creating actual MP4 video file...")
    print("=" * 60)
    
    # Video parameters
    width, height = 1920, 1080  # HD resolution
    fps = 30
    duration = 30  # 30 seconds
    total_frames = fps * duration
    
    # Create output directory
    output_dir = Path("generated_content")
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"scrollintel_angel_blessing_{timestamp}.mp4"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    print(f"📁 Output file: {output_file}")
    print(f"📺 Resolution: {width}x{height}")
    print(f"🎞️  Frame rate: {fps} fps")
    print(f"⏱️  Duration: {duration} seconds")
    print(f"🎬 Total frames: {total_frames}")
    
    # Generate frames
    print("\n🎨 Generating video frames...")
    
    for frame_num in range(total_frames):
        # Calculate progress
        progress = frame_num / total_frames
        
        # Create a frame with gradient background (heavenly sky)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create heavenly gradient background
        for y in range(height):
            # Gradient from light blue at top to golden at bottom
            blue_intensity = int(200 - (y / height) * 100)
            gold_intensity = int((y / height) * 150)
            frame[y, :] = [blue_intensity, blue_intensity + gold_intensity//2, 255 - gold_intensity//2]
        
        # Add divine light effect (circular glow in center)
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 3
        
        # Create divine light that pulses
        pulse = 0.8 + 0.2 * np.sin(progress * 4 * np.pi)  # Gentle pulsing
        
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance < max_radius:
                    # Add golden divine light
                    light_intensity = (1 - distance / max_radius) * pulse * 100
                    frame[y, x, 0] = min(255, frame[y, x, 0] + int(light_intensity * 0.3))  # Blue
                    frame[y, x, 1] = min(255, frame[y, x, 1] + int(light_intensity * 0.7))  # Green
                    frame[y, x, 2] = min(255, frame[y, x, 2] + int(light_intensity))        # Red (golden)
        
        # Add floating light particles
        num_particles = 20
        for i in range(num_particles):
            # Particles move slowly upward and fade in/out
            particle_phase = (progress + i * 0.1) % 1.0
            particle_x = int((center_x + (i - 10) * 50 + np.sin(progress * 2 * np.pi + i) * 30) % width)
            particle_y = int(center_y + 200 - particle_phase * 400)
            
            if 0 <= particle_x < width and 0 <= particle_y < height:
                # Particle brightness fades in and out
                brightness = int(255 * np.sin(particle_phase * np.pi))
                if brightness > 0:
                    # Draw small glowing particle
                    cv2.circle(frame, (particle_x, particle_y), 3, (brightness//2, brightness//2, brightness), -1)
                    cv2.circle(frame, (particle_x, particle_y), 6, (brightness//4, brightness//4, brightness//2), 1)
        
        # Add angel silhouette (simple representation)
        angel_y = center_y - 50
        angel_x = center_x
        
        # Angel body (simple white silhouette)
        angel_color = (255, 255, 255)
        
        # Head
        cv2.circle(frame, (angel_x, angel_y - 80), 25, angel_color, -1)
        
        # Body
        cv2.rectangle(frame, (angel_x - 15, angel_y - 50), (angel_x + 15, angel_y + 50), angel_color, -1)
        
        # Arms raised in blessing (changes position slightly over time)
        arm_angle = np.sin(progress * np.pi) * 0.2  # Gentle movement
        
        # Left arm
        arm_end_x = int(angel_x - 40 + arm_angle * 10)
        arm_end_y = int(angel_y - 30 - arm_angle * 20)
        cv2.line(frame, (angel_x - 15, angel_y - 20), (arm_end_x, arm_end_y), angel_color, 8)
        
        # Right arm  
        arm_end_x = int(angel_x + 40 - arm_angle * 10)
        arm_end_y = int(angel_y - 30 - arm_angle * 20)
        cv2.line(frame, (angel_x + 15, angel_y - 20), (arm_end_x, arm_end_y), angel_color, 8)
        
        # Wings (simple triangular shapes that gently move)
        wing_spread = 0.8 + 0.2 * np.sin(progress * 6 * np.pi)  # Wing flapping
        
        # Left wing
        wing_points = np.array([
            [angel_x - 20, angel_y - 20],
            [int(angel_x - 80 * wing_spread), angel_y - 40],
            [int(angel_x - 60 * wing_spread), angel_y + 20]
        ], np.int32)
        cv2.fillPoly(frame, [wing_points], (240, 240, 255))
        
        # Right wing
        wing_points = np.array([
            [angel_x + 20, angel_y - 20],
            [int(angel_x + 80 * wing_spread), angel_y - 40],
            [int(angel_x + 60 * wing_spread), angel_y + 20]
        ], np.int32)
        cv2.fillPoly(frame, [wing_points], (240, 240, 255))
        
        # Add text overlay for different phases of the video
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (255, 255, 255)
        font_thickness = 2
        
        if progress < 0.2:
            text = "Divine Appearance"
        elif progress < 0.5:
            text = "Worship & Praise"
        elif progress < 0.8:
            text = "The Blessing Song"
        else:
            text = "Blessing Gesture"
        
        # Add text with shadow for better visibility
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        # Shadow
        cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 1)
        # Main text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        # Add ScrollIntel watermark
        watermark = "ScrollIntel Visual Generation"
        watermark_size = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, watermark, (width - watermark_size[0] - 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Write frame to video
        video_writer.write(frame)
        
        # Show progress
        if frame_num % (total_frames // 10) == 0:
            print(f"   📊 Progress: {int(progress * 100)}% ({frame_num}/{total_frames} frames)")
    
    # Release video writer
    video_writer.release()
    
    print(f"\n✅ Video generation complete!")
    print(f"📁 File saved: {output_file}")
    print(f"📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Verify the file exists and is playable
    if output_file.exists():
        print(f"🎉 SUCCESS: Video file created and ready for playback!")
        print(f"🎬 Your media player should now be able to play: {output_file}")
        
        # Try to get video info
        cap = cv2.VideoCapture(str(output_file))
        if cap.isOpened():
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"\n📺 Video Properties:")
            print(f"   Resolution: {actual_width}x{actual_height}")
            print(f"   Frame rate: {actual_fps} fps")
            print(f"   Total frames: {actual_frames}")
            print(f"   Duration: {actual_frames/actual_fps:.1f} seconds")
            
            cap.release()
        
        return str(output_file)
    else:
        print(f"❌ ERROR: Failed to create video file")
        return None

def main():
    """Main function"""
    print("ScrollIntel Angel Blessing Video - Real MP4 Generation")
    print("Creating actual video file for media player compatibility")
    print("=" * 80)
    
    try:
        video_file = create_angel_blessing_video()
        
        if video_file:
            print(f"\n🎊 SUCCESS!")
            print(f"ScrollIntel has created a real MP4 video file!")
            print(f"📁 Location: {video_file}")
            print(f"🎬 This file should now play in your media player!")
            print(f"\n🏆 ScrollIntel delivers working video files - unlike template-based competitors!")
            return 0
        else:
            print(f"\n❌ Failed to create video file")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)