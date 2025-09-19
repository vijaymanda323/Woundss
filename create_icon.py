#!/usr/bin/env python3
"""
Create a simple app icon for the wound healing tracker
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_app_icon():
    # Create a 1024x1024 image (standard app icon size)
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    
    # Create a gradient-like background
    for i in range(size):
        # Create a blue gradient
        color = (int(102 + i * 0.1), int(126 + i * 0.1), int(234 + i * 0.05), 255)
        draw.line([(0, i), (size, i)], fill=color)
    
    # Draw a medical cross in the center
    cross_size = 300
    cross_thickness = 60
    center_x, center_y = size // 2, size // 2
    
    # Vertical part of cross
    draw.rectangle([
        center_x - cross_thickness // 2,
        center_y - cross_size // 2,
        center_x + cross_thickness // 2,
        center_y + cross_size // 2
    ], fill=(255, 255, 255, 255))
    
    # Horizontal part of cross
    draw.rectangle([
        center_x - cross_size // 2,
        center_y - cross_thickness // 2,
        center_x + cross_size // 2,
        center_y + cross_thickness // 2
    ], fill=(255, 255, 255, 255))
    
    # Add a subtle border
    border_width = 8
    draw.rectangle([0, 0, size-1, size-1], outline=(255, 255, 255, 200), width=border_width)
    
    # Save the icon
    icon_path = 'assets/icon.png'
    img.save(icon_path, 'PNG')
    print(f"✅ Created app icon: {icon_path}")
    
    # Also create a smaller version for different uses
    small_size = 512
    small_img = img.resize((small_size, small_size), Image.Resampling.LANCZOS)
    small_img.save('assets/images/icon.png', 'PNG')
    print(f"✅ Created small icon: assets/images/icon.png")
    
    return icon_path

if __name__ == "__main__":
    try:
        create_app_icon()
        print("🎉 App icon created successfully!")
    except ImportError:
        print("❌ PIL (Pillow) not installed. Installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'Pillow'], check=True)
        print("✅ Pillow installed. Creating icon...")
        create_app_icon()
        print("🎉 App icon created successfully!")
    except Exception as e:
        print(f"❌ Error creating icon: {e}")
        print("Creating a simple placeholder instead...")
        
        # Create a simple colored square as fallback
        img = Image.new('RGB', (1024, 1024), (102, 126, 234))
        img.save('assets/icon.png', 'PNG')
        print("✅ Created placeholder icon: assets/icon.png")




