"""
Generate the team's approved fixed automotive studio background image.
Creates a high-definition (1920x1080) automotive showroom cyclorama with:
- Curved cyclorama cove wall (smooth gradient)
- Overhead softbox diffuser lighting
- Polished dark reflective epoxy floor
- Subtle ambient LED light bars
"""

import os
import cv2
import numpy as np


def generate_approved_studio(
    width: int = 1920,
    height: int = 1080,
    output_path: str = "studio_assets/backgrounds/approved_studio.png"
) -> np.ndarray:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Base canvas (BGR)
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    horizon_y = int(height * 0.62)  # Floor line at ~62%
    
    # 2. Cyclorama Wall (y: 0 to horizon_y)
    # Dark charcoal gradient with overhead softbox glow
    xx, yy = np.meshgrid(np.arange(width), np.arange(horizon_y))
    
    # Overhead softbox center at (width // 2, 0)
    dist_from_softbox = np.sqrt(((xx - width / 2) / (width * 0.55)) ** 2 + ((yy - 40) / (horizon_y * 0.9)) ** 2)
    softbox_glow = np.clip(1.0 - dist_from_softbox * 0.75, 0.0, 1.0)
    softbox_glow = softbox_glow ** 2.2
    
    # Vertical wall gradient (dark top, slightly illuminated cove transition)
    wall_v_grad = (yy / horizon_y) ** 1.5
    
    # Wall base colors (deep charcoal navy)
    # BGR format: dark blue-grey (28, 26, 25) to (50, 48, 46)
    wall_b = 22.0 + 24.0 * wall_v_grad + 45.0 * softbox_glow
    wall_g = 24.0 + 26.0 * wall_v_grad + 48.0 * softbox_glow
    wall_r = 28.0 + 28.0 * wall_v_grad + 52.0 * softbox_glow
    
    img[:horizon_y, :, 0] = wall_b
    img[:horizon_y, :, 1] = wall_g
    img[:horizon_y, :, 2] = wall_r
    
    # 3. Soft Cove transition (blurring the horizon edge)
    cove_radius = 45
    cove_y1 = max(0, horizon_y - cove_radius)
    cove_y2 = min(height, horizon_y + cove_radius)
    
    # 4. Polished Floor (y: horizon_y to height)
    floor_h = height - horizon_y
    f_xx, f_yy = np.meshgrid(np.arange(width), np.arange(floor_h))
    
    # Floor depth perspective gradient (further back near horizon is darker, foreground has subtle ambient)
    floor_depth = f_yy / float(floor_h)
    
    # Floor softbox specular reflection pool
    floor_spec_dist = np.sqrt(((f_xx - width / 2) / (width * 0.42)) ** 2 + ((f_yy - floor_h * 0.25) / (floor_h * 0.65)) ** 2)
    floor_spec = np.clip(1.0 - floor_spec_dist * 0.85, 0.0, 1.0) ** 2.5
    
    # Floor base tones (rich, ultra-dark mirror finish)
    floor_b = 14.0 + 16.0 * floor_depth + 32.0 * floor_spec
    floor_g = 16.0 + 17.0 * floor_depth + 34.0 * floor_spec
    floor_r = 18.0 + 18.0 * floor_depth + 36.0 * floor_spec
    
    img[horizon_y:, :, 0] = floor_b
    img[horizon_y:, :, 1] = floor_g
    img[horizon_y:, :, 2] = floor_r
    
    # 5. Horizon cove blend
    # Create smooth transition between cove wall and floor
    for dy in range(-cove_radius, cove_radius):
        y = horizon_y + dy
        if 0 <= y < height:
            blend_factor = (dy + cove_radius) / (2.0 * cove_radius)
            # Smooth cosine blend
            blend_factor = 0.5 - 0.5 * np.cos(blend_factor * np.pi)
            wall_sample = img[max(0, horizon_y - 2), :, :]
            floor_sample = img[min(height - 1, horizon_y + 2), :, :]
            img[y, :, :] = wall_sample * (1.0 - blend_factor) + floor_sample * blend_factor
            
    # 6. Overhead Studio Light Fixture (Subtle LED softbox edge)
    softbox_w = int(width * 0.52)
    softbox_x1 = (width - softbox_w) // 2
    softbox_x2 = softbox_x1 + softbox_w
    # Soft rectangular light bar at the top ceiling
    cv2.line(img, (softbox_x1, 15), (softbox_x2, 15), (140, 145, 150), 3, cv2.LINE_AA)
    
    # Subtle ambient studio background grid lines (very faint showroom tiles)
    for tx in range(0, width, 160):
        # Floor perspective lines converging towards horizon center
        start_pt = (int(width / 2 + (tx - width / 2) * 0.15), horizon_y)
        end_pt = (tx, height)
        cv2.line(img, start_pt, end_pt, (25, 27, 30), 1, cv2.LINE_AA)
        
    for ty in [int(horizon_y + floor_h * 0.25), int(horizon_y + floor_h * 0.55), int(horizon_y + floor_h * 0.85)]:
        cv2.line(img, (0, ty), (width, ty), (25, 27, 30), 1, cv2.LINE_AA)

    # 7. Convert to uint8 and apply gentle gaussian filter for smooth studio lighting
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    img_uint8 = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    
    # Save the approved studio background
    cv2.imwrite(output_path, img_uint8)
    print(f"Generated approved studio background at {output_path} ({width}x{height})")
    return img_uint8


if __name__ == "__main__":
    generate_approved_studio()
