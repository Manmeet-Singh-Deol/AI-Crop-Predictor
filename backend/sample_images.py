"""
Realistic Sample Leaf Image Generator
Creates ready-to-test synthetic crop leaf specimens for 1-click evaluation of diagnosis, Grad-CAM, and severity.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, List, Optional

def create_leaf_base(width: int = 400, height: int = 400, color: tuple = (46, 125, 50)) -> Image.Image:
    """Create a realistic organic leaf silhouette with main and lateral veins."""
    img = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw leaf blade (oval / teardrop organic polygon)
    cx, cy = width // 2, height // 2
    points = []
    num_pts = 60
    for i in range(num_pts):
        angle = (2 * np.pi * i) / num_pts
        # Asymmetrical leaf shape formula
        r = (width * 0.38) * (1.0 + 0.35 * np.cos(angle) - 0.15 * np.sin(2 * angle))
        x = cx + r * np.sin(angle)
        y = cy - r * np.cos(angle) * 1.15
        # Add slight serration jitter
        x += np.random.uniform(-3, 3)
        y += np.random.uniform(-3, 3)
        points.append((x, y))
        
    draw.polygon(points, fill=color)
    
    # Draw central midrib vein
    draw.line([(cx, cy + int(height * 0.4)), (cx, cy - int(height * 0.42))], fill=(color[0] + 30, color[1] + 35, color[2] + 15), width=4)
    
    # Draw secondary side veins
    for dy in range(-int(height * 0.35), int(height * 0.35), 35):
        y_pos = cy + dy
        draw.line([(cx, y_pos), (cx - 70, y_pos - 25)], fill=(color[0] + 25, color[1] + 30, color[2] + 10), width=2)
        draw.line([(cx, y_pos), (cx + 70, y_pos - 25)], fill=(color[0] + 25, color[1] + 30, color[2] + 10), width=2)
        
    return img

def generate_tomato_late_blight() -> Image.Image:
    """Generate Tomato leaf with dark water-soaked necrotic lesions and yellow halos."""
    base = create_leaf_base(400, 400, (46, 125, 50))
    draw = ImageDraw.Draw(base)
    
    # Yellow chlorotic halo zones
    draw.ellipse([100, 120, 240, 260], fill=(190, 180, 40))
    draw.ellipse([210, 220, 310, 320], fill=(185, 175, 35))
    
    # Dark brown/black necrotic lesions
    draw.ellipse([120, 140, 220, 240], fill=(55, 38, 25))
    draw.ellipse([225, 235, 295, 305], fill=(45, 30, 20))
    draw.ellipse([140, 260, 210, 330], fill=(60, 42, 28))
    
    # Smooth blur for realistic fungal progression
    return base.filter(ImageFilter.GaussianBlur(radius=1.5))

def generate_potato_early_blight() -> Image.Image:
    """Generate Potato leaf with concentric ring target-spot lesions."""
    base = create_leaf_base(400, 400, (56, 142, 60))
    draw = ImageDraw.Draw(base)
    
    # Target spot 1 (center right)
    draw.ellipse([180, 130, 280, 230], fill=(195, 175, 45))  # Halo
    draw.ellipse([195, 145, 265, 215], fill=(85, 55, 30))   # Outer ring
    draw.ellipse([210, 160, 250, 200], fill=(120, 85, 45))  # Mid ring
    draw.ellipse([222, 172, 238, 188], fill=(50, 30, 15))   # Bullseye center
    
    # Target spot 2 (lower left)
    draw.ellipse([110, 230, 190, 310], fill=(190, 170, 40))
    draw.ellipse([125, 245, 175, 295], fill=(80, 50, 25))
    draw.ellipse([140, 260, 160, 280], fill=(110, 75, 35))
    
    return base.filter(ImageFilter.GaussianBlur(radius=1.2))

def generate_apple_scab() -> Image.Image:
    """Generate Apple leaf with olive-brown velvety circular scabby spots."""
    base = create_leaf_base(400, 400, (67, 160, 71))
    draw = ImageDraw.Draw(base)
    
    # Scab spots
    scab_color = (62, 54, 30)
    for x, y, r in [(160, 160, 28), (230, 200, 22), (180, 250, 35), (250, 130, 18), (130, 220, 15)]:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=scab_color)
        # Irregular feathered borders
        for _ in range(8):
            jx = x + np.random.randint(-r - 3, r + 3)
            jy = y + np.random.randint(-r - 3, r + 3)
            draw.ellipse([jx - 4, jy - 4, jx + 4, jy + 4], fill=(75, 68, 38))
            
    return base.filter(ImageFilter.GaussianBlur(radius=1.0))

def generate_corn_blight() -> Image.Image:
    """Generate elongated Corn leaf blade with cigar-shaped tan lesions."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw linear elongated corn leaf strap
    draw.polygon([(150, 20), (250, 20), (270, 380), (130, 380)], fill=(76, 175, 80))
    # Parallel veins
    for x in range(140, 265, 15):
        draw.line([(x, 20), (x + 10, 380)], fill=(90, 195, 95), width=2)
        
    # Cigar shaped tan lesions
    draw.ellipse([160, 120, 220, 240], fill=(188, 170, 120))  # Tan lesion 1
    draw.ellipse([170, 140, 210, 220], fill=(130, 110, 70))   # Dark center
    draw.ellipse([180, 260, 240, 340], fill=(188, 170, 120))  # Tan lesion 2
    
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))

def generate_squash_powdery_mildew() -> Image.Image:
    """Generate Squash/Cucurbit broad leaf with white powdery talcum patches."""
    base = create_leaf_base(400, 400, (46, 125, 50))
    draw = ImageDraw.Draw(base)
    
    mildew_white = (235, 238, 235)
    draw.ellipse([130, 120, 200, 190], fill=mildew_white)
    draw.ellipse([210, 150, 290, 230], fill=mildew_white)
    draw.ellipse([140, 220, 230, 310], fill=mildew_white)
    
    return base.filter(ImageFilter.GaussianBlur(radius=2.5))

def generate_healthy_tomato() -> Image.Image:
    """Generate clean, lush, disease-free Tomato leaf."""
    return create_leaf_base(400, 400, (46, 125, 50)).filter(ImageFilter.GaussianBlur(radius=1.0))

def generate_sugarcane_red_rot() -> Image.Image:
    """Generate Sugarcane leaf with characteristic red midrib lesion and straw-colored necrotic center."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw long linear sugarcane blade
    draw.polygon([(140, 20), (260, 20), (280, 380), (120, 380)], fill=(40, 130, 50))
    
    # White central midrib
    draw.line([(200, 20), (200, 380)], fill=(225, 240, 220), width=12)
    
    # Red rot longitudinal crimson midrib lesion
    draw.rectangle([195, 120, 205, 280], fill=(165, 20, 25))
    # Straw colored necrotic center with dark dots
    draw.rectangle([197, 150, 203, 250], fill=(210, 185, 130))
    for y_dot in range(160, 240, 12):
        draw.ellipse([198, y_dot, 202, y_dot + 4], fill=(30, 15, 10))
        
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))

def generate_sugarcane_rust() -> Image.Image:
    """Generate Sugarcane leaf blade with parallel brownish-orange rust pustules."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    
    # Sugarcane linear blade
    draw.polygon([(140, 20), (260, 20), (280, 380), (120, 380)], fill=(50, 140, 60))
    draw.line([(200, 20), (200, 380)], fill=(210, 235, 205), width=10)
    
    # Orange and brown rust pustules parallel to veins
    for px, py in [(160, 90), (170, 140), (230, 110), (240, 170), (165, 210), (225, 260), (175, 300), (235, 320)]:
        draw.ellipse([px - 4, py - 12, px + 4, py + 12], fill=(215, 120, 25)) # Orange border
        draw.ellipse([px - 2, py - 8, px + 2, py + 8], fill=(110, 50, 15))    # Dark brown uredinium
        
    return img.filter(ImageFilter.GaussianBlur(radius=0.9))

def generate_wheat_yellow_rust() -> Image.Image:
    """Generate narrow linear Wheat leaf with yellow stripe rust pustule chains."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # Long erect linear blade
    draw.polygon([(160, 10), (240, 10), (240, 390), (160, 390)], fill=(70, 155, 60))
    draw.line([(200, 10), (200, 390)], fill=(150, 195, 130), width=3)
    # Yellow stripe rust linear chains
    for x_stripe in [175, 188, 212, 225]:
        for y_dot in range(60, 340, 16):
            draw.ellipse([x_stripe - 3, y_dot - 5, x_stripe + 3, y_dot + 5], fill=(245, 195, 25))
            draw.ellipse([x_stripe - 1, y_dot - 2, x_stripe + 1, y_dot + 2], fill=(255, 225, 60))
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))

def generate_cotton_bacterial_blight() -> Image.Image:
    """Generate Cotton 3-lobed palmate leaf with dark angular water-soaked spots."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # 3-lobed cotton leaf outline
    cx, cy = 200, 200
    pts = [
        (200, 40), (230, 120), (320, 100), (270, 200), (330, 280),
        (240, 270), (200, 360), (160, 270), (70, 280), (130, 200),
        (80, 100), (170, 120)
    ]
    draw.polygon(pts, fill=(50, 135, 55))
    # Radial primary veins
    for px, py in [(200, 40), (320, 100), (330, 280), (70, 280), (80, 100)]:
        draw.line([(cx, cy + 30), (px, py)], fill=(90, 175, 95), width=3)
    # Angular water-soaked lesions bounded by veins
    draw.polygon([(180, 140), (210, 145), (205, 175), (175, 165)], fill=(40, 25, 15))
    draw.polygon([(220, 180), (255, 190), (245, 220), (215, 210)], fill=(45, 30, 20))
    draw.polygon([(135, 210), (165, 215), (155, 245), (125, 235)], fill=(35, 20, 10))
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))

def generate_banana_black_sigatoka() -> Image.Image:
    """Generate broad paddle-shaped Banana leaf with black sigatoka necrotic streaks."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # Broad paddle blade
    draw.ellipse([90, 30, 310, 370], fill=(45, 145, 55))
    # Heavy thick midrib
    draw.line([(200, 20), (200, 380)], fill=(210, 240, 195), width=12)
    # Dense parallel lateral venation
    for y in range(50, 350, 20):
        draw.line([(200, y), (105, y - 15)], fill=(65, 165, 75), width=2)
        draw.line([(200, y), (295, y - 15)], fill=(65, 165, 75), width=2)
    # Sigatoka dark brown/black necrotic streaks with yellow halos
    for sx, sy in [(145, 120), (250, 160), (140, 220), (260, 260), (150, 300)]:
        draw.ellipse([sx - 28, sy - 10, sx + 28, sy + 10], fill=(195, 185, 45)) # Yellow halo
        draw.ellipse([sx - 20, sy - 6, sx + 20, sy + 6], fill=(40, 28, 18))      # Black necrotic center
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))

def generate_coffee_leaf_rust() -> Image.Image:
    """Generate dark glossy Coffee leaf with bright orange powdery rust patches."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # Elliptical glossy leaf
    draw.ellipse([110, 40, 290, 360], fill=(28, 105, 40))
    draw.line([(200, 35), (200, 365)], fill=(75, 145, 80), width=4)
    # Orange powdery rust pustule patches
    for rx, ry in [(155, 130), (245, 170), (160, 230), (240, 270)]:
        draw.ellipse([rx - 18, ry - 18, rx + 18, ry + 18], fill=(235, 130, 20))
        draw.ellipse([rx - 10, ry - 10, rx + 10, ry + 10], fill=(255, 175, 40))
    return img.filter(ImageFilter.GaussianBlur(radius=1.1))

# Sample registry
SAMPLES_METADATA: List[Dict[str, Any]] = [
    {
        "id": "sample_wheat_yellow_rust",
        "title": "Wheat Yellow Stripe Rust",
        "crop": "Wheat",
        "expected_class": "Wheat___Yellow_Rust",
        "pathogen": "Puccinia striiformis (Fungal)",
        "description": "Diagnostic parallel bright yellow powdery spore chains along leaf veins.",
        "generator": generate_wheat_yellow_rust
    },
    {
        "id": "sample_cotton_blight",
        "title": "Cotton Bacterial Blight",
        "crop": "Cotton",
        "expected_class": "Cotton___Bacterial_Blight",
        "pathogen": "Xanthomonas citri pv. malvacearum (Bacterial)",
        "description": "Angular water-soaked dark brown spots delimited by leaf veins.",
        "generator": generate_cotton_bacterial_blight
    },
    {
        "id": "sample_banana_sigatoka",
        "title": "Banana Black Sigatoka",
        "crop": "Banana",
        "expected_class": "Banana___Black_Sigatoka",
        "pathogen": "Pseudocercospora fijiensis (Fungal)",
        "description": "Dark black elliptical necrotic streaks with chlorotic halos causing canopy collapse.",
        "generator": generate_banana_black_sigatoka
    },
    {
        "id": "sample_coffee_rust",
        "title": "Coffee Leaf Rust",
        "crop": "Coffee",
        "expected_class": "Coffee___Leaf_Rust",
        "pathogen": "Hemileia vastatrix (Fungal)",
        "description": "Vibrant orange powdery spore colonies causing severe defoliation.",
        "generator": generate_coffee_leaf_rust
    },
    {
        "id": "sample_sugarcane_red_rot",
        "title": "Sugarcane Red Rot",
        "crop": "Sugarcane",
        "expected_class": "Sugarcane___Red_Rot",
        "pathogen": "Colletotrichum falcatum (Fungal)",
        "description": "Cancer of sugarcane showing diagnostic red midrib lesion with white center.",
        "generator": generate_sugarcane_red_rot
    },
    {
        "id": "sample_sugarcane_rust",
        "title": "Sugarcane Brown Rust",
        "crop": "Sugarcane",
        "expected_class": "Sugarcane___Rust",
        "pathogen": "Puccinia melanocephala (Fungal)",
        "description": "Elongated brownish-orange powdery rust pustules across parallel veins.",
        "generator": generate_sugarcane_rust
    },
    {
        "id": "sample_tomato_late_blight",
        "title": "Tomato Late Blight",
        "crop": "Tomato",
        "expected_class": "Tomato___Late_blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "description": "Large water-soaked dark brown necrotic lesions with chlorotic yellow boundaries.",
        "generator": generate_tomato_late_blight
    },
    {
        "id": "sample_potato_early_blight",
        "title": "Potato Early Blight",
        "crop": "Potato",
        "expected_class": "Potato___Early_blight",
        "pathogen": "Alternaria solani (Fungal)",
        "description": "Characteristic bullseye concentric rings with yellow surrounding halos.",
        "generator": generate_potato_early_blight
    },
    {
        "id": "sample_apple_scab",
        "title": "Apple Scab",
        "crop": "Apple",
        "expected_class": "Apple___Apple_scab",
        "pathogen": "Venturia inaequalis (Fungal)",
        "description": "Olive-brown velvety scabby lesions deforming leaf tissue.",
        "generator": generate_apple_scab
    },
    {
        "id": "sample_corn_nclb",
        "title": "Corn Northern Leaf Blight",
        "crop": "Corn (Maize)",
        "expected_class": "Corn_(maize)___Northern_Leaf_Blight",
        "pathogen": "Exserohilum turcicum (Fungal)",
        "description": "Long elliptical tan lesions stretching across parallel veins.",
        "generator": generate_corn_blight
    },
    {
        "id": "sample_squash_powdery_mildew",
        "title": "Squash Powdery Mildew",
        "crop": "Squash",
        "expected_class": "Squash___Powdery_mildew",
        "pathogen": "Podosphaera xanthii (Fungal)",
        "description": "Superficial white powdery talcum-like mycelial patches.",
        "generator": generate_squash_powdery_mildew
    },
    {
        "id": "sample_healthy_tomato",
        "title": "Tomato Healthy Leaf",
        "crop": "Tomato",
        "expected_class": "Tomato___healthy",
        "pathogen": "None (Healthy)",
        "description": "Vibrant uniform green foliage without blemishes or lesions.",
        "generator": generate_healthy_tomato
    }
]

def get_all_samples_with_thumbnails() -> List[Dict[str, Any]]:
    """Return all sample leaf choices with pre-rendered base64 thumbnails."""
    results = []
    for s in SAMPLES_METADATA:
        img = s["generator"]()
        buf = io.BytesIO()
        img.resize((140, 140)).save(buf, format="JPEG", quality=85)
        b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        
        results.append({
            "id": s["id"],
            "title": s["title"],
            "crop": s["crop"],
            "expected_class": s["expected_class"],
            "pathogen": s["pathogen"],
            "description": s["description"],
            "thumbnail": b64
        })
    return results

def get_sample_image(sample_id: str) -> Image.Image:
    """Retrieve PIL Image for a specific sample ID."""
    for s in SAMPLES_METADATA:
        if s["id"] == sample_id:
            return s["generator"]()
    return generate_tomato_late_blight()

def get_sample_expected_class(sample_id: str) -> Optional[str]:
    """Retrieve the ground truth class name for a sample ID."""
    for s in SAMPLES_METADATA:
        if s["id"] == sample_id:
            return s["expected_class"]
    return None
