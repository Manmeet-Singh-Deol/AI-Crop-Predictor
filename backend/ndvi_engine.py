"""
Satellite NDVI & Multispectral Remote Sensing Engine for AgroAI
Provides field-scale canopy vigor mapping, zonal stress segmentation,
multi-temporal biomass curves, and Variable-Rate Application (VRA) fertilizer prescriptions.
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple

# Predefined iconic global farm parcels with real-world GIS coordinates
SAMPLE_FARM_FIELDS = [
    {
        "id": "punjab_wheat_field",
        "name": "Ludhiana Precision Wheat Parcel",
        "region": "Punjab, India",
        "country": "India",
        "lat": 30.9010,
        "lon": 75.8573,
        "crop": "Wheat",
        "area_hectares": 14.5,
        "soil_type": "Alluvial Loam",
        "planting_date": "2025-11-15",
        "mean_ndvi_base": 0.68,
        "stress_anomaly": "North-West Patch (Yellow Rust & Nitrogen Deficit)"
    },
    {
        "id": "salinas_strawberry_farm",
        "name": "Salinas Valley Berry Fields",
        "region": "California, USA",
        "country": "USA",
        "lat": 36.6777,
        "lon": -121.6555,
        "crop": "Strawberry",
        "area_hectares": 8.2,
        "soil_type": "Sandy Clay Loam",
        "planting_date": "2025-10-20",
        "mean_ndvi_base": 0.76,
        "stress_anomaly": "South-East Corner (Early Wilt & Drip Clogging)"
    },
    {
        "id": "iowa_corn_belt",
        "name": "Story County Corn & Soybean Block",
        "region": "Iowa, USA",
        "country": "USA",
        "lat": 42.0308,
        "lon": -93.6319,
        "crop": "Corn (Maize)",
        "area_hectares": 35.0,
        "soil_type": "Mollisol Prairie Loam",
        "planting_date": "2025-05-10",
        "mean_ndvi_base": 0.81,
        "stress_anomaly": "Central Swale (Excess Moisture & Leaf Blight Risk)"
    },
    {
        "id": "valencia_citrus_grove",
        "name": "Huerta de Valencia Orange Orchard",
        "region": "Valencia, Spain",
        "country": "Spain",
        "lat": 39.4699,
        "lon": -0.3763,
        "crop": "Citrus (Orange)",
        "area_hectares": 12.0,
        "soil_type": "Calcareous Sandy Loam",
        "planting_date": "2020-03-15",
        "mean_ndvi_base": 0.72,
        "stress_anomaly": "Eastern Terrace (Minor Citrus Greening / Aphid Vectors)"
    },
    {
        "id": "kenya_rift_valley_maize",
        "name": "Nakuru Highlands Maize Shamba",
        "region": "Rift Valley, Kenya",
        "country": "Kenya",
        "lat": -0.3031,
        "lon": 36.0800,
        "crop": "Corn (Maize)",
        "area_hectares": 6.5,
        "soil_type": "Volcanic Loam",
        "planting_date": "2025-03-01",
        "mean_ndvi_base": 0.65,
        "stress_anomaly": "South Ridge (Fall Armyworm & Low Moisture)"
    },
    {
        "id": "bordeaux_vineyard",
        "name": "Saint-Émilion Premier Cru Vineyard",
        "region": "Bordeaux, France",
        "country": "France",
        "lat": 44.8944,
        "lon": -0.1558,
        "crop": "Grape (Vineyard)",
        "area_hectares": 10.4,
        "soil_type": "Limestone Clay",
        "planting_date": "2018-04-10",
        "mean_ndvi_base": 0.69,
        "stress_anomaly": "Lower Slope (Powdery Mildew Early Incubation)"
    }
]

def ndvi_to_color(val: float) -> Tuple[int, int, int, str]:
    """
    Map an NDVI value (-1.0 to 1.0) to a standard scientific RGB color gradient.
    """
    clamped = max(-0.2, min(1.0, val))
    if clamped < 0.15:
        # Bare soil / water / rock / dead matter -> Crimson Red / Orange
        r, g, b = 220, 38, 38
        hex_code = "#dc2626"
    elif clamped < 0.35:
        # Severe stress -> Amber / Orange
        r, g, b = 234, 88, 12
        hex_code = "#ea580c"
    elif clamped < 0.50:
        # Mild / moderate stress -> Yellow / Chartreuse
        r, g, b = 234, 179, 8
        hex_code = "#eab308"
    elif clamped < 0.68:
        # Healthy vegetative canopy -> Bright Green
        r, g, b = 74, 222, 128
        hex_code = "#4ade80"
    elif clamped < 0.82:
        # High biomass / vigor -> Emerald Green
        r, g, b = 22, 163, 74
        hex_code = "#16a34a"
    else:
        # Peak dense canopy -> Deep Forest Green
        r, g, b = 5, 150, 105
        hex_code = "#059669"
        
    return r, g, b, hex_code

def ndwi_to_color(val: float) -> Tuple[int, int, int, str]:
    """Map NDWI moisture value (-1.0 to 1.0) to aquatic/moisture blue gradient."""
    if val < -0.1:
        return 239, 68, 68, "#ef4444"    # Water Deficit
    elif val < 0.1:
        return 245, 158, 11, "#f59e0b"   # Low Moisture
    elif val < 0.3:
        return 56, 189, 248, "#38bdf8"   # Optimal Moisture
    else:
        return 14, 165, 233, "#0ea5e9"   # High / Saturated Moisture

def generate_field_raster_matrix(
    lat: float,
    lon: float,
    mean_ndvi_target: float = 0.72,
    grid_size: int = 24,
    crop: str = "Wheat"
) -> Dict[str, Any]:
    """
    Generate high-resolution geo-referenced raster matrix ($24 \times 24$ cells)
    simulating Sentinel-2 10m spatial resolution imagery over the specified farm coordinates.
    """
    # Deterministic pseudo-random seed based on coordinate hash for consistency
    coord_seed = int((abs(lat) * 10000 + abs(lon) * 1000) % 999999)
    rng = random.Random(coord_seed)
    
    delta_coord = 0.0035  # Approximate 400m field bounding box
    lat_min = lat - (delta_coord / 2.0)
    lon_min = lon - (delta_coord / 2.0)
    step = delta_coord / float(grid_size)
    
    # Anomaly epicenter for spatial stress patterning
    anomaly_x = rng.randint(4, 10)
    anomaly_y = rng.randint(4, 10)
    anomaly_radius = rng.uniform(3.5, 6.0)
    
    cells = []
    ndvi_values = []
    ndwi_values = []
    evi_values = []
    savi_values = []
    
    zone_counts = {"high": 0, "moderate": 0, "severe": 0}
    
    for row in range(grid_size):
        row_cells = []
        for col in range(grid_size):
            cell_lat = lat_min + (row * step)
            cell_lon = lon_min + (col * step)
            
            # Base field spatial gradient + noise
            dist_to_center = math.sqrt((row - grid_size/2)**2 + (col - grid_size/2)**2)
            edge_falloff = math.sin((row / grid_size) * math.pi) * math.sin((col / grid_size) * math.pi)
            
            # Anomaly distance calculation
            dist_to_anomaly = math.sqrt((col - anomaly_x)**2 + (row - anomaly_y)**2)
            
            base_ndvi = mean_ndvi_target * (0.88 + 0.24 * edge_falloff) + rng.uniform(-0.04, 0.04)
            
            # Inject localized pathology/moisture stress pocket
            if dist_to_anomaly < anomaly_radius:
                severity_factor = (1.0 - (dist_to_anomaly / anomaly_radius)) * 0.45
                base_ndvi -= severity_factor
                
            ndvi_val = round(max(0.12, min(0.92, base_ndvi)), 3)
            ndwi_val = round(max(-0.25, min(0.45, (ndvi_val * 0.55) - 0.12 + rng.uniform(-0.03, 0.03))), 3)
            evi_val = round(max(0.10, min(0.88, ndvi_val * 0.92 - 0.05)), 3)
            savi_val = round(max(0.10, min(0.85, ndvi_val * 0.88)), 3)
            
            r, g, b, hex_c = ndvi_to_color(ndvi_val)
            _, _, _, ndwi_hex = ndwi_to_color(ndwi_val)
            
            # Classify into 3 precision management zones
            if ndvi_val >= 0.65:
                zone = "high_vigor"
                zone_label = "🟢 High Vigor"
                zone_counts["high"] += 1
            elif ndvi_val >= 0.40:
                zone = "moderate_stress"
                zone_label = "🟡 Moderate Stress"
                zone_counts["moderate"] += 1
            else:
                zone = "severe_anomaly"
                zone_label = "🔴 Severe Anomaly / Deficit"
                zone_counts["severe"] += 1
                
            cell_data = {
                "row": row,
                "col": col,
                "lat": round(cell_lat, 6),
                "lon": round(cell_lon, 6),
                "ndvi": ndvi_val,
                "ndwi": ndwi_val,
                "evi": evi_val,
                "savi": savi_val,
                "color_ndvi": hex_c,
                "color_ndwi": ndwi_hex,
                "zone": zone,
                "zone_label": zone_label
            }
            row_cells.append(cell_data)
            ndvi_values.append(ndvi_val)
            ndwi_values.append(ndwi_val)
            evi_values.append(evi_val)
            savi_values.append(savi_val)
            
        cells.append(row_cells)
        
    total_cells = grid_size * grid_size
    mean_ndvi = round(sum(ndvi_values) / total_cells, 3)
    max_ndvi = round(max(ndvi_values), 3)
    min_ndvi = round(min(ndvi_values), 3)
    
    # Calculate Standard Deviation & Uniformity Index
    variance = sum((x - mean_ndvi) ** 2 for x in ndvi_values) / total_cells
    std_dev = round(math.sqrt(variance), 3)
    uniformity_score = round(max(50.0, 100.0 - (std_dev * 180.0)), 1)
    
    pct_high = round((zone_counts["high"] / total_cells) * 100.0, 1)
    pct_mod = round((zone_counts["moderate"] / total_cells) * 100.0, 1)
    pct_sev = round((zone_counts["severe"] / total_cells) * 100.0, 1)
    
    return {
        "grid_size": grid_size,
        "bounds": {
            "lat_min": round(lat_min, 6),
            "lat_max": round(lat_min + delta_coord, 6),
            "lon_min": round(lon_min, 6),
            "lon_max": round(lon_min + delta_coord, 6)
        },
        "cells": cells,
        "statistics": {
            "mean_ndvi": mean_ndvi,
            "max_ndvi": max_ndvi,
            "min_ndvi": min_ndvi,
            "mean_ndwi": round(sum(ndwi_values) / total_cells, 3),
            "mean_evi": round(sum(evi_values) / total_cells, 3),
            "mean_savi": round(sum(savi_values) / total_cells, 3),
            "standard_deviation": std_dev,
            "field_uniformity_score": uniformity_score
        },
        "zonal_breakdown": {
            "high_vigor_pct": pct_high,
            "moderate_stress_pct": pct_mod,
            "severe_anomaly_pct": pct_sev,
            "high_vigor_cells": zone_counts["high"],
            "moderate_stress_cells": zone_counts["moderate"],
            "severe_anomaly_cells": zone_counts["severe"]
        }
    }

def generate_vra_fertilizer_prescription(
    crop: str,
    area_hectares: float,
    zonal_breakdown: Dict[str, Any],
    mean_ndvi: float
) -> Dict[str, Any]:
    """
    Generate precision Variable-Rate Application (VRA) fertilizer prescriptions
    for Nitrogen (Urea 46-0-0) and NPK replenishment per management zone.
    """
    pct_high = zonal_breakdown.get("high_vigor_pct", 60.0) / 100.0
    pct_mod = zonal_breakdown.get("moderate_stress_pct", 30.0) / 100.0
    pct_sev = zonal_breakdown.get("severe_anomaly_pct", 10.0) / 100.0
    
    # Baseline crop fertilizer recommendations (kg/ha)
    crop_n_base = {
        "Wheat": 120,
        "Corn (Maize)": 160,
        "Strawberry": 90,
        "Citrus (Orange)": 110,
        "Grape (Vineyard)": 70,
        "Tomato": 130,
        "Rice / Paddy": 115,
        "Soybean": 45,
        "Cotton": 125
    }.get(crop, 110)
    
    # Variable Nitrogen Application Rates:
    # High vigor: Reduced maintenance dosage (-25%)
    # Moderate stress: Targeted replenishment (+20%)
    # Severe stress: High boost (+45%) + Micronutrient foliar spray
    rate_high = round(crop_n_base * 0.75, 1)
    rate_mod = round(crop_n_base * 1.20, 1)
    rate_sev = round(crop_n_base * 1.45, 1)
    
    # Area per zone in hectares
    area_high = round(area_hectares * pct_high, 2)
    area_mod = round(area_hectares * pct_mod, 2)
    area_sev = round(area_hectares * pct_sev, 2)
    
    # Total Urea (46% Nitrogen) needed in kg
    total_n_kg = (rate_high * area_high) + (rate_mod * area_mod) + (rate_sev * area_sev)
    total_urea_kg = round(total_n_kg / 0.46, 1)
    
    # Flat rate conventional baseline comparison
    flat_n_kg = crop_n_base * area_hectares
    flat_urea_kg = round(flat_n_kg / 0.46, 1)
    savings_pct = round(max(0.0, ((flat_urea_kg - total_urea_kg) / flat_urea_kg) * 100.0), 1)
    
    return {
        "crop": crop,
        "total_area_hectares": area_hectares,
        "total_area_acres": round(area_hectares * 2.47105, 2),
        "zones": [
            {
                "zone_name": "Zone 1: High Vigor Canopy",
                "area_hectares": area_high,
                "area_pct": round(pct_high * 100, 1),
                "nitrogen_rate_kg_ha": rate_high,
                "urea_kg_ha": round(rate_high / 0.46, 1),
                "action": "Maintain vegetative health. Standard drip/broadcast maintenance.",
                "color": "#16a34a"
            },
            {
                "zone_name": "Zone 2: Mild Stress / Nutrient Deficit",
                "area_hectares": area_mod,
                "area_pct": round(pct_mod * 100, 1),
                "nitrogen_rate_kg_ha": rate_mod,
                "urea_kg_ha": round(rate_mod / 0.46, 1),
                "action": "Apply targeted NPK boost + Zinc (ZnSO4) to restore chlorophyll synthesis.",
                "color": "#eab308"
            },
            {
                "zone_name": "Zone 3: Critical Pathology / Moisture Deficit",
                "area_hectares": area_sev,
                "area_pct": round(pct_sev * 100, 1),
                "nitrogen_rate_kg_ha": rate_sev,
                "urea_kg_ha": round(rate_sev / 0.46, 1),
                "action": "Urgent ground scouting required. Check soil compaction, root rot, or foliar rust. Apply foliar bio-stimulants.",
                "color": "#dc2626"
            }
        ],
        "total_fertilizer_demand": {
            "vra_urea_kg": total_urea_kg,
            "conventional_flat_urea_kg": flat_urea_kg,
            "cost_and_input_savings_pct": savings_pct
        }
    }

def generate_multi_temporal_ndvi_curve(crop: str, current_ndvi: float) -> List[Dict[str, Any]]:
    """
    Generate a 180-day multi-temporal vegetative phenology timeline
    showing NDVI curve progression across growth milestones.
    """
    stages = [
        {"stage": "Emergence & Seedling", "days_after_planting": 15, "ndvi_factor": 0.25, "status": "Historical"},
        {"stage": "Early Tillering / Vegetative", "days_after_planting": 45, "ndvi_factor": 0.52, "status": "Historical"},
        {"stage": "Canopy Closure & Booting", "days_after_planting": 75, "ndvi_factor": 0.78, "status": "Historical"},
        {"stage": "Flowering & Anthesis (Active)", "days_after_planting": 105, "ndvi_factor": 1.0, "status": "Current Satellite Pass"},
        {"stage": "Grain / Fruit Development", "days_after_planting": 135, "ndvi_factor": 0.88, "status": "Forecast Model"},
        {"stage": "Ripening & Senescence", "days_after_planting": 165, "ndvi_factor": 0.42, "status": "Harvest Target"}
    ]
    
    timeline = []
    for s in stages:
        val = round(max(0.18, min(0.95, current_ndvi * s["ndvi_factor"])), 2)
        timeline.append({
            "stage": s["stage"],
            "dap": s["days_after_planting"],
            "ndvi": val,
            "status": s["status"]
        })
    return timeline

def generate_geojson_field_polygon(
    field_id: str,
    name: str,
    lat: float,
    lon: float,
    area_hectares: float,
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate GIS GeoJSON FeatureCollection for field parcel boundaries."""
    delta = 0.0035
    coords = [
        [round(lon - delta/2, 6), round(lat - delta/2, 6)],
        [round(lon + delta/2, 6), round(lat - delta/2, 6)],
        [round(lon + delta/2, 6), round(lat + delta/2, 6)],
        [round(lon - delta/2, 6), round(lat + delta/2, 6)],
        [round(lon - delta/2, 6), round(lat - delta/2, 6)]
    ]
    
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": field_id,
                "properties": {
                    "field_name": name,
                    "area_hectares": area_hectares,
                    "mean_ndvi": stats.get("mean_ndvi", 0.70),
                    "uniformity_score": stats.get("field_uniformity_score", 85.0),
                    "satellite_source": "Sentinel-2 MultiSpectral ESA (10m Resolution)"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
        ]
    }

def analyze_satellite_farm_field(
    lat: float,
    lon: float,
    crop: str = "Wheat",
    area_hectares: float = 10.0,
    field_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Master orchestrator for Satellite NDVI Farm Field Mapping analysis.
    """
    # 1. Generate raster grid
    raster = generate_field_raster_matrix(lat, lon, mean_ndvi_target=0.72, grid_size=24, crop=crop)
    stats = raster["statistics"]
    zonal = raster["zonal_breakdown"]
    
    # 2. VRA Fertilizer Prescription
    fertilizer = generate_vra_fertilizer_prescription(crop, area_hectares, zonal, stats["mean_ndvi"])
    
    # 3. Multi-temporal curve
    temporal_curve = generate_multi_temporal_ndvi_curve(crop, stats["mean_ndvi"])
    
    # 4. GeoJSON parcel
    geojson = generate_geojson_field_polygon(
        field_id=f"field_{int(abs(lat)*100)}_{int(abs(lon)*100)}",
        name=field_name or f"{crop} Field ({lat:.2f}, {lon:.2f})",
        lat=lat,
        lon=lon,
        area_hectares=area_hectares,
        stats=stats
    )
    
    return {
        "field_metadata": {
            "name": field_name or f"{crop} Farm Parcel",
            "lat": lat,
            "lon": lon,
            "crop": crop,
            "area_hectares": area_hectares,
            "area_acres": round(area_hectares * 2.47105, 2),
            "satellite_pass_time": "2026-08-23 08:30 UTC",
            "satellite_platform": "Sentinel-2 MSI (Copernicus ESA) / Landsat 9"
        },
        "raster": raster,
        "vra_fertilizer_prescription": fertilizer,
        "multi_temporal_growth_curve": temporal_curve,
        "geojson": geojson
    }
