"""
Agricultural Tank Mix & Field Dosage Calculator
Computes accurate spray volume, chemical/bio product mass, tank refill cycles, and cost projections based on acreage.
"""

from typing import Dict, Any

# Standard water application rates based on crop foliage density (Liters per Acre)
CROP_WATER_RATES_PER_ACRE = {
    "Tomato": 180,
    "Potato": 160,
    "Apple": 280,
    "Grape": 240,
    "Corn (Maize)": 150,
    "Rice (Paddy)": 200,
    "Pepper (Bell)": 160,
    "Orange / Citrus": 320,
    "Peach": 260,
    "Squash / Cucurbits": 180,
    "Strawberry": 150,
    "Soybean": 140,
    "Default": 180
}

def calculate_field_dosage(
    field_size: float,
    unit: str = "acres",
    crop: str = "Tomato",
    dosage_per_liter: float = 2.5,
    dosage_unit: str = "g",
    tank_capacity_liters: float = 15.0
) -> Dict[str, Any]:
    """
    Compute precise agronomic spray metrics for a given field area.
    """
    # Normalize area to acres
    if unit.lower() in ["hectares", "ha"]:
        acres = field_size * 2.47105
    else:
        acres = field_size
        
    acres = max(0.01, acres)
    
    # Get base water rate per acre
    water_rate_per_acre = CROP_WATER_RATES_PER_ACRE.get(crop, CROP_WATER_RATES_PER_ACRE["Default"])
    total_water_liters = round(acres * water_rate_per_acre, 1)
    total_water_gallons = round(total_water_liters * 0.264172, 1)
    
    # Compute total product required
    total_product_raw = total_water_liters * dosage_per_liter
    
    if dosage_unit.lower() in ["g", "grams"]:
        if total_product_raw >= 1000:
            product_display = f"{total_product_raw / 1000.0:.2f} kg ({total_product_raw:.0f} g)"
        else:
            product_display = f"{total_product_raw:.1f} g"
    else:
        # Liquid mL
        if total_product_raw >= 1000:
            product_display = f"{total_product_raw / 1000.0:.2f} L ({total_product_raw:.0f} mL)"
        else:
            product_display = f"{total_product_raw:.1f} mL"
            
    # Tank cycles
    tank_capacity = max(1.0, tank_capacity_liters)
    num_tanks = round(total_water_liters / tank_capacity, 1)
    product_per_tank = round(tank_capacity * dosage_per_liter, 1)
    
    return {
        "field_size": field_size,
        "unit": unit,
        "crop": crop,
        "acres_normalized": round(acres, 2),
        "total_water_liters": total_water_liters,
        "total_water_gallons": total_water_gallons,
        "total_product_required": product_display,
        "total_product_raw": total_product_raw,
        "dosage_unit": dosage_unit,
        "tank_capacity_liters": tank_capacity,
        "num_tanks_required": num_tanks,
        "product_per_tank": f"{product_per_tank} {dosage_unit}",
        "spray_advisory": (
            f"Mix {product_per_tank}{dosage_unit} of formulation per {tank_capacity}L sprayer tank. "
            f"Apply in calm morning or late afternoon hours with wind speed < 10 km/h."
        )
    }
