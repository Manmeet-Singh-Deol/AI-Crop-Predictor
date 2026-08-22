"""
Microclimate Disease Outbreak Risk Forecaster
Integrates with Open-Meteo REST API (free, open, no API key needed) and computes epidemiological risk indices.
"""

import httpx
from typing import Dict, Any, List, Optional

# Default major agricultural regions for quick selection
POPULAR_LOCATIONS = [
    {"name": "Salinas Valley, CA", "lat": 36.6777, "lon": -121.6555, "country": "USA"},
    {"name": "Punjab, India", "lat": 30.9010, "lon": 75.8573, "country": "India"},
    {"name": "Nashik, Maharashtra", "lat": 19.9975, "lon": 73.7898, "country": "India"},
    {"name": "Iowa Corn Belt, USA", "lat": 42.0308, "lon": -93.6319, "country": "USA"},
    {"name": "Kent (Garden of England)", "lat": 51.2787, "lon": 0.5217, "country": "UK"},
    {"name": "Valencia Citrus Groves", "lat": 39.4699, "lon": -0.3763, "country": "Spain"},
    {"name": "Sao Paulo Coffee/Soy", "lat": -23.5505, "lon": -46.6333, "country": "Brazil"},
    {"name": "Hokkaido Agri Plains", "lat": 43.0618, "lon": 141.3545, "country": "Japan"}
]

def calculate_disease_risks(temp_c: float, humidity_pct: float, rain_mm: float, wind_kmh: float) -> Dict[str, Any]:
    """
    Calculate epidemiological pathogen outbreak risks based on microclimate variables.
    """
    # 1. Fungal Outbreak Risk (Favored by high humidity 75-95%, moderate temp 15-28°C, rain/leaf wetness)
    fungal_score = 0.0
    if humidity_pct >= 85:
        fungal_score += 45
    elif humidity_pct >= 70:
        fungal_score += 30
    elif humidity_pct >= 55:
        fungal_score += 15
        
    if 18.0 <= temp_c <= 28.0:
        fungal_score += 35
    elif 12.0 <= temp_c <= 32.0:
        fungal_score += 20
    else:
        fungal_score += 5
        
    if rain_mm > 5.0:
        fungal_score += 20
    elif rain_mm > 0.0:
        fungal_score += 12
        
    fungal_risk = min(100.0, max(5.0, fungal_score))
    
    # 2. Bacterial Outbreak Risk (Favored by warm temp 24-32°C, wind-driven rain, high humidity)
    bacterial_score = 0.0
    if 24.0 <= temp_c <= 33.0:
        bacterial_score += 40
    elif 18.0 <= temp_c <= 36.0:
        bacterial_score += 20
        
    if humidity_pct >= 80:
        bacterial_score += 30
    elif humidity_pct >= 60:
        bacterial_score += 15
        
    if rain_mm > 2.0 and wind_kmh > 15.0:
        bacterial_score += 30  # Driving rain spreads bacteria rapidly
    elif rain_mm > 0.0:
        bacterial_score += 15
        
    bacterial_risk = min(100.0, max(5.0, bacterial_score))
    
    # 3. Viral / Insect Vector Risk (Whiteflies, Aphids, Thrips thrive in warm, dry weather 26-38°C, low rain)
    viral_score = 0.0
    if 26.0 <= temp_c <= 38.0:
        viral_score += 45
    elif 20.0 <= temp_c < 26.0:
        viral_score += 25
        
    if humidity_pct < 55:
        viral_score += 35
    elif humidity_pct < 70:
        viral_score += 20
        
    if rain_mm == 0:
        viral_score += 20
    else:
        viral_score -= 15  # Rain washes away vector insects
        
    viral_risk = min(100.0, max(5.0, viral_score))
    
    # Overall Aggregate Risk
    overall_score = round((fungal_risk * 0.45 + bacterial_risk * 0.35 + viral_risk * 0.20), 1)
    
    if overall_score >= 70:
        threat_level = "High / Critical Risk"
        threat_color = "#ef4444"
        summary = "Current microclimate is highly conducive to rapid pathogen multiplication and sporulation. Immediate preventive or systemic spraying is strongly advised before impending precipitation."
    elif overall_score >= 45:
        threat_level = "Moderate Risk"
        threat_color = "#f59e0b"
        summary = "Favorable weather conditions for early pathogen colonization. Intensify field scouting, ensure adequate canopy ventilation, and prepare bio-fungicides/protective sprays."
    else:
        threat_level = "Low Risk"
        threat_color = "#10b981"
        summary = "Environmental conditions are generally suppressive to major foliar epidemics. Maintain standard irrigation schedules and routine cultural monitoring."
        
    return {
        "fungal_risk_score": round(fungal_risk, 1),
        "bacterial_risk_score": round(bacterial_risk, 1),
        "viral_risk_score": round(viral_risk, 1),
        "overall_outbreak_risk": overall_score,
        "threat_level": threat_level,
        "threat_color": threat_color,
        "advisory_summary": summary
    }

async def fetch_weather_by_coords(lat: float, lon: float, location_name: str = "Target Farm") -> Dict[str, Any]:
    """
    Fetch live weather and 5-day forecast from Open-Meteo REST API.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max&timezone=auto"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        
    current = data.get("current", {})
    daily = data.get("daily", {})
    
    temp = current.get("temperature_2m", 22.0)
    humidity = current.get("relative_humidity_2m", 65.0)
    rain = current.get("precipitation", 0.0)
    wind = current.get("wind_speed_10m", 8.0)
    
    risks = calculate_disease_risks(temp, humidity, rain, wind)
    
    # Process 5-day daily forecast risks
    daily_forecasts = []
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip_sums = daily.get("precipitation_sum", [])
    precip_probs = daily.get("precipitation_probability_max", [])
    
    for i in range(min(5, len(dates))):
        d_date = dates[i]
        d_max = max_temps[i] if i < len(max_temps) else temp
        d_min = min_temps[i] if i < len(min_temps) else temp
        d_avg_temp = (d_max + d_min) / 2.0
        d_precip = precip_sums[i] if i < len(precip_sums) else 0.0
        d_prob = precip_probs[i] if i < len(precip_probs) else 20
        d_est_humidity = min(95.0, max(40.0, humidity + (d_precip * 5.0)))
        
        d_risk = calculate_disease_risks(d_avg_temp, d_est_humidity, d_precip, wind)
        
        daily_forecasts.append({
            "date": d_date,
            "max_temp": d_max,
            "min_temp": d_min,
            "precip_mm": d_precip,
            "precip_probability": d_prob,
            "outbreak_risk": d_risk["overall_outbreak_risk"],
            "threat_level": d_risk["threat_level"],
            "threat_color": d_risk["threat_color"]
        })
        
    return {
        "location": location_name,
        "latitude": lat,
        "longitude": lon,
        "current_weather": {
            "temperature_c": temp,
            "humidity_pct": humidity,
            "precipitation_mm": rain,
            "wind_speed_kmh": wind,
            "apparent_temp_c": current.get("apparent_temperature", temp),
            "weather_code": current.get("weather_code", 0)
        },
        "epidemiological_risk": risks,
        "five_day_forecast": daily_forecasts
    }

async def search_city_and_get_risk(city_name: str) -> Dict[str, Any]:
    """
    Geocode city name and retrieve full weather risk analysis with sanitized query parameters.
    """
    clean_city = str(city_name).strip()[:100]  # Bound string length
    
    # Check if popular location first
    for loc in POPULAR_LOCATIONS:
        if clean_city.lower() in loc["name"].lower():
            return await fetch_weather_by_coords(loc["lat"], loc["lon"], loc["name"])
            
    # Geocoding via Open-Meteo with structured params (prevents parameter injection)
    geo_base = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": clean_city,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            geo_res = await client.get(geo_base, params=geo_params)
            geo_res.raise_for_status()
            geo_data = geo_res.json()
            
        results = geo_data.get("results", [])
        if results:
            first = results[0]
            name = f"{first.get('name')}, {first.get('country', '')}"
            lat = first.get("latitude")
            lon = first.get("longitude")
            return await fetch_weather_by_coords(lat, lon, name)
    except Exception as e:
        print(f"[WeatherRisk] Geocoding API notice ({e}), using default location.")
        
    # Default fallback to Salinas Valley
    return await fetch_weather_by_coords(36.6777, -121.6555, f"{clean_city} (Estimated Region)")
