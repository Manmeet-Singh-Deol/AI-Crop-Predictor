"""
Optimal Spray Window & Rainfastness Weather Engine (AgroAI)
Psychrometric Delta-T modeling, chemical absorption kinetics, wind drift hazard detection,
and 48-hour hourly spray suitability forecasting.
"""

import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Agricultural Chemical Rainfastness & Absorption Database
# ---------------------------------------------------------------------------
RAINFASTNESS_DB: Dict[str, Dict[str, Any]] = {
    "systemic_fungicide": {
        "name": "Systemic Fungicide (e.g. Azoxystrobin, Difenoconazole, Tebuconazole)",
        "type": "Systemic",
        "rainfast_hours": 2.5,
        "ideal_delta_t_min": 2.0,
        "ideal_delta_t_max": 8.0,
        "max_wind_kmh": 16.0,
        "min_wind_kmh": 3.0,
        "max_temp_c": 30.0,
        "min_temp_c": 8.0,
        "description": "Translocates into plant tissue. Requires 2-3 hours rain-free to absorb before precipitation."
    },
    "contact_fungicide": {
        "name": "Contact / Protective Fungicide (e.g. Mancozeb, Chlorothalonil, Copper)",
        "type": "Contact Surface Protectant",
        "rainfast_hours": 5.0,
        "ideal_delta_t_min": 2.0,
        "ideal_delta_t_max": 7.0,
        "max_wind_kmh": 14.0,
        "min_wind_kmh": 3.0,
        "max_temp_c": 28.0,
        "min_temp_c": 10.0,
        "description": "Forms protective surface barrier. Requires 4-6 hours to dry and adhere completely."
    },
    "bio_fungicide": {
        "name": "Biologicals & Bio-Fungicides (e.g. Trichoderma, Bacillus subtilis)",
        "type": "Biological Agent",
        "rainfast_hours": 4.0,
        "ideal_delta_t_min": 2.0,
        "ideal_delta_t_max": 6.0,
        "max_wind_kmh": 12.0,
        "min_wind_kmh": 3.0,
        "max_temp_c": 28.0,
        "min_temp_c": 12.0,
        "description": "Live microbial colonies. Sensitive to high UV; best applied late afternoon or overcast hours."
    },
    "neem_botanical": {
        "name": "Botanical Oils / Neem Extract (Azadirachtin)",
        "type": "Botanical / Organic",
        "rainfast_hours": 3.0,
        "ideal_delta_t_min": 2.0,
        "ideal_delta_t_max": 7.0,
        "max_wind_kmh": 14.0,
        "min_wind_kmh": 3.0,
        "max_temp_c": 30.0,
        "min_temp_c": 10.0,
        "description": "Natural emulsified oil. Avoid high noon heat to prevent foliar oil burn (phytotoxicity)."
    },
    "systemic_insecticide": {
        "name": "Systemic Insecticide (e.g. Imidacloprid, Thiamethoxam)",
        "type": "Systemic Insecticide",
        "rainfast_hours": 2.0,
        "ideal_delta_t_min": 2.0,
        "ideal_delta_t_max": 8.0,
        "max_wind_kmh": 15.0,
        "min_wind_kmh": 3.0,
        "max_temp_c": 32.0,
        "min_temp_c": 10.0,
        "description": "Rapid vascular uptake. Strictly avoid spraying during bee foraging / flowering periods."
    },
    "foliar_fertilizer": {
        "name": "Foliar Micronutrients / Liquid NPK (19:19:19)",
        "type": "Foliar Nutrition",
        "rainfast_hours": 2.0,
        "ideal_delta_t_min": 2.0,
        "ideal_delta_t_max": 6.0,
        "max_wind_kmh": 16.0,
        "min_wind_kmh": 2.0,
        "max_temp_c": 26.0,
        "min_temp_c": 8.0,
        "description": "Direct stomatal absorption. Lower Delta-T (higher humidity) prevents salt burn on foliage."
    }
}

# ---------------------------------------------------------------------------
# 2. Psychrometric Physics: Stull Wet-Bulb & Delta-T Formula
# ---------------------------------------------------------------------------
def calculate_wet_bulb_temperature(temp_c: float, humidity_pct: float) -> float:
    """
    Calculate Wet-Bulb Temperature (Tw) using Stull's empirical psychrometric formula.
    Accurate to within 0.3°C for ambient agricultural ranges.
    """
    t = float(temp_c)
    rh = max(1.0, min(100.0, float(humidity_pct)))
    
    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw, 2)

def calculate_delta_t(temp_c: float, humidity_pct: float) -> Dict[str, Any]:
    """
    Delta-T = Dry-Bulb Temp (T) - Wet-Bulb Temp (Tw)
    The gold standard agricultural metric for droplet evaporation rate & survival.
    """
    tw = calculate_wet_bulb_temperature(temp_c, humidity_pct)
    delta_t = round(float(temp_c) - tw, 1)
    
    if delta_t < 2.0:
        status = "Low (Risk of Run-off / Slow Drying)"
        color = "#3b82f6"  # Blue
        rating = "marginal"
        recommendation = "Droplets survive very long. High humidity slows drying, which can promote fungal infection or cause product run-off. Use medium droplets."
    elif 2.0 <= delta_t <= 8.0:
        status = "Ideal (Optimal Droplet Survival & Uptake)"
        color = "#10b981"  # Emerald
        rating = "optimal"
        recommendation = "Perfect evaporative rate! Droplets maintain optimal lifespan on leaf surface for complete cuticular absorption with minimal drift."
    elif 8.0 < delta_t <= 10.0:
        status = "Marginal (Elevated Evaporation)"
        color = "#f59e0b"  # Amber
        rating = "marginal"
        recommendation = "Droplets evaporate rapidly. Increase droplet size (switch to coarse / air-induction nozzles), lower boom height, or spray during cooler hours."
    else:
        status = "Unsuitable / High Hazard (Severe Evaporation & Volatilization)"
        color = "#ef4444"  # Red
        rating = "unsuitable"
        recommendation = "Extreme evaporation! Fine droplets evaporate into airborne aerosol within seconds before reaching target. Severe chemical loss and drift danger."
        
    return {
        "dry_bulb_c": round(float(temp_c), 1),
        "wet_bulb_c": tw,
        "delta_t_c": delta_t,
        "status": status,
        "rating": rating,
        "color": color,
        "recommendation": recommendation
    }

# ---------------------------------------------------------------------------
# 3. Wind Drift & Thermal Inversion Hazard Evaluation
# ---------------------------------------------------------------------------
def evaluate_wind_drift_hazard(wind_speed_kmh: float, gusts_kmh: float = 0.0) -> Dict[str, Any]:
    """
    Evaluates drift risk and dangerous surface temperature inversion risk.
    """
    w = float(wind_speed_kmh)
    g = float(gusts_kmh or w)
    
    if w < 3.0:
        status = "Surface Temperature Inversion Risk"
        color = "#a855f7"  # Purple
        rating = "caution"
        description = "Calm air (<3 km/h) indicates potential thermal inversion where fine droplets suspend in floating air layers and travel unpredictably off-target for miles."
    elif 3.0 <= w <= 15.0:
        status = "Optimal Wind Velocity"
        color = "#10b981"  # Emerald
        rating = "optimal"
        description = "Gentle, steady breeze (3-15 km/h) moves spray droplets reliably into crop canopy with minimal off-target drift."
    elif 15.0 < w <= 22.0:
        status = "Marginal Drift Hazard (Use Low-Drift Nozzles)"
        color = "#f59e0b"  # Amber
        rating = "marginal"
        description = f"Moderate wind ({w:.1f} km/h, gusts {g:.1f} km/h). Mandatory use of drift-reducing coarse nozzles (AI/TTI) and low boom height (<50cm above canopy)."
    else:
        status = "Severe Drift Hazard (Spray Shutdown Required)"
        color = "#ef4444"  # Red
        rating = "unsuitable"
        description = f"Excessive wind ({w:.1f} km/h)! High risk of off-target chemical loss, contamination of adjacent fields, waterways, and non-target organisms."
        
    return {
        "wind_speed_kmh": round(w, 1),
        "gusts_kmh": round(g, 1),
        "status": status,
        "color": color,
        "rating": rating,
        "description": description
    }

# ---------------------------------------------------------------------------
# 4. Hourly Spray Suitability & Rainfastness Washout Analysis
# ---------------------------------------------------------------------------
def evaluate_single_hour_spray_suitability(
    temp_c: float,
    humidity_pct: float,
    wind_kmh: float,
    gusts_kmh: float,
    precip_mm: float,
    precip_prob: float,
    uv_index: float,
    future_rain_in_window_mm: float,
    chemical_key: str = "systemic_fungicide"
) -> Dict[str, Any]:
    """
    Computes a 0-100% suitability score for a single hour given microclimate variables
    and chemical rainfastness requirements.
    """
    chem = RAINFASTNESS_DB.get(chemical_key, RAINFASTNESS_DB["systemic_fungicide"])
    delta_t_info = calculate_delta_t(temp_c, humidity_pct)
    wind_info = evaluate_wind_drift_hazard(wind_kmh, gusts_kmh)
    
    score = 100.0
    hazard_reasons = []
    
    # 1. Rain Washout Hazard (Absolute Showstopper)
    if precip_mm > 0.0:
        score = 0.0
        hazard_reasons.append(f"Active rainfall ({precip_mm:.1f} mm) — instant spray wash-off")
    elif future_rain_in_window_mm > 0.3:
        penalty = min(90.0, future_rain_in_window_mm * 35.0)
        score -= penalty
        hazard_reasons.append(f"Imminent rain ({future_rain_in_window_mm:.1f} mm) within {chem['rainfast_hours']}h rainfast window (washout risk)")
    elif precip_prob >= 60.0:
        score -= 40.0
        hazard_reasons.append(f"High rain probability ({precip_prob:.0f}%) within absorption window")
    elif precip_prob >= 40.0:
        score -= 20.0
        hazard_reasons.append(f"Moderate rain probability ({precip_prob:.0f}%)")

    # 2. Wind Speed & Drift Penalties
    if wind_kmh > 22.0:
        score = min(score, 10.0)
        hazard_reasons.append(f"Excessive wind speed ({wind_kmh:.1f} km/h > 22 km/h limit)")
    elif wind_kmh > 15.0:
        score -= 25.0
        hazard_reasons.append(f"Wind drift caution ({wind_kmh:.1f} km/h)")
    elif wind_kmh < 3.0:
        score -= 20.0
        hazard_reasons.append("Calm air (<3 km/h) thermal inversion danger")

    # 3. Delta-T Evaporation Penalties
    dt = delta_t_info["delta_t_c"]
    if dt > 10.0:
        score = min(score, 15.0)
        hazard_reasons.append(f"Extreme Delta-T ({dt:.1f}°C) — rapid droplet evaporation")
    elif dt > 8.0:
        score -= 25.0
        hazard_reasons.append(f"Elevated Delta-T ({dt:.1f}°C)")
    elif dt < 2.0:
        score -= 15.0
        hazard_reasons.append(f"Low Delta-T ({dt:.1f}°C) — very slow drying")

    # 4. Temperature Extremes & Phytotoxicity
    if temp_c > 32.0:
        score -= 30.0
        hazard_reasons.append(f"High temperature ({temp_c:.1f}°C) — leaf scorch hazard")
    elif temp_c < 6.0:
        score -= 25.0
        hazard_reasons.append(f"Low temperature ({temp_c:.1f}°C) — restricted cuticular uptake")

    # 5. UV degradation for biologicals
    if "bio" in chemical_key and uv_index > 6.0:
        score -= 20.0
        hazard_reasons.append(f"High solar UV ({uv_index:.1f}) degrades microbial spores")

    final_score = round(max(0.0, min(100.0, score)), 1)
    
    if final_score >= 80.0:
        status = "Optimal"
        badge_color = "#10b981"  # Emerald
        summary = "Excellent conditions! Maximum spray deposition and absorption."
    elif final_score >= 50.0:
        status = "Caution / Marginal"
        badge_color = "#f59e0b"  # Amber
        summary = "Marginal conditions. Acceptable with coarse anti-drift nozzles and caution."
    else:
        status = "Unsuitable / Do Not Spray"
        badge_color = "#ef4444"  # Red
        summary = "Unsafe spraying conditions. Chemical loss or crop damage likely."

    return {
        "suitability_score": final_score,
        "status": status,
        "badge_color": badge_color,
        "summary": summary,
        "delta_t": delta_t_info,
        "wind_drift": wind_info,
        "hazard_reasons": hazard_reasons
    }

# ---------------------------------------------------------------------------
# 5. Full 48-Hour Hourly Timeline & Next Safe Window Forecaster
# ---------------------------------------------------------------------------
def compute_spray_window_timeline(
    hourly_weather: Dict[str, Any],
    chemical_key: str = "systemic_fungicide",
    max_hours: int = 48
) -> Dict[str, Any]:
    """
    Parses Open-Meteo hourly weather and produces 48-hour suitability analysis
    with the next best spray window.
    """
    times = hourly_weather.get("time", [])
    temps = hourly_weather.get("temperature_2m", [])
    humidities = hourly_weather.get("relative_humidity_2m", [])
    precips = hourly_weather.get("precipitation", [])
    precip_probs = hourly_weather.get("precipitation_probability", [])
    winds = hourly_weather.get("wind_speed_10m", [])
    gusts = hourly_weather.get("wind_gusts_10m", [])
    uvs = hourly_weather.get("uv_index", [])
    weather_codes = hourly_weather.get("weather_code", [])

    total_pts = min(max_hours, len(times))
    chem = RAINFASTNESS_DB.get(chemical_key, RAINFASTNESS_DB["systemic_fungicide"])
    rainfast_hours_int = max(1, int(math.ceil(chem["rainfast_hours"])))

    hourly_timeline = []
    optimal_windows = []
    current_window_start = None

    for i in range(total_pts):
        t_time = times[i]
        t_temp = temps[i] if i < len(temps) else 22.0
        t_rh = humidities[i] if i < len(humidities) else 65.0
        t_precip = precips[i] if i < len(precips) else 0.0
        t_prob = precip_probs[i] if i < len(precip_probs) else 10.0
        t_wind = winds[i] if i < len(winds) else 8.0
        t_gust = gusts[i] if i < len(gusts) else t_wind
        t_uv = uvs[i] if i < len(uvs) else 3.0
        t_code = weather_codes[i] if i < len(weather_codes) else 0

        # Calculate forward rain in rainfastness window
        forward_rain = 0.0
        for offset in range(1, rainfast_hours_int + 1):
            if i + offset < len(precips):
                forward_rain += precips[i + offset]

        hour_eval = evaluate_single_hour_spray_suitability(
            temp_c=t_temp,
            humidity_pct=t_rh,
            wind_kmh=t_wind,
            gusts_kmh=t_gust,
            precip_mm=t_precip,
            precip_prob=t_prob,
            uv_index=t_uv,
            future_rain_in_window_mm=forward_rain,
            chemical_key=chemical_key
        )

        formatted_time = t_time.split("T")[-1][:5] if "T" in t_time else t_time
        formatted_date = t_time.split("T")[0] if "T" in t_time else ""

        entry = {
            "iso_time": t_time,
            "display_time": formatted_time,
            "display_date": formatted_date,
            "temperature_c": round(t_temp, 1),
            "humidity_pct": round(t_rh, 1),
            "wind_speed_kmh": round(t_wind, 1),
            "precipitation_mm": round(t_precip, 1),
            "precipitation_probability": round(t_prob, 0),
            "forward_rain_in_window_mm": round(forward_rain, 1),
            "weather_code": t_code,
            "suitability_score": hour_eval["suitability_score"],
            "status": hour_eval["status"],
            "badge_color": hour_eval["badge_color"],
            "delta_t_c": hour_eval["delta_t"]["delta_t_c"],
            "delta_t_status": hour_eval["delta_t"]["status"],
            "wind_status": hour_eval["wind_drift"]["status"],
            "hazard_reasons": hour_eval["hazard_reasons"],
            "summary": hour_eval["summary"]
        }
        hourly_timeline.append(entry)

        # Track contiguous optimal/acceptable windows
        if hour_eval["suitability_score"] >= 65.0:
            if current_window_start is None:
                current_window_start = i
        else:
            if current_window_start is not None:
                optimal_windows.append((current_window_start, i - 1))
                current_window_start = None

    if current_window_start is not None:
        optimal_windows.append((current_window_start, total_pts - 1))

    # Identify Next Recommended Spray Window
    next_window_info = {
        "available": False,
        "headline": "No safe spraying window detected in the next 48 hours.",
        "start_time": "",
        "end_time": "",
        "duration_hours": 0,
        "avg_score": 0.0,
        "avg_delta_t": 0.0,
        "avg_wind": 0.0,
        "recommendation": "Delay application. Heavy rain, high wind, or extreme Delta-T conditions will compromise chemical efficacy."
    }

    if optimal_windows:
        best_win = optimal_windows[0]
        w_start, w_end = best_win
        w_items = hourly_timeline[w_start:w_end + 1]
        
        duration = len(w_items)
        avg_score = round(sum(it["suitability_score"] for it in w_items) / duration, 1)
        avg_dt = round(sum(it["delta_t_c"] for it in w_items) / duration, 1)
        avg_w = round(sum(it["wind_speed_kmh"] for it in w_items) / duration, 1)
        
        start_str = f"{w_items[0]['display_date']} {w_items[0]['display_time']}"
        end_str = f"{w_items[-1]['display_date']} {w_items[-1]['display_time']}"
        
        next_window_info = {
            "available": True,
            "headline": f"Next Safe Spray Window: {w_items[0]['display_time']} to {w_items[-1]['display_time']} ({duration} hrs)",
            "start_time": start_str,
            "end_time": end_str,
            "duration_hours": duration,
            "avg_score": avg_score,
            "avg_delta_t": avg_dt,
            "avg_wind": avg_w,
            "recommendation": f"Ideal application opportunity ({duration}h continuous window). Delta-T averages {avg_dt}°C with calm wind ({avg_w} km/h) and zero precipitation during the {chem['rainfast_hours']}h rainfastness period."
        }

    # Current Hour Status
    current_status = hourly_timeline[0] if hourly_timeline else {
        "suitability_score": 50.0,
        "status": "Awaiting Data",
        "badge_color": "#94a3b8",
        "delta_t_c": 4.0,
        "summary": "No meteorological data available."
    }

    return {
        "chemical_selected": chem,
        "current_status": current_status,
        "next_safe_window": next_window_info,
        "hourly_timeline": hourly_timeline,
        "available_chemicals": [
            {"key": k, "name": v["name"], "type": v["type"], "rainfast_hours": v["rainfast_hours"]}
            for k, v in RAINFASTNESS_DB.items()
        ]
    }
