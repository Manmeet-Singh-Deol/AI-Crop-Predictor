"""
AI Agronomist Chatbot Engine for AgroAI Platform
Provides conversational agricultural intelligence, disease diagnostics,
chemical and organic treatment protocols, dosage calculations, and weather advisories.
Works out-of-the-box with comprehensive local expert agronomy knowledge,
with optional plug-and-play LLM provider support (Gemini / OpenAI / Groq).
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from backend.advisory_db import ADVISORY_DATABASE, get_advisory

SYSTEM_PROMPT = """You are AgroBot, an expert precision agronomist and crop pathologist AI assistant for the AgroAI platform.
Your mission is to assist farmers, agronomists, and home gardeners with practical, scientifically accurate, and actionable advice.
Always emphasize:
1. Exact commercial brand names (Syngenta, Bayer, FMC, Corteva, BASF, Tata Rallis) and dosage rates per liter.
2. Eco-friendly organic alternatives and cultural preventive practices.
3. Pre-Harvest Intervals (PHI) and safety precautions.
4. Clear bullet points, formatted in friendly markdown.
"""

def generate_expert_response(
    message: str,
    history: List[Dict[str, str]] = None,
    context: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Generate agronomic advisory response with contextual awareness of active scan.
    """
    msg_lower = message.lower().strip()
    history = history or []
    
    # Extract diagnostic context if available
    active_crop = context.get("crop") if context else None
    active_disease = context.get("disease") if context else None
    active_severity = context.get("severity_pct") if context else None
    active_stage = context.get("severity_stage") if context else None
    
    # Check if user query matches any crop or disease in database
    matched_advisory_key = None
    if context and context.get("class_name") and context.get("class_name") in ADVISORY_DATABASE:
        matched_advisory_key = context.get("class_name")
    
    if not matched_advisory_key:
        for k in ADVISORY_DATABASE.keys():
            k_lower = k.lower().replace("_", " ").replace("___", " ")
            parts = k.split("___")
            c_name = parts[0].replace("_", " ").lower()
            d_name = parts[1].replace("_", " ").lower() if len(parts) > 1 else ""
            
            if (c_name in msg_lower and d_name in msg_lower) or (d_name and d_name in msg_lower):
                matched_advisory_key = k
                break
                
    if not matched_advisory_key:
        for k in ADVISORY_DATABASE.keys():
            parts = k.split("___")
            c_name = parts[0].replace("_", " ").lower()
            if c_name in msg_lower and len(c_name) > 3:
                matched_advisory_key = k
                break

    # Determine topic intent
    is_organic = any(w in msg_lower for w in ["organic", "natural", "bio", "home remedy", "neem", "chemical-free", "eco"])
    is_chemical = any(w in msg_lower for w in ["chemical", "spray", "fungicide", "insecticide", "pesticide", "brand", "best seller", "medicine", "dawa", "dawaii", "formula", "product", "syngenta", "bayer", "fmc"])
    is_dosage = any(w in msg_lower for w in ["dose", "dosage", "how much", "quantity", "tank", "liter", "acre", "hectare", "dilution", "ratio", "calculator"])
    is_symptoms = any(w in msg_lower for w in ["symptom", "identify", "how to tell", "look like", "sign", "cause", "spread"])
    is_weather = any(w in msg_lower for w in ["weather", "rain", "temperature", "wind", "humidity", "spray today", "when to spray"])
    is_prevention = any(w in msg_lower for w in ["prevent", "avoid", "next season", "rotation", "soil", "cultural", "stop"])
    is_greeting = any(msg_lower == w or msg_lower.startswith(w + " ") for w in ["hi", "hello", "hey", "namaste", "sasrikal", "hola", "bonjour", "help", "who are you"])

    # Fallback to active context if user asks general question like "how to treat it?"
    if not matched_advisory_key and active_crop and active_disease:
        # Search for active crop in db
        for k in ADVISORY_DATABASE.keys():
            if active_crop.lower() in k.lower() and active_disease.lower().replace(" ", "_") in k.lower():
                matched_advisory_key = k
                break
        if not matched_advisory_key:
            for k in ADVISORY_DATABASE.keys():
                if active_crop.lower() in k.lower():
                    matched_advisory_key = k
                    break

    advisory = ADVISORY_DATABASE.get(matched_advisory_key) if matched_advisory_key else None
    
    # Build dynamic agronomic response
    reply_lines = []
    suggested_chips = []
    
    # 1. GREETING INTENT
    if is_greeting and not (is_organic or is_chemical or is_dosage or is_symptoms):
        if language == "hi":
            reply_lines.append("👋 **नमस्ते! मैं AgroAI कृषि विशेषज्ञ बॉट हूँ।**")
            reply_lines.append("मैं फसल रोगों की पहचान, उपचार, बाजार के सर्वोत्तम कीटनाशक/फफूंदनाशक, जैविक उपाय और स्प्रे खुराक की गणना में आपकी सहायता कर सकता हूँ।")
            reply_lines.append("\n**आप मुझसे क्या पूछना चाहते हैं?**")
            suggested_chips = ["🍅 टमाटर के रोग और उपचार", "🎋 गन्ने का लाल सड़न (Red Rot)", "🧪 सर्वोत्तम बाजार उत्पाद", "🌿 जैविक खेती के नुस्खे"]
        elif language == "pa":
            reply_lines.append("👋 **ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ AgroAI ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ ਹਾਂ।**")
            reply_lines.append("ਮੈਂ ਫਸਲਾਂ ਦੀਆਂ ਬਿਮਾਰੀਆਂ, ਕੀਟਨਾਸ਼ਕਾਂ ਦੀ ਸਹੀ ਖੁਰਾਕ, ਜੈਵਿਕ ਇਲਾਜ ਅਤੇ ਮੌਸਮ ਅਨੁਸਾਰ ਸਪਰੇਅ ਦੇ ਸਮੇਂ ਬਾਰੇ ਜਾਣਕਾਰੀ ਦੇ ਸਕਦਾ ਹਾਂ।")
            suggested_chips = ["🍅 ਟਮਾਟਰ ਦਾ ਪਛੇਤਾ ਝੁਲਸਾ ਰੋਗ", "🎋 ਕਮਾਦ ਦਾ ਰੱਤਾ ਰੋਗ (Red Rot)", "🧪 ਸਪਰੇਅ ਦੀ ਸਹੀ ਖੁਰਾਕ", "🌾 ਝੋਨੇ ਦਾ ਬਲਾਸਟ ਰੋਗ"]
        else:
            reply_lines.append("👋 **Hello! I am your AI Agronomist & Crop Protection Specialist.**")
            reply_lines.append("I can help you with crop disease treatment, chemical & insecticide market best-sellers (Syngenta, Bayer, Corteva, FMC), organic remedies, spray dosage calculation, and weather-safe spraying advice.")
            if active_crop and active_disease:
                reply_lines.append(f"\n📌 **Currently reviewing your active scan:** `{active_crop} - {active_disease}` (Severity: {active_severity or 0}% - {active_stage or 'Stage'}).")
                suggested_chips = [f"🧪 Best chemicals for {active_disease}", f"🌿 Organic controls for {active_crop}", "🌧️ Can I spray in current weather?", "📏 Calculate tank dosage"]
            else:
                reply_lines.append("\n**How can I assist your farm today?**")
                suggested_chips = ["🍅 Tomato Late Blight remedy", "🎋 Sugarcane Red Rot & Rust", "🧪 Top fungicide brands", "📏 Spray tank calculator help"]
                
        return {
            "reply": "\n".join(reply_lines),
            "suggested_actions": suggested_chips,
            "crop": active_crop or (advisory["crop"] if advisory else None),
            "disease": active_disease or (advisory["disease"] if advisory else None)
        }

    # 2. CONTEXTUAL OR ADVISORY MATCHED INTENT
    if advisory:
        crop = advisory["crop"]
        disease = advisory["disease"]
        chems = advisory.get("chemical_controls", [])
        organics = advisory.get("organic_controls", [])
        symptoms = advisory.get("symptoms", [])
        cultural = advisory.get("cultural_practices", [])
        weather_trig = advisory.get("environmental_triggers", {})
        
        # Header
        reply_lines.append(f"### 🌾 Expert Advisory: **{crop} — {disease}**")
        reply_lines.append(f"🔬 **Pathogen:** *{advisory.get('scientific_name', 'Pathogen Profile')}* | **Risk Level:** `{advisory.get('severity_risk', 'High')}`\n")
        
        # Chemical inquiry
        if is_chemical or (not is_organic and not is_symptoms and not is_prevention and not is_weather):
            reply_lines.append("#### 🧪 Top Commercial Brand Formulations & Market Best Sellers:")
            for c in chems:
                reply_lines.append(f"- **{c.get('product', 'Commercial Brand')}**")
                reply_lines.append(f"  • **Active Ingredient:** `{c.get('active_ingredient', 'Standard')}`")
                reply_lines.append(f"  • **Dosage:** `{c.get('dosage', 'Standard')}` | **Type:** *{c.get('type', 'Fungicide / Insecticide')}*")
                reply_lines.append(f"  • **Application:** {c.get('timing', 'At onset')}")
                reply_lines.append(f"  • **Safety Interval (PHI):** `{c.get('interval', '14 days')}`\n")
            suggested_chips.append("🌿 Show organic bio-controls")
            suggested_chips.append("📏 How to calculate tank dosage?")
            suggested_chips.append("🔍 What are the key symptoms?")

        # Organic inquiry
        if is_organic:
            reply_lines.append("#### 🌿 Recommended Biological & Organic Treatments:")
            for org in organics:
                reply_lines.append(f"- 🌱 {org}")
            reply_lines.append("\n> **💡 Agronomist Tip:** Apply organic bio-fungicides (like *Bacillus subtilis* or *Trichoderma*) early in the morning or late evening to protect living spores from UV breakdown.")
            suggested_chips.append("🧪 Compare with chemical brands")
            suggested_chips.append("🚜 Cultural field prevention")

        # Symptoms inquiry
        if is_symptoms:
            reply_lines.append("#### 🔍 Diagnostic Visual Symptoms:")
            for sym in symptoms:
                reply_lines.append(f"- 🔎 {sym}")
            reply_lines.append(f"\n⚡ **Weather Triggers:** {weather_trig.get('high_risk_weather', 'High humidity and temperature swings.')}")
            suggested_chips.append(f"🧪 Chemical cure for {disease}")
            suggested_chips.append(f"🌿 Organic spray for {crop}")

        # Cultural / Prevention inquiry
        if is_prevention or "culture" in msg_lower or "rotate" in msg_lower:
            reply_lines.append("#### 🚜 Cultural Practices & Field Management:")
            for cul in cultural:
                reply_lines.append(f"- 🚜 {cul}")
            suggested_chips.append(f"🧪 Best chemical products")
            suggested_chips.append("🌧️ Weather spray precautions")

        # Weather / Spray timing inquiry
        if is_weather:
            reply_lines.append("#### 🌧️ Weather & Spraying Safety Guidelines:")
            reply_lines.append(f"- **Critical Humidity:** {weather_trig.get('critical_humidity_pct', 80)}% | **Optimal Pathogen Temp:** {weather_trig.get('optimal_temp_c', '20-28°C')}")
            reply_lines.append("- **Rainfastness:** Ensure at least 3-4 hours of dry weather after applying systemic fungicides (e.g. Score, Ridomil Gold) so the chemical absorbs into leaf cuticles.")
            reply_lines.append("- **Wind Speed:** Do not spray if wind speed exceeds 15 km/h to prevent spray drift and uneven canopy coverage.")
            reply_lines.append("- **Temperature:** Avoid mid-day spraying when temperatures exceed 35°C to prevent rapid evaporation and foliage scorch.")
            suggested_chips.append("🧪 Recommended chemical dosage")
            suggested_chips.append("🌿 Organic bio-stimulants")

        # Dosage calculation helper
        if is_dosage:
            reply_lines.append("#### 📏 Spray Tank & Dosage Guidelines:")
            reply_lines.append("- **Standard Backpack Knapsack Tank:** 15 Liters")
            reply_lines.append("- **Standard Tractor Boom Tank:** 200 - 400 Liters")
            reply_lines.append("- **Standard Water Requirement per Acre:** 150 to 200 Liters of water.")
            if chems:
                c0 = chems[0]
                reply_lines.append(f"- **Example for {c0.get('brand_name', 'Top Product')}:** At `{c0.get('dosage')}`, add approximately **30-40 g (or ml)** per 15L backpack tank.")
            reply_lines.append("\n> **💡 Pro-Tip:** You can also use our **Field Spray Tank & Dosage Calculator** below on this page for instant multi-tank calculations!")
            suggested_chips.append("🧪 View all chemical brands")
            suggested_chips.append("🌿 View organic recipes")

        return {
            "reply": "\n".join(reply_lines),
            "suggested_actions": suggested_chips[:4],
            "crop": crop,
            "disease": disease
        }

    # 3. GENERAL AGRONOMY QUERIES (Fertilizers, Pests, Weeds, General Spraying)
    if "fertilizer" in msg_lower or "npk" in msg_lower or "urea" in msg_lower or "dap" in msg_lower:
        reply_lines.append("### 🌾 Balanced Crop Nutrition & Fertilizer Guidelines")
        reply_lines.append("- **Nitrogen (N):** Promotes lush vegetative growth. *Caution:* Excess nitrogen causes succulent foliage, dramatically increasing vulnerability to fungal blights and sucking pests.")
        reply_lines.append("- **Phosphorus (P - DAP / SSP):** Stimulates deep root elongation, sturdy tillering, and early blossom vigor.")
        reply_lines.append("- **Potassium (K - MOP / SOP):** Thickens plant cell walls, enhances drought resistance, and drastically reduces fungal spore penetration.")
        reply_lines.append("- **Secondary & Micronutrients:** Apply Zinc Sulfate (ZnSO₄) and Boron during vegetative and flowering flushes to prevent blossom drop and leaf chlorosis.")
        suggested_chips = ["🍅 Tomato nutrition guide", "🎋 Sugarcane fertilizer schedule", "🌾 Rice NPK requirements", "🌿 Organic compost & bio-fertilizers"]

    elif "sucking pest" in msg_lower or "whitefly" in msg_lower or "aphid" in msg_lower or "thrip" in msg_lower:
        reply_lines.append("### 🐛 Sucking Pest Management Protocol (Whiteflies, Aphids, Thrips)")
        reply_lines.append("Sucking pests damage crops directly by extracting sap and act as primary viral vectors (e.g. TYLCV, Mosaic).")
        reply_lines.append("\n#### ⭐ Top Commercial Market Best Sellers:")
        reply_lines.append("- **Confidor 200 SL (Bayer) ⭐ #1 Best Seller** — *Imidacloprid 17.8% SL* (0.5 ml/L)")
        reply_lines.append("- **Actara 25 WG (Syngenta)** — *Thiamethoxam 25% WG* (0.3 g/L)")
        reply_lines.append("- **Movento Energy (Bayer)** — *Spirotetramat + Imidacloprid* (1.0 ml/L, 2-way systemic)")
        reply_lines.append("\n#### 🌿 Organic Controls:")
        reply_lines.append("- Install **Yellow Sticky Traps** (20-30 traps/acre) at canopy level.")
        reply_lines.append("- Spray **Cold-Pressed Neem Oil (10,000 ppm)** at 3-5 ml/L + liquid soap emulsifier.")
        suggested_chips = ["🍅 Tomato Yellow Leaf Curl Virus", "🎋 Sugarcane aphid vector control", "🧪 Dosage calculator help", "🌧️ Weather spraying guidelines"]

    elif "tank mix" in msg_lower or "compatibility" in msg_lower:
        reply_lines.append("### 🧪 Pesticide Tank Mixing Rules & Compatibility")
        reply_lines.append("Always use the **WALES / DALES method** for adding products into your spray tank:")
        reply_lines.append("1. **W** — **W**P / WDG (Wettable powders & granules) pre-slurried in water.")
        reply_lines.append("2. **A** — **A**gitate tank thoroughly with 50% water filled.")
        reply_lines.append("3. **L** — **L**iquid flowables & SC / SL formulations.")
        reply_lines.append("4. **E** — **E**C (Emulsifiable concentrates).")
        reply_lines.append("5. **S** — **S**urfactants & bio-stimulants last; top off with water.")
        reply_lines.append("\n> **⚠️ Warning:** Never mix copper fungicides with acidic foliar fertilizers or organophosphates in the same tank without a jar compatibility test.")
        suggested_chips = ["📏 Calculate field tank volumes", "🧪 Top fungicide brands", "🌿 Organic spray alternatives"]

    else:
        # General Help / Fallback
        reply_lines.append("### 🤖 AgroAI Smart Agronomist Assistant")
        reply_lines.append("I am ready to help you optimize your crop health and yields. You can ask me about:")
        reply_lines.append("- **Specific Crop Disease Treatments:** e.g., *'How to cure Tomato Late Blight?'* or *'What is the dosage for Sugarcane Red Rot?'*")
        reply_lines.append("- **Commercial Brand Recommendations:** Top products from Syngenta, Bayer, FMC, Corteva, BASF, and Tata Rallis.")
        reply_lines.append("- **Organic Biological Remedies:** Neem oil, Trichoderma, Bacillus subtilis, potassium bicarbonate, and copper.")
        reply_lines.append("- **Spraying & Weather Precautions:** Rainfastness, temperature limits, and humidity outbreak risks.")
        reply_lines.append("- **Field Dosage Calculations:** Tank refills and chemical quantities for your farm.")
        suggested_chips = ["🍅 Tomato diseases & remedies", "🎋 Sugarcane Red Rot & Rust", "🌾 Rice Blast & Brown Spot", "🧪 Top market best sellers"]

    return {
        "reply": "\n".join(reply_lines),
        "suggested_actions": suggested_chips[:4],
        "crop": active_crop,
        "disease": active_disease
    }
