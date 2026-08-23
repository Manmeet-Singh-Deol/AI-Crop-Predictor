"""
AI Agronomist Chatbot Engine for AgroAI Platform
Provides multi-lingual conversational agricultural intelligence, disease diagnostics,
commercial and organic treatment protocols, dosage calculations, and weather advisories.
Supports Hindi, Punjabi, English, Spanish, and French with Gemini API acceleration & local expert fallback.
"""

import os
import re
import json
import httpx
from typing import Dict, Any, List, Optional
from backend.advisory_db import ADVISORY_DATABASE, get_advisory

# ---------------------------------------------------------------------------
# Multilingual Crop & Disease Translation Mapping for Intent Understanding
# ---------------------------------------------------------------------------
CROP_TRANSLATIONS: Dict[str, str] = {
    # Hindi / Hinglish
    "टमाटर": "Tomato", "tamatar": "Tomato",
    "गेहूं": "Wheat", "gehu": "Wheat", "gehun": "Wheat", "kanak": "Wheat",
    "गन्ना": "Sugarcane", "ganna": "Sugarcane", "kamad": "Sugarcane",
    "धान": "Rice", "चावल": "Rice", "dhan": "Rice", "chawal": "Rice", "jhona": "Rice",
    "आलू": "Potato", "aloo": "Potato", "alu": "Potato",
    "कपास": "Cotton", "kapaas": "Cotton", "narma": "Cotton",
    "मक्का": "Corn", "मकई": "Corn", "makka": "Corn", "makki": "Corn", "corn": "Corn",
    "सेब": "Apple", "seb": "Apple",
    "अंगूर": "Grape", "angoor": "Grape", "dakh": "Grape",
    "केला": "Banana", "kela": "Banana",
    "मिर्च": "Pepper", "mirch": "Pepper", "shimla mirch": "Pepper",
    "सोयाबीन": "Soybean", "soyabean": "Soybean",
    "चाय": "Tea", "chai": "Tea",
    "कॉफ़ी": "Coffee", "coffee": "Coffee",
    "संतरा": "Orange", "santara": "Orange", "kinnu": "Orange",
    "स्ट्रॉबेरी": "Strawberry", "strawberry": "Strawberry",
    "चेरी": "Cherry", "cherry": "Cherry",
    "आड़ू": "Peach", "aadoo": "Peach"
}

DISEASE_TRANSLATIONS: Dict[str, str] = {
    "झुलसा": "blight", "jhulsa": "blight", "pacheti jhulsa": "late_blight", "agati jhulsa": "early_blight",
    "रतुआ": "rust", "ratua": "rust", "peela ratua": "yellow_rust", "brown rust": "brown_rust",
    "सड़न": "rot", "sadan": "rot", "laal sadan": "red_rot", "black rot": "black_rot",
    "ब्लास्ट": "blast", "blast rog": "leaf_blast",
    "पाउडरी मिल्ड्यू": "powdery_mildew", "churna asita": "powdery_mildew", "mildew": "powdery_mildew",
    "मोज़ेक": "mosaic", "mosaic virus": "mosaic",
    "धब्बा": "spot", "dhabba": "leaf_spot", "tikka": "leaf_spot", "scab": "scab",
    "सफेद मक्खी": "whitefly", "safed makkhi": "whitefly", "churda murda": "curl_virus",
    "माइट": "spider_mites", "keeda": "pest", "sundi": "caterpillar"
}

def detect_language(text: str, default_lang: str = "en") -> str:
    """Detect if input text is in Hindi (Devanagari), Punjabi (Gurmukhi), or other."""
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    if re.search(r'[\u0A00-\u0A7F]', text):
        return "pa"
    
    # Check common Hinglish tokens
    hinglish_tokens = ["kya", "kaise", "kare", "karna", "dawa", "dawaii", "ilaj", "upchar", "rog", "khad", "paani", "kitna", "dalen", "tamatar", "gehu", "ganna", "dhan", "kheti", "kisana"]
    words = text.lower().split()
    if sum(1 for w in words if w in hinglish_tokens) >= 2:
        return "hi"

    # Default to user's selected UI language
    if default_lang in ["hi", "pa", "es", "fr"]:
        return default_lang
    return "en"

def call_gemini_agronomist(prompt: str, context: Optional[Dict[str, Any]], lang: str) -> Optional[str]:
    """Call Google Gemini API if GEMINI_API_KEY / GOOGLE_API_KEY is available."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    system_instruction = (
        "You are AgroBot, an expert precision agronomist and plant pathologist for the AgroAI platform.\n"
        "Your mission is to provide concise, scientifically accurate, and highly actionable advice to farmers.\n"
        "Rules:\n"
        f"1. RESPOND IN THE USER'S NATIVE LANGUAGE: '{lang}' (e.g. if 'hi', respond in natural, clear Hindi; if 'pa', in Punjabi; if 'en', in English).\n"
        "2. Provide exact commercial brand names (Syngenta, Bayer, FMC, BASF, Corteva, Tata Rallis), active ingredients, and exact dosage per liter and per 15L backpack pump.\n"
        "3. Always include eco-friendly organic remedies (Neem oil, Trichoderma, Pseudomonas, cow urine formulations).\n"
        "4. Mention safety precautions, Pre-Harvest Intervals (PHI), and weather spraying conditions (rainfastness, Delta-T, wind drift).\n"
        "5. Format with clear markdown bullet points and emojis for quick readability."
    )

    ctx_str = ""
    if context:
        ctx_str = f"\n[Active Leaf Diagnosis Context: Crop: {context.get('crop')}, Disease: {context.get('disease')}, Severity: {context.get('severity_pct')}%, Urgency: {context.get('severity_stage')}]"

    full_prompt = f"{ctx_str}\nFarmer Inquiry: {prompt}"

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"System: {system_instruction}\n\nUser: {full_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800
            }
        }
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
    except Exception as e:
        print(f"[Chatbot] Gemini API notice: {e}. Falling back to built-in agronomist engine.")
    return None

# ---------------------------------------------------------------------------
# Core Multilingual Advisory Generator (Local Fallback & Built-in Engine)
# ---------------------------------------------------------------------------
def generate_expert_response(
    message: str,
    history: List[Dict[str, str]] = None,
    context: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Generate agronomic advisory response with contextual awareness of active scan,
    supporting Hindi, Punjabi, Spanish, French, and English.
    """
    msg_raw = message.strip()
    msg_lower = msg_raw.lower()
    history = history or []
    
    # 1. Determine active language
    lang = detect_language(msg_raw, default_lang=language)

    # 2. Check if Google Gemini is enabled for direct AI generation
    gemini_reply = call_gemini_agronomist(msg_raw, context, lang)
    if gemini_reply:
        suggested_chips = [
            "🧪 Best commercial products" if lang != "hi" else "🧪 सर्वोत्तम कीटनाशक उत्पाद",
            "🌿 Organic treatments" if lang != "hi" else "🌿 जैविक व प्राकृतिक उपाय",
            "🌧️ Spray window & weather" if lang != "hi" else "🌧️ छिड़काव मौसम व सावधानी",
            "📏 Tank dosage calculator" if lang != "hi" else "📏 स्प्रे पंप दवा कैलकुलेटर"
        ]
        return {
            "reply": gemini_reply,
            "suggested_actions": suggested_chips,
            "crop": context.get("crop") if context else None,
            "disease": context.get("disease") if context else None
        }

    # 3. Built-in Multilingual Rule Engine
    active_crop = context.get("crop") if context else None
    active_disease = context.get("disease") if context else None
    active_severity = context.get("severity_pct") if context else None
    active_stage = context.get("severity_stage") if context else None

    # Resolve Crop from Query (Hindi or English)
    target_crop = None
    for k_token, mapped_crop in CROP_TRANSLATIONS.items():
        if k_token in msg_lower:
            target_crop = mapped_crop
            break
            
    # Resolve Disease from Query (Hindi or English)
    target_disease = None
    for k_token, mapped_disease in DISEASE_TRANSLATIONS.items():
        if k_token in msg_lower:
            target_disease = mapped_disease
            break

    # Match database record
    matched_advisory_key = None
    if context and context.get("class_name") and context.get("class_name") in ADVISORY_DATABASE:
        matched_advisory_key = context.get("class_name")

    if not matched_advisory_key:
        for k in ADVISORY_DATABASE.keys():
            k_clean = k.lower().replace("_", " ").replace("___", " ")
            parts = k.split("___")
            c_name = parts[0].replace("_", " ").lower()
            d_name = parts[1].replace("_", " ").lower() if len(parts) > 1 else ""
            if (c_name in msg_lower and d_name in msg_lower) or (d_name and d_name in msg_lower):
                matched_advisory_key = k
                break

    if not matched_advisory_key and target_crop:
        for k in ADVISORY_DATABASE.keys():
            if target_crop.lower() in k.lower():
                if target_disease and target_disease.lower() in k.lower():
                    matched_advisory_key = k
                    break
                elif not matched_advisory_key:
                    matched_advisory_key = k

    if not matched_advisory_key:
        for k in ADVISORY_DATABASE.keys():
            parts = k.split("___")
            c_name = parts[0].replace("_", " ").lower()
            if c_name in msg_lower and len(c_name) > 3:
                matched_advisory_key = k
                break

    if not matched_advisory_key and active_crop:
        for k in ADVISORY_DATABASE.keys():
            if active_crop.lower() in k.lower():
                matched_advisory_key = k
                break

    # Intent Classification
    is_organic = any(w in msg_lower for w in [
        "organic", "natural", "bio", "home remedy", "neem", "chemical-free", "eco",
        "जैविक", "देसी", "नीम", "प्राकृतिक", "घरेलू", "jaivik", "desi"
    ])
    is_chemical = any(w in msg_lower for w in [
        "chemical", "spray", "fungicide", "insecticide", "pesticide", "brand", "medicine",
        "dawa", "dawaii", "formula", "syngenta", "bayer", "fmc", "basf",
        "दवा", "दवाई", "कीटनाशक", "फफूंदनाशक", "स्प्रे", "उपचार", "इलाज", "ilaj", "upchar"
    ])
    is_dosage = any(w in msg_lower for w in [
        "dose", "dosage", "how much", "quantity", "tank", "liter", "acre", "dilution", "pump",
        "मात्रा", "खुराक", "कितना", "पंप", "टैंक", "matra", "khurak", "kitna"
    ])
    is_symptoms = any(w in msg_lower for w in [
        "symptom", "identify", "look like", "sign", "cause",
        "लक्षण", "पहचान", "कारण", "रोग", "lakshan", "rog"
    ])
    is_weather = any(w in msg_lower for w in [
        "weather", "rain", "temperature", "wind", "humidity", "spray today", "when to spray",
        "मौसम", "बारिश", "छिड़काव", "हवा", "तापमान", "mausam", "barish"
    ])
    is_greeting = any(msg_lower == w or msg_lower.startswith(w + " ") for w in [
        "hi", "hello", "hey", "namaste", "sasrikal", "hola", "bonjour", "help",
        "नमस्ते", "प्रणाम", "सत श्री अकाल", "राम राम", "kisan"
    ])

    advisory = ADVISORY_DATABASE.get(matched_advisory_key) if matched_advisory_key else None
    crop_name = target_crop or active_crop or (advisory["crop"] if advisory else "फसल / Crop")
    disease_name = (advisory["disease"] if advisory else (active_disease or "पादप रोग / Disease"))

    reply_lines = []
    suggested_chips = []

    # -----------------------------------------------------------------------
    # HINDI RESPONSE GENERATION
    # -----------------------------------------------------------------------
    if lang == "hi":
        if is_greeting and not (is_organic or is_chemical or is_dosage or is_symptoms or is_weather):
            reply_lines.append("👋 **नमस्ते किसान भाई! मैं आपका AgroAI कृषि विशेषज्ञ बॉट हूँ।**")
            reply_lines.append("मैं फसल रोगों की पहचान, बाज़ार की सबसे असरदार दवाइयों (Syngenta, Bayer, FMC), जैविक नुस्खों, स्प्रे पंप की सही खुराक और मौसम के अनुसार छिड़काव की सलाह दे सकता हूँ।")
            if active_crop and active_disease:
                reply_lines.append(f"\n📌 **वर्तमान स्कैन:** `{active_crop} - {active_disease}` (गंभीरता: {active_severity or 0}% - {active_stage or 'चरण'})")
                suggested_chips = [f"🧪 {active_disease} की सबसे अच्छी दवा", f"🌿 {active_crop} के जैविक उपाय", "🌧️ क्या आज स्प्रे करना सुरक्षित है?", "📏 15 लीटर पंप की खुराक"]
            else:
                reply_lines.append("\n**आप अपनी फसल के बारे में क्या जानना चाहते हैं?**")
                suggested_chips = ["🍅 टमाटर के रोग और दवा", "🌾 गेहूं का पीला रतुआ (Yellow Rust)", "🎋 गन्ने का लाल सड़न (Red Rot)", "🧪 टॉप फफूंदनाशक उत्पाद"]
            return {
                "reply": "\n".join(reply_lines),
                "suggested_actions": suggested_chips,
                "crop": active_crop,
                "disease": active_disease
            }

        if advisory:
            chems = advisory.get("chemical_controls", [])
            organics = advisory.get("organic_controls", [])
            symptoms = advisory.get("symptoms", [])
            cultural = advisory.get("cultural_practices", [])

            reply_lines.append(f"### 🌾 कृषि विशेषज्ञ परामर्श: **{crop_name} — {disease_name}**")
            reply_lines.append(f"🔬 **रोगजनक (Pathogen):** *{advisory.get('scientific_name', 'रोग प्रोफाइल')}* | **जोखिम स्तर:** `{advisory.get('severity_risk', 'उच्च / High')}`\n")

            if is_chemical or (not is_organic and not is_symptoms and not is_weather and not is_dosage):
                reply_lines.append("#### 🧪 बाज़ार में उपलब्ध सर्वश्रेष्ठ रासायनिक दवाइयाँ व ब्रांड:")
                for c in chems:
                    reply_lines.append(f"- **{c.get('product', 'Commercial Brand')}**")
                    reply_lines.append(f"  • **सक्रिय घटक (Active Ingredient):** `{c.get('active_ingredient', 'Standard')}`")
                    reply_lines.append(f"  • **खुराक:** `{c.get('dosage', 'Standard')}` *(15 लीटर के पंप में लगभग 15-30 मिली/ग्राम)*")
                    reply_lines.append(f"  • **छिड़काव का समय:** {c.get('timing', 'रोग के प्रारंभिक लक्षण दिखने पर')}")
                    reply_lines.append(f"  • **सुरक्षा अंतराल (PHI):** `{c.get('interval', '14 दिन')}` बाद ही फसल तोड़ें।\n")
                suggested_chips = ["🌿 जैविक व देसी नुस्खे दिखाएं", "📏 स्प्रे पंप दवा कैलकुलेटर", "🌧️ छिड़काव के लिए मौसम कैसा है?", "🔍 मुख्य रोग लक्षण क्या हैं?"]

            if is_organic:
                reply_lines.append("#### 🌿 अनुशंसित जैविक व प्राकृतिक नियंत्रण विधि:")
                for org in organics:
                    reply_lines.append(f"- 🌱 {org}")
                reply_lines.append("\n> **💡 विशेषज्ञ सलाह:** जैविक फफूंदनाशक (जैसे *ट्राइकोडर्मा* या *स्यूडोमोनास*) और *नीम का तेल* सुबह जल्दी या शाम को धूप ढलने के बाद ही स्प्रे करें ताकि धूप (UV किरणों) से जीवाणु नष्ट न हों।")
                suggested_chips = ["🧪 रासायनिक ब्रांड्स से तुलना करें", "🚜 खेत प्रबंधन और रोकथाम", "📏 स्प्रे पंप की मात्रा"]

            if is_symptoms:
                reply_lines.append("#### 🔍 रोग के प्रमुख दृश्य लक्षण:")
                for sym in symptoms:
                    reply_lines.append(f"- 🔎 {sym}")
                suggested_chips = [f"🧪 {disease_name} का रासायनिक इलाज", f"🌿 {crop_name} के लिए जैविक स्प्रे", "🌧️ मौसम संबंधी सावधानी"]

            if is_weather:
                reply_lines.append("#### 🌧️ मौसम एवं छिड़काव सुरक्षा नियम:")
                reply_lines.append("- **वर्षा से बचाव (Rainfastness):** सिस्टेमिक दवाइयों (जैसे Amistar Top, Nativo) के छिड़काव के बाद कम से कम **2 से 3 घंटे** तक बारिश नहीं होनी चाहिए।")
                reply_lines.append("- **हवा की गति (Wind Speed):** यदि हवा की गति **15 किमी/घंटा** से अधिक हो तो स्प्रे न करें, वरना दवा हवा में उड़कर व्यर्थ हो जाएगी।")
                reply_lines.append("- **तापमान (Delta-T):** दोपहर की तेज धूप (32°C से ऊपर) में स्प्रे करने से पत्तियों के जलने का खतरा रहता है।")
                suggested_chips = ["🧪 दवा की सही खुराक", "🌿 जैविक कीटनाशक", "📏 खेत स्प्रे कैलकुलेटर"]

            if is_dosage:
                reply_lines.append("#### 📏 स्प्रे पंप एवं दवा की मात्रा का हिसाब:")
                reply_lines.append("- **साधारण पीठ वाला पंप (Knapsack Pump):** 15 लीटर पानी")
                reply_lines.append("- **प्रति एकड़ पानी की आवश्यकता:** 150 से 200 लीटर पानी (लगभग 10-12 पंप)")
                if chems:
                    c0 = chems[0]
                    reply_lines.append(f"- **{c0.get('product', 'दवा')} के लिए:** `{c0.get('dosage')}` की दर से प्रति 15 लीटर पंप में **लगभग 15-30 मिली/ग्राम** दवा घोलें।")
                reply_lines.append("\n> **💡 टिप:** सटीक हिसाब के लिए नीचे दिए गए **'खेत स्प्रे कैलकुलेटर'** का उपयोग करें!")
                suggested_chips = ["🧪 सभी रासायनिक ब्रांड देखें", "🌿 जैविक नुस्खे देखें"]

            return {
                "reply": "\n".join(reply_lines),
                "suggested_actions": suggested_chips[:4],
                "crop": crop_name,
                "disease": disease_name
            }
        else:
            # General Hindi Fallback
            reply_lines.append("### 🤖 AgroAI कृषि सहायक")
            reply_lines.append(f"मैं आपकी **{crop_name}** या किसी भी फसल की बीमारी के बारे में पूरी जानकारी दे सकता हूँ। आप मुझसे पूछ सकते हैं:")
            reply_lines.append("- **फसल की दवा और ब्रांड:** जैसे *'टमाटर में झुलसा रोग की दवा क्या है?'* या *'गन्ने के लाल सड़न रोग का इलाज'*")
            reply_lines.append("- **जैविक नुस्खे:** नीम तेल, ट्राइकोडर्मा और जीवामृत का प्रयोग।")
            reply_lines.append("- **स्प्रे की खुराक:** प्रति पंप या प्रति एकड़ दवा की सही मात्रा।")
            reply_lines.append("- **मौसम का हाल:** आज स्प्रे करना सुरक्षित है या बारिश से दवा धुल जाएगी।")
            suggested_chips = ["🍅 टमाटर के रोग व दवा", "🌾 गेहूं का पीला रतुआ", "🎋 गन्ने का लाल सड़न", "🧪 टॉप फफूंदनाशक"]
            return {
                "reply": "\n".join(reply_lines),
                "suggested_actions": suggested_chips,
                "crop": active_crop,
                "disease": active_disease
            }

    # -----------------------------------------------------------------------
    # ENGLISH / STANDARD MULTILINGUAL ENGINE
    # -----------------------------------------------------------------------
    if is_greeting and not (is_organic or is_chemical or is_dosage or is_symptoms or is_weather):
        reply_lines.append("👋 **Hello! I am your AI Agronomist & Crop Protection Specialist.**")
        reply_lines.append("I can help you with crop disease treatment, chemical market best-sellers (Syngenta, Bayer, Corteva, FMC), organic remedies, spray dosage calculation, and weather-safe spraying advice.")
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

    if advisory:
        crop = advisory["crop"]
        disease = advisory["disease"]
        chems = advisory.get("chemical_controls", [])
        organics = advisory.get("organic_controls", [])
        symptoms = advisory.get("symptoms", [])
        cultural = advisory.get("cultural_practices", [])
        weather_trig = advisory.get("environmental_triggers", {})
        
        reply_lines.append(f"### 🌾 Expert Advisory: **{crop} — {disease}**")
        reply_lines.append(f"🔬 **Pathogen:** *{advisory.get('scientific_name', 'Pathogen Profile')}* | **Risk Level:** `{advisory.get('severity_risk', 'High')}`\n")
        
        if is_chemical or (not is_organic and not is_symptoms and not is_weather and not is_dosage):
            reply_lines.append("#### 🧪 Top Commercial Brand Formulations & Market Best Sellers:")
            for c in chems:
                reply_lines.append(f"- **{c.get('product', 'Commercial Brand')}**")
                reply_lines.append(f"  • **Active Ingredient:** `{c.get('active_ingredient', 'Standard')}`")
                reply_lines.append(f"  • **Dosage:** `{c.get('dosage', 'Standard')}` | **Type:** *{c.get('type', 'Fungicide / Insecticide')}*")
                reply_lines.append(f"  • **Application:** {c.get('timing', 'At onset')}")
                reply_lines.append(f"  • **Safety Interval (PHI):** `{c.get('interval', '14 days')}`\n")
            suggested_chips = ["🌿 Show organic bio-controls", "📏 How to calculate tank dosage?", "🔍 What are the key symptoms?", "🌧️ Weather spray precautions"]

        if is_organic:
            reply_lines.append("#### 🌿 Recommended Biological & Organic Treatments:")
            for org in organics:
                reply_lines.append(f"- 🌱 {org}")
            reply_lines.append("\n> **💡 Agronomist Tip:** Apply organic bio-fungicides (like *Bacillus subtilis* or *Trichoderma*) early in the morning or late evening to protect living spores from UV breakdown.")
            suggested_chips = ["🧪 Compare with chemical brands", "🚜 Cultural field prevention", "📏 Spray tank dosage"]

        if is_symptoms:
            reply_lines.append("#### 🔍 Diagnostic Visual Symptoms:")
            for sym in symptoms:
                reply_lines.append(f"- 🔎 {sym}")
            reply_lines.append(f"\n⚡ **Weather Triggers:** {weather_trig.get('high_risk_weather', 'High humidity and temperature swings.')}")
            suggested_chips = [f"🧪 Chemical cure for {disease}", f"🌿 Organic spray for {crop}", "🌧️ Weather precautions"]

        if is_weather:
            reply_lines.append("#### 🌧️ Weather & Spraying Safety Guidelines:")
            reply_lines.append(f"- **Critical Humidity:** {weather_trig.get('critical_humidity_pct', 80)}% | **Optimal Pathogen Temp:** {weather_trig.get('optimal_temp_c', '20-28°C')}")
            reply_lines.append("- **Rainfastness:** Ensure at least 2.5–4 hours of dry weather after applying systemic fungicides so the chemical absorbs into leaf cuticles.")
            reply_lines.append("- **Wind Speed:** Do not spray if wind speed exceeds 15 km/h to prevent spray drift and chemical loss.")
            reply_lines.append("- **Temperature:** Avoid mid-day spraying when temperatures exceed 32°C to prevent rapid droplet evaporation.")
            suggested_chips = ["🧪 Recommended chemical dosage", "🌿 Organic bio-stimulants", "📏 Tank calculator help"]

        if is_dosage:
            reply_lines.append("#### 📏 Spray Tank & Dosage Guidelines:")
            reply_lines.append("- **Standard Backpack Knapsack Tank:** 15 Liters")
            reply_lines.append("- **Standard Water Requirement per Acre:** 150 to 200 Liters of water (10-12 tanks/acre).")
            if chems:
                c0 = chems[0]
                reply_lines.append(f"- **Example for {c0.get('product', 'Top Product')}:** At `{c0.get('dosage')}`, add approximately **15-30 g (or ml)** per 15L backpack tank.")
            reply_lines.append("\n> **💡 Pro-Tip:** You can also use our **Field Spray Tank & Dosage Calculator** below on this page for instant multi-tank calculations!")
            suggested_chips = ["🧪 View all chemical brands", "🌿 View organic recipes"]

        return {
            "reply": "\n".join(reply_lines),
            "suggested_actions": suggested_chips[:4],
            "crop": crop,
            "disease": disease
        }

    # General Fallback
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
