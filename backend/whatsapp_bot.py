"""
WhatsApp Diagnostic Bot & Farm Advisory Engine for AgroAI
Handles incoming WhatsApp messages (Twilio and Meta Cloud API formats),
leaf photo diagnosis, GPS location spray window checks, and conversational agronomy advice.
"""

import os
import io
import re
import json
import base64
import httpx
from PIL import Image
from typing import Dict, Any, Optional, Tuple

from backend.advisory_db import ADVISORY_DATABASE, get_advisory
from backend.chatbot_engine import generate_expert_response, detect_language

def format_whatsapp_prescription(diag_data: Dict[str, Any], language: str = "en") -> str:
    """
    Format a complete leaf diagnosis into a clean, mobile-friendly WhatsApp prescription.
    """
    top_pred = diag_data.get("top_prediction", {})
    crop = top_pred.get("crop", "Crop")
    disease = top_pred.get("disease", "Disease")
    confidence = top_pred.get("confidence", 0.0)
    
    severity = diag_data.get("severity", {})
    sev_pct = severity.get("severity_percentage", 0.0)
    sev_stage = severity.get("severity_stage", "Stage 1")
    
    advisory = diag_data.get("advisory", {})
    chems = advisory.get("chemical_controls", [])
    organics = advisory.get("organic_controls", [])
    pathogen_type = advisory.get("pathogen_type", "Pathogen")
    
    is_hi = (language == "hi")
    
    lines = []
    if is_hi:
        lines.append("🌾 *AgroAI फसल निदान एवं उपचार पर्ची* 🌾")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🌱 *फसल:* {crop}")
        lines.append(f"🦠 *पहचाना गया रोग:* *{disease}*")
        lines.append(f"🔬 *रोगजनक प्रकार:* {pathogen_type}")
        lines.append(f"📊 *सटीकता स्कोर:* `{confidence:.1f}%`")
        lines.append(f"⚠️ *संक्रमण स्तर:* `{sev_pct:.1f}%` ({sev_stage})")
        lines.append("")
        
        if chems:
            lines.append("🧪 *अनुशंसित बाज़ार की सर्वश्रेष्ठ दवाइयाँ:*")
            for c in chems[:2]:
                lines.append(f"• *{c.get('product', 'Commercial Product')}*")
                lines.append(f"  - सक्रिय घटक: `{c.get('active_ingredient', '')}`")
                lines.append(f"  - खुराक: `{c.get('dosage', '')}` (15L पंप में लगभग 20-30ml/g)")
                lines.append(f"  - सुरक्षा अंतराल (PHI): `{c.get('interval', '14 दिन')}`")
            lines.append("")
            
        if organics:
            lines.append("🌿 *जैविक एवं प्राकृतिक उपचार:*")
            for org in organics[:2]:
                lines.append(f"• {org}")
            lines.append("")
            
        lines.append("🌧️ *छिड़काव सावधानी:*")
        lines.append("• सिस्टेमिक दवा के छिड़काव के बाद 2-3 घंटे तक बारिश नहीं होनी चाहिए।")
        lines.append("• 15 किमी/घंटे से तेज़ हवा में स्प्रे न करें।")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("💬 _अधिक जानकारी के लिए कोई भी प्रश्न पूछें या नया फोटो भेजें।_")
    else:
        lines.append("🌾 *AgroAI Crop Pathology Prescription* 🌾")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🌱 *Crop:* {crop}")
        lines.append(f"🦠 *Diagnosed Disease:* *{disease}*")
        lines.append(f"🔬 *Pathogen Type:* {pathogen_type}")
        lines.append(f"📊 *AI Confidence:* `{confidence:.1f}%`")
        lines.append(f"⚠️ *Infection Severity:* `{sev_pct:.1f}%` ({sev_stage})")
        lines.append("")
        
        if chems:
            lines.append("🧪 *Top Commercial Brand Interventions:*")
            for c in chems[:2]:
                lines.append(f"• *{c.get('product', 'Commercial Product')}*")
                lines.append(f"  - Active Ingredient: `{c.get('active_ingredient', '')}`")
                lines.append(f"  - Recommended Dosage: `{c.get('dosage', '')}` (~20-30ml/g per 15L tank)")
                lines.append(f"  - Pre-Harvest Interval (PHI): `{c.get('interval', '14 days')}`")
            lines.append("")
            
        if organics:
            lines.append("🌿 *Biological & Organic Controls:*")
            for org in organics[:2]:
                lines.append(f"• {org}")
            lines.append("")
            
        lines.append("🌧️ *Spray Safety Guidelines:*")
        lines.append("• Ensure 3-4 hours of dry weather after spraying systemic formulations.")
        lines.append("• Avoid spraying when wind speeds exceed 15 km/h to prevent drift.")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("💬 _Type any farming question or send another leaf photo to continue._")
        
    return "\n".join(lines)

def format_whatsapp_location_advisory(lat: float, lon: float, weather_data: Dict[str, Any], language: str = "en") -> str:
    """
    Format real-time microclimate weather and spray window into WhatsApp advisory.
    """
    cur = weather_data.get("current", {})
    temp = cur.get("temperature_2m", 25.0)
    rh = cur.get("relative_humidity_2m", 60.0)
    wind = cur.get("wind_speed_10m", 8.0)
    rain = cur.get("precipitation", 0.0)
    
    spray_analysis = weather_data.get("spray_window_analysis", {})
    delta_t = spray_analysis.get("current_conditions", {}).get("delta_t", {})
    dt_val = delta_t.get("delta_t_c", 5.0)
    dt_rating = delta_t.get("rating", "optimal").upper()
    
    wind_eval = spray_analysis.get("current_conditions", {}).get("wind_evaluation", {})
    wind_status = wind_eval.get("status", "Safe Wind Speed")
    
    washout = spray_analysis.get("current_conditions", {}).get("washout_protection", {})
    washout_status = washout.get("status", "Clear for Spraying")
    
    next_win = spray_analysis.get("next_safe_window", {})
    is_safe_now = (next_win.get("status") == "safe_now")
    
    is_hi = (language == "hi")
    lines = []
    
    if is_hi:
        lines.append("🌤️ *AgroAI मौसम एवं स्प्रे सुरक्षा रिपोर्ट* 🌤️")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📍 *स्थान निर्देशांक:* Lat {lat:.2f}, Lon {lon:.2f}")
        lines.append(f"🌡️ *तापमान / नमी:* {temp}°C / {rh}%")
        lines.append(f"💨 *हवा की गति:* {wind} km/h ({wind_status})")
        lines.append(f"💧 *डेल्टा-टी (ΔT):* `{dt_val:.1f}°C` ({dt_rating})")
        lines.append(f"🌧️ *वर्षा धुलाई जोखिम:* {washout_status}")
        lines.append("")
        if is_safe_now:
            lines.append("✅ *छिड़काव निर्णय:* *अभी स्प्रे करना बिल्कुल सुरक्षित है!*")
            lines.append(f"• अनुकूल समय: {next_win.get('summary', 'अगले 4 घंटे अनुकूल परिस्थितियां')}")
        else:
            lines.append("⚠️ *छिड़काव निर्णय:* *अभी स्प्रे करने से बचें।*")
            lines.append(f"• अगला सुरक्षित समय: *{next_win.get('summary', 'मौसम साफ होने की प्रतीक्षा करें')}*")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("🌾 _AgroAI द्वारा संचालित_")
    else:
        lines.append("🌤️ *AgroAI Microclimate & Spray Advisory* 🌤️")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📍 *GPS Location:* Lat {lat:.2f}, Lon {lon:.2f}")
        lines.append(f"🌡️ *Temp / Humidity:* {temp}°C / {rh}%")
        lines.append(f"💨 *Wind Speed:* {wind} km/h ({wind_status})")
        lines.append(f"💧 *Delta-T (ΔT):* `{dt_val:.1f}°C` ({dt_rating})")
        lines.append(f"🌧️ *Washout Risk:* {washout_status}")
        lines.append("")
        if is_safe_now:
            lines.append("✅ *Spray Decision:* *SAFE TO SPRAY NOW!*")
            lines.append(f"• Window Details: {next_win.get('summary', 'Next hours have ideal microclimate')}")
        else:
            lines.append("⚠️ *Spray Decision:* *NOT RECOMMENDED RIGHT NOW.*")
            lines.append(f"• Next Safe Opportunity: *{next_win.get('summary', 'Wait for favorable weather')}*")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("🌾 _Powered by AgroAI Precision Engine_")
        
    return "\n".join(lines)

async def download_image_from_url(url: str) -> Optional[Image.Image]:
    """Download image bytes from WhatsApp media URL and convert to PIL Image."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code == 200:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"[WhatsApp] Media download failed: {e}")
    return None

async def process_whatsapp_incoming(
    from_number: str,
    message_text: Optional[str] = None,
    media_url: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    image_base64: Optional[str] = None,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified processor for WhatsApp messages, media, and location pins.
    Returns structured response containing reply text and diagnostic metadata.
    """
    from backend.main import run_full_diagnosis
    from backend.weather_risk import fetch_weather_by_coords
    
    # 1. Determine language from text or query
    detected_lang = language or (detect_language(message_text or "") if message_text else "en")
    
    # 2. Case A: Leaf Photo Attachment Diagnosis
    image_obj = None
    if media_url:
        image_obj = await download_image_from_url(media_url)
    elif image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            img_bytes = base64.b64decode(image_base64)
            image_obj = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"[WhatsApp] Base64 decode note: {e}")

    if image_obj:
        try:
            diag_result = run_full_diagnosis(
                image_obj,
                alpha=0.55,
                colormap="JET",
                target_crop="auto",
                location=f"WhatsApp: {from_number}"
            )
            reply = format_whatsapp_prescription(diag_result, language=detected_lang)
            return {
                "type": "diagnosis",
                "reply": reply,
                "diagnosis_data": diag_result,
                "from": from_number,
                "language": detected_lang
            }
        except Exception as e:
            err_msg = (
                "⚠️ *फसल फोटो जांचने में समस्या आई।* कृपया स्पष्ट और अच्छी रोशनी में पत्ती की फोटो भेजें।"
                if detected_lang == "hi" else
                "⚠️ *Error processing leaf image.* Please send a clear, well-lit photo of the affected plant leaf."
            )
            return {
                "type": "error",
                "reply": err_msg,
                "error": str(e),
                "from": from_number
            }

    # 3. Case B: GPS Location Pin for Outbreak Risk & Spray Window
    if latitude is not None and longitude is not None:
        try:
            w_data = await fetch_weather_by_coords(lat=latitude, lon=longitude, chemical_key="systemic_fungicide")
            reply = format_whatsapp_location_advisory(latitude, longitude, w_data, language=detected_lang)
            return {
                "type": "weather_advisory",
                "reply": reply,
                "weather_data": w_data,
                "from": from_number,
                "language": detected_lang
            }
        except Exception as e:
            err_msg = (
                "⚠️ *मौसम डेटा प्राप्त नहीं हो सका।* कृपया पुनः प्रयास करें।"
                if detected_lang == "hi" else
                "⚠️ *Unable to retrieve weather telemetry for your location.* Please try again."
            )
            return {
                "type": "error",
                "reply": err_msg,
                "error": str(e),
                "from": from_number
            }

    # 4. Case C: Conversational Q&A / Text Agronomic Inquiry
    text = (message_text or "").strip()
    if text:
        chat_resp = generate_expert_response(
            message=text,
            history=[],
            context=None,
            language=detected_lang
        )
        return {
            "type": "chat_advisory",
            "reply": chat_resp.get("reply", ""),
            "suggested_actions": chat_resp.get("suggested_actions", []),
            "from": from_number,
            "language": detected_lang
        }

    # 5. Default Help / Greeting
    if detected_lang == "hi":
        help_text = (
            "👋 *नमस्ते किसान भाई! मैं AgroAI व्हाट्सएप कृषि डॉक्टर हूँ।* 🌾\n\n"
            "आप मुझे:\n"
            "1. 📸 *पत्ती की फोटो भेजें* - तुरंत बीमारी की पहचान व दवा की पर्ची पाएं।\n"
            "2. 📍 *अपनी लोकेशन (GPS Pin) भेजें* - आज स्प्रे करने का सही समय व मौसम का हाल जानें।\n"
            "3. 💬 *कोई भी सवाल लिखें या बोलें* - जैसे 'टमाटर में झुलसा की दवा' या 'गेहूं में पीला रतुआ का इलाज'।\n\n"
            "_फसल स्वास्थ्य और बेहतर पैदावार के लिए AgroAI हमेशा आपके साथ है!_"
        )
    else:
        help_text = (
            "👋 *Welcome to AgroAI WhatsApp Crop Doctor!* 🌾\n\n"
            "How I can help you today:\n"
            "1. 📸 *Send a leaf photo* - Instant disease diagnosis & treatment prescription.\n"
            "2. 📍 *Send your GPS Location pin* - Live Delta-T spray safety & rainfastness check.\n"
            "3. 💬 *Ask any farming question* - e.g. 'Best fungicide for Tomato Blight' or 'Wheat Yellow Rust dosage'.\n\n"
            "_Powered by AgroAI Precision Pathology Core._"
        )
        
    return {
        "type": "help",
        "reply": help_text,
        "from": from_number,
        "language": detected_lang
    }

def generate_twiml_response(reply_text: str) -> str:
    """Generate Twilio TwiML XML response."""
    # XML escape text
    clean_text = reply_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{clean_text}</Message>
</Response>"""
