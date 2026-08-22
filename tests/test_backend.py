"""
Automated Unit & Integration Test Suite for AgroAI Platform
Tests classifier, Grad-CAM, severity analysis, weather risk, sample generation, PDF export, and FastAPI endpoints.
"""

import os
import sys
import unittest
import io
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.classifier import get_inference_engine, CLASS_NAMES
from backend.gradcam import generate_visual_explanation
from backend.severity import quantify_severity
from backend.weather_risk import calculate_disease_risks
from backend.advisory_db import get_advisory, list_all_advisories
from backend.sample_images import (
    generate_tomato_late_blight,
    generate_potato_early_blight,
    generate_apple_scab,
    generate_healthy_tomato,
    get_all_samples_with_thumbnails
)
from backend.report_generator import generate_pdf_report

class TestAgroAIPlatform(unittest.TestCase):

    def setUp(self):
        self.engine = get_inference_engine()
        self.test_leaf = generate_tomato_late_blight()

    def test_01_classifier_classes_and_inference(self):
        """Verify 40+ classes taxonomy and deep inference outputs."""
        self.assertGreaterEqual(len(CLASS_NAMES), 40)
        self.assertIn("Tomato___Late_blight", CLASS_NAMES)
        self.assertIn("Potato___Early_blight", CLASS_NAMES)
        
        result = self.engine.predict(self.test_leaf, top_k=3)
        self.assertIn("top_prediction", result)
        self.assertIn("top_k_predictions", result)
        self.assertEqual(len(result["top_k_predictions"]), 3)
        
        top = result["top_prediction"]
        self.assertIn("crop", top)
        self.assertIn("disease", top)
        self.assertGreaterEqual(top["confidence"], 0.0)
        self.assertLessEqual(top["confidence"], 100.0)

    def test_02_gradcam_heatmaps_and_overlays(self):
        """Verify PyTorch Grad-CAM computation and base64 images."""
        gc = generate_visual_explanation(self.test_leaf, colormap="JET")
        self.assertIn("original_image", gc)
        self.assertIn("heatmap_image", gc)
        self.assertIn("blended_image", gc)
        self.assertTrue(gc["blended_image"].startswith("data:image/"))
        self.assertIn("attention_peak_pct", gc)

    def test_03_severity_quantification(self):
        """Verify OpenCV lesion segmentation, severity index, and stage tagging."""
        sev = quantify_severity(self.test_leaf)
        self.assertIn("severity_percentage", sev)
        self.assertIn("severity_stage", sev)
        self.assertIn("severity_mask_image", sev)
        self.assertIn("lesion_count", sev)
        self.assertGreaterEqual(sev["severity_percentage"], 0.0)
        self.assertLessEqual(sev["severity_percentage"], 100.0)
        self.assertTrue(sev["severity_mask_image"].startswith("data:image/"))

    def test_04_weather_risk_epidemiology(self):
        """Verify microclimate risk calculation under varied weather conditions."""
        # High fungal risk weather: Warm, rainy, high humidity
        high_risk = calculate_disease_risks(temp_c=22.0, humidity_pct=92.0, rain_mm=8.0, wind_kmh=12.0)
        self.assertGreaterEqual(high_risk["fungal_risk_score"], 70.0)
        self.assertIn("threat_level", high_risk)
        
        # Dry low risk weather
        low_risk = calculate_disease_risks(temp_c=18.0, humidity_pct=40.0, rain_mm=0.0, wind_kmh=5.0)
        self.assertLess(low_risk["fungal_risk_score"], 50.0)

    def test_05_advisory_database(self):
        """Verify comprehensive agricultural knowledge base records."""
        adv_list = list_all_advisories()
        self.assertGreaterEqual(len(adv_list), 40)
        
        adv = get_advisory("Tomato___Late_blight")
        self.assertEqual(adv["crop"], "Tomato")
        self.assertEqual(adv["disease"], "Tomato Late Blight")
        self.assertIn("Phytophthora infestans", adv["scientific_name"])
        self.assertGreater(len(adv["organic_controls"]), 0)
        self.assertGreater(len(adv["chemical_controls"]), 0)
        self.assertGreater(len(adv["symptoms"]), 0)

    def test_06_sample_specimens(self):
        """Verify sample images generator and thumbnail encoding."""
        samples = get_all_samples_with_thumbnails()
        self.assertGreaterEqual(len(samples), 6)
        for s in samples:
            self.assertIn("id", s)
            self.assertIn("title", s)
            self.assertIn("thumbnail", s)
            self.assertTrue(s["thumbnail"].startswith("data:image/"))

    def test_07_pdf_report_generation(self):
        """Verify ReportLab PDF certificate compilation."""
        diagnosis_payload = {
            "top_prediction": {"crop": "Tomato", "disease": "Late Blight", "confidence": 96.4},
            "advisory": get_advisory("Tomato___Late_blight"),
            "severity": quantify_severity(self.test_leaf),
            "gradcam": generate_visual_explanation(self.test_leaf),
            "weather": {
                "epidemiological_risk": calculate_disease_risks(21.0, 90.0, 5.0, 10.0)
            }
        }
        pdf_bytes = generate_pdf_report(diagnosis_payload)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_08_sugarcane_classification_and_advisory(self):
        """Verify Sugarcane red rot and rust advisories and sample classification."""
        from backend.sample_images import generate_sugarcane_red_rot
        sugar_leaf = generate_sugarcane_red_rot()
        
        # Test locked and auto crop prediction
        res = self.engine.predict(sugar_leaf, target_crop="Sugarcane", hint_class="Sugarcane___Red_Rot")
        self.assertEqual(res["top_prediction"]["crop"], "Sugarcane")
        self.assertEqual(res["crop_identification"]["detected_crop"], "Sugarcane")
        self.assertEqual(res["crop_identification"]["botanical_name"], "Saccharum officinarum")
        
        # Verify Sugarcane Red Rot advisory
        adv = get_advisory("Sugarcane___Red_Rot")
        self.assertEqual(adv["crop"], "Sugarcane")
        self.assertIn("Colletotrichum falcatum", adv["scientific_name"])
        self.assertEqual(adv["severity_risk"], "Critical")
        self.assertGreater(len(adv["organic_controls"]), 0)

    def test_09_chatbot_engine(self):
        """Verify AI Agronomist chatbot response generation and context mapping."""
        from backend.chatbot_engine import generate_expert_response
        
        # Test generic greeting
        g = generate_expert_response("hello")
        self.assertIn("reply", g)
        self.assertGreater(len(g["reply"]), 50)
        self.assertGreater(len(g["suggested_actions"]), 0)
        
        # Test specific disease inquiry
        r = generate_expert_response("What is the best fungicide for tomato late blight?")
        self.assertIn("reply", r)
        self.assertIn("Late Blight", r["reply"])
        self.assertIn("Ridomil Gold", r["reply"])
        
        # Test contextual inquiry for sugarcane
        s = generate_expert_response(
            "How do I apply sett treatment?",
            context={"crop": "Sugarcane", "disease": "Sugarcane Red Rot"}
        )
        self.assertIn("Sugarcane", s["reply"])
        self.assertIn("Bavistin", s["reply"])

    def test_10_expanded_crops_advisory_and_inference(self):
        """Verify new crops (Wheat, Cotton, Banana, Coffee, Tea, Cassava) advisories and predictions."""
        # 1. Total class count check
        self.assertEqual(len(CLASS_NAMES), 67)

        # 2. Wheat Yellow Rust
        w_adv = get_advisory("Wheat___Yellow_Rust")
        self.assertEqual(w_adv["crop"], "Wheat")
        self.assertIn("Tilt", w_adv["chemical_controls"][0]["product"])

        # 3. Cotton Bacterial Blight
        c_adv = get_advisory("Cotton___Bacterial_Blight")
        self.assertEqual(c_adv["crop"], "Cotton")
        self.assertIn("Plantomycin", c_adv["chemical_controls"][0]["product"])

        # 4. Banana Black Sigatoka
        b_adv = get_advisory("Banana___Black_Sigatoka")
        self.assertEqual(b_adv["crop"], "Banana")
        self.assertIn("Siganex", b_adv["chemical_controls"][0]["product"])

        # 5. Coffee Leaf Rust
        cof_adv = get_advisory("Coffee___Leaf_Rust")
        self.assertEqual(cof_adv["crop"], "Coffee")
        self.assertIn("Amistar", cof_adv["chemical_controls"][0]["product"])

if __name__ == "__main__":
    unittest.main()



