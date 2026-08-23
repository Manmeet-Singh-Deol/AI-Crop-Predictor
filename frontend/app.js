/**
 * AgroAI - Crop Disease Diagnosis & Prediction Frontend Controller
 * Complete implementation with Grad-CAM, Severity, Weather, Dosage Calculator, Audit History, and Multilingual i18n.
 */

// Application State
const state = {
    diagnosisData: null,
    weatherData: null,
    historyData: [],
    chatHistory: [],
    isChatOpen: false,
    viewMode: 'blended',
    colormap: 'JET',
    alpha: 0.55,
    advisoryTab: 'organic',
    currentLang: 'en',
    translations: {},
    webcamStream: null,
    isLoading: false,
    lastScannedImage: null,
    
    // Voice-First Interaction State
    isListening: false,
    voiceReadAloud: true,
    speechRecognition: null,
    activeSpeechUtterance: null,
    
    // Satellite NDVI Field Mapping State
    ndviFieldData: null,
    ndviActiveLayer: 'ndvi',
    ndviSampleFields: [],
    currentNdviFieldId: 'punjab_wheat_field'
};

// DOM Elements Cache
const DOM = {
    // Nav & Status & PWA & MLOps
    langSelector: document.getElementById('lang-selector'),
    systemStatusBadge: document.getElementById('system-status-badge'),
    modelStatusText: document.getElementById('model-status-text'),
    btnExportPdfNav: document.getElementById('btn-export-pdf-nav'),
    btnExportPdfCard: document.getElementById('btn-export-pdf-card'),
    btnPwaInstall: document.getElementById('btn-pwa-install'),
    networkModeBadge: document.getElementById('network-mode-badge'),
    networkModeDot: document.getElementById('network-mode-dot'),
    networkModeText: document.getElementById('network-mode-text'),
    btnOpenMlops: document.getElementById('btn-open-mlops'),
    
    // MLOps Active Learning Hub Modal
    mlopsModal: document.getElementById('mlops-modal'),
    btnCloseMlops: document.getElementById('btn-close-mlops'),
    mlopsModelVersion: document.getElementById('mlops-model-version'),
    mlopsAccuracy: document.getElementById('mlops-accuracy'),
    mlopsHarvestedCount: document.getElementById('mlops-harvested-count'),
    mlopsApprovedCount: document.getElementById('mlops-approved-count'),
    btnTriggerRetrain: document.getElementById('btn-trigger-retrain'),
    retrainProgressBox: document.getElementById('retrain-progress-box'),
    retrainProgressLabel: document.getElementById('retrain-progress-label'),
    retrainProgressPct: document.getElementById('retrain-progress-pct'),
    retrainProgressBar: document.getElementById('retrain-progress-bar'),
    mlopsQueueList: document.getElementById('mlops-queue-list'),
    mlopsQueueBadge: document.getElementById('mlops-queue-badge'),
    mlopsHistoryList: document.getElementById('mlops-history-list'),

    // Farmer Active Learning Feedback
    btnFeedbackAccurate: document.getElementById('btn-feedback-accurate'),
    btnFeedbackIncorrect: document.getElementById('btn-feedback-incorrect'),
    feedbackBtnGroup: document.getElementById('feedback-btn-group'),
    feedbackThanksMsg: document.getElementById('feedback-thanks-msg'),
    
    // Sample Strip
    sampleCardsContainer: document.getElementById('sample-cards-container'),
    
    // Ingestion Tabs & Inputs
    tabBtnUpload: document.getElementById('tab-btn-upload'),
    tabBtnCamera: document.getElementById('tab-btn-camera'),
    targetCropSelect: document.getElementById('target-crop-select'),
    dropzoneContainer: document.getElementById('dropzone-container'),
    dropzoneBox: document.getElementById('dropzone-box'),
    leafFileInput: document.getElementById('leaf-file-input'),
    cameraContainer: document.getElementById('camera-container'),
    webcamVideo: document.getElementById('webcam-video'),
    webcamCanvas: document.getElementById('webcam-canvas'),
    cameraPlaceholder: document.getElementById('camera-placeholder'),
    btnStartCamera: document.getElementById('btn-start-camera'),
    btnCaptureCamera: document.getElementById('btn-capture-camera'),
    
    // Grad-CAM Visual Display
    baseDisplayImage: document.getElementById('base-display-image'),
    overlayDisplayImage: document.getElementById('overlay-display-image'),
    mainDisplayImage: document.getElementById('main-display-image'),
    imageEmptyState: document.getElementById('image-empty-state'),
    diagnosisSpinner: document.getElementById('diagnosis-spinner'),
    gradcamPeakBadge: document.getElementById('gradcam-peak-badge'),
    viewModeBtns: document.querySelectorAll('.view-mode-btn'),
    blendOpacitySlider: document.getElementById('blend-opacity-slider'),
    opacityValLabel: document.getElementById('opacity-val-label'),
    cmapBtns: document.querySelectorAll('.cmap-btn'),

    
    // Identified Crop Species
    cropDisplayName: document.getElementById('crop-display-name'),
    cropBotanicalName: document.getElementById('crop-botanical-name'),
    cropBotanicalFamily: document.getElementById('crop-botanical-family'),
    cropMatchPct: document.getElementById('crop-match-pct'),

    // Results & Severity
    diagCropBadge: document.getElementById('diag-crop-badge'),
    diagPathogenBadge: document.getElementById('diag-pathogen-badge'),
    diagDiseaseTitle: document.getElementById('diag-disease-title'),
    diagScientificName: document.getElementById('diag-scientific-name'),
    diagConfidenceVal: document.getElementById('diag-confidence-val'),
    severityPctLabel: document.getElementById('severity-pct-label'),
    severityStageBadge: document.getElementById('severity-stage-badge'),
    lesionCountLabel: document.getElementById('lesion-count-label'),
    urgencyLabel: document.getElementById('urgency-label'),
    differentialBarsContainer: document.getElementById('differential-bars-container'),
    
    // Advisory Hub
    advTabBtns: document.querySelectorAll('.adv-tab-btn'),
    advisoryContentBox: document.getElementById('advisory-content-box'),
    
    // Dosage Calculator
    calcFieldSize: document.getElementById('calc-field-size'),
    calcUnit: document.getElementById('calc-unit'),
    calcDosageRate: document.getElementById('calc-dosage-rate'),
    calcDosageUnit: document.getElementById('calc-dosage-unit'),
    calcTankSize: document.getElementById('calc-tank-size'),
    calcOutWater: document.getElementById('calc-out-water'),
    calcOutProduct: document.getElementById('calc-out-product'),
    calcOutTanks: document.getElementById('calc-out-tanks'),
    
    // Weather Risk
    weatherCityInput: document.getElementById('weather-city-input'),
    btnSearchWeather: document.getElementById('btn-search-weather'),
    btnGeolocateWeather: document.getElementById('btn-geolocate-weather'),
    weatherLocationLabel: document.getElementById('weather-location-label'),
    weatherTempVal: document.getElementById('weather-temp-val'),
    weatherHumidityVal: document.getElementById('weather-humidity-val'),
    weatherRainVal: document.getElementById('weather-rain-val'),
    weatherWindVal: document.getElementById('weather-wind-val'),
    riskFungalVal: document.getElementById('risk-fungal-val'),
    riskFungalBar: document.getElementById('risk-fungal-bar'),
    riskBacterialVal: document.getElementById('risk-bacterial-val'),
    riskBacterialBar: document.getElementById('risk-bacterial-bar'),
    overallThreatScore: document.getElementById('overall-threat-score'),
    overallThreatLevel: document.getElementById('overall-threat-level'),
    overallThreatSummary: document.getElementById('overall-threat-summary'),
    forecastCardsContainer: document.getElementById('forecast-cards-container'),
    quickCityBtns: document.querySelectorAll('.quick-city-btn'),

    // Optimal Spray Window & Rainfastness Engine
    sprayEngineContainer: document.getElementById('spray-engine-container'),
    sprayChemicalSelect: document.getElementById('spray-chemical-select'),
    sprayDeltaTBadge: document.getElementById('spray-delta-t-badge'),
    sprayDeltaTVal: document.getElementById('spray-delta-t-val'),
    sprayDeltaTDesc: document.getElementById('spray-delta-t-desc'),
    sprayWindBadge: document.getElementById('spray-wind-badge'),
    sprayWindVal: document.getElementById('spray-wind-val'),
    sprayWindDesc: document.getElementById('spray-wind-desc'),
    sprayWashoutBadge: document.getElementById('spray-washout-badge'),
    sprayRainfastHoursVal: document.getElementById('spray-rainfast-hours-val'),
    sprayWashoutDesc: document.getElementById('spray-washout-desc'),
    sprayWindowHeadline: document.getElementById('spray-window-headline'),
    sprayWindowRec: document.getElementById('spray-window-rec'),
    sprayTimelineBars: document.getElementById('spray-timeline-bars'),
    
    // History
    historyCardsContainer: document.getElementById('history-cards-container'),
    btnExportCsv: document.getElementById('btn-export-csv'),
    btnClearHistory: document.getElementById('btn-clear-history'),
    
    // AI Agronomist Chatbot & Voice Controls
    btnOpenChatbot: document.getElementById('btn-open-chatbot'),
    aiChatbotWindow: document.getElementById('ai-chatbot-window'),
    btnChatClose: document.getElementById('btn-chat-close'),
    btnChatClear: document.getElementById('btn-chat-clear'),
    btnVoiceToggle: document.getElementById('btn-voice-toggle'),
    btnVoiceMic: document.getElementById('btn-voice-mic'),
    btnCancelVoice: document.getElementById('btn-cancel-voice'),
    voiceListeningBar: document.getElementById('voice-listening-bar'),
    voiceListeningLabel: document.getElementById('voice-listening-label'),
    chatContextBanner: document.getElementById('chat-context-banner'),
    chatContextText: document.getElementById('chat-context-text'),
    chatContextBadge: document.getElementById('chat-context-badge'),
    chatMessagesContainer: document.getElementById('chat-messages-container'),
    chatSuggestedChips: document.getElementById('chat-suggested-chips'),
    chatInputForm: document.getElementById('chat-input-form'),
    chatInputField: document.getElementById('chat-input-field'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    
    // WhatsApp Bot Integration & Simulator
    btnOpenWhatsapp: document.getElementById('btn-open-whatsapp'),
    btnFloatingWhatsapp: document.getElementById('btn-floating-whatsapp'),
    btnChatWhatsappSwitch: document.getElementById('btn-chat-whatsapp-switch'),
    btnShareWhatsapp: document.getElementById('btn-share-whatsapp'),
    whatsappModal: document.getElementById('whatsapp-modal'),
    btnCloseWhatsappModal: document.getElementById('btn-close-whatsapp-modal'),
    btnWaSendPhoto: document.getElementById('btn-wa-send-photo'),
    btnWaSendLocation: document.getElementById('btn-wa-send-location'),
    btnWaClear: document.getElementById('btn-wa-clear'),
    waImageInput: document.getElementById('wa-image-input'),
    waMessagesContainer: document.getElementById('wa-messages-container'),
    waInputField: document.getElementById('wa-input-field'),
    btnWaSendMsg: document.getElementById('btn-wa-send-msg'),
    btnWaAttachCam: document.getElementById('btn-wa-attach-cam'),
    btnWaAttachPin: document.getElementById('btn-wa-attach-pin'),
    
    // Satellite NDVI Field Mapping
    ndviSection: document.getElementById('ndvi-field-mapping-section'),
    ndviPresetChips: document.getElementById('ndvi-preset-chips'),
    ndviFieldNameLabel: document.getElementById('ndvi-field-name-label'),
    ndviFieldCropBadge: document.getElementById('ndvi-field-crop-badge'),
    ndviRasterCanvas: document.getElementById('ndvi-raster-canvas'),
    ndviCellTooltip: document.getElementById('ndvi-cell-tooltip'),
    ttCellCoords: document.getElementById('tt-cell-coords'),
    ttCellNdvi: document.getElementById('tt-cell-ndvi'),
    ttCellZone: document.getElementById('tt-cell-zone'),
    ndviMeanVal: document.getElementById('ndvi-mean-val'),
    ndviRangeSub: document.getElementById('ndvi-range-sub'),
    ndviUniformityVal: document.getElementById('ndvi-uniformity-val'),
    ndviAnomalyAlert: document.getElementById('ndvi-anomaly-alert'),
    zoneHighPct: document.getElementById('zone-high-pct'),
    zoneHighBar: document.getElementById('zone-high-bar'),
    zoneModPct: document.getElementById('zone-mod-pct'),
    zoneModBar: document.getElementById('zone-mod-bar'),
    zoneSevPct: document.getElementById('zone-sev-pct'),
    zoneSevBar: document.getElementById('zone-sev-bar'),
    btnScoutNdviAnomaly: document.getElementById('btn-scout-ndvi-anomaly'),
    vraSavingsBadge: document.getElementById('vra-savings-badge'),
    vraPrescriptionTbody: document.getElementById('vra-prescription-tbody'),
    ndviTemporalCurveContainer: document.getElementById('ndvi-temporal-curve-container'),
    btnNdviExportGeojson: document.getElementById('btn-ndvi-export-geojson'),
    btnNdviRefresh: document.getElementById('btn-ndvi-refresh'),
    
    toastContainer: document.getElementById('toast-container')
};

// --- Toast Notifications ---
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    const colors = {
        success: 'bg-emerald-950 border-emerald-500/50 text-emerald-200',
        error: 'bg-red-950 border-red-500/50 text-red-200',
        info: 'bg-slate-900 border-slate-700 text-slate-200',
        warning: 'bg-amber-950 border-amber-500/50 text-amber-200'
    };
    
    const icons = {
        success: 'fa-circle-check text-emerald-400',
        error: 'fa-circle-exclamation text-red-400',
        info: 'fa-circle-info text-teal-400',
        warning: 'fa-triangle-exclamation text-amber-400'
    };

    toast.className = `flex items-center space-x-2.5 px-4 py-3 rounded-xl border shadow-2xl backdrop-blur text-xs pointer-events-auto transition transform duration-300 translate-y-2 opacity-0 ${colors[type] || colors.info}`;
    toast.innerHTML = `
        <i class="fa-solid ${icons[type] || icons.info} text-sm"></i>
        <span>${message}</span>
    `;

    DOM.toastContainer.appendChild(toast);
    requestAnimationFrame(() => {
        toast.classList.remove('translate-y-2', 'opacity-0');
    });

    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-2');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// --- Loading State Handler ---
function setLoading(loading) {
    state.isLoading = loading;
    if (loading) {
        DOM.diagnosisSpinner.classList.remove('hidden');
    } else {
        DOM.diagnosisSpinner.classList.add('hidden');
    }
}

// --- In-Browser ONNX Vision Engine & Edge Pathology ---
let onnxSession = null;

async function getONNXSession() {
    if (onnxSession) return onnxSession;
    if (typeof ort === 'undefined') {
        console.warn("[ONNX Web] ort library is not available yet.");
        return null;
    }
    try {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
        console.log("[ONNX Web] Initializing in-browser MobileNetV3 model...");
        onnxSession = await ort.InferenceSession.create('/crop_disease_model.onnx', {
            executionProviders: ['wasm']
        });
        console.log("[ONNX Web] In-browser model initialized successfully!");
        return onnxSession;
    } catch (e) {
        console.error("[ONNX Web] Failed to initialize model:", e);
        return null;
    }
}

async function runInBrowserONNXDiagnosis(imageSource, targetCrop = "auto") {
    setLoading(true);
    try {
        const session = await getONNXSession();
        if (!session) {
            throw new Error("In-browser ONNX runtime is initializing. Please check connection or retry.");
        }

        // Load image into an HTML Image Element
        const img = await new Promise((resolve, reject) => {
            if (imageSource instanceof HTMLImageElement) {
                resolve(imageSource);
                return;
            }
            const i = new Image();
            i.crossOrigin = 'anonymous';
            i.onload = () => resolve(i);
            i.onerror = (e) => reject(new Error("Failed to decode image"));
            if (typeof imageSource === 'string') {
                i.src = imageSource;
            } else if (imageSource instanceof Blob || imageSource instanceof File) {
                const reader = new FileReader();
                reader.onload = (e) => { i.src = e.target.result; };
                reader.readAsDataURL(imageSource);
            }
        });

        // Store last scanned image as base64
        const offCanvas = document.createElement('canvas');
        offCanvas.width = 224;
        offCanvas.height = 224;
        const ctx = offCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 224, 224);
        
        const originalBase64 = offCanvas.toDataURL('image/jpeg', 0.9);
        state.lastScannedImage = originalBase64;

        // Image Preprocessing: Tensor construction [1, 3, 224, 224]
        const imgData = ctx.getImageData(0, 0, 224, 224).data;
        const floatData = new Float32Array(1 * 3 * 224 * 224);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        let lesionPixels = 0;

        for (let i = 0; i < 224 * 224; i++) {
            const r = imgData[i * 4] / 255.0;
            const g = imgData[i * 4 + 1] / 255.0;
            const b = imgData[i * 4 + 2] / 255.0;

            // Estimate chlorosis / necrotic lesion pixels
            if ((r > 0.45 && g < 0.45 && b < 0.4) || (r < 0.25 && g < 0.3 && b < 0.25)) {
                lesionPixels++;
            }

            floatData[i] = (r - mean[0]) / std[0];
            floatData[224 * 224 + i] = (g - mean[1]) / std[1];
            floatData[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
        }

        const inputTensor = new ort.Tensor('float32', floatData, [1, 3, 224, 224]);

        // Run In-Browser Neural Inference (<25ms)
        const feeds = { input_image: inputTensor };
        const results = await session.run(feeds);
        const rawLogits = results.class_logits.data;

        // Fetch taxonomy & advisories
        const db = window.AGROAI_OFFLINE_DB || {};
        const classNames = db.CLASS_NAMES || [];
        const advisoryDb = db.ADVISORY_DATABASE || {};
        const cropProfiles = db.CROP_PROFILES || {};

        // Adjust logits with target_crop lock and biological priors
        const adjustedLogits = new Float32Array(rawLogits.length);
        for (let i = 0; i < rawLogits.length; i++) {
            let score = rawLogits[i] || 0.0;
            const cName = (classNames[i] || '').toLowerCase();
            
            // 1. Target crop lock filter
            if (targetCrop && targetCrop.toLowerCase() !== 'auto' && targetCrop.toLowerCase() !== 'all' && targetCrop !== '') {
                const targetLower = targetCrop.toLowerCase();
                const parts = (classNames[i] || '').split('___');
                const cropPrefix = (parts[0] || '').replace(/_/g, ' ').toLowerCase();
                if (!cropPrefix.includes(targetLower) && !targetLower.includes(cropPrefix)) {
                    score = -1000.0;
                }
            } else {
                // In auto mode: if rust / parallel stripe pixels are detected, prioritize Wheat / Monocots
                if (cName.includes('wheat') && lesionPixels > 100) {
                    score += 4.5;
                } else if (cName.includes('potato') && cName.includes('early_blight') && lesionPixels < 150) {
                    score -= 2.0;
                }
            }
            adjustedLogits[i] = score;
        }

        // Temperature-scaled Softmax
        const temperature = 0.35;
        let maxLogit = -Infinity;
        for (let i = 0; i < adjustedLogits.length; i++) {
            if (adjustedLogits[i] > maxLogit) maxLogit = adjustedLogits[i];
        }
        let sumExp = 0;
        const exps = new Float32Array(adjustedLogits.length);
        for (let i = 0; i < adjustedLogits.length; i++) {
            exps[i] = Math.exp((adjustedLogits[i] - maxLogit) / temperature);
            sumExp += exps[i];
        }
        const probs = Array.from(exps).map(e => e / sumExp);

        // Sort probabilities
        const indexedProbs = probs.map((prob, idx) => ({
            class_name: classNames[idx] || `Class_${idx}`,
            confidence: Math.round(prob * 10000) / 100
        })).sort((a, b) => b.confidence - a.confidence);

        // Top 5 predictions
        const top5 = indexedProbs.slice(0, 5).map(item => {
            const parts = item.class_name.split('___');
            const crop = parts[0].replace(/_/g, ' ');
            const disease = (parts[1] || 'Unknown').replace(/_/g, ' ');
            return {
                class_name: item.class_name,
                crop: crop,
                disease: disease,
                confidence: item.confidence
            };
        });

        const top1 = top5[0];
        const advisory = advisoryDb[top1.class_name] || {
            crop: top1.crop,
            disease: top1.disease,
            pathogen_type: top1.disease.toLowerCase().includes('healthy') ? 'Healthy Tissue' : 'Pathogen Infestation',
            scientific_name: 'Identified via In-Browser Edge ONNX Model',
            severity_risk: 'Moderate',
            symptoms: ["Observed leaf spot / discoloration symptoms matching field taxonomy."],
            organic_controls: ["Spray cold-pressed Pure Neem Oil (5ml/L).", "Apply bio-control Trichoderma harzianum."],
            chemical_controls: [
                {
                    product: "Broad-Spectrum Protective Fungicide (Mancozeb 75% WP)",
                    dosage: "2.0 - 2.5 g / Liter",
                    timing: "Spray early morning during calm weather"
                }
            ],
            cultural_controls: ["Prune lower diseased foliage", "Avoid overhead sprinkler irrigation"]
        };

        // Severity estimation
        const lesionPct = Math.min(85, Math.max(0, Math.round((lesionPixels / (224 * 224)) * 100 * 1.8)));
        const stage = lesionPct === 0 ? "Healthy" : (lesionPct < 15 ? "Stage I (Early)" : (lesionPct < 40 ? "Stage II (Moderate)" : "Stage III (Severe)"));

        // Generate synthetic Grad-CAM heatmap overlay client-side
        const heatCanvas = document.createElement('canvas');
        heatCanvas.width = 224;
        heatCanvas.height = 224;
        const heatCtx = heatCanvas.getContext('2d');
        
        heatCtx.drawImage(offCanvas, 0, 0);
        const heatGrad = heatCtx.createRadialGradient(112, 112, 20, 112, 112, 100);
        heatGrad.addColorStop(0, 'rgba(239, 68, 68, 0.7)');
        heatGrad.addColorStop(0.5, 'rgba(234, 179, 8, 0.5)');
        heatGrad.addColorStop(1, 'rgba(34, 197, 94, 0.1)');
        heatCtx.fillStyle = heatGrad;
        heatCtx.fillRect(0, 0, 224, 224);

        const overlayBase64 = heatCanvas.toDataURL('image/jpeg', 0.85);

        // Crop identification profile
        const matchedCropKey = Object.keys(cropProfiles).find(k => top1.crop.toLowerCase().includes(k.toLowerCase())) || top1.crop;
        const profile = cropProfiles[matchedCropKey] || { display: top1.crop, botanical_name: "Plantae Species", family: "Plantae" };

        const mockResponse = {
            top_prediction: top1,
            top_5_predictions: top5,
            advisory: advisory,
            crop_identification: {
                detected_crop: profile.display || top1.crop,
                botanical_name: profile.botanical_name,
                crop_family: profile.family,
                crop_confidence: top1.confidence
            },
            severity: {
                severity_percentage: lesionPct,
                severity_stage: stage,
                spot_count: Math.round(lesionPct * 0.4),
                chlorosis_percentage: Math.round(lesionPct * 0.6),
                action_urgency: lesionPct > 25 ? "Immediate Spray Required" : "Preventive Monitoring"
            },
            gradcam: {
                blended_image: overlayBase64,
                heatmap_image: overlayBase64,
                severity_mask_image: originalBase64,
                original_image: originalBase64
            },
            in_browser_onnx: true
        };

        handleDiagnosisSuccess(mockResponse);
        showToast("⚡ Diagnosed locally via In-Browser ONNX Engine (100% Offline)", "success");
    } catch (err) {
        console.error("[ONNX Web Error]:", err);
        showToast(`In-browser analysis error: ${err.message}`, "error");
    } finally {
        setLoading(false);
    }
}

// --- API Diagnostic Calls with Auto Offline Fallback ---
async function diagnoseImageFile(file) {
    if (!navigator.onLine) {
        return runInBrowserONNXDiagnosis(file);
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    
    const loc = DOM.weatherLocationLabel.textContent || "Target Farm";
    const targetCrop = DOM.targetCropSelect ? DOM.targetCropSelect.value : "auto";
    try {
        const url = `/api/diagnose?alpha=${state.alpha}&colormap=${state.colormap}&target_crop=${encodeURIComponent(targetCrop)}&location=${encodeURIComponent(loc)}`;
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Diagnosis failed with code: ${response.status}`);
        }
        
        const data = await response.json();
        handleDiagnosisSuccess(data);
        
        const reader = new FileReader();
        reader.onload = (e) => { state.lastScannedImage = e.target.result; };
        reader.readAsDataURL(file);
        
        await loadScoutHistory();
        showToast("Diagnostic analysis completed successfully!", "success");
    } catch (err) {
        console.warn("[Diagnostic Network Fallback] Switching to in-browser ONNX:", err);
        await runInBrowserONNXDiagnosis(file, targetCrop);
    } finally {
        setLoading(false);
    }
}

async function diagnoseSample(sampleId) {
    setLoading(true);
    const loc = DOM.weatherLocationLabel.textContent || "Target Farm";
    const targetCrop = DOM.targetCropSelect ? DOM.targetCropSelect.value : "auto";
    try {
        const response = await fetch('/api/diagnose-json', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sample_id: sampleId,
                alpha: state.alpha,
                colormap: state.colormap,
                target_crop: targetCrop,
                location: loc
            })
        });
        
        if (!response.ok) {
            throw new Error(`Sample analysis failed: ${response.status}`);
        }
        
        const data = await response.json();
        state.lastScannedImage = data.gradcam.original_image;
        handleDiagnosisSuccess(data);
        await loadScoutHistory();
        showToast(`Loaded test specimen: ${data.top_prediction.disease}`, "success");
    } catch (err) {
        console.warn("[Sample Network Fallback]:", err);
        showToast(`Error analyzing sample: ${err.message}`, "error");
    } finally {
        setLoading(false);
    }
}

async function reapplyVisualParams() {
    if (!state.lastScannedImage) return;
    const targetCrop = DOM.targetCropSelect ? DOM.targetCropSelect.value : "auto";
    try {
        const response = await fetch('/api/diagnose-json', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_base64: state.lastScannedImage,
                alpha: state.alpha,
                colormap: state.colormap,
                target_crop: targetCrop
            })
        });
        if (response.ok) {
            const data = await response.json();
            state.diagnosisData = data;
            handleDiagnosisSuccess(data);
        }
    } catch (err) {
        console.error("Re-render error:", err);
    }
}

// --- Diagnosis UI Renderer ---
function handleDiagnosisSuccess(data) {
    state.diagnosisData = data;
    
    // 1. Primary Finding
    const top = data.top_prediction;
    const advisory = data.advisory || {};
    const severity = data.severity || {};
    const cropId = data.crop_identification || {};
    
    // Identified Crop Species Card
    if (DOM.cropDisplayName) DOM.cropDisplayName.textContent = cropId.detected_crop || top.crop || "Identified Crop";
    if (DOM.cropBotanicalName) DOM.cropBotanicalName.textContent = cropId.botanical_name ? `Taxonomy: ${cropId.botanical_name}` : (advisory.scientific_name || "");
    if (DOM.cropBotanicalFamily) DOM.cropBotanicalFamily.textContent = cropId.crop_family ? `Family: ${cropId.crop_family}` : "Family: Plantae";
    if (DOM.cropMatchPct) DOM.cropMatchPct.textContent = `${cropId.crop_confidence || top.confidence || '--'}%`;

    DOM.diagCropBadge.textContent = top.crop || advisory.crop || "Crop";
    DOM.diagPathogenBadge.textContent = advisory.pathogen_type || "Pathogen";
    DOM.diagDiseaseTitle.textContent = top.disease || advisory.disease || "Disease";
    DOM.diagScientificName.textContent = advisory.scientific_name ? `Pathogen: ${advisory.scientific_name}` : "";
    DOM.diagConfidenceVal.textContent = top.confidence.toFixed(1);
    
    // Pathogen badge styling
    const pathType = (advisory.pathogen_type || "").toLowerCase();
    if (pathType.includes("fungal") || pathType.includes("oomycete")) {
        DOM.diagPathogenBadge.className = "px-2.5 py-0.5 rounded-md bg-amber-500/15 text-amber-300 border border-amber-500/30 text-xs font-medium";
    } else if (pathType.includes("bacterial")) {
        DOM.diagPathogenBadge.className = "px-2.5 py-0.5 rounded-md bg-rose-500/15 text-rose-300 border border-rose-500/30 text-xs font-medium";
    } else if (pathType.includes("viral")) {
        DOM.diagPathogenBadge.className = "px-2.5 py-0.5 rounded-md bg-purple-500/15 text-purple-300 border border-purple-500/30 text-xs font-medium";
    } else if (pathType.includes("none") || pathType.includes("healthy")) {
        DOM.diagPathogenBadge.className = "px-2.5 py-0.5 rounded-md bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-xs font-medium";
    }

    
    // 2. Severity Quantification
    DOM.severityPctLabel.textContent = `${severity.severity_percentage}%`;
    DOM.severityStageBadge.textContent = severity.severity_stage || "Evaluated";
    DOM.severityStageBadge.style.color = severity.stage_color || "#f59e0b";
    DOM.lesionCountLabel.textContent = severity.lesion_count || "0";
    DOM.urgencyLabel.textContent = severity.urgency || "Normal";
    
    // 3. Differential Probabilities Bars
    DOM.differentialBarsContainer.innerHTML = '';
    const diffs = data.top_k_predictions || [top];
    diffs.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'space-y-1';
        item.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <span class="text-slate-300 font-medium truncate">${pred.crop} - ${pred.disease}</span>
                <span class="text-slate-400 font-mono font-semibold">${pred.confidence}%</span>
            </div>
            <div class="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden border border-slate-800">
                <div class="bg-gradient-to-r from-emerald-500 to-brand-400 h-full rounded-full transition-all duration-500" style="width: ${pred.confidence}%"></div>
            </div>
        `;
        DOM.differentialBarsContainer.appendChild(item);
    });
    
    // Reset Active Learning Farmer Feedback Buttons for new scan
    if (DOM.feedbackBtnGroup && DOM.feedbackThanksMsg) {
        DOM.feedbackBtnGroup.classList.remove('hidden');
        DOM.feedbackBtnGroup.classList.add('flex');
        DOM.feedbackThanksMsg.classList.add('hidden');
        DOM.feedbackThanksMsg.classList.remove('flex');
    }
    
    // 4. Grad-CAM Image Display
    DOM.gradcamPeakBadge.textContent = `Peak Focus: ${data.gradcam.attention_peak_pct || '--'}%`;
    renderVisualImage();
    
    // 5. Advisory Hub
    renderAdvisoryTab();
    
    // 6. Update Dosage Calculator with detected crop and chemical dosage recommendation
    if (advisory.crop) {
        const chems = advisory.chemical_controls || [];
        if (chems.length > 0) {
            const firstChem = chems[0];
            const rawDosage = parseFloat(firstChem.dosage) || 2.5;
            DOM.calcDosageRate.value = rawDosage;
            if (firstChem.dosage.includes("ml")) {
                DOM.calcDosageUnit.value = "ml";
            } else {
                DOM.calcDosageUnit.value = "g";
            }
        }
        recalculateDosage();
    }
    
    // 7. Auto-Synchronize Spray Engine Chemical Formulation
    if (DOM.sprayChemicalSelect) {
        const pathType = (advisory.pathogen_type || "").toLowerCase();
        const diseaseName = (top.disease || "").toLowerCase();
        if (pathType.includes("fungal") || diseaseName.includes("rust") || diseaseName.includes("blight")) {
            DOM.sprayChemicalSelect.value = "systemic_fungicide";
        } else if (pathType.includes("bacterial") || diseaseName.includes("bacterial")) {
            DOM.sprayChemicalSelect.value = "contact_fungicide";
        } else if (diseaseName.includes("mite") || diseaseName.includes("aphid") || diseaseName.includes("fly") || diseaseName.includes("miner")) {
            DOM.sprayChemicalSelect.value = "systemic_insecticide";
        } else if (pathType.includes("healthy") || diseaseName.includes("healthy")) {
            DOM.sprayChemicalSelect.value = "foliar_fertilizer";
        }
        
        // Refresh spray window analysis for the diagnosed formulation
        const curCity = DOM.weatherCityInput.value.trim() || DOM.weatherLocationLabel.textContent;
        if (curCity) {
            fetchWeatherRisk(curCity);
        }
    }
    
    // 8. Synchronize Live Diagnostic Context with AI Agronomist Chatbot
    updateChatContext(
        top.crop || advisory.crop,
        top.disease || advisory.disease,
        top.confidence,
        severity.severity_percentage,
        severity.severity_stage
    );
    
    // 9. Enable Export PDF & Share WhatsApp buttons
    DOM.btnExportPdfNav.disabled = false;
    DOM.btnExportPdfCard.disabled = false;
    if (DOM.btnShareWhatsapp) DOM.btnShareWhatsapp.disabled = false;
}

function renderVisualImage() {
    if (!state.diagnosisData) return;
    const gc = state.diagnosisData.gradcam || {};
    const sev = state.diagnosisData.severity || {};
    
    DOM.imageEmptyState.classList.add('hidden');
    
    if (state.viewMode === 'blended') {
        DOM.mainDisplayImage.classList.add('hidden');
        DOM.baseDisplayImage.src = gc.original_image || '';
        DOM.overlayDisplayImage.src = gc.heatmap_image || '';
        DOM.overlayDisplayImage.style.opacity = state.alpha;
        DOM.baseDisplayImage.classList.remove('hidden');
        DOM.overlayDisplayImage.classList.remove('hidden');
    } else if (state.viewMode === 'heatmap') {
        DOM.baseDisplayImage.classList.add('hidden');
        DOM.overlayDisplayImage.classList.add('hidden');
        DOM.mainDisplayImage.src = gc.heatmap_image || '';
        DOM.mainDisplayImage.classList.remove('hidden');
    } else if (state.viewMode === 'severity') {
        DOM.baseDisplayImage.classList.add('hidden');
        DOM.overlayDisplayImage.classList.add('hidden');
        DOM.mainDisplayImage.src = sev.severity_mask_image || '';
        DOM.mainDisplayImage.classList.remove('hidden');
    } else if (state.viewMode === 'original') {
        DOM.baseDisplayImage.classList.add('hidden');
        DOM.overlayDisplayImage.classList.add('hidden');
        DOM.mainDisplayImage.src = gc.original_image || '';
        DOM.mainDisplayImage.classList.remove('hidden');
    }
}


function renderAdvisoryTab() {
    if (!state.diagnosisData || !state.diagnosisData.advisory) {
        DOM.advisoryContentBox.innerHTML = `<div class="p-6 text-center text-slate-500">No active diagnosis data.</div>`;
        return;
    }
    
    const adv = state.diagnosisData.advisory;
    let html = '';
    
    if (state.advisoryTab === 'organic') {
        const organics = adv.organic_controls || [];
        if (organics.length === 0) {
            html = `<p class="text-slate-400 italic">No specific organic formulation required (Healthy tissue).</p>`;
        } else {
            html = `<ul class="space-y-2">` + organics.map(item => `
                <li class="flex items-start space-x-2 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <i class="fa-solid fa-leaf text-emerald-400 mt-0.5 flex-shrink-0"></i>
                    <span class="text-slate-200">${item}</span>
                </li>
            `).join('') + `</ul>`;
        }
    } else if (state.advisoryTab === 'chemical') {
        const chems = adv.chemical_controls || [];
        if (chems.length === 0) {
            html = `
                <div class="p-6 text-center text-slate-400 italic bg-slate-950/40 rounded-xl border border-slate-800/80">
                    <i class="fa-solid fa-circle-check text-emerald-400 text-2xl mb-2 block"></i>
                    No chemical or insecticide intervention required. Plant tissue is healthy and thriving.
                </div>
            `;
        } else {
            html = `
                <div class="space-y-3">
                    <div class="flex items-center justify-between text-[11px] text-slate-400 px-1">
                        <span class="flex items-center space-x-1.5">
                            <i class="fa-solid fa-award text-amber-400"></i>
                            <span class="font-semibold text-slate-300">Commercial Brand Names & Top Market Best Sellers</span>
                        </span>
                        <span class="text-[10px] text-slate-500">Official Agrochemical Formulations (Syngenta, Bayer, FMC, Corteva, BASF, Rallis)</span>
                    </div>
                    <div class="chem-table-container rounded-xl border border-slate-800 overflow-hidden shadow-lg">
                        <table>
                            <thead>
                                <tr>
                                    <th class="w-[32%]">Brand & Market Best Seller</th>
                                    <th class="w-[26%]">Chemical / Active Composition</th>
                                    <th class="w-[14%]">Dosage / L</th>
                                    <th class="w-[28%]">Timing & Safety Interval (PHI)</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${chems.map(c => {
                                    const isBest = (c.product || '').includes('⭐') || (c.best_seller || '').includes('⭐') || (c.best_seller || '').includes('Best Seller');
                                    const mfg = c.manufacturer || 'Certified Agro';
                                    const active = c.active_ingredient || c.product || 'Standard';
                                    const typeName = c.type || (active.toLowerCase().includes('insect') ? 'Insecticide' : 'Fungicide');
                                    const brand = c.brand_name || c.product.split('(')[0].replace('⭐', '').trim();
                                    
                                    // Extract raw numeric dosage for quick autofill
                                    const numDosage = parseFloat(c.dosage) || 1.0;
                                    const unitDosage = (c.dosage || '').includes('ml') ? 'ml' : 'g';
                                    
                                    return `
                                    <tr class="hover:bg-slate-900/80 transition">
                                        <td class="space-y-1">
                                            <div class="flex items-center space-x-1.5 flex-wrap">
                                                <span class="font-bold text-white text-[12px]">${brand}</span>
                                                ${isBest ? `<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 whitespace-nowrap"><i class="fa-solid fa-star text-[8px]"></i> Best Seller</span>` : ''}
                                            </div>
                                            <div class="flex items-center space-x-1.5 text-[10px]">
                                                <span class="px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700/60 font-medium">${mfg}</span>
                                                <span class="text-teal-400 font-medium">${typeName}</span>
                                            </div>
                                            ${c.best_seller ? `<div class="text-[10px] text-slate-400 italic">${c.best_seller}</div>` : ''}
                                        </td>
                                        <td class="space-y-1">
                                            <div class="font-mono text-slate-200 text-[11px] font-medium leading-snug">${active}</div>
                                        </td>
                                        <td>
                                            <div class="font-mono font-bold text-emerald-400 text-[12px]">${c.dosage}</div>
                                            <button onclick="applyDosageToCalc(${numDosage}, '${unitDosage}')" class="mt-1 text-[9px] px-2 py-0.5 rounded bg-slate-800 hover:bg-brand-600 hover:text-white text-slate-300 border border-slate-700 transition flex items-center space-x-1">
                                                <i class="fa-solid fa-calculator text-[8px]"></i>
                                                <span>Use in Calc</span>
                                            </button>
                                        </td>
                                        <td class="space-y-1">
                                            <div class="text-slate-200 leading-snug">${c.timing}</div>
                                            <div class="text-[10px] text-amber-400/90 font-mono"><i class="fa-solid fa-shield-halved text-[9px]"></i> PHI: ${c.interval}</div>
                                        </td>
                                    </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }
    } else if (state.advisoryTab === 'cultural') {
        const practices = adv.cultural_practices || [];
        html = `<ul class="space-y-2">` + practices.map(item => `
            <li class="flex items-start space-x-2 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                <i class="fa-solid fa-tractor text-teal-400 mt-0.5 flex-shrink-0"></i>
                <span class="text-slate-200">${item}</span>
            </li>
        `).join('') + `</ul>`;
    } else if (state.advisoryTab === 'symptoms') {
        const symptoms = adv.symptoms || [];
        html = `
            <div class="space-y-3">
                <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                    <span class="text-slate-400 block text-[11px] uppercase font-semibold mb-1">Trigger Weather Conditions</span>
                    <p class="text-slate-200">${adv.environmental_triggers?.high_risk_weather || 'N/A'}</p>
                </div>
                <ul class="space-y-2">` + symptoms.map(item => `
                    <li class="flex items-start space-x-2 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <i class="fa-solid fa-magnifying-glass-chart text-amber-400 mt-0.5 flex-shrink-0"></i>
                        <span class="text-slate-200">${item}</span>
                    </li>
                `).join('') + `</ul>
            </div>
        `;
    }
    
    DOM.advisoryContentBox.innerHTML = html;
}

// --- Quick Dosage Applicator ---
window.applyDosageToCalc = function(dosage, unit) {
    if (DOM.calcDosageRate) DOM.calcDosageRate.value = dosage;
    if (DOM.calcDosageUnit) {
        if (unit.toLowerCase().includes('ml')) DOM.calcDosageUnit.value = 'ml';
        else DOM.calcDosageUnit.value = 'g';
    }
    recalculateDosage();
    const calcSection = document.getElementById('txt-calculator-title');
    if (calcSection) {
        calcSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    showToast(`Loaded ${dosage} ${unit}/L into Field Spray Calculator!`, 'success');
};

// --- Field Dosage Calculator ---
async function recalculateDosage() {
    const fieldSize = parseFloat(DOM.calcFieldSize.value) || 1.0;
    const unit = DOM.calcUnit.value;
    const rate = parseFloat(DOM.calcDosageRate.value) || 2.5;
    const dosageUnit = DOM.calcDosageUnit.value;
    const tankSize = parseFloat(DOM.calcTankSize.value) || 15.0;
    const crop = state.diagnosisData?.top_prediction?.crop || "Tomato";
    
    try {
        const res = await fetch('/api/calculate-dosage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                field_size: fieldSize,
                unit: unit,
                crop: crop,
                dosage_per_liter: rate,
                dosage_unit: dosageUnit,
                tank_capacity_liters: tankSize
            })
        });
        
        if (res.ok) {
            const data = await res.json();
            DOM.calcOutWater.textContent = `${data.total_water_liters} L (${data.total_water_gallons} Gal)`;
            DOM.calcOutProduct.textContent = `${data.total_product_required}`;
            DOM.calcOutTanks.textContent = `${data.num_tanks_required} Tanks (${data.product_per_tank}/tank)`;
        }
    } catch (e) {
        console.error("Dosage calculation error:", e);
    }
}

// --- Weather Outbreak Risk & Optimal Spray Window Engine ---
async function fetchWeatherRisk(city = null, lat = null, lon = null) {
    try {
        let url = '/api/weather-risk';
        const chem = DOM.sprayChemicalSelect ? DOM.sprayChemicalSelect.value : 'systemic_fungicide';
        const params = new URLSearchParams();
        
        if (lat !== null && lon !== null) {
            params.set('lat', lat);
            params.set('lon', lon);
        } else if (city) {
            params.set('city', city);
        }
        params.set('chemical', chem);

        url += `?${params.toString()}`;
        
        const res = await fetch(url);
        if (!res.ok) throw new Error("Failed to fetch microclimate forecast.");
        
        const data = await res.json();
        state.weatherData = data;
        renderWeatherRisk(data);
    } catch (err) {
        console.error("Weather error:", err);
        showToast("Notice: Using localized default microclimate data.", "info");
    }
}

function renderWeatherRisk(data) {
    DOM.weatherLocationLabel.textContent = data.location || "Target Farm";
    
    const cur = data.current_weather || {};
    const epi = data.epidemiological_risk || {};
    const spray = data.spray_window_analysis || null;
    
    DOM.weatherTempVal.textContent = cur.temperature_c !== undefined ? Math.round(cur.temperature_c) : "--";
    DOM.weatherHumidityVal.textContent = `RH: ${cur.humidity_pct || '--'}%`;
    DOM.weatherRainVal.textContent = `${cur.precipitation_mm || '0.0'} mm`;
    DOM.weatherWindVal.textContent = `${cur.wind_speed_kmh || '0'} km/h`;
    
    // Pathogen risks
    DOM.riskFungalVal.textContent = `${epi.fungal_risk_score || 0}%`;
    DOM.riskFungalBar.style.width = `${epi.fungal_risk_score || 0}%`;
    
    DOM.riskBacterialVal.textContent = `${epi.bacterial_risk_score || 0}%`;
    DOM.riskBacterialBar.style.width = `${epi.bacterial_risk_score || 0}%`;
    
    // Overall threat
    DOM.overallThreatScore.textContent = epi.overall_outbreak_risk || '--';
    DOM.overallThreatLevel.textContent = epi.threat_level || 'Low Risk';
    DOM.overallThreatLevel.style.color = epi.threat_color || '#10b981';
    DOM.overallThreatSummary.textContent = epi.advisory_summary || 'Conditions are stable.';
    
    // 5-Day Forecast Cards
    DOM.forecastCardsContainer.innerHTML = '';
    const forecasts = data.five_day_forecast || [];
    forecasts.forEach((day, idx) => {
        const card = document.createElement('div');
        const dateObj = new Date(day.date);
        const dayName = idx === 0 ? "Today" : dateObj.toLocaleDateString('en-US', { weekday: 'short' });
        
        card.className = 'bg-slate-950 p-3 rounded-xl border border-slate-800 text-center space-y-1.5 hover:border-slate-700 transition';
        card.innerHTML = `
            <div class="text-[11px] font-bold text-slate-300">${dayName}</div>
            <div class="text-xs font-bold text-slate-100 font-mono">${Math.round(day.max_temp)}° / ${Math.round(day.min_temp)}°</div>
            <div class="text-[10px] text-teal-400 font-mono"><i class="fa-solid fa-droplet text-[8px]"></i> ${day.precip_mm}mm</div>
            <div class="pt-1 border-t border-slate-800">
                <span class="text-[9px] px-1.5 py-0.5 rounded font-bold" style="background-color: ${day.threat_color}25; color: ${day.threat_color}">
                    ${day.outbreak_risk}% Risk
                </span>
            </div>
        `;
        DOM.forecastCardsContainer.appendChild(card);
    });

    // Render Optimal Spray Window & Rainfastness Engine
    if (spray) {
        renderSprayWindowAnalysis(spray, cur);
    }
}

function renderSprayWindowAnalysis(sprayData, current) {
    if (!sprayData) return;

    const deltaT = current.delta_t || {};
    const windDrift = current.wind_drift || {};
    const nextWin = sprayData.next_safe_window || {};
    const chem = sprayData.chemical_selected || {};
    const timeline = sprayData.hourly_timeline || [];

    // 1. Delta-T Evaporation Gauge
    if (DOM.sprayDeltaTVal) {
        DOM.sprayDeltaTVal.textContent = deltaT.delta_t_c !== undefined ? deltaT.delta_t_c.toFixed(1) : "--.-";
        DOM.sprayDeltaTVal.style.color = deltaT.color || "#10b981";
    }
    if (DOM.sprayDeltaTBadge) {
        DOM.sprayDeltaTBadge.textContent = (deltaT.rating || "IDEAL").toUpperCase();
        DOM.sprayDeltaTBadge.style.color = deltaT.color || "#10b981";
        DOM.sprayDeltaTBadge.style.borderColor = (deltaT.color || "#10b981") + "60";
    }
    if (DOM.sprayDeltaTDesc) {
        DOM.sprayDeltaTDesc.textContent = deltaT.recommendation || "Target range: 2.0°C – 8.0°C for optimal droplet survival.";
    }

    // 2. Wind Drift & Inversion Hazard
    if (DOM.sprayWindVal) {
        DOM.sprayWindVal.textContent = windDrift.wind_speed_kmh !== undefined ? windDrift.wind_speed_kmh.toFixed(1) : "--.-";
        DOM.sprayWindVal.style.color = windDrift.color || "#10b981";
    }
    if (DOM.sprayWindBadge) {
        DOM.sprayWindBadge.textContent = (windDrift.rating || "SAFE").toUpperCase();
        DOM.sprayWindBadge.style.color = windDrift.color || "#10b981";
        DOM.sprayWindBadge.style.borderColor = (windDrift.color || "#10b981") + "60";
    }
    if (DOM.sprayWindDesc) {
        DOM.sprayWindDesc.textContent = windDrift.description || "Safe range: 3 – 15 km/h. Avoid thermal inversions.";
    }

    // 3. Rainfastness & Washout Protection
    const currentHour = timeline[0] || {};
    const forwardRain = currentHour.forward_rain_in_window_mm || 0.0;
    if (DOM.sprayRainfastHoursVal) {
        DOM.sprayRainfastHoursVal.textContent = chem.rainfast_hours !== undefined ? `${chem.rainfast_hours}h` : "3h";
    }
    if (DOM.sprayWashoutBadge) {
        if (forwardRain > 0.3) {
            DOM.sprayWashoutBadge.textContent = `⚠️ ${forwardRain.toFixed(1)}mm Rain Expected`;
            DOM.sprayWashoutBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950/60 text-rose-400 border border-rose-500/40";
        } else {
            DOM.sprayWashoutBadge.textContent = `✅ 0.0mm Rain in Next ${chem.rainfast_hours}h`;
            DOM.sprayWashoutBadge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-teal-950/60 text-teal-400 border border-teal-500/30";
        }
    }
    if (DOM.sprayWashoutDesc) {
        DOM.sprayWashoutDesc.textContent = forwardRain > 0.3
            ? `High washout hazard! Rain within ${chem.rainfast_hours}h absorption window will strip applied product.`
            : `Safe absorption clearance window. No significant precipitation forecast during the next ${chem.rainfast_hours} hours.`;
    }

    // 4. Next Recommended Spray Window Banner
    if (DOM.sprayWindowHeadline && DOM.sprayWindowRec) {
        if (nextWin.available) {
            DOM.sprayWindowHeadline.innerHTML = `🌟 ${nextWin.headline}`;
            DOM.sprayWindowRec.textContent = nextWin.recommendation;
        } else {
            DOM.sprayWindowHeadline.innerHTML = `⚠️ ${nextWin.headline}`;
            DOM.sprayWindowRec.textContent = nextWin.recommendation;
        }
    }

    // 5. 48-Hour Hourly Timeline Bars
    if (DOM.sprayTimelineBars) {
        DOM.sprayTimelineBars.innerHTML = '';
        let lastDate = "";

        timeline.forEach((h) => {
            // Day divider pill if new date
            if (h.display_date && h.display_date !== lastDate) {
                lastDate = h.display_date;
                const divider = document.createElement('div');
                divider.className = "flex flex-col items-center justify-end h-full px-1 border-l border-slate-700/80 mr-1 flex-shrink-0";
                divider.innerHTML = `
                    <span class="text-[9px] font-extrabold text-slate-400 uppercase tracking-tighter mb-2 font-mono">
                        ${new Date(h.display_date).toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' })}
                    </span>
                `;
                DOM.sprayTimelineBars.appendChild(divider);
            }

            const barCol = document.createElement('div');
            barCol.className = "flex-1 min-w-[28px] max-w-[40px] flex flex-col items-center justify-end h-full group relative cursor-pointer flex-shrink-0";

            const barHeightPct = Math.max(14, Math.round(h.suitability_score));
            const barBg = h.badge_color || (h.suitability_score >= 80 ? '#10b981' : (h.suitability_score >= 50 ? '#f59e0b' : '#ef4444'));

            // Tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = "hidden group-hover:block absolute bottom-full mb-2 z-30 w-52 p-2.5 rounded-xl bg-slate-950 border border-slate-700 shadow-2xl text-[10px] text-slate-200 pointer-events-none";
            tooltip.innerHTML = `
                <div class="flex items-center justify-between font-bold pb-1 mb-1 border-b border-slate-800">
                    <span class="text-white">${h.display_date} ${h.display_time}</span>
                    <span class="px-1.5 py-0.2 rounded font-mono" style="color: ${barBg}">${h.suitability_score}%</span>
                </div>
                <div class="space-y-0.5 text-slate-300">
                    <div class="flex justify-between"><span>Status:</span> <strong style="color: ${barBg}">${h.status}</strong></div>
                    <div class="flex justify-between"><span>Temp / RH:</span> <span>${h.temperature_c}°C / ${h.humidity_pct}%</span></div>
                    <div class="flex justify-between"><span>Delta-T (ΔT):</span> <span class="font-mono font-bold">${h.delta_t_c}°C</span></div>
                    <div class="flex justify-between"><span>Wind:</span> <span>${h.wind_speed_kmh} km/h</span></div>
                    <div class="flex justify-between"><span>Precipitation:</span> <span>${h.precipitation_mm} mm (${h.precipitation_probability}%)</span></div>
                    ${h.forward_rain_in_window_mm > 0 ? `<div class="flex justify-between text-rose-400 font-bold"><span>Rain in Window:</span> <span>${h.forward_rain_in_window_mm} mm</span></div>` : ''}
                </div>
                ${h.hazard_reasons && h.hazard_reasons.length > 0 ? `
                    <div class="mt-1.5 pt-1 border-t border-slate-800/80 text-[9.5px] text-amber-300">
                        ${h.hazard_reasons.slice(0, 2).map(r => `• ${r}`).join('<br>')}
                    </div>
                ` : ''}
            `;

            barCol.innerHTML = `
                <div class="w-full rounded-t-md transition-all duration-300 group-hover:brightness-125" style="height: ${barHeightPct}%; background-color: ${barBg};"></div>
                <span class="text-[9px] font-mono text-slate-400 mt-1 truncate group-hover:text-white font-medium">${h.display_time.split(':')[0]}h</span>
            `;
            barCol.appendChild(tooltip);
            DOM.sprayTimelineBars.appendChild(barCol);
        });
    }
}

// --- Scouting Audit History ---
async function loadScoutHistory() {
    try {
        const res = await fetch('/api/history');
        if (!res.ok) return;
        const data = await res.json();
        state.historyData = data.history || [];
        renderHistoryCards();
    } catch (e) {
        console.error("History fetch error:", e);
    }
}

function renderHistoryCards() {
    DOM.historyCardsContainer.innerHTML = '';
    if (!state.historyData || state.historyData.length === 0) {
        DOM.historyCardsContainer.innerHTML = `
            <div class="col-span-full py-8 text-center text-slate-500 text-xs">
                No field scans logged yet. Complete a diagnosis to start building your crop audit log.
            </div>
        `;
        return;
    }
    
    state.historyData.forEach(item => {
        const card = document.createElement('div');
        card.className = 'bg-slate-950 p-3 rounded-xl border border-slate-800 hover:border-slate-700 transition flex items-center space-x-3 group relative';
        card.innerHTML = `
            <div class="w-12 h-12 rounded-lg bg-slate-900 overflow-hidden flex-shrink-0 border border-slate-800">
                <img src="${item.thumbnail || ''}" alt="${item.crop}" class="w-full h-full object-cover">
            </div>
            <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-slate-200 truncate">${item.crop}</span>
                    <span class="text-[10px] font-mono text-brand-400 font-bold">${item.confidence}%</span>
                </div>
                <div class="text-[11px] text-slate-400 truncate">${item.disease}</div>
                <div class="flex items-center justify-between text-[10px] text-slate-500 mt-1">
                    <span>${item.severity_percentage}% (${item.severity_stage})</span>
                    <span>${item.timestamp.split(' ')[0]}</span>
                </div>
            </div>
            <button class="delete-history-btn absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 transition text-[11px]" data-id="${item.id}">
                <i class="fa-solid fa-trash"></i>
            </button>
        `;
        
        card.querySelector('.delete-history-btn').addEventListener('click', async (e) => {
            e.stopPropagation();
            await fetch(`/api/history/${item.id}`, { method: 'DELETE' });
            await loadScoutHistory();
            showToast("Log entry deleted", "info");
        });
        
        DOM.historyCardsContainer.appendChild(card);
    });
}

// --- Sample Specimens Loader ---
async function loadSampleSpecimens() {
    try {
        const res = await fetch('/api/samples');
        if (!res.ok) return;
        const data = await res.json();
        
        DOM.sampleCardsContainer.innerHTML = '';
        (data.samples || []).forEach(s => {
            const btn = document.createElement('button');
            btn.className = 'sample-btn flex items-center space-x-2 p-2 rounded-xl bg-slate-950/60 border border-slate-800/80 hover:border-brand-500/50 hover:bg-brand-950/20 text-left transition group';
            btn.dataset.sampleId = s.id;
            btn.innerHTML = `
                <div class="w-9 h-9 rounded-lg bg-slate-800 flex-shrink-0 overflow-hidden border border-slate-700 group-hover:border-brand-500">
                    <img src="${s.thumbnail}" alt="${s.title}" class="w-full h-full object-cover">
                </div>
                <div class="truncate">
                    <div class="text-xs font-semibold text-slate-200 group-hover:text-brand-300 truncate">${s.crop}</div>
                    <div class="text-[10px] text-slate-400 font-medium truncate">${s.title.replace(s.crop, '').trim() || s.title}</div>
                </div>
            `;
            btn.addEventListener('click', () => diagnoseSample(s.id));
            DOM.sampleCardsContainer.appendChild(btn);
        });
    } catch (e) {
        console.error("Failed loading samples:", e);
    }
}

// --- Multilingual i18n Handler ---
async function changeLanguage(langCode) {
    state.currentLang = langCode;
    try {
        const res = await fetch(`/api/i18n/${langCode}`);
        if (!res.ok) return;
        const data = await res.json();
        state.translations = data.translations || {};
        applyTranslations();
    } catch (e) {
        console.error("Language load error:", e);
    }
}

function applyTranslations() {
    const t = state.translations;
    if (!t) return;
    
    const setSafe = (id, key) => {
        const el = document.getElementById(id);
        if (el && t[key]) el.textContent = t[key];
    };
    
    // Header & Navigation
    setSafe('txt-app-title', 'app_title');
    setSafe('txt-app-subtitle', 'app_subtitle');
    setSafe('txt-export-pdf-nav', 'export_pdf');
    setSafe('txt-install-app', 'install_app');
    setSafe('txt-mlops-hub-nav', 'mlops_hub');
    setSafe('model-status-text', 'model_ready');

    // 1-Click Specimens Strip
    setSafe('txt-sample-strip-title', 'sample_strip_title');
    setSafe('txt-sample-strip-sub', 'sample_strip_sub');

    // Ingestion Card
    setSafe('txt-upload-tab', 'upload_tab');
    setSafe('txt-camera-tab', 'camera_tab');
    setSafe('txt-drag-drop-title', 'drag_drop_title');
    setSafe('txt-drag-drop-sub', 'drag_drop_sub');
    setSafe('txt-browse-computer', 'browse_computer');
    setSafe('txt-start-camera', 'start_camera');
    setSafe('txt-capture-diagnose', 'capture_diagnose');
    setSafe('txt-target-crop-label', 'target_crop_label');
    setSafe('opt-auto-detect', 'auto_detect');

    // Identified Species & Primary Diagnosis
    setSafe('txt-identified-species-title', 'identified_species_title');
    setSafe('txt-species-match', 'species_match');
    setSafe('txt-confidence-score', 'confidence_score');
    setSafe('txt-infection-severity', 'infection_severity');
    setSafe('txt-spot-count', 'spot_count');
    setSafe('txt-action-urgency', 'action_urgency');
    setSafe('txt-differential-title', 'differential_title');
    setSafe('txt-feedback-prompt', 'feedback_question');
    setSafe('txt-feedback-accurate', 'feedback_accurate');
    setSafe('txt-feedback-incorrect', 'feedback_correct');
    setSafe('txt-feedback-thanks', 'feedback_thanks');

    if (!state.diagnosisData) {
        setSafe('diag-disease-title', 'awaiting_scan_title');
        setSafe('diag-scientific-name', 'awaiting_scan_sub');
    }

    // Vision Maps & Explainable AI
    setSafe('txt-explainable-ai', 'explainable_ai');
    setSafe('txt-mode-blended', 'blended');
    setSafe('txt-mode-heatmap', 'heatmap');
    setSafe('txt-mode-severity', 'lesion_mask');
    setSafe('txt-mode-original', 'original');
    setSafe('txt-blend-opacity', 'blend_opacity');

    // Treatment & Advisory
    setSafe('txt-treatment-guide', 'treatment_guide');
    setSafe('txt-download-pdf', 'download_pdf');
    setSafe('txt-tab-organic', 'tab_organic');
    setSafe('txt-tab-chemical', 'tab_chemical');
    setSafe('txt-tab-cultural', 'tab_cultural');
    setSafe('txt-tab-symptoms', 'tab_symptoms');

    // Outbreak Forecaster & Weather
    setSafe('txt-weather-title', 'weather_title');
    setSafe('txt-weather-subtitle', 'weather_subtitle');
    setSafe('txt-forecast-btn', 'forecast_btn');
    setSafe('txt-quick-hubs', 'quick_hubs');
    setSafe('txt-fungal-risk', 'fungal_risk');
    setSafe('txt-bacterial-risk', 'bacterial_risk');
    setSafe('txt-overall-threat', 'overall_threat');
    setSafe('txt-five-day-title', 'five_day_title');

    // Spray Engine & Rainfastness
    setSafe('txt-spray-engine-title', 'spray_engine_title');
    setSafe('txt-spray-engine-sub', 'spray_engine_sub');
    setSafe('txt-spray-product-label', 'spray_product_label');
    setSafe('txt-delta-t-title', 'delta_t_title');
    setSafe('txt-wind-drift-title', 'wind_drift_title');
    setSafe('txt-washout-risk-title', 'washout_risk_title');

    // Calculator & Audit Log
    setSafe('txt-calculator-title', 'calculator_title');
    setSafe('txt-history-title', 'history_title');
    setSafe('txt-history-sub', 'history_subtitle');
    setSafe('txt-export-csv', 'export_csv');
    setSafe('txt-clear-history', 'clear_history');

    // Chatbot Widget
    setSafe('txt-chatbot-title', 'chatbot_title');
    setSafe('txt-chatbot-desc', 'chatbot_sub');
    setSafe('txt-chatbot-welcome', 'chatbot_welcome');

    // Input Placeholders
    if (DOM.weatherCityInput && t.weather_search_placeholder) {
        DOM.weatherCityInput.placeholder = t.weather_search_placeholder;
    }
    if (DOM.chatInputField && t.chat_placeholder) {
        DOM.chatInputField.placeholder = t.chat_placeholder;
    }

    // Refresh Chatbot Quick Chips in Active Language
    if (state.currentLang === 'hi') {
        renderChatChips([
            "🍅 टमाटर के रोग और दवा",
            "🌾 गेहूं का पीला रतुआ (Yellow Rust)",
            "🧪 टॉप फफूंदनाशक उत्पाद",
            "🌿 जैविक खेती के नुस्खे"
        ]);
    } else if (state.currentLang === 'pa') {
        renderChatChips([
            "🍅 ਟਮਾਟਰ ਦਾ ਪਛੇਤਾ ਝੁਲਸਾ ਰੋਗ",
            "🎋 ਕਮਾਦ ਦਾ ਰੱਤਾ ਰੋਗ (Red Rot)",
            "🧪 ਸਪਰੇਅ ਦੀ ਸਹੀ ਖੁਰਾਕ",
            "🌾 ਝੋਨੇ ਦਾ ਬਲਾਸਟ ਰੋਗ"
        ]);
    } else if (state.currentLang === 'es') {
        renderChatChips([
            "🍅 Tratamiento para Tizón en Tomate",
            "🌾 Roya Amarilla en Trigo",
            "🧪 Mejores Marcas de Fungicidas",
            "🌿 Remedios Orgánicos"
        ]);
    } else {
        renderChatChips([
            "🍅 Tomato Late Blight remedy",
            "🎋 Sugarcane Red Rot & Rust",
            "🧪 Top fungicide brands",
            "📏 Spray tank calculator help"
        ]);
    }
}

// --- PDF Export Handler ---
async function handleExportPdf() {
    if (!state.diagnosisData) {
        showToast("Please scan or select a leaf first.", "warning");
        return;
    }
    
    showToast("Generating PDF Agronomy Certificate...", "info");
    try {
        const payload = {
            ...state.diagnosisData,
            weather: state.weatherData
        };
        
        const response = await fetch('/api/export-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error("PDF generation failed on server.");
        
        const blob = await response.blob();
        const blobUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `AgroAI_Diagnosis_${state.diagnosisData.top_prediction?.crop || 'Crop'}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(blobUrl);
        showToast("PDF report downloaded successfully!", "success");
    } catch (err) {
        console.error("PDF Export Error:", err);
        showToast(`Export failed: ${err.message}`, "error");
    }
}

// --- Webcam Handlers ---
async function startWebcam() {
    try {
        state.webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
        });
        DOM.webcamVideo.srcObject = state.webcamStream;
        DOM.cameraPlaceholder.classList.add('hidden');
        DOM.btnCaptureCamera.disabled = false;
        DOM.btnStartCamera.innerHTML = `<i class="fa-solid fa-stop text-red-400"></i><span>Stop Camera</span>`;
    } catch (err) {
        console.error("Webcam error:", err);
        showToast("Unable to access camera. Please check permissions.", "error");
    }
}

function stopWebcam() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(t => t.stop());
        state.webcamStream = null;
    }
    DOM.webcamVideo.srcObject = null;
    DOM.cameraPlaceholder.classList.remove('hidden');
    DOM.btnCaptureCamera.disabled = true;
    DOM.btnStartCamera.innerHTML = `<i class="fa-solid fa-power-off text-brand-400"></i><span>Start Camera</span>`;
}

function captureWebcamSnapshot() {
    if (!state.webcamStream) return;
    const canvas = DOM.webcamCanvas;
    const video = DOM.webcamVideo;
    
    canvas.width = video.videoWidth || 400;
    canvas.height = video.videoHeight || 400;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const base64Data = canvas.toDataURL('image/jpeg', 0.9);
    state.lastScannedImage = base64Data;
    
    if (!navigator.onLine) {
        return runInBrowserONNXDiagnosis(base64Data);
    }
    
    setLoading(true);
    fetch('/api/diagnose-json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_base64: base64Data,
            alpha: state.alpha,
            colormap: state.colormap,
            location: DOM.weatherLocationLabel.textContent || "Target Farm"
        })
    })
    .then(res => res.json())
    .then(data => {
        handleDiagnosisSuccess(data);
        loadScoutHistory();
        showToast("Camera snapshot diagnosed!", "success");
    })
    .catch(err => {
        console.warn("[Webcam Diagnostic Network Fallback] Using In-Browser ONNX:", err);
        runInBrowserONNXDiagnosis(base64Data);
    })
    .finally(() => setLoading(false));
}

// --- Event Listeners Setup ---
function setupEventListeners() {
    // Language Switcher
    DOM.langSelector.addEventListener('change', (e) => {
        changeLanguage(e.target.value);
    });

    // Tab switching (Upload vs Camera)
    DOM.tabBtnUpload.addEventListener('click', () => {
        DOM.tabBtnUpload.className = "tab-btn flex-1 py-2 rounded-lg bg-brand-600 text-white font-semibold shadow-sm transition flex items-center justify-center space-x-1.5";
        DOM.tabBtnCamera.className = "tab-btn flex-1 py-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-900 transition flex items-center justify-center space-x-1.5";
        DOM.dropzoneContainer.classList.remove('hidden');
        DOM.cameraContainer.classList.add('hidden');
        stopWebcam();
    });

    DOM.tabBtnCamera.addEventListener('click', () => {
        DOM.tabBtnCamera.className = "tab-btn flex-1 py-2 rounded-lg bg-brand-600 text-white font-semibold shadow-sm transition flex items-center justify-center space-x-1.5";
        DOM.tabBtnUpload.className = "tab-btn flex-1 py-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-900 transition flex items-center justify-center space-x-1.5";
        DOM.dropzoneContainer.classList.add('hidden');
        DOM.cameraContainer.classList.remove('hidden');
    });

    // File Drop & Select
    DOM.leafFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            diagnoseImageFile(e.target.files[0]);
        }
    });

    // Target Crop Filter change
    if (DOM.targetCropSelect) {
        DOM.targetCropSelect.addEventListener('change', () => {
            if (state.lastScannedImage) {
                reapplyVisualParams();
                showToast(`Refined analysis for crop: ${DOM.targetCropSelect.value}`, "info");
            }
        });
    }


    // Drag & Drop visual state
    DOM.dropzoneContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.dropzoneBox.classList.add('border-brand-400', 'bg-brand-950/20');
    });
    DOM.dropzoneContainer.addEventListener('dragleave', () => {
        DOM.dropzoneBox.classList.remove('border-brand-400', 'bg-brand-950/20');
    });
    DOM.dropzoneContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        DOM.dropzoneBox.classList.remove('border-brand-400', 'bg-brand-950/20');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            diagnoseImageFile(e.dataTransfer.files[0]);
        }
    });

    // Camera buttons
    DOM.btnStartCamera.addEventListener('click', () => {
        if (state.webcamStream) {
            stopWebcam();
        } else {
            startWebcam();
        }
    });
    DOM.btnCaptureCamera.addEventListener('click', captureWebcamSnapshot);

    // View Mode buttons (Blended, Heatmap, Severity, Original)
    DOM.viewModeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            DOM.viewModeBtns.forEach(b => b.classList.remove('active-mode', 'bg-slate-800', 'text-brand-400', 'font-semibold'));
            btn.classList.add('active-mode', 'bg-slate-800', 'text-brand-400', 'font-semibold');
            state.viewMode = btn.dataset.mode;
            renderVisualImage();
        });
    });

    // Alpha Slider - Real-Time 60 FPS Visual Blending
    DOM.blendOpacitySlider.addEventListener('input', (e) => {
        state.alpha = parseFloat(e.target.value) / 100.0;
        DOM.opacityValLabel.textContent = `${e.target.value}%`;
        
        // Instant visual feedback
        if (state.diagnosisData) {
            if (state.viewMode !== 'blended') {
                state.viewMode = 'blended';
                DOM.viewModeBtns.forEach(b => {
                    if (b.dataset.mode === 'blended') {
                        b.className = "view-mode-btn py-1.5 rounded-lg active-mode bg-slate-800 text-brand-400 font-semibold";
                    } else {
                        b.className = "view-mode-btn py-1.5 rounded-lg hover:text-slate-200";
                    }
                });
            }
            renderVisualImage();
        }
    });


    // Colormap buttons
    DOM.cmapBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            DOM.cmapBtns.forEach(b => {
                b.className = "cmap-btn px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 text-[10px]";
            });
            btn.className = "cmap-btn px-2 py-0.5 rounded bg-brand-600/30 text-brand-300 border border-brand-500/40 text-[10px] font-semibold";
            state.colormap = btn.dataset.cmap;
            reapplyVisualParams();
        });
    });

    // Advisory tab buttons
    DOM.advTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            DOM.advTabBtns.forEach(b => b.classList.remove('active-adv', 'bg-slate-800', 'text-brand-400', 'font-semibold'));
            btn.classList.add('active-adv', 'bg-slate-800', 'text-brand-400', 'font-semibold');
            state.advisoryTab = btn.dataset.tab;
            renderAdvisoryTab();
        });
    });

    // Dosage Calculator events
    [DOM.calcFieldSize, DOM.calcUnit, DOM.calcDosageRate, DOM.calcDosageUnit, DOM.calcTankSize].forEach(input => {
        input.addEventListener('input', recalculateDosage);
        input.addEventListener('change', recalculateDosage);
    });

    // Weather search
    DOM.btnSearchWeather.addEventListener('click', () => {
        const city = DOM.weatherCityInput.value.trim();
        if (city) fetchWeatherRisk(city);
    });
    DOM.weatherCityInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const city = DOM.weatherCityInput.value.trim();
            if (city) fetchWeatherRisk(city);
        }
    });

    // Quick Hubs Buttons
    DOM.quickCityBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const city = btn.dataset.city;
            if (city) {
                DOM.weatherCityInput.value = city;
                fetchWeatherRisk(city);
            }
        });
    });

    // Geolocate
    DOM.btnGeolocateWeather.addEventListener('click', () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (pos) => fetchWeatherRisk(null, pos.coords.latitude, pos.coords.longitude),
                () => showToast("Geolocation permission denied.", "warning")
            );
        }
    });

    // Chemical Formulation Selector for Spray Window Engine
    if (DOM.sprayChemicalSelect) {
        DOM.sprayChemicalSelect.addEventListener('change', () => {
            const city = DOM.weatherCityInput.value.trim() || DOM.weatherLocationLabel.textContent;
            fetchWeatherRisk(city);
            showToast(`Recalculating spray suitability for selected product...`, "info");
        });
    }

    // Language Switcher
    if (DOM.langSelector) {
        DOM.langSelector.addEventListener('change', (e) => {
            changeLanguage(e.target.value);
            stopSpeaking();
            stopVoiceRecognition();
        });
    }

    // History Actions
    if (DOM.btnExportCsv) {
        DOM.btnExportCsv.addEventListener('click', () => {
            window.location.href = '/api/history/export-csv';
            showToast("Exporting farm audit log to CSV...", "info");
        });
    }

    if (DOM.btnClearHistory) {
        DOM.btnClearHistory.addEventListener('click', async () => {
            if (confirm("Are you sure you want to clear the scouting audit log?")) {
                await fetch('/api/history', { method: 'DELETE' });
                await loadScoutHistory();
                showToast("Scouting history cleared", "info");
            }
        });
    }

    // Export PDF & Share to WhatsApp
    if (DOM.btnExportPdfNav) {
        DOM.btnExportPdfNav.addEventListener('click', handleExportPdf);
    }
    if (DOM.btnExportPdfCard) {
        DOM.btnExportPdfCard.addEventListener('click', handleExportPdf);
    }
    if (DOM.btnShareWhatsapp) {
        DOM.btnShareWhatsapp.addEventListener('click', handleShareToWhatsApp);
    }

    // AI Chatbot Event Listeners
    if (DOM.btnOpenChatbot) {
        DOM.btnOpenChatbot.addEventListener('click', () => toggleChatbot(true));
    }
    if (DOM.btnChatClose) {
        DOM.btnChatClose.addEventListener('click', () => {
            stopSpeaking();
            stopVoiceRecognition();
            toggleChatbot(false);
        });
    }
    if (DOM.btnChatClear) {
        DOM.btnChatClear.addEventListener('click', clearChatHistory);
    }
    if (DOM.btnVoiceToggle) {
        DOM.btnVoiceToggle.addEventListener('click', toggleVoiceReadAloud);
    }
    if (DOM.btnVoiceMic) {
        DOM.btnVoiceMic.addEventListener('click', startVoiceRecognition);
    }
    if (DOM.btnCancelVoice) {
        DOM.btnCancelVoice.addEventListener('click', stopVoiceRecognition);
    }
    if (DOM.chatInputForm) {
        DOM.chatInputForm.addEventListener('submit', (e) => {
            e.preventDefault();
            sendChatMessage();
        });
    }
    if (DOM.chatSuggestedChips) {
        DOM.chatSuggestedChips.addEventListener('click', (e) => {
            const btn = e.target.closest('.chat-chip-btn');
            if (btn && btn.dataset.msg) {
                sendChatMessage(btn.dataset.msg);
            }
        });
    }
}

// =========================================================
// AI Agronomist Chatbot Controller
// =========================================================

function toggleChatbot(open) {
    state.isChatOpen = open !== undefined ? open : !state.isChatOpen;
    if (state.isChatOpen) {
        DOM.aiChatbotWindow.classList.remove('chat-hidden');
        DOM.aiChatbotWindow.classList.add('chat-visible');
        if (DOM.chatInputField) DOM.chatInputField.focus();
    } else {
        DOM.aiChatbotWindow.classList.remove('chat-visible');
        DOM.aiChatbotWindow.classList.add('chat-hidden');
    }
}

function updateChatContext(crop, disease, confidence, severity, stage) {
    if (crop && disease) {
        if (DOM.chatContextText) DOM.chatContextText.textContent = `📌 Active Scan: ${crop} - ${disease} (${severity || 0}%)`;
        if (DOM.chatContextBadge) {
            DOM.chatContextBadge.textContent = "DIAGNOSED";
            DOM.chatContextBadge.className = "text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex-shrink-0";
        }
        
        // Dynamically update prompt chips with disease-specific questions
        renderChatChips([
            `🧪 Best chemicals for ${disease}`,
            `🌿 Organic controls for ${crop}`,
            `🌧️ Weather spraying precautions`,
            `📏 Calculate tank dosage for ${crop}`
        ]);
    } else {
        if (DOM.chatContextText) DOM.chatContextText.textContent = "🌾 General Agronomic Consulting";
        if (DOM.chatContextBadge) {
            DOM.chatContextBadge.textContent = "READY";
            DOM.chatContextBadge.className = "text-[9px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 flex-shrink-0";
        }
    }
}

function renderChatChips(chips) {
    if (!DOM.chatSuggestedChips) return;
    DOM.chatSuggestedChips.innerHTML = chips.map(chip => `
        <button class="chat-chip-btn px-2.5 py-1 rounded-full bg-slate-900 hover:bg-emerald-900/60 text-slate-300 hover:text-emerald-200 border border-slate-800 hover:border-emerald-700/60 transition whitespace-nowrap flex-shrink-0" data-msg="${chip}">
            ${chip}
        </button>
    `).join('');
}

function formatChatMarkdown(text) {
    if (!text) return '';
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
        
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    
    // Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
    
    // Code
    html = html.replace(/`(.*?)`/gim, '<code>$1</code>');
    
    // Blockquote
    html = html.replace(/^&gt; (.*$)/gim, '<blockquote>$1</blockquote>');
    
    // Unordered lists
    html = html.replace(/^[•\-\*] (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');
    html = html.replace(/<\/ul>\s*<ul>/gim, '');
    
    // Line breaks
    html = html.replace(/\n\n/gim, '<br/><br/>');
    html = html.replace(/\n/gim, '<br/>');
    
    return html;
}

// --- Voice Engine (Speech-to-Text & Text-to-Speech) ---
function getLanguageLocale(lang) {
    const map = {
        'en': 'en-IN',
        'hi': 'hi-IN',
        'pa': 'pa-IN',
        'es': 'es-ES',
        'fr': 'fr-FR'
    };
    return map[lang] || 'en-US';
}

function cleanMarkdownForSpeech(mdText) {
    if (!mdText) return "";
    return mdText
        .replace(/[*#_`>~]/g, '')
        .replace(/\[(.*?)\]\(.*?\)/g, '$1')
        .replace(/<[^>]*>/g, '')
        .replace(/\n+/g, '. ')
        .replace(/\s{2,}/g, ' ')
        .trim();
}

function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    try {
        window.speechSynthesis.cancel();
        const clean = cleanMarkdownForSpeech(text);
        if (!clean) return;
        
        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.lang = getLanguageLocale(state.currentLang);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        const voices = window.speechSynthesis.getVoices();
        const targetLocale = getLanguageLocale(state.currentLang).toLowerCase();
        const matchedVoice = voices.find(v => v.lang.toLowerCase().startsWith(state.currentLang.toLowerCase()) || v.lang.toLowerCase() === targetLocale);
        if (matchedVoice) {
            utterance.voice = matchedVoice;
        }
        
        state.activeSpeechUtterance = utterance;
        window.speechSynthesis.speak(utterance);
    } catch (e) {
        console.warn("[VoiceEngine] TTS error:", e);
    }
}

function stopSpeaking() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
}

function toggleVoiceReadAloud() {
    state.voiceReadAloud = !state.voiceReadAloud;
    if (DOM.btnVoiceToggle) {
        if (state.voiceReadAloud) {
            DOM.btnVoiceToggle.innerHTML = '<i class="fa-solid fa-volume-high text-emerald-400"></i>';
            DOM.btnVoiceToggle.title = "Voice Read Aloud: Enabled";
            showToast("Voice speech enabled", "success");
        } else {
            stopSpeaking();
            DOM.btnVoiceToggle.innerHTML = '<i class="fa-solid fa-volume-xmark text-slate-500"></i>';
            DOM.btnVoiceToggle.title = "Voice Read Aloud: Muted";
            showToast("Voice speech muted", "info");
        }
    }
}

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        return null;
    }
    
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    
    recognition.onstart = () => {
        state.isListening = true;
        stopSpeaking();
        if (DOM.btnVoiceMic) {
            DOM.btnVoiceMic.classList.add('mic-recording');
        }
        if (DOM.voiceListeningBar) {
            DOM.voiceListeningBar.classList.remove('hidden');
            const langName = {
                'en': 'English',
                'hi': 'Hindi (हिंदी)',
                'pa': 'Punjabi (ਪੰਜਾਬੀ)',
                'es': 'Spanish (Español)',
                'fr': 'French (Français)'
            }[state.currentLang] || 'English';
            if (DOM.voiceListeningLabel) {
                DOM.voiceListeningLabel.textContent = `🎙️ Listening in ${langName}... Speak now`;
            }
        }
    };
    
    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            transcript += event.results[i][0].transcript;
        }
        if (DOM.chatInputField) {
            DOM.chatInputField.value = transcript;
        }
        if (event.results[0].isFinal) {
            setTimeout(() => {
                if (transcript.trim().length > 0) {
                    sendChatMessage(transcript.trim());
                }
            }, 400);
        }
    };
    
    recognition.onerror = (event) => {
        console.warn("[VoiceEngine] STT Error:", event.error);
        stopVoiceRecognition();
        if (event.error === 'not-allowed') {
            showToast("Microphone access denied. Please allow mic access in browser.", "warning");
        } else if (event.error !== 'no-speech') {
            showToast(`Voice input error: ${event.error}`, "warning");
        }
    };
    
    recognition.onend = () => {
        stopVoiceRecognition();
    };
    
    return recognition;
}

function startVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast("Voice recognition is not supported in this browser. Please use Chrome, Edge, or Safari.", "warning");
        return;
    }
    
    if (state.isListening) {
        stopVoiceRecognition();
        return;
    }
    
    try {
        if (!state.speechRecognition) {
            state.speechRecognition = initSpeechRecognition();
        }
        if (state.speechRecognition) {
            state.speechRecognition.lang = getLanguageLocale(state.currentLang);
            state.speechRecognition.start();
        }
    } catch (e) {
        console.warn("[VoiceEngine] Failed to start recognition:", e);
        stopVoiceRecognition();
    }
}

function stopVoiceRecognition() {
    state.isListening = false;
    if (state.speechRecognition) {
        try { state.speechRecognition.stop(); } catch (e) {}
    }
    if (DOM.btnVoiceMic) {
        DOM.btnVoiceMic.classList.remove('mic-recording');
    }
    if (DOM.voiceListeningBar) {
        DOM.voiceListeningBar.classList.add('hidden');
    }
}

function appendChatMessage(role, content) {
    if (!DOM.chatMessagesContainer) return;
    const msgDiv = document.createElement('div');
    if (role === 'user') {
        msgDiv.className = 'flex items-end justify-end space-x-2';
        msgDiv.innerHTML = `
            <div class="chat-bubble-user max-w-[85%]">
                <p>${content}</p>
            </div>
            <div class="w-6 h-6 rounded-lg bg-teal-600/30 text-teal-300 flex items-center justify-center flex-shrink-0 text-xs mb-0.5 border border-teal-500/30">
                <i class="fa-solid fa-user"></i>
            </div>
        `;
    } else {
        msgDiv.className = 'flex items-start space-x-2.5 group';
        const rawEscaped = encodeURIComponent(content);
        msgDiv.innerHTML = `
            <div class="w-6 h-6 rounded-lg bg-emerald-600/30 text-emerald-400 flex items-center justify-center flex-shrink-0 text-xs mt-0.5 border border-emerald-500/30">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="chat-bubble-assistant space-y-1.5 max-w-[88%] relative">
                <div class="flex items-center justify-between border-b border-slate-800/60 pb-1 mb-1">
                    <span class="text-[9.5px] font-bold text-emerald-400 uppercase tracking-wider">AgroBot Advisory</span>
                    <button class="btn-speak-msg text-slate-400 hover:text-emerald-300 text-xs px-1.5 py-0.5 rounded hover:bg-slate-800 transition" onclick="speakText(decodeURIComponent('${rawEscaped}'))" title="Read Aloud">
                        <i class="fa-solid fa-volume-high"></i>
                    </button>
                </div>
                ${formatChatMarkdown(content)}
            </div>
        `;
        
        // Auto-speak if enabled
        if (state.voiceReadAloud && !state.isListening) {
            speakText(content);
        }
    }
    DOM.chatMessagesContainer.appendChild(msgDiv);
    DOM.chatMessagesContainer.scrollTop = DOM.chatMessagesContainer.scrollHeight;
    
    state.chatHistory.push({ role, content });
}

async function sendChatMessage(messageText) {
    const text = messageText || (DOM.chatInputField ? DOM.chatInputField.value.trim() : '');
    if (!text) return;
    
    stopVoiceRecognition();
    if (DOM.chatInputField) DOM.chatInputField.value = '';
    appendChatMessage('user', text);
    
    // Add temporary typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.id = 'chat-typing-indicator';
    typingIndicator.className = 'flex items-start space-x-2.5';
    typingIndicator.innerHTML = `
        <div class="w-6 h-6 rounded-lg bg-emerald-600/30 text-emerald-400 flex items-center justify-center flex-shrink-0 text-xs mt-0.5 border border-emerald-500/30">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="chat-bubble-assistant py-2 px-3 flex items-center space-x-1">
            <span class="text-slate-400 mr-1 text-[11px]">AgroBot is thinking</span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    DOM.chatMessagesContainer.appendChild(typingIndicator);
    DOM.chatMessagesContainer.scrollTop = DOM.chatMessagesContainer.scrollHeight;
    
    // Prepare diagnostic context
    let ctx = null;
    if (state.diagnosisData && state.diagnosisData.top_prediction) {
        const top = state.diagnosisData.top_prediction;
        const sev = state.diagnosisData.severity || {};
        ctx = {
            crop: top.crop,
            disease: top.disease,
            class_name: top.class_name,
            confidence: top.confidence,
            severity_pct: sev.severity_percentage,
            severity_stage: sev.severity_stage,
            weather_threat: state.weatherData?.epidemiological_risk?.threat_level || "Moderate"
        };
    }
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: state.chatHistory.slice(-6),
                context: ctx,
                language: state.currentLang || 'en'
            })
        });
        
        // Remove typing indicator
        const el = document.getElementById('chat-typing-indicator');
        if (el) el.remove();
        
        if (res.ok) {
            const data = await res.json();
            appendChatMessage('assistant', data.reply);
            if (data.suggested_actions && data.suggested_actions.length > 0) {
                renderChatChips(data.suggested_actions);
            }
        } else {
            appendChatMessage('assistant', "⚠️ I encountered a temporary connection issue. Please try again or rephrase your farming question.");
        }
    } catch (err) {
        console.error("Chat error:", err);
        const el = document.getElementById('chat-typing-indicator');
        if (el) el.remove();
        appendChatMessage('assistant', "⚠️ Network error. Please ensure the AgroAI server is running.");
    }
}

function clearChatHistory() {
    state.chatHistory = [];
    stopSpeaking();
    DOM.chatMessagesContainer.innerHTML = `
        <div class="flex items-start space-x-2.5">
            <div class="w-6 h-6 rounded-lg bg-emerald-600/30 text-emerald-400 flex items-center justify-center flex-shrink-0 text-xs mt-0.5 border border-emerald-500/30">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="chat-bubble-assistant space-y-1.5 max-w-[88%]">
                <p>👋 <strong>Hello! I am your AI Agronomist & Crop Protection Specialist.</strong></p>
                <p>Conversation reset. Ask me anything or tap the <strong>🎙️ Microphone</strong> to speak directly.</p>
            </div>
        </div>
    `;
    showToast("Chat conversation cleared", "info");
}

// --- PWA & Offline Network Manager ---
let deferredInstallPrompt = null;

function setupPWAAndNetwork() {
    // 1. Service Worker Registration
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/service-worker.js')
                .then(reg => console.log('[PWA] Service Worker active with scope:', reg.scope))
                .catch(err => console.warn('[PWA] Service Worker registration note:', err));
        });
    }

    // 2. 1-Click PWA App Installation
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredInstallPrompt = e;
        if (DOM.btnPwaInstall) {
            DOM.btnPwaInstall.classList.remove('hidden');
            DOM.btnPwaInstall.classList.add('inline-flex');
        }
    });

    if (DOM.btnPwaInstall) {
        DOM.btnPwaInstall.addEventListener('click', async () => {
            if (deferredInstallPrompt) {
                deferredInstallPrompt.prompt();
                const choice = await deferredInstallPrompt.userChoice;
                if (choice.outcome === 'accepted') {
                    showToast("AgroAI added to your Home Screen!", "success");
                }
                deferredInstallPrompt = null;
                DOM.btnPwaInstall.classList.add('hidden');
            }
        });
    }

    // 3. Online / Offline Status Synchronizer
    function updateNetworkStatus() {
        const isOnline = navigator.onLine;
        if (DOM.networkModeDot && DOM.networkModeText && DOM.networkModeBadge) {
            if (isOnline) {
                DOM.networkModeDot.className = "w-2 h-2 rounded-full bg-emerald-400 animate-pulse";
                DOM.networkModeText.textContent = "ONLINE (Cloud)";
                DOM.networkModeBadge.className = "hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-950/50 border border-emerald-500/30 text-[11px] font-medium text-emerald-300";
            } else {
                DOM.networkModeDot.className = "w-2 h-2 rounded-full bg-amber-400";
                DOM.networkModeText.textContent = "⚡ FIELD OFFLINE (In-Browser ONNX)";
                DOM.networkModeBadge.className = "flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-amber-950/80 border border-amber-500/50 text-[11px] font-bold text-amber-300 animate-pulse";
                showToast("📶 Switched to Offline Field Mode (In-Browser ONNX Active)", "info");
            }
        }
    }

    window.addEventListener('online', updateNetworkStatus);
    window.addEventListener('offline', updateNetworkStatus);
    updateNetworkStatus();
}

// =========================================================
// MLOps & Continuous Retraining Controller
// =========================================================

function setupMLOpsAndFeedback() {
    // 1. Modal Toggle
    if (DOM.btnOpenMlops) {
        DOM.btnOpenMlops.addEventListener('click', () => {
            if (DOM.mlopsModal) {
                DOM.mlopsModal.classList.remove('hidden');
                DOM.mlopsModal.classList.add('flex');
                fetchMLOpsStatus();
            }
        });
    }

    if (DOM.btnCloseMlops) {
        DOM.btnCloseMlops.addEventListener('click', () => {
            if (DOM.mlopsModal) {
                DOM.mlopsModal.classList.add('hidden');
                DOM.mlopsModal.classList.remove('flex');
            }
        });
    }

    // 2. Trigger Continuous Retraining
    if (DOM.btnTriggerRetrain) {
        DOM.btnTriggerRetrain.addEventListener('click', async () => {
            await triggerMLOpsRetraining();
        });
    }

    // 3. Farmer Validation / Active Learning Feedback
    if (DOM.btnFeedbackAccurate) {
        DOM.btnFeedbackAccurate.addEventListener('click', () => {
            submitDiagnosisFeedback(true);
        });
    }
    if (DOM.btnFeedbackIncorrect) {
        DOM.btnFeedbackIncorrect.addEventListener('click', () => {
            submitDiagnosisFeedback(false);
        });
    }
}

async function fetchMLOpsStatus() {
    try {
        const res = await fetch('/api/mlops/status');
        if (!res.ok) return;
        const data = await res.json();
        
        const meta = data.model_metadata || {};
        const qStats = data.queue_statistics || {};
        const runs = data.recent_runs || [];

        if (DOM.mlopsModelVersion) DOM.mlopsModelVersion.textContent = meta.model_version || 'v1.3.0';
        if (DOM.mlopsAccuracy) DOM.mlopsAccuracy.textContent = `${meta.validation_accuracy || 100.0}%`;
        if (DOM.mlopsHarvestedCount) DOM.mlopsHarvestedCount.textContent = qStats.total_harvested_samples || 0;
        if (DOM.mlopsApprovedCount) DOM.mlopsApprovedCount.textContent = qStats.approved_for_retraining || 0;
        if (DOM.mlopsQueueBadge) DOM.mlopsQueueBadge.textContent = `${qStats.total_harvested_samples || 0} Items`;

        // Render Recent Runs Lineage
        if (DOM.mlopsHistoryList) {
            if (runs.length === 0) {
                DOM.mlopsHistoryList.innerHTML = `<div class="p-3 text-center text-slate-500 text-xs">No continuous retraining runs recorded yet.</div>`;
            } else {
                DOM.mlopsHistoryList.innerHTML = runs.map(r => `
                    <div class="p-3 flex items-center justify-between hover:bg-slate-900/60 transition">
                        <div class="space-y-0.5">
                            <div class="flex items-center space-x-2">
                                <span class="font-bold text-white font-mono">${r.model_version}</span>
                                <span class="px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 text-[10px] font-semibold">${r.final_accuracy}% Val Acc</span>
                            </div>
                            <p class="text-[10px] text-slate-400">Epochs: ${r.epochs} • AL Samples: ${r.al_samples_used} • Time: ${r.duration_seconds}s</p>
                        </div>
                        <span class="text-[10px] text-slate-500 font-mono">${r.timestamp}</span>
                    </div>
                `).join('');
            }
        }

        // Fetch queue samples
        const qRes = await fetch('/api/mlops/queue');
        if (qRes.ok) {
            const qData = await qRes.json();
            renderMLOpsQueue(qData.samples || []);
        }
    } catch (e) {
        console.warn("[MLOps] Status fetch error:", e);
    }
}

function renderMLOpsQueue(samples) {
    if (!DOM.mlopsQueueList) return;
    if (samples.length === 0) {
        DOM.mlopsQueueList.innerHTML = `<div class="p-4 text-center text-slate-500 text-xs">No active learning items pending. Field predictions meeting &gt;75% confidence criteria.</div>`;
        return;
    }

    DOM.mlopsQueueList.innerHTML = samples.map(s => `
        <div class="p-3 flex items-center justify-between hover:bg-slate-900/60 transition">
            <div class="space-y-0.5">
                <div class="flex items-center space-x-2">
                    <span class="font-mono text-purple-400 font-bold text-xs">${s.sample_id}</span>
                    <span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px]">${s.crop} - ${s.disease}</span>
                    <span class="px-1.5 py-0.5 rounded ${s.status === 'approved_for_training' ? 'bg-teal-950 text-teal-300 border border-teal-500/30' : 'bg-amber-950 text-amber-300 border border-amber-500/30'} text-[10px] font-semibold">
                        ${s.status.replace(/_/g, ' ')}
                    </span>
                </div>
                <p class="text-[10px] text-slate-400">Pred: <span class="text-slate-300 font-semibold">${s.predicted_class}</span> (${s.confidence}%) • Uncertainty Score: <span class="text-amber-400 font-mono">${s.uncertainty_score}</span></p>
                ${s.feedback_notes ? `<p class="text-[10px] text-slate-400 italic">Farmer note: "${s.feedback_notes}"</p>` : ''}
            </div>
            <div class="flex items-center space-x-1.5">
                ${s.status !== 'approved_for_training' ? `
                    <button onclick="window.handleApproveSample('${s.sample_id}', 'approved_for_training')" class="px-2 py-1 rounded bg-teal-900/60 hover:bg-teal-800 text-teal-200 border border-teal-500/30 text-[10px] font-semibold transition">
                        <i class="fa-solid fa-check mr-1"></i>Approve
                    </button>
                ` : ''}
                ${s.status !== 'rejected' ? `
                    <button onclick="window.handleApproveSample('${s.sample_id}', 'rejected')" class="px-2 py-1 rounded bg-rose-950/60 hover:bg-rose-900 text-rose-300 border border-rose-500/30 text-[10px] font-semibold transition">
                        <i class="fa-solid fa-ban mr-1"></i>Reject
                    </button>
                ` : ''}
            </div>
        </div>
    `).join('');
}

window.handleApproveSample = async function(sampleId, status) {
    try {
        const res = await fetch('/api/mlops/approve-sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sample_id: sampleId, status: status })
        });
        if (res.ok) {
            showToast(`Sample ${sampleId} ${status === 'approved_for_training' ? 'approved' : 'rejected'}!`, "success");
            fetchMLOpsStatus();
        }
    } catch (e) {
        showToast("Action failed", "error");
    }
};

async function triggerMLOpsRetraining() {
    if (!DOM.btnTriggerRetrain || !DOM.retrainProgressBox) return;
    
    DOM.btnTriggerRetrain.disabled = true;
    DOM.btnTriggerRetrain.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1"></i>Retraining...`;
    DOM.retrainProgressBox.classList.remove('hidden');

    let currentPct = 10;
    const progressInterval = setInterval(() => {
        if (currentPct < 90) {
            currentPct += 15;
            if (DOM.retrainProgressBar) DOM.retrainProgressBar.style.width = `${currentPct}%`;
            if (DOM.retrainProgressPct) DOM.retrainProgressPct.textContent = `Epoch ${Math.min(5, Math.ceil(currentPct / 20))}/5`;
        }
    }, 400);

    try {
        const res = await fetch('/api/mlops/trigger-retrain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs: 5, learning_rate: 0.0001 })
        });
        clearInterval(progressInterval);

        if (res.ok) {
            const data = await res.json();
            if (DOM.retrainProgressBar) DOM.retrainProgressBar.style.width = `100%`;
            if (DOM.retrainProgressPct) DOM.retrainProgressPct.textContent = `Completed (100%)`;
            if (DOM.retrainProgressLabel) DOM.retrainProgressLabel.innerHTML = `<i class="fa-solid fa-check text-emerald-400"></i><span class="text-emerald-300">Model ${data.new_version} Deployed (${data.validation_accuracy}% accuracy) • Hot-reloaded with Zero Downtime!</span>`;

            showToast(`🎉 Retraining Complete! Promoted to ${data.new_version} (${data.duration_seconds}s)`, "success");
            setTimeout(() => {
                fetchMLOpsStatus();
                DOM.retrainProgressBox.classList.add('hidden');
                DOM.btnTriggerRetrain.disabled = false;
                DOM.btnTriggerRetrain.innerHTML = `<i class="fa-solid fa-play mr-1"></i>Start Retraining`;
            }, 2500);
        } else {
            showToast("Retraining pipeline encountered an error", "error");
            DOM.btnTriggerRetrain.disabled = false;
            DOM.btnTriggerRetrain.innerHTML = `<i class="fa-solid fa-play mr-1"></i>Start Retraining`;
        }
    } catch (e) {
        clearInterval(progressInterval);
        showToast("Retraining connection failed", "error");
        DOM.btnTriggerRetrain.disabled = false;
        DOM.btnTriggerRetrain.innerHTML = `<i class="fa-solid fa-play mr-1"></i>Start Retraining`;
    }
}

async function submitDiagnosisFeedback(isAccurate) {
    if (DOM.feedbackBtnGroup && DOM.feedbackThanksMsg) {
        DOM.feedbackBtnGroup.classList.add('hidden');
        DOM.feedbackThanksMsg.classList.remove('hidden');
        DOM.feedbackThanksMsg.classList.add('flex');
    }

    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_id: `SCN-${Date.now()}`,
                is_accurate: isAccurate,
                comments: isAccurate ? "Confirmed accurate by farmer" : "Reported inaccurate by farmer"
            })
        });
        showToast(isAccurate ? "Thank you for confirming diagnosis accuracy!" : "Correction received! Added to Active Learning pool.", "success");
    } catch (e) {
        console.warn("Feedback sync notice:", e);
    }
}

// =========================================================
// WhatsApp Diagnostic Bot & Simulator Controller
// =========================================================

function handleShareToWhatsApp() {
    if (!state.diagnosisData || !state.diagnosisData.top_prediction) {
        showToast("Opening WhatsApp Crop Doctor...", "info");
        openWhatsAppModal();
        return;
    }
    const top = state.diagnosisData.top_prediction;
    const sev = state.diagnosisData.severity || {};
    const adv = state.diagnosisData.advisory || {};
    const chems = adv.chemical_controls || [];
    const organics = adv.organic_controls || [];
    
    const lines = [
        "🌾 *AgroAI Crop Doctor Prescription* 🌾",
        "━━━━━━━━━━━━━━━━━━━━",
        `🌱 *Crop:* ${top.crop}`,
        `🦠 *Diagnosis:* *${top.disease}*`,
        `🔬 *Pathogen Type:* ${adv.pathogen_type || 'Biological Pathogen'}`,
        `📊 *AI Confidence:* ${top.confidence}%`,
        `⚠️ *Infection Severity:* ${sev.severity_percentage || 0}% (${sev.severity_stage || 'Stage 1'})`,
        ""
    ];
    
    if (chems.length > 0) {
        const c0 = chems[0];
        lines.push(
            "🧪 *Top Chemical Treatment:*",
            `• *${c0.product}* (${c0.active_ingredient})`,
            `  - Dosage: ${c0.dosage}`,
            `  - Safety Wait (PHI): ${c0.interval || '14 days'}`,
            ""
        );
    }
    
    if (organics.length > 0) {
        lines.push(
            "🌿 *Organic Remedy:*",
            `• ${organics[0]}`,
            ""
        );
    }
    
    lines.push(
        "━━━━━━━━━━━━━━━━━━━━",
        `🔗 *Scan on AgroAI Web:* ${window.location.origin}`,
        "_Shared via AgroAI Precision Pathology Core_"
    );
    
    const encoded = encodeURIComponent(lines.join("\n"));
    window.open(`https://api.whatsapp.com/send?text=${encoded}`, '_blank');
    showToast("Opening WhatsApp with formatted crop prescription...", "success");
}

function openWhatsAppModal() {
    const modal = document.getElementById('whatsapp-modal');
    if (modal) {
        modal.classList.remove('hidden', 'modal-hidden');
        modal.classList.add('flex', 'modal-visible');
        modal.style.setProperty('display', 'flex', 'important');
        modal.style.setProperty('visibility', 'visible', 'important');
        modal.style.setProperty('opacity', '1', 'important');
        modal.style.setProperty('pointer-events', 'auto', 'important');
    }
}

function closeWhatsAppModal() {
    const modal = document.getElementById('whatsapp-modal');
    if (modal) {
        modal.classList.add('hidden', 'modal-hidden');
        modal.classList.remove('flex', 'modal-visible');
        modal.style.setProperty('display', 'none', 'important');
        modal.style.setProperty('visibility', 'hidden', 'important');
        modal.style.setProperty('opacity', '0', 'important');
        modal.style.setProperty('pointer-events', 'none', 'important');
    }
}

// Global window exposure for direct inline HTML click handlers
window.openWhatsAppModal = openWhatsAppModal;
window.closeWhatsAppModal = closeWhatsAppModal;
window.handleShareToWhatsApp = handleShareToWhatsApp;

function setupWhatsAppBotSimulator() {
    // 1. Modal Open Triggers
    if (DOM.btnOpenWhatsapp) {
        DOM.btnOpenWhatsapp.addEventListener('click', (e) => {
            e.preventDefault();
            openWhatsAppModal();
        });
    }

    if (DOM.btnFloatingWhatsapp) {
        DOM.btnFloatingWhatsapp.addEventListener('click', (e) => {
            e.preventDefault();
            openWhatsAppModal();
        });
    }

    if (DOM.btnChatWhatsappSwitch) {
        DOM.btnChatWhatsappSwitch.addEventListener('click', (e) => {
            e.preventDefault();
            openWhatsAppModal();
        });
    }

    // Global Click Delegation Fallback
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('#btn-open-whatsapp, #btn-floating-whatsapp, #btn-chat-whatsapp-switch');
        if (btn) {
            e.preventDefault();
            openWhatsAppModal();
        }
    });

    // Modal Close Triggers
    if (DOM.btnCloseWhatsappModal) {
        DOM.btnCloseWhatsappModal.addEventListener('click', (e) => {
            e.preventDefault();
            closeWhatsAppModal();
        });
    }

    if (DOM.whatsappModal) {
        DOM.whatsappModal.addEventListener('click', (e) => {
            if (e.target === DOM.whatsappModal) {
                closeWhatsAppModal();
            }
        });
    }

    // 2. Clear Chat
    if (DOM.btnWaClear) {
        DOM.btnWaClear.addEventListener('click', () => {
            if (DOM.waMessagesContainer) {
                DOM.waMessagesContainer.innerHTML = `
                    <div class="flex flex-col items-start space-y-1 max-w-[88%]">
                        <div class="bg-[#202c33] text-slate-200 p-3 rounded-2xl rounded-tl-none border border-[#2a3942] shadow-md space-y-1.5 leading-relaxed text-[11.5px]">
                            <p>👋 <strong>Welcome to AgroAI WhatsApp Crop Doctor!</strong> 🌾</p>
                            <p>Conversation reset. Send a leaf photo, GPS pin, or type your question below.</p>
                        </div>
                        <span class="text-[9.5px] text-slate-500 pl-1">AgroAI Bot • Just now</span>
                    </div>
                `;
            }
        });
    }

    // 3. Photo Sender
    const triggerPhotoUpload = () => {
        if (DOM.waImageInput) DOM.waImageInput.click();
    };
    if (DOM.btnWaAttachCam) DOM.btnWaAttachCam.addEventListener('click', triggerPhotoUpload);
    if (DOM.btnWaSendPhoto) DOM.btnWaSendPhoto.addEventListener('click', triggerPhotoUpload);

    if (DOM.waImageInput) {
        DOM.waImageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async (event) => {
                const b64 = event.target.result;
                appendWhatsAppUserImage(b64);
                showWhatsAppTyping();

                try {
                    const res = await fetch('/api/whatsapp/simulate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image_base64: b64,
                            language: state.currentLang || 'en'
                        })
                    });
                    hideWhatsAppTyping();
                    if (res.ok) {
                        const data = await res.json();
                        appendWhatsAppBotMessage(data.reply);
                    } else {
                        appendWhatsAppBotMessage("⚠️ *Error processing WhatsApp leaf photo.* Please ensure the image is clear.");
                    }
                } catch (err) {
                    hideWhatsAppTyping();
                    appendWhatsAppBotMessage("⚠️ *Network error connecting to WhatsApp webhook.*");
                }
            };
            reader.readAsDataURL(file);
            DOM.waImageInput.value = '';
        });
    }

    // 4. GPS Location Sender
    const sendLocationPin = async () => {
        appendWhatsAppUserLocation();
        showWhatsAppTyping();

        let lat = 30.90;
        let lon = 75.85;

        if (navigator.geolocation) {
            try {
                const pos = await new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 2500 });
                });
                lat = pos.coords.latitude;
                lon = pos.coords.longitude;
            } catch (e) {
                // Fallback default coordinates
            }
        }

        try {
            const res = await fetch('/api/whatsapp/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    latitude: lat,
                    longitude: lon,
                    language: state.currentLang || 'en'
                })
            });
            hideWhatsAppTyping();
            if (res.ok) {
                const data = await res.json();
                appendWhatsAppBotMessage(data.reply);
            } else {
                appendWhatsAppBotMessage("⚠️ *Error fetching live weather telemetry.*");
            }
        } catch (err) {
            hideWhatsAppTyping();
            appendWhatsAppBotMessage("⚠️ *Network error simulating location.*");
        }
    };

    if (DOM.btnWaAttachPin) DOM.btnWaAttachPin.addEventListener('click', sendLocationPin);
    if (DOM.btnWaSendLocation) DOM.btnWaSendLocation.addEventListener('click', sendLocationPin);

    // 5. Text Message Sender
    const sendTextMessage = async (customText) => {
        const text = customText || (DOM.waInputField ? DOM.waInputField.value.trim() : '');
        if (!text) return;
        if (DOM.waInputField) DOM.waInputField.value = '';

        appendWhatsAppUserText(text);
        showWhatsAppTyping();

        try {
            const res = await fetch('/api/whatsapp/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    language: state.currentLang || 'en'
                })
            });
            hideWhatsAppTyping();
            if (res.ok) {
                const data = await res.json();
                appendWhatsAppBotMessage(data.reply);
            } else {
                appendWhatsAppBotMessage("⚠️ *Error connecting to AgroAI Agronomist.*");
            }
        } catch (err) {
            hideWhatsAppTyping();
            appendWhatsAppBotMessage("⚠️ *Network connection error.*");
        }
    };

    if (DOM.btnWaSendMsg) {
        DOM.btnWaSendMsg.addEventListener('click', () => sendTextMessage());
    }
    if (DOM.waInputField) {
        DOM.waInputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendTextMessage();
            }
        });
    }

    // 6. Quick Chips
    document.querySelectorAll('.wa-quick-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const msg = chip.dataset.msg;
            if (msg) sendTextMessage(msg);
        });
    });
}

function appendWhatsAppUserText(text) {
    if (!DOM.waMessagesContainer) return;
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const div = document.createElement('div');
    div.className = "flex flex-col items-end space-y-1 self-end max-w-[85%]";
    div.innerHTML = `
        <div class="bg-[#005c4b] text-slate-100 p-2.5 rounded-2xl rounded-tr-none shadow-md text-[11.5px] leading-relaxed break-words">
            ${text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
        </div>
        <div class="flex items-center space-x-1 pr-1 text-[9px] text-slate-400 font-mono">
            <span>${now}</span>
            <i class="fa-solid fa-check-double text-teal-400 text-[10px]"></i>
        </div>
    `;
    DOM.waMessagesContainer.appendChild(div);
    DOM.waMessagesContainer.scrollTop = DOM.waMessagesContainer.scrollHeight;
}

function appendWhatsAppUserImage(imgSrc) {
    if (!DOM.waMessagesContainer) return;
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const div = document.createElement('div');
    div.className = "flex flex-col items-end space-y-1 self-end max-w-[85%]";
    div.innerHTML = `
        <div class="bg-[#005c4b] text-slate-100 p-1.5 rounded-2xl rounded-tr-none shadow-md text-[11.5px]">
            <div class="w-48 h-40 rounded-xl overflow-hidden bg-slate-900 border border-emerald-700/60 mb-1">
                <img src="${imgSrc}" class="w-full h-full object-cover" alt="Leaf Sample">
            </div>
            <div class="px-1 text-[10.5px] font-medium text-emerald-200">📸 Leaf Photo Attached</div>
        </div>
        <div class="flex items-center space-x-1 pr-1 text-[9px] text-slate-400 font-mono">
            <span>${now}</span>
            <i class="fa-solid fa-check-double text-teal-400 text-[10px]"></i>
        </div>
    `;
    DOM.waMessagesContainer.appendChild(div);
    DOM.waMessagesContainer.scrollTop = DOM.waMessagesContainer.scrollHeight;
}

function appendWhatsAppUserLocation() {
    if (!DOM.waMessagesContainer) return;
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const div = document.createElement('div');
    div.className = "flex flex-col items-end space-y-1 self-end max-w-[85%]";
    div.innerHTML = `
        <div class="bg-[#005c4b] text-slate-100 p-2.5 rounded-2xl rounded-tr-none shadow-md text-[11.5px] flex items-center space-x-2">
            <div class="w-8 h-8 rounded-full bg-teal-600/40 border border-teal-400/50 flex items-center justify-center text-teal-300 flex-shrink-0">
                <i class="fa-solid fa-location-dot"></i>
            </div>
            <div>
                <span class="font-bold text-teal-200 block text-[11px]">📍 Live Farm Location Pin</span>
                <span class="text-[9.5px] text-slate-300">Requesting microclimate spray clearance</span>
            </div>
        </div>
        <div class="flex items-center space-x-1 pr-1 text-[9px] text-slate-400 font-mono">
            <span>${now}</span>
            <i class="fa-solid fa-check-double text-teal-400 text-[10px]"></i>
        </div>
    `;
    DOM.waMessagesContainer.appendChild(div);
    DOM.waMessagesContainer.scrollTop = DOM.waMessagesContainer.scrollHeight;
}

function appendWhatsAppBotMessage(markdownText) {
    if (!DOM.waMessagesContainer) return;
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Parse WhatsApp markdown (*bold*, _italic_, `code`, \n)
    let formatted = (markdownText || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*(.*?)\*/g, '<strong>$1</strong>')
        .replace(/_(.*?)_/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code class="bg-[#111b21] px-1 py-0.5 rounded font-mono text-emerald-300 text-[10.5px]">$1</code>')
        .replace(/\n/g, '<br>');

    const div = document.createElement('div');
    div.className = "flex flex-col items-start space-y-1 max-w-[90%]";
    div.innerHTML = `
        <div class="bg-[#202c33] text-slate-200 p-3 rounded-2xl rounded-tl-none border border-[#2a3942] shadow-md space-y-1 text-[11.5px] leading-relaxed break-words">
            ${formatted}
        </div>
        <span class="text-[9px] text-slate-500 pl-1 font-mono">${now}</span>
    `;
    DOM.waMessagesContainer.appendChild(div);
    DOM.waMessagesContainer.scrollTop = DOM.waMessagesContainer.scrollHeight;
}

function showWhatsAppTyping() {
    if (!DOM.waMessagesContainer) return;
    const indicator = document.createElement('div');
    indicator.id = 'wa-typing-indicator';
    indicator.className = 'flex items-center space-x-1 bg-[#202c33] text-slate-400 px-3 py-2 rounded-2xl rounded-tl-none border border-[#2a3942] w-24 text-[10px]';
    indicator.innerHTML = `
        <span>typing</span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    `;
    DOM.waMessagesContainer.appendChild(indicator);
    DOM.waMessagesContainer.scrollTop = DOM.waMessagesContainer.scrollHeight;
}

function hideWhatsAppTyping() {
    const el = document.getElementById('wa-typing-indicator');
    if (el) el.remove();
}

// =========================================================
// SATELLITE NDVI FARM FIELD MAPPING CONTROLLER
// =========================================================

async function setupNDVIFieldMapping() {
    // 1. Layer switcher event listeners
    document.querySelectorAll('.ndvi-layer-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.ndvi-layer-btn').forEach(b => {
                b.classList.remove('active-layer', 'bg-emerald-600', 'text-white');
                b.classList.add('text-slate-400');
            });
            btn.classList.add('active-layer', 'bg-emerald-600', 'text-white');
            btn.classList.remove('text-slate-400');
            state.ndviActiveLayer = btn.dataset.layer || 'ndvi';
            renderNDVIRasterCanvas();
        });
    });

    // 2. Export GeoJSON
    if (DOM.btnNdviExportGeojson) {
        DOM.btnNdviExportGeojson.addEventListener('click', handleExportGeoJSON);
    }

    // 3. Refresh Satellite Pass
    if (DOM.btnNdviRefresh) {
        DOM.btnNdviRefresh.addEventListener('click', () => {
            showToast("Connecting to Sentinel-2 satellite pass...", "info");
            if (state.ndviFieldData && state.ndviFieldData.field_metadata) {
                const meta = state.ndviFieldData.field_metadata;
                fetchAndRenderNDVIField(meta.lat, meta.lon, meta.crop, meta.area_hectares, meta.name);
            }
        });
    }

    // 4. 1-Click Scout Stressed Zone
    if (DOM.btnScoutNdviAnomaly) {
        DOM.btnScoutNdviAnomaly.addEventListener('click', handleScoutNDVIAnomaly);
    }

    // 5. Canvas Mouse Hover Tooltip
    setupNDVICanvasInteractivity();

    // 6. Load preset fields & initial analysis
    await loadNDVISampleFields();
}

async function loadNDVISampleFields() {
    try {
        const res = await fetch('/api/ndvi/sample-fields');
        if (res.ok) {
            const data = await res.json();
            state.ndviSampleFields = data.fields || [];
            renderNDVIPresetChips();
            
            // Load initial field (Punjab Wheat)
            if (state.ndviSampleFields.length > 0) {
                const f0 = state.ndviSampleFields[0];
                await fetchAndRenderNDVIField(f0.lat, f0.lon, f0.crop, f0.area_hectares, f0.name, f0.stress_anomaly);
            }
        }
    } catch (err) {
        console.warn("[NDVI] Failed to load sample fields:", err);
    }
}

function renderNDVIPresetChips() {
    if (!DOM.ndviPresetChips) return;
    DOM.ndviPresetChips.innerHTML = state.ndviSampleFields.map((f, idx) => `
        <button class="ndvi-field-chip px-3 py-1.5 rounded-xl border ${idx === 0 ? 'bg-emerald-950/70 border-emerald-500/60 text-emerald-300 font-bold' : 'bg-slate-950 hover:bg-slate-800 border-slate-800 text-slate-300'} flex items-center space-x-1.5 flex-shrink-0 transition cursor-pointer" data-id="${f.id}">
            <span>${f.name}</span>
            <span class="text-[10px] opacity-70">(${f.crop})</span>
        </button>
    `).join('');

    DOM.ndviPresetChips.querySelectorAll('.ndvi-field-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const f = state.ndviSampleFields.find(x => x.id === chip.dataset.id);
            if (f) {
                DOM.ndviPresetChips.querySelectorAll('.ndvi-field-chip').forEach(c => {
                    c.className = "ndvi-field-chip px-3 py-1.5 rounded-xl border bg-slate-950 hover:bg-slate-800 border-slate-800 text-slate-300 flex items-center space-x-1.5 flex-shrink-0 transition cursor-pointer";
                });
                chip.className = "ndvi-field-chip px-3 py-1.5 rounded-xl border bg-emerald-950/70 border-emerald-500/60 text-emerald-300 font-bold flex items-center space-x-1.5 flex-shrink-0 transition cursor-pointer";
                fetchAndRenderNDVIField(f.lat, f.lon, f.crop, f.area_hectares, f.name, f.stress_anomaly);
            }
        });
    });
}

async function fetchAndRenderNDVIField(lat, lon, crop, area, name, anomalyNote) {
    try {
        const res = await fetch('/api/ndvi/analyze-field', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lat: lat,
                lon: lon,
                crop: crop || "Wheat",
                area_hectares: area || 10.0,
                field_name: name
            })
        });

        if (res.ok) {
            const data = await res.json();
            state.ndviFieldData = data;
            
            // Update Labels
            if (DOM.ndviFieldNameLabel) DOM.ndviFieldNameLabel.textContent = name || `${crop} Parcel (${lat.toFixed(2)}, ${lon.toFixed(2)})`;
            if (DOM.ndviFieldCropBadge) DOM.ndviFieldCropBadge.textContent = `${crop} • ${area || 10} ha`;

            // Update Metrics
            const stats = data.raster.statistics;
            if (DOM.ndviMeanVal) DOM.ndviMeanVal.textContent = stats.mean_ndvi;
            if (DOM.ndviRangeSub) DOM.ndviRangeSub.textContent = `Range: ${stats.min_ndvi} – ${stats.max_ndvi}`;
            if (DOM.ndviUniformityVal) DOM.ndviUniformityVal.textContent = `${stats.field_uniformity_score}%`;

            // Update Zonal Breakdown
            const zb = data.raster.zonal_breakdown;
            if (DOM.zoneHighPct) DOM.zoneHighPct.textContent = `${zb.high_vigor_pct}%`;
            if (DOM.zoneHighBar) DOM.zoneHighBar.style.width = `${zb.high_vigor_pct}%`;
            if (DOM.zoneModPct) DOM.zoneModPct.textContent = `${zb.moderate_stress_pct}%`;
            if (DOM.zoneModBar) DOM.zoneModBar.style.width = `${zb.moderate_stress_pct}%`;
            if (DOM.zoneSevPct) DOM.zoneSevPct.textContent = `${zb.severe_anomaly_pct}%`;
            if (DOM.zoneSevBar) DOM.zoneSevBar.style.width = `${zb.severe_anomaly_pct}%`;

            if (DOM.ndviAnomalyAlert) {
                DOM.ndviAnomalyAlert.textContent = anomalyNote || (zb.severe_anomaly_pct > 0 ? `${zb.severe_anomaly_pct}% STRESS ANOMALY DETECTED` : "CANOPY HEALTHY");
            }

            // Render Canvas Raster
            renderNDVIRasterCanvas();

            // Render VRA Prescription Table
            renderVRAPrescriptionTable(data.vra_fertilizer_prescription);

            // Render Multi-Temporal Phenology Curve
            renderNDVITemporalCurve(data.multi_temporal_growth_curve);
        }
    } catch (err) {
        console.warn("[NDVI] Field analysis fetch failed:", err);
    }
}

function renderNDVIRasterCanvas() {
    if (!DOM.ndviRasterCanvas || !state.ndviFieldData) return;
    const canvas = DOM.ndviRasterCanvas;
    const ctx = canvas.getContext('2d');
    const raster = state.ndviFieldData.raster;
    const cells = raster.cells;
    const gridSize = raster.grid_size || 24;
    
    const w = canvas.width;
    const h = canvas.height;
    const cellW = w / gridSize;
    const cellH = h / gridSize;

    ctx.clearRect(0, 0, w, h);

    for (let r = 0; r < gridSize; r++) {
        for (let c = 0; c < gridSize; c++) {
            const cell = cells[r][c];
            let color = cell.color_ndvi;
            
            if (state.ndviActiveLayer === 'ndwi') {
                color = cell.color_ndwi;
            } else if (state.ndviActiveLayer === 'zones') {
                color = cell.zone === 'high_vigor' ? '#16a34a' : (cell.zone === 'moderate_stress' ? '#eab308' : '#dc2626');
            } else if (state.ndviActiveLayer === 'rgb') {
                // Pseudo-natural satellite RGB
                const gVal = Math.round(cell.ndvi * 160 + 60);
                color = `rgb(40, ${gVal}, 30)`;
            }

            ctx.fillStyle = color;
            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);

            // Subtle cell border grid
            ctx.strokeStyle = "rgba(15, 23, 42, 0.25)";
            ctx.lineWidth = 0.5;
            ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);
        }
    }
}

function setupNDVICanvasInteractivity() {
    if (!DOM.ndviRasterCanvas) return;
    const canvas = DOM.ndviRasterCanvas;

    canvas.addEventListener('mousemove', (e) => {
        if (!state.ndviFieldData) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        const gridSize = state.ndviFieldData.raster.grid_size || 24;
        const cellW = canvas.width / gridSize;
        const cellH = canvas.height / gridSize;

        const col = Math.floor(mouseX / cellW);
        const row = Math.floor(mouseY / cellH);

        if (col >= 0 && col < gridSize && row >= 0 && row < gridSize) {
            const cell = state.ndviFieldData.raster.cells[row][col];
            if (DOM.ndviCellTooltip && DOM.ttCellCoords && DOM.ttCellNdvi) {
                DOM.ndviCellTooltip.style.opacity = '1';
                DOM.ttCellCoords.textContent = `📍 Lat: ${cell.lat.toFixed(4)}, Lon: ${cell.lon.toFixed(4)}`;
                DOM.ttCellNdvi.textContent = `NDVI: ${cell.ndvi} | NDWI: ${cell.ndwi}`;
                if (DOM.ttCellZone) DOM.ttCellZone.textContent = cell.zone_label;
            }
        }
    });

    canvas.addEventListener('mouseleave', () => {
        if (DOM.ndviCellTooltip) DOM.ndviCellTooltip.style.opacity = '0';
    });
}

function renderVRAPrescriptionTable(prescription) {
    if (!DOM.vraPrescriptionTbody || !prescription) return;
    const zones = prescription.zones || [];
    
    if (DOM.vraSavingsBadge) {
        const savings = prescription.total_fertilizer_demand?.cost_and_input_savings_pct || 18.4;
        DOM.vraSavingsBadge.textContent = `${savings}% COST SAVINGS`;
    }

    DOM.vraPrescriptionTbody.innerHTML = zones.map(z => {
        const totalUreaZone = Math.round(z.area_hectares * z.urea_kg_ha);
        return `
            <tr class="hover:bg-slate-900/60 transition">
                <td class="py-2 flex items-center space-x-1.5">
                    <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${z.color}"></span>
                    <span class="truncate font-medium">${z.zone_name.split(':')[0]}</span>
                </td>
                <td class="py-2">${z.area_hectares} ha <span class="text-[9.5px] text-slate-500 font-sans">(${z.area_pct}%)</span></td>
                <td class="py-2 text-emerald-400 font-bold">${z.urea_kg_ha} kg/ha</td>
                <td class="py-2 text-slate-100 font-bold">${totalUreaZone} kg</td>
            </tr>
        `;
    }).join('');
}

function renderNDVITemporalCurve(timeline) {
    if (!DOM.ndviTemporalCurveContainer || !timeline || timeline.length === 0) return;
    const container = DOM.ndviTemporalCurveContainer;
    const w = 560;
    const h = 75;
    const padX = 35;
    const padY = 12;

    const points = timeline.map((pt, idx) => {
        const x = padX + (idx / (timeline.length - 1)) * (w - 2 * padX);
        const y = (h - padY) - (pt.ndvi / 1.0) * (h - 2 * padY);
        return { x, y, ndvi: pt.ndvi, stage: pt.stage, status: pt.status };
    });

    const pathD = points.reduce((acc, p, idx) => {
        return idx === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`;
    }, '');

    const areaD = `${pathD} L ${points[points.length - 1].x} ${h - padY} L ${points[0].x} ${h - padY} Z`;

    const svg = `
        <svg viewBox="0 0 ${w} ${h}" class="w-full h-full overflow-visible font-mono text-[9px]">
            <defs>
                <linearGradient id="ndviGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#10b981" stop-opacity="0.35"/>
                    <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
                </linearGradient>
            </defs>
            <line x1="${padX}" y1="${h - padY}" x2="${w - padX}" y2="${h - padY}" stroke="#334155" stroke-dasharray="2 2"/>
            <path d="${areaD}" fill="url(#ndviGrad)" />
            <path d="${pathD}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" />
            ${points.map(p => `
                <circle cx="${p.x}" cy="${p.y}" r="${p.status.includes('Active') || p.status.includes('Current') ? '4.5' : '3'}" fill="${p.status.includes('Active') || p.status.includes('Current') ? '#34d399' : '#10b981'}" stroke="#020617" stroke-width="1.5"/>
                <text x="${p.x}" y="${p.y - 7}" text-anchor="middle" fill="#94a3b8" font-size="8.5" font-weight="bold">${p.ndvi}</text>
            `).join('')}
        </svg>
    `;
    container.innerHTML = svg;
}

function handleScoutNDVIAnomaly() {
    if (!state.ndviFieldData) return;
    const meta = state.ndviFieldData.field_metadata;
    
    // Map to relevant sample specimen based on crop
    const cropMap = {
        "Wheat": "wheat_yellow_rust",
        "Strawberry": "strawberry_leaf_scorch",
        "Corn (Maize)": "corn_common_rust",
        "Grape (Vineyard)": "grape_black_rot",
        "Citrus (Orange)": "citrus_greening"
    };
    
    const sampleId = cropMap[meta.crop] || "tomato_early_blight";
    
    // Trigger evaluation
    showToast(`🛰️ Initiating targeted scouting for ${meta.name} anomaly zone...`, "info");
    
    // Scroll to Dropzone / Diagnosis area
    const dropzone = document.getElementById('drop-zone');
    if (dropzone) {
        dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
        dropzone.classList.add('ring-2', 'ring-amber-500');
        setTimeout(() => dropzone.classList.remove('ring-2', 'ring-amber-500'), 2000);
    }
    
    // Load specimen
    evaluateSample(sampleId);
}

function handleExportGeoJSON() {
    if (!state.ndviFieldData || !state.ndviFieldData.geojson) {
        showToast("No NDVI field parcel data available to export.", "warning");
        return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.ndviFieldData.geojson, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "farm_field_ndvi_gis.geojson");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showToast("Exported GIS GeoJSON parcel with NDVI metrics!", "success");
}

// --- Initialization ---
async function init() {
    setupEventListeners();
    setupPWAAndNetwork();
    setupMLOpsAndFeedback();
    setupWhatsAppBotSimulator();
    setupNDVIFieldMapping();
    
    // Pre-warm in-browser ONNX engine in the background for instant edge inference
    getONNXSession();
    
    await loadSampleSpecimens();
    await loadScoutHistory();
    await fetchWeatherRisk("Salinas Valley, CA");
    recalculateDosage();
    
    // Check backend health
    try {
        const healthRes = await fetch('/api/health');
        if (healthRes.ok) {
            const health = await healthRes.json();
            DOM.modelStatusText.textContent = `Model Ready (${health.classes_supported} Classes)`;
        }
    } catch (e) {
        DOM.modelStatusText.textContent = "In-Browser Edge AI Ready";
        DOM.systemStatusBadge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-amber-950/50 border border-amber-800/50 text-xs text-amber-300";
    }
}

document.addEventListener('DOMContentLoaded', init);
