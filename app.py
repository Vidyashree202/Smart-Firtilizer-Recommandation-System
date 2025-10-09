import os
import warnings
from flask import Flask, render_template, request, jsonify, redirect
import pickle
import pandas as pd
import numpy as np
try:
    import requests as _requests
except Exception:
    _requests = None

# Point this lightweight app to the actual assets inside the subfolder
app = Flask(
    __name__,
    template_folder="Fertilizer_Recommendation_System-main/templates",
    static_folder="Fertilizer_Recommendation_System-main/static",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBAPP_DIR = os.path.join(BASE_DIR, "Fertilizer_Recommendation_System-main")

# Load model and encoders from the subapp directory if available; fallback to root
def _load_pickle(filename: str):
    paths = [
        os.path.join(SUBAPP_DIR, filename),
        os.path.join(BASE_DIR, filename),
    ]
    for p in paths:
        if os.path.exists(p):
            return pickle.load(open(p, 'rb'))
    raise FileNotFoundError(filename)

model = _load_pickle('fertilizer_model.pkl')
encoders = _load_pickle('label_encoders.pkl')
ferti = encoders['Fertilizer']

# Static fertilizer details used to enrich predictions
FERTILIZER_INFO = {
    'Urea': {
        'name': 'Urea',
        'type': 'Nitrogen fertilizer (46% N)',
        'description': 'Fast-acting nitrogen source that promotes vegetative growth and deep green color.',
        'dosage': '40–80 kg/acre per application (split doses). Adjust to soil test.',
        'application': 'Broadcast and incorporate into moist soil; avoid direct contact with seed. Prefer split application and apply before irrigation/rain to reduce volatilization.',
        'notes': 'Best applied when pH is not highly alkaline; losses increase on hot, dry, high-pH soils.'
    },
    'DAP': {
        'name': 'DAP',
        'type': 'Diammonium phosphate (18-46-0)',
        'description': 'Provides both nitrogen and high phosphorus for strong root development and early growth.',
        'dosage': '25–50 kg/acre at sowing or early growth. Adjust to soil test.',
        'application': 'Place near the seed/roots (band placement) at sowing or early stage. Avoid seed contact.',
        'notes': 'Useful for P-deficient soils; complement later with N or K if required.'
    },
    'MOP': {
        'name': 'MOP',
        'type': 'Muriate of potash (0-0-60)',
        'description': 'High potassium source to improve drought tolerance, disease resistance, and grain/fruit quality.',
        'dosage': '20–40 kg/acre depending on K deficiency and crop. Adjust to soil test.',
        'application': 'Broadcast and incorporate or band apply near root zone. Apply before critical K-demand stages.',
        'notes': 'For chloride-sensitive crops consider SOP; otherwise MOP is widely used.'
    },
    '10-26-26': {
        'name': '10-26-26',
        'type': 'NPK complex',
        'description': 'Balanced P and K blend with some N; supports rooting, flowering, and fruiting.',
        'dosage': '40–60 kg/acre as basal; top up N later as needed.',
        'application': 'Apply as basal dose at sowing/transplanting; band near root zone.',
        'notes': 'Pair with additional N fertilizer during vegetative growth if crop requires.'
    },
    '14-35-14': {
        'name': '14-35-14',
        'type': 'NPK complex',
        'description': 'High phosphorus blend to boost early root and shoot development.',
        'dosage': '30–50 kg/acre as basal depending on soil P status.',
        'application': 'Basal band placement at sowing/transplanting for efficient P use.',
        'notes': 'Monitor P levels; avoid overuse in high-P soils.'
    },
    '17-17-17': {
        'name': '17-17-17',
        'type': 'NPK complex (balanced)',
        'description': 'Balanced NPK for general growth and maintenance across stages.',
        'dosage': '35–55 kg/acre split into 1–2 applications.',
        'application': 'Basal plus early top-dress; band or broadcast and incorporate.',
        'notes': 'Supplement with micronutrients as per soil test.'
    },
    '20-20': {
        'name': '20-20',
        'type': 'NPK mixture (approx. 20-20-0/20-20-0+trace)',
        'description': 'Balanced N and P for vegetative growth and root support; may need K separately.',
        'dosage': '30–50 kg/acre split. Add K separately if crop needs it.',
        'application': 'Basal and early top-dress; avoid seed contact.',
        'notes': 'Confirm exact product composition; add MOP if K is low.'
    },
    '28-28': {
        'name': '28-28',
        'type': 'High N and P blend',
        'description': 'High analysis N and P for rapid vegetative growth and strong early establishment.',
        'dosage': '25–45 kg/acre, typically split with later K addition if required.',
        'application': 'Basal band placement; top-up N later based on crop stage.',
        'notes': 'Check K status; supplement with MOP if needed.'
    },
    'NPK': {
        'name': 'NPK',
        'type': 'Balanced NPK fertilizer (various grades)',
        'description': 'Provides a balanced supply of Nitrogen (N), Phosphorus (P), and Potassium (K) for overall plant growth.',
        'dosage': '30–60 kg/acre as basal or split, depending on soil test and crop stage.',
        'application': 'Apply as basal near root zone (band) during sowing/transplanting; top-dress in splits as per crop schedule.',
        'notes': 'Choose the specific grade (e.g., 17-17-17 or 10-26-26) based on soil test and crop needs; supplement micronutrients if required.'
    },
    'Compost': {
        'name': 'Compost',
        'type': 'Organic soil conditioner',
        'description': 'Improves soil structure, water-holding, and microbial activity; supplies slow-release nutrients.',
        'dosage': '400–800 kg/acre or 1–2 tons/acre depending on soil organic matter.',
        'application': 'Broadcast and incorporate into topsoil before planting; can be used as mulch around plants.',
        'notes': 'Maturity and quality of compost affect results; avoid undecomposed materials touching stems.'
    },
    'FYM': {
        'name': 'FYM',
        'type': 'Farmyard manure (organic)',
        'description': 'Adds organic matter and a broad spectrum of nutrients; improves soil tilth.',
        'dosage': '1–3 tons/acre based on soil test and crop.',
        'application': 'Apply and incorporate 2–3 weeks before sowing/transplanting to allow mineralization.',
        'notes': 'Use well-decomposed manure to avoid pests and nutrient tie-up.'
    },
    'Vermicompost': {
        'name': 'Vermicompost',
        'type': 'Organic biofertilizer from earthworms',
        'description': 'Rich in humus and beneficial microbes; improves nutrient availability and root growth.',
        'dosage': '200–400 kg/acre as basal; 1–2 kg per plant for horticultural crops.',
        'application': 'Place near root zone and cover with soil; can be mixed with compost.',
        'notes': 'Keep moist, not waterlogged; store in shade to preserve microbes.'
    },
}

# Kannada static details to ensure offline/full-Kannada display when LLM is unavailable
KANNADA_FERT_INFO = {
    'Urea': {
        'type': 'ಸಾರಜನಕ ಗೊಬ್ಬರ (46% N)',
        'description': 'ಬೆಳೆಗಳ ಶಾಖೀಯ ಬೆಳವಣಿಗೆಗೆ ತ್ವರಿತವಾಗಿ ಪರಿಣಾಮ ನೀಡುವ ಸಾರಜನಕ ಮೂಲ; ಎಲೆಗಳಿಗೆ ಗಾಢ ಹಸಿರು ಬಣ್ಣ ಬರುತ್ತದೆ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 40–80 ಕೆಜಿ (ಭಾಗಗಳಾಗಿ ಕೊಡಿ). ಮಣ್ಣು ಪರೀಕ್ಷೆಯ ಪ್ರಕಾರ ಸರಿಪಡಿಸಿ.',
        'application': 'ಮಣ್ಣಿನ ಮೇಲೆ ಚೆಲ್ಲಿ ತೇವ ಮಣ್ಣಿನಲ್ಲಿ ಮಿಶ್ರಗೊಳಿಸಿ; ಬೀಜದ ನೇರ ಸಂಪರ್ಕ ತಪ್ಪಿಸಿ. ನೀರಾವರಿಗೆ ಮುಂಚೆ/ಮಳೆಗೆ ಮುಂಚೆ ಭಾಗಗಳಾಗಿ ನೀಡುವುದು ಉತ್ತಮ.',
        'notes': 'ಅತಿ ಆಲ್ಕಲೈನ್ pH ಹೊಂದಿದ ಮಣ್ಣಿನಲ್ಲಿ ನಷ್ಟಗಳು ಹೆಚ್ಚಾಗಬಹುದು; ಬಿಸಿಲು/ಒಣ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ವಾತನಾಶಕ ನಷ್ಟ ತಪ್ಪಿಸಲು ಮಿಶ್ರಗೊಳಿಸಿ.'
    },
    'DAP': {
        'type': 'ಡೈಅಮೋನಿಯಮ್ ಫಾಸ್ಫೇಟ್ (18-46-0)',
        'description': 'ಸಾರಜನಕ ಮತ್ತು ಹೆಚ್ಚು ಫಾಸ್ಫರಸ್ ಒದಗಿಸಿ ಬೆಳ್ಳೆಣ್ಣೆಯ ಬೇರು ಮತ್ತು ಆರಂಭಿಕ ಬೆಳವಣಿಗೆಗೆ ಸಹಾಯಕ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 25–50 ಕೆಜಿ (ಬಿತ್ತನೆ/ಆರಂಭದಲ್ಲಿ). ಮಣ್ಣು ಪರೀಕ್ಷೆಯ ಪ್ರಕಾರ.',
        'application': 'ಬೀಜ/ಬೇರುವಿನ ಪಕ್ಕದಲ್ಲಿ ಬ್ಯಾಂಡ್ ವಿಧಾನದಲ್ಲಿ ಇಡಿ; ಬೀಜಕ್ಕೆ ನೇರ ಸ್ಪರ್ಶ ತಪ್ಪಿಸಿ.',
        'notes': 'P ಕೊರತೆಯ ಮಣ್ಣಿನಲ್ಲಿ ಉಪಯುಕ್ತ; ನಂತರ ಅಗತ್ಯವಿದ್ದರೆ N ಅಥವಾ K ಪೂರಕ ನೀಡಿ.'
    },
    'MOP': {
        'type': 'ಮ್ಯೂರಿಯೇಟ್ ಆಫ್ ಪೋಟ್ಯಾಶ್ (0-0-60)',
        'description': 'ಉನ್ನತ K ಮೂಲ; ಒಣಹೆಗೆ ತಡೆ, ರೋಗನಿರೋಧಕತೆ ಮತ್ತು ಧಾನ್ಯ/ಹಣ್ಣಿನ ಗುಣಮಟ್ಟ ಹೆಚ್ಚಿಸಲು ಸಹಾಯಕ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 20–40 ಕೆಜಿ (ಮಣ್ಣಿನ K ಕೊರತೆಯ ಮೇಲೆ ಅವಲಂಬಿತ).',
        'application': 'ಬ್ಯಾಂಡ್ ಅಥವಾ ಪ್ರಸರಣವಾಗಿ ನೀಡಿ, ಮಣ್ಣಿನಲ್ಲಿ ಹೊಂದಿಸಿ; K ಅಗತ್ಯ ಹಂತಗಳ ಮುಂಚೆ ಅನ್ವಯಿಸಿ.',
        'notes': 'ಕ್ಲೋರೈಡ್-ಸಂವೇದನಶೀಲ ಬೆಳೆಗಳಿಗೆ SOP ಪರಿಗಣಿಸಿ; ಬೇರೆಡೆ MOP ಸಾಮಾನ್ಯ.'
    },
    '10-26-26': {
        'type': 'ಎನ್‌ಪಿಕೆ ಸಂಕೀರ್ಣ',
        'description': 'P ಮತ್ತು K ಸಮತೋಲನ ಹೊಂದಿದ ಮಿಶ್ರಣ; ಬೇರು, ಹೂವು ಮತ್ತು ಫಲದ ಹಂತಗಳಿಗೆ ಸಹಕಾರಿ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 40–60 ಕೆಜಿ ಬಸಲ್; ನಂತರ N ಪೂರಕ ನೀಡಬಹುದು.',
        'application': 'ಬಿತ್ತನೆ/ನಗರಿಕೆಯಲ್ಲಿ ಬೇರುವಿನ ಪಕ್ಕ ಬ್ಯಾಂಡ್ ಮಾಡಿ.',
        'notes': 'ಬೆಳೆ ಅವಶ್ಯಕತೆಗೆ ಅನುಗುಣವಾಗಿ ನಂತರ N ಹೆಚ್ಚಿಸಿ.'
    },
    '14-35-14': {
        'type': 'ಎನ್‌ಪಿಕೆ ಸಂಕೀರ್ಣ (ಹೆಚ್ಚು P)',
        'description': 'ಹೆಚ್ಚು ಫಾಸ್ಫರಸ್ ಮಿಶ್ರಣ; ಆರಂಭಿಕ ಬೇರು ಮತ್ತು ಕಾಂಡ ಬೆಳವಣಿಗೆಗೆ ಉತ್ತೇಜನ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 30–50 ಕೆಜಿ ಬಸಲ್ (ಮಣ್ಣಿನ P ಮಟ್ಟದ ಆಧಾರ).',
        'application': 'ಬಿತ್ತನೆ/ನಗರಿಕೆಯಲ್ಲಿ ಬ್ಯಾಂಡ್ ವಿಧಾನ.',
        'notes': 'P ಅಧಿಕ ಇರುವ ಮಣ್ಣಿನಲ್ಲಿ ಮಿತಿ ಮೀರಿ ಬಳಸಬೇಡಿ.'
    },
    '17-17-17': {
        'type': 'ಎನ್‌ಪಿಕೆ ಸಮತೋಲನ (ಬ್ಯಾಲೆನ್ಸ್)',
        'description': 'ಸಾಮಾನ್ಯ ಬೆಳವಣಿಗೆಗೆ ತಕ್ಕ ಸಮತೋಲನ NPK.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 35–55 ಕೆಜಿ, 1–2 ಭಾಗಗಳಲ್ಲಿ.',
        'application': 'ಬಸಲ್ ಹಾಗೂ ಆರಂಭಿಕ ಟಾಪ್-ಡ್ರೆಸ್; ಬ್ಯಾಂಡ್/ಪ್ರಸರಣವಾಗಿ ನೀಡಿ ಮತ್ತು ಮಿಶ್ರಗೊಳಿಸಿ.',
        'notes': 'ಮಣ್ಣು ಪರೀಕ್ಷೆಯ ಪ್ರಕಾರ ಸೂಕ್ಷ್ಮಪೋಷಕಗಳನ್ನು ಸೇರಿಸಿ.'
    },
    '20-20': {
        'type': 'N ಮತ್ತು P ಸಮಬಾಳಿತ ಮಿಶ್ರಣ',
        'description': 'ಸಾರಜನಕ ಮತ್ತು ಫಾಸ್ಫರಸ್ ಒದಗಿಸಿ ಶಾಖೀಯ ಬೆಳವಣಿಗೆ ಮತ್ತು ಬೇರು ಬೆಂಬಲಿಸುತ್ತವೆ; K ಬೇರೆಡೆ ಅಗತ್ಯವಾಗಬಹುದು.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 30–50 ಕೆಜಿ ಭಾಗಗಳಲ್ಲಿ; K ಕಡಿಮೆಯಿದ್ದರೆ ಪ್ರತ್ಯೇಕವಾಗಿ MOP ನೀಡಿ.',
        'application': 'ಬಸಲ್ ಹಾಗೂ ಆರಂಭಿಕ ಟಾಪ್-ಡ್ರೆಸ್; ಬೀಜ ಸ್ಪರ್ಶ ತಪ್ಪಿಸಿ.',
        'notes': 'ಉತ್ಪನ್ನದ ನಿಖರ ಸಂಯೋಜನೆ ಪರಿಶೀಲಿಸಿ.'
    },
    '28-28': {
        'type': 'ಹೆಚ್ಚು N ಮತ್ತು P ಮಿಶ್ರಣ',
        'description': 'ಆರಂಭಿಕ ಸ್ಥಾಪನೆ ಮತ್ತು ವೇಗದ ಶಾಖೀಯ ಬೆಳವಣಿಗೆಗೆ ಸಹಾಯಕ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 25–45 ಕೆಜಿ; ನಂತರ ಅಗತ್ಯವಿದ್ದರೆ K ಸೇರಿಸಿ.',
        'application': 'ಬಸಲ್ ಬ್ಯಾಂಡ್; ಹಂತಾನುಸಾರ N ಟಾಪ್-ಡ್ರೆಸ್.',
        'notes': 'K ಮಟ್ಟಗಳನ್ನು ಗಮನಿಸಿ; ಅಗತ್ಯವಿದ್ದರೆ MOP ಪೂರಕ.'
    },
    'NPK': {
        'type': 'ಸಮತೋಲನ ಎನ್‌ಪಿಕೆ ಗೊಬ್ಬರ (ವಿವಿಧ ಗ್ರೇಡ್‌ಗಳು)',
        'description': 'ಎನ್, ಪಿ, ಕೆ ಪೋಷಕಾಂಶಗಳನ್ನು ಸಮಬಾಳಿತವಾಗಿ ಒದಗಿಸಿ ಒಟ್ಟಾರೆ ಬೆಳವಣಿಗೆಗೆ ಸಹಾಯ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 30–60 ಕೆಜಿ (ಬಸಲ್/ಭಾಗಗಳಲ್ಲಿ), ಮಣ್ಣು ಪರೀಕ್ಷೆಯ ಮೇರೆಗೆ.',
        'application': 'ಬಿತ್ತನೆ/ನಗರಿಕೆಯಲ್ಲಿ ಬೇರುವಿನ ಪಕ್ಕ ಬ್ಯಾಂಡ್ ಮಾಡಿ; ಬೆಳೆ ವೇಳಾಪಟ್ಟಿಗೆ ಅನುಗುಣವಾಗಿ ಭಾಗಗಳಲ್ಲಿ ನೀಡಿರಿ.',
        'notes': 'ಮಣ್ಣಿನ ಸ್ಥಿತಿ ಹಾಗೂ ಬೆಳೆ ಅವಶ್ಯಕತೆಗೆ ತಕ್ಕ ಗ್ರೇಡ್ (ಉದಾ., 17-17-17, 10-26-26) ಆಯ್ಕೆ ಮಾಡಿ; ಸೂಕ್ಷ್ಮಪೋಷಕಗಳನ್ನು ಅಗತ್ಯವಿದ್ದರೆ ಸೇರಿಸಿ.'
    },
    'Compost': {
        'type': 'ಜೈವಿಕ ಮಣ್ಣಿನ ಸಂಡಣಿ',
        'description': 'ಮಣ್ಣಿನ ರಚನೆ, ನೀರಿನ ಹಿಡಿತ ಮತ್ತು ಜೀವಾಣು ಚಟುವಟಿಕೆಯನ್ನು ಹೆಚ್ಚಿಸಿ; ಪೋಷಕಾಂಶಗಳನ್ನು ನಿಧಾನವಾಗಿ ಒದಗಿಸುತ್ತದೆ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 400–800 ಕೆಜಿ ಅಥವಾ 1–2 ಟನ್ (ಮಣ್ಣಿನ ಕಾರ್ಬನ್ ಮಟ್ಟದ ಆಧಾರ).',
        'application': 'ಬಿತ್ತನೆಯ ಮೊದಲು ಮೇಲ್ಮಣ್ಣಿನಲ್ಲಿ ಚೆಲ್ಲಿ ಮಿಶ್ರಗೊಳಿಸಿ; ಗಿಡಗಳ ಸುತ್ತ ಮಲ್ಚ್‌ ಆಗಿ ಬಳಸಬಹುದು.',
        'notes': 'ಪೂರ್ಣವಾಗಿ ಕುಟ್ಟಿದ (ಮ್ಯಾಚ್ಯೂರ್) ಕಂಪೋಸ್ಟ್ ಬಳಸಿರಿ; ಅಸಂಪೂರ್ಣ ಪದಾರ್ಥಗಳನ್ನು ಕಾಂಡಕ್ಕೆ ತಾಕದಂತೆ ನೋಡಿಕೊಳ್ಳಿ.'
    },
    'FYM': {
        'type': 'ಫಾರ್ಮ್ಯಾರ್ಡ್ ಮ್ಯಾನ್ಯುರ್ (ಜೈವಿಕ)',
        'description': 'ಜೈವಿಕ ಪದಾರ್ಥ ಮತ್ತು ವಿವಿಧ ಪೋಷಕಾಂಶಗಳನ್ನು ಸೇರಿಸಿ ಮಣ್ಣಿನ ಸಾಂದ್ರತೆಯನ್ನು ಸುಧಾರಿಸುತ್ತದೆ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 1–3 ಟನ್ (ಮಣ್ಣು/ಬೆಳೆ ಆಧಾರ).',
        'application': 'ಬಿತ್ತನೆ/ನಗರಿಕೆಗೆ 2–3 ವಾರಗಳ ಮುಂಚೆ ಹಾಸಿ ಮಿಶ್ರಗೊಳಿಸಿ, ಖನಿಜೀಕರಣಕ್ಕೆ ಸಮಯ ನೀಡಿ.',
        'notes': 'ಚನ್ನಾಗಿ ಕುಟ್ಟಿದ (ವೆಲ್-ಡೀಕಂಪೋಸ್‌ಡ್) ಮ್ಯಾನ್ಯುರ್ ಬಳಸಿ; ಕೀಟ/ಪೋಷಕಾಂಶ ತಡೆ ತಪ್ಪಿಸಲು.'
    },
    'Vermicompost': {
        'type': 'ಹುಳದ ಮೂಲಕ ತಯಾರಾದ ಜೈವಿಕ ಗೊಬ್ಬರ',
        'description': 'ಹ್ಯೂಮಸ್ ಮತ್ತು ಹಿತಕಾರಿ ಜೀವಾಣುಗಳಲ್ಲಿ ಸಮೃದ್ಧ; ಪೋಷಕಾಂಶ ಲಭ್ಯತೆ ಮತ್ತು ಬೇರು ಬೆಳವಣಿಗೆಯನ್ನು ಉತ್ತೇಜಿಸುತ್ತದೆ.',
        'dosage': 'ಪ್ರತಿ ಎಕರೆ 200–400 ಕೆಜಿ; ತೋಟಗಾರಿಕೆಗೆ ಪ್ರತಿ ಗಿಡಕ್ಕೆ 1–2 ಕೆಜಿ.',
        'application': 'ಬೇರುವಿನ ಪಕ್ಕದಲ್ಲಿ ಇಟ್ಟು ಮಣ್ಣಿನಿಂದ ಮುಚ್ಚಿರಿ; ಕಂಪೋಸ್ಟ್‌ಗೆ ಮಿಶ್ರಣ ಮಾಡಬಹುದು.',
        'notes': 'ತೇವವಾಗಿರಲಿ; ನೀರಿನಿಂದ ತುಂಬಿಸಬೇಡಿ; ನೆರಳಿನಲ್ಲಿ ಸಂಗ್ರಹಿಸಿ ಜೀವಾಣುಗಳನ್ನು ಉಳಿಸಿ.'
    },
}

def _normalize_fert_name(name: str) -> str:
    """Normalize predicted fertilizer name for lookup."""
    if not name:
        return ''
    s = str(name).strip()
    # Remove surrounding quotes/brackets e.g., "['Urea']" -> Urea
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.strip().strip("'\"")
    # If comma-separated, take first
    if ',' in s:
        s = s.split(',')[0].strip()
    # Common uppercase alias for matching
    return s

def _alias_to_key(name: str) -> str:
    """Resolve various aliases and cases to a canonical fertilizer key present in our maps."""
    if not name:
        return ''
    s = _normalize_fert_name(name)
    su = s.upper().replace(' ', '').replace('-', '').replace('.', '')
    # Common aliases
    alias_map = {
        'UREA': 'Urea',
        'DIAMMONIUMPHOSPHATE': 'DAP',
        'DAP': 'DAP',
        'MURIATEOFPOTASH': 'MOP',
        'POTASH': 'MOP',
        'KCL': 'MOP',
        'COMPOST': 'Compost',
        'FYM': 'FYM',
        'FARMYARDMANURE': 'FYM',
        'VERMICOMPOST': 'Vermicompost',
        'NPK': 'NPK',
        'NPK171717': '17-17-17',
        'NPK102626': '10-26-26',
        'NPK143514': '14-35-14',
        'NPK2020': '20-20',
        'NPK2828': '28-28',
    }
    if su in alias_map:
        return alias_map[su]
    # Try direct key matches in our maps (case-insensitive)
    for key in list(FERTILIZER_INFO.keys()) + list(KANNADA_FERT_INFO.keys()):
        if key.lower() == s.lower():
            return key
    # Partial match fallback
    for key in list(FERTILIZER_INFO.keys()) + list(KANNADA_FERT_INFO.keys()):
        if s.lower() in key.lower() or key.lower() in s.lower():
            return key
    return s

# Load CSV data
def _read_csv(filename: str) -> pd.DataFrame:
    for p in [os.path.join(BASE_DIR, filename), os.path.join(SUBAPP_DIR, filename)]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(filename)

# (Removed) Previously loaded soil_defaults.csv for NPK by location; no longer used

# Basic mode defaults from karnataka.csv
try:
    _f2_df = _read_csv('karnataka.csv')
    _f2_df = _f2_df.rename(columns={'Temparature': 'Temperature', 'Phosphorous': 'Phosphorus', 'PH': 'pH'})
    
    # Fill missing pH values with a default value (6.5 is neutral)
    _f2_df['pH'] = _f2_df['pH'].fillna(6.5)
    
    full_key_cols = ['Soil_Type', 'Crop_Type', 'Temperature', 'Humidity', 'Moisture']
    npk_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH']
    _full_means = (
        _f2_df[full_key_cols + npk_cols]
        .groupby(full_key_cols, dropna=False)
        .mean(numeric_only=True)
        .round()
        .reset_index()
    )
    npk_defaults_full = {
        (row['Soil_Type'], row['Crop_Type'], int(row['Temperature']), int(row['Humidity']), int(row['Moisture'])): {
            'Nitrogen': int(row['Nitrogen']),
            'Phosphorus': int(row['Phosphorus']),
            'Potassium': int(row['Potassium']),
            'pH': round(float(row['pH']), 1) if pd.notna(row['pH']) else 6.5,
        }
        for _, row in _full_means.iterrows()
    }
    sc_means = (
        _f2_df[['Soil_Type', 'Crop_Type'] + npk_cols]
        .groupby(['Soil_Type', 'Crop_Type'], dropna=False)
        .mean(numeric_only=True)
        .round()
        .reset_index()
    )
    npk_defaults_sc = {
        (row['Soil_Type'], row['Crop_Type']): {
            'Nitrogen': int(row['Nitrogen']),
            'Phosphorus': int(row['Phosphorus']),
            'Potassium': int(row['Potassium']),
            'pH': round(float(row['pH']), 1) if pd.notna(row['pH']) else 6.5,
        }
        for _, row in sc_means.iterrows()
    }
    crop_means = (
        _f2_df[['Crop_Type'] + npk_cols]
        .groupby(['Crop_Type'], dropna=False)
        .mean(numeric_only=True)
        .round()
        .reset_index()
    )
    npk_defaults_crop = {
        row['Crop_Type']: {
            'Nitrogen': int(row['Nitrogen']),
            'Phosphorus': int(row['Phosphorus']),
            'Potassium': int(row['Potassium']),
            'pH': round(float(row['pH']), 1) if pd.notna(row['pH']) else 6.5,
        }
        for _, row in crop_means.iterrows()
    }
    soil_means = (
        _f2_df[['Soil_Type'] + npk_cols]
        .groupby(['Soil_Type'], dropna=False)
        .mean(numeric_only=True)
        .round()
        .reset_index()
    )
    npk_defaults_soil = {
        row['Soil_Type']: {
            'Nitrogen': int(row['Nitrogen']),
            'Phosphorus': int(row['Phosphorus']),
            'Potassium': int(row['Potassium']),
            'pH': round(float(row['pH']), 1) if pd.notna(row['pH']) else 6.5,
        }
        for _, row in soil_means.iterrows()
    }
    overall_means = _f2_df[npk_cols].mean(numeric_only=True).round()
    npk_defaults_overall = {
        'Nitrogen': int(overall_means['Nitrogen']),
        'Phosphorus': int(overall_means['Phosphorus']),
        'Potassium': int(overall_means['Potassium']),
        'pH': round(float(overall_means['pH']), 1) if pd.notna(overall_means['pH']) else 6.5,
    }
except Exception:
    npk_defaults_full = {}
    npk_defaults_sc = {}
    npk_defaults_crop = {}
    npk_defaults_soil = {}
    npk_defaults_overall = {}


@app.route("/")
def home():
    # Landing page with links using url_for('Model1'), etc.
    return render_template("plantindex.html")


@app.route("/Model1")
def Model1():
    return render_template("Model1.html")


@app.route("/Detail")
def Detail():
    return render_template("Detail.html")


@app.route("/Advanced")
def Advanced():
    return render_template("Advanced.html")


@app.route("/assistant")
def assistant_page():
    return redirect("http://localhost:3000")



# ---------- API Endpoints (ported from sub-app) ----------

@app.route('/defaults-basic')
def defaults_basic():
    soil = request.args.get('soil')
    crop = request.args.get('crop')
    def _to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None
    temp = _to_int(request.args.get('temp'))
    humi = _to_int(request.args.get('humi'))
    mois = _to_int(request.args.get('mois'))
    if not soil or not crop:
        return jsonify({}), 400
    data = None
    if temp is not None and humi is not None and mois is not None:
        data = npk_defaults_full.get((soil, crop, temp, humi, mois))
    if not data:
        data = npk_defaults_sc.get((soil, crop))
    if not data:
        data = npk_defaults_crop.get(crop)
    if not data:
        data = npk_defaults_soil.get(soil)
    if not data:
        data = npk_defaults_overall or {}
    if not data:
        return jsonify({}), 404
    return jsonify(data)


@app.route('/predict-ajax', methods=['POST'])
def predict_ajax():
    try:
        temp = int(float(request.form.get('Temperature')))
        humi = int(float(request.form.get('Humidity')))
        mois = int(float(request.form.get('Moisture')))
        nitro = int(float(request.form.get('Nitrogen')))
        phosp = int(float(request.form.get('Phosphorus')))
        pota = int(float(request.form.get('Potassium')))
        ph = float(request.form.get('pH'))
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')
        soil_map = {'Black': 0, 'Clayey': 1, 'Loamy': 2, 'Red': 3, 'Sandy': 4}
        crop_map = {'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10,'coffee':11,'kidneybeans':12,'orange':13,'pomegranate':14,'rice':15,'watermelon':16}
        if soil_str not in soil_map or crop_str not in crop_map:
            return 'Invalid input. Unknown soil or crop type.', 400
        # Add default location (Belagavi = 0) since Location is required by the model
        location_encoded = 0  # Default to Belagavi
        features = [temp, humi, mois, soil_map[soil_str], crop_map[crop_str], nitro, pota, phosp, location_encoded, ph]
        prediction = model.predict(np.array([features]))
        res = ferti.classes_[prediction]
        return str(res[0]) if hasattr(res, '__getitem__') else str(res)
    except Exception as e:
        return f'Error: {str(e)}', 400


@app.route('/predict', methods=['POST'])
def predict():
    try:
        required = ['Temperature','Humidity','Moisture','Nitrogen','Phosphorus','Potassium','pH','Soil_Type','Crop_Type']
        if any(request.form.get(k) in (None, '') for k in required):
            return render_template('Model1.html', x='Please fill all fields in Basic mode.')
        temp = int(float(request.form.get('Temperature')))
        humi = int(float(request.form.get('Humidity')))
        mois = int(float(request.form.get('Moisture')))
        nitro = int(float(request.form.get('Nitrogen')))
        phosp = int(float(request.form.get('Phosphorus')))
        pota = int(float(request.form.get('Potassium')))
        ph = float(request.form.get('pH'))
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')
        soil_map = {'Loamy':0, 'Peaty':1, 'Acidic':2, 'Neutral':3, 'Alkaline':4, 'Clayey':5, 'Red':6, 'Black':7, 'Sandy':8}
        crop_map = {'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10,'coffee':11,'kidneybeans':12,'orange':13,'pomegranate':14,'rice':15,'watermelon':16}
        if soil_str not in soil_map or crop_str not in crop_map:
            return render_template('Model1.html', x='Invalid input. Unknown soil or crop type.')
        # Add default location (Belagavi = 0) since Location is required by the model
        location_encoded = 0  # Default to Belagavi
        features = [temp, humi, mois, soil_map[soil_str], crop_map[crop_str], nitro, pota, phosp, location_encoded, ph]
        prediction = model.predict(np.array([features]))
        res = ferti.classes_[prediction]
        # If fetch/ajax, return plain text
        if (request.headers.get('X-Requested-With') == 'fetch' or 
            request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and 
            request.headers.get('X-Requested-With')):
            try:
                return str(res[0])
            except Exception:
                return str(res)
        return render_template('Model1.html', x=res)
    except Exception:
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')


@app.route('/predict-advanced', methods=['POST'])
def predict_advanced():
    try:
        nitro = int(float(request.form.get('Nitrogen')))
        phosp = int(float(request.form.get('Phosphorus')))
        pota = int(float(request.form.get('Potassium')))
        ph = float(request.form.get('pH'))
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')
        location_str = request.form.get('Location')
        if None in (soil_str, crop_str, location_str) or any(v in (None, '') for v in [nitro, phosp, pota, ph]):
            return render_template('Advanced.html', x='Please fill all fields. No defaults are used in Advanced mode.')
        soil_map = {'Loamy Soil':0, 'Peaty Soil':1, 'Acidic Soil':2, 'Neutral Soil':3, 'Alkaline Soil':4, 'Loamy':0, 'Peaty':1, 'Acidic':2, 'Neutral':3, 'Alkaline':4, 'Clayey':5, 'Red':6, 'Black':7, 'Sandy':8}
        crop_map = {'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10,'coffee':11,'kidneybeans':12,'orange':13,'pomegranate':14,'rice':15,'watermelon':16}
        if soil_str not in soil_map or crop_str not in crop_map:
            return render_template('Advanced.html', x='Invalid input. Unknown soil or crop type.')
        temp, humi, mois = 25, 60, 50
        # Encode Location using the saved label encoder from training (karnataka.csv)
        loc_norm_map = {
            'Bangalore': 'Bengaluru',
            'Mysore': 'Mysuru',
            'Chamrajnagar': 'Chamarajanagar',
        }
        loc_clean = (location_str or '').strip()
        loc_clean = loc_norm_map.get(loc_clean, loc_clean)
        le_loc = encoders.get('Location')
        if le_loc is not None and hasattr(le_loc, 'classes_'):
            try:
                location_encoded = int(le_loc.transform([loc_clean])[0])
            except Exception:
                try:
                    # case-insensitive fallback match
                    ci = next((c for c in le_loc.classes_ if c.lower() == loc_clean.lower()), None)
                    if ci is not None:
                        location_encoded = int(le_loc.transform([ci])[0])
                    else:
                        location_encoded = int(le_loc.transform([le_loc.classes_[0]])[0])
                except Exception:
                    location_encoded = 0
        else:
            location_encoded = 0
        features = [temp, humi, mois, soil_map[soil_str], crop_map[crop_str], nitro, pota, phosp, location_encoded, ph]
        prediction = model.predict(np.array([features]))
        res = ferti.classes_[prediction]
        if request.headers.get('X-Requested-With') == 'fetch':
            try:
                return str(res[0])
            except Exception:
                return str(res)
        return render_template('Advanced.html', x=res)
    except Exception:
        return render_template('Advanced.html', x='Invalid input. Please provide numeric values for all fields.')


@app.route('/fertilizer-info')
def fertilizer_info():
    try:
        name = _alias_to_key(request.args.get('name') or '')
        if not name:
            return jsonify({}), 400
        if name in FERTILIZER_INFO:
            return jsonify(FERTILIZER_INFO[name])
        for k, v in FERTILIZER_INFO.items():
            if k.lower() == name.lower():
                return jsonify(v)
        for k, v in FERTILIZER_INFO.items():
            if name.lower() in k.lower() or k.lower() in name.lower():
                return jsonify(v)
        return jsonify({}), 404
    except Exception:
        return jsonify({}), 500


# ------------- Kannada explanation + assistant -------------

def _build_kannada_prompt(payload: dict) -> str:
    fertilizer_name = str(payload.get('fertilizer') or '').strip()
    soil = str(payload.get('soil') or '').strip()
    crop = str(payload.get('crop') or '').strip()
    temp = str(payload.get('temperature') or '').strip()
    humi = str(payload.get('humidity') or '').strip()
    mois = str(payload.get('moisture') or '').strip()
    n = str(payload.get('nitrogen') or '').strip()
    p = str(payload.get('phosphorus') or '').strip()
    k = str(payload.get('potassium') or '').strip()
    ph = str(payload.get('ph') or '').strip()
    prompt = (
        "You are an agriculture assistant. Provide a short, clear explanation in Kannada (2-5 sentences) "
        "for why the recommended fertilizer is suitable. Avoid technical jargon; use farmer-friendly language.\n\n"
        f"Fertilizer: {fertilizer_name}\n"
        f"Soil Type: {soil}\n"
        f"Crop: {crop}\n"
        f"Temperature: {temp} °C\n"
        f"Humidity: {humi} %\n"
        f"Moisture: {mois} %\n"
        f"Nitrogen (N): {n}\nPhosphorus (P): {p}\nPotassium (K): {k}\npH: {ph}\n\n"
        "Answer only in Kannada."
    )
    return prompt


def _build_kannada_details_prompt(fertilizer_name: str) -> str:
    prompt = (
        "You are an agriculture expert. For the fertilizer named '{fertilizer_name}', provide the following details in simple, farmer-friendly Kannada:\n"
        "1. A brief description of the fertilizer.\n"
        "2. Recommended crops it is used for.\n"
        "3. Application instructions (how and when to use).\n"
        "4. General dosage guidelines (e.g., kg per acre), mentioning that exact dosage depends on soil tests.\n\n"
        "Answer only in Kannada."
    ).format(fertilizer_name=fertilizer_name)
    return prompt


def _call_ollama(prompt: str) -> str:
    model_name = os.getenv('OLLAMA_MODEL', 'llama3')
    url = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
    if _requests is None:
        return ''
    try:
        resp = _requests.post(url, json={'model': model_name, 'prompt': prompt, 'stream': False}, timeout=5)
        if resp.status_code != 200:
            return ''
        data = resp.json()
        return str(data.get('response') or '').strip()
    except Exception:
        return ''


def _call_openai(prompt: str) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or _requests is None:
        return ''
    model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    try:
        resp = _requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={
                'model': model_name,
                'messages': [
                    { 'role': 'system', 'content': 'Reply only in Kannada, be concise and farmer-friendly.' },
                    { 'role': 'user', 'content': prompt }
                ],
                'temperature': 0.4,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return ''
        data = resp.json()
        ch = (data.get('choices') or [{}])[0]
        msg = (ch.get('message') or {}).get('content')
        return (msg or '').strip()
    except Exception:
        return ''


@app.route('/explain-kn', methods=['POST'])
def explain_kannada():
    try:
        payload = request.get_json(silent=True) or {}
        prompt = _build_kannada_prompt(payload)
        answer = _call_ollama(prompt) or _call_openai(prompt)
        if not answer:
            fert = payload.get('fertilizer') or 'ನಿಮ್ಮ ಗೊಬ್ಬರ'
            answer = (
                f"ಶಿಫಾರಸು ಮಾಡಿದ ಗೊಬ್ಬರ: {fert}. ಇದು ನಿಮ್ಮ ಮಣ್ಣಿನ ಸ್ಥಿತಿ ಮತ್ತು ಬೆಳೆಗಾಗಿ ಸೂಕ್ತವಾಗಿದೆ. "
                "ಎನ್-ಪಿ-ಕೆ ಅಂಶಗಳ ಸಮತೋಲನದಿಂದ ಬೆಳೆ ಉತ್ತಮವಾಗಿ ಬೆಳೆಯಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ."
            )
        return answer
    except Exception:
        return "ಕ್ಷಮಿಸಿ, ವಿವರವನ್ನು ಈಗ ನೀಡಲಾಗುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."


def _get_fallback_response(user_text: str) -> str:
    text_lower = (user_text or '').lower()
    if any(word in text_lower for word in ['ಗೊಬ್ಬರ', 'fertilizer', 'fertilizer']):
        return "ಗೊಬ್ಬರದ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳಿಗೆ, ನಮ್ಮ ಫಲಕಾರಿ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆಯನ್ನು ಬಳಸಿ. ಮುಖ್ಯ ಪುಟದಲ್ಲಿ ನಿಮ್ಮ ಮಣ್ಣು, ಬೆಳೆ ಮತ್ತು ಪರಿಸರದ ಮಾಹಿತಿ ನಮೂದಿಸಿ."
    elif any(word in text_lower for word in ['ಬೆಳೆ', 'crop', 'ಬೆಳೆಗಳು']):
        return "ಬೆಳೆಗಳ ಬಗ್ಗೆ, ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ಸೂಕ್ತವಾದ ಬೆಳೆಗಳನ್ನು ಆಯ್ಕೆ ಮಾಡಿ. ಮಣ್ಣಿನ ಪ್ರಕಾರ ಮತ್ತು ಹವಾಮಾನವನ್ನು ಗಮನದಲ್ಲಿಟ್ಟುಕೊಂಡು ಬೆಳೆಗಳನ್ನು ಆರಿಸಿ."
    elif any(word in text_lower for word in ['ಮಣ್ಣು', 'soil', 'ಮಣ್ಣಿನ']):
        return "ಮಣ್ಣಿನ ಗುಣಮಟ್ಟವನ್ನು ಪರಿಶೀಲಿಸಲು, ನಿಮ್ಮ ಪ್ರದೇಶದ ಮಣ್ಣಿನ ಮಾದರಿಯನ್ನು ಪರೀಕ್ಷಿಸಿ. ವಿವಿಧ ಮಣ್ಣಿನ ಪ್ರಕಾರಗಳು ವಿಭಿನ್ನ ಬೆಳೆಗಳಿಗೆ ಸೂಕ್ತವಾಗಿವೆ."
    elif any(word in text_lower for word in ['ನೀರು', 'water', 'ನೀರಾವರಿ']):
        return "ನೀರಾವರಿ ಮತ್ತು ನೀರಿನ ನಿರ್ವಹಣೆ ಬೆಳೆಗಳಿಗೆ ಮುಖ್ಯ. ಸಮಯಕ್ಕೆ ಸರಿಯಾಗಿ ನೀರುಣಿಸಿ ಮತ್ತು ಮಣ್ಣಿನ ತೇವಾಂಶವನ್ನು ನಿಯಂತ್ರಿಸಿ."
    elif any(word in text_lower for word in ['ಹವಾಮಾನ', 'weather', 'ತಾಪಮಾನ']):
        return "ಹವಾಮಾನವು ಬೆಳೆಗಳ ಬೆಳವಣಿಗೆಗೆ ಮುಖ್ಯ. ನಿಮ್ಮ ಪ್ರದೇಶದ ಹವಾಮಾನಕ್ಕೆ ಸೂಕ್ತವಾದ ಬೆಳೆಗಳನ್ನು ಆಯ್ಕೆ ಮಾಡಿ."
    elif any(word in text_lower for word in ['ನಮಸ್ಕಾರ', 'hello', 'hi', 'ಹಲೋ']):
        return "ನಮಸ್ಕಾರ! ನಾನು ಕೃಷಿ ಸಹಾಯಕ. ಗೊಬ್ಬರ, ಬೆಳೆ, ಮಣ್ಣು ಅಥವಾ ಕೃಷಿ ಸಂಬಂಧಿತ ಯಾವುದೇ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ."
    else:
        return "ಕ್ಷಮಿಸಿ, ನಾನು ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಗೊಬ್ಬರ, ಬೆಳೆ, ಮಣ್ಣು ಅಥವಾ ಕೃಷಿ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ. ನಮ್ಮ ಫಲಕಾರಿ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆಯನ್ನು ಬಳಸಿ."


@app.route('/api/assistant', methods=['POST'])
def assistant_api():
    try:
        body = request.get_json(silent=True) or {}
        user_text = str(body.get('message') or '').strip()
        context = body.get('context') or {}
        if not user_text:
            return jsonify({ 'error': 'empty_message' }), 400
        system_kn = (
            "You are a helpful agriculture assistant. Reply only in Kannada. "
            "Be short and farmer-friendly. If asked about fertilizer results, "
            "use the provided context when helpful."
        )
        prompt = f"{system_kn}\n\nUser question: {user_text}\n\nContext (optional): {context}\n"
        answer = _call_ollama(prompt) or _call_openai(prompt) or _get_fallback_response(user_text)
        return jsonify({ 'reply': answer })
    except Exception:
        return jsonify({ 'reply': _get_fallback_response('') }), 200


@app.route('/explain-kn-details')
def explain_kn_details():
    try:
        fert_name = _alias_to_key(request.args.get('name') or '')
        if not fert_name:
            return "", 400
        # Prefer static Kannada if available for consistent Kannada output
        kn = None
        if fert_name in KANNADA_FERT_INFO:
            kn = KANNADA_FERT_INFO.get(fert_name)
        if not kn:
            for k, v in KANNADA_FERT_INFO.items():
                if k.lower() == fert_name.lower():
                    kn = v; break
        if not kn:
            for k, v in KANNADA_FERT_INFO.items():
                if fert_name.lower() in k.lower() or k.lower() in fert_name.lower():
                    kn = v; break
        if kn:
            parts = []
            if kn.get('type'):
                parts.append(f"ಪ್ರಕಾರ: {kn['type']}")
            if kn.get('description'):
                parts.append(f"ವಿವರಣೆ: {kn['description']}")
            if kn.get('dosage'):
                parts.append(f"ಮಾತ್ರೆ: {kn['dosage']}")
            if kn.get('application'):
                parts.append(f"ಬಳಕೆ ವಿಧಾನ: {kn['application']}")
            if kn.get('notes'):
                parts.append(f"ಟಿಪ್ಪಣಿ: {kn['notes']}")
            return "\n".join(parts)
        # Otherwise call LLMs as a fallback
        prompt = _build_kannada_details_prompt(fert_name)
        answer = _call_ollama(prompt) or _call_openai(prompt)
        if answer:
            return answer
        # Final fallback: English info with Kannada labels
        info = FERTILIZER_INFO.get(fert_name) or {}
        parts = []
        if info.get('type'):
            parts.append(f"ಪ್ರಕಾರ: {info['type']}")
        if info.get('description'):
            parts.append(f"ವಿವರಣೆ: {info['description']}")
        if info.get('dosage'):
            parts.append(f"ಮಾತ್ರೆ: {info['dosage']}")
        if info.get('application'):
            parts.append(f"ಬಳಕೆ ವಿಧಾನ: {info['application']}")
        if info.get('notes'):
            parts.append(f"ಟಿಪ್ಪಣಿ: {info['notes']}")
        return "\n".join(parts) or "ವಿವರಗಳು ಲಭ್ಯವಿಲ್ಲ."
    except Exception:
        return "", 500


@app.route('/llm-status')
def llm_status():
    try:
        test = _call_ollama("ಸರಿ")
        provider = 'ollama'
        model_name = os.getenv('OLLAMA_MODEL', 'llama3')
        url = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
        if not test:
            test = _call_openai("ಸರಿ")
            if test:
                provider = 'openai'
                model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
                url = 'https://api.openai.com/v1/chat/completions'
        ok = bool(test)
        return jsonify({ 'ok': ok, 'provider': provider, 'model': model_name, 'url': url })
    except Exception:
        return jsonify({ 'ok': False, 'provider': 'unknown' })


if __name__ == "__main__":
    app.run(debug=True)