import os
import warnings

# Simple warning suppression
warnings.filterwarnings('ignore')

from flask import Flask, request, render_template
import pickle
import math
import pandas as pd
from flask import jsonify
import numpy as np
try:
    import requests as _requests
except Exception:
    _requests = None

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# importing pickle files
model = pickle.load(open('fertilizer_model.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))
ferti = encoders['Fertilizer']

# Load defaults (Location → N, P, K)
try:
    _defaults_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'soil_defaults.csv'))
    _defaults_df['Location'] = _defaults_df['Location'].str.strip()
    location_to_npk = {
        row['Location']: (row['Nitrogen'], row['Phosphorus'], row['Potassium'])
        for _, row in _defaults_df.iterrows()
    }
except Exception:
    location_to_npk = {}

# Basic mode defaults from f2.csv grouped by Soil_Type+Crop_Type and also by
# (Soil_Type, Crop_Type, Temperature, Humidity, Moisture) for N/P/K/pH
try:
    _f2_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'f2.csv'))
    # Normalize column names that might vary slightly
    _f2_df = _f2_df.rename(columns={
        'Temparature': 'Temperature',
        'Phosphorous': 'Phosphorus',
        'PH': 'pH'
    })
    # Full key for NPK defaults
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
            'pH': row['pH'],
        }
        for _, row in _full_means.iterrows()
    }

    # Fallback averages by soil+crop only
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
            'pH': row['pH'],
        }
        for _, row in sc_means.iterrows()
    }

    # Crop-only and Soil-only fallbacks
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
            'pH': row['pH'],
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
            'pH': row['pH'],
        }
        for _, row in soil_means.iterrows()
    }

    overall_means = _f2_df[npk_cols].mean(numeric_only=True).round()
    npk_defaults_overall = {
        'Nitrogen': int(overall_means['Nitrogen']),
        'Phosphorus': int(overall_means['Phosphorus']),
        'Potassium': int(overall_means['Potassium']),
        'pH': overall_means['pH'],
    }
except Exception:
    npk_defaults_full = {}
    npk_defaults_sc = {}
    npk_defaults_crop = {}
    npk_defaults_soil = {}
    npk_defaults_overall = {}

@app.route('/')
def home():
    return render_template('plantindex.html')

@app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@app.route('/Detail')
def Detail():
    return render_template('Detail.html')

@app.route('/Advanced')
def Advanced():
    return render_template('Advanced.html')

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
    """AJAX endpoint that returns only the fertilizer name"""
    # Use the same logic as the main predict function but return only the result
    try:
        # Numeric fields from the form (all required in basic mode now)
        temp_str = request.form.get('Temperature')
        humi_str = request.form.get('Humidity')
        mois_str = request.form.get('Moisture')
        nitro_str = request.form.get('Nitrogen')
        phosp_str = request.form.get('Phosphorus')
        pota_str = request.form.get('Potassium')
        ph_str = request.form.get('pH')

        # Categorical selections
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')

        # Convert numeric strings to integers
        temp = int(float(temp_str))
        humi = int(float(humi_str))
        mois = int(float(mois_str))
        nitro = int(float(nitro_str))
        phosp = int(float(phosp_str))
        pota = int(float(pota_str))
        ph = float(ph_str)

        # Encode categorical values to integers expected by the model
        soil_map = {
            'Loamy Soil': 0,
            'Peaty Soil': 1,
            'Acidic Soil': 2,
            'Neutral Soil': 3,
            'Alkaline Soil': 4,
        }
        crop_map = {
            'Barley': 0,
            'Cotton': 1,
            'Ground Nuts': 2,
            'Maize': 3,
            'Millets': 4,
            'Oil Seeds': 5,
            'Paddy': 6,
            'Pulses': 7,
            'Sugarcane': 8,
            'Tobacco': 9,
            'Wheat': 10,
            'coffee': 11,
            'kidneybeans': 12,
            'orange': 13,
            'pomegranate': 14,
            'rice': 15,
            'watermelon': 16,
        }

        if soil_str not in soil_map or crop_str not in crop_map:
            return 'Invalid input. Unknown soil or crop type.', 400

        soil = soil_map[soil_str]
        crop = crop_map[crop_str]

        features = [temp, humi, mois, soil, crop, nitro, pota, phosp]
        # Convert to numpy array to avoid feature name warnings
        features_array = np.array([features])
        prediction = model.predict(features_array)
        res = ferti.classes_[prediction]
        
        try:
            return str(res[0])
        except Exception:
            return str(res)
    except Exception as e:
        return f'Error: {str(e)}', 400

@app.route('/predict', methods=['POST'])
def predict():
    # Map form field names from the template to expected numeric features
    try:
        # Numeric fields from the form (all required in basic mode now)
        temp_str = request.form.get('Temperature')
        humi_str = request.form.get('Humidity')
        mois_str = request.form.get('Moisture')
        nitro_str = request.form.get('Nitrogen')
        phosp_str = request.form.get('Phosphorus')
        pota_str = request.form.get('Potassium')
        ph_str = request.form.get('pH')

        # Categorical selections
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')

        # Validate presence
        if None in (soil_str, crop_str, temp_str, humi_str, mois_str, nitro_str, phosp_str, pota_str, ph_str) or '' in (
            str(temp_str or ''), str(humi_str or ''), str(mois_str or ''), str(nitro_str or ''), str(phosp_str or ''), str(pota_str or ''), str(ph_str or '')):
            return render_template('Model1.html', x='Please fill all fields in Basic mode.')

        # Convert numeric strings to integers
        temp = int(float(temp_str))
        humi = int(float(humi_str))
        mois = int(float(mois_str))
        nitro = int(float(nitro_str))
        phosp = int(float(phosp_str))
        pota = int(float(pota_str))
        ph = float(ph_str)

        # Encode categorical values to integers expected by the model
        soil_map = {
            'Loamy Soil': 0,
            'Peaty Soil': 1,
            'Acidic Soil': 2,
            'Neutral Soil': 3,
            'Alkaline Soil': 4,
        }
        crop_map = {
            'Barley': 0,
            'Cotton': 1,
            'Ground Nuts': 2,
            'Maize': 3,
            'Millets': 4,
            'Oil Seeds': 5,
            'Paddy': 6,
            'Pulses': 7,
            'Sugarcane': 8,
            'Tobacco': 9,
            'Wheat': 10,
            'coffee': 11,
            'kidneybeans': 12,
            'orange': 13,
            'pomegranate': 14,
            'rice': 15,
            'watermelon': 16,
        }

        if soil_str not in soil_map or crop_str not in crop_map:
            return render_template('Model1.html', x='Invalid input. Unknown soil or crop type.')

        soil = soil_map[soil_str]
        crop = crop_map[crop_str]

        features = [temp, humi, mois, soil, crop, nitro, pota, phosp]
        # Convert to numpy array to avoid feature name warnings
        features_array = np.array([features])
        prediction = model.predict(features_array)
        res = ferti.classes_[prediction]
        # If AJAX/fetch request, return plain text result
        print(f"Headers: {dict(request.headers)}")
        print(f"X-Requested-With: {request.headers.get('X-Requested-With')}")
        print(f"Content-Type: {request.headers.get('Content-Type')}")
        
        # Check for AJAX request
        if (request.headers.get('X-Requested-With') == 'fetch' or 
            request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and 
            request.headers.get('X-Requested-With')):
            print("Returning plain text")
            try:
                return str(res[0])
            except Exception:
                return str(res)
        print("Returning template")
        return render_template('Model1.html', x=res)
    except Exception:
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

@app.route('/predict-advanced', methods=['POST'])
def predict_advanced():
    try:
        # Read required fields; do not auto-default N/P/K
        nitro_str = request.form.get('Nitrogen')
        phosp_str = request.form.get('Phosphorus')
        pota_str = request.form.get('Potassium')
        ph_str = request.form.get('pH')

        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')
        location_str = request.form.get('Location')

        if None in (soil_str, crop_str, location_str, ph_str) or '' in (nitro_str or '', phosp_str or '', pota_str or '', ph_str or ''):
            return render_template('Advanced.html', x='Please fill all fields. No defaults are used in Advanced mode.')

        # Convert numeric inputs; Advanced requires explicit numbers
        nitro = int(float(nitro_str))
        phosp = int(float(phosp_str))
        pota = int(float(pota_str))
        ph = float(ph_str)

        # Same categorical maps
        soil_map = {
            'Loamy Soil': 0,
            'Peaty Soil': 1,
            'Acidic Soil': 2,
            'Neutral Soil': 3,
            'Alkaline Soil': 4,
        }
        crop_map = {
            'Barley': 0,
            'Cotton': 1,
            'Ground Nuts': 2,
            'Maize': 3,
            'Millets': 4,
            'Oil Seeds': 5,
            'Paddy': 6,
            'Pulses': 7,
            'Sugarcane': 8,
            'Tobacco': 9,
            'Wheat': 10,
            'coffee': 11,
            'kidneybeans': 12,
            'orange': 13,
            'pomegranate': 14,
            'rice': 15,
            'watermelon': 16,
        }

        if soil_str not in soil_map or crop_str not in crop_map:
            return render_template('Advanced.html', x='Invalid input. Unknown soil or crop type.')

        soil = soil_map[soil_str]
        crop = crop_map[crop_str]

        # Advanced page does not expose temp/humidity/moisture; keep same defaults
        temp = 25
        humi = 60
        mois = 50

        features = [temp, humi, mois, soil, crop, nitro, pota, phosp]
        # Convert to numpy array to avoid feature name warnings
        features_array = np.array([features])
        prediction = model.predict(features_array)
        res = ferti.classes_[prediction]
        if request.headers.get('X-Requested-With') == 'fetch':
            try:
                return str(res[0])
            except Exception:
                return str(res)
        return render_template('Advanced.html', x=res)
    except Exception:
        return render_template('Advanced.html', x='Invalid input. Please provide numeric values for all fields.')

## app.run moved to the end of file so all routes are registered before starting

# -------------------- Kannada LLM Integration --------------------

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

    # Instruction: reply in Kannada, short and farmer-friendly
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


def _call_ollama(prompt: str) -> str:
    model = os.getenv('OLLAMA_MODEL', 'llama3')  # Use most reliable model
    url = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
    if _requests is None:
        return ''
    try:
        resp = _requests.post(
            url,
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=5,  # Faster timeout
        )
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
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    try:
        resp = _requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': model,
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
        answer = _call_ollama(prompt)
        if not answer:
            answer = _call_openai(prompt)
        if not answer:
            # Fallback minimal Kannada message
            fert = payload.get('fertilizer') or 'ನಿಮ್ಮ ಗೊಬ್ಬರ'
            answer = (
                f"ಶಿಫಾರಸು ಮಾಡಿದ ಗೊಬ್ಬರ: {fert}. ಇದು ನಿಮ್ಮ ಮಣ್ಣಿನ ಸ್ಥಿತಿ ಮತ್ತು ಬೆಳೆಗಾಗಿ ಸೂಕ್ತವಾಗಿದೆ. "
                "ಎನ್-ಪಿ-ಕೆ ಅಂಶಗಳ ಸಮತೋಲನದಿಂದ ಬೆಳೆ ಉತ್ತಮವಾಗಿ ಬೆಳೆಯಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ."
            )
        return answer
    except Exception:
        return (
            "ಕ್ಷಮಿಸಿ, ವಿವರವನ್ನು ಈಗ ನೀಡಲಾಗುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.")


# -------------------- Kannada Assistant (Chat) --------------------

@app.route('/assistant')
def assistant_page():
    # Opens a simple chat UI
    return render_template('assistant.html')


def _get_fallback_response(user_text: str) -> str:
    """Provide fallback responses when LLM is unavailable"""
    text_lower = user_text.lower()
    
    # Common agriculture questions in Kannada
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

        # Try LLM first
        system_kn = (
            "You are a helpful agriculture assistant. Reply only in Kannada. "
            "Be short and farmer-friendly. If asked about fertilizer results, "
            "use the provided context when helpful."
        )
        prompt = (
            f"{system_kn}\n\n"
            f"User question: {user_text}\n\n"
            f"Context (optional): {context}\n"
        )
        
        answer = _call_ollama(prompt)
        if not answer:
            answer = _call_openai(prompt)
        
        # If LLM fails, use fallback responses
        if not answer:
            answer = _get_fallback_response(user_text)
            
        return jsonify({ 'reply': answer })
    except Exception:
        return jsonify({ 'reply': _get_fallback_response(user_text) }), 200


@app.route('/llm-status')
def llm_status():
    # quick connectivity check
    try:
        test = _call_ollama("ಸರಿ")
        provider = 'ollama'
        model = os.getenv('OLLAMA_MODEL', 'llama3')
        url = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
        if not test:
            test = _call_openai("ಸರಿ")
            if test:
                provider = 'openai'
                model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
                url = 'https://api.openai.com/v1/chat/completions'
        ok = bool(test)
        return jsonify({ 'ok': ok, 'provider': provider, 'model': model, 'url': url })
    except Exception:
        return jsonify({ 'ok': False, 'provider': 'unknown' })

if __name__ == "__main__":
    app.run(debug=True)
