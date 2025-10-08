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

# Load CSV data
def _read_csv(filename: str) -> pd.DataFrame:
    for p in [os.path.join(BASE_DIR, filename), os.path.join(SUBAPP_DIR, filename)]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(filename)

# Location → N,P,K defaults
try:
    _defaults_df = _read_csv('soil_defaults.csv')
    _defaults_df['Location'] = _defaults_df['Location'].str.strip()
    location_to_npk = {
        row['Location']: (row['Nitrogen'], row['Phosphorus'], row['Potassium'])
        for _, row in _defaults_df.iterrows()
    }
except Exception:
    location_to_npk = {}

# Basic mode defaults from f2.csv
try:
    _f2_df = _read_csv('f2.csv')
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
        soil_map = {'Loamy Soil':0,'Peaty Soil':1,'Acidic Soil':2,'Neutral Soil':3,'Alkaline Soil':4}
        crop_map = {'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10,'coffee':11,'kidneybeans':12,'orange':13,'pomegranate':14,'rice':15,'watermelon':16}
        if soil_str not in soil_map or crop_str not in crop_map:
            return 'Invalid input. Unknown soil or crop type.', 400
        features = [temp, humi, mois, soil_map[soil_str], crop_map[crop_str], nitro, pota, phosp]
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
        soil_map = {'Loamy Soil':0,'Peaty Soil':1,'Acidic Soil':2,'Neutral Soil':3,'Alkaline Soil':4}
        crop_map = {'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10,'coffee':11,'kidneybeans':12,'orange':13,'pomegranate':14,'rice':15,'watermelon':16}
        if soil_str not in soil_map or crop_str not in crop_map:
            return render_template('Model1.html', x='Invalid input. Unknown soil or crop type.')
        features = [temp, humi, mois, soil_map[soil_str], crop_map[crop_str], nitro, pota, phosp]
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
        soil_map = {'Loamy Soil':0,'Peaty Soil':1,'Acidic Soil':2,'Neutral Soil':3,'Alkaline Soil':4}
        crop_map = {'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10,'coffee':11,'kidneybeans':12,'orange':13,'pomegranate':14,'rice':15,'watermelon':16}
        if soil_str not in soil_map or crop_str not in crop_map:
            return render_template('Advanced.html', x='Invalid input. Unknown soil or crop type.')
        temp, humi, mois = 25, 60, 50
        features = [temp, humi, mois, soil_map[soil_str], crop_map[crop_str], nitro, pota, phosp]
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
