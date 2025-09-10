from flask import Flask, request, render_template
import pickle
import math
import pandas as pd
from flask import jsonify
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# importing pickle files
model = pickle.load(open('classifier.pkl', 'rb'))
ferti = pickle.load(open('fertilizer.pkl', 'rb'))

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
# (Soil_Type, Crop_Type, Temperature, Humidity, Moisture) for N/P/K
try:
    _f2_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'f2.csv'))
    # Normalize column names that might vary slightly
    _f2_df = _f2_df.rename(columns={
        'Temparature': 'Temperature',
        'Phosphorous': 'Phosphorus'
    })
    # Full key for NPK defaults
    full_key_cols = ['Soil_Type', 'Crop_Type', 'Temperature', 'Humidity', 'Moisture']
    npk_cols = ['Nitrogen', 'Phosphorus', 'Potassium']
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
        }
        for _, row in soil_means.iterrows()
    }

    overall_means = _f2_df[npk_cols].mean(numeric_only=True).round()
    npk_defaults_overall = {
        'Nitrogen': int(overall_means['Nitrogen']),
        'Phosphorus': int(overall_means['Phosphorus']),
        'Potassium': int(overall_means['Potassium']),
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

        # Categorical selections
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')

        # Validate presence
        if None in (soil_str, crop_str, temp_str, humi_str, mois_str, nitro_str, phosp_str, pota_str) or '' in (
            str(temp_str or ''), str(humi_str or ''), str(mois_str or ''), str(nitro_str or ''), str(phosp_str or ''), str(pota_str or '')):
            return render_template('Model1.html', x='Please fill all fields in Basic mode.')

        # Convert numeric strings to integers
        temp = int(float(temp_str))
        humi = int(float(humi_str))
        mois = int(float(mois_str))
        nitro = int(float(nitro_str))
        phosp = int(float(phosp_str))
        pota = int(float(pota_str))

        # Encode categorical values to integers expected by the model
        soil_map = {
            'Black': 0,
            'Clayey': 1,
            'Loamy': 2,
            'Red': 3,
            'Sandy': 4,
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
        prediction = model.predict([features])
        res = ferti.classes_[prediction]
        # If AJAX/fetch request, return plain text result
        if request.headers.get('X-Requested-With') == 'fetch':
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
        # Read required fields; do not auto-default N/P/K
        nitro_str = request.form.get('Nitrogen')
        phosp_str = request.form.get('Phosphorus')
        pota_str = request.form.get('Potassium')

        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')
        location_str = request.form.get('Location')

        if None in (soil_str, crop_str, location_str) or '' in (nitro_str or '', phosp_str or '', pota_str or ''):
            return render_template('Advanced.html', x='Please fill all fields. No defaults are used in Advanced mode.')

        # Convert numeric inputs; Advanced requires explicit numbers
        nitro = int(float(nitro_str))
        phosp = int(float(phosp_str))
        pota = int(float(pota_str))

        # Same categorical maps
        soil_map = {
            'Black': 0,
            'Clayey': 1,
            'Loamy': 2,
            'Red': 3,
            'Sandy': 4,
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
        prediction = model.predict([features])
        res = ferti.classes_[prediction]
        if request.headers.get('X-Requested-With') == 'fetch':
            try:
                return str(res[0])
            except Exception:
                return str(res)
        return render_template('Advanced.html', x=res)
    except Exception:
        return render_template('Advanced.html', x='Invalid input. Please provide numeric values for all fields.')

if __name__ == "__main__":
    app.run(debug=True)