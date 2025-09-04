from flask import Flask, request, render_template
import pickle
import math
import pandas as pd

app = Flask(__name__)

# importing pickle files
model = pickle.load(open('classifier.pkl', 'rb'))
ferti = pickle.load(open('fertilizer.pkl', 'rb'))

# Load defaults (Location → N, P, K)
try:
    _defaults_df = pd.read_csv('soil_defaults.csv')
    _defaults_df['Location'] = _defaults_df['Location'].str.strip()
    location_to_npk = {
        row['Location']: (row['Nitrogen'], row['Phosphorus'], row['Potassium'])
        for _, row in _defaults_df.iterrows()
    }
except Exception:
    location_to_npk = {}

@app.route('/')
def home():
    return render_template('plantindex.html')

@app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@app.route('/Detail')
def Detail():
    return render_template('Detail.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Map form field names from the template to expected numeric features
    try:
        # Numeric fields from the form
        temp_str = request.form.get('Temperature')
        humi_str = request.form.get('Humidity')
        mois_str = request.form.get('Moisture')
        nitro_str = request.form.get('Nitrogen')
        phosp_str = request.form.get('Phosphorus')
        pota_str = request.form.get('Potassium')

        # Categorical selections
        soil_str = request.form.get('Soil_Type')
        crop_str = request.form.get('Crop_Type')
        location_str = request.form.get('Location')

        # Validate presence
        # Temperature and Humidity are optional when removed from the form
        if None in (soil_str, crop_str):
            return render_template('Model1.html', x='Invalid input. Missing one or more fields.')

        # Convert numeric strings to integers (optional for temp/humidity)
        temp = int(float(temp_str)) if temp_str not in (None, '') else 25

        # Provide sensible defaults if empty
        def _parse_or_default(value_str, default_value):
            if value_str is None or str(value_str).strip() == '':
                return int(default_value)
            return int(float(value_str))

        # Default humidity to mid-range; moisture removed from form, default to 50
        humi = _parse_or_default(humi_str, 60)
        mois = _parse_or_default(mois_str, 50)

        # Map Location from UI → CSV key
        loc_map = {
            'Mysore': 'Mysuru',
            'Mandya': 'Mandya',
            'Bangalore': 'Bangalore Rural',
            'Hassan': 'Hassan',
            'Chamrajnagar': 'Chamarajanagar',
        }
        csv_loc = loc_map.get(location_str, location_str)
        n_default, p_default, k_default = (0, 0, 0)
        if csv_loc in location_to_npk:
            n_default, p_default, k_default = location_to_npk[csv_loc]

        nitro = _parse_or_default(nitro_str, round(n_default))
        phosp = _parse_or_default(phosp_str, round(p_default))
        pota = _parse_or_default(pota_str, round(k_default))

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

if __name__ == "__main__":
    app.run(debug=True)