# ============================================================
#   Waste Classification — Flask Web App
#   Student : Kirtan Khasiya | SVIT Vasad
# ============================================================

import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('model', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_models():
    with open('model/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le, scaler

try:
    model, le, scaler = load_models()
    print("✅ Model loaded successfully!")
except:
    model, le, scaler = None, None, None
    print("⚠️ Model not found! Run waste_classification.py first!")

# ─────────────────────────────────────────────────────────────
# DISPOSAL INFO
# ─────────────────────────────────────────────────────────────
DISPOSAL_INFO = {
    'Organic': {
        'bin'       : '🟢 Green Bin',
        'tip'       : 'Compost this waste. Great for making fertilizer!',
        'recyclable': False,
        'color'     : '#2ecc71',
        'icon'      : '🌿'
    },
    'Recyclable': {
        'bin'       : '🔵 Blue Bin',
        'tip'       : 'Clean and dry before recycling. Check local guidelines.',
        'recyclable': True,
        'color'     : '#3498db',
        'icon'      : '♻️'
    }
}

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded! Run waste_classification.py first!'}), 500

    try:
        # Get form data
        material_type    = float(request.form.get('material_type', 1))
        weight_kg        = float(request.form.get('weight_kg', 0.5))
        size_cm          = float(request.form.get('size_cm', 10))
        is_contaminated  = float(request.form.get('is_contaminated', 0))
        moisture_level   = float(request.form.get('moisture_level', 50))
        decompose_days   = float(request.form.get('decompose_days', 30))

        # Prepare input
        features = np.array([[
            material_type,
            weight_kg,
            size_cm,
            is_contaminated,
            moisture_level,
            decompose_days
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction      = model.predict(features_scaled)
        probability     = model.predict_proba(features_scaled)
        class_name      = le.inverse_transform(prediction)[0]
        confidence      = round(float(np.max(probability)) * 100, 2)
        disposal        = DISPOSAL_INFO[class_name]

        return jsonify({
            'success'    : True,
            'category'   : class_name,
            'confidence' : confidence,
            'bin'        : disposal['bin'],
            'tip'        : disposal['tip'],
            'recyclable' : disposal['recyclable'],
            'color'      : disposal['color'],
            'icon'       : disposal['icon'],
            'probs'      : {
                le.classes_[i]: round(float(probability[0][i]) * 100, 2)
                for i in range(len(le.classes_))
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model' : 'loaded' if model else 'not loaded'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)