# ============================================
# app.py — Stellar Analytics Flask Backend
# Serves two endpoints:
#   POST /classify      → Task A
#   POST /predict-radius → Task B
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # allows React frontend to communicate

# ============================================
# LOAD ALL MODELS ON SERVER START
# We load once — not on every request
# Loading on every request would make the 
# API extremely slow
# ============================================
BASE = os.path.join(os.path.dirname(__file__), 'models')

try:
    classifier       = joblib.load(os.path.join(BASE, 'best_classifier.pkl'))
    clf_scaler       = joblib.load(os.path.join(BASE, 'classifier_scaler.pkl'))
    regressor        = joblib.load(os.path.join(BASE, 'best_regressor.pkl'))
    reg_scaler       = joblib.load(os.path.join(BASE, 'regressor_scaler.pkl'))
    clf_feature_names = joblib.load(os.path.join(BASE, 'clf_feature_names.pkl'))
    reg_feature_names = joblib.load(os.path.join(BASE, 'reg_feature_names.pkl'))

    # Load lasso indices if they exist
    lasso_path = os.path.join(BASE, 'lasso_selected_indices.pkl')
    lasso_indices = joblib.load(lasso_path) if os.path.exists(lasso_path) else None

    print("✅ All models loaded successfully!")
    print(f"   Classifier features : {len(clf_feature_names)}")
    print(f"   Regressor features  : {len(reg_feature_names)}")
    if lasso_indices:
        print(f"   Lasso selected      : {len(lasso_indices)} features")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise


# ============================================
# PREPROCESSING HELPERS
# These replicate EXACTLY what we did in 
# Colab during training — same transformations
# same order — otherwise predictions break
# ============================================

LOG_COLS = [
    'koi_period', 'koi_duration', 'koi_depth',
    'koi_model_snr', 'koi_num_transits',
    'st_radius', 'st_dens'
]

def apply_log_transform(df):
    """Apply log1p to the same skewed columns we transformed during training"""
    df = df.copy()
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def add_clf_engineered_features(df):
    """Recreate classification engineered features — must match Cell 3 exactly"""
    df = df.copy()
    df['snr_per_transit']    = df['koi_model_snr'] / (df['koi_num_transits'] + 1e-6)
    df['signal_strength']    = df['koi_depth']     / (df['koi_duration']     + 1e-6)
    df['flux_ratio']         = df['koi_depth']     / (df['st_radius'] ** 2   + 1e-6)
    df['grazing_transit']    = (df['koi_impact'] > 0.9).astype(int)
    df['orbital_compactness']= df['koi_period']    / (df['st_radius']        + 1e-6)
    return df


def add_reg_engineered_features(df):
    """Recreate regression engineered features — must match Cell 3 exactly"""
    df = df.copy()
    # 109.1 = conversion: 1 solar radius = 109.1 Earth radii
    df['physics_radius_est'] = df['st_radius'] * 109.1 * np.sqrt(df['koi_depth'] / 1e6)
    df['transit_geometry']   = df['koi_duration']  / (df['koi_period']         + 1e-6)
    df['stellar_luminosity'] = (df['st_radius']**2) * ((df['st_teff'] / 5778) ** 4)
    df['star_compactness']   = df['st_mass']        / (df['st_radius'] ** 3    + 1e-6)
    df['snr_per_transit']    = df['koi_model_snr']  / (df['koi_num_transits']  + 1e-6)
    return df


def categorize_planet(radius):
    """Convert predicted radius into a human readable category"""
    if radius < 1.5:
        return "Rocky Planet (Earth-like)"
    elif radius < 4:
        return "Super Earth"
    elif radius < 10:
        return "Neptune-like"
    else:
        return "Gas Giant (Jupiter-like)"


def validate_inputs(data, required_fields):
    """
    Check that all required fields are present
    and contain valid numeric values.
    Returns error message string or None if valid.
    """
    for field in required_fields:
        if field not in data:
            return f"Missing field: '{field}'"
        try:
            val = float(data[field])
            if np.isnan(val) or np.isinf(val):
                return f"Invalid value for '{field}': must be a finite number"
        except (TypeError, ValueError):
            return f"'{field}' must be a number, got: {data[field]}"
    return None


# ============================================
# BASE FIELDS — minimum inputs from frontend
# These are the raw measurements the user enters
# before any engineering or transformation
# ============================================
CLF_BASE_FIELDS = [
    'koi_period', 'koi_duration', 'koi_depth',
    'koi_impact', 'koi_model_snr', 'koi_num_transits',
    'koi_prad', 'st_teff', 'st_logg', 'st_met',
    'st_mass', 'st_radius', 'st_dens'
]

REG_BASE_FIELDS = [
    'koi_period', 'koi_duration', 'koi_depth',
    'koi_impact', 'koi_model_snr', 'koi_num_transits',
    'st_teff', 'st_logg', 'st_met',
    'st_mass', 'st_radius', 'st_dens'
]


# ============================================
# ROUTE 1 — Health Check
# Visit http://localhost:5000/ to verify 
# the server is running correctly
# ============================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status":  "running",
        "message": "Stellar Analytics API is live!",
        "endpoints": {
            "classify":       "POST /classify",
            "predict_radius": "POST /predict-radius"
        }
    })


# ============================================
# ROUTE 2 — Classification Endpoint
# Task A: Is this signal a real exoplanet
#         or a false positive?
#
# Expects JSON with all CLF_BASE_FIELDS
# Returns prediction label + probabilities
# ============================================
@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Validate inputs
        error = validate_inputs(data, CLF_BASE_FIELDS)
        if error:
            return jsonify({"error": error}), 400

        # Build raw dataframe from incoming JSON
        raw = {field: float(data[field]) for field in CLF_BASE_FIELDS}
        df  = pd.DataFrame([raw])

        # Apply same transformations as training
        df = apply_log_transform(df)
        df = add_clf_engineered_features(df)

        # Arrange columns in exact training order
        df = df[clf_feature_names]

        # Scale
        df_scaled = clf_scaler.transform(df)

        # Predict
        prediction  = classifier.predict(df_scaled)[0]
        probability = classifier.predict_proba(df_scaled)[0]

        label            = "CONFIRMED" if prediction == 1 else "FALSE POSITIVE"
        confidence       = round(float(probability[int(prediction)]) * 100, 2)
        confirmed_prob   = round(float(probability[1]) * 100, 2)
        false_pos_prob   = round(float(probability[0]) * 100, 2)

        return jsonify({
            "prediction":               label,
            "confidence":               confidence,
            "confirmed_probability":    confirmed_prob,
            "false_positive_probability": false_pos_prob,
            "interpretation": (
                "This signal shows strong characteristics of a real exoplanet."
                if label == "CONFIRMED"
                else "This signal is likely caused by instrumental noise or a binary star system."
            )
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ============================================
# ROUTE 3 — Regression Endpoint
# Task B: How large is this exoplanet
#         in Earth radii?
#
# Expects JSON with all REG_BASE_FIELDS
# Returns predicted radius + size category
# ============================================
@app.route('/predict-radius', methods=['POST'])
def predict_radius():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Validate inputs
        error = validate_inputs(data, REG_BASE_FIELDS)
        if error:
            return jsonify({"error": error}), 400

        # Build raw dataframe
        raw = {field: float(data[field]) for field in REG_BASE_FIELDS}
        df  = pd.DataFrame([raw])

        # Apply same transformations as training
        df = apply_log_transform(df)
        df = add_reg_engineered_features(df)

        # Arrange columns in exact training order
        df = df[reg_feature_names]

        # Apply lasso feature selection if winning model used it
        df_scaled = reg_scaler.transform(df)
        if lasso_indices is not None:
            df_scaled = df_scaled[:, lasso_indices]

        # Predict — model outputs log scale, reverse with expm1
        pred_log    = regressor.predict(df_scaled)[0]
        pred_radius = float(np.expm1(pred_log))
        pred_radius = round(pred_radius, 3)

        return jsonify({
            "predicted_radius_earth_radii": pred_radius,
            "size_category":  categorize_planet(pred_radius),
            "comparison": get_comparison(pred_radius)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ============================================
# ROUTE 4 — Combined Endpoint
# Runs both tasks in one API call
# Useful for the frontend to show both 
# results simultaneously
# ============================================
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Runs classification first.
    If confirmed, also predicts radius.
    Returns both results in one response.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # ---- Classification ----
        error = validate_inputs(data, CLF_BASE_FIELDS)
        if error:
            return jsonify({"error": error}), 400

        raw_clf = {f: float(data[f]) for f in CLF_BASE_FIELDS}
        df_clf  = pd.DataFrame([raw_clf])
        df_clf  = apply_log_transform(df_clf)
        df_clf  = add_clf_engineered_features(df_clf)
        df_clf  = df_clf[clf_feature_names]
        clf_scaled  = clf_scaler.transform(df_clf)

        prediction  = classifier.predict(clf_scaled)[0]
        probability = classifier.predict_proba(clf_scaled)[0]
        label       = "CONFIRMED" if prediction == 1 else "FALSE POSITIVE"

        result = {
            "classification": {
                "prediction":                 label,
                "confidence":                 round(float(probability[int(prediction)]) * 100, 2),
                "confirmed_probability":      round(float(probability[1]) * 100, 2),
                "false_positive_probability": round(float(probability[0]) * 100, 2),
            },
            "regression": None
        }

        # ---- Regression (only if confirmed) ----
        if label == "CONFIRMED":
            error = validate_inputs(data, REG_BASE_FIELDS)
            if not error:
                raw_reg = {f: float(data[f]) for f in REG_BASE_FIELDS}
                df_reg  = pd.DataFrame([raw_reg])
                df_reg  = apply_log_transform(df_reg)
                df_reg  = add_reg_engineered_features(df_reg)
                df_reg  = df_reg[reg_feature_names]
                reg_scaled  = reg_scaler.transform(df_reg)
                if lasso_indices is not None:
                    reg_scaled = reg_scaled[:, lasso_indices]

                pred_radius = float(np.expm1(regressor.predict(reg_scaled)[0]))
                pred_radius = round(pred_radius, 3)

                result["regression"] = {
                    "predicted_radius_earth_radii": pred_radius,
                    "size_category":  categorize_planet(pred_radius),
                    "comparison":     get_comparison(pred_radius)
                }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# ============================================
# HELPER — Planet size comparison
# Makes results more interpretable for users
# ============================================
def get_comparison(radius):
    if radius < 0.5:
        return "Smaller than Mars"
    elif radius < 1.0:
        return "Similar in size to Venus or Mars"
    elif radius < 1.5:
        return "Similar in size to Earth"
    elif radius < 2.5:
        return "About twice the size of Earth"
    elif radius < 4.0:
        return "Similar to a large Super Earth"
    elif radius < 6.0:
        return "Similar in size to Neptune"
    elif radius < 11.0:
        return "Similar in size to Saturn"
    else:
        return "Similar in size to Jupiter or larger"


# ============================================
# RUN SERVER
# debug=True shows detailed errors during dev
# Turn off debug=True before deploying!
# ============================================
if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')