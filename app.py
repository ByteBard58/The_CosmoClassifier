from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model + column names
pipe = joblib.load("models/pipe.pkl")
column_names = joblib.load("models/column_names.pkl")

# Human-readable labels for inputs
feature_labels = {
    "ra": "Right Ascension (degrees)",
    "dec": "Declination (degrees)",
    "u": "Ultraviolet Magnitude (u band)",
    "g": "Green Magnitude (g band)",
    "r": "Red Magnitude (r band)",
    "i": "Infrared Magnitude (i band)",
    "z": "Far Infrared Magnitude (z band)",
    "petroRad_u": "Petrosian Radius (u band)",
    "petroRad_g": "Petrosian Radius (g band)",
    "petroRad_i": "Petrosian Radius (i band)",
    "petroRad_r": "Petrosian Radius (r band)",
    "petroRad_z": "Petrosian Radius (z band)",
    "petroFlux_u": "Petrosian Flux (u band)",
    "petroFlux_g": "Petrosian Flux (g band)",
    "petroFlux_i": "Petrosian Flux (i band)",
    "petroFlux_r": "Petrosian Flux (r band)",
    "petroFlux_z": "Petrosian Flux (z band)",
    "petroR50_u": "Petrosian 50% Light Radius (u band)",
    "petroR50_g": "Petrosian 50% Light Radius (g band)",
    "petroR50_i": "Petrosian 50% Light Radius (i band)",
    "petroR50_r": "Petrosian 50% Light Radius (r band)",
    "petroR50_z": "Petrosian 50% Light Radius (z band)",
    "psfMag_u": "PSF Magnitude (u band)",
    "psfMag_g": "PSF Magnitude (g band)",
    "psfMag_r": "PSF Magnitude (r band)",
    "psfMag_i": "PSF Magnitude (i band)",
    "psfMag_z": "PSF Magnitude (z band)",
    "expAB_u": "Exponential Axis Ratio (u band)",
    "expAB_g": "Exponential Axis Ratio (g band)",
    "expAB_r": "Exponential Axis Ratio (r band)",
    "expAB_i": "Exponential Axis Ratio (i band)",
    "expAB_z": "Exponential Axis Ratio (z band)",
    "redshift": "Redshift Value"
}


@app.route("/")
def home():
    readable_names = [feature_labels.get(col, col) for col in column_names if col != "class"]
    return render_template("index.html", columns=column_names, labels=readable_names,zip=zip)


@app.route("/predict", methods=["POST"])
def predict():
    data = {}
    for col in column_names:
        if col == "class":
            continue
        value = request.form.get(col)
        data[col] = float(value) if value.strip() else np.nan  # let imputer handle missing

    df = pd.DataFrame([data])
    pred_class = pipe.predict(df)[0]
    probs = pipe.predict_proba(df)[0]

    classes = list(pipe.classes_)
    response = {
        "prediction": str(pred_class),
        "probabilities": {cls: round(float(prob), 3) for cls, prob in zip(classes, probs)}
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
