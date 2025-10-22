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
    "redshift": "Redshift Value",
    "psfMag_r": "PSF Magnitude (r band)",
    "u_g_color": "UV Magnitude - Green Magnitude",
    "g_r_color": "Green Magnitude - Red Magnitude",
    "r_i_color": "Red Magnitude - Infrared Magnitude",
    "i_z_color": "Infrared Magnitude - Far Infrared Magnitude"
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
        # coerce to float if possible, empty or non-numeric -> np.nan for imputer
        try:
            data[col] = float(value) if value is not None and str(value).strip() != "" else np.nan
        except ValueError:
            data[col] = np.nan

    df = pd.DataFrame([data])
    # pass numpy array to pipeline to avoid feature-name mismatch warnings from SimpleImputer
    arr = df.to_numpy()
    pred_class = pipe.predict(arr)[0]
    probs = pipe.predict_proba(arr)[0]

    classes = list(pipe.classes_)
    # map numeric classes to human-readable labels
    label_map = {0: "GALAXY", 1: "STAR", 2: "QSO"}
    # predicted class label
    pred_label = label_map.get(int(pred_class), str(pred_class))
    # probabilities mapped to label names
    probs_by_label = {label_map.get(int(cls), str(cls)): round(float(prob), 3)
                      for cls, prob in zip(classes, probs)}
    response = {"prediction": pred_label, "probabilities": probs_by_label}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
