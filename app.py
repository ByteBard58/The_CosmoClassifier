from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from fit import main
import os

app = Flask(__name__)

# Check for model and column names files, create them if they don't exist
def load_or_create_models():
    global pipe, column_names
    model_path = "models/estimator.pkl"
    columns_path = "models/column_names.pkl"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if both files exist
    if not (os.path.exists(model_path) and os.path.exists(columns_path)):
        print("Model or column names file not found. Running fit.py...")
        main()  # Call the dumping method to create the .pkl files
    else:
        print("All artifacts are found! Loading them now....")
    
    # Load model and column names
    try:
        pipe = joblib.load(model_path)
        column_names = joblib.load(columns_path)
        print("Artifacts are loaded successfully! Ready for prediction....")
    except Exception as e:
        print(f"Artifacts could not be loaded ! Error: {e}")

# Load or create models at startup
load_or_create_models()

# Human-readable labels for inputs
# Human-readable labels for inputs
feature_labels = {
    "ra": "Right Ascension (degrees)",
    "dec": "Declination (degrees)",
    "redshift": "Redshift Value",
    "psfMag_r": "PSF Magnitude (r band)",
    "u": "u (Ultraviolet Band)",
    "g": "g (Green Band)",
    "r": "r (Red Band)",
    "i": "i (Near Infrared Band)",
    "z": "z (Infrared Band)"
}

# Define the fields we want the user to see, in order
DISPLAY_COLUMNS = ["ra", "dec", "redshift", "psfMag_r", "u", "g", "r", "i", "z"]

@app.route("/")
def home():
    # Pass the display columns and their labels to the template
    readable_names = [feature_labels.get(col, col) for col in DISPLAY_COLUMNS]
    return render_template("index.html", columns=DISPLAY_COLUMNS, labels=readable_names, zip=zip)

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Collect raw inputs from the form
    raw_input = {}
    for col in DISPLAY_COLUMNS:
        val = request.form.get(col)
        try:
            raw_input[col] = float(val) if val is not None and str(val).strip() != "" else np.nan
        except ValueError:
            raw_input[col] = np.nan

    # 2. Compute the derived color features
    # logic: u_g_color = u - g, etc.
    # Note: If any operand is NaN, the result will be NaN, which the imputer handles.
    derived_data = {}
    derived_data["ra"] = raw_input.get("ra", np.nan)
    derived_data["dec"] = raw_input.get("dec", np.nan)
    derived_data["redshift"] = raw_input.get("redshift", np.nan)
    derived_data["psfMag_r"] = raw_input.get("psfMag_r", np.nan)
    
    # helper to safely subtract
    def safe_sub(a, b):
        return raw_input.get(a, np.nan) - raw_input.get(b, np.nan)

    derived_data["u_g_color"] = safe_sub("u", "g")
    derived_data["g_r_color"] = safe_sub("g", "r")
    derived_data["r_i_color"] = safe_sub("r", "i")
    derived_data["i_z_color"] = safe_sub("i", "z")

    # 3. Assemble the final feature vector in the order the model expects
    # column_names contains the features used during training (loaded from pickle)
    final_features = []
    # We iterate over column_names from the loaded model to ensure correct order
    # column_names includes "class", which we skip.
    for col in column_names:
        if col == "class":
            continue
        final_features.append(derived_data.get(col, np.nan))

    # 4. Predict
    # Reshape to 2D array: (1, n_features)
    arr = np.array([final_features])
    
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
    app.run()