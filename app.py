from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# load artifacts
PIPE_PATH = os.path.join("models","pipe.pkl")
COL_PATH = os.path.join("models","column_names.pkl")
pipe = joblib.load(PIPE_PATH)
colnames = joblib.load(COL_PATH)
# column names include the target at the end
feature_names = list(colnames)[:-1]
label_map = {0: "GALAXY", 1: "STAR", 2: "QSO"}

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # collect feature values in the same order as feature_names
    try:
        values = [float(data.get(f, 0.0)) for f in feature_names]
    except Exception:
        return jsonify({"error": "Invalid input. All features must be numeric."}), 400

    arr = np.array(values).reshape(1, -1)
    pred = pipe.predict(arr)[0]
    probs = pipe.predict_proba(arr)[0]
    # prepare response
    resp = {
        "prediction": label_map.get(int(pred), str(pred)),
        "probs": {
            label_map[i]: float(probs[i]) for i in range(len(probs))
        }
    }
    return jsonify(resp)

if __name__ == '__main__':
    app.run(debug=True)