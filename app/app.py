from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from typing import Tuple,List
from sklearn.pipeline import Pipeline
from models.fit import main
from .schema.validation import UserInput
from pathlib import Path
from contextlib import asynccontextmanager
import joblib
import numpy as np
import os

def load_or_create_models() -> Tuple[Pipeline,np.ndarray]:
    model_path = Path("models","estimator.pkl")
    columns_path = Path("models","column_names.pkl")
    
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
        return pipe,column_names
    except Exception as e:
        raise RuntimeError(f"Artifacts could not be loaded: {e}")

@asynccontextmanager
async def lifespan(app:FastAPI):
    # Load pipeline at start
    pipe,column_names = load_or_create_models()
    
    app.state.pipe = pipe
    app.state.column_names = column_names

    yield

app = FastAPI(title="CosmoClassifier", version="2.0(FastAPI)", lifespan=lifespan)

# Helper for providing the pipeline and column names
def get_model(request:Request) -> Tuple[Pipeline,np.ndarray]:
    return request.app.state.pipe, request.app.state.column_names

# Helper for subtraction in post route
def safe_sub(a:float,b:float,val:dict):
    return val.get(a,None) - val.get(b,None)

@app.get("/")
def home():
    msg = "Welcome to CosmoClassifier API. Provide the designated inputs " \
    "in the `predict` route to run predictions." \
    " Check the GitHub Repository for more."
    return msg

@app.post("/predict",status_code=201)
def prediction_ops(value:UserInput, dep:Tuple[Pipeline,np.ndarray] = Depends(get_model)):
    pipe, column_names = dep
    column_names:List[str] = column_names.tolist()

    # Preprocessing
    value:dict = value.model_dump(mode="json")
    kick = ["u","g","r","i","z"]
    final_value = {key:val for key,val in value.items() if key not in kick}
    final_value["u_g_color"] = safe_sub("u","g",value)
    final_value["g_r_color"] = safe_sub("g","r",value)
    final_value["r_i_color"] = safe_sub("r","i",value)
    final_value["i_z_color"] = safe_sub("i","z",value)

    # Order Check and running prediction
    final_res = []
    for col in column_names:
        if col == "class":
            continue
        else:
            final_res.append(final_value.get(col,None))
    final_res = np.array(final_res).reshape(1,-1)

    pred_label = int(pipe.predict(final_res)[0])
    pred_proba = pipe.predict_proba(final_res)[0].tolist()

    # Postprocessing
    label_map = {0: "GALAXY", 1: "STAR", 2: "QSO"}
    pred_label = label_map.get(pred_label)
    pred_proba = {lmv:round(proba,3) for lmv,proba in zip(label_map.values(), pred_proba)}

    msg = {"message":"prediction successful","predicted_class":pred_label, "prediction_probability":pred_proba}
    return JSONResponse(
        status_code=201, content=msg
    )

