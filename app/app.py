from pydantic import BaseModel
from fastapi import FastAPI, Depends, Request, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError, HTTPException
from typing import Tuple,List
from sklearn.pipeline import Pipeline
from models.fit import main
from .schema.validation import UserInput
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import joblib
import os

# Helper for loading and self-healing the artifacts
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
    
# Helper for performing feature engineering
def preprocess_data(value:BaseModel) -> dict:
    # Preprocessing
    value:dict = value.model_dump(mode="json")
    kick = ["u","g","r","i","z"]
    final_value = {key:val for key,val in value.items() if key not in kick}
    final_value["u_g_color"] = safe_sub("u","g",value)
    final_value["g_r_color"] = safe_sub("g","r",value)
    final_value["r_i_color"] = safe_sub("r","i",value)
    final_value["i_z_color"] = safe_sub("i","z",value)

    return final_value

# Helper for validating user-provided csv files
def upload_validator(df:pd.DataFrame,col_names:List[str]) -> pd.DataFrame:
    if df.columns.tolist() != col_names:
        raise HTTPException(
            status_code=422, detail="Uploaded csv file does not match the expected " \
            "columns or their order"
        )
    
    try:
        df = df.astype(float)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="All values must be numeric (float-compatible)"
        )
    return df

@asynccontextmanager
async def lifespan(app:FastAPI):
    # Load pipeline at start
    pipe,column_names = load_or_create_models()
    
    app.state.pipe = pipe
    app.state.column_names = column_names

    yield

app = FastAPI(title="CosmoClassifier", version="2.0(FastAPI)", lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    # Flatten error messages for the frontend
    error_msgs = []
    for err in errors:
        field = err['loc'][-1]
        msg = err['msg']
        error_msgs.append(f"Invalid {field}: {msg}")
    
    return JSONResponse(
        status_code=422,
        content={"message": "validation failed", "error": "; ".join(error_msgs)},
    )

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Helper for providing the pipeline and column names
def get_model(request:Request) -> Tuple[Pipeline,np.ndarray]:
    return request.app.state.pipe, request.app.state.column_names

# Helper for subtraction in post route
def safe_sub(a:float,b:float,val:dict):
    return val.get(a,None) - val.get(b,None)

@app.get("/health",status_code=200)
def health():
    msg = {
        "title":"CosmoClassifier",
        "version":"2.0(FastAPI)",
        "status":"All systems operational"
    }
    return JSONResponse(status_code=200,content=msg)

@app.get("/")
def home():
    index_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(index_path)

@app.post("/predict",status_code=201)
def prediction_ops(value:UserInput, dep:Tuple[Pipeline,np.ndarray] = Depends(get_model)):
    pipe, column_names = dep
    column_names:List[str] = column_names.tolist()
    final_value:dict = preprocess_data(value)

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

    msg = {
        "message": "prediction successful",
        "prediction": pred_label, 
        "probabilities": pred_proba
    }
    return JSONResponse(
        status_code=201, content=msg
    )

@app.post("/predict/file")
async def prediction_via_file_ops(payload:UploadFile, dep: Tuple[Pipeline,np.ndarray] = Depends(get_model)):
    pipe, column_names = dep
    expected_upload_cols = ['ra', 'dec', 'redshift', 'psfMag_r', 'u', 'g', 'r', 'i', 'z']

    accepted_exts = [".csv"]
    extension = Path(payload.filename).suffix
    if extension.lower() not in accepted_exts:
        raise HTTPException(
            status_code=422, 
            detail=f"Uploaded data must be in '.csv' format, got {extension} instead"
        )

    try:
        df = pd.read_csv(payload.file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV file tracking: {str(e)}")
    df = upload_validator(df, expected_upload_cols)
    
    # Feature Engineering (Vectorized)
    df['u_g_color'] = df['u'] - df['g']
    df['g_r_color'] = df['g'] - df['r']
    df['r_i_color'] = df['r'] - df['i']
    df['i_z_color'] = df['i'] - df['z']
    df = df.drop(columns=['u', 'g', 'r', 'i', 'z'])
    
    # Reorder columns to match the pipeline's expected order (excluding 'class')
    model_features = [col for col in column_names.tolist() if col != "class"]
    df = df[model_features]

    pred_label:list[float] = pipe.predict(df).tolist()
    pred_proba:list[list[float]] = pipe.predict_proba(df).tolist()

    # Postprocessing
    label_map = {0: "GALAXY", 1: "STAR", 2: "QSO"}
    pred_label:list[str] = [label_map.get(pred) for pred in pred_label]
    pred_proba = [[round(r, 3) for r in pred] for pred in pred_proba]

    msg = {
        "message": "batch prediction successful",
        "prediction": pred_label, 
        "probabilities": pred_proba
    }
    return JSONResponse(
        status_code=201, content=msg
    )