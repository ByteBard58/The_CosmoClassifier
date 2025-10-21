from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
import joblib

def data_collection(file_path = "Datasets/SDSS_DR18.csv"):
  # read the raw CSV into a DataFrame
  df_raw = pd.read_csv(file_path)

  # drop identifier and metadata columns which may lead to leakage
  df_raw = df_raw.drop(columns=
    ["objid", "specobjid", "run", "rerun", "camcol", 
    "field", "plate", "mjd", "fiberid"])
  # work on a copy to avoid mutating the original frame
  df_1 = df_raw.copy()

  # map string classes to numeric labels
  df_1["class"] = df_1["class"].map({
    "GALAXY":0,
    "STAR":1,
    "QSO":2
  })

  # finalize DataFrame and split into features and target
  df = df_1.copy()
  column_names = df.columns
  y = df.iloc[:,-1].to_numpy()    # Target Column
  x = df.iloc[:,:-1].to_numpy()     # Feature Column
  
  return x,y,column_names

def model(x,y,column_names):
  # split data, keeping class balance in train/test
  x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=2/10,random_state=120,shuffle=True,stratify=y
  )

  # Random forest for final classification
  rf_model = RandomForestClassifier(
    n_estimators=150,max_depth=10,random_state=104,class_weight="balanced",n_jobs=-1)
  lda = LDA(n_components=2)    
  # There are only 2 possible values for n_components since there are only 3 classes.  
  # n_estimaors=2 gave the best score. So I kept it for the final version
  # preprocessing: impute, scale, then reduce dimensionality
  preprocessor = Pipeline([
    ("imputation",SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lda",lda)
  ])
  # full pipeline: preprocessing followed by the classifier
  pipe = Pipeline([
    ("preprocessor",preprocessor),
    ("model",rf_model)
  ])
  pipe.fit(x_train,y_train)

  # evaluate on the held-out test set
  y_true = y_test
  y_pred = pipe.predict(x_test)
  print(classification_report(y_true,y_pred))

  return pipe,column_names

def dumping(pipe,column_names):
  # ensure models directory exists and save artifacts
  os.makedirs("models",exist_ok=True)
  joblib.dump(pipe, "models/pipe.pkl")
  joblib.dump(column_names, "models/column_names.pkl")

def main():
  x,y,column_names = data_collection()
  pipe,clmn_names = model(x,y,column_names)
  dumping(pipe,clmn_names)

if __name__ == "__main__":
  main()
