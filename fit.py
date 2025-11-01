from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import train_test_split,learning_curve,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import time
import os
import numpy as np
import pandas as pd
import joblib

def data_collection(file_path = "Datasets/SDSS_DR18.csv") -> np.ndarray:
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

  # Feature Reduction
  df_2 = df_1[["ra","dec","redshift","u","g","r","i","z","psfMag_r","class"]].copy()
  
  # Feature Engineering color contrast columns
  df_2["u_g_color"] = df_2["u"] - df_2["g"]
  df_2["g_r_color"] = df_2["g"] - df_2["r"]
  df_2["r_i_color"] = df_2["r"] - df_2["i"]
  df_2["i_z_color"] = df_2["i"] - df_2["z"]
  df_2 = df_2.drop(columns=["u","g","r","i","z"])

  # Moving the `class` column to the end
  popped_class = df_2.pop("class")
  df_2.insert(len(df_2.columns), "class", popped_class)

  # finalize DataFrame and split into features and target
  df = df_2.copy()
  column_names = df.columns.to_numpy()
  y = df.iloc[:,-1].to_numpy()    # Target Column
  x = df.iloc[:,:-1].to_numpy()     # Feature Column
  
  return x,y,column_names

def model(x,y) -> BaseEstimator:
  # split data, keeping class balance in train/test
  x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=2/10,random_state=120,shuffle=True,stratify=y
  )

  # RF, SVC, LR, XGB
  rf_model = RandomForestClassifier(random_state=40)
  lr_model = LogisticRegression(random_state=42,max_iter=10_000)
  xgb_model = XGBClassifier(random_state=43)

  pca = PCA(random_state=44)
  lda = LDA(n_components=2)

  pipe = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scale",StandardScaler()),
    ("smote",SMOTE(random_state=101)),
    ("dimen",pca),
    ("model",rf_model)
  ])
  param_list = [
    { # Random Forest, PCA On
      "model": [rf_model],"model__n_estimators":np.arange(150,650,100),
      "model__max_depth":np.arange(7,14,2), "dimen" : [pca], "dimen__n_components": np.arange(5,8,1)
    },
    { # Logistic Regression, No dimen. reduction, l1 penalty, `saga` solver
      "model": [lr_model], "model__C": [0.01,0.1,1,10], "model__penalty":["l1"], "model__solver":["saga"],
      "dimen": ["passthrough"]
    },
    { # Logistic Regression, No dimen. reduction, l2 penalty, `lbfgs` solver
      "model": [lr_model], "model__C": [0.01,0.1,1,10], "model__penalty":["l2"], "model__solver":["lbfgs"],
      "dimen": ["passthrough"]
    },
    { # XGBoost, PCA On
      "dimen": [pca], "dimen__n_components": np.arange(5,8,1),
      "model": [xgb_model], "model__n_estimators" : np.linspace(500,1100,3,dtype=int),"model__learning_rate": [0.01,0.1], "model__max_depth":np.arange(7,14,3)
    },
    { # XGBoost, LDA On
      "dimen": [lda],
      "model": [xgb_model], "model__n_estimators" : [500,700,900],"model__learning_rate": [0.01,0.1], "model__max_depth":np.arange(7,14,3)
    },
    { # XGBoost, No dimen. reduction
      "dimen": ["passthrough"],
      "model": [xgb_model], "model__n_estimators" : [500,700,900],"model__learning_rate": [0.01,0.1], "model__max_depth":np.arange(7,14,3)
    }
  ]

  rscv = RandomizedSearchCV(
    estimator=pipe,param_distributions=param_list,n_iter=8,cv=5,n_jobs=-1,random_state=50,refit=True
  )

  print(f"🤖 Starting Model Training....")
  print(f"‼️ Training may take a lot of time, so please sit tight....")
  t1 = time.time()
  rscv.fit(x_train,y_train)
  t2 = time.time()
  minutes,seconds = np.divmod((t2-t1),60)
  print(f"⌛️ Time Elapsed: {minutes} Minutes {seconds:.2f} Seconds")
  estimator = rscv.best_estimator_
  y_true = y_test
  y_pred = estimator.predict(x_test)
  print(classification_report(y_true,y_pred))

  return estimator

def dumping(pipe,column_names):
  # ensure models directory exists and save artifacts
  try:
    os.makedirs("models",exist_ok=True)
    joblib.dump(pipe, "models/estimator.pkl")
    joblib.dump(column_names, "models/column_names.pkl")
    print(f"Saved models/pipe.pkl and models/column_names.pkl successfully ✅")
  except Exception as e:
    print(f"Something went wrong while dumping. Message: {e}")

def main():
  x,y,column_names = data_collection()
  estimator = model(x,y)
  dumping(estimator,column_names)

if __name__ == "__main__":
  main()
