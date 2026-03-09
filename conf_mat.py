from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

file_path = "Datasets/SDSS_DR18.csv"
model = joblib.load("models/estimator.pkl")


def data_collection(file_path = file_path) -> np.ndarray:
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

def plotting(x,y) -> None:
  x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=2/10,random_state=120,shuffle=True,stratify=y
  )
  labels= ["GALAXY","STAR","QSO"]
  y_true = y_test
  y_pred = model.predict(x_test)
  cm = confusion_matrix(y_true,y_pred)
  plt.figure(figsize=(10, 7))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.savefig("static/confusion_matrix.png")
  plt.close()
  print("Saved and closed confusion matrix ✅")

def main() -> None:
  x,y,column_names = data_collection()
  plotting(x,y)

if __name__ == "__main__":
  main()

