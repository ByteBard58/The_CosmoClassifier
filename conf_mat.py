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
  
  # Set the style to match the professional theme
  plt.style.use('dark_background')
  
  # Create figure with custom sizing
  fig, ax = plt.subplots(figsize=(10, 8))
  fig.patch.set_facecolor('#12121c')
  ax.set_facecolor('#12121c')
  
  # Custom colormap matching the theme (purple to cyan gradient)
  colors = ['#1a1a2e', '#2d1b4e', '#3d2075', '#4c2a96', '#5c35a8', 
            '#6b40ba', '#7a4bcc', '#8956de', '#9861f0', '#7000ff']
  cmap = sns.color_palette(colors, as_cmap=True)
  
  # Create heatmap with theme colors
  sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=labels, 
    yticklabels=labels,
    ax=ax,
    annot_kws={
      'fontsize': 16,
      'fontweight': 'bold',
      'color': '#ffffff'
    },
    cbar_kws={
      'label': 'Count',
      'shrink': 0.8
    },
    linewidths=2,
    linecolor='rgba(255,255,255,0.1)',
    square=True
  )
  
  # Customize labels
  ax.set_xlabel('Predicted Classification', fontsize=14, fontweight='bold', color='#8b8b9e', labelpad=10)
  ax.set_ylabel('True Classification', fontsize=14, fontweight='bold', color='#8b8b9e', labelpad=10)
  ax.set_title('CosmoClassifier - Confusion Matrix\nSDSS DR18 Test Set Performance', 
               fontsize=16, fontweight='bold', color='#ffffff', pad=20)
  
  # Style the tick labels
  ax.tick_params(axis='both', colors='#8b8b9e', labelsize=12)
  
  # Add subtle grid
  for spine in ax.spines.values():
    spine.set_color('rgba(255,255,255,0.1)')
    spine.set_linewidth(1)
  
  # Style colorbar
  cbar = ax.collections[0].colorbar
  cbar.ax.tick_params(colors='#8b8b9e')
  cbar.set_label('Count', color='#8b8b9e', fontsize=12)
  
  # Adjust layout
  plt.tight_layout()
  
  # Save with theme colors
  plt.savefig("static/confusion_matrix.png", dpi=150, facecolor='#12121c', edgecolor='none', bbox_inches='tight')
  plt.close()
  print("Saved and closed confusion matrix ✅")

def main() -> None:
  x,y,column_names = data_collection()
  plotting(x,y)

if __name__ == "__main__":
  main()
