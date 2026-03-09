from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
  
  # Use dark background style
  plt.style.use('dark_background')
  
  # Theme colors matching the frontend CSS
  BG_COLOR = '#0a0a12'        # --bg-secondary from CSS
  CARD_BG = '#12121c'         # --bg-tertiary from CSS
  ACCENT_PURPLE = '#7000ff'   # --accent-primary from CSS
  ACCENT_CYAN = '#00d4ff'     # --accent-secondary from CSS
  TEXT_PRIMARY = '#ffffff'    # --text-primary from CSS
  TEXT_SECONDARY = '#8b8b9e'  # --text-secondary from CSS
  GALAXY_COLOR = '#7000ff'
  STAR_COLOR = '#00d4ff'
  QSO_COLOR = '#ff6b6b'
  
  # Create figure with custom sizing
  fig, ax = plt.subplots(figsize=(10, 8))
  fig.patch.set_facecolor(BG_COLOR)
  ax.set_facecolor(CARD_BG)
  
  # Create custom colormap: from dark purple to bright cyan/purple
  # This matches the gradient in the frontend: #7000ff to #00d4ff
  colors = [
    '#0a0a12',  # Dark background
    '#1a0a2e',  # Very dark purple
    '#2d1066',  # Dark purple
    '#4a1a99',  # Purple
    '#6b25cc',  # Medium purple
    '#8030e0',  # Purple
    '#00a8cc',  # Cyan-ish
    '#00c8e6',  # Cyan
    '#00e8ff',  # Bright cyan
  ]
  cmap = mcolors.LinearSegmentedColormap.from_list('cosmo', colors, N=256)
  
  # Create heatmap with custom colormap
  sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap=cmap,
    xticklabels=labels, 
    yticklabels=labels,
    ax=ax,
    annot_kws={
      'fontsize': 18,
      'fontweight': 'bold',
      'color': TEXT_PRIMARY
    },
    cbar_kws={
      'label': 'Count',
      'shrink': 0.8
    },
    linewidths=3,
    linecolor=BG_COLOR,
    square=True,
    vmin=0,
    vmax=cm.max()
  )
  
  # Customize labels
  ax.set_xlabel('Predicted Classification', fontsize=14, fontweight='bold', color=TEXT_SECONDARY, labelpad=15)
  ax.set_ylabel('True Classification', fontsize=14, fontweight='bold', color=TEXT_SECONDARY, labelpad=15)
  ax.set_title('CosmoClassifier - Confusion Matrix\nSDSS DR18 Test Set Performance', 
               fontsize=18, fontweight='bold', color=TEXT_PRIMARY, pad=25)
  
  # Style the tick labels with class-specific colors
  ax.tick_params(axis='x', colors=TEXT_SECONDARY, labelsize=12, rotation=0)
  ax.tick_params(axis='y', colors=TEXT_SECONDARY, labelsize=12, rotation=0)
  
  # Style x-axis labels with class colors
  for i, label in enumerate(ax.get_xticklabels()):
    if labels[i] == 'GALAXY':
      label.set_color(GALAXY_COLOR)
    elif labels[i] == 'STAR':
      label.set_color(STAR_COLOR)
    elif labels[i] == 'QSO':
      label.set_color(QSO_COLOR)
  
  # Style y-axis labels with class colors  
  for i, label in enumerate(ax.get_yticklabels()):
    if labels[i] == 'GALAXY':
      label.set_color(GALAXY_COLOR)
    elif labels[i] == 'STAR':
      label.set_color(STAR_COLOR)
    elif labels[i] == 'QSO':
      label.set_color(QSO_COLOR)
  
  # Style the spines
  for spine in ax.spines.values():
    spine.set_color(BG_COLOR)
    spine.set_linewidth(3)
  
  # Style colorbar
  cbar = ax.collections[0].colorbar
  cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=11)
  cbar.set_label('Count', color=TEXT_SECONDARY, fontsize=12, labelpad=10)
  cbar.outline.set_edgecolor(BG_COLOR)
  cbar.outline.set_linewidth(2)
  
  # Adjust layout
  plt.tight_layout()
  
  # Save with theme colors
  plt.savefig("static/confusion_matrix.png", dpi=150, facecolor=BG_COLOR, edgecolor='none', bbox_inches='tight')
  plt.close()
  print("Saved and closed confusion matrix ✅")

def main() -> None:
  x,y,column_names = data_collection()
  plotting(x,y)

if __name__ == "__main__":
  main()
