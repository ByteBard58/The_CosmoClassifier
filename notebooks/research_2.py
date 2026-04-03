import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *This notebook is a continued version of `research.py` notebook. Please read that first. You can open the `research.html` file to do that too.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the time of writing this, I have already trained the model, developed the front-end and the web app. Now, I have realized that putting 33 feature columns as input is a bad call. This will be a fraustrating experience for any user. So, in this notebook, I am going to reduce some features and feature-enginner if necessary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Importing modules**
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier,VotingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from sklearn.model_selection import train_test_split,learning_curve,RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns
    return (
        LDA,
        LogisticRegression,
        PCA,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SMOTE,
        SVC,
        SimpleImputer,
        StandardScaler,
        XGBClassifier,
        classification_report,
        np,
        pd,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Importing the dataset
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("Datasets/SDSS_DR18.csv")
    df_raw.columns
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dropping the identifier columns which may lead to data leakage
    """)
    return


@app.cell
def _(df_raw):
    df_raw_1 = df_raw.drop(columns=['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid'])
    df_raw_1.columns
    return (df_raw_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Identifying and mapping the classes
    """)
    return


@app.cell
def _(df_raw_1):
    print(df_raw_1['class'].value_counts())
    df_1 = df_raw_1.copy()
    return (df_1,)


@app.cell
def _(df_1):
    df_1["class"] = df_1["class"].map({
      "GALAXY":0,
      "STAR":1,
      "QSO":2
    })
    df_1["class"].head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Reduction and Engineering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dropping all raw fluxes, all concentration indices, all radii, all error terms, and all duplicate magnitudes. I have copied a subset of features from `df_1` to `df_2`. This subset excludes all of those features.
    """)
    return


@app.cell
def _(df_1):
    df_2 = df_1[["ra","dec","redshift","u","g","r","i","z","psfMag_r","class"]].copy()
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Feature engineering color contrast columns
    """)
    return


@app.cell
def _(df_2):
    df_2["u_g_color"] = df_2["u"] - df_2["g"]
    df_2["g_r_color"] = df_2["g"] - df_2["r"]
    df_2["r_i_color"] = df_2["r"] - df_2["i"]
    df_2["i_z_color"] = df_2["i"] - df_2["z"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dropping the raw color features
    """)
    return


@app.cell
def _(df_2):
    df_2.head(2)
    return


@app.cell
def _(df_2):
    df_2_1 = df_2.drop(columns=['u', 'g', 'r', 'i', 'z'])
    return (df_2_1,)


@app.cell
def _(df_2_1):
    df_2_1.head(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Moving the class column to the end
    """)
    return


@app.cell
def _(df_2_1):
    popped_class = df_2_1.pop('class')
    df_2_1.insert(len(df_2_1.columns), 'class', popped_class)
    return


@app.cell
def _(df_2_1):
    df_2_1.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Copying `df_2` into the main dataframe and separating feature and target columns
    """)
    return


@app.cell
def _(df_2_1):
    df = df_2_1.copy()
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    feature_columns = df.columns[:-1]
    feature_columns
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ML Preprocessing, Model training, Evaluation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Performing train-test split
    """)
    return


@app.cell
def _(train_test_split, x, y):
    x_train,x_test,y_train,y_test = train_test_split(
      x,y,test_size=2/10,random_state=120,shuffle=True,stratify=y)
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Defining the pipeline and the model
    """)
    return


@app.cell
def _(
    LDA,
    LogisticRegression,
    PCA,
    Pipeline,
    RandomForestClassifier,
    SMOTE,
    SVC,
    SimpleImputer,
    StandardScaler,
    XGBClassifier,
):
    # RF, SVC, LR, XGB
    rf_model = RandomForestClassifier(random_state=40)
    svc_model = SVC(random_state=41)
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
    return lda, lr_model, pca, pipe, rf_model, xgb_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Performing Randomized Search CV
    """)
    return


@app.cell
def _(
    RandomizedSearchCV,
    lda,
    lr_model,
    np,
    pca,
    pipe,
    rf_model,
    x_train,
    xgb_model,
    y_train,
):
    param_list = [
      { # Random Forest, PCA On
        "model": [rf_model],"model__n_estimators":np.arange(150,650,100),
        "model__max_depth":np.arange(7,14,2), "dimen" : [pca], "dimen__n_components": np.arange(5,8,1)
      },

      # { # SVC, No dimen. reduction
      #   "model": [svc_model], "model__C":[0.01,0.1,1,10], "model__kernel":["rbf"], "model__gamma":[0.01,0.1,1,10],
      #   "dimen":["passthrough"]
      # },

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

    rscv.fit(x_train,y_train)
    estimator = rscv.best_estimator_
    score = rscv.best_score_
    config = rscv.best_params_
    print(f"Best Configuration:\n{config}")
    print(f"Best Score = {score}")
    return (rscv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's calculate the **Classification Report**
    """)
    return


@app.cell
def _(classification_report, rscv, x_test, y_test):
    y_true = y_test
    y_pred = rscv.predict(x_test)
    print(classification_report(y_true=y_true,y_pred=y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, the overall accuracy decreased about 1% from before. But it is a huge win considering the simplicity we have successfully added by performing the feature reduction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we have explored how we can make the model suitable for user usage by reducing the number of features. We have seen that it has made the model a lot simplier by only sacrificing 1% accuracy.

    We will use the code from this notebook to write `fit.py` from the ground up.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
