import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The CosmoClassifier
    In this project, I have used the Data Release 18 version of Sloan Digital Sky Survey (SDSS) dataset to train a classifier algorithm to predict whether the given credentials corresponds to a Galaxy(class 0), Star(class 1) or Quasar(class 2). This notebook is used as a playground to test different hyperparameter settings as well as preprocessing approaches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Importing the libraries
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
        learning_curve,
        np,
        pd,
        plt,
        sns,
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
    Checking for null values
    """)
    return


@app.cell
def _(df_1):
    df_1.isna().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No null values were found, so we are going to skip dropping nulls.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Copying the dataset and specifying the target & feature columns
    """)
    return


@app.cell
def _(df_1):
    df = df_1.copy()
    column_names = df.columns
    y = df.iloc[:,-1].to_numpy()      # Target Column
    x = df.iloc[:,:-1]    # Feature Column
    feature_names = x.columns
    x = x.to_numpy()
    return feature_names, x, y


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
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=2/10,random_state=120,shuffle=True,stratify=y)
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
def _(lda, lr_model, np, pca, rf_model, xgb_model):
    param_list = [
      { # Random Forest, PCA On
        "model": [rf_model],"model__n_estimators":np.arange(150,650,100),
        "model__max_depth":np.arange(7,14,2), "dimen" : [pca], "dimen__n_components":[18,20,22]
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
        "dimen": [pca], "dimen__n_components": [18,20,22],
        "model": [xgb_model], "model__n_estimators" : np.linspace(500,1100,3,dtype=int),"model__learning_rate": [0.01,0.1], "model__max_depth":np.arange(7,14,3)
      },
      { # XGBoost, LDA On
        "dimen": [lda],
        "model": [xgb_model], "model__n_estimators" : [500,700,900],"model__learning_rate": [0.01,0.1], "model__max_depth":np.arange(7,14,3)
      }
    ]
    return (param_list,)


@app.cell
def _(RandomizedSearchCV, param_list, pipe, x_train, y_train):
    rscv = RandomizedSearchCV(
      estimator=pipe,param_distributions=param_list,n_iter=8,cv=5,n_jobs=-1,random_state=50,refit=True
    )

    rscv.fit(x_train,y_train)
    estimator = rscv.best_estimator_
    score = rscv.best_score_
    config = rscv.best_params_
    print(f"Best Configuration:\n{config}")
    print(f"Best Score = {score}")
    return estimator, rscv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, the **XGBoost** algorithm paired with PCA dominated the randomized search with the best score of **0.9875**, which is a great score overall. Now, let's calculate the **Classification Report** to find out other metrices.
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
    #### PCA Loading heatmap
    """)
    return


@app.cell
def _(estimator, feature_names, np, pd):
    pca_1 = estimator.named_steps['dimen']
    comp_df = pd.DataFrame(pca_1.components_, index=[f'PC{i + 1}' for i in range(pca_1.n_components_)], columns=feature_names)
    exp_var = pca_1.explained_variance_

    def mod(x):
        return x * np.sqrt(exp_var)
    loading_df = comp_df.copy().apply(mod)
    loading_df.head(3)
    return (loading_df,)


@app.cell
def _(loading_df, plt, sns):
    plt.figure(figsize=(19,9))
    sns.heatmap(loading_df,cmap="icefire",annot=True,fmt=".1f",center=0)
    plt.ylabel("Principle Components",fontdict={"fontsize":14})
    plt.title("Heatmap of PCA Loadings",fontdict={"fontsize":19})
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This heatmap visualizes the PCA loadings for 20 Principal Components (PCs) across numerous features. PC1 and PC2 are primarily defined by the spectral bands (g, r, i etc.) and flux values, capturing the bulk of the variance. Loadings quickly drop to near-zero by PC11, indicating that the first 10-12 components are sufficient for dimension reduction and interpretation in this dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Learning Curve
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's plot the **Learning Curve** plot to understand whether the model is fit well or is underfitting/overfitting.
    """)
    return


@app.cell
def _(estimator, learning_curve, np, x_train, y_train):
    train_size, train_acc, val_acc = learning_curve(
      estimator,x_train,y_train,train_sizes=np.linspace(0.1,1.0,10),
      cv=5,n_jobs=-1,random_state=9,shuffle=True
    )
    return train_acc, train_size, val_acc


@app.cell
def _(np, plt, train_acc, train_size, val_acc):
    train_mean = np.mean(train_acc, axis=1)
    train_std = np.std(train_acc,axis=1)
    val_mean = np.mean(val_acc,axis=1)
    val_std = np.std(val_acc,axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(train_size, train_mean, color="red",marker="s",markersize=4,label="Training Accuracy")
    plt.fill_between(train_size, train_mean + train_std , train_mean - train_std, color="red",alpha=0.3)

    plt.plot(train_size, val_mean, color="orange",marker="v",markersize=4,label="Validation Accuracy")
    plt.fill_between(train_size, val_mean + val_std, val_mean - val_std, color="orange",alpha=0.3)

    plt.title("Learning Curve (Random Forest with PCA)",fontdict={"fontsize":16})
    plt.xlabel("Train Size",fontdict={"fontsize":13})
    plt.ylabel("Accuracy",fontdict={"fontsize":13})
    plt.ylim(0.85,1.03)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This learning curve for a Random Forest model after applying PCA shows excellent performance, with both Training Accuracy (red) and Validation Accuracy (orange) consistently high, clustering around 0.98. Crucially, the minimal gap between the two curves indicates that the model exhibits low variance and is not significantly overfitting the data, even at smaller training sizes (around 10,000 samples). The validation accuracy appears to have converged, suggesting that further increases in the training set size are unlikely to yield substantial improvements in generalization performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    In this notebook, we have performed proper tests to develop the best possible model for our classification task. We will take the code from here to built `fit.py` script from the ground up.

    Check out `research_2.ipynb` file to see the feature reductions I tried out. It is a crucial step to make the app realistic and actually useful.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
