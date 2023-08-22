# 1. Imports
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import LocalOutlierFactor

def train_lof_model(X_train, contamination=0.0025):
    """
    Train the Local Outlier Factor (LOF) model.

    Parameters:
    - X_train: The training data.
    - contamination: The proportion of outliers in the data set.

    Returns:
    - A trained LOF model.
    """
    lof = LocalOutlierFactor(novelty=True, contamination=contamination)
    lof.fit(X_train)
    return lof


def train_model_wf(X_train, y_train):
    # Create a pipeline with StandardScaler and SVC
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        # ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('svc', SVC(max_iter=1000000))
    ])
    pipe.fit(X_train, y_train)

    return pipe

def evaluate_model_wf(model, X_test, y_test):
    predictions = model.predict(X_test)
    return confusion_matrix(y_test, predictions)


