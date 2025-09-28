import pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "model.pkl"

def preprocess_data(data_path="data/framingham.csv"):
    df = pd.read_csv(data_path)
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns)

    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]
    return X, y

def train_model(data_path="data/framingham.csv"):
    X, y = preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, X_test, y_test

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_proba(features: list):
    model = load_model()
    X = np.array(features).reshape(1, -1)
    return float(model.predict_proba(X)[0][1])  # probability of heart disease
