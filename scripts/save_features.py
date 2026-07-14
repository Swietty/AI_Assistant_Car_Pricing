import pandas as pd
import joblib


df = pd.read_csv(
    "data/cars_ml_onehot.csv"
)


X = df.drop(
    columns=[
        "price_in_euro",
        "price_log"
    ]
)


joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)


print("Feature columns saved!")