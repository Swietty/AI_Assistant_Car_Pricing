import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# 1. Daten laden
# ==========================

df = pd.read_csv(
    "data/cars_ml_onehot.csv"
)

print("Dataset geladen:")
print(df.shape)

# ==========================
# 2. Features und Target
# ==========================

X = df.drop(
    columns=[
        "price_in_euro",
        "price_log"
    ]
)

y = df["price_log"]


# ==========================
# 3. Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train samples:", X_train.shape)
print("Test samples:", X_test.shape)


# ==========================
# 4. Random Forest Modell
# ==========================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


print("Training gestartet...")

rf.fit(
    X_train,
    y_train
)


print("Training abgeschlossen.")

# ==========================
# 5. Modell Evaluation
# ==========================

pred_log = rf.predict(
    X_test
)

# Zurück in Euro umwandeln
pred_price = np.exp(pred_log)
true_price = np.exp(y_test)

mae = mean_absolute_error(
    true_price,
    pred_price
)

rmse = np.sqrt(
    mean_squared_error(
        true_price,
        pred_price
    )
)

r2 = r2_score(
    true_price,
    pred_price
)

print("\nModel Evaluation")
print("-----------------------")
print(f"MAE : {mae:.2f} €")
print(f"RMSE: {rmse:.2f} €")
print(f"R²  : {r2:.4f}")

# ==========================
# 6. Modell speichern
# ==========================

model_path = "models/price_model.pkl"

joblib.dump(
    rf,
    model_path
)

print("\nModell gespeichert:")
print(model_path)