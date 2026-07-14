import streamlit as st
import pandas as pd
import numpy as np

from datetime import datetime
from damage_rules import calculate_damage_discount
import joblib

from tensorflow.keras.models import load_model
from PIL import Image

# ==========================
# Einstellungen
# ==========================

st.set_page_config(
    page_title="Fahrzeugpreis Schätzung",
    page_icon="🚗"
)

# ==========================
# Daten laden
# ==========================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/cleaned_cars_2.csv"
    )

@st.cache_data
def load_training_columns():

    train_df = pd.read_csv(
        "data/cars_ml_onehot.csv",
        nrows=1
    )
    return [
        col for col in train_df.columns
        if col not in [
            "price_in_euro",
            "price_log"
        ]
    ]


# ==========================
# Modelle laden
# ==========================

@st.cache_resource
def load_models():

    price_model = joblib.load(
        "models/price_model.pkl"
    )

    damage_model = load_model(
        "models/damage_classifier_v2.keras",
        compile=False
    )

    return price_model, damage_model


df = load_data()

model_columns = load_training_columns()

price_model, damage_model = load_models()


if not hasattr(price_model, "predict"):

    st.error(
        "damage_classifier_v2.keras ist keine ML-Modell-Datei"
    )

    st.stop()


# ==========================
# Funktionen
# ==========================

def predict_damage(image):
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = damage_model.predict(
        img_array,
        verbose=0
    )

    probabilities = prediction[0]

    class_id = np.argmax(
        probabilities
    )

    confidence = float(
        probabilities[class_id]
    )

    label_map = {

        0:"Crack",
        1:"Scratch",
        2:"Flat tire",
        3:"Dent",
        4:"Glass shatter",
        5:"Lamp broken"

    }


    damage_type = label_map[class_id]

    return damage_type, confidence

def create_price_input(
        brand,
        model,
        color,
        fuel,
        transmission,
        power_kw,
        fuel_consumption,
        co2,
        mileage,
        car_age
):


    X = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns
    )

    # numerical features
    X["power_kw"] = power_kw
    X["fuel_consumption_l_100km"] = (
        fuel_consumption
    )
    X["co2_emissions_g_km"] = co2
    X["mileage_in_km"] = mileage
    X["car_age"] = car_age
    X["has_fuel_consumption"] = 1
    X["has_co2"] = 1
    X["is_electric"] = (
        1
        if fuel == "Electric"
        else 0
    )

    X["model_frequency"] = (
        df["model"]
        .value_counts()
        .get(model,0)
    )

    # one hot encoding
    categories = {
        "brand_": brand,
        "fuel_type_": fuel,
        "color_": color,
        "transmission_type_": transmission
    }

    for prefix,value in categories.items():
        column = (
            prefix
            +
            value.lower()
        )
        if column in X.columns:
            X[column] = 1
    return X

# ==========================
# Listen
# ==========================

brands = sorted(
    df["brand"]
    .dropna()
    .unique()
)

colors = sorted(
    df["color"]
    .dropna()
    .unique()
)

transmissions = sorted(
    df["transmission_type"]
    .dropna()
    .unique()
)

fuels = sorted(
    df["fuel_type"]
    .dropna()
    .unique()
)

models_by_brand = (

    df[
        [
            "brand",
            "model"
        ]
    ]
    .dropna()
    .drop_duplicates()
    .groupby("brand")["model"]
    .apply(
        lambda x:
        sorted(
            x.unique()
        )
    )
    .to_dict()
)

technical_data = (
    df.groupby(
        [
            "brand",
            "model"
        ]
    )
    [
        [
            "power_kw",
            "fuel_consumption_l_100km",
            "co2_emissions_g_km"
        ]
    ]
    .mean()
)

# ==========================
# UI
# ==========================

st.title(
    "🚗 Fahrzeugpreis Schätzung"
)

st.write(
    "Geben Sie die Fahrzeugdaten ein:"
)

brand = st.selectbox(
    "Marke",
    brands
)

model = st.selectbox(
    "Modell",
    models_by_brand.get(
        brand,
        []
    )
)

color = st.selectbox(
    "Farbe",
    colors
)

year = st.number_input(
    "Baujahr",
    min_value=1980,
    max_value=datetime.now().year,
    value=2020
)

mileage = st.number_input(
    "Kilometerstand",
    min_value=0,
    value=80000
)

transmission = st.selectbox(
    "Getriebe",
    transmissions
)

fuel = st.selectbox(
    "Kraftstoff",
    fuels
)

# ==========================
# Technische Daten
# ==========================

try:
    defaults = technical_data.loc[
        (brand,model)
    ]

except:
    defaults = (
        df[df["brand"] == brand]
        [
            [
                "power_kw",
                "fuel_consumption_l_100km",
                "co2_emissions_g_km"
            ]
        ]
        .mean()
    )

default_power = int(
    defaults["power_kw"]
)

default_fuel = round(
    defaults["fuel_consumption_l_100km"],
    1
)

default_co2 = round(
    defaults["co2_emissions_g_km"],
    1
)

manual = st.checkbox(
    "Technische Daten manuell eingeben"
)

if manual:
    with st.expander(
        "Zusätzliche technische Daten",
        expanded=True
    ):

        power_kw = st.number_input(
            "Leistung (kW)",
            value=default_power
        )

        fuel_consumption = st.number_input(
            "Verbrauch (l/100 km)",
            value=default_fuel
        )

        co2 = st.number_input(
            "CO₂-Emissionen (g/km)",
            value=default_co2
        )

else:
    power_kw = default_power
    fuel_consumption = default_fuel
    co2 = default_co2
    st.info(
        """
Technische Daten wurden automatisch
für das ausgewählte Fahrzeug übernommen.
"""
    )

car_age = (
    datetime.now().year
    -
    year
)

# ==========================
# Schaden
# ==========================

damage = st.radio(
    "Hat das Fahrzeug Schäden?",
    [
        "Nein",
        "Ja"
    ]
)

image = None

if damage == "Ja":
    image = st.file_uploader(
        "Schadenfoto hochladen",
        type=[
            "jpg",
            "png",
            "jpeg"
        ]
    )

# ==========================
# Prediction
# ==========================

if st.button("Preis berechnen"):

    damage_score = None
    damage_percent = 0

    X = create_price_input(
        brand,
        model,
        color,
        fuel,
        transmission,
        power_kw,
        fuel_consumption,
        co2,
        mileage,
        car_age
    )

    predicted_log_price = float(
        price_model.predict(X)[0]
    )

    predicted_price = round(
        np.exp(predicted_log_price)
    )

    # ======================
    # CNN Schaden erkennen
    # ======================

    damage_type = None
    confidence = None
    damage_percent = 0


    if image is not None:

        st.image(
            image,
            caption="Hochgeladenes Schadenfoto",
            width=300
        )

        damage_type, confidence = predict_damage(
            image
        )

        st.write(
            f"Schaden erkannt: {damage_type}"
        )

        st.write(
            f"Wahrscheinlichkeit: {confidence:.2%}"
        )

        # Rule Based Damage Calculation
        damage_percent = calculate_damage_discount(

            brand,
            car_age,
            mileage,
            damage_type,
            confidence
        )

        st.write(
            f"Wertminderung durch Schaden: {damage_percent}%"
        )

    # ======================
    # Endpreis
    # ======================

    final_price = round(
        predicted_price *
        ( 1 -
            damage_percent / 100
        )
    )

    st.success(
        "Preis erfolgreich berechnet"
    )

    st.metric(
        "Geschätzter Fahrzeugpreis",
        f"{predicted_price:,} €"
    )

    if damage_percent > 0:
        st.warning(
            f"""
Schaden erkannt

Typ:
{damage_type}

Wahrscheinlichkeit:
{confidence:.2%}

Wertminderung:
{damage_percent} %

Endpreis:
{final_price:,} €
"""
        )

    else:
        st.info(
            f"""
Keine Wertminderung

Endpreis:
{final_price:,} €
"""
        )