# 🚗 AI Assistant Car Pricing

Machine Learning application for car price estimation and damage classification.

## Features

- 🚘 Car price prediction
- 🔍 Damage detection from images
- 🤖 Deep Learning CNN classifier
- 🌐 Streamlit web application


## Models

### Price prediction
Machine Learning regression model.

Input:
- brand
- model
- year
- mileage
- engine
- fuel type


### Damage detection

CNN model detects:

- Dent
- Scratch
- Crack
- Glass shatter
- Lamp broken
- Flat tire


## Technologies

Python

- TensorFlow
- Scikit-learn
- Pandas
- Streamlit


## Run locally


Install dependencies:
'''
pip install -r requirements.txt

'''
Start application:
'''
python -m streamlit run app\app.py

'''

## Author

Svitlana Ihnat

AI_Assistant_Car_Pricing/  

│  
├── app/  
│   │  
│   ├── app.py                    ← главное приложение
│   ├── price_prediction.py       ← работа с Random Forest
│   ├── damage_prediction.py      ← работа с CNN
│   ├── damage_rules.py           ← расчет снижения цены
│     
│  
├── models/  
│   ├── price_model.pkl  
│   └── damage_cnn.keras  
│  
├── data/  
│   └── cars_ml_onehot.csv  
│  
├── scripts/
│   └── random_forest_price.py  
    ├── train_damage_model.py  
│   └── predict_damage.py  
│
└── ml_notebooks/
    ├── damage_detection.ipynb
    ├── damage_model_evaluation.ipynb
    ├──

