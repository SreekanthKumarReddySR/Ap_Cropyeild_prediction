import __main__
import json
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
from Model3 import EnhancedRandomForestRegressor
from Model3 import EnhancedDecisionTreeRegressor

# The bundled pickle was created when Model3.py was run as a script, so the
# stored class path points to __main__. Re-register the classes before loading.
__main__.EnhancedRandomForestRegressor = EnhancedRandomForestRegressor
__main__.EnhancedDecisionTreeRegressor = EnhancedDecisionTreeRegressor

# Load the saved model and label encoders
with open('enhanced_random_forest_regressor.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

DEFAULT_FEATURE_COLUMNS = [
    'Crop',
    'District',
    'Season',
    'Area',
    'Annual_Temp',
    'Fertilizer',
    'Annual_Rainfall',
    'Rainfall_Fertilizer',
]


def load_feature_columns():
    metadata_path = Path('model_metadata.json')
    if not metadata_path.exists():
        return DEFAULT_FEATURE_COLUMNS

    with metadata_path.open('r', encoding='utf-8') as metadata_file:
        metadata = json.load(metadata_file)
    return metadata.get('feature_columns', DEFAULT_FEATURE_COLUMNS)


FEATURE_COLUMNS = load_feature_columns()

# Preprocessing function
def preprocess_user_input(user_input, label_encoders):
    processed_input = {
        'Crop': label_encoders['Crop'].transform([user_input['crop']])[0],
        'District': label_encoders['District'].transform([user_input['district']])[0],
        'Season': label_encoders['Season'].transform([user_input['season']])[0],
        'Area': user_input['area'],
        'Annual_Temp': user_input['annual_temp'],
        'Fertilizer': user_input['fertilizer'],
        'Annual_Rainfall': user_input['rainfall'],
    }
    processed_input['Rainfall_Fertilizer'] = (
        processed_input['Annual_Rainfall'] * processed_input['Fertilizer']
    )

    final_input = pd.DataFrame([[processed_input[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return final_input

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'crop': request.form['crop'],
            'district': request.form['district'],
            'season': request.form['season'],
            'area': float(request.form['area']),
            'annual_temp': float(request.form['annual_temp']),
            'fertilizer': float(request.form['fertilizer']),
            'rainfall': float(request.form['rainfall']),
        }
        processed_input = preprocess_user_input(user_input, label_encoders)

        # Predict yield using the loaded model
        yield_prediction = rf_model.predict(processed_input)

        # Render result
        return render_template('index.html', prediction=yield_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)

    
