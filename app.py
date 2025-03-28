from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load trained model, scaler, encoders, and feature names
model = joblib.load("model/stroke_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_names = joblib.load("model/feature_names.pkl")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/features', methods=['GET'])
def features():
    return render_template('features.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        user_input = {
            "Gender": request.form['gender'],
            "Age": float(request.form['age']),
            "Hypertension": int(request.form['hypertension']),
            "Heart Disease": int(request.form['heart_disease']),
            "Marital Status": request.form['marital_status'],
            "Residence Type": request.form['residence_type'],
            "Average Glucose Level": float(request.form['avg_glucose_level']),  # Corrected key
            "Body Mass Index (BMI)": float(request.form['bmi']),  # Corrected key
            "Smoking Status": request.form['smoking_status'],
            "Alcohol Intake": request.form['alcohol_intake'],
            "Physical Activity": request.form['physical_activity'],
            "Stroke History": int(request.form.get('stroke_history', 0)),  # Default to 0 if not provided
            "Family History of Stroke": int(request.form.get('family_history', 0))  # Default to 0 if not provided
        }
        
        # Encode categorical variables
        for col in label_encoders:
            if col in user_input:
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        # Convert input to a DataFrame
        input_df = pd.DataFrame([user_input])

        # Reorder columns to match the training data
        input_df = input_df[feature_names]  # Use saved feature names
        
        # Standardize numerical values
        input_df_scaled = scaler.transform(input_df)

        # Predict using the model
        prediction = model.predict(input_df_scaled)
        result = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"

        return render_template('result.html', prediction=result, user_input=user_input)
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
