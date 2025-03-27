from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and encoders
model = joblib.load("model/stroke_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

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
            "Marital Status": request.form['ever_married'],
            "Residence Type": request.form['Residence_type'],
            "Avg Glucose Level": float(request.form['avg_glucose_level']),
            "BMI": float(request.form['bmi']),
            "Smoking Status": request.form['smoking_status'],
            "Alcohol Intake": request.form.get('alcohol_intake', "No"),  # Default "No"
            "Physical Activity": request.form.get('physical_activity', "Low")  # Default "Low"
        }
        
        # Encode categorical variables
        for col in label_encoders:
            if col in user_input:
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        # Convert input to a DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Standardize numerical values
        input_df_scaled = scaler.transform(input_df)

        # Predict using the model
        prediction = model.predict(input_df_scaled)
        result = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
