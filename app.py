from flask import Flask, render_template, request, make_response
import numpy as np
import joblib
import pandas as pd
from xhtml2pdf import pisa
from io import BytesIO
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Gemini API using .env variable
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

app = Flask(__name__)

# Load trained model, scaler, encoders, and feature names
model = joblib.load("model/stroke_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# Helper function for activity level scoring
def get_activity_level_score(activity_level):
    try:
        levels = ['Low', 'Moderate', 'High']
        if activity_level in levels:
            return 1 - (levels.index(activity_level) / 2)
        return 0.5
    except Exception as e:
        print(f"Error calculating activity score: {str(e)}")
        return 0.5

# Clean and format Gemini response
def process_section_content(content):
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            line = line.replace('•', '').replace('*', '').strip()
            line = line.lstrip('123456789.)-').strip()
            line = line.replace('**', '').replace('__', '')
            if not line.startswith('•'):
                line = f'• {line}'
            formatted_lines.append(line)
    return '<br>'.join(formatted_lines)

# Generate recommendations from Gemini
def get_recommendations(user_input, prediction_result):
    try:
        prompt = f"""
        Based on this patient's profile and stroke risk assessment, provide specific recommendations in these categories. Use bullet points (•) for each recommendation:

        Patient Profile:
        - Age: {user_input['Age']} years
        - BMI: {user_input['Body Mass Index (BMI)']}
        - Hypertension: {'Present' if user_input['Hypertension'] == 1 else 'Not Present'}
        - Heart Disease: {'Present' if user_input['Heart Disease'] == 1 else 'Not Present'}
        - Physical Activity: {user_input['Physical Activity']} level
        - Smoking: {user_input['Smoking Status']}
        - Alcohol: {user_input['Alcohol Intake']}

        Risk Assessment: {prediction_result}

        Please provide recommendations in exactly this format:

        Physical Activities:
        • [specific exercise recommendation]
        • [frequency and duration]
        • [intensity level]
        • [additional activity suggestions]

        Dietary Guidelines:
        • [specific food recommendations]
        • [foods to avoid]
        • [meal timing]
        • [portion control]

        Lifestyle Changes:
        • [daily habits]
        • [stress management]
        • [sleep recommendations]
        • [social activities]

        Prevention Measures:
        • [regular check-ups]
        • [monitoring requirements]
        • [preventive measures]
        • [warning signs to watch]
        """

        response = gemini_model.generate_content(prompt)
        text = response.text

        recommendations = {
            'activities': '',
            'diet': '',
            'lifestyle': '',
            'prevention': ''
        }

        sections = text.split('\n\n')
        for section in sections:
            if not section.strip():
                continue
            if 'Physical Activities:' in section:
                content = section.replace('Physical Activities:', '').strip()
                recommendations['activities'] = process_section_content(content)
            elif 'Dietary Guidelines:' in section:
                content = section.replace('Dietary Guidelines:', '').strip()
                recommendations['diet'] = process_section_content(content)
            elif 'Lifestyle Changes:' in section:
                content = section.replace('Lifestyle Changes:', '').strip()
                recommendations['lifestyle'] = process_section_content(content)
            elif 'Prevention Measures:' in section:
                content = section.replace('Prevention Measures:', '').strip()
                recommendations['prevention'] = process_section_content(content)

        print("Generated recommendations:", recommendations)
        return recommendations

    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        error_message = "• Unable to generate recommendations at this time.<br>• Please consult with your healthcare provider."
        return {
            'activities': error_message,
            'diet': error_message,
            'lifestyle': error_message,
            'prevention': error_message
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        original_input = {
            "Gender": request.form['gender'],
            "Age": float(request.form['age']),
            "Hypertension": int(request.form['hypertension']),
            "Heart Disease": int(request.form['heart_disease']),
            "Marital Status": request.form['marital_status'],
            "Residence Type": request.form['residence_type'],
            "Body Mass Index (BMI)": float(request.form['bmi']),
            "Smoking Status": request.form['smoking_status'],
            "Alcohol Intake": request.form['alcohol_intake'],
            "Physical Activity": request.form['physical_activity'],
            "Stroke History": int(request.form.get('stroke_history', 0)),
            "Family History of Stroke": int(request.form.get('family_history', 0))
        }

        user_input = original_input.copy()

        for col in label_encoders:
            if col in user_input:
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_names]
        input_df_scaled = scaler.transform(input_df)

        prediction = model.predict(input_df_scaled)
        prediction_proba = model.predict_proba(input_df_scaled)
        risk_percentage = prediction_proba[0][1] * 100

        predicted_class = int(prediction[0])
        result = "High Risk of Stroke" if predicted_class == 1 else "Low Risk of Stroke"

        recommendations = get_recommendations(original_input, result)

        user_input_str = {k: str(v) for k, v in original_input.items()}

        health_indicators = {
            'Hypertension': float(original_input['Hypertension']),
            'Heart Disease': float(original_input['Heart Disease']),
            'BMI Risk': min(max((float(original_input['Body Mass Index (BMI)']) - 18.5) / 15, 0), 1),
            'Physical Activity': get_activity_level_score(original_input['Physical Activity']),
            'Smoking Risk': float(original_input['Smoking Status'] != 'Non-smoker'),
            'Alcohol Risk': float(original_input['Alcohol Intake'] in ['Frequent Drinker']),
            'Age Risk': min(max((float(original_input['Age']) - 40) / 40, 0), 1),
            'Family History': float(original_input['Family History of Stroke'])
        }

        return render_template('result.html',
                               prediction=result,
                               risk_percentage=f"{risk_percentage:.1f}",
                               user_input=user_input_str,
                               recommendations=recommendations,
                               health_indicators=health_indicators)

    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        print(f"Input data: {original_input}")
        return f"An error occurred: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
