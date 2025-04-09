from flask import Flask, render_template, request, make_response
import numpy as np
import joblib
import pandas as pd
from xhtml2pdf import pisa
from io import BytesIO
import io

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
            "Body Mass Index (BMI)": float(request.form['bmi']),  # Corrected key
            "Smoking Status": request.form['smoking_status'],
            "Alcohol Intake": request.form['alcohol_intake'],
            "Physical Activity": request.form['physical_activity'],
            "Stroke History": int(request.form.get('stroke_history', 0)),
            "Family History of Stroke": int(request.form.get('family_history', 0))
        }

        # Encode categorical variables
        for col in label_encoders:
            if col in user_input:
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        # Convert input to DataFrame and reorder
        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_names]
        input_df_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_df_scaled)
        predicted_class = int(prediction[0])
        result = "High Risk of Stroke" if predicted_class == 1 else "Low Risk of Stroke"

        # Convert all user_input values to string to avoid int64 serialization issues
        user_input_str = {k: str(v) for k, v in user_input.items()}

        return render_template('result.html', prediction=result, user_input=user_input_str)

    except Exception as e:
        return f"Error: {e}", 400

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    prediction = request.form.get("prediction")
    user_input = {key: request.form[key] for key in request.form if key != "prediction"}

    # Render the PDF template
    rendered = render_template("pdf_template.html", prediction=prediction, user_input=user_input)

    # Convert rendered HTML to PDF
    # Convert rendered HTML to PDF
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(BytesIO(rendered.encode("utf-8")), dest=pdf)

    if pisa_status.err:
        return "Error creating PDF", 500

    response = make_response(pdf.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=stroke_prediction_result.pdf"
    return response

if __name__ == "__main__":
    app.run(debug=True)