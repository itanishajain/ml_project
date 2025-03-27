import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset
file_path = os.path.abspath("healthcare.csv")
print(f"🔍 Checking dataset at: {file_path}")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset not found at: {file_path}. Please check the file location.")

df = pd.read_csv(file_path)

# Encode categorical columns
label_encoders = {}
def encode_column(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

categorical_columns = ["Gender", "Marital Status", "Residence Type", "Smoking Status", "Alcohol Intake", "Physical Activity"]
for col in categorical_columns:
    if col in df.columns:
        encode_column(df, col)
    else:
        print(f"⚠️ Warning: Column '{col}' not found in dataset. Skipping encoding.")

# Convert "Result" column (Target Variable) to binary
if "Result" in df.columns:
    df["Result"] = df["Result"].map({"No Stroke": 0, "Stroke": 1})
else:
    raise KeyError("❌ Error: Column 'Result' not found in dataset.")

# Define Features (X) and Target (y)
X = df.drop(columns=["Result"])
y = df["Result"]

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle missing and infinite values
X.fillna(0, inplace=True)
X.replace([np.inf, -np.inf], 0, inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/stroke_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

# Save feature names in the correct order
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")

print("✅ Model trained and saved successfully!")
