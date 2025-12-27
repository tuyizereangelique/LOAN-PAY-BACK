from flask import Flask, request, jsonify, render_template, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import joblib
import io
import csv
import os

app = Flask(__name__)


#  TO LOAD SAVED Machine learing ARTIFACTS

model = joblib.load("best_loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
encoders = joblib.load("encoders.pkl")


#  A CSV LOG FILE

LOG_FILE = "prediction_logs.csv"


# TO CONFIGURE MYSQL DB

app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://loan_user:password123@localhost:3306/loan_db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# DATABASE MODEL

class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(100))
    prediction = db.Column(db.String(20))
    probability = db.Column(db.Float)
    loan_amount = db.Column(db.Float)
    annual_income = db.Column(db.Float)
    debt_to_income_ratio = db.Column(db.Float)
    credit_score = db.Column(db.Float)
    interest_rate = db.Column(db.Float)
    gender = db.Column(db.String(20))
    marital_status = db.Column(db.String(20))
    education_level = db.Column(db.String(50))
    employment_status = db.Column(db.String(50))
    loan_purpose = db.Column(db.String(50))
    grade_subgrade = db.Column(db.String(10))
    timestamp = db.Column(db.String(50))

# Create table if not exists
with app.app_context():
    db.create_all()


# CSV LOGGER 

def log_to_csv(row):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# HOME PAGE

@app.route("/")
def home():
    return render_template("index.html")


# SINGLE PREDICTION

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    ml_input = {}

    for col in feature_columns:
        value = data.get(col, 0)

        try:
            ml_input[col] = float(value)
        except (ValueError, TypeError):
            ml_input[col] = 0.0

    input_df = pd.DataFrame([ml_input])
    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]
    prediction_text = "Will Repay" if probability >= 0.6 else "Will Not Repay"

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "full_name": data.get("full_name", ""),
        "loan_amount": float(data.get("loan_amount", 0)),
        "annual_income": float(data.get("annual_income", 0)),
        "debt_to_income_ratio": float(data.get("debt_to_income_ratio", 0)),
        "credit_score": float(data.get("credit_score", 0)),
        "interest_rate": float(data.get("interest_rate", 0)),
        "gender": data.get("gender", ""),
        "marital_status": data.get("marital_status", ""),
        "education_level": data.get("education_level", ""),
        "employment_status": data.get("employment_status", ""),
        "loan_purpose": data.get("loan_purpose", ""),
        "grade_subgrade": data.get("grade_subgrade", ""),
        "prediction": prediction_text,
        "probability": round(probability, 4)
    }

    # Save CSV + DB
    log_to_csv(log_data)
    db.session.add(Prediction(**log_data))
    db.session.commit()

    return jsonify({
        "prediction": prediction_text,
        "probability": round(probability, 4)
    })


# BATCH PREDICTION

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    
    required_cols = feature_columns + ["full_name", "loan_amount", "annual_income",
                                       "debt_to_income_ratio", "credit_score", "interest_rate",
                                       "gender", "marital_status", "education_level",
                                       "employment_status", "loan_purpose", "grade_subgrade"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    
    for col in ["gender", "marital_status", "education_level", "employment_status",
                "loan_purpose", "grade_subgrade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    
    categorical_maps = {
        "gender": {"Male": 0, "Female": 1},
        "marital_status": {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3},
        "education_level": {"Secondary": 0, "Bachelor": 1, "Master": 2},
        "employment_status": {"Employed": 0, "Self-Employed": 1, "Unemployed": 2},
        "loan_purpose": {"Business": 0, "Education": 1, "Personal": 2, "Housing": 3},
        "grade_subgrade": {"A1": 0, "A2": 1, "A3": 2, "B1": 3, "B2": 4, "B3": 5,
                           "C1": 6, "C2": 7, "C3": 8, "D1": 9, "D2": 10, "D3": 11, "D4": 12, "C4": 13, "C5": 14}
    }

    for col, mapping in categorical_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  

    # Scale inputs
    input_scaled = scaler.transform(df[feature_columns])
    probs = model.predict_proba(input_scaled)[:, 1]

    
    df["prediction"] = ["Will Repay" if p >= 0.6 else "Will Not Repay" for p in probs]
    df["prediction"] = df["prediction"].str.strip()  
    df["probability"] = probs.round(4)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log to CSV and DB
    records = []
    for _, row in df.iterrows():
        record = {col: row[col] for col in ["full_name", "prediction", "probability",
                                           "loan_amount", "annual_income", "debt_to_income_ratio",
                                           "credit_score", "interest_rate", "gender", "marital_status",
                                           "education_level", "employment_status", "loan_purpose",
                                           "grade_subgrade", "timestamp"]}
        log_to_csv(record)
        records.append(Prediction(**record))

    db.session.bulk_save_objects(records)
    db.session.commit()

    
    total = len(df)
    will_repay = sum(df["prediction"] == "Will Repay")
    will_not_repay = total - will_repay

    return jsonify({
        "message": "  Completed Successfully",
        "summary": {
            "total_records": total,
            "will_repay": will_repay,
            "will_not_repay": will_not_repay
        },
        "records": df.to_dict(orient="records")
    })



# DOWNLOAD SAMPLE CSV

@app.route("/download_sample_users")
def download_sample_users():
    sample_users = [

        {
            "full_name": "Ange T",
            "gender": "Male",
            "marital_status": "Single",
            "education_level": "Bachelor",
            "employment_status": "Employed",
            "loan_amount": 10000,
            "annual_income": 960000,
            "debt_to_income_ratio": 15,
            "credit_score": 710,
            "interest_rate": 4.5,
            "loan_purpose": "Business",
            "grade_subgrade": "A1",
            "expected_outcome": "Repay"
        },
        {
            "full_name": "Emmy",
            "gender": "Male",
            "marital_status": "Married",
            "education_level": "Bachelor",
            "employment_status": "Self-Employed",
            "loan_amount": 18000,
            "annual_income": 150000,
            "debt_to_income_ratio": 18,
            "credit_score": 690,
            "interest_rate": 8.0,
            "loan_purpose": "Personal",
            "grade_subgrade": "A2",
            "expected_outcome": "Repay"
        },
        {
            "full_name": "Karuhanga",
            "gender": "Female",
            "marital_status": "single",
            "education_level": "Master",
            "employment_status": "Employed",
            "loan_amount": 19000,
            "annual_income": 17000,
            "debt_to_income_ratio": 14,
            "credit_score": 860,
            "interest_rate": 4.2,
            "loan_purpose": "Education",
            "grade_subgrade": "B2",
            "expected_outcome": "Repay"
        },

        {
            "full_name": "Katushabe",
            "gender": "Male",
            "marital_status": "Single",
            "education_level": "Secondary",
            "employment_status": "Unemployed",
            "loan_amount": 18000,
            "annual_income": 22000,
            "debt_to_income_ratio": 45,
            "credit_score": 520,
            "interest_rate": 12.5,
            "loan_purpose": "Personal",
            "grade_subgrade": "D3",
            "expected_outcome": "Not Repay"
        },
        {
            "full_name": "Aline Uwase",
            "gender": "Female",
            "marital_status": "Single",
            "education_level": "Secondary",
            "employment_status": "Unemployed",
            "loan_amount": 14000,
            "annual_income": 20000,
            "debt_to_income_ratio": 40,
            "credit_score": 540,
            "interest_rate": 11.8,
            "loan_purpose": "Personal",
            "grade_subgrade": "D2",
            "expected_outcome": "Not Repay"
        },
        {
            "full_name": "Mbabazi wa John",
            "gender": "Male",
            "marital_status": "Divorced",
            "education_level": "Bachelor",
            "employment_status": "Self-Employed",
            "loan_amount": 30000,
            "annual_income": 22000,
            "debt_to_income_ratio": 38,
            "credit_score": 560,
            "interest_rate": 10.5,
            "loan_purpose": "Business",
            "grade_subgrade": "C5",
            "expected_outcome": "Not Repay"
        },
        {
            "full_name": "Tumwine",
            "gender": "Female",
            "marital_status": "Widowed",
            "education_level": "Secondary",
            "employment_status": "Unemployed",
            "loan_amount": 26000,
            "annual_income": 45000,
            "debt_to_income_ratio": 42,
            "credit_score": 510,
            "interest_rate": 13.0,
            "loan_purpose": "Housing",
            "grade_subgrade": "D4",
            "expected_outcome": "Not Repay"
        },
        {
            "full_name": "kayirebwa",
            "gender": "Male",
            "marital_status": "Married",
            "education_level": "Bachelor",
            "employment_status": "Self-Employed",
            "loan_amount": 33000,
            "annual_income": 24000,
            "debt_to_income_ratio": 36,
            "credit_score": 580,
            "interest_rate": 9.8,
            "loan_purpose": "Business",
            "grade_subgrade": "C4",
            "expected_outcome": "Not Repay"
        },
        {
            "full_name": "Keza",
            "gender": "Female",
            "marital_status": "Single",
            "education_level": "Secondary",
            "employment_status": "Unemployed",
            "loan_amount": 18000,
            "annual_income": 21000,
            "debt_to_income_ratio": 39,
            "credit_score": 530,
            "interest_rate": 11.2,
            "loan_purpose": "Personal",
            "grade_subgrade": "D1",
            "expected_outcome": "Not Repay"
        },
        {
            "full_name": "ndekwe",
            "gender": "Male",
            "marital_status": "Married",
            "education_level": "Bachelor",
            "employment_status": "Self-Employed",
            "loan_amount": 44000,
            "annual_income": 36000,
            "debt_to_income_ratio": 37,
            "credit_score": 570,
            "interest_rate": 10.9,
            "loan_purpose": "Business",
            "grade_subgrade": "C5",
            "expected_outcome": "Not Repay"
        }
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=sample_users[0].keys())
    writer.writeheader()
    writer.writerows(sample_users)
    output.seek(0)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=sample_users.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


#  TO FETCH LAST 20 LOGS

@app.route("/logs", methods=["GET"])
def get_logs():
    predictions = Prediction.query.order_by(Prediction.id.desc()).limit(20).all()

    return jsonify([
        {
            "timestamp": p.timestamp,
            "full_name": p.full_name,
            "loan_amount": p.loan_amount,
            "annual_income": p.annual_income,
            "debt_to_income_ratio": p.debt_to_income_ratio,
            "credit_score": p.credit_score,
            "interest_rate": p.interest_rate,
            "gender": p.gender,
            "marital_status": p.marital_status,
            "education_level": p.education_level,
            "employment_status": p.employment_status,
            "loan_purpose": p.loan_purpose,
            "grade_subgrade": p.grade_subgrade,
            "prediction": p.prediction,
            "probability": p.probability
        } for p in predictions
    ])


# TO RUN SERVER

if __name__ == "__main__":
    app.run(debug=True)
