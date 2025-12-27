import joblib
import pandas as pd
import numpy as np
from datetime import datetime


model = joblib.load("best_loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print("Model and scaler loaded successfully")
print(f"Expected features: {feature_columns}")


sample_data = [
    {
        'full_name': 'Ange T',
        'age': 35,
        'gender': '35',
        'marital_status': 'Married',
        'education_level': 'Bachelor',
        'employment_status': 'Employed',
        'loan_amount': 30000,
        'annual_income': 100000,
        'debt_to_income_ratio': 25.5,
        'interest_rate': 8.5,
        'credit_score': 720,
        'loan_purpose': 'business',
        'grade_subgrade': 'A1'
    },
    {
        'full_name': 'kaka',
        'age': 28,
        'gender': 'Female',
        'marital_status': 'Single',
        'education_level': 'Master',
        'employment_status': 'Employed',
        'loan_amount': 170000,
        'annual_income': 17000,
        'debt_to_income_ratio': 30.0,
        'interest_rate': 7.2,
        'credit_score': 750,
        'loan_purpose': 'business',
        'grade_subgrade': 'A2'
    }
]

# Convert to DataFrame
data = pd.DataFrame(sample_data)



metadata_columns = ['full_name']
metadata = data[metadata_columns].copy() if 'full_name' in data.columns else None

# Prepare features for prediction
feature_data = data.drop(columns=[col for col in metadata_columns if col in data.columns], errors='ignore')

# Define categorical columns that need encoding
categorical_columns = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# Encode categorical variables using one-hot encoding (same as training)
feature_data_encoded = pd.get_dummies(feature_data, columns=categorical_columns, drop_first=False)

print(f"\nAfter encoding, features: {list(feature_data_encoded.columns)}")


missing_features = set(feature_columns) - set(feature_data_encoded.columns)
if missing_features:
    print(f"Adding missing features: {missing_features}")
    for feature in missing_features:
        feature_data_encoded[feature] = 0


extra_features = set(feature_data_encoded.columns) - set(feature_columns)
if extra_features:
    print(f"Removing extra features: {extra_features}")
    feature_data_encoded = feature_data_encoded.drop(columns=list(extra_features))


X = feature_data_encoded[feature_columns]

print(f"\nPreparing {len(X)} records for prediction...")
print(f"Features shape: {X.shape}")
print(f"Sample of prepared features:\n{X.head()}")

# Scale the features using the loaded scaler
X_scaled = scaler.transform(X)


predictions = model.predict(X_scaled)


try:
    prediction_probabilities = model.predict_proba(X_scaled)
    print("Prediction probabilities generated successfully")
except AttributeError:
    prediction_probabilities = None
    print("Model doesn't support probability predictions")


results = {
    'predictions': predictions,
    'probabilities': prediction_probabilities,
    'feature_columns': feature_columns,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_predictions': len(predictions)
}

# Save predictions to pickle file
joblib.dump(results, "predictions.pkl")
print(f"\n Predictions saved to predictions.pkl")
print(f" Number of predictions: {len(predictions)}")

# Create a comprehensive output DataFrame
output_df = pd.DataFrame({
    'prediction': predictions,
    'prediction_label': ['Will Repay' if p == 1 else 'Will Not Repay	' for p in predictions]
})

# Add metadata if available
if metadata is not None:
    for col in metadata.columns:
        output_df[col] = metadata[col].values

# Add probability columns
if prediction_probabilities is not None:
    output_df['probability_default'] = prediction_probabilities[:, 0]
    output_df['probability_payback'] = prediction_probabilities[:, 1]
    output_df['confidence'] = np.max(prediction_probabilities, axis=1)

# Add original features (before encoding)
for col in feature_data.columns:
    if col not in output_df.columns:
        output_df[col] = feature_data[col].values

# Add timestamp
output_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Save to CSV for easy viewing
output_df.to_csv("prediction_logs.csv", index=False)
print(f" Predictions also saved to predictions.csv")

# Display sample predictions
print("\n" + "="*80)
print("SAMPLE PREDICTIONS:")
print("="*80)
for idx, row in output_df.head().iterrows():
    print(f"\nRecord {idx + 1}:")
    if 'full_name' in row:
        print(f"  Name: {row['full_name']}")
    print(f"  Prediction: {row['prediction_label']}")
    if 'probability_payback' in row:
        print(f"  Payback Probability: {row['probability_payback']*100:.2f}%")
        print(f"  Confidence: {row['confidence']*100:.2f}%")

print("\n" + "="*80)
print("Prediction summary:")
if prediction_probabilities is not None:
    payback_count = np.sum(predictions == 1)
    default_count = np.sum(predictions == 0)
    print(f"  - Will Repay: {payback_count} ({payback_count/len(predictions)*100:.2f}%)")
    print(f"  - Will Not Repay: {default_count} ({default_count/len(predictions)*100:.2f}%)")
print("="*80)