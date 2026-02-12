import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Select important columns (simple for beginners)
df = df[['employment_type', 'required_experience', 'required_education', 'fraudulent']]

# Handle missing values
df.fillna("Unknown", inplace=True)

# Encode categorical values
encoder = LabelEncoder()
for col in ['employment_type', 'required_experience', 'required_education']:
    df[col] = encoder.fit_transform(df[col])

X = df.drop('fraudulent', axis=1)
y = df['fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ XGBoost model trained and saved as xgboost_model.pkl")
