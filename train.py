import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# MLflow Configuration
# -----------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("IPL Winner Prediction")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/matches.csv")

# Select required columns
df = df[[
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",
    "winner"
]]

# Remove missing values
df = df.dropna()

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("winner", axis=1)
y = df["winner"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy :", accuracy)

    # Log Parameters
    mlflow.log_param("Model", "RandomForest")
    mlflow.log_param("Trees", 100)

    # Log Metric
    mlflow.log_metric("Accuracy", accuracy)

    # Save model in MLflow
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="IPL_Model"
        )
    except Exception as e:
        print("MLflow model logging skipped:", e)

# -----------------------------
# Save Model Locally
# -----------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Training Completed Successfully!")