from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "IPL Winner Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame([[
        data["team1"],
        data["team2"],
        data["venue"],
        data["toss_winner"],
        data["toss_decision"]
    ]], columns=[
        "team1",
        "team2",
        "venue",
        "toss_winner",
        "toss_decision"
    ])

    prediction = model.predict(input_data)

    return jsonify({
        "Predicted Winner": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)