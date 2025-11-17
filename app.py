from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")
encoder = joblib.load("description_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    description = request.form.get("description")
    debit = float(request.form.get("debit"))
    credit = float(request.form.get("credit"))
    balance = float(request.form.get("balance"))

    try:
        desc_encoded = encoder.transform([description])[0]
    except:
        desc_encoded = -1  # unseen description

    input_data = np.array([[desc_encoded, debit, credit, balance]])
    pred = model.predict(input_data)[0]

    result = "❌ FRAUD Transaction" if pred == 1 else "✔ Normal Transaction"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
