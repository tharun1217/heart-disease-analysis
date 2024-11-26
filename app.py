from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Retrieve form data
            age = float(request.form["age"])
            anaemia = int(request.form["anaemia"])
            creatinine_phosphokinase = float(request.form["creatinine_phosphokinase"])
            diabetes = int(request.form["diabetes"])
            ejection_fraction = float(request.form["ejection_fraction"])
            high_blood_pressure = int(request.form["high_blood_pressure"])
            platelets = float(request.form["platelets"])
            serum_creatinine = float(request.form["serum_creatinine"])
            serum_sodium = float(request.form["serum_sodium"])
            sex = int(request.form["sex"])
            smoking = int(request.form["smoking"])
            time = float(request.form["time"])

            # Combine inputs into a single array
            input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                                    ejection_fraction, high_blood_pressure, platelets,
                                    serum_creatinine, serum_sodium, sex, smoking, time]])
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make a prediction
            prediction = model.predict(input_data_scaled)
            prediction = "Heart disease is present" if prediction[0] == 1 else "No heart disease"

        except Exception as e:
            prediction = f"Error in prediction: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
