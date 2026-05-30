from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Collect form data
    passenger_data = {
        "Pclass": int(request.form["Pclass"]),
        "Sex": request.form["Sex"],
        "Age": float(request.form["Age"]),
        "SibSp": int(request.form["SibSp"]),
        "Parch": int(request.form["Parch"]),
        "Fare": float(request.form["Fare"]),
        "Embarked": request.form["Embarked"]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([passenger_data])

    # Prediction
    prediction = model.predict(input_df)[0]

    result = "Survived" if prediction == 1 else "Did Not Survive"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)