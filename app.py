from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])

        if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
            return render_template(
                "index.html",
                error="Please enter positive values."
            )

        features = pd.DataFrame(
            [[area, bedrooms, bathrooms]],
            columns=["area", "bedrooms", "bathrooms"]
        )

        result = model.predict(features)[0]
        prediction = f"₹{result:,.2f}"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)