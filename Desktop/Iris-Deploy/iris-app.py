from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder="Web")

# โหลดโมเดลและ label encoder
model = pickle.load(open("iris_model.sav", "rb"))
le = pickle.load(open("label_encoder.sav", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]
        prediction = model.predict([np.array(features)])
        species = le.inverse_transform(prediction)[0]
        return render_template("index.html", result=f"Predicted species: {species}")
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
