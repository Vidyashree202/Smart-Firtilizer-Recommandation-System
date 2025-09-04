from flask import Flask, render_template, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
model = pickle.load(open("fertilizer_model.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Load Karnataka soil defaults
soil_defaults = pd.read_csv("soil_defaults.csv")

# Redirect root to Model1
@app.route("/")
def home():
    return redirect(url_for('model1'))

@app.route("/model1")
def model1():
    return render_template("Model1.html")

if __name__ == "__main__":
    app.run(debug=True)
