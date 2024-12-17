from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd


app = Flask(__name__)

@app.route("/resultat")
def resultat():
    prediction = request.args.get("prediction")
    return render_template("resultat.html", prediction=prediction)


print(url_for("resultat", prediction=prediction))
