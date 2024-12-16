from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Charger le modèle
model = joblib.load("model/best_random_forest_model.pkl")

# Charger le scaler
scaler = joblib.load("model/scaler.pkl")

# Initialiser le LabelEncoder pour les variables catégorielles
label_encoder = {
    'Sex': LabelEncoder(),
    'Housing': LabelEncoder(),
    'Saving accounts': LabelEncoder(),
    'Purpose': LabelEncoder()
}

# Fonction de prétraitement des données utilisateur
def preprocess_data(form_data):
    # Créer un DataFrame avec les données de l'utilisateur
    data = pd.DataFrame({
        'Age': [form_data['age']],
        'Sex': [form_data['sex']],  # Sexe sous forme de chaîne (male/female)
        'Job': [form_data['job']],
        'Housing': [form_data['housing']],  # Type de logement sous forme de chaîne (own/rent)
        'Saving accounts': [form_data['saving']],  # Compte d'épargne sous forme de chaîne
        'Checking account': [form_data['checking']],  # Compte courant sous forme de chaîne
        'Credit amount': [form_data['credit_amount']],
        'Duration': [form_data['duration']],
        'Purpose': [form_data['purpose']]  # But du crédit sous forme de chaîne
    })

    # Encoder les variables catégorielles
    data['Sex'] = label_encoder['Sex'].fit_transform(data['Sex'])
    data['Housing'] = label_encoder['Housing'].fit_transform(data['Housing'])
    data['Saving accounts'] = label_encoder['Saving accounts'].fit_transform(data['Saving accounts'])
    data['Purpose'] = label_encoder['Purpose'].fit_transform(data['Purpose'])

    # Appliquer la normalisation sur les colonnes numériques
    data[['Age', 'Duration', 'Credit amount']] = scaler.transform(data[['Age', 'Duration', 'Credit amount']])

    return data

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form
        input_data = preprocess_data(form_data)
        result = model.predict(input_data)
        prediction = "Bon Risque" if result[0] == 1 else "Mauvais Risque"
        return redirect(url_for("resultat", prediction=prediction))
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    input_data = preprocess_data(form_data)
    result = model.predict(input_data)
    prediction = "Bon Risque" if result[0] == 1 else "Mauvais Risque"
    return redirect(url_for("resultat", prediction=prediction))

@app.route("/resultat")
def resultat():
    prediction = request.args.get("prediction")
    return render_template('resultat.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
