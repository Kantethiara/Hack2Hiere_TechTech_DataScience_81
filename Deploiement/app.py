from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import joblib
import joblib
import pandas as pd
import numpy as np
#from io import StringIO

app = Flask(__name__)

# Charger le modèle
model = joblib.load("model/best_random_forest_model.pkl")
# Charger le scaler
scaler = joblib.load("model/scaler.pkl")
# Charger les encodeurs
encoders = joblib.load("model/encoders.pkl")

# Fonction de prétraitement des données utilisateur
def preprocess_data(form_data):
    # Créer un DataFrame avec les données de l'utilisateur
    data = pd.DataFrame({
        'Age': [form_data['age']],
        'Sex': [form_data['sex']],  # Sexe sous forme de chaîne (male/female)
        'Job': [form_data['job']],
        'Housing': [form_data['housing']],  # Type de logement sous forme de chaîne (own/rent)
        'Saving accounts': [form_data['saving']],  # Compte d'épargne sous forme de chaîne
        'Checking account': [form_data['Checking']],  # Compte courant sous forme de chaîne
        'Credit amount': [form_data['credit_amount']],
        'Duration': [form_data['duration']],
        'Purpose': [form_data['purpose']]  # But du crédit sous forme de chaîne
    })

    # Transformation des colonnes catégorielles
    for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
        try:
            data[col] = encoders[col].transform(data[col].values)
        except KeyError:
            # Si un label inconnu est rencontré, le transformer en une valeur par défaut
            data[col] = 0  # Vous pouvez ajuster cette valeur par défaut selon vos besoins

    # Appliquer la normalisation sur les colonnes numériques
    data[['Age', 'Credit amount', 'Duration']] = scaler.transform(data[['Age', 'Credit amount', 'Duration']])

    return data

# Route pour l'index (formulaire de saisie simple)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form
        input_data = preprocess_data(form_data)
        result = model.predict(input_data)
        prediction = "Bon Risque" if result[0] == 1 else "Mauvais Risque"
        return redirect(url_for("resultat", prediction=prediction))
    return render_template("index.html")

# Route pour la prédiction à partir de données envoyées par le formulaire
@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    input_data = preprocess_data(form_data)
    result = model.predict(input_data)
    prediction = "Bon Risque" if result[0] == 1 else "Mauvais Risque"
    return redirect(url_for("resultat", prediction=prediction))

# Route pour afficher le résultat de la prédiction
@app.route("/resultat")
def resultat():
    prediction = request.args.get("prediction")
    return render_template('resultat.html', prediction=prediction)


# Fonction de prétraitement pour le batch sans modifier l'original
def preprocess_data_batch(df):
    df_copy = df.copy()  # Créer une copie pour ne pas altérer les données originales

    # Encoder les variables catégorielles
    for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
        if col in df_copy.columns:
            try:
                df_copy[col] = encoders[col].transform(df_copy[col])
            except KeyError:
                df_copy[col] = 0  # Valeur par défaut pour des labels inconnus

    # Appliquer la normalisation sur les colonnes numériques
    df_copy[['Age', 'Credit amount', 'Duration']] = scaler.transform(df_copy[['Age', 'Credit amount', 'Duration']])
    
    return df_copy

@app.route("/predict_batch", methods=["POST"])
def bulk_predict():
    if 'file' not in request.files:
        flash("Aucun fichier téléchargé", "error")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        flash("Fichier non pris en charge. Veuillez télécharger un fichier CSV ou Excel.", "error")
        return redirect(request.url)

    # Vérifier les colonnes requises
    required_columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    for column in required_columns:
        if column not in data.columns:
            flash(f"Colonne manquante : {column}", "error")
            return redirect(request.url)

    # Prétraiter les données pour le modèle
    processed_data = preprocess_data_batch(data)

    # Prédictions
    predictions = model.predict(processed_data)

    # Ajouter la colonne "Prediction" sans modifier les données originales
    resultat_data = data.copy()
    resultat_data['Prediction'] = ["Bon Risque" if pred == 1 else "Mauvais Risque" for pred in predictions]

    # Sauvegarder les résultats
    output_path = "static/predictions_resultat.csv"
    resultat_data.to_csv(output_path, index=False)

    return render_template('bulk_predict.html', predictions_file='predictions_resultat.csv',
                           message="Les prédictions ont été réalisées avec succès. Cliquez ci-dessous pour télécharger le résultat.")

if __name__ == '__main__':
    app.run(debug=True)
