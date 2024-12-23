import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
import tempfile
import os
import joblib  

# Charger le modèle
model = joblib.load("model/best_random_forest_model.pkl")

# Charger le scalerP
scaler = joblib.load("model/scaler.pkl")

# Charger les encodeurs pour les variables catégorielles
# Initialiser les LabelEncoders pour chaque variable catégorielle
label_encoder = {
    'Sex': LabelEncoder(),
    'Housing': LabelEncoder(),
    'Saving accounts': LabelEncoder(),
    'Purpose': LabelEncoder(),
    'Job': LabelEncoder(),  # Assurez-vous d'avoir un LabelEncoder pour 'Job'
    'Checking account': LabelEncoder()  # Et un autre pour 'Checking account'
}

# Fonction pour charger les données
@st.cache_data
def load_data():
    data = pd.read_csv("model/german_credit_data.csv")
    return data

# Page d'accueil
def home_page():
    st.title("Prédiction du Score de Crédit")
    
    # Ajout d'une image à la page d'accueil
    st.image("download.jpeg")

    st.write("""
   Le Credit Scoring est un outil crucial dans le secteur financier, utilisé pour évaluer la capacité d'un individu ou d'une entreprise à rembourser un crédit. Grâce à des données historiques et des algorithmes de machine learning, les institutions financières peuvent prédire la probabilité qu'un client soit en mesure de respecter ses engagements financiers. Ce processus améliore la gestion des risques tout en facilitant une prise de décision rapide et précise pour l'octroi de crédits.

Dans ce projet, nous travaillons avec la base de données German Credit Data, qui contient des informations détaillées sur les caractéristiques socio-économiques et financières des individus. L'objectif principal est de développer un modèle de prédiction permettant de déterminer si une demande de crédit représente un bon risque ou un mauvais risque.
    """)
    st.write("Explorez les données, testez les modèles et effectuez des prédictions dans cette application.")

# Page d'exploration des données
def data_exploration_page():
    st.title("Exploration et Préparation des Données")
    data = load_data()
    st.write("Aperçu des données :")
    st.dataframe(data.head())
      # Génération du rapport de profilage
    show_profiling = st.checkbox("Afficher l'option pour télécharger le rapport de profilage", value=False)
    
    if show_profiling:
        # Mettre en couleur pour attirer l'attention
        st.markdown("### :blue[Rapport de Profilage des Données]")
        # Génération du rapport de profilage
        profile = ProfileReport(data, title="Rapport de Profilage", explorative=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            profile.to_file(tmpfile.name)
            st.download_button(
                label="Télécharger le rapport de profilage",
                data=open(tmpfile.name, "rb").read(),
                file_name="profiling_report.html",
                mime="text/html",
            )
    # Visualisation 1 : Distribution des valeurs de "Credit amount"
    st.subheader("Distribution du Montant du Crédit")
    fig1 = px.histogram(data, x="Credit amount", nbins=30, title="Répartition des Montants de Crédit")
    st.plotly_chart(fig1)
    
    # Visualisation 2 : Distribution des durées de crédit
    st.subheader("Distribution de la Durée du Crédit")
    fig2 = px.histogram(data, x="Duration", nbins=30, title="Répartition des Durées de Crédit")
    st.plotly_chart(fig2)

    # Visualisation 3 : Relation entre "Age" et "Credit amount"
    st.subheader("Relation entre l'Durée et le Montant du Crédit")
    fig3 = px.scatter(data, x="Duration", y="Credit amount", color="Risk",
                      title="Durée vs Montant du Crédit (colorié par le Risque)",
                      labels={"Duration": "Durée", "Credit amount": "Montant du Crédit"})
    st.plotly_chart(fig3)
    

    # Sélectionner uniquement les colonnes numériques pour la corrélation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()

    # Créer une figure pour la heatmap
    plt.figure(figsize=(10, 8))

    # Afficher la heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8})

    # Afficher la heatmap dans Streamlit
    st.pyplot(plt)

# Fonction de prétraitement des données utilisateur
def preprocess_data(form_data):
    # Créer un DataFrame avec les données de l'utilisateur
    data = pd.DataFrame({
        'Age': [form_data['age']],
        'Sex': [form_data['sex']],  # Sexe sous forme de chaîne (male/female)
        'Job': [form_data['Job']],
        'Housing': [form_data['housing']],  # Type de logement sous forme de chaîne (own/rent)
        'Saving accounts': [form_data['saving']],  # Compte d'épargne sous forme de chaîne
        'Checking account': [form_data['Checking']],  # Compte courant sous forme de chaîne
        'Credit amount': [form_data['credit_amount']],
        'Duration': [form_data['duration']],
        'Purpose': [form_data['purpose']]  # But du crédit sous forme de chaîne
    })

    # Encoder les variables catégorielles avec le LabelEncoder
    data['Sex'] = label_encoder['Sex'].fit_transform(data['Sex'])
    data['Housing'] = label_encoder['Housing'].fit_transform(data['Housing'])
    data['Saving accounts'] = label_encoder['Saving accounts'].fit_transform(data['Saving accounts'])
    data['Purpose'] = label_encoder['Purpose'].fit_transform(data['Purpose'])
    data['Checking account'] = label_encoder['Checking account'].fit_transform(data['Checking account'])  # Idem pour "Checking account"
    
    
        # Encoder la colonne "Job" comme une variable catégorielle
    job_encoder = LabelEncoder()
    data['Job'] = job_encoder.fit_transform(data['Job'])

    # Appliquer la normalisation sur les colonnes numériques
    data[['Age', 'Credit amount', 'Duration']] = scaler.transform(data[['Age', 'Credit amount', 'Duration']])

    return data

 # Page de prédiction
def prediction_page():
    st.title("Prédiction du Score de Crédit")
    
    # Formulaire de saisie utilisateur
    st.write("Entrez les caractéristiques pour effectuer une prédiction :")
    
    age = st.number_input("Âge", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sexe", ["male", "female"])
    Job = st.selectbox("Job", ["unskilled and non-resident", "unskilled and resident", "skilled", "highly skilled"])
    housing = st.selectbox("Type de logement", ["own", "rent", "free"])
    saving = st.selectbox("Comptes d'épargne", ["little", "moderate", "quite rich", "rich"])
    Checking = st.selectbox("Compte courant", ["little", "moderate", "rich"])
    credit_amount = st.number_input("Montant du crédit", min_value=0, value=5000)
    duration = st.number_input("Durée du crédit (en mois)", min_value=1, value=12)
    purpose = st.selectbox("But du crédit", ["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"])
    
    form_data = {
        'age': age,
        'sex': sex,
        'Job': Job,
        'housing': housing,
        'saving': saving,
        'Checking': Checking,
        'credit_amount': credit_amount,
        'duration': duration,
        'purpose': purpose
    }
    
    # Effectuer la prédiction lorsque l'utilisateur appuie sur le bouton
    if st.button("Faire la Prédiction"):
        input_data = preprocess_data(form_data)
        result = model.predict(input_data)
        prediction = "Bon Risque" if result[0] == 1 else "Mauvais Risque"
        st.write(f"Le client a un : **{prediction}**")

# Fonction principale pour la navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choisissez une section", ["Accueil", "Exploration et Préparation des Données","Prédiction"])
    
    if page == "Accueil":
        home_page()
    elif page == "Exploration et Préparation des Données":
        data_exploration_page()
        
    elif page == "Prédiction":
        prediction_page()
   
# Lancer l'application Streamlit
if __name__ == '__main__':
    main()
