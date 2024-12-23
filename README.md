Voici la version mise à jour de votre fichier README avec les informations supplémentaires sur le déploiement avec **Flask** et **Streamlit** :  

---

# 🚀 Hack2Hiere_TechTech_DataScience_81  

Ce projet est un pipeline de **Data Science** orienté vers le **scoring de crédit** à partir d'un dataset de données de crédit allemandes. L'objectif est de développer un système de calcul du score de crédit en utilisant des modèles de machine learning tout en déployant le projet dans un environnement automatisé.  

---

## 📂 **Structure du projet**  

### **1. Données 📊**  
- `german_credit_data.csv` : Jeu de données utilisé pour l'analyse du scoring de crédit.  
  - Contient des informations telles que la durée des crédits, le montant et le niveau de risque associé.  

---

### **2. Notebooks Jupyter 📘**  
- `Databeez_credit_score.ipynb` :  
  - Notebook développé dans **Google Colab** pour réaliser l'analyse exploratoire des données (EDA) et appliquer des modèles de machine learning pour prédire le score de crédit.  
  - Inclut les étapes :  
     - Nettoyage et prétraitement des données.  
     - Analyse exploratoire pour comprendre les relations (ex : Montant total et durée par risque).  
     - Implémentation d'algorithmes de classification comme la **Régression logistique**, le **Random forest** et le **Gradient boosting**.  

---

### **3. Visualisations 📈**  
- `Databeez_Tableau_de_bord.pbix` :  
  - Rapport interactif réalisé dans **Power BI** pour visualiser :
     - Des KPI  
     - Le montant total des crédits par niveau de risque.  
     - La durée totale par catégorie de risque (bon/mauvais).  
  - Les graphiques permettent d'extraire des insights sur le dataset.  

---

### **4. Déploiement 🚢**  
- **Dossier `Deploiement`** :  
  - Contient les fichiers nécessaires pour le déploiement du projet avec **Flask** et **Streamlit**.  
  - Les deux applications (Flask et Streamlit) ont été conteneurisées à l'aide d'un **Dockerfile**.  
  - **Objectif initial** : Déployer l'application Streamlit sur **Streamlit Cloud**, mais par manque de temps, le déploiement en ligne n'a pas été effectué.  

---

### **5. Documents 🗂**  
- `CV_Data_science_Thiara_Kanteye.pdf` :  
  - Profil détaillé de Thiara Kanteye, responsable du développement du projet.  

---

## 🛠 **Technologies utilisées**  
- **Python** : Nettoyage et modélisation des données.  
- **Google Colab** : Développement et exécution du notebook.  
- **Power BI** : Création de tableaux de bord interactifs.  
- **GitHub** : Gestion du versionnement du projet.  
- **Flask** et **Streamlit** : Interfaces pour déployer le projet.  
- **Docker** : Conteneurisation pour simplifier l'exécution et le déploiement.  

---

## 🚧 **Étapes à venir**  
- **Déploiement en ligne avec Streamlit Cloud**.  
- Ajout d'une optimisation des modèles pour une meilleure précision.  

---

## 📥 **Cloner et exécuter le projet**  
Pour cloner le projet sur ta machine locale, exécute la commande suivante :  

```bash
git clone https://github.com/Kantethiara/Hack2Hiere_TechTech_DataScience_81.git
cd Hack2Hiere_TechTech_DataScience_81
```

---

## 👤 **Auteur**  
**Thiara Kanteye**  
- *Aspiring Data Scientist*  
- **LinkedIn** : [https://www.linkedin.com/in/thiara-kanteye-a137a3271/](#)  
- **Contact** : thiarakante@gmail.com  

--- 

Si vous souhaitez des modifications supplémentaires ou un meilleur formatage, faites-le-moi savoir ! 😊
