# Choisir une image de base
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers locaux dans le conteneur
COPY . /app

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
