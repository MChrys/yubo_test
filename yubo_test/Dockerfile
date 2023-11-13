# Utiliser une image de base Python officielle
FROM python:3.12

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer Poetry
RUN pip install poetry

# Copier les fichiers de configuration de Poetry dans le conteneur
COPY ../pyproject.toml ../poetry.lock* /app/

# Configurer Poetry pour qu'il n'utilise pas un environnement virtuel
# et installer les dépendances
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copier le reste du code source de l'application dans le conteneur
COPY . /app

# Exposer le port sur lequel votre application va s'exécuter
EXPOSE 8000

# Définir la commande pour exécuter l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
