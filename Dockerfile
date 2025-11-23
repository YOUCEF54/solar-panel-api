FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port (facultatif mais pratique)
EXPOSE 8000

# Démarrage avec fallback si PORT non défini
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
