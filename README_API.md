# Face Recognition API

API Flask pour la reconnaissance faciale avec TensorFlow.

## Installation locale

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
python app.py
```

L'API sera disponible sur `http://localhost:5000`

## Endpoints

### Health Check
```bash
GET /health
```

Retourne:
```json
{
  "status": "ok",
  "model_status": "loaded|not_loaded",
  "timestamp": "2025-12-04T..."
}
```

### Reconnaissance faciale
```bash
POST /recognize
Content-Type: application/json

{
  "image": "<base64_encoded_image>"
}
```

Retourne:
```json
{
  "success": true,
  "name": "jered",
  "confidence": 0.95,
  "percentage": 95.0,
  "employee_id": "EMP_JERED",
  "timestamp": "2025-12-04T...",
  "mode": "DEMO|PRODUCTION"
}
```

### Liste des employés
```bash
GET /employees
```

Retourne:
```json
{
  "classes": ["jered", "gracia", "Ben", "Leo"],
  "count": 4
}
```

## Déploiement sur Render

1. Connectez le repo `ml_api` sur Render
2. Render détectera automatiquement `render.yaml`
3. L'API sera déployée avec auto-redéploiement

## Modèle

Le fichier `face.h5` contient le modèle TensorFlow entraîné pour la reconnaissance faciale.

### Mode de fonctionnement

- **Avec modèle** : Reconnaissance faciale précise (70%+ de confiance)
- **Sans modèle (DEMO)** : Résultats heuristiques basés sur l'analyse d'image

## Environnement

- Python 3.11+
- TensorFlow 2.13+
- Flask 3.0+
