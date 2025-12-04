# API Flask - Guide de démarrage

## 📦 Installation

### 1. Créer un environnement virtuel Python

```bash
cd api
python -m venv venv
```

### 2. Activer l'environnement

**Sur Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Sur Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Sur macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 🚀 Lancer l'API

```bash
python app.py
```

Vous verrez:
```
==================================================
🚀 Face Recognition API Flask
==================================================
Serveur démarré sur http://localhost:5000
Health check: GET http://localhost:5000/health
Reconnaissance: POST http://localhost:5000/recognize
==================================================
```

## 🧪 Tester l'API

### 1. Test de santé
```bash
curl http://localhost:5000/health
```

### 2. Test avec une image (Python)

```python
import requests
import base64

# Charger une image
with open('test_image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Envoyer à l'API
response = requests.post(
    'http://localhost:5000/recognize',
    json={'image': image_base64}
)

print(response.json())
```

### 3. Récupérer la liste des employés
```bash
curl http://localhost:5000/employees
```

## 📱 Connecter l'app React Native

1. Assurez-vous que l'API Flask tourne sur `http://localhost:5000`
2. Ouvrez `services/ApiService.js` et mettez à jour:

```javascript
const API_URL = 'http://192.168.x.x:5000'; // Remplacez par l'IP de votre ordinateur
// Ou si sur le même ordinateur:
const API_URL = 'http://localhost:5000';
```

3. Testez depuis votre téléphone/émulateur!

## 🔧 Configuration avancée

### Utiliser une vraie base de données

Remplacez `EMPLOYEES_DB` par:
```python
import sqlite3
# ou
from sqlalchemy import create_engine
```

### Ajouter une vraie reconnaissance faciale

```python
import face_recognition

# Au lieu de détecter juste les visages:
face_encodings = face_recognition.face_encodings(cv_image)
# Comparer avec les encodings stockés
```

### Déployer en production

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

---

**Besoin d'aide?** Vérifiez les logs Flask! 📊
