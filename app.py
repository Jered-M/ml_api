from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import logging
from datetime import datetime
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle TensorFlow
# Chemins possibles à essayer (ordre de priorité)
possible_paths = [
    # Priorité 1: À la racine du projet (meilleur pour Render)
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face.h5"),
    # Priorité 2: Dans le dossier api/
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "face.h5"),
    # Priorité 3: Face.h5 à la racine (chemin absolu)
    "/app/face.h5",
    "face.h5",
]

model = None
MODEL_PATH = None

for path in possible_paths:
    full_path = os.path.abspath(path) if not path.startswith('/app') else path
    if os.path.exists(full_path):
        try:
            logger.info(f"📁 Tentative de chargement depuis: {full_path}")
            model = tf.keras.models.load_model(full_path)
            MODEL_PATH = full_path
            logger.info(f"✅ Modèle TensorFlow chargé avec succès")
            logger.info(f"   Chemin: {MODEL_PATH}")
            break
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de {full_path}: {e}")
            continue

if model is None:
    logger.warning("⚠️ Aucun modèle trouvé aux emplacements attendus:")
    for path in possible_paths:
        logger.warning(f"   - {path}")
    logger.info("Mode DEMO activé - retourne des résultats de test")

# Classes de reconnaissance
CLASSES = ["jered", "gracia", "Ben", "Leo"]

# Seuil minimum pour accepter une reconnaissance
THRESHOLD = 0.70  # 70%

@app.route('/', methods=['GET'])
def index():
    """Servir l'interface HTML"""
    try:
        return send_file('interface.html', mimetype='text/html')
    except FileNotFoundError:
        return jsonify({
            "error": "Interface HTML non trouvée",
            "available_endpoints": [
                "GET /health",
                "POST /recognize",
                "GET /employees"
            ]
        }), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifier que l'API est active"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        "status": "ok",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Reconnaissance faciale avec TensorFlow
    Reçoit une image en base64
    """
    try:
        # Récupérer l'image en base64
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({
                "success": False,
                "error": "Aucune image fournie"
            }), 400
        
        # Décoder l'image
        logger.info("Décodage de l'image...")
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Redimensionner à 224x224 (taille attendue par le modèle)
        img = img.resize((224, 224))
        logger.info("Image redimensionnée: 224x224")
        
        return process_image(img)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/recognize-file', methods=['POST'])
def recognize_face_file():
    """
    Reconnaissance faciale avec TensorFlow
    Reçoit un fichier image (multipart/form-data)
    """
    try:
        # Récupérer le fichier
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "Aucune image fournie"
            }), 400
        
        file = request.files['image']
        logger.info(f"📁 Fichier reçu: {file.filename}")
        
        # Lire l'image
        img = Image.open(file.stream).convert("RGB")
        
        # Redimensionner à 224x224
        img = img.resize((224, 224))
        logger.info("Image redimensionnée: 224x224")
        
        return process_image(img)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def process_image(img):
    """
    Traite l'image et retourne les résultats
    """
    try:
        if model is not None:
            # Prétraitement (comme dans ML.ipynb)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prédiction
            logger.info("Exécution du modèle...")
            prediction = model.predict(img_array, verbose=0)[0]
            
            # Classe la plus probable
            confidence = float(np.max(prediction))
            percentage = round(confidence * 100, 2)
            index = int(np.argmax(prediction))
            
            logger.info(f"Prédiction: {CLASSES[index]} - Confiance: {percentage}%")
            
            # Vérifier le seuil
            if confidence < THRESHOLD:
                logger.warning(f"Confiance trop faible: {percentage}%")
                return jsonify({
                    "success": False,
                    "name": "Inconnu",
                    "confidence": confidence,
                    "percentage": percentage,
                    "error": "Confiance insuffisante"
                }), 200
            
            # Résultats positifs
            response = {
                "success": True,
                "name": CLASSES[index],
                "confidence": confidence,
                "percentage": percentage,
                "employee_id": f"EMP_{CLASSES[index].upper()}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Mode DEMO - sans modèle, analyse basée sur l'image
            logger.info("Mode DEMO - Analyse basée sur les caractéristiques de l'image")
            
            # Analyser l'image pour déterminer le visage
            img_array = np.array(img)
            
            # Calcul de caractéristiques simples
            brightness = np.mean(img_array)  # Luminosité moyenne
            contrast = np.std(img_array)     # Contraste
            
            # Hash simple de l'image pour pseudo-déterminer l'identité
            image_hash = int(np.sum(img_array) % 1000)
            
            # Choix pseudo-déterministe basé sur le hash
            if image_hash % 2 == 0:
                name = "jered"
                confidence = 0.82 + (image_hash % 100) / 500  # 0.82 - 0.98
            else:
                name = "gracia"
                confidence = 0.75 + (image_hash % 150) / 500  # 0.75 - 0.95
            
            # Ajustement basé sur la luminosité
            if brightness < 50:
                confidence -= 0.1  # Moins confiant si sombre
            elif brightness > 200:
                confidence -= 0.05  # Moins confiant si très lumineux
            
            # Limiter la confiance entre 0 et 1
            confidence = max(0.50, min(0.99, confidence))
            percentage = round(confidence * 100, 2)
            
            logger.info(f"Mode DEMO - {name} avec {percentage}% de confiance")
            logger.info(f"  Luminosité: {brightness:.1f}, Contraste: {contrast:.1f}")
            
            response = {
                "success": True,
                "name": name,
                "confidence": confidence,
                "percentage": percentage,
                "employee_id": f"EMP_{name.upper()}",
                "timestamp": datetime.now().isoformat(),
                "mode": "DEMO"
            }
        
        logger.info(f"✅ Reconnaissance réussie: {response['name']}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/register', methods=['POST'])
def register_face():
    """
    Enregistrer un nouveau visage
    """
    try:
        data = request.json
        name = data.get('name', '').strip()
        image_base64 = data.get('image')
        
        if not name or not image_base64:
            return jsonify({
                "success": False,
                "error": "Nom et image requis"
            }), 400
        
        # Créer le dossier des visages enregistrés s'il n'existe pas
        registered_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face1")
        person_dir = os.path.join(registered_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Décoder et sauvegarder l'image
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data))
        
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)
        
        # Sauvegarder
        img.save(filepath)
        logger.info(f"✅ Visage enregistré: {filepath}")
        
        return jsonify({
            "success": True,
            "message": f"Visage de {name} enregistré avec succès",
            "filename": filename,
            "path": filepath
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/employees', methods=['GET'])
def get_employees():
    """Récupérer la liste des employés"""
    employees = [
        {
            "name": "Jered",
            "employee_id": "EMP_JERED"
        },
        {
            "name": "Gracia",
            "employee_id": "EMP_GRACIA"
        }
    ]
    return jsonify({
        "success": True,
        "employees": employees
    }), 200

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Endpoint de test"""
    return jsonify({
        "success": True,
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Face Recognition API - TensorFlow")
    print("=" * 60)
    print(f"Modèle: {'✅ Chargé' if model is not None else '❌ Non disponible'}")
    print(f"Classes: {CLASSES}")
    print(f"Seuil: {THRESHOLD * 100}%")
    print()
    print("Serveur démarré sur http://localhost:5000")
    print()
    print("Endpoints disponibles:")
    print("  ✓ GET  http://localhost:5000/health")
    print("  ✓ POST http://localhost:5000/recognize")
    print("  ✓ GET  http://localhost:5000/employees")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )

