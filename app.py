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
import threading
import time
import requests

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
THRESHOLD = 0.50  # 50% (baissé pour permettre reconnaissance avec données limitées)

# ============================================================================
# 🔄 KEEP-ALIVE: Maintenir l'API active sur Render
# ============================================================================

keep_alive_active = True
keep_alive_thread = None
API_BASE_URL = None

def determine_api_url():
    """Déterminer l'URL de l'API"""
    # En production sur Render
    if 'RENDER' in os.environ:
        return "https://ml-api-3jf9.onrender.com"
    # En local
    else:
        return "http://localhost:5000"

def keep_alive_self_ping():
    """Ping l'API lui-même toutes les 30 secondes pour éviter le sleep sur Render"""
    global keep_alive_active
    
    logger.info("=" * 70)
    logger.info("🔄 KEEP-ALIVE DÉMARRÉ")
    logger.info(f"   URL: {API_BASE_URL}")
    logger.info("   Intervalle: 30 secondes")
    logger.info("   But: Éviter le suspension de l'API sur Render (free tier)")
    logger.info("=" * 70)
    
    ping_count = 0
    while keep_alive_active:
        try:
            # Attendre 30 secondes
            time.sleep(30)
            
            # Envoyer un ping
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            elapsed = time.time() - start_time
            ping_count += 1
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = "✅" if response.status_code == 200 else "⚠️"
            
            logger.info(f"[{timestamp}] 🔄 Ping #{ping_count}: {status} Status {response.status_code} ({elapsed:.2f}s)")
            
        except Exception as e:
            logger.warning(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Ping failed: {str(e)[:50]}")

def start_keep_alive():
    """Démarrer le keep-alive en arrière-plan"""
    global keep_alive_active, keep_alive_thread, API_BASE_URL
    
    API_BASE_URL = determine_api_url()
    
    if keep_alive_thread is None or not keep_alive_thread.is_alive():
        keep_alive_active = True
        keep_alive_thread = threading.Thread(target=keep_alive_self_ping, daemon=True)
        keep_alive_thread.start()
        logger.info("✅ Keep-Alive ping thread lancé en arrière-plan")
    else:
        logger.info("⚠️ Keep-Alive déjà actif")

def stop_keep_alive():
    """Arrêter le keep-alive"""
    global keep_alive_active
    keep_alive_active = False
    logger.info("❌ Keep-Alive ping arrêté")


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
        
        # Nettoyer le préfixe data:image si présent
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Redimensionner à 224x224 (taille attendue par le modèle)
        img = img.resize((224, 224))
        logger.info(f"Image redimensionnée: 224x224 - Mode: {img.mode}")
        logger.info(f"Image dtype: {np.array(img).dtype}")
        
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
            # Mode DEMO - sans modèle, retourner erreur
            logger.warning("⚠️ Mode DEMO - Modèle non disponible")
            return jsonify({
                "success": False,
                "name": "Inconnu",
                "confidence": 0,
                "percentage": 0,
                "error": "Modèle non disponible"
            }), 503

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

@app.route('/train', methods=['POST'])
def train_model():
    """
    Entraîner le modèle avec les images disponibles
    Endpoint de maintenance - à appeler après ajout de nouvelles images
    """
    try:
        logger.info("=" * 70)
        logger.info("🚀 DEBUT DE L'ENTRAINEMENT DU MODELE")
        logger.info("=" * 70)
        
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.utils import to_categorical
        
        # Chemin du dataset - essayer plusieurs emplacements
        possible_face_dirs = [
            # Render
            "/app/face1",
            # Local development
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face1"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face1"),
            "face1"
        ]
        
        face_dir = None
        for path in possible_face_dirs:
            if os.path.exists(path):
                face_dir = path
                logger.info(f"✅ Dataset trouvé: {face_dir}")
                break
        
        if not face_dir:
            logger.error(f"Dataset non trouvé aux chemins: {possible_face_dirs}")
            return jsonify({
                "success": False,
                "error": f"Dossier face1 non trouvé. Chemins testés: {possible_face_dirs}"
            }), 400
        
        logger.info(f"Utilisation du dataset: {face_dir}")
        
        # Charger les images
        X = []
        y = []
        
        for label, person in enumerate(CLASSES):
            person_dir = os.path.join(face_dir, person)
            if not os.path.exists(person_dir):
                logger.warning(f"Dossier non trouvé: {person_dir}")
                continue
            
            images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"  {person}: {len(images)} images")
            
            for img_file in images:
                img_path = os.path.join(person_dir, img_file)
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img).astype("float32") / 255.0
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    logger.warning(f"  Erreur avec {img_file}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        total_images = len(X)
        logger.info(f"✅ Total images chargées: {total_images}")
        
        if total_images < 100:
            return jsonify({
                "success": False,
                "error": f"Pas assez d'images pour entraîner ({total_images} < 100)"
            }), 400
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        
        y_train_cat = to_categorical(y_train, num_classes=len(CLASSES))
        y_test_cat = to_categorical(y_test, num_classes=len(CLASSES))
        
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Créer et entraîner le modèle
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base.trainable = False
        
        new_model = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        logger.info("⏳ Entraînement en cours (15 epochs)...")
        history = new_model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=15,
            batch_size=32,
            verbose=0
        )
        
        # Évaluer
        final_accuracy = history.history['val_accuracy'][-1]
        logger.info(f"✅ Accuracy final: {final_accuracy*100:.2f}%")
        
        # Sauvegarder le modèle - chercher le meilleur emplacement
        model_save_paths = [
            "/app/face.h5",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "face.h5"),
        ]
        
        model_path = None
        for path in model_save_paths:
            try:
                new_model.save(path)
                model_path = path
                logger.info(f"✅ Modèle sauvegardé: {path}")
                break
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder en {path}: {e}")
        
        if not model_path:
            logger.error("Impossible de sauvegarder le modèle")
            return jsonify({
                "success": False,
                "error": "Impossible de sauvegarder le modèle"
            }), 500
        
        # Recharger le modèle global
        global model
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Modèle rechargé en mémoire")
        
        logger.info("=" * 70)
        logger.info("✅ ENTRAINEMENT TERMINE")
        logger.info("=" * 70)
        
        return jsonify({
            "success": True,
            "message": "Modele entraine avec succes",
            "total_images": total_images,
            "final_accuracy": float(final_accuracy),
            "accuracy_percent": f"{final_accuracy*100:.2f}%"
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur entrainement: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
    
    # Démarrer le keep-alive
    start_keep_alive()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )

