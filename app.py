import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import bleach

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # autoriser ton frontend local Next.js

# Chargement des éléments essentiels
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model_svm = joblib.load('model_svm.pkl')

@app.route('/')
def home():
    return "✅ API de classification d'articles avec SVM"

@app.route('/classify-article', methods=['POST'])
def classify_article():
    data = request.get_json()
    article = data.get('text')

    # Validation
    if not article or not isinstance(article, str):
        return jsonify({'error': 'Champ "text" requis (string).'}), 400

    # Nettoyage simple (sécurité)
    article = bleach.clean(article, tags=[], attributes={}, strip=True)

    try:
        # Vectorisation et prédiction
        vect = tfidf_vectorizer.transform([article])
        prediction = model_svm.predict(vect)[0]
        categorie = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            'prediction_svm': categorie
        })

    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction : {str(e)}'}), 500

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


