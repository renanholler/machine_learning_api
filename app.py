from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from models.classification import random_forest_classification


from models.clustering import cluster_data

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        csv_data = file.read().decode('utf-8')
        result = random_forest_classification(csv_data)
        return jsonify(result)

@app.route('/cluster', methods=['POST'])
def cluster():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        csv_data = file.read().decode('utf-8')
        method = request.form.get('method')
        use_pca = request.form.get('usePca') == 'true'
        n_clusters = request.form.get('nClusters')
        if n_clusters:
            n_clusters = int(n_clusters)
        result = cluster_data(csv_data, method, use_pca, n_clusters)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))