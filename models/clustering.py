import matplotlib
matplotlib.use('Agg')  # Use o backend 'Agg' para evitar problemas com GUI

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from io import StringIO

def preprocess_data(data):
    for column in data.columns:
        if data[column].dtype == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generate_3d_plot(data, labels, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    encoded_image = base64.b64encode(image_png).decode('utf-8')
    return encoded_image

# Método KMeans Elbow
def kmeans_elbow(data, use_pca=False):
    if isinstance(data, str):
        df = pd.read_csv(io.StringIO(data))
    else:
        df = data
    
    df_processed = preprocess_data(df)
    X = df_processed.drop('classe', axis=1)
    X_scaled = normalize_data(X)
    
    if use_pca:
        pca = PCA(n_components=3)
        X_scaled = pca.fit_transform(X_scaled)
    
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(X_scaled)
        distortions.append(kmeanModel.inertia_)
    
    optimal_k = distortions.index(min(distortions)) + 1
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Gráfico do cotovelo
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Quadrados das Distâncias Internas (Inertia)')
    plt.title('Método do Cotovelo')
    elbow_plot_base64 = plot_to_base64(fig1)
    
    # Gráfico 3D dos clusters
    cluster_plot_base64 = None
    if X_scaled.shape[1] >= 3:
        cluster_plot_base64 = generate_3d_plot(X_scaled, clusters, 'K-means clustering (3D visualization)')
    
    return {
        'method': 'elbow',
        'optimal_k': optimal_k,
        'distortions': distortions,
        'clusters': clusters.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'plot_2d': elbow_plot_base64,
        'plot_3d': cluster_plot_base64
    }

# Método KMeans Coeficiente
def kmeans_coeficiente(data, use_pca=False):
    if isinstance(data, str):
        df = pd.read_csv(io.StringIO(data))
    else:
        df = data
    
    df_processed = preprocess_data(df)
    X_scaled = normalize_data(df_processed)
    
    if use_pca:
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(X_scaled)
    else:
        data_pca = X_scaled
    
    range_n_clusters = range(2, 10)
    silhouette_avg = []
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data_pca)
        cluster_labels = kmeans.labels_
        silhouette_avg.append(silhouette_score(data_pca, cluster_labels))
    
    best_n_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    kmeans = KMeans(n_clusters=best_n_clusters)
    clusters = kmeans.fit_predict(data_pca)

    # Gráfico de Silhouette
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_avg, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Coeficiente de Silhouette')
    plt.title('Coeficiente de Silhouette para diferentes números de clusters')
    silhouette_plot_base64 = plot_to_base64(fig1)
    
    # Gráfico 3D dos clusters
    cluster_plot_base64 = None
    if data_pca.shape[1] >= 3:
        cluster_plot_base64 = generate_3d_plot(data_pca, clusters, 'K-means clustering (3D visualization)')
    
    return {
        'method': 'coeficiente',
        'best_n_clusters': best_n_clusters,
        'silhouette_avg': silhouette_avg,
        'clusters': clusters.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'plot_2d': silhouette_plot_base64,
        'plot_3d': cluster_plot_base64
    }

# Método KMeans Manual
def kmeans_manual(data, n_clusters, use_pca=False):
    if isinstance(data, str):
        df = pd.read_csv(io.StringIO(data))
    else:
        df = data
    
    df_processed = preprocess_data(df)
    X_scaled = normalize_data(df_processed)
    
    if use_pca:
        pca = PCA(n_components=3)
        X_scaled = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Gráfico 3D dos clusters
    cluster_plot_base64 = generate_3d_plot(X_scaled, clusters, f'KMeans Clustering (n_clusters={n_clusters})')
    
    return {
        'method': 'manual',
        'n_clusters': n_clusters,
        'clusters': clusters.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'plot_3d': cluster_plot_base64
    }

def cluster_data(csv_data, method, use_pca, n_clusters=None):
    data = pd.read_csv(StringIO(csv_data))
    if method == 'elbow':
        return kmeans_elbow(data, use_pca)
    elif method == 'coeficiente':
        return kmeans_coeficiente(data, use_pca)
    elif method == 'manual':
        return kmeans_manual(data, n_clusters, use_pca)
    else:
        return {'error': 'Invalid method'}