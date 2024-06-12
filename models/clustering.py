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

def kmeans_elbow(data, use_pca=False):
    # Verificar se data é uma string CSV ou um DataFrame
    if isinstance(data, str):
        df = pd.read_csv(io.StringIO(data))
    else:
        df = data
    
    # Preprocessar os dados
    df_processed = preprocess_data(df)
    X = df_processed.drop('classe', axis=1)
    
    # Normalizar os dados
    X_scaled = normalize_data(X)
    
    if use_pca:
        pca = PCA(n_components=2)
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
    
    # Gerar gráfico do cotovelo
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Quadrados das Distâncias Internas (Inertia)')
    plt.title('Método do Cotovelo')
    elbow_plot_base64 = plot_to_base64(fig1)
    
    # Gerar gráfico 3D dos clusters (se não usar PCA)
    cluster_plot_base64 = None
    if not use_pca and X_scaled.shape[1] >= 3:
        fig2 = plt.figure(figsize=(12, 10))
        ax = fig2.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=clusters, cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.set_title('K-means clustering without PCA (3D visualization)')
        ax.set_xlabel('Normalized Feature 1')
        ax.set_ylabel('Normalized Feature 2')
        ax.set_zlabel('Normalized Feature 3')
        cluster_plot_base64 = plot_to_base64(fig2)
    
    return {
        'method': 'elbow',
        'optimal_k': optimal_k,
        'distortions': distortions,
        'clusters': clusters.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'elbow_plot': elbow_plot_base64,
        'cluster_plot': cluster_plot_base64
    }

def kmeans_coeficiente(data, use_pca=False):
    data = preprocess_data(data)
    
    if use_pca:
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data)
    else:
        data_pca = data
    
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

    df_pca = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])
    df_pca['Cluster'] = clusters

    # Plot dos componentes principais
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette=sns.color_palette("hsv", best_n_clusters), data=df_pca, legend="full", alpha=0.7)
    plt.title(f'KMeans Clustering com PCA (n_clusters={best_n_clusters})')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    encoded_image_2d = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    # Plot em 3D se aplicável
    plot_3d = None
    if use_pca and data_pca.shape[1] >= 3:
        plot_3d = generate_3d_plot(data_pca, clusters, f'KMeans Clustering com PCA (n_clusters={best_n_clusters})')

    return {
        'method': 'coeficiente',
        'best_n_clusters': best_n_clusters,
        'silhouette_avg': silhouette_avg,
        'clusters': clusters.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'plot_2d': encoded_image_2d,
        'plot_3d': plot_3d
    }

def kmeans_manual(data, n_clusters, use_pca=False):
    data = preprocess_data(data)
    
    if use_pca:
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    
    return {
        'method': 'manual',
        'n_clusters': n_clusters,
        'clusters': clusters.tolist(),
        'centroids': kmeans.cluster_centers_.tolist()
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