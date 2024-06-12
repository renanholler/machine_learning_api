# pin3_api

Esta API fornece serviços de classificação e clusterização usando modelos de aprendizado de máquina.

## Pré-requisitos

- Python 3.9 ou superior
- Flask
- Pandas
- Scikit-learn
- Flask-CORS

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/renanholler/pin3_api.git
    cd pin3_api
    ```

2. Crie e ative um ambiente virtual:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Inicie o servidor Flask:

    ```bash
    python app.py
    ```

2. A API estará disponível em `http://localhost:5001`.

## Endpoints

### /classify

Endpoint para classificação usando Random Forest.

- **Método**: `POST`
- **Parâmetros**: 
    - `file`: Arquivo CSV contendo os dados para classificação.

- **Resposta**:
    - JSON contendo o resultado da classificação.

### /cluster

Endpoint para clusterização usando diferentes métodos (Elbow, Coeficiente de Silhouette, Manual).

- **Método**: `POST`
- **Parâmetros**: 
    - `file`: Arquivo CSV contendo os dados para clusterização.
    - `method`: Método de clusterização (`elbow`, `coeficiente` ou `manual`).
    - `usePca`: Se `true`, aplica PCA nos dados.
    - `nClusters`: (Opcional) Número de clusters (usado apenas no método `manual`).

- **Resposta**:
    - JSON contendo o resultado da clusterização.

## Estrutura do Projeto
pin3_api/
├── models/
│   ├── classification.py
│   ├── clustering.py
├── app.py
├── requirements.txt

