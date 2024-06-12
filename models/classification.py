import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from io import StringIO

def random_forest_classification(csv_data):
    try:
        frutas_df = pd.read_csv(StringIO(csv_data))
    except pd.errors.EmptyDataError:
        return {'error': 'Empty CSV file'}

    if frutas_df.empty:
        return {'error': 'CSV file is empty'}

    if 'classe' not in frutas_df.columns:
        return {'error': 'Missing target column: classe'}

    X = frutas_df.drop('classe', axis=1)
    y = frutas_df['classe']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)

    return report