import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib
import os

def main():
    print("Loading telemetry data...")
    data_path = 'data/telemetry.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generator.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Separate features and labels
    features = ['request_rate', 'cpu_usage', 'memory_usage', 'latency_ms', 'error_rate']
    X = df[features]
    y_true = df['is_anomaly']
    
    print("Training Isolation Forest baseline...")
    # contamination=0.01 because we simulated 1% anomaly rate
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)
    
    # IsolationForest returns -1 for anomalies and 1 for inliers
    # We map this to 1 (anomaly) and 0 (normal) to match our labels
    preds = model.predict(X)
    y_pred = [1 if p == -1 else 0 for p in preds]
    
    print("\nEvaluation Metrics:")
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/isolation_forest.joblib'
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    main()
