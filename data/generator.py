import numpy as np
import pandas as pd
import os

# Set a random seed so our random numbers are identical every time we run this
np.random.seed(42)

def generate_normal_traffic(num_samples: int) -> pd.DataFrame:
    """Generates healthy server metrics."""
    return pd.DataFrame({
        'request_rate': np.random.normal(loc=1000, scale=100, size=num_samples),
        'cpu_usage': np.random.normal(loc=45, scale=5, size=num_samples),
        'memory_usage': np.random.normal(loc=50, scale=8, size=num_samples),
        'latency_ms': np.random.normal(loc=80, scale=10, size=num_samples),
        'error_rate': np.random.uniform(low=0.0, high=0.01, size=num_samples),
        'is_anomaly': 0 # 0 means normal
    })

def generate_anomalous_traffic(num_samples: int) -> pd.DataFrame:
    """Generates unhealthy server metrics (e.g., during an incident)."""
    return pd.DataFrame({
        'request_rate': np.random.normal(loc=2500, scale=500, size=num_samples), # Traffic spike
        'cpu_usage': np.random.normal(loc=90, scale=5, size=num_samples),        # CPU maxing out
        'memory_usage': np.random.normal(loc=85, scale=5, size=num_samples),
        'latency_ms': np.random.normal(loc=800, scale=200, size=num_samples),    # High latency
        'error_rate': np.random.uniform(low=0.05, high=0.20, size=num_samples),  # 5-20% errors
        'is_anomaly': 1 # 1 means anomaly
    })

def main():
    print("Generating telemetry data...")
    
    # 1. Generate 9,900 normal rows
    df_normal = generate_normal_traffic(9900)
    
    # 2. Generate 100 anomalous rows (simulating 1% incident rate)
    df_anomaly = generate_anomalous_traffic(100)
    
    # 3. Combine them together
    df_combined = pd.concat([df_normal, df_anomaly], ignore_index=True)
    
    # 4. Shuffle the data so anomalies aren't all at the very end
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # 6. Save to CSV
    output_path = 'data/telemetry.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == '__main__':
    main()
