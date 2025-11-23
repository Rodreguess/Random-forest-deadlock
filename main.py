import random
import pandas as pd
import numpy as np

def gerar_snapshot(snapshot_id: int):
    """
    Gera uma simulação de um snapshot do sistema com dados técnicos simples.
    """
    num_processes = random.randint(3, 15)
    num_threads = num_processes * random.randint(2, 8)

    # Threads que estão esperando recursos
    num_waiting_threads = random.randint(0, num_threads // 2)
    pct_threads_waiting = num_waiting_threads / num_threads

    # Métricas médias
    avg_cpu_usage = round(random.uniform(5, 90), 2)  # em %
    avg_blocked_time_ms = round(random.uniform(10, 2000), 2)
    avg_resources_held = round(random.uniform(0.5, 3.0), 2)
    avg_resources_waiting = round(random.uniform(0.0, 2.5), 2)

    # Geração pseudoaleatória de deadlock
    # Deadlocks tendem a ocorrer quando há:
    # - muitos threads esperando
    # - alto tempo bloqueado
    # - recursos disputados
    if (
        num_waiting_threads > num_threads * 0.4 and
        avg_blocked_time_ms > 1000 and
        avg_resources_waiting > 1.5
    ):
        has_deadlock = 1
    else:
        has_deadlock = 0

    return {
        "snapshot_id": f"S{snapshot_id:05d}",
        "num_processes": num_processes,
        "num_threads": num_threads,
        "num_waiting_threads": num_waiting_threads,
        "pct_threads_waiting": round(pct_threads_waiting, 2),
        "avg_cpu_usage": avg_cpu_usage,
        "avg_blocked_time_ms": avg_blocked_time_ms,
        "avg_resources_held": avg_resources_held,
        "avg_resources_waiting": avg_resources_waiting,
        "has_deadlock": has_deadlock
    }

def gerar_dataset(n=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data = [gerar_snapshot(i + 1) for i in range(n)]
    df = pd.DataFrame(data)

    # embaralha e salva
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("dataset_simplificado.csv", index=False)
    print(f"✅ Dataset gerado com {len(df)} snapshots.")
    print(f"📁 Arquivo salvo como 'dataset_simplificado.csv'")
    print(df.head(10))
    return df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def previsao():
    df = pd.read_csv("dataset_simplificado.csv")

    X = df.drop(columns=["snapshot_id", "has_deadlock"])
    y = df["has_deadlock"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

if __name__ == "__main__":
    gerar_dataset(10000)
    previsao()