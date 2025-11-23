import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def gerar_dados(id_instantaneo: int):
    """Gera um conjunto de dados simulando o estado do sistema"""
    cenario_critico = random.choice([True, False])

    num_processos = random.randint(3, 15)
    num_threads = num_processos * random.randint(2, 8)

    if cenario_critico:
        # Força métricas altas para gerar Deadlock
        pct_espera = random.uniform(0.4, 0.7) # Muita gente esperando
        tempo_medio_bloqueio_ms = random.uniform(1300, 2000) # Bloqueio alto
        recursos_medio_espera = random.uniform(1.9, 3.0)
    else:
        # Cenário normal
        pct_espera = random.uniform(0.0, 0.3)
        tempo_medio_bloqueio_ms = random.uniform(10, 1000)
        recursos_medio_espera = random.uniform(0.0, 1.5)

    # Métricas
    num_threads_esperando = int(num_threads * pct_espera)
    pct_threads_esperando = num_threads_esperando / num_threads
    uso_medio_cpu = round(random.uniform(5, 90), 2)
    recursos_medio_em_uso = round(random.uniform(0.5, 3.0), 2)
    tempo_max_bloqueio_ms = round(random.uniform(tempo_medio_bloqueio_ms, tempo_medio_bloqueio_ms * 1.8), 2)
    desvio_padrao_bloqueio_ms = round(random.uniform(5, 300), 2)
    taxa_contention = round(recursos_medio_espera / (recursos_medio_em_uso + 0.001), 2)
    pct_processos_esperando = round(random.uniform(0, 0.6), 2)
    delta_tempo_bloqueio = round(random.uniform(-100, 500), 2)

    # pontuação probabilística para deadlock
    pontuacao = (
        0.4 * (num_threads_esperando / num_threads)
        + 0.3 * (tempo_medio_bloqueio_ms / 2000)
        + 0.2 * (recursos_medio_espera / 2.5)
        + 0.1 * taxa_contention
    )
    tem_deadlock = 1 if pontuacao > 0.7 else 0

    return {
        "id_instantaneo": f"S{id_instantaneo:05d}",
        "num_processos": num_processos,
        "num_threads": num_threads,
        "num_threads_esperando": num_threads_esperando,
        "pct_threads_esperando": round(pct_threads_esperando, 2),
        "uso_medio_cpu": uso_medio_cpu,
        "tempo_medio_bloqueio_ms": tempo_medio_bloqueio_ms,
        "recursos_medio_em_uso": recursos_medio_em_uso,
        "recursos_medio_espera": recursos_medio_espera,
        "tempo_max_bloqueio_ms": tempo_max_bloqueio_ms,
        "desvio_padrao_bloqueio_ms": desvio_padrao_bloqueio_ms,
        "taxa_contention": taxa_contention,
        "pct_processos_esperando": pct_processos_esperando,
        "delta_tempo_bloqueio": delta_tempo_bloqueio,
        "tem_deadlock": tem_deadlock
    }


def gerar_base_dados(qtd=1000, semente=42):
    random.seed(semente)
    np.random.seed(semente)
    dados = [gerar_dados(i + 1) for i in range(qtd)]
    df = pd.DataFrame(dados)

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("base_deadlock.csv", index=False)
    print(f"✅ Base de dados gerada com {len(df)} instantes de sistema.")
    print(df.head(10))
    return df

def previsao():
    df = pd.read_csv("base_deadlock.csv")

    X = df.drop(columns=["id_instantaneo", "tem_deadlock"])
    y = df["tem_deadlock"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.15, random_state=42)

    modelo = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        class_weight='balanced'
    )

    modelo.fit(X_treino, y_treino)

    print("Threshold padrão (0.5):")
    print(classification_report(y_teste, modelo.predict(X_teste)))

if __name__ == "__main__":
    gerar_base_dados(10000)
    previsao()