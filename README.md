🧠 Detecção Inteligente de Deadlocks com Random Forest:

Este projeto implementa um sistema de detecção automática de deadlocks utilizando Machine Learning (Random Forest).

O sistema gera uma base sintética contendo milhares de snapshots do estado de um ambiente concorrente, incluindo métricas como:

Número de processos e threads;

Threads bloqueadas e esperando;

Uso médio de CPU;

Tempo médio de bloqueio;

Taxa de contenção;

Recursos e métricas derivadas;

Indicador final de deadlock;

Com esses dados, um modelo RandomForestClassifier é treinado para classificar automaticamente se um determinado instante representa ou não um deadlock.

🚀 Funcionalidades:

✔️ Geração automática de milhares de instantes de execução;

✔️ Simulação realista de estados concorrentes;

✔️ Classificação entre deadlock e não-deadlock;

✔️ Treinamento completo usando Random Forest;

✔️ Relatórios de avaliação do modelo (accuracy, recall, precision, F1);

✔️ Código modular, claro e de fácil manutenção;

📂 Estrutura e Detalhes Técnicos:

O arquivo main.py está dividido em três partes principais:

🔹 1. Geração dos Dados Sintéticos:

A função gerar_dados() cria snapshots simulados do sistema, contendo diversas métricas relevantes do ambiente concorrente.

🔹 2. Construção da Base de Dados:

A função gerar_base_dados(qtd=10000) gera um arquivo CSV com milhares de exemplos rotulados.

🔹 3. Treinamento e Avaliação:

A função previsao() treina o modelo Random Forest e exibe métricas como:

    Accuracy;
    
    Precision;
    
    Recall;
    
    F1-score.

🛠️ Requisitos Para Rodar o Projeto:

✔️ 1. Python 3.10+ (recomendado)

✔️ 2. Instalar dependências

Execute no terminal:
pip install pandas numpy scikit-learn

✔️ 3. Executar o projeto:

Gerar base de dados:
python main.py --gerar

📊 Exemplos de Métricas Utilizadas:

    num_processos;
    
    num_threads;
    
    num_threads_esperando;
    
    pct_threads_esperando;
    
    uso_medio_cpu;
    
    tempo_medio_bloqueio_ms;
    
    taxa_contention;
    
    recursos_medio_espera;
    
    tem_deadlock (label final).

📚 Tecnologias Utilizadas:

Python:

    Pandas;
    
    NumPy;
    
    Scikit-learn;
    
    RandomForestClassifier.

👨‍💻 Autores:
Gabriel Rodrigues da Silva.
Deibson dos Santos Lima.
