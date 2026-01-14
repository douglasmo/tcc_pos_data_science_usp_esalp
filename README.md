# Previsão de Preços de Bitcoin - TCC Pós-Graduação MBA DSA

Este projeto tem como objetivo desenvolver um sistema de previsão de topos (máximas) e fundos (mínimas) de curto/médio prazo para o par BTC/USDT, utilizando técnicas de Machine Learning e Deep Learning. O foco é identificar pontos de reversão de tendência em um tempo gráfico de 4 horas (4h).

## Estrutura do Projeto

O projeto está dividido em módulos funcionais para facilitar a manutenção e o processamento dos dados:

1.  **`data_handler.py`**: 
    *   Responsável pela coleta de dados via API Pública da Binance.
    *   Busca dados históricos de Klines (velas) nos intervalos de 1h, 4h e 1d.
    *   Realiza a conversão técnica de tipos e salva os dados brutos em formato `.parquet` na pasta `data/`.

2.  **`feature_engineering.py`**:
    *   **Indicadores Técnicos**: Calcula SMA20, RSI, MACD, OBV, MFI e volatilidade (Bollinger Band Width).
    *   **Multi-Timeframe (MTF)**: Integra informações de tempos gráficos superiores (1 dia) e inferiores (1 hora) na base de 4 horas para fornecer mais contexto ao modelo.
    *   **Rotulagem (ZigZag)**: Utiliza a lógica de ZigZag com um limiar de variação percentual (ex: 3%) para identificar retrospectivamente onde ocorreram os topos e fundos reais.

3.  **`model_trainer.py`**:
    *   **Pré-processamento**: Realiza o escalonamento dos dados usando `RobustScaler` e cria sequências temporais (Time Steps) para o modelo LSTM.
    *   **Treinamento e Otimização**:
        *   **Regressão Logística**: Otimizada via `RandomizedSearchCV`.
        *   **Random Forest**: Otimizada com foco em lidar com o desbalanceamento de classes (pesos balanceados).
        *   **Rede Neural LSTM (Long Short-Term Memory)**: Implementada para capturar padrões sequenciais e dependências temporais.
    *   **Avaliação**: Gera matrizes de confusão, relatórios de classificação (Precisão, Recall, F1-Score) e salva as métricas em `output_metrics.txt`.
    *   **Visualização**: Gera gráficos na pasta `plots/` sobrepondo as previsões do modelo ao gráfico de preços real.

## Como Executar

Certifique-se de ter as bibliotecas instaladas (`pandas`, `numpy`, `requests`, `ta`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `pyarrow`).

1.  **Coleta de Dados**:
    ```bash
    python data_handler.py
    ```
2.  **Engenharia de Features e Rotulagem**:
    ```bash
    python feature_engineering.py
    ```
3.  **Treinamento e Avaliação**:
    ```bash
    python model_trainer.py
    ```

## Requisitos de Dados
*   Os dados são salvos na pasta `data/`.
*   Os resultados visuais (matrizes de confusão e marcações de preço) são salvos na pasta `plots/`.

## Observação sobre o Git
Arquivos de dados (`.parquet`), resultados de treinamento (`plots/`) e pastas de cache do Python (`__pycache__`) estão ignorados pelo `.gitignore` para manter o repositório leve.
