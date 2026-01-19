import pandas as pd
import numpy as np
import ta
import os
import matplotlib.pyplot as plt

def apply_indicators(df, prefix=""):
    """
    Aplica indicadores técnicos comuns a um dataframe.
    """
    # SMA (Média Móvel Simples)
    df[f'sma_20{prefix}'] = ta.trend.sma_indicator(df['close'], window=20)
    
    # RSI (Índice de Força Relativa)
    df[f'rsi{prefix}'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df[f'macd_diff{prefix}'] = macd.macd_diff()
    
    # Posição Relativa
    df[f'close_vs_sma20{prefix}'] = (df['close'] - df[f'sma_20{prefix}']) / df[f'sma_20{prefix}']
    
    # Volatilidade
    df[f'bb_width{prefix}'] = ta.volatility.bollinger_wband(df['close'])
    
    return df

def apply_feature_engineering_mtf(df_1h, df_4h, df_1d):
    """
    Mescla indicadores de 1h e 1d no dataframe de 4h.
    A base agora é 4h.
    """
    print("Calculando features de 4h (Base)...")
    df_4h = apply_indicators(df_4h) # Sem prefixo para 4h, pois é a base
    df_4h['obv'] = ta.volume.on_balance_volume(df_4h['close'], df_4h['volume'])
    df_4h['mfi'] = ta.volume.money_flow_index(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume'], window=14)
    df_4h['log_return'] = np.log(df_4h['close'] / df_4h['close'].shift(1))
    df_4h['momentum'] = ta.momentum.roc(df_4h['close'], window=10)
    
    for col in ['rsi', 'macd_diff', 'close_vs_sma20']:
        df_4h[f'{col}_lag1'] = df_4h[col].shift(1)
        df_4h[f'{col}_lag2'] = df_4h[col].shift(2)

    print("Calculando features de referência de 1h e 1d...")
    df_1h = apply_indicators(df_1h, prefix="_1h")
    df_1d = apply_indicators(df_1d, prefix="_1d")

    # FIX Corregindo Look-ahead Bias MTF: Shift de 1 para garantir que usamos apenas dados do passado
    # Isso evita que a vela de 4h saiba o fechamento do dia antes dele acabar.
    cols_1h = ['rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h']
    cols_1d = ['rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d']
    
    df_1h[cols_1h] = df_1h[cols_1h].shift(1)
    df_1d[cols_1d] = df_1d[cols_1d].shift(1)
    
    # Ordena para o merge_asof
    df_1h = df_1h.sort_values('timestamp')
    df_4h = df_4h.sort_values('timestamp')
    df_1d = df_1d.sort_values('timestamp')
    
    print("Mesclando tempos gráficos na base de 4h...")
    # Mescla 1h em 4h (pegando o estado de 1h no momento da vela de 4h)
    df_merged = pd.merge_asof(df_4h, df_1h[['timestamp', 'rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h']], on='timestamp')
    # Mescla 1d em 4h
    df_merged = pd.merge_asof(df_merged, df_1d[['timestamp', 'rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d']], on='timestamp')
    
    return df_merged

def label_zigzag(df, threshold=0.03):
    """
    Rotula topos e fundos usando a lógica ZigZag baseada na variação percentual.
    threshold: alteração percentual necessária para confirmar uma reversão (ex: 0.03 = 3%)
    """
    df['label'] = 0
    highs = df['high'].values
    lows = df['low'].values
    
    trend = 0  # 1 para alta, -1 para baixa
    current_max = highs[0]
    current_max_idx = 0
    current_min = lows[0]
    current_min_idx = 0
    
    for i in range(1, len(df)):
        if trend == 0:
            if highs[i] > current_max:
                current_max = highs[i]
                current_max_idx = i
            if lows[i] < current_min:
                current_min = lows[i]
                current_min_idx = i
            
            if highs[i] >= current_min * (1 + threshold):
                trend = 1
                current_max = highs[i]
                current_max_idx = i
            elif lows[i] <= current_max * (1 - threshold):
                trend = -1
                current_min = lows[i]
                current_min_idx = i
                
        elif trend == 1:  # Tendência de alta, procurando por um topo
            if highs[i] > current_max:
                current_max = highs[i]
                current_max_idx = i
            
            if lows[i] <= current_max * (1 - threshold):
                df.at[df.index[current_max_idx], 'label'] = 1  # Topo
                trend = -1
                current_min = lows[i]
                current_min_idx = i
                
        elif trend == -1:  # Tendência de baixa, procurando por um fundo
            if lows[i] < current_min:
                current_min = lows[i]
                current_min_idx = i
                
            if highs[i] >= current_min * (1 + threshold):
                df.at[df.index[current_min_idx], 'label'] = 2  # Fundo
                trend = 1
                current_max = highs[i]
                current_max_idx = i
                
    return df

def plot_zigzag(df, filename="plots/zigzag_ground_truth.png"):
    """
    Plota o preço com os topos e fundos reais identificados pelo ZigZag para visualização.
    """
    plt.figure(figsize=(15, 7))
    
    # Vamos pegar os últimos 300 pontos para uma visualização clara (Janela de ~50 dias em 4h)
    df_plot = df.iloc[-300:].copy()
    
    plt.plot(df_plot['timestamp'], df_plot['close'], label='Preço (Base 4h)', color='black', alpha=0.5)
    
    # Topos Reais (label = 1)
    tops = df_plot[df_plot['label'] == 1]
    plt.scatter(tops['timestamp'], tops['high'], color='red', label='Topo Confirmado (ZigZag)', marker='v', s=100)
    
    # Fundos Reais (label = 2)
    bottoms = df_plot[df_plot['label'] == 2]
    plt.scatter(bottoms['timestamp'], bottoms['low'], color='green', label='Fundo Confirmado (ZigZag)', marker='^', s=100)
    
    plt.title('Metodologia ZigZag: Rótulos de Topos e Fundos Reais (Target)')
    plt.xlabel('Data/Hora')
    plt.ylabel('Preço BTC (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico do Ground Truth salvo com sucesso em: {filename}")

if __name__ == "__main__":
    if os.path.exists("data/btc_historical_1h.parquet"):
        df_1h = pd.read_parquet("data/btc_historical_1h.parquet")
        df_4h = pd.read_parquet("data/btc_historical_4h.parquet")
        df_1d = pd.read_parquet("data/btc_historical_1d.parquet")
        
        print("Iniciando Engenharia de Features Multi-Timeframe centrada em 4h...")
        df_processed = apply_feature_engineering_mtf(df_1h, df_4h, df_1d)
        
        print("Rotulando pivôs de 4h com ZigZag (Threshold: 3%)...")
        df_processed = label_zigzag(df_processed, threshold=0.03)
        
        df_processed = df_processed.dropna().reset_index(drop=True)
        
        output_path = "data/btc_processed_4h.parquet"
        df_processed.to_parquet(output_path, index=False)
        print(f"Dados processados 4h MTF salvos: {len(df_processed)} registros.")
        print(f"Distribuição:\n{df_processed['label'].value_counts()}")
        
        # Exporta o gráfico dos alvos (Ground Truth) para o TCC
        plot_zigzag(df_processed)
    else:
        print("Execute data_handler.py primeiro.")
