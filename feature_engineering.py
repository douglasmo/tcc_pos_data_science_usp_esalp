import pandas as pd
import numpy as np
import ta
import os

def apply_indicators(df, prefix=""):
    """
    Apply common technical indicators to a dataframe.
    """
    # SMA
    df[f'sma_20{prefix}'] = ta.trend.sma_indicator(df['close'], window=20)
    
    # RSI
    df[f'rsi{prefix}'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df[f'macd_diff{prefix}'] = macd.macd_diff()
    
    # Relative Position
    df[f'close_vs_sma20{prefix}'] = (df['close'] - df[f'sma_20{prefix}']) / df[f'sma_20{prefix}']
    
    # Volatility
    df[f'bb_width{prefix}'] = ta.volatility.bollinger_wband(df['close'])
    
    return df

def apply_feature_engineering_mtf(df_1h, df_4h, df_1d):
    """
    Merge 1h and 1d indicators into the 4h dataframe.
    Base is now 4h.
    """
    print("Calculating 4h features (Base)...")
    df_4h = apply_indicators(df_4h) # No prefix for 4h as it is the base
    df_4h['obv'] = ta.volume.on_balance_volume(df_4h['close'], df_4h['volume'])
    df_4h['mfi'] = ta.volume.money_flow_index(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume'], window=14)
    df_4h['log_return'] = np.log(df_4h['close'] / df_4h['close'].shift(1))
    df_4h['momentum'] = ta.momentum.roc(df_4h['close'], window=10)
    
    for col in ['rsi', 'macd_diff', 'close_vs_sma20']:
        df_4h[f'{col}_lag1'] = df_4h[col].shift(1)
        df_4h[f'{col}_lag2'] = df_4h[col].shift(2)

    print("Calculating 1h and 1d reference features...")
    df_1h = apply_indicators(df_1h, prefix="_1h")
    df_1d = apply_indicators(df_1d, prefix="_1d")
    
    # Sort for merge_asof
    df_1h = df_1h.sort_values('timestamp')
    df_4h = df_4h.sort_values('timestamp')
    df_1d = df_1d.sort_values('timestamp')
    
    print("Merging timeframes into 4h base...")
    # Merge 1h into 4h (taking the state of 1h at the time of 4h candle)
    df_merged = pd.merge_asof(df_4h, df_1h[['timestamp', 'rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h']], on='timestamp')
    # Merge 1d into 4h
    df_merged = pd.merge_asof(df_merged, df_1d[['timestamp', 'rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d']], on='timestamp')
    
    return df_merged

def label_zigzag(df, threshold=0.03):
    """
    Label tops and bottoms using ZigZag logic based on percentage variation.
    threshold: percentage change required to confirm a reversal (e.g., 0.03 = 3%)
    """
    df['label'] = 0
    highs = df['high'].values
    lows = df['low'].values
    
    trend = 0  # 1 for up, -1 for down
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
                
        elif trend == 1:  # Up trend, looking for a top
            if highs[i] > current_max:
                current_max = highs[i]
                current_max_idx = i
            
            if lows[i] <= current_max * (1 - threshold):
                df.at[df.index[current_max_idx], 'label'] = 1  # Top
                trend = -1
                current_min = lows[i]
                current_min_idx = i
                
        elif trend == -1:  # Down trend, looking for a bottom
            if lows[i] < current_min:
                current_min = lows[i]
                current_min_idx = i
                
            if highs[i] >= current_min * (1 + threshold):
                df.at[df.index[current_min_idx], 'label'] = 2  # Bottom
                trend = 1
                current_max = highs[i]
                current_max_idx = i
                
    return df

if __name__ == "__main__":
    if os.path.exists("data/btc_historical_1h.parquet"):
        df_1h = pd.read_parquet("data/btc_historical_1h.parquet")
        df_4h = pd.read_parquet("data/btc_historical_4h.parquet")
        df_1d = pd.read_parquet("data/btc_historical_1d.parquet")
        
        print("Starting 4h-Centered Multi-Timeframe Feature Engineering...")
        df_processed = apply_feature_engineering_mtf(df_1h, df_4h, df_1d)
        
        print("Labeling 4h pivots with ZigZag (Threshold: 3%)...")
        df_processed = label_zigzag(df_processed, threshold=0.03)
        
        df_processed = df_processed.dropna().reset_index(drop=True)
        
        output_path = "data/btc_processed_4h.parquet"
        df_processed.to_parquet(output_path, index=False)
        print(f"4h MTF processed data saved: {len(df_processed)} records.")
        print(f"Distribution:\n{df_processed['label'].value_counts()}")
    else:
        print("Run data_handler.py first.")
