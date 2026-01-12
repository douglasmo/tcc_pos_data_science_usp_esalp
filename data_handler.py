import requests
import pandas as pd
import time
from datetime import datetime
import os

def fetch_klines(symbol='BTCUSDT', interval='1h', limit=1000, start_time=None):
    """
    Fetch Klines from Binance Public REST API.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time:
        params['startTime'] = start_time
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def get_historical_data(symbol='BTCUSDT', interval='1h', days=365):
    """
    Get historical data for a specified number of days.
    """
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    all_dfs = []
    current_start = start_time
    
    print(f"Fetching data for {symbol} ({interval}) since {datetime.fromtimestamp(start_time/1000)}...")
    
    while current_start < end_time:
        try:
            df = fetch_klines(symbol, interval, limit=1000, start_time=current_start)
            if df.empty:
                break
            
            all_dfs.append(df)
            # Update current_start to the last timestamp + 1ms to get the next batch
            last_ts = int(df.iloc[-1]['timestamp'].timestamp() * 1000)
            current_start = last_ts + 1
            
            print(f"Fetched until {df.iloc[-1]['timestamp']}")
            time.sleep(0.1) # Small delay to respect rate limits
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    if not all_dfs:
        return None
        
    full_df = pd.concat(all_dfs).drop_duplicates(subset='timestamp').sort_values('timestamp')
    return full_df

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetching 1 year of data for different timeframes
    intervals = ['1h', '4h', '1d']
    
    for interval in intervals:
        df = get_historical_data(symbol='BTCUSDT', interval=interval, days=365)
        if df is not None:
            file_path = os.path.join(output_dir, f"btc_historical_{interval}.parquet")
            df.to_parquet(file_path, index=False)
            print(f"Data saved to {file_path}. Total records: {len(df)}")
