import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight

def train_eval_models(df):
    # Features and Target
    features = [
        'close_vs_sma20', 'rsi', 'macd_diff', 'log_return', 'momentum', 'bb_width',
        'obv', 'mfi', 'rsi_lag1', 'rsi_lag2', 'macd_diff_lag1', 'macd_diff_lag2',
        'rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h',
        'rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d'
    ]
    X = df[features]
    y = df['label']

    # Split: 80% train+val, 20% test
    split_idx = int(len(df) * 0.8)
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]
    
    # Scaling - RobustScaler handles outliers better in financial data
    scaler = RobustScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Store test price for plotting
    test_dates = df['timestamp'].iloc[split_idx:]
    test_close = df['close'].iloc[split_idx:]

    print(f"Train size: {len(X_train_val_scaled)}")
    print(f"Test size: {len(X_test)}")

    results = {}

    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_val_scaled, y_train_val)
    y_pred_lr = lr.predict(X_test_scaled)
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'f1_macro': f1_score(y_test, y_pred_lr, average='macro'),
        'report': classification_report(y_test, y_pred_lr),
        'y_pred': y_pred_lr
    }

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(X_train_val_scaled, y_train_val)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1_macro': f1_score(y_test, y_pred_rf, average='macro'),
        'report': classification_report(y_test, y_pred_rf),
        'y_pred': y_pred_rf
    }

    # 3. Neural Network (MLP)
    print("Training Neural Network...")
    
    # Calculate class weights for Keras
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_val), y=y_train_val)
    cw_dict = {i: cw[i] for i in range(len(cw))}
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(features),)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # We use class weights instead of resampling to avoid leakage in validation_split
    model.fit(X_train_val_scaled, y_train_val, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0, class_weight=cw_dict)
    
    y_pred_nn_probs = model.predict(X_test_scaled)
    y_pred_nn = np.argmax(y_pred_nn_probs, axis=1)
    
    results['Neural Network'] = {
        'accuracy': accuracy_score(y_test, y_pred_nn),
        'f1_macro': f1_score(y_test, y_pred_nn, average='macro'),
        'report': classification_report(y_test, y_pred_nn),
        'y_pred': y_pred_nn
    }

    return results, y_test, test_dates, test_close

def plot_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nothing', 'Top', 'Bottom'], yticklabels=['Nothing', 'Top', 'Bottom'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/cm_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_predictions_on_price(dates, close, y_test, y_pred, model_name):
    """
    Plot price with markers for correct and incorrect predictions of tops/bottoms.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(dates, close, label='Price', color='gray', alpha=0.5)
    
    # Filter only recent data if it's too large for visibility
    # Let's take the last 200 points for a clear view
    view_slice = -200
    dates_s = dates.iloc[view_slice:]
    close_s = close.iloc[view_slice:]
    y_pred_s = y_pred[view_slice:]
    
    # Predicted Tops (1)
    tops = close_s[y_pred_s == 1]
    plt.scatter(dates_s.iloc[y_pred_s == 1], tops, color='red', label='Predicted Top', marker='v', s=100)
    
    # Predicted Bottoms (2)
    bottoms = close_s[y_pred_s == 2]
    plt.scatter(dates_s.iloc[y_pred_s == 2], bottoms, color='green', label='Predicted Bottom', marker='^', s=100)
    
    plt.title(f'Price Reversion Predictions - {model_name} (Last 200 hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/predictions_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    data_path = "data/btc_processed_4h.parquet"
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        
        results, y_test, test_dates, test_close = train_eval_models(df)
        
        print("\nSaving results and plots...")
        with open("output_metrics.txt", "w", encoding="utf-8") as f:
            for model_name, metrics in results.items():
                header = f"\n--- {model_name} ---\n"
                f.write(header)
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Macro: {metrics['f1_macro']:.4f}\n")
                f.write("Classification Report:\n")
                f.write(metrics['report'])
                f.write("\n" + "="*40 + "\n")
                
                # Print to console too
                print(header)
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                
                # Plot Confusion Matrix
                plot_confusion_matrix(y_test, metrics['y_pred'], model_name)
                
                # Plot Price Predictions
                plot_predictions_on_price(test_dates, test_close, y_test, metrics['y_pred'], model_name)
            
        print("\nTraining completed. All plots and metrics saved.")
    else:
        print(f"File {data_path} not found.")
