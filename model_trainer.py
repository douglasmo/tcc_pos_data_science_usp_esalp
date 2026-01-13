import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight

def create_sequences(X, y, time_steps=10):
    """
    Transform 2D data into 3D sequences for LSTM.
    [samples, features] -> [samples - time_steps, time_steps, features]
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

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
    
    # Time steps for LSTM (look back 10 candles)
    TIME_STEPS = 10
    
    # Store test metadata before sequence creation
    test_dates = df['timestamp'].iloc[split_idx + TIME_STEPS:]
    test_close = df['close'].iloc[split_idx + TIME_STEPS:]
    y_test_final = y_test[TIME_STEPS:]

    # Create Sequences for Neural Network
    X_train_seq, y_train_seq = create_sequences(X_train_val_scaled, y_train_val, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

    print(f"Train size: {len(X_train_val_scaled)}, LSTM Sequences: {X_train_seq.shape}")
    print(f"Test size: {len(X_test_scaled)}, LSTM Sequences: {X_test_seq.shape}")

    results = {}
    tscv = TimeSeriesSplit(n_splits=3)

    # 1. Logistic Regression with Random Search
    print("\nTraining Logistic Regression (Optimizing with Random Search)...")
    lr_base = LogisticRegression(max_iter=2000, class_weight='balanced')
    
    # Hyperparameter Grid for LR
    param_dist_lr = {
        'C': np.logspace(-4, 4, 50), # Explore from 0.0001 to 10000
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'] # Compatible with l1 and l2
    }
    
    lr_random = RandomizedSearchCV(
        estimator=lr_base,
        param_distributions=param_dist_lr,
        n_iter=15,
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=-1,
        scoring='f1_macro'
    )
    
    lr_random.fit(X_train_val_scaled, y_train_val)
    best_lr = lr_random.best_estimator_
    print(f"Best LR Params: {lr_random.best_params_}")
    
    y_pred_lr = best_lr.predict(X_test_scaled)
    # Align with TIME_STEPS shift for consistent comparison
    y_pred_lr_aligned = y_pred_lr[TIME_STEPS:] 
    
    results['Logistic Regression (Optimized)'] = {
        'accuracy': accuracy_score(y_test_final, y_pred_lr_aligned),
        'f1_macro': f1_score(y_test_final, y_pred_lr_aligned, average='macro'),
        'report': classification_report(y_test_final, y_pred_lr_aligned),
        'y_pred': y_pred_lr_aligned
    }

    # 2. Random Forest with Random Search
    print("Training Random Forest (Optimizing with Random Search)...")
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Hyperparameter Grid
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Random Search with TimeSeriesSplit (crucial for financial data!)
    rf_random = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=param_dist, 
        n_iter=10, # Test 10 random combinations
        cv=tscv, 
        verbose=0, 
        random_state=42, 
        n_jobs=-1,
        scoring='f1_macro'
    )
    
    rf_random.fit(X_train_val_scaled, y_train_val)
    best_rf = rf_random.best_estimator_
    print(f"Best RF Params: {rf_random.best_params_}")
    
    y_pred_rf = best_rf.predict(X_test_scaled)
    y_pred_rf_aligned = y_pred_rf[TIME_STEPS:]

    results['Random Forest (Optimized)'] = {
        'accuracy': accuracy_score(y_test_final, y_pred_rf_aligned),
        'f1_macro': f1_score(y_test_final, y_pred_rf_aligned, average='macro'),
        'report': classification_report(y_test_final, y_pred_rf_aligned),
        'y_pred': y_pred_rf_aligned
    }

    # Calculate class weights for LSTM
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    cw_dict = {i: cw[i] for i in range(len(cw))}
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0, class_weight=cw_dict)
    
    y_pred_nn_probs = model.predict(X_test_seq)
    
    # Confidence Threshold Logic: Increase precision by only accepting certain predictions
    CONF_THRESHOLD = 0.45  # Lowered to capture more significant pivots while still filtering noise
    y_pred_nn = []
    for probs in y_pred_nn_probs:
        p_max_idx = np.argmax(probs)
        if p_max_idx != 0 and probs[p_max_idx] > CONF_THRESHOLD:
            y_pred_nn.append(p_max_idx)
        else:
            y_pred_nn.append(0)
    y_pred_nn = np.array(y_pred_nn)
    
    results['Neural Network (LSTM)'] = {
        'accuracy': accuracy_score(y_test_seq, y_pred_nn),
        'f1_macro': f1_score(y_test_seq, y_pred_nn, average='macro'),
        'report': classification_report(y_test_seq, y_pred_nn),
        'y_pred': y_pred_nn
    }

    return results, y_test_seq, test_dates, test_close

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
