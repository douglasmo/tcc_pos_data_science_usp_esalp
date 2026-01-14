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
    Transforma dados 2D em sequências 3D para LSTM.
    [samples, features] -> [samples - time_steps, time_steps, features]
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_eval_models(df):
    # Variáveis explicativas (Features) e Alvo (Target)
    features = [
        'close_vs_sma20', 'rsi', 'macd_diff', 'log_return', 'momentum', 'bb_width',
        'obv', 'mfi', 'rsi_lag1', 'rsi_lag2', 'macd_diff_lag1', 'macd_diff_lag2',
        'rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h',
        'rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d'
    ]
    X = df[features]
    y = df['label']

    # Divisão: 80% treino+val, 20% teste
    split_idx = int(len(df) * 0.8)
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]
    
    # Escalonamento - RobustScaler lida melhor com outliers em dados financeiros
    scaler = RobustScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Time steps para LSTM (retrocesso de 10 velas)
    TIME_STEPS = 10
    
    # Armazena metadados de teste antes da criação das sequências
    test_dates = df['timestamp'].iloc[split_idx + TIME_STEPS:]
    test_close = df['close'].iloc[split_idx + TIME_STEPS:]
    y_test_final = y_test[TIME_STEPS:]

    # Cria sequências para a Rede Neural
    X_train_seq, y_train_seq = create_sequences(X_train_val_scaled, y_train_val, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

    print(f"Tamanho do treino: {len(X_train_val_scaled)}, Sequências LSTM: {X_train_seq.shape}")
    print(f"Tamanho do teste: {len(X_test_scaled)}, Sequências LSTM: {X_test_seq.shape}")

    results = {}
    tscv = TimeSeriesSplit(n_splits=3)

    # 1. Regressão Logística com Random Search
    print("\nTreinando Regressão Logística (Otimizando com Random Search)...")
    lr_base = LogisticRegression(max_iter=2000, class_weight='balanced')
    
    # Grade de hiperparâmetros para Regressão Logística
    param_dist_lr = {
        'C': np.logspace(-4, 4, 50), # Explora de 0.0001 a 10000
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'] # Compatível com l1 e l2
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
    print(f"Melhores parâmetros RL: {lr_random.best_params_}")
    
    y_pred_lr = best_lr.predict(X_test_scaled)
    # Alinha com o deslocamento de TIME_STEPS para uma comparação consistente
    y_pred_lr_aligned = y_pred_lr[TIME_STEPS:] 
    
    results['Logistic Regression (Optimized)'] = {
        'accuracy': accuracy_score(y_test_final, y_pred_lr_aligned),
        'f1_macro': f1_score(y_test_final, y_pred_lr_aligned, average='macro'),
        'report': classification_report(y_test_final, y_pred_lr_aligned),
        'y_pred': y_pred_lr_aligned
    }

    # 2. Random Forest com Random Search
    print("Treinando Random Forest (Otimizando com Random Search)...")
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Grade de hiperparâmetros
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Random Search com TimeSeriesSplit (crucial para dados financeiros!)
    rf_random = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=param_dist, 
        n_iter=10, # Testa 10 combinações aleatórias
        cv=tscv, 
        verbose=0, 
        random_state=42, 
        n_jobs=-1,
        scoring='f1_macro'
    )
    
    rf_random.fit(X_train_val_scaled, y_train_val)
    best_rf = rf_random.best_estimator_
    print(f"Melhores parâmetros RF: {rf_random.best_params_}")
    
    y_pred_rf = best_rf.predict(X_test_scaled)
    y_pred_rf_aligned = y_pred_rf[TIME_STEPS:]

    results['Random Forest (Optimized)'] = {
        'accuracy': accuracy_score(y_test_final, y_pred_rf_aligned),
        'f1_macro': f1_score(y_test_final, y_pred_rf_aligned, average='macro'),
        'report': classification_report(y_test_final, y_pred_rf_aligned),
        'y_pred': y_pred_rf_aligned
    }

    # Calcula pesos das classes para LSTM
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
    
    model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1, class_weight=cw_dict)
    
    y_pred_nn_probs = model.predict(X_test_seq)
    
    # Lógica de Limiar de Confiança: Aumenta a precisão aceitando apenas certas previsões
    CONF_THRESHOLD = 0.45  # Reduzido para capturar pivôs mais significativos, mantendo a filtragem de ruído
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

def plot_confusion_matrix(y_test, y_pred, model_name, show=False):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nada', 'Topo', 'Fundo'], yticklabels=['Nada', 'Topo', 'Fundo'])
    plt.title(f'Matrix de Confusão - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/cm_{model_name.lower().replace(" ", "_")}.png')
    if show:
        plt.show()
    else:
        plt.close()

def plot_predictions_on_price(dates, close, y_test, y_pred, model_name, show=False):
    """
    Plota o preço com marcadores para previsões corretas e incorretas de topos/fundos.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(dates, close, label='Preço', color='gray', alpha=0.5)
    
    # Filtra apenas dados recentes se forem muito grandes para visualização
    # Vamos pegar os últimos 200 pontos para uma visão clara
    view_slice = -200
    dates_s = dates.iloc[view_slice:]
    close_s = close.iloc[view_slice:]
    y_pred_s = y_pred[view_slice:]
    
    # Topos Previstos (1)
    tops = close_s[y_pred_s == 1]
    plt.scatter(dates_s.iloc[y_pred_s == 1], tops, color='red', label='Topo Previsto', marker='v', s=100)
    
    # Fundos Previstos (2)
    bottoms = close_s[y_pred_s == 2]
    plt.scatter(dates_s.iloc[y_pred_s == 2], bottoms, color='green', label='Fundo Previsto', marker='^', s=100)
    
    plt.title(f'Previsões de Reversão de Preço - {model_name} (Últimas 200 horas)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/predictions_{model_name.lower().replace(" ", "_")}.png')
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    data_path = "data/btc_processed_4h.parquet"
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        
        results, y_test, test_dates, test_close = train_eval_models(df)
        
        print("\nSalvando resultados e gráficos...")
        with open("output_metrics.txt", "w", encoding="utf-8") as f:
            for model_name, metrics in results.items():
                header = f"\n--- {model_name} ---\n"
                f.write(header)
                f.write(f"Acurácia: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Macro: {metrics['f1_macro']:.4f}\n")
                f.write("Relatório de Classificação:\n")
                f.write(metrics['report'])
                f.write("\n" + "="*40 + "\n")
                
                # Imprime no console também
                print(header)
                print(f"Acurácia: {metrics['accuracy']:.4f}")
                
                # Plota Matriz de Confusão
                plot_confusion_matrix(y_test, metrics['y_pred'], model_name)
                
                # Plota Previsões de Preço
                plot_predictions_on_price(test_dates, test_close, y_test, metrics['y_pred'], model_name)
            
        print("\nTreinamento concluído. Todos os gráficos e métricas salvos.")
    else:
        print(f"Arquivo {data_path} não encontrado.")
