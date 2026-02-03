import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
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

def _make_class_weight(y, boost=None):
    """
    Calcula pesos balanceados e aplica multiplicadores por classe.
    boost: dict {classe: multiplicador}
    """
    classes = np.unique(y)
    cw = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    cw_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    if boost:
        for c, mult in boost.items():
            if c in cw_dict:
                cw_dict[c] *= float(mult)
    return cw_dict
def _optimize_thresholds(y_true, y_probs, t_min=0.2, t_max=0.8, step=0.05):
    """
    Busca limiares por classe (1 e 2) maximizando recall medio das classes 1 e 2 na validacao.
    """
    best_t1, best_t2, best_score = 0.45, 0.45, -1.0
    thresholds = np.arange(t_min, t_max + 1e-9, step)
    for t1 in thresholds:
        for t2 in thresholds:
            y_pred = []
            for probs in y_probs:
                p_max_idx = int(np.argmax(probs))
                if p_max_idx == 1 and probs[1] >= t1:
                    y_pred.append(1)
                elif p_max_idx == 2 and probs[2] >= t2:
                    y_pred.append(2)
                else:
                    y_pred.append(0)
            y_pred = np.array(y_pred)
            score = recall_score(y_true, y_pred, labels=[1, 2], average='macro', zero_division=0)
            if score > best_score:
                best_score = score
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2, best_score


def _build_lstm(input_shape, num_classes=2):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1 if num_classes == 2 else num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model
def train_eval_models(df):
    # Variaveis explicativas (Features) e Alvo (Target)
    features = [
        'close_vs_sma20', 'rsi', 'macd_diff', 'log_return', 'momentum', 'bb_width',
        'obv', 'mfi', 'rsi_lag1', 'rsi_lag2', 'macd_diff_lag1', 'macd_diff_lag2',
        'rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h',
        'rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d'
    ]
    X = df[features]
    y = df['label']

    # Divisao: 80% treino+val, 20% teste
    split_idx = int(len(df) * 0.8)
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]

    # Escalonamento - RobustScaler lida melhor com outliers em dados financeiros
    scaler = RobustScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # Time steps para LSTM (retrocesso de 10 velas)
    TIME_STEPS = 10
    # Validacao temporal (sem vazamento)
    split_tv = int(len(X_train_val_scaled) * 0.8)
    X_train, X_val = X_train_val_scaled[:split_tv], X_train_val_scaled[split_tv:]
    y_train, y_val = y_train_val.iloc[:split_tv], y_train_val.iloc[split_tv:]

    # Armazena metadados de teste antes da criacao das sequencias
    test_dates = df['timestamp'].iloc[split_idx + TIME_STEPS:]
    test_close = df['close'].iloc[split_idx + TIME_STEPS:]
    y_test_final = y_test[TIME_STEPS:]

    # Cria sequencias para a Rede Neural (sem cruzar fronteiras temporais)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

    print(f"Tamanho do treino: {len(X_train_val_scaled)}, Sequencias LSTM: {X_train_seq.shape}")
    print(f"Tamanho do teste: {len(X_test_scaled)}, Sequencias LSTM: {X_test_seq.shape}")

    results = {}
    tscv = TimeSeriesSplit(n_splits=3)

    # 1. Regressao Logistica com Random Search
    print("\nTreinando Regressao Logistica (Otimizando com Random Search)...")
    cw_dict = _make_class_weight(y_train_val, boost={1: 3.0})
    lr_base = LogisticRegression(max_iter=2000, class_weight=cw_dict)

    param_dist_lr = {
        'C': np.logspace(-4, 4, 50),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    lr_random = RandomizedSearchCV(
        estimator=lr_base,
        param_distributions=param_dist_lr,
        n_iter=15,
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=1,
        scoring='f1_macro'
    )

    lr_random.fit(X_train_val_scaled, y_train_val)
    best_lr = lr_random.best_estimator_
    print(f"Melhores parametros RL: {lr_random.best_params_}")

    y_pred_lr = best_lr.predict(X_test_scaled)
    y_pred_lr_aligned = y_pred_lr[TIME_STEPS:]

    results['Logistic Regression (Optimized)'] = {
        'accuracy': accuracy_score(y_test_final, y_pred_lr_aligned),
        'f1_macro': f1_score(y_test_final, y_pred_lr_aligned, average='macro'),
        'report': classification_report(y_test_final, y_pred_lr_aligned),
        'y_pred': y_pred_lr_aligned
    }

    # 2. Random Forest com Random Search
    print("Treinando Random Forest (Otimizando com Random Search)...")
    rf_base = RandomForestClassifier(class_weight=cw_dict, random_state=42)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=10,
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=1,
        scoring='f1_macro'
    )

    rf_random.fit(X_train_val_scaled, y_train_val)
    best_rf = rf_random.best_estimator_
    print(f"Melhores parametros RF: {rf_random.best_params_}")

    y_pred_rf = best_rf.predict(X_test_scaled)
    y_pred_rf_aligned = y_pred_rf[TIME_STEPS:]

    results['Random Forest (Optimized)'] = {
        'accuracy': accuracy_score(y_test_final, y_pred_rf_aligned),
        'f1_macro': f1_score(y_test_final, y_pred_rf_aligned, average='macro'),
        'report': classification_report(y_test_final, y_pred_rf_aligned),
        'y_pred': y_pred_rf_aligned
    }

    # 3. LSTM
    cw_lstm = _make_class_weight(y_train_seq, boost={1: 3.0})

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

    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stop],
        verbose=1,
        class_weight=cw_lstm,
        shuffle=False
    )

    y_pred_nn_probs = model.predict(X_test_seq)

    # Otimiza limiares por classe usando validacao temporal
    y_val_probs = model.predict(X_val_seq)
    best_t1, best_t2, best_score = _optimize_thresholds(y_val_seq, y_val_probs)

    y_pred_nn = []
    for probs in y_pred_nn_probs:
        p_max_idx = int(np.argmax(probs))
        if p_max_idx == 1 and probs[1] >= best_t1:
            y_pred_nn.append(1)
        elif p_max_idx == 2 and probs[2] >= best_t2:
            y_pred_nn.append(2)
        else:
            y_pred_nn.append(0)
    y_pred_nn = np.array(y_pred_nn)

    results['Neural Network (LSTM)'] = {
        'accuracy': accuracy_score(y_test_seq, y_pred_nn),
        'f1_macro': f1_score(y_test_seq, y_pred_nn, average='macro'),
        'report': classification_report(y_test_seq, y_pred_nn),
        'thresholds': (best_t1, best_t2, best_score),
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
