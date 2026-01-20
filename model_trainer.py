import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
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

def train_eval_models(df, models_to_run=['lr', 'rf', 'xgb', 'lstm']):
    # Variáveis explicativas (Features) e Alvo (Target)
    features = [
        'close_vs_sma20', 'rsi', 'macd_diff', 'log_return', 'momentum', 'bb_width',
        'obv', 'mfi', 'atr', 'adx', 'close_vs_vwap', 
        'rsi_lag1', 'rsi_lag2', 'macd_diff_lag1', 'macd_diff_lag2',
        'rsi_1h', 'macd_diff_1h', 'close_vs_sma20_1h', 'adx_1h', 'close_vs_vwap_1h',
        'rsi_1d', 'macd_diff_1d', 'close_vs_sma20_1d', 'adx_1d', 'close_vs_vwap_1d'
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
    if 'lr' in models_to_run:
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
        
        y_pred_lr_probs = best_lr.predict_proba(X_test_scaled)
        
        # Aplicar limiar de confiança (Reduzir alarmes falsos)
        CONF_THRESHOLD_CLASSIC = 0.55
        y_pred_lr = []
        for probs in y_pred_lr_probs:
            p_max_idx = np.argmax(probs)
            if p_max_idx != 0 and probs[p_max_idx] > CONF_THRESHOLD_CLASSIC:
                y_pred_lr.append(p_max_idx)
            else:
                y_pred_lr.append(0)
        y_pred_lr = np.array(y_pred_lr)
        
        y_pred_lr_aligned = y_pred_lr[TIME_STEPS:] 
        
        results['Logistic Regression (Optimized)'] = {
            'accuracy': accuracy_score(y_test_final, y_pred_lr_aligned),
            'f1_macro': f1_score(y_test_final, y_pred_lr_aligned, average='macro'),
            'report': classification_report(y_test_final, y_pred_lr_aligned),
            'y_pred': y_pred_lr_aligned
        }

    # 2. Random Forest com Random Search
    if 'rf' in models_to_run:
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
        
        y_pred_rf_probs = best_rf.predict_proba(X_test_scaled)
        
        # Aplicar msm limiar para RF
        y_pred_rf = []
        for probs in y_pred_rf_probs:
            p_max_idx = np.argmax(probs)
            if p_max_idx != 0 and probs[p_max_idx] > 0.55: # Sobe o sarrafo para o RF (Reduz falsos positivos)
                y_pred_rf.append(p_max_idx)
            else:
                y_pred_rf.append(0)
        y_pred_rf = np.array(y_pred_rf)
        
        y_pred_rf_aligned = y_pred_rf[TIME_STEPS:]

        results['Random Forest (Optimized)'] = {
            'accuracy': accuracy_score(y_test_final, y_pred_rf_aligned),
            'f1_macro': f1_score(y_test_final, y_pred_rf_aligned, average='macro'),
            'report': classification_report(y_test_final, y_pred_rf_aligned),
            'y_pred': y_pred_rf_aligned
        }

    # 3. XGBoost com Random Search
    if 'xgb' in models_to_run:
        print("Treinando XGBoost (Otimizando com Random Search)...")
        xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss')
        
        param_dist_xgb = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0]
        }
        
        xgb_random = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_dist_xgb,
            n_iter=10,
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=-1,
            scoring='f1_macro'
        )
        
        xgb_random.fit(X_train_val_scaled, y_train_val)
        best_xgb = xgb_random.best_estimator_
        print(f"Melhores parâmetros XGB: {xgb_random.best_params_}")
        
        y_pred_xgb_probs = best_xgb.predict_proba(X_test_scaled)
        
        y_pred_xgb = []
        for probs in y_pred_xgb_probs:
            p_max_idx = np.argmax(probs)
            if p_max_idx != 0 and probs[p_max_idx] > 0.55:
                y_pred_xgb.append(p_max_idx)
            else:
                y_pred_xgb.append(0)
        y_pred_xgb = np.array(y_pred_xgb)
        
        y_pred_xgb_aligned = y_pred_xgb[TIME_STEPS:]

        results['XGBoost (Optimized)'] = {
            'accuracy': accuracy_score(y_test_final, y_pred_xgb_aligned),
            'f1_macro': f1_score(y_test_final, y_pred_xgb_aligned, average='macro'),
            'report': classification_report(y_test_final, y_pred_xgb_aligned),
            'y_pred': y_pred_xgb_aligned
        }

    # 4. Rede Neural LSTM
    if 'lstm' in models_to_run:
        print("Treinando Rede Neural LSTM...")
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
        
        # Lógica de Limiar de Confiança (Aproveitando apenas previsões fortes)
        CONF_THRESHOLD_NN = 0.65
        y_pred_nn = []
        for probs in y_pred_nn_probs:
            p_max_idx = np.argmax(probs)
            if p_max_idx != 0 and probs[p_max_idx] > CONF_THRESHOLD_NN:
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

    # 5. Transformer (Self-Attention)
    if 'transformer' in models_to_run:
        print("\nTreinando Modelo Transformer (Self-Attention)...")
        
        # Calcula pesos das classes
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
        cw_dict = {i: cw[i] for i in range(len(cw))}
        
        # Arquitetura Transformer Simples
        inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        
        # Self-Attention Layer
        attention_out = MultiHeadAttention(num_heads=4, key_dim=X_train_seq.shape[2])(inputs, inputs)
        attention_out = Dropout(0.3)(attention_out)
        
        # Add & Norm (Residual connection)
        x = LayerNormalization(epsilon=1e-6)(inputs + attention_out)
        
        # Feed Forward Part
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(3, activation='softmax')(x)
        
        transformer_model = Model(inputs, outputs)
        transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        transformer_model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1, class_weight=cw_dict)
        
        y_pred_trans_probs = transformer_model.predict(X_test_seq)
        
        # Lógica de Limiar de Confiança (Mesma do LSTM para manter consistência)
        CONF_THRESHOLD_TRANS = 0.55 # Ajustado para 0.55 para voltar a dar sinais de forma moderada
        y_pred_trans = []
        for probs in y_pred_trans_probs:
            p_max_idx = np.argmax(probs)
            if p_max_idx != 0 and probs[p_max_idx] > CONF_THRESHOLD_TRANS:
                y_pred_trans.append(p_max_idx)
            else:
                y_pred_trans.append(0)
        y_pred_trans = np.array(y_pred_trans)
        
        results['Transformer (Attention)'] = {
            'accuracy': accuracy_score(y_test_seq, y_pred_trans),
            'f1_macro': f1_score(y_test_seq, y_pred_trans, average='macro'),
            'report': classification_report(y_test_seq, y_pred_trans),
            'y_pred': y_pred_trans
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
        
        # Selecione os modelos que deseja rodar: ['lr', 'rf', 'xgb', 'lstm', 'transformer']
        modelos_selecionados = ['lr', 'rf', 'xgb', 'lstm', 'transformer']
        
        results, y_test_seq, test_dates, test_close = train_eval_models(df, models_to_run=modelos_selecionados)
        
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
                
                print(header)
                print(f"Acurácia: {metrics['accuracy']:.4f}")
                
                plot_confusion_matrix(y_test_seq, metrics['y_pred'], model_name)
                plot_predictions_on_price(test_dates, test_close, y_test_seq, metrics['y_pred'], model_name)
            
        print("\nTreinamento concluído. Todos os gráficos e métricas salvos.")
    else:
        print(f"Arquivo {data_path} não encontrado.")
