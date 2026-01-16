import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def run_statistical_tests(df_path):
    if not os.path.exists(df_path):
        print(f"Erro: Arquivo {df_path} não encontrado.")
        return

    df = pd.read_parquet(df_path)
    
    # 0: Nada, 1: Topo, 2: Fundo
    labels = {0: 'Nada', 1: 'Topo', 2: 'Fundo'}
    
    # Variáveis de interesse
    vars_to_test = ['rsi', 'bb_width']
    
    # Resultados para o arquivo txt
    output_lines = []
    output_lines.append("TESTE DE HIPÓTESE ESTATÍSTICA (TOPOS E FUNDOS VS MERCADO NORMAL)")
    output_lines.append("="*60 + "\n")
    output_lines.append("Objetivo: Verificar se os momentos de Topo e Fundo são estatisticamente")
    output_lines.append("extremos em relação ao comportamento normal do mercado.\n")

    # Criar subplots para visualização
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, var in enumerate(vars_to_test):
        output_lines.append(f"--- Análise para Variável: {var.upper()} ---")
        
        # Separar grupos
        group0 = df[df['label'] == 0][var]
        group1 = df[df['label'] == 1][var]
        group2 = df[df['label'] == 2][var]
        
        # Médias
        output_lines.append(f"Média Geral (Nada): {group0.mean():.4f}")
        output_lines.append(f"Média Topos: {group1.mean():.4f}")
        output_lines.append(f"Média Fundos: {group2.mean():.4f}\n")
        
        # Testes t de Student (Independentes)
        # Topo vs Nada
        t_stat_top, p_val_top = stats.ttest_ind(group1, group0, equal_var=False)
        output_lines.append(f"Teste t (Topo vs Nada):")
        output_lines.append(f"  t-stat: {t_stat_top:.4f}, p-valor: {p_val_top:.8f}")
        output_lines.append(f"  Significativo (p < 0.05)? {'SIM' if p_val_top < 0.05 else 'NÃO'}")
        
        # Fundo vs Nada
        t_stat_bot, p_val_bot = stats.ttest_ind(group2, group0, equal_var=False)
        output_lines.append(f"Teste t (Fundo vs Nada):")
        output_lines.append(f"  t-stat: {t_stat_bot:.4f}, p-valor: {p_val_bot:.8f}")
        output_lines.append(f"  Significativo (p < 0.05)? {'SIM' if p_val_bot < 0.05 else 'NÃO'}")
        
        # Topo vs Fundo
        t_stat_tf, p_val_tf = stats.ttest_ind(group1, group2, equal_var=False)
        output_lines.append(f"Teste t (Topo vs Fundo):")
        output_lines.append(f"  t-stat: {t_stat_tf:.4f}, p-valor: {p_val_tf:.8f}")
        output_lines.append(f"  Significativo (p < 0.05)? {'SIM' if p_val_tf < 0.05 else 'NÃO'}\n")
        
        # Visualização: Boxplot
        sns.boxplot(x='label', y=var, data=df, ax=axes[i], palette='viridis')
        axes[i].set_xticklabels(['Nada', 'Topo', 'Fundo'])
        axes[i].set_title(f'Distribuição de {var.upper()} por Categoria')
        axes[i].set_xlabel('Estado do Mercado')
        axes[i].set_ylabel(var)

    # Salvar Gráfico
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plot_path = 'plots/analise_estatistica_labels.png'
    plt.savefig(plot_path)
    print(f"Gráfico salvo em: {plot_path}")
    
    # Salvar TXT
    txt_path = 'out_put_teste_estatistico.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    print(f"Resultados salvos em: {txt_path}")

if __name__ == "__main__":
    run_statistical_tests("data/btc_processed_4h.parquet")
