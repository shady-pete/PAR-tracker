import pandas as pd
import numpy as np

# Leggi il CSV
df = pd.read_csv('./models_v2/testing_result.csv')

def print_all_metrics_for_model(df, model_idx, title):
    print(f"\n{title}")
    print("="*50)
    print(f"Indice modello: {model_idx}")
    
    tasks = ['Gender', 'Bag', 'Hat']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Balanced_Accuracy']
    
    for task in tasks:
        print(f"\n{task}:")
        for metric in metrics:
            value = df.loc[model_idx, f"{task}_{metric}"]
            print(f"- {metric}: {value:.4f}")

def find_best_model_per_metric(df, metric):
    metric_cols = [f"{attr}_{metric}" for attr in ['Gender', 'Bag', 'Hat']]
    df['mean_' + metric] = df[metric_cols].mean(axis=1)
    best_idx = df['mean_' + metric].idxmax()
    
    print_all_metrics_for_model(df, best_idx, f"MIGLIOR MODELLO PER {metric}")
    return best_idx

def find_best_overall_model(df):
    metrics = ['Precision', 'Recall', 'Accuracy', 'Balanced_Accuracy', 'F1']
    all_metric_cols = [f"{attr}_{metric}" for attr in ['Gender', 'Bag', 'Hat'] 
                      for metric in metrics]
    
    df['overall_mean'] = df[all_metric_cols].mean(axis=1)
    best_idx = df['overall_mean'].idxmax()
    
    print_all_metrics_for_model(df, best_idx, "MIGLIOR MODELLO COMPLESSIVO")
    return best_idx

def find_best_model_per_task(df):
    tasks = ['Gender', 'Bag', 'Hat']
    metrics = ['Precision', 'Recall', 'Accuracy', 'Balanced_Accuracy', 'F1']
    
    for task in tasks:
        task_cols = [f"{task}_{metric}" for metric in metrics]
        df[f'{task}_mean'] = df[task_cols].mean(axis=1)
        best_idx = df[f'{task}_mean'].idxmax()
        
        print_all_metrics_for_model(df, best_idx, f"MIGLIOR MODELLO PER IL TASK {task}")

def find_best_model_per_f1(df):
    tasks = ['Gender', 'Bag', 'Hat']
    
    for task in tasks:
        best_idx = df[f'{task}_F1'].idxmax()
        print_all_metrics_for_model(df, best_idx, f"MIGLIOR MODELLO PER F1 SCORE - {task}")

print("\n1. MIGLIORI MODELLI PER SINGOLA METRICA")
print("="*70)
for metric in ['Precision', 'Recall', 'Accuracy', 'Balanced_Accuracy']:
    find_best_model_per_metric(df, metric)

print("\n\n2. MIGLIOR MODELLO COMPLESSIVO")
print("="*70)
find_best_overall_model(df)

print("\n\n3. MIGLIORI MODELLI PER TASK")
print("="*70)
find_best_model_per_task(df)

print("\n\n4. MIGLIORI MODELLI PER F1 SCORE")
print("="*70)
find_best_model_per_f1(df)

df = pd.read_csv('./models/testing_result.csv')
print("\n\n")
print("="*70)
print("Models_v1")
print("="*70)
print("\n1. MIGLIORI MODELLI PER SINGOLA METRICA")
print("="*70)
for metric in ['Precision', 'Recall', 'Accuracy', 'Balanced_Accuracy']:
    find_best_model_per_metric(df, metric)

print("\n\n2. MIGLIOR MODELLO COMPLESSIVO")
print("="*70)
find_best_overall_model(df)

print("\n\n3. MIGLIORI MODELLI PER TASK")
print("="*70)
find_best_model_per_task(df)

print("\n\n4. MIGLIORI MODELLI PER F1 SCORE")
print("="*70)
find_best_model_per_f1(df)