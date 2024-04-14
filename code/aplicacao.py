# Imports
import os
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycaret.classification as pc
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss, f1_score
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

# Função para plotar a matriz de confusão
def plot_confusion_matrix(model, data, target):

    # Prever os resultados do modelo
    predictions = pc.predict_model(model, data=data.drop(target, axis=1))

    # Calcular a matriz de confusão
    cm = confusion_matrix(data[target], predictions['prediction_label'])

    # Normalizar a matriz de confusão
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Criar um DataFrame para a matriz de confusão
    df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])

    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12})
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')

    # Retorna a referência da figura atual
    return plt.gcf()  

# Definindo nome do Experimento
experiment_name = 'Projeto Kobe'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

# Criando experimento MLflow "PipelineAplicacao"
with mlflow.start_run(experiment_id=experiment_id, run_name='PipelineAplicacao'):
    
    # Carregando o modelo
    model_uri = f"models:/modelo_kobe@staging"
    model = mlflow.sklearn.load_model(model_uri)

    # Carregando a base de produção
    df_prod_raw = pd.read_parquet("../data/raw/dataset_kobe_prod.parquet")

    # Removendo dados faltantes
    df_prod = df_prod_raw.dropna()

    # Filtrando data frame
    colunas = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    df_prod = df_prod[colunas]

     # Configurando o ambiente do PyCaret
    exp = pc.setup(data=df_prod, target='shot_made_flag')

    # Aplicando o modelo
    predic_label = model.predict(df_prod.drop(columns=['shot_made_flag']))

    # Calculando as métricas usando as previsões e os rótulos reais
    log_loss_prod = log_loss(df_prod['shot_made_flag'], predic_label)
    f1_score_prod = f1_score(df_prod['shot_made_flag'], predic_label)

    # Plot matrix confusão Regressão Logística
    matrix_conf_prod = plot_confusion_matrix(model, df_prod, 'shot_made_flag')

    # Salvando plot matrix de validação 
    matrix_conf_prod.savefig('../plots/matrix_conf_prod.png')

    # Registrando os parâmetros e métricas no MLFlow
    mlflow.log_metric("log_loss", log_loss_prod)
    mlflow.log_metric("f1_score", f1_score_prod)
    mlflow.log_artifact("../plots/matrix_conf_prod.png", artifact_path="plots")
    mlflow.log_artifact("../data/processed/prediction_prod.parquet")

    # Salvar resultados como um arquivo Parquet
    predic_prob = model.predict_proba(df_prod.drop(columns=['shot_made_flag']))
    df_prod['prediction_label'] = predic_label
    df_prod['prediction_score'] = predic_prob[:, 1]
    df_prod.to_parquet("../data/processed/prediction_prod.parquet")