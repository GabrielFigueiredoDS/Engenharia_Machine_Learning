import subprocess
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# Titulo
st.title('🏀 Projeto NBA Kobe Bryant 🏀')

# Espaço em branco
st.write("")
st.write("")

# Carregando a imagem
img = st.image("capa.png",use_column_width=True)

st.write("## 🎯 Predição")

# Criando um botão com o nome "PREDICT"
if st.button('PREDICT'):
    # Executando o arquivo de treinamento como um processo separado
    subprocess.run(["python", "aplicacao.py"])
    st.write('O modelo foi treinado!')
    st.write("")
    st.write("")
    st.write("")

    # # Leitura dos dados
    df_prod = pd.read_parquet('..\data\processed\prediction_prod.parquet')
    df_dev = pd.read_parquet('..\data\processed\prediction_test.parquet')

    # Adicionando conteúdo à primeira coluna
    col1, col2 = st.columns(2)

    with col1:
        st.write("###### 📉 Dashboard Monitoramento Dados Prod")

        # Definindo o tamanho do gráfico
        fig_dev = plt.figure(figsize=(6, 4))

        # Gráfico para os dados de teste
        sns.distplot(df_dev.prediction_score_1,
                    label='Teste',
                    ax=plt.gca());

        plt.title('Monitoramento Desvio Saída do Modelo - Dados Dev')
        plt.ylabel('Densidade Estimada')
        plt.xlabel('Probabilidade Acerto de Cesta')
        plt.grid(True)
        plt.legend(loc='best')

        st.pyplot(fig_dev)

    with col2:
        st.write("###### 📉 Dashboard Monitoramento Dados Prod")

        fig_prod = plt.figure(figsize=(6, 4))

        # Gráfico para os dados de produção
        sns.distplot(df_prod.prediction_score,
                    label='Produção',
                    ax=plt.gca())

        plt.title('Monitoramento Desvio Saída do Modelo - Dados Prod')
        plt.ylabel('Densidade Estimada')
        plt.xlabel('Probabilidade Acerto de Cesta')
        plt.grid(True)
        plt.legend(loc='best')

        st.pyplot(fig_prod)

