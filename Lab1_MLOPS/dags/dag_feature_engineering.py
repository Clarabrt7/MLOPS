import requests
from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import datetime
import pandas as pd


@dag(
    dag_id="teste_extracao_f1",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False
)
def f1_pipeline():

    @task()
    def extract_task():
        session = Variable.get("F1_SESSION", default_var="9158")
        driver = Variable.get("F1_DRIVER", default_var="1")
        url = f"https://api.openf1.org/v1/laps?session_key={session}&driver_number={driver}"
        
        response = requests.get(url)
        response.raise_for_status() 
        dados = response.json()
        
        # O print aparecerá nos logs do Airflow para você confirmar visualmente
        print(f"SUCESSO! Foram extraídos {len(dados)} registros da corrida.")
        print(f"Exemplo do primeiro registro: {dados[0]}")
        
        return dados

    @task()
    def transform_task(dados):
        df = pd.DataFrame(dados)

        print("Colunas disponíveis na OpenF1:", df.columns.tolist())

        colunas_interesse = ['driver_number', 'lap_number', 'lap_duration', 'is_pit_out_lap']
        colunas_presentes = [col for col in colunas_interesse if col in df.columns]
        df_filtrado = df[colunas_presentes].copy()    

        if 'lap_duration' in df_filtrado.columns:
            df_filtrado = df_filtrado.dropna(subset=['lap_duration'])

        # Cria a feature de Lag
        df_filtrado['prev_lap_duration'] = df_filtrado['lap_duration'].shift(1)

        # Transforma boolean em inteiro (0 e 1)
        if 'is_pit_out_lap' in df_filtrado.columns:
            df_filtrado['is_pit_out_lap'] = df_filtrado['is_pit_out_lap'].astype(int)

        # Remove o nulo gerado pela primeira volta do lag
        df_filtrado = df_filtrado.dropna()
        
        return df_filtrado.to_dict('records')
    
    @task()
    def load_task(dados_transformados):

        df_final = pd.DataFrame(dados_transformados) 
        caminho_arquivo = "/home/clarabrt7/laboratorio1_MARIA_CLARA/features.csv"  
        df_final.to_csv(caminho_arquivo, index=False)
        print(f"Carga concluída com sucesso! Arquivo salvo em: {caminho_arquivo}")


    dados_extraidos = extract_task()
    dados_transformados = transform_task(dados_extraidos)
    load_task(dados_transformados)

# Instancia o DAG
pipeline = f1_pipeline()