import joblib
import pandas as pd


def PrevisaoUmidadeSolo(entrada):
    features_umidade = [
        'th1_dia1', 'MinTemp_dia2', 'MaxTemp_dia2', 'MinTemp_dia3', 'MaxTemp_dia3',
        'Precipitation_dia2', 'Precipitation_dia3', 'ReferenceET_dia2', 'ReferenceET_dia3'
    ]

    modelo_umidade = joblib.load('modelo_umidade_treinado.joblib')

    entrada_df = pd.DataFrame([entrada], columns=features_umidade)
    previsao = modelo_umidade.predict(entrada_df)
    return previsao


def PrevisaoIrrigacao(entrada):
    features_irrigacao = [
        'th1_dia1', 'th1_dia2', 'th1_dia3',
        'MinTemp_dia1', 'MinTemp_dia2', 'MinTemp_dia3',
        'MaxTemp_dia1', 'MaxTemp_dia2', 'MaxTemp_dia3',
        'Precipitation_dia1', 'Precipitation_dia2', 'Precipitation_dia3',
        'ReferenceET_dia1', 'ReferenceET_dia2', 'ReferenceET_dia3'
    ]

    modelo_irrigacao = joblib.load('modelo_irrigacao_treinado.joblib')

    entrada_df = pd.DataFrame([entrada], columns=features_irrigacao)
    previsao = modelo_irrigacao.predict(entrada_df)
    return previsao


# Exemplo de uso
print(PrevisaoUmidadeSolo({
    'th1_dia1': 0.2,
    'MinTemp_dia2': 15.0,
    'MaxTemp_dia2': 25.0,
    'MinTemp_dia3': 14.0,
    'MaxTemp_dia3': 26.0,
    'Precipitation_dia2': 5.0,
    'Precipitation_dia3': 10.0,
    'ReferenceET_dia2': 4.0,
    'ReferenceET_dia3': 5.0
}))

print(PrevisaoIrrigacao({
    'th1_dia1': 0.2,
    'th1_dia2': 0.25,
    'th1_dia3': 0.3,
    'MinTemp_dia1': 15.0,
    'MinTemp_dia2': 16.0,
    'MinTemp_dia3': 17.0,
    'MaxTemp_dia1': 25.0,
    'MaxTemp_dia2': 26.0,
    'MaxTemp_dia3': 27.0,
    'Precipitation_dia1': 5.0,
    'Precipitation_dia2': 6.0,
    'Precipitation_dia3': 7.0,
    'ReferenceET_dia1': 4.0,
    'ReferenceET_dia2': 5.0,
    'ReferenceET_dia3': 6.0
}))
