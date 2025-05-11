import joblib
import pandas as pd
import Pacotes.Calculos.ETo as ETo
import Pacotes.Calculos as Calculos
import Pacotes.Calculos.Estresse as Estresse

# Parâmetros para a API
api_key = "01ad5794da6140bfb00162541240210"
cidade = "São Paulo"
altitude = 760

# 1. Dados da API
dados_api = ETo.obter_dados_weatherapi(cidade, api_key)
previsoes = dados_api['forecast']['forecastday']

# 2. Extrair variáveis para cada dia
tmax = [dia['day']['maxtemp_c'] for dia in previsoes]
tmin = [dia['day']['mintemp_c'] for dia in previsoes]
rh_max = [dia['day']['avghumidity'] for dia in previsoes]
rh_min = [h - 10 for h in rh_max]
u10 = [2 for _ in previsoes]
rn = [dia['day']['daily_will_it_rain'] *
      5 for dia in previsoes]
g = [0 for _ in previsoes]
precip = [dia['day']['totalprecip_mm'] for dia in previsoes]

# 3. Calculo ETo 
eto = []
for i in range(3):
    eto.append(ETo.calcular_eto(
        tmax[i], tmin[i], rh_max[i], rh_min[i], u10[i], rn[i], g[i], altitude
    ))

# print(eto[0], eto[1], eto[2])

# 4. Dict ETo e dados meteorológicos
eto_dict = {
    'ReferenceET_dia1': eto[0],
    'ReferenceET_dia2': eto[1],
    'ReferenceET_dia3': eto[2]
}


dados_meteorologicos = {
    'MinTemp_dia1': tmin[0], 'MinTemp_dia2': tmin[1], 'MinTemp_dia3': tmin[2],
    'MaxTemp_dia1': tmax[0], 'MaxTemp_dia2': tmax[1], 'MaxTemp_dia3': tmax[2],
    'Precipitation_dia1': precip[0], 'Precipitation_dia2': precip[1], 'Precipitation_dia3': precip[2]
}
dados_solo = {'th1_dia1': 22}

#print("Dados meteorológicos:", dados_meteorologicos)

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


# 5. Prever umidade do solo
entrada_umidade = [
    dados_solo['th1_dia1'],
    dados_meteorologicos['MinTemp_dia2'],
    dados_meteorologicos['MaxTemp_dia2'],
    dados_meteorologicos['MinTemp_dia3'],
    dados_meteorologicos['MaxTemp_dia3'],
    dados_meteorologicos['Precipitation_dia2'],
    dados_meteorologicos['Precipitation_dia3'],
    eto_dict['ReferenceET_dia2'],
    eto_dict['ReferenceET_dia3']
]
previsao_umidade = PrevisaoUmidadeSolo(entrada_umidade)

# 6. Calcular estresse hídrico
umidades_solo = [dados_solo['th1_dia1']] + list(previsao_umidade[0])
etos = [eto_dict['ReferenceET_dia1'],
        eto_dict['ReferenceET_dia2'], eto_dict['ReferenceET_dia3']]

estresses_hidricos = Estresse.calcular_estresse_hidrico(umidades_solo, etos)
estresses_hidricos = [float(e) for e in estresses_hidricos]

#print("Estresse hídrico:", estresses_hidricos)

# 7. Prever irrigação diária
entrada_irrigacao = [
    dados_solo['th1_dia1'], previsao_umidade[0][0], previsao_umidade[0][1],
    dados_meteorologicos['MinTemp_dia1'], dados_meteorologicos['MinTemp_dia2'], dados_meteorologicos['MinTemp_dia3'],
    dados_meteorologicos['MaxTemp_dia1'], dados_meteorologicos['MaxTemp_dia2'], dados_meteorologicos['MaxTemp_dia3'],
    dados_meteorologicos['Precipitation_dia1'], dados_meteorologicos[
        'Precipitation_dia2'], dados_meteorologicos['Precipitation_dia3'],
    eto_dict['ReferenceET_dia1'], eto_dict['ReferenceET_dia2'], eto_dict['ReferenceET_dia3']
]
previsao_irrigacao = PrevisaoIrrigacao(entrada_irrigacao)

print("Previsão de umidade do solo:", previsao_umidade)
print("Estresse hídrico:", estresses_hidricos)
print("Previsão de irrigação(hoje):", previsao_irrigacao)
