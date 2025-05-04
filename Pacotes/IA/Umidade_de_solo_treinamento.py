import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data_umidade = pd.read_csv(
    r"C:\Users\thiga\OneDrive\Documentos\GitHub\Projeto_1\Dados_de_Treinamento\Cordoba\ciclos_3dias(cordoba)Periodo1-2_SemIrrigacao.csv")

features_umidade = [
    'th1_dia1', 'MinTemp_dia2', 'MaxTemp_dia2', 'MinTemp_dia3', 'MaxTemp_dia3',
    'Precipitation_dia2', 'Precipitation_dia3', 'ReferenceET_dia2', 'ReferenceET_dia3'
]

target_umidade = [
    'th1_dia2', 'th1_dia3'
]

X_umidade = data_umidade[features_umidade]
y_umidade = data_umidade[target_umidade]

X_train_umidade, X_test_umidade, y_train_umidade, y_test_umidade = train_test_split(
    X_umidade,
    y_umidade,
    test_size=0.25,
    random_state=42
)

modelo_umidade = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_umidade.fit(X_train_umidade, y_train_umidade)

joblib.dump(modelo_umidade, 'modelo_umidade_treinado.joblib')

# ANALISANDO O DESEMPENHO DO MODELO
y_pred_umidade = modelo_umidade.predict(X_test_umidade)

mse = mean_squared_error(y_test_umidade, y_pred_umidade)
r2 = r2_score(y_test_umidade, y_pred_umidade)

print("Desempenho do modelo na base de teste:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

dias = ["th1_dia2", "th1_dia3"]
