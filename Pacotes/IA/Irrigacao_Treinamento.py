from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv(
    r"C:\Users\thiga\OneDrive\Documentos\GitHub\Projeto_1\Dados_de_Treinamento\Cordoba\ciclos_3dias(cordoba)Irrigacao_periodo1-2.csv")

features = [
    'th1_dia1', 'th1_dia2', 'th1_dia3',
    'MinTemp_dia1', 'MinTemp_dia2', 'MinTemp_dia3',
    'MaxTemp_dia1', 'MaxTemp_dia2', 'MaxTemp_dia3',
    'Precipitation_dia1', 'Precipitation_dia2', 'Precipitation_dia3',
    'ReferenceET_dia1', 'ReferenceET_dia2', 'ReferenceET_dia3'
]

X = df[features]
y = df['IrrDia_1']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

modelo = RandomForestRegressor(n_estimators=150, random_state=42)
modelo.fit(X_train, y_train)


joblib.dump(modelo, 'modelo_irrigacao_treinado.joblib')


# ANALISANDO O DESEMPENHO DO MODELO
previsoes = modelo.predict(X_test)
erro = mean_absolute_error(y_test, previsoes)

print(f'Erro médio absoluto (MAE): {erro:.3f} mm')

y_pred = modelo.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)


r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.3f} mm')
print(f'MSE: {mse:.3f} mm²')
print(f'R²: {r2:.3f}')




