import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np  # Adicionado para a transformação

# Caminho para o arquivo CSV fornecido pelo usuário
caminho_arquivo_csv = r"C:\Users\thiga\OneDrive\Documentos\GitHub\Projeto_1\Dados_de_Treinamento\Cordoba\ciclos_3dias(cordoba)Irrigacao_periodo1-2.csv"

# Carregar os dados
try:
    dataframe = pd.read_csv(caminho_arquivo_csv)
    print(f"Dados carregados com sucesso de: {caminho_arquivo_csv}")
    print("Primeiras 5 linhas do DataFrame carregado:")
    print(dataframe.head())
    print("\nColunas disponíveis no DataFrame original:")
    print(dataframe.columns.tolist())
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em: {caminho_arquivo_csv}")
    print("Por favor, verifique o caminho e o nome do arquivo.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo CSV: {e}")
    exit()

# --- CRIAÇÃO DA COLUNA ALVO 'Deve_Irrigar_Dia1' ---
coluna_base_para_alvo = 'IrrDia_1'  # Coluna que será usada para criar o alvo
novo_target_classificador = 'Deve_Irrigar_Dia1'  # Nome da nova coluna alvo

if coluna_base_para_alvo in dataframe.columns:
    # Se IrrDia_1 > 0, então Deve_Irrigar_Dia1 = 1, senão 0
    dataframe[novo_target_classificador] = np.where(
        dataframe[coluna_base_para_alvo] > 0, 1, 0)
    print(
        f"\nColuna alvo '{novo_target_classificador}' criada a partir de '{coluna_base_para_alvo}'.")
    print(f"Distribuição da nova coluna alvo '{novo_target_classificador}':")
    print(dataframe[novo_target_classificador].value_counts())
else:
    print(
        f"\nErro: A coluna '{coluna_base_para_alvo}', necessária para criar o alvo, não foi encontrada no CSV.")
    print("Por favor, verifique o nome da coluna no seu arquivo CSV.")
    exit()

# ----- IMPORTANTE: Verifique e ajuste os nomes das colunas de FEATURES abaixo -----
features_classificador = [
    'th1_dia1', 'th1_dia2', 'th1_dia3',
    'MinTemp_dia1', 'MinTemp_dia2', 'MinTemp_dia3',
    'MaxTemp_dia1', 'MaxTemp_dia2', 'MaxTemp_dia3',
    'Precipitation_dia1', 'Precipitation_dia2', 'Precipitation_dia3',
    'ReferenceET_dia1', 'ReferenceET_dia2', 'ReferenceET_dia3',
    'Estresse_hidrico_dia1', 'Estresse_hidrico_dia2', 'Estresse_hidrico_dia3'
    # Certifique-se de que os nomes das colunas no seu CSV correspondem a estes.
]

# Verificar se todas as features existem no DataFrame
colunas_ausentes_features = [
    col for col in features_classificador if col not in dataframe.columns]

if colunas_ausentes_features:
    print(
        f"\nErro: As seguintes colunas de FEATURES estão faltando no seu arquivo CSV ({caminho_arquivo_csv}):")
    for col in colunas_ausentes_features:
        print(f"- {col}")
    print("\nPor favor, verifique os nomes das colunas no seu arquivo CSV e ajuste a lista 'features_classificador' no script.")
    exit()

X = dataframe[features_classificador]
y = dataframe[novo_target_classificador]  # Usando a coluna alvo recém-criada

# Verificar se a coluna alvo contém apenas 0s e 1s (deve estar correto pela lógica np.where)
valores_unicos_target = y.unique()
print(
    f"\nValores únicos encontrados na coluna target ('{novo_target_classificador}'): {valores_unicos_target}")
if not all(val in [0, 1] for val in valores_unicos_target):
    print(
        f"Atenção: A coluna target '{novo_target_classificador}' deveria conter apenas valores 0 e 1.")
    # Isso não deve acontecer com a lógica np.where, mas é uma verificação extra.

# Dividir os dados em conjuntos de treinamento e teste
# stratify=y é útil se as classes (0 e 1) estiverem desbalanceadas
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
    print(f"\nErro ao dividir os dados em treino/teste: {e}")
    print("Isso pode acontecer se uma das classes (0 ou 1) tiver muito poucas amostras.")
    print("Verifique a distribuição da sua coluna alvo impressa acima.")
    exit()


print(f"\nFormato de X_train: {X_train.shape}")
print(f"Formato de y_train: {y_train.shape}")
print(f"Formato de X_test: {X_test.shape}")
print(f"Formato de y_test: {y_test.shape}")

# Inicializar e treinar o RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = classifier.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Classificador: {accuracy:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=[
      'Não Irrigar (0)', 'Irrigar (1)']))  # Adicionado target_names
print("\nMatriz de Confusão:")
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() # Descomente se quiser os valores individuais
print(confusion_matrix(y_test, y_pred))


# Salvar o modelo treinado
nome_modelo_salvo = 'modelo_classificador_irrigacao.joblib'
joblib.dump(classifier, nome_modelo_salvo)
print(f"\nModelo classificador salvo como '{nome_modelo_salvo}'")

print("\n--- Treinamento do Classificador Concluído ---")
print(
    f"Lembre-se: As features usadas para treinar este modelo foram: {features_classificador}")
print(f"O modelo espera estas mesmas features (e na mesma ordem) ao fazer novas previsões.")
