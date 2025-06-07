#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import joblib
import math
import serial
import time
import os
import warnings
import paho.mqtt.client as mqtt 
import uuid
# Suprime aviso de unpickle de versão do sklearn
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator .* from version .* when using version .*"
)


# --- Configurações de caminhos absolutos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TH_PATH = os.path.join(BASE_DIR, "modelo_umidade_treinado.joblib")
MODEL_IRR_PATH = os.path.join(BASE_DIR, "modelo_irrigacao_treinado.joblib")
# Caminho para o novo modelo classificador
MODEL_CLASS_PATH = os.path.join(
    BASE_DIR, "modelo_classificador_irrigacao.joblib")

# --------------------
# Funções de hardware
# --------------------


def ler_umidade_solo(
    porta='/dev/ttyACM0', baud=9600, timeout=15,
    canal=0, min_valor=200, max_valor=900, inverter=True,
    default=0.0
):
    """
    Lê umidade do solo via Arduino.
    Envia comando "R 0\n" e retorna valor normalizado em %
    Retorna `default` em caso de falha.
    """
    try:
        # No Windows, a porta pode ser algo como 'COM3', 'COM4', etc.
        # No Linux, geralmente '/dev/ttyACM0' ou '/dev/ttyUSB0'
        # Ajuste 'porta' conforme necessário ou passe como argumento.
        with serial.Serial(porta, baud, timeout=timeout) as ser:
            time.sleep(2)  # Tempo para a conexão serial estabilizar
            ser.reset_input_buffer()  # Limpa buffer de entrada
            comando = f"R {canal}\n"  # Supondo que o canal seja relevante
            ser.write(comando.encode())
            linha = ""
            deadline = time.time() + timeout
            while time.time() < deadline:  # Loop de leitura com timeout
                if ser.in_waiting > 0:
                    linha = ser.readline().decode('utf-8', errors='ignore').strip()
                    if linha:  # Sai do loop se uma linha não vazia for lida
                        break
                time.sleep(0.1)  # Pequena pausa para não sobrecarregar CPU

            if not linha:
                raise RuntimeError(
                    "Sem resposta do Arduino ou resposta vazia na leitura do sensor.")

            valor_bruto = int(linha)
    except serial.SerialException as e:
        print(
            f"[Aviso] Falha ao abrir/conectar à porta serial {porta}: {e}. Usando valor padrão {default}.")
        return default
    except ValueError as e:
        print(
            f"[Aviso] Falha ao converter valor do sensor para inteiro ('{linha}'): {e}. Usando valor padrão {default}.")
        return default
    except Exception as e:
        print(
            f"[Aviso] Falha na leitura do sensor: {e}. Usando valor padrão {default}.")
        return default

    # Normalização
    # Clampa o valor dentro do range
    valor = max(min_valor, min(max_valor, valor_bruto))
    frac = (valor - min_valor) / float(max_valor - min_valor)
    if inverter:  # Se o sensor lê maior para seco, inverte
        frac = 1.0 - frac
    return frac * 100.0


def acionar_bomba_irrigacao(
    quantidade_agua_mm, area_m2, vazao_bomba_lph,
    porta='/dev/ttyACM0', baudrate=9600, timeout_arduino=10
):
    """
    Aciona a bomba via Arduino: envia "ON <segundos>\n" e aguarda retorno "OK".
    """
    if quantidade_agua_mm <= 0:
        print("Quantidade de água para irrigar é 0 mm ou negativa. Bomba não acionada.")
        return 0, 0, "NO_ACTION"

    litros = quantidade_agua_mm * area_m2
    # Evita divisão por zero se vazao_bomba_lph for 0
    dur_s = math.ceil(
        litros * 3600 / vazao_bomba_lph) if vazao_bomba_lph > 0 else 0

    if dur_s <= 0:
        print(
            f"Tempo de acionamento calculado é {dur_s}s. Bomba não acionada.")
        return litros, dur_s, "NO_ACTION_ZERO_TIME"

    print(f"Irrigação necessária: {quantidade_agua_mm:.2f} mm")
    print(f"Volume calculado: {litros:.2f} L")
    print(f"Tempo de acionamento da bomba: {dur_s} s")

    ack = None
    try:
        with serial.Serial(porta, baudrate, timeout=timeout_arduino) as ser:
            time.sleep(2)  # Aguarda a conexão serial estabilizar
            comando = f"ON {dur_s}\n"
            ser.write(comando.encode())
            print(f"Comando enviado ao Arduino: {comando.strip()}")

            # Aguarda confirmação do Arduino
            ack = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"Resposta do Arduino: '{ack}'")

            if ack != "OK":
                print(
                    f"[Alerta] Arduino não confirmou 'OK'. Resposta: '{ack}'")

    except serial.SerialException as e:
        print(
            f"[Erro Critico] Falha ao comunicar com o Arduino na porta {porta} para acionar bomba: {e}")
    except Exception as e:
        print(f"[Erro Critico] Falha inesperada ao tentar acionar bomba: {e}")

    return litros, dur_s, ack

# --------------------
# Funções de cálculo
# --------------------


def obter_dados_weatherapi(cidade, chave_api):
    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={chave_api}&q={cidade}&days=3&aqi=no&alerts=no"
    )
    try:
        resp = requests.get(url, timeout=10)  # Timeout aumentado para 10s
        resp.raise_for_status()  # Levanta um erro para códigos HTTP 4xx/5xx
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[Erro Critico] Falha ao obter dados da WeatherAPI: {e}")
        return None  # Retorna None para indicar falha


def calcular_eto(tmax, tmin, rh_max, rh_min, u10, rn, g, z):
    # Garante que rh_min não seja menor que 0 ou maior que rh_max
    # -1 para garantir que rh_min < rh_max
    rh_min_ajustado = max(0, min(rh_min, rh_max - 1))

    u2 = u10 * (4.87 / math.log(67.8 * 10 - 5.42))
    es_tmax = 0.6108 * math.exp((17.27 * tmax) / (tmax + 237.3))
    es_tmin = 0.6108 * math.exp((17.27 * tmin) / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2
    # Usa rh_min_ajustado e garante que rh_max e rh_min_ajustado estejam entre 0-100
    ea = ((es_tmax * min(100, max(0, rh_min_ajustado))/100) +
          (es_tmin * min(100, max(0, rh_max))/100)) / 2
    delta = (4098 * es) / (((tmax + tmin)/2 + 237.3)**2)
    patm = 101.3 * (((293 - 0.0065*z) / 293)**5.26)
    gamma = 0.665e-3 * patm

    # Evita divisão por zero no denominador
    denominador = delta + gamma * (1 + 0.34 * u2)
    if denominador == 0:
        return 0.0  # Ou algum outro valor de fallback apropriado

    eto = (
        (0.408 * delta * (rn - g)) +
        (gamma * (900/((tmax + tmin)/2 + 273)) * u2 * (es - ea))
    ) / denominador
    return max(0, eto)  # ETo não deve ser negativo


# float para ponto_alerta
def calcular_estresse_hidrico(umidades, etos, ponto_alerta=15.0, coef_tol=1.0):
    estresses = []
    for umidade_percentual, eto_diario in zip(umidades, etos):
        # umidade_percentual já deve estar em % (e não fração 0-1)
        if umidade_percentual <= ponto_alerta:
            # O cálculo original (eto/10) pode precisar de revisão dependendo da unidade de 'u'
            # Se 'u' é %, e ponto_alerta é %, a diferença é em %
            # Multiplicar por (eto/10) parece um fator empírico.
            estresse = max(0, (ponto_alerta - umidade_percentual)
                           * coef_tol * (eto_diario / 10.0))
        else:
            estresse = 0.0
        estresses.append(estresse)
    return estresses

# --------------------
# Funções de previsão
# --------------------


def previsao_umidade_solo(entrada_umidade_list):
    # Colunas esperadas pelo modelo de umidade do solo
    features_umidade = [
        'th1_dia1', 'MinTemp_dia2', 'MaxTemp_dia2', 'MinTemp_dia3', 'MaxTemp_dia3',
        'Precipitation_dia2', 'Precipitation_dia3', 'ReferenceET_dia2', 'ReferenceET_dia3'
    ]
    try:
        modelo_umidade = joblib.load(MODEL_TH_PATH)
        entrada_df = pd.DataFrame(
            [entrada_umidade_list], columns=features_umidade)
        previsao = modelo_umidade.predict(entrada_df)
        # A previsão pode ser uma lista de listas ou array 2D, pegamos o primeiro elemento
        return [float(x) for x in previsao[0]]
    except FileNotFoundError:
        print(
            f"[Erro Critico] Arquivo do modelo de umidade não encontrado: {MODEL_TH_PATH}")
        return None  # Retorna None em caso de erro
    except Exception as e:
        print(f"[Erro Critico] Falha ao prever umidade do solo: {e}")
        return None


def deve_irrigar_hoje(entrada_classificador_list):
    
    features_classificador = [
        'th1_dia1', 'th1_dia2', 'th1_dia3',
        'MinTemp_dia1', 'MinTemp_dia2', 'MinTemp_dia3',
        'MaxTemp_dia1', 'MaxTemp_dia2', 'MaxTemp_dia3',
        'Precipitation_dia1', 'Precipitation_dia2', 'Precipitation_dia3',
        'ReferenceET_dia1', 'ReferenceET_dia2', 'ReferenceET_dia3',
        'Estresse_hidrico_dia1', 'Estresse_hidrico_dia2', 'Estresse_hidrico_dia3'
    ]
    try:
        modelo_classificador = joblib.load(MODEL_CLASS_PATH)
        entrada_df = pd.DataFrame(
            [entrada_classificador_list], columns=features_classificador)
        predicao_classe = modelo_classificador.predict(entrada_df)
        # Retorna True se a classe prevista for 1, False caso contrário.
        return int(predicao_classe[0]) == 1
    except FileNotFoundError:
        print(
            f"[Erro Critico] Arquivo do modelo classificador não encontrado: {MODEL_CLASS_PATH}")
        return False  # Default para não irrigar em caso de erro crítico
    except Exception as e:
        print(f"[Erro Critico] Falha ao usar modelo classificador: {e}")
        return False  # Default para não irrigar


def previsao_irrigacao(entrada_irrigacao_list):
    # Colunas esperadas pelo modelo de previsão de irrigação (regressão)
    features_irrigacao = [
        'th1_dia1', 'th1_dia2', 'th1_dia3',
        'MinTemp_dia1', 'MinTemp_dia2', 'MinTemp_dia3',
        'MaxTemp_dia1', 'MaxTemp_dia2', 'MaxTemp_dia3',
        'Precipitation_dia1', 'Precipitation_dia2', 'Precipitation_dia3',
        'ReferenceET_dia1', 'ReferenceET_dia2', 'ReferenceET_dia3'
    ]
    try:
        modelo_irrigacao = joblib.load(MODEL_IRR_PATH)
        entrada_df = pd.DataFrame(
            [entrada_irrigacao_list], columns=features_irrigacao)
        predicao_qtde = modelo_irrigacao.predict(entrada_df)
        return float(predicao_qtde[0])
    except FileNotFoundError:
        print(
            f"[Erro Critico] Arquivo do modelo de irrigação não encontrado: {MODEL_IRR_PATH}")
        return 0.0  # Retorna 0 mm em caso de erro
    except Exception as e:
        print(f"[Erro Critico] Falha ao prever quantidade de irrigação: {e}")
        return 0.0

# --------------------
# Bloco principal
# --------------------

if __name__ == "__main__":
    print("--- Iniciando Sistema Autônomo de Irrigação ---")

    # --- Configurações MQTT ---
    MQTT_BROKER_HOST = "localhost"  # Ou o IP do seu Raspberry Pi se o broker estiver nele
    MQTT_BROKER_PORT = 1883
    # Usar um client_id único ajuda a evitar desconexões inesperadas se outro script usar o mesmo id
    MQTT_CLIENT_ID = f"irrigation_system_publisher_{uuid.uuid4()}" 
    
    MQTT_TOPIC_SOIL_MOISTURE = "projeto/irrigacao/umidade_solo"
    MQTT_TOPIC_WATER_STRESS_TODAY = "projeto/irrigacao/estresse_hidrico_hoje"
    MQTT_TOPIC_IRRIGATION_DECISION_TODAY = "projeto/irrigacao/decisao_irrigar_hoje" # "Sim" ou "Não"
    MQTT_TOPIC_IRRIGATION_AMOUNT_TODAY = "projeto/irrigacao/qtde_irrigada_hoje_mm"
    MQTT_TOPIC_IRRIGATION_DURATION_TODAY = "projeto/irrigacao/duracao_bomba_hoje_s"

    mqtt_client = None # Inicializa como None
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=MQTT_CLIENT_ID)
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        mqtt_client.loop_start() # Inicia uma thread para gerenciar o tráfego de rede e reconexões
        print("Conectado ao MQTT Broker.")
    except Exception as e:
        print(f"[Erro Critico] Falha ao conectar ao MQTT Broker: {e}")
        # O script continuará, mas não publicará dados MQTT

    # Parâmetros do sistema
    API_KEY   = "01ad5794da6140bfb00162541240210" # Sua chave da WeatherAPI
    CIDADE    = "Sao Paulo" # Cidade para previsão
    ALTITUDE  = 760         # Altitude local em metros
    AREA_M2   = 1           # Área da plantação em m²
    VAZAO_LPH = 1200        # Vazão da bomba em Litros por Hora
    PORTA_ARDUINO = '/dev/ttyACM0' # Ajuste para sua porta real

    # Inicializar variáveis que serão publicadas via MQTT para garantir que sempre existam
    umidade_para_mqtt = 0.0
    estresse_hoje_para_mqtt = 0.0
    decisao_irrigar_str_para_mqtt = "Não"
    final_qtde_irrigada_mm = 0.0
    final_duracao_bomba_s = 0

    # 1) Obter dados meteorológicos
    print("\n1. Obtendo dados meteorológicos...")
    dados_api = obter_dados_weatherapi(CIDADE, API_KEY)
    if dados_api is None:
        print("Não foi possível obter dados meteorológicos. Abortando execução e MQTT.")
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        exit()
        
    dias_previsao = dados_api['forecast']['forecastday']

    # 2) Extrair variáveis para 3 dias
    print("2. Extraindo variáveis meteorológicas...")
    tmax_list   = [dia['day']['maxtemp_c'] for dia in dias_previsao]
    tmin_list   = [dia['day']['mintemp_c'] for dia in dias_previsao]
    rh_max_list = [dia['day']['avghumidity'] for dia in dias_previsao]
    rh_min_list = [max(0, rh - 20) for rh in rh_max_list]
    u10_list    = [dia['day'].get('maxwind_kph', 2 * 3.6) / 3.6 for dia in dias_previsao]
    rn_list     = [(1 if dia['day']['daily_will_it_rain'] == 1 else 0.2) * 5 for dia in dias_previsao]
    g_list      = [0.0] * 3
    precip_list = [dia['day']['totalprecip_mm'] for dia in dias_previsao]

    # 3) Calcular ETo para 3 dias
    print("3. Calculando Evapotranspiração de Referência (ETo)...")
    eto_list = [calcular_eto(tmax_list[i], tmin_list[i], rh_max_list[i], rh_min_list[i], u10_list[i], rn_list[i], g_list[i], ALTITUDE) for i in range(3)]

    # 4) Ler umidade atual do solo
    print("4. Lendo umidade atual do solo...")
    umidade_atual_solo_percent = ler_umidade_solo(porta=PORTA_ARDUINO, default=30.0)
    umidade_para_mqtt = umidade_atual_solo_percent # Salva para MQTT
    print(f"   Umidade atual do solo: {umidade_atual_solo_percent:.1f}%")

    # 5) Prever umidade do solo para os próximos 2 dias
    print("5. Prevendo umidade do solo para os próximos dias...")
    umidade_atual_fracao_param_modelo = umidade_atual_solo_percent / 100.0
    entrada_previsao_umidade = [
        umidade_atual_fracao_param_modelo, tmin_list[1], tmax_list[1], tmin_list[2], tmax_list[2],
        precip_list[1], precip_list[2], eto_list[1], eto_list[2]
    ]
    previsao_umidade_dias_2_3_fracao = previsao_umidade_solo(entrada_previsao_umidade)
    
    if previsao_umidade_dias_2_3_fracao is None:
        print("Não foi possível prever a umidade do solo. Abortando irrigação e MQTT.")
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        exit()
    
    umidades_solo_3dias_percent = [
        umidade_atual_solo_percent,
        previsao_umidade_dias_2_3_fracao[0] * 100.0,
        previsao_umidade_dias_2_3_fracao[1] * 100.0
    ]
    print(f"   Umidades do solo para 3 dias (atual, D+1, D+2) [%]: {[f'{u:.1f}' for u in umidades_solo_3dias_percent]}")

    # 6) Calcular estresse hídrico para 3 dias
    print("6. Calculando estresse hídrico...")
    estresse_hidrico_3dias = calcular_estresse_hidrico(umidades_solo_3dias_percent, eto_list)
    if estresse_hidrico_3dias and len(estresse_hidrico_3dias) > 0:
        estresse_hoje_para_mqtt = estresse_hidrico_3dias[0] # Salva para MQTT
    print(f"   Estresse hídrico (hoje): {estresse_hoje_para_mqtt:.2f}")


    # 7) Decidir se a irrigação é necessária HOJE usando o classificador
    print("7. Verificando necessidade de irrigação com classificador...")
    entrada_classificador_list = [
        umidade_atual_fracao_param_modelo, previsao_umidade_dias_2_3_fracao[0], previsao_umidade_dias_2_3_fracao[1],
        tmin_list[0], tmin_list[1], tmin_list[2], tmax_list[0], tmax_list[1], tmax_list[2],
        precip_list[0], precip_list[1], precip_list[2], eto_list[0], eto_list[1], eto_list[2],
        estresse_hidrico_3dias[0], estresse_hidrico_3dias[1], estresse_hidrico_3dias[2]
    ]
    irrigacao_necessaria_hoje = deve_irrigar_hoje(entrada_classificador_list)
    decisao_irrigar_str_para_mqtt = "Sim" if irrigacao_necessaria_hoje else "Não" # Salva para MQTT
    print(f"   Decisão do classificador: {'IRRIGAR HOJE' if irrigacao_necessaria_hoje else 'NÃO IRRIGAR HOJE'}")

    # 8) Se irrigação for necessária, prever a quantidade e acionar a bomba
    if irrigacao_necessaria_hoje:
        print("8. Classificador indicou necessidade de irrigação. Calculando quantidade...")
        entrada_previsao_irrigacao = [
            umidade_atual_fracao_param_modelo, previsao_umidade_dias_2_3_fracao[0], previsao_umidade_dias_2_3_fracao[1],
            tmin_list[0], tmin_list[1], tmin_list[2], tmax_list[0], tmax_list[1], tmax_list[2],
            precip_list[0], precip_list[1], precip_list[2], eto_list[0], eto_list[1], eto_list[2]
        ]
        quantidade_irrigar_mm = previsao_irrigacao(entrada_previsao_irrigacao)
        
        if quantidade_irrigar_mm > 0:
            final_qtde_irrigada_mm = quantidade_irrigar_mm # Salva para MQTT
            print(f"   Quantidade de irrigação prevista para hoje: {final_qtde_irrigada_mm:.2f} mm")
            print("9. Acionando bomba de irrigação...")
            litros_gastos, dur_s, ack_bomba = acionar_bomba_irrigacao(
                quantidade_agua_mm=final_qtde_irrigada_mm,
                area_m2=AREA_M2,
                vazao_bomba_lph=VAZAO_LPH,
                porta=PORTA_ARDUINO
            )
            final_duracao_bomba_s = dur_s # Salva para MQTT
        else:
            print(f"   Modelo de irrigação previu {quantidade_irrigar_mm:.2f} mm. Nenhuma irrigação será realizada.")
    else:
        print("8. Irrigação não é necessária hoje conforme o classificador. Nenhuma ação de irrigação será tomada.")

    # --- Publicar dados via MQTT ---
    if mqtt_client and mqtt_client.is_connected(): # Verifica se o cliente existe e está conectado
        print("\n10. Publicando dados via MQTT...")
        try:
            # Publica com retain=True para que o Node-RED sempre pegue o último valor ao se conectar/reiniciar
            mqtt_client.publish(MQTT_TOPIC_SOIL_MOISTURE, payload=f"{umidade_para_mqtt:.1f}", qos=1, retain=True)
            mqtt_client.publish(MQTT_TOPIC_WATER_STRESS_TODAY, payload=f"{estresse_hoje_para_mqtt:.2f}", qos=1, retain=True)
            mqtt_client.publish(MQTT_TOPIC_IRRIGATION_DECISION_TODAY, payload=decisao_irrigar_str_para_mqtt, qos=1, retain=True)
            mqtt_client.publish(MQTT_TOPIC_IRRIGATION_AMOUNT_TODAY, payload=f"{final_qtde_irrigada_mm:.2f}", qos=1, retain=True)
            mqtt_client.publish(MQTT_TOPIC_IRRIGATION_DURATION_TODAY, payload=str(final_duracao_bomba_s), qos=1, retain=True) # Envia como string
            
            print(f"   Dados publicados nos tópicos '{MQTT_TOPIC_SOIL_MOISTURE}', '{MQTT_TOPIC_WATER_STRESS_TODAY}', etc.")
        except Exception as e:
            print(f"Falha ao publicar dados via MQTT: {e}")
    elif mqtt_client: # Se mqtt_client existe mas não está conectado (falha na conexão inicial)
        print("\n10. Não foi possível publicar dados via MQTT (cliente não conectado).")


    print("\n--- Sistema Autônomo de Irrigação Concluído ---")

    # --- Desconectar do MQTT Broker ---
    if mqtt_client:
        mqtt_client.loop_stop() # Para a thread de rede
        mqtt_client.disconnect()
        print("Desconectado do MQTT Broker.")