#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
# Suprime aviso de unpickle de versão do sklearn
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator .* from version .* when using version .*"
)

import os
import time
import serial
import math
import joblib
import pandas as pd
import requests

# --- Configurações de caminhos absolutos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TH_PATH = os.path.join(BASE_DIR, "modelo_umidade_treinado.joblib")
MODEL_IRR_PATH = os.path.join(BASE_DIR, "modelo_irrigacao_treinado.joblib")

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
        with serial.Serial(porta, baud, timeout=timeout) as ser:
            time.sleep(2)
            ser.reset_input_buffer()
            ser.write(b"R 0\n")
            linha = ""
            deadline = time.time() + timeout
            while time.time() < deadline:
                linha = ser.readline().decode().strip()
                if linha:
                    break
            if not linha:
                raise RuntimeError("sem resposta do Arduino em leitura")
            valor_bruto = int(linha)
    except Exception as e:
        print(f"[Aviso] Falha na leitura do sensor: {e}. Usando valor padrão {default}.")
        return default

    valor = max(min_valor, min(max_valor, valor_bruto))
    frac  = (valor - min_valor) / float(max_valor - min_valor)
    if inverter:
        frac = 1.0 - frac
    return frac * 100.0


def acionar_bomba_irrigacao(
    quantidade_agua_mm, area_m2, vazao_bomba_lph,
    porta='/dev/ttyACM0', baudrate=9600
):
    """
    Aciona a bomba via Arduino: envia "ON <segundos>\n" e aguarda retorno "OK".
    """
    litros = quantidade_agua_mm * area_m2
    dur_s  = math.ceil(litros * 3600 / vazao_bomba_lph)
    print(f"Litros necessários: {litros:.2f} L")
    print(f"Bomba ligada por: {dur_s} s")
    try:
        with serial.Serial(porta, baudrate, timeout=5) as ser:
            time.sleep(2)
            ser.write(f"ON {dur_s}\n".encode())
            ack = ser.readline().decode().strip()
        print("Resposta do Arduino:", ack)
    except Exception as e:
        print(f"[Erro] Falha ao acionar bomba: {e}")
        ack = None
    return litros, dur_s, ack

# --------------------
# Funções de cálculo
# --------------------

def obter_dados_weatherapi(cidade, chave_api):
    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={chave_api}&q={cidade}&days=3&aqi=no&alerts=no"
    )
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()


def calcular_eto(tmax, tmin, rh_max, rh_min, u10, rn, g, z):
    u2       = u10 * (4.87 / math.log(67.8 * 10 - 5.42))
    es_tmax  = 0.6108 * math.exp((17.27 * tmax) / (tmax + 237.3))
    es_tmin  = 0.6108 * math.exp((17.27 * tmin) / (tmin + 237.3))
    es       = (es_tmax + es_tmin) / 2
    ea       = ((es_tmax * rh_max/100) + (es_tmin * rh_min/100)) / 2
    delta    = (4098 * es) / (((tmax + tmin)/2 + 237.3)**2)
    patm     = 101.3 * (((293 - 0.0065*z) / 293)**5.26)
    gamma    = 0.665e-3 * patm
    eto      = (
        (0.408 * delta * (rn - g)) +
        (gamma * (900/((tmax + tmin)/2 + 273)) * u2 * (es - ea))
    ) / (delta + gamma * (1 + 0.34 * u2))
    return eto


def calcular_estresse_hidrico(umidades, etos, ponto_alerta=15, coef_tol=1):
    est = []
    for u, eto in zip(umidades, etos):
        if u <= ponto_alerta:
            e = max(0, (ponto_alerta - u) * coef_tol * (eto/10))
        else:
            e = 0
        est.append(e)
    return est

# --------------------
# Funções de previsão
# --------------------

def previsao_umidade_solo(entrada):
    cols = [
        'th1_dia1','MinTemp_dia2','MaxTemp_dia2',
        'MinTemp_dia3','MaxTemp_dia3',
        'Precipitation_dia2','Precipitation_dia3',
        'ReferenceET_dia2','ReferenceET_dia3'
    ]
    m = joblib.load(MODEL_TH_PATH)
    df = pd.DataFrame([entrada], columns=cols)
    return [float(x) for x in m.predict(df)[0]]


def previsao_irrigacao(entrada):
    cols = [
        'th1_dia1','th1_dia2','th1_dia3',
        'MinTemp_dia1','MinTemp_dia2','MinTemp_dia3',
        'MaxTemp_dia1','MaxTemp_dia2','MaxTemp_dia3',
        'Precipitation_dia1','Precipitation_dia2','Precipitation_dia3',
        'ReferenceET_dia1','ReferenceET_dia2','ReferenceET_dia3'
    ]
    m = joblib.load(MODEL_IRR_PATH)
    df = pd.DataFrame([entrada], columns=cols)
    return float(m.predict(df)[0])

# --------------------
# Bloco principal
# --------------------

if __name__ == "__main__":
    # Parâmetros do sistema
    API_KEY   = "01ad5794da6140bfb00162541240210"
    CIDADE    = "Sao Paulo"
    ALTITUDE  = 760
    AREA_M2   = 1
    VAZAO_LPH = 1200

    # 1) Obter dados meteorológicos
    data = obter_dados_weatherapi(CIDADE, API_KEY)
    dias = data['forecast']['forecastday']

    # 2) Extrair variáveis para 3 dias
    tmax   = [d['day']['maxtemp_c'] for d in dias]
    tmin   = [d['day']['mintemp_c'] for d in dias]
    rh_max = [d['day']['avghumidity'] for d in dias]
    rh_min = [h - 10 for h in rh_max]
    u10    = [2] * 3
    rn     = [d['day']['daily_will_it_rain']*5 for d in dias]
    g      = [0] * 3
    precip = [d['day']['totalprecip_mm'] for d in dias]


    for i, v in enumerate(tmax, start=1):
        print(f"Dia {i} → Tmax: {v} °C")
    for i, v in enumerate(tmin, start=1):
        print(f"Dia {i} → Tmin: {v} °C")
    for i, v in enumerate(precip, start=1):
        print(f"Dia {i} → Precipitação: {v} mm")


    

    # 3) Calcular ETo para 3 dias
    etos = [
        calcular_eto(tmax[i], tmin[i], rh_max[i], rh_min[i],
                     u10[i], rn[i], g[i], ALTITUDE)
        for i in range(3)
    ]
    for i, val in enumerate(etos, start=1):
        print(f"ETo dia {i}: {val:.2f} mm/dia")

    # 4) Ler umidade atual do solo
    um_atual = ler_umidade_solo()/100
    print(f"Umidade atual do solo: {um_atual:.1f}%")

    # 5) Prever umidade para dias 2 e 3
    entrada_umidade = [
        um_atual,
        tmin[1], tmax[1],
        tmin[2], tmax[2],
        precip[1], precip[2],
        etos[1], etos[2]
    ]
    pred_um = previsao_umidade_solo(entrada_umidade)
    print("Previsão umidade (dias 2 e 3):", pred_um)

    # 6) Calcular estresse hídrico
    umidades = [um_atual] + pred_um
    estresse = calcular_estresse_hidrico(umidades, etos)
    print("Estresses hídricos:", estresse)

    # 7) Prever irrigação para hoje (dia 1)
    entrada_irrigacao = [
        um_atual, pred_um[0], pred_um[1],
        tmin[0], tmin[1], tmin[2],
        tmax[0], tmax[1], tmax[2],
        precip[0], precip[1], precip[2],
        etos[0], etos[1], etos[2]
    ]
    pred_ir = previsao_irrigacao(entrada_irrigacao)
    print(f"Previsão de irrigação (hoje): {pred_ir:.2f} mm")

    # 8) Acionar bomba de irrigação
    acionar_bomba_irrigacao(
        quantidade_agua_mm=pred_ir,
        area_m2=AREA_M2,
        vazao_bomba_lph=VAZAO_LPH
    )
