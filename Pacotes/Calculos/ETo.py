import requests
import math


def obter_dados_weatherapi(cidade, chave_api):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={chave_api}&q={cidade}&days=3&aqi=no&alerts=no"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Erro ao acessar a WeatherAPI: {response.status_code}")


def calcular_eto(tmax, tmin, rh_max, rh_min, u10, rn, g, z):
    u2 = u10 * (4.87 / math.log(67.8 * 10 - 5.42))
    es_tmax = 0.6108 * math.exp((17.27 * tmax) / (tmax + 237.3))
    es_tmin = 0.6108 * math.exp((17.27 * tmin) / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2
    ea = ((es_tmax * rh_max / 100) + (es_tmin * rh_min / 100)) / 2
    delta = (4098 * es) / ((tmax + tmin) / 2 + 237.3)**2
    patm = 101.3 * ((293 - 0.0065 * z) / 293)**5.26
    gamma = 0.665 * 10**-3 * patm
    eto = ((0.408 * delta * (rn - g)) + (gamma * (900 / ((tmax + tmin) /
           2 + 273)) * u2 * (es - ea))) / (delta + gamma * (1 + 0.34 * u2))
    return eto


def calcular_agua_estimada(api_key, cidade, kc=1, area=1, altitude=760):
    dados = obter_dados_weatherapi(cidade, api_key)
    previsoes = dados['forecast']['forecastday']
    agua_estimada = []

    for dia in previsoes:
        tmax = dia['day']['maxtemp_c']
        tmin = dia['day']['mintemp_c']
        rh_max = dia['day']['avghumidity']
        rh_min = rh_max - 10
        rn = dia['day']['daily_will_it_rain'] * 5
        u10 = 2
        g = 0
        z = altitude
        eto = calcular_eto(tmax, tmin, rh_max, rh_min, u10, rn, g, z)
        etc = eto * kc
        volume_agua = etc * area
        agua_estimada.append({
            "data": dia['date'],
            "eto": eto,
            "etc": etc,
            "volume_agua": volume_agua
        })
    return agua_estimada


api_key = "01ad5794da6140bfb00162541240210"
cidade = "São Paulo"



