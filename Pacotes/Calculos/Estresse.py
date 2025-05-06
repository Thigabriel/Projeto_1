

def calcular_estresse_hidrico(umidades_solo, eto, ponto_alerta=15, coef_tolerancia=1):

    estresses_hidricos = []

    for i, umidade in enumerate(umidades_solo):
        if umidade <= ponto_alerta:
            estresse = max(0, (ponto_alerta - umidade) *
                           coef_tolerancia * (eto[i] / 10))
        else:
            estresse = 0

        estresses_hidricos.append(estresse)

    return estresses_hidricos
