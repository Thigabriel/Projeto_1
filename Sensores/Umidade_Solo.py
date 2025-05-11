import serial
import time

def ler_umidade_solo(porta_serial='/dev/ttyACM0',
                     baudrate=9600,
                     timeout=2,
                     canal_analogico=0,
                     min_valor=200,
                     max_valor=900,
                     inverter=True):
    """
    Lê umidade do solo via Arduino conectado na porta USB.

    Parâmetros opcionais:
    - porta_serial: ex. '/dev/ttyACM0' (padrão) ou 'COM5'
    - baudrate: padrão 9600
    - timeout: tempo de espera pela resposta, em segundos
    - canal_analogico: 0 para A0, 1 para A1, etc.
    - min_valor: leitura bruta no solo mais seco
    - max_valor: leitura bruta no solo mais úmido
    - inverter: True se o sensor retorna 0=úmido e 1023=seco
    """

    try:
        with serial.Serial(porta_serial, baudrate, timeout=timeout) as arduino:
            time.sleep(2)  # espera o Arduino reiniciar
            # envia comando para ler o sensor
            comando = f"R {canal_analogico}\n"
            arduino.write(comando.encode())
            linha = arduino.readline().decode().strip()
            valor_bruto = int(linha)
    except serial.SerialException as e:
        raise RuntimeError(f"Erro de comunicação: {e}")
    except ValueError:
        raise RuntimeError(f"Resposta inválida do Arduino: '{linha}'")

    # calibração e normalização
    valor = max(min_valor, min(max_valor, valor_bruto))
    frac  = (valor - min_valor) / float(max_valor - min_valor)
    if inverter:
        frac = 1.0 - frac

    return frac * 100.0  # retorna umidade em %




