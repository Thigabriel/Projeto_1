import serial
import time

def ler_umidade_solo(porta_serial='/dev/ttyACM0',
                     baudrate=9600,
                     timeout=2,
                     canal_analogico=0,
                     min_valor=200,
                     max_valor=900,
                     inverter=True):
  

    try:
        with serial.Serial(porta_serial, baudrate, timeout=timeout) as arduino:
            time.sleep(2)  
          
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




