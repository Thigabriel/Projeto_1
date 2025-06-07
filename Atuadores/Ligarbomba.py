import serial
import time
import math

def acionar_bomba_irrigacao(quantidade_agua_mm,
                            area_m2,
                            vazao_bomba_lph,
                            porta_serial='/dev/ttyACM0',
                            baudrate=9600):
   
    # 1 mm = 1 L/m²
    litros = quantidade_agua_mm * area_m2

    # calcula tempo em segundos
    tempo_s = litros * 3600 / vazao_bomba_lph
    duracao_s = math.ceil(tempo_s)   # arredonda para cima

    print(f"Litros necessários: {litros:.2f} L")
    print(f"Bomba ligada por: {duracao_s} s")

    comando = f"ON {duracao_s}\n"     # protocolo simples: "ON 30"

    try:
        with serial.Serial(porta_serial, baudrate, timeout=2) as arduino:
            time.sleep(2)              # aguarda reset do Arduino
            arduino.write(comando.encode())
            ack = arduino.readline().decode().strip()
            print("Resposta do Arduino:", ack)
    except serial.SerialException as e:
        print("Erro na porta serial:", e)
        ack = None
    except Exception as e:
        print("Erro inesperado:", e)
        ack = None

    return litros, duracao_s, ack



