import qi
import time

NAO_IP = "192.168.0.30"
NAO_PORT = 9559

# --- PARÁMETROS DEL RETO ---
DISTANCIA_OBJETIVO = 1.0  # Metros
VELOCIDAD_X = 0.5         # Velocidad relativa (0.0 a 1.0)

# --- CONSTANTES PID ---
Kp = 0
Ki = 0.00
Kd = 0.0


def comprobar_tacto_cabeza(memory):
    f = memory.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
    m = memory.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
    r = memory.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
    return (f > 0.5 or m > 0.5 or r > 0.5)

def main():
    app = qi.Application([], url=f"tcp://{NAO_IP}:{NAO_PORT}")
    app.start()
    
    motion = app.session.service("ALMotion")
    pose = app.session.service("ALRobotPosture")
    memory = app.session.service("ALMemory")
    
    # 1. PREPARACIÓN Y SEGURIDAD
    motion.wakeUp()
    pose.goToPosture("StandInit", 0.5)
    
    config = [
        ["MaxStepX", 0.2],
        ["MaxStepY", 0.12],
        ["StepHeight", 0.8],
        ["MaxStepTheta", 0.04],
        ["MaxStepFrequency", 0.8]
    ] 
    
    motion.moveInit()
    
    posicion_inicial = motion.getRobotPosition(True)
    x_inicial = posicion_inicial[0]
    theta_objetivo = posicion_inicial[2]
    
    error_previo = 0.0
    integral = 0.0
    distancia_recorrida = 0.0
    tiempo_previo = time.time()

    interrupcion_manual = False

    print(f"Iniciando avance PID. Objetivo: {DISTANCIA_OBJETIVO} m")
    print("Nota: Toca la cabeza en cualquier momento para cancelar.")

    # 2. BUCLE PRINCIPAL (PID + MONITOR TÁCTIL)
    while distancia_recorrida < DISTANCIA_OBJETIVO:
        # A. Chequeo prioritario del sensor táctil
        if comprobar_tacto_cabeza(memory):
            print("\n[INTERRUPCIÓN] Se tocó la cabeza. Cancelando caminata PID...")
            interrupcion_manual = True
            break
            
        tiempo_actual = time.time()
        dt = tiempo_actual - tiempo_previo
        if dt <= 0:
            dt = 0.01
        
        # B. Leer odometría
        pos_actual = motion.getRobotPosition(True)
        x_actual = pos_actual[0]
        theta_actual = pos_actual[2]
        
        # C. Distancia recorrida en X
        distancia_recorrida = x_actual - x_inicial
        
        # D. Cálculo PID
        error = theta_objetivo - theta_actual
        proporcional = Kp * error
        integral += error * dt
        derivativo = Kd * (error - error_previo) / dt
        
        correccion_theta = proporcional + integral + derivativo
        
        # Clamping de seguridad
        max_gir_rate = 0.2
        correccion_theta = max(-max_gir_rate, min(max_gir_rate, correccion_theta))
        
        # E. Enviar comando de velocidad
        motion.moveToward(VELOCIDAD_X, 0.0, correccion_theta, config)
        
        error_previo = error
        tiempo_previo = tiempo_actual
        time.sleep(0.05)
        
        print(f"Dist: {distancia_recorrida:.2f}m | Error: {error:.4f} | Corr: {correccion_theta:.4f}")

    # 3. RUTINA DE APAGADO / CANCELACIÓN
    if interrupcion_manual:
        print("Ejecutando parada de emergencia y sentando al robot...")
        motion.stopMove()
        pose.goToPosture("Sit", 0.5)
        motion.rest()
    else:
        print("Objetivo completado con éxito. Deteniendo...")
        motion.moveToward(0.0, 0.0, 0.0, config)
        motion.stopMove()
        pose.goToPosture("Sit", 0.5)
        motion.rest()

if __name__ == "__main__":
    main()