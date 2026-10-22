from os import name

import qi
import time

NAO_IP = "192.168.0.30"
NAO_PORT = 9559

def comprobar_tacto_cabeza(memory):
    
    frontal = memory.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
    central = memory.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
    trasero = memory.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
    
    return (frontal > 0.5 or central > 0.5 or trasero > 0.5)

def main():
    app = qi.Application([], url=f"tcp://{NAO_IP}:{NAO_PORT}")
    app.start()
    
    motion = app.session.service("ALMotion")
    pose = app.session.service("ALRobotPosture")
    memory = app.session.service("ALMemory")
    
    print("Parando NAO")
    motion.wakeUp()
    pose.goToPosture("StandInit", 0.5)
    
    print("\n[MODO PRUEBA SENSOR] Toca la cabeza de NAO para interrumpir y sentarlo...")
    
    cancelado = False
    
    # 2. BUCLE DE MONITOREO
    while True:
        if comprobar_tacto_cabeza(memory):
            print("\nSensor detectado. Rutina cancelada")
            cancelado = True
            break
        
        time.sleep(0.05) 

    if cancelado:
        print("Deteniendo movimientos")
        motion.stopMove()
        
        print("Sentando robot")
        pose.goToPosture("Sit", 0.5)
        
        print("Activar pose de seguridad")
        motion.rest()
        print("Rutina finalizada")

if __name__ == "__main__":
    main()