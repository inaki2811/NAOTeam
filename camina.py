import qi
import time
from sys import argv

NAO_IP = "192.168.0.151"  # 127.0.0.1 si usas el robot virtual en Linux,
                  # o la IP de Windows si corres el script desde WSL
NAO_PORT = 9561

def main():
    app = qi.Application([], url=f"tcp://{NAO_IP}:{NAO_PORT}")
    app.start()
    
    motion = app.session.service("ALMotion")
    pose = app.session.service("ALRobotPosture")
    
    motion.wakeUp()
    pose.goToPosture("Stand", 0.5)
        
    config = [
        ["MaxStepX", 0.2],
        ["MaxStepY", 0.12],
        ["StepHeight", 0.8],
        ["MaxStepTheta", 0.04],
        ["MaxStepFrequency", 0.8]
    ] 
    
    motion.moveToward(1, 0, 0, config)
    prevoius_time = time.time()
    
    while time.time() - prevoius_time < 10:
        print("Waiiiiitiiiiingggggggggg")
    motion.moveToward(0, 0, 0, config)
    
    motion.rest()

if __name__ == "__main__":
    main()