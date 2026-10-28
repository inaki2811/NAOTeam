import csv
import datetime
import json
import logging
import math
import os
import time

import qi

NAO_IP = "192.168.0.30"
NAO_PORT = 9559

# --- PARÁMETROS DEL RETO ---
DISTANCIA_OBJETIVO = 1.0   # Metros
VELOCIDAD_X = 0.5          # Velocidad relativa máxima (0.0 a 1.0)
VELOCIDAD_MIN = 0.15       # Evita que el NAO deje de dar pasos antes de llegar
TOLERANCIA_M = 0.02        # Se considera meta alcanzada a +-2 cm
PERIODO_S = 0.05           # Periodo del bucle de control
TIEMPO_MAX_S = 60.0        # Seguridad: aborta si tarda demasiado

# --- CONSTANTES PID (valores de partida, hay que afinarlos con las pruebas) ---
# Rumbo (salida = velocidad angular relativa, theta de moveToward)
KP_RUMBO, KI_RUMBO, KD_RUMBO = 1.0, 0.0, 0.05
MAX_GIRO = 0.2
# Distancia (salida = velocidad relativa hacia adelante, x de moveToward)
KP_DIST, KI_DIST, KD_DIST = 1.5, 0.0, 0.0

# Configuración de marcha deseada. Se recorta a los límites reales del robot
# (getMoveConfig "Min"/"Max"), y cada ajuste queda registrado en el log.
CONFIG_DESEADA = {
    "MaxStepX": 0.3,
    "MaxStepY": 0.12,
    "StepHeight": 0.8,
    "MaxStepTheta": 0.04,
    "MaxStepFrequency": 0.8,
}


def wrap_pi(a):
    """Normaliza un ángulo a [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, i_max=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_max = i_max            # anti-windup de la integral
        self.integral = 0.0
        self.prev = None

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.i_max, min(self.i_max, self.integral))
        deriv = 0.0 if self.prev is None else (error - self.prev) / dt
        self.prev = error
        salida = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(self.out_min, min(self.out_max, salida))


def comprobar_tacto_cabeza(memory):
    f = memory.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
    m = memory.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
    r = memory.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
    return (f > 0.5 or m > 0.5 or r > 0.5)


def ajustar_config(motion, deseada):
    """Recorta la config de marcha a los límites reales del robot."""
    try:
        minimos = dict(motion.getMoveConfig("Min"))
        maximos = dict(motion.getMoveConfig("Max"))
    except Exception:
        logging.exception("No se pudieron leer los límites de marcha; uso la config tal cual")
        return [[k, v] for k, v in deseada.items()]

    config = []
    for nombre, valor in deseada.items():
        lo = minimos.get(nombre, valor)
        hi = maximos.get(nombre, valor)
        ajustado = max(lo, min(hi, valor))
        if ajustado != valor:
            logging.warning("%s=%s fuera de rango [%s, %s]; se usa %s",
                            nombre, valor, lo, hi, ajustado)
        config.append([nombre, ajustado])
    return config


def crear_corrida(params):
    """Crea logs/<fecha_hora>/ con params.json y eventos.log."""
    sello = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(base, "logs", sello)
    os.makedirs(carpeta, exist_ok=True)

    with open(os.path.join(carpeta, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(carpeta, "eventos.log")),
            logging.StreamHandler(),
        ],
    )
    return carpeta


def main():
    params = {
        "distancia_objetivo": DISTANCIA_OBJETIVO,
        "velocidad_x": VELOCIDAD_X,
        "velocidad_min": VELOCIDAD_MIN,
        "tolerancia_m": TOLERANCIA_M,
        "Kp_rumbo": KP_RUMBO, "Ki_rumbo": KI_RUMBO, "Kd_rumbo": KD_RUMBO,
        "max_giro": MAX_GIRO,
        "Kp_dist": KP_DIST, "Ki_dist": KI_DIST, "Kd_dist": KD_DIST,
        "config_deseada": CONFIG_DESEADA,
    }
    carpeta = crear_corrida(params)
    logging.info("Corrida en %s", carpeta)

    app = qi.Application([], url=f"tcp://{NAO_IP}:{NAO_PORT}")
    app.start()

    motion = app.session.service("ALMotion")
    pose = app.session.service("ALRobotPosture")
    memory = app.session.service("ALMemory")

    pid_dist = PID(KP_DIST, KI_DIST, KD_DIST, 0.0, VELOCIDAD_X)
    pid_rumbo = PID(KP_RUMBO, KI_RUMBO, KD_RUMBO, -MAX_GIRO, MAX_GIRO)

    motivo = "error"
    x0 = y0 = th0 = None
    c = s = 0.0
    t_inicio = time.time()

    archivo_csv = open(os.path.join(carpeta, "telemetria.csv"), "w", newline="")
    escritor = csv.writer(archivo_csv)
    escritor.writerow(["t", "dt", "x", "y", "theta", "avance", "lateral",
                       "err_dist", "err_theta", "vx", "w"])

    try:
        # 1. PREPARACIÓN
        motion.wakeUp()
        pose.goToPosture("StandInit", 0.5)
        config = ajustar_config(motion, CONFIG_DESEADA)
        motion.moveInit()

        x0, y0, th0 = motion.getRobotPosition(True)
        c, s = math.cos(th0), math.sin(th0)

        logging.info("Iniciando avance PID. Objetivo: %.2f m", DISTANCIA_OBJETIVO)
        logging.info("Toca la cabeza en cualquier momento para cancelar.")

        t_prev = time.time()
        filas = 0

        # 2. BUCLE PRINCIPAL (PID de rumbo + PID de distancia + monitor táctil)
        while True:
            if comprobar_tacto_cabeza(memory):
                logging.info("[INTERRUPCIÓN] Se tocó la cabeza. Cancelando caminata.")
                motivo = "tacto"
                break

            t = time.time()
            if t - t_inicio > TIEMPO_MAX_S:
                logging.error("Tiempo máximo excedido (%.0f s). Abortando.", TIEMPO_MAX_S)
                motivo = "timeout"
                break
            dt = max(t - t_prev, 1e-3)
            t_prev = t

            # Odometría (marco mundo) -> avance y desviación respecto al rumbo inicial
            x, y, th = motion.getRobotPosition(True)
            dx, dy = x - x0, y - y0
            avance = dx * c + dy * s        # distancia recorrida a lo largo del rumbo inicial
            lateral = -dx * s + dy * c      # desviación lateral (+ = izquierda)

            err_dist = DISTANCIA_OBJETIVO - avance
            if err_dist <= TOLERANCIA_M:
                logging.info("Meta alcanzada (avance %.3f m).", avance)
                motivo = "meta"
                break

            err_theta = wrap_pi(th0 - th)

            vx = pid_dist.update(err_dist, dt)
            vx = max(vx, VELOCIDAD_MIN)
            w = pid_rumbo.update(err_theta, dt)

            motion.moveToward(vx, 0.0, w, config)

            escritor.writerow([f"{t - t_inicio:.3f}", f"{dt:.4f}", f"{x:.4f}", f"{y:.4f}",
                               f"{th:.4f}", f"{avance:.4f}", f"{lateral:.4f}",
                               f"{err_dist:.4f}", f"{err_theta:.4f}",
                               f"{vx:.3f}", f"{w:.3f}"])
            filas += 1
            if filas % 20 == 0:
                archivo_csv.flush()

            print(f"Dist: {avance:.2f}m | Lat: {lateral:+.3f}m | "
                  f"ErrRumbo: {err_theta:+.4f} | vx: {vx:.2f} | w: {w:+.3f}")
            time.sleep(PERIODO_S)

    except KeyboardInterrupt:
        logging.warning("Interrumpido con Ctrl+C.")
        motivo = "ctrl_c"
    except Exception:
        logging.exception("Error durante la ejecución")
        motivo = "error"
    finally:
        # 3. PARADA SEGURA (siempre se ejecuta)
        avance_final = None
        try:
            motion.stopMove()
            time.sleep(1.0)  # deja que terminen los pasos por inercia
            if x0 is not None:
                x, y, _ = motion.getRobotPosition(True)
                avance_final = (x - x0) * c + (y - y0) * s
                logging.info("Avance final estable (odometría): %.3f m", avance_final)
        except Exception:
            logging.exception("Error al detener el movimiento")

        try:
            pose.goToPosture("Sit", 0.5)
            motion.rest()
        except Exception:
            logging.exception("Error al sentar al robot")

        archivo_csv.close()
        resultado = {
            "motivo": motivo,
            "tiempo_s": round(time.time() - t_inicio, 2),
            "avance_final_estable": avance_final,
            "medido_cm": None,  # llénalo a mano con cinta métrica
        }
        with open(os.path.join(carpeta, "resultado.json"), "w") as f:
            json.dump(resultado, f, indent=2)
        logging.info("Fin de la corrida: %s", resultado)


if __name__ == "__main__":
    main()