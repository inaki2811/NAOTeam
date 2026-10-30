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
VELOCIDAD_MIN = 0.15       # Evita que el NAO deje de dar pasos antes de llegar
TOLERANCIA_M = 0.02        # Se considera meta alcanzada a +-2 cm
PERIODO_S = 0.05           # Periodo del bucle de control
TIEMPO_MAX_S = 60.0        # Seguridad: aborta si tarda demasiado
ACELERACION = 1.5          # Rampa de arranque (vel. relativa por segundo): evita sacudidas

# --- PERFIL DE MARCHA ---
# "seguro": equivale a lo que ya funcionó en tus logs (~0.06 m/s).
# "rapido": velocidad completa + inclinación de torso hacia adelante (como al caminar rápido).
PERFIL = "rapido"

PERFILES = {
    "seguro": {
        "velocidad_x": 0.5,
        "config": {
            "MaxStepX": 0.075, "MaxStepY": 0.15, "StepHeight": 0.03,
            "MaxStepTheta": 0.04, "MaxStepFrequency": 0.8,
        },
    },
    "rapido": {
        "velocidad_x": 1.0,
        "config": {
            "MaxStepX": 0.08, "MaxStepY": 0.15, "StepHeight": 0.025,
            "MaxStepTheta": 0.08, "MaxStepFrequency": 1.0,
            "TorsoWx": 0.0, "TorsoWy": 0.05,   # inclinación del torso (rad); +Wy = hacia adelante
        },
    },
}

# --- CONSTANTES PID (valores de partida, afinar con los logs) ---
# Rumbo (salida = velocidad angular relativa, theta de moveToward)
KP_RUMBO, KI_RUMBO, KD_RUMBO = 0.5, 0.0, 0.0
MAX_GIRO = 0.3
# Distancia (salida = velocidad relativa hacia adelante). Con Kp=3 empieza a
# frenar a ~0.33 m de la meta, para no pasarse por la inercia de los pasos.
KP_DIST, KI_DIST, KD_DIST = 3.0, 0.0, 0.0

# --- CABEZA Y ESTABILIDAD ---
DESACTIVAR_VIDA_AUTONOMA = True   # evita que Autonomous Life/BasicAwareness mueva la cabeza
ESTABILIZAR_CABEZA = True         # la cabeza compensa la inclinación del torso (como un humano)
GANANCIA_CABEZA = 1.0
LIM_CABEZA = 0.3                  # rad, máxima compensación
LIM_CAIDA = 0.5                   # rad, inclinación del torso que se considera caída

CLAVES_SENSORES = [
    "Device/SubDeviceList/Head/Touch/Front/Sensor/Value",
    "Device/SubDeviceList/Head/Touch/Middle/Sensor/Value",
    "Device/SubDeviceList/Head/Touch/Rear/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value",
]


def wrap_pi(a):
    """Normaliza un ángulo a [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, i_max=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_max = i_max            # anti-windup de la integral
        self.integral = 0.0
        self.prev = None

    def update(self, error, dt):
        self.integral = limitar(self.integral + error * dt, -self.i_max, self.i_max)
        deriv = 0.0 if self.prev is None else (error - self.prev) / dt
        self.prev = error
        salida = self.kp * error + self.ki * self.integral + self.kd * deriv
        return limitar(salida, self.out_min, self.out_max)


def leer_sensores(memory):
    """Una sola llamada a ALMemory: tacto de cabeza + inclinación del torso."""
    f, m, r, ang_x, ang_y = memory.getListData(CLAVES_SENSORES)
    tocado = (f > 0.5 or m > 0.5 or r > 0.5)
    return tocado, ang_x, ang_y


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
        if nombre not in minimos or nombre not in maximos:
            logging.warning("%s no existe en esta versión de NAOqi; se omite", nombre)
            continue
        lo, hi = minimos[nombre], maximos[nombre]
        ajustado = limitar(valor, lo, hi)
        if ajustado != valor:
            logging.warning("%s=%s fuera de rango [%s, %s]; se usa %s",
                            nombre, valor, lo, hi, ajustado)
        config.append([nombre, ajustado])
    logging.info("Config de marcha efectiva: %s", config)
    return config


def desactivar_vida_autonoma(app):
    try:
        vida = app.session.service("ALAutonomousLife")
        estado = vida.getState()
        if estado != "disabled":
            logging.info("Desactivando Autonomous Life (estaba en '%s')", estado)
            vida.setState("disabled")
    except Exception:
        logging.warning("No se pudo desactivar Autonomous Life", exc_info=True)


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


def rms(valores):
    return math.sqrt(sum(v * v for v in valores) / len(valores)) if valores else None


def main():
    perfil = PERFILES[PERFIL]
    velocidad_x = perfil["velocidad_x"]

    params = {
        "perfil": PERFIL,
        "distancia_objetivo": DISTANCIA_OBJETIVO,
        "velocidad_x": velocidad_x,
        "velocidad_min": VELOCIDAD_MIN,
        "aceleracion": ACELERACION,
        "tolerancia_m": TOLERANCIA_M,
        "Kp_rumbo": KP_RUMBO, "Ki_rumbo": KI_RUMBO, "Kd_rumbo": KD_RUMBO,
        "max_giro": MAX_GIRO,
        "Kp_dist": KP_DIST, "Ki_dist": KI_DIST, "Kd_dist": KD_DIST,
        "estabilizar_cabeza": ESTABILIZAR_CABEZA,
        "ganancia_cabeza": GANANCIA_CABEZA,
        "config_deseada": perfil["config"],
    }
    carpeta = crear_corrida(params)
    logging.info("Corrida en %s (perfil %s)", carpeta, PERFIL)

    app = qi.Application([], url=f"tcp://{NAO_IP}:{NAO_PORT}")
    app.start()

    motion = app.session.service("ALMotion")
    pose = app.session.service("ALRobotPosture")
    memory = app.session.service("ALMemory")

    pid_dist = PID(KP_DIST, KI_DIST, KD_DIST, 0.0, velocidad_x)
    pid_rumbo = PID(KP_RUMBO, KI_RUMBO, KD_RUMBO, -MAX_GIRO, MAX_GIRO)

    motivo = "error"
    x0 = y0 = th0 = None
    c = s = 0.0
    angs_x, angs_y = [], []
    t_inicio = time.time()

    archivo_csv = open(os.path.join(carpeta, "telemetria.csv"), "w", newline="")
    escritor = csv.writer(archivo_csv)
    escritor.writerow(["t", "dt", "x", "y", "theta", "avance", "lateral",
                       "err_dist", "err_theta", "vx", "w", "ang_x", "ang_y", "cab_pitch"])

    try:
        # 1. PREPARACIÓN
        if DESACTIVAR_VIDA_AUTONOMA:
            desactivar_vida_autonoma(app)
        motion.wakeUp()
        pose.goToPosture("StandInit", 0.5)   # posturas válidas: Stand, StandInit, StandZero, Crouch, Sit, SitRelax...
        config = ajustar_config(motion, perfil["config"])
        motion.moveInit()
        try:
            motion.setMoveArmsEnabled(True, True)   # balanceo natural de brazos al caminar
        except Exception:
            logging.warning("No se pudo habilitar el balanceo de brazos", exc_info=True)

        # Cabeza recta y rígida
        motion.setStiffnesses("Head", 1.0)
        motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
        time.sleep(0.8)  # deja asentarse al robot antes de medir la referencia

        x0, y0, th0 = motion.getRobotPosition(True)
        c, s = math.cos(th0), math.sin(th0)
        _, ax0, ay0 = leer_sensores(memory)   # inclinación de referencia (parado)
        ang_y_f = ay0

        logging.info("Iniciando avance PID. Objetivo: %.2f m", DISTANCIA_OBJETIVO)
        logging.info("Toca la cabeza en cualquier momento para cancelar.")

        t_prev = time.time()
        vx_cmd = 0.0
        filas = 0

        # 2. BUCLE PRINCIPAL (PID rumbo + PID distancia + estabilidad + monitor táctil)
        while True:
            tocado, ang_x, ang_y = leer_sensores(memory)
            if tocado:
                logging.info("[INTERRUPCIÓN] Se tocó la cabeza. Cancelando caminata.")
                motivo = "tacto"
                break
            if abs(ang_x - ax0) > LIM_CAIDA or abs(ang_y - ay0) > LIM_CAIDA:
                logging.error("Inclinación excesiva (x=%.2f, y=%.2f rad). Posible caída.",
                              ang_x - ax0, ang_y - ay0)
                motivo = "caida"
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

            # Velocidad hacia adelante: PID de distancia + rampa de arranque
            vx_obj = max(pid_dist.update(err_dist, dt), VELOCIDAD_MIN)
            vx = min(vx_obj, vx_cmd + ACELERACION * dt)   # sube con rampa, baja de inmediato
            vx_cmd = vx
            w = pid_rumbo.update(err_theta, dt)

            motion.moveToward(vx, 0.0, w, config)

            # Cabeza: compensa la inclinación del torso para mantener la mirada nivelada
            cab = 0.0
            ang_y_f = 0.7 * ang_y_f + 0.3 * ang_y   # filtro pasa-bajas
            if ESTABILIZAR_CABEZA:
                cab = limitar(-GANANCIA_CABEZA * (ang_y_f - ay0), -LIM_CABEZA, LIM_CABEZA)
                motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, cab], 0.3)

            angs_x.append(ang_x - ax0)
            angs_y.append(ang_y - ay0)
            escritor.writerow([f"{t - t_inicio:.3f}", f"{dt:.4f}", f"{x:.4f}", f"{y:.4f}",
                               f"{th:.4f}", f"{avance:.4f}", f"{lateral:.4f}",
                               f"{err_dist:.4f}", f"{err_theta:.4f}",
                               f"{vx:.3f}", f"{w:.3f}",
                               f"{ang_x - ax0:.4f}", f"{ang_y - ay0:.4f}", f"{cab:.4f}"])
            filas += 1
            if filas % 20 == 0:
                archivo_csv.flush()

            print(f"Dist: {avance:.2f}m | Lat: {lateral:+.3f}m | "
                  f"ErrRumbo: {err_theta:+.4f} | vx: {vx:.2f} | w: {w:+.3f} | "
                  f"AngY: {ang_y - ay0:+.3f}")
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
            motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
            motion.rest()
        except Exception:
            logging.exception("Error al pasar a reposo")

        archivo_csv.close()
        resultado = {
            "motivo": motivo,
            "perfil": PERFIL,
            "tiempo_s": round(time.time() - t_inicio, 2),
            "avance_final_estable": avance_final,
            "rms_ang_x": rms(angs_x),
            "rms_ang_y": rms(angs_y),
            "max_ang": max([abs(a) for a in angs_x + angs_y], default=None),
            "medido_cm": None,  # llénalo a mano con cinta métrica
        }
        with open(os.path.join(carpeta, "resultado.json"), "w") as f:
            json.dump(resultado, f, indent=2)
        logging.info("Fin de la corrida: %s", resultado)


if __name__ == "__main__":
    main()