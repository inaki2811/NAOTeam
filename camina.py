import collections
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
DISTANCIA_OBJETIVO = 5.0   # Metros
VELOCIDAD_MIN = 0.15       # Evita que el NAO deje de dar pasos antes de llegar
TOLERANCIA_M = 0.02        # Se considera meta alcanzada a +-2 cm
PERIODO_S = 0.05           # Periodo del bucle de control
TIEMPO_MAX_S = 120.0       # Seguridad: aborta si tarda demasiado
ACELERACION = 1.5          # Rampa de arranque (vel. relativa por segundo)

# --- PERFIL DE MARCHA ---
# "seguro": lo que ya funcionó (~0.06 m/s).
# "estable": intermedio, pasos más cortos y menos frecuentes.
# "rapido": cerca del límite del robot (~0.19 m/s con vx=1.0). El gobernador lo frena si se balancea.
PERFIL = "rapido"

PERFILES = {
    "seguro": {
        "velocidad_x": 0.5,
        "config": {
            "MaxStepX": 0.075, "MaxStepY": 0.15, "StepHeight": 0.03,
            "MaxStepTheta": 0.04, "MaxStepFrequency": 0.8,
        },
    },
    "estable": {
        "velocidad_x": 0.7,
        "config": {
            "MaxStepX": 0.06, "MaxStepY": 0.15, "StepHeight": 0.025,
            "MaxStepTheta": 0.08, "MaxStepFrequency": 0.85,
            "TorsoWx": 0.0, "TorsoWy": 0.0,
        },
    },
    "rapido": {
        "velocidad_x": 0.9,
        "config": {
            "MaxStepX": 0.079, "MaxStepY": 0.15, "StepHeight": 0.025,
            "MaxStepTheta": 0.08, "MaxStepFrequency": 1.0,
            # Tus logs muestran el torso ya inclinado hacia adelante (+0.02 a +0.04 rad de media)
            # y balanceos de cabeceo crecientes: se quita la inclinación extra.
            # Si quieres probar el efecto contrario, usa un valor pequeño negativo (p. ej. -0.02).
            "TorsoWx": 0.0, "TorsoWy": 0.0,
        },
    },
}

# --- CONSTANTES PID ---
# Rumbo (salida = velocidad angular relativa, theta de moveToward)
KP_RUMBO, KI_RUMBO, KD_RUMBO = 0.5, 0.0, 0.0
MAX_GIRO = 0.3
ZONA_MUERTA_RUMBO = 0.015  # rad (~0.9 deg): dentro de esto no se corrige, para no perturbar la marcha
# Distancia (salida = velocidad relativa hacia adelante)
KP_DIST, KI_DIST, KD_DIST = 2.5, 0.0, 0.0

# --- GOBERNADOR DE ESTABILIDAD (ajusta la velocidad según cuánto se balancea el torso) ---
# La velocidad máxima permitida (vx_lim) baja rápido si el torso se balancea mucho y sube
# despacio cuando está tranquilo, de modo que se asienta sola en la velocidad más alta estable.
GOBERNADOR = True
VX_LIM_INICIAL = 0.6       # empieza conservador y sube si el robot va estable
VX_LIM_MIN = 0.3           # nunca baja de aquí por el gobernador
VENTANA_S = 1.5            # ventana para medir la amplitud de balanceo
AMP_ALTA = 0.10            # rad: por encima, se reduce la velocidad
AMP_BAJA = 0.06            # rad: por debajo, se permite subir poco a poco
BAJADA = 0.5               # vel. relativa por segundo que se resta cuando se balancea mucho
SUBIDA = 0.08              # vel. relativa por segundo que se suma cuando va estable
ALERTA_RAD = 0.22          # rad: freno inmediato a VX_LIM_MIN
LIM_CAIDA = 0.35           # rad: se detiene la marcha (inclinación crítica)

# --- CABEZA ---
DESACTIVAR_VIDA_AUTONOMA = True   # evita que Autonomous Life/BasicAwareness mueva la cabeza
ESTABILIZAR_CABEZA = True         # la cabeza compensa la inclinación del torso
GANANCIA_CABEZA = 1.0
LIM_CABEZA = 0.2                  # rad, máxima compensación

# --- ESTADO DEL HARDWARE ---
BATERIA_MIN = 0.20         # por debajo, los motores pierden fuerza y la marcha se vuelve inestable
TEMP_AVISO = 65.0          # °C en articulaciones de piernas: si se calientan, el robot reduce rigidez
PERIODO_HW_S = 5.0
JUNTAS_PIERNA = [
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
]

CLAVES_SENSORES = [
    "Device/SubDeviceList/Head/Touch/Front/Sensor/Value",
    "Device/SubDeviceList/Head/Touch/Middle/Sensor/Value",
    "Device/SubDeviceList/Head/Touch/Rear/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value",
]
CLAVES_HW = (["Device/SubDeviceList/Battery/Charge/Sensor/Value"] +
             [f"Device/SubDeviceList/{j}/Temperature/Sensor/Value" for j in JUNTAS_PIERNA])


def wrap_pi(a):
    """Normaliza un ángulo a [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def fmt(valor, decimales=4):
    return "" if valor is None else f"{valor:.{decimales}f}"


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


def leer_estado_hw(memory):
    """Batería (0..1) y temperatura máxima de las articulaciones de las piernas."""
    try:
        vals = memory.getListData(CLAVES_HW)
        bateria = vals[0]
        temps = [v for v in vals[1:] if v is not None]
        return bateria, (max(temps) if temps else None)
    except Exception:
        logging.warning("No se pudo leer batería/temperaturas", exc_info=True)
        return None, None


def avisar_hw(bateria, temp_max):
    logging.info("Batería: %s | Temp. máx. piernas: %s °C", fmt(bateria, 2), fmt(temp_max, 1))
    if bateria is not None and bateria < BATERIA_MIN:
        logging.warning("Batería baja (%.0f%%): puede causar inestabilidad. Cárgalo.", bateria * 100)
    if temp_max is not None and temp_max >= TEMP_AVISO:
        logging.warning("Piernas calientes (%.0f °C): deja enfriar el robot entre pruebas.", temp_max)


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


def pedir_medicion():
    """Pregunta en la terminal la distancia medida con cinta (cm). Enter para omitir."""
    for _ in range(3):
        try:
            texto = input("\nDistancia medida con cinta en cm (Enter para omitir): ").strip().replace(",", ".")
        except (EOFError, KeyboardInterrupt):
            return None
        if not texto:
            return None
        try:
            valor = float(texto)
            if valor > 0:
                return valor
        except ValueError:
            pass
        print("Escribe solo un número positivo, por ejemplo 98.5")
    return None


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
        "max_giro": MAX_GIRO, "zona_muerta_rumbo": ZONA_MUERTA_RUMBO,
        "Kp_dist": KP_DIST, "Ki_dist": KI_DIST, "Kd_dist": KD_DIST,
        "gobernador": GOBERNADOR,
        "vx_lim_inicial": VX_LIM_INICIAL, "vx_lim_min": VX_LIM_MIN,
        "amp_alta": AMP_ALTA, "amp_baja": AMP_BAJA, "bajada": BAJADA, "subida": SUBIDA,
        "alerta_rad": ALERTA_RAD, "lim_caida": LIM_CAIDA,
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
    vx_lim = min(VX_LIM_INICIAL, velocidad_x)
    vx_lim_min_vista = vx_lim
    bateria, temp_max = None, None
    bateria_ini = temp_ini = None
    t_inicio = time.time()

    archivo_csv = open(os.path.join(carpeta, "telemetria.csv"), "w", newline="")
    escritor = csv.writer(archivo_csv)
    escritor.writerow(["t", "dt", "x", "y", "theta", "avance", "lateral",
                       "err_dist", "err_theta", "vx", "w", "ang_x", "ang_y", "cab_pitch",
                       "vx_lim", "amp", "bateria", "temp_max"])

    try:
        # 1. PREPARACIÓN
        bateria, temp_max = leer_estado_hw(memory)
        bateria_ini, temp_ini = bateria, temp_max
        avisar_hw(bateria, temp_max)

        if DESACTIVAR_VIDA_AUTONOMA:
            desactivar_vida_autonoma(app)
        motion.wakeUp()
        pose.goToPosture("StandInit", 0.5)   # posturas válidas: Stand, StandInit, StandZero, Crouch, Sit...
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
        t_hw = t_prev
        vx_cmd = 0.0
        ventana = collections.deque()   # (t, amplitud) de los últimos VENTANA_S segundos
        filas = 0

        # 2. BUCLE PRINCIPAL (PID + gobernador de estabilidad + monitor táctil)
        while True:
            tocado, ang_x, ang_y = leer_sensores(memory)
            if tocado:
                logging.info("[INTERRUPCIÓN] Se tocó la cabeza. Cancelando caminata.")
                motivo = "tacto"
                break
            rel_x, rel_y = ang_x - ax0, ang_y - ay0
            if abs(rel_x) > LIM_CAIDA or abs(rel_y) > LIM_CAIDA:
                logging.error("Inclinación crítica (x=%.2f, y=%.2f rad). Deteniendo.", rel_x, rel_y)
                motivo = "inclinacion"
                break

            t = time.time()
            if t - t_inicio > TIEMPO_MAX_S:
                logging.error("Tiempo máximo excedido (%.0f s). Abortando.", TIEMPO_MAX_S)
                motivo = "timeout"
                break
            dt = max(t - t_prev, 1e-3)
            t_prev = t

            # Estado del hardware, cada PERIODO_HW_S
            if t - t_hw >= PERIODO_HW_S:
                t_hw = t
                bateria, temp_max = leer_estado_hw(memory)
                avisar_hw(bateria, temp_max)

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

            # --- Gobernador: amplitud de balanceo del torso en la última ventana ---
            # (el balanceo lateral normal de cada paso pesa menos que el cabeceo)
            ventana.append((t, max(abs(rel_y), 0.75 * abs(rel_x))))
            while ventana and t - ventana[0][0] > VENTANA_S:
                ventana.popleft()
            amp = max(a for _, a in ventana)
            if GOBERNADOR:
                if amp >= ALERTA_RAD:
                    if vx_lim > VX_LIM_MIN:
                        logging.warning("Balanceo alto (%.2f rad): freno inmediato", amp)
                    vx_lim = VX_LIM_MIN
                elif amp > AMP_ALTA:
                    vx_lim -= BAJADA * dt
                elif amp < AMP_BAJA:
                    vx_lim += SUBIDA * dt
                vx_lim = limitar(vx_lim, VX_LIM_MIN, velocidad_x)
                vx_lim_min_vista = min(vx_lim_min_vista, vx_lim)
            else:
                vx_lim = velocidad_x

            # Velocidad hacia adelante: PID de distancia, tope del gobernador, rampa de arranque
            vx_obj = min(max(pid_dist.update(err_dist, dt), VELOCIDAD_MIN), vx_lim)
            vx = min(vx_obj, vx_cmd + ACELERACION * dt)   # sube con rampa, baja de inmediato
            vx_cmd = vx

            # Giro: con zona muerta para no perturbar la marcha por errores diminutos
            w = 0.0 if abs(err_theta) < ZONA_MUERTA_RUMBO else pid_rumbo.update(err_theta, dt)
            if w == 0.0:
                pid_rumbo.integral = 0.0

            motion.moveToward(vx, 0.0, w, config)

            # Cabeza: compensa la inclinación del torso para mantener la mirada nivelada
            cab = 0.0
            ang_y_f = 0.7 * ang_y_f + 0.3 * ang_y   # filtro pasa-bajas
            if ESTABILIZAR_CABEZA:
                cab = limitar(-GANANCIA_CABEZA * (ang_y_f - ay0), -LIM_CABEZA, LIM_CABEZA)
                motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, cab], 0.3)

            angs_x.append(rel_x)
            angs_y.append(rel_y)
            escritor.writerow([f"{t - t_inicio:.3f}", f"{dt:.4f}", f"{x:.4f}", f"{y:.4f}",
                               f"{th:.4f}", f"{avance:.4f}", f"{lateral:.4f}",
                               f"{err_dist:.4f}", f"{err_theta:.4f}",
                               f"{vx:.3f}", f"{w:.3f}",
                               f"{rel_x:.4f}", f"{rel_y:.4f}", f"{cab:.4f}",
                               f"{vx_lim:.3f}", f"{amp:.4f}", fmt(bateria, 3), fmt(temp_max, 1)])
            filas += 1
            if filas % 20 == 0:
                archivo_csv.flush()

            print(f"Dist: {avance:.2f}m | Lat: {lateral:+.3f}m | vx: {vx:.2f} (lim {vx_lim:.2f}) | "
                  f"w: {w:+.3f} | AngY: {rel_y:+.3f} | Amp: {amp:.3f}")
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
        medido_cm = pedir_medicion()   # el robot ya está en reposo: mide con cinta y escribe el valor
        resultado = {
            "motivo": motivo,
            "perfil": PERFIL,
            "tiempo_s": round(time.time() - t_inicio, 2),
            "avance_final_estable": avance_final,
            "rms_ang_x": rms(angs_x),
            "rms_ang_y": rms(angs_y),
            "max_ang": max([abs(a) for a in angs_x + angs_y], default=None),
            "vx_lim_final": vx_lim,
            "vx_lim_minimo": vx_lim_min_vista,
            "bateria_ini": bateria_ini,
            "temp_max_ini": temp_ini,
            "medido_cm": medido_cm,
        }
        with open(os.path.join(carpeta, "resultado.json"), "w") as f:
            json.dump(resultado, f, indent=2)
        logging.info("Fin de la corrida: %s", resultado)


if __name__ == "__main__":
    main()