# -*- coding: utf-8 -*-
"""
maze_nao.py (v3) -- Laberinto Copa NAO CCM 2026. Corre DENTRO del NAO (Python 2.7, NAOqi 2.x).

Reglas vigentes (aviso de los organizadores):
  - 5x5 celdas de 48.8 cm, piso de madera, META EN LA CELDA CENTRAL (piso rojo).
  - Salida en una esquina, con paredes en 3 lados.
  - Los ids de los AprilTags pueden estar REVUELTOS: el id NO dice la posicion.
    El robot aprende en el intento 1 que id tiene cada celda y lo usa en el intento 2.

Como se orienta el robot sin ids posicionales
  Define su propio sistema de coordenadas: la salida es la celda (4, 0) y el robot mira al Este.
  El laberinto queda hacia adelante (columnas 0..4) y hacia UN lado: o a la izquierda (filas 0..4)
  o a la derecha (filas 4..8). Al principio no sabe cual; lo resuelve en cuanto ve una pared
  abierta hacia un lado de la fila de salida, o el piso rojo. La meta es el centro de ese 5x5:
  (2, 2) si el laberinto quedo a la izquierda, (6, 2) si quedo a la derecha.

USO EN COMPETENCIA (un solo comando para los dos intentos):
  cd ~/vision
  nohup python maze_nao.py > maze.out 2>&1 &
    1. Arranca la vision (nao_apriltag.py) si no esta corriendo.
    2. Se para y espera: TOCAR LA CABEZA -> intento 1 (explora hacia el centro, aprende los ids).
    3. Al terminar se agacha. Llevarlo a la salida, en la MISMA orientacion, y TOCAR LA CABEZA -> intento 2.
    Tocar la cabeza DURANTE un intento = PARO DE EMERGENCIA (se detiene y se sienta).

Otros modos:
  python maze_nao.py --run 1 | --run 2        # un solo intento (el 2 carga mapa.json)
  python maze_nao.py --check                  # robot quieto: que paredes/tags/rojo ve (no camina)
  python maze_nao.py --sim --seed 3           # simulacion de los dos intentos (Mac/PC, Python 3)
  python maze_nao.py --sim --batch 300        # estadistica

Convenciones
  Celdas (fila, col). Direcciones: N=0 (fila-1), E=1, S=2, W=3.
  Relativas: 0 frente, 1 derecha, 2 atras, 3 izquierda  ->  absoluta = (rumbo + rel) % 4.
  Paredes: UNKNOWN < WEAK (no se vio tag) < WALL < OPEN (cruzada o vista a traves).
"""
from __future__ import print_function, division

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
SIZE = 5                    # laberinto SIZE x SIZE
CELL = 0.488                # m (aviso de los organizadores)
WALL_DIST = CELL / 2 - 0.0075   # m, centro de la celda -> superficie de la pared (pared 1-2 cm)
N, E, S, W = 0, 1, 2, 3
DNAME = "NESW"
DR = [-1, 0, 1, 0]
DC = [0, 1, 0, -1]
UNKNOWN, WEAK, WALL, OPEN = 0, 1, 2, 3

# Marco propio del robot: salida en (S0, 0) mirando al Este
S0 = SIZE - 1
START = (S0, 0)
START_HEADING = E
# Filas validas: [RMIN, RMAX]. Antes de saber de que lado queda el laberinto: las dos opciones.
RMIN, RMAX = 0, 2 * S0

# Tags (FRAME_ROBOT: origen entre los tobillos, x adelante, y izquierda)
NEAR_MIN, NEAR_MAX = 0.10, 0.42
FAR_MIN, FAR_MAX = 0.53, 0.98
NEAR_TOL, FAR_TOL = 35.0, 20.0
REL_BEARING = {0: 0.0, 3: 90.0, 1: -90.0}

ALIGN_MIN_XY, ALIGN_MAX_XY = 0.03, 0.12
ALIGN_MIN_TH, ALIGN_MAX_TH = 4.0, 20.0

GOAL_MIN_POINTS = 25
RED_MAX_RANGE = 1.40


def configure_size(size):
    global SIZE, S0, START, RMIN, RMAX
    SIZE = size
    S0 = SIZE - 1
    START = (S0, 0)
    RMIN, RMAX = 0, 2 * S0


def set_rows(rmin, rmax):
    global RMIN, RMAX
    RMIN, RMAX = rmin, rmax


def goal_candidates():
    """Centro del laberinto en el marco propio, segun el lado en que quede."""
    c = SIZE // 2
    if RMIN == 0 and RMAX == 2 * S0:
        return [(S0 - c, c), (S0 + c, c)]
    return [(RMIN + c, c)]


def nb(rc, d):
    return (rc[0] + DR[d], rc[1] + DC[d])


def inside(rc):
    return RMIN <= rc[0] <= RMAX and 0 <= rc[1] < SIZE


def wrap180(a):
    while a > 180.0:
        a -= 360.0
    while a <= -180.0:
        a += 360.0
    return a


def median(v):
    v = sorted(v)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def dir_between(a, b):
    for d in range(4):
        if nb(a, d) == b:
            return d
    return None


class EmergencyStop(Exception):
    pass


# --------------------------------------------------------------------------
# Mapa (en el marco propio del robot)
# --------------------------------------------------------------------------
class MazeMap(object):
    def __init__(self):
        self.w = {}
        self.visited = set()
        self.start = START
        self.start_heading = START_HEADING
        self.goal = None
        self.side = None            # None, "left" (filas 0..S0) o "right" (filas S0..2*S0)
        self.ids = {}               # celda -> {id: votos}
        self._borders()

    def _borders(self):
        for r in range(RMIN, RMAX + 1):
            for c in range(SIZE):
                for d in range(4):
                    if not inside(nb((r, c), d)):
                        self.w[(r, c, d)] = WALL

    def get(self, rc, d):
        return self.w.get((rc[0], rc[1], d), UNKNOWN)

    def set(self, rc, d, st, force=False):
        if not inside(rc):
            return False
        if st <= self.get(rc, d) and not force:
            return False
        self.w[(rc[0], rc[1], d)] = st
        o = nb(rc, d)
        if inside(o):
            self.w[(o[0], o[1], (d + 2) % 4)] = st
            # Una pared ABIERTA que sale de la fila de salida revela de que lado esta el laberinto
            if st == OPEN and self.side is None and rc[0] != o[0]:
                if min(rc[0], o[0]) == S0 - 1:
                    self.resolve("left")
                elif max(rc[0], o[0]) == S0 + 1:
                    self.resolve("right")
        return True

    def resolve(self, side):
        if self.side is not None:
            return
        self.side = side
        if side == "left":
            set_rows(0, S0)
        else:
            set_rows(S0, 2 * S0)
        self._borders()
        self.goal = goal_candidates()[0]

    def passable(self, rc, d, allowed):
        return inside(nb(rc, d)) and self.get(rc, d) in allowed

    def unknown_edges(self, rc):
        return [d for d in range(4) if self.get(rc, d) == UNKNOWN and inside(nb(rc, d))]

    # ---- ids aprendidos ----
    def learn_id(self, rc, tag_id, w=1):
        if not inside(rc):
            return
        cnt = self.ids.setdefault(rc, {})
        cnt[tag_id] = cnt.get(tag_id, 0) + w

    def id_of(self, rc):
        cnt = self.ids.get(rc)
        if not cnt:
            return None
        return max(cnt, key=cnt.get)

    def cell_with_id(self, tag_id, min_votes=2):
        """Celda cuyo id dominante es tag_id (con suficiente evidencia y sin ambiguedad)."""
        hits = [c for c, cnt in self.ids.items()
                if inside(c) and max(cnt, key=cnt.get) == tag_id and cnt[tag_id] >= min_votes]
        return hits[0] if len(hits) == 1 else None

    # ---- persistencia ----
    def save(self, path):
        data = {"size": SIZE, "rows": [RMIN, RMAX], "side": self.side,
                "walls": [[k[0], k[1], k[2], v] for k, v in self.w.items()],
                "visited": sorted(list(self.visited)),
                "ids": [[c[0], c[1], t, v] for c, cnt in self.ids.items() for t, v in cnt.items()],
                "start": self.start, "start_heading": self.start_heading, "goal": self.goal}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.rename(tmp, path)

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
        configure_size(data.get("size", SIZE))
        set_rows(*data["rows"])
        m = MazeMap()
        m.side = data["side"]
        for r, c, d, v in data["walls"]:
            m.w[(r, c, d)] = v
        m.visited = set(tuple(x) for x in data["visited"])
        for r, c, t, v in data.get("ids", []):
            m.ids.setdefault((r, c), {})[t] = v
        m.start = tuple(data["start"])
        m.start_heading = data["start_heading"]
        m.goal = tuple(data["goal"]) if data["goal"] else None
        return m

    def render(self, pos=None, heading=None, path=None):
        path = set(path or [])
        arrow = {N: "^", E: ">", S: "v", W: "<"}
        hz = {WALL: "---", OPEN: "   ", WEAK: " ~ ", UNKNOWN: " ? "}
        vt = {WALL: "|", OPEN: " ", WEAK: "~", UNKNOWN: ":"}
        out = []
        for r in range(RMIN, RMAX + 1):
            top, mid = "+", ""
            for c in range(SIZE):
                top += hz[self.get((r, c), N)] + "+"
                mid += vt[self.get((r, c), W)]
                if (r, c) == pos and heading is not None:
                    ch = " " + arrow[heading] + " "
                elif (r, c) == self.goal or (self.goal is None and (r, c) in goal_candidates()):
                    ch = " M " if self.goal else " m "
                elif (r, c) == self.start:
                    ch = " S "
                elif (r, c) in path:
                    ch = " * "
                elif (r, c) in self.visited:
                    ch = " . "
                else:
                    ch = "   "
                mid += ch
            mid += vt[self.get((r, SIZE - 1), E)]
            out += [top, mid]
        out.append("+" + "---+" * SIZE)
        return "\n".join(out)


# --------------------------------------------------------------------------
# A* -- del codigo del equipo (laberinto.txt), adaptado a paredes ENTRE celdas
# --------------------------------------------------------------------------
class Node(object):
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0

    def __eq__(self, other):
        return self.position == other.position


def a_star_search(mapa, inicio, meta, permitidos=(OPEN,), costo_giro=0.5, costo_desconocido=0.0):
    """A* del equipo: vecino valido = no hay pared entre celdas; girar cuesta extra.
    Con UNKNOWN en 'permitidos' es optimista (supone abierto lo que no ha visto): asi se explora
    hacia la meta conocida, como en micromouse."""
    lista_abierta = [Node(None, inicio)]
    lista_cerrada = []
    nodo_final = Node(None, meta)
    while len(lista_abierta) > 0:
        nodo_actual = lista_abierta[0]
        indice_actual = 0
        for index, item in enumerate(lista_abierta):
            if item.f < nodo_actual.f:
                nodo_actual = item
                indice_actual = index
        lista_abierta.pop(indice_actual)
        lista_cerrada.append(nodo_actual)

        if nodo_actual == nodo_final:
            camino = []
            actual = nodo_actual
            while actual is not None:
                camino.append(actual.position)
                actual = actual.parent
            return camino[::-1]

        for d in range(4):
            if not mapa.passable(nodo_actual.position, d, permitidos):
                continue
            vecino = Node(nodo_actual, nb(nodo_actual.position, d))
            if vecino in lista_cerrada:
                continue
            extra = 0.0
            if nodo_actual.parent is not None:
                if dir_between(nodo_actual.parent.position, nodo_actual.position) != d:
                    extra += costo_giro
            if mapa.get(nodo_actual.position, d) != OPEN:
                extra += costo_desconocido
            vecino.g = nodo_actual.g + 1.0 + extra
            vecino.h = abs(vecino.position[0] - meta[0]) + abs(vecino.position[1] - meta[1])
            vecino.f = vecino.g + vecino.h
            en_lista_abierta = False
            for nodo_abierto in lista_abierta:
                if vecino == nodo_abierto:
                    en_lista_abierta = True
                    if vecino.g < nodo_abierto.g:
                        nodo_abierto.g = vecino.g
                        nodo_abierto.f = vecino.f
                        nodo_abierto.parent = nodo_actual
                    break
            if not en_lista_abierta:
                lista_abierta.append(vecino)
    return None


def path_cost(path, mapa=None):
    if not path:
        return float("inf")
    turns = sum(1 for s in zip(path, path[1:], path[2:])
                if dir_between(s[0], s[1]) != dir_between(s[1], s[2]))
    return len(path) - 1 + 0.5 * turns


def segments(path):
    segs = []
    for a, b in zip(path, path[1:]):
        d = dir_between(a, b)
        if segs and segs[-1][0] == d:
            segs[-1][1] += 1
        else:
            segs.append([d, 1])
    return [tuple(s) for s in segs]


# --------------------------------------------------------------------------
# Percepcion
# --------------------------------------------------------------------------
def classify(tags):
    out = dict((rel, {"near": [], "far": []}) for rel in REL_BEARING)
    for t in tags:
        x, y = t[1], t[2]
        r = math.hypot(x, y)
        b = math.degrees(math.atan2(y, x))
        for rel, center in REL_BEARING.items():
            db = abs(wrap180(b - center))
            if NEAR_MIN <= r <= NEAR_MAX and db <= NEAR_TOL:
                out[rel]["near"].append(t)
            elif FAR_MIN <= r <= FAR_MAX and db <= FAR_TOL:
                out[rel]["far"].append(t)
    return out


def alignment(cls, tags):
    yaw_err = [wrap180(t[3] - 90.0 * round(t[3] / 90.0)) for t in tags
               if len(t) > 3 and t[3] is not None]
    dth = median(yaw_err) if yaw_err else None

    def axis(near, far, coord, sign):
        if near:
            return median([t[coord] for t in near]) - sign * WALL_DIST
        if far:
            return median([t[coord] for t in far]) - sign * (WALL_DIST + CELL)
        return None

    dx = axis(cls[0]["near"], cls[0]["far"], 1, +1)
    ys = [v for v in (axis(cls[3]["near"], cls[3]["far"], 2, +1),
                      axis(cls[1]["near"], cls[1]["far"], 2, -1)) if v is not None]
    return dx, (sum(ys) / len(ys) if ys else None), dth


def cluster_tags(tags, tol=0.06):
    groups = []
    for t in tags:
        for g in groups:
            if g[0][0] == t[0] and math.hypot(g[0][1] - t[1], g[0][2] - t[2]) < tol:
                g.append(t)
                break
        else:
            groups.append([t])
    out = []
    for g in groups:
        yaws = [t[3] for t in g if t[3] is not None]
        out.append((g[0][0], median([t[1] for t in g]), median([t[2] for t in g]),
                    median(yaws) if yaws else None))
    return out


# --------------------------------------------------------------------------
# PID de caminata recta -- portado de caminata_v2.py del equipo (Python 3 -> 2.7)
# --------------------------------------------------------------------------
VELOCIDAD_X = 0.5
VELOCIDAD_MIN = 0.15
TOLERANCIA_M = 0.02
PERIODO_S = 0.05
KP_RUMBO, KI_RUMBO, KD_RUMBO = 0.0, 0.0, 0.05
MAX_GIRO = 0.2
KP_DIST, KI_DIST, KD_DIST = 1.0, 0.0, 0.0
CONFIG_DESEADA = {"MaxStepX": 0.08, "MaxStepY": 0.03, "StepHeight": 0.8,
                  "MaxStepTheta": 0.04, "MaxStepFrequency": 0.8}


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class PID(object):
    def __init__(self, kp, ki, kd, out_min, out_max, i_max=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_max = i_max
        self.integral = 0.0
        self.prev = None

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.i_max, min(self.i_max, self.integral))
        deriv = 0.0 if self.prev is None else (error - self.prev) / dt
        self.prev = error
        salida = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(self.out_min, min(self.out_max, salida))


def ajustar_config(motion, deseada, log):
    try:
        minimos = dict(motion.getMoveConfig("Min"))
        maximos = dict(motion.getMoveConfig("Max"))
    except Exception:
        log("[!] no pude leer limites de marcha; uso la config tal cual")
        return [[k, v] for k, v in deseada.items()]
    config = []
    for nombre, valor in deseada.items():
        lo, hi = minimos.get(nombre, valor), maximos.get(nombre, valor)
        ajustado = max(lo, min(hi, valor))
        if ajustado != valor:
            log("[cfg] %s=%s fuera de rango [%s, %s]; uso %s" % (nombre, valor, lo, hi, ajustado))
        config.append([nombre, ajustado])
    return config


# --------------------------------------------------------------------------
# Robot real (NAOqi)
# --------------------------------------------------------------------------
TOUCH_KEYS = ["Device/SubDeviceList/Head/Touch/Front/Sensor/Value",
              "Device/SubDeviceList/Head/Touch/Middle/Sensor/Value",
              "Device/SubDeviceList/Head/Touch/Rear/Sensor/Value"]


class NaoRobot(object):
    NEAR_PITCH = -7.5
    LOOK_YAW = {0: 0.0, 3: 90.0, 1: -90.0}
    FOV_V5 = (60.97, 47.64)

    def __init__(self, ip, port, log, test_goal=None):
        from naoqi import ALProxy
        import numpy
        self.np = numpy
        self.log = log
        self.motion = ALProxy("ALMotion", ip, port)
        self.posture = ALProxy("ALRobotPosture", ip, port)
        self.memory = ALProxy("ALMemory", ip, port)
        self.video = ALProxy("ALVideoDevice", ip, port)
        try:
            self.sonar = ALProxy("ALSonar", ip, port)
            self.sonar.subscribe("maze_nao")
        except Exception:
            self.sonar = None
        self.color = None
        try:
            for s in self.video.getSubscribers():
                if s.startswith("maze_red"):
                    self.video.unsubscribe(s)
            self.color = self.video.subscribeCamera("maze_red", 1, 0, 13, 5)   # abajo, 160x120, BGR
        except Exception as e:
            log("[!] sin camara de color para la meta: %s" % e)
        self.config = ajustar_config(self.motion, CONFIG_DESEADA, log)
        self.test_goal = test_goal
        self.armed = False
        self.t0 = time.time()
        self.last_red = []
        self.vision_proc = None

    # ---- arranque / paro ----
    def head_touched(self):
        try:
            return any(self.memory.getData(k) > 0.5 for k in TOUCH_KEYS)
        except RuntimeError:
            return False

    def check(self):
        if self.armed and self.head_touched():
            raise EmergencyStop()

    def wait_start(self, msg):
        self.armed = False
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)
        self.motion.setStiffnesses("Head", 1.0)
        self.head(0.0, self.NEAR_PITCH)
        self.log("[..] %s" % msg)
        while not self.head_touched():
            time.sleep(0.05)
        while self.head_touched():
            time.sleep(0.05)
        time.sleep(0.3)
        self.armed = True
        self.t0 = time.time()
        self.log("[ok] arranque")

    def emergency(self):
        self.armed = False
        try:
            self.motion.stopMove()
            self.posture.goToPosture("Sit", 0.5)
            self.motion.rest()
        except Exception:
            pass

    def crouch(self):
        self.armed = False
        try:
            self.motion.stopMove()
            self.head(0.0, 0.0)
            self.motion.rest()
        except Exception:
            pass

    def close(self):
        for fn in (lambda: self.video.unsubscribe(self.color) if self.color else None,
                   lambda: self.sonar.unsubscribe("maze_nao") if self.sonar else None,
                   lambda: self.vision_proc.terminate() if self.vision_proc else None):
            try:
                fn()
            except Exception:
                pass

    def elapsed(self):
        return time.time() - self.t0

    # ---- vision ----
    def vision_alive(self):
        try:
            return time.time() - self.memory.getData("Vision/AprilTags/stamp") < 1.0
        except RuntimeError:
            return False

    def ensure_vision(self):
        if self.vision_alive():
            self.log("[ok] vision ya corriendo")
            return
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "nao_apriltag.py")
        self.log("[..] arrancando vision: %s" % script)
        logf = open(os.path.join(here, "vision.log"), "a")
        self.vision_proc = subprocess.Popen(
            [sys.executable, script, "--camera", "both", "--publish", "--quiet"],
            stdout=logf, stderr=subprocess.STDOUT, cwd=here)
        t_end = time.time() + 25
        while time.time() < t_end:
            if self.vision_alive():
                self.log("[ok] vision publicando")
                return
            if self.vision_proc.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError("La vision no arranco; revisa vision.log")

    def head(self, yaw, pitch):
        self.motion.angleInterpolationWithSpeed(["HeadYaw", "HeadPitch"],
                                                [math.radians(yaw), math.radians(pitch)], 0.4)

    def _read_tags(self, window):
        got, t_from, last = [], time.time(), None
        t_end = t_from + window
        while time.time() < t_end:
            self.check()
            try:
                stamp = self.memory.getData("Vision/AprilTags/stamp")
                if stamp != last and stamp > t_from:
                    last = stamp
                    for r in self.memory.getData("Vision/AprilTags"):
                        got.append((int(r[0]), r[1], r[2], r[7] if len(r) > 7 else None))
            except RuntimeError:
                pass
            time.sleep(0.05)
        return cluster_tags(got)

    def _ground(self, step=2, only_red=True):
        """Proyecta pixeles de la camara de abajo al piso. Devuelve (puntos, es_rojo) en FRAME_ROBOT."""
        np = self.np
        if self.color is None:
            return [], []
        img = self.video.getImageRemote(self.color)
        if img is None:
            return [], []
        w, h = img[0], img[1]
        T = np.array(self.motion.getTransform("CameraBottom", 2, True)).reshape(4, 4)
        a = np.frombuffer(img[6], dtype=np.uint8).reshape(h, w, 3)[::step, ::step].astype(np.int32)
        b, g, r = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        red = (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)
        if only_red:
            vs, us = np.nonzero(red)
            is_red = np.ones(len(us), dtype=bool)
        else:
            vs, us = np.nonzero(np.ones(red.shape, dtype=bool))
            is_red = red[vs, us]
        if len(us) == 0:
            return [], []
        us = us * step
        vs = vs * step
        fx = (w / 2.0) / math.tan(math.radians(self.FOV_V5[0] / 2))
        fy = (h / 2.0) / math.tan(math.radians(self.FOV_V5[1] / 2))
        xn = (us - w / 2.0) / fx
        yn = (vs - h / 2.0) / fy
        d = T[:3, :3].dot(np.vstack([np.ones(len(xn)), -xn, -yn]))
        t = T[:3, 3]
        ok = d[2] < -1e-3
        s = -t[2] / d[2][ok]
        px = t[0] + s * d[0][ok]
        py = t[1] + s * d[1][ok]
        keep = np.hypot(px, py) < RED_MAX_RANGE
        return (list(zip(px[keep].tolist(), py[keep].tolist())), is_red[ok][keep].tolist())

    def red_floor_points(self):
        return self._ground(step=2, only_red=True)[0]

    def red_below(self):
        self.head(0.0, 25.0)
        time.sleep(0.4)
        try:
            pts, red = self._ground(step=4, only_red=False)
        finally:
            self.head(0.0, self.NEAR_PITCH)
        zone = [rd for (x, y), rd in zip(pts, red) if 0.08 < x < 0.22 and abs(y) < 0.18]
        frac = sum(zone) / float(len(zone)) if zone else 0.0
        self.log("[meta?] rojo bajo los pies: %.0f%% de %d puntos" % (100 * frac, len(zone)))
        return len(zone) >= 20 and frac > 0.4

    def look(self, rels):
        tags, self.last_red = [], []
        for rel in rels:
            self.check()
            self.head(self.LOOK_YAW[rel], self.NEAR_PITCH)
            time.sleep(0.1)
            tags += self._read_tags(0.5)
            try:
                self.last_red += self.red_floor_points()
            except Exception as e:
                self.log("[!] rojo: %s" % e)
        self.head(0.0, self.NEAR_PITCH)
        return cluster_tags(tags)

    def front_range(self):
        if self.sonar is None:
            return None
        try:
            return min(self.memory.getData("Device/SubDeviceList/US/Left/Sensor/Value"),
                       self.memory.getData("Device/SubDeviceList/US/Right/Sensor/Value"))
        except RuntimeError:
            return None

    # ---- locomocion ----
    def move(self, dx, dy, dth_deg):
        """UNICO punto de locomocion. Recto -> PID del equipo; giros/correcciones -> moveTo."""
        if dx > 0.2 and abs(dy) < 1e-6 and abs(dth_deg) < 1e-6:
            self.straight_pid(dx)
            return
        self.motion.post.moveTo(dx, dy, math.radians(dth_deg), self.config)
        time.sleep(0.3)
        try:
            while self.motion.moveIsActive():
                self.check()
                time.sleep(0.05)
        except EmergencyStop:
            self.motion.stopMove()
            raise

    def straight_pid(self, distancia):
        pid_dist = PID(KP_DIST, KI_DIST, KD_DIST, 0.0, VELOCIDAD_X)
        pid_rumbo = PID(KP_RUMBO, KI_RUMBO, KD_RUMBO, -MAX_GIRO, MAX_GIRO)
        self.motion.moveInit()
        x0, y0, th0 = self.motion.getRobotPosition(True)
        c, s = math.cos(th0), math.sin(th0)
        t_ini = t_prev = time.time()
        t_max = 8.0 + distancia / 0.04
        try:
            while True:
                self.check()
                t = time.time()
                if t - t_ini > t_max:
                    self.log("[!] PID: tiempo maximo del tramo")
                    break
                dt = max(t - t_prev, 1e-3)
                t_prev = t
                x, y, th = self.motion.getRobotPosition(True)
                err_dist = distancia - ((x - x0) * c + (y - y0) * s)
                if err_dist <= TOLERANCIA_M:
                    break
                vx = max(pid_dist.update(err_dist, dt), VELOCIDAD_MIN)
                w = pid_rumbo.update(wrap_pi(th0 - th), dt)
                self.motion.moveToward(vx, 0.0, w, self.config)
                time.sleep(PERIODO_S)
        finally:
            self.motion.stopMove()
        time.sleep(0.6)

    def is_goal(self, pos):
        return self.test_goal is not None and pos == self.test_goal


# --------------------------------------------------------------------------
# Simulador: mundo REAL 5x5 con salida en una esquina cualquiera e ids revueltos.
# No comparte coordenadas con el agente (como el robot real).
# --------------------------------------------------------------------------
def w_inside(rc):
    return 0 <= rc[0] < SIZE and 0 <= rc[1] < SIZE


def w_corners():
    return [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)]


class SimRobot(object):
    def __init__(self, walls, start, start_heading, goal, ids, rng, p_miss=0.05, p_red=0.9,
                 noise_xy=0.01, noise_yaw=2.0, step_err=0.06, drift_deg=3.0, turn_err=5.0):
        self.walls, self.goal, self.ids, self.rng = walls, goal, ids, rng
        self.p_miss, self.p_red = p_miss, p_red
        self.noise_xy, self.noise_yaw = noise_xy, noise_yaw
        self.step_err, self.drift_deg, self.turn_err = step_err, drift_deg, turn_err
        self.X = start[1] * CELL + CELL / 2
        self.Y = -(start[0] * CELL + CELL / 2)
        self.th = [90.0, 0.0, -90.0, 180.0][start_heading] + rng.gauss(0, 3.0)
        self.t = 0.0
        self.hits = 0
        self.touches = 0
        self.last_red = []

    def elapsed(self):
        return self.t

    def cell(self):
        return (int(math.floor(-self.Y / CELL)), int(math.floor(self.X / CELL)))

    def is_wall(self, rc, d):
        return (rc[0], rc[1], d) in self.walls

    def _to_robot(self, wx, wy):
        dx, dy = wx - self.X, wy - self.Y
        a = math.radians(self.th)
        return (math.cos(a) * dx + math.sin(a) * dy, -math.sin(a) * dx + math.cos(a) * dy)

    def _tag(self, rc, d):
        cx, cy = rc[1] * CELL + CELL / 2, -(rc[0] * CELL + CELL / 2)
        ux, uy = [(0, 1), (1, 0), (0, -1), (-1, 0)][d]
        x, y = self._to_robot(cx + ux * WALL_DIST, cy + uy * WALL_DIST)
        yaw = wrap180([90.0, 0.0, -90.0, 180.0][d] - self.th)
        return (self.ids[rc], x + self.rng.gauss(0, self.noise_xy),
                y + self.rng.gauss(0, self.noise_xy), yaw + self.rng.gauss(0, self.noise_yaw))

    def _abs_dir(self, rel):
        th = wrap180(self.th + REL_BEARING[rel])
        return {90: N, 0: E, -90: S, 180: W, -180: W}[int(90 * round(th / 90.0))]

    def look(self, rels):
        self.t += 1.0 * len(rels)
        rc = self.cell()
        out, self.last_red = [], []
        for rel in rels:
            look_b = REL_BEARING[rel]
            for d in range(4):
                for cell_rc, through in ((rc, None), (nb(rc, d), d)):
                    if not w_inside(cell_rc) or not self.is_wall(cell_rc, d):
                        continue
                    if through is not None and self.is_wall(rc, d):
                        continue
                    t = self._tag(cell_rc, d)
                    if abs(wrap180(math.degrees(math.atan2(t[2], t[1])) - look_b)) > 28.0:
                        continue
                    if self.rng.random() < self.p_miss:
                        continue
                    out.append(t)
            d = self._abs_dir(rel)
            c = rc
            for _ in range(2):
                if self.is_wall(c, d) or not w_inside(nb(c, d)):
                    break
                c = nb(c, d)
                if c == self.goal and self.rng.random() < self.p_red:
                    cx, cy = c[1] * CELL + CELL / 2, -(c[0] * CELL + CELL / 2)
                    for _k in range(60):
                        px, py = self._to_robot(cx + self.rng.uniform(-0.22, 0.22),
                                                cy + self.rng.uniform(-0.22, 0.22))
                        if abs(wrap180(math.degrees(math.atan2(py, px)) - look_b)) < 28.0:
                            self.last_red.append((px, py))
        return cluster_tags(out)

    def front_range(self):
        a = math.radians(self.th)
        x, y = self.X, self.Y
        for k in range(1, 260):
            nx, ny = x + 0.01 * math.cos(a), y + 0.01 * math.sin(a)
            if self._crosses_wall(x, y, nx, ny):
                return max(0.20, 0.01 * k - 0.05 + self.rng.gauss(0, 0.02))
            x, y = nx, ny
        return 2.5

    def _crosses_wall(self, x0, y0, x1, y1):
        c0 = (int(math.floor(-y0 / CELL)), int(math.floor(x0 / CELL)))
        c1 = (int(math.floor(-y1 / CELL)), int(math.floor(x1 / CELL)))
        if c0 == c1:
            return False
        d = dir_between(c0, c1)
        if d is None:
            return True
        return self.is_wall(c0, d) or not w_inside(c1)

    def _touching(self, half_width=0.13):
        rc = self.cell()
        cx, cy = rc[1] * CELL + CELL / 2, -(rc[0] * CELL + CELL / 2)
        off = {N: self.Y - cy, S: cy - self.Y, E: self.X - cx, W: cx - self.X}
        return any(self.is_wall(rc, d) and WALL_DIST - off[d] < half_width for d in range(4))

    def move(self, dx, dy, dth_deg):
        self.t += 0.6 + math.hypot(dx, dy) / 0.10 + abs(dth_deg) / 25.0
        self.th += dth_deg * (1 + self.rng.gauss(0, self.turn_err / 90.0))
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        dist *= 1 + self.rng.gauss(0, self.step_err)
        local = math.atan2(dy, dx)
        steps = max(1, int(dist / 0.01))
        drift = self.rng.gauss(0, self.drift_deg) * dist / CELL
        touched = False
        for _ in range(steps):
            self.th += drift / steps
            a = math.radians(self.th) + local
            nx, ny = self.X + 0.01 * math.cos(a), self.Y + 0.01 * math.sin(a)
            if self._crosses_wall(self.X, self.Y, nx, ny):
                self.hits += 1
                self.t += 10.0
                return
            self.X, self.Y = nx, ny
            touched = touched or self._touching()
        if touched:
            self.touches += 1

    def is_goal(self, pos):
        return False

    def red_below(self):
        self.t += 1.5
        return self.cell() == self.goal and self.rng.random() < 0.95

    def check(self):
        pass


# --------------------------------------------------------------------------
# Agente
# --------------------------------------------------------------------------
class Agent(object):
    def __init__(self, robot, mapa, time_limit, log, map_path=None):
        self.r, self.m = robot, mapa
        self.time_limit, self.log, self.map_path = time_limit, log, map_path
        self.pos = None
        self.heading = None
        self.red_votes = {}
        self.goal_doubts = 0

    def save(self):
        if self.map_path:
            self.m.save(self.map_path)

    # ---- percepcion ----
    def sense(self, rels=(0, 3, 1), align=True):
        tags = self.r.look(list(rels))
        cls = classify(tags)
        near_ids = [t[0] for rel in cls for t in cls[rel]["near"]]
        own = max(set(near_ids), key=near_ids.count) if near_ids else None

        # Relocalizar con ids aprendidos (mi celda o, por los tags lejanos, una vecina)
        where = None
        if own is not None:
            where = self.m.cell_with_id(own)
        if where is None:
            for rel in rels:
                d = (self.heading + rel) % 4
                for t in cls[rel]["far"]:
                    c = self.m.cell_with_id(t[0])
                    if c is not None and inside(nb(c, (d + 2) % 4)):
                        where = nb(c, (d + 2) % 4)
                        break
        if where is not None and where != self.pos:
            self.log("[!] relocalizo por ids: creia estar en %s, estoy en %s" % (self.pos, where))
            self.pos = where

        # Aprender ids (solo si el id no pertenece ya, con certeza, a otra celda)
        if own is not None:
            known = self.m.cell_with_id(own)
            if known is None or known == self.pos:
                self.m.learn_id(self.pos, own)
        for rel in rels:
            d = (self.heading + rel) % 4
            o = nb(self.pos, d)
            for t in cls[rel]["far"]:
                known = self.m.cell_with_id(t[0])
                if known is None or known == o:
                    self.m.learn_id(o, t[0])

        # Paredes
        for rel in rels:
            d = (self.heading + rel) % 4
            if cls[rel]["near"]:
                self.m.set(self.pos, d, WALL)
            elif cls[rel]["far"]:
                self.m.set(self.pos, d, OPEN)
                o = nb(self.pos, d)
                if inside(o):
                    self.m.set(o, d, WALL)
            else:
                self.m.set(self.pos, d, WEAK)
        self.see_red(getattr(self.r, "last_red", []))
        if align:
            dx, dy, dth = alignment(cls, tags)
            self.correct(dx, dy, dth)
        return own, cls

    def see_red(self, points):
        """Rojo en una de las celdas candidatas a centro -> confirma la meta (y el lado)."""
        if not points or self.m.side is not None:
            return
        fwd, left = self.heading, (self.heading + 3) % 4
        cands = goal_candidates()
        for x, y in points:
            f = int(math.floor((x + CELL / 2) / CELL))
            l = int(math.floor((y + CELL / 2) / CELL))
            c = (self.pos[0] + f * DR[fwd] + l * DR[left], self.pos[1] + f * DC[fwd] + l * DC[left])
            if c in cands:
                self.red_votes[c] = self.red_votes.get(c, 0) + 1
        for c in cands:
            if self.red_votes.get(c, 0) >= GOAL_MIN_POINTS:
                self.log("[META VISTA] piso rojo en %s" % (c,))
                self.m.resolve("left" if c[0] < S0 else "right")
                self.save()
                return

    def correct(self, dx, dy, dth):
        mx = dx if dx is not None and abs(dx) >= ALIGN_MIN_XY else 0.0
        my = dy if dy is not None and abs(dy) >= ALIGN_MIN_XY else 0.0
        mt = dth if dth is not None and abs(dth) >= ALIGN_MIN_TH else 0.0
        mx = max(-ALIGN_MAX_XY, min(ALIGN_MAX_XY, mx))
        my = max(-ALIGN_MAX_XY, min(ALIGN_MAX_XY, my))
        mt = max(-ALIGN_MAX_TH, min(ALIGN_MAX_TH, mt))
        if mt:
            self.r.move(0.0, 0.0, mt)
            a = math.radians(-mt)
            mx, my = mx * math.cos(a) - my * math.sin(a), mx * math.sin(a) + my * math.cos(a)
        if mx or my:
            self.r.move(mx, my, 0.0)

    def front_blocked(self):
        own, cls = self.sense(rels=(0,), align=True)
        if cls[0]["near"]:
            return True
        fr = self.r.front_range()
        return fr is not None and fr < 0.40

    def unknown_rels(self):
        rels = [0]
        for rel in (3, 1):
            if self.m.get(self.pos, (self.heading + rel) % 4) == UNKNOWN:
                rels.append(rel)
        return rels

    def turn_to(self, d):
        k = (d - self.heading) % 4
        if k == 0:
            return False
        self.r.move(0, 0, {1: -90.0, 3: 90.0, 2: 180.0}[k])
        self.heading = d
        self.sense(rels=(0, 3, 1), align=True)
        return True

    def forward(self, n=1, rels=None):
        origin, d = self.pos, self.heading
        line = [origin]
        for _ in range(n):
            line.append(nb(line[-1], d))
        self.r.move(CELL * n, 0.0, 0.0)
        self.pos = line[-1]
        own, cls = self.sense(rels=rels if rels is not None else self.unknown_rels(), align=True)
        if self.pos in line:
            k = line.index(self.pos)
            for i in range(k):
                self.m.set(line[i], d, OPEN)
                self.m.visited.add(line[i + 1])
            if k < n:
                self.log("[!] avance incompleto: quede en %s (esperaba %s)" % (self.pos, line[-1]))
                fr = self.r.front_range()
                if fr is not None and fr < 0.40:
                    self.m.set(self.pos, d, WALL, force=True)
        else:
            self.m.visited.add(self.pos)
        return own, cls

    def at_goal(self):
        if self.r.is_goal(self.pos):
            return True
        if self.m.goal is None or self.pos != self.m.goal:
            return False
        if self.r.red_below():
            return True
        self.goal_doubts += 1
        self.log("[?] creo estar en el centro %s pero no veo rojo abajo (%d)" % (self.pos, self.goal_doubts))
        if self.goal_doubts >= 3:
            self.log("[!] acepto: la meta es el centro por reglamento")
            return True
        self.sense(rels=(0, 3, 1), align=True)
        return False

    # ---- intento 1 ----
    def bootstrap(self):
        """Salida: esquina con paredes atras y a los lados; el frente es la unica salida (3.1.4)."""
        self.pos, self.heading = START, START_HEADING
        self.m.visited.add(START)
        for rel in (1, 2, 3):
            self.m.set(START, (START_HEADING + rel) % 4, WALL)
        self.sense(rels=(0, 3, 1), align=True)
        self.m.set(START, START_HEADING, OPEN)
        self.save()

    def plan(self):
        """Ruta optimista (lo no visto se supone abierto) a la meta o, si aun no se sabe el lado,
        a la mas cercana de las dos candidatas."""
        best = None
        targets = [self.m.goal] if self.m.goal else goal_candidates()
        for g in targets:
            p = a_star_search(self.m, self.pos, g, (OPEN, WEAK, UNKNOWN), costo_desconocido=0.3)
            if p is not None and (best is None or path_cost(p) < path_cost(best)):
                best = p
        return best

    def explore(self, bootstrap=True):
        if bootstrap:
            self.bootstrap()
        while self.r.elapsed() < self.time_limit:
            if self.at_goal():
                self.save()
                self.log("[META] en %s a los %.0f s" % (self.pos, self.r.elapsed()))
                return True
            if self.m.goal is not None and self.pos == self.m.goal:
                continue
            path = self.plan()
            if path is None or len(path) < 2:
                self.log("[!] sin ruta posible hacia el centro")
                return False
            d = dir_between(self.pos, path[1])
            self.turn_to(d)
            n = 0
            while (n + 1 < len(path) and n < 2 and dir_between(path[n], path[n + 1]) == d
                   and self.m.get(path[n], d) == OPEN and path[n + 1] in self.m.visited):
                n += 1
            if n >= 1:
                self.forward(n)
                self.log("  %5.0f s  viaje rapido x%d -> %s" % (self.r.elapsed(), n, self.pos))
                continue
            if self.m.get(self.pos, d) != OPEN and self.front_blocked():
                self.log("[!] pared al frente en %s %s" % (self.pos, DNAME[d]))
                self.m.set(self.pos, d, WALL)
                continue
            self.forward(1)
            self.save()
            self.log("  %5.0f s  celda %s  rumbo %s%s" % (
                self.r.elapsed(), self.pos, DNAME[self.heading],
                "" if self.m.side else "  (lado aun desconocido)"))
        self.save()
        self.log("[!] tiempo agotado")
        return False

    # ---- intento 2 ----
    def run_optimal(self, max_chain=2):
        m = self.m
        self.pos, self.heading = m.start, m.start_heading
        self.goal_doubts = 0
        self.sense(rels=(0, 3, 1), align=True)
        while self.r.elapsed() < self.time_limit:
            if self.at_goal():
                self.log("[META] a los %.0f s" % self.r.elapsed())
                return True
            if m.goal is not None and self.pos == m.goal:
                continue
            path = None
            if m.goal is not None:
                path = a_star_search(m, self.pos, m.goal, (OPEN,)) or \
                    a_star_search(m, self.pos, m.goal, (OPEN, WEAK))
            if path is None:
                self.log("[..] sin ruta conocida al centro: exploro")
                return self.explore(bootstrap=False)
            d, n_total = segments(path)[0]
            self.turn_to(d)
            n = min(n_total, max_chain)
            if m.get(self.pos, d) != OPEN and self.front_blocked():
                self.m.set(self.pos, d, WALL)
                continue
            self.forward(n, rels=(0, 3, 1) if n_total > n else (0,))
            self.log("  %5.0f s  tramo %s x%d -> %s" % (self.r.elapsed(), DNAME[d], n, self.pos))
        self.log("[!] tiempo agotado")
        return False


# --------------------------------------------------------------------------
# Simulacion
# --------------------------------------------------------------------------
def random_maze(rng, loops=0.10):
    """Mundo real SIZE x SIZE: arbol aleatorio + ciclos; salida en una esquina con 3 paredes."""
    while True:
        walls = set((r, c, d) for r in range(SIZE) for c in range(SIZE) for d in range(4))
        seen, stack = set([(0, 0)]), [(0, 0)]
        while stack:
            u = stack[-1]
            opts = [d for d in range(4) if w_inside(nb(u, d)) and nb(u, d) not in seen]
            if not opts:
                stack.pop()
                continue
            d = rng.choice(opts)
            v = nb(u, d)
            walls.discard((u[0], u[1], d))
            walls.discard((v[0], v[1], (d + 2) % 4))
            seen.add(v)
            stack.append(v)
        start = rng.choice(w_corners())
        inner = [(r, c, d) for (r, c, d) in walls if w_inside(nb((r, c), d)) and d in (E, S)]
        rng.shuffle(inner)
        for (r, c, d) in inner[:int(loops * len(inner))]:
            if (r, c) == start or nb((r, c), d) == start:
                continue
            walls.discard((r, c, d))
            o = nb((r, c), d)
            walls.discard((o[0], o[1], (d + 2) % 4))
        open_dirs = [d for d in range(4) if (start[0], start[1], d) not in walls]
        if len(open_dirs) == 1:
            return walls, start, open_dirs[0]


def w_bfs_len(walls, a, b):
    dist, q, i = {a: 0}, [a], 0
    while i < len(q):
        u = q[i]
        i += 1
        if u == b:
            return dist[u]
        for d in range(4):
            v = nb(u, d)
            if w_inside(v) and (u[0], u[1], d) not in walls and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return None


def sim_once(seed, time_limit=300.0, verbose=False, p_miss=0.05, p_red=0.9, scrambled=True):
    configure_size(SIZE)
    rng = random.Random(seed)
    walls, start, sh = random_maze(rng)
    goal = (SIZE // 2, SIZE // 2)
    pool = list(range(64)) if scrambled else None
    if scrambled:
        rng.shuffle(pool)
        ids = dict(((r, c), pool[r * SIZE + c]) for r in range(SIZE) for c in range(SIZE))
    else:
        ids = dict(((r, c), r * SIZE + c) for r in range(SIZE) for c in range(SIZE))
    log = (lambda s: print(s)) if verbose else (lambda s: None)

    r1 = SimRobot(walls, start, sh, goal, ids, rng, p_miss=p_miss, p_red=p_red)
    m = MazeMap()
    a1 = Agent(r1, m, time_limit, log)
    try:
        ok1 = a1.explore() and r1.cell() == goal
    except RuntimeError as e:
        log("[!] %s" % e)
        ok1 = False
    res = {"seed": seed, "ok1": ok1, "t1": r1.elapsed() if ok1 else time_limit,
           "hits1": r1.hits, "touch1": r1.touches, "cells1": len(m.visited),
           "side_ok": m.side is not None}
    if verbose:
        print("\nMapa aprendido (marco propio; S salida, M meta, . visitada):")
        print(m.render())
        log("\n--- intento 2 ---")

    r2 = SimRobot(walls, start, sh, goal, ids, rng, p_miss=p_miss, p_red=p_red)
    a2 = Agent(r2, m, time_limit, log)
    try:
        ok2 = a2.run_optimal() and r2.cell() == goal
    except RuntimeError as e:
        log("[!] %s" % e)
        ok2 = False
    end = r2.cell()
    res.update(ok2=ok2, t2=r2.elapsed() if ok2 else time_limit, hits2=r2.hits, touch2=r2.touches,
               dist2=w_bfs_len(walls, end, goal) if w_inside(end) else None,
               opt_len=w_bfs_len(walls, start, goal))
    res["final"] = 0.3 * res["t1"] + 0.7 * res["t2"]
    if verbose:
        print("\nIntento 1: %s en %.0f s, %d celdas, %d golpes | Intento 2: %s en %.0f s, %d golpes"
              % ("META" if res["ok1"] else "no", res["t1"], res["cells1"], res["hits1"],
                 "META" if res["ok2"] else "no", res["t2"], res["hits2"]))
        print("Ruta optima real: %s celdas.  Tiempo final = 0.3*T1 + 0.7*T2 = %.0f s"
              % (res["opt_len"], res["final"]))
    return res


def sim_batch(n, time_limit, p_miss, p_red, scrambled):
    rs = [sim_once(s, time_limit, False, p_miss, p_red, scrambled) for s in range(n)]
    k = float(len(rs))
    ok1 = [r for r in rs if r["ok1"]]
    ok2 = [r for r in rs if r["ok2"]]
    print("Laberintos %dx%d: %d  ids %s  p_miss=%.2f  p_rojo=%.2f  limite=%.0f s"
          % (SIZE, SIZE, len(rs), "revueltos" if scrambled else "en orden", p_miss, p_red, time_limit))
    print("Intento 1 llega: %3.0f%%   tiempo mediano %.0f s   celdas medianas %d"
          % (100 * len(ok1) / k, median([r["t1"] for r in ok1]) or 0, median([r["cells1"] for r in rs])))
    print("Intento 2 llega: %3.0f%%   tiempo mediano %.0f s"
          % (100 * len(ok2) / k, median([r["t2"] for r in ok2]) or 0))
    print("Golpes por intento: I1 %.2f  I2 %.2f   |  roces: I1 %.2f  I2 %.2f"
          % (sum(r["hits1"] for r in rs) / k, sum(r["hits2"] for r in rs) / k,
             sum(r["touch1"] for r in rs) / k, sum(r["touch2"] for r in rs) / k))
    fails = [r["dist2"] for r in rs if not r["ok2"] and r["dist2"] is not None]
    print("Lado resuelto en I1: %3.0f%%  |  ruta optima real mediana: %d celdas  |  tiempo final mediano %.0f s"
          % (100 * sum(r["side_ok"] for r in rs) / k, median([r["opt_len"] for r in rs]),
             median([r["final"] for r in rs])))
    if fails:
        print("Sin meta en I2: a %d celdas (mediana)" % median(fails))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def make_logger(path):
    def log(s):
        line = "%s %s" % (time.strftime("%H:%M:%S"), s)
        print(line)
        sys.stdout.flush()
        if path:
            try:
                with open(path, "a") as f:
                    f.write(line + "\n")
            except IOError:
                pass
    return log


def main():
    ap = argparse.ArgumentParser(description="Laberinto Copa NAO 2026")
    ap.add_argument("--run", choices=["both", "1", "2"], default="both")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--map", default="mapa.json")
    ap.add_argument("--time-limit", type=float, default=300.0, help="s por intento")
    ap.add_argument("--test-goal", default=None,
                    help="SOLO PRUEBAS: 'fila,col' (marco propio) donde declarar meta sin rojo")
    ap.add_argument("--check", action="store_true", help="robot quieto: imprime lo que percibe")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--p-miss", type=float, default=0.05)
    ap.add_argument("--p-red", type=float, default=0.9)
    ap.add_argument("--ids-ordered", action="store_true", help="sim: ids en orden (no revueltos)")
    args = ap.parse_args()

    if args.sim:
        if args.batch:
            sim_batch(args.batch, args.time_limit, args.p_miss, args.p_red, not args.ids_ordered)
        else:
            sim_once(args.seed, args.time_limit, True, args.p_miss, args.p_red, not args.ids_ordered)
        return

    here = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(here, args.map)
    log = make_logger(os.path.join(here, "maze_log.txt"))
    log("===== maze_nao v3 %s  %dx%d  celda %.3f m =====" % (
        "check" if args.check else args.run, SIZE, SIZE, CELL))
    test_goal = tuple(int(v) for v in args.test_goal.split(",")) if args.test_goal else None
    if test_goal:
        log("[!] --test-goal %s activo: SOLO PARA PRUEBAS" % (test_goal,))
    robot = NaoRobot(args.ip, args.port, log, test_goal)
    try:
        robot.ensure_vision()

        if args.check:
            robot.motion.wakeUp()
            robot.posture.goToPosture("StandInit", 0.5)
            tags = robot.look([0, 3, 1])
            cls = classify(tags)
            print("Tags vistos (id, x, y, yaw):")
            for t in tags:
                print("  id=%2d  x=%+.3f  y=%+.3f  yaw=%s" % (
                    t[0], t[1], t[2], "%+.1f" % t[3] if t[3] is not None else "-"))
            for rel, name in ((0, "FRENTE"), (3, "IZQUIERDA"), (1, "DERECHA")):
                c = cls[rel]
                print("  %-9s %s" % (name, "PARED" if c["near"] else
                                     ("abierta (veo la del fondo)" if c["far"] else "abierta?")))
            dx, dy, dth = alignment(cls, tags)
            f = lambda v, u: ("%+.3f %s" % (v, u)) if v is not None else "-"
            print("Correccion: dx %s  dy %s  giro %s" % (f(dx, "m"), f(dy, "m"), f(dth, "grados")))
            print("Sonar al frente: %s" % robot.front_range())
            red = robot.last_red
            if red:
                p = min(red, key=lambda q: math.hypot(*q))
                print("Piso rojo enfrente/lados: %d puntos; el mas cercano en x=%.2f y=%.2f" % (len(red), p[0], p[1]))
            else:
                print("Piso rojo enfrente/lados: no veo")
            print("Rojo bajo los pies: %s" % ("SI" if robot.red_below() else "no"))
            robot.crouch()
            return

        mapa = None
        if args.run in ("both", "1"):
            if os.path.exists(map_path):
                os.rename(map_path, map_path + ".anterior")
            configure_size(SIZE)
            mapa = MazeMap()
            agent = Agent(robot, mapa, args.time_limit, log, map_path)
            robot.wait_start("INTENTO 1: toca la cabeza para arrancar")
            ok = agent.explore()
            robot.crouch()
            log("\n" + mapa.render(agent.pos, agent.heading))
            log("[fin intento 1] %s en %.0f s. ids aprendidos: %d celdas. Mapa en %s"
                % ("META" if ok else "sin meta", robot.elapsed(), len(mapa.ids), map_path))

        if args.run in ("both", "2"):
            if mapa is None:
                mapa = MazeMap.load(map_path)
            path = a_star_search(mapa, mapa.start, mapa.goal, (OPEN,)) if mapa.goal else None
            log("\n" + mapa.render(path=path))
            log("Ruta: %s" % ("  ".join("%s%d" % (DNAME[d], k) for d, k in segments(path))
                              if path else "ninguna conocida (seguira explorando)"))
            agent = Agent(robot, mapa, args.time_limit, log, map_path)
            robot.wait_start("INTENTO 2: pon el robot en la salida (misma orientacion) y toca la cabeza")
            ok = agent.run_optimal()
            robot.crouch()
            log("[fin intento 2] %s en %.0f s" % ("META" if ok else "sin meta", robot.elapsed()))

    except EmergencyStop:
        log("[PARO] se toco la cabeza durante el intento")
        robot.emergency()
    except KeyboardInterrupt:
        log("[!] Control+C")
        robot.emergency()
    except Exception as e:
        log("[ERROR] %s" % e)
        import traceback
        log(traceback.format_exc())
        robot.emergency()
    finally:
        robot.close()


if __name__ == "__main__":
    main()
