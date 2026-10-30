# -*- coding: utf-8 -*-
"""
penal_nao.py -- Tiro penal Copa NAO CCM 2026. Corre DENTRO del NAO (Python 2.7, NAOqi 2.x).

Por que no el timeline de Choregraphe: un timeline son angulos fijos calibrados para el centro de
masa de UN robot (Apollo). Aqui la patada usa el BALANCEADOR DE CUERPO COMPLETO de NAOqi
(wbEnable / wbGoToBalance): el robot mantiene su centro de masa sobre el pie de apoyo en tiempo real
mientras la otra pierna patea, sin importar las diferencias entre robots.

Secuencia
  1. Se para, baja la cabeza para ver la pelota y ESPERA: tocar la cabeza = arranque (al silbatazo).
  2. Busca la pelota naranja con la camara de abajo y camina hasta dejarla frente al pie que patea.
     (Si no la ve, usa --ball-dist: distancia de la punta de los pies al centro de la pelota.)
  3. Pasa el peso al pie de apoyo, lleva el pie de patada atras y lo dispara hacia adelante.
  4. Regresa a dos pies, se para y se agacha.
  Tocar la cabeza durante la rutina = PARO (reglamento 3.3.3: desactivacion fuera de las piernas).

PRUEBAS EN ORDEN (con alguien listo para atraparlo):
  python penal_nao.py --check-ball               # robot quieto: donde ve la pelota (no se mueve)
  python penal_nao.py --stage balance --no-wait  # solo pasa el peso al pie izquierdo y regresa
  python penal_nao.py --stage lift --no-wait     # pasa el peso y levanta el pie derecho despacio
  python penal_nao.py --stage kick --no-wait --t-kick 0.8   # patada LENTA, sin caminar a la pelota
  python penal_nao.py --stage kick --no-wait --t-kick 0.4   # mas rapida (ir bajando de 0.1 en 0.1)
  python penal_nao.py --t-kick 0.35              # rutina completa: espera cabeza, camina, patea

Competencia:
  nohup python penal_nao.py --t-kick <el que funciono> > penal.out 2>&1 &
"""
from __future__ import print_function, division

import argparse
import math
import sys
import time

TOUCH_KEYS = ["Device/SubDeviceList/Head/Touch/Front/Sensor/Value",
              "Device/SubDeviceList/Head/Touch/Middle/Sensor/Value",
              "Device/SubDeviceList/Head/Touch/Rear/Sensor/Value"]
FRAME_WORLD, FRAME_ROBOT = 1, 2
AXIS_MASK_ALL = 63

BALL_R = 0.0325             # m (pelota oficial 65 mm)
TOE_X = 0.105               # m, punta del pie delante del tobillo (origen de FRAME_ROBOT)
FOOT_Y = 0.05               # m, separacion lateral de cada pie respecto al centro
BALL_PITCH = 20.0           # grados de cabeza para ver la pelota cerca de los pies
FOV_V5 = (60.97, 47.64)


class EmergencyStop(Exception):
    pass


def log(s):
    print("%s %s" % (time.strftime("%H:%M:%S"), s))
    sys.stdout.flush()


class Penal(object):
    def __init__(self, args):
        from naoqi import ALProxy
        import numpy
        self.np = numpy
        self.a = args
        self.motion = ALProxy("ALMotion", args.ip, args.port)
        self.posture = ALProxy("ALRobotPosture", args.ip, args.port)
        self.memory = ALProxy("ALMemory", args.ip, args.port)
        self.video = ALProxy("ALVideoDevice", args.ip, args.port)
        self.cam = None
        self.armed = False
        self.kick = "RLeg" if args.leg == "R" else "LLeg"
        self.support = "LLeg" if args.leg == "R" else "RLeg"
        self.side = -1.0 if args.leg == "R" else 1.0       # y del pie que patea

    # ------------------------------------------------------------------
    # Cabeza: arranque y paro
    # ------------------------------------------------------------------
    def touched(self):
        try:
            return any(self.memory.getData(k) > 0.5 for k in TOUCH_KEYS)
        except RuntimeError:
            return False

    def check(self):
        if self.armed and self.touched():
            raise EmergencyStop()

    def wait(self, task_id):
        """Espera una tarea .post revisando el paro."""
        while self.motion.isRunning(task_id):
            if self.armed and self.touched():
                try:
                    self.motion.stop(task_id)
                except Exception:
                    pass
                raise EmergencyStop()
            time.sleep(0.02)

    def head(self, yaw, pitch):
        self.motion.angleInterpolationWithSpeed(["HeadYaw", "HeadPitch"],
                                                [math.radians(yaw), math.radians(pitch)], 0.3)

    def stand(self):
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)
        self.motion.setStiffnesses("Head", 1.0)
        self.head(0.0, BALL_PITCH)

    def wait_start(self):
        log("[..] listo. Toca la cabeza al silbatazo")
        while not self.touched():
            time.sleep(0.03)
        while self.touched():               # esperar a que suelten (si no, el paro se dispara solo)
            time.sleep(0.03)
        self.armed = True
        log("[ok] arranque")

    # ------------------------------------------------------------------
    # Pelota
    # ------------------------------------------------------------------
    def subscribe(self):
        for s in self.video.getSubscribers():
            if s.startswith("penal_ball"):
                self.video.unsubscribe(s)
        self.cam = self.video.subscribeCamera("penal_ball", 1, 1, 13, 15)   # abajo, 320x240, BGR

    def close(self):
        try:
            if self.cam:
                self.video.unsubscribe(self.cam)
        except Exception:
            pass

    def ball_mask(self, img):
        np = self.np
        b, g, r = img[:, :, 0].astype(np.int32), img[:, :, 1].astype(np.int32), img[:, :, 2].astype(np.int32)
        # Naranja saturado: r >> g >> b. La duela de madera (cafe/naranja palido) no pasa g > 1.8*b.
        return (r > 100) & (r > 1.55 * g) & (g > 1.8 * b) & (g > 35)

    def find_ball(self, save=None):
        """Posicion (x, y) del centro de la pelota en FRAME_ROBOT, o None."""
        np = self.np
        import cv2
        img = self.video.getImageRemote(self.cam)
        if img is None:
            return None
        w, h = img[0], img[1]
        a = np.frombuffer(img[6], dtype=np.uint8).reshape(h, w, 3)
        T = np.array(self.motion.getTransform("CameraBottom", FRAME_ROBOT, True)).reshape(4, 4)
        mask = self.ball_mask(a).astype(np.uint8) * 255
        if save:
            cv2.imwrite(save, np.hstack([a, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = res[-2]
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 25:
            return None
        m = cv2.moments(c)
        u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
        fx = (w / 2.0) / math.tan(math.radians(FOV_V5[0] / 2))
        fy = (h / 2.0) / math.tan(math.radians(FOV_V5[1] / 2))
        d = T[:3, :3].dot(np.array([1.0, -(u - w / 2.0) / fx, -(v - h / 2.0) / fy]))
        t = T[:3, 3]
        if d[2] > -1e-3:
            return None
        s = (BALL_R - t[2]) / d[2]                 # rayo al plano del CENTRO de la pelota
        x, y = t[0] + s * d[0], t[1] + s * d[1]
        if not (0.0 < x < 1.5 and abs(y) < 0.8):
            return None
        return (x, y, area)

    def ball_median(self, n=5):
        got = []
        for _ in range(n):
            self.check()
            b = self.find_ball()
            if b:
                got.append(b)
            time.sleep(0.07)
        if len(got) < max(2, n // 2):
            return None
        xs = sorted(g[0] for g in got)
        ys = sorted(g[1] for g in got)
        return (xs[len(xs) // 2], ys[len(ys) // 2])

    # ------------------------------------------------------------------
    # Acercamiento
    # ------------------------------------------------------------------
    def move(self, dx, dy, dth):
        tid = self.motion.post.moveTo(dx, dy, dth)
        self.wait(tid)

    def approach(self):
        """Deja la pelota en (kick_x, +-FOOT_Y) frente al pie que patea, apuntando a --aim grados."""
        kx, ky = self.a.kick_x, self.side * FOOT_Y
        aim = math.radians(self.a.aim)
        ball = None if self.a.no_ball else self.ball_median()
        if ball is None:
            ball = (TOE_X + self.a.ball_dist, 0.0)
            log("[..] no veo la pelota: uso --ball-dist -> pelota en x=%.2f y=%.2f" % ball)
        else:
            log("[ok] pelota en x=%.3f y=%.3f" % ball)
        # Pose objetivo: girado 'aim' y con la pelota en (kx, ky) de mi nuevo marco
        c, s = math.cos(aim), math.sin(aim)
        px = ball[0] - (c * kx - s * ky)
        py = ball[1] - (s * kx + c * ky)
        log("[..] acercamiento: dx=%.3f dy=%.3f giro=%.1f" % (px, py, self.a.aim))
        if abs(px) > 0.01 or abs(py) > 0.01 or abs(aim) > 0.02:
            self.move(px, py, aim)
        # Ajuste fino (1-2 veces) si se ve la pelota
        for _ in range(2):
            if self.a.no_ball:
                break
            time.sleep(0.3)
            b = self.ball_median()
            if b is None:
                log("[!] perdi la pelota en el ajuste fino; pateo asi")
                break
            ex, ey = b[0] - kx, b[1] - ky
            log("[..] ajuste fino: pelota en x=%.3f y=%.3f (error %.3f, %.3f)" % (b[0], b[1], ex, ey))
            if abs(ex) < 0.02 and abs(ey) < 0.015:
                break
            self.move(max(-0.06, min(0.10, ex)), max(-0.06, min(0.06, ey)), 0.0)

    # ------------------------------------------------------------------
    # Patada con balanceo de cuerpo completo
    # ------------------------------------------------------------------
    def _offset_tf(self, tf, dx, dy, dz, yaw):
        """tf (12 o 16 floats, FRAME_WORLD) desplazado (dx, dy, dz) en ejes del ROBOT."""
        c, s = math.cos(yaw), math.sin(yaw)
        out = list(tf[:12])
        out[3] += c * dx - s * dy
        out[7] += s * dx + c * dy
        out[11] += dz
        return out

    def kick_motion(self, stage):
        """Si hay paro o error a media patada NO apaga el balanceador: safe_end() regresa el pie
        al piso y a dos pies con el balanceador activo, y despues lo apaga."""
        a = self.a
        m = self.motion
        m.wbEnable(True)
        self.wb_on = True
        m.wbFootState("Fixed", "Legs")
        m.wbEnableBalanceConstraint(True, "Legs")
        log("[..] peso al pie de apoyo (%s)" % self.support)
        self.wait(m.post.wbGoToBalance(self.support, a.t_shift))
        if stage == "balance":
            time.sleep(1.0)
        else:
            tf0 = m.getTransform(self.kick, FRAME_WORLD, False)
            self.tf0 = list(tf0[:12])
            m.wbFootState("Free", self.kick)
            self.foot_free = True
            yaw = m.getRobotPosition(False)[2]
            if stage == "lift":
                path = [self._offset_tf(tf0, 0.0, 0.0, a.lift, yaw), list(tf0[:12])]
                times = [1.5, 3.0]
            else:
                back = self._offset_tf(tf0, -a.back, 0.0, a.lift, yaw)
                fwd = self._offset_tf(tf0, a.fwd, 0.0, a.lift * 0.6, yaw)
                times = [a.t_back, a.t_back + a.t_kick, a.t_back + a.t_kick + a.t_return]
                path = [back, fwd, list(tf0[:12])]
                log("[..] patada: atras %.2f m en %.2f s, adelante %.2f m en %.2f s"
                    % (a.back, a.t_back, a.fwd, a.t_kick))
            self.wait(m.post.transformInterpolations(self.kick, FRAME_WORLD, path,
                                                     AXIS_MASK_ALL, times))
            m.wbFootState("Fixed", "Legs")
            self.foot_free = False
        log("[..] regreso a dos pies")
        self.wait(m.post.wbGoToBalance("Legs", a.t_shift))
        m.wbEnable(False)
        self.wb_on = False

    def safe_end(self):
        """Tras paro o error: pie de regreso al piso, peso en dos pies, balanceador fuera, sentarse."""
        self.armed = False
        m = self.motion
        try:
            m.stopMove()
        except Exception:
            pass
        if getattr(self, "wb_on", False):
            if getattr(self, "foot_free", False) and getattr(self, "tf0", None):
                try:
                    log("[..] regresando el pie al piso")
                    m.transformInterpolations(self.kick, FRAME_WORLD, [self.tf0], AXIS_MASK_ALL, [1.5])
                except Exception:
                    pass
            try:
                m.wbFootState("Fixed", "Legs")
                m.wbGoToBalance("Legs", 1.5)
            except Exception:
                pass
            try:
                m.wbEnable(False)
            except Exception:
                pass
        try:
            self.posture.goToPosture("Sit", 0.5)
            m.rest()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def run(self):
        a = self.a
        self.subscribe()
        self.stand()
        if a.check_ball:
            for i in range(5):
                b = self.find_ball(save="ball_dbg.png" if i == 0 else None)
                log("pelota: %s" % ("x=%.3f y=%.3f (area %d px)" % b if b else "no la veo"))
                time.sleep(0.4)
            log("Imagen de depuracion (camara | mascara naranja): ball_dbg.png")
            log("Para patear, la pelota debe quedar en x=%.2f y=%+.2f" % (a.kick_x, self.side * FOOT_Y))
            return
        if a.no_wait:
            self.armed = True
            log("[..] --no-wait: arranco ya (tocar la cabeza = paro)")
        else:
            self.wait_start()
        t0 = time.time()
        if a.stage == "full":
            self.approach()
            stage = "kick"
        else:
            stage = a.stage
        self.head(0.0, 0.0)
        self.posture.goToPosture("StandInit", 0.5)
        self.kick_motion(stage)
        self.posture.goToPosture("StandInit", 0.5)
        log("[ok] rutina terminada en %.1f s" % (time.time() - t0))
        self.armed = False
        time.sleep(1.5)
        self.motion.rest()


def main():
    ap = argparse.ArgumentParser(description="Tiro penal NAO")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--leg", choices=["R", "L"], default="R", help="pierna que patea (Apollo: derecha)")
    ap.add_argument("--stage", choices=["full", "balance", "lift", "kick"], default="full",
                    help="full = buscar pelota + acercarse + patear; los demas son pruebas sin caminar")
    ap.add_argument("--no-wait", action="store_true", help="no esperar la cabeza (solo pruebas)")
    ap.add_argument("--check-ball", action="store_true", help="solo mostrar donde ve la pelota")
    ap.add_argument("--no-ball", action="store_true", help="no buscar la pelota: usar --ball-dist")
    ap.add_argument("--ball-dist", type=float, default=0.20,
                    help="m, de la PUNTA de los pies al CENTRO de la pelota (si no la ve)")
    ap.add_argument("--aim", type=float, default=0.0, help="grados; + = tirar hacia la izquierda")
    ap.add_argument("--kick-x", type=float, default=0.17,
                    help="m; donde debe quedar el centro de la pelota (desde el tobillo) antes de patear")
    ap.add_argument("--t-shift", type=float, default=1.5, help="s para pasar el peso de pie")
    ap.add_argument("--t-back", type=float, default=0.8, help="s para llevar el pie atras")
    ap.add_argument("--t-kick", type=float, default=0.6, help="s del golpe (menor = mas fuerte y mas riesgo)")
    ap.add_argument("--t-return", type=float, default=0.8, help="s para regresar el pie")
    ap.add_argument("--back", type=float, default=0.05, help="m que el pie va hacia atras")
    ap.add_argument("--fwd", type=float, default=0.10, help="m que el pie va hacia adelante")
    ap.add_argument("--lift", type=float, default=0.035, help="m que se levanta el pie")
    args = ap.parse_args()

    p = Penal(args)
    try:
        p.run()
    except EmergencyStop:
        log("[PARO] se toco la cabeza")
        p.safe_end()
    except KeyboardInterrupt:
        log("[!] Control+C")
        p.safe_end()
    except Exception as e:
        import traceback
        log("[ERROR] %s\n%s" % (e, traceback.format_exc()))
        p.safe_end()
    finally:
        p.close()


if __name__ == "__main__":
    main()
