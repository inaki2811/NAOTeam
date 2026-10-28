# -*- coding: utf-8 -*-
"""
walk_test.py (v2) -- Prueba de vision MIENTRAS el NAO camina hacia un AprilTag.

Requiere nao_apriltag.py corriendo en segundo plano con --publish:
  nohup python nao_apriltag.py --camera both --publish --quiet > vision.log 2>&1 &

Luego:
  python walk_test.py --no-walk      # se para, pone la cabeza y solo lee (sin caminar)
  python walk_test.py                # camina hacia el tag y se detiene a --stop-x

Cabeza (del barrido con head_sweep.py en Tesla):
  - lejos  (x > --switch-x): HeadPitch = --far-pitch  (+10 grados, ve con la camara de arriba)
  - cerca  (x <= --switch-x): HeadPitch = --near-pitch (-7.5 grados, ve con la de abajo)

Seguridad: robot EN EL PISO y alguien al lado. Se detiene si:
  - el tag queda a menos de --stop-x metros,
  - la vision no ve tags por mas de --lost s,
  - pasan --max-time s, o presionas Ctrl+C.

Guarda un registro walk_log.csv para revisar despues (t, x, y, cam, pitch, caminando).
"""
from __future__ import print_function, division

import argparse
import math
import time

from naoqi import ALProxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--speed", type=float, default=0.3, help="fraccion de la velocidad maxima (0-1)")
    ap.add_argument("--stop-x", type=float, default=0.30, help="m; se detiene si el tag esta mas cerca")
    ap.add_argument("--switch-x", type=float, default=0.55, help="m; cambia a cabeza baja por debajo de esto")
    ap.add_argument("--far-pitch", type=float, default=10.0)
    ap.add_argument("--near-pitch", type=float, default=-7.5)
    ap.add_argument("--lost", type=float, default=1.5, help="s sin ver tag antes de detenerse")
    ap.add_argument("--max-time", type=float, default=25.0)
    ap.add_argument("--log", default="walk_log.csv")
    ap.add_argument("--no-walk", action="store_true")
    args = ap.parse_args()

    motion = ALProxy("ALMotion", args.ip, args.port)
    posture = ALProxy("ALRobotPosture", args.ip, args.port)
    memory = ALProxy("ALMemory", args.ip, args.port)

    motion.wakeUp()
    posture.goToPosture("StandInit", 0.5)
    motion.setAngles("HeadYaw", 0.0, 0.2)
    pitch = args.far_pitch
    motion.setAngles("HeadPitch", math.radians(pitch), 0.2)
    time.sleep(1.0)
    print("[ok] de pie, cabeza a %+.1f" % pitch)

    log = open(args.log, "w")
    log.write("t,id,x,y,cam,pitch,walking\n")

    t0 = time.time()
    t_seen = time.time()
    walking = False
    try:
        while time.time() - t0 < args.max_time:
            try:
                tags = memory.getData("Vision/AprilTags")
                stamp = memory.getData("Vision/AprilTags/stamp")
            except RuntimeError:
                print("[!] No hay datos en ALMemory. Esta corriendo nao_apriltag.py --publish?")
                break
            now = time.time()
            if (now - stamp) < 0.5 and tags:
                t_seen = now
                near = min(tags, key=lambda r: r[1])
                log.write("%.3f,%d,%.4f,%.4f,%d,%.1f,%d\n" % (
                    now - t0, near[0], near[1], near[2], near[6], pitch, int(walking)))
                print("id=%3d x=%+.3f y=%+.3f cam=%s pitch=%+.1f %s" % (
                    near[0], near[1], near[2], "BOT" if near[6] == 1 else "TOP",
                    pitch, "(caminando)" if walking else ""))

                if near[1] < args.stop_x:
                    print("[ok] llegue a x=%.2f m del tag" % near[1])
                    break
                if near[1] <= args.switch_x and pitch != args.near_pitch:
                    pitch = args.near_pitch
                    motion.setAngles("HeadPitch", math.radians(pitch), 0.3)
                    t_seen = now + 0.5             # tolerancia mientras la cabeza baja
                    print("[cabeza] -> %+.1f (cerca)" % pitch)
                if not walking and not args.no_walk:
                    motion.moveToward(args.speed, 0.0, 0.0)
                    walking = True
            elif now - t_seen > args.lost:
                print("[!] %.1f s sin ver tag: me detengo" % (now - t_seen))
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[!] Ctrl+C")
    finally:
        motion.stopMove()
        log.close()
        print("[ok] detenido. Registro en %s" % args.log)
        print("     Para sentarlo: python -c \"from naoqi import ALProxy; "
              "ALProxy('ALMotion','127.0.0.1',9559).rest()\"")


if __name__ == "__main__":
    main()
