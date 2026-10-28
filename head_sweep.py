# -*- coding: utf-8 -*-
"""
head_sweep.py -- Barre el angulo de la cabeza (HeadPitch) y reporta que camara ve el tag en cada angulo.

Robot DE PIE y QUIETO; solo se mueve la cabeza. Pon el tag fijo en la pared y corre:
  python head_sweep.py                       # barre de -20 a +25 grados, paso 2.5
  python head_sweep.py --from -10 --to 15 --step 1

Repitelo con el tag a ~17 cm y a ~67 cm de la punta del pie. El angulo bueno es el
que sale "OK" en las dos corridas.

Necesita nao_apriltag.py en la misma carpeta.
"""
from __future__ import print_function, division

import argparse
import math
import time

import numpy as np
from naoqi import ALProxy

import nao_apriltag as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--from", dest="a0", type=float, default=-20.0)
    ap.add_argument("--to", dest="a1", type=float, default=25.0)
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--frames", type=int, default=8, help="frames por camara en cada angulo")
    ap.add_argument("--tag-size", type=float, default=0.15)
    args = ap.parse_args()

    video = ALProxy("ALVideoDevice", args.ip, args.port)
    motion = ALProxy("ALMotion", args.ip, args.port)

    detect = V.make_aruco_detector() if V.aruco_available() else V.BuiltinDetector()
    obj = V.tag_object_points(args.tag_size)
    w, h = V.RESOLUTIONS[2]
    K, D = V.nominal_intrinsics("v5", w, h)

    start = motion.getAngles("HeadPitch", True)[0]
    print("[ok] HeadPitch actual: %+.1f grados" % math.degrees(start))

    V.cleanup_stale(video)
    handles = {}
    for cam in (0, 1):
        handles[cam] = video.subscribeCamera(V.SUB_NAME, cam, 2, V.K_Y_COLORSPACE, 15)

    motion.setStiffnesses("Head", 1.0)
    motion.setAngles("HeadYaw", 0.0, 0.15)

    angles = []
    a = args.a0
    while a <= args.a1 + 1e-6:
        angles.append(a)
        a += args.step

    print("\n pitch |  TOP vistos  x      |  BOT vistos  x      | veredicto")
    print("-------+---------------------+---------------------+----------")
    rows = []
    try:
        for ang in angles:
            motion.setAngles("HeadPitch", math.radians(ang), 0.15)
            time.sleep(0.8)
            cols = []
            for cam in (0, 1):
                hits, xs = 0, []
                for _ in range(args.frames):
                    img = video.getImageRemote(handles[cam])
                    if img is None:
                        continue
                    gray = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0]).copy()
                    T = V.robot_from_camera(motion, V.CAM_NAMES[cam])
                    dets = detect(gray)
                    if dets:
                        hits += 1
                        ok, rv, tv = V.solve_tag_pose(dets[0][1], obj, K, D)
                        if ok:
                            p = T.dot(V.opencv_to_nao_optical(np.array(tv, dtype=np.float64).ravel()))
                            xs.append(p[0])
                cols.append((hits, float(np.median(xs)) if xs else None))
            best = max(c[0] for c in cols)
            verdict = "OK" if best >= args.frames - 1 else ("debil" if best > 0 else "-")
            rows.append((ang, verdict))
            fmt = lambda c: "%2d/%-2d  %s" % (c[0], args.frames, ("%.3f" % c[1]) if c[1] is not None else "  -  ")
            print(" %+5.1f |  %-17s  |  %-17s  | %s" % (ang, fmt(cols[0]), fmt(cols[1]), verdict))
    except KeyboardInterrupt:
        pass
    finally:
        for hd in handles.values():
            try:
                video.unsubscribe(hd)
            except Exception:
                pass
        motion.setAngles("HeadPitch", start, 0.15)
        time.sleep(0.8)

    ok = [r[0] for r in rows if r[1] == "OK"]
    if ok:
        print("\nAngulos OK: de %+.1f a %+.1f grados" % (min(ok), max(ok)))
    else:
        print("\nNingun angulo vio el tag de forma estable.")
    print("[ok] cabeza regresada a %+.1f grados" % math.degrees(start))


if __name__ == "__main__":
    main()
