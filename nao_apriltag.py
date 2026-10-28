# -*- coding: utf-8 -*-
"""
nao_apriltag.py  (v2) -- Deteccion de AprilTags tag36h11 + posicion (x, y) en el NAO.

Pensado para correr DENTRO del robot (Python 2.7, NAOqi 2.x), sin ventana grafica.
Tambien corre desde una laptop con pynaoqi (agregando --ip del robot).

Dos detectores:
  - aruco   : cv2.aruco (OpenCV >= 3.4.? con contrib). Rapido y robusto.
  - builtin : detector propio (umbral + cuadrilateros + codebook 36h11), solo
              usa funciones que existen desde OpenCV 2.4. Es el respaldo.
  --backend auto elige aruco si existe y si no, builtin.

Uso tipico en Tesla (NAO v5):
  python nao_apriltag.py                         # camara de arriba, imprime detecciones
  python nao_apriltag.py --camera both           # alterna arriba/abajo (cerca + lejos)
  python nao_apriltag.py --save-dir dbg          # guarda JPG anotados para revisarlos en la Mac
  python nao_apriltag.py --snap-dir calib        # guarda frames crudos cada 2 s (calibracion)
  nohup python nao_apriltag.py --camera both --publish --quiet > vision.log 2>&1 &

Salida por tag (FRAME_ROBOT de NAOqi: origen en el piso entre los pies,
x adelante, y a la izquierda, z arriba; metros):
  id, x, y, z, dist (camara-tag), bearing (grados, + = izquierda), cam (0=top, 1=bottom)
Con --publish se escribe en ALMemory:
  "Vision/AprilTags"        -> [[id, x, y, z, dist, bearing, cam], ...]
  "Vision/AprilTags/stamp"  -> time.time() del ultimo frame procesado
"""
from __future__ import print_function, division

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("No se encontro cv2 (OpenCV) en este Python.")

from naoqi import ALProxy

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
K_Y_COLORSPACE = 0              # solo luminancia: 1 byte/pixel
FRAME_ROBOT = 2
RESOLUTIONS = {1: (320, 240), 2: (640, 480), 3: (1280, 960)}
CAM_NAMES = {0: "CameraTop", 1: "CameraBottom"}
SUB_NAME = "vision_apriltag"

# Campo de vision nominal (datasheet). Solo para arrancar sin calibracion.
FOV_DEG = {
    "v5": (60.97, 47.64),       # MT9M114
    "v6": (56.3, 43.7),         # OV5640
}

# tag36h11: 587 codigos de 36 bits. Bits = rejilla 6x6 interior, por renglones,
# empezando arriba-izquierda, 1 = blanco. (Extraido de OpenCV DICT_APRILTAG_36h11
# y verificado contra un tag impreso.)
TAG36H11 = [
    0x21a146bab, 0x92d18fe9b, 0x7089014bb, 0x193979e27, 0x44153d3d7, 0x35cd5b8cf,
    0xa10ba56a0, 0x2b874a608, 0xb57fb8d44, 0x4e20b5a64, 0x61d897f2c, 0xab3469ffc,
    0x594ca45c2, 0xfa1c2d2e2, 0x97c24b972, 0x75928624a, 0x1caafe99a, 0x3236ddc16,
    0x884e527d6, 0x5771a8c7e, 0xf52925b81, 0x2e59cc1a1, 0x7fa58ab31, 0xeb43384f9,
    0x3913d5345, 0xd79b1a3b5, 0x0dfbe768d, 0x5cd7a031d, 0xb9c8f7a4b, 0x46987e06b,
    0x6e0c5d527, 0x5bc567bb4, 0x1e6302bc2, 0x8ac7d046a, 0xa65fe2326, 0x5400eb616,
    0xf7a066836, 0x2dd0afdf6, 0x3b9cb60b1, 0xad6a1caf9, 0x93b62efb5, 0xbb210d93d,
    0xdee55094b, 0xa0cd1479b, 0x2a23abb27, 0x4f67f4b5f, 0x58a815818, 0x743427d64,
    0x1dec40874, 0x892ab3a02, 0x38d5742ee, 0x0c43567a1, 0x5db311cb1, 0xc957ca1f9,
    0xa22457823, 0x29ace98f3, 0xccc52c528, 0xda8b3578c, 0xd5b4b9e36, 0xac9cdd3ce,
    0xcbdaa8391, 0x1986256b1, 0xbae6cc889, 0xdf91b10e5, 0xa1a9d5a75, 0x2b651aa9d,
    0x4d5c2e19d, 0x5b163736b, 0xf753d60a8, 0x5e84b37d2, 0xfce43a9f2, 0xbb926d1a6,
    0xed8d236e5, 0x6af846f37, 0xea56a5138, 0x5d497146c, 0x470360eca, 0x8c4bcd67a,
    0x295f3901e, 0xf6b577e97, 0x520602f5a, 0xb5119492e, 0x48e299ab7, 0xb79f5737a,
    0xa814c683e, 0x2da900583, 0xd4aba4797, 0x982a0e65c, 0xf480ccc15, 0x0b20ea732,
    0xa9506710a, 0x0486e2ee1, 0xb0a972615, 0xe73f3a257, 0x17e205b98, 0xec9ab2b24,
    0xd9339e076, 0xeb8874f99, 0x9420a55cc, 0xe364ff992, 0x8a4cb8f0a, 0xf72aee766,
    0x73cc6b569, 0xc99ca6a59, 0x366a0c535, 0x509e51943, 0xba6d227d1, 0xd2acf163b,
    0xd989fef68, 0x641707bb2, 0xf9bd312ab, 0x77739c207, 0x68bf8d1bf, 0xdedd84a6e,
    0x42eaec49b, 0x302b68d53, 0x4ebb930ce, 0x439c5d9b5, 0xe1fc9228d, 0xf8a6a8d4a,
    0xb53902a18, 0xb163475ee, 0xad2f5af29, 0xaab5dc850, 0xb3ec25786, 0x4799c4405,
    0x8bc68dc04, 0x702f15dde, 0xb42293da9, 0x1acad0382, 0x813363971, 0x0e692ad70,
    0x8ad05c5b9, 0x85e654c1b, 0x3d0ac2b3b, 0x4bbfa0552, 0x722dde306, 0x369d1acd2,
    0xa2d994e60, 0x5fad9139d, 0x57535ae96, 0x5f5ef1d46, 0xebc3e6acc, 0x0a99c835d,
    0x5463e49a5, 0x25676893a, 0xabed37064, 0x59bdded94, 0x64fa04c09, 0x250e44e75,
    0xe58a374bc, 0x24caa5741, 0xe42cf6588, 0x6ea238538, 0xf75813025, 0x0ceb33bbd,
    0x786ae8670, 0xf0de8c969, 0x26c81efe6, 0x69f489086, 0x3fbf92a4d, 0xd6555a760,
    0x9ccae236d, 0xb256c16a3, 0xd78ac7221, 0x2dc1fe262, 0xaf2051090, 0x932c08afe,
    0x53feca775, 0x5d292497a, 0x2f101a22b, 0xc6ad16b14, 0xd2eca4b20, 0xa6fa603d4,
    0x3057c3b69, 0xe817b54d2, 0x40abfc3e0, 0xf53ad16e7, 0x36f3a06e8, 0x27782e418,
    0x3e9062146, 0x01fd7ad4e, 0xc8765bc64, 0xad3d8215f, 0x96317a27e, 0x05f813ca6,
    0x0e7adc22e, 0xbf52c4a50, 0x724f6ae2e, 0x540fd7ea8, 0x9c962f741, 0x264042b01,
    0x197cb7754, 0x1f2bb850d, 0xc87d6c2ad, 0xdc8d2dede, 0xcb6b01738, 0xbe4210368,
    0xbb4622a77, 0x733226ef8, 0x4933ade39, 0x1f1c896c7, 0x897665aba, 0x92f46ba12,
    0x297981767, 0xad702cf8d, 0x473198b90, 0xec2e8e06b, 0x36a1d6980, 0xa5ce8650f,
    0x637870348, 0xf3ab41d46, 0xe6032a8b5, 0x4c11eec73, 0x261de50d6, 0x296616e32,
    0x37cff1a0a, 0xc457847ab, 0x776e52ff9, 0x980819bb9, 0x50cb2b5d8, 0x8b6918063,
    0x12d9c2ce9, 0x158c2e738, 0x6b84015ea, 0x67695db9f, 0x3e72745c1, 0x01ca1a571,
    0x22e7036f5, 0xd5dcedae1, 0x809532c5b, 0xf7e4e9d54, 0x59d84f9cb, 0xf42ed0851,
    0x5db4a91fa, 0x0d58e5b1f, 0x69fb9fa43, 0x29ba51f60, 0x587c67427, 0xd8b9021ed,
    0xbb594449e, 0xaa231f35a, 0x0d95943f4, 0xdec461257, 0x05a16930d, 0xe2512eb58,
    0xab48e2dc0, 0x0cb68494f, 0x927a6b5b1, 0xbb90f3450, 0x632e8fad0, 0xc2eeaf2a7,
    0x1eb2af39d, 0x511548c52, 0x4693d5de4, 0x888e9161d, 0x68836ff4f, 0x51bd2e441,
    0x7a0ac40e9, 0x8bb762b4a, 0x1caf83a1a, 0xc2700bbbd, 0xfb2ed09aa, 0xf8520294e,
    0x7aa104ca6, 0x6a96d2bcb, 0x28f2ec75e, 0x607065eb4, 0x7ad9b3a8c, 0x3572d6cbf,
    0xd426efb8f, 0x64ee4dbaa, 0x983d7a8f1, 0x8ca31f499, 0x025457b46, 0xd53ae4149,
    0x9d3c64396, 0x795332ee5, 0x70884d664, 0x1dd3528d8, 0xd8e51d246, 0xd6029ca38,
    0xd94ab610c, 0x766bb8880, 0xfd0c6b319, 0x3bff8b9d1, 0xe786f4d4e, 0xd4e6b9b79,
    0xcfa8bb369, 0x20a4366c6, 0x5cefb9536, 0x0f06b6605, 0xbb51d25e7, 0xf7e262641,
    0x81fda9855, 0x2241dbc59, 0x56fb4b280, 0xeda64c6d0, 0xce7b080f4, 0x6192ee7e9,
    0x943ff5124, 0x88b3beadf, 0x9b27e58d0, 0xbe0b45fd2, 0x69fb3a72a, 0x61383a832,
    0x6688639b3, 0x54d01c992, 0x90b10c02a, 0xf667fc6d1, 0x7acd75813, 0x3abd07f61,
    0x543012473, 0x067794d58, 0xa855b1201, 0x9cb4d7fdb, 0x31a193e6a, 0x3da3b76d4,
    0x4153f1e04, 0x5544335b5, 0x92f61249d, 0x269a35f8a, 0x042677896, 0x9487cdc6f,
    0x4ea7e7d9c, 0x2e367345b, 0x3a5e8f304, 0x76d9faa77, 0x64e4ea1df, 0x12a7b434d,
    0x4973d9553, 0xcdb418975, 0x8bda76d2e, 0xf54a4dec8, 0x811616f8a, 0x4036f4e6b,
    0x2736282a6, 0xa413d229f, 0x901fe4c17, 0x9abfce56e, 0x2d325b9ef, 0xd9901f6a9,
    0x3aa64fade, 0x65288d60b, 0x0a53e960b, 0xe9e09c3e4, 0x4d9b7863d, 0xd98bb4833,
    0x34b181ed7, 0x5b373f490, 0x8572b7b91, 0x2c5d056a4, 0xe4671296d, 0x2a08d6f78,
    0x452949c61, 0x2501ec5b7, 0xb8f54779c, 0xa3b7ea4b3, 0x443d6b9b4, 0xbe2dcb848,
    0xb51fc47ec, 0x6f38cf988, 0x90f2fe934, 0x07fd540ab, 0xee53c3d08, 0xf7098ebca,
    0x65467c309, 0xae6fd797e, 0x8bc4739e9, 0x971930fe4, 0x9a752df57, 0xbd47fe53e,
    0x6f1530928, 0x5de1cc489, 0x111ce3dcd, 0xa4c7e157c, 0x552691112, 0x9d6511805,
    0x6663c1cb3, 0xbfc594002, 0x3a3500387, 0x76d4f679c, 0x446f2667b, 0x17ae59e97,
    0x36132d63e, 0x8e07492bf, 0x188acddf1, 0xed9052d02, 0x5b6bd01f5, 0x84a4d070e,
    0xc9b2a3af4, 0xab18852fc, 0x9de918390, 0x3576ee9e0, 0xe8d00d210, 0x709b5a95c,
    0xa9022a314, 0xa03ea2b9c, 0x1e589ec5e, 0x74470eedd, 0x36bd44159, 0xb7f8d2516,
    0xdcde4bfa5, 0x9eda99c30, 0x8ffb98a8b, 0x9ac777e71, 0x8d0a53cb3, 0xc9ed1dced,
    0xc238535ea, 0xee4aa6437, 0xcd8345a1e, 0x5a19e05ae, 0x96d780d25, 0xe2c885185,
    0xc5314f8db, 0xa47c37ee1, 0x1d82c6f8c, 0x325c5c6c8, 0x5d9d1bae0, 0x9549e075e,
    0x8a6845c15, 0xadf9f10fe, 0x95cef30b0, 0x71933b20d, 0x1f0de3a6f, 0x596f5bba6,
    0xcd61d9f34, 0xa478ad328, 0xdade4012f, 0xd15764370, 0x3a2bf5d22, 0x2303e348d,
    0x41b9bf9a9, 0x5a4e93ee3, 0x462b44512, 0x4e01719c1, 0x73a0d9137, 0x932b416dd,
    0xff4b615a7, 0x2c880f1ac, 0x6f3bb6aa1, 0x476d21ae6, 0xf9c290688, 0x768111210,
    0x623458f55, 0xf31919e1f, 0xef47a7dd1, 0xc6c777808, 0xeec96f096, 0x9b3c9bdaf,
    0x6fd432573, 0x60e7d0aa9, 0x0651cda34, 0xe2acc04e7, 0x9460d1a22, 0x82b79b177,
    0x4eb189f4b, 0xcbf9c3b1e, 0x6f6c64841, 0xe54e50a2e, 0x632f30f23, 0xb3bb3de6c,
    0x6331f5383, 0xbca1dafe5, 0xd1816cab9, 0x0d8407aca, 0x794e1d7e5, 0x92de65442,
    0x651497949, 0x33f751712, 0x3c9ce9195, 0x7ecb81d7a, 0xede140178, 0xc09b0ee2c,
    0x14e61f774, 0x6d336649a, 0x865dd0880, 0xf7214bfb8, 0xd8b9c2e22, 0x99a70737f,
    0xc0ecf90c9, 0xbb8b26e01, 0x05678fcac, 0x33b483834, 0xeb9d84b83, 0x7ca277ea2,
    0x02af5f66b, 0x4a678c122, 0x121592bf3, 0xcc82b48c9, 0x77cbe90fc, 0x5ef7d7931,
    0x117947995, 0x344d5f970, 0xdf2f12e72, 0x114277b37, 0x0b8fa73d5, 0x36e16f423,
    0x057402b7e, 0xc342423a7, 0x758c34dab, 0x67dc61614, 0x29b222068, 0x7101af0fa,
    0x54d22bc75, 0x908ba3af9, 0xeb68c9b51, 0x2605d442c, 0x6ab88b7f4, 0x207c7657d,
    0x377940d20, 0xac2eb77b6, 0x3e87318ec, 0x47afea1aa, 0x46de35b74, 0x48a36ba10,
    0xf0711536b, 0x963f87ed4, 0xd83ab9965, 0x7b2eb5a0f, 0xb6adfaf32, 0x87e07a4ea,
    0x7b4518f4e, 0x9064a3738, 0x3c242ca12, 0x96b25c4a4, 0x8b01d83ac, 0xe51cceb25,
    0x480f58db0, 0xbae295867, 0x89fc0fba0, 0xdd06475e9, 0xf7fe46b4c, 0x43cef026d,
    0x6393585a2, 0x87b404456, 0x4fa47cbea, 0x7619f955a, 0x51b9a1330, 0x1533637c7,
    0x7587a972a, 0x60de11622, 0x7d4ce68ae, 0xaf135e81c, 0x31c9b647c, 0xac43d8155,
    0xd853e1cfb, 0x4fcdc5cf0, 0xee696fd2d, 0xb71618475, 0xb9b4b8781, 0x328579505,
    0x95e947771, 0x9aa0302e2, 0xec1828d17, 0xbc5079635, 0x8785c1b79, 0xd37aadb7a,
    0x817f02ce0, 0xfd63dfb0a, 0x510ac1af2, 0x3d1edf8d3, 0xf4b73a402, 0x37d789e9c,
    0x8f11c77ca, 0xc4bd036be, 0x7dfa995c9, 0x7b14e8251, 0xca070e7f4, 0x5c22dbcd5,
    0xa0db726c8, 0x6b7776de2, 0x80f58e748, 0xe1ac1ff4a, 0x1bbaea85c, 0x26ac557fc,
    0xe5d1ef357, 0x1f29d4e10, 0x459fbb15b, 0xd4b049559, 0x6cb70acf8, 0xf8617a518,
    0xc391dac34, 0x51235e2cc, 0x74407c4fe, 0xdf1609a24, 0xced27dc17,
]


# --------------------------------------------------------------------------
# Intrinsecos
# --------------------------------------------------------------------------
def nominal_intrinsics(robot, w, h):
    hfov, vfov = [math.radians(a) for a in FOV_DEG[robot]]
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    fy = (h / 2.0) / math.tan(vfov / 2.0)
    K = np.array([[fx, 0.0, w / 2.0],
                  [0.0, fy, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64)


def load_intrinsics(path, w, h):
    data = np.load(path)
    K = data["K"].astype(np.float64)
    D = data["D"].astype(np.float64).ravel()
    cw, ch = int(data["width"]), int(data["height"])
    if (cw, ch) != (w, h):
        sx, sy = w / float(cw), h / float(ch)
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        print("[calib] %s: %dx%d reescalada a %dx%d" % (path, cw, ch, w, h))
    return K, D


# --------------------------------------------------------------------------
# Detector 1: cv2.aruco
# --------------------------------------------------------------------------
def aruco_available():
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_APRILTAG_36h11")


def make_aruco_detector():
    ar = cv2.aruco
    dictionary = ar.getPredefinedDictionary(ar.DICT_APRILTAG_36h11)
    if hasattr(ar, "ArucoDetector"):            # OpenCV >= 4.7
        params = ar.DetectorParameters()
    else:
        params = ar.DetectorParameters_create()
    params.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX

    if hasattr(ar, "ArucoDetector"):
        det = ar.ArucoDetector(dictionary, params)

        def run(gray):
            corners, ids, _ = det.detectMarkers(gray)
            return _pack(corners, ids)
    else:
        def run(gray):
            corners, ids, _ = ar.detectMarkers(gray, dictionary, parameters=params)
            return _pack(corners, ids)
    return run


def _pack(corners, ids):
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, np.array(ids).flatten()):
        out.append((int(i), np.array(c, dtype=np.float32).reshape(4, 2)))
    return out


# --------------------------------------------------------------------------
# Detector 2: builtin (OpenCV 2.4+ y numpy)
# --------------------------------------------------------------------------
class BuiltinDetector(object):
    CELL = 6                        # pixeles por celda en la imagen rectificada
    GRID = 8                        # borde negro (1) + datos (6) + borde negro (1)

    def __init__(self, max_hamming=3, min_side=12):
        bits = np.zeros((len(TAG36H11), 36), dtype=np.uint8)
        for i, code in enumerate(TAG36H11):
            for b in range(36):
                bits[i, b] = (code >> (35 - b)) & 1
        self.codebook = bits
        self.max_hamming = max_hamming
        self.min_side = min_side
        n = self.GRID * self.CELL
        self.dst = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32)
        self.crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)

    def __call__(self, gray):
        h, w = gray.shape[:2]
        block = max(3, (min(w, h) // 20) | 1)       # impar, ~1/20 de la imagen
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, block, 7)
        res = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[-2]                          # 2.4 / 3.x / 4.x

        min_perim = 4 * self.min_side
        best = {}
        for cnt in contours:
            if len(cnt) < 4:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < min_perim:
                continue
            approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            # lado minimo
            sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
            if min(sides) < self.min_side:
                continue
            # orden horario en pantalla (y hacia abajo): area con signo > 0
            x, y = quad[:, 0], quad[:, 1]
            area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            if area < 0:
                quad = quad[::-1].copy()

            dec = self._decode(gray, quad)
            if dec is None:
                continue
            tag_id, rot, ham = dec
            quad = np.roll(quad, -rot, axis=0)
            prev = best.get(tag_id)
            if prev is None or ham < prev[0] or (ham == prev[0] and abs(area) > prev[1]):
                best[tag_id] = (ham, abs(area), quad)

        out = []
        for tag_id, (_, _, quad) in best.items():
            refined = quad.reshape(-1, 1, 2).copy()
            try:
                cv2.cornerSubPix(gray, refined, (3, 3), (-1, -1), self.crit)
            except cv2.error:
                pass
            out.append((tag_id, refined.reshape(4, 2)))
        return out

    def _decode(self, gray, quad):
        M = cv2.getPerspectiveTransform(quad, self.dst)
        n = self.GRID * self.CELL
        warp = cv2.warpPerspective(gray, M, (n, n), flags=cv2.INTER_LINEAR)
        c = self.CELL
        # promedio del centro de cada celda (zona de c/2 x c/2)
        q = c // 4
        cells = np.zeros((self.GRID, self.GRID), dtype=np.float32)
        for r in range(self.GRID):
            for k in range(self.GRID):
                patch = warp[r * c + q:(r + 1) * c - q, k * c + q:(k + 1) * c - q]
                cells[r, k] = patch.mean()
        lo, hi = cells.min(), cells.max()
        if hi - lo < 30:
            return None
        thr = (lo + hi) / 2.0
        bits = (cells > thr).astype(np.uint8)

        border = np.concatenate([bits[0, :], bits[-1, :], bits[1:-1, 0], bits[1:-1, -1]])
        if border.sum() > 2:                         # el borde debe ser negro
            return None

        inner = bits[1:-1, 1:-1]
        best = None
        for rot in range(4):
            cand = np.rot90(inner, rot).flatten()
            d = (self.codebook != cand).sum(axis=1)
            idx = int(d.argmin())
            ham = int(d[idx])
            if ham <= self.max_hamming and (best is None or ham < best[2]):
                best = (idx, rot, ham)
        return best


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------
def tag_object_points(size):
    # Orden de esquinas: sup-izq, sup-der, inf-der, inf-izq (marco del tag, y hacia arriba)
    s = size / 2.0
    return np.array([[-s, s, 0.0],
                     [s, s, 0.0],
                     [s, -s, 0.0],
                     [-s, -s, 0.0]], dtype=np.float64)


def solve_tag_pose(corners, obj_pts, K, D):
    img_pts = np.ascontiguousarray(corners.reshape(4, 1, 2).astype(np.float64))
    obj = np.ascontiguousarray(obj_pts.reshape(4, 1, 3))
    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        ret = cv2.solvePnP(obj, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    else:
        ret = cv2.solvePnP(obj, img_pts, K, D)      # OpenCV 2.4/3.x: iterativo
    ok, rvec, tvec = ret[0], ret[1], ret[2]
    return bool(ok), rvec, tvec


def robot_from_camera(motion, cam_name):
    # 16 floats, matriz 4x4 por renglones, convencion NAO (x adelante, y izq, z arriba)
    return np.array(motion.getTransform(cam_name, FRAME_ROBOT, True), dtype=np.float64).reshape(4, 4)


def opencv_to_nao_optical(p):
    # OpenCV: z adelante, x derecha, y abajo  ->  NAO: x adelante, y izquierda, z arriba
    return np.array([p[2], -p[0], -p[1], 1.0], dtype=np.float64)


# --------------------------------------------------------------------------
# Dibujo (solo para --save-dir); funciones existentes desde OpenCV 2.4
# --------------------------------------------------------------------------
def annotate(gray, dets, results):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    by_id = dict((r["id"], r) for r in results)
    for tag_id, c in dets:
        pts = [tuple(int(round(v)) for v in p) for p in c]
        for i in range(4):
            cv2.line(vis, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        cv2.circle(vis, pts[0], 4, (0, 0, 255), -1)          # esquina sup-izq del tag
        r = by_id.get(tag_id)
        label = "id%d" % tag_id
        if r is not None:
            label += " x%.2f y%.2f" % (r["x"], r["y"])
        cv2.putText(vis, label, (pts[0][0], max(12, pts[0][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return vis


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="AprilTag 36h11 en NAO (v2)")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--robot", choices=["v5", "v6"], default="v5")
    ap.add_argument("--camera", choices=["top", "bottom", "both"], default="top")
    ap.add_argument("--res", type=int, choices=[1, 2, 3], default=2,
                    help="1=320x240, 2=640x480 (default), 3=1280x960")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--tag-size", type=float, default=0.15,
                    help="lado del CUADRO NEGRO en metros (Copa NAO: 0.15)")
    ap.add_argument("--backend", choices=["auto", "aruco", "builtin"], default="auto")
    ap.add_argument("--calib-top", default=None, help=".npz de calibrate_camera.py (CameraTop)")
    ap.add_argument("--calib-bottom", default=None, help=".npz de calibrate_camera.py (CameraBottom)")
    ap.add_argument("--head-pitch", type=float, default=None,
                    help="grados; inclina la cabeza (+ = abajo). Endurece SOLO la cabeza.")
    ap.add_argument("--publish", action="store_true", help="publica en ALMemory")
    ap.add_argument("--quiet", action="store_true", help="solo imprime el estado cada 5 s")
    ap.add_argument("--save-dir", default=None, help="guarda JPG anotados aqui")
    ap.add_argument("--save-every", type=float, default=1.0, help="segundos entre JPG anotados")
    ap.add_argument("--snap-dir", default=None, help="guarda PNG crudos aqui (calibracion)")
    ap.add_argument("--snap-every", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0, help="segundos; 0 = infinito")
    return ap.parse_args()


def cleanup_stale(video):
    # Si una corrida anterior murio sin desuscribirse, NAOqi deja el handle vivo
    try:
        for s in video.getSubscribers():
            if s.startswith(SUB_NAME):
                video.unsubscribe(s)
                print("[ok] limpie suscripcion vieja", s)
    except Exception:
        pass


def main():
    args = parse_args()

    video = ALProxy("ALVideoDevice", args.ip, args.port)
    motion = ALProxy("ALMotion", args.ip, args.port)
    memory = ALProxy("ALMemory", args.ip, args.port) if args.publish else None

    # Detector
    if args.backend == "aruco" or (args.backend == "auto" and aruco_available()):
        if not aruco_available():
            sys.exit("Pediste --backend aruco pero este OpenCV (%s) no lo tiene." % cv2.__version__)
        detect, backend = make_aruco_detector(), "aruco"
    else:
        detect, backend = BuiltinDetector(), "builtin"

    w, h = RESOLUTIONS[args.res]
    cams = [0, 1] if args.camera == "both" else [0 if args.camera == "top" else 1]

    # Cabeza (opcional)
    if args.head_pitch is not None:
        motion.setStiffnesses("Head", 1.0)
        motion.setAngles("HeadYaw", 0.0, 0.2)
        motion.setAngles("HeadPitch", math.radians(args.head_pitch), 0.2)
        time.sleep(1.0)

    cleanup_stale(video)
    streams = []
    for cam in cams:
        calib = args.calib_top if cam == 0 else args.calib_bottom
        if calib:
            K, D = load_intrinsics(calib, w, h)
        else:
            K, D = nominal_intrinsics(args.robot, w, h)
        handle = video.subscribeCamera(SUB_NAME, cam, args.res, K_Y_COLORSPACE, args.fps)
        streams.append({"cam": cam, "name": CAM_NAMES[cam], "handle": handle, "K": K, "D": D})
        print("[ok] %s %dx%d %s" % (CAM_NAMES[cam], w, h,
                                   "calibrada" if calib else "(intrinsecos nominales)"))
    print("[ok] OpenCV %s | backend=%s | tag=%.3f m" % (cv2.__version__, backend, args.tag_size))

    for d in (args.save_dir, args.snap_dir):
        if d and not os.path.isdir(d):
            os.makedirs(d)

    obj_pts = tag_object_points(args.tag_size)
    latest = {}                     # cam -> lista de resultados del ultimo frame
    t0 = time.time()
    t_status = t0
    t_save = 0.0
    t_snap = 0.0
    frames = 0
    n_img = 0
    i = 0
    try:
        while True:
            s = streams[i % len(streams)]
            i += 1
            img = video.getImageRemote(s["handle"])
            if img is None:
                time.sleep(0.01)
                continue
            iw, ih = img[0], img[1]
            gray = np.frombuffer(img[6], dtype=np.uint8).reshape(ih, iw).copy()
            T = robot_from_camera(motion, s["name"])
            now = time.time()

            dets = detect(gray)
            results = []
            for tag_id, c in dets:
                ok, rvec, tvec = solve_tag_pose(c, obj_pts, s["K"], s["D"])
                if not ok:
                    continue
                p_cv = np.array(tvec, dtype=np.float64).ravel()
                p = T.dot(opencv_to_nao_optical(p_cv))
                results.append({
                    "id": tag_id, "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                    "dist": float(np.linalg.norm(p_cv)),
                    "bearing": math.degrees(math.atan2(p[1], p[0])),
                    "cam": s["cam"],
                })
            latest[s["cam"]] = results
            frames += 1

            if not args.quiet:
                for r in results:
                    print("%s id=%3d  fila=%d col=%d  x=%+.3f  y=%+.3f  z=%+.3f  dist=%.3f  bearing=%+6.1f"
                          % ("TOP" if r["cam"] == 0 else "BOT", r["id"], r["id"] // 8, r["id"] % 8,
                             r["x"], r["y"], r["z"], r["dist"], r["bearing"]))

            if memory is not None:
                merged = {}
                for cam in sorted(latest.keys(), reverse=True):     # abajo primero (mas cerca)
                    for r in latest[cam]:
                        merged.setdefault(r["id"], r)
                payload = [[r["id"], r["x"], r["y"], r["z"], r["dist"], r["bearing"], r["cam"]]
                           for r in merged.values()]
                memory.insertData("Vision/AprilTags", payload)
                memory.insertData("Vision/AprilTags/stamp", now)

            if args.save_dir and now - t_save >= args.save_every:
                t_save = now
                fn = os.path.join(args.save_dir, "%s_%04d.jpg" % (s["name"], n_img))
                cv2.imwrite(fn, annotate(gray, dets, results))
                n_img += 1
            if args.snap_dir and now - t_snap >= args.snap_every:
                t_snap = now
                fn = os.path.join(args.snap_dir, "%s_%04d.png" % (s["name"], n_img))
                cv2.imwrite(fn, gray)
                n_img += 1
                print("[snap]", fn)

            period = 5.0 if args.quiet else 1.0
            if now - t_status >= period:
                print("[estado] %.1f Hz | tags vistos: %s" % (
                    frames / (now - t_status),
                    sorted(set(r["id"] for res in latest.values() for r in res)) or "ninguno"))
                sys.stdout.flush()
                frames = 0
                t_status = now

            if args.duration and now - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            try:
                video.unsubscribe(s["handle"])
            except Exception:
                pass
        print("[ok] camaras liberadas")


if __name__ == "__main__":
    main()
