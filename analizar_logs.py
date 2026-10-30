#!/usr/bin/env python3
"""Analiza las corridas de logs/ del NAO y sugiere qué parámetro mover.

Uso:
    python analizar_logs.py                # últimas 5 corridas de ./logs
    python analizar_logs.py -n 10          # últimas 10
    python analizar_logs.py --todas        # todas
    python analizar_logs.py --ruta D:\\NAO\\logs

Requiere: pandas y numpy (pip install pandas numpy).
Los umbrales (diccionario U) son ESTIMACIONES hechas con las corridas del 2026-10-30:
recalíbralos cuando tengas corridas que consideres buenas.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

U = dict(
    dt_lento=0.12,        # s, periodo medio del lazo (esperado ~0.10)
    quieto_largo=2.5,     # s sin avanzar al inicio
    rms_y_alerta=0.05,    # rad, cabeceo RMS
    rms_y_malo=0.08,
    max_ang_alerta=0.15,  # rad, inclinación máxima
    max_ang_malo=0.22,
    crec_env=0.015,       # rad/s, crecimiento de la envolvente del cabeceo
    lean=0.04,            # rad, inclinación media
    rms_rumbo=0.02,       # rad
    lateral=0.05,         # m
    sobrepaso=0.03,       # m, error de parada
    odom_err=0.05,        # fracción, odometría vs cinta
    bateria=0.20,
    temp=65.0,
)


def rms(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")


def leer_json(ruta):
    try:
        with open(ruta) as f:
            return json.load(f)
    except Exception:
        return {}


def analizar(carpeta):
    nombre = os.path.basename(carpeta.rstrip("/\\"))
    try:
        df = pd.read_csv(os.path.join(carpeta, "telemetria.csv"))
    except Exception:
        return nombre, None, [("info", "Sin telemetría (la corrida no llegó a caminar o falló al arrancar).")]
    if len(df) < 10:
        return nombre, None, [("info", f"Solo {len(df)} filas de telemetría: corrida demasiado corta.")]

    p = leer_json(os.path.join(carpeta, "params.json"))
    r = leer_json(os.path.join(carpeta, "resultado.json"))
    m, avisos = {}, []

    m["perfil"] = r.get("perfil") or p.get("perfil", "?")
    m["motivo"] = r.get("motivo", "?")
    m["t_s"] = float(df.t.iloc[-1])
    m["dt"] = float(df.dt.mean())
    dist = p.get("distancia_objetivo")

    # Arranque y velocidad real
    mov = df[df.avance > 0.02]
    m["quieto_s"] = float(mov.t.iloc[0] - df.t.iloc[0]) if len(mov) else float("nan")
    cam = df[df.avance > 0.05]
    if len(cam) >= 6 and cam.t.iloc[-1] > cam.t.iloc[0]:
        m["vel"] = float((cam.avance.iloc[-1] - cam.avance.iloc[0]) / (cam.t.iloc[-1] - cam.t.iloc[0]))
    else:
        m["vel"] = float("nan")
    base = cam if len(cam) >= 6 else df

    # Rumbo y lateral
    m["rms_rumbo"] = rms(base.err_theta)
    m["lat_max"] = float(base.lateral.abs().max())

    # Estabilidad del torso (solo si el log trae los ángulos)
    tiene_ang = "ang_y" in df.columns
    if tiene_ang:
        ay, ax = base.ang_y.values, base.ang_x.values
        m["rms_y"], m["rms_x"] = rms(ay), rms(ax)
        m["max_ang"] = float(max(np.abs(ay).max(), np.abs(ax).max()))
        m["lean"] = float(np.mean(ay))
        t = base.t.values
        env = []
        for a in np.arange(t[0], t[-1], 1.0):
            sel = (t >= a) & (t < a + 1.0)
            if sel.sum() > 3:
                env.append((a, float(np.abs(ay[sel]).max())))
        m["crec_env"] = float(np.polyfit(*zip(*env), 1)[0]) if len(env) >= 4 else float("nan")
        if len(t) > 20:
            tt = np.arange(t[0], t[-1], 0.1)
            s = np.interp(tt, t, ay)
            fr = np.fft.rfftfreq(len(tt), 0.1)
            sp = np.abs(np.fft.rfft(s - s.mean()))
            m["f_cabeceo"] = float(fr[sp[1:].argmax() + 1])
        else:
            m["f_cabeceo"] = float("nan")

    # Gobernador
    if "vx_lim" in df.columns:
        vmin = p.get("vx_lim_min", 0.3)
        m["vxlim_min"] = float(df.vx_lim.min())
        m["pct_en_min"] = float((df.vx_lim <= vmin + 0.01).mean() * 100)
        m["vxlim_fin"] = float(df.vx_lim.iloc[-1])
        m["amp_max"] = float(df.amp.max())

    # Parada y odometría
    fin = r.get("avance_final_estable")
    m["final"] = fin
    if fin is not None and dist and m["motivo"] == "meta":
        m["err_parada"] = float(fin - dist)
    med = r.get("medido_cm")
    if med and fin:
        m["err_odom"] = float((med / 100.0 - fin) / max(fin, 1e-6))
    m["bateria"] = r.get("bateria_ini")
    m["temp"] = r.get("temp_max_ini")
    pct_w0 = float((base.w.abs() < 1e-6).mean() * 100)
    m["pct_w0"] = pct_w0

    # ---------------- Diagnóstico ----------------
    def add(nivel, msg):
        avisos.append((nivel, msg))

    if m["motivo"] in ("inclinacion", "tacto") and tiene_ang and m["max_ang"] > U["max_ang_alerta"]:
        add("MAL", f"Terminó por '{m['motivo']}' con inclinación máxima {m['max_ang']:.2f} rad: estaba perdiendo el equilibrio.")
    if m["motivo"] == "error":
        add("MAL", "Terminó por error: revisa eventos.log (traceback).")
    if m["motivo"] == "timeout":
        add("MAL", "Timeout: no llegó a la meta. Revisa velocidad real y si el robot se quedó pisando en el lugar.")
    if m["dt"] > U["dt_lento"]:
        add("ATENCIÓN", f"Lazo lento (dt medio {m['dt']:.3f} s). Revisa wifi/cable; con latencia los PID reaccionan tarde.")
    if m["quieto_s"] == m["quieto_s"] and m["quieto_s"] > U["quieto_largo"]:
        add("ATENCIÓN", f"Tardó {m['quieto_s']:.1f} s en empezar a avanzar. Si no hay balanceo, sube ACELERACION o VELOCIDAD_MIN; si lo hubo, revisa moveInit/postura.")
    if tiene_ang:
        if m["rms_y"] > U["rms_y_malo"] or m["max_ang"] > U["max_ang_malo"]:
            add("MAL", f"Cabeceo alto (RMS {m['rms_y']:.3f}, máx {m['max_ang']:.2f} rad): BAJA velocidad_x o MaxStepFrequency, o usa perfil 'estable'.")
        elif m["rms_y"] > U["rms_y_alerta"] or m["max_ang"] > U["max_ang_alerta"]:
            add("ATENCIÓN", f"Cabeceo moderado (RMS {m['rms_y']:.3f}, máx {m['max_ang']:.2f} rad): cerca del límite; no subas más la velocidad.")
        else:
            add("OK", f"Cabeceo bajo control (RMS {m['rms_y']:.3f}, máx {m['max_ang']:.2f} rad).")
        if m["crec_env"] == m["crec_env"] and m["crec_env"] > U["crec_env"]:
            add("MAL", f"La oscilación CRECE con el tiempo (+{m['crec_env']:.3f} rad/s): señal de que la velocidad está en el límite. Baja velocidad_x ~10% o activa/endurece el gobernador (AMP_ALTA↓, BAJADA↑).")
        if m["lean"] > U["lean"]:
            add("ATENCIÓN", f"Torso inclinado hacia adelante en promedio (+{m['lean']:.3f} rad): baja TorsoWy (prueba -0.02) o reduce velocidad.")
        elif m["lean"] < -U["lean"]:
            add("ATENCIÓN", f"Torso inclinado hacia atrás en promedio ({m['lean']:.3f} rad): sube TorsoWy un poco (+0.02).")
    else:
        add("info", "Este log es de una versión anterior: no trae ángulos del torso, no se puede evaluar estabilidad.")
    if m["rms_rumbo"] > U["rms_rumbo"] or m["lat_max"] > U["lateral"]:
        add("ATENCIÓN", f"Se desvía (rumbo RMS {m['rms_rumbo']:.3f} rad, lateral máx {m['lat_max']*100:.1f} cm): sube KP_RUMBO (+0.2) o MaxStepTheta, o baja ZONA_MUERTA_RUMBO.")
    elif pct_w0 > 90 and m["lat_max"] < 0.02:
        add("OK", f"Rumbo muy bien (lateral máx {m['lat_max']*100:.1f} cm; sin giro el {pct_w0:.0f}% del tiempo).")
    else:
        add("OK", f"Rumbo bien (RMS {m['rms_rumbo']:.3f} rad, lateral máx {m['lat_max']*100:.1f} cm).")
    if "vxlim_min" in m:
        if m["pct_en_min"] > 50:
            add("ATENCIÓN", f"El gobernador pasó {m['pct_en_min']:.0f}% del tiempo en el mínimo: el perfil es demasiado agresivo; baja MaxStepFrequency/velocidad_x o usa 'estable'.")
        elif m["vxlim_min"] >= p.get("velocidad_x", 1) - 0.01 and tiene_ang and m["rms_y"] < U["rms_y_alerta"]:
            add("info", "El gobernador nunca frenó y el torso estuvo tranquilo: hay margen para subir velocidad_x o MaxStepX.")
    if "err_parada" in m:
        if m["err_parada"] > U["sobrepaso"]:
            add("ATENCIÓN", f"Se pasó {m['err_parada']*100:.1f} cm de la meta: baja KP_DIST/velocidad_x o sube TOLERANCIA_M (para frenar antes).")
        elif m["err_parada"] < -U["sobrepaso"]:
            add("ATENCIÓN", f"Quedó {-m['err_parada']*100:.1f} cm corto: sube KP_DIST o VELOCIDAD_MIN, o baja TOLERANCIA_M.")
        else:
            add("OK", f"Parada precisa ({m['err_parada']*100:+.1f} cm).")
    if "err_odom" in m and abs(m["err_odom"]) > U["odom_err"]:
        add("ATENCIÓN", f"La odometría difiere {m['err_odom']*100:+.1f}% de la cinta: el 'avance' del log no es confiable en este piso.")
    if m["bateria"] is not None and m["bateria"] < U["bateria"]:
        add("MAL", f"Batería inicial baja ({m['bateria']*100:.0f}%): carga antes de ajustar parámetros.")
    if m["temp"] is not None and m["temp"] >= U["temp"]:
        add("MAL", f"Piernas calientes al iniciar ({m['temp']:.0f} °C): deja enfriar; los motores pierden rigidez.")
    return nombre, m, avisos


def f(v, fmt="{:.3f}"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def main():
    ap = argparse.ArgumentParser()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    ap.add_argument("--ruta", default=base)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--todas", action="store_true")
    a = ap.parse_args()

    carpetas = sorted(c for c in glob.glob(os.path.join(a.ruta, "*")) if os.path.isdir(c))
    if not a.todas:
        carpetas = carpetas[-a.n:]
    if not carpetas:
        print("No hay corridas en", a.ruta)
        return

    filas = []
    for c in carpetas:
        nombre, m, avisos = analizar(c)
        print("=" * 78)
        print(nombre)
        for nivel, msg in avisos:
            print(f"  [{nivel}] {msg}")
        if m:
            filas.append((nombre, m))

    if filas:
        print("\n" + "=" * 78)
        print(f"{'corrida':<16}{'perfil':<8}{'motivo':<12}{'t(s)':>6}{'v(m/s)':>8}{'rmsY':>7}{'máxAng':>8}"
              f"{'lean':>7}{'latmax':>8}{'final(m)':>9}")
        for nombre, m in filas:
            print(f"{nombre:<16}{str(m['perfil']):<8}{m['motivo']:<12}{f(m['t_s'],'{:.1f}'):>6}{f(m['vel']):>8}"
                  f"{f(m.get('rms_y')):>7}{f(m.get('max_ang')):>8}{f(m.get('lean')):>7}"
                  f"{f(m['lat_max']):>8}{f(m['final']):>9}")
        print("\nRecuerda: cambia UN parámetro por vez y repite 3 corridas antes de decidir.")


if __name__ == "__main__":
    main()
