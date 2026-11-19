"""
Prova di MediaPipe prima del programma completo: controlla che l'installazione funzioni e misura,
sul tuo portatile e con le tue camere, velocità e qualità dei tre modelli (lite, full, heavy).

Cosa fa:
  1. scarica i modelli in data/modelli (solo la prima volta);
  2. dopo un conto alla rovescia di 10 secondi registra 60 coppie dalle camere, una ogni 0,1 s
     (6 secondi): in quel tempo una persona sta nel cerchio e muove le braccia lentamente;
  3. per ogni modello trova le 33 articolazioni in tutte le immagini, misura il tempo, triangola
     le articolazioni visibili in entrambe le camere e calcola:
       - millisecondi per immagine e coppie al secondo possibili (due immagini per coppia);
       - in quante immagini la persona è stata trovata;
       - errore di riproiezione mediano (px): quanto i due raggi di ogni articolazione si incontrano;
       - lunghezza di braccio, avambraccio, coscia e gamba e quanto varia nel tempo (un osso
         vero non cambia lunghezza: più varia, più la ricostruzione è rumorosa);
  4. salva un'immagine con le articolazioni disegnate per ciascun modello.

Uso:   python prova_mediapipe.py
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from acquisizione import AcquisizioneStereo
from triangolazione import Triangolatore

CARTELLA_MODELLI = os.path.join("data", "modelli")
CARTELLA_USCITA = os.path.join("data", "prova_mediapipe")
URL_MODELLO = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{0}/"
               "float16/latest/pose_landmarker_{0}.task")
MODELLI = ("lite", "full", "heavy")
COPPIE = 60
PASSO_S = 0.1
VISIBILITA_MIN = 0.5

# Ossa usate per il controllo di stabilità: (nome, articolazione, articolazione) con la numerazione di MediaPipe
OSSA = [("braccio sinistro", 11, 13), ("braccio destro", 12, 14), ("avambraccio sinistro", 13, 15),
        ("avambraccio destro", 14, 16), ("coscia sinistra", 23, 25), ("coscia destra", 24, 26),
        ("gamba sinistra", 25, 27), ("gamba destra", 26, 28)]
# Segmenti da disegnare
SCHELETRO = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
             (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
             (0, 11), (0, 12)]


def percorso_modello(nome):
    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso = os.path.join(CARTELLA_MODELLI, f"pose_landmarker_{nome}.task")
    if not os.path.exists(percorso):
        print(f"Scarico il modello {nome}...")
        urllib.request.urlretrieve(URL_MODELLO.format(nome), percorso)
    return percorso


def crea_rilevatore(nome):
    """Un PoseLandmarker in modalità VIDEO: segue la persona da un fotogramma al successivo."""
    opzioni = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=percorso_modello(nome)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1)
    return mp.tasks.vision.PoseLandmarker.create_from_options(opzioni)


def articolazioni(rilevatore, immagine_bgr, istante_ms):
    """Pixel (33, 2) e visibilità (33,) della prima persona trovata, oppure None."""
    rgb = np.ascontiguousarray(cv2.cvtColor(immagine_bgr, cv2.COLOR_BGR2RGB))
    risultato = rilevatore.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), istante_ms)
    if not risultato.pose_landmarks:
        return None
    h, w = immagine_bgr.shape[:2]
    punti = risultato.pose_landmarks[0]
    pixel = np.array([[p.x * w, p.y * h] for p in punti])
    visibilita = np.array([p.visibility for p in punti])
    return pixel, visibilita


def disegna(immagine, pixel, visibilita):
    for a, b in SCHELETRO:
        if visibilita[a] > VISIBILITA_MIN and visibilita[b] > VISIBILITA_MIN:
            cv2.line(immagine, tuple(pixel[a].astype(int)), tuple(pixel[b].astype(int)), (0, 255, 0), 3, cv2.LINE_AA)
    for (x, y), v in zip(pixel, visibilita):
        cv2.circle(immagine, (int(x), int(y)), 5, (0, 0, 255) if v > VISIBILITA_MIN else (128, 128, 128), -1)


def registra_coppie():
    acq = AcquisizioneStereo()
    try:
        time.sleep(2)
        for s in range(10, 0, -1):
            print(f"  la registrazione parte tra {s}... (mettiti nel cerchio)")
            time.sleep(1)
        print(f"  REGISTRO {COPPIE} coppie in {COPPIE * PASSO_S:.0f} s: muovi le braccia lentamente")
        coppie, prossimo = [], time.perf_counter()
        while len(coppie) < COPPIE:
            prossimo += PASSO_S
            coppia = acq.prossima_coppia(timeout_s=2)
            if coppia is not None:
                coppie.append((coppia.cam1.immagine.copy(), coppia.cam2.immagine.copy(),
                               coppia.cam1.t_scena, coppia.cam2.t_scena))
            time.sleep(max(0.0, prossimo - time.perf_counter()))
        print("  fatto.")
        return coppie
    finally:
        acq.chiudi()


def prova_modello(nome, coppie, tri):
    rilevatori = [crea_rilevatore(nome), crea_rilevatore(nome)]
    tempi, trovate, errori = [], [0, 0], []
    lunghezze = {o[0]: [] for o in OSSA}
    ultimo_ms = [-1, -1]
    t0 = [coppie[0][2], coppie[0][3]]
    esempio = None
    try:
        for im1, im2, t1, t2 in coppie:
            risultati = []
            for c, (im, t) in enumerate(((im1, t1), (im2, t2))):
                istante = max(int((t - t0[c]) * 1000), ultimo_ms[c] + 1)    # deve crescere sempre
                ultimo_ms[c] = istante
                inizio = time.perf_counter()
                r = articolazioni(rilevatori[c], im, istante)
                tempi.append((time.perf_counter() - inizio) * 1000)
                trovate[c] += r is not None
                risultati.append(r)
            if any(r is None for r in risultati):
                continue
            (p1, v1), (p2, v2) = risultati
            esempio = (im1, im2, p1, v1, p2, v2)
            buone = (v1 > VISIBILITA_MIN) & (v2 > VISIBILITA_MIN)
            if not buone.any():
                continue
            X = np.full((33, 3), np.nan)
            X[buone], e = tri.triangola(p1[buone], p2[buone])
            errori.extend(e.ravel())
            for osso, a, b in OSSA:
                if buone[a] and buone[b]:
                    lunghezze[osso].append(np.linalg.norm(X[a] - X[b]))
    finally:
        for r in rilevatori:
            r.close()
    if esempio:
        im1, im2, p1, v1, p2, v2 = esempio
        im1, im2 = im1.copy(), im2.copy()
        disegna(im1, p1, v1)
        disegna(im2, p2, v2)
        os.makedirs(CARTELLA_USCITA, exist_ok=True)
        cv2.imwrite(os.path.join(CARTELLA_USCITA, f"articolazioni_{nome}.jpg"), cv2.hconcat([im1, im2]))
    return np.array(tempi), trovate, np.array(errori), lunghezze


def main():
    print(f"MediaPipe {mp.__version__}")
    for nome in MODELLI:
        percorso_modello(nome)
    tri = Triangolatore()
    print("Registrazione delle coppie di prova:")
    coppie = registra_coppie()

    righe = [f"Prova di MediaPipe {mp.__version__} - {time.strftime('%Y-%m-%d %H:%M')} - {len(coppie)} coppie", ""]
    for nome in MODELLI:
        print(f"Elaboro con il modello {nome}...")
        tempi, trovate, errori, lunghezze = prova_modello(nome, coppie, tri)
        ms = np.median(tempi[2:]) if len(tempi) > 2 else np.nan     # i primi includono l'avvio
        righe += [f"Modello {nome}:",
                  f"  tempo per immagine: mediana {ms:.0f} ms, 95% sotto {np.percentile(tempi[2:], 95):.0f} ms"
                  f"  -> circa {1000 / (2 * ms):.0f} coppie al secondo elaborando le due immagini una dopo l'altra",
                  f"  persona trovata: Redmi {trovate[0]}/{len(coppie)}, S20FE {trovate[1]}/{len(coppie)}",
                  f"  errore di riproiezione delle articolazioni visibili: mediana {np.median(errori):.1f} px, "
                  f"90% sotto {np.percentile(errori, 90):.1f} px" if len(errori) else "  nessuna articolazione triangolata",
                  "  lunghezza delle ossa (media e variazione nel tempo, dev. std):"]
        for osso, valori in lunghezze.items():
            if len(valori) >= 5:
                righe.append(f"    {osso:22s} {np.mean(valori) * 100:5.1f} cm   ± {np.std(valori) * 100:4.1f} cm"
                             f"   ({len(valori)} coppie)")
            else:
                righe.append(f"    {osso:22s} poche coppie con entrambe le articolazioni visibili")
        righe.append("")
    righe.append(f"Immagini con le articolazioni disegnate: {CARTELLA_USCITA}\\articolazioni_<modello>.jpg")
    testo = "\n".join(righe)
    print("\n" + testo)
    os.makedirs(CARTELLA_USCITA, exist_ok=True)
    with open(os.path.join(CARTELLA_USCITA, "prova_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()
