"""
Misura di punti della scena con la coppia stereo calibrata (anticipo del test T12, usato qui come
verifica indipendente della scala della calibrazione stereo).

Si fotografa la scena con le due camere, si clicca lo stesso punto (per esempio un segno di nastro
sul pavimento) prima nell'immagine della camera 1 e poi in quella della camera 2: il programma
triangola il punto. Dopo almeno due punti stampa le distanze tra tutti i punti, con ciascuna delle
calibrazioni stereo disponibili, da confrontare con le distanze misurate con il metro sul pavimento.

Il confronto è indipendente sia dal pattern A1 sia dalla misura della baseline: dice quale
calibrazione ricostruisce la scena alla scala giusta.

Uso:   python misura_punti.py              (scatta una coppia dal vivo)
       python misura_punti.py --coppia data/calibrazione/punti_20261111_1610   (riusa una coppia salvata)
Tasti: clic sinistro = punto (prima nell'immagine di sinistra, poi in quella di destra)
       Z = annulla l'ultimo clic     ESC = fine
"""
import argparse
import glob
import itertools
import os
import time

import cv2
import numpy as np

from acquisizione import AcquisizioneStereo, STREAM
from calibrazione_mono import CARTELLA

FINESTRA = "Misura punti  (clic: sinistra poi destra, Z = annulla, ESC = fine)"
SCALA = 0.55                 # le due immagini 720 x 1280 affiancate, ridotte per stare sullo schermo
LENTE_PX = 40                # mezza larghezza della zona ingrandita intorno al cursore (pixel originali)
LENTE_ZOOM = 4
EPIPOLARE_MAX_PX = 3.0       # oltre: i due clic probabilmente non sono lo stesso punto


def leggi_stereo(percorso):
    fs = cv2.FileStorage(percorso, cv2.FILE_STORAGE_READ)
    dati = {k: fs.getNode(k).mat() for k in ("K1", "dist1", "K2", "dist2", "R", "T", "F")}
    dati["nome"] = fs.getNode("intrinseci").string() or os.path.basename(percorso)
    fs.release()
    return dati


def triangola(cal, u1, u2):
    """Punto 3D (sistema della camera 1, metri) dai due pixel, più l'errore epipolare in pixel."""
    a = cv2.undistortPoints(np.array([[u1]], np.float64), cal["K1"], cal["dist1"], P=cal["K1"]).reshape(2)
    b = cv2.undistortPoints(np.array([[u2]], np.float64), cal["K2"], cal["dist2"], P=cal["K2"]).reshape(2)
    P1 = cal["K1"] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = cal["K2"] @ np.hstack([cal["R"], cal["T"].reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, a.reshape(2, 1), b.reshape(2, 1)).ravel()
    linea = cal["F"] @ np.array([a[0], a[1], 1.0])
    epi = abs(np.array([b[0], b[1], 1.0]) @ linea) / np.hypot(linea[0], linea[1])
    return X[:3] / X[3], epi


def scatta():
    acq = AcquisizioneStereo()
    try:
        time.sleep(3)                       # qualche secondo per la stima della griglia
        coppia = acq.prossima_coppia(timeout_s=5)
    finally:
        acq.chiudi()
    if coppia is None:
        raise RuntimeError("Nessuna coppia dalle camere")
    base = os.path.join(CARTELLA, f"punti_{time.strftime('%Y%m%d_%H%M%S')}")
    nomi = list(STREAM)
    cv2.imwrite(f"{base}_{nomi[0]}.png", coppia.cam1.immagine)
    cv2.imwrite(f"{base}_{nomi[1]}.png", coppia.cam2.immagine)
    print(f"Coppia salvata: {base}_*.png  (si può riusare con --coppia {base})")
    return coppia.cam1.immagine, coppia.cam2.immagine


def main():
    parser = argparse.ArgumentParser(description="Misura di punti con la coppia stereo")
    parser.add_argument("--coppia", help="prefisso di una coppia salvata (senza _Redmi.png / _S20FE.png)")
    args = parser.parse_args()

    calibrazioni = [leggi_stereo(p) for p in sorted(glob.glob(os.path.join(CARTELLA, "stereo_*.yaml")))]
    if not calibrazioni:
        calibrazioni = [leggi_stereo(os.path.join(CARTELLA, "stereo.yaml"))]
    nomi = list(STREAM)
    if args.coppia:
        im1, im2 = (cv2.imread(f"{args.coppia}_{n}.png") for n in nomi)
    else:
        im1, im2 = scatta()
    larghezza = im1.shape[1]
    clic, cursore = [], None                 # clic: lista di (x, y) in pixel originali, alternati 1 e 2

    def mouse(evento, x, y, flags, param):
        nonlocal cursore
        X, Y = x / SCALA, y / SCALA
        cursore = (X, Y)
        if evento == cv2.EVENT_LBUTTONDOWN:
            atteso_sinistra = len(clic) % 2 == 0
            if (X < larghezza) == atteso_sinistra:
                clic.append((X if atteso_sinistra else X - larghezza, Y))

    cv2.namedWindow(FINESTRA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(FINESTRA, mouse)
    while True:
        vista = cv2.hconcat([im1, im2])
        for i, (x, y) in enumerate(clic):
            ox = 0 if i % 2 == 0 else larghezza
            cv2.drawMarker(vista, (int(ox + x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(vista, str(i // 2 + 1), (int(ox + x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)
        if len(clic) % 2 == 1 and calibrazioni:
            # aiuto: la linea epipolare del punto appena cliccato a sinistra, su cui cercare il punto a destra
            cal = calibrazioni[0]
            a = cv2.undistortPoints(np.array([[clic[-1]]], np.float64), cal["K1"], cal["dist1"], P=cal["K1"]).reshape(2)
            l = cal["F"] @ np.array([a[0], a[1], 1.0])
            y0, y1 = -l[2] / l[1], -(l[2] + l[0] * (larghezza - 1)) / l[1]
            cv2.line(vista, (larghezza, int(y0)), (2 * larghezza - 1, int(y1)), (0, 255, 255), 1, cv2.LINE_AA)
        piccola = cv2.resize(vista, None, fx=SCALA, fy=SCALA, interpolation=cv2.INTER_AREA)
        if cursore is not None:
            # lente d'ingrandimento nell'angolo in alto a sinistra
            cx, cy = int(cursore[0]), int(cursore[1])
            x0, y0 = max(0, cx - LENTE_PX), max(0, cy - LENTE_PX)
            ritaglio = vista[y0:cy + LENTE_PX, x0:cx + LENTE_PX]
            if ritaglio.size:
                lente = cv2.resize(ritaglio, None, fx=LENTE_ZOOM, fy=LENTE_ZOOM, interpolation=cv2.INTER_NEAREST)
                lente = lente[:piccola.shape[0], :piccola.shape[1] // 2]
                cv2.drawMarker(lente, ((cx - x0) * LENTE_ZOOM, (cy - y0) * LENTE_ZOOM), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 1)
                # la lente va nell'angolo in alto dell'ALTRA immagine, così non copre il punto da cliccare
                x_lente = piccola.shape[1] - lente.shape[1] if cx < larghezza else 0
                piccola[:lente.shape[0], x_lente:x_lente + lente.shape[1]] = lente
        testo = "clicca il punto nell'immagine di SINISTRA" if len(clic) % 2 == 0 else \
            "ora lo STESSO punto nell'immagine di DESTRA (lungo la linea gialla)"
        cv2.putText(piccola, testo, (10, piccola.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(FINESTRA, piccola)
        tasto = cv2.waitKey(20) & 0xFF
        if tasto in (ord("z"), ord("Z")) and clic:
            clic.pop()
        if tasto == 27:
            break
    cv2.destroyAllWindows()

    coppie = [(clic[i], clic[i + 1]) for i in range(0, len(clic) - 1, 2)]
    if len(coppie) < 2:
        print("Servono almeno due punti.")
        return
    righe = [f"Misura punti - {time.strftime('%Y-%m-%d %H:%M')} - {len(coppie)} punti", ""]
    risultati = {}
    for cal in calibrazioni:
        punti = [triangola(cal, a, b) for a, b in coppie]
        risultati[cal["nome"]] = punti
        righe.append(f"Calibrazione '{cal['nome']}':")
        for i, (X, epi) in enumerate(punti):
            avviso = "   <- i due clic forse non sono lo stesso punto" if epi > EPIPOLARE_MAX_PX else ""
            righe.append(f"  punto {i + 1}: ({X[0]:+.3f}, {X[1]:+.3f}, {X[2]:+.3f}) m, errore epipolare {epi:.1f} px{avviso}")
    righe += ["", "Distanze tra i punti (m):",
              "  " + f"{'punti':8s}" + "".join(f"{n:>18s}" for n in risultati) + "      metro"]
    for i, j in itertools.combinations(range(len(coppie)), 2):
        valori = "".join(f"{np.linalg.norm(p[i][0] - p[j][0]):18.3f}" for p in risultati.values())
        righe.append(f"  {i + 1}-{j + 1}     {valori}      ______")
    righe.append("Scrivi nella colonna 'metro' le distanze misurate sul pavimento e confrontale.")
    testo = "\n".join(righe)
    print("\n" + testo)
    with open(os.path.join(CARTELLA, f"punti_{time.strftime('%Y%m%d_%H%M%S')}.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()