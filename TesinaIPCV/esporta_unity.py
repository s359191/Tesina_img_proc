"""
Fase Unity, passo 1: esporta la scena reale per Unity (test T20).

Unity deve ricostruire lo spazio della stanza 1:1. Per farlo gli servono, nel sistema del pavimento
(teoria sezione 4.11):
  - le croci sul pavimento (dove sono nella stanza);
  - le due camere: dove stanno, dove guardano e con quale obiettivo (K), così che una camera di Unity
    messa nello stesso punto veda la scena come il telefono vero.

Il programma legge stereo.yaml, pavimento.yaml e il file delle croci usato per il pavimento, e scrive
data/unity/scena.json, già negli assi di Unity:

  sistema del pavimento (destrorso)      Unity (sinistrorso)
  x  verso destra guardando le camere -> X  (destra)
  z  verso l'alto                     -> Y  (alto)
  y  verso le camere                  -> Z  (avanti)

Uso:   python esporta_unity.py
Va rieseguito ogni volta che si rifanno la calibrazione stereo o il sistema del pavimento.
"""
import json
import os
import time

import cv2
import numpy as np

from acquisizione import STREAM
from calibrazione_mono import CARTELLA
from triangolazione import Triangolatore, FILE_STEREO, FILE_PAVIMENTO, leggi_punti_3d

CARTELLA_UNITY = os.path.join("data", "unity")
FILE_SCENA = os.path.join(CARTELLA_UNITY, "scena.json")


def in_unity(v):
    """Un punto o una direzione dal sistema del pavimento agli assi di Unity (X, Y, Z) = (x, z, y)."""
    x, y, z = (float(c) for c in v)
    return {"x": round(x, 5), "y": round(z, 5), "z": round(y, 5)}


def leggi_testo(percorso, chiave):
    fs = cv2.FileStorage(percorso, cv2.FILE_STORAGE_READ)
    valore = fs.getNode(chiave)
    risultato = valore.string() if valore.isString() else valore.real()
    fs.release()
    return risultato


def main():
    tri = Triangolatore()
    if tri.R_pav is None:
        raise SystemExit("Manca data/calibrazione/pavimento.yaml: prima esegui triangolazione.py --pavimento")
    larghezza = int(leggi_testo(FILE_STEREO, "larghezza"))
    altezza = int(leggi_testo(FILE_STEREO, "altezza"))

    # Le croci: lo stesso file usato per definire il pavimento
    file_croci = os.path.join(CARTELLA, leggi_testo(FILE_PAVIMENTO, "punti"))
    numeri, X = leggi_punti_3d(file_croci)
    croci = tri.nel_pavimento(X)

    # Le camere. Le righe di R_cam sono gli assi della camera (x destra, y giù, z avanti)
    # espressi nel sistema della camera 1: per la camera 1 è l'identità, per la camera 2 è R
    # (X2 = R X1 + T, quindi l'asse k della camera 2 nel sistema 1 è la riga k di R).
    nomi = list(STREAM)
    camere = []
    for nome, K, centro, R_cam in ((nomi[0], tri.K1, np.zeros(3), np.eye(3)),
                                   (nomi[1], tri.K2, tri.centro2, tri.R)):
        posizione = tri.nel_pavimento(centro)[0]
        avanti = tri.R_pav @ R_cam[2]          # direzione in cui guarda l'obiettivo
        su = -(tri.R_pav @ R_cam[1])           # "su" dell'immagine (l'asse y dell'immagine va in giù)
        camere.append({"nome": nome,
                       "posizione": in_unity(posizione), "avanti": in_unity(avanti), "su": in_unity(su),
                       "larghezza": larghezza, "altezza": altezza,
                       "fx": round(float(K[0, 0]), 3), "fy": round(float(K[1, 1]), 3),
                       "cx": round(float(K[0, 2]), 3), "cy": round(float(K[1, 2]), 3)})

    scena = {"data": time.strftime("%Y-%m-%d %H:%M"),
             "calibrazione_stereo": tri.data_stereo,
             "pavimento": leggi_testo(FILE_PAVIMENTO, "data"),
             "croci": [dict(numero=int(n), **in_unity(p)) for n, p in zip(numeri, croci)],
             "camere": camere}
    os.makedirs(CARTELLA_UNITY, exist_ok=True)
    with open(FILE_SCENA, "w", encoding="utf-8") as f:
        json.dump(scena, f, indent=2)

    # Riepilogo da controllare con il metro (stessi numeri di pavimento_riepilogo.txt)
    print(f"Scena per Unity - calibrazione stereo del {tri.data_stereo}, pavimento del {scena['pavimento']}")
    print("Croci (Unity: X destra, Y alto, Z verso le camere), in metri:")
    for c in scena["croci"]:
        print(f"  croce {c['numero']}: X {c['x']:+.3f}  Y {c['y'] * 1000:+.1f} mm  Z {c['z']:+.3f}")
    for c in camere:
        p, a = c["posizione"], c["avanti"]
        inclinazione = np.degrees(np.arcsin(-a["y"]))
        campo = np.degrees(2 * np.arctan(c["altezza"] / 2 / c["fy"]))
        print(f"  {c['nome']}: X {p['x']:+.3f}  Y (altezza) {p['y']:.3f}  Z {p['z']:+.3f} m, "
              f"guarda {inclinazione:.1f} gradi verso il basso, campo verticale {campo:.1f} gradi")
    print(f"\nSalvato {os.path.abspath(FILE_SCENA)}")


if __name__ == "__main__":
    main()
