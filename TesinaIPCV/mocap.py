"""
Motion capture: articolazioni del corpo dalle due camere, triangolate in 3D nel sistema del pavimento.

Per ogni coppia di fotogrammi:
  1. MediaPipe (PoseLandmarker, modalità VIDEO) trova le 33 articolazioni in ciascuna immagine;
  2. le articolazioni ben visibili in entrambe le immagini vengono triangolate (Triangolatore,
     fase 7) e portate nel sistema del pavimento: metri, origine al centro del cerchio, z in alto,
     y verso le camere, x verso destra guardando le camere;
  3. si scartano le articolazioni con errore di riproiezione troppo alto (i due raggi non si
     incontrano: rilevamento sbagliato in una delle due immagini);
  4. lo scheletro viene salvato in un file CSV e, se richiesto, inviato a Unity (UDP, JSON);
  5. si mostrano le due immagini con lo scheletro, una vista dall'alto e una di lato.

Controllo continuo: l'errore di riproiezione mediano delle articolazioni dice se le camere sono
ancora nella posizione della calibrazione. Se cresce molto, un telefono si è probabilmente mosso.

Sorgenti:
  dal vivo (predefinito)    le camere; R avvia e ferma la registrazione delle coppie
  --da CARTELLA             una registrazione fatta prima (rielaborazione, anche con un altro modello)

Uso:
  python mocap.py                                   dal vivo, modello full
  python mocap.py --udp 127.0.0.1:5065              dal vivo, inviando lo scheletro a Unity
  python mocap.py --da data/registrazioni/20261118_193000 --modello heavy
  python mocap.py --da data/registrazioni/20261118_193000 --modello heavy --nome visibilita03
                                                    come sopra, ma i file di uscita hanno l'etichetta
                                                    nel nome (scheletro_heavy_visibilita03.csv): prove
                                                    diverse sulla stessa registrazione non si sovrascrivono
Tasti: R = avvia/ferma la registrazione (solo dal vivo)   SPAZIO = pausa   ESC = fine
"""
import argparse
import csv
import json
import os
import queue
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np

from prova_mediapipe import percorso_modello
from triangolazione import Triangolatore

CARTELLA_REGISTRAZIONI = os.path.join("data", "registrazioni")
CARTELLA_SCHELETRI = os.path.join("data", "scheletri")
FINESTRA = "Motion capture  (R = registra, SPAZIO = pausa, ESC = fine)"

VISIBILITA_MIN = 0.5           # sotto: l'articolazione non viene triangolata
RIPROIEZIONE_MAX_PX = 15.0     # sopra (in almeno una camera): il punto 3D viene scartato
ALLARME_RIPROIEZIONE_PX = 8.0  # mediana recente sopra questa soglia: camere probabilmente mosse
FINESTRA_STATISTICHE = 30      # coppie su cui si calcolano velocità ed errore recente
QUALITA_JPEG = 95

NOMI = ["naso", "occhio_sx_interno", "occhio_sx", "occhio_sx_esterno", "occhio_dx_interno", "occhio_dx",
        "occhio_dx_esterno", "orecchio_sx", "orecchio_dx", "bocca_sx", "bocca_dx", "spalla_sx", "spalla_dx",
        "gomito_sx", "gomito_dx", "polso_sx", "polso_dx", "mignolo_sx", "mignolo_dx", "indice_sx", "indice_dx",
        "pollice_sx", "pollice_dx", "anca_sx", "anca_dx", "ginocchio_sx", "ginocchio_dx", "caviglia_sx",
        "caviglia_dx", "tallone_sx", "tallone_dx", "punta_piede_sx", "punta_piede_dx"]
SCHELETRO = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
             (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
             (0, 11), (0, 12)]
OSSA = [("braccio sx", 11, 13), ("braccio dx", 12, 14), ("avambraccio sx", 13, 15), ("avambraccio dx", 14, 16),
        ("coscia sx", 23, 25), ("coscia dx", 24, 26), ("gamba sx", 25, 27), ("gamba dx", 26, 28)]


# ---------------------------------------------------------------------------------------
# 1. MediaPipe sulle due immagini
# ---------------------------------------------------------------------------------------

class Rilevatore:
    """Un PoseLandmarker per camera (modalità VIDEO, segue la persona nel tempo). Le due immagini di
    una coppia vengono elaborate in parallelo su due thread."""

    def __init__(self, modello):
        percorso = percorso_modello(modello)
        self.landmarker = []
        for _ in range(2):
            opzioni = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=percorso),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1)
            self.landmarker.append(mp.tasks.vision.PoseLandmarker.create_from_options(opzioni))
        self.ultimo_ms = [-1, -1]
        self.t0 = None
        self.esecutore = ThreadPoolExecutor(max_workers=2)

    def _una(self, camera, immagine, t):
        istante = max(int((t - self.t0[camera]) * 1000), self.ultimo_ms[camera] + 1)   # sempre crescente
        self.ultimo_ms[camera] = istante
        rgb = np.ascontiguousarray(cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB))
        risultato = self.landmarker[camera].detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), istante)
        if not risultato.pose_landmarks:
            return None
        h, w = immagine.shape[:2]
        punti = risultato.pose_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in punti]), np.array([p.visibility for p in punti])

    def elabora(self, im1, im2, t1, t2):
        if self.t0 is None:
            self.t0 = (t1, t2)
        futuri = [self.esecutore.submit(self._una, 0, im1, t1), self.esecutore.submit(self._una, 1, im2, t2)]
        return [f.result() for f in futuri]

    def chiudi(self):
        self.esecutore.shutdown()
        for l in self.landmarker:
            l.close()


def ricostruisci(tri, rilevati):
    """Punti 3D nel sistema del pavimento (33, 3; NaN dove non validi), maschera dei validi e errori
    di riproiezione (33,; NaN dove non triangolati)."""
    X = np.full((33, 3), np.nan)
    errori = np.full(33, np.nan)
    if any(r is None for r in rilevati):
        return X, np.zeros(33, bool), errori
    (p1, v1), (p2, v2) = rilevati
    buone = (v1 > VISIBILITA_MIN) & (v2 > VISIBILITA_MIN)
    if buone.any():
        Xc, e = tri.triangola(p1[buone], p2[buone])
        errori[buone] = e.max(axis=1)
        X[buone] = tri.nel_pavimento(Xc)
    valide = buone & (errori <= RIPROIEZIONE_MAX_PX)
    X[~valide] = np.nan
    return X, valide, errori


# ---------------------------------------------------------------------------------------
# 2. Sorgenti di coppie, registrazione, uscite
# ---------------------------------------------------------------------------------------

def coppie_dal_vivo(acq):
    while True:
        coppia = acq.prossima_coppia(timeout_s=2)
        if coppia is not None:
            yield coppia.cam1.immagine, coppia.cam2.immagine, coppia.cam1.t_scena, coppia.cam2.t_scena


def coppie_da_registrazione(cartella, nomi):
    with open(os.path.join(cartella, "coppie.csv"), encoding="utf-8") as f:
        for riga in csv.DictReader(f, delimiter=";"):
            ims = [cv2.imread(os.path.join(cartella, f"{riga['coppia']}_{n}.jpg")) for n in nomi]
            if any(im is None for im in ims):
                continue
            yield ims[0], ims[1], float(riga["t_scena_1"]), float(riga["t_scena_2"])


class Registratore:
    """Salva le coppie (JPEG di alta qualità) e i loro istanti in un thread separato, per non rallentare."""

    def __init__(self, nomi):
        self.nomi = nomi
        self.cartella = os.path.join(CARTELLA_REGISTRAZIONI, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.cartella, exist_ok=True)
        self.indice = 0
        self.coda = queue.Queue(maxsize=300)
        self.persi = 0
        with open(os.path.join(self.cartella, "coppie.csv"), "w", encoding="utf-8") as f:
            f.write("coppia;t_scena_1;t_scena_2\n")
        self.thread = threading.Thread(target=self._scrivi, daemon=True)
        self.thread.start()

    def aggiungi(self, im1, im2, t1, t2):
        try:
            self.coda.put_nowait((f"coppia_{self.indice:06d}", im1, im2, t1, t2))
            self.indice += 1
        except queue.Full:
            self.persi += 1                     # il disco non tiene il passo: meglio perdere che bloccare

    def _scrivi(self):
        while True:
            elemento = self.coda.get()
            if elemento is None:
                return
            nome, im1, im2, t1, t2 = elemento
            for n, im in zip(self.nomi, (im1, im2)):
                cv2.imwrite(os.path.join(self.cartella, f"{nome}_{n}.jpg"), im,
                            [cv2.IMWRITE_JPEG_QUALITY, QUALITA_JPEG])
            with open(os.path.join(self.cartella, "coppie.csv"), "a", encoding="utf-8") as f:
                f.write(f"{nome};{t1:.6f};{t2:.6f}\n")

    def chiudi(self):
        self.coda.put(None)
        self.thread.join()
        print(f"Registrazione salvata in {self.cartella}: {self.indice} coppie"
              + (f" ({self.persi} perse perché il disco era lento)" if self.persi else ""))


class InvioUDP:
    """Invia a Unity, per ogni coppia, un messaggio JSON: {"t": istante, "punti": [[x, y, z] o null, ...]}
    con le coordinate nel sistema del pavimento (metri, z in alto)."""

    def __init__(self, indirizzo):
        host, porta = indirizzo.rsplit(":", 1)
        self.destinazione = (host, int(porta))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def invia(self, t, X, valide):
        punti = [[round(float(v), 4) for v in X[i]] if valide[i] else None for i in range(33)]
        self.socket.sendto(json.dumps({"t": round(t, 4), "punti": punti}).encode("utf-8"), self.destinazione)


# ---------------------------------------------------------------------------------------
# 3. Visualizzazione
# ---------------------------------------------------------------------------------------

def disegna_2d(immagine, rilevato):
    if rilevato is None:
        return
    pixel, vis = rilevato
    for a, b in SCHELETRO:
        if vis[a] > VISIBILITA_MIN and vis[b] > VISIBILITA_MIN:
            cv2.line(immagine, tuple(pixel[a].astype(int)), tuple(pixel[b].astype(int)), (0, 255, 0), 3, cv2.LINE_AA)
    for (x, y), v in zip(pixel, vis):
        cv2.circle(immagine, (int(x), int(y)), 5, (0, 0, 255) if v > VISIBILITA_MIN else (128, 128, 128), -1)


def vista(X, valide, camere, dall_alto, lato_px=320):
    """Proiezione ortogonale dello scheletro e delle camere, come le vede chi sta dalla parte delle
    camere: l'asse x del pavimento (destra della persona che guarda le camere) appare a sinistra.
    dall_alto=True: pianta, camere in basso, cerchio di 2 m disegnato. False: vista di fronte, pavimento in basso."""
    tela = np.full((lato_px, lato_px, 3), 30, np.uint8)
    if dall_alto:
        scala, cx, cy = 45, lato_px / 2, 110
        def p(v):
            return int(cx - v[0] * scala), int(cy + v[1] * scala)
        cv2.circle(tela, p(np.zeros(3)), int(1.0 * scala), (90, 90, 90), 1)
        titolo = "dall'alto (camere in basso)"
    else:
        scala, cx, base = 120, lato_px / 2, lato_px - 15
        def p(v):
            return int(cx - v[0] * scala), int(base - v[2] * scala)
        cv2.line(tela, (0, base), (lato_px, base), (90, 90, 90), 1)
        titolo = "di fronte (come dalle camere)"
    for c in camere:
        cv2.drawMarker(tela, p(c), (255, 200, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
    for a, b in SCHELETRO:
        if valide[a] and valide[b]:
            cv2.line(tela, p(X[a]), p(X[b]), (0, 255, 0), 2, cv2.LINE_AA)
    for i in np.flatnonzero(valide):
        cv2.circle(tela, p(X[i]), 3, (0, 0, 255), -1)
    cv2.putText(tela, titolo, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return tela


def schermata(im1, im2, X, valide, camere, testi, altezza=640):
    piccole = [cv2.resize(im, (round(im.shape[1] * altezza / im.shape[0]), altezza), interpolation=cv2.INTER_AREA)
               for im in (im1, im2)]
    viste = cv2.vconcat([vista(X, valide, camere, True), vista(X, valide, camere, False)])
    viste = cv2.resize(viste, (round(viste.shape[1] * altezza / viste.shape[0]), altezza))
    riga = cv2.hconcat(piccole + [viste])
    fascia = np.zeros((20 + 26 * len(testi), riga.shape[1], 3), np.uint8)
    for k, (testo, allarme) in enumerate(testi):
        cv2.putText(fascia, testo, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if allarme else (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.vconcat([riga, fascia])


# ---------------------------------------------------------------------------------------
# 4. Programma
# ---------------------------------------------------------------------------------------

def main():
    from acquisizione import AcquisizioneStereo, STREAM

    parser = argparse.ArgumentParser(description="Motion capture con due camere e MediaPipe")
    parser.add_argument("--modello", default="full", choices=("lite", "full", "heavy"))
    parser.add_argument("--da", help="cartella di una registrazione da rielaborare")
    parser.add_argument("--udp", help="indirizzo:porta a cui inviare lo scheletro (Unity), es. 127.0.0.1:5065")
    parser.add_argument("--nome", help="etichetta aggiunta al nome dei file di uscita, es. visibilita03")
    args = parser.parse_args()

    nomi = list(STREAM)
    tri = Triangolatore()
    if tri.R_pav is None:
        raise SystemExit("Manca data/calibrazione/pavimento.yaml: prima esegui triangolazione.py --pavimento")
    camere = tri.nel_pavimento(np.vstack([np.zeros(3), tri.centro2]))
    rilevatore = Rilevatore(args.modello)
    udp = InvioUDP(args.udp) if args.udp else None

    # Etichetta facoltativa nel nome dei file: "_visibilita03" con --nome visibilita03, niente senza
    etichetta = f"_{args.nome}" if args.nome else ""
    acq = None
    if args.da:
        sorgente = coppie_da_registrazione(args.da, nomi)
        os.makedirs(args.da, exist_ok=True)
        percorso_csv = os.path.join(args.da, f"scheletro_{args.modello}{etichetta}.csv")
    else:
        acq = AcquisizioneStereo()
        sorgente = coppie_dal_vivo(acq)
        os.makedirs(CARTELLA_SCHELETRI, exist_ok=True)
        percorso_csv = os.path.join(CARTELLA_SCHELETRI, f"scheletro_{time.strftime('%Y%m%d_%H%M%S')}{etichetta}.csv")
    if os.path.exists(percorso_csv):
        print(f"ATTENZIONE: {percorso_csv} esiste già e verrà sovrascritto (per tenerlo usa --nome)")
    file_csv = open(percorso_csv, "w", encoding="utf-8", newline="")
    file_csv.write("coppia;t_scena_1;t_scena_2;registrazione;"
                   + ";".join(f"{n}_{a}" for n in NOMI for a in "xyz") + "\n")

    registratore = None
    istanti = deque(maxlen=FINESTRA_STATISTICHE)
    errori_recenti = deque(maxlen=FINESTRA_STATISTICHE)
    lunghezze = {o[0]: [] for o in OSSA}
    tutti_errori, coppie, con_persona = [], 0, 0
    pausa = False
    cv2.namedWindow(FINESTRA, cv2.WINDOW_AUTOSIZE)
    try:
        for im1, im2, t1, t2 in sorgente:
            tasto = cv2.waitKey(1) & 0xFF
            if tasto == 27:
                break
            if tasto == ord(" "):
                pausa = not pausa
            if tasto in (ord("r"), ord("R")) and acq is not None:
                if registratore is None:
                    registratore = Registratore(nomi)
                    print(f"Registrazione avviata in {registratore.cartella}")
                else:
                    registratore.chiudi()
                    registratore = None
            while pausa:
                tasto = cv2.waitKey(50) & 0xFF
                if tasto in (ord(" "), 27):
                    pausa = False
            if registratore is not None:
                registratore.aggiungi(im1, im2, t1, t2)

            rilevati = rilevatore.elabora(im1, im2, t1, t2)
            X, valide, errori = ricostruisci(tri, rilevati)
            coppie += 1
            con_persona += bool(valide.any())
            istanti.append(time.perf_counter())
            buoni = errori[np.isfinite(errori)]
            if len(buoni):
                errori_recenti.append(np.median(buoni))
                tutti_errori.extend(buoni)
            for osso, a, b in OSSA:
                if valide[a] and valide[b]:
                    lunghezze[osso].append(np.linalg.norm(X[a] - X[b]))

            valori = ";".join((f"{X[i, 0]:.4f};{X[i, 1]:.4f};{X[i, 2]:.4f}" if valide[i] else ";;")
                              for i in range(33))
            file_csv.write(f"{coppie};{t1:.6f};{t2:.6f};{int(registratore is not None)};{valori}\n")
            if udp is not None:
                udp.invia(t1, X, valide)

            disegna_2d(im1, rilevati[0])
            disegna_2d(im2, rilevati[1])
            velocita = (len(istanti) - 1) / (istanti[-1] - istanti[0]) if len(istanti) > 1 else 0.0
            errore = float(np.median(errori_recenti)) if errori_recenti else float("nan")
            allarme = errore > ALLARME_RIPROIEZIONE_PX
            altezza = np.nanmax(X[:, 2]) if valide.any() else float("nan")
            testi = [(f"{velocita:4.1f} coppie/s   modello {args.modello}   articolazioni valide {int(valide.sum())}/33"
                      f"   punto più alto {altezza:.2f} m", False),
                     (f"errore di riproiezione recente {errore:.1f} px"
                      + ("  -> ALTO: un telefono si è mosso? controlla con rettificazione.py --vivo" if allarme else ""),
                      allarme),
                     ("REGISTRAZIONE IN CORSO (R per fermare)" if registratore is not None else
                      ("R = registra" if acq is not None else f"rielaborazione di {args.da}"), registratore is not None)]
            cv2.imshow(FINESTRA, schermata(im1, im2, X, valide, camere, testi))
    finally:
        if registratore is not None:
            registratore.chiudi()
        file_csv.close()
        rilevatore.chiudi()
        if acq is not None:
            acq.chiudi()
        cv2.destroyAllWindows()

    righe = [f"Motion capture - {time.strftime('%Y-%m-%d %H:%M')} - modello {args.modello}"
             + (f" - registrazione {args.da}" if args.da else " - dal vivo")
             + (f" - prova '{args.nome}'" if args.nome else ""),
             f"coppie elaborate: {coppie}, con la persona ricostruita: {con_persona}"]
    if tutti_errori:
        righe.append(f"errore di riproiezione: mediana {np.median(tutti_errori):.1f} px, "
                     f"90% sotto {np.percentile(tutti_errori, 90):.1f} px")
    righe.append("lunghezza delle ossa (media ± dev. std nel tempo):")
    for osso, valori in lunghezze.items():
        if len(valori) >= 5:
            righe.append(f"  {osso:16s} {np.mean(valori) * 100:5.1f} cm ± {np.std(valori) * 100:4.1f} cm  ({len(valori)} coppie)")
    righe.append(f"scheletro salvato in {percorso_csv}")
    testo = "\n".join(righe)
    print("\n" + testo)
    with open(percorso_csv.replace(".csv", "_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()
