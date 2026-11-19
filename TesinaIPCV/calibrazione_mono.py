"""
Fase 4 - Calibrazione monoculare di una camera (test T06; requisiti RF4, RF6, RC1).

Due passi nello stesso programma:
  1. ACQUISIZIONE - la camera dal vivo con il pattern A3. Il programma salva da solo un'immagine quando
     il pattern è fermo, a una distanza adatta al fuoco fisso e in una posizione nuova; una griglia
     colorata mostra le zone dell'immagine già coperte dagli angoli delle immagini salvate.
  2. CALCOLO - rileva gli angoli in tutte le immagini salvate, calibra con cv2.calibrateCameraExtended,
     controlla il risultato (T06) e salva K e i coefficienti di distorsione in data/calibrazione/<camera>.yaml

Uso:
  python calibrazione_mono.py --camera S20FE                   acquisizione, poi calcolo
  python calibrazione_mono.py --camera S20FE --solo-calcolo    ricalcola dalle immagini già salvate
  python calibrazione_mono.py --camera S20FE --nuova           cancella le immagini salvate e ricomincia
  python calibrazione_mono.py --camera S20FE --verifica-distanza
        verifica indipendente della focale: distanza del pattern stimata con la calibrazione,
        da confrontare con quella misurata con il metro (SPAZIO = registra la misura, ESC = esci)
Tasti durante l'acquisizione:
  SPAZIO = salva subito (se il pattern è fermo)   C = calcola   ESC = esci senza calcolare
Prima di avviarlo: IP Webcam avviata, adb forward attivo, imposta_camere.py -> PRONTO.
"""
import argparse
import glob
import os
import threading
import time

import cv2
import numpy as np

from acquisizione import STREAM, ROTAZIONE, LettoreStream

FILE_PATTERN = "charuco_boards.yaml"
CARTELLA = os.path.join("data", "calibrazione")
FINESTRA = "Calibrazione  (SPAZIO = salva, C = calcola, ESC = esci)"
SCALA_VISTA = 0.7            # 720x1280 non sta in verticale sullo schermo di un portatile

FOTO_OBIETTIVO = 25          # immagini da raccogliere (20-30)
ANGOLI_MINIMI = 12           # su 24: almeno metà pattern visibile, altrimenti la posa è poco determinata
FERMO_FOTOGRAMMI = 6         # il pattern deve restare fermo per tanti fotogrammi consecutivi (0,2 s)...
FERMO_PX = 0.7               # ...con spostamento massimo degli angoli inferiore a tanto (pixel)
NUOVA_POSA_PX = 40           # spostamento medio minimo degli angoli rispetto a ogni immagine già salvata
NUOVA_INCLINAZIONE = 10      # oppure inclinazione diversa di almeno tanti gradi
PAUSA_S = 1.0                # tempo minimo tra due salvataggi automatici
DISTANZA_M = (0.6, 1.7)      # fuori da qui il pattern è troppo sfocato (fuoco fisso a 1,7 m) o piccolo
LATO_CELLA_PX = 90           # griglia della copertura
FOCALE_GUIDA_PX = 950        # focale approssimata, solo per stimare distanza e inclinazione dal vivo
# Focale attesa dai dati nominali del sensore (Camera2): solo indicativa. Il flusso video può usare
# una parte ritagliata del sensore, e la focale nominale è arrotondata: la verifica vera della focale
# è quella con il metro (--verifica-distanza).
FOCALE_ATTESA = {"S20FE": 952}
VERIFICA_FOTOGRAMMI = 30     # fotogrammi mediati per una misura di distanza

# Criteri del test T06
RMS_OTTIMO, RMS_ACCETTABILE = 0.5, 1.0
RAPPORTO_FOCALI_MAX = 0.01   # |fx/fy - 1|
COPERTURA_MIN = 0.70         # frazione delle celle della griglia con almeno un angolo
INCLINATE_MIN = 5            # immagini con il pattern inclinato più di 20 gradi
INCERTEZZA_FOCALE_MAX = 0.01      # incertezza della focale: sotto l'1%


# ---------------------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------------------

def leggi_pattern(nome, percorso=FILE_PATTERN):
    """Legge charuco_boards.yaml (formato semplice 'chiave: valore', con sezioni per pattern)."""
    valori, sezione = {}, None
    with open(percorso, encoding="utf-8") as f:
        for riga in f:
            riga = riga.split("#")[0].rstrip()
            if not riga.strip():
                continue
            chiave, _, valore = riga.strip().partition(":")
            if not riga.startswith(" "):
                sezione = chiave
                if valore.strip():
                    valori[chiave] = valore.strip()
            elif sezione == nome:
                valori[chiave] = valore.strip()
    dizionario = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, valori["dictionary"]))
    colonne, righe = int(valori["squares_x"]), int(valori["squares_y"])
    primo = int(valori["first_id"])
    ids = np.arange(primo, primo + (colonne * righe) // 2, dtype=np.int32)
    # stessa costruzione di genera_charuco.py: lunghezze in metri
    lato = float(valori["square_length_m"])
    board = cv2.aruco.CharucoBoard((colonne, righe), lato, float(valori["marker_length_m"]), dizionario, ids)
    # Stampa non uniforme: le stampanti di grande formato possono allungare il foglio in una sola
    # direzione. Se nel file ci sono i lati misurati separatamente (square_length_x_m lungo le 7
    # colonne, square_length_y_m lungo le 5 righe), le coordinate degli angoli vengono scalate.
    lato_x = float(valori.get("square_length_x_m", lato))
    lato_y = float(valori.get("square_length_y_m", lato))
    _SCALA_PATTERN[id(board)] = np.array([lato_x / lato, lato_y / lato, 1.0])
    valori["lato_x"], valori["lato_y"] = lato_x, lato_y
    return board, valori


_SCALA_PATTERN = {}          # id del pattern -> fattori di scala (x, y, z) delle coordinate degli angoli


def rileva(detector, board, immagine):
    """Angoli ChArUco in un'immagine: (punti 3D sul pattern, punti 2D nell'immagine, id) oppure None."""
    angoli, ids, _, _ = detector.detectBoard(immagine)
    if ids is None or len(ids) < ANGOLI_MINIMI:
        return None
    oggetto, immagine_pt = board.matchImagePoints(angoli, ids)
    oggetto = oggetto.reshape(-1, 3) * _SCALA_PATTERN.get(id(board), 1.0)
    return oggetto.astype(np.float32), immagine_pt.reshape(-1, 2), ids.ravel()


def inclinazione_gradi(rvec):
    """Angolo tra la normale al pattern e l'asse ottico: 0 = pattern frontale."""
    R, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


# ---------------------------------------------------------------------------------------
# 1. Acquisizione
# ---------------------------------------------------------------------------------------

class Copertura:
    """Griglia di celle sull'immagine: quali celle hanno già ricevuto almeno un angolo."""

    def __init__(self, larghezza, altezza):
        self.colonne = max(1, round(larghezza / LATO_CELLA_PX))
        self.righe = max(1, round(altezza / LATO_CELLA_PX))
        self.dim = (larghezza, altezza)
        self.conteggi = np.zeros((self.righe, self.colonne), int)

    def aggiungi(self, punti):
        for x, y in punti:
            c = min(self.colonne - 1, int(x / self.dim[0] * self.colonne))
            r = min(self.righe - 1, int(y / self.dim[1] * self.righe))
            self.conteggi[r, c] += 1

    def frazione(self):
        return float(np.mean(self.conteggi > 0))

    def angoli_immagine(self):
        """Quanti dei 4 angoli dell'immagine (blocchi di 2x2 celle) contengono angoli del pattern."""
        r, c = self.righe, self.colonne
        blocchi = [self.conteggi[:2, :2], self.conteggi[:2, c - 2:], self.conteggi[r - 2:, :2],
                   self.conteggi[r - 2:, c - 2:]]
        return sum(int(b.sum() > 0) for b in blocchi)

    def disegna(self, immagine):
        sovrapposta = immagine.copy()
        for r in range(self.righe):
            for c in range(self.colonne):
                if self.conteggi[r, c] > 0:
                    x0, y0 = int(c * self.dim[0] / self.colonne), int(r * self.dim[1] / self.righe)
                    x1, y1 = int((c + 1) * self.dim[0] / self.colonne), int((r + 1) * self.dim[1] / self.righe)
                    cv2.rectangle(sovrapposta, (x0, y0), (x1, y1), (0, 180, 0), -1)
        return cv2.addWeighted(sovrapposta, 0.25, immagine, 0.75, 0)


def stima_posa(oggetto, punti, dimensione):
    """Distanza (m) e inclinazione (gradi) del pattern con una focale approssimata: solo come guida."""
    w, h = dimensione
    K = np.array([[FOCALE_GUIDA_PX, 0, w / 2], [0, FOCALE_GUIDA_PX, h / 2], [0, 0, 1]], float)
    ok, rvec, tvec = cv2.solvePnP(oggetto, punti, K, None)
    if not ok:
        return None, None
    return float(np.linalg.norm(tvec)), inclinazione_gradi(rvec)


def acquisisci(camera, cartella, board, detector):
    """Mostra la camera e salva le immagini adatte. Restituisce True se si deve passare al calcolo."""
    os.makedirs(cartella, exist_ok=True)
    salvate = sorted(glob.glob(os.path.join(cartella, "img_*.png")))
    lettore = LettoreStream(camera, STREAM[camera], threading.Condition(), rotazione=ROTAZIONE.get(camera))
    copertura, pose_salvate = None, []     # pose: (dizionario id -> punto 2D, inclinazione)
    # le immagini già presenti (programma riavviato) contano per copertura e pose
    for percorso in salvate:
        im = cv2.imread(percorso)
        if copertura is None:
            copertura = Copertura(im.shape[1], im.shape[0])
        r = rileva(detector, board, im)
        if r is not None:
            oggetto, punti, ids = r
            copertura.aggiungi(punti)
            pose_salvate.append((dict(zip(ids, punti)), stima_posa(oggetto, punti, (im.shape[1], im.shape[0]))[1]))
    storia = []                            # ultime rilevazioni, per decidere se il pattern è fermo
    ultimo_indice, ultimo_salvataggio = 0, 0.0
    cv2.namedWindow(FINESTRA, cv2.WINDOW_NORMAL)
    spazio = False                         # SPAZIO premuto e non ancora usato
    try:
        while True:
            # i tasti si leggono a ogni giro, anche quando non c'è un fotogramma nuovo, sennò si perdono
            tasto = cv2.waitKey(5) & 0xFF
            if tasto == 27:
                return False
            if tasto in (ord("c"), ord("C")):
                return True
            spazio = spazio or tasto == ord(" ")
            with lettore._cond:
                fot = lettore.buffer[-1] if lettore.buffer else None
            if fot is None or fot.indice == ultimo_indice:
                continue
            ultimo_indice = fot.indice
            immagine = fot.immagine
            h, w = immagine.shape[:2]
            if copertura is None:
                copertura = Copertura(w, h)
            r = rileva(detector, board, immagine)
            vista = copertura.disegna(immagine)
            stato, colore, adatta, fermo = "pattern non trovato (servono almeno 12 angoli su 24)", (0, 0, 255), False, False
            if r is None:
                storia.clear()
            else:
                oggetto, punti, ids = r
                cv2.aruco.drawDetectedCornersCharuco(vista, punti.reshape(-1, 1, 2).astype(np.float32),
                                                     ids.reshape(-1, 1), (255, 0, 0))
                attuale = dict(zip(ids, punti))
                storia = [s for s in storia[-(FERMO_FOTOGRAMMI - 1):]] + [attuale]
                # fermo: in tutte le ultime rilevazioni gli angoli comuni si sono spostati poco
                fermo = len(storia) == FERMO_FOTOGRAMMI and all(
                    len(set(s) & set(attuale)) >= ANGOLI_MINIMI and
                    max(np.linalg.norm(s[i] - attuale[i]) for i in set(s) & set(attuale)) < FERMO_PX
                    for s in storia[:-1])
                distanza, inclinazione = stima_posa(oggetto, punti, (w, h))
                if distanza is None:
                    distanza, inclinazione = 0.0, 0.0
                nuova = all(
                    len(set(p) & set(attuale)) < 6 or
                    np.mean([np.linalg.norm(p[i] - attuale[i]) for i in set(p) & set(attuale)]) > NUOVA_POSA_PX or
                    (incl is not None and abs(inclinazione - incl) > NUOVA_INCLINAZIONE)
                    for p, incl in pose_salvate)
                vicino_ok = DISTANZA_M[0] <= distanza <= DISTANZA_M[1]
                adatta = fermo and nuova and vicino_ok
                if not vicino_ok:
                    stato = f"distanza {distanza:.2f} m: tieni il pattern tra 0,7 e 1,5 m"
                elif not fermo:
                    stato, colore = "tieni il pattern fermo", (0, 200, 255)
                elif not nuova:
                    stato, colore = "posizione già salvata: spostalo o inclinalo", (0, 200, 255)
                else:
                    stato, colore = "ok", (0, 255, 0)
                stato += f"   ({len(ids)} angoli, {distanza:.2f} m, inclinazione {inclinazione:.0f} gradi)"
            adesso = time.perf_counter()
            salva = r is not None and ((adatta and adesso - ultimo_salvataggio > PAUSA_S) or (spazio and fermo))
            if r is None or fermo:
                spazio = False                 # usato, oppure senza pattern non ha senso tenerlo
            if salva:
                numero = len(glob.glob(os.path.join(cartella, "img_*.png")))
                cv2.imwrite(os.path.join(cartella, f"img_{numero:03d}.png"), immagine)   # PNG: nessuna perdita
                copertura.aggiungi(punti)
                pose_salvate.append((attuale, inclinazione))
                ultimo_salvataggio = adesso
                vista = cv2.addWeighted(vista, 0.5, np.full_like(vista, 255), 0.5, 0)     # lampo bianco
            n = len(pose_salvate)
            testi = [(f"{camera}: salvate {n}/{FOTO_OBIETTIVO}   copertura {copertura.frazione() * 100:.0f}%   "
                      f"angoli dell'immagine raggiunti {copertura.angoli_immagine()}/4", (255, 255, 255)),
                     (stato, colore)]
            if n >= FOTO_OBIETTIVO:
                testi.append(("Obiettivo raggiunto: premi C per calcolare (o continua a coprire le zone vuote)",
                              (0, 255, 0)))
            scala = w / 720
            for riga, (testo, c) in enumerate(testi):
                pos = (10, int((40 + 34 * riga) * scala))
                cv2.putText(vista, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scala, (0, 0, 0), int(5 * scala), cv2.LINE_AA)
                cv2.putText(vista, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scala, c, max(1, int(2 * scala)), cv2.LINE_AA)
            cv2.imshow(FINESTRA, cv2.resize(vista, None, fx=SCALA_VISTA, fy=SCALA_VISTA, interpolation=cv2.INTER_AREA))
    finally:
        lettore.chiudi()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------
# 2. Calcolo e controlli
# ---------------------------------------------------------------------------------------

NOMI_PARAMETRI = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
CRITERI = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)


def calibra(oggetti, punti, dimensione, flags):
    rms, K, dist, rvecs, tvecs, std_int, _, errori = cv2.calibrateCameraExtended(
        oggetti, punti, dimensione, None, None, flags=flags, criteria=CRITERI)
    return rms, K, dist.ravel()[:5], rvecs, std_int.ravel()[:9], errori.ravel()


def calcola(camera, cartella, board, detector, nome_pattern):
    percorsi = sorted(glob.glob(os.path.join(cartella, "img_*.png")))
    righe = [f"Calibrazione {camera} - {time.strftime('%Y-%m-%d %H:%M')} - pattern {nome_pattern}, "
             f"rotazione {ROTAZIONE.get(camera)}", ""]
    oggetti, punti, usati, dimensione = [], [], [], None
    for p in percorsi:
        im = cv2.imread(p)
        dimensione = (im.shape[1], im.shape[0])
        r = rileva(detector, board, im)
        if r is None:
            righe.append(f"  {os.path.basename(p)}: pattern non trovato, esclusa")
            continue
        oggetti.append(r[0].astype(np.float32))
        punti.append(r[1].astype(np.float32))
        usati.append(p)
    if len(usati) < 10:
        righe.append(f"Solo {len(usati)} immagini utilizzabili: ne servono almeno 10 (meglio 20-30).")
        return righe, False

    # Prima calibrazione con il modello completo (k1, k2, p1, p2, k3)
    flags = 0
    rms, K, dist, rvecs, std, errori = calibra(oggetti, punti, dimensione, flags)
    # Immagini anomale: errore molto più alto delle altre (mossa, sfocata, angoli sbagliati)
    soglia = max(3 * np.median(errori), 1.0)
    anomale = [i for i, e in enumerate(errori) if e > soglia]
    if anomale:
        for i in anomale:
            righe.append(f"  {os.path.basename(usati[i])}: errore {errori[i]:.2f} px, oltre {soglia:.2f} px: esclusa")
        tieni = [i for i in range(len(usati)) if i not in anomale]
        oggetti, punti, usati = [oggetti[i] for i in tieni], [punti[i] for i in tieni], [usati[i] for i in tieni]
        rms, K, dist, rvecs, std, errori = calibra(oggetti, punti, dimensione, flags)
    # k3 poco determinato (incertezza più grande del valore): si fissa a zero
    if std[8] > abs(dist[4]):
        righe.append(f"  k3 = {dist[4]:.3f} ± {std[8]:.3f}: non determinato dai dati, fissato a 0")
        flags |= cv2.CALIB_FIX_K3
        rms, K, dist, rvecs, std, errori = calibra(oggetti, punti, dimensione, flags)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = dimensione

    copertura = Copertura(w, h)
    for p in punti:
        copertura.aggiungi(p)
    inclinazioni = [inclinazione_gradi(rv) for rv in rvecs]
    n_inclinate = sum(i > 20 for i in inclinazioni)

    # Controlli del test T06
    controlli = []
    controlli.append(("RMS di riproiezione", f"{rms:.3f} px",
                      f"< {RMS_OTTIMO} px (accettabile < {RMS_ACCETTABILE})", rms < RMS_ACCETTABILE))
    controlli.append(("fx / fy", f"{fx / fy:.4f}", f"1 ± {RAPPORTO_FOCALI_MAX:.2f}", abs(fx / fy - 1) <= RAPPORTO_FOCALI_MAX))
    controlli.append(("punto principale", f"({cx:.1f}, {cy:.1f})", f"entro il 5% dal centro ({w / 2:.0f}, {h / 2:.0f})",
                      abs(cx - w / 2) <= 0.05 * w and abs(cy - h / 2) <= 0.05 * h))
    controlli.append(("incertezza della focale", f"± {std[0]:.2f} / {std[1]:.2f} px",
                      f"< {INCERTEZZA_FOCALE_MAX * 100:.1f}% della focale",
                      max(std[0] / fx, std[1] / fy) <= INCERTEZZA_FOCALE_MAX))
    controlli.append(("copertura dell'immagine", f"{copertura.frazione() * 100:.0f}% delle celle, "
                      f"{copertura.angoli_immagine()}/4 angoli", f">= {COPERTURA_MIN * 100:.0f}% e 4/4 angoli",
                      copertura.frazione() >= COPERTURA_MIN and copertura.angoli_immagine() == 4))
    controlli.append(("immagini inclinate oltre 20 gradi", f"{n_inclinate} su {len(usati)}",
                      f">= {INCLINATE_MIN}", n_inclinate >= INCLINATE_MIN))

    righe += ["", f"Immagini usate: {len(usati)} su {len(percorsi)}   dimensione {w} x {h}", "",
              "Parametri (valore ± incertezza):"]
    valori = [fx, fy, cx, cy, *dist]
    for nome, v, s in zip(NOMI_PARAMETRI, valori, std):
        fissato = nome == "k3" and flags & cv2.CALIB_FIX_K3
        righe.append(f"  {nome:3s} = {v:10.4f}" + ("   (fissato)" if fissato else f"  ± {s:.4f}"))
    righe += ["", "Controlli del test T06:"]
    for nome, valore, criterio, ok in controlli:
        righe.append(f"  {'OK ' if ok else 'NO '} {nome:34s} {valore:28s} criterio: {criterio}")
    righe += ["", f"Errore per immagine: mediana {np.median(errori):.3f} px, massimo {errori.max():.3f} px "
              f"({os.path.basename(usati[int(np.argmax(errori))])})",
              f"Inclinazioni: da {min(inclinazioni):.0f} a {max(inclinazioni):.0f} gradi"]
    if camera in FOCALE_ATTESA:
        attesa = FOCALE_ATTESA[camera]
        righe.append(f"Focale media {(fx + fy) / 2:.1f} px: {((fx + fy) / 2 / attesa - 1) * 100:+.1f}% rispetto ai "
                     f"{attesa} px nominali (indicativo: la verifica è con il metro, --verifica-distanza)")
    esito = all(ok for *_, ok in controlli)
    if esito and rms >= RMS_OTTIMO:
        righe.append(f"RMS tra {RMS_OTTIMO} e {RMS_ACCETTABILE} px: accettabile, ma si può migliorare con immagini più nitide")

    # File: parametri (RF6), errori per immagine, confronto originale/corretta per il controllo visivo
    base = os.path.join(CARTELLA, camera)
    fs = cv2.FileStorage(f"{base}.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("camera", camera)
    fs.write("data", time.strftime("%Y-%m-%d %H:%M"))
    fs.write("larghezza", w)
    fs.write("altezza", h)
    fs.write("rotazione", str(ROTAZIONE.get(camera)))
    fs.write("pattern", nome_pattern)
    fs.write("K", K)
    fs.write("dist", dist.reshape(1, 5))
    fs.write("rms_px", float(rms))
    fs.write("incertezze", np.array(std[:9]).reshape(1, 9))
    fs.write("immagini", len(usati))
    fs.write("k3_fissato", int(bool(flags & cv2.CALIB_FIX_K3)))
    fs.release()
    with open(f"{base}_errori.csv", "w") as f:
        f.write("immagine;errore_px;inclinazione_gradi;angoli\n")
        f.writelines(f"{os.path.basename(p)};{e:.3f};{i:.1f};{len(pt)}\n"
                     for p, e, i, pt in zip(usati, errori, inclinazioni, punti))
    # immagine con gli angoli più lontani dal centro: lì la distorsione si vede di più
    lontana = max(range(len(usati)), key=lambda i: np.max(np.hypot(punti[i][:, 0] - cx, punti[i][:, 1] - cy)))
    originale = cv2.imread(usati[lontana])
    corretta = cv2.undistort(originale, K, dist)
    cv2.imwrite(f"{base}_confronto.png", cv2.hconcat([originale, corretta]))
    righe += ["", f"Salvati: {base}.yaml, {base}_errori.csv, {base}_confronto.png (sinistra originale, destra corretta)",
              f"ESITO T06 {camera}: {'SUPERATO' if esito else 'NON SUPERATO'}"]
    return righe, esito


def leggi_calibrazione(camera):
    """K e coefficienti di distorsione salvati da calcola() (requisito RF6)."""
    fs = cv2.FileStorage(os.path.join(CARTELLA, f"{camera}.yaml"), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Manca {os.path.join(CARTELLA, camera)}.yaml: prima calibra la camera")
    K, dist = fs.getNode("K").mat(), fs.getNode("dist").mat()
    fs.release()
    return K, dist


def verifica_distanza(camera, board, detector, valori):
    """Mostra la distanza dalla camera al centro del pattern stimata con la calibrazione.
    Un errore della focale del p% dà un errore della distanza del p%: a 1,2 m, 1% = 1,2 cm."""
    K, dist = leggi_calibrazione(camera)
    lato = float(valori["square_length_m"])
    centro = np.array([int(valori["squares_x"]) * valori["lato_x"] / 2, int(valori["squares_y"]) * valori["lato_y"] / 2, 0.0])
    lettore = LettoreStream(camera, STREAM[camera], threading.Condition(), rotazione=ROTAZIONE.get(camera))
    recenti, ultimo_indice, misure, spazio = [], 0, [], False
    cv2.namedWindow(FINESTRA, cv2.WINDOW_NORMAL)
    try:
        while True:
            with lettore._cond:
                fot = lettore.buffer[-1] if lettore.buffer else None
            tasto = cv2.waitKey(5) & 0xFF
            if tasto == 27:
                break
            spazio = spazio or tasto == ord(" ")
            if fot is None or fot.indice == ultimo_indice:
                continue
            ultimo_indice = fot.indice
            vista = fot.immagine.copy()
            r = rileva(detector, board, fot.immagine)
            testi = ["Misura con il metro la distanza tra l'obiettivo e il centro del pattern,",
                     "tieni il pattern fermo e quasi frontale, poi SPAZIO.   ESC = esci"]
            if r is None:
                recenti.clear()
                testi.append("pattern non trovato")
            else:
                ok, rvec, tvec = cv2.solvePnP(r[0], r[1], K, dist)
                R, _ = cv2.Rodrigues(rvec)
                recenti = (recenti + [float(np.linalg.norm(R @ centro + tvec.ravel()))])[-VERIFICA_FOTOGRAMMI:]
                cv2.aruco.drawDetectedCornersCharuco(vista, r[1].reshape(-1, 1, 2).astype(np.float32),
                                                     r[2].reshape(-1, 1), (255, 0, 0))
                testi.append(f"distanza stimata {np.mean(recenti):.3f} m  (dev.std {np.std(recenti) * 1000:.1f} mm "
                             f"su {len(recenti)} fotogrammi)   inclinazione {inclinazione_gradi(rvec):.0f} gradi")
                if spazio and len(recenti) == VERIFICA_FOTOGRAMMI:
                    spazio = False
                    misure.append(np.mean(recenti))
                    print(f"Misura {len(misure)}: distanza stimata {misure[-1]:.3f} m")
            for riga, testo in enumerate(testi):
                cv2.putText(vista, testo, (10, 40 + 34 * riga), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(vista, testo, (10, 40 + 34 * riga), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(FINESTRA, cv2.resize(vista, None, fx=SCALA_VISTA, fy=SCALA_VISTA, interpolation=cv2.INTER_AREA))
    finally:
        lettore.chiudi()
        cv2.destroyAllWindows()
    if misure:
        with open(os.path.join(CARTELLA, f"{camera}_distanze.csv"), "a", encoding="utf-8") as f:
            f.writelines(f"{time.strftime('%Y-%m-%d %H:%M')};{m:.4f};\n" for m in misure)
        print(f"Misure aggiunte a {os.path.join(CARTELLA, camera)}_distanze.csv (data;stimata_m;metro_m): "
              "scrivi nella terza colonna i valori del metro")


def main():
    parser = argparse.ArgumentParser(description="Calibrazione monoculare (test T06)")
    parser.add_argument("--camera", required=True, choices=list(STREAM), help="camera da calibrare")
    parser.add_argument("--pattern", default="A3", help="pattern di charuco_boards.yaml (default A3)")
    parser.add_argument("--solo-calcolo", action="store_true", help="non acquisire: calcola dalle immagini salvate")
    parser.add_argument("--nuova", action="store_true", help="cancella le immagini già salvate per questa camera")
    parser.add_argument("--verifica-distanza", action="store_true",
                        help="confronta la distanza stimata con quella misurata con il metro")
    args = parser.parse_args()

    board, valori = leggi_pattern(args.pattern)
    detector = cv2.aruco.CharucoDetector(board)
    cartella = os.path.join(CARTELLA, args.camera)
    print(f"Pattern {args.pattern}: quadrato {valori['lato_x'] * 1000:.2f} x {valori['lato_y'] * 1000:.2f} mm, "
          f"marker {float(valori['marker_length_m']) * 1000:.2f} mm")
    if args.verifica_distanza:
        verifica_distanza(args.camera, board, detector, valori)
        return
    if args.nuova:
        for p in glob.glob(os.path.join(cartella, "img_*.png")):
            os.remove(p)
    if not args.solo_calcolo and not acquisisci(args.camera, cartella, board, detector):
        print("Uscita senza calcolo. Le immagini salvate restano: si riparte da lì al prossimo avvio.")
        return
    righe, _ = calcola(args.camera, cartella, board, detector, args.pattern)
    testo = "\n".join(righe)
    print("\n" + testo)
    os.makedirs(CARTELLA, exist_ok=True)
    with open(os.path.join(CARTELLA, f"{args.camera}_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()