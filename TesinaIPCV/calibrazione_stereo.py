"""
Fase 5 - Calibrazione stereo (test T07 e T08; requisiti RF5, RF6, RC2, RC3, RC4).

Stima la posa della camera 2 rispetto alla camera 1 (R, T) e le matrici E ed F, con i parametri
intrinseci fissi a quelli della calibrazione monoculare (data/calibrazione/<camera>.yaml).

Due passi nello stesso programma:
  1. ACQUISIZIONE - le due camere affiancate con il pattern A1. Il programma salva da solo una coppia
     quando il pattern è fermo e visibile in entrambe le camere, in una posizione nuova. Una coppia su
     cinque viene tenuta da parte per la verifica (test T08) e non entra nella calibrazione.
  2. CALCOLO - cv2.stereoCalibrateExtended, controlli dei test T07 e T08, salvataggio in
     data/calibrazione/stereo.yaml

Uso:
  python calibrazione_stereo.py --baseline 2.445                 acquisizione, poi calcolo
  python calibrazione_stereo.py --baseline 2.445 --solo-calcolo  ricalcola dalle coppie già salvate
  python calibrazione_stereo.py --baseline 2.445 --nuova         cancella le coppie salvate e ricomincia
  (--baseline = distanza tra gli obiettivi misurata con il metro, in metri)
Tasti: SPAZIO = salva subito (se il pattern è fermo)   C = calcola   ESC = esci senza calcolare
Prima di avviarlo: imposta_camere.py -> PRONTO, sincronizzazione misurata nella sessione.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from acquisizione import AcquisizioneStereo, ROTAZIONE, STREAM
from calibrazione_mono import (leggi_pattern, rileva, leggi_calibrazione, Copertura, inclinazione_gradi,
                               ANGOLI_MINIMI, CARTELLA)
from test_latenza import affianca

CARTELLA_COPPIE = os.path.join(CARTELLA, "stereo")
FILE_RISULTATO = os.path.join(CARTELLA, "stereo.yaml")
FINESTRA = "Calibrazione stereo  (SPAZIO = salva, C = calcola, ESC = esci)"

COPPIE_OBIETTIVO = 25        # di cui una su cinque per la verifica
OGNI_QUANTE_VERIFICA = 5     # la 5a, 10a, 15a... coppia salvata va nel gruppo di verifica
FERMO_COPPIE = 6             # il pattern deve restare fermo per tante coppie consecutive...
FERMO_PX = 1.0               # ...con spostamento massimo degli angoli inferiore a tanto in entrambe le camere
NUOVA_POSA_PX = 30           # spostamento medio minimo degli angoli (in almeno una camera) rispetto alle coppie salvate
NUOVA_INCLINAZIONE = 10      # oppure orientamento diverso di almeno tanti gradi
PAUSA_S = 1.5                # tempo minimo tra due salvataggi automatici
CRITERI = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)

# Criteri dei test T07 e T08
RMS_MAX = 1.0                # RC2
BASELINE_TOLLERANZA_M = 0.02 # RC3
ANGOLO_ATTESO = (40, 60)     # gradi: con la disposizione a 50 gradi
EPIPOLARE_MAX = 1.0          # RC4: media in pixel sulle coppie di verifica


def angoli_comuni(r1, r2):
    """Dei due rilevamenti tiene solo gli angoli visti da ENTRAMBE le camere, nello stesso ordine."""
    oggetto1, punti1, ids1 = r1
    _, punti2, ids2 = r2
    comuni = np.intersect1d(ids1, ids2)
    i1 = [int(np.flatnonzero(ids1 == i)[0]) for i in comuni]
    i2 = [int(np.flatnonzero(ids2 == i)[0]) for i in comuni]
    return oggetto1[i1].astype(np.float32), punti1[i1].astype(np.float32), punti2[i2].astype(np.float32), comuni


# ---------------------------------------------------------------------------------------
# 1. Acquisizione
# ---------------------------------------------------------------------------------------

def acquisisci(board, detector, calibrazioni):
    os.makedirs(CARTELLA_COPPIE, exist_ok=True)
    acq = AcquisizioneStereo()
    nomi = [l.nome for l in acq.lettori]
    salvate = sorted(glob.glob(os.path.join(CARTELLA_COPPIE, f"coppia_*_{nomi[0]}.png")))
    coperture, pose_salvate = {}, []        # pose: per camera (dizionario id -> punto, inclinazione)
    for p1 in salvate:
        immagini = [cv2.imread(p1), cv2.imread(p1.replace(nomi[0], nomi[1]))]
        rilevamenti = [rileva(detector, board, im) for im in immagini]
        for nome, im, r in zip(nomi, immagini, rilevamenti):
            coperture.setdefault(nome, Copertura(im.shape[1], im.shape[0]))
            if r is not None:
                coperture[nome].aggiungi(r[1])
        if all(r is not None for r in rilevamenti):
            pose_salvate.append([(dict(zip(r[2], r[1])), posa_inclinazione(r, calibrazioni[n]))
                                 for n, r in zip(nomi, rilevamenti)])
    storia, ultimo_salvataggio, spazio = [], 0.0, False
    cv2.namedWindow(FINESTRA, cv2.WINDOW_NORMAL)
    try:
        while True:
            tasto = cv2.waitKey(1) & 0xFF
            if tasto == 27:
                return False
            if tasto in (ord("c"), ord("C")):
                return True
            spazio = spazio or tasto == ord(" ")
            coppia = acq.prossima_coppia(timeout_s=0.05)
            if coppia is None:
                continue
            immagini = [coppia.cam1.immagine, coppia.cam2.immagine]
            rilevamenti = [rileva(detector, board, im) for im in immagini]
            viste = []
            for nome, im, r in zip(nomi, immagini, rilevamenti):
                coperture.setdefault(nome, Copertura(im.shape[1], im.shape[0]))
                vista = coperture[nome].disegna(im)
                if r is not None:
                    cv2.aruco.drawDetectedCornersCharuco(vista, r[1].reshape(-1, 1, 2).astype(np.float32),
                                                         r[2].reshape(-1, 1), (255, 0, 0))
                viste.append(vista)
            fermo, adatta = False, False
            if any(r is None for r in rilevamenti):
                storia.clear()
                mancanti = [n for n, r in zip(nomi, rilevamenti) if r is None]
                stato = f"pattern non trovato in: {', '.join(mancanti)} (servono almeno {ANGOLI_MINIMI} angoli)"
            else:
                comuni = angoli_comuni(*rilevamenti)[3]
                attuale = [dict(zip(r[2], r[1])) for r in rilevamenti]
                storia = storia[-(FERMO_COPPIE - 1):] + [attuale]
                fermo = len(storia) == FERMO_COPPIE and all(
                    len(set(s[c]) & set(attuale[c])) >= ANGOLI_MINIMI and
                    max(np.linalg.norm(s[c][i] - attuale[c][i]) for i in set(s[c]) & set(attuale[c])) < FERMO_PX
                    for s in storia[:-1] for c in range(2))
                inclinazioni = [posa_inclinazione(r, calibrazioni[n]) for n, r in zip(nomi, rilevamenti)]
                nuova = all(any(
                    len(set(p) & set(attuale[c])) < 6 or
                    np.mean([np.linalg.norm(p[i] - attuale[c][i]) for i in set(p) & set(attuale[c])]) > NUOVA_POSA_PX or
                    abs(inclinazioni[c] - incl) > NUOVA_INCLINAZIONE
                    for c, (p, incl) in enumerate(salvata)) for salvata in pose_salvate)
                abbastanza = len(comuni) >= ANGOLI_MINIMI
                adatta = fermo and nuova and abbastanza
                if not abbastanza:
                    stato = f"solo {len(comuni)} angoli visti da entrambe: gira il pattern verso il centro tra le camere"
                elif not fermo:
                    stato = "tieni il pattern fermo"
                elif not nuova:
                    stato = "posizione già salvata: spostalo, alzalo/abbassalo o ruotalo"
                else:
                    stato = "ok"
                stato += f"   ({len(comuni)} angoli comuni, inclinazioni {inclinazioni[0]:.0f} e {inclinazioni[1]:.0f} gradi)"
            adesso = time.perf_counter()
            if not any(r is None for r in rilevamenti) and (
                    (adatta and adesso - ultimo_salvataggio > PAUSA_S) or (spazio and fermo)):
                numero = len(glob.glob(os.path.join(CARTELLA_COPPIE, f"coppia_*_{nomi[0]}.png")))
                for nome, im in zip(nomi, immagini):
                    cv2.imwrite(os.path.join(CARTELLA_COPPIE, f"coppia_{numero:03d}_{nome}.png"), im)
                for nome, r in zip(nomi, rilevamenti):
                    coperture[nome].aggiungi(r[1])
                pose_salvate.append(list(zip(attuale, inclinazioni)))
                ultimo_salvataggio = adesso
                viste = [cv2.addWeighted(v, 0.5, np.full_like(v, 255), 0.5, 0) for v in viste]   # lampo bianco
            if any(r is None for r in rilevamenti) or fermo:
                spazio = False
            n = len(pose_salvate)
            testi = [f"coppie salvate {n}/{COPPIE_OBIETTIVO}   copertura {nomi[0]} {coperture[nomi[0]].frazione() * 100:.0f}%, "
                     f"{nomi[1]} {coperture[nomi[1]].frazione() * 100:.0f}%", stato]
            if n >= COPPIE_OBIETTIVO:
                testi.append("Obiettivo raggiunto: premi C per calcolare")
            cv2.imshow(FINESTRA, affianca(viste, testi, altezza=640))
    finally:
        acq.chiudi()
        cv2.destroyAllWindows()


def posa_inclinazione(r, calibrazione):
    """Inclinazione del pattern rispetto all'asse ottico, con la calibrazione monoculare della camera."""
    K, dist = calibrazione
    ok, rvec, _ = cv2.solvePnP(r[0], r[1], K, dist)
    return inclinazione_gradi(rvec) if ok else 0.0


# ---------------------------------------------------------------------------------------
# 2. Calcolo e controlli
# ---------------------------------------------------------------------------------------

def errore_epipolare(F, punti1, punti2, calibrazioni):
    """Distanza in pixel di ogni punto dalla linea epipolare del punto corrispondente, nelle due
    immagini (media delle due). I punti vengono prima corretti dalla distorsione restando in pixel."""
    (K1, d1), (K2, d2) = calibrazioni
    u1 = cv2.undistortPoints(punti1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
    u2 = cv2.undistortPoints(punti2.reshape(-1, 1, 2), K2, d2, P=K2).reshape(-1, 2)
    h1 = np.column_stack([u1, np.ones(len(u1))])
    h2 = np.column_stack([u2, np.ones(len(u2))])
    linee2 = h1 @ F.T                       # linee nell'immagine 2: F u1
    linee1 = h2 @ F                         # linee nell'immagine 1: F^T u2
    d_2 = np.abs(np.sum(h2 * linee2, axis=1)) / np.hypot(linee2[:, 0], linee2[:, 1])
    d_1 = np.abs(np.sum(h1 * linee1, axis=1)) / np.hypot(linee1[:, 0], linee1[:, 1])
    return (d_1 + d_2) / 2


# Affinamento: con molte coppie del pattern A1 a 2-4 m, la calibrazione stereo può correggere anche
# focali e punti principali (non la distorsione, che si misura bene solo con il pattern vicino ai
# bordi dell'immagine, come nella calibrazione monoculare). La lunghezza di T, misurata con il metro,
# resta un controllo indipendente perché non entra nel calcolo.
FLAG_AFFINA = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
               cv2.CALIB_FIX_TANGENT_DIST)


def calibra(oggetti, punti1, punti2, calibrazioni, dimensione, affina=False):
    """Restituisce un dizionario con rms, R, T, E, F, le due calibrazioni usate e gli errori per coppia."""
    (K1, d1), (K2, d2) = calibrazioni
    risultato = cv2.stereoCalibrateExtended(
        oggetti, punti1, punti2, K1.copy(), d1.copy(), K2.copy(), d2.copy(), dimensione, None, None,
        flags=FLAG_AFFINA if affina else cv2.CALIB_FIX_INTRINSIC, criteria=CRITERI)
    rms, K1n, d1n, K2n, d2n, R, T, E, F, _, _, errori = risultato
    return {"rms": rms, "R": R, "T": T.ravel(), "E": E, "F": F, "calibrazioni": [(K1n, d1n), (K2n, d2n)],
            "errori": np.asarray(errori).reshape(len(oggetti), -1)}


def stima(calib, calibrazioni, dimensione, affina, righe):
    """Calibra, toglie le coppie anomale e ricalibra. Restituisce (risultato, coppie usate)."""
    ris = calibra(calib["oggetti"], calib["p1"], calib["p2"], calibrazioni, dimensione, affina)
    # Coppie anomale: errore molto più alto delle altre (pattern mosso tra le due immagini, angoli sbagliati)
    errore_coppia = ris["errori"].max(axis=1)
    soglia = max(3 * np.median(errore_coppia), 1.5)
    anomale = [i for i, e in enumerate(errore_coppia) if e > soglia]
    if anomale:
        for i in anomale:
            righe.append(f"  {calib['nomi'][i]}: errore {errore_coppia[i]:.2f} px, oltre {soglia:.2f} px: esclusa")
        tieni = [i for i in range(len(calib["oggetti"])) if i not in anomale]
        calib = {k: [v[i] for i in tieni] for k, v in calib.items()}
        ris = calibra(calib["oggetti"], calib["p1"], calib["p2"], calibrazioni, dimensione, affina)
    return ris, calib


def analisi_per_coppia(gruppo, errori, calibrazioni, nomi, baseline_m, centro_pattern):
    """Stima R e T da OGNI coppia separatamente (posa del pattern in ciascuna camera con solvePnP) e
    le confronta nell'ordine in cui le coppie sono state salvate. Se un telefono si è mosso durante
    l'acquisizione, angolo e baseline cambiano di colpo da un certo punto in poi, e gli errori salgono."""
    righe_csv, basi, distanze, angoli = [], [], [], []
    for nome, o, a, b, err in zip(gruppo["nomi"], gruppo["oggetti"], gruppo["p1"], gruppo["p2"], errori):
        pose = []
        for punti, (K, d) in zip((a, b), calibrazioni):
            ok, rvec, tvec = cv2.solvePnP(o, punti, K, d)
            R_i, _ = cv2.Rodrigues(rvec)
            pose.append((R_i, tvec.ravel()))
        (R1, t1), (R2, t2) = pose
        R_c = R2 @ R1.T
        T_c = t2 - R_c @ t1
        d1, d2 = np.linalg.norm(R1 @ centro_pattern + t1), np.linalg.norm(R2 @ centro_pattern + t2)
        basi.append(np.linalg.norm(T_c))
        angoli.append(np.degrees(np.linalg.norm(cv2.Rodrigues(R_c)[0])))
        distanze.append((d1, d2))
        righe_csv.append(f"{nome};{d1:.3f};{d2:.3f};{basi[-1]:.4f};{angoli[-1]:.2f};{err[0]:.3f};{err[1]:.3f}\n")
    with open(os.path.join(CARTELLA, "stereo_per_coppia.csv"), "w") as f:
        f.write(f"coppia;distanza_{nomi[0]}_m;distanza_{nomi[1]}_m;baseline_m;angolo_gradi;"
                f"errore_{nomi[0]}_px;errore_{nomi[1]}_px\n")
        f.writelines(righe_csv)
    basi, angoli, errori = np.array(basi), np.array(angoli), np.asarray(errori)
    righe = ["", "Analisi per coppia, nell'ordine di acquisizione (pose stimate separatamente in ogni camera):",
             f"  {'coppie':10s} {'baseline':>9s} {'angolo':>8s} {'err. ' + nomi[0]:>12s} {'err. ' + nomi[1]:>12s}"]
    for blocco in np.array_split(np.arange(len(basi)), 4):
        righe.append(f"  {gruppo['nomi'][blocco[0]][-3:]}-{gruppo['nomi'][blocco[-1]][-3:]}    {np.median(basi[blocco]):9.3f} "
                     f"{np.median(angoli[blocco]):8.2f} {errori[blocco, 0].mean():12.2f} {errori[blocco, 1].mean():12.2f}")
    righe.append("  (un salto netto di angolo o baseline tra un blocco e l'altro indica un telefono mosso)")
    return righe


def calcola(board, detector, calibrazioni, nomi, baseline_m, affina=False, centro_pattern=None):
    righe = [f"Calibrazione stereo - {time.strftime('%Y-%m-%d %H:%M')} - camera 1 {nomi[0]}, camera 2 {nomi[1]}", ""]
    percorsi = sorted(glob.glob(os.path.join(CARTELLA_COPPIE, f"coppia_*_{nomi[0]}.png")))
    calib = {"oggetti": [], "p1": [], "p2": [], "nomi": []}
    verif = {"oggetti": [], "p1": [], "p2": [], "nomi": []}
    dimensione = None
    for indice, p1 in enumerate(percorsi):
        immagini = [cv2.imread(p1), cv2.imread(p1.replace(f"_{nomi[0]}.png", f"_{nomi[1]}.png"))]
        dimensione = (immagini[0].shape[1], immagini[0].shape[0])
        rilevamenti = [rileva(detector, board, im) for im in immagini]
        nome = os.path.basename(p1).replace(f"_{nomi[0]}.png", "")
        if any(r is None for r in rilevamenti):
            righe.append(f"  {nome}: pattern non trovato in una delle due immagini, esclusa")
            continue
        oggetto, q1, q2, comuni = angoli_comuni(*rilevamenti)
        if len(comuni) < ANGOLI_MINIMI:
            righe.append(f"  {nome}: solo {len(comuni)} angoli comuni, esclusa")
            continue
        gruppo = verif if (indice + 1) % OGNI_QUANTE_VERIFICA == 0 else calib
        for chiave, valore in (("oggetti", oggetto), ("p1", q1), ("p2", q2), ("nomi", nome)):
            gruppo[chiave].append(valore)
    if len(calib["oggetti"]) < 8:
        righe.append(f"Solo {len(calib['oggetti'])} coppie per la calibrazione: ne servono almeno 8 (meglio 20).")
        return righe, False

    # Diagnosi: le due varianti, intrinseci fissi e focali/punti principali affinati, a confronto.
    # Se affinando scendono sia l'RMS sia la differenza dalla baseline del metro, le calibrazioni
    # monoculari avevano focali un po' sbagliate. Se scende l'RMS ma la baseline resta sbagliata,
    # il sospetto è il lato dei quadrati del pattern. Se l'RMS resta alto, qualcosa si è mosso.
    varianti = {}
    for nome_v, aff in (("intrinseci fissi", False), ("focali affinate", True)):
        ris, usate = stima(calib, calibrazioni, dimensione, aff, righe if aff == affina else [])
        epi = np.concatenate([errore_epipolare(ris["F"], a, b, ris["calibrazioni"])
                              for a, b in zip(verif["p1"], verif["p2"])]) if verif["oggetti"] else np.array([np.nan])
        varianti[nome_v] = (ris, usate, epi)
    righe += ["", "Confronto (diagnosi):",
              f"  {'variante':18s} {'RMS':>7s} {'err. ' + nomi[0]:>12s} {'err. ' + nomi[1]:>12s} {'|T|':>8s} "
              f"{'diff. metro':>11s} {'epipolare verifica':>19s}   focali {nomi[0]} / {nomi[1]}"]
    for nome_v, (ris, _, epi) in varianti.items():
        (K1v, _), (K2v, _) = ris["calibrazioni"]
        righe.append(f"  {nome_v:18s} {ris['rms']:7.3f} {ris['errori'][:, 0].mean():12.3f} {ris['errori'][:, 1].mean():12.3f} "
                     f"{np.linalg.norm(ris['T']):8.3f} {(np.linalg.norm(ris['T']) - baseline_m) * 100:+9.1f} cm "
                     f"{np.mean(epi):15.3f} px   {(K1v[0, 0] + K1v[1, 1]) / 2:.1f} / {(K2v[0, 0] + K2v[1, 1]) / 2:.1f}")
    if centro_pattern is not None:
        righe += analisi_per_coppia(varianti["intrinseci fissi"][1], varianti["intrinseci fissi"][0]["errori"],
                                    calibrazioni, nomi, baseline_m, centro_pattern)
    scelta = "focali affinate" if affina else "intrinseci fissi"
    ris, calib, epi_verif = varianti[scelta]
    righe.append(f"  -> si usa: {scelta}" + ("" if affina else "  (per usare l'altra: --affina)"))
    R, T, E, F, rms = ris["R"], ris["T"], ris["E"], ris["F"], ris["rms"]
    calibrazioni = ris["calibrazioni"]
    if not verif["oggetti"]:
        epi_verif = np.array([])
    lunghezza = float(np.linalg.norm(T))
    angolo = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    centro2 = -R.T @ T                         # centro della camera 2 nel sistema della camera 1
    epi_calib = np.concatenate([errore_epipolare(F, a, b, calibrazioni) for a, b in zip(calib["p1"], calib["p2"])])

    controlli = [
        ("RMS stereo", f"{rms:.3f} px", f"< {RMS_MAX} px", rms < RMS_MAX),
        ("lunghezza di T (baseline)", f"{lunghezza:.3f} m (metro {baseline_m:.3f} m, "
         f"differenza {(lunghezza - baseline_m) * 100:+.1f} cm)", f"entro ±{BASELINE_TOLLERANZA_M * 100:.0f} cm",
         abs(lunghezza - baseline_m) <= BASELINE_TOLLERANZA_M),
        ("angolo di R", f"{angolo:.1f} gradi", f"tra {ANGOLO_ATTESO[0]} e {ANGOLO_ATTESO[1]} gradi",
         ANGOLO_ATTESO[0] <= angolo <= ANGOLO_ATTESO[1]),
        # La camera 2 può stare a destra o a sinistra della camera 1: conta solo che le camere siano
        # affiancate (spostamento orizzontale maggiore di quello verticale). Il lato viene solo riportato.
        (f"{nomi[1]} affiancata a {nomi[0]}",
         f"x = {centro2[0]:+.3f} m, y = {centro2[1]:+.3f} m ({'a destra' if centro2[0] > 0 else 'a sinistra'})",
         "|x| > |y|", abs(centro2[0]) > abs(centro2[1])),
    ]
    if len(epi_verif):
        controlli.append(("errore epipolare, coppie di verifica", f"media {epi_verif.mean():.3f} px, "
                          f"95% sotto {np.percentile(epi_verif, 95):.3f}", f"media < {EPIPOLARE_MAX} px",
                          epi_verif.mean() < EPIPOLARE_MAX))
    righe += ["", f"Coppie usate: {len(calib['oggetti'])} per la calibrazione, {len(verif['oggetti'])} per la verifica",
              "", "Risultato:",
              f"  R (vettore di Rodrigues, gradi): {np.degrees(cv2.Rodrigues(R)[0].ravel()).round(2).tolist()}",
              f"  T (m): {T.round(4).tolist()}",
              f"  centro della camera 2 visto dalla camera 1 (m): {centro2.round(3).tolist()}",
              f"  errore epipolare sulle coppie di calibrazione: media {epi_calib.mean():.3f} px", "",
              "Controlli dei test T07 e T08:"]
    for nome, valore, criterio, ok in controlli:
        righe.append(f"  {'OK ' if ok else 'NO '} {nome:38s} {valore:48s} criterio: {criterio}")
    esito = all(ok for *_, ok in controlli) and len(epi_verif) > 0
    if not len(epi_verif):
        righe.append("  Nessuna coppia di verifica: servono almeno 5 coppie salvate per il test T08")

    # Si salvano entrambe le varianti (per confrontarle con misure_pavimento.py) e quella scelta
    for percorso, (ris_v, usate_v, _), nome_v in (
            (os.path.join(CARTELLA, "stereo_fissa.yaml"), varianti["intrinseci fissi"], "intrinseci fissi"),
            (os.path.join(CARTELLA, "stereo_affinata.yaml"), varianti["focali affinate"], "focali affinate"),
            (FILE_RISULTATO, varianti[scelta], scelta)):
        salva(percorso, ris_v, nomi, dimensione, baseline_m, len(usate_v["oggetti"]), nome_v)
    righe += ["", f"Salvato: {FILE_RISULTATO} (variante '{scelta}'); entrambe le varianti in stereo_fissa.yaml "
              "e stereo_affinata.yaml",
              f"ESITO T07 e T08: {'SUPERATO' if esito else 'NON SUPERATO'}"]
    return righe, esito


def salva(percorso, ris, nomi, dimensione, baseline_m, coppie, variante):
    (K1, d1), (K2, d2) = ris["calibrazioni"]
    fs = cv2.FileStorage(percorso, cv2.FILE_STORAGE_WRITE)
    fs.write("data", time.strftime("%Y-%m-%d %H:%M"))
    fs.write("camera1", nomi[0])
    fs.write("camera2", nomi[1])
    fs.write("larghezza", dimensione[0])
    fs.write("altezza", dimensione[1])
    fs.write("rotazione1", str(ROTAZIONE.get(nomi[0])))
    fs.write("rotazione2", str(ROTAZIONE.get(nomi[1])))
    for chiave, valore in (("K1", K1), ("dist1", d1), ("K2", K2), ("dist2", d2), ("R", ris["R"]),
                           ("T", ris["T"].reshape(3, 1)), ("E", ris["E"]), ("F", ris["F"])):
        fs.write(chiave, valore)
    fs.write("rms_px", float(ris["rms"]))
    fs.write("baseline_metro_m", float(baseline_m))
    fs.write("coppie", coppie)
    fs.write("intrinseci", variante)
    fs.release()


def main():
    parser = argparse.ArgumentParser(description="Calibrazione stereo (test T07 e T08)")
    parser.add_argument("--baseline", type=float, required=True,
                        help="distanza tra gli obiettivi misurata con il metro, in metri")
    parser.add_argument("--pattern", default="A1", help="pattern di charuco_boards.yaml (default A1)")
    parser.add_argument("--solo-calcolo", action="store_true", help="non acquisire: calcola dalle coppie salvate")
    parser.add_argument("--nuova", action="store_true", help="cancella le coppie già salvate")
    parser.add_argument("--affina", action="store_true",
                        help="salva la variante con focali e punti principali affinati dalla calibrazione stereo")
    args = parser.parse_args()

    board, valori = leggi_pattern(args.pattern)
    detector = cv2.aruco.CharucoDetector(board)
    nomi = list(STREAM)                         # camera 1, camera 2
    calibrazioni = {n: leggi_calibrazione(n) for n in nomi}
    print(f"Pattern {args.pattern}: quadrato {valori['lato_x'] * 1000:.2f} x {valori['lato_y'] * 1000:.2f} mm")
    if args.nuova:
        for p in glob.glob(os.path.join(CARTELLA_COPPIE, "coppia_*.png")):
            os.remove(p)
    if not args.solo_calcolo and not acquisisci(board, detector, calibrazioni):
        print("Uscita senza calcolo. Le coppie salvate restano: si riparte da lì al prossimo avvio.")
        return
    centro = np.array([int(valori["squares_x"]) * valori["lato_x"] / 2, int(valori["squares_y"]) * valori["lato_y"] / 2, 0.0])
    righe, _ = calcola(board, detector, [calibrazioni[n] for n in nomi], nomi, args.baseline, args.affina, centro)
    testo = "\n".join(righe)
    print("\n" + testo)
    with open(os.path.join(CARTELLA, "stereo_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()