"""
Fase 6 - Rettificazione (test T09; requisiti RF7 e RC5).

Trasforma le immagini delle due camere come se le camere fossero parallele, affiancate e orientate
allo stesso modo (teoria, sezione 5.10). Dopo la rettificazione lo stesso punto della scena cade
sulla STESSA RIGA nelle due immagini: lo scarto verticale tra i due pixel deve essere circa zero.

Il test T09 misura questo scarto sugli angoli del pattern A1 nelle coppie salvate dalla calibrazione
stereo (data/calibrazione/stereo). Quelle immagini sono state scattate con le camere esattamente
nella posizione descritta da stereo.yaml, quindi il test vale anche se nel frattempo un telefono si
è mosso. Criterio (RC5): scarto verticale medio <= 1 px sulle coppie di verifica.

Le camere del progetto convergono di circa 50 gradi: ogni immagine viene ruotata virtualmente di
circa 25 gradi e risulta molto deformata (teoria, sezione 5.12). Per questo ogni immagine rettificata
viene centrata sul proprio contenuto: le due immagini condividono focale e riga del centro ottico
(quello che serve perché le righe coincidano), ma non la colonna del centro ottico.

Uso:
  python rettificazione.py                 test T09 sulle coppie della calibrazione stereo
  python rettificazione.py --vivo          vista dal vivo rettificata (solo con camere ferme dopo la calibrazione)
Tasti nella finestra:  N = coppia successiva   P = precedente   S = salva l'immagine   ESC = esci
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from acquisizione import AcquisizioneStereo, STREAM
from calibrazione_mono import leggi_pattern, rileva, ANGOLI_MINIMI, CARTELLA
from calibrazione_stereo import angoli_comuni, CARTELLA_COPPIE, OGNI_QUANTE_VERIFICA

FILE_STEREO = os.path.join(CARTELLA, "stereo.yaml")
CARTELLA_USCITA = os.path.join("data", "rettificazione")
FINESTRA = "Rettificazione  (N/P = coppia, S = salva, ESC = esci)"
SCARTO_MAX_PX = 1.0          # RC5: scarto verticale medio sulle coppie di verifica
PASSO_RIGHE_PX = 40          # distanza tra le righe orizzontali di riferimento disegnate
SCALA_VISTA = 0.5            # la vista affiancata (2 x 720 x 1280) ridotta per stare sullo schermo


# ---------------------------------------------------------------------------------------
# 1. Calcolo della rettificazione
# ---------------------------------------------------------------------------------------

def leggi_stereo(percorso=FILE_STEREO):
    """Parametri della calibrazione stereo: K, distorsione, R, T e dimensione delle immagini."""
    fs = cv2.FileStorage(percorso, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Manca {percorso}: prima esegui calibrazione_stereo.py")
    cal = {k: fs.getNode(k).mat() for k in ("K1", "dist1", "K2", "dist2", "R", "T")}
    cal["dimensione"] = (int(fs.getNode("larghezza").real()), int(fs.getNode("altezza").real()))
    cal["data"] = fs.getNode("data").string()
    fs.release()
    return cal


def prepara(cal):
    """Rotazioni virtuali (R1, R2), nuove matrici di proiezione (P1, P2) e mappe per cv2.remap.

    cv2.stereoRectify calcola le due rotazioni che rendono le camere parallele. Le nuove P però
    metterebbero il centro ottico nella stessa colonna per entrambe le camere, e con camere così
    convergenti ognuna delle due immagini finirebbe quasi tutta fuori dal riquadro. Qui si sceglie
    una focale unica che faccia stare entrambe le immagini nel riquadro, una riga del centro ottico
    unica (necessaria perché le righe coincidano) e una colonna diversa per ogni camera."""
    larghezza, altezza = cal["dimensione"]
    R1, R2, P1, _, _, _, _ = cv2.stereoRectify(cal["K1"], cal["dist1"], cal["K2"], cal["dist2"],
                                               cal["dimensione"], cal["R"], cal["T"].reshape(3, 1), alpha=-1)
    f = P1[0, 0]
    # dove finisce il bordo di ogni immagine originale dopo la rotazione virtuale (focale f, centro in 0)
    s = np.linspace(0, 1, 60)
    bordo = np.concatenate([np.c_[s * (larghezza - 1), 0 * s], np.c_[s * (larghezza - 1), 0 * s + altezza - 1],
                            np.c_[0 * s, s * (altezza - 1)], np.c_[0 * s + larghezza - 1, s * (altezza - 1)]])
    P0 = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    estensioni = []
    for K, d, Ri in ((cal["K1"], cal["dist1"], R1), (cal["K2"], cal["dist2"], R2)):
        p = cv2.undistortPoints(bordo.reshape(-1, 1, 2).astype(np.float64), K, d, R=Ri, P=P0).reshape(-1, 2)
        estensioni.append((p.min(0), p.max(0)))
    y_min = min(e[0][1] for e in estensioni)
    y_max = max(e[1][1] for e in estensioni)
    larghezza_max = max(e[1][0] - e[0][0] for e in estensioni)
    scala = min(larghezza / larghezza_max, altezza / (y_max - y_min))
    f_nuova = f * scala
    cy = altezza / 2 - scala * (y_min + y_max) / 2
    rett = {"R1": R1, "R2": R2, "f": f_nuova, "cy": cy, "dimensione": (larghezza, altezza)}
    for i, (K, d, Ri) in enumerate(((cal["K1"], cal["dist1"], R1), (cal["K2"], cal["dist2"], R2)), start=1):
        cx = larghezza / 2 - scala * (estensioni[i - 1][0][0] + estensioni[i - 1][1][0]) / 2
        Pi = np.array([[f_nuova, 0, cx], [0, f_nuova, cy], [0, 0, 1.0]])
        rett[f"P{i}"] = Pi
        rett[f"mappa{i}"] = cv2.initUndistortRectifyMap(K, d, Ri, Pi, (larghezza, altezza), cv2.CV_32FC1)
    return rett


def rettifica_immagini(rett, im1, im2):
    """Le due immagini rettificate (cv2.remap con le mappe preparate una volta sola)."""
    return [cv2.remap(im, *rett[f"mappa{i}"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            for i, im in ((1, im1), (2, im2))]


def rettifica_punti(rett, cal, punti, camera):
    """Coordinate rettificate di punti dell'immagine originale (camera = 1 o 2)."""
    K, d = cal[f"K{camera}"], cal[f"dist{camera}"]
    return cv2.undistortPoints(np.asarray(punti, np.float64).reshape(-1, 1, 2), K, d,
                               R=rett[f"R{camera}"], P=rett[f"P{camera}"]).reshape(-1, 2)


def scarti_verticali(rett, cal, rilevamenti):
    """Per gli angoli visti da entrambe le camere: y nella camera 1 - y nella camera 2, dopo la
    rettificazione. Restituisce (scarti in px, punti rettificati 1, punti rettificati 2, id) o None."""
    if any(r is None for r in rilevamenti):
        return None
    _, q1, q2, comuni = angoli_comuni(*rilevamenti)
    if len(comuni) < ANGOLI_MINIMI:
        return None
    a = rettifica_punti(rett, cal, q1, 1)
    b = rettifica_punti(rett, cal, q2, 2)
    return a[:, 1] - b[:, 1], a, b, comuni


# ---------------------------------------------------------------------------------------
# 2. Disegno
# ---------------------------------------------------------------------------------------

def colore(i):
    """Un colore diverso e stabile per ogni id di angolo (lo stesso nelle due immagini)."""
    tinta = np.uint8([[[int(i) * 37 % 180, 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(tinta, cv2.COLOR_HSV2BGR)[0, 0])


def vista_affiancata(r1, r2, misura=None, titolo=""):
    """Le due immagini rettificate affiancate, con righe orizzontali ogni PASSO_RIGHE_PX pixel e,
    se c'è il pattern, gli angoli con lo stesso colore nelle due immagini."""
    vista = cv2.hconcat([r1, r2])
    larghezza = r1.shape[1]
    for y in range(PASSO_RIGHE_PX // 2, vista.shape[0], PASSO_RIGHE_PX):
        cv2.line(vista, (0, y), (vista.shape[1] - 1, y), (0, 200, 0), 1, cv2.LINE_AA)
    cv2.line(vista, (larghezza, 0), (larghezza, vista.shape[0] - 1), (255, 255, 255), 2)
    testi = [titolo]
    if misura is not None:
        scarti, a, b, ids = misura
        for (x1, y1), (x2, y2), i in zip(a, b, ids):
            c = colore(i)
            cv2.circle(vista, (int(round(x1)), int(round(y1))), 6, c, 2, cv2.LINE_AA)
            cv2.circle(vista, (int(round(x2 + larghezza)), int(round(y2))), 6, c, 2, cv2.LINE_AA)
        testi.append(f"scarto verticale: medio {np.mean(np.abs(scarti)):.2f} px, massimo {np.max(np.abs(scarti)):.2f} px"
                     f"  ({len(scarti)} angoli)")
    else:
        testi.append("pattern non visto da entrambe le camere")
    for riga, testo in enumerate(t for t in testi if t):
        for spessore, c in ((6, (0, 0, 0)), (2, (0, 255, 255))):
            cv2.putText(vista, testo, (15, 45 + 45 * riga), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, spessore, cv2.LINE_AA)
    return vista


def mostra(vista):
    cv2.imshow(FINESTRA, cv2.resize(vista, None, fx=SCALA_VISTA, fy=SCALA_VISTA, interpolation=cv2.INTER_AREA))


# ---------------------------------------------------------------------------------------
# 3. Test T09 sulle coppie salvate
# ---------------------------------------------------------------------------------------

def test_coppie(cal, rett, board, detector, nomi):
    percorsi = sorted(glob.glob(os.path.join(CARTELLA_COPPIE, f"coppia_*_{nomi[0]}.png")))
    if not percorsi:
        raise SystemExit(f"Nessuna coppia in {CARTELLA_COPPIE}: prima esegui calibrazione_stereo.py")
    os.makedirs(CARTELLA_USCITA, exist_ok=True)
    righe_csv = ["coppia;gruppo;angoli;scarto_medio_px;scarto_massimo_px;scarto_con_segno_px"]
    gruppi = {"calibrazione": [], "verifica": []}
    viste = []
    for indice, p1 in enumerate(percorsi):
        nome = os.path.basename(p1).replace(f"_{nomi[0]}.png", "")
        im1, im2 = cv2.imread(p1), cv2.imread(p1.replace(f"_{nomi[0]}.png", f"_{nomi[1]}.png"))
        if im1 is None or im2 is None:
            continue
        gruppo = "verifica" if (indice + 1) % OGNI_QUANTE_VERIFICA == 0 else "calibrazione"
        misura = scarti_verticali(rett, cal, [rileva(detector, board, im1), rileva(detector, board, im2)])
        vista = vista_affiancata(*rettifica_immagini(rett, im1, im2), misura, f"{nome} ({gruppo})")
        viste.append((nome, vista))
        if misura is None:
            righe_csv.append(f"{nome};{gruppo};0;;;")
            continue
        scarti = misura[0]
        gruppi[gruppo].append(scarti)
        righe_csv.append(f"{nome};{gruppo};{len(scarti)};{np.mean(np.abs(scarti)):.3f};"
                         f"{np.max(np.abs(scarti)):.3f};{np.mean(scarti):+.3f}")
        if gruppo == "verifica":
            cv2.imwrite(os.path.join(CARTELLA_USCITA, f"rettificata_{nome}.png"), vista)
    with open(os.path.join(CARTELLA_USCITA, "rettificazione_per_coppia.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(righe_csv) + "\n")

    righe = [f"Rettificazione - {time.strftime('%Y-%m-%d %H:%M')} - calibrazione stereo del {cal['data']}", "",
             f"Rotazioni virtuali: camera 1 {np.degrees(np.linalg.norm(cv2.Rodrigues(rett['R1'])[0])):.1f} gradi, "
             f"camera 2 {np.degrees(np.linalg.norm(cv2.Rodrigues(rett['R2'])[0])):.1f} gradi",
             f"Immagini rettificate: focale {rett['f']:.1f} px, riga del centro ottico {rett['cy']:.1f}, "
             f"colonna {rett['P1'][0, 2]:.1f} (camera 1) e {rett['P2'][0, 2]:.1f} (camera 2)", "",
             "Scarto verticale tra gli stessi angoli nelle due immagini rettificate:"]
    for gruppo, lista in gruppi.items():
        if lista:
            tutti = np.abs(np.concatenate(lista))
            righe.append(f"  coppie di {gruppo:12s} {len(lista):3d} coppie, {len(tutti):4d} angoli: medio {tutti.mean():.3f} px,"
                         f" 95% sotto {np.percentile(tutti, 95):.3f} px, massimo {tutti.max():.3f} px")
    esito = bool(gruppi["verifica"]) and np.mean(np.abs(np.concatenate(gruppi["verifica"]))) <= SCARTO_MAX_PX
    if gruppi["verifica"]:
        medio = np.mean(np.abs(np.concatenate(gruppi["verifica"])))
        righe += ["", "Controllo del test T09:",
                  f"  {'OK ' if esito else 'NO '} scarto verticale medio, coppie di verifica   {medio:.3f} px"
                  f"   criterio: <= {SCARTO_MAX_PX} px"]
    else:
        righe += ["", "Nessuna coppia di verifica con il pattern visto da entrambe le camere."]
    righe += ["", f"Salvati in {CARTELLA_USCITA}: rettificazione_per_coppia.csv e le immagini rettificate delle "
              "coppie di verifica", f"ESITO T09: {'SUPERATO' if esito else 'NON SUPERATO'}"]
    testo = "\n".join(righe)
    print("\n" + testo)
    with open(os.path.join(CARTELLA_USCITA, "rettificazione_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")
    return viste


def sfoglia(viste):
    """Finestra per guardare le coppie rettificate una alla volta."""
    i = 0
    cv2.namedWindow(FINESTRA, cv2.WINDOW_AUTOSIZE)
    while True:
        nome, vista = viste[i]
        mostra(vista)
        tasto = cv2.waitKey(0) & 0xFF
        if tasto == 27:
            break
        if tasto in (ord("n"), ord("N")):
            i = (i + 1) % len(viste)
        elif tasto in (ord("p"), ord("P")):
            i = (i - 1) % len(viste)
        elif tasto in (ord("s"), ord("S")):
            percorso = os.path.join(CARTELLA_USCITA, f"rettificata_{nome}.png")
            cv2.imwrite(percorso, vista)
            print(f"Salvata {percorso}")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------
# 4. Vista dal vivo
# ---------------------------------------------------------------------------------------

def vivo(cal, rett, board, detector):
    """Coppie dal vivo rettificate. Se il pattern è in vista, mostra lo scarto verticale: se cresce
    molto oltre 1 px, un telefono si è mosso dopo la calibrazione stereo."""
    os.makedirs(CARTELLA_USCITA, exist_ok=True)
    acq = AcquisizioneStereo()
    cv2.namedWindow(FINESTRA, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            coppia = acq.prossima_coppia(timeout_s=2)
            tasto = cv2.waitKey(1) & 0xFF
            if tasto == 27:
                break
            if coppia is None:
                continue
            im1, im2 = coppia.cam1.immagine, coppia.cam2.immagine
            misura = scarti_verticali(rett, cal, [rileva(detector, board, im1), rileva(detector, board, im2)])
            vista = vista_affiancata(*rettifica_immagini(rett, im1, im2), misura, "dal vivo")
            mostra(vista)
            if tasto in (ord("s"), ord("S")):
                percorso = os.path.join(CARTELLA_USCITA, f"vivo_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(percorso, vista)
                print(f"Salvata {percorso}")
    finally:
        acq.chiudi()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Rettificazione della coppia stereo (test T09)")
    parser.add_argument("--vivo", action="store_true", help="vista dal vivo invece del test sulle coppie salvate")
    parser.add_argument("--pattern", default="A1", help="pattern di charuco_boards.yaml (default A1)")
    args = parser.parse_args()

    cal = leggi_stereo()
    rett = prepara(cal)
    board, _ = leggi_pattern(args.pattern)
    detector = cv2.aruco.CharucoDetector(board)
    if args.vivo:
        vivo(cal, rett, board, detector)
    else:
        sfoglia(test_coppie(cal, rett, board, detector, list(STREAM)))


if __name__ == "__main__":
    main()
