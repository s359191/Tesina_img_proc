"""
Fase 7 - Triangolazione e sistema del pavimento (test T10, T11, T13; requisiti RF8, RF9, RC6).

Contiene due cose:

1. La classe Triangolatore, che userà anche la fase di MediaPipe: dai due pixel dello stesso punto
   (camera 1 e camera 2) calcola il punto 3D in metri (DLT, teoria sezione 6), il suo errore di
   riproiezione in ciascuna camera e, se il sistema del pavimento è stato definito, le sue
   coordinate rispetto al pavimento (teoria sezione 4.11).

2. Tre programmi di misura:

   --distanze            test T10 e T11. Il pattern tenuto FERMO in varie posizioni; per ogni
                         posizione (SPAZIO) si triangolano tutti gli angoli visti da entrambe le
                         camere, mediati su 10 coppie, e si confrontano le distanze tra gli angoli
                         con quelle vere del pattern; si misura anche quanto gli angoli si scostano
                         dal piano e quanto "tremano" da una coppia all'altra (rumore).

   --pavimento FILE      definisce il sistema del pavimento (non dal vivo). FILE è il file
                         punti_<data>_3d.csv salvato da misura_punti.py con le croci a terra:
                         il piano del pavimento passa per le croci, l'origine è la croce 1,
                         l'asse z va in alto, l'asse y verso le camere, l'asse x verso destra
                         (guardando le camere). Salva data/calibrazione/pavimento.yaml.
                         (Il pattern A1 appoggiato a terra non va bene: visto così di sbieco da
                         3 m i suoi marker sono troppo schiacciati per essere riconosciuti.)

   --verifica-pavimento  test T13, verticale: il pattern appoggiato bene contro un muro o una
                         porta deve risultare inclinato di 90 gradi rispetto al pavimento.
                         (La quota 0 si verifica con le croci stesse e con l'altezza delle camere.)

Uso:
  python triangolazione.py --distanze
  python triangolazione.py --pavimento data/calibrazione/punti_20261118_170500_3d.csv
  python triangolazione.py --verifica-pavimento
Tasti: SPAZIO = registra la posizione (quando il pattern è fermo da 10 coppie)   ESC = fine
"""
import argparse
import os
import time

import cv2
import numpy as np

from acquisizione import AcquisizioneStereo
from calibrazione_mono import leggi_pattern, rileva, ANGOLI_MINIMI, CARTELLA
from calibrazione_stereo import angoli_comuni
from test_latenza import affianca

FILE_STEREO = os.path.join(CARTELLA, "stereo.yaml")
FILE_PAVIMENTO = os.path.join(CARTELLA, "pavimento.yaml")
CARTELLA_USCITA = os.path.join("data", "triangolazione")
FINESTRA = "Triangolazione  (SPAZIO = registra, ESC = fine)"

COPPIE_MEDIA = 10            # coppie mediate per ogni posizione del pattern
MOVIMENTO_MAX_M = 0.005      # oltre: il pattern si è mosso, la media riparte
LUNGHE_MIN_M = 0.40          # distanze tra angoli considerate "lunghe" (5 quadrati in orizzontale, ecc.)
# Criteri del test T10 (pattern a circa 3 m) e T13
T10_DISTANZA_M = (2.5, 3.5)
T10_ERRORE_MAX_MM = 5.0
T10_PIANO_MAX_MM = 5.0
T13_QUOTA_MAX_MM = 10.0     # punti a terra: distanza dal piano del pavimento
T13_ANGOLO_MAX_GRADI = 1.0


# ---------------------------------------------------------------------------------------
# 1. Il triangolatore
# ---------------------------------------------------------------------------------------

class Triangolatore:
    """Dai pixel corrispondenti delle due camere ai punti 3D (in metri)."""

    def __init__(self, file_stereo=FILE_STEREO, file_pavimento=FILE_PAVIMENTO):
        fs = cv2.FileStorage(file_stereo, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Manca {file_stereo}: prima esegui calibrazione_stereo.py")
        for chiave in ("K1", "dist1", "K2", "dist2", "R", "T"):
            setattr(self, chiave, fs.getNode(chiave).mat())
        self.data_stereo = fs.getNode("data").string()
        fs.release()
        self.T = self.T.reshape(3, 1)
        # Matrici di proiezione (teoria 4.9): camera 1 nell'origine, camera 2 in (R, T)
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.K2 @ np.hstack([self.R, self.T])
        self.rvec2 = cv2.Rodrigues(self.R)[0]
        self.centro2 = (-self.R.T @ self.T).ravel()     # centro della camera 2 nel sistema della camera 1
        # Sistema del pavimento (facoltativo)
        self.R_pav, self.origine = None, None
        if file_pavimento and os.path.exists(file_pavimento):
            fs = cv2.FileStorage(file_pavimento, cv2.FILE_STORAGE_READ)
            self.R_pav = fs.getNode("R_pavimento").mat()
            self.origine = fs.getNode("origine").mat().ravel()
            fs.release()

    def triangola(self, pixel1, pixel2):
        """pixel1, pixel2: (N, 2) pixel dello stesso punto nelle immagini (con distorsione).
        Restituisce (punti 3D (N, 3) nel sistema della camera 1, errore di riproiezione (N, 2) in px
        nella camera 1 e nella camera 2)."""
        p1 = np.asarray(pixel1, np.float64).reshape(-1, 1, 2)
        p2 = np.asarray(pixel2, np.float64).reshape(-1, 1, 2)
        # 1. via la distorsione: pixel "raddrizzati", ancora in pixel (P = K)
        a = cv2.undistortPoints(p1, self.K1, self.dist1, P=self.K1).reshape(-1, 2).T
        b = cv2.undistortPoints(p2, self.K2, self.dist2, P=self.K2).reshape(-1, 2).T
        # 2. DLT (teoria 6.3): risultato in coordinate omogenee, 4 x N
        X4 = cv2.triangulatePoints(self.P1, self.P2, a, b)
        X = (X4[:3] / X4[3]).T
        # 3. errore di riproiezione (teoria 6.6): si riproietta il punto, con la distorsione
        r1 = cv2.projectPoints(X, np.zeros(3), np.zeros(3), self.K1, self.dist1)[0].reshape(-1, 2)
        r2 = cv2.projectPoints(X, self.rvec2, self.T, self.K2, self.dist2)[0].reshape(-1, 2)
        errori = np.column_stack([np.linalg.norm(r1 - p1.reshape(-1, 2), axis=1),
                                  np.linalg.norm(r2 - p2.reshape(-1, 2), axis=1)])
        return X, errori

    def nel_pavimento(self, X):
        """Punti dal sistema della camera 1 al sistema del pavimento (x, y orizzontali, z in alto)."""
        if self.R_pav is None:
            raise RuntimeError(f"Manca {FILE_PAVIMENTO}: prima esegui triangolazione.py --pavimento")
        return (np.asarray(X).reshape(-1, 3) - self.origine) @ self.R_pav.T


# ---------------------------------------------------------------------------------------
# 2. Misure sul pattern
# ---------------------------------------------------------------------------------------

def angoli_triangolati(tri, coppia, board, detector):
    """Angoli del pattern visti da entrambe le camere in una coppia, triangolati.
    Restituisce un dizionario o None se il pattern non è visto bene da entrambe."""
    r = [rileva(detector, board, coppia.cam1.immagine), rileva(detector, board, coppia.cam2.immagine)]
    if any(x is None for x in r):
        return None
    oggetto, q1, q2, ids = angoli_comuni(*r)
    if len(ids) < ANGOLI_MINIMI:
        return None
    X, errori = tri.triangola(q1, q2)
    return {"ids": ids, "X": X, "oggetto": oggetto, "q1": q1, "q2": q2, "errori": errori}


class Media:
    """Media dei punti 3D di ogni angolo sulle ultime COPPIE_MEDIA coppie, finché il pattern è fermo."""

    def __init__(self, n=COPPIE_MEDIA):
        self.n = n
        self.coppie = []

    def aggiungi(self, misura):
        centro = misura["X"].mean(axis=0)
        if self.coppie and np.linalg.norm(centro - self.coppie[-1]["X"].mean(axis=0)) > MOVIMENTO_MAX_M:
            self.coppie = []                            # il pattern si è mosso: si ricomincia
        self.coppie = (self.coppie + [misura])[-self.n:]

    @property
    def pronta(self):
        return len(self.coppie) >= self.n

    def risultato(self):
        """Angoli visti in almeno metà delle coppie: id, posizione media, posizione sul pattern,
        rumore (dev. std media della posizione, m), errore di riproiezione medio (px)."""
        tutti = {}
        for m in self.coppie:
            for i, x, o in zip(m["ids"], m["X"], m["oggetto"]):
                tutti.setdefault(int(i), ([], o))[0].append(x)
        ids = sorted(i for i, (xs, _) in tutti.items() if len(xs) >= self.n / 2)
        X = np.array([np.mean(tutti[i][0], axis=0) for i in ids])
        oggetto = np.array([tutti[i][1] for i in ids], np.float64)
        rumore = float(np.mean([np.linalg.norm(np.std(tutti[i][0], axis=0)) for i in ids]))
        errore = float(np.mean(np.concatenate([m["errori"].ravel() for m in self.coppie])))
        return np.array(ids), X, oggetto, rumore, errore


def confronta_distanze(X, oggetto):
    """Distanze tra tutte le coppie di angoli: misurate (triangolate) contro vere (dal pattern).
    Restituisce gli errori in mm separati in "adiacenti" (un quadrato) e "lunghe" (>= 40 cm)."""
    i, j = np.triu_indices(len(X), 1)
    vere = np.linalg.norm(oggetto[i] - oggetto[j], axis=1)
    misurate = np.linalg.norm(X[i] - X[j], axis=1)
    errori = (misurate - vere) * 1000
    adiacenti = vere < 1.2 * vere.min()
    lunghe = vere >= LUNGHE_MIN_M
    return errori[adiacenti], errori[lunghe]


def posa_rigida(oggetto, X):
    """Rotazione e traslazione che portano i punti del pattern (sul piano Z = 0) sui punti
    triangolati, ai minimi quadrati (metodo di Kabsch). Restituisce (R, t, scarti in m)."""
    ca, cb = oggetto.mean(axis=0), X.mean(axis=0)
    U, _, Vt = np.linalg.svd((oggetto - ca).T @ (X - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    scarti = np.linalg.norm(X - (oggetto @ R.T + t), axis=1)
    return R, t, scarti


def scarto_dal_piano(X):
    """Distanza di ogni punto dal piano che meglio li approssima (m) e normale del piano."""
    c = X.mean(axis=0)
    normale = np.linalg.svd(X - c)[2][2]
    return (X - c) @ normale, normale


def posizione_in_immagine(misura, larghezza):
    """Posizione orizzontale del pattern in ciascuna immagine, in % della larghezza."""
    return (100 * np.mean(misura["q1"][:, 0]) / larghezza, 100 * np.mean(misura["q2"][:, 0]) / larghezza)


# ---------------------------------------------------------------------------------------
# 3. Ciclo dal vivo, comune ai tre programmi
# ---------------------------------------------------------------------------------------

def ciclo(tri, board, detector, descrivi, registra, istruzione):
    """Legge coppie, triangola il pattern, ne fa la media e chiama registra(...) a ogni SPAZIO.
    descrivi(ids, X, oggetto, rumore, errore) -> righe di testo da mostrare con il pattern fermo."""
    acq = AcquisizioneStereo()
    media = Media()
    spazio, ultima = False, None
    cv2.namedWindow(FINESTRA, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            tasto = cv2.waitKey(1) & 0xFF
            if tasto == 27:
                break
            spazio = spazio or tasto == ord(" ")
            coppia = acq.prossima_coppia(timeout_s=2)
            if coppia is None:
                continue
            immagini = [coppia.cam1.immagine.copy(), coppia.cam2.immagine.copy()]
            misura = angoli_triangolati(tri, coppia, board, detector)
            testi = [istruzione]
            if misura is None:
                media.coppie = []
                testi.append("pattern non visto da entrambe le camere (servono almeno 12 angoli comuni)")
            else:
                media.aggiungi(misura)
                for im, q in zip(immagini, (misura["q1"], misura["q2"])):
                    for x, y in q:
                        cv2.circle(im, (int(x), int(y)), 7, (0, 255, 0), 2, cv2.LINE_AA)
                distanza = float(np.linalg.norm(misura["X"].mean(axis=0)))
                testi.append(f"{len(misura['ids'])} angoli comuni, centro del pattern a {distanza:.2f} m dalla camera 1"
                             f"   media: {len(media.coppie)}/{COPPIE_MEDIA} coppie")
                if media.pronta:
                    ids, X, oggetto, rumore, errore = media.risultato()
                    testi += descrivi(ids, X, oggetto, rumore, errore)
                    if spazio:
                        spazio = False
                        ultima = registra(ids, X, oggetto, rumore, errore, misura, immagini[0].shape[1])
                else:
                    testi.append("tieni il pattern FERMO...")
            if ultima:
                testi.append(f"ultima registrazione: {ultima}")
            cv2.imshow(FINESTRA, affianca(immagini, testi, altezza=640))
    finally:
        acq.chiudi()
        cv2.destroyAllWindows()


def salva_testo(nome, righe):
    os.makedirs(CARTELLA_USCITA, exist_ok=True)
    testo = "\n".join(righe)
    print("\n" + testo)
    with open(os.path.join(CARTELLA_USCITA, nome), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


# ---------------------------------------------------------------------------------------
# 4. Test T10 e T11: distanze note
# ---------------------------------------------------------------------------------------

def programma_distanze(tri, board, detector):
    registrazioni = []

    def descrivi(ids, X, oggetto, rumore, errore):
        adiacenti, lunghe = confronta_distanze(X, oggetto)
        piano, _ = scarto_dal_piano(X)
        testo = f"lato quadrato: errore medio {adiacenti.mean():+.1f} mm"
        if len(lunghe):
            testo += f"   distanze lunghe: {lunghe.mean():+.1f} mm"
        return [testo + f"   piano: {np.sqrt(np.mean(piano ** 2)) * 1000:.1f} mm   rumore {rumore * 1000:.1f} mm",
                "SPAZIO = registra questa posizione"]

    def registra(ids, X, oggetto, rumore, errore, misura, larghezza):
        adiacenti, lunghe = confronta_distanze(X, oggetto)
        piano, _ = scarto_dal_piano(X)
        x1, x2 = posizione_in_immagine(misura, larghezza)
        r = {"distanza": float(np.linalg.norm(X.mean(axis=0))), "angoli": len(ids), "x1": x1, "x2": x2,
             "adiacenti": adiacenti, "lunghe": lunghe, "piano_rms": float(np.sqrt(np.mean(piano ** 2))),
             "piano_max": float(np.abs(piano).max()), "rumore": rumore, "riproiezione": errore}
        registrazioni.append(r)
        print(f"Posizione {len(registrazioni)}: {r['distanza']:.2f} m, lato {adiacenti.mean():+.2f} mm, "
              f"lunghe {lunghe.mean() if len(lunghe) else float('nan'):+.2f} mm, piano {r['piano_rms'] * 1000:.2f} mm")
        return f"posizione {len(registrazioni)} a {r['distanza']:.2f} m"

    ciclo(tri, board, detector, descrivi, registra,
          "T10/T11: pattern FERMO in una posizione, poi SPAZIO. Cambia distanza (2, 3, 4 m) e posizione.")
    if not registrazioni:
        print("Nessuna posizione registrata.")
        return

    righe = [f"Triangolazione - distanze note (T10, T11) - {time.strftime('%Y-%m-%d %H:%M')}",
             f"calibrazione stereo del {tri.data_stereo}", "",
             "Errori = distanza triangolata - distanza vera sul pattern (media con segno, media assoluta).",
             "Posizione in immagine: centro del pattern in % della larghezza (0 = bordo sinistro).", "",
             f"  {'pos':>3s} {'dist m':>6s} {'Redmi%':>6s} {'S20%':>5s} {'angoli':>6s}   {'lato mm':>13s}   "
             f"{'lunghe mm':>13s}   {'piano rms/max mm':>16s}  {'rumore mm':>9s}  {'riproiez px':>11s}  T10"]
    csv = ["posizione;distanza_m;x_redmi_pct;x_s20fe_pct;angoli;lato_medio_mm;lato_assoluto_mm;lunghe_medio_mm;"
           "lunghe_assoluto_mm;piano_rms_mm;piano_max_mm;rumore_mm;riproiezione_px"]
    esiti_t10 = []
    for k, r in enumerate(registrazioni, start=1):
        lat, lun = r["adiacenti"], r["lunghe"]
        lun_m = lun.mean() if len(lun) else np.nan
        lun_a = np.abs(lun).mean() if len(lun) else np.nan
        nel_t10 = T10_DISTANZA_M[0] <= r["distanza"] <= T10_DISTANZA_M[1]
        esito = ""
        if nel_t10:
            ok = (np.abs(lat).mean() <= T10_ERRORE_MAX_MM and (not len(lun) or lun_a <= T10_ERRORE_MAX_MM)
                  and r["piano_rms"] * 1000 <= T10_PIANO_MAX_MM)
            esiti_t10.append(ok)
            esito = "OK" if ok else "NO"
        righe.append(f"  {k:3d} {r['distanza']:6.2f} {r['x1']:6.0f} {r['x2']:5.0f} {r['angoli']:6d}   "
                     f"{lat.mean():+6.2f} {np.abs(lat).mean():5.2f}   {lun_m:+6.2f} {lun_a:5.2f}   "
                     f"{r['piano_rms'] * 1000:7.2f} {r['piano_max'] * 1000:7.2f}  {r['rumore'] * 1000:9.2f}  "
                     f"{r['riproiezione']:11.2f}  {esito}")
        csv.append(f"{k};{r['distanza']:.3f};{r['x1']:.0f};{r['x2']:.0f};{r['angoli']};{lat.mean():.3f};"
                   f"{np.abs(lat).mean():.3f};{lun_m:.3f};{lun_a:.3f};{r['piano_rms'] * 1000:.3f};"
                   f"{r['piano_max'] * 1000:.3f};{r['rumore'] * 1000:.3f};{r['riproiezione']:.3f}")

    # T11: andamento con la distanza
    righe += ["", "T11 - andamento con la distanza (media delle posizioni in ogni fascia):",
              f"  {'fascia':>12s} {'posizioni':>9s} {'lato ass. mm':>12s} {'lunghe ass. mm':>14s} {'rumore mm':>9s}"]
    for basso, alto in ((0, 2.5), (2.5, 3.5), (3.5, 10)):
        gruppo = [r for r in registrazioni if basso <= r["distanza"] < alto]
        if not gruppo:
            continue
        lun = [np.abs(r["lunghe"]).mean() for r in gruppo if len(r["lunghe"])]
        righe.append(f"  {f'{basso}-{alto if alto < 10 else ''} m':>12s} {len(gruppo):9d} "
                     f"{np.mean([np.abs(r['adiacenti']).mean() for r in gruppo]):12.2f} "
                     f"{np.mean(lun) if lun else float('nan'):14.2f} {np.mean([r['rumore'] for r in gruppo]) * 1000:9.2f}")

    righe += ["", f"Criteri T10 (posizioni tra {T10_DISTANZA_M[0]} e {T10_DISTANZA_M[1]} m): errore assoluto medio "
              f"<= {T10_ERRORE_MAX_MM} mm sul lato e sulle distanze lunghe, scarto dal piano rms <= {T10_PIANO_MAX_MM} mm"]
    if esiti_t10:
        righe.append(f"ESITO T10: {'SUPERATO' if all(esiti_t10) else 'NON SUPERATO'} "
                     f"({sum(esiti_t10)} posizioni su {len(esiti_t10)} entro i criteri)")
    else:
        righe.append("ESITO T10: nessuna posizione tra 2,5 e 3,5 m")
    salva_testo("distanze_riepilogo.txt", righe)
    with open(os.path.join(CARTELLA_USCITA, "distanze.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")


# ---------------------------------------------------------------------------------------
# 5. Sistema del pavimento (definito con le croci sul pavimento) e test T13
# ---------------------------------------------------------------------------------------

def leggi_punti_3d(percorso):
    """Il file punti_<data>_3d.csv di misura_punti.py: numero del punto e coordinate (camera 1, m)."""
    righe = [r.strip().split(";") for r in open(percorso, encoding="utf-8") if r.strip()][1:]
    return [int(r[0]) for r in righe], np.array([[float(v) for v in r[1:4]] for r in righe])


def sistema_pavimento(tri, X):
    """Sistema del pavimento dai punti a terra X (il primo è l'origine).
    Asse z: normale al piano che meglio approssima i punti, verso le camere (in alto).
    Asse y: orizzontale, dall'origine verso il punto medio tra le due camere ("avanti").
    Asse x: y x z, cioè verso destra per chi sta nell'origine guardando le camere.
    Restituisce (R_pav con gli assi come righe, origine, scarti dei punti dal piano in m)."""
    centro = X.mean(axis=0)
    su = np.linalg.svd(X - centro)[2][2]
    medio_camere = tri.centro2 / 2
    if su @ (medio_camere - centro) < 0:
        su = -su
    scarti = (X - centro) @ su
    origine = X[0] - ((X[0] - centro) @ su) * su
    avanti = medio_camere - origine
    avanti -= (avanti @ su) * su
    avanti /= np.linalg.norm(avanti)
    destra = np.cross(avanti, su)
    return np.vstack([destra, avanti, su]), origine, scarti


def programma_pavimento(tri, percorso_punti):
    numeri, X = leggi_punti_3d(percorso_punti)
    if len(X) < 3:
        raise SystemExit("Servono almeno 3 punti sul pavimento (meglio 5, ben distribuiti)")
    R_pav, origine, scarti = sistema_pavimento(tri, X)
    os.makedirs(CARTELLA, exist_ok=True)
    fs = cv2.FileStorage(FILE_PAVIMENTO, cv2.FILE_STORAGE_WRITE)
    fs.write("data", time.strftime("%Y-%m-%d %H:%M"))
    fs.write("calibrazione_stereo", tri.data_stereo)
    fs.write("punti", os.path.basename(percorso_punti))
    fs.write("R_pavimento", R_pav)
    fs.write("origine", origine.reshape(3, 1))
    fs.release()
    tri.R_pav, tri.origine = R_pav, origine
    punti = tri.nel_pavimento(X)
    c1, c2 = tri.nel_pavimento(np.vstack([np.zeros(3), tri.centro2]))
    esito = bool(np.all(np.abs(scarti) * 1000 <= T13_QUOTA_MAX_MM))
    righe = [f"Sistema del pavimento - {time.strftime('%Y-%m-%d %H:%M')} - calibrazione stereo del {tri.data_stereo}",
             f"punti: {percorso_punti} (il punto {numeri[0]} è l'origine)", "",
             "Assi: z verso l'alto, y orizzontale verso le camere, x verso destra guardando le camere.", "",
             "Punti nel sistema del pavimento (z = distanza dal piano del pavimento):"]
    for n, (x, y, z) in zip(numeri, punti):
        righe.append(f"  punto {n}: x {x:+.3f}  y {y:+.3f}  z {z * 1000:+.1f} mm")
    righe += [f"  scarto massimo dal piano {np.abs(scarti).max() * 1000:.1f} mm (criterio T13: entro ±{T13_QUOTA_MAX_MM:.0f} mm)"
              f"   {'OK' if esito else 'NO'}", "",
              "Posizione delle camere (da confrontare con il metro):",
              f"  camera 1: altezza da terra {c1[2]:.3f} m, in pianta a {np.hypot(c1[0], c1[1]):.3f} m dall'origine",
              f"  camera 2: altezza da terra {c2[2]:.3f} m, in pianta a {np.hypot(c2[0], c2[1]):.3f} m dall'origine",
              f"  distanza tra le camere misurata sul pavimento (come con il filo a piombo): {np.hypot(*(c1[:2] - c2[:2])):.3f} m",
              "", f"Salvato: {FILE_PAVIMENTO}"]
    salva_testo("pavimento_riepilogo.txt", righe)


def assi_del_pattern(tri, X, oggetto):
    """Posa del pattern ricostruita (Kabsch) e la sua normale, orientata verso le camere."""
    R, t, scarti = posa_rigida(oggetto, X)
    normale = R[:, 2]
    if normale @ (tri.centro2 / 2 - X.mean(axis=0)) < 0:
        normale = -normale
    return R, t, normale, scarti


def programma_verifica_pavimento(tri, board, detector):
    if tri.R_pav is None:
        raise SystemExit(f"Manca {FILE_PAVIMENTO}: prima esegui python triangolazione.py --pavimento PUNTI_3D.csv")
    registrazioni = []

    def inclinazione(X, oggetto):
        """Angolo tra il pattern e il pavimento: 0 = orizzontale, 90 = verticale."""
        _, _, normale, _ = assi_del_pattern(tri, X, oggetto)
        n = tri.R_pav @ normale
        return float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))

    def descrivi(ids, X, oggetto, rumore, errore):
        Xp = tri.nel_pavimento(X)
        return [f"inclinazione rispetto al pavimento {inclinazione(X, oggetto):.2f} gradi   centro del pattern: "
                f"x {Xp[:, 0].mean():+.2f}  y {Xp[:, 1].mean():+.2f}  altezza {Xp[:, 2].mean():.2f} m",
                "pattern appoggiato bene contro un muro o una porta: SPAZIO = registra"]

    def registra(ids, X, oggetto, rumore, errore, misura, larghezza):
        r = {"inclinazione": inclinazione(X, oggetto), "centro": tri.nel_pavimento(X).mean(axis=0)}
        registrazioni.append(r)
        return f"posizione {len(registrazioni)}: inclinazione {r['inclinazione']:.2f} gradi"

    ciclo(tri, board, detector, descrivi, registra,
          "T13: pattern A1 appoggiato bene contro un muro o una porta (deve risultare verticale)")
    if not registrazioni:
        print("Nessuna posizione registrata.")
        return
    righe = [f"Verifica della verticale del sistema del pavimento (T13) - {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Criterio: pattern contro un muro verticale, inclinazione 90 ± {T13_ANGOLO_MAX_GRADI} gradi", ""]
    esiti = []
    for k, r in enumerate(registrazioni, start=1):
        x, y, z = r["centro"]
        ok = abs(r["inclinazione"] - 90) <= T13_ANGOLO_MAX_GRADI
        esiti.append(ok)
        righe.append(f"  {k}. centro in ({x:+.2f}, {y:+.2f}) m a {z:.2f} m da terra: inclinazione "
                     f"{r['inclinazione']:.2f} gradi (scarto da 90: {r['inclinazione'] - 90:+.2f})   {'OK' if ok else 'NO'}")
    righe += ["", f"ESITO verticale T13: {'SUPERATO' if all(esiti) else 'NON SUPERATO'}"]
    salva_testo("verifica_pavimento_riepilogo.txt", righe)


def main():
    parser = argparse.ArgumentParser(description="Triangolazione e sistema del pavimento (T10, T11, T13)")
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--distanze", action="store_true", help="test T10 e T11")
    modo.add_argument("--pavimento", metavar="PUNTI_3D_CSV",
                      help="definisce il sistema del pavimento dai punti a terra salvati da misura_punti.py")
    modo.add_argument("--verifica-pavimento", action="store_true", help="test T13")
    parser.add_argument("--pattern", default="A1", help="pattern di charuco_boards.yaml (default A1)")
    args = parser.parse_args()

    board, valori = leggi_pattern(args.pattern)
    detector = cv2.aruco.CharucoDetector(board)
    tri = Triangolatore()
    if args.distanze:
        programma_distanze(tri, board, detector)
    elif args.pavimento:
        programma_pavimento(tri, args.pavimento)
    else:
        programma_verifica_pavimento(tri, board, detector)


if __name__ == "__main__":
    main()
