"""
Confronto tra due immagini della STESSA camera ferma: misura se l'inquadratura è cambiata.

Se il telefono non si è mosso, lo sfondo della stanza deve trovarsi negli stessi pixel. Il programma
trova migliaia di punti caratteristici (SIFT) in entrambe le immagini, li abbina e stima la
trasformazione che porta la prima nella seconda:
  - scala diversa da 1   -> è cambiato l'ingrandimento (ritaglio del sensore, zoom): K non vale più
  - solo uno spostamento  -> il telefono si è mosso o ruotato un po' (oppure la stabilizzazione ottica)
  - niente                -> inquadratura identica
Le persone e il pattern, che cambiano posizione tra le due immagini, vengono scartati da RANSAC
come abbinamenti incoerenti: conta lo sfondo.

Se il telefono è stato anche ruotato tra le due immagini, uno spostamento di molti pixel falsa la
stima della scala (la rotazione della camera non è una semplice traslazione dell'immagine). Con
--camera NOME il programma usa la calibrazione di quella camera: toglie la distorsione e separa la
rotazione del telefono dal cambio di ingrandimento (modello H = K' R K^-1 con K' = K ingrandita).

Uso:
  python confronta_immagini.py IMMAGINE_A IMMAGINE_B [--camera Redmi]
  python confronta_immagini.py IMMAGINE_A --vivo Redmi     (confronta con il fotogramma attuale della camera)
"""
import argparse
import threading
import time

import cv2
import numpy as np


def fotogramma_vivo(camera):
    from acquisizione import STREAM, ROTAZIONE, LettoreStream
    lettore = LettoreStream(camera, STREAM[camera], threading.Condition(), rotazione=ROTAZIONE.get(camera))
    try:
        time.sleep(2)
        with lettore._cond:
            return lettore.buffer[-1].immagine.copy()
    finally:
        lettore.chiudi()


def abbina(a, b):
    """Punti SIFT abbinati tra le due immagini (test di Lowe)."""
    sift = cv2.SIFT_create(nfeatures=8000)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    coppie = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    buoni = [m for m, n in (c for c in coppie if len(c) == 2) if m.distance < 0.75 * n.distance]
    return np.float32([ka[m.queryIdx].pt for m in buoni]), np.float32([kb[m.trainIdx].pt for m in buoni])


def rotazione_e_ingrandimento(pa, pb, K, dist):
    """Telefono ruotato (R) e forse ingrandimento cambiato (s): H = K_s R K^-1, con K_s = K con le focali
    moltiplicate per s. Si stima H con RANSAC sui punti senza distorsione, poi si cerca la s per cui
    K_s^-1 H K è più vicina a una rotazione. Restituisce (s, angolo di rotazione in gradi, inlier, totale)."""
    ua = cv2.undistortPoints(pa.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    ub = cv2.undistortPoints(pb.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    H, inlier = cv2.findHomography(ua, ub, cv2.RANSAC, 3.0, maxIters=10000, confidence=0.999)
    migliore = None
    for s in np.arange(0.90, 1.10, 0.0002):
        Ks = K.copy()
        Ks[0, 0] *= s
        Ks[1, 1] *= s
        A = np.linalg.inv(Ks) @ H @ K
        A = A / np.cbrt(np.linalg.det(A))
        scarto = np.linalg.norm(A.T @ A - np.eye(3))
        if migliore is None or scarto < migliore[0]:
            migliore = (scarto, s, A)
    _, s, A = migliore
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    angolo = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    return float(s), angolo, int(inlier.sum()), len(pa)


def confronta(a, b):
    """Restituisce (scala, rotazione in gradi, spostamento in px, abbinamenti coerenti, abbinamenti totali,
    spostamento mediano dei punti in px)."""
    pa, pb = abbina(a, b)
    buoni = pa
    # Similitudine (scala uniforme + rotazione + traslazione) stimata con RANSAC
    M, coerenti = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.5,
                                              maxIters=5000, confidence=0.999)
    coerenti = coerenti.ravel().astype(bool)
    scala = float(np.hypot(M[0, 0], M[1, 0]))
    rotazione = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    spostamento = np.linalg.norm(pb[coerenti] - pa[coerenti], axis=1)
    return scala, rotazione, M[:, 2], int(coerenti.sum()), len(buoni), float(np.median(spostamento))


def main():
    parser = argparse.ArgumentParser(description="Confronto dell'inquadratura tra due immagini della stessa camera")
    parser.add_argument("immagine_a")
    parser.add_argument("immagine_b", nargs="?")
    parser.add_argument("--vivo", help="nome della camera: confronta immagine_a con il fotogramma attuale")
    parser.add_argument("--camera", help="nome della camera: usa la sua calibrazione per separare rotazione e ingrandimento")
    args = parser.parse_args()
    a = cv2.imread(args.immagine_a)
    b = fotogramma_vivo(args.vivo) if args.vivo else cv2.imread(args.immagine_b)
    if a is None or b is None or a.shape != b.shape:
        raise SystemExit("Servono due immagini della stessa camera, con le stesse dimensioni")
    scala, rotazione, traslazione, coerenti, totali, mediano = confronta(a, b)
    print(f"Abbinamenti coerenti con lo sfondo: {coerenti} su {totali}")
    print(f"Scala:        {scala:.5f}  ({(scala - 1) * 100:+.2f}%)")
    print(f"Rotazione:    {rotazione:+.3f} gradi")
    print(f"Traslazione:  ({traslazione[0]:+.1f}, {traslazione[1]:+.1f}) px,  spostamento mediano dei punti {mediano:.2f} px")
    camera = args.camera or args.vivo
    if camera:
        from calibrazione_mono import leggi_calibrazione
        K, dist = leggi_calibrazione(camera)
        pa, pb = abbina(a, b)
        s, angolo, inl, tot = rotazione_e_ingrandimento(pa, pb, K, dist)
        print(f"\nCon la calibrazione di {camera} (rotazione del telefono separata dall'ingrandimento):")
        print(f"  telefono ruotato di {angolo:.2f} gradi,  ingrandimento {s:.4f} ({(s - 1) * 100:+.2f}%),  "
              f"abbinamenti coerenti {inl} su {tot}")
        if abs(s - 1) > 0.005:
            print("  => L'INGRANDIMENTO è cambiato anche tenendo conto della rotazione: K non vale più.")
        else:
            print("  => Ingrandimento invariato (entro lo 0,5%): " +
                  ("il telefono si è solo mosso." if angolo > 0.2 else "inquadratura identica."))
        return
    if coerenti < 100:
        print("Pochi abbinamenti: le immagini sono troppo diverse (luce, oggetti spostati) per un confronto affidabile.")
    elif abs(scala - 1) > 0.003:
        print("=> L'INGRANDIMENTO è cambiato: la calibrazione fatta con l'immagine A non vale per l'immagine B.")
    elif mediano > 1.0:
        print("=> Stesso ingrandimento, ma l'inquadratura si è spostata: il telefono si è mosso o ruotato.")
    if mediano > 5:
        print("ATTENZIONE: spostamento di molti pixel, il telefono si è mosso: la scala qui sopra non è affidabile.\n"
              "Ripeti con --camera NOME per separare la rotazione del telefono dal cambio di ingrandimento.")
    else:
        print("=> Inquadratura identica (differenze sotto il pixel).")


if __name__ == "__main__":
    main()