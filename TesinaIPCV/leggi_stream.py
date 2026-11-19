"""
Fase 0 - ultimo controllo: legge i due stream di IP Webcam, li mostra affiancati
e misura per ciascuno i fotogrammi al secondo effettivamente ricevuti.

Prima di avviarlo:
  1. IP Webcam avviata su entrambi i telefoni
  2. inoltro delle porte attivo:
       adb -s SERIALE_REDMI forward tcp:8080 tcp:8080
       adb -s SERIALE_S20   forward tcp:8081 tcp:8080
  3. python imposta_camere.py   ->  deve dire PRONTO

Uso:    python leggi_stream.py
Tasti:  q = esci     s = salva la coppia di fotogrammi correnti
"""
import os
import threading
import time
from collections import deque

import cv2

# Indirizzo dello stream MJPEG di ciascun telefono (IP Webcam lo pubblica su /video)
STREAM = {
    "Redmi": "http://localhost:8080/video",
    "S20FE": "http://localhost:8081/video",
}
CARTELLA_SALVATAGGI = os.path.join("data", "prove_stream")
SCALA_FINESTRA = 0.5   # solo per la visualizzazione: due immagini 1280 px non stanno sullo schermo


class LettoreStream:
    """Legge uno stream in un thread separato e conserva solo l'ultimo fotogramma ricevuto."""

    def __init__(self, nome, url):
        self.nome = nome
        # CAP_FFMPEG: usiamo la libreria FFmpeg, inclusa in opencv-python, che sa leggere gli stream HTTP
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"{nome}: impossibile aprire {url}. "
                               "IP Webcam è avviata? L'inoltro adb è attivo?")

        self.lock = threading.Lock()      # protegge i dati condivisi tra i due thread
        self.frame = None                 # ultimo fotogramma ricevuto
        self.t_arrivo = None              # istante di arrivo (orologio del PC, in secondi)
        self.n_frame = 0                  # quanti fotogrammi sono arrivati in tutto
        self.arrivi = deque(maxlen=60)    # istanti degli ultimi 60 arrivi, per misurare gli fps

        self.attivo = True
        self.thread = threading.Thread(target=self._ciclo_lettura, daemon=True)
        self.thread.start()

    def _ciclo_lettura(self):
        """Gira nel thread: legge di continuo, così nessun fotogramma resta in attesa nel buffer."""
        while self.attivo:
            ok, frame = self.cap.read()       # si blocca finché non arriva un fotogramma
            t = time.perf_counter()           # lo prendiamo SUBITO dopo l'arrivo
            if not ok:
                time.sleep(0.01)              # stream interrotto: piccola pausa e riprova
                continue
            with self.lock:
                self.frame = frame
                self.t_arrivo = t
                self.n_frame += 1
                self.arrivi.append(t)

    def ultimo(self):
        """Restituisce (fotogramma, istante di arrivo, numero di fotogrammi ricevuti)."""
        with self.lock:
            return self.frame, self.t_arrivo, self.n_frame

    def fps(self):
        """Fotogrammi al secondo calcolati sugli ultimi arrivi."""
        with self.lock:
            if len(self.arrivi) < 2:
                return 0.0
            return (len(self.arrivi) - 1) / (self.arrivi[-1] - self.arrivi[0])

    def chiudi(self):
        self.attivo = False
        self.thread.join(timeout=2)
        self.cap.release()


def scrivi(immagine, testo, riga, colore):
    """Testo leggibile su qualsiasi sfondo: contorno nero sotto, colore sopra."""
    pos = (10, 30 + 30 * riga)
    cv2.putText(immagine, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(immagine, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colore, 2, cv2.LINE_AA)


def attendi_primi_fotogrammi(lettori, timeout_s=10):
    inizio = time.perf_counter()
    while any(l.ultimo()[0] is None for l in lettori):
        if time.perf_counter() - inizio > timeout_s:
            mancanti = [l.nome for l in lettori if l.ultimo()[0] is None]
            raise RuntimeError(f"Nessun fotogramma entro {timeout_s} s da: {', '.join(mancanti)}")
        time.sleep(0.05)


def main():
    lettori = []
    try:
        for nome, url in STREAM.items():
            lettori.append(LettoreStream(nome, url))
        attendi_primi_fotogrammi(lettori)
        os.makedirs(CARTELLA_SALVATAGGI, exist_ok=True)
        n_salvati = 0

        while True:
            adesso = time.perf_counter()
            originali, viste = [], []
            for l in lettori:
                frame, t_arrivo, n = l.ultimo()
                originali.append((frame, t_arrivo))
                vista = frame.copy()              # disegniamo su una copia, l'originale resta pulito
                h, w = vista.shape[:2]
                eta_ms = (adesso - t_arrivo) * 1000
                colore = (0, 255, 0) if eta_ms < 200 else (0, 0, 255)   # rosso = stream fermo
                scrivi(vista, f"{l.nome}  {w}x{h}  {l.fps():.1f} fps", 0, colore)
                scrivi(vista, f"fotogrammi: {n}   ultimo arrivato {eta_ms:.0f} ms fa", 1, colore)
                viste.append(vista)

            # per affiancarle con hconcat le immagini devono avere la stessa altezza
            altezza = min(v.shape[0] for v in viste)
            viste = [v if v.shape[0] == altezza else
                     cv2.resize(v, (round(v.shape[1] * altezza / v.shape[0]), altezza)) for v in viste]
            affiancate = cv2.hconcat(viste)
            affiancate = cv2.resize(affiancate, None, fx=SCALA_FINESTRA, fy=SCALA_FINESTRA,
                                    interpolation=cv2.INTER_AREA)
            cv2.imshow("Due stream  (q = esci, s = salva)", affiancate)

            tasto = cv2.waitKey(1) & 0xFF        # waitKey serve anche a far ridisegnare la finestra
            if tasto == ord("q"):
                break
            if tasto == ord("s"):
                for l, (frame, _) in zip(lettori, originali):
                    percorso = os.path.join(CARTELLA_SALVATAGGI, f"coppia_{n_salvati:03d}_{l.nome}.png")
                    cv2.imwrite(percorso, frame)   # PNG: nessuna seconda compressione
                scarto_ms = (originali[0][1] - originali[1][1]) * 1000
                print(f"Salvata coppia {n_salvati:03d}  (differenza tra gli arrivi: {scarto_ms:+.0f} ms)")
                n_salvati += 1
    finally:
        for l in lettori:
            l.chiudi()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()