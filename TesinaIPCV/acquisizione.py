"""
Fase 2 - Modulo di acquisizione stereo.

Contiene:
  Tempi               i tempi di un fotogramma: arrivo, posto sulla griglia, istante sulla griglia
  StimatoreGriglia    stima in tempo reale la griglia regolare degli scatti di una camera
  Fotogramma          un'immagine con i suoi tempi
  Coppia              due fotogrammi (camera 1 e camera 2) abbinati
  LettoreStream       legge uno stream in un thread e conserva gli ultimi fotogrammi
  AcquisizioneStereo  unisce i due lettori e restituisce coppie abbinate per istante
                      della scena più vicino (requisito RF2)

Istante della scena = istante sulla griglia meno la correzione di sincronizzazione della camera,
misurata con test_sincronizzazione.py e salvata in data/sincronizzazione.json (requisito RY1).

Eseguito direttamente mostra le coppie affiancate con le statistiche.
Prima di avviarlo: IP Webcam avviata, adb forward attivi, imposta_camere.py -> PRONTO.

Uso:    python acquisizione.py
Tasti:  q = esci     s = salva la coppia corrente (PNG + riga nel file CSV)
"""
import csv
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Camera 1 = Redmi, camera 2 = S20 FE: l'ordine conta, la calibrazione stereo lo userà
STREAM = {
    "Redmi": "http://localhost:8080/video",
    "S20FE": "http://localhost:8081/video",
}
# I telefoni sono montati in verticale ma IP Webcam trasmette in orizzontale: ogni immagine viene
# raddrizzata appena arriva. Valori: "orario", "antiorario", "180" oppure None (nessuna rotazione).
# Si sceglie guardando l'anteprima (python acquisizione.py): le persone devono apparire in piedi.
# ATTENZIONE: la calibrazione vale solo per la rotazione con cui è stata fatta. Non cambiarla dopo.
ROTAZIONE = {
    "Redmi": "orario",
    "S20FE": "orario",
}
CODICI_ROTAZIONE = {"orario": cv2.ROTATE_90_CLOCKWISE, "antiorario": cv2.ROTATE_90_COUNTERCLOCKWISE,
                    "180": cv2.ROTATE_180, None: None}
PERIODO_ATTESO_S = 1 / 30                    # 30 fps
SOGLIA_ABBINAMENTO_S = PERIODO_ATTESO_S / 2  # 16,7 ms: mezzo fotogramma (requisito RF2)
CARTELLA_COPPIE = os.path.join("data", "coppie")
FILE_SINCRONIZZAZIONE = os.path.join("data", "sincronizzazione.json")
SCALA_FINESTRA = 0.5


# ---------------------------------------------------------------------------------------
# La griglia degli scatti
# ---------------------------------------------------------------------------------------

@dataclass
class Tempi:
    """Tempi di un fotogramma. Lo stimatore della griglia può correggerli anche dopo l'arrivo."""
    indice: int        # numero progressivo del fotogramma ricevuto: 1, 2, 3, ...
    t_arrivo: float    # istante di arrivo al programma, in secondi (orologio time.perf_counter)
    posto: int         # posto n sulla griglia degli scatti: il numero dello scatto
    t_griglia: float   # istante sulla griglia q + P*n; uguale a t_arrivo finché la griglia non è stimata


class StimatoreGriglia:
    """
    Stima la griglia regolare degli scatti di una camera dai soli istanti di arrivo.

    Modello:  t_arrivo = q + P * posto + rumore      (una retta: pendenza P, intercetta q)
      1. ogni fotogramma prende il posto successivo a quello del fotogramma precedente;
      2. ogni `ogni` fotogrammi, P e q si stimano con i minimi quadrati sugli ultimi
         `finestra` fotogrammi (finestra mobile: segue le variazioni lente);
      3. P resta entro `tolleranza` (0,2%) dal periodo nominale: il sensore è stabilissimo, e
         senza vincolo i falsi "fotogrammi persi" inclinerebbero la retta, generandone altri;
      4. se la mediana degli scarti (arrivo - griglia) degli ultimi `k` fotogrammi supera
         mezzo periodo, un fotogramma è andato perso: i fotogrammi dopo il buco avanzano di posto.
    Parametri scelti provando le alternative sui dati reali del test T02 (9 novembre 2026).
    """

    def __init__(self, periodo_nominale=PERIODO_ATTESO_S, finestra=300, ogni=10, k=9, tolleranza=0.002,
                 riscaldamento=30):
        self.riscaldamento = riscaldamento   # primi fotogrammi esclusi: all'apertura arrivano a raffica
        self.P0 = periodo_nominale
        self.P = periodo_nominale   # periodo: nominale finché non c'è la prima stima
        self.P_libero = None        # stima senza vincolo, solo per diagnosi
        self.q = None               # None = griglia non ancora stimata
        self.storia = deque(maxlen=finestra)   # Tempi degli ultimi fotogrammi (10 s a 30 fps)
        self.ogni = ogni
        self.k = k
        self.tolleranza = tolleranza
        self.ricevuti = 0
        self.correzioni = 0         # quante volte i posti sono stati corretti

    def nuovo_arrivo(self, t):
        """Registra un arrivo e restituisce i suoi Tempi (oggetto condiviso: può essere corretto dopo)."""
        self.ricevuti += 1
        if self.ricevuti <= self.riscaldamento:
            # Riscaldamento: nessun posto (-1) e istante sulla griglia = istante di arrivo
            return Tempi(self.ricevuti, t, -1, t)
        posto = self.storia[-1].posto + 1 if self.storia else 0
        t_griglia = t if self.q is None else self.q + self.P * posto
        tempi = Tempi(self.ricevuti, t, posto, t_griglia)
        self.storia.append(tempi)
        if self.q is not None:
            self._controlla_posti()
        if len(self.storia) >= self.ogni and self.ricevuti % self.ogni == 0:
            self._stima_retta()
        return tempi

    def mancanti(self):
        """Fotogrammi persi dalla fine del riscaldamento: posti vuoti della griglia (stima)."""
        if not self.storia:
            return 0
        return self.storia[-1].posto + 1 - (self.ricevuti - self.riscaldamento)

    def _stima_retta(self):
        # Minimi quadrati per la retta t = q + P * n:
        #   P = somma((n - n_medio) * (t - t_medio)) / somma((n - n_medio)^2),  q = t_medio - P * n_medio
        n = np.array([r.posto for r in self.storia], dtype=float)
        t = np.array([r.t_arrivo for r in self.storia])
        n_medio, t_medio = n.mean(), t.mean()
        self.P_libero = float(np.sum((n - n_medio) * (t - t_medio)) / np.sum((n - n_medio) ** 2))
        # Vincolo sul periodo: P resta tra P0*(1 - tolleranza) e P0*(1 + tolleranza)
        self.P = min(max(self.P_libero, self.P0 * (1 - self.tolleranza)), self.P0 * (1 + self.tolleranza))
        self.q = float(t_medio - self.P * n_medio)   # la retta passa per il punto medio
        for r in self.storia:
            r.t_griglia = self.q + self.P * r.posto

    def _controlla_posti(self):
        if len(self.storia) <= self.k:
            return
        finestra = [self.storia[-i] for i in range(self.k + 1, 0, -1)]   # k+1 fotogrammi, dal più vecchio
        ultimi = finestra[1:]
        scarti = [r.t_arrivo - (self.q + self.P * r.posto) for r in ultimi]
        salto = int(round(float(np.median(scarti)) / self.P))   # di quanti posti sono spostati
        if salto > 0:
            # Fotogrammi persi: il buco sta prima del fotogramma con l'intervallo di arrivo più lungo
            _, dopo_buco = max(zip(finestra, ultimi), key=lambda c: c[1].t_arrivo - c[0].t_arrivo)
            self._sposta(dopo_buco.indice, salto)
        elif salto < 0:
            # Posti di troppo (una correzione sbagliata): si toglie un buco, se tra gli ultimi ce n'è uno
            for prima, dopo in zip(finestra, ultimi):
                buco = dopo.posto - prima.posto - 1
                if buco > 0:
                    self._sposta(dopo.indice, max(salto, -buco))
                    break

    def _sposta(self, da_indice, s):
        for r in self.storia:
            if r.indice >= da_indice:
                r.posto += s
                r.t_griglia = self.q + self.P * r.posto
        self.correzioni += 1


# ---------------------------------------------------------------------------------------
# Fotogrammi, coppie e lettura degli stream
# ---------------------------------------------------------------------------------------

@dataclass
class Fotogramma:
    immagine: np.ndarray        # immagine BGR, 720 x 1280 x 3
    tempi: Tempi                # condiviso con lo stimatore della griglia
    correzione_s: float = 0.0   # ritardo costante di questa camera rispetto all'altra (sincronizzazione)

    @property
    def indice(self):
        return self.tempi.indice

    @property
    def t_arrivo(self):
        return self.tempi.t_arrivo

    @property
    def t_griglia(self):
        return self.tempi.t_griglia

    @property
    def t_scena(self):
        """Istante sulla griglia corretto: confrontabile con quello dell'altra camera."""
        return self.tempi.t_griglia - self.correzione_s


@dataclass
class Coppia:
    cam1: Fotogramma
    cam2: Fotogramma

    @property
    def scarto_s(self):
        """Differenza tra gli istanti della scena (camera 1 - camera 2), in secondi."""
        return self.cam1.t_scena - self.cam2.t_scena

    @property
    def scarto_arrivi_s(self):
        """Differenza tra gli istanti di arrivo, solo per confronto."""
        return self.cam1.t_arrivo - self.cam2.t_arrivo

    @property
    def entro_soglia(self):
        """True se lo scarto rispetta la soglia di mezzo fotogramma (RF2)."""
        return abs(self.scarto_s) <= SOGLIA_ABBINAMENTO_S


class LettoreStream:
    """Legge uno stream in un thread separato e conserva gli ultimi fotogrammi ricevuti."""

    def __init__(self, nome, url, condizione, dimensione_buffer=10, registra=False, correzione_s=0.0,
                 rotazione=None):
        self.nome = nome
        self.url = url
        self.correzione_s = correzione_s
        self.rotazione = rotazione
        self._codice_rotazione = CODICI_ROTAZIONE[rotazione]
        self._cond = condizione                          # condivisa con l'altro lettore (vedi sotto)
        self.buffer = deque(maxlen=dimensione_buffer)    # ultimi 10 fotogrammi, circa 1/3 di secondo
        self._arrivi = deque(maxlen=61)                  # istanti degli ultimi arrivi, per gli fps
        self.griglia = StimatoreGriglia()
        self.registro = [] if registra else None         # Tempi di tutti i fotogrammi (test T02)
        self._cap = self._apri()
        self._attivo = True
        self._thread = threading.Thread(target=self._ciclo_lettura, daemon=True)
        self._thread.start()

    def _apri(self):
        # I timeout evitano che la lettura resti bloccata a lungo se il telefono smette di trasmettere
        parametri = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000]
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, parametri)
        if not cap.isOpened():
            raise RuntimeError(f"{self.nome}: impossibile aprire {self.url}. "
                               "IP Webcam è avviata? L'inoltro adb è attivo?")
        return cap

    def _ciclo_lettura(self):
        inizio_errori = None
        while self._attivo:
            ok, immagine = self._cap.read()
            t = time.perf_counter()          # preso subito dopo l'arrivo, prima di qualsiasi elaborazione
            if ok and self._codice_rotazione is not None:
                immagine = cv2.rotate(immagine, self._codice_rotazione)
            if not ok:
                # Stream interrotto: dopo 2 s di errori proviamo a riaprirlo
                if inizio_errori is None:
                    inizio_errori = t
                elif t - inizio_errori > 2.0:
                    self._cap.release()
                    time.sleep(0.5)
                    try:
                        self._cap = self._apri()      # stessi timeout dell'apertura iniziale
                    except RuntimeError:
                        pass                          # telefono ancora assente: si riprova tra 2 s
                    inizio_errori = None
                else:
                    time.sleep(0.01)
                continue
            inizio_errori = None
            with self._cond:
                tempi = self.griglia.nuovo_arrivo(t)
                self.buffer.append(Fotogramma(immagine, tempi, self.correzione_s))
                self._arrivi.append(t)
                if self.registro is not None:
                    self.registro.append(tempi)
                self._cond.notify_all()       # sveglia chi sta aspettando una coppia nuova

    def fps(self):
        with self._cond:
            if len(self._arrivi) < 2:
                return 0.0
            return (len(self._arrivi) - 1) / (self._arrivi[-1] - self._arrivi[0])

    def mancanti(self):
        with self._cond:
            return self.griglia.mancanti()

    def chiudi(self):
        self._attivo = False
        self._thread.join(timeout=3)
        self._cap.release()


def leggi_sincronizzazione(percorso=FILE_SINCRONIZZAZIONE):
    """Legge le correzioni di sincronizzazione salvate da test_sincronizzazione.py.
    Restituisce il contenuto del file (dizionario) oppure None se il file non esiste."""
    if not os.path.exists(percorso):
        return None
    with open(percorso, encoding="utf-8") as f:
        return json.load(f)


class AcquisizioneStereo:
    """Due lettori e l'abbinamento dei loro fotogrammi per istante della scena più vicino."""

    def __init__(self, stream=STREAM, registra=False, sincronizza=True):
        # Un'unica Condition per entrambi i lettori: il suo lock protegge i due buffer,
        # e la sua notifica sveglia prossima_coppia() quando arriva un fotogramma qualsiasi.
        self._cond = threading.Condition()
        self.lettori = []
        self.sincronizzazione = leggi_sincronizzazione() if sincronizza else None
        correzioni = self.sincronizzazione["correzione_ms"] if self.sincronizzazione else {}
        if sincronizza and self.sincronizzazione is None:
            print(f"ATTENZIONE: {FILE_SINCRONIZZAZIONE} non esiste, le camere non sono sincronizzate. "
                  "Eseguire: python test_sincronizzazione.py --salva")
        try:
            for nome, url in stream.items():
                self.lettori.append(LettoreStream(nome, url, self._cond, registra=registra,
                                                  correzione_s=correzioni.get(nome, 0.0) / 1000,
                                                  rotazione=ROTAZIONE.get(nome)))
        except Exception:
            self.chiudi()
            raise
        self._ultima = None   # numeri dei fotogrammi dell'ultima coppia restituita

    def prossima_coppia(self, timeout_s=2.0):
        """Aspetta una coppia nuova e la restituisce. None se entro timeout_s non ce n'è una."""
        scadenza = time.perf_counter() + timeout_s
        with self._cond:
            while True:
                coppia = self._cerca_coppia()
                if coppia is not None:
                    return coppia
                rimasto = scadenza - time.perf_counter()
                if rimasto <= 0:
                    return None
                self._cond.wait(rimasto)    # rilascia il lock e dorme finché un lettore notifica

    def fotogrammi_dopo(self, ultimi):
        """Per ogni camera, i fotogrammi del buffer con indice maggiore di ultimi[nome].
        Serve ai test che devono esaminare OGNI fotogramma (per esempio T04), non solo le coppie:
        chiamandola più spesso di 3 volte al secondo non se ne perde nessuno (il buffer ne tiene 10)."""
        with self._cond:
            return {l.nome: [f for f in l.buffer if f.indice > ultimi.get(l.nome, 0)] for l in self.lettori}

    def _cerca_coppia(self):
        b1, b2 = self.lettori[0].buffer, self.lettori[1].buffer
        if not b1 or not b2:
            return None
        # Riferimento = l'ultimo fotogramma della camera "rimasta indietro" (istante della scena).
        # L'altra camera ha già un fotogramma più avanti del riferimento, quindi tra i suoi
        # fotogrammi ce n'è uno prima e uno dopo: il più vicino dista al massimo mezzo periodo.
        if b1[-1].t_scena <= b2[-1].t_scena:
            rif = b1[-1]
            altro = min(b2, key=lambda f: abs(f.t_scena - rif.t_scena))
            coppia = Coppia(rif, altro)
        else:
            rif = b2[-1]
            altro = min(b1, key=lambda f: abs(f.t_scena - rif.t_scena))
            coppia = Coppia(altro, rif)
        # Ogni fotogramma si usa una volta sola: entrambi devono essere più recenti di quelli
        # dell'ultima coppia. Se una camera si ferma, il suo indice non cresce più e non escono
        # coppie: prossima_coppia() restituisce None allo scadere del timeout.
        if self._ultima is not None and (coppia.cam1.indice <= self._ultima[0] or
                                         coppia.cam2.indice <= self._ultima[1]):
            return None
        # Soglia di mezzo periodo (RF2): senza perdite il compagno più vicino è sempre entro
        # mezzo periodo; se è più lontano, il compagno vero è andato perso e la coppia si scarta.
        if not coppia.entro_soglia:
            return None
        self._ultima = (coppia.cam1.indice, coppia.cam2.indice)
        return coppia

    def chiudi(self):
        for lettore in self.lettori:
            lettore.chiudi()


# ---------------------------------------------------------------------------------------
# Programma di prova: visualizzazione, statistiche e salvataggio delle coppie
# ---------------------------------------------------------------------------------------

def scrivi(immagine, testo, riga, colore=(0, 255, 0)):
    pos = (10, 30 + 30 * riga)
    cv2.putText(immagine, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(immagine, testo, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colore, 2, cv2.LINE_AA)


def prepara_vista(coppia, lettori, scarti_ms):
    viste = []
    for fot, lettore in zip((coppia.cam1, coppia.cam2), lettori):
        vista = fot.immagine.copy()
        scrivi(vista, f"{lettore.nome}  {lettore.fps():.1f} fps  persi: {lettore.mancanti()}", 0)
        viste.append(vista)
    altezza = min(v.shape[0] for v in viste)
    viste = [v if v.shape[0] == altezza else
             cv2.resize(v, (round(v.shape[1] * altezza / v.shape[0]), altezza)) for v in viste]
    affiancate = cv2.hconcat(viste)
    # Lo scarto sulla griglia è sempre entro mezzo periodo (le coppie più lontane vengono
    # scartate): qui si mostra quanto vale, la verifica con lo scarto vero si fa nella fase 3.
    medio = np.mean(np.abs(scarti_ms))
    scrivi(affiancate, f"scarto coppia: {scarti_ms[-1]:+.1f} ms   medio {medio:.1f} ms "
                       f"(ultime {len(scarti_ms)})", 1)
    return cv2.resize(affiancate, None, fx=SCALA_FINESTRA, fy=SCALA_FINESTRA,
                      interpolation=cv2.INTER_AREA)


def main():
    acq = AcquisizioneStereo()
    nomi = [l.nome for l in acq.lettori]
    os.makedirs(CARTELLA_COPPIE, exist_ok=True)
    sessione = time.strftime("%Y%m%d_%H%M%S")
    percorso_csv = os.path.join(CARTELLA_COPPIE, "coppie.csv")
    intestazione = ["sessione", "coppia", f"file_{nomi[0]}", f"file_{nomi[1]}",
                    f"t_scena_{nomi[0]}_s", f"t_scena_{nomi[1]}_s", "scarto_ms",
                    "scarto_arrivi_ms", f"indice_{nomi[0]}", f"indice_{nomi[1]}"]
    if os.path.exists(percorso_csv):
        with open(percorso_csv, newline="") as f:
            if f.readline().strip() != ";".join(intestazione):
                # file scritto da una versione precedente, con colonne diverse: lo mettiamo da parte
                os.replace(percorso_csv, percorso_csv.replace(".csv", f"_fino_al_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    nuovo_file = not os.path.exists(percorso_csv)
    file_csv = open(percorso_csv, "a", newline="")
    tabella = csv.writer(file_csv, delimiter=";")
    if nuovo_file:
        tabella.writerow(intestazione)
    t0 = time.perf_counter()
    scarti_ms = deque(maxlen=300)
    tutti_scarti_ms = []
    n_salvate = 0
    try:
        while True:
            coppia = acq.prossima_coppia(timeout_s=2.0)
            if coppia is None:
                print("Nessuna coppia nuova da 2 s: controlla telefoni e inoltro adb.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            scarti_ms.append(coppia.scarto_s * 1000)
            tutti_scarti_ms.append(coppia.scarto_s * 1000)
            cv2.imshow("Acquisizione stereo  (q = esci, s = salva)",
                       prepara_vista(coppia, acq.lettori, scarti_ms))
            tasto = cv2.waitKey(1) & 0xFF
            if tasto == ord("q"):
                break
            if tasto == ord("s"):
                file_png = []
                for nome, fot in zip(nomi, (coppia.cam1, coppia.cam2)):
                    nome_file = f"{sessione}_{n_salvate:03d}_{nome}.png"
                    cv2.imwrite(os.path.join(CARTELLA_COPPIE, nome_file), fot.immagine)
                    file_png.append(nome_file)
                tabella.writerow([sessione, n_salvate, *file_png,
                                  f"{coppia.cam1.t_scena - t0:.4f}", f"{coppia.cam2.t_scena - t0:.4f}",
                                  f"{coppia.scarto_s * 1000:.1f}", f"{coppia.scarto_arrivi_s * 1000:.1f}",
                                  coppia.cam1.indice, coppia.cam2.indice])
                file_csv.flush()
                print(f"Salvata coppia {n_salvate:03d}  (scarto {coppia.scarto_s * 1000:+.1f} ms)")
                n_salvate += 1
    finally:
        durata = time.perf_counter() - t0
        acq.chiudi()
        file_csv.close()
        cv2.destroyAllWindows()
        if tutti_scarti_ms:
            s = np.abs(np.array(tutti_scarti_ms))
            print(f"\nRiepilogo ({durata:.0f} s, {len(s)} coppie, {len(s) / durata:.1f} coppie/s)")
            print(f"  scarto sulla griglia: medio {s.mean():.1f} ms, massimo {s.max():.1f} ms "
                  f"(le coppie oltre {SOGLIA_ABBINAMENTO_S * 1000:.1f} ms vengono scartate)")
            for l in acq.lettori:
                print(f"  {l.nome}: fotogrammi senza compagno {l.griglia.ricevuti - len(s)}, "
                      f"persi (stima) {l.griglia.mancanti()}, "
                      f"periodo stimato {l.griglia.P * 1000:.3f} ms ({1 / l.griglia.P:.3f} fps)")


if __name__ == "__main__":
    main()