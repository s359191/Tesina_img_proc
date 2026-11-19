"""
Fase 2 - Test T04: latenza dalla scena al programma (requisito RA4, obiettivo <= 200 ms).

Idea: il programma stesso crea un evento nella scena, di cui conosce l'istante esatto.
Lo schermo del PC, a tutto schermo, passa da nero a bianco (e viceversa) a istanti casuali;
i due telefoni inquadrano lo schermo. Per ogni cambio si cerca il primo fotogramma di ciascuna
camera in cui lo schermo appare cambiato:

    latenza = istante di arrivo di quel fotogramma  -  istante in cui il programma ha cambiato lo schermo

Entrambi gli istanti sono letti con lo stesso orologio (time.perf_counter del PC), quindi la
differenza non dipende dagli orologi dei telefoni.

Fasi del programma:
  1. PUNTAMENTO  - finestra con le due camere: si orientano i telefoni verso lo schermo, poi SPAZIO
  2. MISURA      - schermo intero nero/bianco con un orologio in ms; dura circa 1 s per cambio
  3. ANALISI     - zona dello schermo in ogni immagine, luminosità di ogni fotogramma, latenze

Uso:   python test_latenza.py --transizioni 60 --etichetta prova
Tasti: SPAZIO = inizia la misura     ESC = interrompi
Risultati: a schermo e nella cartella data/test_latenza/
"""
import argparse
import os
import random
import time

import cv2
import numpy as np

from acquisizione import STREAM, PERIODO_ATTESO_S, AcquisizioneStereo

CARTELLA = os.path.join("data", "test_latenza")
FINESTRA = "T04 latenza  (SPAZIO = inizia, ESC = esci)"
SOGLIA_LATENZA_MS = 200             # requisito RA4
DIM_RIDOTTA = (128, 72)             # per la luminosità media bastano immagini piccole (larghezza, altezza)
INTERVALLO_S = (0.8, 1.2)           # tempo casuale tra due cambi dello schermo
NERO_INIZIALE_S = 2.0               # schermo nero prima del primo cambio: livello di partenza
ATTESA_FINALE_S = 1.0               # dopo l'ultimo cambio: tempo per ricevere i suoi fotogrammi
FINESTRA_LIVELLO_S = 0.3            # fotogrammi usati per misurare il livello prima/dopo un cambio
CONTRASTO_MINIMO = 20               # livelli di grigio: sotto, il cambio non si vede abbastanza
CAMPIONI_OGNI_S = 5.0               # ogni quanto conservare un'immagine intera per la verifica a mano


class Raccolta:
    """Raccoglie, per ogni camera, TUTTI i fotogrammi arrivati durante la misura, in versione ridotta."""

    def __init__(self, acq, t_inizio, campioni_ogni_s=CAMPIONI_OGNI_S):
        self.acq = acq
        self.t_inizio = t_inizio
        self.campioni_ogni_s = campioni_ogni_s
        self.ultimi = {}                                         # indice dell'ultimo fotogramma letto
        self.tempi = {l.nome: [] for l in acq.lettori}           # oggetti Tempi (arrivo e griglia)
        self.ridotte = {l.nome: [] for l in acq.lettori}         # immagini 160x90 in grigio
        self.ultima_intera = {l.nome: None for l in acq.lettori}
        self.campioni = {l.nome: [] for l in acq.lettori}        # (t_arrivo relativo, immagine intera)
        self.saltati = {l.nome: 0 for l in acq.lettori}          # fotogrammi persi dalla raccolta

    def aggiorna(self):
        for nome, nuovi in self.acq.fotogrammi_dopo(self.ultimi).items():
            for f in nuovi:
                precedente = self.ultimi.get(nome)
                if precedente is not None and f.indice > precedente + 1:
                    self.saltati[nome] += f.indice - precedente - 1   # usciti dal buffer prima di leggerli
                self.ultimi[nome] = f.indice
                grigio = cv2.cvtColor(f.immagine, cv2.COLOR_BGR2GRAY)
                self.ridotte[nome].append(cv2.resize(grigio, DIM_RIDOTTA, interpolation=cv2.INTER_AREA))
                self.tempi[nome].append(f.tempi)
                self.ultima_intera[nome] = f.immagine
                t_rel = f.t_arrivo - self.t_inizio
                if not self.campioni[nome] or t_rel - self.campioni[nome][-1][0] >= self.campioni_ogni_s:
                    self.campioni[nome].append((t_rel, f.immagine))


FOTOGRAMMI_PRIMA_DI_MISURARE = 330   # riscaldamento (30) + una finestra intera della griglia (300)


def affianca(immagini, testi, altezza=360):
    """Immagini affiancate alla stessa altezza (qualsiasi orientamento) con una fascia di testo sotto."""
    viste = [cv2.resize(im, (round(im.shape[1] * altezza / im.shape[0]), altezza), interpolation=cv2.INTER_AREA)
             for im in immagini]
    riga_immagini = cv2.hconcat(viste)
    larghezza = max(riga_immagini.shape[1], 900)
    if riga_immagini.shape[1] < larghezza:
        bordo = larghezza - riga_immagini.shape[1]
        riga_immagini = cv2.copyMakeBorder(riga_immagini, 0, 0, bordo // 2, bordo - bordo // 2,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
    fascia = np.zeros((20 + 28 * len(testi), larghezza, 3), np.uint8)
    for riga, testo in enumerate(testi):
        cv2.putText(fascia, testo, (10, 30 + 28 * riga), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.vconcat([riga_immagini, fascia])


def puntamento(acq, istruzioni=("Punta i due telefoni verso questo schermo:",
                                "lo schermo deve occupare circa meta' di ogni immagine, al centro.")):
    """Mostra le due camere finché non si preme SPAZIO (True) o ESC (False).
    SPAZIO funziona solo quando la griglia degli scatti di entrambe le camere è stimata bene."""
    cv2.namedWindow(FINESTRA, cv2.WINDOW_NORMAL)
    while True:
        mancano = max(FOTOGRAMMI_PRIMA_DI_MISURARE - l.griglia.ricevuti for l in acq.lettori)
        ultimi = acq.fotogrammi_dopo({})
        immagini = []
        for l in acq.lettori:
            im = ultimi[l.nome][-1].immagine.copy() if ultimi[l.nome] else np.zeros((360, 640, 3), np.uint8)
            cv2.putText(im, l.nome, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            immagini.append(im)
        ultima = ("SPAZIO = inizia la misura   ESC = esci" if mancano <= 0 else
                  f"Attendi {mancano / 30:.0f} s: la griglia degli scatti si sta stimando   ESC = esci")
        cv2.imshow(FINESTRA, affianca(immagini, list(istruzioni) + [ultima]))
        tasto = cv2.waitKey(30) & 0xFF
        if tasto == ord(" ") and mancano <= 0:
            return True
        if tasto == 27:
            return False


def misura(acq, n_transizioni, campioni_ogni_s=CAMPIONI_OGNI_S):
    """Fa lampeggiare lo schermo e raccoglie i fotogrammi. Restituisce (eventi, raccolta, t_fine)."""
    cv2.setWindowProperty(FINESTRA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(FINESTRA, np.zeros((720, 1280, 3), np.uint8))
    cv2.waitKey(300)                                    # tempo per passare a schermo intero
    _, _, larghezza, altezza = cv2.getWindowImageRect(FINESTRA)
    if larghezza <= 0 or altezza <= 0:
        larghezza, altezza = 1280, 720
    fondi = [np.full((altezza, larghezza, 3), v, np.uint8) for v in (0, 255)]   # nero, bianco
    scala = larghezza / 900                             # orologio largo circa metà schermo
    spessore = max(2, int(4 * scala))

    t_inizio = time.perf_counter()
    raccolta = Raccolta(acq, t_inizio, campioni_ogni_s)
    eventi = []                                         # (istante in cui il cambio è stato mostrato, 0/1)
    stato = 0
    prossimo = t_inizio + NERO_INIZIALE_S
    t_fine = None
    while True:
        adesso = time.perf_counter()
        cambia = t_fine is None and adesso >= prossimo
        if cambia:
            stato = 1 - stato
        immagine = fondi[stato].copy()
        # Orologio in ms, grigio medio: si legge sia sul nero sia sul bianco (verifica a mano)
        testo = f"{(adesso - t_inizio) * 1000:7.0f} ms"
        (w, h), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, scala, spessore)
        cv2.putText(immagine, testo, ((larghezza - w) // 2, (altezza + h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scala, (128, 128, 128), spessore, cv2.LINE_AA)
        cv2.imshow(FINESTRA, immagine)
        tasto = cv2.waitKey(1) & 0xFF                  # qui OpenCV disegna davvero la finestra
        t_mostrato = time.perf_counter()
        if cambia:
            eventi.append((t_mostrato, stato))
            prossimo = t_mostrato + random.uniform(*INTERVALLO_S)
            if len(eventi) == n_transizioni:
                t_fine = t_mostrato + ATTESA_FINALE_S
        raccolta.aggiorna()
        if tasto == 27:
            print("Interrotto: analizzo i cambi già misurati.")
            t_fine = t_mostrato
        if t_fine is not None and t_mostrato >= t_fine:
            return eventi, raccolta, t_mostrato


def zona_schermo(ridotte):
    """Trova i pixel che cambiano con il lampeggio: la zona più grande con forte escursione di luminosità.
    Restituisce (maschera booleana 90x160 oppure None, escursione massima)."""
    passo = max(1, len(ridotte) // 600)               # al massimo 600 immagini: basta e risparmia memoria
    pila = np.stack(ridotte[::passo]).astype(np.float32)
    escursione = np.percentile(pila, 95, axis=0) - np.percentile(pila, 5, axis=0)
    # massimo dopo una leggera sfocatura: un pixel isolato rumoroso non conta, una zona piccola
    # (un telefono a 3 m occupa meno dell'1% dell'immagine) sì
    massima = float(cv2.GaussianBlur(escursione, (3, 3), 0).max())
    if massima < CONTRASTO_MINIMO:
        return None, massima
    maschera = (escursione >= 0.5 * massima).astype(np.uint8)
    n, etichette, statistiche, _ = cv2.connectedComponentsWithStats(maschera, connectivity=8)
    if n <= 1:
        return None, massima
    piu_grande = 1 + int(np.argmax(statistiche[1:, cv2.CC_STAT_AREA]))
    return etichette == piu_grande, massima


def latenze(eventi, t_arr, t_gr, luminosita, t_fine):
    """Per ogni cambio: il primo fotogramma arrivato dopo il cambio che ha superato la metà strada
    tra il livello prima e quello dopo. Restituisce una lista di dizionari (None = cambio non misurabile)."""
    risultati = []
    for k, (e, stato) in enumerate(eventi):
        e_dopo = eventi[k + 1][0] if k + 1 < len(eventi) else t_fine
        prima = luminosita[(t_arr > e - FINESTRA_LIVELLO_S) & (t_arr <= e)]
        dopo = luminosita[(t_arr > e_dopo - FINESTRA_LIVELLO_S) & (t_arr <= e_dopo)]
        if len(prima) < 3 or len(dopo) < 3:
            risultati.append(None)
            continue
        livello_prima, livello_dopo = float(np.median(prima)), float(np.median(dopo))
        contrasto = livello_dopo - livello_prima
        # il contrasto deve essere abbastanza grande e nel verso giusto (verso il bianco se stato = 1)
        if abs(contrasto) < CONTRASTO_MINIMO or (contrasto > 0) != (stato == 1):
            risultati.append(None)
            continue
        meta = (livello_prima + livello_dopo) / 2
        cambiati = np.flatnonzero((t_arr > e) & (t_arr <= e_dopo) & ((luminosita - meta) * np.sign(contrasto) > 0))
        if len(cambiati) == 0:
            risultati.append(None)
            continue
        i = cambiati[0]
        risultati.append({"arrivo_ms": (t_arr[i] - e) * 1000, "griglia_ms": (t_gr[i] - e) * 1000,
                          "contrasto": abs(contrasto)})
    return risultati


def descrivi(valori):
    v = np.array(valori)
    return (f"media {v.mean():.0f} ms, mediana {np.median(v):.0f} ms, 95% sotto {np.percentile(v, 95):.0f} ms, "
            f"min {v.min():.0f} ms, max {v.max():.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="Test T04: latenza dalla scena al programma")
    parser.add_argument("--transizioni", type=int, default=60, help="numero di cambi nero/bianco (default 60)")
    parser.add_argument("--etichetta", default="prova", help="nome della prova, usato nei file salvati")
    args = parser.parse_args()

    acq = AcquisizioneStereo(STREAM, sincronizza=False)   # qui servono gli istanti non corretti
    try:
        if not puntamento(acq):
            return
        eventi, raccolta, t_fine = misura(acq, args.transizioni)
    finally:
        acq.chiudi()
        cv2.destroyAllWindows()
    if not eventi:
        print("Nessun cambio misurato.")
        return

    os.makedirs(CARTELLA, exist_ok=True)
    base = os.path.join(CARTELLA, args.etichetta)
    t0 = raccolta.t_inizio
    righe = [f"Test T04 '{args.etichetta}' - {time.strftime('%Y-%m-%d %H:%M')} - {len(eventi)} cambi dello schermo", ""]
    per_camera, esiti = {}, []
    for nome in raccolta.tempi:
        tempi, ridotte = raccolta.tempi[nome], raccolta.ridotte[nome]
        if len(ridotte) < 30:
            righe += [f"{nome}: troppo pochi fotogrammi ({len(ridotte)})", ""]
            esiti.append(False)
            continue
        maschera, escursione = zona_schermo(ridotte)
        if maschera is None:
            righe += [f"{nome}: lo schermo non si vede (escursione di luminosità {escursione:.0f} livelli). "
                      "Il telefono inquadra lo schermo?", ""]
            esiti.append(False)
            continue
        t_arr = np.array([t.t_arrivo for t in tempi])
        t_gr = np.array([t.t_griglia for t in tempi])      # valori finali, dopo le correzioni della griglia
        luminosita = np.array([r[maschera].mean() for r in ridotte])
        risultati = latenze(eventi, t_arr, t_gr, luminosita, t_fine)
        per_camera[nome] = risultati
        validi = [r for r in risultati if r is not None]

        # File per le verifiche: segnale di luminosità e zona trovata
        with open(f"{base}_{nome}_segnale.csv", "w") as f:
            f.write("indice;t_arrivo_s;t_griglia_s;luminosita\n")
            f.writelines(f"{t.indice};{t.t_arrivo - t0:.5f};{t.t_griglia - t0:.5f};{l:.1f}\n"
                         for t, l in zip(tempi, luminosita))
        intera = raccolta.ultima_intera[nome].copy()
        grande = cv2.resize(maschera.astype(np.uint8), (intera.shape[1], intera.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        contorni, _ = cv2.findContours(grande, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(intera, contorni, -1, (0, 0, 255), 3)
        cv2.imwrite(f"{base}_{nome}_zona.png", intera)
        for t_rel, immagine in raccolta.campioni[nome]:
            cv2.imwrite(f"{base}_{nome}_arrivo_{t_rel * 1000:.0f}ms.png", immagine)

        righe.append(f"{nome}: {len(tempi)} fotogrammi, zona dello schermo {maschera.mean() * 100:.0f}% dell'immagine, "
                     f"cambi misurati {len(validi)} su {len(eventi)}"
                     + (f", fotogrammi saltati nella raccolta {raccolta.saltati[nome]}" if raccolta.saltati[nome] else ""))
        if len(validi) < 0.8 * len(eventi):
            righe += ["  troppi cambi non misurabili: controlla l'immagine della zona e il contrasto", ""]
            esiti.append(False)
            continue
        arrivo = [r["arrivo_ms"] for r in validi]
        righe.append(f"  latenza (arrivo - cambio):   {descrivi(arrivo)}")
        righe.append(f"  istante sulla griglia - cambio: media {np.mean([r['griglia_ms'] for r in validi]):.0f} ms")
        ok = np.mean(arrivo) <= SOGLIA_LATENZA_MS
        righe += [f"  T04 (latenza media <= {SOGLIA_LATENZA_MS} ms): {'SUPERATO' if ok else 'NON SUPERATO'}", ""]
        esiti.append(ok)

    # Differenza tra le camere sui cambi misurati da entrambe: anticipa la misura della fase 3
    nomi = list(per_camera)
    if len(nomi) == 2:
        coppie = [(a, b) for a, b in zip(per_camera[nomi[0]], per_camera[nomi[1]]) if a and b]
        if len(coppie) >= 10:
            d_gr = np.array([a["griglia_ms"] - b["griglia_ms"] for a, b in coppie])
            d_arr = np.array([a["arrivo_ms"] - b["arrivo_ms"] for a, b in coppie])
            righe.append(f"Differenza {nomi[0]} - {nomi[1]} su {len(coppie)} cambi (anticipo della fase 3):")
            righe.append(f"  sulla griglia: media {d_gr.mean():+.1f} ms (incertezza ±{d_gr.std() / np.sqrt(len(d_gr)):.1f} ms), "
                         f"dev.std {d_gr.std():.1f} ms")
            righe.append(f"  sugli arrivi:  media {d_arr.mean():+.1f} ms, dev.std {d_arr.std():.1f} ms")
        with open(f"{base}_cambi.csv", "w") as f:
            f.write("cambio;t_cambio_s;verso;" + ";".join(f"latenza_arrivo_{n}_ms;latenza_griglia_{n}_ms" for n in nomi) + "\n")
            for k, (e, stato) in enumerate(eventi):
                valori = []
                for n in nomi:
                    r = per_camera[n][k]
                    valori += [f"{r['arrivo_ms']:.1f}", f"{r['griglia_ms']:.1f}"] if r else ["", ""]
                f.write(f"{k + 1};{e - t0:.4f};{'bianco' if stato else 'nero'};" + ";".join(valori) + "\n")
    righe.append(f"ESITO T04: {'SUPERATO' if esiti and all(esiti) else 'NON SUPERATO'}")

    testo = "\n".join(righe)
    print("\n" + testo)
    with open(f"{base}_riepilogo.txt", "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()