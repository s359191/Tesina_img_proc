"""
Fase 3 - Test T05: sfasamento costante tra le camere (requisiti RF2 e RY1).

Usa lo stesso lampeggio dello schermo del test T04 (test_latenza.py). Per ogni cambio k e per
ogni camera si prende il primo fotogramma in cui lo schermo appare cambiato e si calcola

    g = istante sulla griglia di quel fotogramma - istante del cambio  =  ritardo costante + attesa dello scatto

La differenza tra le camere,  d = g_camera1 - g_camera2,  contiene il ritardo costante di una
camera rispetto all'altra (quello che la griglia non può togliere) più un termine casuale a media
nulla (le due attese dello scatto). La media di d su molti cambi è la correzione da sottrarre agli
istanti della camera 1; la sua incertezza scende come 1 / radice(numero di cambi).

Due modi di produrre il lampeggio:
  - schermo del portatile (predefinito): il programma conosce l'istante di ogni cambio;
  - --esterno: lampeggia un altro dispositivo (telefono o tablet con la pagina lampeggio.html) messo
    al centro dell'area di ripresa. L'istante del cambio non è noto, ma non serve: nella differenza
    d = g1 - g2 l'istante del cambio si cancella. Il programma ricava i cambi dal video della camera 2
    e li anticipa di ANTICIPO_S, così che il primo fotogramma cambiato di ENTRAMBE le camere arrivi dopo.

Uso:
  python test_sincronizzazione.py --transizioni 200 --salva [--esterno]
        misura la correzione e la salva in data/sincronizzazione.json
  python test_sincronizzazione.py --transizioni 100 --etichetta verifica [--esterno]
        nuova misura indipendente: controlla che, con la correzione salvata, il residuo sia ~0
Tasti: SPAZIO = inizia la misura     ESC = interrompi
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from acquisizione import (STREAM, PERIODO_ATTESO_S, FILE_SINCRONIZZAZIONE, AcquisizioneStereo,
                          leggi_sincronizzazione)
from test_latenza import puntamento, misura, zona_schermo, latenze, affianca, FINESTRA, CONTRASTO_MINIMO

CARTELLA = os.path.join("data", "test_sincronizzazione")
INCERTEZZA_MAX_MS = 1.5      # la correzione salvata deve essere nota con questa precisione
RESIDUO_MAX_MS = 3.0         # verifica: dopo la correzione la differenza media deve stare entro questo
# Dispersione delle differenze: con sfasamento phi tra le due griglie, la differenza di un cambio vale
# -phi oppure P - phi (dipende da quale camera scatta per prima dopo il cambio). La media è comunque il
# ritardo costante; la dev. std vale P * radice(p (1 - p)) con p = phi / P, quindi al massimo P/2.
DISPERSIONE_MASSIMA_MS = PERIODO_ATTESO_S * 1000 / 2
CAMPIONI_OGNI_S = 30.0       # immagini intere conservate per controllo: poche, il test è lungo
# Salute della griglia intorno a un cambio: media di (arrivo - istante sulla griglia) sui fotogrammi
# arrivati entro 2 s dal cambio. Se la griglia segue bene la camera vale circa 0; durante una raffica
# di latenza (e un eventuale falso fotogramma perso) si allontana di molti ms e quel cambio è inaffidabile.
SALUTE_FINESTRA_S = 2.0
SALUTE_MAX_MS = 3.0
# Modo --esterno
RICERCA_ZONA_S = 6.0         # primi secondi: si cerca la zona che lampeggia (il dispositivo deve già lampeggiare)
RIDUZIONE = 0.25             # immagini a un quarto della risoluzione: il dispositivo è piccolo nell'immagine
ANTICIPO_S = 0.15            # il cambio stimato viene anticipato di tanto rispetto al fotogramma della camera 2
PAUSA_MINIMA_S = 0.5         # due cambi più vicini di così sono rumore, non cambi veri


def dati_da_raccolta(raccolta, righe):
    """Modo schermo del portatile: zona e luminosità di ogni fotogramma dalle immagini ridotte raccolte."""
    dati = {}
    for nome in raccolta.tempi:
        ridotte = raccolta.ridotte[nome]
        if len(ridotte) < 30:
            righe.append(f"{nome}: troppo pochi fotogrammi ({len(ridotte)})")
            return None
        maschera, escursione = zona_schermo(ridotte)
        if maschera is None:
            righe.append(f"{nome}: lo schermo non si vede (escursione {escursione:.0f} livelli)")
            return None
        dati[nome] = (raccolta.tempi[nome], np.array([r[maschera].mean() for r in ridotte]), maschera)
    return dati


def registra_esterno(acq, durata_s, righe):
    """Modo --esterno: registra per durata_s la luminosità della zona che lampeggia in ogni camera.
    Nei primi RICERCA_ZONA_S secondi conserva le immagini ridotte e vi cerca la zona; poi conserva solo
    la luminosità media della zona, un numero per fotogramma. Restituisce (dati, t_inizio) o None."""
    nomi = [l.nome for l in acq.lettori]
    ultimi, tempi, lum, ridotte, maschere = {}, {n: [] for n in nomi}, {n: [] for n in nomi}, {n: [] for n in nomi}, {}
    t_inizio = time.perf_counter()
    while True:
        trascorso = time.perf_counter() - t_inizio
        nuovi = acq.fotogrammi_dopo(ultimi)
        for nome, lista in nuovi.items():
            for f in lista:
                ultimi[nome] = f.indice
                piccola = cv2.resize(cv2.cvtColor(f.immagine, cv2.COLOR_BGR2GRAY), None, fx=RIDUZIONE, fy=RIDUZIONE,
                                     interpolation=cv2.INTER_AREA)
                tempi[nome].append(f.tempi)
                if nome in maschere:
                    lum[nome].append(float(piccola[maschere[nome]].mean()))
                else:
                    ridotte[nome].append(piccola)
        if not maschere and trascorso >= RICERCA_ZONA_S:
            for nome in nomi:
                maschera, escursione = zona_schermo(ridotte[nome])
                if maschera is None:
                    righe.append(f"{nome}: nessuna zona che lampeggia (escursione {escursione:.0f} livelli). "
                                 "Il dispositivo lampeggiava già? È visibile dalla camera?")
                    return None
                maschere[nome] = maschera
                lum[nome] = [float(r[maschera].mean()) for r in ridotte[nome]]
                ridotte[nome] = None
        # Anteprima con la zona trovata
        immagini = []
        for l in acq.lettori:
            fot = nuovi[l.nome][-1].immagine.copy() if nuovi[l.nome] else None
            if fot is None:
                continue
            if l.nome in maschere:
                grande = cv2.resize(maschere[l.nome].astype(np.uint8), (fot.shape[1], fot.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                contorni, _ = cv2.findContours(grande, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(fot, contorni, -1, (0, 0, 255), 4)
            immagini.append(fot)
        if len(immagini) == len(nomi):
            stato = (f"Ricerca della zona che lampeggia: {RICERCA_ZONA_S - trascorso:.0f} s" if not maschere else
                     f"Registrazione: mancano {durata_s - trascorso:.0f} s   (ESC = interrompi)")
            cv2.imshow(FINESTRA, affianca(immagini, ["Il contorno rosso deve stare sul dispositivo che lampeggia.",
                                                     stato]))
        tasto = cv2.waitKey(30) & 0xFF
        if tasto == 27 or trascorso >= durata_s:
            break
    if not maschere:
        righe.append("Interrotto prima di trovare la zona che lampeggia")
        return None
    return {n: (tempi[n], np.array(lum[n]), maschere[n]) for n in nomi}, t_inizio


def cambi_dal_video(t_arr, luminosita):
    """Cambi ricavati dalla luminosità di una camera, con isteresi: si passa a 'bianco' sopra il 70%
    dell'escursione e a 'nero' sotto il 30%, così il rumore vicino alla soglia non crea cambi falsi.
    L'istante restituito è l'arrivo del fotogramma cambiato meno ANTICIPO_S."""
    basso, alto = np.percentile(luminosita, 10), np.percentile(luminosita, 90)
    if alto - basso < CONTRASTO_MINIMO:
        return []
    soglia_su, soglia_giu = basso + 0.7 * (alto - basso), basso + 0.3 * (alto - basso)
    stato = int(luminosita[0] > (alto + basso) / 2)
    eventi = []
    for t, l in zip(t_arr, luminosita):
        nuovo = 1 if l > soglia_su else 0 if l < soglia_giu else stato
        if nuovo != stato:
            stato = nuovo
            e = t - ANTICIPO_S
            if eventi and e - eventi[-1][0] < PAUSA_MINIMA_S:
                eventi.pop()          # due cambi troppo vicini: rumore, si scartano entrambi
                continue
            eventi.append((e, stato))
    return eventi


def differenze(eventi, dati, t_fine, t0, righe, base, esterno=False):
    """Calcola g per ogni camera e cambio, poi d = g1 - g2. Restituisce (nomi, d in ms) oppure None.
    dati = {nome: (lista di Tempi, luminosità di ogni fotogramma, maschera della zona)}.
    Salva anche i file per la diagnosi: fotogrammi di ogni camera e valori di ogni cambio."""
    nomi = list(dati)
    g, salute = {}, {}
    for nome in nomi:
        lista_tempi, luminosita, maschera = dati[nome]
        # Istante sulla griglia NON corretto: è quello che si vuole misurare
        t_arr = np.array([t.t_arrivo for t in lista_tempi])
        t_gr = np.array([t.t_griglia for t in lista_tempi])
        risultati = latenze(eventi, t_arr, t_gr, luminosita, t_fine)
        g[nome] = risultati
        scarti = (t_arr - t_gr) * 1000
        salute[nome] = np.array([scarti[(t_arr > e - SALUTE_FINESTRA_S) & (t_arr <= e + SALUTE_FINESTRA_S)].mean()
                                 for e, _ in eventi])
        validi = [r["griglia_ms"] for r in risultati if r is not None]
        with open(f"{base}_{nome}_fotogrammi.csv", "w") as f:
            f.write("indice;t_arrivo_s;posto;t_griglia_s;luminosita\n")
            f.writelines(f"{t.indice};{t.t_arrivo - t0:.5f};{t.posto};{t.t_griglia - t0:.5f};{l:.1f}\n"
                         for t, l in zip(lista_tempi, luminosita))
        # Ritmo della camera DURANTE la misura: se rallenta (esposizione lunga) la griglia non è affidabile
        durante = t_arr[t_arr >= t0]
        intervalli = np.diff(durante) * 1000
        fps = (len(durante) - 1) / (durante[-1] - durante[0])
        # posizione della zona: con l'otturatore a scorrimento conta (vedi documento)
        ys, xs = np.nonzero(maschera)
        cy, cx = (ys.mean() + 0.5) / maschera.shape[0] * 100, (xs.mean() + 0.5) / maschera.shape[1] * 100
        ritardo = (f"ritardo medio sulla griglia {np.mean(validi):.0f} ms" if not esterno else
                   "(lampeggio esterno: ritardo assoluto non misurabile)")
        righe.append(f"{nome}: cambi misurati {len(validi)} su {len(eventi)}, zona {maschera.mean() * 100:.1f}% "
                     f"dell'immagine, centro della zona al {cx:.0f}% della larghezza e al {cy:.0f}% dell'altezza, "
                     + ritardo)
        righe.append(f"  durante la misura: {fps:.2f} fps, intervalli oltre 50 ms {np.mean(intervalli > 50) * 100:.1f}%, "
                     f"luminosità della zona da {np.percentile(luminosita, 5):.0f} a {np.percentile(luminosita, 95):.0f}")
    with open(f"{base}_cambi.csv", "w") as f:
        f.write(f"cambio;t_cambio_s;verso;griglia_{nomi[0]}_ms;griglia_{nomi[1]}_ms;differenza_ms;"
                f"salute_{nomi[0]}_ms;salute_{nomi[1]}_ms\n")
        for k, ((e, stato), a, b) in enumerate(zip(eventi, g[nomi[0]], g[nomi[1]])):
            valori = [f"{a['griglia_ms']:.2f}" if a else "", f"{b['griglia_ms']:.2f}" if b else "",
                      f"{a['griglia_ms'] - b['griglia_ms']:.2f}" if a and b else "",
                      f"{salute[nomi[0]][k]:.2f}", f"{salute[nomi[1]][k]:.2f}"]
            f.write(f"{k + 1};{e - t0:.4f};{'bianco' if stato else 'nero'};" + ";".join(valori) + "\n")
    # Si usano solo i cambi misurati da entrambe le camere con le griglie in salute
    buona = {n: np.abs(salute[n]) <= SALUTE_MAX_MS for n in nomi}
    for n in nomi:
        if not buona[n].all():
            righe.append(f"Esclusi {np.sum(~buona[n])} cambi: griglia di {n} fuori posto (ritardo medio "
                         f"entro {SALUTE_FINESTRA_S:.0f} s dal cambio oltre {SALUTE_MAX_MS:.0f} ms, massimo "
                         f"{np.abs(salute[n]).max():.0f} ms)")
    usati = [k for k, (a, b) in enumerate(zip(g[nomi[0]], g[nomi[1]]))
             if a and b and buona[nomi[0]][k] and buona[nomi[1]][k]]
    verso = np.array([eventi[k][1] for k in usati])
    d = np.array([g[nomi[0]][k]["griglia_ms"] - g[nomi[1]][k]["griglia_ms"] for k in usati])
    if len(d) < 10:
        righe.append(f"Solo {len(d)} cambi misurati da entrambe le camere: troppo pochi")
        return None
    # Le differenze buone stanno in due gruppi distanti un periodo (vedi sopra DISPERSIONE_MASSIMA_MS).
    # Un fotogramma perso messo nel posto sbagliato sposta la griglia di un ALTRO periodo: quei cambi
    # finiscono oltre 1,5 periodi dalla mediana e vanno esclusi. Una soglia di 1 periodo sarebbe
    # sbagliata: taglierebbe una parte del secondo gruppo e sposterebbe la media.
    lontani = np.abs(d - np.median(d)) > 1.5 * PERIODO_ATTESO_S * 1000
    if lontani.any():
        righe.append(f"Esclusi {lontani.sum()} cambi con differenza lontana più di 1,5 periodi dalla mediana")
    d, verso = d[~lontani], verso[~lontani]
    # Controlli di stabilità: la media non deve dipendere dal verso del cambio né dal momento della misura
    righe.append(f"Media della differenza per verso: verso il bianco {d[verso == 1].mean():+.1f} ms, "
                 f"verso il nero {d[verso == 0].mean():+.1f} ms")
    quarti = np.array_split(d, 4)
    righe.append("Media della differenza per quarti della misura: "
                 + "  ".join(f"{q.mean():+.1f}" for q in quarti) + " ms")
    return nomi, d


def main():
    parser = argparse.ArgumentParser(description="Test T05: sfasamento costante tra le camere")
    parser.add_argument("--transizioni", type=int, default=200, help="numero di cambi nero/bianco (default 200)")
    parser.add_argument("--etichetta", default="misura", help="nome della prova, usato nei file salvati")
    parser.add_argument("--salva", action="store_true", help="salva la correzione misurata")
    parser.add_argument("--esterno", action="store_true",
                        help="il lampeggio lo fa un altro dispositivo (lampeggio.html), non lo schermo del portatile")
    args = parser.parse_args()

    salvata = leggi_sincronizzazione()
    righe = []
    acq = AcquisizioneStereo(STREAM, sincronizza=False)
    try:
        if args.esterno:
            if not puntamento(acq, ("Fai lampeggiare il dispositivo al centro dell'area (lampeggio.html)",
                                    "e controlla che si veda in entrambe le immagini.")):
                return
            # circa un cambio al secondo con la pagina lampeggio.html
            registrazione = registra_esterno(acq, RICERCA_ZONA_S + args.transizioni + 2, righe)
        else:
            if not puntamento(acq):
                return
            eventi, raccolta, t_fine = misura(acq, args.transizioni, CAMPIONI_OGNI_S)
    finally:
        acq.chiudi()
        cv2.destroyAllWindows()

    risultato = None
    if args.esterno:
        if registrazione is not None:
            dati, t0 = registrazione
            nomi = list(dati)
            # cambi dalla camera 2: è la più regolare (esposizione fissa)
            t_rif = np.array([t.t_arrivo for t in dati[nomi[1]][0]])
            eventi = cambi_dal_video(t_rif, dati[nomi[1]][1])
            t_fine = min(dati[n][0][-1].t_arrivo for n in nomi)
        else:
            eventi, dati = [], None
    else:
        t0 = raccolta.t_inizio
        dati = dati_da_raccolta(raccolta, righe)
    origine = "del dispositivo esterno" if args.esterno else "dello schermo"
    righe = [f"Test T05 '{args.etichetta}' - {time.strftime('%Y-%m-%d %H:%M')} - {len(eventi)} cambi {origine}", ""] + righe
    os.makedirs(CARTELLA, exist_ok=True)
    if dati is not None and len(eventi) >= 10:
        risultato = differenze(eventi, dati, t_fine, t0, righe, os.path.join(CARTELLA, args.etichetta), args.esterno)
    elif dati is not None:
        righe.append("Troppo pochi cambi rilevati: il dispositivo lampeggiava? Si vedeva bene in entrambe le camere?")
    esito = False
    if risultato is not None:
        nomi, d = risultato
        media, dispersione = float(d.mean()), float(d.std(ddof=1))
        # Incertezza: std / radice(N) vale solo se i cambi sono indipendenti. Se la griglia di una camera
        # oscilla lentamente, cambi vicini sbagliano nello stesso verso: allora conta quanto variano le
        # medie di blocchi di 20 cambi. Si tiene la più grande delle due stime.
        incertezza = dispersione / np.sqrt(len(d))
        blocchi = [b.mean() for b in np.array_split(d, len(d) // 20)] if len(d) >= 60 else []
        if blocchi:
            incertezza = max(incertezza, float(np.std(blocchi, ddof=1) / np.sqrt(len(blocchi))))
        righe += ["", f"Differenza {nomi[0]} - {nomi[1]} sulla griglia, su {len(d)} cambi:",
                  f"  media {media:+.1f} ms, incertezza ±{incertezza:.1f} ms, dev.std {dispersione:.1f} ms"]
        righe.append(f"  (la dev.std dipende dallo sfasamento tra le griglie: da 0 a {DISPERSIONE_MASSIMA_MS:.1f} ms; "
                     f"molto di più indica una griglia instabile)")

        if salvata is not None:
            corr = salvata["correzione_ms"]
            gia = corr.get(nomi[0], 0.0) - corr.get(nomi[1], 0.0)
            residuo = media - gia
            residuo_ok = abs(residuo) <= RESIDUO_MAX_MS
            righe += [f"Correzione salvata il {salvata['data']}: {gia:+.1f} ms (±{salvata['incertezza_ms']:.1f})",
                      f"  residuo dopo la correzione: {residuo:+.1f} ms  "
                      f"(entro ±{RESIDUO_MAX_MS:.0f} ms: {'sì' if residuo_ok else 'NO'})"]
        if args.salva:
            precisa = incertezza <= INCERTEZZA_MAX_MS
            if precisa:
                os.makedirs(os.path.dirname(FILE_SINCRONIZZAZIONE), exist_ok=True)
                dati = {"data": time.strftime("%Y-%m-%d %H:%M"),
                        "descrizione": f"ritardo costante di ciascuna camera rispetto a {nomi[1]}, da sottrarre "
                                       "agli istanti sulla griglia",
                        "correzione_ms": {nomi[0]: round(media, 2), nomi[1]: 0.0},
                        "incertezza_ms": round(incertezza, 2), "dispersione_ms": round(dispersione, 2),
                        "cambi": int(len(d)), "etichetta": args.etichetta,
                        "lampeggio": "esterno" if args.esterno else "schermo del portatile"}
                with open(FILE_SINCRONIZZAZIONE, "w", encoding="utf-8") as f:
                    json.dump(dati, f, indent=2, ensure_ascii=False)
                righe.append(f"SALVATA in {FILE_SINCRONIZZAZIONE}: correzione {nomi[0]} {media:+.1f} ms")
            else:
                righe.append(f"NON salvata: incertezza ±{incertezza:.1f} ms oltre ±{INCERTEZZA_MAX_MS} ms, "
                             "servono più cambi (--transizioni)")
            esito = precisa
        else:
            esito = salvata is not None and residuo_ok
            if salvata is None:
                righe.append(f"Nessuna correzione salvata da verificare: eseguire prima con --salva")

        os.makedirs(CARTELLA, exist_ok=True)
        np.savetxt(os.path.join(CARTELLA, f"{args.etichetta}_differenze.csv"), d, fmt="%.2f",
                   header=f"differenza_{nomi[0]}_{nomi[1]}_ms", comments="")
    righe.append(f"ESITO {'MISURA' if args.salva else 'VERIFICA'} T05: {'SUPERATO' if esito else 'NON SUPERATO'}")

    testo = "\n".join(righe)
    print("\n" + testo)
    os.makedirs(CARTELLA, exist_ok=True)
    with open(os.path.join(CARTELLA, f"{args.etichetta}_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()