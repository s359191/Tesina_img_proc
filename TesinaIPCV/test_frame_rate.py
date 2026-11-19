"""
Fase 2 - Test T02: stabilità del frame rate.

Registra l'istante di arrivo di OGNI fotogramma delle due camere, senza finestra
(così il PC non interferisce), e analizza gli intervalli tra fotogrammi consecutivi.
La forma della distribuzione degli intervalli indica la causa di eventuali problemi:
  - intervalli di ~33 ms con alcuni di ~66 ms   -> il telefono salta fotogrammi
  - intervalli regolari ma più lunghi di 33 ms   -> la camera rallenta (es. esposizione)
  - buchi seguiti da intervalli quasi nulli       -> fotogrammi arrivati "a grappoli"

Uso:   python test_frame_rate.py --durata 600 --etichetta redmi_q80
       (Ctrl+C per interrompere prima: l'analisi viene fatta comunque)
Risultati: a schermo e nella cartella data/test_fps/
"""
import argparse
import os
import time

import numpy as np

from acquisizione import STREAM, PERIODO_ATTESO_S, SOGLIA_ABBINAMENTO_S, AcquisizioneStereo

CARTELLA = os.path.join("data", "test_fps")
P_MS = PERIODO_ATTESO_S * 1000                  # 33,3 ms
CLASSI_MS = [0, 10, 25, 42, 58, 75, 1e9]        # limiti delle classi dell'istogramma
NOMI_CLASSI = ["< 10 ms (grappolo)", "10-25 ms", "25-42 ms (normale)", "42-58 ms",
               "58-75 ms (1 perso)", "> 75 ms (2+ persi)"]


def analizza(nome, registro, griglia):
    """Restituisce le righe di testo del riepilogo e l'esito del test per una camera."""
    t = np.array([r.t_arrivo for r in registro])
    if len(t) < 3:
        return [f"{nome}: troppo pochi fotogrammi ({len(t)})"], False
    iv = np.diff(t) * 1000                                        # intervalli in ms
    fps = (len(t) - 1) / (t[-1] - t[0])
    oltre_50 = np.mean(iv > 1.5 * P_MS) * 100                     # almeno un fotogramma perso
    oltre_66 = np.mean(iv > 66) * 100                             # soglia del requisito RA1
    grappoli = np.mean(iv < 10) * 100
    # fotogrammi attesi nella durata del test meno quelli ricevuti: un buco seguito da un
    # grappolo non conta come perdita, perché i fotogrammi sono arrivati tutti
    persi = max(0, int(round((t[-1] - t[0]) / (P_MS / 1000))) + 1 - len(t))
    superato = fps >= 29 and oltre_66 < 1
    righe = [f"{nome}: {len(t)} fotogrammi in {t[-1] - t[0]:.0f} s -> {fps:.2f} fps medi",
             f"  intervalli: mediana {np.median(iv):.1f} ms, 95% sotto {np.percentile(iv, 95):.1f} ms, "
             f"massimo {iv.max():.0f} ms",
             f"  oltre 50 ms: {oltre_50:.2f}%   oltre 66 ms: {oltre_66:.2f}%   "
             f"sotto 10 ms: {grappoli:.2f}%   mancanti rispetto a 30 fps: {persi}"]
    conteggi, _ = np.histogram(iv, bins=CLASSI_MS)
    for nome_classe, c in zip(NOMI_CLASSI, conteggi):
        quota = c / len(iv) * 100
        righe.append(f"    {nome_classe:<20} {quota:6.2f}%  {'#' * int(round(quota / 2))}")
    # Griglia degli scatti: ritardo variabile = arrivo - istante sulla griglia (dopo la prima stima)
    stimati = [r for r in registro if r.t_griglia != r.t_arrivo]
    if stimati and griglia.P_libero is not None:
        ritardo = np.array([(r.t_arrivo - r.t_griglia) * 1000 for r in stimati])
        righe.append(f"  griglia: periodo {griglia.P * 1000:.3f} ms (stima libera {griglia.P_libero * 1000:.3f} ms), "
                     f"ritardo variabile dev.std {ritardo.std():.1f} ms, posti vuoti {griglia.mancanti()}")
        if abs(griglia.P_libero / griglia.P0 - 1) > 0.01:
            righe.append("  ATTENZIONE: il periodo stimato si discosta più dell'1% da quello nominale: "
                         "la camera non lavora a 30 fps e la griglia non è affidabile")
    righe.append(f"  T02 (>= 29 fps e oltre 66 ms < 1%): {'SUPERATO' if superato else 'NON SUPERATO'}")
    return righe, superato


def main():
    parser = argparse.ArgumentParser(description="Test T02: stabilità del frame rate")
    parser.add_argument("--durata", type=float, default=600, help="durata in secondi (default 600)")
    parser.add_argument("--etichetta", default="prova", help="nome della prova, usato nei file salvati")
    args = parser.parse_args()

    acq = AcquisizioneStereo(STREAM, registra=True)
    print(f"Test di {args.durata:.0f} s avviato ('{args.etichetta}'). Ctrl+C per interrompere.")
    scarti_ms, scarti_arrivi_ms = [], []
    inizio = time.perf_counter()
    prossimo_avviso = inizio + 10
    try:
        while time.perf_counter() - inizio < args.durata:
            coppia = acq.prossima_coppia(timeout_s=2.0)
            if coppia is None:
                print("  nessuna coppia da 2 s: un telefono ha smesso di trasmettere?")
            else:
                scarti_ms.append(coppia.scarto_s * 1000)
                scarti_arrivi_ms.append(coppia.scarto_arrivi_s * 1000)
            if time.perf_counter() >= prossimo_avviso:
                stato = "   ".join(f"{l.nome} {l.fps():.1f} fps, persi (stima) {l.mancanti()}" for l in acq.lettori)
                print(f"  {time.perf_counter() - inizio:5.0f} s   {stato}")
                prossimo_avviso += 10
    except KeyboardInterrupt:
        print("Interrotto: analizzo i dati raccolti.")
    finally:
        acq.chiudi()

    durata = time.perf_counter() - inizio
    righe = [f"Test T02 '{args.etichetta}' - {time.strftime('%Y-%m-%d %H:%M')} - {durata:.0f} s", ""]
    os.makedirs(CARTELLA, exist_ok=True)
    esiti = []
    # Origine dei tempi COMUNE alle due camere: così nei due CSV gli istanti restano confrontabili
    # (con un'origine per camera si perderebbe lo sfasamento tra le due).
    t0 = min((l.registro[0].t_arrivo for l in acq.lettori if l.registro), default=0)
    for l in acq.lettori:
        r, ok = analizza(l.nome, l.registro, l.griglia)
        righe += r + [""]
        esiti.append(ok)
        # tutti i tempi su file, per analisi successive (es. grafici nella tesina)
        with open(os.path.join(CARTELLA, f"{args.etichetta}_{l.nome}.csv"), "w") as f:
            f.write("indice;t_arrivo_s;posto;t_griglia_s\n")
            f.writelines(f"{r.indice};{r.t_arrivo - t0:.5f};{r.posto};{r.t_griglia - t0:.5f}\n"
                         for r in l.registro)
    soglia_ms = SOGLIA_ABBINAMENTO_S * 1000
    if scarti_ms:
        s, sa = np.abs(scarti_ms), np.abs(scarti_arrivi_ms)
        senza = ", ".join(f"{l.nome} {len(l.registro) - len(s)}" for l in acq.lettori)
        righe.append(f"Coppie: {len(s)} ({len(s) / durata:.1f}/s)   fotogrammi senza compagno: {senza}")
        # Per costruzione lo scarto sulla griglia non supera mai la soglia (le coppie più lontane
        # vengono scartate): la percentuale "entro soglia" non avrebbe significato. Lo scarto vero
        # tra gli scatti si potrà misurare solo nella fase 3.
        righe.append(f"  scarto stimato sulla griglia: medio {s.mean():.1f} ms, massimo {s.max():.1f} ms "
                     f"(su registrazioni lunghe tende a circa {soglia_ms / 2:.1f} ms)")
        righe.append(f"  scarto tra gli arrivi delle stesse coppie: medio {sa.mean():.1f} ms, massimo {sa.max():.1f} ms, "
                     f"oltre {soglia_ms:.1f} ms: {np.mean(sa > soglia_ms) * 100:.1f}%")
    righe.append(f"ESITO T02: {'SUPERATO' if all(esiti) else 'NON SUPERATO'}")
    testo = "\n".join(righe)
    print("\n" + testo)
    with open(os.path.join(CARTELLA, f"{args.etichetta}_riepilogo.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()