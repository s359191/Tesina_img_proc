"""
Invia a entrambi i telefoni tutte le impostazioni di IP Webcam e controlla
che siano state accettate.

Uso:   python imposta_camere.py            imposta e controlla
       python imposta_camere.py --elenco   mostra anche TUTTE le impostazioni e i valori ammessi
Da eseguire ogni volta che IP Webcam viene riaperta sui telefoni.
Usa solo librerie standard di Python: non serve installare nulla.
"""
import json
import sys
import time
import urllib.request

# Indirizzi dei telefoni: aggiornali se cambiano.
# Con il collegamento USB diventeranno "http://localhost:8080" e "http://localhost:8081".
TELEFONI = {
    "Redmi Note 12 Pro": "http://localhost:8080",   # adb -s SERIALE_REDMI forward tcp:8080 tcp:8080
    "Samsung S20 FE": "http://localhost:8081",      # adb -s SERIALE_S20 forward tcp:8081 tcp:8080
}

# Impostazioni uguali per i due telefoni, NELL'ORDINE in cui vanno inviate:
# prima il formato dell'immagine (può far ripartire la fotocamera),
# poi le funzioni da spegnere, poi colore, fuoco ed esposizione.
COMUNI = [
    ("video_size", "1280x720"),
    ("orientation", "landscape"),
    ("mirror_flip", "none"),
    ("ffc", "off"),
    ("zoom", "100"),
    ("crop_x", "50"),
    ("crop_y", "50"),
    ("quality", "80"),
    ("overlay", "off"),
    ("night_vision", "off"),
    ("motion_detect", "off"),
    ("motion_display", "off"),
    ("focus_homing", "off"),
    ("torch", "off"),
    ("flashmode", "off"),
    ("gps_active", "off"),
    ("coloreffect", "none"),
    ("scenemode", "auto"),
    ("whitebalance", "fluorescent"),   # stesso preset su entrambi
    ("focusmode", "off"),
    ("focus_distance", "0.60"),        # diottrie: 0.60 = circa 1,7 m
]

# Impostazioni diverse per ciascun telefono (aggiunte in fondo).
SPECIFICHE = {
    # Redmi: esposizione AUTOMATICA. I gradini di esposizione che IP Webcam offre sul Redmi
    # sono 0,1 ms, 300 ms, 600 ms...: 10 ms non è ottenibile (l'app usa il gradino più vicino,
    # 0,1 ms, e l'immagine esce quasi nera). L'antibanding a 50 Hz fa scegliere all'esposizione
    # automatica tempi che evitano lo sfarfallio delle lampade.
    "Redmi Note 12 Pro": [
        ("manual_sensor", "off"),
        ("antibanding", "50hz"),
    ],
    # S20 FE: esposizione MANUALE (gradini da ~1,25 ms, quindi 10 ms è ottenibile).
    # Il controllo manuale va acceso prima di esposizione, durata del fotogramma e ISO.
    "Samsung S20 FE": [
        ("antibanding", "off"),          # non ha l'opzione 50hz; in manuale non serve
        ("manual_sensor", "on"),
        ("exposure_ns", "10000000"),     # 10 ms: un ciclo intero di luce a 100 Hz
        ("frame_duration", "33333333"),  # 33,3 ms = 30 fps
        ("iso", "400"),
    ],
}

# Queste due non si impostano con /settings (il telefono risponde 404): si controlla solo che siano spente.
# La torcia, se accesa, si spegne con l'indirizzo /disabletorch.
SOLO_CONTROLLO = {"focus_homing", "torch"}

# Dopo queste impostazioni la fotocamera può ripartire: aspettiamo un attimo.
RIAVVIO_CAMERA = {"video_size", "orientation", "ffc"}


def invia(base, nome, valore):
    """Manda un'impostazione al telefono. Restituisce True se la richiesta è andata a buon fine."""
    url = f"{base}/settings/{nome}?set={valore}"
    try:
        with urllib.request.urlopen(url, timeout=5):
            pass
        return True
    except Exception as errore:
        print(f"   ERRORE inviando {nome}={valore}: {errore}")
        return False


def leggi_stato(base, completo=False):
    """Legge i valori attuali del telefono (la sezione 'curvals' di status.json).
    Con completo=True restituisce tutto status.json, compresi i valori ammessi ('avail')."""
    with urllib.request.urlopen(f"{base}/status.json?show_avail=1", timeout=5) as risposta:
        dati = json.load(risposta)
    return dati if completo else dati["curvals"]


# La stabilizzazione video ritaglia l'immagine e sposta il ritaglio seguendo il giroscopio: per la
# calibrazione è come se la focale e il punto principale cambiassero. Va spenta. Il nome
# dell'impostazione dipende dalla versione di IP Webcam, quindi la cerchiamo per parole chiave.
PAROLE_STABILIZZAZIONE = ("stab", "ois", "eis")


def spegni_stabilizzazione(base):
    """Spegne ogni impostazione di stabilizzazione esposta da IP Webcam. Restituisce True se è tutto spento."""
    dati = leggi_stato(base, completo=True)
    valori, ammessi = dati.get("curvals", {}), dati.get("avail", {})
    chiavi = [k for k in valori if any(p in k.lower() for p in PAROLE_STABILIZZAZIONE)]
    if not chiavi:
        print("   Stabilizzazione: IP Webcam non espone nessuna impostazione. Controlla nell'app "
              "(preferenze video) che sia spenta.")
        return True
    tutto_ok = True
    for k in chiavi:
        spento = next((v for v in ("off", "false", "0", "disabled", "none") if v in ammessi.get(k, ["off"])), "off")
        prima = valori[k]
        if prima != spento:
            invia(base, k, spento)
            time.sleep(1.0)
        dopo = leggi_stato(base).get(k)
        ok = dopo == spento
        tutto_ok &= ok
        print(f"   Stabilizzazione '{k}': era {prima}, ora {dopo}" + ("" if ok else "   <- NON spenta!"))
    return tutto_ok


def elenca(base):
    """Stampa tutte le impostazioni con il valore attuale e i valori ammessi."""
    dati = leggi_stato(base, completo=True)
    for k, v in sorted(dati.get("curvals", {}).items()):
        ammessi = dati.get("avail", {}).get(k)
        print(f"   {k:28s} = {v!s:20s}" + (f"  ammessi: {ammessi}" if ammessi else ""))


def configura(nome_telefono, base):
    print(f"\n=== {nome_telefono} ({base}) ===")
    impostazioni = COMUNI + SPECIFICHE[nome_telefono]

    # Prima di tutto: il telefono risponde?
    try:
        leggi_stato(base)
    except Exception as errore:
        print(f"   Il telefono non risponde ({errore}).")
        print("   Controlla che IP Webcam sia avviata e che l'indirizzo sia giusto.")
        return False

    for nome, valore in impostazioni:
        if nome in SOLO_CONTROLLO:
            continue
        invia(base, nome, valore)
        time.sleep(2.0 if nome in RIAVVIO_CAMERA else 0.2)
    if leggi_stato(base).get("torch") == "on":
        try:
            urllib.request.urlopen(f"{base}/disabletorch", timeout=5).close()
        except Exception as errore:
            print(f"   ERRORE spegnendo la torcia: {errore}")
    stabilizzazione_ok = spegni_stabilizzazione(base)

    # Controllo: ogni valore letto dal telefono deve coincidere con quello inviato.
    time.sleep(1.0)
    try:
        stato = leggi_stato(base)
    except Exception as errore:
        print(f"   Impossibile leggere lo stato: {errore}")
        return False

    tutto_ok = True
    for nome, valore in impostazioni:
        letto = stato.get(nome)
        if letto != valore:
            print(f"   DIVERSO  {nome}: inviato {valore}, il telefono riporta {letto}")
            tutto_ok = False
    tutto_ok = tutto_ok and stabilizzazione_ok
    print("   Tutte le impostazioni confermate." if tutto_ok else "   Controlla le righe DIVERSO.")
    if "--elenco" in sys.argv:
        elenca(base)
    return tutto_ok


if __name__ == "__main__":
    risultati = [configura(nome, base) for nome, base in TELEFONI.items()]
    print("\nPRONTO: entrambi i telefoni sono configurati." if all(risultati)
          else "\nATTENZIONE: almeno un telefono non è configurato correttamente.")