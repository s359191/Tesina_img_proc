"""
Invia a Unity uno scheletro già salvato da mocap.py, con i tempi originali, come se fosse dal vivo.

Serve a lavorare sulla scena Unity senza montare i telefoni: i pacchetti UDP sono identici a
quelli di mocap.py --udp ({"t": istante, "punti": [[x, y, z] o null, ...]}, sistema del pavimento).

Uso:
  python invia_scheletro.py data/scheletri/scheletro_20261118_193110.csv
  python invia_scheletro.py FILE --ripeti            (ricomincia da capo alla fine)
  python invia_scheletro.py FILE --udp 127.0.0.1:5065 --velocita 0.5    (al rallentatore)
Ctrl+C = fine.
"""
import argparse
import json
import socket
import time


def leggi_scheletro(percorso):
    """Righe del CSV di mocap.py: (istante della camera 1, 33 punti o None)."""
    righe = []
    with open(percorso, encoding="utf-8") as f:
        next(f)                                           # intestazione
        for riga in f:
            campi = riga.rstrip("\n").split(";")
            if len(campi) < 4 + 33 * 3:
                continue
            valori = campi[4:4 + 33 * 3]
            punti = []
            for i in range(33):
                x, y, z = valori[3 * i:3 * i + 3]
                punti.append([float(x), float(y), float(z)] if x else None)
            righe.append((float(campi[1]), punti))
    return righe


def main():
    parser = argparse.ArgumentParser(description="Invia a Unity uno scheletro salvato")
    parser.add_argument("file", help="scheletro_<data>.csv salvato da mocap.py")
    parser.add_argument("--udp", default="127.0.0.1:5065", help="indirizzo:porta di Unity")
    parser.add_argument("--velocita", type=float, default=1.0, help="1 = tempo reale, 0.5 = metà velocità")
    parser.add_argument("--ripeti", action="store_true", help="ricomincia quando il file finisce")
    args = parser.parse_args()

    righe = leggi_scheletro(args.file)
    if not righe:
        raise SystemExit("Nessuna riga valida nel file")
    host, porta = args.udp.rsplit(":", 1)
    destinazione = (host, int(porta))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    durata = righe[-1][0] - righe[0][0]
    print(f"{len(righe)} coppie, {durata:.0f} s. Invio a {host}:{porta} ... (Ctrl+C = fine)")
    try:
        while True:
            inizio, t0 = time.perf_counter(), righe[0][0]
            for n, (t, punti) in enumerate(righe):
                attesa = (t - t0) / args.velocita - (time.perf_counter() - inizio)
                if attesa > 0:
                    time.sleep(attesa)
                sock.sendto(json.dumps({"t": round(t, 4), "punti": punti}).encode("utf-8"), destinazione)
                if n % 300 == 0:
                    print(f"  {t - t0:6.1f} s di {durata:.0f}")
            if not args.ripeti:
                break
            print("  ricomincio")
    except KeyboardInterrupt:
        pass
    print("Fine.")


if __name__ == "__main__":
    main()
