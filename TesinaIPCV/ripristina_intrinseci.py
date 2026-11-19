"""
Ripristina le calibrazioni monoculari (K e distorsione) dalla calibrazione stereo salvata.

Serve quando i file data/calibrazione/<camera>.yaml sono stati sovrascritti da una calibrazione
monoculare peggiore, ma la calibrazione stereo (stereo.yaml, variante «intrinseci fissi») contiene
ancora al suo interno i valori buoni: K1/dist1 per la camera 1 e K2/dist2 per la camera 2.

I file attuali non vengono persi: sono rinominati in <camera>_sostituita.yaml.

Uso (PRIMA di rifare la calibrazione stereo, che sovrascrive stereo.yaml):
  python ripristina_intrinseci.py
  python ripristina_intrinseci.py --da data/calibrazione/stereo.yaml
"""
import argparse
import os
import shutil
import time

import cv2

from calibrazione_mono import CARTELLA


def main():
    parser = argparse.ArgumentParser(description="Ripristino di K e distorsione da stereo.yaml")
    parser.add_argument("--da", default=os.path.join(CARTELLA, "stereo.yaml"), help="file stereo da cui leggere")
    args = parser.parse_args()

    fs = cv2.FileStorage(args.da, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"Non trovo {args.da}")
    variante = fs.getNode("intrinseci").string()
    if variante and variante != "intrinseci fissi":
        raise SystemExit(f"{args.da} è la variante '{variante}': le sue K non sono quelle della calibrazione "
                         "monoculare. Usa la variante 'intrinseci fissi'.")
    data_stereo = fs.getNode("data").string()
    larghezza, altezza = int(fs.getNode("larghezza").real()), int(fs.getNode("altezza").real())
    camere = [(fs.getNode("camera1").string(), fs.getNode("K1").mat(), fs.getNode("dist1").mat(),
               fs.getNode("rotazione1").string()),
              (fs.getNode("camera2").string(), fs.getNode("K2").mat(), fs.getNode("dist2").mat(),
               fs.getNode("rotazione2").string())]
    fs.release()

    for camera, K, dist, rotazione in camere:
        if not camera or K is None or dist is None:
            raise SystemExit(f"{args.da} non contiene camera, K e distorsione di entrambe le camere")
        percorso = os.path.join(CARTELLA, f"{camera}.yaml")
        if os.path.exists(percorso):
            vecchio = os.path.join(CARTELLA, f"{camera}_sostituita.yaml")
            shutil.copy2(percorso, vecchio)
            print(f"{percorso} -> copia di sicurezza in {vecchio}")
        out = cv2.FileStorage(percorso, cv2.FILE_STORAGE_WRITE)
        out.write("camera", camera)
        out.write("data", time.strftime("%Y-%m-%d %H:%M"))
        out.write("ripristinata_da", f"{args.da} ({data_stereo})")
        out.write("larghezza", larghezza)
        out.write("altezza", altezza)
        out.write("rotazione", rotazione)
        out.write("K", K)
        out.write("dist", dist.reshape(1, -1))
        out.release()
        print(f"{camera}: fx = {K[0, 0]:.1f}  fy = {K[1, 1]:.1f}  cx = {K[0, 2]:.1f}  cy = {K[1, 2]:.1f}  -> {percorso}")


if __name__ == "__main__":
    main()
