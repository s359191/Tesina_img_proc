python -m venv env  crea la cartella per l'ambiente (una sola volta da fare)

.\env\bin\Activate.ps1 (attiva l'ambiente)
in alternativa .\env\Scripts\activate
su Mac o Linux source env/bin/activate

per installare le librerie: 
pip install -r requirements.txt
In caso di problemi di versione: python.exe -m pip install --upgrade pip


Usare una versione di Python compresa tra la 3.8 e la 3.11 (le versioni 3.12, 3.13 o 3.14 non vanno bene).

Avere Python rigorosamente a 64-bit.