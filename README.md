### Link Utili
**Fondamenti teorici della visione stereo**: https://claude.ai/artifact/Fm2rUse2UzDmFBVbaksq8j#283d43a7-2b84

**Requisiti e piano dei test**: https://claude.ai/artifact/QCUceVMf9Cbcof7rUUrFXc#eda94ba0-8a2d

**Riepilogo lavoro svolto**: https://claude.ai/artifact/R4qgWu9iaC3YuZiTex76vf#4f8d6d0c-5e19

**Procuedura sessione di riprese**: https://claude.ai/artifact/7LBAqDTU2MJFd8Sbd81Xma#1bfe1251-72c1

**Codice spiegato**: https://claude.ai/artifact/2qyVyFnQ3hkDoXzri1ymkE#19f79d1c-3e91


### Le registrazioni sono su Google Drive
link: https://drive.google.com/drive/folders/10bBDKsIlyw7VacUE37H_D6e6Ulsk-A55?usp=drive_link

Per usarle:
1. scaricare la cartella della registrazione (es. 20261118_202321, con dentro coppie.csv e le immagini);
2. metterla in "TesinaIPCV\data\registrazioni\" 
3. per rielaborarla:
   `python mocap.py --da data\registrazioni\20261118_202321 --modello heavy --nome versione01`
   che scrive scheletro_heavy_versione01.csv accanto agli altri. Per non sovrascrivere altre rielaborazioni.

### Per provare Unity
Premere ▶︎ in Unity in Game e dopo lanciare da python `python invia_scheletro.py data\registrazioni\20261118_202321\scheletro_heavy.csv --ripeti`
In Unity: tasti (o pulsanti sullo schermo) **0** vista libera, **1** dal Redmi, **2** dall'S20 FE, **3** dall'alto.

### Problemi aperti e idee su cui lavorare
#### 1. Articolazioni nascoste (occlusioni)
Con le braccia incrociate o con il corpo girato di lato, alcune articolazioni spariscono dallo scheletro.
Un'articolazione viene scartata se:
- in almeno una delle due immagini MediaPipe la dà con **visibilità < 0,5** (`VISIBILITA_MIN` in `mocap.py`), oppure
- i due raggi non si incontrano: **errore di riproiezione > 15 px** (`RIPROIEZIONE_MAX_PX`), di solito perché ciascuna camera "indovina" il punto nascosto in un posto diverso, o perché MediaPipe scambia destra e sinistra.
