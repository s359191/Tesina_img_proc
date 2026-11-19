using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

// Riceve lo scheletro da mocap.py --udp (o da invia_scheletro.py) e lo mostra con 33 sfere
// (le articolazioni di MediaPipe) e dei bastoncini (le ossa).
// Ogni pacchetto UDP è un testo JSON: {"t": istante, "punti": [[x, y, z] oppure null, ... 33 volte]}
// con le coordinate nel sistema del pavimento (x destra, y verso le camere, z in alto; metri).
// In Unity diventano (X, Y, Z) = (x, z, y): stessi assi di scena.json (teoria 4.11).
public class RicevitoreScheletro : MonoBehaviour
{
    [SerializeField] int porta = 5065;
    [SerializeField] float diametroArticolazione = 0.05f;
    [SerializeField] float spessoreOsso = 0.02f;
    [Tooltip("Dopo quanti secondi senza pacchetti lo scheletro sparisce")]
    [SerializeField] float secondiSenzaDati = 0.5f;
    [Tooltip("Mostra le sfere e le ossa (si possono nascondere quando c'è l'avatar)")]
    [SerializeField] bool mostraSfere = true;

    [Header("Filtro One Euro (toglie il tremolio)")]
    [SerializeField] bool filtroAttivo = true;
    [Tooltip("Hz. Più basso = meno tremolio da fermo")]
    [SerializeField] float frequenzaMinima = 1.0f;
    [Tooltip("Più alto = meno ritardo nei movimenti veloci")]
    [SerializeField] float beta = 1.5f;

    // Coppie di articolazioni da unire (stessa numerazione di MediaPipe e di mocap.py)
    static readonly int[,] Ossa =
    {
        {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {11, 23}, {12, 24}, {23, 24},
        {23, 25}, {25, 27}, {27, 29}, {29, 31}, {27, 31}, {24, 26}, {26, 28}, {28, 30}, {30, 32}, {28, 32},
        {0, 11}, {0, 12},
    };

    // --- parte condivisa tra il thread di ricezione e il thread principale di Unity ---
    readonly object blocco = new object();
    string ultimoMessaggio;          // l'ultimo pacchetto arrivato (quelli più vecchi non servono)
    int pacchettiRicevuti;

    UdpClient client;
    Thread threadRicezione;
    volatile bool inAscolto;

    readonly Transform[] articolazioni = new Transform[33];
    Transform[] ossa;
    readonly Vector3[] grezze = new Vector3[33];      // come arrivano da mocap.py
    readonly Vector3[] posizioni = new Vector3[33];   // dopo il filtro
    readonly bool[] valide = new bool[33];
    readonly FiltroOneEuro[] filtri = new FiltroOneEuro[33];
    float ultimoArrivo = -100f;
    double ultimoIstante, istantePrecedente = -1;     // double: l'orologio di Python può valere centinaia di migliaia di secondi

    // ---- Per gli altri script (AvatarDaScheletro): punti filtrati, in metri, negli assi di Unity ----
    public bool HaDati => Time.time - ultimoArrivo <= secondiSenzaDati;
    public bool Valido(int i) => valide[i];
    public Vector3 Punto(int i) => posizioni[i];
    int puntiValidi, pacchettiAlSecondo, contatore;
    float inizioConteggio;

    void Start()
    {
        CreaScheletro();
        for (int i = 0; i < 33; i++) filtri[i] = new FiltroOneEuro(frequenzaMinima, beta);
        // Il socket UDP si mette in ascolto sulla porta; Receive() blocca finché arriva un pacchetto,
        // per questo sta in un thread separato: il thread principale di Unity non deve mai fermarsi.
        try
        {
            client = new UdpClient(porta);
        }
        catch (SocketException)
        {
            Debug.LogError($"La porta UDP {porta} è già occupata (un altro Play o un altro programma?)");
            return;
        }
        inAscolto = true;
        threadRicezione = new Thread(Ricevi) { IsBackground = true, Name = "Ricezione UDP" };
        threadRicezione.Start();
        Debug.Log($"In ascolto sulla porta UDP {porta}");
    }

    void Ricevi()
    {
        IPEndPoint mittente = new IPEndPoint(IPAddress.Any, 0);
        while (inAscolto)
        {
            try
            {
                byte[] dati = client.Receive(ref mittente);
                string testo = Encoding.UTF8.GetString(dati);
                lock (blocco)
                {
                    ultimoMessaggio = testo;
                    pacchettiRicevuti++;
                }
            }
            catch (SocketException) { }              // succede anche quando il socket viene chiuso
            catch (ObjectDisposedException) { break; }
        }
    }

    void Update()
    {
        // 1. prende l'ultimo pacchetto (se ne è arrivato uno nuovo)
        string messaggio;
        lock (blocco)
        {
            messaggio = ultimoMessaggio;
            ultimoMessaggio = null;
            contatore += pacchettiRicevuti;
            pacchettiRicevuti = 0;
        }
        if (Time.time - inizioConteggio >= 1f)
        {
            pacchettiAlSecondo = contatore;
            contatore = 0;
            inizioConteggio = Time.time;
        }

        // 2. lo legge e sposta sfere e ossa
        if (messaggio != null && Leggi(messaggio, grezze, valide, out ultimoIstante))
        {
            ultimoArrivo = Time.time;
            // secondi tra questo pacchetto e il precedente, dagli istanti di scatto (non dal frame di Unity)
            float dt = istantePrecedente < 0 ? 0f : (float)(ultimoIstante - istantePrecedente);
            istantePrecedente = ultimoIstante;
            puntiValidi = 0;
            for (int i = 0; i < 33; i++)
            {
                if (!valide[i]) filtri[i].Azzera();
                else if (filtroAttivo)
                {
                    filtri[i].frequenzaMinima = frequenzaMinima;   // così si possono cambiare durante il Play
                    filtri[i].beta = beta;
                    posizioni[i] = filtri[i].Filtra(grezze[i], dt);
                }
                else posizioni[i] = grezze[i];
                articolazioni[i].gameObject.SetActive(valide[i]);
                if (valide[i])
                {
                    articolazioni[i].position = posizioni[i];
                    puntiValidi++;
                }
            }
            for (int k = 0; k < ossa.Length; k++)
            {
                int a = Ossa[k, 0], b = Ossa[k, 1];
                bool visibile = valide[a] && valide[b];
                ossa[k].gameObject.SetActive(visibile);
                if (visibile) Forme.Segmento(ossa[k], posizioni[a], posizioni[b], spessoreOsso);
            }
        }

        // 3. se non arriva più niente (o se le sfere sono spente), nasconde lo scheletro
        GameObject persona = articolazioni[0].parent.gameObject;
        bool mostra = HaDati && mostraSfere;
        if (persona.activeSelf != mostra) persona.SetActive(mostra);
    }

    // Legge il JSON di mocap.py. JsonUtility di Unity non sa leggere liste di liste né i null,
    // quindi lo leggiamo a mano: dopo "punti" ci sono 33 elementi, ciascuno "null" oppure "[x, y, z]".
    static bool Leggi(string testo, Vector3[] posizioni, bool[] valide, out double istante)
    {
        istante = 0f;
        int t = testo.IndexOf("\"t\":", StringComparison.Ordinal);
        if (t >= 0)
        {
            int fine = testo.IndexOf(',', t);
            double.TryParse(testo.Substring(t + 4, fine - t - 4), NumberStyles.Float, CultureInfo.InvariantCulture, out istante);
        }
        int i = testo.IndexOf("\"punti\"", StringComparison.Ordinal);
        if (i < 0) return false;
        i = testo.IndexOf('[', i) + 1;                           // dentro la lista esterna
        for (int k = 0; k < 33; k++)
        {
            while (i < testo.Length && (char.IsWhiteSpace(testo[i]) || testo[i] == ',')) i++;
            if (i >= testo.Length) return false;
            if (testo[i] == 'n')                                  // null: articolazione non ricostruita
            {
                valide[k] = false;
                i += 4;
            }
            else if (testo[i] == '[')
            {
                int fine = testo.IndexOf(']', i);
                string[] c = testo.Substring(i + 1, fine - i - 1).Split(',');
                // InvariantCulture: il punto decimale è "." anche su un Windows in italiano (che usa ",")
                float x = float.Parse(c[0], CultureInfo.InvariantCulture);
                float y = float.Parse(c[1], CultureInfo.InvariantCulture);
                float z = float.Parse(c[2], CultureInfo.InvariantCulture);
                posizioni[k] = new Vector3(x, z, y);              // pavimento (x, y, z) -> Unity (X, Y, Z) = (x, z, y)
                valide[k] = true;
                i = fine + 1;
            }
            else return false;
        }
        return true;
    }

    void CreaScheletro()
    {
        Transform persona = new GameObject("Persona").transform;
        persona.SetParent(transform, false);
        for (int i = 0; i < 33; i++)
        {
            // Colori: lato sinistro della persona blu, lato destro rosso, naso bianco.
            // Numerazione di MediaPipe: 1-3 occhio sx, 4-6 occhio dx, poi dispari = sinistra, pari = destra.
            bool sinistra = (i >= 1 && i <= 3) || (i >= 7 && i % 2 == 1);
            Color colore = i == 0 ? Color.white : sinistra ? new Color(0.2f, 0.4f, 1f) : new Color(1f, 0.25f, 0.2f);
            GameObject sfera = Forme.Crea(PrimitiveType.Sphere, $"Articolazione {i}", persona, colore);
            float diametro = i <= 10 ? diametroArticolazione * 0.6f : diametroArticolazione;   // viso più piccolo
            sfera.transform.localScale = Vector3.one * diametro;
            articolazioni[i] = sfera.transform;
        }
        ossa = new Transform[Ossa.GetLength(0)];
        for (int k = 0; k < ossa.Length; k++)
        {
            ossa[k] = Forme.Crea(PrimitiveType.Cylinder, $"Osso {Ossa[k, 0]}-{Ossa[k, 1]}", persona,
                                 new Color(0.9f, 0.9f, 0.9f)).transform;
            ossa[k].gameObject.SetActive(false);
        }
        persona.gameObject.SetActive(false);
    }

    void OnGUI()
    {
        string stato = Time.time - ultimoArrivo > secondiSenzaDati
            ? $"UDP {porta}: nessun dato (avvia mocap.py --udp 127.0.0.1:{porta} o invia_scheletro.py)"
            : $"UDP {porta}: {pacchettiAlSecondo} pacchetti/s, {puntiValidi}/33 articolazioni, t = {ultimoIstante:F2}";
        GUI.Label(new Rect(10, 50, 800, 22), stato);
    }

    void OnDestroy()
    {
        // Chiudere il socket sblocca Receive() nel thread, che così termina
        inAscolto = false;
        client?.Close();
        threadRicezione?.Join(500);
    }
}
