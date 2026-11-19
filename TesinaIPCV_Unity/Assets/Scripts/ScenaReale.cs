using System;
using System.IO;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// Ricostruisce in Unity, in scala 1:1, la stanza delle riprese: pavimento, croci e i due telefoni,
// leggendo scena.json scritto da esporta_unity.py. Le coordinate del file sono già negli assi di Unity
// (X destra guardando le camere, Y in alto, Z verso le camere; origine = croce 1; metri).
//
// Tasti durante Play:  0 = vista libera da dietro le camere   1 = vista dalla camera 1 (Redmi)
//                      2 = vista dalla camera 2 (S20FE)        3 = vista dall'alto
// Nelle viste 1 e 2 la Main Camera prende posizione, orientamento e obiettivo (K) del telefono vero.
public class ScenaReale : MonoBehaviour
{
    [Tooltip("Percorso di scena.json (esporta_unity.py). Un percorso relativo parte da Assets/StreamingAssets.")]
    [SerializeField] string fileScena = @"C:\Users\clara\PycharmProjects\TesinaIPCV\data\unity\scena.json";
    [Tooltip("Lato del pavimento grigio, in metri")]
    [SerializeField] float latoPavimento = 6f;
    [Tooltip("Lunghezza dei bracci delle croci, in metri")]
    [SerializeField] float braccioCroce = 0.20f;
    [Tooltip("La camera che cambia vista con i tasti 0-3 (di solito la Main Camera)")]
    [SerializeField] Camera telecamera;

    // ---- Struttura di scena.json: JsonUtility riempie questi campi con gli stessi nomi del file ----
    [Serializable] public class Croce { public int numero; public float x, y, z; }
    [Serializable]
    public class DatiCamera
    {
        public string nome;
        public Vector3 posizione, avanti, su;     // un Vector3 nel JSON è {"x": .., "y": .., "z": ..}
        public int larghezza, altezza;
        public float fx, fy, cx, cy;
    }
    [Serializable]
    public class DatiScena
    {
        public string data, calibrazione_stereo, pavimento;
        public Croce[] croci;
        public DatiCamera[] camere;
    }

    public DatiScena Scena { get; private set; }
    string vistaAttuale = "";

    void Start()
    {
        if (telecamera == null) telecamera = Camera.main;
        string percorso = Path.IsPathRooted(fileScena) ? fileScena
                          : Path.Combine(Application.streamingAssetsPath, fileScena);
        if (!File.Exists(percorso))
        {
            Debug.LogError($"Non trovo {percorso}: esegui python esporta_unity.py e controlla il campo File Scena.");
            return;
        }
        Scena = JsonUtility.FromJson<DatiScena>(File.ReadAllText(percorso));
        Debug.Log($"Scena caricata: calibrazione stereo del {Scena.calibrazione_stereo}, " +
                  $"pavimento del {Scena.pavimento}, {Scena.croci.Length} croci, {Scena.camere.Length} camere");

        CreaPavimento();
        CreaAssi();
        foreach (Croce c in Scena.croci) CreaCroce(c);
        foreach (DatiCamera c in Scena.camere) CreaTelefono(c);
        VistaLibera();
    }

    void Update()
    {
        if (Scena == null) return;
        for (int n = 0; n <= 3; n++)
            if (Premuto(n)) CambiaVista(n);
    }

    // Usata sia dai tasti sia dai pulsanti sullo schermo
    void CambiaVista(int n)
    {
        if (n == 0) VistaLibera();
        else if (n == 1) VistaDaTelefono(Scena.camere[0]);
        else if (n == 2) VistaDaTelefono(Scena.camere[1]);
        else VistaDallAlto();
        Debug.Log($"Vista cambiata: {vistaAttuale}");
    }

    // ---------------------------------------------------------------- oggetti della stanza

    void CreaPavimento()
    {
        // Il Plane di Unity è 10 x 10 unità con scala 1: scala = lato / 10. Un millimetro sotto lo
        // zero, così le croci (a quota 0) restano visibili sopra.
        GameObject pavimento = Forme.Crea(PrimitiveType.Plane, "Pavimento", transform, new Color(0.55f, 0.55f, 0.55f));
        pavimento.transform.localPosition = new Vector3(0f, -0.001f, 0f);
        pavimento.transform.localScale = Vector3.one * (latoPavimento / 10f);
    }

    void CreaAssi()
    {
        // Assi dell'origine (croce 1), lunghi 30 cm: X rosso, Y verde, Z blu
        Transform assi = new GameObject("Assi").transform;
        assi.SetParent(transform, false);
        Vector3[] direzioni = { Vector3.right, Vector3.up, Vector3.forward };
        Color[] colori = { Color.red, Color.green, Color.blue };
        string[] nomi = { "X", "Y", "Z" };
        for (int i = 0; i < 3; i++)
        {
            GameObject asse = Forme.Crea(PrimitiveType.Cylinder, nomi[i], assi, colori[i]);
            Forme.Segmento(asse.transform, Vector3.zero, direzioni[i] * 0.3f, 0.01f);
        }
    }

    void CreaCroce(Croce c)
    {
        // Due bracci piatti incrociati (3 cm x 2 mm), allineati agli assi X e Z. La croce 1 è rossa.
        Color colore = c.numero == 1 ? new Color(0.9f, 0.1f, 0.1f) : new Color(1f, 0.85f, 0.1f);
        Transform croce = new GameObject($"Croce {c.numero}").transform;
        croce.SetParent(transform, false);
        croce.localPosition = new Vector3(c.x, c.y, c.z);
        GameObject a = Forme.Crea(PrimitiveType.Cube, "braccio X", croce, colore);
        a.transform.localScale = new Vector3(braccioCroce, 0.002f, 0.03f);
        GameObject b = Forme.Crea(PrimitiveType.Cube, "braccio Z", croce, colore);
        b.transform.localScale = new Vector3(0.03f, 0.002f, braccioCroce);
    }

    void CreaTelefono(DatiCamera c)
    {
        // Il telefono: una scatola 7,5 x 16,5 x 0,8 cm centrata sull'obiettivo, orientata come la camera
        Transform telefono = new GameObject($"Telefono {c.nome}").transform;
        telefono.SetParent(transform, false);
        telefono.SetPositionAndRotation(c.posizione, Quaternion.LookRotation(c.avanti, c.su));
        GameObject corpo = Forme.Crea(PrimitiveType.Cube, "corpo", telefono, new Color(0.15f, 0.15f, 0.2f));
        corpo.transform.localScale = new Vector3(0.075f, 0.165f, 0.008f);

        // Il campo visivo: 4 spigoli dalla camera ai 4 angoli dell'immagine, a 1 m di distanza.
        // Un pixel (u, v) corrisponde alla direzione ((u - cx) / fx, (v - cy) / fy, 1) nel sistema
        // della camera (teoria 5.1); in Unity l'asse y dell'immagine (in giù) diventa -Y (in su).
        Vector2[] angoli = { new Vector2(0, 0), new Vector2(c.larghezza, 0), new Vector2(c.larghezza, c.altezza), new Vector2(0, c.altezza) };
        Vector3[] estremi = new Vector3[4];
        for (int i = 0; i < 4; i++)
        {
            Vector3 locale = new Vector3((angoli[i].x - c.cx) / c.fx, -(angoli[i].y - c.cy) / c.fy, 1f);
            estremi[i] = telefono.TransformPoint(locale);
            GameObject spigolo = Forme.Crea(PrimitiveType.Cylinder, $"spigolo {i}", telefono, Color.white);
            Forme.Segmento(spigolo.transform, c.posizione, estremi[i], 0.004f);
        }
        for (int i = 0; i < 4; i++)
        {
            GameObject bordo = Forme.Crea(PrimitiveType.Cylinder, $"bordo {i}", telefono, Color.white);
            Forme.Segmento(bordo.transform, estremi[i], estremi[(i + 1) % 4], 0.004f);
        }
    }

    // ---------------------------------------------------------------- viste

    void VistaLibera()
    {
        // Dietro e sopra il punto medio dei telefoni, guardando il centro dell'area all'altezza del busto
        Vector3 medio = (Scena.camere[0].posizione + Scena.camere[1].posizione) * 0.5f;
        Vector3 indietro = new Vector3(medio.x, 0f, medio.z).normalized;
        telecamera.usePhysicalProperties = false;
        telecamera.fieldOfView = 60f;
        telecamera.transform.position = medio + indietro * 1.0f + Vector3.up * 0.7f;
        telecamera.transform.LookAt(new Vector3(0f, 0.9f, 0f));
        vistaAttuale = "libera";
    }

    void VistaDallAlto()
    {
        telecamera.usePhysicalProperties = false;
        telecamera.fieldOfView = 60f;
        telecamera.transform.SetPositionAndRotation(new Vector3(0f, 7f, 1.5f), Quaternion.Euler(90f, 180f, 0f));
        vistaAttuale = "dall'alto";
    }

    void VistaDaTelefono(DatiCamera c)
    {
        // Camera "fisica": il campo visivo nasce dal rapporto tra sensore e focale. Usando i pixel come
        // unità per entrambi (sensore = immagine, focale = fy) il campo visivo è quello del telefono:
        // 2 * atan(altezza / 2 / fy). Il lens shift sposta il centro come cx, cy (punto principale).
        telecamera.usePhysicalProperties = true;
        telecamera.sensorSize = new Vector2(c.larghezza, c.altezza);
        telecamera.focalLength = c.fy;
        telecamera.lensShift = new Vector2(-(c.cx - c.larghezza / 2f) / c.larghezza, (c.cy - c.altezza / 2f) / c.altezza);
        telecamera.gateFit = Camera.GateFitMode.Vertical;
        telecamera.nearClipPlane = 0.05f;
        telecamera.transform.SetPositionAndRotation(c.posizione, Quaternion.LookRotation(c.avanti, c.su));
        vistaAttuale = c.nome;
    }

    void OnGUI()
    {
        if (Scena == null) return;
        GUI.Label(new Rect(10, 10, 600, 22),
                  $"Vista: {vistaAttuale}   (0 libera, 1 {Scena.camere[0].nome}, 2 {Scena.camere[1].nome}, 3 dall'alto)");
        GUI.Label(new Rect(10, 30, 600, 22),
                  $"Calibrazione stereo del {Scena.calibrazione_stereo}, pavimento del {Scena.pavimento}");
        // Pulsanti cliccabili con il mouse: funzionano anche se la tastiera non arriva al Game
        string[] etichette = { "0 libera", "1 " + Scena.camere[0].nome, "2 " + Scena.camere[1].nome, "3 dall'alto" };
        for (int n = 0; n <= 3; n++)
            if (GUI.Button(new Rect(10 + n * 110, 75, 105, 28), etichette[n])) CambiaVista(n);
    }

    static bool Premuto(int numero)
    {
#if ENABLE_INPUT_SYSTEM
        Keyboard tastiera = Keyboard.current;
        if (tastiera == null) return false;
        // Sia i numeri sopra le lettere sia quelli del tastierino numerico
        if (numero == 0) return tastiera.digit0Key.wasPressedThisFrame || tastiera.numpad0Key.wasPressedThisFrame;
        if (numero == 1) return tastiera.digit1Key.wasPressedThisFrame || tastiera.numpad1Key.wasPressedThisFrame;
        if (numero == 2) return tastiera.digit2Key.wasPressedThisFrame || tastiera.numpad2Key.wasPressedThisFrame;
        return tastiera.digit3Key.wasPressedThisFrame || tastiera.numpad3Key.wasPressedThisFrame;
#else
        return Input.GetKeyDown(KeyCode.Alpha0 + numero) || Input.GetKeyDown(KeyCode.Keypad0 + numero);
#endif
    }
}