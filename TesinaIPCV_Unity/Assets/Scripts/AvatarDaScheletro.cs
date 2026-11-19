using System.Collections.Generic;
using UnityEngine;

// Muove un avatar umanoide (importato con Rig -> Animation Type = Humanoid) con i punti di MediaPipe
// ricevuti da RicevitoreScheletro.
//
// L'avatar ha ossa di lunghezza fissa che si muovono solo ruotando: dai punti si ricavano quindi
// le ROTAZIONI delle ossa ("retargeting"), confrontando ogni direzione misurata con la stessa
// direzione nell'avatar a riposo (posa a T, all'avvio del Play):
//   1. bacino e torace: un sistema di assi da anche e spalle (destra = da anca sx ad anca dx,
//      su = dal centro delle anche al centro delle spalle); il bacino va anche nella posizione misurata;
//   2. testa: assi da orecchie e spalle;
//   3. arti, dal tronco verso le estremità: la rotazione più piccola che porta la direzione a riposo
//      dell'osso (già girata insieme al genitore) sulla direzione misurata (es. spalla -> gomito);
//   4. infine l'avatar si alza o abbassa perché il piede più basso tocchi il pavimento.
// Lo script va sull'oggetto dell'avatar (quello con il componente Animator), in posa a T, rivolto
// verso +Z (verso le camere), con Position e Rotation a zero.
[RequireComponent(typeof(Animator))]
public class AvatarDaScheletro : MonoBehaviour
{
    [SerializeField] RicevitoreScheletro ricevitore;
    [Tooltip("Statura della persona ripresa, in metri: l'avatar viene scalato a questa altezza")]
    [SerializeField] float statura = 1.64f;
    [Tooltip("Alza o abbassa l'avatar perché il piede più basso tocchi il pavimento")]
    [SerializeField] bool appoggiaAlPavimento = true;

    // Un osso di un arto: la sua direzione va dall'articolazione MediaPipe 'da' ad 'a'
    // (o al punto medio tra 'a' e 'a2'); a riposo, dall'osso al suo 'figlio'.
    class Arto
    {
        public HumanBodyBones osso, figlio;
        public int da, a, a2;
        public Transform t;
        public Vector3 direzioneRiposo;
        public Arto(HumanBodyBones osso, HumanBodyBones figlio, int da, int a, int a2 = -1)
        { this.osso = osso; this.figlio = figlio; this.da = da; this.a = a; this.a2 = a2; }
    }

    // In ordine: prima i genitori, poi i figli
    readonly Arto[] arti =
    {
        new Arto(HumanBodyBones.LeftUpperArm,  HumanBodyBones.LeftLowerArm,  11, 13),
        new Arto(HumanBodyBones.LeftLowerArm,  HumanBodyBones.LeftHand,      13, 15),
        new Arto(HumanBodyBones.LeftHand,      HumanBodyBones.LeftMiddleProximal,  15, 17, 19),
        new Arto(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, 12, 14),
        new Arto(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,     14, 16),
        new Arto(HumanBodyBones.RightHand,     HumanBodyBones.RightMiddleProximal, 16, 18, 20),
        new Arto(HumanBodyBones.LeftUpperLeg,  HumanBodyBones.LeftLowerLeg,  23, 25),
        new Arto(HumanBodyBones.LeftLowerLeg,  HumanBodyBones.LeftFoot,      25, 27),
        new Arto(HumanBodyBones.LeftFoot,      HumanBodyBones.LeftToes,      27, 31),
        new Arto(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, 24, 26),
        new Arto(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot,     26, 28),
        new Arto(HumanBodyBones.RightFoot,     HumanBodyBones.RightToes,     28, 32),
    };

    Animator animator;
    Transform bacino, testa, collo, piedeSx, piedeDx;
    readonly List<Transform> colonna = new List<Transform>();     // ossa della colonna presenti nel modello
    readonly List<float> pesiColonna = new List<float>();          // quanto ciascuna segue le spalle
    readonly Dictionary<Transform, Quaternion> rotazioneRiposo = new Dictionary<Transform, Quaternion>();
    Quaternion assiBacinoRiposo, assiSpalleRiposo, assiTestaRiposo;
    Vector3 scostamentoBacino;            // dal centro delle anche all'osso del bacino, a riposo
    float altezzaPiedeSx, altezzaPiedeDx; // altezza delle caviglie da terra, a riposo
    bool pronto;

    void Start()
    {
        animator = GetComponent<Animator>();
        if (animator.avatar == null || !animator.isHuman)
        {
            Debug.LogError("L'avatar non è Humanoid: seleziona il file FBX, scheda Rig, Animation Type = Humanoid, Apply.");
            return;
        }
        if (ricevitore == null)
        {
            Debug.LogError("Trascina l'oggetto Scheletro nel campo Ricevitore di AvatarDaScheletro.");
            return;
        }
        ScalaAllaStatura();
        MemorizzaRiposo();
        pronto = true;
    }

    Transform Osso(HumanBodyBones b) => animator.GetBoneTransform(b);

    // L'avatar prende la statura della persona: in uno spazio 1:1 anche lui deve avere le sue misure
    void ScalaAllaStatura()
    {
        Bounds limiti = new Bounds();
        bool primo = true;
        foreach (Renderer r in GetComponentsInChildren<Renderer>())
        {
            if (primo) { limiti = r.bounds; primo = false; }
            else limiti.Encapsulate(r.bounds);
        }
        float altezza = limiti.size.y;
        if (altezza < 0.1f) return;
        transform.localScale *= statura / altezza;
        Debug.Log($"Avatar alto {altezza:F2} m nel modello, scalato a {statura:F2} m");
    }

    static Quaternion Assi(Vector3 destra, Vector3 su)
    {
        // Sistema di assi di Unity da due direzioni: avanti = destra x su (Unity è sinistrorso)
        return Quaternion.LookRotation(Vector3.Cross(destra, su), su);
    }

    void MemorizzaRiposo()
    {
        foreach (Transform t in GetComponentsInChildren<Transform>()) rotazioneRiposo[t] = t.rotation;

        bacino = Osso(HumanBodyBones.Hips);
        collo = Osso(HumanBodyBones.Neck);
        testa = Osso(HumanBodyBones.Head);
        piedeSx = Osso(HumanBodyBones.LeftFoot);
        piedeDx = Osso(HumanBodyBones.RightFoot);
        // Le ossa della colonna sono 1-3 a seconda del modello: il peso dice quanto seguono le spalle
        HumanBodyBones[] ossaColonna = { HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.UpperChest };
        float[] pesi = { 0.33f, 0.66f, 1f };
        for (int i = 0; i < 3; i++)
        {
            Transform t = Osso(ossaColonna[i]);
            if (t != null) { colonna.Add(t); pesiColonna.Add(pesi[i]); }
        }

        Vector3 ancaSx = Osso(HumanBodyBones.LeftUpperLeg).position, ancaDx = Osso(HumanBodyBones.RightUpperLeg).position;
        Vector3 spallaSx = Osso(HumanBodyBones.LeftUpperArm).position, spallaDx = Osso(HumanBodyBones.RightUpperArm).position;
        Vector3 centroAnche = (ancaSx + ancaDx) * 0.5f, centroSpalle = (spallaSx + spallaDx) * 0.5f;
        assiBacinoRiposo = Assi(ancaDx - ancaSx, centroSpalle - centroAnche);
        assiSpalleRiposo = Assi(spallaDx - spallaSx, centroSpalle - centroAnche);
        assiTestaRiposo = Assi(spallaDx - spallaSx, testa.position - centroSpalle);
        scostamentoBacino = Quaternion.Inverse(assiBacinoRiposo) * (bacino.position - centroAnche);

        foreach (Arto arto in arti)
        {
            arto.t = Osso(arto.osso);
            Transform figlio = Osso(arto.figlio);
            if (arto.t != null && figlio != null)
                arto.direzioneRiposo = (figlio.position - arto.t.position).normalized;
            else arto.t = null;            // osso mancante in questo modello (es. dita): si salta
        }
        altezzaPiedeSx = piedeSx.position.y - transform.position.y;
        altezzaPiedeDx = piedeDx.position.y - transform.position.y;
    }

    // LateUpdate: dopo tutti gli Update (quindi dopo il ricevitore) e dopo l'Animator
    void LateUpdate()
    {
        if (!pronto || !ricevitore.HaDati) return;       // senza dati l'avatar resta nell'ultima posa
        if (!(V(11) && V(12) && V(23) && V(24))) return; // senza spalle e anche non si sa dov'è il corpo

        // 1. bacino e colonna
        Vector3 centroAnche = (P(23) + P(24)) * 0.5f, centroSpalle = (P(11) + P(12)) * 0.5f;
        Vector3 su = centroSpalle - centroAnche;
        Quaternion giroBacino = Assi(P(24) - P(23), su) * Quaternion.Inverse(assiBacinoRiposo);
        Quaternion giroSpalle = Assi(P(12) - P(11), su) * Quaternion.Inverse(assiSpalleRiposo);
        bacino.rotation = giroBacino * rotazioneRiposo[bacino];
        bacino.position = centroAnche + Assi(P(24) - P(23), su) * scostamentoBacino;
        for (int k = 0; k < colonna.Count; k++)
            colonna[k].rotation = Quaternion.Slerp(giroBacino, giroSpalle, pesiColonna[k]) * rotazioneRiposo[colonna[k]];

        // 2. collo e testa (orecchie 7 e 8)
        if (V(7) && V(8))
        {
            Vector3 centroOrecchie = (P(7) + P(8)) * 0.5f;
            Quaternion giroTesta = Assi(P(8) - P(7), centroOrecchie - centroSpalle) * Quaternion.Inverse(assiTestaRiposo);
            if (collo != null) collo.rotation = Quaternion.Slerp(giroSpalle, giroTesta, 0.5f) * rotazioneRiposo[collo];
            testa.rotation = giroTesta * rotazioneRiposo[testa];
        }

        // 3. arti
        foreach (Arto arto in arti)
        {
            if (arto.t == null || !V(arto.da) || !V(arto.a) || (arto.a2 >= 0 && !V(arto.a2))) continue;
            Vector3 fine = arto.a2 >= 0 ? (P(arto.a) + P(arto.a2)) * 0.5f : P(arto.a);
            Vector3 misurata = fine - P(arto.da);
            if (misurata.sqrMagnitude < 1e-6f) continue;
            // quanto è girato il genitore rispetto al riposo: la direzione a riposo gira con lui
            Transform genitore = arto.t.parent;
            Quaternion giroGenitore = genitore.rotation * Quaternion.Inverse(rotazioneRiposo[genitore]);
            Vector3 riposoGirata = giroGenitore * arto.direzioneRiposo;
            arto.t.rotation = Quaternion.FromToRotation(riposoGirata, misurata) * giroGenitore * rotazioneRiposo[arto.t];
        }

        // 4. piedi a terra: il piede più basso alla sua altezza di riposo
        if (appoggiaAlPavimento)
        {
            float eccesso = Mathf.Min(piedeSx.position.y - altezzaPiedeSx, piedeDx.position.y - altezzaPiedeDx);
            bacino.position -= Vector3.up * eccesso;
        }
    }

    bool V(int i) => ricevitore.Valido(i);
    Vector3 P(int i) => ricevitore.Punto(i);
}
