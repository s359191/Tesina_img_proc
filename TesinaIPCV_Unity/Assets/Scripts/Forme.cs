using UnityEngine;

// Piccoli aiuti per creare dal codice gli oggetti della scena: sfere, cubi e "bastoncini"
// (cilindri sottili) tra due punti. Li usano ScenaReale e RicevitoreScheletro.
public static class Forme
{
    // Crea una forma semplice (sfera, cubo, cilindro...) con un nome, un genitore e un colore.
    public static GameObject Crea(PrimitiveType tipo, string nome, Transform genitore, Color colore)
    {
        GameObject oggetto = GameObject.CreatePrimitive(tipo);
        oggetto.name = nome;
        oggetto.transform.SetParent(genitore, false);
        // Il collider serve alla fisica e ai clic: qui non serve, lo togliamo.
        Object.Destroy(oggetto.GetComponent<Collider>());
        // .material crea una copia del materiale solo per questo oggetto, così il colore non
        // cambia anche agli altri.
        oggetto.GetComponent<Renderer>().material.color = colore;
        return oggetto;
    }

    // Mette un cilindro tra i punti a e b. Il cilindro di Unity è alto 2 unità lungo il suo asse Y
    // locale e largo 1: lo si orienta lungo a->b e lo si scala.
    public static void Segmento(Transform cilindro, Vector3 a, Vector3 b, float spessore)
    {
        Vector3 direzione = b - a;
        float lunghezza = direzione.magnitude;
        if (lunghezza < 1e-5f)
        {
            cilindro.localScale = Vector3.zero;
            return;
        }
        cilindro.position = (a + b) * 0.5f;
        cilindro.up = direzione / lunghezza;
        cilindro.localScale = new Vector3(spessore, lunghezza * 0.5f, spessore);
    }
}
