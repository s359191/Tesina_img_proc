using UnityEngine;

// Filtro "One Euro" (Casiry, Roussel, Vogel, 2012) per un punto 3D che si muove nel tempo.
//
// Un filtro passa-basso normale toglie il tremolio ma introduce ritardo: più filtra, più il punto
// "insegue" la persona. Il One Euro cambia quanto filtra in base alla velocità:
//   - punto quasi fermo  -> frequenza di taglio bassa (frequenzaMinima): filtra molto, niente tremolio;
//   - punto veloce       -> frequenza di taglio alta (frequenzaMinima + beta * velocità): filtra poco,
//                           niente ritardo, e nel movimento il tremolio non si nota.
// Regolazione: se trema da fermo, abbassa frequenzaMinima; se è in ritardo nei movimenti veloci, alza beta.
public class FiltroOneEuro
{
    public float frequenzaMinima;          // Hz
    public float beta;                     // quanto la frequenza di taglio cresce con la velocità (Hz per m/s)
    public float frequenzaDerivata = 1f;   // Hz: filtro sulla stima della velocità

    Vector3 valore, velocita;
    bool inizializzato;

    public FiltroOneEuro(float frequenzaMinima, float beta)
    {
        this.frequenzaMinima = frequenzaMinima;
        this.beta = beta;
    }

    // Da chiamare quando il punto sparisce: alla ricomparsa il filtro riparte dal nuovo valore
    public void Azzera() { inizializzato = false; }

    // Peso del valore nuovo per un passa-basso del primo ordine con quella frequenza di taglio
    static float Alfa(float frequenza, float dt)
    {
        float tau = 1f / (2f * Mathf.PI * frequenza);
        return 1f / (1f + tau / dt);
    }

    // x: misura nuova; dt: secondi trascorsi dalla misura precedente
    public Vector3 Filtra(Vector3 x, float dt)
    {
        if (!inizializzato || dt > 0.5f || dt < 0f)   // primo punto, pausa lunga o tempo che torna indietro
        {
            valore = x;
            velocita = Vector3.zero;
            inizializzato = true;
            return x;
        }
        if (dt == 0f) return valore;               // stesso istante di prima: niente di nuovo
        Vector3 velocitaGrezza = (x - valore) / dt;
        velocita = Vector3.Lerp(velocita, velocitaGrezza, Alfa(frequenzaDerivata, dt));
        float frequenza = frequenzaMinima + beta * velocita.magnitude;
        valore = Vector3.Lerp(valore, x, Alfa(frequenza, dt));
        return valore;
    }
}
