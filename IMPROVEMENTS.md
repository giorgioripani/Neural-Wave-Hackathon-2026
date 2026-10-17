# IMPROVEMENTS — revisione di `main` (09d3ab3), 17/10

Revisione fatta con un agente (Claude) per Francesco. Ogni numero viene da un'esecuzione reale.
Le proposte P1-P3 sono state provate su una copia del codice: le patch qui sotto funzionano così come sono.

## 0. Cosa è stato verificato su `main`

| Controllo | Esito |
| --- | --- |
| `pytest` | 162 test passano (167 su `fix/bug-fixes`) |
| `demo/check_env.py` | AMBIENTE OK (Python 3.12.10, versioni uguali a `requirements.txt`) |
| `inference.py data/fake_test.csv`, lanciato da un'altra cartella | 40 righe, 20 TRUE, 3 estensioni, 9 s, nessuna riserva usata; punteggio in-sample 78,38 |
| Artefatti riproducibili | `run_detect.py` rigenera `detect_params.json` identico; `train_capacity --method median,geo_b` rigenera gli stessi parametri di `capacity.json` |
| `python -m src.eval.loo_pipeline` | 79,19 / 78,10 / 77,93, uguali a STATUS |
| `requirements.txt` | ogni versione ha un wheel per Python 3.12 su Linux, macOS e Windows (PyPI) |
| Dati AIL o chiavi nei file tracciati | nessun id cliente né chiave trovati |

## 1. Bug corretti nel branch `fix/bug-fixes` (serve solo la PR)

| Bug su `main` | Effetto | Correzione |
| --- | --- | --- |
| Un solo `timestamp_epoch` vuoto o illeggibile | `inference.py` si ferma (`IntCastingNaNError`) e non scrive nessun file | righe senza epoch tolte con avviso; un cliente rimasto senza righe esce comunque, FALSE |
| Intestazione con spazi (`" customer_id "`) e id solo numerici | `dtype={"customer_id": str}` non si applica: crash `AttributeError`, zeri iniziali persi | tutte le colonne lette come testo, poi convertite (`_to_number`) |
| Righe ripetute (stesso cliente e stesso epoch) | tenute in inferenza, tolte da `load_series` in sviluppo | `drop_duplicates(keep="first")` con avviso, come `load_series` |
| `call_date` | `NaT` o `None` diventavano la data `"NaT"`; `"2024-5-1"` non veniva normalizzata | `NaT` va alla riserva; data sempre `strftime("%Y-%m-%d")` |
| `call_capacity` | accettava `inf` (scritto `inf` nel CSV) | richiede un valore finito |
| `loo_pipeline.py` | alias `src.eval.score_date` rimasto (il commento dice che `run_detect` lo usa, non è più vero) | tolto |

Sul test finto le uscite sono identiche byte per byte a `main`. Ci sono 5 test nuovi in `tests/test_inference.py`: falliscono su `main` e passano sul branch. 17 casi limite sintetici passano tutti: BOM, CRLF, tabulazione, virgola decimale, id vuoti, cliente tutto NaN o tutto zero, cliente di 1 riga, solo inverno, file vuoto.

## 2. Proposte, in ordine di priorità

### P1. detect isolato per cliente (`inference.py`, Giorgio, 10 minuti). Da fare.
**Problema.** `detect.predict` gira una volta sola su tutto il file. Se un solo cliente lo fa fallire, la riserva mette TRUE a tutti: F1 scende da circa 0,91 a 0,71 (baseline "sempre TRUE" in STATUS), circa 20 punti persi.
**Come.** Sostituire `call_detect` con questa versione. Le probabilità non cambiano, perché detect calcola le feature cliente per cliente. I test attuali passano senza modifiche.
```python
def call_detect(mod, df: pd.DataFrame) -> dict:
    """detect.predict(serie di tutti i clienti) -> {customer_id: probabilita`}.
    Se fallisce sul file intero riprova cliente per cliente: solo chi fallisce resta
    senza probabilita` (riserva TRUE in predict_all). Se falliscono tutti, eccezione."""
    try:
        raw = dict(mod.predict(df))
    except Exception as e:
        warn("detect (tutti i clienti insieme)", "*", e)
        raw = {}
        for cid, g in df.groupby("customer_id", sort=False):
            try:
                raw.update(dict(mod.predict(g)))
            except Exception as e2:
                warn("detect", cid, e2)
        if not raw:
            raise RuntimeError("detect fallito per tutti i clienti")
    out = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            out[str(k)] = float("nan")  # valore illeggibile: come un cliente mancante
    return out
```
**Test da aggiungere.** Un finto detect che solleva un'eccezione se nel DataFrame c'è un certo cliente. Atteso: solo quel cliente resta senza probabilità, gli altri tengono la loro. Provato: con 4 clienti sintetici, 3 hanno la probabilità vera e 1 va a TRUE.

### P2. duckdb non obbligatorio per i kWp (`src/loading/load.py`, Andrea, 5 minuti). Da fare.
**Problema.** La catena `capacity` → `capacity_common` → `load` importa `duckdb` all'avvio. Se duckdb non si installa, `capacity` non si importa e tutti i kWp diventano la mediana (P da circa 0,44 a 0,415). Verificato simulando l'assenza di duckdb: `ModuleNotFoundError`.
**Come.** In `src/loading/load.py`:
1. In cima, mettere `from __future__ import annotations` e togliere `import duckdb`.
2. Firma: `def _read_csv(con, path: Path) -> "duckdb.DuckDBPyRelation":`
3. Come prima riga del corpo di `_read_csv` e di `load_series`: `import duckdb`.
**Verifica.** Provato: con `sys.modules["duckdb"] = None`, `capacity.estimate` dà un kWp vero (135,72 su un cliente sintetico) e non la riserva. Tutti i test passano.

### P3. Validazione "ultimi 2 anni" per il report (`src/eval/validate.py` e `loo_pipeline.py`, Andrea, 15 minuti). Consigliata.
**Problema.** La validazione mette la data vera fra il 10% e il 90% della finestra. Nel dev, 9 clienti su 22 con fotovoltaico hanno l'installazione prima della finestra "ultimi 2 anni": per loro D vale 0 qualunque cosa si preveda. Non sappiamo come sono tagliate le finestre del test (domanda aperta agli organizzatori). Il report dovrebbe dare entrambi i numeri.

| Finestra (LOO, `tutto_loo`) | F1 | D | P | Punteggio |
| --- | --- | --- | --- | --- |
| data vera a caso nel 10-90% (attuale) | 0,909 | 0,608 | 0,439 | 77,93 ± 2,42 |
| ultimi 2 anni di ogni cliente | 0,952 | 0,372 | 0,436 | 78,20 |

**Come.** Patch provata.
- `validate.window`, subito dopo `if span <= days: return first, end`:
  ```python
      if pos is None:  # finestra "ultimi 2 anni", come demo/make_fake_test.py
          return end - pd.Timedelta(days=days), end
  ```
- `validate.validate`: aggiungere il parametro `last: bool = False` e sostituire la riga di `window(...)` con:
  ```python
              pos = rng.uniform(*POS_RANGE)  # estratto sempre: stessi numeri casuali nei due modi
              start, end = window(*bounds[cid], true, None if last else pos, days)
  ```
- `loo_pipeline.main`: aggiungere `ap.add_argument("--last", action="store_true")`, passare `last=a.last` a `validate(...)` e aggiungere `_last` al nome del CSV di uscita.
- Comando: `python -m src.eval.loo_pipeline --last`.

### P4. Consegna (Flavio, Giorgio, Francesco). Da fare prima del congelamento.
- `README.md` ha ancora 5 segnaposto `[...]`: Cosa fa, Metodo e risultati, Limiti, Uso di agenti, Fonti. Li scrivono le persone; i numeri sono in STATUS e qui in P3.
- `README.md` elenca `src/viz/`, che in `main` non esiste: creare la cartella con i grafici oppure togliere la riga.
- La riga di Francesco in `STATUS.md` diceva "branch pronto per la PR", ma `francesco/capacity` è già tutto in `main` (aggiornata in questo branch).
- `DatabaseReadme.md` (descrizione dei dati scritta da AIL) è stato tolto dal repo prima della pubblicazione; resta nella storia git dei commit precedenti.
- La prova in ambiente pulito resta da fare (domenica alle 7:00). Comandi: quelli del README, poi `python demo/check_env.py` e `python inference.py data/fake_test.csv`.

### P5. Estensioni (Leonardo). Solo da sapere, nessuna azione consigliata.
Sul test finto le 3 estensioni previste cadono nello stesso giorno della data iniziale prevista o prima, e sono tutte falsi allarmi. Vengono dal ramo `pv_gia_presente` di `refine_date0`, con kWp sempre uguale a 1,1 × kwp0 (limite già in LOG). È una scommessa coperta: il file principale punta su date0 come installazione, quello delle estensioni su date0 come estensione. I falsi allarmi non tolgono punti al punteggio principale. Spostare invece la data principale peggiora D (vedi sotto). Se si vuole un file coerente, si possono filtrare in `predict_extensions_all` le righe con `extension_date <= pv_installation_date`. Va misurato prima con `python src/models/inject_ext_dev.py --reestimate-date0 --refine`, perché toglierebbe anche i casi in cui date0 è davvero un'estensione.

## 3. Provate e scartate (righe aggiunte in `docs/LOG.md`)
- **Ristimare la data sul tratto prima di date0** quando `refine_date0` dice `pv_gia_presente`: D sul test finto scende da 0,372 a 0,338. Usare il primo giorno della serie dà lo stesso 0,338. Scartata.
- **Stessa regola sui buchi in tutti i moduli** (oggi solo date.py scarta i giorni con oltre il 25% di misure mancanti): nel dev c'è un solo giorno parziale per cliente, quello di bordo. Nessun effetto, non serve.
- **Velocità**: 9-12 s per 40 clienti su 2 anni. Non serve ottimizzare.
- **Minore**: `python src/models/run_detect.py --help` prende `--help` come percorso delle etichette. È solo cosmetico.
