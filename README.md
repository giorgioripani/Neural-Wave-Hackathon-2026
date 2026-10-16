# Neural Wave 2026: rilevamento del fotovoltaico dai profili di consumo (AIL)

## Team
| Nome e cognome | Ruolo |
| --- | --- |
| Giorgio Ripani | Integrazione, `inference.py`, consegna |
| Leonardo Lodolini | Task 2: data di messa in servizio |
| Andrea Gallo | Dati, valutazione, report |
| Flavio Davide Borriello | Task 1: rilevamento del fotovoltaico, README |
| Francesco Tassi | Task 3: capacità in kWp, grafici, video |

## Cosa fa
[2-3 frasi: dalla sola curva di consumo a 15 minuti di un cliente, il programma stima se ha un impianto fotovoltaico, la data di messa in servizio e la capacità in kWp.]

## Come eseguire
Richiede **Python 3.12** (testato su Windows, macOS e Linux). Tutte le dipendenze, con versioni fissate, sono in `requirements.txt`.

Linux / macOS:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py percorso/test_set.csv
```

Windows:
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python inference.py percorso\test_set.csv
```

Il programma scrive nella cartella corrente due file (UTF-8, virgola come separatore, punto decimale):

- `predictions.csv`, una riga per ogni cliente del test:
  `customer_id,pv_installed,pv_installation_date,pv_capacity_kwp`
  con `TRUE`/`FALSE`, date `AAAA-MM-GG`, capacità in kWp; data e capacità vuote se `pv_installed=FALSE`.
  I valori si riferiscono all'impianto iniziale.
- `pv_extensions_predictions.csv` (compito facoltativo), una riga per ogni estensione prevista:
  `customer_id,extension_date,pv_capacity_after_extension_kwp`
  con la capacità totale dopo l'estensione. Se non ci sono estensioni contiene solo l'intestazione.

Non servono rete né interventi manuali. I modelli addestrati sono in `artifacts/` e vengono caricati da `inference.py`.

## Dati
I dati appartengono ad AIL e **non sono inclusi** nel repository. Per riaddestrare i modelli, copiare i file forniti in `data/raw/` e prepararli con:
```
python src/loading/prepare_data.py data/raw/<file>.csv data/clean/<nome>.parquet
```
La soluzione usa solo i consumi e le informazioni di calendario (ora, giorno, mese): nessun dato meteo, di irraggiamento o di posizione.

## Struttura del repository
```
inference.py        punto di ingresso
requirements.txt    dipendenze con versioni fissate
src/loading/        caricamento e pulizia dei dati (filtro customer_id POD)
src/models/         rilevamento, data, capacità
src/eval/           punteggio ufficiale e validazione
src/viz/            grafici
artifacts/          modelli salvati
docs/               report, registro del lavoro (LOG.md)
tests/              test automatici
```

## Metodo e risultati
[Un paragrafo per compito e una tabella con F1, D, P e punteggio misurato con validazione a un cliente alla volta, su finestre di due anni come nel test. Dettagli nel report.]

## Limiti
[Cosa non funziona o funziona male, onestamente.]

## Uso di agenti e assistenti IA
[Quali strumenti, da chi e per cosa. Gli assistenti hanno visto solo dati aggregati per cliente, mai le serie grezze. Nessun LLM fa parte della pipeline di `inference.py`. Dettagli in `docs/LOG.md` e nel report.]

## Fonti e crediti
[Librerie (numpy, pandas, scipy, scikit-learn, pyarrow, duckdb, matplotlib, ruptures), articoli. Il dataset appartiene ad AIL ed è fornito per Neural Wave: non è incluso nel repository.]
