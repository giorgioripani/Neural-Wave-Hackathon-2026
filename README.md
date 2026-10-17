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
Dalla sola curva dei prelievi dalla rete a 15 minuti di ogni cliente, il programma stima:
1. se il cliente ha un impianto fotovoltaico;
2. la data di messa in servizio dell'impianto iniziale;
3. la capacità dell'impianto iniziale in kWp;
4. (facoltativo) le estensioni successive dell'impianto, con la capacità totale dopo ciascuna.

Il segnale è uno solo: dopo l'installazione, a mezzogiorno il cliente preleva meno dalla rete, molto di più nei giorni sereni d'estate che d'inverno. Tutti i modelli sono semplici (regole a soglia, regressioni lineari) e usano solo i consumi e il calendario, con fasce orarie fisse in UTC.

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

Facoltativo, prima di `inference.py`: `python demo/check_env.py` controlla versione di Python, pacchetti, artefatti e moduli, e termina con `AMBIENTE OK` o con l'elenco dei problemi.

Il programma scrive nella cartella corrente due file (UTF-8, punto e virgola come separatore, punto decimale):

- `predictions.csv`, una riga per ogni cliente del test:
  `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`
  con `TRUE`/`FALSE`, date `AAAA-MM-GG`, capacità in kWp; data e capacità vuote se `pv_installed=FALSE`.
  I valori si riferiscono all'impianto iniziale.
- `pv_extensions_predictions.csv` (compito facoltativo), una riga per ogni estensione prevista:
  `customer_id;extension_date;pv_capacity_after_extension_kwp`
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
Per ogni cliente si calcolano cifre giornaliere (giorni UTC, così l'ora legale non crea problemi): consumo medio per quarto d'ora nelle fasce notte (0-4 UTC), mattino, mezzogiorno (10-13 UTC) e pomeriggio.

**Task 1, c'è il fotovoltaico?** (`src/models/detect.py`). Regola a soglia su due cifre:
- `dip_spread`: in estate, ogni giorno si misura il "buco" di mezzogiorno rispetto a mattino e pomeriggio, e si prende la differenza fra i giorni con il buco più forte e quelli con il buco più debole (90° e 10° percentile). Con il fotovoltaico il buco va e viene con le nuvole, senza fotovoltaico resta quasi costante.
- `rec_drop`, solo per chi la prima regola dice "no": buco degli ultimi 90 giorni rispetto agli stessi giorni dell'anno prima. Serve per gli impianti installati da poco.

Le soglie massimizzano l'F1 su finestre di 2 anni dei clienti di sviluppo (`src/models/run_detect.py`) e sono salvate in `artifacts/detect_params.json`. Le due cifre usano solo gli ultimi 2 anni di ogni cliente, anche se il test ne ha circa 4: così la previsione vede la stessa lunghezza delle finestre di addestramento e meno cambi di abitudini. Un impianto installato prima resta visibile, perché c'è ancora alla fine dei dati. La data e i kWp usano invece tutta la serie.

**Task 2, data di messa in servizio** (`src/models/date.py`). Il logaritmo del rapporto mezzogiorno/notte di ogni giorno è modellato con una costante, una stagionalità annuale (seno e coseno) e un gradino. Si prova ogni data candidata e vince quella con l'errore quadratico minimo. Se nessun gradino migliora l'errore di almeno il 5%, l'impianto c'era già all'inizio dei dati e si risponde con il primo giorno. Nessun parametro è addestrato sui dati.

**Task 3, capacità in kWp** (`src/models/capacity.py`). Dopo la data stimata si misura di quanti kW cala il prelievo a mezzogiorno rispetto al livello atteso senza fotovoltaico (stimato dalla notte, che non vede il sole). Una regressione lineare trasforma il calo in kWp, poi la stima viene avvicinata alla mediana dei kWp di sviluppo con una media geometrica. Parametri in `artifacts/capacity.json` (`src/models/train_capacity.py`).

**Estensioni (facoltativo)** (`src/models/extensions.py`). Stesso modello del Task 2, cercando un secondo gradino verso il basso almeno 60 giorni dopo l'installazione e almeno 60 giorni prima della fine dei dati. Si scrivono al massimo 3 estensioni in tutto, le più nette.

**Robustezza** (`inference.py`, `src/loading/load.py`). Il CSV del test è letto con un lettore tollerante: righe con data non valida tolte, consumi non numerici lasciati vuoti, nessun cliente perso. Ogni modulo ha una riserva: se il rilevamento fallisce il cliente è segnato con fotovoltaico, e se data o kWp falliscono si usano il primo giorno dei dati e la mediana dei kWp. Se fallisce tutta la pipeline, si scrivono comunque i file con tutti i clienti.

**Risultati.** Validazione a un cliente alla volta sui 40 clienti etichettati (`python -m src.eval.loo_pipeline`): ogni cliente è previsto con soglie di detect e coefficienti dei kWp ricalcolati senza di lui, sulla sua serie intera come nel test (2022-2025). Punteggio = F1 × (70 + 15 × D + 15 × P).

| Versione | F1 | D (data) | P (kWp) | Punteggio |
| --- | --- | --- | --- | --- |
| Riferimento: tutti con fotovoltaico, data a metà dei dati, kWp mediano | 0,710 | 0,045 | 0,415 | 54,58 |
| Versione precedente (`v2-consegna`): rilevamento su tutta la serie | 0,905 | 0,665 | 0,452 | 78,49 |
| Pipeline finale, serie intera | 0,952 | 0,632 | 0,429 | **81,82** |
| Pipeline finale, finestre di 2 anni (5 giri, prova di robustezza) | 0,909 | 0,608 | 0,439 | 77,93 ± 2,42 |

Sulla serie intera: 20 impianti trovati su 22, nessun falso allarme su 18 clienti senza fotovoltaico. Rispetto alla versione precedente cambiano 2 clienti (un impianto trovato in più, un falso allarme in meno); D e P calano un poco perché l'impianto in più ha data e kWp molto sbagliati. Date: errore mediano di 1 giorno, 13 su 20 entro 7 giorni. kWp: errore mediano del 22%. Estensioni: nei dati di sviluppo ce n'è una sola etichettata, che il metodo non trova (bonus 0 sul dev).

## Limiti
- **Pochi esempi.** 40 clienti etichettati, 22 con fotovoltaico: un cliente in più o in meno sposta il punteggio di circa 2 punti. Le soglie e i coefficienti sono ricalcolati in validazione, ma la scelta delle cifre e dei metodi è stata fatta guardando tutti i 40 clienti, quindi il punteggio è un po' ottimista.
- **Impianti non visti.** 2 impianti su 22 hanno un segnale troppo debole: sono grandi utenze che di notte consumano da 1,6 a 5,4 volte i kWp dell'impianto.
- **Soglia del rilevamento.** Sui 1147 clienti senza etichetta la soglia di `dip_spread` cade sulla coda del gruppo senza fotovoltaico, non in una valle fra i due gruppi: sul test qualche falso allarme è possibile anche se sul dev non ce ne sono. Un cliente con pochi dati negli ultimi 2 anni risulta senza fotovoltaico.
- **kWp.** Vediamo solo i prelievi, non la produzione: l'energia autoconsumata non si vede. Se l'impianto copre tutto il consumo, il prelievo si ferma a zero e i grandi impianti sono sottostimati. 8 clienti su 20 hanno un errore oltre il 50%, e la sola mediana arriva già a P 0,415.
- **Date.** 5 date su 20 sbagliano di oltre 30 giorni: impianti presenti dall'inizio dei dati o con un avvio graduale.
- **Estensioni.** Senza meteo, un mese più soleggiato di quello dell'anno prima sposta il segnale quanto un'estensione. Un'estensione recente, con pochi mesi di dati dopo, è quasi invisibile. Con un solo esempio etichettato, il metodo non è validato su casi veri.

## Uso di agenti e assistenti IA
Più membri del team hanno usato Claude Code (Anthropic) come assistente di programmazione. Lo hanno usato per scrivere e rivedere codice e test, per gli script di validazione e per la ricerca dei casi limite su dati sintetici. Regola del team (in `CLAUDE.md`): gli assistenti vedono solo dati aggregati per cliente (cifre, metriche, previsioni contro etichette), mai le righe grezze dei consumi. Nessun LLM fa parte della pipeline di `inference.py`. Ogni uso importante di un agente e ogni idea scartata, con i numeri, è registrato in `docs/LOG.md`.

## Fonti e crediti
- Librerie: numpy, pandas, scipy, scikit-learn, pyarrow, duckdb, matplotlib, ruptures, pytest (versioni in `requirements.txt`).
- Dati: il dataset appartiene ad AIL SA ed è stato fornito per la sfida Neural Wave 2026. Non è incluso nel repository.
