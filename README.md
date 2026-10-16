# Neural Wave 2026: rilevamento del fotovoltaico dai profili di consumo (AIL)

> Modello da completare. Sostituire ogni [parentesi quadra] e cancellare questa riga.

## Team
| Nome e cognome | Ruolo |
| --- | --- |
| Giorgio [cognome] | Integrazione, `inference.py`, consegna |
| Leonardo [cognome] | Task 2: data di messa in servizio |
| Andrea [cognome] | Dati, valutazione, report |
| Flavio [cognome] | Task 1: rilevamento del fotovoltaico, README |
| Francesco [cognome] | Task 3: capacità in kWp, grafici, video |

## Cosa fa
[2-3 frasi: dalla sola curva di consumo a 15 minuti di un cliente, il programma stima se ha un impianto fotovoltaico, la data di messa in servizio e la capacità in kWp.]

## Come eseguire
Testato con Python [versione].

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py percorso/test_set.csv
```

Scrive nella cartella corrente `predictions.csv` e `pv_extensions_predictions.csv`
(UTF-8, punto e virgola come separatore, punto decimale). Non servono rete né interventi manuali.
I modelli addestrati sono in `artifacts/` e vengono caricati da `inference.py`.

## Struttura del repository
```
inference.py        punto di ingresso
src/loading/        caricamento e pulizia dei dati (filtro customer_id POD)
src/models/         rilevamento, data, capacità
src/eval/           punteggio ufficiale e validazione
src/viz/            grafici
artifacts/          modelli salvati
docs/               report, registro del lavoro
```

## Metodo e risultati
[Un paragrafo per compito e una tabella con F1, D, P e punteggio misurato con validazione a un cliente alla volta. Dettagli nel report.]

## Limiti
[Cosa non funziona o funziona male, onestamente.]

## Uso di agenti e assistenti IA
[Quali strumenti, da chi e per cosa. Dettagli in `docs/LOG.md` e nel report.]

## Fonti e crediti
[Librerie, articoli, dataset. Il dataset appartiene ad AIL ed è fornito per Neural Wave: non è incluso nel repository.]
