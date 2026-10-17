# STATUS.md — Neural Wave (AIL, rilevamento fotovoltaico)

Ognuno modifica solo la propria riga e aggiorna quando chiude un pezzo (non a ogni commit).
Gli agenti leggono questo file per sapere cosa fanno gli altri: tenerlo breve e vero.

Ultimo aggiornamento: 17/10 da Andrea
Consegna: domenica 18 ottobre, ore 11:00 (obiettivo: pubblicato entro le 9:30). Poi presentazione da 5 minuti.

## Chi fa cosa
| Persona | Compito | Cartella | Branch | Stato |
| --- | --- | --- | --- | --- |
| Giorgio | Integrazione, `inference.py`, repo dell'organizzazione, Swisstransfer | `inference.py`, `demo/` | [branch] | [stato] |
| Andrea | Dati e punteggio ufficiale; report di 3 pagine | `src/loading/`, `src/eval/` | `andrea/eval` (in main) | Fatto: prepare_data, load_series, daily_features, score.py, validate.py. Prossimo: punteggio della pipeline intera, grafici e tabelle per il report |
| Flavio | Task 1: c'è il fotovoltaico?; README e presentazione da 5 minuti | `src/models/detect.py` | [branch] | [stato] |
| Leonardo | Task 2: data (poi estensioni); `docs/LOG.md` | `src/models/date.py` | `Leonardo/data1` | Task 2 v2 + fix fillna in main. Prossimo: test sintetico realistico |
| Francesco | Task 3: kWp; grafici e video da 2 minuti | `src/models/capacity.py`, `src/viz/` | `francesco/capacity` | branch pronto per la PR verso main; in uso `geo_b` (media geometrica B lineare x mediana): P 0,437 con data vera e con data stimata da date.py, mediana 0,415 (validate.py, seed 42; con seed 7 0,429). `geo_b_loglog`, taglia e altre combinazioni scartate |

## Traguardi
- [x] Ven sera: dati puliti in Parquet, punteggio ufficiale pronto, `inference.py` provvisorio
- [ ] Sab 13:00: primo punteggio dell'intera pipeline
- [ ] Sab 20:00: codice congelato; poi bozza del report (3 pagine) e copione del video
- [ ] Sab sera: prova di caricamento sul repo dell'organizzazione e `docs/LOG.md` aggiornato
- [ ] Dom 7:00: prova da zero in ambiente pulito (`pip install -r requirements.txt`, poi `inference.py`)
- [ ] Dom 9:30: repo, report e video pubblicati; link Swisstransfer inviato (scadenza 11:00)
- [ ] Dopo la consegna: presentazione da 5 minuti

## Fatto (già in main)
- 17/10 Leonardo: Task 2 v2 (stagionalità + gradino, regola "nessun gradino") in main; su 22 clienti, finestre di 2 anni dentro i dati: D 0,597, mediana errore 2,5 giorni, 7 clienti oltre 30 giorni.
- 17/10 03:13 Andrea: dati puliti (dev 40 clienti, senza etichette 1147 clienti), `load_series`, `daily_features`, `score.py` (formula ufficiale, unico file di punteggio), `validate.py` (leave-one-out, finestre di 2 anni). Baseline 54,58.
- 17/10 Giorgio: `inference.py` provvisorio con riserve; sul test finto scrive 40 righe nel formato giusto.

## In corso
- Leonardo: Task 2, test sintetico realistico.
- Andrea: punteggio della pipeline intera appena detect e capacity sono in main.

## Bloccato o serve da altri
- Andrea aspetta `detect.py` da Flavio e `capacity.py` da Francesco in main, per il punteggio dell'intera pipeline (oggi: 0, perche' senza detect tutti i clienti escono FALSE).
- Tutti: risposta degli organizzatori su impianti installati prima dei 2 anni di dati del test (decide quanto vale D).

## Decisioni
- 2026-10-17 Andrea: kWh negativi -> 0 (riga tenuta) in `load_series`; giorni = giorni UTC (ora legale); D e P solo sui veri positivi (piano, pag. 2).
- 2026-10-17: CSV di uscita con punto e virgola (confermato dagli organizzatori).

## Punteggio misurato (validazione a un cliente alla volta)
| Data e ora | Versione (commit) | F1 | D | P | Punteggio |
| --- | --- | --- | --- | --- | --- |
| 17/10 | 066bc37, baseline (sempre TRUE, data a metà finestra, kWp mediano) | 0,710 | 0,045 | 0,415 | 54,58 ± 0,41 |

## Quota Claude (controllare con /status circa ogni ora)
| Persona | Quota usata | Crediti extra |
| --- | --- | --- |
| Giorgio | [%] | [si/no] |
| Andrea | [%] | [si/no] |
| Flavio | [%] | [si/no] |
| Leonardo | [%] | [si/no] |
| Francesco | [%] | [si/no] |
