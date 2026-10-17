# STATUS.md — Neural Wave (AIL, rilevamento fotovoltaico)

Ognuno modifica solo la propria riga e aggiorna quando chiude un pezzo (non a ogni commit).
Gli agenti leggono questo file per sapere cosa fanno gli altri: tenerlo breve e vero.

Ultimo aggiornamento: [data e ora] da [nome]
Consegna: domenica 18 ottobre, ore 11:00 (obiettivo: pubblicato entro le 9:30). Poi presentazione da 5 minuti.

## Chi fa cosa
| Persona | Compito | Cartella | Branch | Stato |
| --- | --- | --- | --- | --- |
| Giorgio | Integrazione, `inference.py`, repo dell'organizzazione, Swisstransfer | `inference.py`, `demo/` | [branch] | [stato] |
| Andrea | Dati e punteggio ufficiale; report di 3 pagine | `src/loading/`, `src/eval/` | andrea/score | Fatto: prepare_data, load_series, daily_features, score.py, validate.py (baseline 54,6) |
| Flavio | Task 1: c'è il fotovoltaico?; README e presentazione da 5 minuti | `src/models/detect.py` | [branch] | [stato] |
| Leonardo | Task 2: data (poi estensioni); `docs/LOG.md` | `src/models/date.py` | `Leonardo/data1` | “Task 2 v2 + fix fillna in main; Prossimo: test sintetico realistico”
| Francesco | Task 3: kWp; grafici e video da 2 minuti | `src/models/capacity.py`, `src/viz/` | `francesco/capacity` | `estimate` collegata; P (validate.py, data vera): mediana 0,415, B 0,412, A 0,378: in uso la mediana. Prossimo: feature contro la saturazione |

## Traguardi
- [ ] Ven sera: dati puliti in Parquet, punteggio ufficiale pronto, `inference.py` provvisorio
- [ ] Sab 13:00: primo punteggio dell'intera pipeline
- [ ] Sab 20:00: codice congelato; poi bozza del report (3 pagine) e copione del video
- [ ] Sab sera: prova di caricamento sul repo dell'organizzazione e `docs/LOG.md` aggiornato
- [ ] Dom 7:00: prova da zero in ambiente pulito (`pip install -r requirements.txt`, poi `inference.py`)
- [ ] Dom 9:30: repo, report e video pubblicati; link Swisstransfer inviato (scadenza 11:00)
- [ ] Dopo la consegna: presentazione da 5 minuti

## Fatto (già in main)
- [data ora] [nome]: [cosa, in una riga]
- 17/10 02:20 Leonardo: Task 2 v2 (stagionalità + gradino, regola "nessun gradino") in main; su 22 clienti, finestre di 2 anni: D 0,58 (circa ±0,17), mediana errore 4 giorni.
## In corso
- [nome]: [cosa, quando pensa di finire]
- 17/10 02:20 Leonardo: Task 2 v2 (stagionalità + gradino, regola "nessun gradino") in main; su 22 clienti, finestre di 2 anni: D 0,58 (circa ±0,17), mediana errore 4 giorni.

## Bloccato o serve da altri
- [nome] aspetta [cosa] da [nome]

## Decisioni
- 2026-10-17 Andrea: kWh negativi -> 0 (riga tenuta) in `load_series`; giorni = giorni UTC (ora legale); D e P solo sui veri positivi (piano, pag. 2).
- [data ora]: [decisione e motivo in una riga]

## Punteggio misurato (validazione a un cliente alla volta)
| Data e ora | Versione (commit) | F1 | D | P | Punteggio |
| --- | --- | --- | --- | --- | --- |
| [..] | [..] | [..] | [..] | [..] | [..] |

## Quota Claude (controllare con /status circa ogni ora)
| Persona | Quota usata | Crediti extra |
| --- | --- | --- |
| Giorgio | [%] | [si/no] |
| Andrea | [%] | [si/no] |
| Flavio | [%] | [si/no] |
| Leonardo | [%] | [si/no] |
| Francesco | [%] | [si/no] |
