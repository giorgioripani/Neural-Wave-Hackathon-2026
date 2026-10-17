# STATUS.md — Neural Wave (AIL, rilevamento fotovoltaico)

Ognuno modifica solo la propria riga e aggiorna quando chiude un pezzo (non a ogni commit).
Gli agenti leggono questo file per sapere cosa fanno gli altri: tenerlo breve e vero.

Ultimo aggiornamento: 17/10 da Andrea (consegna finale)
Consegna: domenica 18 ottobre, ore 11:00 (obiettivo: pubblicato entro le 9:30). Poi presentazione da 5 minuti.

## Chi fa cosa
| Persona | Compito | Cartella | Branch | Stato |
| --- | --- | --- | --- | --- |
| Giorgio | Integrazione, `inference.py`, repo dell'organizzazione, Swisstransfer | `inference.py`, `demo/` | [branch] | [stato] |
| Andrea | Dati e punteggio ufficiale; report di 3 pagine | `src/loading/`, `src/eval/` | `andrea/load-test`, `feature/extensions` (PR) | Fatto: prepare_data, load_series, daily_features, score.py, validate.py, LOO pipeline intera (77,93 ± 2,42, in main), score_ext.py (PR #5), load_test + rete di sicurezza in inference.py (PR). Prossimo: grafici e tabelle per il report |
| Flavio | Task 1: c'è il fotovoltaico?; README e presentazione da 5 minuti | `src/models/detect.py` | `flavio/detect2` (in main) | In main: `dip_spread` + secondo passo `rec_drop` (installazioni recenti), F1 LOO 0,952 (0 FP, 2 FN); regola in `artifacts/detect_params.json` |
| Leonardo | Task 2: data (poi estensioni); `docs/LOG.md` | `src/models/date.py`, `src/models/extensions.py` | `lodoscatenato` (in main) | Task 2 v2 in main (v4 provata, non di default). Estensioni in main e collegate in `inference.py` (`predict_extensions_refined`, top 3): F1_ext atteso 0,29-0,50 sull'iniezione sintetica, 0 sull'unica estensione vera del dev |
| Francesco | Task 3: kWp; grafici e video da 2 minuti | `src/models/capacity.py`, `src/viz/` | `francesco/capacity` (in main); `fix/bug-fixes` | `geo_b` in main: P 0,437 (validate.py, seed 42), mediana 0,415. Revisione di main: bug di lettura del test corretti in `fix/bug-fixes` (PR da fare), proposte in `IMPROVEMENTS.md` |

## Traguardi
- [x] Ven sera: dati puliti in Parquet, punteggio ufficiale pronto, `inference.py` provvisorio
- [ ] Sab 13:00: primo punteggio dell'intera pipeline
- [ ] Sab 20:00: codice congelato; poi bozza del report (3 pagine) e copione del video
- [ ] Sab sera: prova di caricamento sul repo dell'organizzazione e `docs/LOG.md` aggiornato
- [ ] Dom 7:00: prova da zero in ambiente pulito (`pip install -r requirements.txt`, poi `inference.py`)
- [ ] Dom 9:30: repo, report e video pubblicati; link Swisstransfer inviato (scadenza 11:00)
- [ ] Dopo la consegna: presentazione da 5 minuti

## Fatto (già in main)
- 17/10 Andrea (d'accordo con Giorgio): merge PR #7 (README completo, tolto DatabaseReadme.md); verificato su clone pulito da GitHub: AMBIENTE OK, 205 test, previsioni identiche a f82db85 (fake_test 78,66 in-sample), LOO 78,49. Tag `v2-consegna` spostato sull'ultima main.
- 17/10 Giorgio: merge finale in main (f82db85, tag `v2-consegna`): ImprovementDetectAlgorythm, PR #6 (load_test), fix/bug-fixes, PR #5 (score_ext), Leonardo/date-agg; detect con crash -> TRUE; test 2022-2025 (fake_test serie intera, fake_test_2y); 205 test; previsioni sul test a 2 anni identiche a prima dei merge.
- 17/10 Leonardo: estensioni in main (c18a288) e collegate in `inference.py` (09e9ce2): al massimo 3 righe in `pv_extensions_predictions.csv`, riserva `ext` nel riepilogo; `predictions.csv` invariato (78,38), tempo su fake_test 14 s.
- 17/10 Leonardo: Task 2 v2 (stagionalità + gradino, regola "nessun gradino") in main; su 22 clienti, finestre di 2 anni dentro i dati: D 0,597, mediana errore 2,5 giorni, 7 clienti oltre 30 giorni.
- 17/10 03:13 Andrea: dati puliti (dev 40 clienti, senza etichette 1147 clienti), `load_series`, `daily_features`, `score.py` (formula ufficiale, unico file di punteggio), `validate.py` (leave-one-out, finestre di 2 anni). Baseline 54,58.
- 17/10 Giorgio: `inference.py` provvisorio con riserve; sul test finto scrive 40 righe nel formato giusto.
- 17/10 Flavio: `detect.py` (`dip_spread` + `rec_drop`, F1 LOO 0,952) in main (ebe5364); pipeline intera sul test finto: 20 clienti con fotovoltaico, punteggio 78,38 (in-sample).

## In corso
- Leonardo: Task 2, test sintetico realistico.
- Andrea: punteggio della pipeline intera appena detect e capacity sono in main.
- Flavio: feature per installazioni recenti e dip normalizzato per i grandi impianti commerciali (sab pomeriggio)

## Bloccato o serve da altri
- Nessun blocco aperto. Risolti: detect e capacity sono in main (pipeline intera misurata); gli organizzatori hanno risposto sul formato del test (vedi Decisioni).

## Decisioni
- 2026-10-17 Andrea: kWh negativi -> 0 (riga tenuta) in `load_series`; giorni = giorni UTC (ora legale); D e P solo sui veri positivi (piano, pag. 2).
- 2026-10-17: CSV di uscita con punto e virgola (confermato dagli organizzatori).
- 2026-10-17 (organizzatori): il test copre il periodo 2022-2025 come i dati di sviluppo (circa 4 anni per cliente), già filtrato sui POD; nessuna installazione prima del 2022. `data/fake_test.csv` = serie intera, `data/fake_test_2y.csv` = ultimi 2 anni (prova di robustezza); LOO principale sulla serie intera (`loo_pipeline --protocol serie_intera`, default).
- Sab 17/10 ore 10:00: scelta `dip_spread` (variabilità del calo di mezzogiorno con le nuvole) come feature per il Task 1; prossimi passi: installazioni recenti e grandi impianti

## Punteggio misurato (validazione a un cliente alla volta)
| Data e ora | Versione (commit) | F1 | D | P | Punteggio |
| --- | --- | --- | --- | --- | --- |
| 17/10 | 066bc37, baseline (sempre TRUE, data a metà finestra, kWp mediano) | 0,710 | 0,045 | 0,415 | 54,58 ± 0,41 |
| 17/10 | ebe5364, pipeline intera (detect + date + capacity) su fake_test, IN-SAMPLE (ottimistico, non LOO) | 0,952 | 0,372 | 0,448 | 78,38 |
| 17/10 | 45d6f80 (main 9858eb4), pipeline intera LOO con artefatti attuali (detect e capacity calcolati anche sul cliente lasciato fuori: OTTIMISTA), `python -m src.eval.loo_pipeline` | 0,922 | 0,608 | 0,452 | 79,19 ± 2,92 |
| 17/10 | 45d6f80, pipeline intera LOO, soglie detect ricalcolate senza il cliente lasciato fuori, capacity da artefatto | 0,909 | 0,608 | 0,452 | 78,10 ± 2,42 |
| 17/10 | 45d6f80, pipeline intera LOO, detect E capacity (geo_b) ricalcolati senza il cliente lasciato fuori, finestre di 2 anni (superato: ora vale la serie intera, riga f82db85) | 0,909 | 0,608 | 0,439 | 77,93 ± 2,42 |
| 17/10 | 09e9ce2, estensioni su fake_test, IN-SAMPLE: 3 previste, 1 vera (TP 0, FP 3, FN 1), F1_ext 0, bonus 0 (con 10 x F1_ext, formula non ufficiale) | - | - | - | 78,38 + 0 |
| 17/10 | f82db85 (`v2-consegna`), pipeline intera LOO tutto_loo sulla SERIE INTERA (protocollo serie_intera, 1 giro deterministico): NUMERO PER IL REPORT | 0,905 | 0,665 | 0,452 | 78,49 |
| 17/10 | f82db85, in-sample su fake_test serie intera / fake_test_2y (ottimistici) | 0,905 / 0,952 | 0,665 / 0,372 | 0,465 / 0,448 | 78,66 / 78,38 |

## Quota Claude (controllare con /status circa ogni ora)
| Persona | Quota usata | Crediti extra |
| --- | --- | --- |
| Giorgio | [%] | [si/no] |
| Andrea | [%] | [si/no] |
| Flavio | [%] | [si/no] |
| Leonardo | [%] | [si/no] |
| Francesco | [%] | [si/no] |
