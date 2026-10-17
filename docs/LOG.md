# Registro del lavoro

Una riga per ogni idea scartata (cosa, perché, numeri) e per ogni uso importante di un agente.
- 2026-10-17 Giorgio + agente: creati inference.py (try/except con riserve per detect/date/capacity), demo/make_fake_test.py, tests/test_inference.py. Su fake_test (40 clienti, ultimi 2 anni): 40 righe, 0 PV perche' detect.py non esiste ancora.
v1 (change point sul rapporto, finestre UTC fisse): D 0,547; mediana 5,5 gg; 9 clienti su 22 oltre 30 gg.
v2 (stagionalità + gradino costante, regola “nessun gradino”): D 0,582; mediana 4,0 gg; 7 clienti oltre 30 gg. Scelta.
v3 (gradino stagionale con termini post·sin e post·cos): D 0,568; mediana 4,5 gg; 6 clienti oltre 30 gg. Scartata: sistema POD22 e POD23 ma crea gradini falsi quando ci sono meno di circa sei mesi di dati prima dell’installazione (POD34, POD35).

La scelta della v2 tra v1, v2 e v3 è stata fatta sugli stessi 22 clienti: possibile leggero ottimismo. La soglia del 5% è stata fissata prima dei risultati.

Tolto fillna(0.0): NaN lasciati, media sugli intervalli presenti, EPS diviso per il numero di intervalli (senza buchi il gradino resta identico), giorno scartato se in una finestra manca più del 25% degli intervalli (soglia fissata a priori). D sui 22 clienti: 0,581 (prima 0,582); mediana 4,0 gg; 7 clienti oltre 30 gg. La piccola differenza viene dai giorni scartati per timestamp mancanti.
- 2026-10-17 Andrea (agente): `prepare_data.py` con tipi dichiarati per tutte le colonne (errore invece di testo se il decimale è sbagliato), conteggio duplicati; nuovo `src/eval/score.py` con la formula ufficiale (D e P solo sui veri positivi) e test.
- 2026-10-17 Andrea (agente): `load_series`/`daily_features` in `src/loading/load.py` e `src/eval/validate.py` (leave-one-out, finestre di 730 giorni, 5 giri, seed 42). Baseline (sempre TRUE, data a metà finestra, kWp mediano): 54,58 ± 0,41 (F1 0,710, D 0,045, P 0,415).
- 2026-10-17 Andrea: scartata la posizione FISSA della data nella finestra (25/50/75%): premia chi indovina la metà della finestra; usata posizione casuale tra 10% e 90%.
- 2026-10-17 Andrea: scartato l'ORDER BY globale in `prepare_data.py` sul file da 7,9 GB (senza tetto di RAM ha esaurito la memoria e lasciato un Parquet vuoto). Ora: tetto 4 GB, scrittura su .tmp rinominato solo a fine corretta, ordinamento in `load_series`.
- 2026-10-17 Andrea (agente): pulito il file senza etichette: 153.683.799 righe lette, 39.622 scartate (id nullo), 704 scartate ('Evoluzione consumi gas', non POD), 153.643.473 tenute (1147 clienti), 9512 kWh negativi, 0 duplicati, 0 clienti in comune con i 40 di sviluppo. 172 s, 4 GB di RAM.
- 2026-10-17 Francesco (agente Claude Code): Task 3 in `capacity.py` (punto d'ingresso, metodo scelto da `artifacts/capacity.json`), `capacity_a.py` (Ridge 5 feature su log kWp), `capacity_b.py` (calo di mezzogiorno), `capacity_common.py` (su `daily_features`), `train_capacity.py` (validazione con `validate.py`, data vera).
- 2026-10-17 Francesco: P con validate.py (5 giri, 2 anni, data vera, 22 clienti PV): mediana 0,415 ± 0,000; B lineare `drop_kw` 0,412 ± 0,042; A Ridge 0,378 ± 0,042; B `k x peak_kw` 0,212 ± 0,049. Nessuno batte la mediana: in `capacity.json` va la mediana (169,5 kWp). Scartato `k x peak_kw`: saturazione, d'estate meta` dei quarti d'ora di mezzogiorno sono a prelievo zero (mediana 0,49; 16/22 clienti > 0,1), il calo misura il carico e non l'impianto.
- 2026-10-17 Francesco: nota su validate.py: la data vera cade sempre tra il 10% e il 90% della finestra, quindi c'e` sempre un "prima" di almeno 73 giorni; il caso "data all'inizio della serie" non viene misurato.
- 2026-10-17 Francesco (agente Claude Code): nuovi metodi kWp `capacity_size.py` (taglia) e `capacity_blend.py` (base + mediana di training), stessa validazione (validate.py, 5 giri, seed 42, data vera). Regola fissata prima: si tiene solo con P >= mediana + 0,02 (0,435).
- 2026-10-17 Francesco: scartato `size_night` (log kWp = a + b log notte nei giorni di riferimento): P 0,286 ± 0,006, err < 50% 53%. La taglia del carico notturno non segue i kWp.
- 2026-10-17 Francesco: scartato `size_all` (idem con la media giornaliera): P 0,259 ± 0,010, err < 50% 56%.
- 2026-10-17 Francesco: TENUTO `geo_b` (media geometrica tra B lineare e mediana di training): P 0,437 ± 0,027, err < 50% 61%, err mediano 0,22; batte la mediana di 0,022 (soglia 0,02). Nessuna perdita: la mediana viene sempre da <= 21 clienti di training. Limiti: scelto tra 7 candidati sugli stessi 22 clienti; con seed 7 (solo verifica) P 0,429 ± 0,013 contro 0,415, sotto la soglia.
- 2026-10-17 Francesco: scartato `clip_b` (B limitato tra 0,5x e 2x la mediana): P 0,370 ± 0,033, err < 50% 65%; peggio di B senza limiti (0,412).
- 2026-10-17 Francesco: scartato `geo_size_night` (taglia notte in media geometrica con la mediana): P 0,357 ± 0,002, err < 50% 55%.
- 2026-10-17 Francesco: scartato `geo_size_all` (taglia giornaliera in media geometrica con la mediana): P 0,368 ± 0,005, err < 50% 55%.
- 2026-10-17 Francesco: robustezza alla data: con la data STIMATA da date.py (v2, errore mediano 5 gg; stesse finestre, seed 42; training sempre con data vera) `geo_b` P 0,437 ± 0,025 (data vera 0,437 ± 0,027), mediana 0,415 ± 0,000. `geo_b` resta in `capacity.json`.
- 2026-10-17 Francesco: scartato `geo_b_loglog` (B log-log su drop_kw in media geometrica con la mediana): P 0,397 ± 0,026 con data stimata (0,394 ± 0,031 con data vera), sotto `geo_b`; serviva +0,02 su `geo_b`.
- 2026-10-17 Andrea (agente): un solo file di punteggio: tolto `src/eval/score_date.py` (era solo un rimando), `run_date_dev.py` e `test_date.py` importano da `src/eval/score.py`. Tolto `prova.txt`. Verifica di main dopo i merge: nessun conflitto, 39 test passano, `inference.py` sul test finto (40 clienti, 2 anni) scrive 40 righe nel formato giusto; punteggio 0 finché mancano detect e capacity.
- 2026-10-17 Andrea (agente, con il consenso di Leonardo): `run_date_dev.py` ora misura di default il metodo della pipeline (v2), legge con `load_series` e tiene le finestre di 730 giorni DENTRO i dati (stessa regola di `validate.py`; prima una finestra poteva uscire dai dati e restare di pochi mesi). Nuovi numeri sui 22 clienti: v1 D 0,523 (mediana 6,5 gg, 9 oltre 30 gg); v2 D 0,597 (mediana 2,5 gg, 7 oltre 30 gg); v3 D 0,570 (mediana 4,5 gg, 6 oltre 30 gg). Classifica invariata, v2 resta il metodo scelto.


FEATURES PER CAPIRE LA PRESENZA DI UN PV - FLAVIO
#	What we tried	F1	TP	FP	FN	Verdict
0	Predict TRUE for everyone	0.710	22	18	0	Baseline
1	drop, first version (last 2 years only, no windows)	0.682	15	7	7	Rejected: below the baseline
2	drop (winter minus summer), on the windows	0.737	14	2	8	Rejected
3	winter_level	0.593	16	16	6	Rejected
4	duck (evening ramp-up)	0.714	15	5	7	Rejected: about the same as the baseline
5	step (lasting drop)	0.778	14	0	8	Rejected
6	var_summer (day-to-day swings)	0.780	16	3	6	Rejected
7	summer_level	0.783	18	6	4	Rejected
8	latest_summer	0.857	18	2	4	Kept only as a candidate
9	weekend_summer	0.889	20	3	2	Kept only as a candidate
10	dip_summer (midday against morning and afternoon)	0.894	21	4	1	Best single feature
11	clamp_summer (one clamping day is enough)	0.927	19	0	3	Rejected: fragile, relies on near-zero values
12	dip_summer + clamp tiebreaker (clamp needs ≥5% of summer days)	0.872	17	0	5	Rejected: worse than dip_summer alone
13	Logistic regression: dip_summer + weekend_summer	0.905	19	1	3	Best overall
14	Logistic regression: dip_summer + latest_summer	0.850	17	1	5	Rejected
15	Logistic regression: all three	0.878	18	1	4	Rejected
16	dip_spread (p90 - p10 of the daily summer dip: PV dip comes and goes with clouds)	0.927	19	0	3	Kept: new main feature, LOO threshold stable 0.836-0.856; 1-year windows 0.900 (0 FP). Thin margin: max no-PV 0.83 vs threshold 0.85
17	dip_season (dip winter minus summer), dip_ext (Apr-Sep), dip_q25, dip_min_month, dip_recent_change	<=0.894	-	-	-	Rejected: not better than dip_summer
18	Logistic regression: dip_spread + dip_summer / weekend_summer / dip_season / clamp (C=1 and 0.1)	0.842-0.872	-	0-1	5-6	Rejected: more FN than dip_spread alone
19	OR rule: dip_spread high OR dip_summer very low (both thresholds fitted in LOO)	0.864	19	3	3	Rejected: second threshold unstable across folds
20	Agent (Claude Code, 2026-10-17): proposed and tested features 16-19 on per-window aggregates only; remaining FN are large plants (166, 178, 959 kWp), one installed 75 days before data end
21	Second pass rec_drop (dip one year earlier minus dip in last 90 days), only on customers dip_spread calls no-PV	0.952	20	0	2	Kept: finds the 178 kWp plant installed 75 days before data end; stage-2 threshold stable 0.12-0.14 in LOO; no-PV max 0.05 vs threshold 0.12
22	Size normalisation: dip_spread divided by morning/afternoon spread (spread_norm); kWh dip spread / median shoulder kWh (spread_kwh)	0.878 / 0.696	18 / 16	1 / 8	4 / 6	Rejected: worse than dip_spread (log ratio is already size-free)
23	Agent (Claude Code, 2026-10-17): tested features 21-22 on per-window aggregates; checked new vs committed predict() end to end (0.927 -> 0.952)
