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

Le estensioni sono pronte sul branch lodoscatenato. In inference.py, per ogni cliente segnato TRUE, chiama extensions.predict_extensions(serie_cliente, data_iniziale, kwp_iniziale) dentro un try/except (in caso di errore: nessuna estensione). Raccogli le righe di tutti i clienti con il loro customer_id, applica extensions.top_k(righe, k=3) e scrivile in pv_extensions_predictions.csv (date AAAA-MM-GG, kWp con il punto decimale). Se non ci sono righe, resta la sola intestazione. Il bonus non può togliere punti al punteggio principale. Valutazione con iniezione sintetica, in modo realistico: F1_ext atteso circa 0,29 (circa 2,5 punti). Dettagli in docs/LOG.md.

(.venv) lodo@lodo-XPS-13-9370:~/Desktop/AI-uto-$ git add src/models/date.py src/models/run_date_dev.py src/models/inject_ext_dev.py tests/
git commit -m "Data v4 (due gradini): D 0,628 vs 0,597 sui 22 clienti"
git push

python src/models/inject_ext_dev.py --reestimate-date0 --date-method v2
python src/models/inject_ext_dev.py --reestimate-date0 --date-method v4
[lodoscatenato f24706a] Data v4 (due gradini): D 0,628 vs 0,597 sui 22 clienti
 5 files changed, 164 insertions(+), 21 deletions(-)
Enumerating objects: 19, done.
Counting objects: 100% (19/19), done.
Delta compression using up to 8 threads
Compressing objects: 100% (10/10), done.
Writing objects: 100% (10/10), 3.96 KiB | 3.96 MiB/s, done.
Total 10 (delta 9), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (9/9), completed with 9 local objects.
To https://github.com/giorgioripani/AI-uto-.git
   77c4e76..f24706a  lodoscatenato -> lodoscatenato
Clienti usati: 19, saltati: 2 (pochi_mesi 2)
date0 ristimato (v2) entro 30 gg dalla data iniettata: 27/114 (r0.3=5/38 r0.6=9/38 r1.0=13/38)
soglia 0.050: recall 0.17 (r0.3=0.11 r0.6=0.13 r1.0=0.26), err gg 2, err kWp 42%, FA/cliente 0.00, F1_ext 0.29
soglia 0.075: recall 0.11 (r0.3=0.08 r0.6=0.11 r1.0=0.13), err gg 2, err kWp 53%, FA/cliente 0.00, F1_ext 0.19
soglia 0.100: recall 0.07 (r0.3=0.00 r0.6=0.08 r1.0=0.13), err gg 1, err kWp 41%, FA/cliente 0.00, F1_ext 0.13
soglia 0.150: recall 0.04 (r0.3=0.00 r0.6=0.05 r1.0=0.05), err gg 1, err kWp 53%, FA/cliente 0.00, F1_ext 0.07
Clienti usati: 19, saltati: 2 (pochi_mesi 2)
date0 ristimato (v4) entro 30 gg dalla data iniettata: 27/114 (r0.3=5/38 r0.6=9/38 r1.0=13/38)
soglia 0.050: recall 0.19 (r0.3=0.11 r0.6=0.18 r1.0=0.29), err gg 2, err kWp 66%, FA/cliente 0.00, F1_ext 0.32
soglia 0.075: recall 0.14 (r0.3=0.08 r0.6=0.16 r1.0=0.18), err gg 2, err kWp 67%, FA/cliente 0.00, F1_ext 0.25
soglia 0.100: recall 0.11 (r0.3=0.00 r0.6=0.13 r1.0=0.18), err gg 1, err kWp 56%, FA/cliente 0.00, F1_ext 0.19
soglia 0.150: recall 0.07 (r0.3=0.00 r0.6=0.11 r1.0=0.11), err gg 1, err kWp 67%, FA/cliente 0.00, F1_ext 0.13

Diagnosi della data rubata dall’estensione: 26 casi su 27 hanno l’installazione vera prima della finestra (o nei primi 60 giorni). In quei casi nella finestra si vede un solo gradino e il metodo lo prende per l’installazione. Soluzione individuata: decidere se prima del gradino c’è già fotovoltaico con il rilevatore del Task 1.

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


Data v4 (due gradini, il primo come installazione): sui 22 clienti D 0,628 contro 0,597 della v2 (1 cliente migliorato, 0 peggiorati); nell’iniezione realistica la data rubata dall’estensione resta 27/114 come con la v2, la recall delle estensioni passa da 0,17 a 0,19 e l’errore kWp da 42% a 66%. Non adottata: il secondo criterio di accettazione, fissato prima, non è soddisfatto e il guadagno di D si basa su un solo cliente.
Diagnosi della data rubata dall’estensione: 26 casi su 27 hanno l’installazione vera prima della finestra (o nei primi 60 giorni). Nella finestra si vede un solo gradino e il metodo lo prende per l’installazione. Soluzione individuata: decidere con il rilevatore del Task 1 se prima del gradino c’è già fotovoltaico.

refine_date0 (detect sul tratto prima del gradino) per la data principale: D 0,556 contro 0,597 della v2 (3 clienti peggiorati) → respinto. Per le estensioni: data rubata 27→6 su 114, recall 0,17→0,33, F1_ext atteso 0,29→0,50 → da usare solo nella ricerca delle estensioni, se i falsi allarmi sui clienti reali restano bassi.

- 2026-10-17 Leonardo (agente Claude Code): predict_extensions_refined (refine_date0 solo come inizio della ricerca delle estensioni, date0 e kwp0 invariati), --refine in run_ext_dev.py e inject_ext_dev.py (anche sulle serie originali), _kwp_ratio con s1 non stimabile se l'installazione e` al primo giorno (kWp di riserva 1,1 x kwp0); test in tests/test_refine.py.

## 17/10 — Punteggio LOO della pipeline intera (Andrea, con agente Claude)
Agente usato per scrivere src/eval/loo_pipeline.py: pipeline di inference.py dentro validate.py (2 anni, 5 giri, seed 42).
Artefatti attuali 79,19 ± 2,92; soglie detect ricalcolate senza il cliente 78,10 ± 2,42; detect e capacity ricalcolati 77,93 ± 2,42 (F1 0,909, D 0,608, P 0,439). La perdita per l'in-sample è piccola (circa 1,3 punti).
Limite: si ricalcolano soglie e coefficienti, non la scelta di feature e metodo (dip_spread + rec_drop, geo_b), fatta guardando tutti i 40 clienti. Neanche date.py v2 è stato scelto in LOO.
24	Decision tree on all 11 features (depth 1-3, min leaf 5/10, balanced or not), LOO on 2-year windows	0.927-0.952	20	0	2	Rejected: best ties the cascade; root split is clamp_summer > 0.002 (one zero-midday day), fragile; it rediscovers the second pass rec_drop > 0.135
25	Same trees without clamp_summer / random forest (200 trees, depth 3)	0.927 (d2), 0.884 (d3) / 0.857	-	0-2	3-4	Rejected: no gain, deeper models overfit 40 customers
26	Agent (Claude Code, 2026-10-17): wrote src/models/tree_detect.py (experiment, not in inference) and ran the tree comparison
27	Agent (Claude Code, 2026-10-17): detect.py safety: invalid epoch rows dropped, per-customer try (failure -> no PV for that customer only), rec_drop missing keeps main proba. LOO unchanged 0.952; 19/19 synthetic edge cases OK

## 17/10 — Lettore tollerante del test (Andrea, con agente Claude, richiesta di Flavio)
Prima: un epoch vuoto o testo in una riga del test faceva fallire tutto inference.py (read_test fuori da ogni try), e un cliente con sole righe non valide spariva dall'output.
Ora: src.loading.load.load_test (via veloce come il vecchio read_test; se ci sono righe malformate, lettura come testo con le righe saltate), all_ids da tutte le righe del file, rete di sicurezza in main() (tutti TRUE). fake_test: predictions.csv e estensioni identici byte per byte, 19 s contro 17. Test con CSV sintetici in tests/test_load_validate.py.

## 17/10 — Revisione di main (Francesco, con agente Claude)
Agente usato per la revisione del codice di main (09d3ab3), i test di integrazione e i casi limite su dati sintetici. Correzioni nel branch fix/bug-fixes, proposte in IMPROVEMENTS.md.
Bug corretti in inference.py: crash con un timestamp_epoch vuoto; crash e zeri persi con intestazione con spazi e id numerici; duplicati non tolti (diverso da load_series); data "NaT" e kWp inf accettati. Uscite sul test finto identiche a main.
Scartata: se refine_date0 dice pv_gia_presente, ristimare la data sul tratto prima di date0. D sul test finto da 0,372 a 0,338; usare il primo giorno della serie dà lo stesso 0,338.
Misura per il report: LOO tutto_loo con finestre "ultimi 2 anni" 78,20 (F1 0,952, D 0,372, P 0,436) contro 77,93 (D 0,608) con la data vera nel 10-90% della finestra. Nel dev 9 clienti su 22 con PV hanno l'installazione prima degli ultimi 2 anni.

## 17/10 — Estensioni: audit e confronto anno su anno (Andrea, con agente Claude)
Audit: extensions.py non è rotto (formato, id, kWp totali e collegamento giusti; sintetico F1_ext 0,29-0,50). Sull'unica estensione vera fa 0 perché cade 48 giorni prima della fine dei dati: il gradino migliore è un altro, 474 giorni prima; con END_MARGIN_DAYS 30 invece di 60 non cambia nulla.
Scartato: confronto anno su anno (28 gg dopo e prima di t, meno gli stessi giorni dell'anno prima) su mezzogiorno, 7-16 UTC e spalle, log o lineare, mediana o quantile 0,1-0,2, finestre 28 o 42 gg. Data dell'estensione vera giusta entro ±6 gg con la mediana, ma il suo calo è dentro il rumore dei 19 clienti senza estensione (8-15 di loro superano lo stesso calo con soglia -0,5): niente soglia con 0 falsi allarmi. Il meteo di un mese tra due anni sposta la firma quanto un raddoppio dei kWp.
Nuovo src/eval/score_ext.py: punteggio ufficiale del bonus (abbinamento uno a uno entro 30 gg). fake_test con il metodo attuale: TP 0 FP 3 FN 1, bonus 0.

Prova finale della data (regola fissata prima). Serie intera: v2 0,617 (mediana 1,5 gg, 6/22 oltre 30 gg); refined 0,577; v2+refine 0,617; refined+refine 0,577; selezione annidata sceglie la v2 22/22. Ultimi 730 giorni: v2 0,338; refined 0,397 (3 migliorati / 0 peggiorati, IC90% [+0,002; +0,147]); selezione annidata sceglie refined 22/22. Tre finestre: v2 0,597; refined 0,618. Decisione: v2, salvo conferma degli organizzatori che le serie del test siano tagliate a circa 2 anni (in quel caso refined, che passa tutte le condizioni su quel protocollo).

Risposta degli organizzatori (sabato 15:30): il test copre il periodo 2022–2025 come i dati di sviluppo, già filtrato sui POD; in precedenza: nessuna installazione prima del 2022 nei dati etichettati. Quindi il protocollo di riferimento è la serie intera: resta la v2 (D 0,617, mediana 1,5 gg, 6/22 oltre 30 gg). Il raffinamento locale, migliore solo con serie tagliate a 2 anni, non viene adottato. Da correggere nel CLAUDE.md l’indicazione “test di circa 2 anni”.
- 2026-10-17 Giorgio (agente Claude Code): merge finale su merge/finale e poi main (f82db85, tag v2-consegna) di 5 branch con controlli dopo ogni merge (test, check_env, uscite su fake_test a 2 anni identiche byte per byte); un solo lettore del test (load_test), da fix/bug-fixes portati i controlli di call_date/call_capacity; detect con crash -> NaN (riserva TRUE); fake_test e LOO passati alla serie intera (LOO tutto_loo 78,49, F1 0,905, D 0,665, P 0,452; a 2 anni 77,93 ± 2,42).
- 2026-10-17 Flavio (agente Claude Code): detect.py usa solo gli ultimi 2 anni della serie (LAST_DAYS = 730, dentro features_from_daily, uguale alle finestre di addestramento di run_detect). Test 4 anni: F1 LOO 0,905 -> 0,952 (1 FP e 1 FN in meno); pipeline intera LOO serie intera 78,49 -> 81,82 (D 0,665 -> 0,632, P 0,452 -> 0,429: cambiano i TP su cui si medono). fake_test 4 anni e 2 anni: stesse risposte PV per tutti i 40 clienti. Soglie invariate.
- 2026-10-17 Flavio (agente Claude Code): controllo di sanita' di detect sui 1147 clienti senza etichetta (src/models/unlabelled_detect.py, 34 min; non cambia il modello). Classificati PV 392 (34%): 353 dalla prima regola, 39 da rec_drop. Taglio di Otsu su log10(dip_spread) senza etichette 1,083 contro soglia 0,853: la soglia sta sulla coda del gruppo senza PV, non nella valle (densita' in calo continuo da 0,5 a 2,5; 132 clienti tra 0,68 e 1,07). Stima del rischio: log-normale sui 18 no-PV etichettati -> P(dip_spread > 0,853) circa 7%. Soglia NON cambiata: sulle etichette alzarla a 1,083 perde 1 PV vero. Uscite: outputs/unlabelled_detect_features.csv (serve anche alle pseudo-etichette di Giorgio) e 2 grafici.
- 2026-10-17 Flavio (agente Claude Code): verifica del secondo passo `rec_drop` con un "test dello specchio" sui 794 clienti senza etichetta che la prima regola chiama no-PV: prima di un impianto `rec_drop` e` simmetrico intorno a 0, quindi gli scatti della regola specchiata (-rec_drop > soglia) stimano i falsi positivi. Regola salvata: 39 scatti contro 40 specchiati, cioe` rumore (circa 5% dei no-PV). Varianti piu` severe provate (src/models/recent_check.py, LOO sui 40 + specchio): step>0 F1 0,952, 31 vs 27; step>0,3 F1 0,952, 18 vs 19; step>0,5 F1 0,927, 14 vs 8; dip_spread>0,5 F1 0,927, 14 vs 19; nessun secondo passo F1 0,927. Nessuna variante separa segnale da rumore senza perdere l'unico PV recuperato (installato 75 giorni prima della fine serie). Scartate; modello NON cambiato: sul test (circa 9 no-PV e 11 PV) il secondo passo vale circa +0,5 TP e +0,45 FP attesi, un pareggio.
- 2026-10-17 Flavio (agente Claude Code): revisione finale della parte Task 1 (detect.py, run_detect.py, tree_detect.py, unlabelled_detect.py, test): nessun bug trovato; tutte le librerie importate sono in requirements.txt; suite completa 214 test passati (pytest 9.1.1). Dopo il merge di flavio/detect3 vanno aggiornati i numeri del README (LOO serie intera 78,49 -> 81,82; F1 0,905 -> 0,952) e rifatta la verifica su clone pulito.

## 17/10 — README finale e controllo di consegna (Andrea, con agente Claude)
Agente usato per il controllo completo di main (8ddd85a): installazione da zero in un venv nuovo (AMBIENTE OK), 205 test, fake_test serie intera 78,66 in-sample, CSV sporchi/vuoti/senza colonne (sempre exit 0 e file scritti), LOO serie intera 78,49 e finestre di 2 anni 77,93 ± 2,42 rifatti. Sezioni del README riempite (metodo, risultati, limiti, uso di agenti, crediti); tolto DatabaseReadme.md (materiale di AIL).
- 2026-10-17 Flavio (agente Claude Code): README aggiornato a v3 (rilevamento sugli ultimi 2 anni): LOO serie intera 81,82 (F1 0,952, D 0,632, P 0,429), riga della versione precedente 78,49, finestre di 2 anni invariate 77,93 ± 2,42 (rieseguito), 20/22 trovati e 0 falsi allarmi, date 13/20 entro 7 giorni e 5/20 oltre 30, kWp 8/20 oltre il 50%; limiti aggiornati (2 grandi utenze non viste, soglia sulla coda del gruppo senza PV). Tag v3.1-consegna.
