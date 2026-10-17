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
