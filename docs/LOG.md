
Una riga per ogni idea scartata (cosa, perché, numeri) e per ogni uso importante di un agente.
- 2026-10-17 Giorgio + agente: creati inference.py (try/except con riserve per detect/date/capacity), demo/make_fake_test.py, tests/test_inference.py. Su fake_test (40 clienti, ultimi 2 anni): 40 righe, 0 PV perche' detect.py non esiste ancora.
v1 (change point sul rapporto, finestre UTC fisse): D 0,547; mediana 5,5 gg; 9 clienti su 22 oltre 30 gg.
v2 (stagionalità + gradino costante, regola “nessun gradino”): D 0,582; mediana 4,0 gg; 7 clienti oltre 30 gg. Scelta.
v3 (gradino stagionale con termini post·sin e post·cos): D 0,568; mediana 4,5 gg; 6 clienti oltre 30 gg. Scartata: sistema POD22 e POD23 ma crea gradini falsi quando ci sono meno di circa sei mesi di dati prima dell’installazione (POD34, POD35).

La scelta della v2 tra v1, v2 e v3 è stata fatta sugli stessi 22 clienti: possibile leggero ottimismo. La soglia del 5% è stata fissata prima dei risultati.
