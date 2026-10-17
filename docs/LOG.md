# Registro del lavoro

Una riga per ogni idea scartata (cosa, perché, numeri) e per ogni uso importante di un agente.

- 2026-10-17 Francesco (agente Claude Code): bozza di `src/models/capacity.py` (calo del prelievo a mezzogiorno d'estate + regressione verso i kWp), test e `artifacts/capacity.json`.
- 2026-10-17 Francesco: scartate per ora `mean_kw` log-log (kWp in proporzione alla taglia del cliente), `drop_kw` log-log e `peak_kw` log-log: LOO sui 22 clienti PV, finestra di 2 anni casuale, errore mediano 0,43 / 0,43 / 0,44 contro 0,28 della `drop_kw` lineare. Nessun modello batte nettamente la mediana costante (P medio 0,41 contro 0,39).
- 2026-10-17 Francesco (agente Claude Code): diviso il Task 3 in `capacity_a.py` (Ridge 5 feature su log kWp), `capacity_b.py` (calo di mezzogiorno), `capacity_common.py`, `capacity_compare.py`; `capacity.py` sceglie il metodo.
- 2026-10-17 Francesco: scartato B "puro" kWp = k x peak_kw: P medio 0,25 / 0,16 / 0,18 / 0,10 (random-vera / last-vera / random-stimata / last-stimata) contro 0,41 della mediana. Causa: saturazione, quota mediana di quarti d'ora a prelievo zero d'estate 0,49 (16/22 clienti > 0,1); kWp/peak_kw varia da 1,7 a 5,4 (quartili), k=2,35.
- 2026-10-17 Francesco: A (Ridge) e B lineare in pareggio: P medio A 0,33 / 0,42 / 0,35 / 0,29, B lineare 0,39 / 0,34 / 0,38 / 0,27; nessuno batte la mediana in modo stabile. Con la data di date.py (errore mediano 114 giorni) entrambi peggiorano.
