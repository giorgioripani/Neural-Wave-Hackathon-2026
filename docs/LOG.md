# Registro del lavoro

Una riga per ogni idea scartata (cosa, perché, numeri) e per ogni uso importante di un agente.

- 2026-10-17 Francesco (agente Claude Code): bozza di `src/models/capacity.py` (calo del prelievo a mezzogiorno d'estate + regressione verso i kWp), test e `artifacts/capacity.json`.
- 2026-10-17 Francesco: scartate per ora `mean_kw` log-log (kWp in proporzione alla taglia del cliente), `drop_kw` log-log e `peak_kw` log-log: LOO sui 22 clienti PV, finestra di 2 anni casuale, errore mediano 0,43 / 0,43 / 0,44 contro 0,28 della `drop_kw` lineare. Nessun modello batte nettamente la mediana costante (P medio 0,41 contro 0,39).
