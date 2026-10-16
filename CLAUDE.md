# CLAUDE.md — Neural Wave: rilevamento fotovoltaico (AIL)

Leggi questo file a inizio sessione. Tienilo corto: costa token a ogni sessione.

## Obiettivo
Dalla sola curva di consumo a 15 minuti di un cliente (prelievo dalla rete, kWh) stimare:
1. `pv_installed` (TRUE/FALSE)
2. `pv_installation_date` (AAAA-MM-GG, vuoto se FALSE): data dell'impianto INIZIALE
3. `pv_capacity_kwp` (kWp, vuoto se FALSE): capacità dell'impianto INIZIALE
Bonus facoltativo (max 10 punti, solo a MVP finito): estensioni dell'impianto.
Punteggio = F1 x (70 + 15 x D + 15 x P). D: data (0 a 30 giorni di errore). P: kWp (0 al 50% di errore).
Consegna: domenica 18 ottobre, ore 11:00. `python inference.py path/test.csv` deve scrivere
`predictions.csv` e `pv_extensions_predictions.csv` (anche solo con l'intestazione) nella cartella corrente.

## Consegna (regole dell'organizzazione)
- Repo `NW_26_<azienda>_<squadra>` dell'organizzazione Neural Wave; ambiente con un semplice `pip install -r requirements.txt`; nessuno fa debug.
- Servono: `README.md` (panoramica, come eseguire, dipendenze, nome e cognome di tutti con il ruolo), report di max 3 pagine, video di max 2 minuti.
- Report: lo scrivono le persone, non tu. Aiuta solo con grafici, tabelle e controllo dei numeri. Non scrivere testo di report.
- Tieni `docs/LOG.md`: una riga per ogni idea scartata (cosa, perche`, numeri) e una per ogni uso importante di un agente. Il repo diventera` open source: niente dati di AIL ne` chiavi.

## Dati
- 40 clienti di sviluppo con etichette, 20 di test nascosti, 1148 curve senza etichette (facoltative). 2022-2025.
- Colonne: `customer_id`, `energy_consumption_kwh`, `timestamp_local`, `timestamp_epoch`.
- VIETATO: meteo, posizione, irraggiamento, riconoscere i clienti. Permesse: feature di calendario (ora, giorno, mese).
- Errore del prof: nei dati di sviluppo tenere solo `customer_id LIKE 'POD%'` (toglie anche gli id nulli):
  `python src/loading/prepare_data.py <csv> data/clean/<nome>.parquet`.
- In `inference.py` NON scartare nessun cliente: una riga di output per ogni `customer_id` del test.
- Ora legale: esistono giorni da 92 e da 100 misure. Ordinare con `timestamp_epoch`. Mai assumere 96 righe al giorno.
- `customer_id` sempre come stringa. Dati fuori da Git (`data/` è nel `.gitignore`).

## Formato di uscita (dal PDF)
UTF-8, separatore punto e virgola, punto decimale, `TRUE`/`FALSE`, date AAAA-MM-GG, campi vuoti se non applicabili.
Colonne esatte e nell'ordine: `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`
e `customer_id;extension_date;pv_capacity_after_extension_kwp`.
[Da confermare con gli organizzatori: gli esempi del PDF usano le virgole.]

## Struttura e proprietari (un modulo a persona)
| Cartella | Persona | Contenuto |
| --- | --- | --- |
| `inference.py`, `demo/` | Giorgio | Entry point, README, merge, prova in ambiente pulito |
| `src/loading/`, `src/eval/` | Andrea | Caricamento, filtro POD, cifre giornaliere, punteggio ufficiale, validazione a un cliente alla volta |
| `src/models/detect.py` | Flavio | Task 1: c'è il fotovoltaico? |
| `src/models/date.py` | Leonardo | Task 2: data (poi estensioni) |
| `src/models/capacity.py`, `src/viz/` | Francesco | Task 3: kWp, grafici |
Lavora solo nel modulo della persona che ti usa. Se serve toccare un altro modulo, dillo prima.

## Interfacce (proposta, Andrea e Giorgio le fissano venerdì sera)
- `load_series(path) -> DataFrame`: `customer_id` (str), `ts_epoch` (int), `kwh` (float), `ts_local` (str).
- `daily_features(df) -> DataFrame`: una riga per cliente e giorno (consumo di mezzogiorno, notte, sera e rapporti).
- `detect.predict(feat) -> {customer_id: probabilità}`; `date.estimate(feat_cliente) -> 'AAAA-MM-GG'`;
  `capacity.estimate(feat_cliente, data) -> kWp`.
[Aggiornare qui ogni cambio di firma.]

## Metodo (poco codice, modelli semplici)
- Segnale: calo dei prelievi a mezzogiorno, più forte in estate che in inverno; salto stabile alla data di messa in servizio.
- Modelli semplici (regola, regressione logistica, alberi piccoli, regressione lineare, change point). Niente reti profonde.
- Validare sempre a un cliente alla volta (leave-one-out). Mai scegliere parametri sul test. Seed fissi (42).
- Per ogni cliente classificato con fotovoltaico, riempire sempre data e kWp (un valore mancante vale 0).
- Modelli e artefatti salvati in `artifacts/` e caricati da `inference.py`. Niente rete in inferenza.

## Risparmio token (importante)
- Non aprire, non stampare e non leggere interi i CSV. Lavorare su Parquet; usare `head(5)`, `describe()`, `shape`.
- Gli script stampano al massimo 10 righe (metriche) e salvano i grafici come file in `outputs/`.
- Non guardare immagini se non richiesto. Richieste piccole, con file e criterio di verifica.
- Cifre per cliente calcolate una volta e salvate; non rielaborare i dati grezzi a ogni sessione.
- Niente subagenti né esplorazioni ampie del repo se non richiesto. Risposte brevi: cosa hai fatto, risultato, problemi.

## Qualità
- Funzione nuova = test in `tests/`; esegui i test prima di dire che hai finito.
- Se un risultato sembra troppo buono, sospetta perdita di informazione dal futuro o un errore nello split, e dillo.
- Dichiara i limiti del metodo. Nessun numero inventato: ogni cifra viene da un'esecuzione reale.

## Git
- Mai su `main`: branch `nome/compito`, commit piccoli e frequenti, pull request, merge di Giorgio.
- `git pull` a inizio sessione, poi `git log -10`. Niente `push --force`, `reset --hard` o dati nei commit senza conferma.
- Aggiorna `STATUS.md` (una riga, la tua) quando chiudi un pezzo.
