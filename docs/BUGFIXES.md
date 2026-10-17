# Bug corretti nel branch `fix/bug-fixes`

Base: `main` 09d3ab3. Commit: `baaacc0` (inference.py e test) e `8bd2526` (loo_pipeline.py).
Verifica generale: 167 test passano. Sul test finto (40 clienti, ultimi 2 anni) `predictions.csv` e
`pv_extensions_predictions.csv` sono identici byte per byte a quelli di `main`. Le correzioni cambiano
solo i casi anomali, non le previsioni sui dati puliti.

## 1. Un `timestamp_epoch` vuoto blocca tutta l'inferenza
- **File:** `inference.py`, `read_test`.
- **Sintomo:** se anche una sola riga del CSV di test ha `timestamp_epoch` vuoto o illeggibile, il programma
  si ferma con `IntCastingNaNError` e non scrive né `predictions.csv` né `pv_extensions_predictions.csv`.
  È il caso peggiore possibile: zero punti.
- **Causa:** `df["timestamp_epoch"].astype("int64")` non accetta NaN.
- **Correzione:** l'epoch viene convertito con `pd.to_numeric(..., errors="coerce")`. Le righe senza epoch
  valido vengono tolte con un avviso, perché non si sa a che istante appartengono. L'elenco completo dei
  clienti viene salvato prima in `df.attrs["customer_ids"]`. Se un cliente resta senza nessuna riga,
  `predict_all` scrive comunque la sua riga, con `FALSE`. Così resta vera la regola "una riga per ogni
  customer_id del test".
- **Test:** `test_epoch_missing_rows_dropped_customer_kept`.

## 2. Intestazione con spazi e id numerici: crash e zeri iniziali persi
- **File:** `inference.py`, `read_test`.
- **Sintomo:** con un'intestazione come `" customer_id ";...` e id fatti solo di cifre (`000123`), il
  programma va in errore (`AttributeError: Can only use .str accessor with string values`). Senza il crash,
  `000123` diventerebbe `123`, e il PDF chiede di conservare gli zeri iniziali.
- **Causa:** `dtype={"customer_id": str}` si applica al nome di colonna originale, quello con gli spazi.
  Gli spazi venivano tolti solo dopo la lettura, quindi la colonna era già stata letta come numero.
- **Correzione:** tutte le colonne vengono lette come testo (`dtype=str`). kWh ed epoch vengono poi
  convertiti con `_to_number`, che accetta anche la virgola decimale. Gli id vuoti diventano NaN.
- **Test:** `test_header_with_spaces_keeps_leading_zeros`.

## 3. Righe ripetute tenute in inferenza
- **File:** `inference.py`, `read_test`.
- **Sintomo:** nessun errore, ma le righe con stesso cliente e stesso epoch venivano contate due volte
  nelle medie.
- **Causa:** in sviluppo `load_series` toglie i duplicati, `read_test` no. Il modello era quindi valutato
  su dati diversi da quelli che vede in gara.
- **Correzione:** dopo l'ordinamento, `drop_duplicates(["customer_id", "timestamp_epoch"], keep="first")`
  con un avviso, come fa `load_series`.
- **Test:** `test_read_test_drops_duplicates`.

## 4. Date "NaT" e kWp infiniti accettati dagli adattatori
- **File:** `inference.py`, `call_date` e `call_capacity`.
- **Sintomo:** se `date.estimate` restituisce `NaT` o `None`, nel CSV finisce la data `NaT`, che vale 0
  per D. Se restituisce `"2024-5-1"`, la data esce in un formato diverso da AAAA-MM-GG. Se
  `capacity.estimate` restituisce `inf`, nel CSV finisce `inf`.
- **Causa:** `pd.Timestamp("NaT")` non solleva eccezioni, e la data veniva presa così com'era con
  `str(out)[:10]`. Il controllo `kwp > 0` è vero anche per `inf`.
- **Correzione:** una data `NaT` solleva un'eccezione, quindi si usa la riserva (primo giorno della serie).
  La data viene sempre scritta con `strftime("%Y-%m-%d")`. Il kWp deve essere finito, altrimenti si usa
  la riserva (la mediana).
- **Nota:** oggi `date.py` e `capacity.py` non producono questi valori. È una protezione degli adattatori,
  che esistono proprio per questo.
- **Test:** `test_call_date_rejects_nat_and_normalizes`, `test_call_capacity_rejects_inf`.

## 5. Alias obsoleto in `loo_pipeline.py`
- **File:** `src/eval/loo_pipeline.py`.
- **Sintomo:** nessun errore, ma il codice registrava un modulo finto `src.eval.score_date`. Il commento
  diceva che `run_detect.py` lo usava ancora, il che non è più vero da 8609d26.
- **Correzione:** tolte le tre righe. `test_loo_pipeline.py` passa e la validazione dà gli stessi
  punteggi (79,19 / 78,10 / 77,93).

## Come riprovare
```
python -m pytest -q
python -m pytest -q tests/test_inference.py -k "epoch_missing or header_with_spaces or drops_duplicates or rejects_nat or rejects_inf"
```
I cinque test della seconda riga falliscono sull'`inference.py` di `main` e passano su questo branch.
