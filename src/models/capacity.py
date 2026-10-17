"""Task 3 (kWp): punto d'ingresso per inference.py.

Sceglie quale metodo usare. Entrambi espongono estimate(series, date) -> kWp:
    capacity_a  regressione Ridge su piu` feature solari verso log(kWp)
    capacity_b  picco del calo di mezzogiorno x una costante
Per cambiare metodo basta cambiare METHOD. Confronto: python -m src.models.capacity_compare
"""
import importlib

METHOD = "b"

_impl = importlib.import_module(f"src.models.capacity_{METHOD}")


def estimate(series, date: str) -> float:
    """Serie a 15 min di UN cliente + data 'AAAA-MM-GG' -> kWp (float > 0)."""
    return _impl.estimate(series, date)
