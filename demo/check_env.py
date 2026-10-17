#!/usr/bin/env python3
"""Controllo dell'ambiente prima di inference.py: python demo/check_env.py

Controlla versione di Python, pacchetti e versioni di requirements.txt,
artefatti in artifacts/ e import dei moduli di src/models/. Non legge dati.
Finisce con "AMBIENTE OK" (codice 0) o con l'elenco dei problemi (codice 1).
"""
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# nome pip -> nome import, quando sono diversi
IMPORT_NAMES = {"scikit-learn": "sklearn"}
ARTIFACTS = ["capacity.json", "detect_params.json"]
MODULES = ["detect", "date", "capacity"]


def read_requirements(path: Path) -> dict[str, str]:
    """requirements.txt -> {pacchetto: versione} (solo righe 'nome==versione')."""
    reqs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            reqs[name.strip()] = version.strip()
    return reqs


def check_python(problems: list[str]) -> None:
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    if (v.major, v.minor) != (3, 12):
        problems.append(f"Python {v.major}.{v.minor}: serve Python 3.12")


def check_packages(reqs: dict[str, str], problems: list[str]) -> None:
    for name, wanted in reqs.items():
        try:
            importlib.import_module(IMPORT_NAMES.get(name, name))
            found = metadata.version(name)
        except Exception as e:
            problems.append(f"pacchetto {name}=={wanted} mancante o rotto ({type(e).__name__})")
            continue
        status = "ok" if found == wanted else f"VERSIONE DIVERSA, richiesta {wanted}"
        print(f"  {name} {found}: {status}")
        if found != wanted:
            problems.append(f"{name} {found} invece di {wanted}")


def check_artifacts(problems: list[str]) -> None:
    for name in ARTIFACTS:
        path = ROOT / "artifacts" / name
        try:
            json.loads(path.read_text(encoding="utf-8"))
            print(f"  artifacts/{name}: ok")
        except FileNotFoundError:
            problems.append(f"artifacts/{name} non esiste")
        except Exception as e:
            problems.append(f"artifacts/{name} non e` un JSON valido ({type(e).__name__}: {e})")


def check_modules(problems: list[str]) -> None:
    for name in MODULES:
        try:
            importlib.import_module(f"src.models.{name}")
            print(f"  src.models.{name}: ok")
        except Exception as e:
            problems.append(f"src.models.{name} non si importa ({type(e).__name__}: {e})")


def main() -> int:
    problems: list[str] = []
    check_python(problems)
    reqs_path = ROOT / "requirements.txt"
    if reqs_path.exists():
        check_packages(read_requirements(reqs_path), problems)
    else:
        problems.append("requirements.txt non esiste")
    check_artifacts(problems)
    check_modules(problems)
    if problems:
        print("PROBLEMI:")
        for p in problems:
            print(f"- {p}")
        return 1
    print("AMBIENTE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
