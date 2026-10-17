#!/usr/bin/env python3
"""Environment check before inference.py: python demo/check_env.py

Checks the Python version, the packages and versions of requirements.txt,
the artifacts in artifacts/ and the import of the src/models/ modules. Reads no data.
Ends with "ENVIRONMENT OK" (exit code 0) or with the list of problems (exit code 1).
"""
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pip name -> import name, when they differ
IMPORT_NAMES = {"scikit-learn": "sklearn"}
ARTIFACTS = ["capacity.json", "detect_params.json"]
MODULES = ["detect", "date", "capacity", "extensions"]


def read_requirements(path: Path) -> dict[str, str]:
    """requirements.txt -> {package: version} (only 'name==version' lines)."""
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
        problems.append(f"Python {v.major}.{v.minor}: Python 3.12 is required")


def check_packages(reqs: dict[str, str], problems: list[str]) -> None:
    for name, wanted in reqs.items():
        try:
            importlib.import_module(IMPORT_NAMES.get(name, name))
            found = metadata.version(name)
        except Exception as e:
            problems.append(f"package {name}=={wanted} missing or broken ({type(e).__name__})")
            continue
        status = "ok" if found == wanted else f"DIFFERENT VERSION, required {wanted}"
        print(f"  {name} {found}: {status}")
        if found != wanted:
            problems.append(f"{name} {found} instead of {wanted}")


def check_artifacts(problems: list[str]) -> None:
    for name in ARTIFACTS:
        path = ROOT / "artifacts" / name
        try:
            json.loads(path.read_text(encoding="utf-8"))
            print(f"  artifacts/{name}: ok")
        except FileNotFoundError:
            problems.append(f"artifacts/{name} does not exist")
        except Exception as e:
            problems.append(f"artifacts/{name} is not valid JSON ({type(e).__name__}: {e})")


def check_modules(problems: list[str]) -> None:
    for name in MODULES:
        try:
            importlib.import_module(f"src.models.{name}")
            print(f"  src.models.{name}: ok")
        except Exception as e:
            problems.append(f"src.models.{name} cannot be imported ({type(e).__name__}: {e})")


def main() -> int:
    problems: list[str] = []
    check_python(problems)
    reqs_path = ROOT / "requirements.txt"
    if reqs_path.exists():
        check_packages(read_requirements(reqs_path), problems)
    else:
        problems.append("requirements.txt does not exist")
    check_artifacts(problems)
    check_modules(problems)
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"- {p}")
        return 1
    print("ENVIRONMENT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
