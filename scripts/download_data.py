"""Descarga el dataset de resultados de fútbol internacional desde Kaggle.

Lee KAGGLE_USERNAME y KAGGLE_KEY desde el archivo .env y descomprime
el dataset en ./data
"""
import os
from pathlib import Path

from prode import DATA_DIR, PROJECT_ROOT

DATASET = "martj42/international-football-results-from-1872-to-2017"


def load_env(env_path: Path) -> None:
    """Carga variables del .env al entorno (sin dependencias extra)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    load_env(PROJECT_ROOT / ".env")

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "Faltan credenciales. Completá KAGGLE_USERNAME y KAGGLE_KEY en el archivo .env"
        )

    # Importar después de setear las env vars (kaggle valida al importar)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Descargando {DATASET} en {DATA_DIR} ...")
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)
    print("Listo. Archivos:")
    for f in sorted(DATA_DIR.glob("*")):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
