from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
MODELS_DIR = PACKAGE_ROOT / "models"
DATA_DIR = PACKAGE_ROOT / "data"
CONFIG_DIR = PACKAGE_ROOT / "configs"


def discover_dataset(explicit_path: str | Path | None = None) -> Path:
    """Find the event CSV without assuming it has already been copied."""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset not found: {path}")

    candidates = list(DATA_DIR.glob("*.csv")) + list(REPO_ROOT.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No CSV dataset found. Put the event CSV in event_intelligence/data/ "
            "or the repository root."
        )
    return max(candidates, key=lambda p: p.stat().st_size)
