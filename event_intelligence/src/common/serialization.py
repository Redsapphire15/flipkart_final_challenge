from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
