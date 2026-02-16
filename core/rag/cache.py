import hashlib
import json
import os
from typing import Dict


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


class IngestionCache:
    def __init__(self, cache_path: str = ".rag_cache.json"):
        self.cache_path = cache_path
        self._state: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}
        return {}

    def save(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def is_unchanged(self, path: str) -> bool:
        current_hash = sha256_file(path)
        return self._state.get(path) == current_hash

    def mark(self, path: str) -> None:
        self._state[path] = sha256_file(path)
