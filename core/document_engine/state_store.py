import json
import os
from typing import Optional

from .models import DocumentCheckpoint


class JsonCheckpointStore:
    def __init__(self, root_dir: str = ".document_checkpoints"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path_for(self, job_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in job_id)
        return os.path.join(self.root_dir, f"{safe}.json")

    def load(self, job_id: str) -> Optional[DocumentCheckpoint]:
        path = self._path_for(job_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return DocumentCheckpoint(**payload)

    def save(self, checkpoint: DocumentCheckpoint) -> None:
        path = self._path_for(checkpoint.job_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.model_dump(), f, indent=2)
