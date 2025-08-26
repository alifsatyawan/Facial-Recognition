import os
import pathlib
from typing import List, Tuple, Dict


class DatasetManager:
    def __init__(self, data_dir: pathlib.Path):
        self.data_dir = pathlib.Path(data_dir)

    def list_persons(self) -> List[Tuple[str, int]]:
        results: List[Tuple[str, int]] = []
        if not self.data_dir.exists():
            return results
        for person in sorted([p for p in self.data_dir.iterdir() if p.is_dir()]):
            cnt = len([f for f in person.glob("*.jpg")])
            results.append((person.name, cnt))
        return results

    def index_dataset(self) -> Tuple[List[str], List[str], Dict[str, int]]:
        image_paths: List[str] = []
        labels: List[str] = []
        label_to_index: Dict[str, int] = {}
        next_idx = 0
        for person in sorted([p for p in self.data_dir.iterdir() if p.is_dir()]):
            files = sorted([str(p) for p in person.glob("*.jpg")])
            if len(files) == 0:
                continue
            if person.name not in label_to_index:
                label_to_index[person.name] = next_idx
                next_idx += 1
            for f in files:
                image_paths.append(f)
                labels.append(person.name)
        return image_paths, labels, label_to_index


