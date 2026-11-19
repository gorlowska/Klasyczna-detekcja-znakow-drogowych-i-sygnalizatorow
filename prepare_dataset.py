"""Połącz zdjęcia i etykiety BDD100K po nazwie obrazu."""
import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

DATASET = Path("experiment_data/dataset.json")
CLASSES = ("traffic sign", "traffic light")


def prepare(archive):
    images = {
        path.stem: path
        for split in ("train", "val", "test")
        for path in (Path("100k") / split).glob("*.jpg")
    }
    dataset = {"train": [], "val": []}
    with ZipFile(archive) as labels:
        for entry in labels.namelist():
            path = PurePosixPath(entry)
            if path.suffix != ".json" or path.parent.name not in dataset:
                continue
            if path.stem not in images:
                continue
            label = json.loads(labels.read(entry))
            objects = [
                obj for obj in label["frames"][0]["objects"]
                if obj["category"] in CLASSES and "box2d" in obj
            ]
            dataset[path.parent.name].append({
                "image": images[path.stem].as_posix(),
                "objects": objects,
            })

    for split, rows in dataset.items():
        if not rows:
            raise SystemExit(f"Brak dopasowanych obrazów dla podziału detekcji {split}.")
        rows.sort(key=lambda row: row["image"])
        print(f"{split}: {len(rows)} par zdjęcie–etykieta")
    DATASET.parent.mkdir(exist_ok=True)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Archiwum bdd100k_labels.zip")
    prepare(parser.parse_args().archive)
