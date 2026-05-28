from pathlib import Path
from typing import Literal

from datasets import Dataset, load_dataset


class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])
    DATASET_TYPES = frozenset(["flat", "sequential"])
    DATASET_SIZES = frozenset(["50m", "500m", "5b"])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        if dataset_type not in self.DATASET_TYPES:
            raise ValueError(f"Unsupported dataset_type: {dataset_type!r}")
        if dataset_size not in self.DATASET_SIZES:
            raise ValueError(f"Unsupported dataset_size: {dataset_size!r}")
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]) -> Dataset:
        if event_type not in self.INTERACTIONS:
            raise ValueError(f"Unsupported event_type: {event_type!r}")
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    def save_interaction(self, event_type: str, output_path: str | Path) -> Path:
        dataset = self.interaction(event_type)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(target))
        return target

    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        return load_dataset(
            "yandex/yambda",
            data_dir=data_dir,
            data_files={"train": f"{file}.parquet"},
            split="train",
        )


def download_default_parts(
    output_dir: str | Path = "dataset-parts",
    dataset_type: Literal["flat", "sequential"] = "flat",
    dataset_size: Literal["50m", "500m", "5b"] = "50m",
) -> dict[str, Path]:
    dataset = YambdaDataset(dataset_type, dataset_size)
    output_dir = Path(output_dir)
    saved_paths: dict[str, Path] = {}

    for event_type in ("likes", "dislikes", "listens"):
        saved_paths[event_type] = dataset.save_interaction(
            event_type,
            output_dir / f"yambda_{event_type}",
        )

    return saved_paths


likes = "dataset-parts/yambda_likes"
dislikes = "dataset-parts/yambda_dislikes"
listens = "dataset-parts/yambda_listens"


if __name__ == "__main__":
    saved = download_default_parts()
    for event_type, path in saved.items():
        print(f"{event_type}: saved to {path}")
