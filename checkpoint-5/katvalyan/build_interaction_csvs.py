from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parent
INTERACTION_DIR = ROOT / "interaction"
RAW_LIKES_DIR = ROOT.parent / "checkpoint-5-bogdan-egor" / "bogdan" / "dataset-parts" / "yambda_likes"
WEEK_TICKS = (7 * 24 * 3600) // 5
YEAR_TICKS = (365 * 24 * 3600) // 5
CHUNK_ROWS = 200_000


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source_dir: Path
    sign_value: int
    weighted_mode: str


@dataclass
class DatasetSummary:
    dataset: str
    total_rows: int
    organic_rows: int
    train_rows: int
    unique_users_train: int
    unique_items_train: int
    q70_user_events: float
    q95_user_events: float
    final_pairs: int
    sign_rule: str
    weighted_rule: str


def arrow_files(path: Path) -> list[Path]:
    files = sorted(path.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"Missing data-*.arrow in {path}")
    return files


def iter_record_batches(path: Path) -> Iterator[pa.RecordBatch]:
    for file_path in arrow_files(path):
        with pa.memory_map(str(file_path), "r") as source:
            reader = ipc.open_stream(source)
            while True:
                try:
                    yield reader.read_next_batch()
                except StopIteration:
                    break


def max_organic_timestamp(path: Path) -> int:
    max_time: int | None = None
    for batch in iter_record_batches(path):
        is_organic = batch.column(batch.schema.get_field_index("is_organic")).to_numpy(zero_copy_only=False)
        if not is_organic.any():
            continue
        timestamps = batch.column(batch.schema.get_field_index("timestamp")).to_numpy(zero_copy_only=False)
        batch_max = int(timestamps[is_organic == 1].max())
        if max_time is None or batch_max > max_time:
            max_time = batch_max
    if max_time is None:
        raise ValueError(f"No organic events found in {path}")
    return max_time


def flush_chunk(
    pair_latest: dict[int, tuple[int, float]],
    uid_parts: list[np.ndarray],
    item_parts: list[np.ndarray],
    ts_parts: list[np.ndarray],
    *,
    weighted_mode: str,
    weight_parts: list[np.ndarray] | None = None,
) -> None:
    if not uid_parts:
        return

    chunk = pd.DataFrame(
        {
            "uid": np.concatenate(uid_parts),
            "item_id": np.concatenate(item_parts),
            "timestamp": np.concatenate(ts_parts),
        }
    )

    if weight_parts is not None:
        chunk["raw_weight"] = np.concatenate(weight_parts)

    chunk = chunk.sort_values(["uid", "item_id", "timestamp"], kind="mergesort")
    chunk = chunk.drop_duplicates(subset=["uid", "item_id"], keep="last")

    uids = chunk["uid"].to_numpy(dtype=np.uint32, copy=False)
    item_ids = chunk["item_id"].to_numpy(dtype=np.uint32, copy=False)
    timestamps = chunk["timestamp"].to_numpy(dtype=np.uint32, copy=False)

    if weighted_mode == "constant_0.1":
        weights = np.full(len(chunk), 0.1, dtype=np.float32)
    elif weighted_mode == "played_ratio_last":
        weights = chunk["raw_weight"].to_numpy(dtype=np.float32, copy=False)
    else:
        raise ValueError(f"Unknown weighted_mode: {weighted_mode}")

    for uid, item_id, timestamp, weight in zip(uids, item_ids, timestamps, weights, strict=True):
        key = (int(uid) << 32) | int(item_id)
        prev = pair_latest.get(key)
        if prev is None or int(timestamp) >= prev[0]:
            pair_latest[key] = (int(timestamp), float(weight))


def process_dataset(config: DatasetConfig, train_start: int, test_start: int) -> DatasetSummary:
    total_rows = 0
    organic_rows = 0
    train_rows = 0
    user_activity: Counter[int] = Counter()
    pair_latest: dict[int, tuple[int, float]] = {}

    uid_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] | None = [] if config.weighted_mode == "played_ratio_last" else None
    buffered_rows = 0

    for batch in iter_record_batches(config.source_dir):
        total_rows += batch.num_rows

        uid = batch.column(batch.schema.get_field_index("uid")).to_numpy(zero_copy_only=False)
        item_id = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
        timestamp = batch.column(batch.schema.get_field_index("timestamp")).to_numpy(zero_copy_only=False)
        is_organic = batch.column(batch.schema.get_field_index("is_organic")).to_numpy(zero_copy_only=False)

        organic_mask = is_organic == 1
        organic_rows += int(organic_mask.sum())

        train_mask = organic_mask & (timestamp >= train_start) & (timestamp < test_start)
        if not train_mask.any():
            continue

        train_rows += int(train_mask.sum())
        uid_train = uid[train_mask]
        item_train = item_id[train_mask]
        ts_train = timestamp[train_mask]

        users, counts = np.unique(uid_train, return_counts=True)
        user_activity.update({int(user): int(count) for user, count in zip(users, counts, strict=True)})

        uid_parts.append(uid_train)
        item_parts.append(item_train)
        ts_parts.append(ts_train)

        if weight_parts is not None:
            played_ratio = batch.column(batch.schema.get_field_index("played_ratio_pct")).to_numpy(zero_copy_only=False)
            weight_parts.append(played_ratio[train_mask].astype(np.float32) / 100.0)

        buffered_rows += len(uid_train)
        if buffered_rows >= CHUNK_ROWS:
            flush_chunk(
                pair_latest,
                uid_parts,
                item_parts,
                ts_parts,
                weighted_mode=config.weighted_mode,
                weight_parts=weight_parts,
            )
            uid_parts.clear()
            item_parts.clear()
            ts_parts.clear()
            if weight_parts is not None:
                weight_parts.clear()
            buffered_rows = 0

    flush_chunk(
        pair_latest,
        uid_parts,
        item_parts,
        ts_parts,
        weighted_mode=config.weighted_mode,
        weight_parts=weight_parts,
    )

    if user_activity:
        activity_values = np.fromiter(user_activity.values(), dtype=np.int64)
        q70 = float(np.quantile(activity_values, 0.7))
        q95 = float(np.quantile(activity_values, 0.95))
    else:
        q70 = float("nan")
        q95 = float("nan")

    keys = np.fromiter(pair_latest.keys(), dtype=np.uint64, count=len(pair_latest))
    weights = np.fromiter((value[1] for value in pair_latest.values()), dtype=np.float32, count=len(pair_latest))
    uids = (keys >> np.uint64(32)).astype(np.uint32)
    item_ids = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    order = np.lexsort((item_ids, uids))

    sign_df = pd.DataFrame(
        {
            "uid": uids[order],
            "item_id": item_ids[order],
            f"interaction_sign": np.full(len(order), config.sign_value, dtype=np.int8),
        }
    )
    weighted_df = pd.DataFrame(
        {
            "uid": uids[order],
            "item_id": item_ids[order],
            "interaction_weighted": weights[order],
        }
    )

    sign_path = INTERACTION_DIR / config.name / f"user_item_sign_{config.name}.csv"
    weighted_path = INTERACTION_DIR / config.name / f"user_item_weighted_{config.name}.csv"
    sign_df.to_csv(sign_path)
    weighted_df.to_csv(weighted_path)

    unique_items_train = int(np.unique(item_ids).shape[0]) if len(item_ids) else 0

    return DatasetSummary(
        dataset=config.name,
        total_rows=total_rows,
        organic_rows=organic_rows,
        train_rows=train_rows,
        unique_users_train=len(user_activity),
        unique_items_train=unique_items_train,
        q70_user_events=q70,
        q95_user_events=q95,
        final_pairs=len(pair_latest),
        sign_rule=str(config.sign_value),
        weighted_rule="0.1" if config.weighted_mode == "constant_0.1" else "played_ratio_pct / 100 from last event",
    )


def main() -> None:
    configs = [
        DatasetConfig(
            name="dislikes",
            source_dir=INTERACTION_DIR / "dislikes",
            sign_value=-1,
            weighted_mode="constant_0.1",
        ),
        DatasetConfig(
            name="listens",
            source_dir=INTERACTION_DIR / "listens",
            sign_value=1,
            weighted_mode="played_ratio_last",
        ),
    ]

    reference_dirs = [RAW_LIKES_DIR, *(config.source_dir for config in configs)]
    reference_max_time = max(max_organic_timestamp(path) for path in reference_dirs)
    test_start = reference_max_time - WEEK_TICKS
    train_start = test_start - YEAR_TICKS

    print(f"reference_max_time={reference_max_time}")
    print(f"train_start={train_start}, test_start={test_start}")

    summaries: list[DatasetSummary] = []
    for config in configs:
        print(f"Processing {config.name} ...")
        summary = process_dataset(config, train_start=train_start, test_start=test_start)
        summaries.append(summary)
        print(summary)

    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    summary_path = INTERACTION_DIR / "interaction_eda_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
