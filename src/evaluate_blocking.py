"""Benchmark the existing blocking implementation on an eligible target sample.

Run from the repository root:
    python src/evaluate_blocking.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

try:
    from src.blocking import (
        BlockingConfig,
        BlockingStats,
        build_index_from_frames,
        generate_candidates_from_frames,
    )
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    from blocking import (
        BlockingConfig,
        BlockingStats,
        build_index_from_frames,
        generate_candidates_from_frames,
    )


REQUIRED_TARGET_COLUMNS = [
    "entity_id",
    "business_name_clean",
    "business_address_clean",
    "country",
]
SOURCE1_COLUMNS = REQUIRED_TARGET_COLUMNS
OUTPUT_COLUMNS = ["source1_entity_id", "target_entity_id", "target_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing the cleaned Parquet files and ground truth TSV.",
    )
    parser.add_argument("--source1-limit", type=int, default=10_000)
    parser.add_argument("--non-matching-per-source", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-posting-list-size", type=int, default=1_000)
    parser.add_argument("--max-candidates-per-source1", type=int, default=1_000)
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _read_source1(path: Path, row_limit: int) -> pd.DataFrame:
    """Read only the selected Source 1 prefix needed by this benchmark."""
    parquet_file = pq.ParquetFile(path)
    batches: list[pd.DataFrame] = []
    rows_read = 0
    for batch in parquet_file.iter_batches(
        batch_size=min(row_limit, 100_000),
        columns=SOURCE1_COLUMNS,
    ):
        rows_remaining = row_limit - rows_read
        batches.append(batch.slice(0, rows_remaining).to_pandas())
        rows_read += len(batches[-1])
        if rows_read >= row_limit:
            break
    if not batches:
        return pd.DataFrame(columns=SOURCE1_COLUMNS)
    return pd.concat(batches, ignore_index=True)


def _parse_target_ids(value: object) -> set[str]:
    """Parse comma-separated ground-truth IDs while tolerating missing values."""
    if value is None or pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _ground_truth(
    ground_truth_path: Path,
    source1_ids: set[str],
) -> dict[str, set[str]]:
    frame = pd.read_csv(ground_truth_path, sep="\t", dtype="string").fillna("")
    required = {"source1_entity_id", "matched_entity_ids"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")
    selected = frame[frame["source1_entity_id"].isin(source1_ids)]
    truth: dict[str, set[str]] = {source1_id: set() for source1_id in source1_ids}
    for row in selected.itertuples(index=False):
        truth.setdefault(str(row.source1_entity_id), set()).update(
            _parse_target_ids(row.matched_entity_ids)
        )
    return truth


def _source_from_id(entity_id: str) -> str | None:
    """Use the challenge's S2-/S3- convention when it is present."""
    upper_id = entity_id.upper()
    if upper_id.startswith("S2-"):
        return "source2"
    if upper_id.startswith("S3-"):
        return "source3"
    return None


def _target_is_true(entity_id: str, source: str, true_ids: set[str]) -> bool:
    identified_source = _source_from_id(entity_id)
    return entity_id in true_ids and (
        identified_source is None or identified_source == source
    )


def _truth_pairs_by_source1(
    truth: dict[str, set[str]],
) -> dict[str, set[tuple[str, str]]]:
    """Expand each truth ID into source-qualified pairs using its ID prefix."""
    pairs_by_source1: dict[str, set[tuple[str, str]]] = {}
    for source1_id, target_ids in truth.items():
        pairs: set[tuple[str, str]] = set()
        for target_id in target_ids:
            identified_source = _source_from_id(target_id)
            sources = (
                (identified_source,)
                if identified_source is not None
                else ("source2", "source3")
            )
            pairs.update((source, target_id) for source in sources)
        pairs_by_source1[source1_id] = pairs
    return pairs_by_source1


def _reservoir_add(
    sample: list[dict[str, object]],
    record: dict[str, object],
    seen: int,
    sample_size: int,
    rng: random.Random,
) -> None:
    """Add a record to a deterministic fixed-size reservoir sample."""
    if len(sample) < sample_size:
        sample.append(record)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < sample_size:
        sample[replacement_index] = record


def _load_benchmark_targets(
    path: Path,
    source: str,
    true_ids: set[str],
    non_matching_per_source: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Scan one target Parquet and retain true rows plus sampled non-matches."""
    parquet_file = pq.ParquetFile(path)
    true_records: list[dict[str, object]] = []
    non_matching_sample: list[dict[str, object]] = []
    non_matching_seen = 0
    seen_true_ids: set[str] = set()

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=REQUIRED_TARGET_COLUMNS,
    ):
        frame = batch.to_pandas()
        for record in frame.to_dict(orient="records"):
            entity_id = str(record["entity_id"])
            if _target_is_true(entity_id, source, true_ids):
                if entity_id not in seen_true_ids:
                    true_records.append(record)
                    seen_true_ids.add(entity_id)
            else:
                non_matching_seen += 1
                _reservoir_add(
                    non_matching_sample,
                    record,
                    non_matching_seen,
                    non_matching_per_source,
                    rng,
                )

    records = true_records + non_matching_sample
    benchmark = pd.DataFrame(records, columns=REQUIRED_TARGET_COLUMNS)
    eligible_pairs = {
        (source, str(record["entity_id"])) for record in true_records
    }
    return benchmark, eligible_pairs


def _metric(value: int, denominator: int) -> float:
    return value / denominator if denominator else 0.0


def run_benchmark(args: argparse.Namespace) -> dict[str, float | int]:
    _validate_positive("source1_limit", args.source1_limit)
    _validate_positive("non_matching_per_source", args.non_matching_per_source)
    _validate_positive("batch_size", args.batch_size)

    data_dir = args.data_dir
    print("Loading Source1 sample...", flush=True)
    source1 = _read_source1(data_dir / "train_source1_clean.parquet", args.source1_limit)
    source1_ids = set(source1["entity_id"].astype(str))
    print("Loading ground truth...", flush=True)
    truth = _ground_truth(data_dir / "train_ground_truth.tsv", source1_ids)
    truth_pairs_by_source1 = _truth_pairs_by_source1(truth)
    true_ids = {target_id for targets in truth.values() for target_id in targets}

    print("Building benchmark target frames...", flush=True)
    target_frames: list[tuple[str, pd.DataFrame]] = []
    eligible_pairs: set[tuple[str, str]] = set()
    for source, filename in (
        ("source2", "train_source2_clean.parquet"),
        ("source3", "train_source3_clean.parquet"),
    ):
        print(f"Scanning {source.title()}...", flush=True)
        frame, source_pairs = _load_benchmark_targets(
            data_dir / filename,
            source,
            true_ids,
            args.non_matching_per_source,
            args.batch_size,
            random.Random(args.seed),
        )
        target_frames.append((source, frame))
        eligible_pairs.update(source_pairs)

    print("Building blocking index...", flush=True)
    config = BlockingConfig(
        max_posting_list_size=args.max_posting_list_size,
        max_candidates_per_source1_record=args.max_candidates_per_source1,
        chunk_size=args.batch_size,
        input_format="parquet",
    )
    index = build_index_from_frames(target_frames, config)
    stats = BlockingStats()
    print("Generating candidates...", flush=True)
    candidates = generate_candidates_from_frames(source1, index, stats)
    candidate_triples = set(
        zip(
            candidates["source1_entity_id"],
            candidates["target_source"],
            candidates["target_entity_id"],
            strict=True,
        )
    )

    print("Calculating metrics...", flush=True)
    eligible_by_source1 = {
        source1_id: truth_pairs_by_source1[source1_id].intersection(eligible_pairs)
        for source1_id in source1_ids
    }
    retrieved_by_source1 = {
        source1_id: {
            (source, target_id)
            for candidate_source1_id, source, target_id in candidate_triples
            if candidate_source1_id == source1_id
            and (source, target_id) in eligible_by_source1[source1_id]
        }
        for source1_id in source1_ids
    }
    retrieved_matches = sum(len(pairs) for pairs in retrieved_by_source1.values())
    records_with_eligible = sum(bool(pairs) for pairs in eligible_by_source1.values())
    records_with_retrieved = sum(bool(pairs) for pairs in retrieved_by_source1.values())
    candidate_counts = candidates.groupby("source1_entity_id").size()
    limit_hit_source1_ids = {
        str(source1_id)
        for source1_id, candidate_count in candidate_counts.items()
        if candidate_count >= args.max_candidates_per_source1
    }
    eligible_limit_hit = sum(
        len(eligible_by_source1[source1_id])
        for source1_id in limit_hit_source1_ids
    )
    retrieved_limit_hit = sum(
        len(retrieved_by_source1[source1_id])
        for source1_id in limit_hit_source1_ids
    )

    metrics: dict[str, float | int] = {
        "source1_records": len(source1_ids),
        "benchmark_source2_records": len(target_frames[0][1]),
        "benchmark_source3_records": len(target_frames[1][1]),
        "total_ground_truth_target_matches": sum(len(targets) for targets in truth.values()),
        "eligible_ground_truth_matches": len(eligible_pairs),
        "ground_truth_matches_retrieved": retrieved_matches,
        "match_level_blocking_recall": _metric(retrieved_matches, len(eligible_pairs)),
        "source1_records_with_eligible_match": records_with_eligible,
        "source1_records_with_true_target_retrieved": records_with_retrieved,
        "record_level_coverage": _metric(records_with_retrieved, records_with_eligible),
        "average_candidates_per_source1": _metric(len(candidates), len(source1_ids)),
        "maximum_candidates_per_source1": int(candidate_counts.max()) if len(candidate_counts) else 0,
        "candidates_generated_by_blocker": stats.candidates_generated,
        "candidates_skipped_by_limit": stats.candidates_skipped_limit,
        "source1_records_hitting_candidate_limit": len(limit_hit_source1_ids),
        "eligible_true_matches_in_limit_hit_records": eligible_limit_hit,
        "true_matches_retrieved_in_limit_hit_records": retrieved_limit_hit,
        "true_matches_missed_due_to_candidate_limit": eligible_limit_hit - retrieved_limit_hit,
        "limit_hit_records_match_level_recall": _metric(
            retrieved_limit_hit, eligible_limit_hit
        ),
    }
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_benchmark(args)
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
