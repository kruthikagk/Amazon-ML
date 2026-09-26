"""Compare frequency-aware posting-list thresholds on the blocking benchmark.

Run from the repository root:
    python src/experiment_blocking_thresholds.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.blocking import (
        BlockingConfig,
        BlockingStats,
        build_index_from_frames,
        generate_candidates_from_frames,
    )
    from src.evaluate_blocking import (
        _ground_truth,
        _load_benchmark_targets,
        _read_source1,
        _truth_pairs_by_source1,
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
    from evaluate_blocking import (
        _ground_truth,
        _load_benchmark_targets,
        _read_source1,
        _truth_pairs_by_source1,
    )


THRESHOLDS = (100, 250, 500, 1_000)
MAX_CANDIDATES_PER_SOURCE1 = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source1-limit", type=int, default=2_000)
    parser.add_argument("--non-matching-per-source", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _metric(value: int, denominator: int) -> float:
    return value / denominator if denominator else 0.0


def _prepare_benchmark(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    dict[str, set[str]],
    dict[str, set[tuple[str, str]]],
    list[tuple[str, pd.DataFrame]],
    set[tuple[str, str]],
]:
    data_dir = args.data_dir
    print("Loading Source1 sample...", flush=True)
    source1 = _read_source1(
        data_dir / "train_source1_clean.parquet", args.source1_limit
    )
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
    return source1, truth, truth_pairs_by_source1, target_frames, eligible_pairs


def _evaluate_threshold(
    threshold: int,
    source1: pd.DataFrame,
    truth: dict[str, set[str]],
    truth_pairs_by_source1: dict[str, set[tuple[str, str]]],
    target_frames: list[tuple[str, pd.DataFrame]],
    eligible_pairs: set[tuple[str, str]],
    batch_size: int,
) -> dict[str, Any]:
    print(f"Building blocking index for threshold {threshold}...", flush=True)
    config = BlockingConfig(
        max_posting_list_size=threshold,
        max_candidates_per_source1_record=MAX_CANDIDATES_PER_SOURCE1,
        chunk_size=batch_size,
        input_format="parquet",
    )
    index = build_index_from_frames(target_frames, config)
    stats = BlockingStats()
    print(f"Generating candidates for threshold {threshold}...", flush=True)
    candidates = generate_candidates_from_frames(source1, index, stats)
    candidate_triples = set(
        zip(
            candidates["source1_entity_id"],
            candidates["target_source"],
            candidates["target_entity_id"],
            strict=True,
        )
    )

    eligible_by_source1 = {
        source1_id: truth_pairs_by_source1[source1_id].intersection(eligible_pairs)
        for source1_id in set(source1["entity_id"].astype(str))
    }
    retrieved_by_source1 = {
        source1_id: {
            (source, target_id)
            for candidate_source1_id, source, target_id in candidate_triples
            if candidate_source1_id == source1_id
            and (source, target_id) in eligible_by_source1[source1_id]
        }
        for source1_id in eligible_by_source1
    }
    retrieved_matches = sum(len(pairs) for pairs in retrieved_by_source1.values())
    eligible_match_count = len(eligible_pairs)
    records_with_eligible = sum(bool(pairs) for pairs in eligible_by_source1.values())
    records_with_retrieved = sum(bool(pairs) for pairs in retrieved_by_source1.values())
    candidate_counts = candidates.groupby("source1_entity_id").size()

    return {
        "threshold": threshold,
        "match_level_blocking_recall": _metric(retrieved_matches, eligible_match_count),
        "record_level_coverage": _metric(records_with_retrieved, records_with_eligible),
        "average_candidates_per_source1": _metric(len(candidates), len(source1)),
        "maximum_candidates_per_source1": int(candidate_counts.max()) if len(candidate_counts) else 0,
        "candidates_generated": stats.candidates_generated,
        "candidates_skipped_by_limit": stats.candidates_skipped_limit,
    }


def main() -> None:
    args = parse_args()
    _validate_positive("source1_limit", args.source1_limit)
    _validate_positive("non_matching_per_source", args.non_matching_per_source)
    _validate_positive("batch_size", args.batch_size)

    source1, truth, truth_pairs_by_source1, target_frames, eligible_pairs = _prepare_benchmark(args)
    print("Running threshold experiments...", flush=True)
    results = [
        _evaluate_threshold(
            threshold,
            source1,
            truth,
            truth_pairs_by_source1,
            target_frames,
            eligible_pairs,
            args.batch_size,
        )
        for threshold in THRESHOLDS
    ]

    print("\nBlocking threshold results")
    print("threshold | recall | coverage | avg_candidates | max_candidates | generated | skipped_limit")
    for result in results:
        print(
            f"{result['threshold']:>9} | "
            f"{result['match_level_blocking_recall']:.6f} | "
            f"{result['record_level_coverage']:.6f} | "
            f"{result['average_candidates_per_source1']:.3f} | "
            f"{result['maximum_candidates_per_source1']:>15} | "
            f"{result['candidates_generated']:>9} | "
            f"{result['candidates_skipped_by_limit']:>13}"
        )


if __name__ == "__main__":
    main()
