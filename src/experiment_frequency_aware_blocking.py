r"""Experiment with true frequency-aware blocking-key cutoffs.

Run from the repository root:
    .\.venv\Scripts\python.exe src/experiment_frequency_aware_blocking.py
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

try:
    from src.blocking import (
        BlockingConfig,
        BlockingStats,
        _informative_tokens,
        normalize_business_address,
        normalize_business_name,
        normalize_country,
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
        _informative_tokens,
        normalize_business_address,
        normalize_business_name,
        normalize_country,
    )
    from evaluate_blocking import (
        _ground_truth,
        _load_benchmark_targets,
        _read_source1,
        _truth_pairs_by_source1,
    )


CUTOFFS = (10, 25, 50, 100, 250, 500)
MAX_CANDIDATES_PER_SOURCE1 = 1_000
KEY_TYPES = ("exact_name", "name_token", "address_token")


class FrequencyAwareIndex:
    """Index only keys whose complete benchmark frequency is within the cutoff."""

    def __init__(self, config: BlockingConfig, cutoff: int) -> None:
        self.config = config
        self.cutoff = cutoff
        self.postings: dict[tuple[str, str, str], list[tuple[str, str]]] = {}

    def add(
        self,
        source: str,
        entity_id: object,
        name: object,
        address: object,
        country: object,
    ) -> None:
        country_key = normalize_country(country)
        if entity_id is None or pd.isna(entity_id) or not country_key:
            return
        entity_key = str(entity_id)
        normalized_name = normalize_business_name(name)
        normalized_address = normalize_business_address(address)
        keys = [("exact_name", country_key, normalized_name)]
        keys.extend(
            ("name_token", country_key, token)
            for token in _informative_tokens(normalized_name, self.config.stop_tokens)
        )
        keys.extend(
            ("address_token", country_key, token)
            for token in _informative_tokens(normalized_address, self.config.stop_tokens)
        )
        for key_type, key_country, key_value in keys:
            if not key_value:
                continue
            self.postings.setdefault((key_type, key_country, key_value), []).append(
                (entity_key, source)
            )

    def finalize(self, frequencies: Counter[tuple[str, str, str]]) -> None:
        """Remove frequent keys after full frequencies have been calculated."""
        self.postings = {
            key: postings
            for key, postings in self.postings.items()
            if frequencies[key] <= self.cutoff
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output/frequency_aware_blocking_results.txt"))
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


def _frame_records(frame: pd.DataFrame) -> Iterator[tuple[object, object, object, object]]:
    columns = ["entity_id", "business_name_clean", "business_address_clean", "country"]
    indices = [frame.columns.get_loc(column) for column in columns]
    for row in frame.itertuples(index=False, name=None):
        yield tuple(row[index] for index in indices)


def _key_values(
    name: object,
    address: object,
    country: object,
    config: BlockingConfig,
) -> Iterator[tuple[str, str, str]]:
    country_key = normalize_country(country)
    normalized_name = normalize_business_name(name)
    normalized_address = normalize_business_address(address)
    if normalized_name and country_key:
        yield ("exact_name", country_key, normalized_name)
    if country_key:
        for token in _informative_tokens(normalized_name, config.stop_tokens):
            yield ("name_token", country_key, token)
        for token in _informative_tokens(normalized_address, config.stop_tokens):
            yield ("address_token", country_key, token)


def _count_key_frequencies(
    target_frames: list[tuple[str, pd.DataFrame]],
    config: BlockingConfig,
) -> Counter[tuple[str, str, str]]:
    print("Counting complete benchmark key frequencies...", flush=True)
    frequencies: Counter[tuple[str, str, str]] = Counter()
    for _, frame in target_frames:
        for entity_id, name, address, country in _frame_records(frame):
            del entity_id
            frequencies.update(_key_values(name, address, country, config))
    return frequencies


def _build_index(
    target_frames: list[tuple[str, pd.DataFrame]],
    config: BlockingConfig,
    cutoff: int,
    frequencies: Counter[tuple[str, str, str]],
) -> FrequencyAwareIndex:
    index = FrequencyAwareIndex(config, cutoff)
    for source, frame in target_frames:
        for entity_id, name, address, country in _frame_records(frame):
            index.add(source, entity_id, name, address, country)
    index.finalize(frequencies)
    return index


def _iter_candidates(
    record: tuple[object, object, object, object],
    index: FrequencyAwareIndex,
    stats: BlockingStats,
) -> Iterator[tuple[str, str, str]]:
    entity_id, name, address, country = record
    if entity_id is None or pd.isna(entity_id):
        stats.observe_record(0)
        return
    source1_id = str(entity_id)
    seen: set[tuple[str, str, str]] = set()
    retained = 0
    for key in _key_values(name, address, country, index.config):
        for target_id, target_source in index.postings.get(key, ()):
            stats.candidates_generated += 1
            candidate = (source1_id, target_id, target_source)
            if candidate in seen:
                continue
            if retained >= index.config.max_candidates_per_source1_record:
                stats.candidates_skipped_limit += 1
                continue
            seen.add(candidate)
            retained += 1
            yield candidate
    stats.observe_record(retained)


def _evaluate_cutoff(
    cutoff: int,
    source1: pd.DataFrame,
    truth: dict[str, set[str]],
    truth_pairs_by_source1: dict[str, set[tuple[str, str]]],
    target_frames: list[tuple[str, pd.DataFrame]],
    eligible_pairs: set[tuple[str, str]],
    frequencies: Counter[tuple[str, str, str]],
    batch_size: int,
) -> dict[str, Any]:
    print(f"Building frequency-aware index for cutoff {cutoff}...", flush=True)
    config = BlockingConfig(
        max_posting_list_size=MAX_CANDIDATES_PER_SOURCE1,
        max_candidates_per_source1_record=MAX_CANDIDATES_PER_SOURCE1,
        chunk_size=batch_size,
        input_format="parquet",
    )
    index = _build_index(target_frames, config, cutoff, frequencies)
    stats = BlockingStats()
    candidates: list[tuple[str, str, str]] = []
    print(f"Generating candidates for cutoff {cutoff}...", flush=True)
    for record in _frame_records(source1):
        candidates.extend(_iter_candidates(record, index, stats))

    # blocking.py emits (source1_id, target_entity_id, target_source);
    # evaluator truth pairs use (target_source, target_entity_id).
    candidate_triples = {
        (source1_id, target_source, target_id)
        for source1_id, target_id, target_source in candidates
    }
    true_target_ids = {
        target_id for target_ids in truth.values() for target_id in target_ids
    }
    matching_ground_truth_candidate_ids = {
        target_id
        for _, target_source, target_id in candidate_triples
        if target_id in true_target_ids
    }
    source1_ids = set(source1["entity_id"].astype(str))
    eligible_by_source1 = {
        source1_id: truth_pairs_by_source1[source1_id].intersection(eligible_pairs)
        for source1_id in source1_ids
    }
    candidates_by_source1: dict[str, set[tuple[str, str]]] = {}
    for candidate_source1_id, source, target_id in candidate_triples:
        candidates_by_source1.setdefault(candidate_source1_id, set()).add(
            (source, target_id)
        )
    retrieved_by_source1 = {
        source1_id: eligible_by_source1[source1_id].intersection(
            candidates_by_source1.get(source1_id, set())
        )
        for source1_id in eligible_by_source1
    }
    eligible_count = len(eligible_pairs)
    retrieved_count = sum(len(pairs) for pairs in retrieved_by_source1.values())
    eligible_records = sum(bool(pairs) for pairs in eligible_by_source1.values())
    retrieved_records = sum(bool(pairs) for pairs in retrieved_by_source1.values())
    candidate_counts = Counter(source1_id for source1_id, _, _ in candidates)
    limit_hit_records = sum(
        count >= MAX_CANDIDATES_PER_SOURCE1 for count in candidate_counts.values()
    )
    example_candidate_id = next(
        (target_id for _, _, target_id in candidate_triples), "(none)"
    )
    example_ground_truth_id = next(iter(true_target_ids), "(none)")
    print(
        f"Cutoff {cutoff} diagnostic: candidates={len(candidates)}, "
        f"candidate IDs matching ground truth={len(matching_ground_truth_candidate_ids)}, "
        f"example candidate ID={example_candidate_id}, "
        f"example ground-truth target ID={example_ground_truth_id}",
        flush=True,
    )
    return {
        "cutoff": cutoff,
        "match_level_blocking_recall": _metric(retrieved_count, eligible_count),
        "record_level_coverage": _metric(retrieved_records, eligible_records),
        "average_candidates_per_source1": _metric(len(candidates), len(source1)),
        "maximum_candidates_per_source1": max(candidate_counts.values(), default=0),
        "candidates_generated": stats.candidates_generated,
        "candidates_skipped": stats.candidates_skipped_limit,
        "source1_records_hitting_candidate_limit": limit_hit_records,
        "eligible_true_matches": eligible_count,
        "retrieved_true_matches": retrieved_count,
        "true_matches_missed": eligible_count - retrieved_count,
        "candidate_count": len(candidates),
        "candidate_ids_matching_ground_truth": len(matching_ground_truth_candidate_ids),
        "example_candidate_id": example_candidate_id,
        "example_ground_truth_target_id": example_ground_truth_id,
    }


def main() -> None:
    args = parse_args()
    _validate_positive("source1_limit", args.source1_limit)
    _validate_positive("non_matching_per_source", args.non_matching_per_source)
    _validate_positive("batch_size", args.batch_size)

    source1, truth, truth_pairs_by_source1, target_frames, eligible_pairs = _prepare_benchmark(args)
    config = BlockingConfig(input_format="parquet")
    frequencies = _count_key_frequencies(target_frames, config)
    print("Running frequency-aware cutoff experiments...", flush=True)
    results = [
        _evaluate_cutoff(
            cutoff,
            source1,
            truth,
            truth_pairs_by_source1,
            target_frames,
            eligible_pairs,
            frequencies,
            args.batch_size,
        )
        for cutoff in CUTOFFS
    ]

    lines = [
        "Frequency-aware blocking results",
        "=================================",
        f"Source1 limit: {args.source1_limit}",
        f"Non-matching targets per source: {args.non_matching_per_source}",
        f"Seed: {args.seed}",
        "",
        "cutoff | recall | coverage | avg_candidates | max_candidates | generated | skipped | limit_hit_records | eligible | retrieved | missed",
    ]
    for result in results:
        lines.append(
            f"{result['cutoff']:>6} | "
            f"{result['match_level_blocking_recall']:.6f} | "
            f"{result['record_level_coverage']:.6f} | "
            f"{result['average_candidates_per_source1']:.3f} | "
            f"{result['maximum_candidates_per_source1']:>14} | "
            f"{result['candidates_generated']:>9} | "
            f"{result['candidates_skipped']:>7} | "
            f"{result['source1_records_hitting_candidate_limit']:>16} | "
            f"{result['eligible_true_matches']:>8} | "
            f"{result['retrieved_true_matches']:>9} | "
            f"{result['true_matches_missed']:>6}"
        )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved results to {output}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
