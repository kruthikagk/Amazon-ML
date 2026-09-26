"""Analyze blocking-key frequencies in cleaned target Parquet files.

Run from the repository root:
    python src/analyze_blocking_keys.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

try:
    from src.blocking import (
        BlockingConfig,
        _informative_tokens,
        normalize_business_address,
        normalize_business_name,
        normalize_country,
    )
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    from blocking import (
        BlockingConfig,
        _informative_tokens,
        normalize_business_address,
        normalize_business_name,
        normalize_country,
    )


COLUMNS = [
    "business_name_clean",
    "business_address_clean",
    "country",
]
THRESHOLDS = (10, 50, 100, 500, 1_000)


@dataclass
class KeyFrequency:
    exact_names: Counter[tuple[str, str]]
    name_tokens: Counter[tuple[str, str]]
    address_tokens: Counter[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output/blocking_key_frequency_report.txt"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def _new_frequency() -> KeyFrequency:
    return KeyFrequency(Counter(), Counter(), Counter())


def _analyze_file(path: Path, batch_size: int, config: BlockingConfig) -> KeyFrequency:
    """Count keys batch by batch so the full Parquet file stays out of RAM."""
    frequencies = _new_frequency()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=COLUMNS):
        columns = batch.to_pydict()
        for name_value, address_value, country_value in zip(
            columns["business_name_clean"],
            columns["business_address_clean"],
            columns["country"],
        ):
            country = normalize_country(country_value)
            if not country:
                continue

            name = normalize_business_name(name_value)
            if name:
                frequencies.exact_names[(country, name)] += 1
            for token in _informative_tokens(name, config.stop_tokens):
                frequencies.name_tokens[(country, token)] += 1

            address = normalize_business_address(address_value)
            for token in _informative_tokens(address, config.stop_tokens):
                frequencies.address_tokens[(country, token)] += 1
    return frequencies


def _merge_frequencies(total: KeyFrequency, current: KeyFrequency) -> None:
    total.exact_names.update(current.exact_names)
    total.name_tokens.update(current.name_tokens)
    total.address_tokens.update(current.address_tokens)


def _format_key(key: tuple[str, str]) -> str:
    country, value = key
    return f"{country} | {value}"


def _write_frequency_section(
    lines: list[str],
    title: str,
    frequencies: Counter[tuple[str, str]],
) -> None:
    lines.append(title)
    lines.append("-" * len(title))
    lines.append("Top 20 keys:")
    if frequencies:
        for rank, (key, count) in enumerate(frequencies.most_common(20), start=1):
            lines.append(f"{rank:>2}. {count:>10}  {_format_key(key)}")
    else:
        lines.append("  (none)")
    lines.append("Key counts above thresholds:")
    for threshold in THRESHOLDS:
        lines.append(f"  > {threshold:>4}: {sum(count > threshold for count in frequencies.values())}")
    lines.append(f"Maximum frequency: {max(frequencies.values(), default=0)}")
    lines.append(f"Distinct keys: {len(frequencies)}")
    lines.append("")


def _report_for_source(source_name: str, frequencies: KeyFrequency) -> list[str]:
    lines = [source_name, "=" * len(source_name), ""]
    _write_frequency_section(lines, "Exact normalized business name", frequencies.exact_names)
    _write_frequency_section(lines, "Informative business-name token", frequencies.name_tokens)
    _write_frequency_section(lines, "Informative business-address token", frequencies.address_tokens)
    return lines


def analyze(data_dir: Path, batch_size: int) -> str:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = BlockingConfig(input_format="parquet")
    lines = [
        "Blocking key frequency report",
        "==============================",
        "",
        f"Batch size: {batch_size}",
        "Keys are country-qualified, matching the blocking index.",
        "Empty country/name/token keys are excluded, matching blocking.py.",
        "",
    ]
    for source_name, filename in (
        ("Source2", "train_source2_clean.parquet"),
        ("Source3", "train_source3_clean.parquet"),
    ):
        print(f"Analyzing {source_name}...", flush=True)
        frequencies = _analyze_file(data_dir / filename, batch_size, config)
        lines.extend(_report_for_source(source_name, frequencies))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = analyze(args.data_dir, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
