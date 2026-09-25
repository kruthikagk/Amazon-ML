"""Memory-conscious blocking for the cleaned business entity datasets."""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd


logger = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["source1_entity_id", "target_entity_id", "target_source"]
REQUIRED_COLUMNS = {"entity_id", "country"}
RAW_NAME_COLUMN = "business_name"
RAW_ADDRESS_COLUMN = "business_address"
CLEAN_NAME_COLUMN = "business_name_clean"
CLEAN_ADDRESS_COLUMN = "business_address_clean"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _normalize_text(value: object) -> str:
    """Normalize one value; null-like values become an empty string."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return " ".join(text.split())


def normalize_business_name(value: object) -> str:
    return _normalize_text(value)


def normalize_business_address(value: object) -> str:
    return _normalize_text(value)


def normalize_country(value: object) -> str:
    return _normalize_text(value)


def _informative_tokens(value: str, stop_tokens: frozenset[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(value):
        if token in stop_tokens or len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True)
class BlockingConfig:
    """Limits and input settings for safe, batch-oriented blocking."""

    max_posting_list_size: int = 1_000
    max_candidates_per_source1_record: int = 1_000
    chunk_size: int = 100_000
    input_format: str = "parquet"
    stop_tokens: frozenset[str] = frozenset(
        {
            "and", "the", "company", "corporation", "inc", "llc", "limited",
            "ltd", "road", "rd", "street", "st", "avenue", "ave",
        }
    )

    def __post_init__(self) -> None:
        if self.max_posting_list_size < 1:
            raise ValueError("max_posting_list_size must be positive")
        if self.max_candidates_per_source1_record < 1:
            raise ValueError("max_candidates_per_source1_record must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.input_format not in {"parquet", "tsv"}:
            raise ValueError("input_format must be 'parquet' or 'tsv'")


@dataclass(frozen=True)
class TargetRecord:
    entity_id: str
    target_source: str


@dataclass
class BlockingStats:
    source1_records: int = 0
    candidates_generated: int = 0
    candidates_retained: int = 0
    candidates_skipped_limit: int = 0
    max_candidates_for_record: int = 0

    def observe_record(self, retained: int) -> None:
        self.source1_records += 1
        self.candidates_retained += retained
        self.max_candidates_for_record = max(self.max_candidates_for_record, retained)

    @property
    def average_candidates_per_record(self) -> float:
        return self.candidates_retained / self.source1_records if self.source1_records else 0.0


class BoundedInvertedIndex:
    """Keep only a bounded prefix for each key, including frequent keys."""

    def __init__(self, max_posting_list_size: int) -> None:
        self.max_posting_list_size = max_posting_list_size
        self._postings: dict[tuple[str, str], list[TargetRecord]] = {}

    def add(self, key: tuple[str, str], record: TargetRecord) -> None:
        if not key[0] or not key[1]:
            return
        postings = self._postings.setdefault(key, [])
        if len(postings) < self.max_posting_list_size:
            postings.append(record)

    def get(self, key: tuple[str, str]) -> Sequence[TargetRecord]:
        """Return the posting list directly, avoiding a per-lookup copy."""
        return self._postings.get(key, ())

    @property
    def number_of_keys(self) -> int:
        return len(self._postings)


class BlockingIndex:
    """Country-aware exact-name, name-token, and address-token indexes."""

    def __init__(self, config: BlockingConfig) -> None:
        self.config = config
        self.exact_names = BoundedInvertedIndex(config.max_posting_list_size)
        self.name_tokens = BoundedInvertedIndex(config.max_posting_list_size)
        self.address_tokens = BoundedInvertedIndex(config.max_posting_list_size)

    def add_values(
        self,
        entity_id: object,
        business_name: object,
        business_address: object,
        country_value: object,
        target_source: str,
    ) -> None:
        country = normalize_country(country_value)
        if entity_id is None or pd.isna(entity_id) or not country:
            return
        name = normalize_business_name(business_name)
        address = normalize_business_address(business_address)
        target = TargetRecord(str(entity_id), target_source)
        self.exact_names.add((country, name), target)
        for token in _informative_tokens(name, self.config.stop_tokens):
            self.name_tokens.add((country, token), target)
        for token in _informative_tokens(address, self.config.stop_tokens):
            self.address_tokens.add((country, token), target)

    def add_record(self, record: Mapping[str, object], target_source: str) -> None:
        self.add_values(
            record.get("entity_id"),
            record.get(CLEAN_NAME_COLUMN, record.get(RAW_NAME_COLUMN)),
            record.get(CLEAN_ADDRESS_COLUMN, record.get(RAW_ADDRESS_COLUMN)),
            record.get("country"),
            target_source,
        )


def _validate_frame_columns(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not (
        {CLEAN_NAME_COLUMN, CLEAN_ADDRESS_COLUMN}.issubset(frame.columns)
        or {RAW_NAME_COLUMN, RAW_ADDRESS_COLUMN}.issubset(frame.columns)
    ):
        raise ValueError("Frame needs cleaned name/address columns or raw fallbacks")


def _frame_values(frame: pd.DataFrame) -> Iterator[tuple[object, object, object, object]]:
    """Yield only needed columns, avoiding a dictionary per input row."""
    _validate_frame_columns(frame)
    name_column = CLEAN_NAME_COLUMN if CLEAN_NAME_COLUMN in frame else RAW_NAME_COLUMN
    address_column = CLEAN_ADDRESS_COLUMN if CLEAN_ADDRESS_COLUMN in frame else RAW_ADDRESS_COLUMN
    columns = ["entity_id", name_column, address_column, "country"]
    indices = [frame.columns.get_loc(column) for column in columns]
    for row in frame.itertuples(index=False, name=None):
        yield tuple(row[index] for index in indices)


def build_index_from_frames(
    target_frames: Iterable[tuple[str, pd.DataFrame]],
    config: BlockingConfig | None = None,
) -> BlockingIndex:
    config = config or BlockingConfig(input_format="tsv")
    index = BlockingIndex(config)
    for target_source, frame in target_frames:
        for entity_id, name, address, country in _frame_values(frame):
            index.add_values(entity_id, name, address, country, target_source)
    return index


def _read_tsv(path: str | Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    # TSV fallback remains chunked; neither format is read as one full DataFrame.
    yield from pd.read_csv(path, sep="\t", dtype="string", chunksize=chunk_size)


def _require_pyarrow() -> object:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "PyArrow is required for Parquet input; install it in the active environment."
        ) from error
    return parquet


def _read_parquet_rows(
    path: str | Path,
    batch_size: int,
    max_rows: int | None = None,
) -> Iterator[tuple[object, object, object, object]]:
    """Yield needed Parquet values batch by batch, never the full dataset.

    ``iter_batches`` bounds the temporary Arrow batch. Only the cleaned fields,
    country, and entity ID are requested, reducing both I/O and batch memory.
    """
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative or None")
    parquet = _require_pyarrow()
    parquet_file = parquet.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = REQUIRED_COLUMNS.union({CLEAN_NAME_COLUMN, CLEAN_ADDRESS_COLUMN}) - available
    if missing:
        raise ValueError(f"Parquet file is missing required cleaned columns: {sorted(missing)}")
    columns = ["entity_id", CLEAN_NAME_COLUMN, CLEAN_ADDRESS_COLUMN, "country"]
    read_rows = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        batch_rows = batch.to_pydict()
        row_count = len(batch_rows["entity_id"])
        if max_rows is not None:
            row_count = min(row_count, max_rows - read_rows)
        for row_index in range(row_count):
            yield tuple(batch_rows[column][row_index] for column in columns)
        read_rows += row_count
        if max_rows is not None and read_rows >= max_rows:
            break


def _read_rows(
    path: str | Path,
    config: BlockingConfig,
    max_rows: int | None = None,
) -> Iterator[tuple[object, object, object, object]]:
    if config.input_format == "parquet":
        yield from _read_parquet_rows(path, config.chunk_size, max_rows)
        return
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative or None")
    read_rows = 0
    for frame in _read_tsv(path, config.chunk_size):
        for row in _frame_values(frame):
            if max_rows is not None and read_rows >= max_rows:
                return
            yield row
            read_rows += 1


def build_index_from_paths(
    source2_path: str | Path,
    source3_path: str | Path,
    config: BlockingConfig | None = None,
    max_target_rows: int | None = None,
) -> BlockingIndex:
    """Build target indexes incrementally; max_target_rows is for benchmarks."""
    config = config or BlockingConfig()
    index = BlockingIndex(config)
    for target_source, path in (("source2", source2_path), ("source3", source3_path)):
        for entity_id, name, address, country in _read_rows(path, config, max_target_rows):
            index.add_values(entity_id, name, address, country, target_source)
    return index


def _record_values(record: Mapping[str, object]) -> tuple[object, str, str, str] | None:
    entity_id = record.get("entity_id")
    if entity_id is None or pd.isna(entity_id):
        return None
    return (
        entity_id,
        normalize_country(record.get("country")),
        normalize_business_name(record.get(CLEAN_NAME_COLUMN, record.get(RAW_NAME_COLUMN))),
        normalize_business_address(record.get(CLEAN_ADDRESS_COLUMN, record.get(RAW_ADDRESS_COLUMN))),
    )


def iter_candidates_for_record(
    record: Mapping[str, object],
    index: BlockingIndex,
    stats: BlockingStats | None = None,
) -> Iterator[tuple[str, str, str]]:
    """Yield bounded, deduplicated candidates for one Source 1 record."""
    values = _record_values(record)
    if values is None:
        if stats is not None:
            stats.observe_record(0)
        return
    entity_id, country, name, address = values
    seen: set[tuple[str, str, str]] = set()
    retained = 0
    keys_and_indexes = [((country, name), index.exact_names)]
    keys_and_indexes.extend(
        ((country, token), index.name_tokens)
        for token in _informative_tokens(name, index.config.stop_tokens)
    )
    keys_and_indexes.extend(
        ((country, token), index.address_tokens)
        for token in _informative_tokens(address, index.config.stop_tokens)
    )
    for key, postings_index in keys_and_indexes:
        for target in postings_index.get(key):
            if stats is not None:
                stats.candidates_generated += 1
            candidate = (str(entity_id), target.entity_id, target.target_source)
            if candidate in seen:
                continue
            if retained >= index.config.max_candidates_per_source1_record:
                if stats is not None:
                    stats.candidates_skipped_limit += 1
                continue
            seen.add(candidate)
            retained += 1
            yield candidate
    if stats is not None:
        stats.observe_record(retained)


def candidates_for_record(
    record: Mapping[str, object], index: BlockingIndex
) -> set[tuple[str, str, str]]:
    return set(iter_candidates_for_record(record, index))


def generate_candidates_from_frames(
    source1_frame: pd.DataFrame,
    index: BlockingIndex,
    stats: BlockingStats | None = None,
) -> pd.DataFrame:
    """Small in-memory helper retained for the toy test."""
    rows: list[tuple[str, str, str]] = []
    for entity_id, name, address, country in _frame_values(source1_frame):
        record = {
            "entity_id": entity_id,
            CLEAN_NAME_COLUMN: name,
            CLEAN_ADDRESS_COLUMN: address,
            "country": country,
        }
        rows.extend(iter_candidates_for_record(record, index, stats))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).drop_duplicates(ignore_index=True)


def _log_stats(stats: BlockingStats, config: BlockingConfig) -> None:
    logger.info(
        "Blocking complete: source1_records=%d candidates_generated=%d "
        "candidates_retained=%d candidates_skipped_limit=%d "
        "average_candidates_per_record=%.2f max_candidates_for_record=%d "
        "max_candidates_per_source1_record=%d",
        stats.source1_records, stats.candidates_generated, stats.candidates_retained,
        stats.candidates_skipped_limit, stats.average_candidates_per_record,
        stats.max_candidates_for_record, config.max_candidates_per_source1_record,
    )


def generate_candidates(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    output_path: str | Path,
    config: BlockingConfig | None = None,
    stats: BlockingStats | None = None,
    max_source1_rows: int | None = None,
    max_target_rows: int | None = None,
) -> BlockingStats:
    """Stream Parquet/TSV inputs and write only the three candidate columns.

    Candidate rows are written immediately. The per-record cap protects both
    RAM and output volume when a blocking key is unusually common.
    """
    config = config or BlockingConfig()
    stats = stats or BlockingStats()
    index = build_index_from_paths(
        source2_path, source3_path, config, max_target_rows=max_target_rows
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for entity_id, name, address, country in _read_rows(
            source1_path, config, max_source1_rows
        ):
            record = {
                "entity_id": entity_id,
                CLEAN_NAME_COLUMN: name,
                CLEAN_ADDRESS_COLUMN: address,
                "country": country,
            }
            for candidate in iter_candidates_for_record(record, index, stats):
                writer.writerow(candidate)
    _log_stats(stats, config)
    return stats


def toy_example() -> pd.DataFrame:
    """Return a small candidate result without reading any external files."""
    source1 = pd.DataFrame(
        [
            {"entity_id": "s1-a", "business_name": "Acme Café", "business_address": "12 Main St", "country": "US"},
            {"entity_id": "s1-b", "business_name": "Bharat Foods", "business_address": "7 MG Road", "country": "India"},
        ]
    )
    source2 = pd.DataFrame(
        [{"entity_id": "s2-a", "business_name": "ACME Cafe", "business_address": "12 Main Street", "country": "US"}]
    )
    source3 = pd.DataFrame(
        [{"entity_id": "s3-b", "business_name": "Bharat Foods Pvt", "business_address": "7 MG Road", "country": "India"}]
    )
    index = build_index_from_frames((("source2", source2), ("source3", source3)))
    return generate_candidates_from_frames(source1, index)
