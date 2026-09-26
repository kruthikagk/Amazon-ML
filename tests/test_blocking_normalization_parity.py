from __future__ import annotations

import re
import unicodedata
import unittest

from src.blocking import _informative_tokens, normalize_business_name, normalize_business_address


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def reference_normalize(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return " ".join(text.split())


def reference_tokens(value: str, stop_tokens: frozenset[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(value):
        if token in stop_tokens or len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


class BlockingNormalizationParityTests(unittest.TestCase):
    def test_normalization_parity_for_text_edge_cases(self) -> None:
        examples = (
            ("ordinary text", "ordinary text"),
            ("  North\tShore\nClinic  ", "north shore clinic"),
            ("Acme_Co., LLC", "acme co llc"),
            ("A.B/C\\D!", "a b c d"),
            ("Cafe\u0301 \u2014 \u6771\u4eac\u3001\u4f01\u696d\uff01", "café 東京 企業"),
            ("ＡＣＭＥ　Ｐａｒｔｎｅｒｓ", "acme partners"),
            (None, ""),
        )
        for raw, expected in examples:
            with self.subTest(raw=raw):
                self.assertEqual(reference_normalize(raw), expected)
                self.assertEqual(normalize_business_name(raw), expected)
                self.assertEqual(normalize_business_name(raw), reference_normalize(raw))
                self.assertEqual(normalize_business_address(raw), expected)
                self.assertEqual(normalize_business_address(raw), reference_normalize(raw))

    def test_token_parity_for_order_repeats_stops_and_short_tokens(self) -> None:
        value = "a acme acme co company company rd road x y 12 main_road"
        stop_tokens = frozenset({"company", "rd", "road"})
        expected = ("acme", "co", "12", "main")
        self.assertEqual(reference_tokens(value, stop_tokens), expected)
        self.assertEqual(_informative_tokens(value, stop_tokens), expected)
        self.assertEqual(
            _informative_tokens(value, stop_tokens),
            reference_tokens(value, stop_tokens),
        )

    def test_token_matching_and_first_occurrence_order(self) -> None:
        value = "éclair cafe\u2019s cafe éclair"
        stop_tokens = frozenset({"s"})
        expected = ("éclair", "cafe")
        self.assertEqual(reference_tokens(value, stop_tokens), expected)
        self.assertEqual(_informative_tokens(value, stop_tokens), expected)


if __name__ == "__main__":
    unittest.main()
