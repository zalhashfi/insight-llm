"""insight-llm: deterministic, dependency-free experiment scaffolding.

This module exposes :func:`make_split`, a pure function that deterministically
partitions a sequence of items into train/valid/test partitions.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["make_split", "SplitResult"]

# Type alias: a split result maps each split name to its ordered list of items.
SplitResult = dict[str, list[Any]]

# Target fractions for the three standard splits (~80/10/10).
DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
SPLIT_NAMES: tuple[str, str, str] = ("train", "valid", "test")


def _stable_key(seed: Any, item: Any) -> float:
    """Return a deterministic, uniform-ish hash key for ``item`` under ``seed``.

    We hash the item's ``repr`` (so arbitrary hashable/unhashable objects work)
    together with the seed, using SHA-256. The digest is treated as a random
    score in [0, 1) for the item, which makes the assignment independent of the
    item's value distribution and of input ordering.
    """
    payload = f"{seed!r}\x00{item!r}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Take the first 8 bytes as a big-endian unsigned int -> score in [0, 1).
    value = int.from_bytes(digest[:8], "big")
    return value / float(1 << 64)


def _split_counts(n: int, sizes: Sequence[float]) -> list[int]:
    """Convert fractional ``sizes`` into integer counts summing exactly to ``n``.

    Uses largest-remainder rounding so the counts always sum to ``n``.
    """
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("split sizes must sum to a positive value")

    raw = [n * (s / total) for s in sizes]
    counts = [int(math.floor(r)) for r in raw]
    remainder = n - sum(counts)

    # Distribute leftover units to the largest fractional parts (stable order).
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in range(remainder):
        counts[order[i % len(counts)]] += 1
    return counts


def _as_list(items: Iterable[Any]) -> list[Any]:
    if isinstance(items, (str, bytes)):
        raise TypeError("items must be a sequence of items, not a string/bytes")
    return list(items)


def make_split(
    items: Iterable[Any],
    seed: Any,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    kind_field: str = "kind",
) -> SplitResult:
    """Deterministically partition ``items`` into train/valid/test.

    The function is pure: it performs no I/O, uses no randomness beyond ``seed``,
    and never mutates its inputs. Given the same ``items`` and ``seed`` it always
    returns identical output, regardless of the order of ``items``.

    Parameters
    ----------
    items:
        Iterable of items to partition. Items may be any objects.
    seed:
        Any hashable/repr-able value; it fully determines the assignment.
    ratios:
        Fractions for ``("train", "valid", "test")``. Defaults to ``(0.8, 0.1, 0.1)``.
    kind_field:
        If an item is a mapping containing this key, stratification is applied
        per distinct ``kind`` value so each kind is split by the same ratios.

    Returns
    -------
    dict
        ``{"train": [...], "valid": [...], "test": [...]}``. Every split key is
        always present (possibly empty). Each input item appears in exactly one
        split, and the concatenation of the splits is a permutation of the input.
    """
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"ratios must have {len(SPLIT_NAMES)} entries")

    unordered = _as_list(items)

    # Bucket indices by their "kind" (empty string when no kind field present),
    # so identical kinds are split with the same ratios.
    buckets: dict[Any, list[int]] = {}
    for index, item in enumerate(unordered):
        kind = item.get(kind_field) if isinstance(item, Mapping) else None
        buckets.setdefault(kind, []).append(index)

    result: SplitResult = {name: [] for name in SPLIT_NAMES}

    # Deterministic bucket order: sort by a stable key so insertion order of the
    # input cannot influence the outcome.
    for kind in sorted(buckets, key=lambda k: (k is not None, repr(k))):
        indices = buckets[kind]
        # Deterministic ranking by score, tie-broken by index for stability.
        ranked = sorted(indices, key=lambda i: (_stable_key(seed, unordered[i]), i))
        counts = _split_counts(len(ranked), ratios)

        cursor = 0
        for name, count in zip(SPLIT_NAMES, counts):
            for index in ranked[cursor : cursor + count]:
                result[name].append(unordered[index])
            cursor += count

    return result
