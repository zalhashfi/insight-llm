"""Tests for :func:`insight_llm.make_split`.

Covers the three required properties: determinism, disjointness/completeness,
and rough proportion (both globally and per stratum).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_llm import make_split  # noqa: E402

ITEMS = [{"id": i, "kind": ["a", "b", "c"][i % 3]} for i in range(100)]
SPLITS = ("train", "valid", "test")


def _flat(result):
    """Return the concatenation of all splits in a fixed order."""
    return [item for name in SPLITS for item in result[name]]


def test_determinism_same_seed_identical():
    first = make_split(ITEMS, seed=42)
    second = make_split(ITEMS, seed=42)
    assert first == second


def test_determinism_order_independent():
    # Shuffling the input must not change the assignment (pure + seed-driven).
    shuffled = list(reversed(ITEMS))
    assert make_split(ITEMS, seed=7) == make_split(shuffled, seed=7)


def test_different_seed_changes_assignment():
    assert make_split(ITEMS, seed=1) != make_split(ITEMS, seed=2)


def test_disjointness_and_union():
    result = make_split(ITEMS, seed=123)
    ids_by_split = [[item["id"] for item in result[key]] for key in SPLITS]

    # No item appears in two splits (pairwise disjoint sets).
    train, valid, test = (set(ids) for ids in ids_by_split)
    assert train.isdisjoint(valid)
    assert train.isdisjoint(test)
    assert valid.isdisjoint(test)

    # Union equals the input; no duplicates introduced.
    assert train | valid | test == {item["id"] for item in ITEMS}
    assert len(_flat(result)) == len(ITEMS)


def test_rough_proportion():
    result = make_split(ITEMS, seed=123)
    n = len(ITEMS)
    assert len(result["train"]) / n == pytest_approx(0.8, abs=0.05)
    assert len(result["valid"]) / n == pytest_approx(0.1, abs=0.05)
    assert len(result["test"]) / n == pytest_approx(0.1, abs=0.05)


def test_stratified_by_kind():
    result = make_split(ITEMS, seed=5)
    for kind in ("a", "b", "c"):
        per_kind = {
            name: [i for i in result[name] if i["kind"] == kind] for name in SPLITS
        }
        total = sum(len(v) for v in per_kind.values())
        # Every item of each kind is accounted for exactly once.
        assert total == sum(1 for item in ITEMS if item["kind"] == kind)
        # And each kind is still roughly 80/10/10.
        train_frac = len(per_kind["train"]) / total
        assert train_frac == pytest_approx(0.8, abs=0.1)


def test_handles_unhashable_and_plain_items():
    # Plain, non-mapping items work; no 'kind' stratification is applied.
    items = [f"item-{i}" for i in range(30)]
    result = make_split(items, seed="abc")
    assert sorted(_flat(result)) == sorted(items)


def test_empty_input():
    result = make_split([], seed=0)
    assert result == {"train": [], "valid": [], "test": []}


def pytest_approx(expected, abs):
    """Import pytest.approx lazily so the module can be imported without pytest."""
    import pytest

    return pytest.approx(expected, abs=abs)
