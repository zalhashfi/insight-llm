"""Tests for the eval harness (:mod:`insight_llm.eval`) — ticket F02.

Four properties are asserted, mirroring the ticket's acceptance criteria:

1. the eval runs deterministically on the fixed set (same input, same score);
2. a score is reported (and it is the mean of the two axes);
3. a regression is detected — perturbing an output lowers the score, and the
   *right* axis moves for the *right* reason;
4. the grounding metric actually consults the real API: a plausible-but-false
   fact is caught, and a true one is not.

Every test is driven by the shipped fixed set
(``eval_cases.json`` / ``eval_answers.json``) — never by a live service — so the
suite is offline and reproducible.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

# Make ``src`` importable without an install step (the repo's own convention,
# as in tests/test_split.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_llm import eval as ev  # noqa: E402
from insight_llm import spec  # noqa: E402

CASES = ev.load_cases()
ANSWERS = ev.load_answers()


@pytest.fixture()
def baseline_report() -> ev.Report:
    """The report the shipped reference answers earn (the baseline run)."""
    return ev.evaluate(CASES, ANSWERS)


@pytest.fixture()
def baseline(baseline_report: ev.Report) -> ev.Baseline:
    """The recorded baseline of the reference run."""
    return ev.record_baseline(baseline_report)


# ---------------------------------------------------------------------------
# The fixed set is well-formed and complete
# ---------------------------------------------------------------------------


def test_fixed_set_is_non_empty_and_unique():
    assert len(CASES) >= 5
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    # Every case carries at least one scorable fact, and every answer matches a
    # case (a stray answer would silently never be scored).
    assert all(case.facts for case in CASES)
    assert set(ANSWERS) == set(ids)


def test_every_fact_kind_is_exercised():
    kinds = {fact.kind for case in CASES for fact in case.facts}
    assert kinds == {"tool", "route", "limit"}


def test_fact_kinds_are_all_scorable():
    for case in CASES:
        for fact in case.facts:
            # An unknown kind is always false; the fixed set must not contain
            # one, since an unscorable fact makes the metric meaningless.
            assert fact.kind in {"tool", "route", "limit"}


# ---------------------------------------------------------------------------
# (1) Determinism
# ---------------------------------------------------------------------------


def test_same_input_scores_the_same(baseline_report: ev.Report):
    again = ev.evaluate(CASES, ANSWERS)
    assert again.score == baseline_report.score
    assert again.grounding_score == baseline_report.grounding_score
    assert again.shape_score == baseline_report.shape_score


def test_same_input_scores_the_same_bit_for_bit(baseline: ev.Baseline):
    # A baseline recorded from the same input must show no regression at all.
    report = ev.evaluate(CASES, ANSWERS)
    regressed, reasons = ev.compare_to_baseline(report, baseline)
    assert regressed is False
    assert reasons == ()


def test_scoring_does_not_mutate_its_inputs():
    cases = copy.deepcopy(CASES)
    answers = copy.deepcopy(ANSWERS)
    ev.evaluate(cases, answers)
    assert cases == CASES
    assert answers == ANSWERS


def test_answer_order_does_not_change_the_score(baseline_report: ev.Report):
    reversed_answers = {cid: ANSWERS[cid] for cid in reversed(list(ANSWERS))}
    assert ev.evaluate(CASES, reversed_answers).score == baseline_report.score


# ---------------------------------------------------------------------------
# (2) A score is reported
# ---------------------------------------------------------------------------


def test_reference_run_scores_a_perfect_one(baseline_report: ev.Report):
    # The shipped reference answers are correct by construction, so they must
    # score exactly 1.0 — a baseline that is already imperfect would make every
    # regression comparison ambiguous.
    assert baseline_report.score == 1.0
    assert baseline_report.grounding_score == 1.0
    assert baseline_report.shape_score == 1.0
    assert baseline_report.failed_facts == ()


def test_score_is_the_mean_of_the_two_axes(baseline_report: ev.Report):
    expected = (baseline_report.grounding_score + baseline_report.shape_score) / 2.0
    assert baseline_report.score == pytest.approx(expected)


def test_per_case_score_is_a_fraction(baseline_report: ev.Report):
    for case in baseline_report.cases:
        assert 0.0 <= case.score <= 1.0
        assert case.grounding_ok <= case.grounding_total
        assert case.shape_ok <= case.shape_total


def test_record_baseline_captures_the_axes(baseline_report: ev.Report, baseline: ev.Baseline):
    assert baseline.score == baseline_report.score
    assert baseline.grounding_score == baseline_report.grounding_score
    assert baseline.shape_score == baseline_report.shape_score
    assert set(baseline.per_case) == {case.id for case in CASES}


# ---------------------------------------------------------------------------
# (3) Regression is detected — the ticket's headline test
# ---------------------------------------------------------------------------


def test_a_hallucinated_answer_scores_below_the_bundled_baseline(baseline: ev.Baseline):
    """The bundled answers must not be the only thing the scorer can praise.

    An answer that INVENTS API (a ``delete_project`` tool, an ``mcp:write``
    scope) must score strictly below the baseline, so a scoring regression
    cannot hide behind "the fixture scores 1.0".
    """
    hallucinated = dict(ANSWERS)
    first_case = next(iter(hallucinated))
    hallucinated[first_case] = {
        **hallucinated[first_case],
        "answer": "Use delete_project; it needs the mcp:write scope.",
        "tools": [
            {"name": "delete_project", "scope": "mcp:write", "description": "Deletes a project."}
        ],
    }
    report = ev.evaluate(CASES, hallucinated)
    assert report.score < baseline.score
    assert report.grounding_score < baseline.grounding_score

def test_perturbing_an_output_lowers_the_score(baseline: ev.Baseline):
    """The headline acceptance test: a degraded output must score lower.

    ``drop_tools_key`` removes the whole ``tools`` array from every answer — an
    incomplete envelope. The score must fall, a regression must be reported, and
    the reason must name the failing cases.
    """
    degraded = {cid: ev.perturb(ans, "drop_tools_key") for cid, ans in ANSWERS.items()}
    report = ev.evaluate(CASES, degraded)

    assert report.score < baseline.score
    regressed, reasons = ev.compare_to_baseline(report, baseline)
    assert regressed is True
    assert reasons, "a regression must carry at least one reason"


@pytest.mark.parametrize(
    "kind",
    ["drop_tools_key", "blank_notes"],
)
def test_shape_regressions_lower_the_shape_axis(kind: str, baseline: ev.Baseline):
    """A shape degradation always moves the shape axis and lowers the score."""
    degraded = {cid: ev.perturb(ans, kind) for cid, ans in ANSWERS.items()}
    report = ev.evaluate(CASES, degraded)

    assert report.score < baseline.score
    assert report.shape_score < baseline.shape_score


def test_blank_notes_is_a_pure_shape_regression(baseline: ev.Baseline):
    """Emptying ``notes`` breaks the envelope without making a false claim.

    Grounding must be untouched — that independence is what makes the report
    diagnostic rather than a single opaque number. (``drop_tools_key`` is *not*
    a pure shape regression: dropping the ``tools`` array also discards the tool
    claims it carried, so it rightly moves both axes — see the test above.)
    """
    degraded = {cid: ev.perturb(ans, "blank_notes") for cid, ans in ANSWERS.items()}
    report = ev.evaluate(CASES, degraded)

    assert report.grounding_score == baseline.grounding_score
    assert report.score < baseline.score


def test_shattered_envelope_costs_both_axes(baseline: ev.Baseline):
    """A wholesale break carries no verifiable claim, so both axes fall."""
    degraded = {cid: ev.perturb(ans, "shatter_shape") for cid, ans in ANSWERS.items()}
    report = ev.evaluate(CASES, degraded)

    assert report.score < baseline.score
    assert report.shape_score < baseline.shape_score
    assert report.grounding_score < baseline.grounding_score


def test_grounding_regression_lowers_the_grounding_axis(baseline: ev.Baseline):
    """A grounding-only degradation moves grounding and leaves shape intact.

    Perturbing one *tool-bearing* case: the tool names are right but a scope is
    bent from read to write. Shape is unchanged; grounding (and the score) fall.
    """
    target = "tools_read_catalogue"
    degraded = dict(ANSWERS)
    degraded[target] = ev.perturb(ANSWERS[target], "wrong_tool_scope", index=0)

    report = ev.evaluate(CASES, degraded)
    case = next(c for c in report.cases if c.case_id == target)

    assert case.grounding_ok < case.grounding_total
    assert report.grounding_score < baseline.grounding_score
    assert report.shape_score == baseline.shape_score
    assert report.score < baseline.score


def test_unknown_tool_name_is_a_grounding_regression(baseline: ev.Baseline):
    """Renaming a tool to a plausible-but-fake verb is caught by grounding."""
    target = "tools_write_catalogue"
    degraded = dict(ANSWERS)
    degraded[target] = ev.perturb(ANSWERS[target], "unknown_tool", index=0)
    original = ANSWERS[target]["tools"][0]["name"]
    assert degraded[target]["tools"][0]["name"] == f"delete_{original}"

    report = ev.evaluate(CASES, degraded)
    case = next(c for c in report.cases if c.case_id == target)

    assert case.grounding_ok < case.grounding_total
    assert any("grounding" in label or " tool " in label for label in case.failed_facts)


def test_compare_to_baseline_is_strict_not_lenient(baseline: ev.Baseline):
    # An equal score is not a regression; the comparison must be strict, or the
    # determinism test above would be vacuous.
    report = ev.evaluate(CASES, ANSWERS)
    assert ev.compare_to_baseline(report, baseline) == (False, ())


def test_missing_answer_is_scored_as_a_failure(baseline: ev.Baseline):
    """A dropped answer must not be a silent skip — that would inflate scores."""
    thinned = dict(ANSWERS)
    dropped = CASES[0].id
    del thinned[dropped]

    report = ev.evaluate(CASES, thinned)
    assert report.score < baseline.score

    case = next(c for c in report.cases if c.case_id == dropped)
    # No answer at all means no well-formed envelope: shape fails outright, and
    # the missing document is scored, not skipped.
    assert case.shape_ok == 0
    assert case.shape_total > 0


# ---------------------------------------------------------------------------
# (4) The metric consults the REAL API
# ---------------------------------------------------------------------------


def test_false_but_plausible_fact_scores_false():
    # None of these exist on the real surface; a harness that guessed would mark
    # them true, so asserting them false pins the metric to the spec table.
    assert ev.score_fact(ev.Fact("tool", "delete_project")) is False
    assert ev.score_fact(ev.Fact("tool", "mcp:write")) is False
    assert ev.score_fact(ev.Fact("route", "/v1/telemetry/:proj", "POST")) is False
    assert ev.score_fact(ev.Fact("route", "/v1/mcp/delete", "POST")) is False


def test_true_facts_score_true():
    assert ev.score_fact(ev.Fact("tool", "list_projects", "mcp:read")) is True
    assert ev.score_fact(ev.Fact("tool", "emit_event", "mcp:manage")) is True
    assert ev.score_fact(ev.Fact("route", "/v1/telemetry", "POST")) is True
    assert ev.score_fact(ev.Fact("route", "/healthz", "GET")) is True


def test_right_tool_wrong_scope_scores_false():
    # The tool is real but the scope is the other class: still false.
    assert ev.score_fact(ev.Fact("tool", "emit_event", "mcp:read")) is False
    assert ev.score_fact(ev.Fact("tool", "list_projects", "mcp:manage")) is False


def test_absent_polarity_inverts_the_result():
    # A spec-only check (no answer): the name really is absent, so the claim
    # "this does not exist" is true, and the claim "this exists" is false.
    assert ev.score_fact(ev.Fact("tool", "delete_project", expect_absent=True)) is True
    assert ev.score_fact(ev.Fact("tool", "list_projects", expect_absent=True)) is False


def test_an_answer_that_invents_an_absent_tool_fails():
    """Hallucinating a non-existent tool must cost grounding, not earn it.

    The absent-polarity fact is true for a *correct* answer (which omits the
    fake name) and false for an answer that names it — that is the whole point
    of scoring the negative space.
    """
    faked = {
        "prompt_id": "tools_absent_verbs",
        "answer": "Yes: delete_project removes a project.",
        "tools": [{"name": "delete_project", "scope": "mcp:manage", "description": "Delete a project."}],
        "notes": "",
    }
    assert ev.score_fact(ev.Fact("tool", "delete_project", expect_absent=True), faked) is False
    assert ev.score_fact(ev.Fact("tool", "delete_project", expect_absent=True), ANSWERS["tools_absent_verbs"]) is True


def test_limit_fact_requires_an_exact_match():
    assert ev.score_fact(ev.Fact("limit", "series_max_window_ms", 2_592_000_000)) is True
    assert ev.score_fact(ev.Fact("limit", "series_max_window_ms", 2_592_000_001)) is False
    assert ev.score_fact(ev.Fact("limit", "firmware_max_bytes", 8 * 1024 * 1024)) is True


def test_spec_and_real_surface_agree_internally():
    # Guards the spec table itself: the per-tool scope map and the two ordered
    # catalogues must describe the same tools, so a fact cannot be scorable via
    # one view and invisible to another.
    assert set(spec.MCP_TOOL_SCOPES) == set(spec.MCP_READ_TOOLS) | set(spec.MCP_WRITE_TOOLS)
    assert set(spec.MCP_TOOL_SCOPES.values()) == set(spec.MCP_SCOPES)
    # No real tool name collides with a name the spec declares absent.
    assert not (set(spec.MCP_TOOL_SCOPES) & set(spec.RETIRED_NAMES))
    assert not (set(spec.ROUTE_METHODS) & set(spec.RETIRED_ROUTES))


def test_route_method_is_checkable_in_both_directions():
    # Right path, wrong method, is false — and right path, right method (in any
    # case) is true.
    assert ev.score_fact(ev.Fact("route", "/v1/telemetry", "GET")) is False
    assert ev.score_fact(ev.Fact("route", "/v1/telemetry", "post")) is True


def test_shape_scorer_rejects_a_non_document():
    ok, total, failures = ev.score_shape(None)
    assert ok == 0
    assert total == len(failures) > 0


def test_shape_scorer_accepts_an_empty_tools_list():
    # A well-formed answer may legitimately name no tools ("does X exist?").
    ok, total, failures = ev.score_shape(
        {"prompt_id": "p", "answer": "no", "tools": [], "notes": "none exist"}
    )
    assert failures == ()
    assert ok == total


def test_perturb_does_not_mutate_its_input():
    original = copy.deepcopy(ANSWERS["tools_read_catalogue"])
    for kind in ev.PERTURBATIONS:
        ev.perturb(original, kind)
        assert ANSWERS["tools_read_catalogue"] == original


def test_unknown_perturbation_is_rejected():
    with pytest.raises(ValueError):
        ev.perturb(ANSWERS["tools_read_catalogue"], "not-a-real-perturbation")
