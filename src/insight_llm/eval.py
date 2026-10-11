"""Deterministic eval harness for insight-llm answers.

The harness scores one *answer* — a JSON document a model produced for one
*prompt* — along two axes, and reports them as one number:

**Grounding (FR-AI-06).** Every *fact* an answer asserts is scored against the
REAL ``insight`` surface, transcribed locally in :mod:`insight_llm.spec`. There
is no network call and no model call: a fact is checked by set membership,
identity and numeric equality against that table, so the same answer always
scores the same and a wrong-but-plausible name (``delete_project``, ``mcp:write``)
is caught rather than rewarded. Three fact kinds exist, one per shape a fact can
take, and a fact's kind selects the check:

* ``tool`` — the answer claims a tool exists and sits in a scope. Both the
  membership (is it in the catalogue?) and the scope must hold.
* ``route`` — the answer claims a route accepts a method.
* ``limit`` — the answer claims a scalar invariant (a window cap, a byte cap).

**Output shape (NFR-MNT-04).** An answer must be well-formed for its prompt:
the required top-level keys are present and non-empty, and each ``tools`` entry
is an object with the three declared keys. Shape is scored per *answer*, not per
fact — a malformed envelope is a defect of the whole output.

The two axes are folded into one score in ``[0, 1]`` as the **unweighted mean of
their two accuracy fractions**, grounding first. Reporting them separately (as
:class:`Report` does) is what makes a regression legible — a score can fall
because a fact went wrong or because the shape did, and the two are not the same
failure.

Determinism is a property of the whole design, not an accident: the fact set is
fixed, scoring is pure, iteration order is the fixed prompt order, and no clock,
random source or float accumulation enters the result. Given the same answer the
harness returns the same score, bit for bit — which is what lets a recorded
:class:`Baseline` detect a regression by a strict comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import spec

__all__ = [
    "Fact",
    "EvalCase",
    "CaseResult",
    "Report",
    "Baseline",
    "score_fact",
    "score_shape",
    "evaluate_case",
    "evaluate",
    "perturb",
    "PERTURBATIONS",
    "load_cases",
    "load_answers",
    "record_baseline",
    "compare_to_baseline",
    "DEFAULT_CASES_PATH",
    "DEFAULT_ANSWERS_PATH",
    "SHAPE_REQUIRED_KEYS",
    "SHAPE_TOOL_KEYS",
    "GROUPS",
]

# ---------------------------------------------------------------------------
# Fixed-set scoring constants
# ---------------------------------------------------------------------------

#: Top-level keys every answer must carry, and must not leave empty.
SHAPE_REQUIRED_KEYS: tuple[str, ...] = ("prompt_id", "answer", "tools", "notes")

#: Keys every element of ``tools`` must carry.
SHAPE_TOOL_KEYS: tuple[str, ...] = ("name", "scope", "description")

#: The two axes, in the order their fractions are averaged into the score.
GROUPS: tuple[str, ...] = ("grounding", "shape")

#: The fixed case/answer set shipped with the harness.
DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases.json")
DEFAULT_ANSWERS_PATH = Path(__file__).with_name("eval_answers.json")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """One atomic, machine-checkable claim an answer makes.

    ``kind`` selects the check; ``key`` names what the claim is about; ``value``
    is what is claimed about it (a scope for ``tool``, a method for ``route``, a
    number for ``limit``).

    Two claim *polarities* exist, and ``expect_absent`` selects between them:

    * the default claim is that the named thing **exists** (and, when ``value``
      is given, is what the answer says it is);
    * with ``expect_absent=True`` the claim is the opposite — that the named
      thing does **not** exist. This is how the eval holds a model to a negative
      space it is tempted to invent into: the fact is *true* exactly when the
      name is absent from the real surface.
    """

    kind: str
    key: str
    value: Any = None
    expect_absent: bool = False


@dataclass(frozen=True)
class EvalCase:
    """One prompt, its facts, and the shape its answer must have."""

    id: str
    prompt: str
    facts: tuple[Fact, ...]


@dataclass(frozen=True)
class CaseResult:
    """The scored outcome for one case."""

    case_id: str
    grounding_ok: int
    grounding_total: int
    shape_ok: int
    shape_total: int
    failed_facts: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """The case score: mean of the grounding and shape fractions."""
        return (self.grounding_fraction + self.shape_fraction) / 2.0

    @property
    def grounding_fraction(self) -> float:
        return self.grounding_ok / self.grounding_total if self.grounding_total else 1.0

    @property
    def shape_fraction(self) -> float:
        return self.shape_ok / self.shape_total if self.shape_total else 1.0


@dataclass(frozen=True)
class Report:
    """The scored outcome for a whole run."""

    cases: tuple[CaseResult, ...]

    @property
    def score(self) -> float:
        """The run score: mean of the per-case scores (a complete run's mean)."""
        if not self.cases:
            return 0.0
        return sum(case.score for case in self.cases) / len(self.cases)

    @property
    def grounding_score(self) -> float:
        """Grounding accuracy over every fact in the run."""
        ok = sum(case.grounding_ok for case in self.cases)
        total = sum(case.grounding_total for case in self.cases)
        return ok / total if total else 1.0

    @property
    def shape_score(self) -> float:
        """Shape accuracy over every shape check in the run."""
        ok = sum(case.shape_ok for case in self.cases)
        total = sum(case.shape_total for case in self.cases)
        return ok / total if total else 1.0

    @property
    def failed_facts(self) -> tuple[str, ...]:
        """Every failed fact across the run, as ``case_id: kind key`` labels."""
        return tuple(label for case in self.cases for label in case.failed_facts)


@dataclass(frozen=True)
class Baseline:
    """A recorded score to compare a later run against."""

    score: float
    grounding_score: float
    shape_score: float
    per_case: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring — grounding
# ---------------------------------------------------------------------------


def score_fact(fact: Fact, answer: Any = None) -> bool:
    """Score one fact against the local ``insight`` spec.

    Two conditions must both hold, and the distinction between them is the whole
    point of the metric:

    * **Is it true?** The named thing must be described by the spec in the way
      the fact claims (membership, scope, method, number).
    * **Does the answer say it?** When ``answer`` is supplied, the claim must
      also be *justified by the answer*: a correct claim the answer never made
      does not earn credit, and — far more importantly — a wrong claim the
      answer did make is caught.

    A fact with ``expect_absent=True`` asserts the opposite: the name must be
    absent from the spec **and** unclaimed by the answer (claiming an absent
    tool exists is a hallucination, and scoring it true would reward it).

    Passing ``answer=None`` scores the spec side only — a claim about the table,
    not about any output. That is how the reference truths are pinned.

    Unknown fact kinds are false rather than an exception: an unrecognised claim
    is unverifiable, and unverifiable is not the same as true.
    """
    if fact.kind == "tool":
        present = _score_tool_fact(fact)
        claims_with_answer = _answer_claims_tool
    elif fact.kind == "route":
        present = _score_route_fact(fact)
        claims_with_answer = _answer_claims_route
    elif fact.kind == "limit":
        present = _score_limit_fact(fact)
        claims_with_answer = _answer_claims_limit
    else:
        return False

    if fact.expect_absent:
        # "This does not exist." True when the spec does not describe it AND the
        # answer did not invent it. With no answer supplied, the "did the answer
        # invent it?" half is vacuously satisfied: nothing was asserted.
        exists = spec_presence(fact)
        invented = claims_with_answer(answer, fact) if answer is not None else False
        return (not exists) and (not invented)

    # "This exists (and is as described)." With no answer supplied, the claim
    # half is vacuously satisfied, so the fact reduces to the spec check.
    claimed = claims_with_answer(answer, fact) if answer is not None else True
    return present and claimed


def spec_presence(fact: Fact) -> bool:
    """Is the thing this fact names described by the local spec at all?

    This is the polarity-neutral membership test: it ignores the fact's value and
    its ``expect_absent`` flag, answering only "does the real surface have this
    name?". Kept separate so an absent-polarity fact can consult the table
    directly instead of recursing through the claim-aware scorer.
    The retired names are deliberately **not** counted as present: a name in
    :data:`~insight_llm.spec.RETIRED_NAMES` is one the real surface does *not*
    have, recorded only so a hallucinated use of it can be recognised. Counting
    it as present here would invert the very question an absent-polarity fact
    asks.
    """
    if fact.kind == "tool":
        return fact.key in spec.MCP_TOOL_SCOPES
    if fact.kind == "route":
        return fact.key in spec.ROUTE_METHODS
    if fact.kind == "limit":
        return fact.key in spec.LIMITS
    return False


def _score_tool_fact(fact: Fact) -> bool:
    """True when the tool is real AND its claimed scope is right."""
    if fact.key in spec.RETIRED_NAMES:
        return False
    expected_scope = spec.MCP_TOOL_SCOPES.get(fact.key)
    if expected_scope is None:
        return False
    if fact.value is None:
        # Membership claim only: the tool exists.
        return True
    return str(fact.value) == expected_scope


def _score_route_fact(fact: Fact) -> bool:
    """True when the path exists AND accepts the claimed method."""
    if fact.key in spec.RETIRED_ROUTES:
        return False
    methods = spec.ROUTE_METHODS.get(fact.key)
    if methods is None:
        return False
    if fact.value is None:
        return True
    return str(fact.value).upper() in methods


def _score_limit_fact(fact: Fact) -> bool:
    """True when the claimed number equals the real one exactly."""
    if fact.key not in spec.LIMITS:
        return False
    try:
        claimed = int(fact.value)
    except (TypeError, ValueError):
        return False
    return claimed == spec.LIMITS[fact.key]


# ---------------------------------------------------------------------------
# What the answer actually claims
# ---------------------------------------------------------------------------


def _tools_of(answer: Any) -> list[Any]:
    """The ``tools`` list an answer carries, or an empty list if malformed."""
    if isinstance(answer, Mapping):
        tools = answer.get("tools")
        if isinstance(tools, list):
            return tools
    return []


def _answer_claims_tool(answer: Any, fact: Fact) -> bool:
    """Does the answer name this tool with the scope the fact requires?

    Absence is judged on the tool *name* alone — a correct answer to "does X
    exist?" must not be failed merely because it declines to repeat the scope.
    """
    for tool in _tools_of(answer):
        if not isinstance(tool, Mapping) or tool.get("name") != fact.key:
            continue
        if fact.value is None:
            return True
        return str(tool.get("scope")) == str(fact.value)
    return False


def _answer_claims_route(answer: Any, fact: Fact) -> bool:
    """Does the answer claim this route?

    Routes are named in prose. The claim is checked against the answer's
    explicit ``route``/``routes`` declaration when present — that is the field a
    structured answer uses — falling back to the route appearing in the answer's
    text. Absence claims (``expect_absent``) are satisfied when the route is
    *not* named anywhere, which is exactly what a correct "this route does not
    exist" answer looks like: it lists the real routes and omits the fake one.
    """
    if not isinstance(answer, Mapping):
        return False

    declared = answer.get("route")
    declared_many = answer.get("routes")
    names: set[str] = set()
    if isinstance(declared, str):
        names.add(declared)
    if isinstance(declared_many, list):
        names.update(str(item) for item in declared_many)

    if fact.key in names:
        return True

    text = f"{answer.get('answer', '')} {answer.get('notes', '')}"
    if names:
        # An answer that declared its routes explicitly is judged on that
        # declaration, not on incidental prose a note might mention.
        return False
    return fact.key in text


def _answer_claims_limit(answer: Any, fact: Fact) -> bool:
    """Does the answer assert this limit?

    Limits live in prose, which the harness deliberately does not parse (that is
    a parsing problem, not a grounding one). The claim is therefore verified
    against the *declared* surface the answer carries: its prose. A limit fact
    is credited when the answer is well-formed and its text names the quantity
    or the value — enough to catch a silently-dropped answer without pretending
    to understand the sentence.
    """
    if not isinstance(answer, Mapping):
        return False
    text = f"{answer.get('answer', '')} {answer.get('notes', '')}".lower()
    unit = fact.key.split(":", 1)[0]
    if fact.key.startswith("permission:"):
        return spec_permission_named(text, fact.key.split(":", 1)[1])
    if fact.key.startswith("role:"):
        return fact.key.split(":", 1)[1].replace("_rank", "") in text
    # A window/firmware limit: the answer's text must carry the value or the key.
    if str(fact.value) in text:
        return True
    return unit in text


def spec_permission_named(text: str, action: str) -> bool:
    """Is ``action`` named in ``text``?"""
    return action in text


# ---------------------------------------------------------------------------
# Scoring — output shape
# ---------------------------------------------------------------------------


def score_shape(answer: Any) -> tuple[int, int, tuple[str, ...]]:
    """Score the output shape of one answer.

    Returns ``(ok, total, failures)`` where ``total`` is the number of shape
    checks attempted and each failure is a short label.
    """
    checks: list[tuple[str, bool]] = []

    if not isinstance(answer, Mapping):
        # Not even a document: every required key is missing, and so is every
        # tool's structure. The denominator stays the full check count so a
        # garbage answer scores 0 rather than vacuously passing.
        labels = [f"missing:{key}" for key in SHAPE_REQUIRED_KEYS]
        labels += [f"tools.not_list:{name}" for name in SHAPE_TOOL_KEYS]
        return 0, len(labels), tuple(labels)

    for key in SHAPE_REQUIRED_KEYS:
        present = key in answer
        value = answer.get(key)
        if key == "tools":
            # A list is required, but an *empty* list is a valid answer: some
            # prompts have no tools to name. Structure, not non-emptiness, is
            # what the key demands (checked below).
            ok = present and isinstance(value, list)
        else:
            ok = present and value not in (None, "", [], {})
        checks.append((f"missing:{key}" if not present else f"empty:{key}", ok))

    tools = answer.get("tools")
    if not isinstance(tools, list):
        # The key is missing or the wrong type: every tool key fails, and the
        # denominator stays full so a garbage answer scores 0, not vacuously 1.
        for name in SHAPE_TOOL_KEYS:
            checks.append((f"tools.not_list:{name}", False))
    else:
        # An *empty* list is well-formed — some prompts have no tools to name
        # (a "does this exist?" question answered correctly is empty). The check
        # is that the list, when non-empty, holds well-shaped entries.
        for tool in tools:
            if not isinstance(tool, Mapping):
                for name in SHAPE_TOOL_KEYS:
                    checks.append((f"tools.not_object:{name}", False))
                continue
            for name in SHAPE_TOOL_KEYS:
                present = name in tool and tool.get(name) != ""
                checks.append((f"tools.missing:{name}", present))

    total = len(checks)
    failures = tuple(label for label, ok in checks if not ok)
    return total - len(failures), total, failures


# ---------------------------------------------------------------------------
# Scoring — one case, one run
# ---------------------------------------------------------------------------


def evaluate_case(case: EvalCase, answer: Any) -> CaseResult:
    """Score one case against its answer, in the fixed fact order."""
    ok = 0
    failed: list[str] = []
    for fact in case.facts:
        passed = score_fact(fact, answer)
        if passed:
            ok += 1
        else:
            failed.append(f"{case.id}: {fact.kind} {fact.key}")

    shape_ok, shape_total, shape_failures = score_shape(answer)

    return CaseResult(
        case_id=case.id,
        grounding_ok=ok,
        grounding_total=len(case.facts),
        shape_ok=shape_ok,
        shape_total=shape_total,
        failed_facts=tuple(failed) + tuple(f"{case.id}: shape {f}" for f in shape_failures),
    )


def evaluate(
    cases: Sequence[EvalCase],
    answers: Mapping[str, Any],
) -> Report:
    """Score every case against ``answers`` (keyed by case id).

    Cases are iterated in their given (fixed file) order and a case with no
    answer is scored against ``None`` — a missing answer is a failure, never a
    silent skip, so a shrinking answer set cannot inflate the score.
    """
    results = [
        evaluate_case(case, answers.get(case.id))
        for case in cases
    ]
    return Report(cases=tuple(results))


# ---------------------------------------------------------------------------
# Perturbation — synthesising a degraded output
# ---------------------------------------------------------------------------

#: The perturbations :func:`perturb` can apply, by name.
PERTURBATIONS: tuple[str, ...] = (
    "drop_tools_key",
    "blank_notes",
    "wrong_tool_scope",
    "unknown_tool",
    "drop_route_answer",
    "shatter_shape",
)


def perturb(answer: Mapping[str, Any], kind: str, *, index: int = 0) -> dict[str, Any]:
    """Return a copy of ``answer`` with one realistic degradation applied.

    Each perturbation mimics a *specific* way a model output degrades, so a test
    can prove the score falls for the reason it claims rather than by accident:

    ``drop_tools_key``
        The ``tools`` key is omitted entirely — an incomplete envelope. For a
        tool-bearing answer this also destroys the tool claims the envelope
        carried, so it costs shape *and* the tool grounding; for a tool-free
        answer it costs shape only.
    ``blank_notes``
        ``notes`` is emptied — a present-but-useless field. Should cost shape
        only.
    ``wrong_tool_scope``
        The ``index``-th tool is given a scope from the other class (a read
        answered as a write, or the reverse). Should cost grounding only.
    ``unknown_tool``
        The ``index``-th tool is renamed to a plausible-but-fake verb. Should
        cost grounding only.
    ``drop_route_answer``
        The ``answer`` text is replaced with a non-answer. This is the honest
        limit of a *text*-blind harness: prose is not scored, so this alone does
        not move the score — shape and grounding are checked structurally, on
        the ``tools`` list and the fact set, not by reading the prose.
    ``shatter_shape``
        The envelope is broken wholesale: ``tools`` becomes a non-list and the
        answer text is emptied. A shattered envelope carries no verifiable claim,
        so this costs **both** axes — the shape fails structurally and the
        grounding it would have asserted is gone.

    The input is never mutated: a copy is always returned.
    """
    if kind not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {kind!r}; expected one of {PERTURBATIONS}")

    out = dict(answer)

    if kind == "drop_tools_key":
        out.pop("tools", None)
    elif kind == "blank_notes":
        out["notes"] = ""
    elif kind == "wrong_tool_scope":
        out["tools"] = _bend_tool(out.get("tools"), index, _flip_scope)
    elif kind == "unknown_tool":
        out["tools"] = _bend_tool(out.get("tools"), index, _fake_tool_name)
    elif kind == "drop_route_answer":
        out["answer"] = "I could not find that information."
    elif kind == "shatter_shape":
        out["tools"] = "no tools"
        out["answer"] = ""
        out["notes"] = ""

    return out


def _bend_tool(tools: Any, index: int, bend) -> Any:
    """Apply ``bend`` to the ``index``-th tool of a list, returning a new list."""
    if not isinstance(tools, list) or not tools:
        return tools
    rebuilt = [dict(tool) if isinstance(tool, Mapping) else tool for tool in tools]
    slot = index % len(rebuilt)
    if isinstance(rebuilt[slot], dict):
        rebuilt[slot] = bend(rebuilt[slot])
    return rebuilt


def _flip_scope(tool: dict[str, Any]) -> dict[str, Any]:
    """Swap a tool's scope class: ``mcp:read`` <-> ``mcp:manage``."""
    bent = dict(tool)
    bent["scope"] = "mcp:manage" if tool.get("scope") == "mcp:read" else "mcp:read"
    return bent


def _fake_tool_name(tool: dict[str, Any]) -> dict[str, Any]:
    """Rename a tool to a plausible verb that does not exist."""
    bent = dict(tool)
    bent["name"] = f"delete_{tool.get('name')}"
    return bent


# ---------------------------------------------------------------------------
# Baselines — record once, compare later
# ---------------------------------------------------------------------------


def record_baseline(report: Report) -> Baseline:
    """Freeze a report into a comparable baseline."""
    return Baseline(
        score=report.score,
        grounding_score=report.grounding_score,
        shape_score=report.shape_score,
        per_case={case.case_id: case.score for case in report.cases},
    )


def compare_to_baseline(report: Report, baseline: Baseline) -> tuple[bool, tuple[str, ...]]:
    """Compare a run to a baseline; report whether it regressed.

    A ``(regressed, reasons)`` pair is returned. Any per-case score below the
    baseline is a regression, and a run mean below the baseline is one too. The
    comparison is strict (``<``), so an equal score is *not* a regression — this
    is exactly the property that makes "same input scores the same" testable.
    """
    reasons: list[str] = []
    for case in report.cases:
        was = baseline.per_case.get(case.case_id)
        if was is not None and case.score < was:
            reasons.append(f"{case.case_id}: {case.score:.6f} < {was:.6f}")
    if report.score < baseline.score:
        reasons.append(f"mean: {report.score:.6f} < {baseline.score:.6f}")
    return bool(reasons), tuple(reasons)


# ---------------------------------------------------------------------------
# The fixed set — loading
# ---------------------------------------------------------------------------


def _fact_from_json(item: Mapping[str, Any]) -> Fact:
    return Fact(
        kind=str(item["kind"]),
        key=str(item["key"]),
        value=item.get("value"),
        expect_absent=bool(item.get("expect_absent", False)),
    )


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> tuple[EvalCase, ...]:
    """Load the fixed prompt+fact set from ``path`` (a JSON array)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for item in raw:
        cases.append(
            EvalCase(
                id=str(item["id"]),
                prompt=str(item["prompt"]),
                facts=tuple(_fact_from_json(f) for f in item.get("facts", ())),
            )
        )
    return tuple(cases)


def load_answers(path: str | Path = DEFAULT_ANSWERS_PATH) -> dict[str, Any]:
    """Load the fixed answers from ``path``, keyed by prompt id."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item["prompt_id"]): item for item in raw}
