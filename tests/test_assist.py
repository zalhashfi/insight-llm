"""Tests for the build assistant (:mod:`insight_llm.assist`) — ticket F01.

Three properties are asserted, mirroring the ticket's acceptance criteria:

1. **Grounded output** — a reply mentions only API that is present in
   :mod:`insight_llm.spec`; every platform name it cites is checked against the
   table, and an invented one is never returned;
2. **Out-of-scope prompts are refused/flagged** — a prompt with no platform
   anchor is refused, and a sketch that cites invented API is flagged and
   corrected rather than answered;
3. **Determinism** — the assistant is a pure function of ``(prompt, sketch,
   spec)``; the same input always yields the same reply.

Every test is driven by the shipped spec table — never by a live model or
service — so the suite is offline and reproducible.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

# Make ``src`` importable without an install step (the repo's own convention,
# as in tests/test_split.py and tests/test_eval.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_llm import assist  # noqa: E402
from insight_llm import spec  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: a real, in-scope ask and a sketch citing invented API
# ---------------------------------------------------------------------------

IN_SCOPE_PROMPTS = (
    "List the MCP read tools and their scopes.",
    "Which HTTP routes ingest telemetry?",
    "What is the maximum series window?",
    "Which roles can delete a project?",
    "Which integrations can send a notification?",
    "How do I send telemetry from my ESP32 sketch?",
)

OUT_OF_SCOPE_PROMPTS = (
    "Write me a poem about cats.",
    "What is the weather in Jakarta tomorrow?",
    "Explain quantum entanglement.",
    "Give me a recipe for rendang.",
)

#: A sketch that posts to a route the platform does not serve and calls a
#: delete tool the surface deliberately lacks.
BAD_SKETCH = '\n'.join(
    [
        'void loop() {',
        '  http.begin(host + "/v1/telemetry/:proj");',
        '  mcp.delete_project(42);',
        '}',
    ]
)


@pytest.fixture()
def bad_sketch_advice() -> assist.Advice:
    """The assistant's reply to a sketch citing invented API."""
    return assist.build_advice("Fix my insight ESP32 sketch", sketch=BAD_SKETCH)


# ---------------------------------------------------------------------------
# (1) The retriever's memory is exactly the spec table
# ---------------------------------------------------------------------------


def test_known_names_are_a_subset_of_the_spec_table():
    names = assist.known_names()
    assert names, "the assistant must know at least one name"
    # Every name the assistant may cite is one the table defines...
    assert names <= (
        set(spec.MCP_READ_TOOLS)
        | set(spec.MCP_WRITE_TOOLS)
        | set(spec.MCP_SCOPES)
        | set(spec.ROUTE_METHODS)
        | set(spec.ROLES)
        | set(spec.PERMISSIONS)
        | set(spec.PROVIDER_IDS)
        | set(spec.LIMITS)
        | {spec.MCP_READ_ROUTE, spec.MCP_WRITE_ROUTE, spec.MCP_WRITE_FLAG, spec.VERSION}
    )
    # ...and no retired name leaks in as if it were real.
    assert not (names & set(spec.RETIRED_NAMES))
    assert not (names & set(spec.RETIRED_ROUTES))


def test_context_block_is_drawn_only_from_the_spec():
    block = assist.context_block("telemetry routes and MCP scopes")
    assert "insight_llm.spec" in block
    # Every route the block names is a real one, and no retired path appears.
    for path in re.findall(r"/v1/[A-Za-z0-9_:/.\-]*", block):
        assert path.rstrip(".,;:") in spec.ROUTE_METHODS
    for retired in spec.RETIRED_ROUTES:
        assert retired not in block


def test_retrieval_is_relevant_and_deterministic():
    first = assist.context_block("what is the maximum series window?")
    again = assist.context_block("what is the maximum series window?")
    assert first == again
    assert "series_max_window_ms" in first

    tools = assist.context_block("list the MCP tools")
    assert "list_projects" in tools and "emit_event" in tools


# ---------------------------------------------------------------------------
# (2) Output mentions ONLY APIs present in spec.py — the headline acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", IN_SCOPE_PROMPTS)
def test_in_scope_reply_is_in_scope(prompt: str):
    advice = assist.build_advice(prompt)
    assert not advice.refused or advice.kind == "correction"
    assert advice.text.strip(), "a reply must say something"


@pytest.mark.parametrize("prompt", IN_SCOPE_PROMPTS)
def test_reply_cites_only_real_api(prompt: str):
    """The core property: no accepted reply names anything outside the table."""
    advice = assist.build_advice(prompt)
    if advice.refused:
        return  # a refusal makes no API claim at all
    assert assist.grounding_check(advice.text) == ()
    for name in advice.cited:
        assert name in assist.known_names()


def test_every_cited_name_exists_in_spec():
    for prompt in IN_SCOPE_PROMPTS:
        for name in assist.build_advice(prompt).cited:
            assert name in assist.known_names(), f"{name!r} is not in spec.py"


def test_citations_are_derived_from_the_text():
    advice = assist.build_advice("list the MCP read tools")
    # ``cited`` is the dual of the grounding check, recomputed from the reply.
    assert set(advice.cited) == set(assist.citations(advice.text))
    assert set(advice.cited).issubset(assist.known_names())


def test_assistant_never_emits_a_retired_name():
    # The retired names are exactly the plausible inventions a model reaches
    # for; a grounded reply must not contain them as if they were real.
    for prompt in IN_SCOPE_PROMPTS + OUT_OF_SCOPE_PROMPTS:
        advice = assist.build_advice(prompt)
        for retired in spec.RETIRED_NAMES:
            assert retired not in advice.text
        for retired in spec.RETIRED_ROUTES:
            assert retired not in advice.text


def test_grounding_check_catches_invented_names():
    assert assist.grounding_check("call delete_project()") == ("delete_project",)
    assert assist.grounding_check("query /v1/telemetry/:proj") == ("/v1/telemetry/:proj",)
    assert assist.grounding_check("we use the mcp:write scope") == ("mcp:write",)
    # A genuine name is not flagged.
    assert assist.grounding_check("call emit_event()") == ()
    assert assist.grounding_check("GET /v1/ota/image") == ()


def test_a_reply_that_would_hallucinate_is_refused_not_returned():
    """The gate is the reason acceptance holds for *any* composer, not just the
    shipped one: a reply citing invented API is downgraded to a refusal.

    This is the ticket's acceptance criterion stated as a property of the
    boundary — if a future model-backed composer invents a name, the check
    catches it at the same place a test can.
    """
    hallucinated = assist.Advice(
        kind="recommendation",
        text="Use the delete_project tool via mcp:write on /v1/mcp/delete.",
    )
    gated = assist._verify(hallucinated)
    assert gated.refused is True
    assert gated.kind == "refused"
    assert gated.text == assist.OUT_OF_SCOPE
    assert "hallucination:" in " ".join(gated.flags)
    assert assist.grounding_check(gated.text) == ()


# ---------------------------------------------------------------------------
# (3) An out-of-scope prompt is refused or flagged, never answered with
#     invented API — the second headline acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", OUT_OF_SCOPE_PROMPTS)
def test_out_of_scope_prompt_is_refused(prompt: str):
    advice = assist.build_advice(prompt)
    assert advice.refused is True
    assert advice.kind == "refused"
    assert advice.flags, "a refusal must be flagged for the caller"
    assert "out of scope" in advice.text.lower()


@pytest.mark.parametrize("prompt", OUT_OF_SCOPE_PROMPTS)
def test_refusal_invents_no_api(prompt: str):
    advice = assist.build_advice(prompt)
    assert assist.grounding_check(advice.text) == ()
    assert advice.cited == ()


def test_scope_gate_is_decided_by_anchors():
    assert assist.is_in_scope("explain the insight MCP gate")
    assert assist.is_in_scope("my ESP32 sketch fails to compile")
    assert not assist.is_in_scope("write me a poem about cats")
    assert not assist.is_in_scope("")


def test_bad_sketch_is_corrected_and_flagged(bad_sketch_advice: assist.Advice):
    advice = bad_sketch_advice
    assert advice.kind == "correction"
    assert advice.flags, "a correction must carry flags naming the problems"
    assert advice.corrected_sketch is not None
    # The correction reply itself cites no invented API.
    assert assist.grounding_check(advice.text) == ()


def test_corrected_sketch_removes_the_bad_lines(bad_sketch_advice: assist.Advice):
    patched = bad_sketch_advice.corrected_sketch or ""
    assert "/v1/telemetry/:proj" not in patched
    assert "delete_project" not in patched
    # The real ingest route survives the patch.
    assert "/v1/telemetry" in patched


def test_correction_prose_does_not_echo_the_invented_names(
    bad_sketch_advice: assist.Advice,
):
    # Quoting the bad name back would itself be a hallucinated citation.
    assert "delete_project" not in bad_sketch_advice.text
    assert "/v1/telemetry/:proj" not in bad_sketch_advice.text


def test_sketch_audit_reports_each_problem_once():
    prompt = "Fix my insight sketch" + "\n" + BAD_SKETCH + "\n" + BAD_SKETCH
    findings = assist._audit_sketch(prompt)
    keys = {(sentence, reason) for sentence, reason in findings}
    assert len(keys) == len(findings), "findings must be de-duplicated"
    assert any("/v1/telemetry/:proj" in sentence for sentence, _ in findings)
    assert any("delete" in reason for _, reason in findings)


def test_platform_agnostic_sketch_code_is_not_flagged():
    # ``loop()`` and ``http.begin`` are ordinary sketch code, not claims about
    # insight — the assistant must not "correct" them.
    findings = assist._audit_sketch('void loop() {\n  http.begin(host);\n}')
    assert findings == ()


# ---------------------------------------------------------------------------
# (4) Determinism and purity
# ---------------------------------------------------------------------------


def test_reply_is_deterministic():
    prompt, sketch = "How do I send telemetry?", BAD_SKETCH
    first = assist.build_advice(prompt, sketch=sketch)
    second = assist.build_advice(prompt, sketch=sketch)
    assert first == second


def test_compose_reply_matches_build_advice():
    prompt = "list the MCP tools"
    assert assist.compose_reply(prompt) == assist.build_advice(prompt)


def test_build_advice_does_not_mutate_its_inputs():
    sketch = BAD_SKETCH
    before = sketch
    assist.build_advice("fix my insight sketch", sketch=sketch)
    assert sketch == before
    before_spec = {name: scope for name, scope in spec.MCP_TOOL_SCOPES.items()}
    assist.build_advice("list the tools")
    assert {name: scope for name, scope in spec.MCP_TOOL_SCOPES.items()} == before_spec


# ---------------------------------------------------------------------------
# (5) The reply is scoreable by the F02 harness (envelope compatibility)
# ---------------------------------------------------------------------------


def test_tool_bearing_reply_matches_the_eval_shape():
    from insight_llm import eval as ev

    advice = assist.build_advice("List the MCP tools and their scopes")
    answer = dict(advice.to_answer(), prompt_id="assist")
    ok, total, failures = ev.score_shape(answer)
    assert ok == total, failures


def test_assistant_citations_are_all_scorable_by_the_eval_harness():
    from insight_llm import eval as ev

    for prompt in IN_SCOPE_PROMPTS:
        for name in assist.build_advice(prompt).cited:
            if name in spec.MCP_TOOL_SCOPES:
                fact = ev.Fact("tool", name, spec.MCP_TOOL_SCOPES[name])
                assert ev.score_fact(fact) is True


# ---------------------------------------------------------------------------
# (6) The module stays offline and dependency-free (the repo's convention)
# ---------------------------------------------------------------------------


def test_assist_module_imports_no_network_libraries():
    source = Path(assist.__file__).read_text(encoding="utf-8")
    for banned in ("requests", "urllib", "httpx", "socket", "http.client"):
        assert f"import {banned}" not in source


def test_public_surface_is_exported():
    for name in assist.__all__:
        assert hasattr(assist, name), f"{name} is missing"
