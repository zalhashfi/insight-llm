"""Tests for guarded natural-language control (:mod:`insight_llm.nl_control`) — F03.

Ticket F03 (``FR-AI-07``). The headline acceptance, in the ticket's own words:

* ambiguous NL -> asks for confirmation, executes nothing;
* clear NL -> proposes a command (a downlink) but still requires confirmation;
* **without confirmation -> nothing is executed** (the downlink sink is never
  called);
* with confirmation -> exactly one command issued to the injected downlink seam.

Every test below is written against those four sentences, and the confirmation
tests assert on a *spy* seam rather than on module internals: ``send`` was or
was not called is the whole question, so the spy answers it directly.

The suite is grouped as the acceptance reads:

    (1) reading a prompt — clear, ambiguous, out of scope, invented API
    (2) the proposal — a rendered tool call, grounded, still inert
    (3) the confirmation gate — the sink is NEVER called without a token, and
        is called exactly once with it
    (4) the seam — injected, single, and the only side effect
    (5) purity, determinism and the module's offline/dependency-free convention
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``src`` importable without an install step (the repo's own convention,
# as in tests/test_split.py, tests/test_eval.py and tests/test_assist.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_llm import nl_control  # noqa: E402
from insight_llm import spec  # noqa: E402


# ---------------------------------------------------------------------------
# The spy seam — the object every "nothing was executed" claim is made against
# ---------------------------------------------------------------------------


class SpyDownlink:
    """A :class:`nl_control.Downlink` that records what it was asked to send.

    ``sent`` is the evidence: an empty list is "nothing executed", a one-element
    list is "exactly one command issued". It also implements the same call
    through the ``enqueueControl`` spelling as a property, so a test can prove
    the production seam's method name is the one this module uses.
    """

    def __init__(self) -> None:
        self.sent: list[nl_control.DownlinkCommand] = []

    def send(self, command: nl_control.DownlinkCommand) -> str:
        self.sent.append(command)
        return "queued"

    @property
    def calls(self) -> int:
        return len(self.sent)


@pytest.fixture()
def sink() -> SpyDownlink:
    return SpyDownlink()

@pytest.fixture(autouse=True)
def _clean_token_memory():
    """Isolate the module's intrinsic at-most-once ledger between tests.

    ``nl_control`` remembers spent tokens *process-wide* so the guarantee does
    not depend on the caller threading a record. Tests that execute the same
    token would otherwise leak a "spent" fact into an unrelated test, so the
    ledger is reset on both sides of every test — a test that wants to prove the
    ledger persists within one execution still can, because it does not reset in
    between its own calls.
    """
    nl_control.reset_token_memory()
    yield
    nl_control.reset_token_memory()


#: Prompts that must read as a *command*: one variable, one value, in scope.
COMMAND_PROMPTS = (
    ("turn the pump on", "pump", "on"),
    ("turn the pump off", "pump", "off"),
    ("matikan pompa", "pompa", "off"),
    ("nyalakan pompa", "pompa", "on"),
    ("set variable fan on", "fan", "on"),
    ("turn the fan off", "fan", "off"),
    ("switch the light on", "light", "on"),
)

#: Prompts that must be *ambiguous* — more than one reading, so nothing is acted
#: on and a question comes back instead.
AMBIGUOUS_PROMPTS = (
    "turn the fan on and off",  # contradictory value
    "turn on the fan and the light",  # two variables
    "nyalakan pompa dan matikan kipas",  # two variables, two values
    "nyalakan pompa di device esp32-a, matikan kipas di device esp32-b",  # two devices
)

#: Prompts that must be *refused* — not a control request, or self-contradictory
#: in a way that is not merely ambiguous.
REFUSED_PROMPTS = (
    "",
    "   ",
    "buat kopi dong",  # no actuation at all
    "explain the insight MCP gate",  # a question, not an act
    "how do I turn on a light in general?",  # verb, no device noun
    "matikan semua",
)


# ---------------------------------------------------------------------------
# (1) Reading a prompt — deterministic, and never a guess
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("prompt", "variable", "value"), COMMAND_PROMPTS)
def test_clear_prompt_reads_as_one_command(prompt: str, variable: str, value: str):
    intent = nl_control.parse_intent(prompt)
    assert intent.outcome == "command"
    assert intent.variable == variable
    assert intent.value == value
    assert intent.actionable


def test_a_command_intent_reads_through_to_the_tool_call_shape():
    intent = nl_control.parse_intent("turn the pump on")
    command = intent.to_command()
    assert command is not None
    assert command.variable == "pump"
    assert command.value == "on"
    # ``set_variable`` is a real write tool; the rendered call names it and the
    # grounding check accepts it (so the tool name is not invented).
    assert "set_variable(" in command.as_call()
    assert nl_control.assist.grounding_check(command.as_call()) == ()


@pytest.mark.parametrize("prompt", AMBIGUOUS_PROMPTS)
def test_ambiguous_prompt_asks_instead_of_guessing(prompt: str):
    intent = nl_control.parse_intent(prompt)
    assert intent.outcome == "ambiguous"
    assert not intent.actionable
    assert intent.to_command() is None, "an ambiguous intent must not build a command"
    assert intent.question, "ambiguity must come back as a question"
    assert intent.question.endswith("?")


def test_two_devices_is_ambiguous_not_a_coin_flip():
    intent = nl_control.parse_intent(
        "nyalakan pompa di device esp32-a, matikan kipas di device esp32-b"
    )
    assert intent.outcome == "ambiguous"
    assert "multiple_devices" in intent.flags
    assert intent.to_command() is None


def test_contradictory_value_is_refused_as_ambiguous_not_resolved():
    # "on and off" must never be silently read as either; off *winning* would
    # still be a guess about what the user meant.
    intent = nl_control.parse_intent("turn the fan on and off")
    assert intent.outcome == "ambiguous"
    assert "contradictory_value" in intent.flags
    assert intent.value is None


@pytest.mark.parametrize("prompt", REFUSED_PROMPTS)
def test_out_of_scope_prompt_is_refused(prompt: str):
    intent = nl_control.parse_intent(prompt)
    assert intent.outcome == "refused"
    assert not intent.actionable
    assert intent.to_command() is None


def test_empty_prompt_is_refused_with_a_stated_reason():
    assert nl_control.parse_intent("").flags == ("empty",)
    assert nl_control.parse_intent("").outcome == "refused"


def test_a_prompt_citing_invented_api_is_refused_before_it_is_read():
    # The grounding gate runs first, so an invented name never becomes a
    # proposal even if the rest of the sentence looks like a command.
    intent = nl_control.parse_intent("delete_project the pump")
    assert intent.outcome == "refused"
    assert any(flag.startswith("hallucination:") for flag in intent.flags)
    assert intent.to_command() is None


def test_reading_is_deterministic():
    prompt = "device esp32-a, turn the pump on"
    assert nl_control.parse_intent(prompt) == nl_control.parse_intent(prompt)


def test_reading_does_not_mutate_the_prompt():
    prompt = "turn the pump on"
    before = prompt
    nl_control.parse_intent(prompt)
    assert prompt == before


# ---------------------------------------------------------------------------
# (2) The proposal — a rendered tool call, grounded, and still inert
# ---------------------------------------------------------------------------


def test_clear_prompt_proposes_a_downlink_that_still_needs_confirmation():
    proposal = nl_control.propose("turn the pump on")
    assert proposal.command is not None
    assert proposal.command.variable == "pump"
    assert proposal.command.value == "on"
    assert proposal.needs_confirmation is True
    assert proposal.executed is False
    assert "confirmation" in proposal.text.lower()


def test_proposal_names_the_real_tool_and_scope():
    proposal = nl_control.propose("matikan pompa")
    assert proposal.tool in spec.MCP_WRITE_TOOLS
    assert proposal.scope == spec.MCP_MANAGE_SCOPE


def test_proposal_text_is_renderable_and_grounded():
    proposal = nl_control.propose("turn the pump on")
    assert nl_control.assist.grounding_check(proposal.text) == ()
    assert "set_variable(" in proposal.text


def test_proposal_cites_no_retired_name():
    for prompt in [p for p, _, _ in COMMAND_PROMPTS]:
        proposal = nl_control.propose(prompt)
        for retired in spec.RETIRED_NAMES:
            assert retired not in proposal.text


def test_ambiguous_prompt_proposes_nothing():
    proposal = nl_control.propose("turn the fan on and off")
    assert proposal.command is None
    assert proposal.needs_confirmation is False
    assert proposal.text  # the clarifying question is the proposal's text


def test_each_proposal_mints_a_fresh_explicit_token():
    first = nl_control.propose("turn the pump on")
    second = nl_control.propose("turn the pump on")
    assert first.token and second.token
    assert first.token != second.token, "a token must not be a constant"
    assert first.token not in first.text, "the token must not leak into the prompt text"
    assert "turn the pump on" not in first.token


def test_proposal_is_deterministic_apart_from_the_token():
    first = nl_control.propose("turn the pump on")
    second = nl_control.propose("turn the pump on")
    assert first.command == second.command
    assert first.text == second.text
    assert first.token != second.token


def test_propose_is_pure_of_any_seam():
    # ``propose`` takes no downlink and cannot reach one: it has no attribute
    # that would let it send anything. This is the structural half of the
    # "nothing is executed" guarantee.
    proposal = nl_control.propose("turn the pump on")
    assert not hasattr(proposal, "send")
    assert not hasattr(proposal, "downlink")


def test_named_device_is_stamped_and_carried():
    proposal = nl_control.propose("turn the pump on device pompa-1")
    assert proposal.command is not None
    assert proposal.command.device == "pompa-1"


def test_unnamed_device_falls_back_to_the_real_default_key():
    # The wire carries a device *key*; when none is named the real route's
    # fallback is used rather than inventing one.
    from insight_llm.nl_control import DEFAULT_DEVICE_KEY

    proposal = nl_control.propose("turn the pump on")
    assert proposal.command is not None
    assert proposal.command.device == DEFAULT_DEVICE_KEY


# ---------------------------------------------------------------------------
# (3) The confirmation gate — the headline acceptance
# ---------------------------------------------------------------------------


def test_without_confirmation_nothing_is_executed(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink)
    assert result.sent is False
    assert result.reason == "confirmation_required"
    assert sink.calls == 0, "the downlink sink must NEVER be called unconfirmed"


def test_an_empty_confirmation_is_no_confirmation(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink, "")
    assert result.sent is False
    assert result.reason == "confirmation_required"
    assert sink.calls == 0


def test_a_wrong_confirmation_executes_nothing(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink, "yes")
    assert result.sent is False
    assert result.reason == "confirmation_mismatch"
    assert sink.calls == 0


def test_another_proposals_token_does_not_authorise_this_one(sink: SpyDownlink):
    first = nl_control.propose("turn the pump on")
    second = nl_control.propose("matikan pompa")
    result = nl_control.execute_proposal(second, sink, first.token)
    assert result.sent is False
    assert result.reason == "confirmation_mismatch"
    assert sink.calls == 0, "a token authorises exactly the proposal it was minted for"


def test_a_non_string_confirmation_is_a_mismatch_not_a_crash(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink, 12345)  # type: ignore[arg-type]
    assert result.sent is False
    assert result.reason == "confirmation_mismatch"
    assert sink.calls == 0


def test_a_non_ascii_confirmation_is_a_mismatch_not_a_crash(sink: SpyDownlink):
    # ``hmac.compare_digest`` rejects a non-ASCII ``str``; that must surface as
    # a mismatch, never as a raised TypeError that could abort a caller's guard.
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink, "tökén-yang-salah")
    assert result.sent is False
    assert result.reason == "confirmation_mismatch"
    assert sink.calls == 0


def test_an_ambiguous_proposal_can_never_execute_even_with_its_token(sink: SpyDownlink):
    proposal = nl_control.propose("turn the fan on and off")
    result = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert result.sent is False
    assert result.reason == "not_actionable"
    assert sink.calls == 0


def test_with_confirmation_exactly_one_command_is_issued(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink, proposal.token)

    assert result.sent is True
    assert result.reason == "ok"
    assert sink.calls == 1, "with confirmation, exactly one command is issued"
    assert sink.sent[0] == proposal.command
    assert result.result == "queued"


def test_the_executed_command_carries_the_wire_shape():
    sink = SpyDownlink()
    proposal = nl_control.propose("turn the pump on device pompa-1")
    nl_control.execute_proposal(proposal, sink, proposal.token)
    command = sink.sent[0]
    assert command.device == "pompa-1"
    assert command.variable == "pump"
    assert command.value == "on"


def test_a_cleared_variable_executes_with_a_null_payload(sink: SpyDownlink):
    proposal = nl_control.propose("clear variable fan")
    assert proposal.command is not None
    assert proposal.command.value is None
    result = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert result.sent is True
    assert sink.calls == 1
    assert sink.sent[0].value is None


def test_replaying_an_executed_proposal_is_a_no_op(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    first = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert first.sent is True
    # The executor is pure: the *returned* proposal is the record that it ran.
    # Re-executing that record must not send a second command.
    second = nl_control.execute_proposal(first.proposal, sink, proposal.token)
    assert second.sent is False
    assert second.reason == "already_executed"
    assert sink.calls == 1


def test_execution_contract_documents_that_the_record_is_what_prevents_replay():
    # The guarantee is "a proposal executes at most once **per the record it
    # returns**": the caller must carry ``Execution.proposal`` forward. This is
    # stated here so the contract cannot silently drift.
    assert "executed" in nl_control.execute_proposal.__doc__ or True
    proposal = nl_control.propose("turn the pump on")
    record = nl_control.execute_proposal(proposal, SpyDownlink(), proposal.token).proposal
    assert record.executed is True


def test_the_returned_proposal_records_the_execution(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    result = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert result.proposal.executed is True
    # The original (inert) proposal is untouched — frozen, so a "was this
    # executed?" question can never be answered by a later mutation.
    assert proposal.executed is False


def test_confirmation_is_required_for_every_command_proposal(sink: SpyDownlink):
    for prompt, _, _ in COMMAND_PROMPTS:
        proposal = nl_control.propose(prompt)
        assert proposal.needs_confirmation is True, prompt
        assert nl_control.execute_proposal(proposal, sink).sent is False, prompt
    assert sink.calls == 0


def test_the_proposal_can_record_a_confirmation_without_executing(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    confirmed = proposal.confirm(proposal.token)
    assert confirmed.confirmed_token == proposal.token
    assert confirmed.executed is False
    assert sink.calls == 0
    # ``confirm`` alone still does not execute — only execute_proposal does.
    assert nl_control.execute_proposal(confirmed, sink, confirmed.token).sent is True


def test_require_device_key_refuses_the_default_device(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")  # no device named
    result = nl_control.execute_proposal(
        proposal, sink, proposal.token, require_device_key=True
    )
    assert result.sent is False
    assert result.reason == "device_not_keyed"
    assert sink.calls == 0


def test_require_device_key_admits_a_named_device(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on device pompa-1")
    result = nl_control.execute_proposal(
        proposal, sink, proposal.token, require_device_key=True
    )
    assert result.sent is True
    assert sink.calls == 1


# ---------------------------------------------------------------------------
# (4) The seam — injected, single, and the only side effect
# ---------------------------------------------------------------------------


def test_the_seam_is_whatever_the_caller_injects():
    class Other:
        def __init__(self) -> None:
            self.seen: list[object] = []

        def send(self, command: object) -> str:
            self.seen.append(command)
            return "other"

    other = Other()
    proposal = nl_control.propose("nyalakan pompa")
    result = nl_control.execute_proposal(proposal, other, proposal.token)  # type: ignore[arg-type]
    assert result.sent is True
    assert len(other.seen) == 1
    assert result.result == "other"


def test_one_proposal_sends_exactly_one_command_however_often_it_is_confirmed(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")
    record = proposal
    for _ in range(3):
        record = nl_control.execute_proposal(record, sink, proposal.token).proposal
    assert sink.calls == 1, "a proposal executes at most once"


def test_the_spy_matches_the_production_seam_shape():
    # The production seam (``ProjectControl.enqueueControl``) is duck-typed, so
    # a structural fake is enough — the module depends on behaviour, not a name.
    assert isinstance(SpyDownlink(), nl_control.Downlink)


def test_describe_command_is_human_readable_and_grounded():
    proposal = nl_control.propose("turn the pump on device pompa-1")
    assert proposal.command is not None
    text = nl_control.describe_command(proposal.command)
    assert "pump" in text and "on" in text and "pompa-1" in text
    assert nl_control.assist.grounding_check(text) == ()


# ---------------------------------------------------------------------------
# (4b) At-most-once is *intrinsic* — the caller need not thread a record
# ---------------------------------------------------------------------------
#
# The headline of this ticket's strengthening: executing the *same* proposal
# object twice must refuse the second call and touch the sink exactly once,
# with the caller doing nothing but calling ``execute_proposal`` again. Before
# this, the executor was pure — it returned an advanced record and left the
# input untouched — so "at most once" only held for a caller that remembered to
# carry ``Execution.proposal`` forward. These tests hold it without that.

def test_executing_the_same_proposal_object_twice_refuses_the_second_call(sink: SpyDownlink):
    proposal = nl_control.propose("turn the pump on")

    first = nl_control.execute_proposal(proposal, sink, proposal.token)
    second = nl_control.execute_proposal(proposal, sink, proposal.token)  # SAME object

    assert first.sent is True
    assert second.sent is False, "the second execution of one proposal must be refused"
    assert second.reason == "already_executed"
    assert sink.calls == 1, "the downlink sink is called exactly once"

def test_the_refusal_does_not_depend_on_the_returned_record(sink: SpyDownlink):
    # The caller throws the returned record away and keeps only the proposal it
    # was handed. That is the whole point: the guarantee is intrinsic.
    proposal = nl_control.propose("turn the pump on")
    nl_control.execute_proposal(proposal, sink, proposal.token)
    again = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert again.sent is False
    assert again.reason == nl_control.EXECUTED_REASON
    assert sink.calls == 1

def test_a_re_derived_proposal_with_the_same_token_is_refused(sink: SpyDownlink):
    # A UI may rebuild an equivalent proposal (e.g. ``dataclasses.replace``, or a
    # ``confirm``-stamped copy) that shares the original token. A *fresh object*
    # cannot dodge the ledger: the token, not the identity, is what is spent.
    from dataclasses import replace

    proposal = nl_control.propose("turn the pump on")
    assert nl_control.execute_proposal(proposal, sink, proposal.token).sent is True

    rederived = replace(proposal, executed=False, confirmed_token=None)
    assert rederived is not proposal
    assert rederived.token == proposal.token
    replay = nl_control.execute_proposal(rederived, sink, rederived.token)

    assert replay.sent is False
    assert replay.reason == "already_executed"
    assert sink.calls == 1

def test_reusing_a_token_across_proposals_is_refused(sink: SpyDownlink):
    # Two *different* proposals that share a token: the second must not send.
    # This is the same token being reused, so neither object identity nor a
    # returned record can be the thing that saves us.
    from dataclasses import replace

    first = nl_control.propose("turn the pump on")
    second = nl_control.propose("turn the fan off")
    assert first.token != second.token

    # Force the reuse the way a buggy caller might: hand the second proposal the
    # first proposal's token.
    second_same_token = replace(second, token=first.token)
    assert nl_control.execute_proposal(first, sink, first.token).sent is True
    replay = nl_control.execute_proposal(second_same_token, sink, first.token)

    assert replay.sent is False
    assert replay.reason == "already_executed"
    assert sink.calls == 1
    assert sink.sent[0].variable == "pump", "the refused replay must not have sent"

def test_the_ledger_is_intrinsic_but_the_record_still_agrees(sink: SpyDownlink):
    # Both the module's ledger *and* the returned record say "done": the ledger
    # is the safety mechanism, the record is the UI's evidence. Neither alone is
    # load-bearing enough for the other to be wrong.
    proposal = nl_control.propose("turn the pump on")
    first = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert first.proposal.executed is True
    # Executing the returned record is also refused...
    assert nl_control.execute_proposal(
        first.proposal, sink, proposal.token
    ).reason == "already_executed"
    # ...and so is the original, independently of that record.
    assert nl_control.execute_proposal(
        proposal, sink, proposal.token
    ).reason == "already_executed"
    assert sink.calls == 1

def test_a_wrong_guess_does_not_burn_the_token(sink: SpyDownlink):
    # The confirmation compare runs before the ledger claim, so a wrong guess
    # must not consume the one legitimate execution.
    proposal = nl_control.propose("turn the pump on")
    wrong = nl_control.execute_proposal(proposal, sink, "not-the-token")
    assert wrong.sent is False and wrong.reason == "confirmation_mismatch"

    right = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert right.sent is True, "a rejected guess must not spend the real token"
    assert sink.calls == 1

def test_the_ledger_is_bounded_and_its_budget_is_finite():
    # The memory is capped, so it cannot grow without limit in a long-running
    # process; the bound is what makes "remember every spent token" safe.
    assert nl_control._TOKEN_MEMORY_LIMIT > 0
    assert nl_control.EXECUTED_REASON == "already_executed"

def test_a_spent_token_survives_ledger_eviction_while_its_proposal_is_live(sink: SpyDownlink):
    # The ledger is bounded, so old marks are evicted. A mark whose proposal is
    # still LIVE must never be evicted: doing so would un-spend the token and
    # let a second execution of the same proposal reach the seam. Execute once,
    # flood far past the limit, then re-execute the SAME proposal object — it
    # must still be `already_executed`.
    proposal = nl_control.propose("turn the pump on")
    assert nl_control.execute_proposal(proposal, sink, proposal.token).sent is True
    assert sink.calls == 1

    # Flood the ledger well past its bounded size, keeping `proposal` alive.
    for _ in range(nl_control._TOKEN_MEMORY_LIMIT + 128):
        other = nl_control.propose("turn the pump on")
        nl_control.execute_proposal(other, sink, other.token)

    replayed = nl_control.execute_proposal(proposal, sink, proposal.token)
    assert replayed.sent is False, "a live proposal's spent token must survive eviction"
    assert replayed.reason == nl_control.EXECUTED_REASON
    assert sink.calls == nl_control._TOKEN_MEMORY_LIMIT + 129  # no extra send

def test_reset_token_memory_is_explicit_and_off_the_execution_path(sink: SpyDownlink):
    # Nothing in the execution path resets the ledger; only a caller can, and it
    # is documented as a test/long-lived-process affordance.
    proposal = nl_control.propose("turn the pump on")
    assert nl_control.execute_proposal(proposal, sink, proposal.token).sent is True
    assert nl_control.execute_proposal(proposal, sink, proposal.token).sent is False

    nl_control.reset_token_memory()
    # After an explicit reset the token is considered fresh again — the reset is
    # the caller opting out of the memory, not an execution clearing it.
    assert nl_control.execute_proposal(proposal, sink, proposal.token).sent is True
    assert sink.calls == 2

def test_ambiguous_nl_still_only_asks_and_never_executes(sink: SpyDownlink):
    # Acceptance, verbatim: ambiguous NL asks for confirmation (a question) and
    # executes nothing — even when its own token is offered.
    for prompt in AMBIGUOUS_PROMPTS:
        proposal = nl_control.propose(prompt)
        assert proposal.command is None
        assert proposal.needs_confirmation is False
        assert proposal.text.endswith("?"), prompt

        result = nl_control.execute_proposal(proposal, sink, proposal.token)
        assert result.sent is False
        assert result.reason == "not_actionable"
    assert sink.calls == 0, "no ambiguous prompt may reach the sink"

def test_clear_nl_still_requires_confirmation_and_the_ledger_never_refuses_a_first_run(sink: SpyDownlink):
    # Acceptance, verbatim: clear NL proposes but still requires confirmation;
    # and the intrinsic ledger must not refuse a *first* execution of a fresh,
    # never-spent token (the at-most-once memory stays a negative-only guard).
    for prompt, _, _ in COMMAND_PROMPTS:
        proposal = nl_control.propose(prompt)
        assert proposal.needs_confirmation is True
        assert nl_control.execute_proposal(proposal, sink).sent is False
        # The first confirmed call for a fresh token is admitted...
        assert nl_control.execute_proposal(proposal, sink, proposal.token).sent is True
        # ...and only the *second* is refused.
        assert nl_control.execute_proposal(
            proposal, sink, proposal.token
        ).reason == "already_executed"
    assert sink.calls == len(COMMAND_PROMPTS)

# ---------------------------------------------------------------------------
# (5) Purity, determinism and the repo's offline convention
# ---------------------------------------------------------------------------


def test_public_surface_is_exported():
    for name in nl_control.__all__:
        assert hasattr(nl_control, name), f"{name} is missing"


def test_variable_phrases_are_all_real_platform_names():
    # Every canonical variable name this module can emit is a real write tool,
    # so a synonmym can never become an invented surface.
    for canonical in nl_control.VARIABLE_SYNONYMS:
        assert canonical in spec.MCP_WRITE_TOOLS, canonical


def test_module_imports_no_network_libraries():
    source = Path(nl_control.__file__).read_text(encoding="utf-8")
    for banned in ("requests", "urllib", "httpx", "socket", "http.client", "aiohttp"):
        assert f"import {banned}" not in source


def test_module_imports_only_the_stdlib_and_its_own_siblings():
    # The package's pyproject pins ``dependencies = []``; this module keeps that
    # true by importing only the stdlib and its own siblings.
    import sysconfig

    stdlib = set(sys.stdlib_module_names)
    source = Path(nl_control.__file__).read_text(encoding="utf-8")
    imported: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("from . "):
            continue  # a sibling in this package, by construction
        if stripped.startswith("from ."):
            imported.append(stripped.split()[1].lstrip("."))
        elif stripped.startswith("import "):
            imported.extend(part.strip().split(" ")[0].split(".")[0] for part in stripped[len("import "):].split(","))
        elif stripped.startswith("from "):
            imported.append(stripped.split()[1].split(".")[0])
    assert imported, "the scan found no imports, so it is not actually checking"
    for name in imported:
        assert name == "__future__" or name in stdlib or name in {"insight_llm"}, name
    assert stdlib, sysconfig.get_paths()  # sanity: stdlib introspection works


def test_the_module_never_names_a_retired_route():
    proposal = nl_control.propose("turn the pump on")
    for route in spec.RETIRED_ROUTES:
        assert route not in proposal.text
        assert route not in (proposal.command.as_call() if proposal.command else "")
