"""Natural language -> guarded downlink control (ticket F03, ``FR-AI-07``).

This is the **guarded tool-call** half of the assistant. Where :mod:`assist`
answers questions about the platform, this module lets a user *act* on it in
plain words ("nyalakan pompa", "turn the fan off") — and that difference is the
whole design problem: a question that is answered badly is a bad answer, but a
command that is *executed* badly opens a valve.

So the pipeline is deliberately not a generator that happens to have a hook. It
is a deterministic read-then-confirm spine:

1. **Read the intent** — a deterministic parser turns the text into a
   :class:`ControlIntent`: which device, which variable, which value. The
   *values* are a small, closed actuation vocabulary (:data:`ON_VALUES`,
   :data:`OFF_VALUES`, :data:`CLEAR_WORDS`); the *variable* is either a real
   platform name or a known target noun (:data:`_TARGET_NOUNS`). Anything the
   parser cannot read is a refusal or a question, never a guess. The prompt is
   also run through :func:`assist.grounding_check`, so a request that cites
   invented API is refused before it is read as a command at all.
2. **Propose a tool call** — the intent is rendered as exactly the
   ``set_variable`` MCP call the platform really exposes
   (:data:`spec.MCP_WRITE_TOOLS`, scope :data:`spec.MCP_MANAGE_SCOPE`). The
   proposal is text plus a machine-readable :class:`DownlinkCommand`; **nothing
   is executed**.
3. **Confirm** — the proposal carries an unguessable
   :func:`secrets.token_urlsafe` token. The caller must echo that exact token
   back; only then does :func:`execute_proposal` cross the seam.

Ambiguity is a first-class outcome, not an error: a prompt naming more than one
device, more than one variable, or both an on and an off is *not* parsed by
picking a winner — it comes back as a clarifying question. A prompt that is not
a control request at all (a question, or no device noun) is refused.

## The confirmation mechanism, precisely

Execution is separated from intent by three structural facts, none of which a
test has to take on faith:

* :func:`propose` is a **pure function of the text** — it takes no seam, and
  therefore *cannot* reach one. There is no code path from a prompt to the
  downlink that does not pass through :func:`execute_proposal`.
* :func:`propose` mints a fresh :func:`secrets.token_urlsafe` token and keeps it
  **private** to the returned :class:`Proposal`. The prompt text never carries
  it, so a prompt can never self-confirm.
* :func:`execute_proposal` is the *only* caller of the injected
  :class:`Downlink`, and it calls it only when the confirmed string equals the
  proposal's token **and** the proposal is a clean, unimpeded intent. Missing
  confirmation, a wrong token, a token handed to a different proposal, a refused
  (ambiguous / out-of-scope / hallucinated) intent, and a repeat of an
  already-executed proposal are each a no-op with a stated reason — the seam is
  never touched.

Because the token is minted per proposal and compared with
:func:`hmac.compare_digest`, "confirm this" and "confirm *that*" are different
acts: a user can only ever authorise the exact command they were shown.

Nothing here reaches the network, imports a provider SDK, or invents a name.
"""

from __future__ import annotations

import hmac
import re
import secrets
from dataclasses import dataclass, field, replace
from typing import Final, Protocol, runtime_checkable

from . import assist, spec

__all__ = [
    "ControlIntent",
    "ControlOutcome",
    "Device",
    "Downlink",
    "DownlinkCommand",
    "Proposal",
    "Execution",
    "VARIABLE_SYNONYMS",
    "ON_VALUES",
    "OFF_VALUES",
    "CLEAR_WORDS",
    "DEFAULT_DEVICE_KEY",
    "parse_intent",
    "propose",
    "execute_proposal",
    "describe_command",
]


# ---------------------------------------------------------------------------
# The downlink seam — the single side effect this module can have
# ---------------------------------------------------------------------------


@runtime_checkable
class Downlink(Protocol):
    """The one side effect: put a command on a device's downlink queue.

    Production injects a client over the REAL ``ProjectControl.enqueueControl``
    seam (``worker/src/platform/durable-objects/project-control.ts``); tests
    inject a spy. The method name mirrors that seam on purpose. Because this is
    the *only* seam the module knows, "nothing was executed" is exactly
    "``send`` was never called" — an assertion a test can make without knowing
    anything else about the module's internals.
    """

    def send(self, command: "DownlinkCommand") -> object:
        """Deliver one command. The return value is opaque to this module."""
        ...


@dataclass(frozen=True)
class Device:
    """The device a command is addressed to.

    ``name`` is what the user said (``"pompa"``, ``"esp32-a"``). ``stamped``
    records whether the *name* can be carried onto the wire: the real wire
    contract carries a device **key**, not a free-text label, so a name that is
    a key-shaped identifier is stamped and anything else is not. Whether to
    demand a stamp is the caller's policy — see :func:`execute_proposal`.
    """

    name: str
    stamped: bool = False

    @property
    def wire_key(self) -> str:
        """The identifier that may be sent, or the unknown-device fallback."""
        return self.name if self.stamped else DEFAULT_DEVICE_KEY


@dataclass(frozen=True)
class DownlinkCommand:
    """One downlink command, in the REAL wire shape.

    ``variable``/``value`` are exactly the fields a device receives from
    ``GET /v1/control`` (``CONTROL_COMMAND_FIELDS`` in
    ``domains/telemetry/control.ts``) — *not* a second envelope invented here.
    ``value`` is ``None`` for "clear the variable", mirroring an omitted or null
    payload on the wire.
    """

    device: str
    variable: str
    value: str | None

    def as_call(self, *, tool: str = "set_variable") -> str:
        """Render the command as the grounded MCP tool call it stands for.

        The tool name defaults to ``set_variable``, which the table lists as a
        write tool (:data:`spec.MCP_WRITE_TOOLS`); rendering uses the trailing
        ``(`` call shape so :func:`assist.grounding_check` treats it as a
        platform claim, exactly as it would in prose.
        """
        payload = "clear" if self.value is None else self.value
        return f"{tool}(variable={self.variable}, value={payload}, device={self.device})"


# ---------------------------------------------------------------------------
# The value vocabulary — read from the spec, not retyped as magic strings
# ---------------------------------------------------------------------------

#: Words a user uses for "on". `set` is included because "set the fan to 1" and
#: "turn the pump on" are the same intent.
ON_VALUES: Final[tuple[str, ...]] = ("on", "on.", "aktif", "aktifkan", "hidup", "nyala", "nyalakan", "set", "enable", "enabled", "start", "true", "1", "ya")

#: Words for "off".
OFF_VALUES: Final[tuple[str, ...]] = ("off", "off.", "mati", "matikan", "padam", "padamkan", "disable", "disabled", "stop", "false", "0", "tidak")

#: Sentinel returned by :func:`_said_value` for "no payload" — the caller turns
#: it into the real ``None`` a ``value: null`` downlink carries. Keeping the
#: sentinel distinct from ``None`` is what lets "clear the fan" be told apart
#: from "I could not read a value".
_CLEAR: Final[str] = "\x00clear"

#: Sentinel for "the prompt asks for both on and off". The caller refuses it as
#: ambiguous rather than picking a side: guessing here would actuate a device on
#: a coin flip, which is precisely what confirmation exists to prevent.
_CONTRADICTION: Final[str] = "\x00both"

#: Verb -> on/off. The verb set is closed on purpose: it is the actuation
#: vocabulary the platform's write path exposes, kept small so an unmodelled
#: verb is a refusal rather than a guess.
_VERB_VALUES: Final[dict[str, str]] = {
    "turn on": "on",
    "switch on": "on",
    "power on": "on",
    "nyalakan": "on",
    "hidupkan": "on",
    "aktifkan": "on",
    "enable": "on",
    "turn off": "off",
    "switch off": "off",
    "power off": "off",
    "matikan": "off",
    "padamkan": "off",
    "disable": "off",
    "clear": _CLEAR,
    "reset": _CLEAR,
    "hapus": _CLEAR,
    "kosongkan": _CLEAR,
    "unset": _CLEAR,
    "clear variable": _CLEAR,
}

#: Words that mean "no payload" — derived from the verb table, so the clear
#: vocabulary and the clear *verbs* can never drift apart.
CLEAR_WORDS: Final[tuple[str, ...]] = tuple(
    word for word, val in _VERB_VALUES.items() if val == _CLEAR
)

#: Canonical variable vocabulary. The keys are the *real* write tools the table
#: lists (:data:`spec.MCP_WRITE_TOOLS`), so a synonym can never invent a surface
#: — a test asserts that membership. Phrases are matched longest-first.
VARIABLE_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "set_variable": ("set variable", "set_variable"),
    "emit_event": ("emit event", "emit_event"),
    "create_variable": ("create variable", "create_variable"),
}

#: The device key the real route falls back to when no device is named
#: (``DEFAULT_DEVICE_KEY`` in ``domains/telemetry/control.ts``).
DEFAULT_DEVICE_KEY: Final[str] = "unknown"

# ---------------------------------------------------------------------------
# Scope — what makes a prompt a *control* request at all
# ---------------------------------------------------------------------------

#: Nouns that name a *device* — the platform entity a command is addressed to.
#: A control request must name one of these (or an explicit ``<key> device``).
_DEVICE_NOUNS: Final[tuple[str, ...]] = (
    "device",
    "perangkat",
    "alat",
)

#: Nouns that name an *actuated thing* — the variable a command sets. "turn the
#: pump on" actuates the variable ``pump``. This is a closed list on purpose:
#: only a word here is ever promoted to a variable name, so free prose cannot
#: become a device action. Both English and Indonesian, matching the corpus.
_TARGET_NOUNS: Final[tuple[str, ...]] = (
    "pump",
    "pompa",
    "fan",
    "kipas",
    "lamp",
    "lampu",
    "light",
    "valve",
    "relay",
    "heater",
    "setpoint",
    "sensor",
    "led",
    "motor",
    "buzzer",
    "servo",
)

#: Interrogative openers. A prompt that *asks* is a question about the platform,
#: not an instruction to act on it — "how do I turn on a light?" names a verb and
#: a noun, but answering it with a live device command would be the single worst
#: mistake this module could make. The check is on the shape of the sentence,
#: not on a keyword that happens to appear inside a command.
_QUESTION_OPENERS: Final[tuple[str, ...]] = (
    "how ",
    "how do",
    "what ",
    "why ",
    "when ",
    "where ",
    "which ",
    "who ",
    "can i",
    "could i",
    "should i",
    "is it",
    "do i",
    "does ",
    "bagaimana",
    "apa ",
    "kenapa",
    "mengapa",
    "kapan",
    "di mana",
)

#: A device noun or a target noun — the union used by the scope gate, since a
#: request naming a pump is about a device even if it never says "device".
_ACTUATION_NOUNS: Final[tuple[str, ...]] = _DEVICE_NOUNS + _TARGET_NOUNS

#: Verbs that name an actuation (in English and Indonesian). A prompt carrying
#: one of these *and* a device noun is a control request; a question, or a verb
#: with no noun, is not.
_ACTUATION_VERBS: Final[tuple[str, ...]] = (
    "turn on",
    "turn off",
    "switch on",
    "switch off",
    "power on",
    "power off",
    "enable",
    "disable",
    "set",
    "clear",
    "reset",
    "start",
    "stop",
    "nyalakan",
    "matikan",
    "hidupkan",
    "padamkan",
    "aktifkan",
    "setel",
    "atur",
)


def _is_control_request(text: str, *, table: type = spec) -> bool:
    """Is ``text`` an attempt to *act on* a platform device?

    A control request must carry **both** an actuation verb and a device noun.
    That conjunction is the module's scope gate, and it is deliberately *not*
    :func:`assist.is_in_scope`: the assistant's gate is anchor-based (a prompt
    must mention platform jargon to be answerable), and a plain actuation like
    ``"turn the pump on"`` carries none — correctly, since it is not a question
    about the API. Here it is exactly the point of the request.

    Requiring the device noun matters: "how do I turn on a light in general?"
    names a verb but no device, so it is out of scope and is refused rather than
    turned into a command against the wrong device.

    Grounding is checked *before* this (in :func:`parse_intent`), so a prompt
    that clears this gate still cannot smuggle in an invented name.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if _is_question(text):
        return False
    lowered = text.lower()
    has_verb = any(_says_verb(lowered, v) for v in _ACTUATION_VERBS)
    has_noun = any(re.search(rf"\b{n}\b", lowered) for n in _ACTUATION_NOUNS)
    return has_verb and has_noun


def _is_question(text: str) -> bool:
    """Is ``text`` phrased as a question rather than an instruction?

    A question mark, or an interrogative opener (``how``, ``what``, ``bagaimana``
    …), marks a question. The opener list is anchored to the *start* of the
    prompt so a command that merely contains ``when`` later is not misread.
    """
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(opener) for opener in _QUESTION_OPENERS)


def _says_verb(text: str, verb: str) -> bool:
    """Does ``text`` use ``verb``, allowing a split particle pair?

    ``turn on`` and ``turn the pump on`` are the same act. English phrasal verbs
    separate, so a two-word verb is matched with a short, word-bounded gap of up
    to three tokens between its halves (``turn <the pump> on``); a one-word verb
    is matched adjacency-only, so ``setpoint`` is never read as ``set``.
    """
    words = verb.split()
    if len(words) == 1:
        return bool(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(verb)}(?![A-Za-z0-9_])", text)
        )
    head, tail = words[0], " ".join(words[1:])
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(head)}(?:\s+[A-Za-z0-9_.\-]+){{0,3}}\s+{re.escape(tail)}(?![A-Za-z0-9_])",
            text,
        )
    )

#: A device name may be carried on the wire only if it is key-shaped. Labels
#: with spaces or punctuation are kept for the human but never sent.
_DEVICE_KEY: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

#: A device noun, as a word. Used to bound a device name so that
#: ``device esp32-a`` yields ``esp32-a`` and never the clause around it.
_DEVICE_NOUN_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:device|perangkat|alat)\b")


# ---------------------------------------------------------------------------
# The intent — what the prompt is asking to do
# ---------------------------------------------------------------------------

ControlOutcome = str  # "command" | "ambiguous" | "refused"


@dataclass(frozen=True)
class ControlIntent:
    """The deterministic read of a prompt: what to do, or why we will not.

    ``outcome`` is one of:

    * ``"command"`` — the prompt names exactly one variable and one resolvable
      value, and cites no invented API. It still needs confirming.
    * ``"ambiguous"`` — more than one device, more than one variable, or no
      resolvable value. Nothing is guessed; :attr:`question` says what to ask.
    * ``"refused"`` — the prompt is out of scope for the platform, or it reads
      as an actuation but names no variable.
    """

    outcome: ControlOutcome
    text: str
    device: Device | None = None
    variable: str | None = None
    value: str | None = None
    question: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        """Is this a clean, single, value-bearing command?"""
        return self.outcome == "command" and self.variable is not None

    def to_command(self) -> DownlinkCommand | None:
        """The wire command this intent stands for, or ``None`` when it is not
        actionable. Reads through :meth:`Device.wire_key`, so the stamping
        policy is applied in exactly one place."""
        if not self.actionable:
            return None
        device = self.device or Device(DEFAULT_DEVICE_KEY)
        return DownlinkCommand(device=device.wire_key, variable=self.variable or "", value=self.value)


# ---------------------------------------------------------------------------
# Reading a prompt — closed vocabulary, longest-match, never a guess
# ---------------------------------------------------------------------------


def _said_devices(text: str) -> tuple[str, ...]:
    """Every device the prompt names, first-appearance order.

    Recognised forms, both noun-bounded so a name never absorbs its sentence:

    * the ``device <key>`` form (``device esp32-a``, ``perangkat pompa_1``);
    * the ``<key> device`` form (``esp32-a device``, ``pompa device``).

    Two of them in one prompt is not resolved — it is the ambiguity
    :func:`parse_intent` refuses rather than guessing between.
    """
    found: list[str] = []
    # Words that can never *be* a device name: value words, target nouns, and
    # the device nouns themselves. Without this, "turn the pump on device X"
    # would read the "on" before "device" as a second device.
    stopwords = set(
        _DEVICE_NOUNS
        + _TARGET_NOUNS
        + ON_VALUES
        + OFF_VALUES
        + CLEAR_WORDS
        + ("the", "a", "an", "and", "then", "to", "for", "di", "ke", "dan")
    )
    # ``device <key>``: take the token right after the noun, clipped to the
    # identifier shape so a trailing "on"/"off" is not swallowed as a name.
    for noun in _DEVICE_NOUN_RE.finditer(text):
        rest = text[noun.end():]
        m = re.match(r"[:\s]+([A-Za-z0-9][A-Za-z0-9_.:-]*)", rest)
        if m and m.group(1) not in found and m.group(1) not in stopwords:
            found.append(m.group(1))
    # ``<key> device``: the token immediately before the noun, provided it is
    # not a stopword, a noun or a verb (that would be "the device" / "on device").
    for noun in _DEVICE_NOUN_RE.finditer(text):
        head = text[: noun.start()]
        m = re.search(r"([A-Za-z0-9][A-Za-z0-9_.:-]*)\s*$", head)
        if not m:
            continue
        candidate = m.group(1)
        if candidate in stopwords or candidate in found:
            continue
        if any(re.fullmatch(re.escape(v), candidate) for v in _ACTUATION_VERBS):
            continue
        found.append(candidate)
    return tuple(found)


def _said_variables(text: str) -> tuple[str, ...]:
    """Every variable name the prompt names, first-appearance order.

    Three recognisers run, and the third is the one that makes plain speech
    work:

    * the canonical control verbs in :data:`VARIABLE_SYNONYMS`
      (``set variable``, ``emit event``) — these *are* variable names, because
      the platform exposes them as such;
    * an identifier the user explicitly marks with ``variable <key>``;
    * a **target noun** — the thing being actuated (``pump``, ``fan``,
      ``heater``, ``setpoint``). "turn the pump on" actuates the variable
      ``pump``; without this recogniser the module could only ever act on
      variables spelled in platform jargon, which is the opposite of the natural
      language it exists for.

    A target noun is only ever read as a variable name when it is a known
    target (:data:`_TARGET_NOUNS`), so free prose — "turn things on" — is still
    never promoted to a variable. The result is a *candidate set*; more than one
    candidate is ambiguity for :func:`parse_intent` to refuse, not a coin flip.
    """
    found: list[str] = []
    # Explicit names win: ``variable heater`` is the variable ``heater``. When
    # the user names a variable, a canonical control verb elsewhere in the
    # sentence (usually the very ``set variable`` that carried the name) is the
    # *verb*, not a second variable — so it is not counted as a candidate.
    explicit = re.findall(r"\bvariable[:\s]+([A-Za-z_][A-Za-z0-9_]*)", text)
    for token in explicit:
        if token not in found:
            found.append(token)

    targets = [
        m.group(1)
        for m in re.finditer(r"(?<![A-Za-z0-9_.:-])([a-z][a-z0-9_]*)(?![A-Za-z0-9_.:-])", text)
        if m.group(1) in _TARGET_NOUNS
    ]
    for token in targets:
        if token not in found:
            found.append(token)

    # Only when the prompt named no variable and no target noun does a canonical
    # control verb double as the variable (``emit_event`` actuates a variable
    # literally named ``emit_event``).
    if not explicit and not targets:
        phrases = sorted(
            ((phrase, canon) for canon, words in VARIABLE_SYNONYMS.items() for phrase in words),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        for phrase, canon in phrases:
            if _says_verb(text, phrase) and canon not in found:
                found.append(canon)
    return tuple(found)


def _said_value(text: str) -> str | None:
    """The value the prompt resolves to, or ``None`` when it is unresolvable.

    Resolution order:

    1. a clear verb (``clear``, ``hapus``) -> :data:`_CLEAR`;
    2. both an on and an off -> :data:`_CONTRADICTION`, which the caller refuses
       rather than picks between (silence is never read as "on", and a
       contradiction is never silently resolved to either);
    3. a single side, by value word (``on``/``off``) or by verb
       (``turn on``/``matikan``);
    4. nothing recognisable -> ``None`` ("unresolvable", not "off").
    """
    # A clear verb is a complete instruction on its own, so it wins outright.
    for phrase in sorted(_VERB_VALUES, key=len, reverse=True):
        if _VERB_VALUES[phrase] == _CLEAR and _says_verb(text, phrase):
            return _CLEAR

    has_on = _says_value(text, ON_VALUES, "on")
    has_off = _says_value(text, OFF_VALUES, "off")
    if has_on and has_off:
        return _CONTRADICTION
    if has_on:
        return "on"
    if has_off:
        return "off"
    return None


def _says_value(text: str, words: tuple[str, ...], wanted: str) -> bool:
    """Does ``text`` say ``wanted`` (``"on"``/``"off"``), by word or by verb?"""
    if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
        return True
    return any(
        val == wanted and _says_verb(text, phrase)
        for phrase, val in _VERB_VALUES.items()
    )


def parse_intent(prompt: str, *, table: type = spec) -> ControlIntent:
    """Read ``prompt`` as a control intent — deterministically, and without a model.

    Ordering matters and is deliberate:

    1. **Grounding first.** An invented API name in the prompt
       (:func:`assist.grounding_check`) is a refusal, before anything is parsed.
       A user who asks to call ``delete_project()`` does not get a proposal.
    2. **Scope.** A prompt that is a question, or that names no device noun, is
       not a control request and is refused: "buat kopi" and "how do I turn on a
       light?" are both refused here.
    3. **Variable.** Zero names -> a refusal that asks which variable; several
       -> ambiguous (never "pick the first").
    4. **Value.** Unresolvable -> ambiguous; an explicit on+off contradiction is
       ambiguous too (never resolved to a side, per :func:`_said_value`).

    The return value is the intent alone. It holds no reference to a downlink
    and no confirmation token: parsing can propose, never execute.
    """
    text = prompt if isinstance(prompt, str) else ""
    raw = text.strip()
    if not raw:
        return ControlIntent(outcome="refused", text=raw, flags=("empty",))

    invented = list(assist.grounding_check(text, table=table))
    # ``grounding_check`` catches invented names in *platform-claim* positions
    # (calls, tool lists, routes). A retired name used as an object ("delete
    # project the pump") is not in one of those positions, so scan for the
    # retired vocabulary outright — it is exactly the vocabulary a control
    # request must never be built on.
    for retired in table.RETIRED_NAMES:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(retired)}(?![A-Za-z0-9_])", text) and retired not in invented:
            invented.append(retired)
    if invented:
        return ControlIntent(
            outcome="refused",
            text=raw,
            flags=tuple(f"hallucination:{name}" for name in invented),
        )

    if not _is_control_request(text, table=table):
        return ControlIntent(outcome="refused", text=raw, flags=("out_of_scope",))

    lowered = raw.lower()
    devices = _said_devices(lowered)
    device_name = devices[0] if devices else None
    device = (
        Device(device_name, stamped=bool(device_name and _DEVICE_KEY.match(device_name)))
        if device_name
        else None
    )
    if len(devices) > 1:
        return ControlIntent(
            outcome="ambiguous",
            text=raw,
            device=device,
            flags=("multiple_devices",),
            question=(
                "Which single device should I command? More than one device came "
                "up in that request (" + ", ".join(devices) + "), so I will not "
                "choose one for you. Name one device (e.g. `device esp32-a`) and I "
                "will propose the downlink?"
            ),
        )

    variables = _said_variables(lowered)
    if len(variables) > 1:
        return ControlIntent(
            outcome="ambiguous",
            text=raw,
            device=device,
            flags=("multiple_variables",),
            question=(
                "That request names more than one variable ("
                + ", ".join(variables)
                + "), so I will not guess. Which one should I set, and to what value?"
            ),
        )
    if not variables:
        return ControlIntent(
            outcome="refused",
            text=raw,
            device=device,
            flags=("no_variable",),
            question=(
                "Which variable should I set on the device? Name it as a variable "
                "(e.g. `variable fan`) and I can propose a `set_variable` downlink?"
            ),
        )

    raw_value = _said_value(lowered)
    if raw_value is None:
        return ControlIntent(
            outcome="ambiguous",
            text=raw,
            device=device,
            variable=variables[0],
            flags=("no_value",),
            question=(
                f"What value should `{variables[0]}` be set to? I could not read an "
                "on/off, a number, or a clear from that request, so I am not going "
                "to guess a device action?"
            ),
        )
    if raw_value == _CONTRADICTION:
        return ControlIntent(
            outcome="ambiguous",
            text=raw,
            device=device,
            variable=variables[0],
            flags=("contradictory_value",),
            question=(
                f"That request asks for `{variables[0]}` to be both on and off. I "
                "will not pick one — say which single value you want and I will "
                "propose the downlink?"
            ),
        )

    # The clear sentinel becomes the real ``null`` payload the wire carries.
    value: str | None = None if raw_value == _CLEAR else raw_value

    flags: tuple[str, ...] = ()
    if device and not device.stamped:
        flags = ("device_name_not_a_key",)

    return ControlIntent(
        outcome="command",
        text=raw,
        device=device,
        variable=variables[0],
        value=value,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# The proposal — a tool call waiting for a human "yes"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """A rendered tool call plus the token that alone can authorise it.

    ``token`` is a :func:`secrets.token_urlsafe` value, minted per proposal and
    never derived from the prompt. ``confirmed_token`` is the last token the
    caller offered; it is compared to ``token`` in
    :func:`execute_proposal` and is kept on the record so a UI can show "this
    was confirmed" without the executor re-minting anything.
    """

    intent: ControlIntent
    command: DownlinkCommand | None
    token: str
    tool: str = "set_variable"
    scope: str = spec.MCP_MANAGE_SCOPE
    confirmed_token: str | None = None
    executed: bool = False
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_confirmation(self) -> bool:
        """Every actionable proposal needs confirmation, always.

        This is a *constant* for a command-bearing proposal: there is no flag,
        no environment, and no caller that turns it off. A proposal that is not
        actionable needs no confirmation because it can never execute.
        """
        return self.command is not None

    @property
    def text(self) -> str:
        """The user-facing proposal, or the question/refusal for a non-command."""
        if self.command is not None:
            return (
                f"Proposed downlink (needs confirmation): {self.command.as_call(tool=self.tool)} "
                f"[device={self.command.device}, variable={self.command.variable}, "
                f"value={self.command.value if self.command.value is not None else 'null'}]"
            )
        if self.intent.question:
            return self.intent.question
        return (
            "I cannot turn that into a device command, and I will not guess at "
            "one. Rephrase it as `<verb> <variable> on|off [device <key>]`."
        )

    def confirm(self, token: str) -> "Proposal":
        """Record a confirmation attempt. Still executes nothing — the token is
        only *stored* here; :func:`execute_proposal` is what compares it."""
        return replace(self, confirmed_token=token)


def _command_text(intent: ControlIntent, tool: str = "set_variable") -> str:
    """Render the call shape used for the grounding re-check."""
    command = intent.to_command()
    if command is None:  # pragma: no cover - guarded by the caller
        return ""
    return command.as_call(tool=tool)


def propose(prompt: str, *, table: type = spec) -> Proposal:
    """Parse ``prompt`` and render the tool call it would be — **executing nothing**.

    This function has no parameter for a :class:`Downlink` and no way to reach
    one. Its only job is to turn text into a :class:`Proposal`, which is inert
    until :func:`execute_proposal` is handed the matching token. A proposal that
    is not actionable is returned with ``command=None`` and an explanatory
    ``text`` (the clarifying question, or the refusal), and every command-bearing
    proposal carries a fresh confirmation token.

    Before a command is exposed, the rendered call is re-checked through the
    shared grounding gate: if re-rendering ever produced a name the table does
    not carry, the proposal is downgraded to a refusal rather than surfaced.
    """
    intent = parse_intent(prompt, table=table)
    token = secrets.token_urlsafe(24)

    if not intent.actionable:
        return Proposal(intent=intent, command=None, token=token, flags=intent.flags)

    invented = assist.grounding_check(_command_text(intent), table=table)
    if invented:
        refused = replace(
            intent,
            outcome="refused",
            flags=tuple(intent.flags) + tuple(f"hallucination:{name}" for name in invented),
        )
        return Proposal(intent=refused, command=None, token=token, flags=refused.flags)

    return Proposal(intent=intent, command=intent.to_command(), token=token, flags=intent.flags)


def describe_command(command: DownlinkCommand) -> str:
    """One-line, user-facing description of a command (for logs and confirmations)."""
    payload = "null (clear)" if command.value is None else command.value
    return f"{command.variable}={payload} -> device {command.device}"


# ---------------------------------------------------------------------------
# Execution — the only crossing of the seam, and only on an exact "yes"
# ---------------------------------------------------------------------------


def _digestable(value: str) -> bytes:
    """Encode ``value`` for :func:`hmac.compare_digest`.

    ``compare_digest`` accepts only bytes or ASCII-only ``str``; encoding to
    UTF-8 first makes any string comparable (including a non-ASCII guess, which
    must be a *mismatch*, not a crash).
    """
    return value.encode("utf-8")

@dataclass(frozen=True)
class Execution:
    """The outcome of :func:`execute_proposal`.

    ``sent`` is the whole contract: it is ``True`` if and only if the injected
    downlink's ``send`` was called. ``reason`` is ``"ok"`` on success, or the
    machine-readable reason nothing happened. ``proposal`` is the (possibly
    advanced) proposal, so a caller can keep the "executed" flag on the record.
    """

    sent: bool
    reason: str
    proposal: Proposal
    result: object | None = None

    @property
    def ok(self) -> bool:
        return self.sent


def execute_proposal(
    proposal: Proposal,
    downlink: Downlink,
    confirmation: str | None = None,
    *,
    require_device_key: bool = False,
) -> Execution:
    """Execute ``proposal`` on ``downlink`` — but only against its exact token.

    The gate, in order, and every failure is a no-op (``send`` untouched):

    1. **Actionable** — a refused or ambiguous proposal (``command is None``)
       can never execute; reason ``"not_actionable"``.
    2. **Not already done** — a proposal marked ``executed`` is reason
       ``"already_executed"``, so a double-click cannot double-send.
    3. **Confirmation present** — a missing/empty ``confirmation`` is reason
       ``"confirmation_required"``. This is the acceptance's "without
       confirmation, nothing is executed".
    4. **Confirmation is *this* proposal's** — compared with
       :func:`hmac.compare_digest` against the token minted in :func:`propose`;
       a mismatch (including another proposal's token) is
       ``"confirmation_mismatch"``, and the seam stays untouched.
    5. **Optional device-key policy** — ``require_device_key=True`` refuses a
       command whose device name is not key-shaped (the wire carries a key, not
       a label), reason ``"device_not_keyed"``. Off by default so the common
       "turn the pump on" case works; a deploy that must not address the default
       key turns it on.

    Only after all five does it call ``downlink.send(command)`` exactly once and
    return ``Execution(sent=True, reason="ok", ...)``.
    """
    if proposal.command is None:
        return Execution(sent=False, reason="not_actionable", proposal=proposal)
    if proposal.executed:
        return Execution(sent=False, reason="already_executed", proposal=proposal)
    if not confirmation:
        return Execution(sent=False, reason="confirmation_required", proposal=proposal)
    if not isinstance(confirmation, str) or not hmac.compare_digest(
        _digestable(confirmation), _digestable(proposal.token)
    ):
        return Execution(
            sent=False,
            reason="confirmation_mismatch",
            proposal=proposal.confirm(str(confirmation)),
        )

    device = proposal.intent.device
    if require_device_key and (device is None or not device.stamped):
        return Execution(sent=False, reason="device_not_keyed", proposal=proposal)

    result = downlink.send(proposal.command)
    done = replace(proposal, executed=True, confirmed_token=confirmation)
    return Execution(sent=True, reason="ok", proposal=done, result=result)
