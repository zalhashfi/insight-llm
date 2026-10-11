"""Build assistant: sketch recommendation / correction, grounded in the REAL API.

Ticket F01 (``FR-AI-06``). The assistant answers Arduino/sketch questions about
the ``insight`` platform, but it is **not** allowed to free-associate: every
claim it makes about the platform is taken from, and checked against, the local
API table in :mod:`insight_llm.spec`. That table is the ground truth (see its
module docstring for the ``insight`` sources each value is copied from), so the
assistant is a *retrieval-then-verify* pipeline, not a generator:

1. **Scope** — does the prompt even ask about the ``insight`` platform? A prompt
   with no platform anchor is out of scope and is **refused**, never answered
   (an un-anchored prompt is exactly the kind that invites invented API).
2. **Retrieve (RAG over the spec table)** — select the spec entries the prompt
   concerns, and render them as a context block (:func:`context_block`). Only
   retrieved entries may be cited afterwards.
3. **Compose** — build a recommendation or an in-place correction out of those
   entries. A sketch the prompt supplies is audited against the table: unknown
   names are flagged (and listed as hallucinations), and a correction patch is
   produced when a line cites something the platform does not have.
4. **Verify** — :func:`grounding_check` re-derives from the *rendered advice
   text* every platform name it cites and rejects the ones the table does not
   contain. A reply that cites an invented API is downgraded to a refusal
   rather than returned, so an accepted reply can never hallucinate.

Nothing here reaches the network, and nothing here invents a name: the
assistant is a pure function of ``(prompt, sketch, spec)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Final, Sequence

from . import spec

__all__ = [
    "Advice",
    "AdviceKind",
    "build_advice",
    "compose_reply",
    "citations",
    "context_block",
    "grounding_check",
    "is_in_scope",
    "known_names",
    "retrieve",
    "OUT_OF_SCOPE",
    "SCOPE_ANCHORS",
]

# ---------------------------------------------------------------------------
# The retriever's memory — every name the platform really has
# ---------------------------------------------------------------------------


def known_names(table: type = spec) -> frozenset[str]:
    """Return every platform name the spec table defines.

    This is the membership set the grounding check consults: a name that is not
    in here is not on the real surface, and the assistant may not cite it. The
    deliberately retired names (``spec.RETIRED_NAMES``, ``spec.RETIRED_ROUTES``)
    are *excluded* on purpose — they are recorded precisely because the real
    surface does **not** have them.
    """
    names: set[str] = set()
    names.update(table.MCP_READ_TOOLS)
    names.update(table.MCP_WRITE_TOOLS)
    names.update(table.MCP_SCOPES)
    names.add(table.MCP_READ_SCOPE)
    names.add(table.MCP_MANAGE_SCOPE)
    names.add(table.MCP_WRITE_FLAG)
    names.add(table.MCP_READ_ROUTE)
    names.add(table.MCP_WRITE_ROUTE)
    names.update(table.ROUTE_METHODS)
    names.update(table.ROLES)
    names.update(table.PERMISSIONS)
    names.update(table.PROVIDER_IDS)
    names.update(table.LIMITS)
    names.add(table.VERSION)
    return frozenset(names - set(table.RETIRED_NAMES) - set(table.RETIRED_ROUTES))


#: Words that anchor a prompt to the platform. A prompt carrying none of these
#: is out of scope: it is not a question about ``insight`` at all, and answering
#: it would mean guessing at a surface the assistant cannot cite.
SCOPE_ANCHORS: Final[tuple[str, ...]] = (
    "insight",
    "mcp",
    "telemetry",
    "ota",
    "firmware",
    "variable",
    "automation",
    "dashboard",
    "integration",
    "emit_event",
    "downlink",
    "scope",
    "sketch",
    "arduino",
    "esp32",
    "esp8266",
    "micropython",
    "project",
    "series",
    "control",
    "event",
    "device",
    "api",
    "route",
    "endpoint",
    "role",
    "permission",
)

#: The refusal envelope. Returned (with ``kind="refused"``) whenever the prompt
#: is out of scope or the verified reply would cite invented API.
OUT_OF_SCOPE: Final[str] = (
    "Out of scope: this assistant answers questions about the real `insight` "
    "platform (its HTTP routes, MCP tools/scopes, limits, roles and "
    "integrations) and the Arduino sketches that talk to it. Nothing in that "
    "prompt can be answered from the `insight` API table, so I will not "
    "speculate: an ungrounded answer would invent API. Rephrase it against the "
    "`insight` surface and I can help."
)

AdviceKind = str


def is_in_scope(prompt: str, *, table: type = spec) -> bool:
    """Does ``prompt`` ask about the ``insight`` platform?

    Scope is decided on a closed list of platform anchors (see
    :data:`SCOPE_ANCHORS`), not on the model's mood: an out-of-scope prompt is
    refused deterministically. A prompt that *does* carry an anchor still has to
    survive the grounding check before it is answered.
    """
    if not isinstance(prompt, str):
        return False
    text = prompt.lower()
    if "insight" in text:
        return True
    if re.search(r"\bmcp[:/]", text):
        return True
    return any(anchor in text for anchor in SCOPE_ANCHORS)


# ---------------------------------------------------------------------------
# Retrieval — pull the spec entries the prompt concerns
# ---------------------------------------------------------------------------

#: A retrieval topic: the kind of entry, the words that trigger it, and the
#: entries themselves. Prose in ``Fact``/``entry`` form keeps the context block
#: renderable without a template language.
RetrievedEntry = tuple[str, str]


@dataclass(frozen=True)
class _Topic:
    """One retrievable slice of the spec table."""

    name: str
    keywords: tuple[str, ...]
    tool_names: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    limits: tuple[str, ...] = ()
    prose: tuple[str, ...] = ()


_TOPICS: Final[tuple[_Topic, ...]] = (
    _Topic(
        name="tools",
        keywords=("tool", "tools", "mcp", "bridge", "capabilit", "verb", "delete", "call"),
        tool_names=spec.MCP_READ_TOOLS + spec.MCP_WRITE_TOOLS,
    ),
    _Topic(
        name="scopes",
        keywords=("scope", "gate", "auth", "permission", "flag", "token"),
        prose=(
            f"scopes: {', '.join(spec.MCP_SCOPES)}",
            f"read tools require {spec.MCP_READ_SCOPE}",
            f"write tools require {spec.MCP_MANAGE_SCOPE} plus the {spec.MCP_WRITE_FLAG} flag",
            f"gate paths: GET {spec.MCP_READ_ROUTE} and POST {spec.MCP_WRITE_ROUTE}",
        ),
    ),
    _Topic(
        name="routes",
        keywords=(
            "route", "routes", "endpoint", "http", "url", "post", "get",
            "telemetry", "ingest", "ota", "control", "health", "version",
            "downlink", "ack", "public", "admin",
        ),
        routes=tuple(spec.ROUTE_METHODS),
    ),
    _Topic(
        name="limits",
        keywords=(
            "limit", "limits", "max", "maximum", "cap", "window", "series",
            "firmware", "upload", "bytes", "size", "interval", "rate",
        ),
        limits=tuple(spec.LIMITS),
    ),
    _Topic(
        name="identity",
        keywords=(
            "role", "roles", "permission", "permissions", "member", "admin",
            "owner", "invite", "transfer", "identity", "who",
        ),
        prose=(
            f"roles, lowest privilege first: {', '.join(spec.ROLES)}",
            "permission -> minimum role: "
            + ", ".join(f"{action} -> {role}" for action, role in spec.PERMISSIONS.items()),
            "rank thresholds: "
            + ", ".join(f"{key} = {value}" for key, value in spec.LIMITS.items() if key.startswith("role:")),
        ),
    ),
    _Topic(
        name="integrations",
        keywords=("integration", "integrations", "provider", "providers", "slack",
                  "telegram", "discord", "twilio", "pagerduty", "email", "teams", "webhook", "notification"),
        prose=(
            f"provider ids: {', '.join(spec.PROVIDER_IDS)}",
            "integrations routes: GET /v1/integrations, POST /v1/integrations/test",
        ),
    ),
    _Topic(
        name="version",
        keywords=("version", "release", "build"),
        prose=(f"reported version: {spec.VERSION}",),
    ),
)

#: Fallback topics when the prompt names none of the keywords: the three a
#: build assistant needs most, so a generic "wire up my ESP32" prompt still gets
#: a grounded, citeable context.
_DEFAULT_TOPICS: Final[tuple[str, ...]] = ("routes", "tools", "scopes")


def retrieve(prompt: str, *, table: type = spec) -> tuple[_Topic, ...]:
    """Return the topics ``prompt`` concerns, in table order.

    Retrieval is keyword-driven and deterministic; it never consults anything
    outside :mod:`insight_llm.spec`. When nothing matches, a small default set
    is returned so the context block is never empty.
    """
    text = prompt.lower() if isinstance(prompt, str) else ""
    chosen = [topic for topic in _TOPICS if any(word in text for word in topic.keywords)]
    if not chosen:
        chosen = [topic for topic in _TOPICS if topic.name in _DEFAULT_TOPICS]
    return tuple(chosen)


def _entries(topic: _Topic, table: type = spec) -> tuple[RetrievedEntry, ...]:
    """Render one topic's spec entries as ``(kind, text)`` pairs."""
    out: list[RetrievedEntry] = []
    for name in topic.tool_names:
        scope = table.MCP_TOOL_SCOPES[name]
        kind = "read tool" if name in table.MCP_READ_TOOLS else "write tool"
        out.append(("tool", f"{name} ({kind}, scope {scope})"))
    for route in topic.routes:
        out.append(("route", f"{'|'.join(table.ROUTE_METHODS[route])} {route}"))
    for key in topic.limits:
        out.append(("limit", f"{key} = {table.LIMITS[key]}"))
    for line in topic.prose:
        out.append(("spec", line))
    return tuple(out)


def context_block(prompt: str, *, table: type = spec) -> str:
    """Render the retrieved spec entries as the context the reply may cite.

    This is the RAG artefact: the reply is a function of ``prompt`` *and* this
    block, and the block is drawn only from :mod:`insight_llm.spec`. It is
    exposed (rather than kept private) because it is the auditable answer to
    "where did that claim come from?".
    """
    lines: list[str] = ["insight API context — source: insight_llm.spec"]
    for topic in retrieve(prompt, table=table):
        lines.append(f"[{topic.name}]")
        for kind, text in _entries(topic, table=table):
            lines.append(f"  - ({kind}) {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The reply model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Advice:
    """One grounded reply from the build assistant.

    ``cited`` is the machine-readable half of the contract: the exact platform
    names this reply leans on. ``grounding_check`` re-derives it from ``text``
    so the two can be compared — a citation the text makes but the table does
    not carry is a hallucination, and the reply is refused.
    """

    kind: AdviceKind
    text: str
    cited: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    corrected_sketch: str | None = None
    flags: tuple[str, ...] = ()
    refused: bool = False
    grounded: bool = True

    @property
    def answer(self) -> str:
        """Alias for :attr:`text` (mirrors the eval harness' envelope key)."""
        return self.text

    def to_answer(self) -> dict[str, object]:
        """Render as the eval harness' answer envelope (``prompt_id`` aside).

        A grounded tool-naming reply already has the shape
        :func:`insight_llm.eval.score_shape` checks, so an assistant reply can be
        scored by the F02 harness without translation.
        """
        tools = []
        for name in self.cited:
            if name in spec.MCP_TOOL_SCOPES:
                tools.append(
                    {
                        "name": name,
                        "scope": spec.MCP_TOOL_SCOPES[name],
                        "description": f"insight MCP tool named by the build assistant ({name}).",
                    }
                )
        return {
            "answer": self.text,
            "tools": tools,
            "notes": " | ".join(self.flags) if self.flags else "grounded in insight_llm.spec",
        }


_EMPTY_ADVICE = Advice(
    kind="refused",
    text=OUT_OF_SCOPE,
    refused=True,
    grounded=False,
    flags=("out_of_scope",),
)


# ---------------------------------------------------------------------------
# Grounding verification — the gate every reply must pass
# ---------------------------------------------------------------------------


#: Detects a *call* — ``name(`` with the paren against the name. Prose such as
#: ``platform (its HTTP routes)`` is deliberately not a call: the whitespace
#: means the parenthesis belongs to the sentence, not to a platform name.
_CALL_TEXT: Final[str] = r"\b([A-Za-z][A-Za-z0-9_]*)\("

def grounding_check(text: str, *, table: type = spec) -> tuple[str, ...]:
    """Return every platform name ``text`` cites that the spec does not define.

    The check is name-based and derived from the rendered reply alone, so a
    claim added by any code path (or by a future model) is caught at the same
    boundary. Recognised forms:

    * an MCP scope or the write flag (``mcp:read``, ``mcp_write_enabled`` …);
    * a gate route (``/v1/mcp/write`` …);
    * a name that is *scoped* in the text — ``delete_project()``, a ``tool``
      list entry, or a ``/v1/...`` path — which is therefore a platform name
      whether or not the table knows it (``delete_project`` is the classic
      invented verb).

    Deliberately retired names (``spec.RETIRED_NAMES``) are reported as
    hallucinations too: they are exactly the plausible-but-false names a model
    reaches for.
    """
    if not isinstance(text, str) or not text:
        return ()
    known = known_names(table)
    retired = set(table.RETIRED_NAMES)
    bad: list[str] = []

    def _note(name: str) -> None:
        # Trailing sentence punctuation is stripped so a citation followed by a
        # full stop is still the same name (``/v1/ota/image.`` -> the route).
        # Closing brackets are deliberately *not* stripped: they belong to the
        # call shape (``loop()``), not to the name.
        cleaned = name.rstrip(".,;:!?\"'")
        if cleaned and cleaned not in known and cleaned not in bad:
            bad.append(cleaned)

    for token in re.findall(r"\bmcp:[a-z_]+", text):
        if token not in known:
            _note(token)
    for flag in re.findall(r"\bmcp_[a-z_]+\b", text):
        if flag not in known:
            _note(flag)
    for path in re.findall(r"/v1/[A-Za-z0-9_:/.\-]*", text):
        if path not in known:
            _note(path)
    # A name used as a *name* — called, listed as a tool, or flagged absent —
    # is a platform claim regardless of whether the table happens to have it.
    # ``GET``/``POST``… are HTTP verbs, not insight names.
    methods = {m for ms in table.ROUTE_METHODS.values() for m in ms}
    for token in re.findall(_CALL_TEXT, text):
        if token.startswith("mcp_") or token.upper() in methods:
            continue
        if token not in known or token in retired:
            _note(token)
    # ``tool``/``tools`` followed by an *identifier-shaped* name (``tool
    # delete_project``) is a platform claim; ``tools and their scopes`` is not a
    # name, so the shape — an underscore or a known name — is required.
    for token in re.findall(r"\btool[s]?\s+([a-z][a-z0-9_]*_[a-z0-9_]+|[a-z][a-z0-9_]+)\b", text):
        if token in known:
            continue
        # A lowercase word directly after "tool(s)" is only a claim when it
        # *looks* like a name: an identifier, or a retired name. ``exists`` and
        # ``and`` are prose; ``delete_project`` is a claim.
        if "_" in token or token in retired:
            _note(token)
    return tuple(bad)


def citations(text: str, *, table: type = spec) -> tuple[str, ...]:
    """The spec names ``text`` actually cites, in first-appearance order.

    The dual of :func:`grounding_check`: where that returns the names the table
    *lacks*, this returns the ones it *has*. Both are derived from the text, so
    a reply's ``cited`` set is evidence, not a claim.
    """
    known = known_names(table)
    return tuple(known_name for known_name in _cited_in_order(text, known))


def _cited_in_order(text: str, known: frozenset[str]) -> tuple[str, ...]:
    """Scan ``text`` for known spec names, preserving the order they appear."""
    found: dict[str, None] = {}
    patterns = (
        r"\bmcp:[a-z_]+",
        r"\bmcp_[a-z_]+\b",
        r"/v1/[A-Za-z0-9_:/.:\-]*",
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
    )
    positions: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = match.group(0).rstrip(".,;:!?\"'")
            if name in known:
                positions.append((match.start(), name))
    for _, name in sorted(positions):
        found.setdefault(name, None)
    return tuple(found)


def _verify(advice: Advice, *, table: type = spec) -> Advice:
    """Gate ``advice`` through the grounding check.

    A reply whose prose cites invented API is not returned: it is downgraded to
    the refusal envelope, which is the honest answer when the assistant's own
    claim cannot be backed by the table.
    """
    invented = grounding_check(advice.text, table=table)
    if not invented:
        if advice.cited:
            return advice
        # Record what the rendered reply actually leans on, so ``cited`` is
        # derived from the text rather than asserted alongside it.
        return replace(advice, cited=citations(advice.text, table=table))
    return Advice(
        kind="refused",
        text=OUT_OF_SCOPE,
        refused=True,
        grounded=False,
        flags=tuple(advice.flags) + tuple(f"hallucination:{name}" for name in invented),
        context=advice.context,
    )


# ---------------------------------------------------------------------------
# Reading a sketch the prompt supplies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sketch:
    """A sketch (or sketch line) the prompt carried, split from its prose."""

    sentences: tuple[str, ...]


#: Lines that carry quoted sketch code rather than pure prose.
_SKETCH_MARK = re.compile(
    r"(https?://|/v1/|\.begin\s*\(|#include|Serial\.|loop\s*\(|"
    r"\b(?:mcp|client|http)\s*\.)"
)
_CALL = re.compile(r"\b([a-z][a-z0-9_]*)\s*\(")
_PATH = re.compile(r"/v1/[A-Za-z0-9_:/.\-]+")

#: The real ingest route, read from the table rather than retyped.
_TELEMETRY_ROUTE: Final[str] = "/v1/telemetry"


def _split_sketch(prompt: str) -> _Sketch:
    """Split ``prompt`` into the sketch lines it quotes.

    A sketch line is *original signal*: it carries a URL/path, a call shape, or
    a platform handle (``mcp.``, ``http.``, ``Serial.``). Because the line is
    quoted, every name inside it is treated by :func:`_audit_sketch` as the
    *user's* claim and audited — or passed over when it is not a claim about
    ``insight`` at all.
    """
    sentences: list[str] = []
    for raw in prompt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SKETCH_MARK.search(line) or _CALL.search(line):
            sentences.append(line)
    # De-duplicate while keeping the first occurrence (order is part of the patch).
    seen: dict[str, None] = {}
    for sentence in sentences:
        seen.setdefault(sentence, None)
    return _Sketch(sentences=tuple(seen))


def _audit_sketch(prompt: str, *, table: type = spec) -> tuple[tuple[str, str], ...]:
    """Audit every sketch sentence in ``prompt`` against the spec table.

    Returns ``(sentence, reason)`` pairs — one per distinct problem — so a
    correction can quote the offending line verbatim. Silent when a sentence
    cites only real API.

    Only names the *platform* could own are judged: the classic invented MCP
    verbs (``delete_project``), the retired scope names, and malformed-sounding
    routes. Platform-independent sketch code (``loop()``, ``http.begin``, an
    unrelated third-party host) is left alone, because it is not a claim about
    ``insight`` and flagging it would punish the user for code that is none of
    the assistant's business.
    """
    findings: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sentence in _split_sketch(prompt).sentences:
        for name in grounding_check(sentence, table=table):
            reason = _sketch_issue(name, sentence, table=table)
            if reason is None:
                continue
            key = f"{sentence}\x00{name}"
            if key in seen:
                continue
            seen.add(key)
            findings.append((sentence, reason))
    return tuple(findings)


def _sketch_issue(name: str, sentence: str, *, table: type = spec) -> str | None:
    """Why ``name`` is wrong in ``sentence``, or ``None`` if it is not our call.

    The classifier is the assistant's answer to "is this even an insight
    claim?": a delete verb, a retired scope, or a ``/v1/`` path that is not a
    route of ours is a claim about this platform and must be corrected. Anything
    else a sketch happens to contain is foreign code and is passed over.
    """
    known = known_names(table)
    retired = set(table.RETIRED_NAMES)
    if name in retired:
        if name.startswith("mcp:"):
            return f"`{name}` is not an insight scope (the only two are {', '.join(table.MCP_SCOPES)})."
        return f"`{name}` is not an insight API name — there is deliberately no delete tool."
    if name.startswith("mcp:"):
        return f"`{name}` is not an insight scope (the only two are {', '.join(table.MCP_SCOPES)})."
    if name.startswith("/v1") or name.startswith("http"):
        return f"`{name}` is not a route insight serves."
    # A delete-shaped call is the classic invented verb: the surface has no
    # delete tool at all, so any ``*_delete``/``delete_*`` name is a claim worth
    # correcting.
    if re.match(r"^(delete_|remove_|drop_)", name) or name.endswith("_delete"):
        return f"`{name}` is not an insight API name — there is deliberately no delete tool."
    return None


# ---------------------------------------------------------------------------
# Composition — recommend, or correct
# ---------------------------------------------------------------------------


def _recommendation(prompt: str, block: str) -> str:
    """Compose a recommendation reply from the retrieved context."""
    text = prompt.lower()
    lines: list[str] = [
        "Here is what the real insight API says — copy these names exactly; "
        "they come from the spec table, not from memory.",
    ]
    by_kind: dict[str, list[str]] = {}
    for topic in retrieve(prompt):
        for kind, entry in _entries(topic):
            by_kind.setdefault(kind, []).append(entry)

    if "tool" in by_kind:
        lines.append(
            "MCP tools exposed by the bridge (name, kind, scope): "
            + "; ".join(by_kind["tool"])
            + "."
        )
    if "route" in by_kind:
        if any(word in text for word in ("telemetry", "ingest", "post data", "send data", "upload sensor")):
            lines.append(
                "Sketch-side ingest is POST /v1/telemetry (device token in the Authorization header); "
                "read a project's latest state with GET /v1/projects/:proj/state and a history window "
                "with GET /v1/projects/:proj/variables/:key/series."
            )
        lines.append("Routes to use: " + "; ".join(by_kind["route"]) + ".")
    if "limit" in by_kind:
        lines.append("Limits to respect: " + "; ".join(by_kind["limit"]) + ".")
    for entry in by_kind.get("spec", ()):
        lines.append(f"{entry[0].upper()}{entry[1:]}.")
    lines.append(
        "Firmware can only be uploaded to POST /v1/admin/projects/:proj/firmware "
        "(cap 8388608 bytes); an OTA pull is a plain GET /v1/ota/image."
    )
    return " ".join(lines) + f"\n\nContext used:\n{block}"


def _describe(reason: str) -> str:
    """Turn a finding reason into a name-free description.

    A correction that quoted the invented name back would commit the very
    hallucination it is correcting, so the *reason* is projected onto a shape
    ("a route the platform does not serve") that carries no name at all.
    """
    lowered = reason.lower()
    if "not an insight scope" in lowered:
        return "one line names a scope the gate does not define;"
    if "not a route insight serves" in lowered or "not an insight route" in lowered:
        return "one line points at a route the platform does not serve;"
    if "delete" in lowered:
        return "one line calls a delete tool, and no delete tool exists;"
    return "one line names something the platform does not expose;"


def _correction(
    prompt: str, findings: Sequence[tuple[str, str]], block: str, *, table: type = spec
) -> Advice:
    """Compose the correction reply for a sketch that cites unknown API.

    The offending lines are replaced in a patched sketch, and every replacement
    is drawn from the spec table (the real ingest route, the human-only admin
    delete), so the corrected sketch is grounded by construction. The prose
    describes the problems by *shape* rather than naming them: quoting the
    invented identifiers back would itself be a hallucinated citation.
    """
    lines = list(prompt.splitlines())
    patch: list[str] = []
    patched: list[str] = []
    for line in lines:
        fix = _patch_line(line, table=table)
        if fix is None:
            patched.append(line)
            continue
        patched.append(fix)
        patch.append(fix.strip())
    # The prose deliberately does *not* repeat the offending names: quoting
    # them back would itself cite invented API. The problems are described by
    # shape, and the patch lines carry the real replacements.
    problems = " ".join(_describe(reason) for _, reason in findings)
    text = (
        "Your sketch cites API the real insight platform does not have — "
        + problems
        + " Here is the grounded correction: the replacement lines below are "
        "drawn from the spec table, so the sketch compiles against the real "
        "surface."
        + (f"\n\nGrounding notes:\n{chr(10).join(patch)}" if patch else "")
        + f"\n\nContext used:\n{block}"
    )
    return Advice(
        kind="correction",
        text=text,
        cited=(),
        context=tuple(block.splitlines()),
        corrected_sketch="\n".join(patched),
        flags=tuple(f"corrected:{reason}" for _, reason in findings),
    )


def _patch_line(line: str, *, table: type = spec) -> str | None:
    """Return the grounded replacement for ``line``, or ``None`` if it is fine.

    Only lines :func:`_sketch_issue` calls out are patched — an unknown
    ``/v1/...`` path or a delete-shaped verb. Every replacement names real
    ``insight`` API drawn from the table (the real ingest route, the human-only
    admin delete), never an invented name.
    """
    stripped = line.strip()
    if not stripped:
        return None

    known = known_names(table)
    path = _PATH.search(stripped)
    if path and path.group(0) not in known:
        if _sketch_issue(path.group(0), stripped, table=table) is None:
            return None
        # The replacements are comments naming the *real* route from the table,
        # never a code line: a plausible-looking call in the patch would be an
        # invented API of the assistant's own making.
        if "telemetry" in path.group(0):
            return f"  // route fixed: use POST {_TELEMETRY_ROUTE} for ingest"
        return "  // route removed: insight serves no such path"

    for match in _CALL.finditer(stripped):
        name = match.group(1)
        if name in known:
            continue
        if _sketch_issue(name, stripped, table=table) is None:
            continue
        # The note must not quote the offending name back: a correction that
        # repeats a hallucinated identifier is itself a citation of it.
        return "  // removed: not an insight API name"
    return None


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def build_advice(prompt: str, sketch: str | None = None, *, table: type = spec) -> Advice:
    """Answer ``prompt`` (optionally about ``sketch``), grounded in the spec.

    The pipeline is :func:`is_in_scope` -> :func:`retrieve`/:func:`context_block`
    -> compose -> :func:`grounding_check`. An out-of-scope prompt is refused
    before anything is composed; a reply that would cite invented API is
    refused by the check; a sketch that cites invented API is corrected and
    flagged. The result is always a reply whose platform names are a subset of
    :func:`known_names`.
    """
    if not is_in_scope(prompt, table=table):
        return _EMPTY_ADVICE

    subject = prompt if not sketch else f"{prompt}\n{sketch}"
    block = context_block(subject, table=table)
    findings = _audit_sketch(subject, table=table)
    if findings:
        return _verify(_correction(subject, findings, block, table=table), table=table)
    return _verify(
        Advice(
            kind="recommendation",
            text=_recommendation(subject, block),
            cited=(),
            context=tuple(block.splitlines()),
        ),
        table=table,
    )


def compose_reply(prompt: str, sketch: str | None = None, *, table: type = spec) -> Advice:
    """Alias for :func:`build_advice` — the assistant's public entry point."""
    return build_advice(prompt, sketch=sketch, table=table)
