"""Local spec of the REAL ``insight`` API — facts the eval checks answers against.

This module is the **ground truth** for the grounding metric. It is a hand-kept,
network-free transcription of the public surface of the ``insight`` platform as
that repository defines it, so an eval run never has to reach a live service (or
even a checkout of it) to know whether a fact is true.

Every value here is copied from the ``insight`` sources and cites the file it
came from, so a drift is auditable rather than mysterious:

* ``MCP_TOOLS`` / ``MCP_TOOL_SCOPES`` — ``worker/src/mcp/tools.ts`` (the frozen
  tool catalogue and the ``SPECS`` table) and ``worker/src/mcp/gate.ts`` (the
  two MCP scopes).
* ``ROUTE_METHODS`` — ``worker/src/app.ts`` (route mounts) and each domain's own
  ``create*`` router.
* ``PERMISSIONS`` / ``ROLES`` — ``worker/src/domains/projects/membership.ts``.
* ``PROVIDER_IDS`` — ``worker/src/shared/integrations/index.ts``.
* ``LIMITS`` — ``domains/telemetry/query.ts``, ``domains/firmware/upload.ts``.

Only facts that a model answer can be scored against live here: canonical
names, the exact set a name belongs to, and scalar limits. Anything a fact
cannot be checked against — prose, ordering of unrelated items — is deliberately
left out.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

__all__ = [
    "MCP_READ_TOOLS",
    "MCP_WRITE_TOOLS",
    "MCP_TOOL_SCOPES",
    "MCP_SCOPES",
    "MCP_READ_SCOPE",
    "MCP_MANAGE_SCOPE",
    "MCP_WRITE_FLAG",
    "MCP_READ_ROUTE",
    "MCP_WRITE_ROUTE",
    "ROUTE_METHODS",
    "ROLES",
    "PERMISSIONS",
    "PROVIDER_IDS",
    "LIMITS",
    "VERSION",
    "RETIRED_NAMES",
    "RETIRED_ROUTES",
]


def _frozen(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Freeze an ordered collection as a tuple (order is part of the spec)."""
    return tuple(values)


# ---------------------------------------------------------------------------
# MCP tool surface — worker/src/mcp/tools.ts
# ---------------------------------------------------------------------------

#: Read tools, in catalogue order (``READ_TOOLS``).
MCP_READ_TOOLS: Final[tuple[str, ...]] = _frozen(
    [
        "list_projects",
        "get_project",
        "list_variables",
        "get_state",
        "get_series",
        "list_dashboards",
        "get_dashboard",
        "list_automations",
        "get_automation",
        "list_integrations",
        "get_integration",
    ]
)

#: Write tools, in catalogue order (``WRITE_TOOLS``).
MCP_WRITE_TOOLS: Final[tuple[str, ...]] = _frozen(
    [
        "create_project",
        "update_project",
        "create_variable",
        "update_variable",
        "set_variable",
        "create_layout",
        "update_layout",
        "create_automation",
        "update_automation",
        "create_integration",
        "update_integration",
        "emit_event",
    ]
)

#: The two MCP scopes (``gate.ts``): ``mcp:read`` and ``mcp:manage`` — no third.
MCP_SCOPES: Final[tuple[str, ...]] = _frozen(["mcp:read", "mcp:manage"])
MCP_READ_SCOPE: Final[str] = "mcp:read"
MCP_MANAGE_SCOPE: Final[str] = "mcp:manage"

#: The flag the write path is additionally gated on (``gate.ts``).
MCP_WRITE_FLAG: Final[str] = "mcp_write_enabled"

#: The gate's two wire paths, relative to the ``/v1/mcp`` mount.
MCP_READ_ROUTE: Final[str] = "/v1/mcp/read"
MCP_WRITE_ROUTE: Final[str] = "/v1/mcp/write"

#: Scope required per tool. Reads take ``mcp:read``; writes take ``mcp:manage``.
MCP_TOOL_SCOPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        **{name: MCP_READ_SCOPE for name in MCP_READ_TOOLS},
        **{name: MCP_MANAGE_SCOPE for name in MCP_WRITE_TOOLS},
    }
)

# ---------------------------------------------------------------------------
# HTTP surface — worker/src/app.ts (+ each domain router)
# ---------------------------------------------------------------------------

#: ``path -> HTTP methods``, upper-cased. Only routes a fact is written about.
ROUTE_METHODS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "/healthz": ("GET",),
        "/v1/version": ("GET",),
        "/v1/telemetry": ("POST",),
        "/v1/projects/:proj/state": ("GET",),
        "/v1/projects/:proj/variables/:key/series": ("GET",),
        "/v1/events": ("POST",),
        "/v1/control": ("GET",),
        "/v1/control/ack": ("POST",),
        "/v1/control/ws": ("GET",),
        "/v1/ota": ("GET",),
        "/v1/ota/image": ("GET",),
        "/v1/admin/projects/:proj": ("DELETE",),
        "/v1/admin/projects/:proj/firmware": ("POST",),
        "/v1/admin/projects/:proj/automations/:id/run": ("POST",),
        "/v1/public/dashboards/:token": ("GET",),
        "/v1/public/dashboards/:token/stream": ("GET",),
        "/v1/mcp/read": ("GET",),
        "/v1/mcp/write": ("POST",),
        "/v1/integrations": ("GET",),
        "/v1/integrations/test": ("POST",),
        "/v1/external/poll": ("POST",),
    }
)

# ---------------------------------------------------------------------------
# Identity / authorization — worker/src/domains/projects/membership.ts
# ---------------------------------------------------------------------------

ROLES: Final[tuple[str, ...]] = _frozen(["member", "admin", "owner"])

#: The action -> minimum-role table (``PERMISSIONS``).
PERMISSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "project:read": "member",
        "project:update": "admin",
        "project:delete": "owner",
        "project:transfer": "owner",
        "member:invite": "admin",
        "member:remove": "admin",
        "member:change_role": "admin",
    }
)

# ---------------------------------------------------------------------------
# Integrations — worker/src/shared/integrations/index.ts
# ---------------------------------------------------------------------------

PROVIDER_IDS: Final[tuple[str, ...]] = _frozen(
    [
        "http_service",
        "email",
        "telegram",
        "slack",
        "discord",
        "twilio",
        "ms_teams",
        "pagerduty",
    ]
)

# ---------------------------------------------------------------------------
# Scalar limits
# ---------------------------------------------------------------------------

LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        # domains/telemetry/query.ts — MAX_WINDOW_MS / DEFAULT_WINDOW_MS.
        "series_max_window_ms": 30 * 24 * 60 * 60 * 1000,
        "series_default_window_ms": 60 * 60 * 1000,
        # domains/firmware/upload.ts — MAX_FIRMWARE_BYTES (8 MiB).
        "firmware_max_bytes": 8 * 1024 * 1024,
        # membership.ts — `roleRank` over ROLES, highest privilege last.
        "role:member_rank": 1,
        "role:admin_rank": 2,
        "role:owner_rank": 3,
        # membership.ts — the rank each permission requires (a projection of
        # PERMISSIONS through `roleRank`, so the two can never disagree).
        "permission:project:read": 1,
        "permission:project:update": 2,
        "permission:project:delete": 3,
        "permission:project:transfer": 3,
        "permission:member:invite": 2,
        "permission:member:remove": 2,
        "permission:member:change_role": 2,
    }
)

#: The version ``GET /v1/version`` reports (``DEFAULT_VERSION``).
VERSION: Final[str] = "0.0.1"

# ---------------------------------------------------------------------------
# Negative space — what does NOT exist
# ---------------------------------------------------------------------------

#: Names a model is tempted to invent that are deliberately absent: there is no
#: delete *tool* at all, and no separate write scope. A fact naming one of these
#: as real is false.
RETIRED_NAMES: Final[tuple[str, ...]] = _frozen(
    [
        "delete_project",
        "delete_variable",
        "delete_dashboard",
        "delete_automation",
        "delete_integration",
        "mcp:write",
        "mcp:admin",
    ]
)

#: Routes that do not exist on purpose (a delete stays on the admin, human-only
#: surface; MCP has no REST mirror of a tool).
RETIRED_ROUTES: Final[tuple[str, ...]] = _frozen(
    [
        "/v1/telemetry/:proj",
        "/v1/mcp/delete",
        "/v1/projects/:proj/delete",
    ]
)
