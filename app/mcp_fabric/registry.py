"""
MCP Tool Fabric
-----------------
A permissioned tool registry mirroring the Model Context Protocol pattern:
agents discover and invoke tools (DB, S3-like storage, search, GitHub,
monitoring) only within their declared permission set. Swap `_TOOLS` and
`_PERMISSIONS` for real MCP server connections in production; the
call-site interface (`MCPFabric.call`) stays the same.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from app.models import Domain, MCPToolCallResult

# --------------------------------------------------------------------------
# Mock tool implementations (stand-ins for real MCP servers)
# --------------------------------------------------------------------------
_FAKE_DB: Dict[str, Dict[str, Any]] = {
    "user:1": {"name": "Ada Lovelace", "plan": "enterprise"},
    "user:2": {"name": "Alan Turing", "plan": "pro"},
}
_FAKE_STORE: Dict[str, str] = {}


def _tool_database(action: str, payload: Dict[str, Any]) -> Any:
    if action == "READ":
        key = payload.get("key", "")
        return _FAKE_DB.get(key, None)
    if action == "DELETE":
        raise PermissionError("DELETE on database is never permitted via this fabric.")
    raise ValueError(f"Unsupported database action: {action}")


def _tool_document_store(action: str, payload: Dict[str, Any]) -> Any:
    if action == "READ":
        return _FAKE_STORE.get(payload.get("key", ""), None)
    if action == "WRITE":
        _FAKE_STORE[payload["key"]] = payload.get("value", "")
        return {"status": "written"}
    raise ValueError(f"Unsupported document_store action: {action}")


def _tool_search(action: str, payload: Dict[str, Any]) -> Any:
    if action == "SEARCH":
        query = payload.get("query", "")
        return {"query": query, "results": [f"result-1 for '{query}'", f"result-2 for '{query}'"]}
    raise ValueError(f"Unsupported search action: {action}")


def _tool_github(action: str, payload: Dict[str, Any]) -> Any:
    if action == "READ":
        return {"repo": payload.get("repo"), "open_issues": 4, "stars": 128}
    raise ValueError(f"Unsupported github action: {action}")


def _tool_monitoring(action: str, payload: Dict[str, Any]) -> Any:
    if action == "READ":
        return {"cpu_pct": 34.2, "p99_latency_ms": 812, "error_rate": 0.004}
    raise ValueError(f"Unsupported monitoring action: {action}")


_TOOLS: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {
    "database": _tool_database,
    "document_store": _tool_document_store,
    "search": _tool_search,
    "github": _tool_github,
    "monitoring": _tool_monitoring,
}

# --------------------------------------------------------------------------
# Per-agent tool + action permissions, mirroring the design doc's example:
#   Research Agent: READ documents ✓, SEARCH knowledge ✓, READ database ✓,
#                    DELETE database ✗, MODIFY production ✗
# --------------------------------------------------------------------------
_PERMISSIONS: Dict[Domain, Dict[str, List[str]]] = {
    Domain.RESEARCH: {
        "document_store": ["READ"],
        "search": ["SEARCH"],
        "database": ["READ"],
    },
    Domain.CODING: {
        "github": ["READ"],
        "document_store": ["READ", "WRITE"],
    },
    Domain.REASONING: {
        "document_store": ["READ"],
        "search": ["SEARCH"],
    },
    Domain.SECURITY: {
        "monitoring": ["READ"],
        "database": ["READ"],
    },
    Domain.ANALYTICS: {
        "monitoring": ["READ"],
        "database": ["READ"],
    },
}


class MCPFabric:
    def call(self, agent: Domain, tool: str, action: str, payload: Dict[str, Any]) -> MCPToolCallResult:
        allowed_actions = _PERMISSIONS.get(agent, {}).get(tool, [])
        if action not in allowed_actions:
            return MCPToolCallResult(
                allowed=False,
                tool=tool,
                action=action,
                result=None,
                reason=f"Agent '{agent.value}' is not permitted to {action} on '{tool}'.",
            )

        handler = _TOOLS.get(tool)
        if handler is None:
            return MCPToolCallResult(
                allowed=False, tool=tool, action=action, reason=f"Unknown tool '{tool}'."
            )

        try:
            result = handler(action, payload)
        except (PermissionError, ValueError) as exc:
            return MCPToolCallResult(allowed=False, tool=tool, action=action, reason=str(exc))

        return MCPToolCallResult(allowed=True, tool=tool, action=action, result=result)

    def permissions_for(self, agent: Domain) -> Dict[str, List[str]]:
        return _PERMISSIONS.get(agent, {})


mcp_singleton = MCPFabric()
