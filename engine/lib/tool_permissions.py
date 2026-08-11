"""Pack-level Ask / Auto-allow gates for native and MCP tools."""

import json
import threading
import time
import uuid

E = None

_lock = threading.Lock()
_pending = {}  # request_id -> {"event": Event, "action": str}

def bind(engine_module):
	global E
	E = engine_module

def _norm_mode(raw):
	mode = str(raw or "ask").strip().lower()
	return "auto" if mode == "auto" else "ask"

def GetPackPermissionMode(pack):
	if not pack:
		return "ask"
	return _norm_mode(pack.get("permission_mode", pack.get("permissionMode")))

def SetPackPermissionMode(pack_id, mode):
	"""Update in-memory pack permission_mode. Returns the pack dict or None."""
	packs = getattr(E, "tool_packs", None) or []
	mode = _norm_mode(mode)
	for p in packs:
		if p.get("id") == pack_id:
			p["permission_mode"] = mode
			return p
	return None

def FindPackForTool(tool_name):
	"""Return the enabled pack that owns tool_name, or None.

	Includes tools that are disabled within the pack so RequirePermission can
	deny them without treating them as unknown.
	"""
	from lib import mcp_client
	from lib.native_tools import NATIVE_PACKS
	from lib import tool_client

	packs = getattr(E, "tool_packs", None) or []
	name = str(tool_name or "")
	for p in packs:
		if not p.get("enabled"):
			continue
		if p.get("kind") == "native":
			mod = NATIVE_PACKS.get(p.get("id"))
			if mod is None:
				continue
			names = {t["name"] for t in mod.list_tools()}
			bare = tool_client.BareNativeName(name)
			if bare in names:
				return p
		elif p.get("kind") == "mcp":
			for schema in mcp_client.list_pack_schemas(p.get("id"), use_cache=True):
				if schema.get("name") == name:
					return p
			for schema in mcp_client.list_enabled_schemas():
				if schema.get("_pack_id") == p.get("id") and schema.get("name") == name:
					return p
	return None

def _ToolDisabledInPack(pack, tool_name):
	from lib import tool_client
	name = str(tool_name or "")
	disabled = tool_client._disabled_set(pack)
	if pack.get("kind") == "native":
		bare = tool_client.BareNativeName(name)
		return tool_client._name_disabled(
			disabled, name, bare, tool_client.PrefixedNativeName(bare),
		)
	return name in disabled

def ResolvePermission(request_id, action):
	action = str(action or "deny").strip().lower()
	if action not in ("deny", "allow", "auto"):
		action = "deny"
	with _lock:
		item = _pending.get(request_id)
		if not item:
			return False
		item["action"] = action
		item["event"].set()
	return True

def RequirePermission(tool_name, arguments=None):
	"""Block until the user allows/denies (or pack is auto).

	Returns True to run the tool, False if denied / stopped.
	When the user chooses Auto-allow, updates the pack in memory and returns True.
	Caller/host persist Auto-allow via the tool_permission_result / side event.
	"""
	from lib import ipc

	pack = FindPackForTool(tool_name)
	if pack is None:
		# Unknown tools fail later in dispatch; do not prompt.
		return True
	if _ToolDisabledInPack(pack, tool_name):
		# Disabled in pack settings — deny without a permission dialog.
		return False
	if GetPackPermissionMode(pack) == "auto":
		return True

	request_id = uuid.uuid4().hex
	event = threading.Event()
	with _lock:
		_pending[request_id] = {"event": event, "action": "deny"}

	args = arguments if isinstance(arguments, dict) else {}
	try:
		preview = json.dumps(args, ensure_ascii=False)
		if len(preview) > 100000:
			preview = preview[:100000] + "…"
	except Exception:
		preview = str(args)[:100000]

	ipc.SendEvent("tool_permission", {
		"request_id": request_id,
		"pack_id": pack.get("id"),
		"pack_name": pack.get("name") or pack.get("id"),
		"tool": tool_name,
		"arguments": args,
		"args_preview": preview,
	})

	deadline = time.time() + 3600
	while time.time() < deadline:
		if ipc.StopRequested():
			ResolvePermission(request_id, "deny")
			break
		if event.wait(timeout=0.15):
			break
	else:
		ResolvePermission(request_id, "deny")

	with _lock:
		item = _pending.pop(request_id, None)
	action = (item or {}).get("action", "deny")
	if action == "auto":
		SetPackPermissionMode(pack.get("id"), "auto")
		return True
	return action == "allow"
