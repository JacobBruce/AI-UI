# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

import json

from lib import mcp_client
from lib.native_tools import NATIVE_PACKS

# Shared prefix for all native packs (avoids collisions with MCP server tools).
NATIVE_PREFIX = "aiui_"

def PrefixedNativeName(bare):
	return NATIVE_PREFIX + str(bare or "")

def BareNativeName(name):
	n = str(name or "")
	if n.startswith(NATIVE_PREFIX):
		return n[len(NATIVE_PREFIX):]
	return n

def _native_expose(tool):
	"""Copy tool schema with aiui_ prefix for the model / catalog."""
	item = dict(tool)
	bare = item.get("name") or ""
	item["_native_name"] = bare
	item["name"] = PrefixedNativeName(bare)
	return item

def _name_disabled(disabled, *names):
	for n in names:
		if n and n in disabled:
			return True
	return False

def _default_packs():
	return [
		{"id": "native:utils", "kind": "native", "name": "Utils", "enabled": True, "disabled_tools": [], "permission_mode": "auto"},
		{"id": "native:math", "kind": "native", "name": "Math", "enabled": True, "disabled_tools": [], "permission_mode": "ask"},
		{"id": "native:web", "kind": "native", "name": "Web", "enabled": True, "disabled_tools": [], "permission_mode": "ask"},
		{
			"id": "native:coding",
			"kind": "native",
			"name": "Coding",
			"enabled": True,
			"disabled_tools": ["run_command"],
			"disabled_defaults_applied": ["run_command"],
			"permission_mode": "ask",
			"sandbox": True,
		},
		{"id": "native:image", "kind": "native", "name": "Image", "enabled": True, "disabled_tools": [], "permission_mode": "ask"},
	]

def _apply_opt_in_disabled(item):
	"""One-time: keep DEFAULT_DISABLED_TOOLS off until the user enables them."""
	if not item or item.get("id") != "native:coding":
		return item
	mod = NATIVE_PACKS.get("native:coding")
	defaults = list(getattr(mod, "DEFAULT_DISABLED_TOOLS", None) or ["run_command"])
	disabled = list(item.get("disabled_tools") or [])
	applied = [str(x) for x in (item.get("disabled_defaults_applied") or []) if str(x or "").strip()]
	for name in defaults:
		bare = str(name or "").strip()
		if not bare or bare in applied:
			continue
		pref = PrefixedNativeName(bare)
		if bare not in disabled and pref not in disabled:
			disabled.append(bare)
		applied.append(bare)
	item["disabled_tools"] = _norm_disabled(disabled)
	item["disabled_defaults_applied"] = applied
	# Keep display name in sync with pack module.
	if mod is not None and getattr(mod, "PACK_NAME", None):
		item["name"] = mod.PACK_NAME
	return item


def _migrate_packs_from_tool_mode(tool_mode):
	"""Best-effort migration from legacy tools_file / tool_mode."""
	packs = _default_packs()
	mode = str(tool_mode or "custom")
	enable_math = mode in ("math", "merge", "custom")
	enable_web = mode in ("research", "merge", "custom")
	for p in packs:
		if p["id"] == "native:math":
			p["enabled"] = enable_math
		elif p["id"] == "native:web":
			p["enabled"] = enable_web
	return packs

def _norm_disabled(raw):
	if not isinstance(raw, list):
		return []
	out = []
	seen = set()
	for name in raw:
		n = str(name or "").strip()
		if not n or n in seen:
			continue
		seen.add(n)
		out.append(n)
	return out

def NormalizeToolPacks(raw_packs, tool_mode=None):
	if isinstance(raw_packs, list) and len(raw_packs) > 0:
		out = []
		for p in raw_packs:
			if not isinstance(p, dict):
				continue
			item = dict(p)
			item["id"] = str(item.get("id", "") or "")
			item["kind"] = str(item.get("kind", "") or "")
			item["name"] = str(item.get("name", "") or item["id"])
			item["enabled"] = bool(item.get("enabled", False))
			item["disabled_tools"] = _norm_disabled(
				item.get("disabled_tools", item.get("disabledTools"))
			)
			pm = str(item.get("permission_mode", item.get("permissionMode", "ask")) or "ask").strip().lower()
			item["permission_mode"] = "auto" if pm == "auto" else "ask"
			if item.get("id") == "native:coding":
				if "sandbox" in item:
					item["sandbox"] = bool(item.get("sandbox"))
				elif "sandbox_enabled" in item:
					item["sandbox"] = bool(item.get("sandbox_enabled"))
				else:
					item["sandbox"] = True
			_apply_opt_in_disabled(item)
			if item["kind"] == "mcp":
				item["command"] = str(item.get("command", "") or "")
				args = item.get("args") or []
				if isinstance(args, str):
					try:
						args = json.loads(args) if args.strip() else []
					except Exception:
						args = []
				item["args"] = list(args) if isinstance(args, list) else []
				env = item.get("env") or {}
				if isinstance(env, str):
					try:
						env = json.loads(env) if env.strip() else {}
					except Exception:
						env = {}
				item["env"] = dict(env) if isinstance(env, dict) else {}
				item["cwd"] = str(item.get("cwd", "") or "")
				item["server_id"] = str(item.get("server_id", "") or item["id"].replace("mcp:", "", 1))
			if any(x.get("id") == item["id"] for x in out):
				continue
			out.append(item)
		if out:
			have = {item["id"] for item in out}
			missing = [d for d in _default_packs() if d["id"] not in have]
			if missing:
				out = missing + out
			return out
	return _migrate_packs_from_tool_mode(tool_mode)

def _schema_for_prompt(tool):
	"""MCP-shaped tool -> transformers-style name/description/parameters dict."""
	params = tool.get("inputSchema") or tool.get("parameters") or {"type": "object", "properties": {}}
	return {
		"name": tool["name"],
		"description": tool.get("description", "") or "",
		"parameters": params,
	}

def _schema_for_hf(tool):
	"""OpenAI tools= format for apply_chat_template."""
	params = tool.get("inputSchema") or tool.get("parameters") or {"type": "object", "properties": {}}
	return {
		"type": "function",
		"function": {
			"name": tool["name"],
			"description": tool.get("description", "") or "",
			"parameters": params,
		},
	}

def _disabled_set(pack):
	return set(_norm_disabled(pack.get("disabled_tools") or pack.get("disabledTools")))

def _enabled_native_modules(packs):
	mods = []
	for p in packs:
		if not p.get("enabled"):
			continue
		if p.get("kind") != "native":
			continue
		mod = NATIVE_PACKS.get(p.get("id"))
		if mod is not None:
			mods.append((p, mod))
	return mods

def _enabled_mcp_cfgs(packs):
	return [p for p in packs if p.get("enabled") and p.get("kind") == "mcp"]

def _find_pack(packs, pack_id):
	for p in packs:
		if p.get("id") == pack_id:
			return p
	return None

def _catalog_entry(tools, disabled):
	out = []
	for tool in tools:
		name = tool.get("name")
		if not name:
			continue
		bare = tool.get("_native_name") or tool.get("_mcp_name") or ""
		# Prefixed names; also honor legacy bare names in disabled_tools.
		is_on = not _name_disabled(disabled, name, bare)
		entry = {
			"name": name,
			"description": tool.get("description", "") or "",
			"enabled": is_on,
		}
		hint = tool.get("readOnlyHint", None)
		if hint is None:
			ann = tool.get("annotations")
			if isinstance(ann, dict) and "readOnlyHint" in ann:
				hint = ann.get("readOnlyHint")
		if hint is not None:
			entry["readOnlyHint"] = bool(hint)
		out.append(entry)
	return out

def _mcp_schema_disabled(pack, schema):
	disabled = _disabled_set(pack)
	name = schema.get("name") or ""
	bare = schema.get("_mcp_name") or ""
	return _name_disabled(disabled, name, bare)

def _emit_catalog(catalog):
	payload = {
		"tools": catalog,
		"mcp_running": mcp_client.running_ids(),
		"mcp_errors": mcp_client.error_map(),
	}
	E.PrintSleep("TOOL_CATALOG:" + json.dumps(payload, separators=(",", ":")).replace("\n", "[AI_UI_BR]"))

def BuildPackCatalog(packs=None):
	"""Full catalog for UI toggles (includes disabled tools)."""
	packs = packs if packs is not None else (getattr(E, "tool_packs", None) or [])
	packs = NormalizeToolPacks(packs, getattr(E, "tool_mode", None))
	fc_key = str(getattr(E, "firecrawl_api_key", "") or "")
	web_cfg = {"firecrawl_api_key": fc_key}
	catalog = {}

	for p in packs:
		pid = p.get("id")
		disabled = _disabled_set(p)
		if p.get("kind") == "native":
			mod = NATIVE_PACKS.get(pid)
			if mod is None:
				catalog[pid] = []
				continue
			if hasattr(mod, "configure"):
				mod.configure(web_cfg)
			# Declared tools (not filtered by Firecrawl availability) for toggle UI.
			tools = [_native_expose(t) for t in mod.list_tools()]
			catalog[pid] = _catalog_entry(tools, disabled)
		elif p.get("kind") == "mcp":
			# Live session schemas, else last list captured when the server started.
			schemas = mcp_client.list_pack_schemas(pid, use_cache=True)
			catalog[pid] = _catalog_entry(schemas, disabled)
		else:
			catalog[pid] = []
	return catalog

def _InputSchemaFor(func_name):
	"""inputSchema / parameters for a dispatched tool name, or {}."""
	name = str(func_name or "")
	bare = BareNativeName(name)
	prefixed = PrefixedNativeName(bare)
	for t in getattr(E, "tool_funcs", None) or []:
		if not isinstance(t, dict):
			continue
		fn = t.get("function") if t.get("type") == "function" else t
		if not isinstance(fn, dict):
			continue
		if fn.get("name") in (name, bare, prefixed):
			return fn.get("parameters") or fn.get("inputSchema") or {}
	packs = getattr(E, "tool_packs", None) or []
	for _p, mod in _enabled_native_modules(packs):
		for tool in mod.list_tools():
			if tool.get("name") in (name, bare):
				return tool.get("inputSchema") or {}
	try:
		for tool in mcp_client.list_enabled_schemas():
			if tool.get("name") in (name, prefixed):
				return tool.get("inputSchema") or tool.get("parameters") or {}
	except Exception:
		pass
	return {}


def DispatchToolFunc(func_name, func_args, aiui_funcs=None):
	"""Run a tool with no permission prompt (caller already gated)."""
	from lib.tool_calls import CoerceArgsToSchema

	func_args = CoerceArgsToSchema(func_args or {}, _InputSchemaFor(func_name))
	packs = getattr(E, "tool_packs", None) or []
	native_bare = BareNativeName(func_name)
	for p, mod in _enabled_native_modules(packs):
		disabled = _disabled_set(p)
		if _name_disabled(disabled, func_name, native_bare):
			continue
		names = {t["name"] for t in mod.list_tools()}
		if native_bare in names:
			try:
				return mod.call_tool(native_bare, func_args or {}, aiui_funcs=aiui_funcs)
			except TypeError:
				# Packs that do not accept aiui_funcs yet.
				return mod.call_tool(native_bare, func_args or {})

	allowed_mcp = []
	for p in _enabled_mcp_cfgs(packs):
		# Disabled check uses prefixed name (and legacy bare via schema match in mcp_client).
		if func_name in _disabled_set(p):
			continue
		# Also skip if disabled under bare suffix for this server.
		sid = str(p.get("server_id") or str(p.get("id") or "").replace("mcp:", "", 1) or "")
		prefix = mcp_client.SanitizeId(sid) + "_"
		if func_name.startswith(prefix) and func_name[len(prefix):] in _disabled_set(p):
			continue
		allowed_mcp.append(p.get("id"))
	if allowed_mcp:
		mcp_result = mcp_client.call_tool(func_name, func_args or {}, allowed_pack_ids=allowed_mcp)
		if mcp_result is not None:
			return mcp_result
	return "ERROR: unknown function"


def CallToolFunc(func_name, func_args, aiui_funcs=None):
	"""Dispatch a tool call among enabled packs/tools (with permission gate)."""
	from lib import tool_permissions

	if not tool_permissions.RequirePermission(func_name, func_args or {}):
		return "ERROR: tool call denied by user"
	return DispatchToolFunc(func_name, func_args, aiui_funcs=aiui_funcs)

def LoadToolFuncs(force_mcp_ids=None):
	"""Load schemas for enabled packs and (re)connect MCP servers.

	MCP tool lists are refreshed when a server starts or restarts - not on UI browse.
	"""
	packs = NormalizeToolPacks(
		getattr(E, "tool_packs", None),
		getattr(E, "tool_mode", None),
	)
	E.tool_packs = packs

	fc_key = str(getattr(E, "firecrawl_api_key", "") or "")
	web_cfg = {"firecrawl_api_key": fc_key}

	prev_tools = getattr(E, "tool_funcs", None)
	try:
		prev_sig = json.dumps(prev_tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if prev_tools else ""
	except Exception:
		prev_sig = str(prev_tools or "")

	schemas = []
	if E.enable_tooluse:
		for p, mod in _enabled_native_modules(packs):
			if hasattr(mod, "configure"):
				mod.configure(web_cfg)
			disabled = _disabled_set(p)
			for tool in mod.list_tools():
				bare = tool.get("name") or ""
				if _name_disabled(disabled, bare, PrefixedNativeName(bare)):
					continue
				schemas.append(_native_expose(tool))

		mcp_cfgs = _enabled_mcp_cfgs(packs)
		errors = mcp_client.sync_servers(mcp_cfgs, force_ids=force_mcp_ids)
		for err in errors:
			E.PrintSleep("TOOL_WARN:MCP connect failed - " + str(err).replace("\n", "[AI_UI_BR]"))
		for tool in mcp_client.list_enabled_schemas():
			pack = _find_pack(packs, tool.get("_pack_id"))
			if pack and _mcp_schema_disabled(pack, tool):
				continue
			schemas.append(tool)

		E.tool_funcs = [_schema_for_hf(t) for t in schemas]
		E.tool_func = CallToolFunc
		# Schemas reach the model via tools= + chat template; format docs live in rules.
		tools_str = ",".join(json.dumps(_schema_for_prompt(t)) for t in schemas)
		E.PrintSleep("TOOL_FUNCS:" + tools_str.replace("\n", "[AI_UI_BR]"))
	else:
		mcp_client.close_all()
		E.tool_funcs = None
		E.tool_func = CallToolFunc
		E.PrintSleep("TOOL_FUNCS:")

	catalog = BuildPackCatalog(packs)
	E.tool_catalog = catalog
	_emit_catalog(catalog)

	try:
		cur_sig = json.dumps(E.tool_funcs, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if E.tool_funcs else ""
	except Exception:
		cur_sig = str(E.tool_funcs or "")
	if prev_sig != cur_sig:
		from lib import remote_tools
		remote_tools.WarnIfAnthropicPrefixMayBreak("tools")
