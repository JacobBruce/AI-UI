"""Native Utils pack - status, datetime, context usage, and hidden chat history."""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PACK_ID = "native:utils"
PACK_NAME = "Utils"
PACK_KIND = "native"

_MAX_HISTORY_MSGS = 32
_MAX_CATALOG_MSGS = 200
_MAX_HISTORY_CHARS = 100_000
_MAX_MSG_CHARS = 32_000

TOOLS = [
	{
		"name": "status",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Runtime snapshot of AI UI: app version, Python, workspace vs Home, "
			"text backend, tool-use flag, enabled packs, running MCP servers, and "
			"regex (live glob_files / grep_files: engine, syntax, glob, ignore, "
			"binary, default_exclude). "
			"No arguments. Does not include API keys or secrets. "
			"Call this first when diagnosing environment or tool-pack issues, "
			"or when you need the glob/grep regex dialect."
		),
		"inputSchema": {
			"type": "object",
			"properties": {},
			"additionalProperties": False,
		},
	},
	{
		"name": "get_datetime",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Current date and time: ISO 8601 datetime, date, time, timezone name, "
			"UTC offset, weekday, and Unix timestamp. Defaults to the system local "
			"timezone. Pass an IANA name such as 'America/New_York' or 'UTC'."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"timezone": {
					"type": "string",
					"description": "IANA timezone name (e.g. 'UTC', 'Australia/Sydney'). Empty = system local.",
					"default": "",
				},
			},
			"additionalProperties": False,
		},
	},
	{
		"name": "context_usage",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Last-known chat token usage vs the Context Token Budget, plus how much "
			"stored history sits outside the current context window. Uses stored "
			"provider/local counts from the latest model reply — does not recount "
			"and does not call a remote token API. After tools, stale is true until "
			"the next model response. budget 0 means context compaction is off and old "
			"turns may have been pruned. hidden / index_range are for chat_history "
			"(-1 = newest hidden message). Does not include API keys."
		),
		"inputSchema": {
			"type": "object",
			"properties": {},
			"additionalProperties": False,
		},
	},
	{
		"name": "chat_history",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Read stored chat messages that are outside the current context window "
			"(compacted or dropped by the keep-tail cap). Prefer this tool over reading "
			"chat.json. Indices are negative only: -1 is the newest hidden message, "
			"more negative is older. start and end are inclusive (end defaults to start). "
			"start may be omitted when query is set (search all hidden messages). "
			"Call context_usage first for the valid index_range. catalog=true returns "
			"role/timestamp/chars without bodies so you can pick a range to fetch. "
			"query is an optional case-insensitive substring filter: a string or list "
			"of strings; checks content and tool name, role is not searched."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"start": {
					"type": "integer",
					"description": "Inclusive start index (negative; -1 = newest hidden). Required unless query is set.",
				},
				"end": {
					"type": "integer",
					"description": "Inclusive end index (negative). Defaults to start.",
				},
				"catalog": {
					"type": "boolean",
					"description": "If true, return metadata only (no message bodies).",
					"default": False,
				},
				"query": {
					"type": ["string", "array"],
					"items": {"type": "string"},
					"description": (
						"Optional case-insensitive substring(s). Pass a string or a list of "
						"strings. A message matches if any term appears in its content or tool name."
					),
				},
			},
			"additionalProperties": False,
		},
	},
]


def configure(_cfg=None):
	return None


def list_tools():
	return list(TOOLS)


def _grep_engine_info():
	try:
		from lib.native_tools.coding_pack import RegexEngineInfo
		return RegexEngineInfo()
	except Exception:
		return {"engine": "python", "syntax": "python-re", "glob": "stdlib", "ignore": "none"}


def _engine():
	from lib.tool_client import E
	return E


def _status():
	from lib import mcp_client

	E = _engine()
	ws = ""
	home = ""
	version = ""
	backend = "local"
	remote_model = ""
	remote_provider = ""
	remote_base_url = ""
	tool_use = False
	agent = ""
	packs = []
	if E is not None:
		ws = str(getattr(E, "workspace_dir", "") or "").strip()
		home = str(getattr(E, "aiui_home", "") or "").strip()
		version = str(getattr(E, "app_version", "") or "").strip()
		backend = str(getattr(E, "text_backend", "local") or "local")
		remote_model = str(getattr(E, "remote_model", "") or "")
		remote_provider = str(getattr(E, "remote_provider", "") or "")
		remote_base_url = str(getattr(E, "remote_base_url", "") or "")
		tool_use = bool(int(getattr(E, "enable_tooluse", 0) or 0))
		agent = str(getattr(E, "active_agent", "") or "")
		raw_packs = getattr(E, "tool_packs", None) or []
		if isinstance(raw_packs, list):
			for p in raw_packs:
				if not isinstance(p, dict):
					continue
				packs.append({
					"id": str(p.get("id", "") or ""),
					"kind": str(p.get("kind", "") or ""),
					"name": str(p.get("name", "") or p.get("id") or ""),
					"enabled": bool(p.get("enabled", False)),
				})
	if not home:
		home = os.path.join(os.path.expanduser("~"), ".aiui")
	root = ws if ws else home
	files_root = os.path.abspath(os.path.expanduser(root))
	try:
		cwd = os.getcwd()
	except OSError:
		cwd = ""
	py = sys.version_info
	out = {
		"version": version,
		"python": f"{py.major}.{py.minor}.{py.micro}",
		"python_executable": sys.executable,
		"platform": sys.platform,
		"cwd": cwd,
		"workspace": files_root,
		"aiui_home": home,
		"agent": agent,
		"text_backend": backend,
		"tool_use": tool_use,
		"packs": packs,
		"mcp_running": mcp_client.running_ids(),
		"mcp_errors": mcp_client.error_map(),
		"regex": _grep_engine_info(),
	}
	if backend == "remote":
		out["remote_model"] = remote_model
		out["remote_provider"] = remote_provider
		out["remote_base_url"] = remote_base_url
	return out


def _get_datetime(timezone=""):
	tz_name = str(timezone or "").strip()
	if tz_name:
		try:
			tz = ZoneInfo(tz_name)
		except ZoneInfoNotFoundError:
			return {
				"error": (
					f"Unknown timezone {tz_name!r}. "
					"Use an IANA name such as 'America/New_York' or 'UTC'."
				)
			}
		now = datetime.now(tz)
	else:
		now = datetime.now().astimezone()
	return {
		"datetime": now.isoformat(),
		"date": now.date().isoformat(),
		"time": now.time().strftime("%H:%M:%S"),
		"timezone": str(now.tzinfo),
		"utc_offset": now.strftime("%z"),
		"weekday": now.strftime("%A"),
		"unix": now.timestamp(),
	}


def _context_usage():
	from lib import context_compact

	E = _engine()
	budget = 0
	gen_max = 32000
	prompt = None
	completion = None
	stale = True
	compact_on = False
	if E is not None:
		budget = int(getattr(E, "compact_at", 0) or 0)
		gen_max = int(getattr(E, "gen_max", 32000) or 32000)
		raw_p = getattr(E, "last_usage_prompt", None)
		raw_c = getattr(E, "last_usage_completion", None)
		stale = bool(getattr(E, "last_usage_stale", True))
		compact_on = context_compact.CompactEnabled()
		if raw_p is not None:
			prompt = int(raw_p)
		if raw_c is not None:
			completion = int(raw_c)
	total = None
	if prompt is not None:
		total = prompt + int(completion or 0)
	free = None
	if budget > 0 and total is not None:
		free = max(0, budget - total)
	hist = context_compact.HiddenHistoryIndexRange() if E is not None else {
		"hidden": 0, "index_range": None,
	}
	return {
		"budget": budget,
		"gen_max": gen_max,
		"prompt_tokens": prompt,
		"completion_tokens": completion,
		"total": total,
		"free": free,
		"stale": stale,
		"compact_enabled": compact_on,
		"hidden": hist["hidden"],
		"index_range": hist["index_range"],
	}


def _as_index(value, name):
	if isinstance(value, bool) or value is None:
		return None, f"{name} must be a negative integer"
	if isinstance(value, float) and value.is_integer():
		value = int(value)
	if not isinstance(value, int):
		return None, f"{name} must be a negative integer"
	return value, None


def _incl_slice(seq, start, end):
	if end == -1:
		return seq[start:]
	return seq[start:end + 1]


def _history_entry(msg, index, catalog, char_budget):
	from lib.context_compact import ContentToText

	text = ContentToText(msg.get("content"))
	chars = len(text)
	item = {
		"index": index,
		"role": str(msg.get("role") or ""),
		"timestamp": str(msg.get("timestamp") or ""),
		"chars": chars,
	}
	if catalog:
		return item, 0
	body = text
	truncated = False
	if char_budget is not None and char_budget <= 0:
		item["content"] = ""
		item["truncated"] = True
		return item, 0
	limit = _MAX_MSG_CHARS
	if char_budget is not None:
		limit = min(limit, char_budget)
	if len(body) > limit:
		body = body[:limit]
		truncated = True
	item["content"] = body
	if truncated:
		item["truncated"] = True
	return item, len(body)


def _norm_query(raw):
	if raw is None or raw is False:
		return [], None
	if isinstance(raw, str):
		from lib.tool_calls import TryParseJsonArray
		parsed = TryParseJsonArray(raw)
		if parsed is not None:
			parts = list(parsed)
		else:
			parts = [raw]
	elif isinstance(raw, (list, tuple)):
		parts = list(raw)
	else:
		return None, "query must be a string or a list of strings"
	terms = []
	seen = set()
	for p in parts:
		if isinstance(p, bool) or p is None:
			continue
		s = str(p).strip()
		if not s:
			continue
		key = s.casefold()
		if key in seen:
			continue
		seen.add(key)
		terms.append(s)
	return terms, None


def _query_hits(msg, terms):
	if not terms:
		return True
	from lib.context_compact import ContentToText
	parts = [ContentToText(msg.get("content"))]
	name = msg.get("name")
	if name:
		parts.append(str(name))
	for call in msg.get("tool_calls") or []:
		if not isinstance(call, dict):
			continue
		fn = call.get("function") if isinstance(call.get("function"), dict) else call
		if isinstance(fn, dict) and fn.get("name"):
			parts.append(str(fn.get("name")))
	blob = "\n".join(parts).casefold()
	return any(t.casefold() in blob for t in terms)


def _chat_history(start=None, end=None, catalog=False, query=None):
	from lib import context_compact

	terms, err = _norm_query(query)
	if err:
		return {"error": err}
	hidden = context_compact.HiddenHistoryMessages()
	n = len(hidden)
	index_range = {"start": -n, "end": -1} if n else None
	if start is None:
		if not terms:
			return {
				"error": "start is required unless query is set",
				"index_range": index_range,
			}
		if end is not None:
			return {
				"error": "end requires start",
				"index_range": index_range,
			}
		start_i = -n if n else -1
		end_i = -1
	else:
		start_i, err = _as_index(start, "start")
		if err:
			return {"error": err, "index_range": index_range}
		if end is None:
			end_i = start_i
		else:
			end_i, err = _as_index(end, "end")
			if err:
				return {"error": err, "index_range": index_range}
		if start_i >= 0 or end_i >= 0:
			return {
				"error": "indices must be negative (-1 is the newest hidden message)",
				"index_range": index_range,
			}
	if n <= 0:
		if not context_compact.CompactEnabled():
			return {
				"error": "no hidden history (context compaction is off; old turns are pruned)",
				"hidden": 0,
				"index_range": None,
			}
		return {
			"error": "no hidden history (nothing compacted yet)",
			"hidden": 0,
			"index_range": None,
		}
	if start is not None and (start_i < -n or end_i < -n):
		return {
			"error": f"index out of range (valid {-n} to -1)",
			"hidden": n,
			"index_range": index_range,
		}
	lo, hi = (end_i, start_i) if start_i > end_i else (start_i, end_i)
	rel_indices = list(range(lo, 0 if hi == -1 else hi + 1))
	picked = _incl_slice(hidden, lo, hi)
	if terms:
		filtered = []
		for rel, msg in zip(rel_indices, picked):
			if _query_hits(msg, terms):
				filtered.append((rel, msg))
		rel_indices = [p[0] for p in filtered]
		picked = [p[1] for p in filtered]
	want_catalog = bool(catalog)
	cap = _MAX_CATALOG_MSGS if want_catalog else _MAX_HISTORY_MSGS
	truncated = False
	matched = len(picked)
	if len(picked) > cap:
		picked = picked[-cap:]
		rel_indices = rel_indices[-cap:]
		truncated = True
	messages = []
	used = 0
	for rel, msg in zip(rel_indices, picked):
		if (not want_catalog) and used >= _MAX_HISTORY_CHARS:
			truncated = True
			break
		budget = None if want_catalog else max(0, _MAX_HISTORY_CHARS - used)
		entry, nchars = _history_entry(msg, rel, want_catalog, budget)
		messages.append(entry)
		used += nchars
	if any(m.get("truncated") for m in messages):
		truncated = True
	out = {
		"start": lo,
		"end": hi,
		"catalog": want_catalog,
		"hidden": n,
		"index_range": index_range,
		"messages": messages,
	}
	if terms:
		out["query"] = terms
		out["matched"] = matched
	if truncated:
		out["truncated"] = True
	return out


def call_tool(name, arguments=None, aiui_funcs=None):
	arguments = arguments or {}
	try:
		if name == "status":
			return _status()
		if name == "get_datetime":
			return _get_datetime(**arguments)
		if name == "context_usage":
			return _context_usage()
		if name == "chat_history":
			return _chat_history(**arguments)
		return {"error": "unknown function"}
	except Exception as e:
		return {"error": f"an exception occured ({e})"}
