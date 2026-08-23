"""Parse <tool_call> inner payloads into a list of {name, arguments}.

Supports:
- JSON object: {"name": "...", "arguments": {...}}
- JSON array:  [{"name": "...", "arguments": {...}}, ...]
- XML:         <function=name><parameter=k>v</parameter></function> (multi-function OK)
"""

from __future__ import annotations

import json
import re

_FUNC_OPEN = re.compile(r"<function\s*=\s*([^>\s]+)\s*>", re.IGNORECASE)
_FUNC_CLOSE = re.compile(r"</function\s*>", re.IGNORECASE)
_PARAM_OPEN = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>", re.IGNORECASE)
_PARAM_CLOSE = re.compile(r"</parameter\s*>", re.IGNORECASE)


def _UnwrapXmlValue(raw: str) -> str:
	"""Drop one wrapping newline after the open tag and before the close tag."""
	s = raw or ""
	if s.startswith("\r\n"):
		s = s[2:]
	elif s.startswith("\n") or s.startswith("\r"):
		s = s[1:]
	if s.endswith("\r\n"):
		s = s[:-2]
	elif s.endswith("\n") or s.endswith("\r"):
		s = s[:-1]
	return s


def _CoerceArgValue(raw: str):
	"""XML parameter text. Numbers / true / false / null become JSON types; else literal string.

	Does not parse JSON arrays/objects here — `{`/`[` stay strings so write_file can
	store JSON bodies. List/object params are coerced later from the tool schema.
	"""
	s = _UnwrapXmlValue(raw)
	if s in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", s):
		try:
			return json.loads(s)
		except Exception:
			pass
	return s


def TryParseJsonArray(raw):
	"""If raw is a JSON array string, return that list; otherwise None."""
	if not isinstance(raw, str):
		return None
	s = raw.strip()
	if not s.startswith("["):
		return None
	try:
		parsed = json.loads(s)
	except Exception:
		return None
	return parsed if isinstance(parsed, list) else None


def TryParseJsonObject(raw):
	"""If raw is a JSON object string, return that dict; otherwise None."""
	if not isinstance(raw, str):
		return None
	s = raw.strip()
	if not s.startswith("{"):
		return None
	try:
		parsed = json.loads(s)
	except Exception:
		return None
	return parsed if isinstance(parsed, dict) else None


def _SchemaTypes(prop):
	if not isinstance(prop, dict):
		return set()
	t = prop.get("type")
	if isinstance(t, str):
		return {t}
	if isinstance(t, (list, tuple)):
		return {str(x) for x in t if x}
	return set()


def CoerceArgsToSchema(args, schema):
	"""Turn XML JSON-array/object strings into lists/dicts when the schema says so.

	String-typed properties (file bodies) are left as strings.
	"""
	if not isinstance(args, dict):
		return args
	props = (schema or {}).get("properties") if isinstance(schema, dict) else None
	if not isinstance(props, dict):
		return args
	out = dict(args)
	for key, val in args.items():
		if not isinstance(val, str):
			continue
		types = _SchemaTypes(props.get(key))
		if "array" in types:
			parsed = TryParseJsonArray(val)
			if parsed is not None:
				out[key] = parsed
				continue
		if "object" in types:
			parsed = TryParseJsonObject(val)
			if parsed is not None:
				out[key] = parsed
	return out


def _NormalizeOne(obj) -> dict | None:
	if not isinstance(obj, dict):
		return None
	name = obj.get("name")
	if name is None and isinstance(obj.get("function"), dict):
		fn = obj["function"]
		name = fn.get("name")
		args = fn.get("arguments", fn.get("parameters", {}))
	else:
		args = obj.get("arguments", obj.get("parameters", {}))
	if name is None or name == "":
		return None
	if isinstance(args, str):
		try:
			args = json.loads(args) if args.strip() else {}
		except Exception:
			args = {"_raw": args}
	if not isinstance(args, dict):
		args = {}
	out = {"name": str(name), "arguments": args}
	tid = obj.get("id")
	if tid:
		out["id"] = str(tid)
	return out


def ParseXmlFunctions(raw: str) -> list[dict]:
	"""Extract one or more <function=…>…</function> blocks."""
	text = raw or ""
	out = []
	pos = 0
	while True:
		m_open = _FUNC_OPEN.search(text, pos)
		if not m_open:
			break
		name = m_open.group(1).strip()
		body_start = m_open.end()
		m_close = _FUNC_CLOSE.search(text, body_start)
		if not m_close:
			break
		body = text[body_start:m_close.start()]
		args = {}
		ppos = 0
		while True:
			pm = _PARAM_OPEN.search(body, ppos)
			if not pm:
				break
			pkey = pm.group(1).strip()
			pval_start = pm.end()
			pc = _PARAM_CLOSE.search(body, pval_start)
			if not pc:
				break
			args[pkey] = _CoerceArgValue(body[pval_start:pc.start()])
			ppos = pc.end()
		out.append({"name": name, "arguments": args})
		pos = m_close.end()
	return out


def ParseToolCallInner(raw: str) -> list[dict]:
	"""Normalize one <tool_call>…</tool_call> body to a list of calls.

	Form is the first non-whitespace character: `{`/`[` JSON, `<` XML.
	"""
	text = (raw or "").strip()
	if not text:
		return []

	if text[0] in "{[":
		try:
			data = json.loads(text)
		except Exception:
			return []
		if isinstance(data, dict):
			one = _NormalizeOne(data)
			return [one] if one else []
		if isinstance(data, list):
			out = []
			for item in data:
				one = _NormalizeOne(item)
				if one:
					out.append(one)
			return out
		return []

	if text[0] == "<":
		return ParseXmlFunctions(text)

	return []


def SerializeToolCallInner(calls: list[dict]) -> str:
	"""Canonical JSON for storage/display (object if one, array if many)."""
	if not calls:
		return ""
	payload = []
	for c in calls:
		item = {"name": c.get("name"), "arguments": c.get("arguments") or {}}
		if c.get("id"):
			item["id"] = c.get("id")
		payload.append(item)
	if len(payload) == 1:
		return json.dumps(payload[0], ensure_ascii=False)
	return json.dumps(payload, ensure_ascii=False)


def SplitToolSegments(content: str):
	"""Yield ('text'|'tool_call'|'tool_response', payload) from assistant XML transcript."""
	text = content or ""
	pos = 0
	n = len(text)
	while pos < n:
		tc_i = text.find("<tool_call>", pos)
		tr_i = text.find("<tool_response>", pos)
		next_i = -1
		kind = None
		open_tag = close_tag = None
		if tc_i >= 0 and (tr_i < 0 or tc_i <= tr_i):
			next_i, kind = tc_i, "tool_call"
			open_tag, close_tag = "<tool_call>", "</tool_call>"
		elif tr_i >= 0:
			next_i, kind = tr_i, "tool_response"
			open_tag, close_tag = "<tool_response>", "</tool_response>"
		if next_i < 0:
			tail = text[pos:].strip()
			if tail:
				yield ("text", tail)
			break
		before = text[pos:next_i].strip()
		if before:
			yield ("text", before)
		close_i = text.find(close_tag, next_i + len(open_tag))
		if close_i < 0:
			# Incomplete tag — keep remainder as text.
			tail = text[next_i:].strip()
			if tail:
				yield ("text", tail)
			break
		inner = text[next_i + len(open_tag):close_i].strip()
		yield (kind, inner)
		pos = close_i + len(close_tag)
