"""Remote tool wire adapters (OpenAI + Anthropic).

UI / E.messages stay AI UI XML. This module maps to provider wire shapes
per request and folds structured tool_calls / tool_use back into XML for the
shared tool loop. Rebuild maps from the current transcript each request.
"""

from __future__ import annotations

import json
import re

E = None


def bind(engine_module):
	global E
	E = engine_module


def RemoteToolFormat():
	"""Return 'openai' or 'anthropic' (default openai). Legacy 'xml' maps to openai."""
	raw = str(getattr(E, "remote_tool_format", "openai") or "openai").strip().lower()
	if raw == "anthropic":
		return "anthropic"
	return "openai"


def IsOpenAITools():
	return RemoteToolFormat() == "openai"


def IsAnthropicTools():
	return RemoteToolFormat() == "anthropic"


# ---------------------------------------------------------------------------
# OpenAI Chat Completions tools=
# ---------------------------------------------------------------------------

def ToolsForOpenAI():
	"""OpenAI tools= list from E.tool_funcs (already HF/OpenAI shaped)."""
	funcs = getattr(E, "tool_funcs", None) or []
	out = []
	for t in funcs:
		if not isinstance(t, dict):
			continue
		if t.get("type") == "function" and isinstance(t.get("function"), dict):
			out.append(t)
		elif t.get("name"):
			out.append({
				"type": "function",
				"function": {
					"name": t.get("name"),
					"description": t.get("description", "") or "",
					"parameters": t.get("parameters") or t.get("inputSchema") or {
						"type": "object",
						"properties": {},
					},
				},
			})
	return out


def ApplyOpenAIToolFields(payload, enable_tooluse=None):
	"""Mutate chat.completions payload for OpenAI remote protocol.

	tooluse on  → tools= + tool_choice auto
	tooluse off → omit tools= and tool_choice (servers reject choice without tools)
	"""
	if not IsOpenAITools():
		return payload
	if enable_tooluse is None:
		enable_tooluse = bool(getattr(E, "enable_tooluse", 0))
	tools = ToolsForOpenAI() if enable_tooluse else []
	if enable_tooluse and tools:
		payload["tools"] = tools
		payload["tool_choice"] = "auto"
	else:
		payload.pop("tools", None)
		payload.pop("tool_choice", None)
	return payload


def ApplyOpenAIThinkingFields(payload):
	"""Map Enable Thinking / Reasoning Effort onto OpenAI-compat chat.completions.

	- Always keep/refresh chat_template_kwargs for transformers serve (Qwen/Inkling).
	- Also set top-level reasoning_effort for native OpenAI Chat Completions
	  (ignored or rejected by servers that do not support it — same class of risk
	  as chat_template_kwargs on non-serve endpoints).
	- Enable Thinking False → reasoning_effort "none" (UI dropped None; this is the off switch).
	"""
	from lib import chat_template

	ct_kw = chat_template.ThinkingKwargs(E)
	if isinstance(ct_kw, dict) and ct_kw:
		payload["chat_template_kwargs"] = ct_kw
	else:
		payload.pop("chat_template_kwargs", None)

	et = str(getattr(E, "enable_thinking", "auto") or "auto").strip().lower()
	re = str(getattr(E, "reasoning_effort", "auto") or "auto").strip().lower()
	if re == "none":
		re = "auto"
	elif re == "minimal":
		re = "low"

	if et in ("false", "0", "no", "off"):
		payload["reasoning_effort"] = "none"
	elif re in ("low", "medium", "high", "xhigh", "max"):
		payload["reasoning_effort"] = re
	else:
		payload.pop("reasoning_effort", None)
	return payload


def MapMessagesOpenAI(messages):
	"""Rebuild OpenAI wire messages from AI UI XML transcript.

	`messages` should already be role-normalized (assistant/user/system) with
	multimodal content parts applied (e.g. from MessagesForRemote).
	Stable tool_call_id: call_{msg_index}_{call_index}.
	"""
	from lib import tool_calls as tc_mod

	out = []
	pending_ids = []

	for msg_i, m in enumerate(messages or []):
		if not isinstance(m, dict):
			continue
		role = str(m.get("role") or "user")
		content = m.get("content")
		if role != "assistant" or not isinstance(content, str):
			out.append(dict(m))
			continue
		if "<tool_call>" not in content and "<tool_response>" not in content:
			out.append(dict(m))
			continue

		buf_text = []
		pending_calls = []
		call_i = 0

		def FlushText():
			nonlocal buf_text
			if not buf_text:
				return
			joined = "\n\n".join(buf_text).strip()
			buf_text = []
			if joined:
				out.append({"role": "assistant", "content": joined})

		def FlushCalls():
			nonlocal pending_calls, buf_text
			if not pending_calls:
				return
			preface = "\n\n".join(buf_text).strip() if buf_text else ""
			buf_text = []
			tc_list = []
			for c in pending_calls:
				args = c.get("arguments") or {}
				arg_s = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
				tc_list.append({
					"id": c["id"],
					"type": "function",
					"function": {
						"name": c.get("name"),
						"arguments": arg_s,
					},
				})
				pending_ids.append(c["id"])
			pending_calls = []
			msg = {"role": "assistant", "tool_calls": tc_list}
			msg["content"] = preface if preface else None
			out.append(msg)

		for kind, payload in tc_mod.SplitToolSegments(content):
			if kind == "text":
				FlushCalls()
				buf_text.append(payload)
			elif kind == "tool_call":
				parsed = tc_mod.ParseToolCallInner(payload)
				if not parsed:
					FlushCalls()
					buf_text.append("<tool_call>" + payload + "</tool_call>")
					continue
				for one in parsed:
					tid = f"call_{msg_i}_{call_i}"
					call_i += 1
					pending_calls.append({
						"id": tid,
						"name": one.get("name"),
						"arguments": one.get("arguments") or {},
					})
			elif kind == "tool_response":
				# Preface + tool_calls share one assistant message when possible.
				FlushCalls()
				FlushText()
				name = "unknown"
				body = payload
				try:
					data = json.loads(payload) if payload else {}
					if isinstance(data, dict):
						name = str(data.get("name") or name)
						body = data.get("content", payload)
						if not isinstance(body, str):
							body = json.dumps(body, ensure_ascii=False)
				except Exception:
					pass
				tid = pending_ids.pop(0) if pending_ids else f"call_{msg_i}_orphan_{call_i}"
				out.append({
					"role": "tool",
					"tool_call_id": tid,
					"name": name,
					"content": body if isinstance(body, str) else str(body),
				})
		FlushCalls()
		FlushText()

	return out


def NewOpenAIToolCallAcc():
	"""index -> {id, name, arguments} accumulator for streaming deltas."""
	return {}


def AccumulateOpenAIToolCalls(acc, deltas):
	"""Merge OpenAI delta.tool_calls fragments into acc (by index)."""
	if not deltas:
		return acc
	for d in deltas:
		if not isinstance(d, dict):
			continue
		try:
			idx = int(d.get("index", 0))
		except (TypeError, ValueError):
			idx = 0
		slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
		tid = d.get("id")
		if isinstance(tid, str) and tid:
			slot["id"] = tid
		fn = d.get("function")
		if not isinstance(fn, dict):
			fn = {}
		name = fn.get("name")
		if isinstance(name, str) and name:
			cur = slot["name"]
			if not cur:
				slot["name"] = name
			elif name.startswith(cur):
				slot["name"] = name
			elif not cur.startswith(name):
				slot["name"] = cur + name
		args = fn.get("arguments")
		if isinstance(args, str) and args:
			slot["arguments"] += args
	return acc


def FinalizeOpenAIToolCalls(acc):
	"""Return list of {name, arguments} ready for SerializeToolCallInner."""
	calls = []
	for idx in sorted(acc.keys()):
		slot = acc[idx]
		name = (slot.get("name") or "").strip()
		if not name:
			continue
		raw = slot.get("arguments") or ""
		args = {}
		if isinstance(raw, str) and raw.strip():
			try:
				parsed = json.loads(raw)
				args = parsed if isinstance(parsed, dict) else {"_raw": raw}
			except Exception:
				args = {"_raw": raw}
		elif isinstance(raw, dict):
			args = raw
		calls.append({"name": name, "arguments": args, "id": slot.get("id") or ""})
	# Drop empty ids from OpenAI finalize for cleaner XML when unused
	for c in calls:
		if not c.get("id"):
			c.pop("id", None)
	return calls


def ToolCallsFromMessage(msg):
	"""Non-stream message.tool_calls → [{name, arguments}, ...]."""
	if msg is None:
		return []
	raw = getattr(msg, "tool_calls", None)
	if raw is None and isinstance(msg, dict):
		raw = msg.get("tool_calls")
	if not raw:
		return []
	acc = NewOpenAIToolCallAcc()
	for i, tc in enumerate(raw):
		if isinstance(tc, dict):
			fn = tc.get("function") or {}
			acc[i] = {
				"id": tc.get("id") or "",
				"name": fn.get("name") or "",
				"arguments": fn.get("arguments") or "",
			}
			continue
		fn = getattr(tc, "function", None)
		name = getattr(fn, "name", None) if fn is not None else None
		arguments = getattr(fn, "arguments", None) if fn is not None else None
		acc[i] = {
			"id": getattr(tc, "id", "") or "",
			"name": name or "",
			"arguments": arguments if isinstance(arguments, str) else (json.dumps(arguments) if arguments else ""),
		}
	return FinalizeOpenAIToolCalls(acc)


def ToolCallsToXml(calls):
	"""Fold structured calls into AI UI <tool_call>…</tool_call> for the shared loop."""
	from lib.tool_calls import SerializeToolCallInner

	if not calls:
		return ""
	inner = SerializeToolCallInner(calls)
	if not inner:
		return ""
	return "<tool_call>" + inner + "</tool_call>"


def AppendToolCallsXml(gen_text, calls):
	"""Append synthesized tool XML to reply text (plain newlines)."""
	xml = ToolCallsToXml(calls)
	if not xml:
		return gen_text or ""
	base = (gen_text or "").rstrip()
	if base:
		return base + "\n\n" + xml
	return xml


# ---------------------------------------------------------------------------
# Anthropic Messages API — same XML storage, different wire
# ---------------------------------------------------------------------------

def ToolsForAnthropic():
	"""Anthropic tools= list from E.tool_funcs."""
	funcs = getattr(E, "tool_funcs", None) or []
	out = []
	for t in funcs:
		if not isinstance(t, dict):
			continue
		fn = t.get("function") if t.get("type") == "function" and isinstance(t.get("function"), dict) else t
		name = fn.get("name")
		if not name:
			continue
		params = fn.get("parameters") or fn.get("inputSchema") or {"type": "object", "properties": {}}
		out.append({
			"name": name,
			"description": fn.get("description", "") or "",
			"input_schema": params,
		})
	return out


def ApplyAnthropicToolFields(payload, enable_tooluse=None):
	if enable_tooluse is None:
		enable_tooluse = bool(getattr(E, "enable_tooluse", 0))
	tools = ToolsForAnthropic() if enable_tooluse else []
	if enable_tooluse and tools:
		payload["tools"] = tools
		payload["tool_choice"] = {"type": "auto"}
	else:
		# Anthropic rejects tool_choice when tools is absent/empty.
		payload.pop("tools", None)
		payload.pop("tool_choice", None)
	return payload


def ApplyAnthropicThinkingFields(payload):
	"""Map Enable Thinking / Reasoning Effort → Anthropic thinking + effort."""
	et = str(getattr(E, "enable_thinking", "auto") or "auto").strip().lower()
	if et in ("false", "0", "no", "off"):
		payload["thinking"] = {"type": "disabled"}
		payload.pop("effort", None)
		return payload
	# true and auto → adaptive + summarized so thoughts show in AI UI
	payload["thinking"] = {"type": "adaptive", "display": "summarized"}
	effort = str(getattr(E, "reasoning_effort", "auto") or "auto").strip().lower()
	if effort == "none":
		effort = "auto"
	elif effort == "minimal":
		effort = "low"
	if effort in ("low", "medium", "high", "xhigh", "max"):
		oc = dict(payload.get("output_config") or {})
		oc["effort"] = effort
		payload["output_config"] = oc
	else:
		oc = payload.get("output_config")
		if isinstance(oc, dict):
			oc.pop("effort", None)
			if not oc:
				payload.pop("output_config", None)
	return payload


def ApplyAnthropicCacheFields(payload):
	"""Top-level automatic ephemeral cache when remote_prompt_cache is on."""
	on = getattr(E, "remote_prompt_cache", 1)
	try:
		on = int(on)
	except (TypeError, ValueError):
		on = 1
	if on:
		payload["cache_control"] = {"type": "ephemeral"}
	else:
		payload.pop("cache_control", None)
	return payload


def _OpenAIImageToAnthropic(part):
	"""Map OpenAI image_url part → Anthropic image block (or None)."""
	if not isinstance(part, dict):
		return None
	url_obj = part.get("image_url") if part.get("type") == "image_url" else None
	if not isinstance(url_obj, dict):
		return None
	url = str(url_obj.get("url") or "").strip()
	if not url:
		return None
	if url.startswith("data:") and ";base64," in url:
		header, b64 = url.split(";base64,", 1)
		media = header[5:] if header.startswith("data:") else "image/png"
		return {
			"type": "image",
			"source": {"type": "base64", "media_type": media or "image/png", "data": b64},
		}
	if url.startswith("http://") or url.startswith("https://"):
		return {"type": "image", "source": {"type": "url", "url": url}}
	return None


def ContentToAnthropic(content):
	"""Convert MessagesForRemote content (str or OpenAI parts) to Anthropic content."""
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	if not isinstance(content, (list, tuple)):
		return str(content)
	blocks = []
	for p in content:
		if not isinstance(p, dict):
			continue
		ptype = p.get("type")
		if ptype == "text":
			t = p.get("text") or ""
			if t:
				blocks.append({"type": "text", "text": t})
		elif ptype == "image_url":
			img = _OpenAIImageToAnthropic(p)
			if img:
				blocks.append(img)
		# video_url / audio skipped on Anthropic path
	if not blocks:
		return ""
	if len(blocks) == 1 and blocks[0].get("type") == "text":
		return blocks[0].get("text") or ""
	return blocks


def _ContentBlocksAsList(content):
	"""Normalize Anthropic content to a list of blocks."""
	if content is None or content == "":
		return []
	if isinstance(content, str):
		return [{"type": "text", "text": content}] if content else []
	if isinstance(content, list):
		return list(content)
	return [{"type": "text", "text": str(content)}]


def _HoistThinkingFirst(blocks):
	"""Move thinking blocks to the front, keeping their relative order.

	A preface stored as its own message merges ahead of the tool-call message that
	carries the sidecar; Claude always thinks before writing, so leading with the
	thinking blocks restores the turn shape the model actually produced.
	"""
	def IsThink(b):
		return isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")

	think = [b for b in blocks if IsThink(b)]
	if not think or IsThink(blocks[0]):
		return blocks
	return think + [b for b in blocks if not IsThink(b)]


_LEADING_THINK_RE = re.compile(
	r"^\s*(?:<(think|thoughts)\b[^>]*>.*?</\1>|AIUI_THINK_START.*?AIUI_THINK_END)\s*",
	re.IGNORECASE | re.DOTALL,
)


def _StripLeadingThink(text):
	"""Drop leading `<think>` blocks from stored assistant text.

	Reasoning always opens a reply, so anchoring to the start leaves think tags
	that appear mid-message (code examples, quoted markup) untouched. On the
	Anthropic wire reasoning belongs in `thinking` blocks, so an inline copy is
	either a duplicate of the signed block or unsigned text Claude should not
	see replayed as its own words.
	"""
	s = str(text or "")
	while True:
		trimmed = _LEADING_THINK_RE.sub("", s, count=1)
		if trimmed == s:
			return s.strip()
		s = trimmed


def ThinkingLeadBlocks(sidecar):
	"""`thinking` / `redacted_thinking` blocks from an `_anthropic_content` sidecar.

	Text and tool_use blocks are deliberately ignored: they are rebuilt from the
	XML transcript (tool_call JSON keeps the server `id`), so replaying them here
	would send every assistant turn twice. Encrypted thinking signatures are the
	only part of a reply that cannot be reconstructed.
	"""
	out = []
	for b in sidecar if isinstance(sidecar, list) else []:
		if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
			out.append(dict(b))
	return out


def MapMessagesAnthropic(messages):
	"""Rebuild Anthropic wire messages from AI UI transcript.

	Returns (system, wire_messages). Thinking blocks come from the
	`_anthropic_content` sidecar and lead the assistant turn they belong to.
	Preserves server tool_use ids from tool-call JSON `"id"` when remapping XML.
	"""
	from lib import tool_calls as tc_mod

	system = None
	out = []
	pending_ids = []

	for msg_i, m in enumerate(messages or []):
		if not isinstance(m, dict):
			continue
		role = str(m.get("role") or "user")
		content = m.get("content")

		if role == "system":
			# Anthropic wants system top-level; accumulate text.
			part = ContentToAnthropic(content)
			if isinstance(part, list):
				texts = [b.get("text") or "" for b in part if isinstance(b, dict) and b.get("type") == "text"]
				s = "\n\n".join(t for t in texts if t).strip()
			else:
				s = (part or "").strip() if isinstance(part, str) else ""
			if s:
				system = (system + "\n\n" + s) if system else s
			continue

		if role != "assistant":
			mapped = ContentToAnthropic(content)
			# Skip attachments that mapped to nothing (video-only, unsupported parts).
			if mapped == "":
				continue
			out.append({"role": "user", "content": mapped})
			continue

		lead = ThinkingLeadBlocks(m.get("_anthropic_content"))
		if isinstance(content, str):
			content = _StripLeadingThink(content)

		if not isinstance(content, str):
			blocks = lead + _ContentBlocksAsList(ContentToAnthropic(content))
			if blocks:
				out.append({"role": "assistant", "content": blocks})
			continue

		if "<tool_call>" not in content and "<tool_response>" not in content:
			if not content:
				continue
			if lead:
				out.append({"role": "assistant", "content": lead + _ContentBlocksAsList(content)})
			else:
				out.append({"role": "assistant", "content": content})
			continue

		buf_text = []
		pending_calls = []
		call_i = 0

		def TakeLead():
			"""Thinking blocks belong at the front of the first block group emitted."""
			nonlocal lead
			blocks = lead
			lead = []
			return blocks

		def FlushText():
			nonlocal buf_text
			joined = "\n\n".join(buf_text).strip() if buf_text else ""
			buf_text = []
			if not joined:
				return
			blocks = TakeLead()
			if blocks:
				blocks.append({"type": "text", "text": joined})
				out.append({"role": "assistant", "content": blocks})
			else:
				out.append({"role": "assistant", "content": joined})

		def FlushCalls():
			nonlocal pending_calls, buf_text
			if not pending_calls:
				return
			preface = "\n\n".join(buf_text).strip() if buf_text else ""
			buf_text = []
			blocks = TakeLead()
			if preface:
				blocks.append({"type": "text", "text": preface})
			for c in pending_calls:
				tid = c["id"]
				pending_ids.append(tid)
				blocks.append({
					"type": "tool_use",
					"id": tid,
					"name": c.get("name"),
					"input": c.get("arguments") or {},
				})
			pending_calls = []
			out.append({"role": "assistant", "content": blocks})

		for kind, payload in tc_mod.SplitToolSegments(content):
			if kind == "text":
				FlushCalls()
				buf_text.append(payload)
			elif kind == "tool_call":
				parsed = tc_mod.ParseToolCallInner(payload)
				if not parsed:
					FlushCalls()
					buf_text.append("<tool_call>" + payload + "</tool_call>")
					continue
				for one in parsed:
					tid = one.get("id") or f"toolu_{msg_i}_{call_i}"
					call_i += 1
					pending_calls.append({
						"id": tid,
						"name": one.get("name"),
						"arguments": one.get("arguments") or {},
					})
			elif kind == "tool_response":
				FlushCalls()
				FlushText()
				name = "unknown"
				body = payload
				try:
					data = json.loads(payload) if payload else {}
					if isinstance(data, dict):
						name = str(data.get("name") or name)
						body = data.get("content", payload)
						if not isinstance(body, str):
							body = json.dumps(body, ensure_ascii=False)
				except Exception:
					pass
				tid = pending_ids.pop(0) if pending_ids else f"toolu_{msg_i}_orphan_{call_i}"
				out.append({
					"role": "user",
					"content": [{
						"type": "tool_result",
						"tool_use_id": tid,
						"content": body if isinstance(body, str) else str(body),
					}],
				})
		FlushCalls()
		FlushText()

	# Anthropic requires alternating user/assistant; merge consecutive same-role.
	merged = []
	for m in out:
		if not merged:
			merged.append(m)
			continue
		prev = merged[-1]
		if prev.get("role") == m.get("role"):
			prev_c = _ContentBlocksAsList(prev.get("content"))
			cur_c = _ContentBlocksAsList(m.get("content"))
			prev["content"] = _HoistThinkingFirst(prev_c + cur_c)
		else:
			merged.append(m)
	# Must start with user, and Anthropic rejects an empty messages list — which
	# compaction can produce when the whole tail folded into the summary.
	if not merged or merged[0].get("role") != "user":
		merged.insert(0, {"role": "user", "content": "(continued)"})
	return system, merged


def NewAnthropicToolUseAcc():
	"""index -> {id, name, input_json} for streaming tool_use blocks."""
	return {}


def AccumulateAnthropicToolUse(acc, *, index=None, block=None, partial_json=None):
	"""Merge content_block_start (tool_use) / input_json_delta into acc."""
	if index is None:
		return acc
	try:
		idx = int(index)
	except (TypeError, ValueError):
		idx = 0
	slot = acc.setdefault(idx, {"id": "", "name": "", "input_json": ""})
	if isinstance(block, dict) and block.get("type") == "tool_use":
		if block.get("id"):
			slot["id"] = block["id"]
		if block.get("name"):
			slot["name"] = block["name"]
		inp = block.get("input")
		if isinstance(inp, dict) and inp:
			slot["input_json"] = json.dumps(inp, ensure_ascii=False)
	if isinstance(partial_json, str) and partial_json:
		slot["input_json"] += partial_json
	return acc


def FinalizeAnthropicToolUse(acc):
	"""Return list of {id, name, arguments} for ToolCallsToXml."""
	calls = []
	for idx in sorted(acc.keys()):
		slot = acc[idx]
		name = (slot.get("name") or "").strip()
		if not name:
			continue
		raw = slot.get("input_json") or ""
		args = {}
		if isinstance(raw, str) and raw.strip():
			try:
				parsed = json.loads(raw)
				args = parsed if isinstance(parsed, dict) else {"_raw": raw}
			except Exception:
				args = {"_raw": raw}
		elif isinstance(raw, dict):
			args = raw
		item = {"name": name, "arguments": args}
		if slot.get("id"):
			item["id"] = slot["id"]
		calls.append(item)
	return calls


def ContentBlocksToWireList(content_blocks):
	"""Serialize SDK content blocks to plain dicts for `_anthropic_content` sidecar."""
	out = []
	for b in content_blocks or []:
		if isinstance(b, dict):
			out.append(dict(b))
			continue
		btype = getattr(b, "type", None)
		if btype == "text":
			out.append({"type": "text", "text": getattr(b, "text", "") or ""})
		elif btype == "thinking":
			item = {
				"type": "thinking",
				"thinking": getattr(b, "thinking", "") or "",
			}
			sig = getattr(b, "signature", None)
			if sig:
				item["signature"] = sig
			out.append(item)
		elif btype == "redacted_thinking":
			item = {"type": "redacted_thinking"}
			data = getattr(b, "data", None)
			if data is not None:
				item["data"] = data
			out.append(item)
		elif btype == "tool_use":
			inp = getattr(b, "input", None) or {}
			out.append({
				"type": "tool_use",
				"id": getattr(b, "id", "") or "",
				"name": getattr(b, "name", "") or "",
				"input": inp if isinstance(inp, dict) else {},
			})
		else:
			# Best-effort dump
			if hasattr(b, "model_dump"):
				try:
					d = b.model_dump()
					if isinstance(d, dict):
						out.append(d)
						continue
				except Exception:
					pass
	return out


def ToolUseFromContentBlocks(content_blocks):
	"""Extract tool_use calls from a completed Anthropic message content list."""
	calls = []
	for b in content_blocks or []:
		if isinstance(b, dict):
			if b.get("type") != "tool_use":
				continue
			calls.append({
				"id": b.get("id") or "",
				"name": b.get("name") or "",
				"arguments": b.get("input") if isinstance(b.get("input"), dict) else {},
			})
			continue
		if getattr(b, "type", None) != "tool_use":
			continue
		inp = getattr(b, "input", None) or {}
		calls.append({
			"id": getattr(b, "id", "") or "",
			"name": getattr(b, "name", "") or "",
			"arguments": inp if isinstance(inp, dict) else {},
		})
	return [c for c in calls if c.get("name")]
