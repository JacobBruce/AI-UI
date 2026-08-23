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

	- chat_template_kwargs only for custom/transformers serve (openai.com and
	  OpenRouter reject unknown body fields).
	- reasoning_effort for OpenRouter / serve; official OpenAI omits it when
	  thinking is off and clamps xhigh/max to high.
	"""
	from lib import chat_template

	ct_kw = chat_template.ThinkingKwargs(E)
	et = str(getattr(E, "enable_thinking", "auto") or "auto").strip().lower()
	re = str(getattr(E, "reasoning_effort", "auto") or "auto").strip().lower()
	if re == "none":
		re = "auto"
	elif re == "minimal":
		re = "low"

	host = str(getattr(E, "remote_base_url", "") or "").strip().lower()
	official_openai = "openai.com" in host or "api.openai" in host
	serve_kw = bool(host) and not official_openai and "openrouter.ai" not in host
	if serve_kw and isinstance(ct_kw, dict) and ct_kw:
		payload["chat_template_kwargs"] = ct_kw
	else:
		payload.pop("chat_template_kwargs", None)

	if et in ("false", "0", "no", "off"):
		if official_openai:
			payload.pop("reasoning_effort", None)
		else:
			payload["reasoning_effort"] = "none"
	elif re in ("low", "medium", "high", "xhigh", "max"):
		if official_openai and re in ("xhigh", "max"):
			re = "high"
		payload["reasoning_effort"] = re
	else:
		payload.pop("reasoning_effort", None)
	return payload


def ParseToolResponseInner(payload):
	"""XML <tool_response> body → (name, content, id)."""
	name = "unknown"
	body = payload
	resp_id = ""
	try:
		data = json.loads(payload) if payload else {}
		if isinstance(data, dict):
			name = str(data.get("name") or name)
			resp_id = str(data.get("id") or "")
			body = data.get("content", payload)
			if not isinstance(body, str):
				body = json.dumps(body, ensure_ascii=False)
	except Exception:
		pass
	return name, body, resp_id


def TakePendingToolId(pending_ids, resp_id=None, fallback=""):
	"""Bind a tool result to a pending call id.

	Prefer an explicit resp_id still in the queue so out-of-order transcripts
	do not FIFO-attach the wrong result. Otherwise take the next call-order id.
	"""
	rid = str(resp_id).strip() if resp_id else ""
	if rid:
		try:
			pending_ids.remove(rid)
			return rid
		except ValueError:
			pass
	if pending_ids:
		return pending_ids.pop(0)
	return fallback


_REASONING_TEXT_FIELDS = {
	"reasoning.text": "text",
	"reasoning.summary": "summary",
	"reasoning.encrypted": "data",
}


def ReasoningDetailToDict(item):
	"""Plain dict from an OpenRouter reasoning_details element (JSON or SDK object)."""
	if isinstance(item, dict):
		return dict(item)
	dump = getattr(item, "model_dump", None)
	if callable(dump):
		try:
			d = dump(exclude_none=False)
			if isinstance(d, dict):
				return d
		except Exception:
			pass
	out = {}
	for k in ("type", "text", "summary", "data", "signature", "id", "format", "index"):
		v = getattr(item, k, None)
		if v is not None:
			out[k] = v
	return out


def _ReasoningDetailSlotKey(item):
	if not isinstance(item, dict):
		return None
	if item.get("index") is not None:
		return ("idx", item.get("index"), item.get("type"))
	if item.get("id"):
		return ("id", item.get("id"), item.get("type"))
	return None


def _MergeReasoningDetail(dst, src):
	kind = str(dst.get("type") or src.get("type") or "")
	field = _REASONING_TEXT_FIELDS.get(kind)
	if field:
		a = dst.get(field) or ""
		b = src.get(field) or ""
		if kind == "reasoning.encrypted":
			if b and (not a or a == "[REDACTED]" or (b != "[REDACTED]" and len(b) >= len(a))):
				dst[field] = b
			elif b and b != "[REDACTED]":
				dst[field] = (a + b) if a else b
		elif b:
			dst[field] = (a + b) if a else b
	sig = src.get("signature")
	if sig:
		dst["signature"] = sig
	for k, v in src.items():
		if k in (field, "signature") or v in (None, ""):
			continue
		if dst.get(k) in (None, ""):
			dst[k] = v


def AccumulateReasoningDetails(acc, details):
	"""Merge streaming reasoning_details chunks into acc (mutates acc)."""
	if not acc:
		acc = acc if acc is not None else []
	if not details:
		return acc
	if not isinstance(details, list):
		details = [details]
	index = {}
	for i, existing in enumerate(acc):
		key = _ReasoningDetailSlotKey(existing)
		if key is not None:
			index[key] = i
	for raw in details:
		item = ReasoningDetailToDict(raw)
		if not item:
			continue
		key = _ReasoningDetailSlotKey(item)
		if key is not None and key in index:
			_MergeReasoningDetail(acc[index[key]], item)
			continue
		acc.append(item)
		if key is not None:
			index[key] = len(acc) - 1
	return acc


def FinalizeReasoningDetails(acc):
	"""Return a JSON-safe list, or None when empty."""
	if not acc:
		return None
	out = [dict(x) for x in acc if isinstance(x, dict) and x]
	if not out:
		return None
	if all(x.get("index") is not None for x in out):
		out.sort(key=lambda x: x.get("index", 0))
	return out


def ReasoningDetailsFromMessage(msg):
	"""Pull reasoning_details off a non-stream OpenAI/OpenRouter message."""
	raw = None
	if msg is None:
		return None
	if isinstance(msg, dict):
		raw = msg.get("reasoning_details")
	else:
		raw = getattr(msg, "reasoning_details", None)
		if raw is None:
			extra = getattr(msg, "model_extra", None)
			if isinstance(extra, dict):
				raw = extra.get("reasoning_details")
		if raw is None:
			dump = SdkModelDump(msg)
			if isinstance(dump, dict):
				raw = dump.get("reasoning_details")
	if not raw:
		return None
	acc = []
	AccumulateReasoningDetails(acc, raw)
	return FinalizeReasoningDetails(acc)


def ReasoningTextFromDetails(details):
	"""Visible reasoning.text fragments (for UI when the string channel is empty)."""
	parts = []
	for raw in details or []:
		item = ReasoningDetailToDict(raw)
		if item.get("type") == "reasoning.text":
			t = item.get("text")
			if isinstance(t, str) and t:
				parts.append(t)
	return "".join(parts)


def HasEncryptedReasoning(details):
	"""True when details include an encrypted / compaction blob bound to one endpoint."""
	for raw in details or []:
		item = ReasoningDetailToDict(raw)
		if str(item.get("type") or "") == "reasoning.encrypted":
			return True
	return False


def NormalizeOpenRouterProvider(raw):
	s = str(raw or "").strip().replace(" ", "-")
	return s.lower() if s else ""


def IsOpenRouterAutoModel(model_id):
	"""True for OpenRouter routers that pick a backend per request.

	Matches `openrouter/auto`, `openrouter/auto-beta`, and `openrouter/free`.
	"""
	s = str(model_id or "").strip().lower()
	return s.startswith("openrouter/auto") or s == "openrouter/free"


def ReasoningPinFromDetails(details, *, model=None, provider=None):
	"""Build a pin dict for the endpoint that produced these details, or None."""
	if not HasEncryptedReasoning(details):
		return None
	pin = {}
	mid = str(model or "").strip()
	if mid and not IsOpenRouterAutoModel(mid):
		pin["model"] = mid
	prov = NormalizeOpenRouterProvider(provider)
	if prov:
		pin["provider"] = prov
	# Only when OpenRouter actually routed to a ZDR variant (e.g. xai/zdr).
	# xAI encrypted reasoning on the normal xai endpoint must not request zdr:
	# accounts that have not opted into Zero Data Retention then 404.
	if "zdr" in prov:
		pin["zdr"] = True
	return pin or None


def SdkModelDump(obj):
	"""Best-effort dict of a pydantic/SDK object, including unset extras."""
	if obj is None:
		return None
	if isinstance(obj, dict):
		return obj
	dump = getattr(obj, "model_dump", None)
	if callable(dump):
		for kwargs in ({"exclude_unset": False}, {}):
			try:
				out = dump(**kwargs)
				if isinstance(out, dict):
					return out
			except TypeError:
				continue
			except Exception:
				break
	as_dict = getattr(obj, "dict", None)
	if callable(as_dict):
		try:
			out = as_dict()
			if isinstance(out, dict):
				return out
		except Exception:
			pass
	extra = getattr(obj, "model_extra", None)
	return extra if isinstance(extra, dict) else None


def TranscriptReasoningPin(messages=None):
	"""Most recent `_reasoning_pin` plus whether the transcript has encrypted details."""
	raw = messages if messages is not None else (getattr(E, "messages", None) or [])
	pin = None
	encrypted = False
	for m in raw:
		if not isinstance(m, dict):
			continue
		details = m.get("_reasoning_details") or m.get("reasoning_details")
		if HasEncryptedReasoning(details):
			encrypted = True
		p = m.get("_reasoning_pin")
		if isinstance(p, dict) and (p.get("model") or p.get("provider") or p.get("zdr")):
			pin = {k: v for k, v in p.items() if v not in (None, "", False)}
			if p.get("zdr"):
				pin["zdr"] = True
	return pin, encrypted


def ApplyReasoningRoutePin(create_kw, messages=None):
	"""Force the producing OpenRouter model/endpoint when encrypted reasoning is in play.

	`openrouter/auto` (and other routers) may pick a different endpoint on the next
	tool round; encrypted xAI blobs 404 unless they go back to the original one.

	Pin and encrypted are taken from the mapped wire (`create_kw["messages"]`) so a
	compacted transcript cannot keep pinning after the blobs left the request.
	"""
	if not isinstance(create_kw, dict):
		return create_kw
	mapped = create_kw.get("messages")
	if isinstance(mapped, list):
		pin, encrypted = TranscriptReasoningPin(mapped)
	else:
		pin, encrypted = TranscriptReasoningPin(messages)
	if not encrypted:
		return create_kw
	if not pin:
		pin = {}
	kw = dict(create_kw)
	extra = dict(kw.get("extra_body") or {})
	cur = str(kw.get("model") or "").strip()
	pinned_model = str(pin.get("model") or "").strip()
	changed = False
	if pinned_model and (IsOpenRouterAutoModel(cur) or not cur or pinned_model != cur):
		kw["model"] = pinned_model
		changed = True
	prov = {}
	slug = NormalizeOpenRouterProvider(pin.get("provider"))
	zdr = bool(pin.get("zdr"))
	order = []
	if slug:
		if zdr and "/" not in slug:
			order = [f"{slug}/zdr", slug]
		else:
			order = [slug]
	if order:
		prov["order"] = order
		prov["allow_fallbacks"] = False
		changed = True
	if zdr:
		prov["zdr"] = True
		changed = True
	if not changed:
		return create_kw
	if prov:
		extra["provider"] = prov
	if extra:
		kw["extra_body"] = extra
	from lib import ipc
	ipc.LogErr(
		f"OpenRouter: pinning encrypted reasoning model={kw.get('model')!r} "
		f"provider={prov or None}"
	)
	return kw


def _StealOpenAIPreface(out):
	"""Merge a preceding bare assistant text turn into the following tool_calls message.

	OpenRouter encrypted reasoning is one completion; storing think text separately
	used to replay as assistant(text) then assistant(tool_calls + sidecar).
	"""
	if not out:
		return ""
	prev = out[-1]
	if not isinstance(prev, dict) or prev.get("role") != "assistant":
		return ""
	if prev.get("tool_calls"):
		return ""
	if prev.get("reasoning_details"):
		return ""
	text = prev.get("content")
	if not isinstance(text, str) or not text.strip():
		return ""
	text = _StripLeadingThink(text)
	if not text:
		return ""
	out.pop()
	return text.strip()


def MapMessagesOpenAI(messages, replay_reasoning=False):
	"""Rebuild OpenAI wire messages from AI UI XML transcript.

	`messages` should already be role-normalized (assistant/user/system) with
	multimodal content parts applied (e.g. from MessagesForRemote).
	tool_call_id prefers the id stored on the XML call, else call_{msg}_{i}.
	`replay_reasoning` adds OpenRouter `reasoning_details` (skip for openai.com).
	Leading `<think>` is stripped from assistant `content` the same way as
	Anthropic: native reasoning stays on the sidecar, not as XML in the body.
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
			item = {k: v for k, v in dict(m).items() if not str(k).startswith("_")}
			out.append(item)
			continue
		sidecar = m.get("_reasoning_details") if replay_reasoning else None
		pin = m.get("_reasoning_pin") if replay_reasoning else None
		sidecar_used = False
		raw_content = content
		content = _StripLeadingThink(content)

		def AttachReasoning(msg):
			nonlocal sidecar_used
			if sidecar_used or not sidecar:
				return
			msg["reasoning_details"] = sidecar
			if isinstance(pin, dict) and pin:
				msg["_reasoning_pin"] = pin
			sidecar_used = True

		if "<tool_call>" not in content and "<tool_response>" not in content:
			if not content:
				if sidecar:
					item = {"role": "assistant", "content": None}
					AttachReasoning(item)
					out.append(item)
				else:
					inner = _WholeMessageThinkInner(raw_content)
					if inner:
						out.append({"role": "assistant", "content": inner})
				continue
			item = {"role": "assistant", "content": content}
			AttachReasoning(item)
			out.append(item)
			continue

		buf_text = []
		pending_calls = []
		call_i = 0
		prior_text = _StealOpenAIPreface(out) if "<tool_call>" in content else ""

		def FlushText():
			nonlocal buf_text
			if not buf_text:
				return
			joined = "\n\n".join(buf_text).strip()
			buf_text = []
			if joined:
				msg = {"role": "assistant", "content": joined}
				AttachReasoning(msg)
				out.append(msg)

		def FlushCalls():
			nonlocal pending_calls, buf_text, prior_text
			if not pending_calls:
				return
			preface = "\n\n".join(buf_text).strip() if buf_text else ""
			buf_text = []
			if prior_text:
				preface = prior_text + ("\n\n" + preface if preface else "")
				prior_text = ""
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
			AttachReasoning(msg)
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
					tid = str(one.get("id") or "").strip() or f"call_{msg_i}_{call_i}"
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
				_name, body, resp_id = ParseToolResponseInner(payload)
				tid = TakePendingToolId(
					pending_ids, resp_id, fallback=f"call_{msg_i}_orphan_{call_i}",
				)
				out.append({
					"role": "tool",
					"tool_call_id": tid,
					"content": body if isinstance(body, str) else str(body),
				})
		FlushCalls()
		if prior_text:
			buf_text.insert(0, prior_text)
			prior_text = ""
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


def _PromptCacheOn():
	on = getattr(E, "remote_prompt_cache", 1)
	try:
		on = int(on)
	except (TypeError, ValueError):
		on = 1
	return bool(on)


def ApplyAnthropicCacheFields(payload):
	"""Top-level automatic ephemeral cache when remote_prompt_cache is on."""
	if _PromptCacheOn():
		payload["cache_control"] = {"type": "ephemeral"}
	else:
		payload.pop("cache_control", None)
	return payload


def ApplyOpenAICacheFields(payload):
	"""OpenRouter automatic cache breakpoint (Anthropic-style top-level cache_control).

	openai.com and transformers serve reject unknown body fields. Grok/OpenAI
	on OpenRouter ignore the field; Claude-via-OpenRouter uses it to advance
	the cache as the conversation grows.
	"""
	host = str(getattr(E, "remote_base_url", "") or "").strip().lower()
	if "openrouter.ai" not in host:
		payload.pop("cache_control", None)
		return payload
	if _PromptCacheOn():
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
	wire, native reasoning belongs in the sidecar (`reasoning_details` /
	Anthropic `thinking` blocks). An inline copy is a duplicate the model
	should not see replayed as its own words.
	"""
	s = str(text or "")
	while True:
		trimmed = _LEADING_THINK_RE.sub("", s, count=1)
		if trimmed == s:
			return s.strip()
		s = trimmed


_WHOLE_THINK_RE = re.compile(
	r"<(think|thoughts)\b[^>]*>(.*?)</\1>",
	re.IGNORECASE | re.DOTALL,
)


def _WholeMessageThinkInner(text):
	"""Inner text when the whole message is one think block, else empty."""
	s = str(text or "").strip()
	m = _WHOLE_THINK_RE.fullmatch(s)
	return m.group(2).strip() if m else ""


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
				_name, body, resp_id = ParseToolResponseInner(payload)
				tid = TakePendingToolId(
					pending_ids, resp_id, fallback=f"toolu_{msg_i}_orphan_{call_i}",
				)
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
