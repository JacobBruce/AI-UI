"""Remote OpenAI-compatible chat backend.

Local transformers path stays in chat.GenReply. This module only handles remote.

OpenAI-compat chat uses the OpenAI Python SDK. Streaming always goes through
`with_streaming_response` plus our tolerant SSE helpers (skip malformed chunks,
soft-keep partials). Empty Remote Base URL → Hugging Face router
(`https://router.huggingface.co/v1`). Anthropic Messages stays on the Anthropic SDK.
"""

from __future__ import annotations

import base64
import json
import os
import threading

E = None

_stop_callbacks = []
_stop_cb_lock = threading.Lock()
_active_close = None
_active_close_lock = threading.Lock()

from lib.media_types import (
	REMOTE_IMAGE_MAX_BYTES,
	REMOTE_VIDEO_MAX_BYTES,
	ImageMime as _ImageMime,
	VideoMime as _VideoMime,
)


class RemoteStreamServerError(RuntimeError):
	"""Remote sent an SSE/JSON error payload mid-stream (e.g. CUDA OOM).

	Not a client bug - keep any partial tokens already received.
	"""


def bind(engine_module):
	global E
	E = engine_module


def IsRemoteText():
	return str(getattr(E, "text_backend", "local") or "local").lower() == "remote"


def RegisterStopCallback(cb):
	with _stop_cb_lock:
		_stop_callbacks.append(cb)


def UnregisterStopCallback(cb):
	with _stop_cb_lock:
		try:
			_stop_callbacks.remove(cb)
		except ValueError:
			pass


def FireStopCallbacks():
	with _stop_cb_lock:
		cbs = list(_stop_callbacks)
	for cb in cbs:
		try:
			cb()
		except Exception:
			pass


def _SetActiveCloser(close_fn):
	global _active_close
	with _active_close_lock:
		_active_close = close_fn


def _CloseActiveStream():
	global _active_close
	with _active_close_lock:
		close_fn = _active_close
		_active_close = None
	if close_fn is None:
		return
	try:
		close_fn()
	except Exception:
		pass


def _RemoteApiKey():
	key = str(getattr(E, "remote_api_key", "") or "").strip()
	if key:
		return key
	return str(getattr(E, "hf_access_token", "") or "").strip()


# Hugging Face Inference Providers OpenAI-compatible router (chat only).
HF_OPENAI_BASE = "https://router.huggingface.co/v1"


def _ResolveOpenAIBaseUrl():
	base = str(getattr(E, "remote_base_url", "") or "").strip().rstrip("/")
	return base or HF_OPENAI_BASE


def _ResolveOpenAIModel(model_id):
	"""Append `:provider` on the HF router when Base URL is empty and model has no suffix."""
	mid = str(model_id or "").strip()
	configured = str(getattr(E, "remote_base_url", "") or "").strip()
	if configured:
		return mid
	provider = str(getattr(E, "remote_provider", "") or "").strip()
	if provider and ":" not in mid:
		return f"{mid}:{provider}"
	return mid


def _EnsureOpenAI():
	try:
		import openai  # noqa: F401
		return True
	except ImportError:
		return False


def _IsOpenRouterBase(base_url):
	host = str(base_url or "").strip().lower()
	return "openrouter.ai" in host


def _BuildOpenAIClient():
	"""OpenAI SDK client for OpenAI-compat remotes (OpenRouter, HF router, local serve, …).

	Empty API key: construct with a callable so the SDK skips Authorization
	(local Ollama / transformers serve). Non-empty key → Bearer as usual.
	OpenRouter: attach ranking attribution headers only when base_url is OpenRouter
	(X-OpenRouter-Title is vendor-specific; Referer alone is harmless but unused elsewhere).
	"""
	from openai import OpenAI
	import httpx
	from lib import ipc

	base_url = _ResolveOpenAIBaseUrl()
	api_key = _RemoteApiKey()
	provider = str(getattr(E, "remote_provider", "") or "").strip() or None
	timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
	kw = {
		"base_url": base_url,
		"timeout": timeout,
		# Callable returning "" → no Authorization header (auth_headers skips empty key).
		"api_key": api_key if api_key else (lambda: ""),
	}
	if _IsOpenRouterBase(base_url):
		kw["default_headers"] = {
			"HTTP-Referer": "https://bitfreak.info/?page=aiui",
			"X-OpenRouter-Title": "AI UI",
		}
	ipc.LogErr(
		f"Remote text backend: base_url={base_url!r} provider={provider!r} "
		f"model={getattr(E, 'remote_model', '')!r}"
	)
	return OpenAI(**kw)



def _IsHttpUrl(s):
	s = str(s or "").strip().lower()
	return s.startswith("http://") or s.startswith("https://")


def _MediaUrlPart(src, ipc, *, part_type, url_key, mime_fn, label, max_bytes):
	"""Build an OpenAI image_url / video_url part from a local path or http(s) URL."""
	src = str(src or "").strip()
	if not src:
		return None
	if _IsHttpUrl(src) or src.startswith("data:"):
		return {"type": part_type, url_key: {"url": src}}
	if not os.path.isfile(src):
		ipc.LogErr(f"WARN: remote {label} not found, skipping: {src}")
		return None
	try:
		size = os.path.getsize(src)
	except OSError as e:
		ipc.LogErr(f"WARN: remote {label} stat failed, skipping {src!r} ({e})")
		return None
	if size > max_bytes:
		ipc.LogErr(
			f"WARN: remote {label} exceeds {max_bytes // (1024 * 1024)} MiB, "
			f"skipping: {src} ({size} bytes)"
		)
		return None
	try:
		with open(src, "rb") as f:
			raw = f.read()
	except OSError as e:
		ipc.LogErr(f"WARN: remote {label} read failed, skipping {src!r} ({e})")
		return None
	mime = mime_fn(src)
	b64 = base64.b64encode(raw).decode("ascii")
	return {"type": part_type, url_key: {"url": f"data:{mime};base64,{b64}"}}


def _ImageUrlPart(src, ipc, warn_once=None):
	return _MediaUrlPart(
		src, ipc, part_type="image_url", url_key="image_url",
		mime_fn=_ImageMime, label="image", max_bytes=REMOTE_IMAGE_MAX_BYTES,
	)


def _VideoUrlPart(src, ipc, warn_once=None):
	return _MediaUrlPart(
		src, ipc, part_type="video_url", url_key="video_url",
		mime_fn=_VideoMime, label="video", max_bytes=REMOTE_VIDEO_MAX_BYTES,
	)


def _PartForRemote(part, ipc, warn_once):
	"""Map one local multimodal content dict to an OpenAI content part (or None to skip)."""
	if not isinstance(part, dict):
		return None
	ptype = str(part.get("type") or "").lower()
	if ptype == "text" or ptype in ("input_text",):
		text = part.get("text")
		if text is None:
			return None
		return {"type": "text", "text": str(text)}
	if ptype == "image":
		# HF / Qwen style: {"type": "image", "image": path|url}
		src = part.get("image")
		if src is None:
			src = part.get("url")
		if src is None:
			for k, v in part.items():
				if k != "type" and isinstance(v, str):
					src = v
					break
		return _ImageUrlPart(src, ipc, warn_once)
	if ptype in ("image_url", "input_image"):
		iu = part.get("image_url")
		if iu is None:
			iu = part.get("url")
		if isinstance(iu, dict) and isinstance(iu.get("url"), str):
			url = iu["url"]
			if _IsHttpUrl(url) or url.startswith("data:"):
				return {"type": "image_url", "image_url": {"url": url}}
			return _ImageUrlPart(url, ipc, warn_once)
		if isinstance(iu, str):
			if _IsHttpUrl(iu) or iu.startswith("data:"):
				return {"type": "image_url", "image_url": {"url": iu}}
			return _ImageUrlPart(iu, ipc, warn_once)
		return None
	if ptype in ("video_url",):
		vu = part.get("video_url")
		if isinstance(vu, dict) and isinstance(vu.get("url"), str):
			return _VideoUrlPart(vu["url"], ipc, warn_once)
		if isinstance(vu, str):
			return _VideoUrlPart(vu, ipc, warn_once)
		return None
	if ptype == "video":
		src = part.get("video") or part.get("url")
		return _VideoUrlPart(src, ipc, warn_once)
	if ptype in ("audio", "input_audio", "audio_url"):
		if "skip_audio" not in warn_once:
			warn_once.add("skip_audio")
			ipc.LogErr("WARN: remote backend does not send audio attachments; skipping")
		return None
	return None


def _ContentForRemote(content, ipc=None, warn_once=None):
	"""Map engine message content to OpenAI string or content-part list.

	Paths stay in E.messages; encoding to data URLs happens here at request time.
	"""
	from lib import ipc as ipc_mod

	if ipc is None:
		ipc = ipc_mod
	if warn_once is None:
		warn_once = set()
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	if not isinstance(content, (list, tuple)):
		return str(content)
	parts = []
	for item in content:
		mapped = _PartForRemote(item, ipc, warn_once)
		if mapped is not None:
			parts.append(mapped)
	if not parts:
		return ""
	# Single plain text part -> string (smaller payloads for text-only attaches)
	if len(parts) == 1 and parts[0].get("type") == "text":
		return parts[0].get("text") or ""
	return parts


def CountRemotePromptTokens(messages, tools=None):
	"""Best-effort remote token count. Returns int or None if unavailable.

	Prefers provider count APIs (Anthropic count_tokens, OpenAI responses/input_tokens).
	When tools are provided and the provider accepts them, includes them in the count;
	otherwise adds a heuristic estimate of the tool JSON on top of the message count.
	"""
	import json
	from lib import remote_tools

	model_id = str(getattr(E, "remote_model", "") or "").strip()
	if not model_id or not messages:
		return None

	def _ToolsHeuristic(schemas):
		if not schemas:
			return 0
		try:
			blob = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
		except Exception:
			blob = str(schemas)
		return max(0, (len(blob) + 3) // 4)

	def _ChatToolsToResponses(schemas):
		"""Chat Completions tools= → Responses input_tokens tools shape."""
		out = []
		for t in schemas or []:
			if not isinstance(t, dict):
				continue
			if t.get("type") == "function" and isinstance(t.get("function"), dict):
				fn = t["function"]
				out.append({
					"type": "function",
					"name": fn.get("name") or "",
					"description": fn.get("description") or "",
					"parameters": fn.get("parameters") or {"type": "object", "properties": {}},
				})
			elif t.get("type") == "function" and t.get("name"):
				out.append({
					"type": "function",
					"name": t.get("name") or "",
					"description": t.get("description") or "",
					"parameters": t.get("parameters") or {"type": "object", "properties": {}},
				})
			elif t.get("name"):
				out.append({
					"type": "function",
					"name": t.get("name") or "",
					"description": t.get("description") or "",
					"parameters": t.get("parameters") or t.get("inputSchema") or {
						"type": "object", "properties": {},
					},
				})
		return out

	def _ContentToResponsesInput(content):
		"""OpenAI chat content → Responses input parts (text + images)."""
		if content is None:
			return ""
		if isinstance(content, str):
			return content
		if not isinstance(content, list):
			return str(content)
		parts = []
		for p in content:
			if not isinstance(p, dict):
				continue
			ptype = str(p.get("type") or "")
			if ptype == "text":
				parts.append({"type": "input_text", "text": str(p.get("text") or "")})
			elif ptype == "image_url":
				img = p.get("image_url")
				url = img.get("url") if isinstance(img, dict) else (img if isinstance(img, str) else "")
				url = str(url or "").strip()
				if url:
					parts.append({"type": "input_image", "image_url": url})
			elif ptype == "input_text":
				parts.append({"type": "input_text", "text": str(p.get("text") or "")})
			elif ptype == "input_image":
				parts.append(p)
		if not parts:
			return ""
		if len(parts) == 1 and parts[0].get("type") == "input_text":
			return parts[0].get("text") or ""
		return parts

	tools_n = _ToolsHeuristic(tools) if tools else 0

	# Anthropic Messages count_tokens
	if remote_tools.IsAnthropicTools():
		if not _EnsureAnthropic():
			return None
		try:
			system, wire = remote_tools.MapMessagesAnthropic(list(messages))
			client = _BuildAnthropicClient()
			kw = {"model": model_id, "messages": wire}
			if system:
				kw["system"] = system
			if tools and getattr(E, "enable_tooluse", 0):
				ant_tools = remote_tools.ToolsForAnthropic()
				if ant_tools:
					kw["tools"] = ant_tools
					tools_n = 0  # included in provider count
			resp = client.messages.count_tokens(**kw)
			n = getattr(resp, "input_tokens", None)
			if n is None and isinstance(resp, dict):
				n = resp.get("input_tokens")
			if n is None:
				return None
			return int(n) + int(tools_n)
		except Exception as e:
			from lib import ipc
			ipc.LogErr(f"WARN: Anthropic count_tokens failed ({e})")
			return None

	# OpenAI-compat: try Responses input_tokens when available
	if not _EnsureOpenAI():
		return None
	try:
		client = _BuildOpenAIClient()
		wire_model = _ResolveOpenAIModel(model_id)
		inp = []
		for m in messages:
			if not isinstance(m, dict):
				continue
			role = str(m.get("role") or "user")
			content = _ContentToResponsesInput(m.get("content"))
			inp.append({"role": role, "content": content if content != "" else ""})
		count_fn = getattr(getattr(getattr(client, "responses", None), "input_tokens", None), "count", None)
		if callable(count_fn):
			kw = {"model": wire_model, "input": inp}
			resp_tools = _ChatToolsToResponses(tools) if (tools and getattr(E, "enable_tooluse", 0)) else []
			if resp_tools:
				kw["tools"] = resp_tools
			resp = count_fn(**kw)
			n = getattr(resp, "input_tokens", None)
			if n is None and isinstance(resp, dict):
				n = resp.get("input_tokens")
			if n is not None:
				# tools included when passed; otherwise keep heuristic add-on
				if resp_tools:
					tools_n = 0
				return int(n) + int(tools_n)
	except Exception as e:
		from lib import ipc
		ipc.LogErr(f"WARN: OpenAI input_tokens count failed ({e})")
	return None


def MessagesForRemote():
	"""Map engine chat messages to OpenAI-style role/content dicts.

	Multimodal list content becomes OpenAI content parts (images as image_url).
	Paths are encoded here - not stored as base64 in E.messages.
	Uses budgeted model messages when Phase 4 compaction is active.
	"""
	from lib import chat as chat_mod
	from lib import ipc

	raw = chat_mod._MessagesForModel()
	user = str(getattr(E, "user_name", "user") or "user")
	bot = str(getattr(E, "bot_name", "assistant") or "assistant")
	warn_once = set()
	out = []
	for m in raw:
		if not isinstance(m, dict):
			continue
		role = str(m.get("role") or "user")
		if role == bot or role == "assistant":
			role = "assistant"
		elif role == user or role == "user":
			role = "user"
		elif role == "system":
			role = "system"
		else:
			role = "user"
		content = _ContentForRemote(m.get("content", ""), ipc, warn_once)
		# Assistant/system stay strings in 3b (flatten accidental lists)
		if role in ("assistant", "system") and isinstance(content, list):
			texts = []
			for p in content:
				if isinstance(p, dict) and p.get("type") == "text":
					texts.append(str(p.get("text") or ""))
			content = "".join(texts)
		item = {"role": role, "content": content}
		# Anthropic wire continuity (thinking signatures / tool_use ids)
		if "_anthropic_content" in m and m.get("_anthropic_content") is not None:
			item["_anthropic_content"] = m["_anthropic_content"]
		if "_reasoning_details" in m and m.get("_reasoning_details") is not None:
			item["_reasoning_details"] = m["_reasoning_details"]
		if "_reasoning_pin" in m and m.get("_reasoning_pin") is not None:
			item["_reasoning_pin"] = m["_reasoning_pin"]
		out.append(item)
	return out


def BuildRemoteChatPayload(model_id, messages, temperature, max_tokens, stream=True, extra=None):
	"""Assemble OpenAI chat.completions kwargs for the OpenAI SDK path.

	ApplyOpenAIToolFields adds tools=/tool_choice for the OpenAI remote protocol.
	Thinking/effort → chat_template_kwargs (serve) + top-level reasoning_effort (native OpenAI).
	Phase 4 may pass already-budgeted messages from a context selector.
	Does not send stop= — server-side EOS/template config owns turn ending.
	openai.com gets no sampling params and `max_completion_tokens` (see below).
	"""
	from lib import remote_tools

	payload = {
		"model": model_id,
		"messages": messages,
		"stream": bool(stream),
		"max_tokens": int(max_tokens),
	}
	if remote_tools.IsOfficialOpenAI():
		# openai.com reasoning models (o-series, gpt-5) reject non-default
		# temperature / top_p / penalties, and reject `max_tokens` in favour of
		# `max_completion_tokens`. gpt-4o-class models accept the new spelling,
		# so one host rule covers every openai.com model - at the cost of
		# sampling control on the models that would still honour it.
		payload["max_completion_tokens"] = payload.pop("max_tokens")
	else:
		payload["temperature"] = float(temperature)
		payload["top_p"] = float(getattr(E, "top_p", 1.0) or 1.0)
		fp = float(getattr(E, "frequency_penalty", 0.0) or 0.0)
		pp = float(getattr(E, "presence_penalty", 0.0) or 0.0)
		if fp != 0.0:
			payload["frequency_penalty"] = fp
		if pp != 0.0:
			payload["presence_penalty"] = pp
	# Final SSE chunk can carry usage when the server honors this (OpenAI + many hosts).
	if payload.get("stream") and _WantStreamUsageMeta():
		payload["stream_options"] = {"include_usage": True}
	if extra:
		payload.update(extra)
	remote_tools.ApplyOpenAIThinkingFields(payload)
	remote_tools.ApplyOpenAIToolFields(payload)
	remote_tools.ApplyOpenAICacheFields(payload)
	return payload


def _WantStreamUsageMeta():
	"""True when the remote host is likely to accept stream_options.include_usage."""
	base = str(getattr(E, "remote_base_url", "") or "").strip().lower()
	if not base:
		return True  # default HF router
	keys = (
		"openai.com", "api.openai", "openrouter.ai", "together.xyz",
		"fireworks.ai", "groq.com", "deepseek.com", "huggingface.co",
	)
	return any(k in base for k in keys)


def _CombineDeltaParts(content, reasoning, think_state):
	"""Merge OpenAI content + optional reasoning_content into AI UI text.

	Older transformers serve only `content` (may already include <think> XML).
	Newer builds may stream chain-of-thought in `reasoning_content` while
	`content` stays null until the final answer - wrap that channel in <think>.
	"""
	if think_state is None:
		think_state = {"open": False}
	parts = []
	# Treat whitespace-only the same as empty so we never open a blank think block.
	if isinstance(reasoning, str) and reasoning.strip() != "":
		if not think_state.get("open"):
			think_state["open"] = True
			parts.append("<think>\n")
		parts.append(reasoning)
	if isinstance(content, str) and content != "":
		if think_state.get("open"):
			think_state["open"] = False
			parts.append("\n</think>\n\n")
		parts.append(content)
	return "".join(parts)


def _FormatRemoteError(err):
	if isinstance(err, str):
		return err
	try:
		return json.dumps(err)
	except Exception:
		return str(err)


def _DeltaFromPayload(obj, think_state=None):
	"""Extract assistant text from an OpenAI-style SSE JSON object."""
	if not isinstance(obj, dict):
		return ""
	if obj.get("error") is not None:
		raise RemoteStreamServerError(_FormatRemoteError(obj.get("error")))
	choices = obj.get("choices") or []
	if not choices:
		return ""
	choice0 = choices[0] if isinstance(choices[0], dict) else {}
	delta = choice0.get("delta") or {}
	if isinstance(delta, dict):
		content = delta.get("content")
		# Newer servers: reasoning_content; some drafts used "reasoning".
		reasoning = delta.get("reasoning_content")
		if reasoning is None:
			reasoning = delta.get("reasoning")
		if not (isinstance(reasoning, str) and reasoning.strip()):
			from lib import remote_tools
			extra = remote_tools.ReasoningTextFromDetails(delta.get("reasoning_details"))
			reasoning = extra or None
		text = _CombineDeltaParts(
			content if isinstance(content, str) else None,
			reasoning if isinstance(reasoning, str) else None,
			think_state,
		)
		if text:
			return text
	msg = choice0.get("message") or {}
	if isinstance(msg, dict):
		content = msg.get("content")
		reasoning = msg.get("reasoning_content")
		if reasoning is None:
			reasoning = msg.get("reasoning")
		if not (isinstance(reasoning, str) and reasoning.strip()):
			from lib import remote_tools
			extra = remote_tools.ReasoningTextFromDetails(msg.get("reasoning_details"))
			reasoning = extra or None
		return _CombineDeltaParts(
			content if isinstance(content, str) else None,
			reasoning if isinstance(reasoning, str) else None,
			think_state,
		)
	return ""


def _ConsumeSseEvent(line, ipc, think_state):
	"""Parse one SSE line. Return event dict, None to skip, or raise StopIteration on [DONE].

	Event keys: text (str), tool_calls (list|None), finish_reason (str|None), usage (obj|None).
	"""
	if line is None:
		return None
	if isinstance(line, bytes):
		line = line.decode("utf-8", errors="replace")
	line = line.strip()
	if not line:
		return None
	# Some servers emit event: / id: lines - ignore those.
	if not line.startswith("data:"):
		return None
	data = line[5:].strip()
	if data == "[DONE]":
		raise StopIteration
	try:
		obj = json.loads(data)
	except json.JSONDecodeError as e:
		ipc.LogErr(f"WARN: skipping malformed SSE chunk ({e}): {data[:160]!r}")
		return None
	if not isinstance(obj, dict):
		return None
	if obj.get("error") is not None:
		raise RemoteStreamServerError(_FormatRemoteError(obj.get("error")))
	usage = obj.get("usage")
	route_model = obj.get("model") if isinstance(obj.get("model"), str) else None
	route_provider = obj.get("provider") if isinstance(obj.get("provider"), str) else None
	choices = obj.get("choices") or []
	if not choices:
		# OpenAI include_usage final chunk often has empty choices + usage only.
		if usage or route_model or route_provider:
			return {
				"text": "",
				"tool_calls": None,
				"finish_reason": None,
				"usage": usage,
				"reasoning_details": None,
				"model": route_model,
				"provider": route_provider,
			}
		return None
	choice0 = choices[0] if isinstance(choices[0], dict) else {}
	delta = choice0.get("delta") or {}
	if not isinstance(delta, dict):
		delta = {}
	text = _DeltaFromPayload(obj, think_state)
	details = None
	if isinstance(delta, dict):
		details = delta.get("reasoning_details")
	if not details:
		msg = choice0.get("message") or {}
		if isinstance(msg, dict):
			details = msg.get("reasoning_details")
	return {
		"text": text or "",
		"tool_calls": delta.get("tool_calls"),
		"finish_reason": choice0.get("finish_reason"),
		"usage": usage,
		"reasoning_details": details,
		"model": route_model,
		"provider": route_provider,
	}


def _ConsumeSseLine(line, ipc, think_state):
	"""Return content delta string, None to skip, or raise StopIteration on [DONE]."""
	evt = _ConsumeSseEvent(line, ipc, think_state)
	if evt is None:
		return None
	return evt.get("text") or ""


def _HandleDelta(piece, state, want_ui, on_block, block_ends, ipc):
	"""Apply one content piece. Returns 'continue', 'tool_stop', or 'cancel'."""
	if state["cancel"].is_set() or ipc.StopRequested():
		state["stopped"] = True
		return "cancel"
	delta = piece.replace("\n", "[AI_UI_BR]").replace("\r", "")
	state["gen_text"] += delta
	plain = state["gen_text"].replace("[AI_UI_BR]", "\n")
	# Remote OpenAI protocol uses structured tool_calls; do not early-stop on XML.
	if state.get("xml_tool_stop") and getattr(E, "enable_tooluse", 0) and "</tool_call>" in plain:
		cut = plain.find("</tool_call>") + len("</tool_call>")
		plain = plain[:cut]
		state["gen_text"] = plain.replace("\n", "[AI_UI_BR]")
		if want_ui:
			ipc.SendStreamDelta(delta, "chat")
		return "tool_stop"
	if want_ui:
		ipc.SendStreamDelta(delta, "chat")
	elif on_block is not None:
		n_blocks = sum(plain.count(tag) for tag in block_ends)
		if n_blocks > state["last_blocks"]:
			state["last_blocks"] = n_blocks
			try:
				on_block(plain)
			except Exception as e:
				ipc.LogException("remote_stream_on_block", e)
	if state["cancel"].is_set() or ipc.StopRequested():
		state["stopped"] = True
		return "cancel"
	return "continue"


def _FlushOpenAIToolCalls(state, want_ui, on_block, block_ends, ipc):
	"""If streamed tool_calls were accumulated, append synthetic XML. Returns True if flushed."""
	from lib import remote_tools

	acc = state.get("tool_acc")
	if not acc:
		return False
	calls = remote_tools.FinalizeOpenAIToolCalls(acc)
	state["tool_acc"] = remote_tools.NewOpenAIToolCallAcc()
	if not calls:
		return False
	_CloseOpenThink(state, want_ui, on_block, block_ends, ipc)
	plain = state["gen_text"].replace("[AI_UI_BR]", "\n")
	plain = remote_tools.AppendToolCallsXml(plain, calls)
	xml = remote_tools.ToolCallsToXml(calls)
	state["gen_text"] = plain.replace("\n", "[AI_UI_BR]")
	if want_ui and xml:
		# Stream only the synthesized XML (preface content already painted).
		ipc.SendStreamDelta(xml.replace("\n", "[AI_UI_BR]").replace("\r", ""), "chat")
	return True


def _RememberStreamRoute(state, model=None, provider=None):
	"""Keep a resolved OpenRouter model if a later chunk repeats the auto id."""
	from lib import remote_tools

	if isinstance(model, str) and model.strip():
		prev = state.get("route_model")
		if (
			not prev
			or remote_tools.IsOpenRouterAutoModel(prev)
			or not remote_tools.IsOpenRouterAutoModel(model)
		):
			state["route_model"] = model
	if isinstance(provider, str) and provider.strip():
		state["route_provider"] = provider


def _ApplyStreamEvent(evt, state, want_ui, on_block, block_ends, ipc):
	"""Handle one parsed stream event. Returns 'continue' or a stop action."""
	from lib import remote_tools
	from lib import context_compact

	if not evt:
		return "continue"
	if evt.get("usage") is not None:
		context_compact.RecordRemoteUsage(evt.get("usage"))
	_RememberStreamRoute(state, evt.get("model"), evt.get("provider"))
	rds = evt.get("reasoning_details")
	if rds:
		acc = state.setdefault("reasoning_acc", [])
		remote_tools.AccumulateReasoningDetails(acc, rds)
		state["reasoning_acc"] = acc
	text = evt.get("text") or ""
	if text:
		action = _HandleDelta(text, state, want_ui, on_block, block_ends, ipc)
		if action != "continue":
			return action
	tcs = evt.get("tool_calls")
	if tcs and state.get("oai_tools"):
		remote_tools.AccumulateOpenAIToolCalls(state["tool_acc"], tcs)
	fr = evt.get("finish_reason")
	if fr == "tool_calls" and state.get("oai_tools"):
		if _FlushOpenAIToolCalls(state, want_ui, on_block, block_ends, ipc):
			return "tool_stop"
	return "continue"


def _StreamViaOpenAI(client, create_kw, state, want_ui, on_block, block_ends, ipc):
	"""OpenAI SDK streaming via with_streaming_response + tolerant SSE parse.

	Owns the line loop so malformed chunks / mid-stream errors soft-keep partials.
	"""
	kw = _CreateKwForOpenAI(create_kw)
	ipc.LogErr(f"Remote SSE via OpenAI SDK model={kw.get('model')!r}")
	resp = None
	try:
		with client.chat.completions.with_streaming_response.create(**kw) as resp:
			_SetActiveCloser(lambda r=resp: r.close())
			try:
				tool_stop = False
				for line in resp.iter_lines():
					if state["cancel"].is_set() or ipc.StopRequested():
						state["stopped"] = True
						break
					try:
						evt = _ConsumeSseEvent(line, ipc, state["think"])
					except StopIteration:
						break
					except RemoteStreamServerError as e:
						ipc.LogErr(f"WARN: remote stream aborted by server; keeping partial reply ({e})")
						state["stopped"] = True
						break
					if not evt:
						continue
					if tool_stop:
						if evt.get("usage") is not None:
							from lib import context_compact
							context_compact.RecordRemoteUsage(evt.get("usage"))
						_RememberStreamRoute(state, evt.get("model"), evt.get("provider"))
						continue
					action = _ApplyStreamEvent(evt, state, want_ui, on_block, block_ends, ipc)
					if action == "tool_stop":
						tool_stop = True
						continue
					if action != "continue":
						break
				if state.get("oai_tools") and state.get("tool_acc"):
					_FlushOpenAIToolCalls(state, want_ui, on_block, block_ends, ipc)
			except RemoteStreamServerError as e:
				ipc.LogErr(f"WARN: remote stream aborted by server; keeping partial reply ({e})")
				state["stopped"] = True
			except Exception as e:
				if state["gen_text"]:
					ipc.LogErr(f"WARN: remote stream failed; keeping partial reply ({e})")
					state["stopped"] = True
				else:
					raise
	except Exception as e:
		# Connection / HTTP errors before or around the stream context.
		if state["gen_text"]:
			ipc.LogErr(f"WARN: remote stream failed; keeping partial reply ({e})")
			state["stopped"] = True
		else:
			raise
	finally:
		_FinalizeOpenThink(state, want_ui, ipc)
		_StoreOpenAIReasoningSidecar(state)
		_SetActiveCloser(None)


def _CloseOpenThink(state, want_ui, on_block, block_ends, ipc):
	"""Close a synthetic think wrapper. Does not go through `_HandleDelta`.

	Stop sets cancel before this runs; `_HandleDelta` would skip the close tag.
	`on_block` / `block_ends` are unused (kept so flush call sites stay the same).
	"""
	del on_block, block_ends
	if not state["think"].get("open"):
		return
	state["think"]["open"] = False
	delta = "\n</think>\n\n".replace("\n", "[AI_UI_BR]")
	state["gen_text"] += delta
	if want_ui:
		ipc.SendStreamDelta(delta, "chat")


def _FinalizeOpenThink(state, want_ui, ipc):
	"""End-of-stream: unwrap a think-only reply, else close the wrapper."""
	if not state["think"].get("open"):
		return
	from lib.files import CloseOrUnwrapOpenThink

	state["think"]["open"] = False
	plain = state["gen_text"].replace("[AI_UI_BR]", "\n")
	new = CloseOrUnwrapOpenThink(plain)
	state["gen_text"] = new.replace("\n", "[AI_UI_BR]")
	if want_ui and new.startswith(plain) and len(new) > len(plain):
		delta = new[len(plain):].replace("\n", "[AI_UI_BR]").replace("\r", "")
		if delta:
			ipc.SendStreamDelta(delta, "chat")


def _AnthropicApiKey():
	"""Anthropic requires remote_api_key (no HF token fallback)."""
	return str(getattr(E, "remote_api_key", "") or "").strip()


def _AnthropicBaseUrl():
	base = str(getattr(E, "remote_base_url", "") or "").strip().rstrip("/")
	if not base:
		return None  # SDK default api.anthropic.com
	# Allow either https://api.anthropic.com or …/v1
	if base.endswith("/v1"):
		base = base[:-3].rstrip("/")
	return base or None


def _AnthropicIsOfficial():
	"""True when the request goes to Anthropic's own API (empty Base URL = default)."""
	base = str(getattr(E, "remote_base_url", "") or "").strip().lower()
	return not base or "api.anthropic.com" in base


def _EnsureAnthropic():
	try:
		import anthropic  # noqa: F401
		return True
	except ImportError:
		return False


def _AnthropicTimeoutSeconds(max_tokens):
	"""Read timeout (seconds) for non-streaming Anthropic `messages.create`.

	The SDK raises ValueError on non-streaming create when the client still has
	DEFAULT_TIMEOUT and it estimates the call will run longer than 10 minutes
	(`3600 * max_tokens / 128000`), or when `max_tokens` exceeds a per-model
	non-streaming cap. Passing an explicit `timeout` on that create skips the
	check. Streaming and `count_tokens` keep the SDK default (10 min read,
	5s connect). Floor is 10 minutes; above that, same estimate as the SDKs.
	"""
	try:
		n = int(max_tokens or 0)
	except (TypeError, ValueError):
		n = 0
	n = max(1, n)
	minimum = 10 * 60
	calculated = (60 * 60 * n) / 128_000.0
	return float(minimum if calculated < minimum else calculated)


def _AnthropicNonStreamTimeout(max_tokens):
	"""Timeout for non-streaming create only — long read, fast connect fail."""
	seconds = _AnthropicTimeoutSeconds(max_tokens)
	for mod_name in ("httpx2", "httpx"):
		try:
			mod = __import__(mod_name)
			return mod.Timeout(seconds, connect=5.0)
		except Exception:
			continue
	return seconds


def _BuildAnthropicClient():
	import anthropic

	api_key = _AnthropicApiKey()
	if not api_key:
		raise RuntimeError("Remote API Key is required for Anthropic Messages")
	# Anthropic's own API authenticates with `x-api-key`; third-party Messages
	# endpoints (gateways, DeepSeek / GLM / Kimi) expect `Authorization: Bearer`.
	# Either explicit kwarg also stops the SDK reading ANTHROPIC_* credentials.
	# Do not set timeout here: streaming and count_tokens should keep the SDK
	# default (10 min read, 5s connect). Non-stream create passes timeout itself.
	if _AnthropicIsOfficial():
		kw = {"api_key": api_key}
	else:
		kw = {"auth_token": api_key}
	base = _AnthropicBaseUrl()
	if base:
		kw["base_url"] = base
	return anthropic.Anthropic(**kw)


def BuildAnthropicMessagesPayload(model_id, system, messages, max_tokens, stream=True):
	"""Assemble Anthropic messages.create kwargs.

	No sampling params: Claude 4.7+ / Sonnet 5 / Fable 5 reject non-default
	temperature / top_p / top_k on every request, and older models reject
	temperature whenever thinking is on. Reasoning Effort is the only lever left.
	"""
	from lib import remote_tools

	payload = {
		"model": model_id,
		"messages": messages,
		"max_tokens": int(max_tokens),
		"stream": bool(stream),
	}
	if system:
		payload["system"] = system
	remote_tools.ApplyAnthropicToolFields(payload)
	remote_tools.ApplyAnthropicThinkingFields(payload)
	remote_tools.ApplyAnthropicCacheFields(payload)
	return payload


def _FlushAnthropicToolUse(state, want_ui, on_block, block_ends, ipc):
	from lib import remote_tools

	acc = state.get("tool_acc")
	if not acc:
		return False
	calls = remote_tools.FinalizeAnthropicToolUse(acc)
	state["tool_acc"] = remote_tools.NewAnthropicToolUseAcc()
	if not calls:
		return False
	_CloseOpenThink(state, want_ui, on_block, block_ends, ipc)
	plain = state["gen_text"].replace("[AI_UI_BR]", "\n")
	plain = remote_tools.AppendToolCallsXml(plain, calls)
	xml = remote_tools.ToolCallsToXml(calls)
	state["gen_text"] = plain.replace("\n", "[AI_UI_BR]")
	if want_ui and xml:
		ipc.SendStreamDelta(xml.replace("\n", "[AI_UI_BR]").replace("\r", ""), "chat")
	return True


def _StoreAnthropicSidecar(content_blocks):
	"""Store the full Anthropic content array for later as-is replay."""
	from lib import remote_tools

	wire = remote_tools.ContentBlocksToWireList(content_blocks)
	E._anthropic_last_content = wire or None


def _WarnAnthropicStopReason(stop_reason, ipc):
	"""Surface truncation — thinking tokens count against max_tokens."""
	if str(stop_reason or "") != "max_tokens":
		return
	ipc.LogErr(
		"WARN: Anthropic reply was cut off at max_tokens "
		f"(Max Response Length = {int(getattr(E, 'gen_max', 0) or 0)}). "
		"Thinking counts toward that budget - raise it or lower Reasoning Effort."
	)


def _StreamViaAnthropic(create_kw, state, want_ui, on_block, block_ends, ipc):
	"""Anthropic SDK messages.stream → text / thinking / tool_use."""
	from lib import remote_tools

	client = _BuildAnthropicClient()
	# stream= is not a create kw when using .stream()
	kw = dict(create_kw)
	kw.pop("stream", None)
	stream_cm = client.messages.stream(**kw)
	stream_obj = stream_cm.__enter__()
	closed = {"done": False}

	def _close_stream():
		if closed["done"]:
			return
		closed["done"] = True
		try:
			stream_cm.__exit__(None, None, None)
		except Exception:
			pass

	_SetActiveCloser(_close_stream)
	final_message = None
	try:
		try:
			for event in stream_obj:
				if state["cancel"].is_set() or ipc.StopRequested():
					state["stopped"] = True
					break
				etype = getattr(event, "type", None)
				if etype == "message_start":
					# Only event carrying input/cache tokens; Stop skips the final message.
					started = getattr(event, "message", None)
					usage = getattr(started, "usage", None) if started is not None else None
					if usage is not None:
						from lib import context_compact
						context_compact.RecordRemoteUsage(usage)
				elif etype == "content_block_start":
					block = getattr(event, "content_block", None)
					idx = getattr(event, "index", 0)
					btype = getattr(block, "type", None) if block is not None else None
					if btype == "tool_use":
						remote_tools.AccumulateAnthropicToolUse(
							state["tool_acc"],
							index=idx,
							block={
								"type": "tool_use",
								"id": getattr(block, "id", "") or "",
								"name": getattr(block, "name", "") or "",
								"input": getattr(block, "input", None) or {},
							},
						)
				elif etype == "content_block_delta":
					delta = getattr(event, "delta", None)
					dtype = getattr(delta, "type", None) if delta is not None else None
					idx = getattr(event, "index", 0)
					if dtype == "text_delta":
						text = getattr(delta, "text", None) or ""
						if text:
							# Closes the <think> wrapper on the thinking → text switch.
							piece = _CombineDeltaParts(text, None, state["think"])
							action = _HandleDelta(piece, state, want_ui, on_block, block_ends, ipc)
							if action != "continue":
								break
					elif dtype == "thinking_delta":
						thinking = getattr(delta, "thinking", None) or ""
						if thinking:
							# Reuse OpenAI-style reasoning → <think> path
							piece = _CombineDeltaParts(None, thinking, state["think"])
							if piece:
								action = _HandleDelta(piece, state, want_ui, on_block, block_ends, ipc)
								if action != "continue":
									break
					elif dtype == "input_json_delta":
						partial = getattr(delta, "partial_json", None) or ""
						if partial:
							remote_tools.AccumulateAnthropicToolUse(
								state["tool_acc"], index=idx, partial_json=partial,
							)
				elif etype == "message_delta":
					delta = getattr(event, "delta", None)
					stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
					usage = getattr(event, "usage", None)
					if usage is not None:
						from lib import context_compact
						context_compact.RecordRemoteUsage(usage)
					if stop_reason:
						state["stop_reason"] = stop_reason
					if stop_reason == "tool_use":
						if _FlushAnthropicToolUse(state, want_ui, on_block, block_ends, ipc):
							break
				elif etype == "message_stop":
					break
			# Final message for sidecar (thinking signatures). get_final_message()
			# reads the stream to completion, so skip it when the user hit Stop.
			if not state["stopped"]:
				try:
					final_message = stream_obj.get_final_message()
				except Exception:
					final_message = None
			if state.get("tool_acc"):
				_FlushAnthropicToolUse(state, want_ui, on_block, block_ends, ipc)
			if final_message is not None:
				_StoreAnthropicSidecar(getattr(final_message, "content", None))
				from lib import context_compact
				context_compact.RecordRemoteUsage(getattr(final_message, "usage", None))
				_WarnAnthropicStopReason(getattr(final_message, "stop_reason", None), ipc)
			elif not state["stopped"]:
				_WarnAnthropicStopReason(state.get("stop_reason"), ipc)
		except Exception as e:
			if state["gen_text"]:
				ipc.LogErr(f"WARN: remote stream failed; keeping partial reply ({e})")
				state["stopped"] = True
			else:
				raise
	finally:
		_FinalizeOpenThink(state, want_ui, ipc)
		_SetActiveCloser(None)
		_close_stream()


def _NonStreamViaAnthropic(create_kw, state, want_ui, ipc):
	from lib import remote_tools

	client = _BuildAnthropicClient()
	kw = dict(create_kw)
	kw.pop("stream", None)
	timeout = _AnthropicNonStreamTimeout(kw.get("max_tokens"))
	resp = client.messages.create(timeout=timeout, **kw)
	content_blocks = getattr(resp, "content", None) or []
	_StoreAnthropicSidecar(content_blocks)
	from lib import context_compact
	context_compact.RecordRemoteUsage(getattr(resp, "usage", None))
	_WarnAnthropicStopReason(getattr(resp, "stop_reason", None), ipc)
	# Build text + thinking as <think>
	think_state = state["think"]
	parts = []
	for b in content_blocks:
		btype = getattr(b, "type", None)
		if btype == "thinking":
			t = getattr(b, "thinking", None) or ""
			if t:
				parts.append(_CombineDeltaParts(None, t, think_state))
		elif btype == "text":
			t = getattr(b, "text", None) or ""
			if t:
				parts.append(_CombineDeltaParts(t, None, think_state))
	state["gen_text"] = "".join(parts)
	calls = remote_tools.ToolUseFromContentBlocks(content_blocks)
	if calls:
		_CloseOpenThink(state, False, None, (), ipc)
		state["gen_text"] = remote_tools.AppendToolCallsXml(state["gen_text"], calls)
	else:
		_FinalizeOpenThink(state, False, ipc)
	if want_ui and state["gen_text"]:
		ipc.SendStreamDelta(state["gen_text"].replace("\n", "[AI_UI_BR]").replace("\r", ""), "chat")


def _RouteMetaFromResponse(resp):
	"""Resolved model/provider from a non-stream OpenAI/OpenRouter completion."""
	from lib import remote_tools

	model = getattr(resp, "model", None)
	provider = getattr(resp, "provider", None)
	extra = getattr(resp, "model_extra", None)
	if isinstance(extra, dict):
		if not model:
			model = extra.get("model")
		if not provider:
			provider = extra.get("provider")
	dump = remote_tools.SdkModelDump(resp)
	if isinstance(dump, dict):
		if not model:
			model = dump.get("model")
		if not provider:
			provider = dump.get("provider")
	if isinstance(resp, dict):
		if not model:
			model = resp.get("model")
		if not provider:
			provider = resp.get("provider")
	model = model if isinstance(model, str) and model.strip() else None
	provider = provider if isinstance(provider, str) and provider.strip() else None
	return model, provider


def _StoreOpenAIReasoningSidecar(state):
	from lib import remote_tools

	details = remote_tools.FinalizeReasoningDetails(state.get("reasoning_acc"))
	E._openai_last_reasoning_details = details
	pin = remote_tools.ReasoningPinFromDetails(
		details,
		model=state.get("route_model"),
		provider=state.get("route_provider"),
	)
	E._openai_last_reasoning_pin = pin


def _CreateKwForOpenAI(create_kw):
	"""OpenAI SDK: merge extra_body; serve-only chat_template_kwargs go under extra_body.

	OpenRouter reasoning_details is not a typed Chat Completions field — the SDK
	strips it from `messages`, so copy the mapped list into extra_body (OpenRouter
	only; openai.com rejects unknown message keys). cache_control and
	prompt_cache_key go through extra_body so an older SDK still forwards them.
	"""
	kw = dict(create_kw)
	extra = dict(kw.get("extra_body") or {})
	ct = kw.pop("chat_template_kwargs", None)
	if isinstance(ct, dict) and ct:
		extra["chat_template_kwargs"] = ct
	cc = kw.pop("cache_control", None)
	if cc is not None and _IsOpenRouterBase(_ResolveOpenAIBaseUrl()):
		extra["cache_control"] = cc
	cache_key = kw.pop("prompt_cache_key", None)
	if cache_key:
		extra["prompt_cache_key"] = cache_key
	msgs = kw.get("messages")
	if isinstance(msgs, list):
		public = [
			{k: v for k, v in m.items() if not str(k).startswith("_")}
			if isinstance(m, dict) else m
			for m in msgs
		]
		kw["messages"] = public
		if (
			_IsOpenRouterBase(_ResolveOpenAIBaseUrl())
			and any(isinstance(m, dict) and m.get("reasoning_details") for m in public)
		):
			extra["messages"] = public
	if extra:
		kw["extra_body"] = extra
	elif "extra_body" in kw:
		kw.pop("extra_body", None)
	return kw


def RemoteChatReply(temperature=None, stream=True, ui_stream=True, on_block=None, reuse_stream=False):
	"""Stream or collect a remote chat completion. Returns plain text (may include tool XML)."""
	from lib import ipc
	from lib import remote_tools

	# Vision (#am_select / mm_processor) is local-only and disabled when remote is on.
	# Do not refuse based on a stuck mm_processor=janus - remote always uses OpenAI-style
	# multimodal parts. Only block when the remote model id itself looks like Janus.
	if "janus" in str(getattr(E, "remote_model", "") or "").lower():
		raise RuntimeError(
			"Janus multimodal is local-only (VLChatProcessor). "
			"Stock transformers serve / OpenAI-compatible remotes are unsupported - "
			"use a non-Janus remote model, or switch Text Backend to local with Vision=Janus."
		)

	model_id = str(getattr(E, "remote_model", "") or "").strip()
	if not model_id:
		raise RuntimeError(
			"Remote model is empty - set Remote Model under Remote Settings."
		)

	messages = MessagesForRemote()
	if not messages:
		raise RuntimeError("No messages to send to the remote model.")

	# Clear prior sidecar unless this round reuses it via messages
	E._anthropic_last_content = None
	E._openai_last_reasoning_details = None
	E._openai_last_reasoning_pin = None

	ant_tools = remote_tools.IsAnthropicTools()
	oai_tools = remote_tools.IsOpenAITools()
	if ant_tools:
		if not _EnsureAnthropic():
			raise RuntimeError(
				"Anthropic Messages requires the anthropic package. "
				"Install with: pip install anthropic"
			)
		system, messages = remote_tools.MapMessagesAnthropic(messages)
		if not messages:
			raise RuntimeError(
				"No messages left after mapping for Anthropic Messages "
				"(transcript may be empty or all turns were filtered)."
			)
	elif oai_tools:
		messages = remote_tools.MapMessagesOpenAI(
			messages,
			replay_reasoning=_IsOpenRouterBase(_ResolveOpenAIBaseUrl()),
		)

	temp = float(E.rando_lvl if temperature is None else temperature)
	max_tokens = int(getattr(E, "gen_max", 32000) or 32000)
	want_stream = bool(stream)
	want_ui = bool(ui_stream) and want_stream

	block_ends = ("</think>", "</thoughts>", "</tool_call>", "</tool_response>")
	state = {
		"gen_text": "",
		"last_blocks": 0,
		"stopped": False,
		"cancel": threading.Event(),
		"think": {"open": False},
		"stop_reason": None,
		"xml_tool_stop": False,
		"oai_tools": oai_tools,
		"ant_tools": ant_tools,
		"reasoning_acc": [],
		"route_model": None,
		"route_provider": None,
		"tool_acc": (
			remote_tools.NewAnthropicToolUseAcc() if ant_tools
			else (remote_tools.NewOpenAIToolCallAcc() if oai_tools else None)
		),
	}

	def _on_stop():
		state["cancel"].set()
		_CloseActiveStream()

	def _done(text):
		from lib.files import TrimTrailingChatClosers
		plain = (text or "").replace("[AI_UI_BR]", "\n")
		return TrimTrailingChatClosers(plain)

	RegisterStopCallback(_on_stop)
	try:
		if want_ui and not reuse_stream:
			ipc.SendStreamStart("chat")

		if ant_tools:
			create_kw = BuildAnthropicMessagesPayload(
				model_id, system, messages, max_tokens, stream=want_stream,
			)
			if not want_stream:
				_NonStreamViaAnthropic(create_kw, state, want_ui, ipc)
				if want_ui:
					ipc.SendStreamEnd("ok", "chat")
				return _done(state["gen_text"])
			_StreamViaAnthropic(create_kw, state, want_ui, on_block, block_ends, ipc)
			if want_ui:
				ipc.SendStreamEnd("stop" if state["stopped"] else "ok", "chat")
			return _done(state["gen_text"])

		if not _EnsureOpenAI():
			raise RuntimeError(
				"OpenAI-compatible remote chat requires the openai package. "
				"Install with: pip install openai"
			)

		wire_model = _ResolveOpenAIModel(model_id)
		create_kw = BuildRemoteChatPayload(
			wire_model, messages, temp, max_tokens, stream=want_stream,
		)
		if _IsOpenRouterBase(_ResolveOpenAIBaseUrl()):
			create_kw = remote_tools.ApplyReasoningRoutePin(create_kw)
		client = _BuildOpenAIClient()

		if not want_stream:
			resp = client.chat.completions.create(**_CreateKwForOpenAI(create_kw))
			try:
				from lib import context_compact
				context_compact.RecordRemoteUsage(getattr(resp, "usage", None))
				msg = resp.choices[0].message
				rm, rp = _RouteMetaFromResponse(resp)
				_RememberStreamRoute(state, rm, rp)
				content = getattr(msg, "content", None)
				reasoning = getattr(msg, "reasoning_content", None)
				if reasoning is None:
					reasoning = getattr(msg, "reasoning", None)
				details = remote_tools.ReasoningDetailsFromMessage(msg)
				if details:
					state["reasoning_acc"] = list(details)
				if not (isinstance(reasoning, str) and reasoning.strip()) and details:
					reasoning = remote_tools.ReasoningTextFromDetails(details) or reasoning
				# Older servers: content only. Newer: optional reasoning_content.
				state["gen_text"] = _CombineDeltaParts(
					content if isinstance(content, str) else None,
					reasoning if isinstance(reasoning, str) else None,
					state["think"],
				)
				if oai_tools:
					calls = remote_tools.ToolCallsFromMessage(msg)
					if calls:
						_CloseOpenThink(state, False, None, (), ipc)
						state["gen_text"] = remote_tools.AppendToolCallsXml(state["gen_text"], calls)
				_FinalizeOpenThink(state, False, ipc)
				_StoreOpenAIReasoningSidecar(state)
			except (IndexError, AttributeError, TypeError) as e:
				ipc.LogErr(f"WARN: remote non-stream parse failed ({e})")
				raise RuntimeError(f"Remote reply could not be parsed ({e})") from e
			except Exception as e:
				ipc.LogException("remote_nonstream_parse", e)
				if not state["gen_text"]:
					raise RuntimeError(f"Remote reply could not be parsed ({e})") from e
			if want_ui and state["gen_text"]:
				ipc.SendStreamDelta(state["gen_text"].replace("\n", "[AI_UI_BR]").replace("\r", ""), "chat")
			if want_ui:
				ipc.SendStreamEnd("ok", "chat")
			return _done(state["gen_text"])

		_StreamViaOpenAI(client, create_kw, state, want_ui, on_block, block_ends, ipc)

		if want_ui:
			ipc.SendStreamEnd("stop" if state["stopped"] else "ok", "chat")
		return _done(state["gen_text"])
	except Exception as e:
		ipc.LogException("remote_chat", e)
		if want_ui:
			try:
				ipc.SendStreamEnd("error", "chat")
			except Exception:
				pass
		partial = _done(state["gen_text"]) if state["gen_text"] else ""
		# Soft-keep painted tokens; hard-fail (no text) so _RunBotTurn shows Generation failed.
		if partial:
			return partial
		raise
	finally:
		UnregisterStopCallback(_on_stop)
		_CloseActiveStream()
