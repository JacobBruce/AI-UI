# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module


import copy
import os

from lib import ipc

COMPACT_ROLE = "compact"

_DEFAULT_PROMPT = """Context compaction was triggered because the conversation exceeded the configured token budget.

Write a concise handoff summary for yourself after compaction (target well under 10K tokens).

After compaction you will only see the summary plus a short recent tail of the conversation.

Retain what is needed to continue: goals, decisions, open tasks, tool results that still matter, etc.

Prefer clarity over length - distill the context down to key information by omitting chatter and redundant tool noise.

If a prior compaction summary is present, merge anything still relevant into one updated summary.

Output ONLY the handoff summary. No preamble, planning, or commentary about writing the summary.
"""

_USER_PROMPT_NAME = "compact_prompt.txt"


def CompactEnabled():
	return int(getattr(E, "compact_at", 0) or 0) > 0


def ResetCompactState():
	E.compact_summary = ""
	E.compact_until = 0
	E.compact_count = 0


def FindLastCompactNotice(messages=None):
	"""Return (index, msg) for the last compact notice, or (-1, None)."""
	raw = list(messages if messages is not None else getattr(E, "messages", None) or [])
	idx = -1
	notice = None
	for i, m in enumerate(raw):
		if IsCompactNotice(m):
			idx = i
			notice = m
	return idx, notice


def DeriveKeepStart(messages=None):
	"""Keep-tail start from the last compact notice (0 when none).

	The notice records `keep_back` - how many messages before it belong to the
	keep-tail - so a compaction that folded the last turn stays folded. Notices
	from older sessions have no `keep_back`, so keep-tail is re-derived as
	FindKeepTailStart(messages[:notice_idx]).
	"""
	raw = list(messages if messages is not None else getattr(E, "messages", None) or [])
	notice_idx, notice = FindLastCompactNotice(raw)
	if notice_idx < 0:
		return 0
	if isinstance(notice, dict) and notice.get("keep_back") is not None:
		try:
			return max(0, notice_idx - int(notice["keep_back"]))
		except (TypeError, ValueError):
			pass
	if notice_idx == 0:
		return min(1, len(raw))
	return FindKeepTailStart(raw[:notice_idx])


def SyncCompactFromMessages(messages=None):
	"""Refresh in-memory compact summary/count from the last compact notice.

	When no notice exists, leave summary/count alone (may be meta-imported);
	keep-tail index is cleared because it is always re-derived for the wire.
	"""
	raw = list(messages if messages is not None else getattr(E, "messages", None) or [])
	_notice_idx, notice = FindLastCompactNotice(raw)
	if notice is None:
		E.compact_until = 0
		return
	E.compact_summary = str(notice.get("content") or "")
	count = int(notice.get("compact_count") or 0)
	if count <= 0:
		count = int(getattr(E, "compact_count", 0) or 0) or 1
	E.compact_count = count
	# Cached derived index for export/debug only — keep-tail always re-derives.
	E.compact_until = DeriveKeepStart(raw)


def GetCompactState():
	"""Snapshot for export/UI. Keep-tail is derived from the last compact notice."""
	SyncCompactFromMessages()
	return {
		"summary": str(getattr(E, "compact_summary", "") or ""),
		"summarized_until": int(getattr(E, "compact_until", 0) or 0),
		"compact_count": int(getattr(E, "compact_count", 0) or 0),
	}


def ApplyCompactState(state):
	"""Restore compact cache after import. Prefers transcript notices over meta."""
	SyncCompactFromMessages()
	# Legacy/orphan meta: use only when the transcript has no compact notice yet.
	if FindLastCompactNotice()[1] is not None:
		return
	if not isinstance(state, dict):
		ResetCompactState()
		return
	E.compact_summary = str(state.get("summary") or "")
	E.compact_count = int(state.get("compact_count") or 0)
	E.compact_until = 0


def OnCompactEnabledChanged(enabled):
	"""When Context Token Budget is turned off, drop stale compact cursor/summary."""
	if not enabled:
		ResetCompactState()


def _UserPromptPath():
	home = str(getattr(E, "aiui_home", "") or "").strip()
	if not home:
		home = os.path.join(os.path.expanduser("~"), ".aiui")
	return os.path.join(home, _USER_PROMPT_NAME)


def LoadCompactPrompt():
	"""Optional ~/.aiui/compact_prompt.txt overrides the built-in default."""
	user = _UserPromptPath()
	try:
		if os.path.isfile(user):
			with open(user, "r", encoding="utf-8") as f:
				text = f.read().strip()
			if text:
				return text
	except Exception as e:
		ipc.LogErr(f"WARN: could not read compact prompt {user}: {e}")
	return _DEFAULT_PROMPT.strip()


def IsCompactNotice(msg):
	if not isinstance(msg, dict):
		return False
	if msg.get("_aiui_compact"):
		return True
	return str(msg.get("role") or "").lower() == COMPACT_ROLE


def _UserRole():
	return str(getattr(E, "user_name", "user") or "user")


def _BotRole():
	return str(getattr(E, "bot_name", "assistant") or "assistant")


def _IsUserMsg(msg):
	role = str(msg.get("role") or "").lower()
	user = _UserRole().lower()
	return role in ("user", user)


def _IsSystemMsg(msg):
	return str(msg.get("role") or "").lower() == "system" and not IsCompactNotice(msg)


def ContentToText(content):
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	if not isinstance(content, (list, tuple)):
		return str(content)
	parts = []
	for item in content:
		if isinstance(item, dict):
			ptype = str(item.get("type") or "").lower()
			if ptype in ("text", "input_text") or "text" in item:
				parts.append(str(item.get("text") or ""))
			elif ptype in ("image", "image_url", "input_image"):
				src = item.get("image") or item.get("url")
				if isinstance(item.get("image_url"), dict):
					src = item["image_url"].get("url") or src
				elif isinstance(item.get("image_url"), str):
					src = item.get("image_url")
				parts.append(f"[image: {os.path.basename(str(src or 'attachment'))}]")
			elif ptype in ("video", "video_url"):
				src = item.get("video") or item.get("url")
				if isinstance(item.get("video_url"), dict):
					src = item["video_url"].get("url") or src
				parts.append(f"[video: {os.path.basename(str(src or 'attachment'))}]")
			else:
				for k, v in item.items():
					if k != "type" and isinstance(v, str):
						parts.append(v)
						break
		else:
			parts.append(str(item))
	return "\n".join(p for p in parts if p)


def StubMediaContent(content):
	"""Replace image/video parts with short text stubs (model wire / counting)."""
	if not isinstance(content, (list, tuple)):
		return content
	out = []
	for item in content:
		if not isinstance(item, dict):
			out.append(item)
			continue
		ptype = str(item.get("type") or "").lower()
		if ptype in ("image", "image_url", "input_image"):
			src = item.get("image") or item.get("url")
			iu = item.get("image_url")
			if isinstance(iu, dict):
				src = iu.get("url") or src
			elif isinstance(iu, str):
				src = iu
			name = os.path.basename(str(src or "attachment"))
			if str(src or "").startswith("data:"):
				name = "attachment"
			out.append({"type": "text", "text": f"[omitted image: {name}]"})
		elif ptype in ("video", "video_url"):
			src = item.get("video") or item.get("url")
			vu = item.get("video_url")
			if isinstance(vu, dict):
				src = vu.get("url") or src
			name = os.path.basename(str(src or "attachment"))
			out.append({"type": "text", "text": f"[omitted video: {name}]"})
		else:
			out.append(item)
	return out


def HeuristicTokensFromText(text):
	n = len(str(text or ""))
	return max(0, (n + 3) // 4)


def HeuristicTokensFromMessages(messages):
	total = 0
	for m in messages or []:
		if not isinstance(m, dict):
			continue
		total += 4
		total += HeuristicTokensFromText(ContentToText(m.get("content")))
	return total


def FindKeepTailStart(messages):
	"""Index of last user message (preferred keep-tail start)."""
	msgs = messages or []
	last_user = -1
	for i, m in enumerate(msgs):
		if isinstance(m, dict) and not IsCompactNotice(m) and _IsUserMsg(m):
			last_user = i
	if last_user < 0:
		for i in range(len(msgs) - 1, -1, -1):
			if isinstance(msgs[i], dict) and not _IsSystemMsg(msgs[i]) and not IsCompactNotice(msgs[i]):
				return i
		return len(msgs)
	return last_user


def _LastUserIndex(messages):
	idx = -1
	for i, m in enumerate(messages or []):
		if isinstance(m, dict) and not IsCompactNotice(m) and _IsUserMsg(m):
			idx = i
	return idx


def _ToolXmlKind(msg):
	"""'call', 'resp', or 'other' from stored assistant XML (not collapsed both)."""
	if not isinstance(msg, dict):
		return "other"
	text = ContentToText(msg.get("content"))
	has_call = "<tool_call>" in text
	has_resp = "<tool_response>" in text
	if has_call and not has_resp:
		return "call"
	if has_resp and not has_call:
		return "resp"
	return "other"


def _TruncateMsgSuffix(msg, char_budget):
	"""Keep the end of a message's text. Preserve tool XML wrappers when present."""
	if not isinstance(msg, dict):
		return msg
	text = ContentToText(msg.get("content"))
	if len(text) <= char_budget:
		return msg
	out = dict(msg)
	kind = _ToolXmlKind(msg)
	wrap = {"call": ("<tool_call>", "</tool_call>"), "resp": ("<tool_response>", "</tool_response>")}.get(kind)
	if wrap:
		open_t, close_t = wrap
		start = text.find(open_t)
		end = text.rfind(close_t)
		if start >= 0 and end > start:
			head = text[:start]
			body = text[start + len(open_t):end]
			tail = text[end + len(close_t):]
			overhead = len(head) + len(open_t) + len(close_t) + len(tail)
			keep = max(0, char_budget - overhead)
			body = body[-keep:] if keep else ""
			out["content"] = head + open_t + body + close_t + tail
			if len(out["content"]) <= char_budget:
				return out
	out["content"] = text[-char_budget:]
	return out


def _DropOldestTailGroup(out, last_user):
	"""Drop the oldest droppable message; tool_call takes following tool_response(s)."""
	if last_user == 0:
		drop_at = 1
	else:
		drop_at = 0
	if drop_at >= len(out) or drop_at == last_user:
		return out, last_user
	end = drop_at + 1
	if _ToolXmlKind(out[drop_at]) == "call":
		while end < len(out) and end != last_user and _ToolXmlKind(out[end]) == "resp":
			end += 1
	n = end - drop_at
	out = out[:drop_at] + out[end:]
	if last_user >= end:
		last_user -= n
	return out, last_user


def _CapTailByTokens(messages, max_tokens):
	"""Trim from the front of the tail until under max_tokens.

	Never drops the last user message (or truncates its text). Oversized non-user
	messages are shortened in place (keep the end) before any message is deleted.
	When a message must go, a `<tool_call>` is dropped with its following
	`<tool_response>`s so remappers never see an orphan result.
	"""
	max_tokens = int(max_tokens or 0)
	if max_tokens <= 0 or not messages:
		return list(messages)
	out = list(messages)
	last_user = _LastUserIndex(out)

	def over_budget():
		return HeuristicTokensFromMessages(out) > max_tokens

	def _fit_char_budget(idx):
		others = HeuristicTokensFromMessages([m for j, m in enumerate(out) if j != idx])
		remain = max_tokens - others
		if remain <= 4:
			return 0
		return max(64, (remain - 4) * 4)

	guard = 0
	while len(out) > 1 and over_budget() and guard < 10000:
		guard += 1
		start = 1 if last_user == 0 else 0
		trimmed = False
		for i in range(start, len(out)):
			if i == last_user:
				continue
			char_budget = _fit_char_budget(i)
			if char_budget <= 0:
				continue
			text = ContentToText(out[i].get("content") if isinstance(out[i], dict) else "")
			if len(text) > char_budget:
				out[i] = _TruncateMsgSuffix(out[i], char_budget)
				new = ContentToText(out[i].get("content") if isinstance(out[i], dict) else "")
				if len(new) < len(text):
					trimmed = True
					break
		if trimmed:
			continue
		prev_n = len(out)
		out, last_user = _DropOldestTailGroup(out, last_user)
		if len(out) >= prev_n:
			break
	if out and over_budget() and last_user != 0 and last_user != len(out) - 1:
		idx = len(out) - 1
		char_budget = _fit_char_budget(idx)
		if char_budget > 0:
			out[idx] = _TruncateMsgSuffix(out[idx], char_budget)
	return out


def _SystemMessages(messages):
	out = []
	for m in messages or []:
		if isinstance(m, dict) and _IsSystemMsg(m) and not IsCompactNotice(m):
			out.append(m)
	return out


def _WireSourceParts(raw):
	"""System + keep-tail originals currently sent (no summary, no copies)."""
	SyncCompactFromMessages(raw)
	until = DeriveKeepStart(raw)
	summary = str(getattr(E, "compact_summary", "") or "")
	tail_budget = int(getattr(E, "compact_tail_tokens", 8000) or 8000)
	system = _SystemMessages(raw[:until] if until else raw[:1])
	if not system:
		system = _SystemMessages(raw)
	tail = []
	for m in raw[until:]:
		if not isinstance(m, dict) or IsCompactNotice(m):
			continue
		if _IsSystemMsg(m) and system:
			continue
		tail.append(m)
	if summary:
		capped = _CapTailByTokens(tail, tail_budget)
	else:
		capped = list(tail)
	return system, summary, capped


def HiddenHistoryMessages():
	"""Stored conversation messages that are not on the current model wire.

	Chronological (oldest first). Skips compact notices and the live system/tail.
	Empty when compaction is off (those turns were pruned) or nothing has been
	folded/capped yet. -1 in tool indices is the last entry of this list.
	"""
	raw = list(getattr(E, "messages", None) or [])
	if getattr(E, "_compact_bypass", False) or not CompactEnabled():
		return []
	system, _summary, capped = _WireSourceParts(raw)
	visible = {id(m) for m in system}
	visible.update(id(m) for m in capped)
	hidden = []
	for m in raw:
		if not isinstance(m, dict) or IsCompactNotice(m):
			continue
		if id(m) in visible:
			continue
		hidden.append(m)
	return hidden


def HiddenHistoryIndexRange():
	n = len(HiddenHistoryMessages())
	if n <= 0:
		return {"hidden": 0, "index_range": None}
	return {"hidden": n, "index_range": {"start": -n, "end": -1}}


def BuildModelMessages(stub_old_media=True, sanitize=True):
	"""Model-facing transcript: system + summary + keep-tail (no compact notices).

	Keep-tail start is derived from the last compact notice in the transcript.
	The recent-tail token cap applies only after a summary exists; until then
	the full conversation is sent (Context Token Budget triggers compact).
	When sanitize=False, keeps internal markers (e.g. _aiui_summary) for usage breakdown.
	"""
	raw = list(getattr(E, "messages", None) or [])
	if getattr(E, "_compact_bypass", False):
		out = [m for m in raw if isinstance(m, dict)]
		return _FinishModelMessages(out, sanitize)

	if not CompactEnabled():
		out = [m for m in raw if isinstance(m, dict) and not IsCompactNotice(m)]
		return _FinishModelMessages(out, sanitize)

	system, summary, capped = _WireSourceParts(raw)
	live_keep = FindKeepTailStart(raw)
	out = []
	for m in system:
		out.append(copy.deepcopy(m))
	if summary:
		out.append({
			"role": "system",
			"content": (
				f"[Conversation summary after compaction "
				f"#{int(getattr(E, 'compact_count', 0) or 0)}]\n{summary}"
			),
			"_aiui_summary": True,
		})
	# Identity map — list.index uses == and can hit an earlier duplicate message.
	raw_id_index = {id(m): i for i, m in enumerate(raw)}
	for m in capped:
		msg = copy.deepcopy(m)
		idx = raw_id_index.get(id(m))
		if stub_old_media and idx is not None and idx < live_keep:
			msg["content"] = StubMediaContent(msg.get("content"))
		out.append(msg)
	return _FinishModelMessages(out, sanitize)


def _FinishModelMessages(messages, sanitize):
	from lib import git_context
	out = git_context.AttachGitContext(messages)
	return _SanitizeModelMessages(out) if sanitize else out


def _SanitizeModelMessages(messages):
	"""Drop internal keys so HF apply_chat_template / remote mappers only see wire fields."""
	keep = {
		"role", "content", "name", "tool_calls", "tool_call_id",
		"_anthropic_content", "_reasoning_details", "_reasoning_pin",
	}
	out = []
	for m in messages or []:
		if not isinstance(m, dict):
			continue
		item = {k: v for k, v in m.items() if k in keep}
		out.append(item)
	return out


def _ActiveToolSchemas():
	"""Tool schemas currently sent to the model (None when tool use is off / empty)."""
	if not getattr(E, "enable_tooluse", 0):
		return None
	funcs = getattr(E, "tool_funcs", None)
	if not funcs:
		return None
	if getattr(E, "use_aiui_template", False) and int(getattr(E, "model_type", 0) or 0) != 2:
		return [t.get("function", t) if isinstance(t, dict) else t for t in funcs]
	return list(funcs)


def _ToolsJsonBlob(schemas=None):
	import json

	schemas = schemas if schemas is not None else _ActiveToolSchemas()
	if not schemas:
		return ""
	try:
		return json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
	except Exception:
		return str(schemas)


def CountTextTokens(text):
	"""Tokenize a plain string (local tokenizer when available, else heuristic).

	Remote / lite installs never require a local tokenizer - only the local
	text backend uses E.tokenizer for counting.
	"""
	s = str(text or "")
	from lib import text_backend as text_backend_mod
	if not text_backend_mod.IsRemoteText():
		tok = getattr(E, "tokenizer", None)
		if tok is not None:
			# Processors (Qwen2.5-VL, etc.) expose .tokenizer; they have no .encode.
			inner = getattr(tok, "tokenizer", None)
			enc = inner if inner is not None else tok
			try:
				return len(enc.encode(s, add_special_tokens=False))
			except Exception:
				pass
	return HeuristicTokensFromText(s)


def CountToolsTokens(schemas=None):
	blob = _ToolsJsonBlob(schemas)
	if not blob:
		return 0
	return CountTextTokens(blob)


def CountMessagesContentTokens(messages):
	"""Approximate tokens for a message list (content + small per-message overhead)."""
	total = 0
	for m in messages or []:
		if not isinstance(m, dict):
			continue
		total += 4
		total += CountTextTokens(ContentToText(m.get("content")))
	return total


def _PartitionWireMessages(wire):
	"""Split unsanitized model wire into system / summary / conversation lists."""
	system, summary, conversation = [], [], []
	for m in wire or []:
		if not isinstance(m, dict):
			continue
		if m.get("_aiui_summary"):
			summary.append(m)
		elif _IsSystemMsg(m):
			system.append(m)
		else:
			conversation.append(m)
	return system, summary, conversation


def CountPromptTokens(messages=None):
	"""Return estimated/exact prompt tokens for the model-facing list (includes tools)."""
	from lib import text_backend as text_backend_mod

	msgs = messages if messages is not None else BuildModelMessages()
	# Always count against sanitized wire for provider/template APIs.
	if messages is not None and any(isinstance(m, dict) and ("_aiui_summary" in m) for m in messages):
		msgs = _SanitizeModelMessages(msgs)
	tools = _ActiveToolSchemas()

	if text_backend_mod.IsRemoteText():
		n = text_backend_mod.CountRemotePromptTokens(msgs, tools=tools)
		if n is not None and n > 0:
			return int(n)
		return HeuristicTokensFromMessages(msgs) + CountToolsTokens(tools)

	tok = getattr(E, "tokenizer", None)
	if tok is None:
		return HeuristicTokensFromMessages(msgs) + CountToolsTokens(tools)
	inner = getattr(tok, "tokenizer", None)
	enc = inner if inner is not None else tok

	def _Heuristic():
		return HeuristicTokensFromMessages(msgs) + CountToolsTokens(tools)

	try:
		if hasattr(tok, "apply_chat_template") and bool(getattr(E, "use_chat_template", False)):
			wire = []
			has_user = False
			for m in msgs:
				role = str(m.get("role") or "user")
				if role == _BotRole() or role == "assistant":
					role = "assistant"
				elif role == _UserRole() or role == "user":
					role = "user"
				elif role != "system":
					role = "user"
				text = ContentToText(m.get("content"))
				if role == "user" and str(text or "").strip():
					has_user = True
				wire.append({"role": role, "content": text})
			# Qwen3.5+ templates raise "No user query found in messages" on
			# system-only wires (new chat / usage ring before the first turn).
			if not has_user:
				return _Heuristic()
			kw = {}
			if tools:
				kw["tools"] = tools
			text = tok.apply_chat_template(
				wire, tokenize=False, add_generation_prompt=True, **kw
			)
			ids = enc.encode(text, add_special_tokens=False)
			return len(ids)
		blob = "\n".join(ContentToText(m.get("content")) for m in msgs)
		ids = enc.encode(blob, add_special_tokens=False)
		return len(ids) + CountToolsTokens(tools)
	except Exception as e:
		err = str(e)
		# Expected for some chat templates when the wire has no plain user turn.
		if "No user query found" not in err:
			ipc.LogErr(f"WARN: local token count failed ({e}); using heuristic")
		return _Heuristic()


def ShouldCompact(prompt_tokens=None):
	if not CompactEnabled():
		return False
	at = int(getattr(E, "compact_at", 0) or 0)
	if at <= 0:
		return False
	n = int(prompt_tokens if prompt_tokens is not None else CountPromptTokens())
	gen = int(getattr(E, "gen_max", 32000) or 32000)
	return n >= at or (n + gen) > at


def EmitContextUsage(prompt_tokens=None, heuristic=False, compacting=None):
	"""Push used/limit (+ local breakdown) to the UI context-usage ring.

	When Context Token Budget is 0, emits limit=0 so the ring hides.
	Remote mode omits the system/conversation/tools split (totals only).
	Pass compacting=True/False to toggle the UI compaction pulse (may be sent
	alone without recounting when used/limit are omitted via EmitCompacting).
	"""
	from lib import text_backend as text_backend_mod

	at = int(getattr(E, "compact_at", 0) or 0)
	gen = int(getattr(E, "gen_max", 32000) or 32000)
	remote = bool(text_backend_mod.IsRemoteText())
	if at <= 0:
		payload = {
			"used": 0, "limit": 0, "gen_max": gen, "remote": remote,
		}
		if compacting is not None:
			payload["compacting"] = bool(compacting)
		ipc.SendEvent("context_usage", payload)
		return 0

	# Local mode also reports a breakdown, so build the wire once and reuse it.
	wire = None if remote else BuildModelMessages(sanitize=False)

	if prompt_tokens is not None:
		n = int(prompt_tokens)
	elif heuristic:
		msgs = wire if wire is not None else BuildModelMessages()
		n = HeuristicTokensFromMessages(msgs) + CountToolsTokens()
	else:
		n = CountPromptTokens(_SanitizeModelMessages(wire) if wire is not None else None)
	n = max(0, int(n))

	payload = {
		"used": n,
		"limit": at,
		"gen_max": gen,
		"remote": remote,
	}
	if wire is not None:
		sys_m, sum_m, conv_m = _PartitionWireMessages(wire)
		sys_n = CountMessagesContentTokens(sys_m)
		sum_n = CountMessagesContentTokens(sum_m)
		tools_n = CountToolsTokens()
		payload["system"] = sys_n
		payload["summary"] = sum_n
		payload["tools"] = tools_n
		# Conversation takes the remainder so the widget's segments add up to the
		# ring: `used` is the exact template count, which also carries per-message
		# scaffolding that the content-only counters cannot see.
		measured = CountMessagesContentTokens(conv_m)
		payload["conversation"] = max(measured, n - sys_n - sum_n - tools_n)
		payload["free"] = max(0, at - n)
	if compacting is not None:
		payload["compacting"] = bool(compacting)

	ipc.SendEvent("context_usage", payload)
	return n


def _UsageGet(obj, *names):
	if obj is None:
		return None
	getter = obj.get if isinstance(obj, dict) else lambda k, d=None: getattr(obj, k, d)
	for name in names:
		val = getter(name)
		if val is None:
			continue
		try:
			return int(val)
		except (TypeError, ValueError):
			continue
	return None


def RecordRemoteUsage(usage):
	"""Store provider response usage on E for mid/end-of-turn ring updates.

	Accepts OpenAI (prompt_tokens/completion_tokens) or Anthropic
	(input_tokens/output_tokens) shapes. Marks usage as fresh (valid until
	tool results grow the transcript past this response).
	"""
	if usage is None:
		return False
	prompt = _UsageGet(usage, "prompt_tokens", "input_tokens")
	completion = _UsageGet(usage, "completion_tokens", "output_tokens")
	if prompt is None and completion is None:
		return False
	E.last_usage_prompt = 0 if prompt is None else max(0, prompt)
	E.last_usage_completion = 0 if completion is None else max(0, completion)
	E.last_usage_stale = False
	return True


def EmitContextUsageAfterGeneration():
	"""After a GenReply: prefer prompt+completion from response metadata."""
	if not getattr(E, "last_usage_stale", True):
		p = getattr(E, "last_usage_prompt", None)
		if p is not None:
			c = int(getattr(E, "last_usage_completion", 0) or 0)
			return EmitContextUsage(prompt_tokens=int(p) + c)
	return EmitContextUsage(heuristic=True)


def EmitContextUsageAfterTools():
	"""After tool results: transcript grew past last response usage."""
	E.last_usage_stale = True
	return EmitContextUsage(heuristic=True)


def EmitContextUsageTurnEnd():
	"""End of turn: fresh response metadata, else local exact, else heuristic.

	Does not call the remote token-count API (idle/end must stay free).
	"""
	from lib import text_backend as text_backend_mod

	if not getattr(E, "last_usage_stale", True):
		p = getattr(E, "last_usage_prompt", None)
		if p is not None:
			c = int(getattr(E, "last_usage_completion", 0) or 0)
			return EmitContextUsage(prompt_tokens=int(p) + c)
	if not text_backend_mod.IsRemoteText():
		return EmitContextUsage()
	return EmitContextUsage(heuristic=True)


def EmitCompacting(active):
	"""Toggle the UI compaction indicator without recounting tokens."""
	ipc.SendEvent("context_usage", {"compacting": bool(active)})


def _RegionForSummary(messages, keep_start):
	"""Messages to fold into the summary (exclude system, compact notices, and keep-tail)."""
	region = []
	for i, m in enumerate(messages or []):
		if not isinstance(m, dict):
			continue
		if IsCompactNotice(m) or _IsSystemMsg(m):
			continue
		if i >= keep_start:
			break
		region.append(m)
	return region


def _FormatRegionText(region, prior_summary):
	lines = []
	if prior_summary:
		lines.append("### Prior compaction summary")
		lines.append(prior_summary.strip())
		lines.append("")
	lines.append("### Conversation to summarize")
	for m in region:
		role = str(m.get("role") or "user")
		lines.append(f"{role}: {ContentToText(m.get('content'))}")
	return "\n".join(lines).strip()


def _CleanSummaryText(text):
	"""Strip think wrappers / roleplay fluff so the stored handoff is plain text."""
	import re

	s = str(text or "").strip()
	prev = None
	while prev != s:
		prev = s
		s = re.sub(r"<(think|thoughts)>\s*</\1>", "", s, flags=re.IGNORECASE)
		s = re.sub(r"AIUI_THINK_START\s*AIUI_THINK_END", "", s)
	s = re.sub(r"<(think|thoughts)\b[^>]*>.*?</\1>", "", s, flags=re.IGNORECASE | re.DOTALL)
	s = re.sub(r"AIUI_THINK_START.*?AIUI_THINK_END", "", s, flags=re.DOTALL)
	s = s.strip()
	s = re.sub(r"^<think>\s*</think>\s*", "", s, flags=re.IGNORECASE)
	return s.strip()


_META_SUMMARY_MARKERS = (
	"okay, let's tackle",
	"let's tackle this",
	"the user wants a compaction summary",
	"the compaction summary needs",
	"i need to understand the key points",
	"first, i need to understand",
	"write the compaction summary",
	"output only the summary text",
)


def _LooksLikeMetaSummary(text):
	"""True when the model narrated writing a summary instead of producing one."""
	s = str(text or "").strip()
	if not s:
		return True
	low = s.lower()
	head = low[:240]
	for marker in _META_SUMMARY_MARKERS:
		if marker in head or marker in low[:500]:
			return True
	# Cut off mid-planning sentence (common when the model stalls).
	if ("i should mention" in low or "needs to condense" in low) and not low.rstrip().endswith((".", "!", "?", '"', "'")):
		return True
	return False


def _SummarizeSideChannel(region_text, instruction, max_tokens, *, strict=False):
	"""Run a no-tools, no-thinking completion; returns summary text.

	max_tokens comes from Compact Summary Max (E.compact_summary_max), not Max Response Length.
	"""
	from lib import text_backend as text_backend_mod
	from lib import chat as chat_mod

	max_tokens = max(256, int(max_tokens or 8192))
	user_blob = region_text
	if len(user_blob) > 500_000:
		user_blob = user_blob[:500_000] + "\n\n[truncated for summarization]"

	strict_note = ""
	if strict:
		strict_note = (
			"\n\nCRITICAL: Your previous attempt was invalid meta-commentary. "
			"Start immediately with the handoff facts. No 'Okay', no planning, "
			"no description of what the summary should contain."
		)

	# Clear instruction + transcript in one user turn (models often ignore system for this).
	user_content = (
		str(instruction or "").strip()
		+ "\n\n---\n\n"
		+ user_blob
		+ "\n\n---\n\n"
		+ "Write the compaction summary now. Output only the summary text."
		+ strict_note
	)

	E._compact_bypass = True
	saved_messages = E.messages
	saved_prompt = getattr(E, "prompt", "")
	saved_enable = getattr(E, "enable_tooluse", False)
	saved_gen_max = getattr(E, "gen_max", 32000)
	saved_gen_min = getattr(E, "gen_min", 1)
	saved_thinking = getattr(E, "enable_thinking", "auto")
	saved_effort = getattr(E, "reasoning_effort", "auto")
	try:
		E.enable_tooluse = False
		E.gen_max = max_tokens
		E.gen_min = 1
		# Force a plain answer - thinking models often echo the last turn instead.
		E.enable_thinking = "false"
		E.reasoning_effort = "auto"
		sys_msg = {
			"role": "system",
			"content": (
				"You compress conversation context for a later turn of the same assistant. "
				"Reply with only the handoff summary — factual bullet points or short "
				"paragraphs. No greeting, no tool calls, no <think> blocks, no narration "
				"about how you will write the summary."
			),
		}
		E.messages = [
			sys_msg,
			{"role": _UserRole(), "content": user_content},
		]
		# Non-template local path reads E.prompt - keep it in sync with the side-channel.
		sep = str(getattr(E, "roles_append_str", ": ") or ": ")
		E.prompt = (
			"system" + sep + sys_msg["content"] + "\n"
			+ _UserRole() + sep + user_content
		)

		if text_backend_mod.IsRemoteText():
			raw = text_backend_mod.RemoteChatReply(
				temperature=0.2,
				stream=False,
				ui_stream=False,
				on_block=None,
				reuse_stream=False,
			) or ""
		else:
			if getattr(E, "model", None) is None:
				chat_mod.LoadModel()
			it = chat_mod.TokenizeChat()
			raw = chat_mod.GenReply(
				it, 0.2, stream=False, reuse_stream=False, ui_stream=False, on_block=None,
			) or ""
		cleaned = _CleanSummaryText(raw)
		ipc.LogErr(
			f"Context compact summarize: raw_len={len(raw)} clean_len={len(cleaned)} "
			f"max_tokens={max_tokens} strict={strict} preview={cleaned[:120]!r}"
		)
		return cleaned
	finally:
		E.messages = saved_messages
		E.prompt = saved_prompt
		E.enable_tooluse = saved_enable
		E.gen_max = saved_gen_max
		E.gen_min = saved_gen_min
		E.enable_thinking = saved_thinking
		E.reasoning_effort = saved_effort
		E._compact_bypass = False


def RunCompactIfNeeded():
	"""If over budget, summarize older turns and update compact state. Returns True if compacted."""
	if not CompactEnabled():
		EmitContextUsage()
		return False
	if getattr(E, "_compact_bypass", False):
		return False

	wire = BuildModelMessages()
	n = CountPromptTokens(wire)
	EmitContextUsage(n)
	if not ShouldCompact(n):
		return False
	return _DoCompact(n)


def RunCompactForced():
	"""Manual compact: same path as auto, but skips the budget gate.

	Returns (ok, error_message). ok False when compact is off or nothing to fold.
	"""
	if not CompactEnabled():
		EmitContextUsage()
		return False, "Context compaction is off (Context Token Budget is 0)."
	if getattr(E, "_compact_bypass", False):
		return False, "Compaction is already running."
	wire = BuildModelMessages()
	n = CountPromptTokens(wire)
	EmitContextUsage(n)
	ok = _DoCompact(n)
	if not ok:
		return False, "Nothing to compact (no older turns to summarize)."
	return True, ""


def _DoCompact(prompt_tokens):
	"""Summarize older turns into compact state. Caller must gate budget / force."""
	raw = list(E.messages or [])
	keep_start = FindKeepTailStart(raw)
	region = _RegionForSummary(raw, keep_start)
	prior = str(getattr(E, "compact_summary", "") or "")

	# Edge: only the keep-tail exists (e.g. one huge user turn). Fold it into the
	# summary region and advance past it so the wire can shrink to summary (+ tiny cap).
	if not region and not prior:
		if keep_start < len(raw):
			region = _RegionForSummary(raw, keep_start + 1)
			keep_start = keep_start + 1
		if not region:
			ipc.LogErr("WARN: compact triggered but nothing to summarize")
			return False

	instruction = LoadCompactPrompt()
	region_text = _FormatRegionText(region, prior)
	summary_max = int(getattr(E, "compact_summary_max", 8192) or 8192)
	ipc.LogErr(
		f"Context compact: prompt_tokens~{prompt_tokens} compact_at={getattr(E, 'compact_at', 0)} "
		f"region_msgs={len(region)} keep_start={keep_start}"
	)
	EmitCompacting(True)
	try:
		summary = (_SummarizeSideChannel(region_text, instruction, summary_max) or "").strip()
		if _LooksLikeMetaSummary(summary):
			ipc.LogErr("WARN: compact summary looked like meta-reasoning; retrying once")
			summary = (_SummarizeSideChannel(
				region_text, instruction, summary_max, strict=True,
			) or "").strip()
		if not summary or _LooksLikeMetaSummary(summary):
			ipc.LogErr("WARN: compaction produced unusable summary; leaving context unchanged")
			return False

		E.compact_summary = summary
		E.compact_until = keep_start
		E.compact_count = int(getattr(E, "compact_count", 0) or 0) + 1
		from lib.chat import StampMsg
		notice = StampMsg({
			"role": COMPACT_ROLE,
			"content": summary,
			"_aiui_compact": True,
			"compact_count": E.compact_count,
			# Distance back to the keep-tail start; survives later appends and
			# front-truncation, unlike an absolute index.
			"keep_back": max(0, len(E.messages) - keep_start),
		})
		E.messages.append(notice)
		ipc.SendEvent("compact", {
			"summary": summary,
			"compact_count": E.compact_count,
			"summarized_until": E.compact_until,
			"ok": True,
		})
		EmitContextUsage()
		return True
	finally:
		EmitCompacting(False)
