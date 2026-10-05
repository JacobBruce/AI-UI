"""Compact wire shape, Anthropic wire/usage, OpenRouter preface glue, reasoning pin."""

import os
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..")
if _ENGINE not in sys.path:
	sys.path.insert(0, _ENGINE)

from lib import context_compact as cc
from lib import remote_tools as rt


def test_map_openai_strips_think_keeps_sidecar():
	enc = [{"type": "reasoning.encrypted", "data": "blob"}]
	msgs = [
		{"role": "user", "content": "hi"},
		{
			"role": "assistant",
			"content": (
				"<think>\nplan the glob\n</think>\n\n"
				"I'll look it up.\n\n"
				'<tool_call>{"name":"glob_files","arguments":{}}</tool_call>'
			),
			"_reasoning_details": enc,
			"_reasoning_pin": {"model": "x-ai/grok-4", "provider": "x-ai"},
		},
	]
	out = rt.MapMessagesOpenAI(msgs, replay_reasoning=True)
	asst = [m for m in out if m.get("role") == "assistant"]
	assert len(asst) == 1
	assert "<think>" not in (asst[0].get("content") or "")
	assert "I'll look it up." in (asst[0].get("content") or "")
	assert asst[0].get("tool_calls")
	assert asst[0].get("reasoning_details") == enc


def test_map_openai_think_only_sidecar_clears_content():
	enc = [{"type": "reasoning.summary", "summary": "the review"}]
	out = rt.MapMessagesOpenAI(
		[{
			"role": "assistant",
			"content": "<think>\nthe review\n</think>",
			"_reasoning_details": enc,
		}],
		replay_reasoning=True,
	)
	assert len(out) == 1
	assert out[0]["content"] is None
	assert out[0]["reasoning_details"] == enc


def test_map_openai_think_only_without_sidecar_uses_body():
	out = rt.MapMessagesOpenAI(
		[{"role": "assistant", "content": "<think>\nthe review\n</think>"}],
		replay_reasoning=True,
	)
	assert len(out) == 1
	assert out[0]["content"] == "the review"
	assert "<think>" not in out[0]["content"]


def test_close_or_unwrap_open_think():
	from lib.files import CloseOrUnwrapOpenThink

	assert CloseOrUnwrapOpenThink("<think>\nLet me look at the bugs") == "Let me look at the bugs"
	sealed = CloseOrUnwrapOpenThink("<think>\nplan\n<tool_call>{}</tool_call>")
	assert sealed.startswith("<think>\nplan")
	assert "</think>" in sealed
	assert sealed.index("</think>") < sealed.index("<tool_call>")
	assert CloseOrUnwrapOpenThink("hello") == "hello"


def test_map_openai_glues_preface_to_tool_calls():
	enc = [{"type": "reasoning.encrypted", "data": "blob"}]
	msgs = [
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "thinking out loud"},
		{
			"role": "assistant",
			"content": '<tool_call>{"name":"glob_files","arguments":{}}</tool_call>',
			"_reasoning_details": enc,
			"_reasoning_pin": {"model": "x-ai/grok-4", "provider": "x-ai"},
		},
	]
	out = rt.MapMessagesOpenAI(msgs, replay_reasoning=True)
	asst = [m for m in out if m.get("role") == "assistant"]
	assert len(asst) == 1
	assert asst[0].get("tool_calls")
	assert "thinking out loud" in (asst[0].get("content") or "")
	assert asst[0].get("reasoning_details") == enc
	assert asst[0].get("_reasoning_pin", {}).get("model") == "x-ai/grok-4"


def test_pin_uses_wire_not_hidden_history():
	wire = [
		{"role": "user", "content": "x"},
		{"role": "assistant", "content": "ok"},
	]
	out = rt.ApplyReasoningRoutePin({"model": "openrouter/auto", "messages": wire})
	assert out.get("model") == "openrouter/auto"
	assert not (out.get("extra_body") or {}).get("provider")


def _xai_encrypted():
	return [{
		"type": "reasoning.encrypted",
		"data": "blob",
		"format": "xai-responses-v1",
	}]


def test_pin_xai_encrypted_does_not_imply_zdr():
	pin = rt.ReasoningPinFromDetails(
		_xai_encrypted(), model="x-ai/grok-4.6", provider="xai",
	)
	assert pin == {"model": "x-ai/grok-4.6", "provider": "xai"}
	assert "zdr" not in pin


def test_pin_zdr_only_when_provider_slug_says_so():
	pin = rt.ReasoningPinFromDetails(
		_xai_encrypted(), model="x-ai/grok-4.6", provider="xai/zdr",
	)
	assert pin["zdr"] is True
	assert pin["provider"] == "xai/zdr"


def test_apply_pin_without_zdr_does_not_request_zdr():
	kw = {
		"model": "openrouter/auto",
		"messages": [{
			"role": "assistant",
			"reasoning_details": [{"type": "reasoning.encrypted", "data": "b"}],
			"_reasoning_pin": {"model": "x-ai/grok-4.6", "provider": "xai"},
		}],
	}
	out = rt.ApplyReasoningRoutePin(kw)
	assert out["model"] == "x-ai/grok-4.6"
	prov = out["extra_body"]["provider"]
	assert prov["order"] == ["xai"]
	assert "zdr" not in prov


def test_pin_from_mapped_sidecar():
	kw = {
		"model": "openrouter/auto",
		"messages": [
			{"role": "user", "content": "x"},
			{
				"role": "assistant",
				"content": None,
				"tool_calls": [{
					"id": "c1",
					"type": "function",
					"function": {"name": "g", "arguments": "{}"},
				}],
				"reasoning_details": [{"type": "reasoning.encrypted", "data": "b"}],
				"_reasoning_pin": {"model": "x-ai/grok-4", "provider": "x-ai"},
			},
		],
	}
	out = rt.ApplyReasoningRoutePin(kw)
	assert out["model"] == "x-ai/grok-4"
	assert out["extra_body"]["provider"]["allow_fallbacks"] is False


def test_reasoning_details_from_model_dump():
	class Msg:
		def model_dump(self, exclude_unset=False):
			return {"reasoning_details": [{"type": "reasoning.text", "text": "hi"}]}

	got = rt.ReasoningDetailsFromMessage(Msg())
	assert got and got[0].get("text") == "hi"


class _CompactE:
	compact_at = 8000
	compact_summary = ""
	compact_until = 0
	compact_count = 0
	compact_summary_max = 8192
	messages = None
	_compact_bypass = False
	_compact_summarize = False
	user_name = "user"
	bot_name = "assistant"
	git_status_context = 0
	enable_tooluse = 0
	tool_funcs = None
	aiui_home = ""
	reasoning_effort = "auto"
	text_backend = "local"
	remote_tool_format = "openai"


def _bind_compact():
	from lib import git_context

	e = _CompactE()
	e.messages = []
	cc.bind(e)
	git_context.bind(e)
	rt.bind(e)
	return e


def test_build_model_messages_after_compact_drops_folded_turns():
	e = _bind_compact()
	e.messages = [
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "old question"},
		{"role": "assistant", "content": '<tool_call>{"name":"x"}</tool_call>'},
		{"role": "assistant", "content": '<tool_response>{"name":"x","content":"ok"}</tool_response>'},
		{"role": "user", "content": "new question"},
		{
			"role": "compact",
			"content": "handoff about old work",
			"_aiui_compact": True,
			"compact_count": 1,
		},
	]
	e.compact_count = 1
	wire = cc.BuildModelMessages(sanitize=False)
	roles = [m.get("role") for m in wire]
	texts = [m.get("content") or "" for m in wire]
	assert roles[0] == "system"
	assert roles[1] == "user"
	assert wire[1].get("_aiui_summary") is True
	assert "handoff about old work" in texts[1]
	assert roles[-1] == "user"
	assert texts[-1] == "new question"
	assert "system" not in roles[1:]
	blob = "\n".join(texts)
	assert "<tool_call>" not in blob
	assert "<tool_response>" not in blob
	assert "old question" not in blob

	clean = cc.BuildModelMessages(sanitize=True)
	assert clean[1]["role"] == "user"
	assert not any(m.get("role") == "system" and "handoff" in str(m.get("content") or "") for m in clean)


def test_hidden_history_lists_folded_turns():
	e = _bind_compact()
	old_user = {"role": "user", "content": "old question"}
	call = {"role": "assistant", "content": '<tool_call>{"name":"x"}</tool_call>'}
	resp = {"role": "assistant", "content": '<tool_response>ok</tool_response>'}
	new_user = {"role": "user", "content": "new question"}
	e.messages = [
		{"role": "system", "content": "sys"},
		old_user,
		call,
		resp,
		new_user,
		{"role": "compact", "content": "summary", "_aiui_compact": True, "compact_count": 1},
	]
	hidden = cc.HiddenHistoryMessages()
	assert hidden == [old_user, call, resp]
	assert new_user not in hidden


def _compacted_messages():
	"""Transcript with one compaction, then a full turn and a new user message."""
	return [
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "old question"},
		{"role": "assistant", "content": "old answer"},
		{"role": "user", "content": "live question"},
		{
			"role": "compact",
			"content": "handoff about old work",
			"_aiui_compact": True,
			"compact_count": 1,
			"keep_back": 1,
		},
		{"role": "assistant", "content": "answer to live question"},
		{"role": "user", "content": "yes do that"},
	]


def test_wire_keeps_turns_added_after_compaction():
	"""Turns after the notice were never summarized, so they must stay verbatim."""
	e = _bind_compact()
	e.compact_count = 1
	e.messages = _compacted_messages()
	wire = cc.BuildModelMessages()
	texts = [str(m.get("content") or "") for m in wire]
	blob = "\n".join(texts)
	assert "live question" in blob
	assert "answer to live question" in blob
	assert texts[-1] == "yes do that"
	assert "old question" not in blob
	assert "old answer" not in blob
	hidden = [str(m.get("content") or "") for m in cc.HiddenHistoryMessages()]
	assert hidden == ["old question", "old answer"]


def test_hidden_history_keep_back_ignores_shared_object():
	"""The same dict can sit in the folded region and the live tail."""
	e = _bind_compact()
	e.compact_count = 1
	same = {"role": "user", "content": "same words"}
	e.messages = [
		{"role": "system", "content": "sys"},
		same,
		{
			"role": "compact",
			"content": "summary",
			"_aiui_compact": True,
			"compact_count": 1,
			"keep_back": 0,
		},
		same,
	]
	hidden = [m.get("content") for m in cc.HiddenHistoryMessages()]
	assert hidden == ["same words"]


def test_compact_region_skips_already_summarized_turns():
	e = _bind_compact()
	e.messages = _compacted_messages()
	raw = list(e.messages)
	region = cc._RegionForSummary(
		raw, cc.FindKeepTailStart(raw), cc.DeriveKeepStart(raw),
	)
	assert [str(m.get("content") or "") for m in region] == [
		"live question", "answer to live question",
	]


def test_compact_prompt_preserve_list_no_token_cap(tmp_path):
	e = _bind_compact()
	e.aiui_home = str(tmp_path)
	text = cc.LoadCompactPrompt()
	assert "10K" not in text
	assert "10k" not in text.lower()
	low = text.lower()
	assert "problems that came up" in low
	assert "options raised" in low
	assert "preference or boundary" in low
	assert "finished vs still open" in low
	assert "names, paths, numbers" in low
	assert "output only the summary" in low


def test_map_anthropic_full_sidecar_as_is():
	sidecar = [
		{"type": "thinking", "thinking": "plan", "signature": "sig"},
		{"type": "text", "text": "I'll look it up."},
		{"type": "tool_use", "id": "toolu_1", "name": "glob_files", "input": {}},
	]
	xml = (
		"<think>\nplan\n</think>\n\nI'll look it up.\n\n"
		'<tool_call>{"id":"toolu_1","name":"glob_files","arguments":{}}</tool_call>'
	)
	_system, out = rt.MapMessagesAnthropic([
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": xml, "_anthropic_content": sidecar},
		{"role": "assistant", "content": '<tool_response>{"id":"toolu_1","content":"ok"}</tool_response>'},
	])
	asst = [m for m in out if m.get("role") == "assistant"]
	assert len(asst) == 1
	assert asst[0]["content"] == sidecar
	results = [
		b for m in out if m.get("role") == "user"
		for b in (m.get("content") if isinstance(m.get("content"), list) else [])
		if isinstance(b, dict) and b.get("type") == "tool_result"
	]
	assert results and results[0]["tool_use_id"] == "toolu_1"


def test_map_anthropic_thinking_only_sidecar_rebuilds():
	sidecar = [{"type": "thinking", "thinking": "plan", "signature": "sig"}]
	xml = "I'll look it up.\n\n" + '<tool_call>{"id":"toolu_1","name":"glob_files","arguments":{}}</tool_call>'
	_system, out = rt.MapMessagesAnthropic([
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": xml, "_anthropic_content": sidecar},
	])
	asst = [m for m in out if m.get("role") == "assistant"]
	assert len(asst) == 1
	blocks = asst[0]["content"]
	assert blocks[0]["type"] == "thinking"
	assert any(b.get("type") == "text" and "I'll look it up." in (b.get("text") or "") for b in blocks)
	assert any(b.get("type") == "tool_use" and b.get("id") == "toolu_1" for b in blocks)


def test_anthropic_thinking_on_is_adaptive_without_block_binding():
	e = _bind_compact()
	e.reasoning_effort = "auto"
	payload = {}
	rt.ApplyAnthropicThinkingFields(payload)
	# block_binding needs the thinking-binding-controls beta header; without it
	# every thinking request 400s.
	assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
	assert "output_config" not in payload


def test_anthropic_thinking_off_omits_thinking():
	e = _bind_compact()
	e.reasoning_effort = "off"
	payload = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
	rt.ApplyAnthropicThinkingFields(payload)
	# `disabled` is rejected by always-thinking models; omit instead.
	assert "thinking" not in payload
	assert "output_config" not in payload


def test_anthropic_max_effort_kept():
	e = _bind_compact()
	e.reasoning_effort = "max"
	payload = {}
	rt.ApplyAnthropicThinkingFields(payload)
	assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
	assert payload["output_config"]["effort"] == "max"


def test_anthropic_payload_sends_no_sampling_params():
	_bind_compact()
	from lib import text_backend as tb

	tb.bind(rt.E)
	payload = tb.BuildAnthropicMessagesPayload(
		"claude-opus-5", "sys", [{"role": "user", "content": "hi"}], 4096, stream=True,
	)
	assert "temperature" not in payload
	assert "top_p" not in payload
	assert "top_k" not in payload
	assert payload["max_tokens"] == 4096


def test_openai_official_payload_drops_sampling_params():
	_bind_compact()
	from lib import text_backend as tb

	tb.bind(rt.E)
	rt.E.top_p = 0.9
	rt.E.frequency_penalty = 0.5
	rt.E.presence_penalty = 0.5
	rt.E.remote_base_url = "https://api.openai.com/v1"
	payload = tb.BuildRemoteChatPayload("gpt-5", [], 0.8, 4096, stream=False)
	for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "max_tokens"):
		assert key not in payload, key
	assert payload["max_completion_tokens"] == 4096


def test_openai_compat_payload_keeps_sampling_params():
	_bind_compact()
	from lib import text_backend as tb

	tb.bind(rt.E)
	rt.E.top_p = 0.9
	rt.E.frequency_penalty = 0.5
	rt.E.presence_penalty = 0.0
	# Local serve / OpenRouter still get the sliders.
	rt.E.remote_base_url = "http://localhost:8000/v1"
	payload = tb.BuildRemoteChatPayload("qwen", [], 0.8, 4096, stream=False)
	assert payload["temperature"] == 0.8
	assert payload["top_p"] == 0.9
	assert payload["frequency_penalty"] == 0.5
	assert "presence_penalty" not in payload
	assert payload["max_tokens"] == 4096
	assert "max_completion_tokens" not in payload


def test_anthropic_auth_header_follows_base_url():
	_bind_compact()
	from lib import text_backend as tb

	tb.bind(rt.E)
	rt.E.remote_api_key = "k1"
	for base in ("", "https://api.anthropic.com", "https://api.anthropic.com/v1"):
		rt.E.remote_base_url = base
		assert tb._AnthropicIsOfficial() is True, base
	for base in (
		"https://api.deepseek.com/anthropic",
		"https://api.z.ai/api/anthropic",
		"http://localhost:4000",
	):
		rt.E.remote_base_url = base
		assert tb._AnthropicIsOfficial() is False, base


def test_remote_usage_counts_anthropic_cache_tokens():
	e = _bind_compact()
	e.last_usage_prompt = None
	e.last_usage_completion = None
	cc.RecordRemoteUsage({
		"input_tokens": 5,
		"output_tokens": 12,
		"cache_read_input_tokens": 90000,
		"cache_creation_input_tokens": 1200,
	})
	assert e.last_usage_prompt == 91205
	assert e.last_usage_completion == 12
	# message_delta reports output only — the prompt must survive
	cc.RecordRemoteUsage({"output_tokens": 30})
	assert e.last_usage_prompt == 91205
	assert e.last_usage_completion == 30


def test_remote_usage_openai_prompt_not_double_counted():
	e = _bind_compact()
	e.last_usage_prompt = None
	cc.RecordRemoteUsage({
		"prompt_tokens": 1000,
		"completion_tokens": 10,
		"prompt_tokens_details": {"cached_tokens": 900},
	})
	assert e.last_usage_prompt == 1000


def test_anthropic_stream_closes_think_before_text(monkeypatch):
	"""thinking → text must close <think>, else reasoning shows as the reply."""
	import threading

	from lib import ipc
	from lib import text_backend as tb

	e = _bind_compact()
	e.remote_api_key = "x"
	e.remote_base_url = ""
	e.last_usage_prompt = None
	e.last_usage_completion = None
	tb.bind(e)

	class Ev:
		def __init__(self, **kw):
			self.__dict__.update(kw)

	events = [
		Ev(type="message_start", message=Ev(usage=Ev(
			input_tokens=5, output_tokens=1, cache_read_input_tokens=9000,
		))),
		Ev(type="content_block_delta", index=0, delta=Ev(type="thinking_delta", thinking="plan")),
		Ev(type="content_block_delta", index=1, delta=Ev(type="text_delta", text="answer")),
		Ev(type="message_delta", delta=Ev(stop_reason="end_turn"), usage=Ev(output_tokens=12)),
		Ev(type="message_stop"),
	]

	class Stream:
		def __iter__(self):
			return iter(events)

		def get_final_message(self):
			return None

	class Cm:
		def __enter__(self):
			return Stream()

		def __exit__(self, *a):
			return False

	class Client:
		class messages:
			@staticmethod
			def stream(**kw):
				return Cm()

	monkeypatch.setattr(tb, "_BuildAnthropicClient", lambda: Client())
	monkeypatch.setattr(ipc, "StopRequested", lambda: False)
	monkeypatch.setattr(ipc, "LogErr", lambda *a, **k: None)

	state = {
		"gen_text": "", "last_blocks": 0, "stopped": False,
		"cancel": threading.Event(), "think": {"open": False},
		"stop_reason": None, "xml_tool_stop": False,
		"tool_acc": rt.NewAnthropicToolUseAcc(),
	}
	tb._StreamViaAnthropic({"model": "m", "messages": []}, state, False, None, (), ipc)
	assert state["gen_text"].replace("[AI_UI_BR]", "\n") == "<think>\nplan\n</think>\n\nanswer"
	assert e.last_usage_prompt == 9005
	assert e.last_usage_completion == 12


def test_map_anthropic_tool_result_is_error():
	_system, out = rt.MapMessagesAnthropic([
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": '<tool_call>{"id":"toolu_1","name":"glob_files","arguments":{}}</tool_call>'},
		{"role": "assistant", "content": '<tool_response>{"id":"toolu_1","content":"ERROR: path is empty","error":true}</tool_response>'},
	])
	results = [
		b for m in out if m.get("role") == "user"
		for b in (m.get("content") if isinstance(m.get("content"), list) else [])
		if isinstance(b, dict) and b.get("type") == "tool_result"
	]
	assert results and results[0]["tool_use_id"] == "toolu_1"
	assert results[0]["is_error"] is True
	assert results[0]["content"].startswith("ERROR:")


def test_tool_result_error_false_not_sniffed_from_prefix():
	_name, body, _tid, err = rt.ParseToolResponseInner(
		'{"id":"t1","content":"ERROR: compiler output","error":false}'
	)
	assert err is False
	assert body.startswith("ERROR:")


def test_tool_result_legacy_error_prefix_without_flag():
	_name, body, _tid, err = rt.ParseToolResponseInner(
		'{"id":"t1","content":"ERROR: path is empty"}'
	)
	assert err is True
	assert body.startswith("ERROR:")


def test_messages_for_summarizer_drops_last_user_keeps_prior_summary():
	e = _bind_compact()
	e.compact_count = 1
	e.messages = _compacted_messages()
	prefix = cc.MessagesForSummarizer()
	blob = "\n".join(str(m.get("content") or "") for m in prefix)
	assert "handoff about old work" in blob
	assert "live question" in blob
	assert "answer to live question" in blob
	assert "yes do that" not in blob
	assert "old question" not in blob
	assert "old answer" not in blob


def test_messages_for_summarizer_first_compact_excludes_only_last_user():
	e = _bind_compact()
	e.messages = [
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "old question"},
		{"role": "assistant", "content": "old answer"},
		{"role": "user", "content": "new question"},
	]
	prefix = cc.MessagesForSummarizer()
	texts = [str(m.get("content") or "") for m in prefix]
	assert "old question" in texts
	assert "old answer" in texts
	assert "new question" not in texts


def test_messages_for_summarizer_empty_when_only_last_user():
	e = _bind_compact()
	e.messages = [
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "only question"},
	]
	assert cc.MessagesForSummarizer() == []


def test_openai_compact_summarize_sets_tool_choice_none():
	e = _bind_compact()
	e.enable_tooluse = 1
	e.remote_tool_format = "openai"
	e.tool_funcs = [{
		"type": "function",
		"function": {"name": "glob_files", "parameters": {"type": "object", "properties": {}}},
	}]
	e._compact_summarize = True
	payload = {}
	rt.ApplyOpenAIToolFields(payload)
	assert payload.get("tools")
	assert payload["tool_choice"] == "none"


def test_anthropic_compact_summarize_sets_tool_choice_none():
	e = _bind_compact()
	e.enable_tooluse = 1
	e.remote_tool_format = "anthropic"
	e.tool_funcs = [{"name": "glob_files", "inputSchema": {"type": "object", "properties": {}}}]
	e._compact_summarize = True
	payload = {}
	rt.ApplyAnthropicToolFields(payload)
	assert payload.get("tools")
	assert payload["tool_choice"] == {"type": "none"}


def test_anthropic_timeout_floors_at_ten_minutes():
	from lib import text_backend as tb

	assert tb._AnthropicTimeoutSeconds(1024) == 600.0
	assert tb._AnthropicTimeoutSeconds(21333) == 600.0
	assert tb._AnthropicTimeoutSeconds(32000) == 900.0
	assert tb._AnthropicTimeoutSeconds(128000) == 3600.0
	assert tb._AnthropicTimeoutSeconds(0) == 600.0


def test_anthropic_nonstream_timeout_keeps_fast_connect():
	from lib import text_backend as tb

	t = tb._AnthropicNonStreamTimeout(32000)
	if hasattr(t, "connect"):
		assert t.connect == 5.0
		assert t.read == 900.0
	else:
		assert t == 900.0


def test_prefix_has_reasoning_artifacts():
	assert cc._PrefixHasReasoningArtifacts([{
		"role": "assistant",
		"content": "ok",
		"_anthropic_content": [
			{"type": "thinking", "thinking": "plan", "signature": "sig"},
			{"type": "text", "text": "ok"},
		],
	}])
	assert cc._PrefixHasReasoningArtifacts([{
		"role": "assistant",
		"content": "ok",
		"_reasoning_details": [{"type": "reasoning.encrypted", "data": "blob"}],
	}])
	assert not cc._PrefixHasReasoningArtifacts([
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "ok"},
	])


def test_summarize_keeps_thinking_when_prefix_has_sidecars(monkeypatch):
	from lib import text_backend as tb
	from lib import ipc

	e = _bind_compact()
	e.reasoning_effort = "off"
	e.text_backend = "remote"
	tb.bind(e)
	seen = {}
	monkeypatch.setattr(
		tb, "RemoteChatReply",
		lambda **kw: seen.update(effort=e.reasoning_effort) or "Handoff facts here.",
	)
	monkeypatch.setattr(ipc, "LogErr", lambda *a, **k: None)
	prefix = [{
		"role": "assistant",
		"content": "ok",
		"_anthropic_content": [
			{"type": "thinking", "thinking": "plan", "signature": "sig"},
			{"type": "text", "text": "ok"},
		],
	}]
	out = cc._SummarizeContinuation(prefix, 256)
	assert "Handoff" in out
	assert seen.get("effort") == "auto"
	assert e.reasoning_effort == "off"


def test_summarize_disables_thinking_without_sidecars(monkeypatch):
	from lib import text_backend as tb
	from lib import ipc

	e = _bind_compact()
	e.reasoning_effort = "auto"
	e.text_backend = "remote"
	tb.bind(e)
	seen = {}
	monkeypatch.setattr(
		tb, "RemoteChatReply",
		lambda **kw: seen.update(effort=e.reasoning_effort) or "Handoff facts here.",
	)
	monkeypatch.setattr(ipc, "LogErr", lambda *a, **k: None)
	prefix = [
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "ok"},
	]
	out = cc._SummarizeContinuation(prefix, 256)
	assert "Handoff" in out
	assert seen.get("effort") == "off"
	assert e.reasoning_effort == "auto"


def test_merge_reasoning_effort_legacy_false():
	from lib.chat_template import MergeReasoningEffort, ThinkingKwargs

	assert MergeReasoningEffort("high", "false") == "off"
	assert MergeReasoningEffort("none") == "off"
	assert MergeReasoningEffort("auto", "true") == "auto"
	e = _bind_compact()
	e.reasoning_effort = "off"
	assert ThinkingKwargs(e) == {"enable_thinking": False}
	e.reasoning_effort = "low"
	kw = ThinkingKwargs(e)
	assert kw["enable_thinking"] is True
	assert kw["reasoning_effort"] == "low"
	e.reasoning_effort = "auto"
	assert ThinkingKwargs(e) == {}


def test_openai_off_sends_none_except_official():
	e = _bind_compact()
	e.reasoning_effort = "off"
	e.remote_base_url = "https://openrouter.ai/api/v1"
	payload = {}
	rt.ApplyOpenAIThinkingFields(payload)
	assert payload.get("reasoning_effort") == "none"
	e.remote_base_url = "https://api.openai.com/v1"
	payload = {"reasoning_effort": "high"}
	rt.ApplyOpenAIThinkingFields(payload)
	assert "reasoning_effort" not in payload


def test_openai_max_becomes_xhigh_then_high_on_official():
	e = _bind_compact()
	e.reasoning_effort = "max"
	e.remote_base_url = "https://openrouter.ai/api/v1"
	payload = {}
	rt.ApplyOpenAIThinkingFields(payload)
	assert payload.get("reasoning_effort") == "xhigh"
	e.remote_base_url = "https://api.openai.com/v1"
	payload = {}
	rt.ApplyOpenAIThinkingFields(payload)
	assert payload.get("reasoning_effort") == "high"


def _partial_tool_transcript():
	return [
		{"role": "system", "content": "same system"},
		{"role": "user", "content": "go"},
		{
			"role": "assistant",
			"content": (
				'<tool_call>{"id":"toolu_1","name":"glob_files","arguments":{}}</tool_call>'
				'<tool_call>{"id":"toolu_2","name":"grep_files","arguments":{}}</tool_call>'
			),
		},
		{
			"role": "assistant",
			"content": '<tool_response>{"id":"toolu_1","content":"ok"}</tool_response>',
		},
		{"role": "user", "content": "next"},
	]


def test_openai_stopped_tool_keeps_finished_result():
	out = rt.MapMessagesOpenAI(_partial_tool_transcript())
	assert [m.get("role") for m in out] == ["system", "user", "assistant", "tool", "tool", "user"]
	assert out[3]["tool_call_id"] == "toolu_1"
	assert out[3]["content"] == "ok"
	assert out[4]["tool_call_id"] == "toolu_2"
	assert out[4]["content"] == rt.STOPPED_BEFORE_TOOL
	assert out[5]["content"] == "next"


def test_anthropic_stopped_tool_is_error_before_next_user():
	_system, out = rt.MapMessagesAnthropic(_partial_tool_transcript())
	results = [
		b for m in out if m.get("role") == "user"
		for b in (m.get("content") if isinstance(m.get("content"), list) else [])
		if isinstance(b, dict) and b.get("type") == "tool_result"
	]
	assert [b["tool_use_id"] for b in results] == ["toolu_1", "toolu_2"]
	assert results[0]["content"] == "ok"
	assert results[0].get("is_error") is not True
	assert results[1]["content"] == rt.STOPPED_BEFORE_TOOL
	assert results[1]["is_error"] is True
	last = out[-1]
	blocks = last.get("content") if isinstance(last.get("content"), list) else []
	texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
	assert "next" in texts
	stopped_at = next(i for i, b in enumerate(blocks) if isinstance(b, dict) and b.get("tool_use_id") == "toolu_2")
	next_at = next(i for i, b in enumerate(blocks) if isinstance(b, dict) and b.get("text") == "next")
	assert stopped_at < next_at


def test_openai_prompt_cache_key_is_stable_and_host_gated():
	from lib import text_backend as tb

	e = _bind_compact()
	tb.bind(e)
	e.remote_prompt_cache = 1
	msgs = [
		{"role": "system", "content": "same system"},
		{"role": "user", "content": "hi"},
	]
	e.remote_base_url = "https://api.openai.com/v1"
	official = {"messages": msgs}
	rt.ApplyOpenAICacheFields(official)
	again = {"messages": [{"role": "system", "content": "same system"}, {"role": "user", "content": "later"}]}
	rt.ApplyOpenAICacheFields(again)
	assert official["prompt_cache_key"] == again["prompt_cache_key"]
	assert "cache_control" not in official
	kw = tb._CreateKwForOpenAI({"model": "gpt-5", "messages": msgs, "prompt_cache_key": official["prompt_cache_key"]})
	assert kw["extra_body"]["prompt_cache_key"] == official["prompt_cache_key"]
	assert "prompt_cache_key" not in kw

	e.remote_base_url = "https://openrouter.ai/api/v1"
	router = {"messages": msgs}
	rt.ApplyOpenAICacheFields(router)
	assert router["prompt_cache_key"] == official["prompt_cache_key"]
	assert router["cache_control"] == {"type": "ephemeral"}

	e.remote_base_url = "http://localhost:11434/v1"
	serve = {"messages": msgs, "prompt_cache_key": "x", "cache_control": {"type": "ephemeral"}}
	rt.ApplyOpenAICacheFields(serve)
	assert "prompt_cache_key" not in serve
	assert "cache_control" not in serve

	e.remote_base_url = "https://api.openai.com/v1"
	e.remote_prompt_cache = 0
	off = {"messages": msgs, "prompt_cache_key": "x", "cache_control": {"type": "ephemeral"}}
	rt.ApplyOpenAICacheFields(off)
	assert "prompt_cache_key" not in off
	assert "cache_control" not in off


def test_anthropic_cache_is_top_level_marker():
	e = _bind_compact()
	e.remote_prompt_cache = 1
	on = {"system": "Stay put."}
	rt.ApplyAnthropicCacheFields(on)
	assert on["cache_control"] == {"type": "ephemeral"}
	assert on["system"] == "Stay put."
	e.remote_prompt_cache = 0
	off = {"system": "Stay put.", "cache_control": {"type": "ephemeral"}}
	rt.ApplyAnthropicCacheFields(off)
	assert "cache_control" not in off
	assert off["system"] == "Stay put."
