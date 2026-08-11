"""AI UI chat template — built dynamically from tokenizer special tokens."""

from __future__ import annotations

import json
import re
from datetime import datetime


def _DateString() -> str:
	"""Human-readable date for the model."""
	return datetime.now().strftime("%d %b %Y")


def _DatePreamble() -> str:
	return (
		"{%- if not date_string is defined %}{%- set date_string = "
		+ _J(_DateString())
		+ " %}{%- endif %}\n"
	)


def _NativeTemplateString(tokenizer) -> str:
	ct = getattr(tokenizer, "chat_template", None) or ""
	if isinstance(ct, list):
		for entry in ct:
			if not isinstance(entry, dict):
				continue
			name = str(entry.get("name", "") or "")
			if name in ("tool_use", "default"):
				tpl = entry.get("template")
				if isinstance(tpl, str) and tpl.strip():
					return tpl
		for entry in ct:
			if isinstance(entry, dict) and isinstance(entry.get("template"), str):
				return entry["template"]
		return ""
	return str(ct)


def _ParseLlamaHeaders(native_ct: str) -> tuple[str, str]:
	ct = str(native_ct or "")
	m = re.search(
		r"'(<\|[^']+\|>)'\s*\+\s*message\['role'\]\s*\+\s*'(<\|[^']+\|>)'",
		ct,
	)
	if m:
		return m.group(1), m.group(2)
	return "<|start_header_id|>", "<|end_header_id|>"


def _WireFormat(tokenizer) -> str:
	ct = _NativeTemplateString(tokenizer)
	if "im_start" in ct or "<|im_start|>" in ct:
		return "chatml"
	if "start_header_id" in ct or "header_id" in ct:
		return "llama"
	return "chatml"


def _J(val) -> str:
	return json.dumps(str(val if val is not None else ""))


def _ContentFmtMacro() -> str:
	"""Flatten string or multimodal content lists (image / image_url / video / video_url)."""
	return (
		"{%- macro fmt_content(content) -%}\n"
		"{%- if content is string -%}\n"
		"{{ content }}\n"
		"{%- elif content is iterable -%}\n"
		"{%- for part in content -%}\n"
		"{%- if part is string -%}{{ part }}\n"
		"{%- elif part.type == 'text' -%}{{ part.text }}\n"
		"{%- elif part.type == 'image' or part.type == 'image_url' -%}[image]\n"
		"{%- elif part.type == 'video' or part.type == 'video_url' -%}[video]\n"
		"{%- elif part.type == 'audio' -%}[audio]\n"
		"{%- endif -%}\n"
		"{%- endfor -%}\n"
		"{%- endif -%}\n"
		"{%- endmacro -%}\n"
	)


def _AssistantContentMacro() -> str:
	return (
		"{%- macro fmt_assistant_content(content) -%}\n"
		"{%- set flat = fmt_content(content) -%}\n"
		"{%- if flat is string and flat.startswith('<tool_call>') -%}\n"
		"{{ flat }}\n"
		"{%- elif flat is string and flat.strip().startswith('{') "
		"and '\"name\"' in flat and not flat.startswith('<tool_response>') -%}\n"
		"<tool_call>\n"
		"{{ flat.strip() }}\n"
		"</tool_call>\n"
		"{%- else -%}\n"
		"{{ flat }}\n"
		"{%- endif -%}\n"
		"{%- endmacro -%}\n"
	)


def _RenderAssistantMacro() -> str:
	"""Emit assistant content + optional reasoning_content / tool_calls (HF-shaped)."""
	return (
		"{%- macro render_assistant(message) -%}\n"
		"{%- set content = message.content -%}\n"
		"{%- set reasoning_content = '' -%}\n"
		"{%- if message.reasoning_content is string -%}\n"
		"{%- set reasoning_content = message.reasoning_content -%}\n"
		"{%- elif content is string and '</think>' in content -%}\n"
		"{%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') -%}\n"
		"{%- set content = content.split('</think>')[-1].lstrip('\\n') -%}\n"
		"{%- endif -%}\n"
		"{%- set reasoning_content = reasoning_content|trim -%}\n"
		"{%- if reasoning_content -%}\n"
		"<think>\\n{{ reasoning_content }}\\n</think>\\n\\n"
		"{%- endif -%}\n"
		"{{ fmt_assistant_content(content) }}"
		"{%- if message.tool_calls is defined and message.tool_calls -%}\n"
		"{%- for tool_call in message.tool_calls -%}\n"
		"{%- if tool_call.function is defined -%}{%- set tc = tool_call.function -%}"
		"{%- else -%}{%- set tc = tool_call -%}{%- endif -%}\n"
		"{%- set args = tc.arguments if tc.arguments is defined else {} -%}\n"
		"\\n<tool_call>\\n"
		"{%- if args is string -%}"
		"{\"name\": {{ tc.name | tojson }}, \"arguments\": {{ args }}}\\n"
		"{%- else -%}"
		"{{ {'name': tc.name, 'arguments': args} | tojson }}\\n"
		"{%- endif -%}"
		"</tool_call>"
		"{%- endfor -%}\n"
		"{%- endif -%}\n"
		"{%- endmacro -%}\n"
	)


def _ToolsBlockMiddle() -> str:
	return (
		"{%- for tool in tools %}\n"
		"{%- if tool.function is defined %}{%- set fn = tool.function %}{%- else %}{%- set fn = tool %}{%- endif %}\n"
		"{{ '\\n' }}{{ fn | tojson }}"
		"{%- endfor %}\n"
	)


def BuildTemplate(tokenizer, include_tool_instructions: bool = True) -> str:
	"""Return a jinja chat_template string tailored to this tokenizer.

	include_tool_instructions: when False, still emit <tools> schemas but omit
	the JSON/XML call-format tutorial (use a rule instead).
	"""
	fmt = _WireFormat(tokenizer)
	content_macro = _ContentFmtMacro()
	macro = _AssistantContentMacro()
	render_asst = _RenderAssistantMacro()
	date_pre = _DatePreamble()
	today = "Current Date: {{ date_string }}\\n\\n"
	tools_mid = _ToolsBlockMiddle()
	if include_tool_instructions:
		tools_pre = (
			"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n"
			"You are provided with function signatures within <tools></tools> XML tags:\\n<tools>"
		)
		tools_suf = (
			"\\n</tools>\\n\\nYou may call one or more functions in a single <tool_call> block.\\n"
			"Use a JSON object for one call, or a JSON array for multiple calls:\\n"
			"<tool_call>\\n"
			"{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n"
			"</tool_call>\\n"
			"or\\n"
			"<tool_call>\\n"
			"[{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}, "
			"{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}]\\n"
			"</tool_call>\\n\\n"
			"Reminders:\\n"
			"- Put any reasoning before the tool call and after the tool response(s)\\n"
			"- Prefer one function when later steps depend on an earlier result; "
			"use a JSON array only when calls are independent and safe to run in parallel\\n"
		)
	else:
		tools_pre = (
			"# Tools\\n\\n"
			"Function signatures within <tools></tools> XML tags:\\n<tools>"
		)
		tools_suf = "\\n</tools>"
	# When enable_thinking is explicitly false, close an empty think block (Qwen3.x style).
	gen_think = (
		"{%- if enable_thinking is defined and enable_thinking == false %}\n"
		"<think>\\n</think>\\n"
		"{%- endif %}\n"
	)

	if fmt == "llama":
		header_start, header_end = _ParseLlamaHeaders(_NativeTemplateString(tokenizer))
		bos = getattr(tokenizer, "bos_token", None) or ""
		eot = getattr(tokenizer, "eos_token", None) or "<|eot_id|>"
		return (
			content_macro
			+ macro
			+ render_asst
			+ date_pre
			+ "{%- if not bos_token is defined %}{%- set bos_token = " + _J(bos) + " %}{%- endif %}\n"
			+ "{%- if not eot_token is defined %}{%- set eot_token = " + _J(eot) + " %}{%- endif %}\n"
			+ "{%- if not header_start is defined %}{%- set header_start = " + _J(header_start) + " %}{%- endif %}\n"
			+ "{%- if not header_end is defined %}{%- set header_end = " + _J(header_end) + " %}{%- endif %}\n"
			+ "{{ bos_token }}\n"
			+ "{%- macro role_header(r) -%}\n"
			+ "{{ header_start + r + header_end + '\\n\\n' }}\n"
			+ "{%- endmacro -%}\n"
			+ "{%- if tools %}\n"
			+ "{%- set loop_messages = messages %}\n"
			+ "{%- set system_text = '' %}\n"
			+ "{%- if loop_messages and loop_messages[0].role == 'system' %}\n"
			+ "{%- set system_text = fmt_content(loop_messages[0].content) + '\\n\\n' %}\n"
			+ "{%- set loop_messages = loop_messages[1:] %}\n"
			+ "{%- endif %}\n"
			+ "{{ role_header('system') }}" + today + "{{ system_text }}"
			+ tools_pre
			+ tools_mid
			+ tools_suf
			+ "{{ eot_token }}\n"
			+ "{%- else %}\n"
			+ "{%- set loop_messages = messages %}\n"
			+ "{%- if loop_messages and loop_messages[0].role == 'system' %}\n"
			+ "{{ role_header('system') }}" + today + "{{ fmt_content(loop_messages[0].content) | trim }}{{ eot_token }}\n"
			+ "{%- set loop_messages = loop_messages[1:] %}\n"
			+ "{%- endif %}\n"
			+ "{%- endif %}\n"
			+ "{%- for message in loop_messages %}\n"
			+ "{%- set content = fmt_content(message.content) %}\n"
			+ "{%- if message.role == 'assistant' %}\n"
			+ "{{ role_header('assistant') }}{{ render_assistant(message) | trim }}{{ eot_token }}\n"
			+ "{%- elif message.role == 'user' %}\n"
			+ "{{ role_header('user') }}{{ content | trim }}{{ eot_token }}\n"
			+ "{%- elif message.role == 'system' %}\n"
			+ "{{ role_header('system') }}{{ content | trim }}{{ eot_token }}\n"
			+ "{%- elif message.role == 'tool' %}\n"
			+ "{{ role_header('user') }}<tool_response>\\n{{ content }}\\n</tool_response>{{ eot_token }}\n"
			+ "{%- else %}\n"
			+ "{{ header_start + message.role + header_end + '\\n\\n' }}{{ content | trim }}{{ eot_token }}\n"
			+ "{%- endif %}\n"
			+ "{%- endfor %}\n"
			+ "{%- if add_generation_prompt %}\n"
			+ "{{ role_header('assistant') }}\n"
			+ gen_think
			+ "{%- endif %}\n"
		)

	im_start = "<|im_start|>"
	im_end = getattr(tokenizer, "eos_token", None) or ""
	return (
		content_macro
		+ macro
		+ render_asst
		+ date_pre
		+ "{%- if not im_start is defined %}{%- set im_start = " + _J(im_start) + " %}{%- endif %}\n"
		+ "{%- if not im_end is defined %}{%- set im_end = " + _J(im_end) + " %}{%- endif %}\n"
		+ "{%- if tools %}\n"
		+ "{{ im_start + 'system\\n' }}" + today
		+ "{%- if messages and messages[0].role == 'system' %}\n"
		+ "{{ fmt_content(messages[0].content) + '\\n\\n' }}\n"
		+ "{%- set loop_messages = messages[1:] %}\n"
		+ "{%- else %}\n"
		+ "{%- set loop_messages = messages %}\n"
		+ "{%- endif %}"
		+ tools_pre
		+ tools_mid
		+ tools_suf
		+ "{{ im_end + '\\n' }}\n"
		+ "{%- else %}\n"
		+ "{%- set loop_messages = messages %}\n"
		+ "{%- if loop_messages and loop_messages[0].role == 'system' %}\n"
		+ "{{ im_start + 'system\\n' }}" + today + "{{ fmt_content(loop_messages[0].content) + im_end + '\\n' }}\n"
		+ "{%- set loop_messages = loop_messages[1:] %}\n"
		+ "{%- endif %}\n"
		+ "{%- endif %}\n"
		+ "{%- for message in loop_messages %}\n"
		+ "{%- set content = fmt_content(message.content) %}\n"
		+ "{%- if message.role == 'assistant' %}\n"
		+ "{{ im_start + 'assistant\\n' }}{{ render_assistant(message) }}{{ im_end + '\\n' }}\n"
		+ "{%- elif message.role == 'user' %}\n"
		+ "{{ im_start + 'user\\n' + content + im_end + '\\n' }}\n"
		+ "{%- elif message.role == 'system' %}\n"
		+ "{{ im_start + 'system\\n' + content + im_end + '\\n' }}\n"
		+ "{%- elif message.role == 'tool' %}\n"
		+ "{{ im_start + 'user\\n' }}<tool_response>\\n{{ content }}\\n</tool_response>{{ im_end + '\\n' }}\n"
		+ "{%- else %}\n"
		+ "{{ im_start + message.role + '\\n' + content + im_end + '\\n' }}\n"
		+ "{%- endif %}\n"
		+ "{%- endfor %}\n"
		+ "{%- if add_generation_prompt %}\n"
		+ "{{ im_start + 'assistant\\n' }}\n"
		+ gen_think
		+ "{%- endif %}\n"
	)


def ThinkingKwargs(engine=None) -> dict:
	"""enable_thinking / reasoning_effort for apply_chat_template (omit when auto)."""
	kw = {}
	if engine is None:
		return kw
	et = str(getattr(engine, "enable_thinking", "auto") or "auto").strip().lower()
	if et in ("true", "1", "yes", "on"):
		kw["enable_thinking"] = True
	elif et in ("false", "0", "no", "off"):
		kw["enable_thinking"] = False
	re = str(getattr(engine, "reasoning_effort", "auto") or "auto").strip().lower()
	if re == "none":
		re = "auto"
	elif re == "minimal":
		re = "low"
	if re and re != "auto" and re in ("low", "medium", "high", "xhigh", "max"):
		kw["reasoning_effort"] = re
	return kw


def TemplateKwargs(tokenizer, engine=None) -> dict:
	kw = {"date_string": _DateString()}
	kw.update(ThinkingKwargs(engine))
	if _WireFormat(tokenizer) == "llama":
		header_start, header_end = _ParseLlamaHeaders(_NativeTemplateString(tokenizer))
		kw["bos_token"] = getattr(tokenizer, "bos_token", None) or ""
		kw["eot_token"] = getattr(tokenizer, "eos_token", None) or "<|eot_id|>"
		kw["header_start"] = header_start
		kw["header_end"] = header_end
	else:
		kw["im_start"] = "<|im_start|>"
		kw["im_end"] = getattr(tokenizer, "eos_token", None) or ""
	return kw


def ApplyToTokenizer(tokenizer, engine=None) -> str:
	"""Replace tokenizer.chat_template with the AI UI template. Returns wire format name."""
	# Snapshot native jinja once. Re-applying AI UI teplate must
	# not overwrite the backup with the AI UI template itself.
	store = engine if engine is not None else tokenizer
	if not getattr(store, "_aiui_native_template", None):
		native = _NativeTemplateString(tokenizer)
		if native:
			store._aiui_native_template = native
	wire = _WireFormat(tokenizer)
	include_instr = True
	if engine is not None:
		include_instr = bool(getattr(engine, "aiui_tool_instructions", True))
	tokenizer.chat_template = BuildTemplate(tokenizer, include_tool_instructions=include_instr)
	tokenizer._aiui_wire = wire
	return wire
