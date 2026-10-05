"""Local / remote chat generation helpers."""

from __future__ import annotations

# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

import os
import sys
import time
import gc
import glob
import random
import threading
import tempfile
import importlib
import ctypes
import wave
import queue
import json
from datetime import datetime, timezone

from lib.local_ml import EnsureLocalMl, ResolveTorchDtype

# Filled by _EnsureChatMl() on first local-text use.
torch = None
transformers = None
TextIteratorStreamer = None
StoppingCriteriaList = None
BitsAndBytesConfig = None
AutoTokenizer = None
AutoProcessor = None
AutoModel = None
AutoModelForCausalLM = None
AutoModelForImageTextToText = None
AutoModelForMultimodalLM = None
process_vision_info = None
_chat_ml_ready = False


def _EnsureChatMl():
	"""Import torch/transformers stack once for local chat."""
	global torch, transformers, TextIteratorStreamer, StoppingCriteriaList
	global BitsAndBytesConfig, AutoTokenizer, AutoProcessor, AutoModel
	global AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM
	global process_vision_info, _chat_ml_ready
	if _chat_ml_ready:
		return
	EnsureLocalMl("Local text chat")
	import torch as _torch
	import transformers as _transformers
	from transformers import (
		TextIteratorStreamer as _TextIteratorStreamer,
		StoppingCriteriaList as _StoppingCriteriaList,
		BitsAndBytesConfig as _BitsAndBytesConfig,
		AutoTokenizer as _AutoTokenizer,
		AutoProcessor as _AutoProcessor,
		AutoModel as _AutoModel,
		AutoModelForCausalLM as _AutoModelForCausalLM,
		AutoModelForImageTextToText as _AutoModelForImageTextToText,
	)
	try:
		from transformers import AutoModelForMultimodalLM as _AutoModelForMultimodalLM
	except ImportError:
		_AutoModelForMultimodalLM = None
	from qwen_vl_utils import process_vision_info as _process_vision_info
	torch = _torch
	transformers = _transformers
	TextIteratorStreamer = _TextIteratorStreamer
	StoppingCriteriaList = _StoppingCriteriaList
	BitsAndBytesConfig = _BitsAndBytesConfig
	AutoTokenizer = _AutoTokenizer
	AutoProcessor = _AutoProcessor
	AutoModel = _AutoModel
	AutoModelForCausalLM = _AutoModelForCausalLM
	AutoModelForImageTextToText = _AutoModelForImageTextToText
	AutoModelForMultimodalLM = _AutoModelForMultimodalLM
	process_vision_info = _process_vision_info
	if isinstance(getattr(E, "torch_dtype", "auto"), str):
		E.torch_dtype = ResolveTorchDtype(E.torch_dtype, default="auto")
	_chat_ml_ready = True


def MsgTimestamp():
	"""UTC timestamp for chat.json messages (ISO-8601 with milliseconds)."""
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def StampMsg(msg):
	"""Ensure a message dict has a timestamp (in-place). Does not overwrite."""
	if isinstance(msg, dict) and not msg.get("timestamp"):
		msg["timestamp"] = MsgTimestamp()
	return msg


def ResolveMmFamily():
	"""Resolve attach #am_select (auto / qwen_vl / janus) for local load + tokenize."""
	mp = str(getattr(E, "mm_processor", "auto") or "auto").strip().lower()
	if mp in ("qwen_vl", "janus"):
		return mp
	mid = str(getattr(E, "model_id", "") or "").lower().replace("\\", "/")
	if "janus" in mid:
		return "janus"
	return "auto"


def _UsesAiuiWire():
	"""True when the AI UI jinja template is actually applied (not multimodal)."""
	return bool(getattr(E, "use_aiui_template", False)) and int(getattr(E, "model_type", 0) or 0) != 2


def _ChatTemplateKwargs():
	from lib import chat_template
	if _UsesAiuiWire():
		return chat_template.TemplateKwargs(E.tokenizer, E)
	return chat_template.ThinkingKwargs(E)


def LoadTokenizer():
	_EnsureChatMl()

	try:
		if E.model_type == 2 and ResolveMmFamily() == "janus":
			from lib import janus_backend
			janus_backend.LoadProcessor()
		elif E.model_type == 2:
			if E.tokenizer_model == '':
				E.tokenizer = AutoProcessor.from_pretrained(E.model_id, gguf_file=E.gguf_file, trust_remote_code=E.custom_token_code)
			else:
				E.tokenizer = AutoProcessor.from_pretrained(E.tokenizer_model, gguf_file=E.gguf_file, trust_remote_code=E.custom_token_code)
		else:
			if E.tokenizer_model == '':
				E.tokenizer = AutoTokenizer.from_pretrained(E.model_id, gguf_file=E.gguf_file, trust_remote_code=E.custom_token_code)
			else:
				E.tokenizer = AutoTokenizer.from_pretrained(E.tokenizer_model, gguf_file=E.gguf_file, trust_remote_code=E.custom_token_code)
	except Exception as e:
		sys.exit(f"ERROR: failed to load text tokenizer ({e})")

	# New processor/tokenizer - drop any prior model's native-template backup.
	E._aiui_native_template = None
	SyncTokenizerTemplate()


def SyncTokenizerTemplate():
	"""Apply or restore chat template based on Template Format."""
	from lib import chat_template
	from lib import ipc
	if E.tokenizer is None:
		return
	# Janus VLChatProcessor: template lives on .tokenizer
	if ResolveMmFamily() == "janus":
		return
	# Multimodal processors need native vision placeholders (<|image_pad|>, etc.)
	if int(getattr(E, "model_type", 0) or 0) == 2:
		native = getattr(E, "_aiui_native_template", None)
		if not native:
			native = chat_template._NativeTemplateString(E.tokenizer)
			if native:
				E._aiui_native_template = native
		if native:
			E.tokenizer.chat_template = native
		if getattr(E, "use_aiui_template", False):
			ipc.LogErr(
				"WARN: AI UI Template is ignored for multimodal models; "
				"using the model chat template so vision tokens work."
			)
		return
	if getattr(E, "use_aiui_template", False):
		chat_template.ApplyToTokenizer(E.tokenizer, E)
	elif getattr(E, "_aiui_native_template", None):
		E.tokenizer.chat_template = E._aiui_native_template


def _ToolsForTemplate():
	"""HF tools= schemas for apply_chat_template (AI UI and native)."""
	if not getattr(E, "enable_tooluse", 0) or not getattr(E, "tool_funcs", None):
		return None
	# AI UI template expects unwrapped function dicts; native templates usually
	# accept OpenAI-style {type, function} or unwrapped — pass HF-shaped list.
	if _UsesAiuiWire():
		return [t.get("function", t) for t in E.tool_funcs]
	return list(E.tool_funcs)


def _SplitToolSegments(content: str):
	"""Yield ('text'|'tool_call'|'tool_response', payload) from assistant XML transcript."""
	from lib.tool_calls import SplitToolSegments
	yield from SplitToolSegments(content)


def _ReshapeToolMessages(msgs):
	"""Map AI UI XML assistant turns to HF tool_calls / role=tool for native templates."""
	from lib import tool_calls as tc_mod

	out = []
	for m in msgs or []:
		role = m.get("role")
		content = m.get("content")
		# Already HF-shaped
		if role == "tool" or m.get("tool_calls"):
			out.append(dict(m))
			continue
		if role not in ("assistant", getattr(E, "bot_name", None), "bot") or not isinstance(content, str):
			out.append(dict(m))
			continue
		if "<tool_call>" not in content and "<tool_response>" not in content:
			out.append(dict(m))
			continue

		bot_role = "assistant"
		buf_text = []
		pending_calls = []

		def FlushText():
			nonlocal buf_text
			if not buf_text:
				return
			joined = "\n\n".join(buf_text).strip()
			buf_text = []
			if joined:
				out.append({"role": bot_role, "content": joined})

		def FlushCalls():
			nonlocal pending_calls
			if not pending_calls:
				return
			tc_list = []
			for c in pending_calls:
				args = c.get("arguments") or {}
				tc_list.append({
					"type": "function",
					"function": {
						"name": c.get("name"),
						"arguments": args if isinstance(args, str) else json.dumps(args),
					},
				})
			pending_calls = []
			msg = {"role": bot_role, "content": "", "tool_calls": tc_list}
			out.append(msg)

		for kind, payload in _SplitToolSegments(content):
			if kind == "text":
				FlushCalls()
				buf_text.append(payload)
			elif kind == "tool_call":
				FlushText()
				parsed = tc_mod.ParseToolCallInner(payload)
				if parsed:
					pending_calls.extend(parsed)
				else:
					# Unparsed — keep raw XML as assistant text.
					FlushCalls()
					buf_text.append("<tool_call>" + payload + "</tool_call>")
			elif kind == "tool_response":
				FlushText()
				FlushCalls()
				name = "unknown"
				body = payload
				try:
					data = json.loads(payload) if payload else {}
					if isinstance(data, dict):
						name = str(data.get("name") or name)
						body = data.get("content", payload)
						if not isinstance(body, str):
							body = json.dumps(body)
				except Exception:
					pass
				out.append({"role": "tool", "name": name, "content": body})
		FlushText()
		FlushCalls()
	return out


def _MessagesForModel():
	"""Conversation for the model (budgeted when context compaction is on)."""
	from lib import context_compact
	return context_compact.BuildModelMessages()


def _PromptForModel():
	"""Plain-text prompt path when chat templates are off."""
	return E.prompt or ""


def LoadModel():
	_EnsureChatMl()
	if E.tokenizer == None: LoadTokenizer()

	if E.model_type == 2 and ResolveMmFamily() == "janus":
		from lib import janus_backend
		janus_backend.LoadModel()
		return

	model_loader = AutoModelForCausalLM

	if E.model_type == 0:
		if E.text_lang_class != None:
			model_loader = getattr(importlib.import_module("transformers"), E.text_lang_class)
	elif E.model_type == 1:
		model_loader = AutoModel
	elif E.model_type == 2:
		if E.multi_modal_class != None:
			model_loader = getattr(importlib.import_module("transformers"), E.multi_modal_class)
		elif ResolveMmFamily() == "qwen_vl":
			model_loader = AutoModelForImageTextToText
		elif AutoModelForMultimodalLM is not None:
			model_loader = AutoModelForMultimodalLM
		else:
			model_loader = AutoModelForImageTextToText
	E._mm_backend = ResolveMmFamily() if E.model_type == 2 else "text"

	# Hold image/gen headroom before placement so device_map packs into remaining free VRAM.
	E.ReserveVRAM()

	# Newer transformers reject load_in_8bit= on from_pretrained; use quantization_config.
	# device_map="auto" uses free VRAM after the ReserveVRAM hold (0.8.0-style; no max_memory).
	load_kwargs = {
		"torch_dtype": E.torch_dtype,
		"attn_implementation": E.attn_implementation,
		"use_safetensors": E.use_safetensors,
		"trust_remote_code": E.custom_model_code,
	}
	if E.gguf_file:
		load_kwargs["gguf_file"] = E.gguf_file

	use_device_map = E.load_in_8bit or E.device == "auto"
	if E.load_in_8bit:
		load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
	if use_device_map:
		load_kwargs["device_map"] = "auto"
		load_kwargs["low_cpu_mem_usage"] = True
	elif E.device != "cuda":
		load_kwargs["low_cpu_mem_usage"] = True

	E.model = model_loader.from_pretrained(E.model_id, **load_kwargs)
	# bitsandbytes / device_map models are already placed; .to() can break them
	if not use_device_map:
		E.model = E.model.to(E.device)

	if E.model_adapter != '':
		E.model.load_adapter(E.model_adapter)

	E.model.eval()

	if torch.__version__ >= "2" and E.torch_compile:
		E.model = torch.compile(E.model, mode=E.compile_mode)

	E.ReserveVRAM()
	_LogDeviceMap()


def _LogDeviceMap():
	"""Log placement after load. Note: transformers only sets hf_device_map when
	accelerate_dispatch runs (device_map uses more than one device or disk). An
	all-CPU map leaves hf_device_map unset even though device_map='auto' was used.
	"""
	model = getattr(E, "model", None)
	from lib import ipc
	from collections import Counter

	hold_mb = int(E.ActiveReserveMb()) if hasattr(E, "ActiveReserveMb") else 0
	held = getattr(E, "reserved_vram", None) is not None
	free_mb = -1
	try:
		import torch
		if torch.cuda.is_available():
			free_b, _tot = torch.cuda.mem_get_info(0)
			free_mb = int(free_b // (1024 * 1024))
	except Exception:
		pass

	hf_map = getattr(model, "hf_device_map", None) if model is not None else None
	param_counts = Counter()
	if model is not None:
		try:
			param_counts = Counter(str(p.device) for p in model.parameters())
		except Exception:
			pass

	dev = str(getattr(E, "device", "") or "")
	if isinstance(hf_map, dict) and hf_map:
		counts = Counter(str(v) for v in hf_map.values())
		cpu_n = sum(n for d, n in counts.items() if d in ("cpu", "disk"))
		gpu_n = sum(n for d, n in counts.items() if d not in ("cpu", "disk", "meta"))
		ipc.LogErr(
			f"text model device_map: {gpu_n} GPU / {cpu_n} CPU/disk "
			f"(device={dev}, hold={hold_mb}MB held={held}, free={free_mb}MB, {dict(counts)})"
		)
		if cpu_n > gpu_n and cpu_n > 0:
			ipc.LogErr(
				"WARN: most modules on CPU/disk - first tokens may be very slow with little GPU use. "
				"Lower im_res_vram_mb / min_res_vram_mb if the text model should use more VRAM."
			)
		return

	ipc.LogErr(
		f"text model has no hf_device_map (device={dev}, hold={hold_mb}MB held={held}, "
		f"free={free_mb}MB, param_devices={dict(param_counts) or 'unknown'})"
	)
	cpu_params = sum(n for d, n in param_counts.items() if d.startswith("cpu"))
	gpu_params = sum(n for d, n in param_counts.items() if d.startswith("cuda"))
	meta_params = sum(n for d, n in param_counts.items() if d.startswith("meta"))
	if cpu_params and not gpu_params:
		ipc.LogErr(
			"WARN: text model weights are entirely on CPU. Generation will be very slow. "
			"Lower im_res_vram_mb / min_res_vram_mb so some layers can fit on the GPU at load time."
		)
	elif meta_params and not gpu_params and not cpu_params:
		ipc.LogErr(
			"WARN: text model parameters are still on the meta device - generate may hang or fail."
		)

class TextGenThread(threading.Thread):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.exception = None
		self.result = None

	def run(self):
		try:
			if torch is not None:
				with torch.no_grad():
					self.result = self._target(*self._args, **self._kwargs)
			else:
				self.result = self._target(*self._args, **self._kwargs)
		except SystemExit:
			# Async kill via PyThreadState_SetAsyncExc
			self.result = None
		except Exception as e:
			self.exception = e

	def ForceKill(self):
		"""Inject SystemExit into this thread. Often ignored mid-CUDA; callers should retry."""
		tid = self.ident
		if tid is None:
			return False
		# Linux often needs c_ulong; try both widths.
		for cast in (ctypes.c_ulong, ctypes.c_long):
			try:
				res = ctypes.pythonapi.PyThreadState_SetAsyncExc(cast(tid), ctypes.py_object(SystemExit))
			except Exception:
				continue
			if res == 0:
				continue
			if res > 1:
				ctypes.pythonapi.PyThreadState_SetAsyncExc(cast(tid), None)
				return False
			return True
		return False


def RunStoppable(target, *args, **kwargs):
	"""Run blocking work on TextGenThread so STOP can ForceKill (best-effort).

	Used for post-reply TTS / avatar animation which otherwise block the engine
	thread with no active gen thread for RequestStop to interrupt.
	Returns (ok, result). ok is False when STOP was requested (before or during).
	"""
	from lib import ipc

	if ipc.StopRequested():
		return False, None
	thread = TextGenThread(target=target, args=args, kwargs=kwargs)
	ipc.SetActiveGenThread(thread)
	thread.start()
	while thread.is_alive():
		if ipc.StopRequested():
			_JoinAfterStop(thread)
			break
		thread.join(timeout=0.15)
	ipc.SetActiveGenThread(None)
	if ipc.StopRequested():
		return False, None
	if thread.exception:
		raise thread.exception
	return True, thread.result


class StoppingCriteriaKeys:
	def __init__(self, keywords_ids: list):
		self.keywords = keywords_ids

	def __call__(self, input_ids, scores, **kwargs):
		_EnsureChatMl()
		done = False
		if self.keywords and len(self.keywords) < len(input_ids[0]):
			key_num = -len(self.keywords)
			done = True
			for keyword in self.keywords:
				if input_ids[0][key_num] != keyword:
					done = False
					break
				key_num += 1
		return torch.full((input_ids.shape[0],), done, device=input_ids.device, dtype=torch.bool)


def _CriteriaBool(input_ids, done: bool):
	_EnsureChatMl()
	return torch.full((input_ids.shape[0],), bool(done), device=input_ids.device, dtype=torch.bool)


class StoppingCriteriaToolCall:
	"""Halt once a complete generated <tool_call>...</tool_call> appears.

	Only inspects tokens after prompt_len. Never matches tool examples baked into
	the prompt. Uses one decode of the generated suffix per step (not the full
	sequence, and not per-token re-decode of the prompt).
	"""

	_OPEN = "<tool_call>"
	_CLOSE = "</tool_call>"

	def __init__(self, tokenizer, prompt_len: int):
		self.tokenizer = tokenizer
		self.prompt_len = max(0, int(prompt_len))

	def __call__(self, input_ids, scores, **kwargs):
		try:
			gen = input_ids[0][self.prompt_len:]
			if gen.numel() == 0:
				return _CriteriaBool(input_ids, False)
			text = self.tokenizer.decode(gen.tolist(), skip_special_tokens=True)
			open_i = text.find(self._OPEN)
			if open_i < 0:
				return _CriteriaBool(input_ids, False)
			close_i = text.find(self._CLOSE, open_i + len(self._OPEN))
			return _CriteriaBool(input_ids, close_i >= 0)
		except Exception:
			return _CriteriaBool(input_ids, False)


def _NormTokenIdList(val):
	"""Normalize eos_token_id (int / list / None) to a list of non-negative ints."""
	if val is None:
		return []
	if isinstance(val, (list, tuple)):
		raw = val
	else:
		raw = [val]
	out = []
	for x in raw:
		try:
			i = int(x)
		except (TypeError, ValueError):
			continue
		if i >= 0:
			out.append(i)
	return out


def _PrimaryEosTokenIds(tok):
	"""EOS ids transformers already exposes (tokenizer / generation_config / config)."""
	ids = _NormTokenIdList(getattr(tok, "eos_token_id", None))
	if ids:
		return ids
	model = getattr(E, "model", None)
	if model is None:
		return []
	gc = getattr(model, "generation_config", None)
	ids = _NormTokenIdList(getattr(gc, "eos_token_id", None) if gc is not None else None)
	if ids:
		return ids
	cfg = getattr(model, "config", None)
	return _NormTokenIdList(getattr(cfg, "eos_token_id", None) if cfg is not None else None)


def _FallbackEosTokenIds(tok):
	"""Last resort: map known EOS markers to single token ids when primary EOS is missing."""
	if tok is None:
		return []
	from lib.files import EOS_MARKERS
	unk = getattr(tok, "unk_token_id", None)
	out = []
	seen = set()
	conv = getattr(tok, "convert_tokens_to_ids", None)
	for marker in EOS_MARKERS:
		tid = None
		if callable(conv):
			try:
				tid = conv(marker)
			except Exception:
				tid = None
		if tid is None or tid == unk or (isinstance(tid, int) and tid < 0):
			try:
				enc = tok.encode(marker, add_special_tokens=False)
			except Exception:
				enc = None
			if not isinstance(enc, list) or len(enc) != 1:
				continue
			tid = enc[0]
		try:
			tid = int(tid)
		except (TypeError, ValueError):
			continue
		if tid < 0 or tid == unk or tid in seen:
			continue
		seen.add(tid)
		out.append(tid)
	return out


def _EosOverrideForGenerate(tok):
	"""Only pass eos_token_id when HF has no primary EOS; then try marker fallbacks."""
	if _PrimaryEosTokenIds(tok):
		return None
	fb = _FallbackEosTokenIds(tok)
	return fb if fb else None


def GetInputDevice():
	"""Device for input tensors when the model may be split across CPU/GPU (device_map).

	Never return `meta` — accelerate can expose embedding weights on meta even when the
	dispatched model is on cuda/cpu; moving inputs there makes generate() blow up in
	_prepare_special_tokens (Tensor.item() on meta tensors).
	"""
	def _real(dev):
		try:
			d = torch.device(dev)
		except Exception:
			return None
		if d.type == "meta":
			return None
		return d

	if E.model is None:
		fallback = _real(E.comp_dev) or _real("cpu")
		return fallback

	try:
		emb = E.model.get_input_embeddings()
		if emb is not None and hasattr(emb, "weight"):
			d = _real(emb.weight.device)
			if d is not None:
				return d
	except Exception:
		pass

	hf_map = getattr(E.model, "hf_device_map", None)
	if hf_map:
		# Prefer embedding-related modules, then any non-meta / non-disk entry.
		ordered = sorted(
			hf_map.items(),
			key=lambda kv: (0 if "embed" in kv[0] else 1, str(kv[0])),
		)
		for _name, dev in ordered:
			if str(dev) in ("disk", "meta"):
				continue
			d = _real(dev)
			if d is not None:
				return d

	try:
		for p in E.model.parameters():
			d = _real(p.device)
			if d is not None:
				return d
	except Exception:
		pass

	if torch.cuda.is_available() and str(getattr(E, "device", "") or "") in ("auto", "cuda"):
		return torch.device("cuda")
	return torch.device("cpu")


def MoveToInputDevice(it):
	"""Place token tensors on the same device as embed_tokens (required for CPU offload)."""
	from collections.abc import Mapping
	dev = GetInputDevice()
	if getattr(dev, "type", None) == "meta":
		# Belt-and-suspenders: never feed meta inputs into generate().
		dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if isinstance(it, torch.Tensor):
		return it.to(dev)
	# BatchEncoding is Mapping but not always isinstance(dict).
	if isinstance(it, Mapping):
		out = {}
		for k, v in it.items():
			if torch.is_tensor(v):
				# Skip moving already-correct tensors; refuse meta destinations above.
				out[k] = v.to(dev) if v.device != dev else v
			else:
				out[k] = v
		return out
	return it


def _MoveBatchToDevice(batch, input_dev):
	"""Move processor/tokenizer outputs to device; reject bare strings."""
	if isinstance(batch, str):
		raise TypeError("chat template returned a string; expected tensors (check tokenize/return_tensors)")
	if hasattr(batch, "to") and not isinstance(batch, dict):
		return batch.to(input_dev)
	if isinstance(batch, dict):
		return {k: (v.to(input_dev) if hasattr(v, "to") else v) for k, v in batch.items()}
	if isinstance(batch, torch.Tensor):
		return batch.to(input_dev)
	raise TypeError(f"unexpected tokenize result type: {type(batch).__name__}")


def _TextHasVisionPlaceholders(text):
	s = str(text or "")
	return any(
		m in s
		for m in (
			"<|image_pad|>",
			"<|video_pad|>",
			"<|vision_start|>",
			"<image>",
			"<video>",
		)
	)


def _TokenizeVlLegacy(messages, apply_kw, input_dev):
	"""Qwen2.5-VL style: string template + process_vision_info + processor()."""
	# Ensure native vision template (AI UI may have been applied earlier this session).
	SyncTokenizerTemplate()
	text = E.tokenizer.apply_chat_template(
	 messages, add_generation_prompt=True, tokenize=False, **apply_kw,
	)
	if not isinstance(text, str):
		return _MoveBatchToDevice(text, input_dev)
	image_inputs, video_inputs = process_vision_info(messages)
	has_media = bool(image_inputs) or bool(video_inputs)
	if has_media and not _TextHasVisionPlaceholders(text):
		raise RuntimeError("Image/video features were prepared but the chat template emitted no vision tokens.")
	result = E.tokenizer(
	 text=[text],
	 images=image_inputs,
	 videos=video_inputs,
	 padding=True,
	 return_tensors="pt",
	)
	return _MoveBatchToDevice(result, input_dev)


def _TokenizeVlModern(messages, apply_kw, input_dev):
	"""Qwen3-VL / Inkling style: processor.apply_chat_template → BatchEncoding."""
	SyncTokenizerTemplate()
	try:
		result = E.tokenizer.apply_chat_template(
		 messages,
		 add_generation_prompt=True,
		 tokenize=True,
		 return_dict=True,
		 return_tensors="pt",
		 **apply_kw,
		)
		if isinstance(result, str):
			raise TypeError("processor returned string")
		return _MoveBatchToDevice(result, input_dev)
	except TypeError:
		return _TokenizeVlLegacy(messages, apply_kw, input_dev)


def TokenizeChat():
	_EnsureChatMl()
	if E.tokenizer == None: LoadTokenizer()
	# Prefer the embedding device when the model is already loaded (device_map may put
	# embed_tokens on CPU). Otherwise keep tensors on CPU and let GenReply move them.
	input_dev = GetInputDevice() if E.model is not None else torch.device("cpu")
	messages = _MessagesForModel()
	tools_arg = _ToolsForTemplate()
	tmpl_kw = _ChatTemplateKwargs()
	family = ResolveMmFamily() if E.model_type == 2 else "text"

	# Native templates expect tool_calls / role=tool; AI UI keeps XML in content.
	if tools_arg is not None and not _UsesAiuiWire():
		messages = _ReshapeToolMessages(messages)

	if family == "janus":
		from lib import janus_backend
		loaded = getattr(E, "_mm_backend", None)
		if E.model is not None and loaded not in (None, "janus"):
			raise RuntimeError(
				"Vision is Janus but the loaded model is not. "
				"Restart the engine after switching Vision to Janus (or set Text Model to a Janus path)."
			)
		return janus_backend.TokenizeChat(messages)

	apply_kw = dict(tmpl_kw)
	if tools_arg is not None:
		apply_kw["tools"] = tools_arg

	if E.use_chat_template:
		# Multimodal: always use VL tokenize paths (tools must not skip vision).
		if E.model_type == 2:
			if family == "qwen_vl":
				return _TokenizeVlLegacy(messages, apply_kw, input_dev)
			return _TokenizeVlModern(messages, apply_kw, input_dev)

		result = E.tokenizer.apply_chat_template(
		 messages, add_generation_prompt=True, return_tensors="pt", **apply_kw,
		)
		return _MoveBatchToDevice(result, input_dev)

	result = E.tokenizer(_PromptForModel(), return_tensors="pt").input_ids
	return _MoveBatchToDevice(result, input_dev)


def _JoinAfterStop(thread, retries=25, pause=0.1):
	"""Keep poking ForceKill while joining — async exc is often ignored mid-CUDA."""
	for _ in range(retries):
		if not thread.is_alive():
			break
		thread.ForceKill()
		thread.join(timeout=pause)
	if thread.is_alive():
		thread.join(timeout=0.5)
		if thread.is_alive():
			from lib import ipc
			ipc.LogErr("WARN: generate thread still running after STOP (ForceKill may be blocked in native code)")


def _SignalStreamerStop(txt_streamer):
	"""Unblock TextIteratorStreamer if generate died without sending the end sentinel."""
	try:
		txt_streamer.text_queue.put_nowait(txt_streamer.stop_signal)
	except Exception:
		pass


def _HfTokenizer():
	"""Underlying HF tokenizer (Processors lack get_vocab needed by streamer / criteria)."""
	tok = getattr(E, "_janus_tokenizer", None) or E.tokenizer
	inner = getattr(tok, "tokenizer", None)
	return inner if inner is not None else tok


def _NewTextStreamer():
	# timeout lets StreamText poll StopRequested while waiting for tokens.
	# Without it, ForceKill can leave the consumer hung forever (no end sentinel)
	# and the UI never gets stream_end → SEND stays stuck on STOP.
	# skip_special_tokens: Janus eos (<｜end▁of▁sentence｜>) must not appear in the UI.
	return TextIteratorStreamer(
		_HfTokenizer(), skip_prompt=True, skip_special_tokens=True, timeout=0.2,
	)


def _PrepareGenKwargs(gen_kwargs):
	"""Strip AI UI private keys; pick generate target (Janus uses language_model)."""
	from lib import janus_backend
	kw = dict(gen_kwargs)
	is_janus = janus_backend.IsJanusGen(kw)
	kw.pop("_janus_generate", None)
	kw.pop("_janus_prompt_len", None)
	if is_janus:
		target = janus_backend.GenerateModel().generate
	else:
		target = E.model.generate
	return target, kw


def StreamText(txt_streamer, gen_kwargs, channel="chat", reuse_stream=False, ui_stream=True, on_block=None):
	"""Consume TextIteratorStreamer.

	ui_stream=True  — send stream_start/delta/end for live token UI.
	ui_stream=False + on_block — still generate with a streamer (early-stop works)
	  but only invoke on_block when a collapse/tool XML close tag appears, so tool
	  turns can update the chat in structural chunks without token streaming.
	"""
	_EnsureChatMl()
	from lib import ipc

	gen_text = ""
	stopped = False
	last_blocks = 0
	block_ends = ("</think>", "</thoughts>", "</tool_call>", "</tool_response>")
	# Do not ClearStop here — a STOP that arrived during tokenize/setup must stick.
	if ipc.StopRequested():
		if ui_stream:
			ipc.SendStreamEnd("stop", channel)
		return ""

	target, run_kw = _PrepareGenKwargs(gen_kwargs)
	thread = TextGenThread(target=target, kwargs=run_kw)
	ipc.SetActiveGenThread(thread)
	thread.start()
	time.sleep(0.5)

	if thread.exception:
		exc = thread.exception
		thread.join()
		ipc.SetActiveGenThread(None)
		if ui_stream:
			ipc.SendStreamEnd("error", channel)
		# Surface the real failure (often Processor quirks with streamer).
		# Callers may retry without streamer.
		raise RuntimeError(f"streaming generate failed: {exc}") from exc

	if ui_stream and not reuse_stream:
		ipc.SendStreamStart(channel)

	while True:
		if thread.exception:
			ipc.LogErr(f"ERROR: model failed to generate text ({thread.exception})")
			break
		if ipc.StopRequested():
			stopped = True
			thread.ForceKill()
			_SignalStreamerStop(txt_streamer)
			break
		try:
			new_text = next(txt_streamer)
		except StopIteration:
			break
		except queue.Empty:
			# generate still running (or stuck in CUDA) - keep polling stop
			if not thread.is_alive():
				_SignalStreamerStop(txt_streamer)
				break
			time.sleep(0.01)
			continue
		# TextStreamer queues "" until a word boundary (space / newline / CJK).
		if not new_text:
			if not thread.is_alive():
				_SignalStreamerStop(txt_streamer)
				break
			time.sleep(0.01)
			continue
		delta = new_text.replace("\n", "[AI_UI_BR]").replace("\r", "")
		if not delta:
			time.sleep(0.01)
			continue
		gen_text += delta
		if ui_stream:
			ipc.SendStreamDelta(delta, channel)
		elif on_block is not None:
			plain = gen_text.replace("[AI_UI_BR]", "\n")
			n_blocks = sum(plain.count(tag) for tag in block_ends)
			if n_blocks > last_blocks:
				last_blocks = n_blocks
				try:
					on_block(plain)
				except Exception as e:
					ipc.LogException("stream_on_block", e)
		if ipc.StopRequested():
			stopped = True
			thread.ForceKill()
			_SignalStreamerStop(txt_streamer)
			break

	if not stopped and ipc.StopRequested():
		stopped = True
		thread.ForceKill()
		_SignalStreamerStop(txt_streamer)

	if stopped:
		_JoinAfterStop(thread)
	else:
		thread.join()
	ipc.SetActiveGenThread(None)
	time.sleep(0.1)
	if ui_stream:
		ipc.SendStreamEnd("stop" if stopped else "ok", channel)

	# Success path: return generate() tensors so GenReply can decode with the
	# same formatting as non-streaming. On STOP, prefer streamed tokens — if
	# ForceKill failed mid-CUDA, thread.result may still be the *full* output.
	if stopped:
		return gen_text.replace("[AI_UI_BR]", "\n")
	if thread.result is None:
		return gen_text.replace("[AI_UI_BR]", "\n")
	return thread.result


def _RunGenerate(gen_kwargs, txt_streamer, channel="chat", reuse_stream=False, ui_stream=True, on_block=None):
	"""Run model.generate with optional streaming and ForceKill STOP."""
	from lib import ipc

	if txt_streamer is not None:
		return StreamText(
			txt_streamer, gen_kwargs, channel,
			reuse_stream=reuse_stream, ui_stream=ui_stream, on_block=on_block,
		)

	if ipc.StopRequested():
		return None

	target, run_kw = _PrepareGenKwargs(gen_kwargs)
	thread = TextGenThread(target=target, kwargs=run_kw)
	ipc.SetActiveGenThread(thread)
	thread.start()
	stopped = False
	while thread.is_alive():
		if ipc.StopRequested():
			stopped = True
			thread.ForceKill()
			_JoinAfterStop(thread)
			break
		time.sleep(0.05)
	else:
		thread.join()
	ipc.SetActiveGenThread(None)
	if stopped:
		return None
	if thread.exception is not None:
		raise thread.exception
	return thread.result


def _PromptLen(it):
	"""Token count of the prompt (excludes tokens that will be generated)."""
	from collections.abc import Mapping
	if isinstance(it, torch.Tensor):
		return int(it.shape[-1])
	if type(it) is list and len(it) > 0 and hasattr(it[0], "shape"):
		return int(it[0].shape[-1])
	ids = None
	if isinstance(it, Mapping) and "input_ids" in it:
		ids = it["input_ids"]
	else:
		ids = getattr(it, "input_ids", None)
	if ids is None:
		return 0
	return int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids[0])


def _BuildChatStoppingCriteria(it):
	"""Stopping criteria for chat generation.

	Tool early-stop uses cheap custom criteria on generated tokens only.
	Non-template mode stops when the model starts a new user turn (ID match).
	Chat-template EOS is left to transformers (eos_token_id); do not add
	</tool_call> here - that belongs in StoppingCriteriaToolCall only.
	"""
	_EnsureChatMl()
	criteria = []
	prompt_len = _PromptLen(it)
	tok = _HfTokenizer()

	if getattr(E, "enable_tooluse", 0):
		criteria.append(StoppingCriteriaToolCall(tok, prompt_len))

	# Legacy plain "role: text" prompts: halt if the model opens a user turn.
	if not getattr(E, "use_chat_template", False):
		s = "\n" + str(getattr(E, "user_name_c", "") or "")
		if s != "\n":
			try:
				ids = tok.encode(s, add_special_tokens=False)
			except Exception:
				ids = []
			if ids:
				criteria.append(StoppingCriteriaKeys(ids))

	return StoppingCriteriaList(criteria) if criteria else None


def GenReply(it, rl, stream=None, reuse_stream=False, ui_stream=None, on_block=None):
	from lib import text_backend as text_backend_mod
	if text_backend_mod.IsRemoteText():
		use_stream = E.gen_mode != 0 if stream is None else bool(stream)
		use_ui_stream = E.gen_mode != 0 if ui_stream is None else bool(ui_stream)
		return text_backend_mod.RemoteChatReply(
			temperature=rl,
			stream=use_stream,
			ui_stream=use_ui_stream and use_stream,
			on_block=on_block if (use_stream and not use_ui_stream) else None,
			reuse_stream=reuse_stream and use_stream,
		)

	if E.model == None: LoadModel()
	from lib import janus_backend
	it = MoveToInputDevice(it)
	E.CleanVRAM()

	use_stream = E.gen_mode != 0 if stream is None else bool(stream)
	# Token-level IPC follows gen_mode unless the caller overrides (tool turns may
	# stream internally for early-stop / block patches while keeping UI quiet).
	use_ui_stream = E.gen_mode != 0 if ui_stream is None else bool(ui_stream)

	if use_stream:
		txt_streamer = _NewTextStreamer()
	else:
		txt_streamer = None

	stop_list = _BuildChatStoppingCriteria(it)

	hf_tok = _HfTokenizer()
	eos_override = _EosOverrideForGenerate(hf_tok)
	if isinstance(it, torch.Tensor) or type(it) is list:
		gen_kwargs = dict(
		 inputs=it, do_sample=E.do_sample, num_beams=E.num_beams, num_beam_groups=E.beam_groups,
		 tokenizer=hf_tok, streamer=txt_streamer, min_new_tokens=E.gen_min, max_new_tokens=E.gen_max,
		 temperature=rl, top_k=E.top_k, top_p=E.top_p, typical_p=E.typical_p, repetition_penalty=E.rep_penalty,
		)
	else:
		try:
			from collections.abc import Mapping
			base = dict(it) if isinstance(it, Mapping) else it
			base['do_sample'] = E.do_sample
			base['num_beams'] = E.num_beams
			base['num_beam_groups'] = E.beam_groups
			base['streamer'] = txt_streamer
			base['min_new_tokens'] = E.gen_min
			base['max_new_tokens'] = E.gen_max
			base['temperature'] = rl
			base['top_k'] = E.top_k
			base['top_p'] = E.top_p
			base['typical_p'] = E.typical_p
			base['repetition_penalty'] = E.rep_penalty
			if not janus_backend.FinalizeGenKwargs(base, hf_tok):
				base['tokenizer'] = hf_tok
			gen_kwargs = base
		except Exception:
			sys.exit("ERROR: unexpected input token format")

	if eos_override is not None:
		gen_kwargs['eos_token_id'] = eos_override

	if stop_list is not None:
		gen_kwargs['stopping_criteria'] = stop_list

	run_kw = dict(
		reuse_stream=reuse_stream and use_stream,
		ui_stream=use_ui_stream and use_stream,
		on_block=on_block if (use_stream and not use_ui_stream) else None,
	)
	try:
		output = _RunGenerate(gen_kwargs, txt_streamer, "chat", **run_kw)
	except Exception as e:
		from lib import ipc
		if txt_streamer is None:
			raise
		ipc.LogErr(f"WARN: streaming generate failed ({e}); retrying without streamer")
		gen_kwargs['streamer'] = None
		output = _RunGenerate(
			gen_kwargs, None, "chat",
			reuse_stream=False, ui_stream=False, on_block=None,
		)

	E.CleanVRAM()
	E.ReserveVRAM()

	if type(output) is str:
		return output
	elif output is None:
		return ""
	janus_txt = janus_backend.DecodeReply(it, output)
	if janus_txt is not None:
		return janus_txt
	prompt_len = _PromptLen(it)
	return hf_tok.decode(
		output[0][prompt_len:],
		skip_special_tokens=E.skip_special_tokens,
		clean_up_tokenization_space=E.cleanup_token_space,
	)


def GenText(it, nt, mt, ds, nb, ng, rl, tk, tp, ty, rp):
	_EnsureChatMl()
	if E.model == None: LoadModel()
	it = MoveToInputDevice(it)
	E.CleanVRAM()

	if E.gen_mode == 0:
		txt_streamer = None
	else:
		txt_streamer = _NewTextStreamer()

	if isinstance(it, torch.Tensor) or type(it) is list:
		gen_kwargs = dict(
		 inputs=it, do_sample=ds, num_beams=nb, num_beam_groups=ng,
		 tokenizer=_HfTokenizer(), streamer=txt_streamer, min_new_tokens=nt, max_new_tokens=mt,
		 temperature=rl, top_k=tk, top_p=tp, typical_p=ty, repetition_penalty=rp
		)
	else:
		try:
			it['do_sample'] = ds
			it['num_beams'] = nb
			it['num_beam_groups'] = ng
			it['tokenizer'] = _HfTokenizer()
			it['streamer'] = txt_streamer
			it['min_new_tokens'] = nt
			it['max_new_tokens'] = mt
			it['temperature'] = rl
			it['top_k'] = tk
			it['top_p'] = tp
			it['typical_p'] = ty
			it['repetition_penalty'] = rp
			gen_kwargs = it
		except Exception:
			sys.exit("ERROR: unexpected input token format")

	output = _RunGenerate(gen_kwargs, txt_streamer, "gen")

	E.CleanVRAM()
	E.ReserveVRAM()

	if type(output) is str:
		return output
	elif output is None:
		return ""
	else:
		return E.tokenizer.decode(output[0], skip_special_tokens=E.skip_special_tokens)


def GetUserNames(user_name=None, bot_name=None):
	E.user_name = "user"
	E.bot_name = "assistant"

	if E.use_chat_template:
		E.user_name_c = E.user_name
		E.bot_name_c = E.bot_name
	else:
		E.user_name_c = E.user_name + E.roles_append_str.rstrip()
		E.bot_name_c = E.bot_name + E.roles_append_str.rstrip()


def DeriveRedoCount(messages=None):
	"""How many trailing bot messages REDO should drop (after session import).

	Matches live _RunBotTurn: only consecutive assistant entries at the end,
	stopping before user / system / compact so a compact notice is never deleted.
	"""
	from lib import context_compact
	raw = list(messages if messages is not None else getattr(E, "messages", None) or [])
	bot = str(getattr(E, "bot_name", "assistant") or "assistant").lower()
	user = str(getattr(E, "user_name", "user") or "user").lower()
	n = 0
	for m in reversed(raw):
		if not isinstance(m, dict):
			break
		if context_compact.IsCompactNotice(m):
			break
		role = str(m.get("role") or "").lower()
		if role == "system" or role == user or role == "user":
			break
		if role == "assistant" or role == bot:
			n += 1
			continue
		break
	return n


def LastTurnAwaitingReply(messages=None):
	"""True when the transcript ends on a user turn (STOP/empty left no assistant)."""
	from lib import context_compact
	raw = list(messages if messages is not None else getattr(E, "messages", None) or [])
	user = str(getattr(E, "user_name", "user") or "user").lower()
	for m in reversed(raw):
		if not isinstance(m, dict):
			return False
		if context_compact.IsCompactNotice(m):
			continue
		role = str(m.get("role") or "").lower()
		if role == "system":
			return False
		return role == user or role == "user"
	return False


def ComposedSystemPrompt():
	from lib import prompt_files
	return prompt_files.ComposeSystemPrompt(
		getattr(E, "aiui_home", "") or "",
		getattr(E, "workspace_dir", "") or "",
		getattr(E, "active_agent", "") or "",
		getattr(E, "enabled_rules", None),
		getattr(E, "enabled_skills", None),
	)


def ApplySystemPrompt(text):
	"""Keep init_prompt as messages[0] without wiping the rest of the transcript."""
	if getattr(E, "prompt_p", 0) == 2:
		return
	body = str(text or "")
	msgs = list(getattr(E, "messages", None) or [])
	old = ""
	if msgs and str(msgs[0].get("role") or "").lower() == "system":
		old = str(msgs[0].get("content") or "")
	E.init_prompt = body
	E.prompt = body
	sys_msg = StampMsg({"role": "system", "content": body})
	if msgs and str(msgs[0].get("role") or "").lower() == "system":
		msgs[0] = sys_msg
		E.messages = msgs
	else:
		E.messages = [sys_msg] + msgs
	if old != body:
		from lib import remote_tools
		remote_tools.WarnIfAnthropicPrefixMayBreak("system prompt")


def InitPrompt(have_prompt=False):
	E.redo_count = 0
	from lib import context_compact
	context_compact.ResetCompactState()
	if E.prompt_p == 2:
		E.prompt = ''
		E.init_prompt = 'No prompt'
		E.messages = []
	else:
		composed = ComposedSystemPrompt()
		E.init_prompt = composed
		E.prompt = composed
		E.messages = [StampMsg({"role": "system", "content": composed})] if composed else []
	if have_prompt:
		E.PrintSleep("CLEAR_DONE:"+(E.init_prompt or "").replace("\n", "[AI_UI_BR]"))
	else:
		E.PrintSleep("INIT_DONE:"+(E.init_prompt or "").replace("\n", "[AI_UI_BR]"))
	context_compact.EmitContextUsage(heuristic=True)


def PrunePrompt():
	from lib import context_compact
	# When compaction is enabled, keep the full transcript; model wire is budgeted separately.
	if not context_compact.CompactEnabled():
		# Compact cursor is meaningless while msg_mem truncation is active.
		context_compact.ResetCompactState()
		i = len(E.messages) - E.msg_mem
		if i > 0:
			E.messages = E.messages[i::]
	else:
		context_compact.SyncCompactFromMessages()
	E.prompt = ''
	i = 0
	l = len(E.messages)
	if (
		not context_compact.CompactEnabled()
		and E.prompt_p == 1
		and len(E.messages) >= E.msg_mem
	):
		if "role" in E.messages[0] and E.messages[0]["role"].lower() != 'system':
			E.messages[0] = StampMsg({"role":"system", "content":E.init_prompt})
	while i < l and "role" in E.messages[i] and "content" in E.messages[i]:
		if context_compact.IsCompactNotice(E.messages[i]):
			i += 1
			continue
		content = E.messages[i]["content"]
		role = E.messages[i]["role"]
		if isinstance(content, str):
			E.prompt += role + E.roles_append_str + content + "\n"
		elif isinstance(content, list):
			E.prompt += role + E.roles_append_str
			for msg_con in content:
				if not isinstance(msg_con, dict):
					continue
				if "user" in msg_con:
					text = msg_con.get("text")
					E.prompt += role + E.roles_append_str + ("" if text is None else str(text)) + "\n"
				else:
					for msg_key, val in msg_con.items():
						if msg_key == "type" or val is None:
							continue
						if isinstance(val, dict):
							val = val.get("url") or val.get("image") or val.get("path") or f"[{msg_key}]"
						E.prompt += str(val) + "\n"
						break
		i += 1
	E.prompt = E.prompt.rstrip("\n")
	if E.prompt_p < 2 and E.prompt == '': E.prompt = E.init_prompt
	# Prefer response metadata / local exact; heuristic when remote has no usage yet.
	context_compact.EmitContextUsageTurnEnd()

