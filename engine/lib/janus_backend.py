"""Local Janus-Pro understanding path (optional `janus` package).

`import lib.janus_backend` is always safe: this module does not import DeepSeek's
`janus` at load time. That package is pulled in only by EnsurePackage() when
loading/tokenizing a Janus model.

Remote / transformers serve is unsupported - callers should refuse remote+janus.
"""

from __future__ import annotations

import os
import sys

E = None


def bind(engine_module):
	global E
	E = engine_module


def LooksLikeJanus(model_id=None) -> bool:
	mid = str(model_id if model_id is not None else getattr(E, "model_id", "") or "").lower().replace("\\", "/")
	return "janus" in mid


def ApplyJanusConfigDataclassCompat():
	"""Skip HF @dataclass wrapping for janus.* PretrainedConfig subclasses.

	transformers 5.5+ applies dataclass(kw_only=True) in PretrainedConfig.__init_subclass__.
	Janus declares `params: AttrDict = {}`, which raises:
	  ValueError: mutable default <class 'dict'> for field params is not allowed
	Must run before importing janus.models.
	"""
	import transformers.configuration_utils as config_utils
	from transformers.configuration_utils import PretrainedConfig

	if getattr(PretrainedConfig, "_aiui_janus_dataclass_compat", False):
		return

	_orig_dataclass = config_utils.dataclass

	def _dataclass(cls=None, /, **kwargs):
		if cls is not None and isinstance(cls, type):
			module = getattr(cls, "__module__", "") or ""
			if module.startswith("janus."):
				return cls
		if cls is None:
			return lambda c: _dataclass(c, **kwargs)
		return _orig_dataclass(cls, **kwargs)

	config_utils.dataclass = _dataclass
	PretrainedConfig._aiui_janus_dataclass_compat = True


def ApplyJanusPostInitPatch():
	"""Ensure MultiModalityCausalLM runs post_init() after construction."""
	from janus.models import modeling_vlm

	cls = modeling_vlm.MultiModalityCausalLM
	if getattr(cls, "_aiui_janus_post_init_patched", False):
		return
	_orig_init = cls.__init__

	def __init__(self, config, *args, **kwargs):
		_orig_init(self, config, *args, **kwargs)
		if hasattr(self, "post_init"):
			self.post_init()

	cls.__init__ = __init__
	cls._aiui_janus_post_init_patched = True


def EnsurePackage():
	"""Require DeepSeek's Janus (janus.models), not the unrelated PyPI async-queue package."""
	ApplyJanusConfigDataclassCompat()
	try:
		from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: F401
		ApplyJanusPostInitPatch()
		return
	except ImportError as e:
		wrong_pypi = False
		try:
			import janus as _j
			# aio-libs/janus exposes AsyncQueue and has no .models
			if getattr(_j, "AsyncQueue", None) is not None and not hasattr(_j, "models"):
				wrong_pypi = True
		except ImportError:
			pass
		hint = (
			"Install DeepSeek Janus from GitHub (not PyPI `janus`):\n"
			"  pip uninstall -y janus\n"
			"  git clone https://github.com/deepseek-ai/Janus.git\n"
			"  cd Janus && pip install -e .\n"
			"See https://github.com/deepseek-ai/Janus"
		)
		if wrong_pypi:
			raise RuntimeError(
				"Wrong package: PyPI `janus` (async queue) is installed, "
				"but Janus-Pro needs deepseek-ai/Janus (`janus.models`).\n" + hint
			) from e
		raise RuntimeError(
			"Janus multimodal mode requires deepseek-ai/Janus (`janus.models`).\n" + hint
			+ f"\n({e})"
		) from e
	except ValueError as e:
		# If compat patch missed (partial import / old transformers layout).
		raise RuntimeError(
			"Janus failed to import under this transformers/Python version "
			f"({e}). AI UI applies a dataclass compat shim for janus.* configs; "
			"update AI UI or pin transformers, or report if this persists."
		) from e


def RepairJanusTokenizer(tok, model_path: str) -> bool:
	"""Restore ByteLevel backend from tokenizer.json.

	transformers' LlamaTokenizer.__init__ rewrites DeepSeek's GPT-2-style
	ByteLevel (Ġ) tokenizer to Metaspace(▁). Encode then drops spaces and
	decode leaves literal Ġ — looks like one long gibberish line in the UI.
	"""
	from tokenizers import Tokenizer

	tj = os.path.join(model_path, "tokenizer.json")
	if tok is None or not os.path.isfile(tj):
		return False
	backend = getattr(tok, "_tokenizer", None)
	if backend is None:
		return False
	# Already ByteLevel (e.g. older transformers) — leave alone.
	try:
		dec = backend.decoder
		if dec is not None and type(dec).__name__ == "ByteLevel":
			return False
	except Exception:
		pass
	tok._tokenizer = Tokenizer.from_file(tj)
	return True


def LoadProcessor():
	EnsurePackage()
	from janus.models import VLChatProcessor
	path = E.tokenizer_model if getattr(E, "tokenizer_model", "") else E.model_id
	E.tokenizer = VLChatProcessor.from_pretrained(path)
	# Streamer / stop strings expect a HF tokenizer.
	E._janus_processor = E.tokenizer
	tok = getattr(E.tokenizer, "tokenizer", None)
	if tok is not None:
		RepairJanusTokenizer(tok, path)
		E._janus_tokenizer = tok
	else:
		E._janus_tokenizer = E.tokenizer


def LoadModel():
	EnsurePackage()
	import torch
	from transformers import AutoModelForCausalLM
	from janus.models import MultiModalityCausalLM  # noqa: F401 — registers remote code

	if getattr(E, "tokenizer", None) is None:
		LoadProcessor()

	# Official Janus examples use bfloat16; float32/auto often yields garbled understanding output.
	dtype = getattr(E, "torch_dtype", "auto")
	if dtype in (None, "auto", ""):
		dtype = torch.bfloat16

	if hasattr(E, "ReserveVRAM"):
		E.ReserveVRAM()

	load_kwargs = {
		"trust_remote_code": True,
		"torch_dtype": dtype,
	}
	use_device_map = getattr(E, "load_in_8bit", False) or getattr(E, "device", "auto") == "auto"
	if use_device_map:
		load_kwargs["device_map"] = "auto"
		# False matches forge loaders — more reliable materialization for Janus vision tower.
		load_kwargs["low_cpu_mem_usage"] = False

	E.model = AutoModelForCausalLM.from_pretrained(E.model_id, **load_kwargs)
	if not use_device_map:
		E.model = E.model.to(E.device, dtype=dtype if isinstance(dtype, torch.dtype) else torch.bfloat16)
	E.model.eval()
	E._mm_backend = "janus"
	if hasattr(E, "ReserveVRAM"):
		E.ReserveVRAM()


def _PartImagePath(part) -> str | None:
	if not isinstance(part, dict):
		return None
	t = str(part.get("type") or "")
	if t == "image":
		val = part.get("image") or part.get("url")
		return str(val) if val else None
	if t == "image_url":
		iu = part.get("image_url")
		if isinstance(iu, dict):
			return str(iu.get("url") or "") or None
		return str(iu) if iu else None
	return None


def _FlattenText(content) -> str:
	if isinstance(content, str):
		return content
	if not isinstance(content, list):
		return str(content or "")
	bits = []
	for part in content:
		if isinstance(part, str):
			bits.append(part)
		elif isinstance(part, dict) and part.get("type") == "text":
			bits.append(str(part.get("text") or ""))
	return "\n".join(b for b in bits if b)


def MessagesToConversation(messages):
	"""Convert AI UI messages to Janus deepseek-template turns.

	Returns (conversation, system_prompt). System text must NOT be a User turn —
	two User messages in a row break the DeepSeek SFT template and produce garbage.
	Roles must match janus deepseek template: <|User|> / <|Assistant|>.
	"""
	system_text = ""
	conv = []
	for m in messages or []:
		if not isinstance(m, dict):
			continue
		role = str(m.get("role") or "")
		content = m.get("content")
		if role == "system":
			text = _FlattenText(content).strip()
			if text:
				system_text = text
			continue
		if role in ("assistant", "bot"):
			conv.append({"role": "<|Assistant|>", "content": _FlattenText(content)})
			continue
		# user
		images = []
		text = ""
		if isinstance(content, list):
			for part in content:
				ip = _PartImagePath(part)
				if ip:
					images.append(ip)
				elif isinstance(part, dict) and part.get("type") == "text":
					text += str(part.get("text") or "")
				elif isinstance(part, str):
					text += part
		else:
			text = _FlattenText(content)
		# Official samples put one placeholder per image before the question.
		tag = getattr(getattr(E, "_janus_processor", None), "image_tag", None) or "<image_placeholder>"
		prefix = (f"{tag}\n" * len(images)) if images else ""
		conv.append({
			"role": "<|User|>",
			"content": (prefix + text).strip(),
			"images": images,
		})
	# Generation prompt: empty assistant turn
	if not conv or conv[-1].get("role") != "<|Assistant|>":
		conv.append({"role": "<|Assistant|>", "content": ""})
	return conv, system_text


def _JanusDevice():
	"""Real device for Janus tensors (never meta)."""
	import torch
	try:
		from lib import chat as chat_mod
		if hasattr(chat_mod, "GetInputDevice"):
			d = chat_mod.GetInputDevice()
			if d is not None and getattr(d, "type", None) != "meta":
				return d
	except Exception:
		pass
	try:
		lm = getattr(E.model, "language_model", None) or E.model
		emb = lm.get_input_embeddings()
		d = emb.weight.device
		if d.type != "meta":
			return d
	except Exception:
		pass
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def TokenizeChat(messages):
	"""Return generate kwargs with inputs_embeds for Janus language_model.generate."""
	EnsurePackage()
	from janus.utils.io import load_pil_images
	import torch

	if getattr(E, "_janus_processor", None) is None:
		LoadProcessor()

	proc = E._janus_processor
	conv, system_text = MessagesToConversation(messages)
	pil_images = load_pil_images(conv)

	# Processor applies system via self.system_prompt (not as a User turn).
	prev_sys = getattr(proc, "system_prompt", "") or ""
	if system_text:
		proc.system_prompt = system_text
	try:
		prepare_inputs = proc(conversations=conv, images=pil_images, force_batchify=True)
	finally:
		proc.system_prompt = prev_sys

	device = _JanusDevice()
	# Match vision/LM dtype (typically bfloat16).
	try:
		dtype = next(E.model.parameters()).dtype
		if dtype.is_floating_point:
			prepare_inputs = prepare_inputs.to(device, dtype=dtype)
		else:
			prepare_inputs = prepare_inputs.to(device)
	except Exception:
		prepare_inputs = prepare_inputs.to(device)

	inputs_embeds = E.model.prepare_inputs_embeds(**prepare_inputs)

	tok = getattr(E, "_janus_tokenizer", None) or getattr(proc, "tokenizer", None)
	eos_id = getattr(tok, "eos_token_id", None)
	# DeepSeek template stop id (also listed in janus conversation.py).
	if eos_id is None:
		eos_id = 100001
	gen = {
		"inputs_embeds": inputs_embeds,
		"attention_mask": prepare_inputs.attention_mask,
		"pad_token_id": eos_id,
		"bos_token_id": getattr(tok, "bos_token_id", None),
		"eos_token_id": eos_id,
		"use_cache": True,
		"_janus_generate": True,
		# With inputs_embeds, generate() returns only NEW tokens — do not slice by prompt len.
		"_janus_prompt_len": 0,
	}
	return gen


def GenerateModel():
	"""Return the module that should run .generate for Janus."""
	lm = getattr(E.model, "language_model", None)
	return lm if lm is not None else E.model


def IsJanusGen(kw) -> bool:
	return isinstance(kw, dict) and bool(kw.get("_janus_generate"))


def FinalizeGenKwargs(base, hf_tok):
	"""Fill tokenizer for Janus generate kwargs. Returns True if Janus."""
	if not IsJanusGen(base):
		return False
	tok = getattr(E, "_janus_tokenizer", None)
	base["tokenizer"] = tok if tok is not None else hf_tok
	return True


def DecodeReply(it, output):
	"""Decode Janus generate tensors, or None if it is not a Janus gen dict."""
	if not IsJanusGen(it):
		return None
	return DecodeOutput(output, it.get("_janus_prompt_len") or 0)


def DecodeOutput(output, prompt_len):
	tok = getattr(E, "_janus_tokenizer", None)
	if tok is None:
		tok = getattr(getattr(E, "_janus_processor", None), "tokenizer", None)
	if tok is None:
		tok = E.tokenizer
	seq = output[0]
	# inputs_embeds path returns only new tokens (prompt_len should be 0).
	if prompt_len and hasattr(seq, "__len__") and len(seq) > prompt_len:
		seq = seq[prompt_len:]
	return tok.decode(
		seq.cpu().tolist() if hasattr(seq, "cpu") else seq,
		skip_special_tokens=True,
	)
