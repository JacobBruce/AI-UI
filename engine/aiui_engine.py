"""
 # © 2026 Bitfreak Software
 # www.bitfreak.info
 # Apache License 2.0
"""

import sys

# Control-plane stdout must stay NDJSON-only. Install the guard before heavy
# imports so library print() during import cannot desync Electron.
sys.stdout.reconfigure(encoding='utf-8')
try:
	sys.stderr.reconfigure(encoding='utf-8')
except Exception:
	pass
from lib import ipc
ipc.InstallStdoutGuard()

# Boot before heavy ML imports so requirements-lite (API-only) can start.
import time, glob, os, gc
import threading
import json
import tempfile
from datetime import datetime
from huggingface_hub import login
from lib.text_norm import NormalizeText
from lib.local_ml import TorchAvailable

ipc.SendHello()
_boot = ipc.WaitBoot()
ipc.LogErr("Engine: boot received, initializing ...")
_app = _boot.get("app") or {}
_tools = _boot.get("tools") or {}
_ai = _boot.get("ai") or {}
_voice = _boot.get("voice") or {}
_chat = _boot.get("chat") or {}

work_dir = str(_app.get("script_dir", "")).rstrip('/\\')
model_id = str(_app.get("model_dir", "")).rstrip('/')
im_model_id = str(_app.get("sd_model", "")).rstrip('/')
tts_model_id = str(_app.get("tts_model", "")).rstrip('/')
sr_model_id = str(_app.get("sr_model", "")).rstrip('/')
model_type = int(_app.get("model_type", 0))
imodel_type = int(_app.get("imodel_type", 0))
smodel_type = int(_app.get("smodel_type", 0))
_model_args_raw = _app.get("model_args", "")
model_args = _model_args_raw.split(',') if isinstance(_model_args_raw, str) else list(_model_args_raw or [])
device = str(_app.get("comp_dev", "auto"))
start_meth = str(_app.get("start_meth", "none"))
avatar_img = str(_app.get("avatar_img", ""))

gen_mode = int(_app.get("gen_mode", 0))
format_mode = int(_app.get("format_mode", 0))
enable_tooluse = int(_app.get("enable_tooluse", 1))
hf_access_token = str(_app.get("hf_token", ""))
text_backend = "remote" if str(_app.get("text_backend", "local") or "local").lower() == "remote" else "local"
remote_base_url = str(_app.get("remote_base_url", "") or "")
remote_model = str(_app.get("remote_model", "") or "")
remote_api_key = str(_app.get("remote_api_key", "") or "")
remote_provider = str(_app.get("remote_provider", "") or "")
_rtf = str(_app.get("remote_tool_format", "openai") or "openai").strip().lower()
remote_tool_format = _rtf if _rtf == "anthropic" else "openai"
remote_prompt_cache = 0 if int(_app.get("remote_prompt_cache", 1) or 0) == 0 else 1
_mp = str(_app.get("mm_processor", "auto") or "auto").strip().lower()
mm_processor = _mp if _mp in ("qwen_vl", "janus") else "auto"
aiui_home = str(_app.get("aiui_home", "")).rstrip('/\\')
app_version = str(_app.get("app_version", "") or "")
workspace_dir = str(_app.get("workspace", "")).rstrip('/\\')
active_agent = str(_app.get("agent", ""))
enabled_rules = _app.get("enabled_rules", None)
enabled_skills = _app.get("enabled_skills", None)

if hf_access_token != '':
	login(token=hf_access_token)

if not os.path.isdir(work_dir):
	sys.exit("ERROR: cannot find engine folder at "+work_dir)

temp_form = str(_tools.get("temp_form", _tools.get("temp_format", "none")))

def _NormTempForm(raw):
	t = str(raw or "none").strip().lower()
	if t == "tool":
		return "aiui"
	if t == "chat":
		return "model"
	if t in ("none", "model", "aiui", "aiui_notoolprompt"):
		return t
	return "none"

temp_form = _NormTempForm(temp_form)
tool_mode = str(_tools.get("tool_mode", _tools.get("tools_file", "custom")))
tool_packs = _tools.get("tool_packs", None)
firecrawl_api_key = str(_tools.get("firecrawl_api_key", "") or "")

msg_mem = int(_ai.get("msg_mem", 10))
compact_at = int(_ai.get("compact_at", 0))
compact_summary_max = int(_ai.get("compact_summary_max", 8192))
git_status_context = int(_ai.get("git_status_context", 0) or 0)
compact_summary = ""
compact_until = 0
compact_count = 0
_compact_bypass = False
# Last remote response usage (prompt/completion); stale after tool results grow the wire.
last_usage_prompt = None
last_usage_completion = None
last_usage_stale = True
gen_max = int(_ai.get("gen_max", _ai.get("max_res", 32000)))
tool_result_max = max(1, int(_ai.get("max_tool_result", 4_000_000) or 4_000_000))
gen_min = int(_ai.get("gen_min", _ai.get("min_res", 1)))
do_sample = int(_ai.get("do_sample", 1))
num_beams = int(_ai.get("num_beams", 1))
beam_groups = int(_ai.get("beam_groups", 1))
rando_lvl = float(_ai.get("rando_lvl", _ai.get("base_temp", 0.8)))
prompt_p = int(_ai.get("prompt_p", 0))
top_k = int(_ai.get("top_k", 50))
top_p = float(_ai.get("top_p", 1.0))
typical_p = float(_ai.get("typical_p", 1.0))
rep_penalty = float(_ai.get("rep_penalty", 1.0))
frequency_penalty = float(_ai.get("frequency_penalty", 0.0))
presence_penalty = float(_ai.get("presence_penalty", 0.0))

from lib.chat_template import MergeReasoningEffort

reasoning_effort = MergeReasoningEffort(
	_ai.get("reasoning_effort", "auto"), _ai.get("enable_thinking"),
)
rando_min = rando_lvl
rando_add = 0.05
rando_sub = 0.01

user_name = 'user'
bot_name = 'assistant'
user_name_c = ''
bot_name_c = ''
roles_append_str = ': '

prompt = ''
init_prompt = ''
msg = ''
redo_count = 0
r = 0

messages = []
responses = []
chat_files = []
tool_funcs = None
tool_func = None
tool_catalog = None
use_tool_config = True
_anthropic_last_content = None  # sidecar from last Anthropic remote reply (thinking/tool_use)
_openai_last_reasoning_details = None  # OpenRouter reasoning_details for the last OpenAI-compat reply
_openai_last_reasoning_pin = None  # {model, provider, zdr} for the endpoint that minted encrypted details

# Provisional defaults; overwritten from GPU size after utils.bind (see below).
min_res_vram_mb = 256
im_res_vram_mb = 0 if im_model_id == '' else 1536
_vram_hold_mb = None  # temporary override for ActiveReserveMb (e.g. image load)
reserved_vram = None

tokenizer_model = ''
vocoder_model = ''
audio_max_new_tokens = 4096
audio_max_chars = 4000
anim_max_secs = 30.0
model_adapter = ''
im_lora_file = ''
im_lora_dir = ''
im_vae_file = ''
im_unet_file = ''
im_tran_file = ''
im_enc_file = ''
im_enc2_file = ''
im_enc3_file = ''
im_lora_scale = 1.0
im_load_in_8bit = False
im_safety_check = True
im_cpu_offload = True
im_att_slicing = False
im_from_single_file = False
im_local_files = False
im_config_file = None
im_model_class = None
im_variant = None
im_use_safetensors = None
text_lang_class = None
multi_modal_class = None
speech_recog_class = None
attn_implementation = None
use_safetensors = None
gguf_file = None
load_in_8bit = False
custom_model_code = False
custom_token_code = False
skip_special_tokens = True
cleanup_token_space = True
torch_compile = False
chunk_audio_secs = 30
im_torch_dtype = "float16"
torch_dtype = "auto"

use_chat_template = True if temp_form != "none" else False
use_aiui_template = temp_form in ("aiui", "aiui_notoolprompt")
# When False, AI UI template still lists <tools> schemas but omits call-format prose.
aiui_tool_instructions = temp_form == "aiui"
_aiui_native_template = None

if imodel_type > 0: im_att_slicing = False
comp_dev = 'cpu'
compile_mode = 'default'
seed_gen = None

if TorchAvailable():
	import torch
	torch.set_float32_matmul_precision('high')
	comp_dev = 'cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu'
	compile_mode = 'default' if comp_dev == 'cpu' else 'max-autotune'
	seed_gen = torch.Generator(device=comp_dev)

###### Library modules (bound to this engine module) ######
ipc.LogErr("Engine: importing libraries ...")
from lib import utils, files, tool_client, tts, asr, chat, img, model_args as model_args_mod, prompt_files
from lib import fs_sandbox, tool_permissions, text_backend as text_backend_mod, janus_backend
from lib import remote_tools as remote_tools_mod
from lib import context_compact as context_compact_mod
from lib import git_context as git_context_mod
from lib import edit_last as edit_last_mod
import sys as _sys
_engine = _sys.modules[__name__]

for _m in (utils, files, tool_client, tts, asr, chat, img, model_args_mod, prompt_files, fs_sandbox, tool_permissions, text_backend_mod, janus_backend, remote_tools_mod, context_compact_mod, git_context_mod):
	_m.bind(_engine)

# Re-export so main-loop names and E.* cross-calls resolve on this module
PrintSleep = utils.PrintSleep
ReserveVRAM = utils.ReserveVRAM
CleanVRAM = utils.CleanVRAM
ActiveReserveMb = utils.ActiveReserveMb
AnimAvatar = utils.AnimAvatar
LoadAvatar = utils.LoadAvatar
CountFiles = files.CountFiles
NewFileName = files.NewFileName
SetWorkingDir = files.SetWorkingDir
ReadTextFile = files.ReadTextFile
WriteTextFile = files.WriteTextFile
SafeString = files.SafeString
StripEmptyThinkTags = files.StripEmptyThinkTags
TrimTrailingChatClosers = files.TrimTrailingChatClosers
FormatThinkMarkers = files.FormatThinkMarkers
FormatToolMarkers = files.FormatToolMarkers
StripEnd = files.StripEnd
StripStart = files.StripStart
LoadToolFuncs = tool_client.LoadToolFuncs
LoadKokoro = tts.LoadKokoro
LoadNeuTTS = tts.LoadNeuTTS
LoadChatTTS = tts.LoadChatTTS
LoadVoxCPM = tts.LoadVoxCPM
LoadKitten = tts.LoadKitten
LoadSpeaker = tts.LoadSpeaker
LoadSysVoices = tts.LoadSysVoices
LoadAIVoices = tts.LoadAIVoices
ConfigTTSEngine = tts.ConfigTTSEngine
LoadTTSVoices = tts.LoadTTSVoices
LoadTTSEngine = tts.LoadTTSEngine
ApplyTalkMode = tts.ApplyTalkMode
CloneVoice = tts.CloneVoice
ModifySound = tts.ModifySound
AddEpsilonNoise = tts.AddEpsilonNoise
SentenceSplit = tts.SentenceSplit
TextToSpeechAI = tts.TextToSpeechAI
TextToSpeechSys = tts.TextToSpeechSys
LoadSRProcessor = asr.LoadSRProcessor
LoadSRModel = asr.LoadSRModel
ResampleSpeech = asr.ResampleSpeech
SpeechToText = asr.SpeechToText
LoadTokenizer = chat.LoadTokenizer
SyncTokenizerTemplate = chat.SyncTokenizerTemplate
LoadModel = chat.LoadModel
TextGenThread = chat.TextGenThread
StoppingCriteriaKeys = chat.StoppingCriteriaKeys
TokenizeChat = chat.TokenizeChat
StreamText = chat.StreamText
GenReply = chat.GenReply
GenText = chat.GenText
GetUserNames = chat.GetUserNames
InitPrompt = chat.InitPrompt
PrunePrompt = chat.PrunePrompt
DiffIMGMode = img.DiffIMGMode
LoadIMGModule = img.LoadIMGModule
LoadIMGModel = img.LoadIMGModel
GenImage = img.GenImage
GenImageIPA = img.GenImageIPA
LoadModels = model_args_mod.LoadModels

# Scale reserve defaults from total VRAM; model_args can override.
min_res_vram_mb, im_res_vram_mb = utils.DefaultVramReserves(im_model_id != '')

model_args_mod.ApplyModelArgs(model_args)

###### TTS STUFF ######

audio_dir = work_dir+'/ai_audio/'
if not os.path.exists(audio_dir): os.mkdir(audio_dir)

speech_wav = audio_dir+'speech.wav'
chat_wav = audio_dir+'chat.wav'
read_wav = audio_dir+'read.wav'

speaker_embeddings = None
synthesizer = None
processor = None
vocoder = None
tts_model = None
tts_ready = False
do_talk = True
do_anim = True
voices = []
ai_voices = []

voice_key = int(_voice.get("voice_key", _voice.get("bot_voice", 0)))
voice_vol = float(_voice.get("voice_vol", _voice.get("speech_vol", 1.0)))
voice_rate = int(_voice.get("voice_rate", _voice.get("speech_rate", 200)))
voice_ps = int(_voice.get("voice_ps", _voice.get("pitch_shift", 0)))
talk_mode = int(_voice.get("talk_mode", 0))
tts_mode = int(_voice.get("tts_mode", 0))
anim_mode = int(_voice.get("anim_mode", -1))

last_smt_val = smodel_type

try:
	SetWorkingDir(anim_mode)
except Exception as e:
	ipc.LogErr(f"WARN: avatar working directory ({e})")

kokoro_loaded = False
chattts_loaded = False
voxcpm_loaded = False
neutts_loaded = False
kitten_loaded = False
audio8_loaded = False
audio8_processor = None

###### SPEECH RECOGNITION STUFF ######

sr_model = None
sr_pipe = None
sr_processor = None

###### TEXT MODEL STUFF ######

#set_seed(int(str(time.time()).replace('.', '')))

model = None
tokenizer = None

###### IMAGE MODEL STUFF ######

im_model = None
orig_shdlr = None
img_gen_mode = False

last_sc_val = im_safety_check
last_vf_val = im_vae_file
last_lf_val = im_lora_file
last_ld_val = im_lora_dir
last_ls_val = im_lora_scale

default_neg_prompt = "low resolution, lowres, low quality, bad quality, jpeg artifacts, pixelated, blurry, error, glitch, cropped, out of frame, out of focus"

person_neg_prompt = "face asymmetry, deformed face, deformed eyes, eyes asymmetry, missing fingers, extra fingers, fused fingers, mutated hands, bad anatomy, wrong proportions, extra limbs, malformed limbs, missing arms, missing legs, extra arms, extra legs"

ipc.LogErr("Engine: loading tool packs ...")
try:
	LoadToolFuncs()
except Exception as e:
	ipc.LogException("LoadToolFuncs", e)
try:
	ipc.LogErr("Engine: loading TTS voices ...")
	LoadTTSVoices()
except Exception as e:
	ipc.LogErr(f"WARN: TTS voices not loaded ({e})")
try:
	ipc.LogErr("Engine: preload models ...")
	LoadModels()
except Exception as e:
	ipc.LogErr(f"WARN: model preload skipped ({e})")
try:
	if anim_mode >= 0:
		ipc.LogErr("Engine: loading avatar ...")
		if not LoadAvatar():
			# Same user-facing path as Apply / update_avatar (boot used to fail silently).
			PrintSleep("GOT_AVATAR:false")
except Exception as e:
	ipc.LogErr(f"WARN: avatar preload skipped ({e})")
	PrintSleep("GOT_AVATAR:false")

ApplyTalkMode()
GetUserNames()
InitPrompt()
ipc.LogErr("Engine: ready")
ipc.SendReady("ready")
_pf_snap = prompt_files.SendPromptFilesEvent(aiui_home, workspace_dir, active_agent, enabled_rules, enabled_skills)
active_agent = _pf_snap.get("agent", "")
enabled_rules = _pf_snap.get("enabled_rules")
enabled_skills = _pf_snap.get("enabled_skills")
if prompt_p < 2:
	chat.ApplySystemPrompt(_pf_snap.get("composed") or "")

def _CmdParams(msg):
	params = msg.get("params")
	if isinstance(params, dict):
		return params
	# Allow flat fields on the cmd object (excluding envelope keys)
	skip = {"v", "type", "name", "params"}
	return {k: v for k, v in msg.items() if k not in skip}

def _NormalizeBotText(response):
	text = (response or "").replace("\r", "").strip(" \n")
	text = TrimTrailingChatClosers(text)
	if not use_chat_template:
		text = StripStart(text, bot_name_c).lstrip(" ")
	return text

def _ParseToolCalls(text):
	"""Extract all complete <tool_call> blocks from one generation.

	Returns (has_tool, preface, calls, leftover):
	- preface: text before the first <tool_call>
	- calls: list of {"kind":"tool","raw": inner} or {"kind":"text","text":...}
	- leftover: text after the last </tool_call> (usually empty with early-stop)

	Early-stop normally yields one block per GenReply; if several complete blocks
	are present (stop miss, non-streaming edge, parallel emit), process all of them.
	"""
	text = text or ""
	open_tag, close_tag = "<tool_call>", "</tool_call>"
	if open_tag not in text:
		return False, "", [], ""
	calls = []
	pos = 0
	preface = ""
	first = True
	while True:
		open_i = text.find(open_tag, pos)
		if open_i < 0:
			break
		close_i = text.find(close_tag, open_i + len(open_tag))
		if close_i < 0:
			break
		before = text[pos:open_i].strip()
		if first:
			preface = before
			first = False
		elif before:
			# Interstitial text between calls — keep visible in the turn.
			calls.append({"kind": "text", "text": before})
		raw_inner = text[open_i + len(open_tag):close_i].strip()
		calls.append({"kind": "tool", "raw": raw_inner})
		pos = close_i + len(close_tag)
	leftover = text[pos:].strip()
	tool_calls = [c for c in calls if c.get("kind") == "tool"]
	if not tool_calls:
		return False, "", [], leftover
	return True, preface, calls, leftover


# Host-wide ceiling for any tool result (native or MCP). Above native 2MB file
# reads / 200k cmd output so those caps still win; 3rd-party MCP dumps cannot
# flood the transcript.
_MAX_TOOL_RESULT_CHARS = 4_000_000


def _ResultToToolContent(tc_result):
	if isinstance(tc_result, (int, float, bool)):
		text = str(tc_result)
	elif not isinstance(tc_result, str):
		text = json.dumps(tc_result)
	else:
		text = tc_result
	n = len(text)
	cap = tool_result_max
	try:
		cap = int(cap)
	except (TypeError, ValueError):
		cap = _MAX_TOOL_RESULT_CHARS
	if cap < 1:
		cap = _MAX_TOOL_RESULT_CHARS
	if n <= cap:
		return text
	note = f"\n\n... truncated at {cap} chars (tool result was {n} chars)"
	keep = max(0, cap - len(note))
	return text[:keep] + note


def _RunToolCallBlock(raw_inner, aiui_funcs, on_result=None):
	"""Permission-gate then run (start on each grant). Returns (display_raw, results, err_count).

	results: list of (name, content_str, is_error) in original call order.
	Tools still run in parallel; on_result(name, content, is_error, call_id) fires
	in call order (flush the ready prefix as later slots finish) so the transcript
	and OpenAI/Anthropic remappers never FIFO-attach a faster result to the wrong call.
	"""
	from concurrent.futures import ThreadPoolExecutor, as_completed
	from lib import tool_calls, tool_permissions
	from lib.tool_client import DispatchToolFunc

	calls = tool_calls.ParseToolCallInner(raw_inner)
	display_raw = tool_calls.SerializeToolCallInner(calls) if calls else (raw_inner or "")
	if not calls:
		content = "ERROR: unknown tool call format"
		if on_result:
			on_result("unknown", content, True)
		return display_raw or raw_inner or "", [("unknown", content, True)], 1

	def Emit(name, content, is_err, call_id=""):
		if on_result:
			on_result(name, content, is_err, call_id)

	# slots[i] = (name, result_or_None, future_or_None, is_error, call_id)
	slots = []
	futures = {}
	emitted = 0

	def FlushReady():
		nonlocal emitted
		while emitted < len(slots):
			name, result, fut, is_err, call_id = slots[emitted]
			if fut is not None:
				return
			Emit(name, result, is_err, call_id)
			emitted += 1

	max_workers = max(1, min(8, len(calls)))
	executor = ThreadPoolExecutor(max_workers=max_workers)
	try:
		for call in calls:
			name = call.get("name") or "unknown"
			args = call.get("arguments") or {}
			call_id = str(call.get("id") or "")
			if ipc.StopRequested():
				content = "Stopped before this tool ran."
				slots.append((name, content, None, True, call_id))
				FlushReady()
				continue
			if not tool_permissions.RequirePermission(name, args):
				content = "ERROR: tool call denied by user"
				slots.append((name, content, None, True, call_id))
				FlushReady()
				continue
			fut = executor.submit(DispatchToolFunc, name, args, aiui_funcs)
			slots.append((name, None, fut, False, call_id))
			futures[fut] = len(slots) - 1

		for fut in as_completed(futures):
			idx = futures[fut]
			name, _, _, _, call_id = slots[idx]
			try:
				result = fut.result()
			except Exception as e:
				result = f"ERROR: {e}"
			content = _ResultToToolContent(result)
			is_err = str(content).startswith("ERROR:")
			slots[idx] = (name, content, None, is_err, call_id)
			FlushReady()
	finally:
		executor.shutdown(wait=True)

	results = []
	err_count = 0
	for i, (name, result, fut, pre_err, call_id) in enumerate(slots):
		if fut is not None:
			try:
				result = _ResultToToolContent(fut.result())
			except Exception as e:
				result = f"ERROR: {e}"
			pre_err = str(result).startswith("ERROR:")
			slots[i] = (name, result, None, pre_err, call_id)
		content = slots[i][1] if isinstance(slots[i][1], str) else _ResultToToolContent(slots[i][1])
		is_err = slots[i][3] or str(content).startswith("ERROR:")
		if is_err:
			err_count += 1
		results.append((name, content, is_err))
	FlushReady()
	return display_raw, results, err_count

def _FormatTurnDisplay(cmb_response):
	"""Collapse think/tool tags for the chat UI + TTS."""
	cmb_response = StripEmptyThinkTags(cmb_response)
	tts_response = cmb_response
	think_tags = { "<thoughts>": "</thoughts>", "<think>": "</think>" }
	tool_tags = { "<tool_call>": "</tool_call>", "<tool_response>": "</tool_response>" }

	for xml_open_tag, xml_close_tag in think_tags.items():
		while xml_open_tag in cmb_response and xml_close_tag in cmb_response:
			tag_start = cmb_response.index(xml_open_tag) + len(xml_open_tag)
			tag_end = cmb_response.index(xml_close_tag)
			if tag_start >= tag_end:
				cmb_response = cmb_response.replace(xml_close_tag, xml_close_tag.replace('<', 'AI_UI_C60;', 1).replace('>', 'AI_UI_C62;', 1), 1)
				continue
			inner_txt = cmb_response[tag_start:tag_end]
			full = xml_open_tag+inner_txt+xml_close_tag
			tts_response = tts_response.replace(full, '', 1)
			cmb_response = cmb_response.replace(full, FormatThinkMarkers(inner_txt), 1)

	for xml_open_tag, xml_close_tag in tool_tags.items():
		while xml_open_tag in cmb_response and xml_close_tag in cmb_response:
			tag_start = cmb_response.index(xml_open_tag) + len(xml_open_tag)
			tag_end = cmb_response.index(xml_close_tag)
			if tag_start >= tag_end:
				cmb_response = cmb_response.replace(xml_close_tag, xml_close_tag.replace('<', 'AI_UI_C60;', 1).replace('>', 'AI_UI_C62;', 1), 1)
				continue
			inner_txt = cmb_response[tag_start:tag_end]
			tts_response = tts_response.replace(xml_open_tag+inner_txt+xml_close_tag, '', 1)
			if xml_open_tag == "<tool_call>":
				cmb_response = cmb_response.replace(
					xml_open_tag+inner_txt+xml_close_tag,
					FormatToolMarkers(inner_txt, "tc"),
					1,
				)
			else:
				cmb_response = cmb_response.replace(
					xml_open_tag+inner_txt+xml_close_tag,
					FormatToolMarkers(inner_txt, "tr"),
					1,
				)

	return StripEmptyThinkTags(cmb_response), tts_response

def _SendTurnPatch(turn_parts):
	partial = "\n\n".join([p for p in turn_parts if p]).strip()
	if partial == "":
		return
	display, _ = _FormatTurnDisplay(partial)
	ipc.SendEvent("turn_patch", {"text": display.replace("\n", "[AI_UI_BR]")})

def _CollapseAssistantRun(start_idx):
	"""Merge consecutive assistant messages from one bot turn into one entry.

	Skip when any message carries an Anthropic `_anthropic_content` or OpenRouter
	`_reasoning_details` sidecar - those must stay on the assistant turn that
	produced them so remappers can replay thinking/tool_use ids.
	"""
	global messages
	if start_idx < 0 or start_idx >= len(messages):
		return
	parts = []
	end = start_idx
	while end < len(messages) and messages[end].get("role") in (bot_name, "assistant"):
		content = messages[end].get("content")
		if isinstance(content, str) and content.strip():
			parts.append(content)
		end += 1
	if len(parts) <= 1:
		return
	for m in messages[start_idx:end]:
		if isinstance(m, dict) and (
			m.get("_anthropic_content") is not None
			or m.get("_reasoning_details") is not None
		):
			return
	combined = "\n\n".join(parts)
	msg = {"role": bot_name, "content": combined}
	earliest_ts = None
	for m in messages[start_idx:end]:
		if not isinstance(m, dict):
			continue
		ts = m.get("timestamp")
		if ts and (earliest_ts is None or str(ts) < str(earliest_ts)):
			earliest_ts = ts
	if earliest_ts:
		msg["timestamp"] = earliest_ts
	else:
		chat.StampMsg(msg)
	messages[start_idx:end] = [msg]

def _AttachRemoteSidecars(msg):
	"""Copy provider thinking blobs onto the assistant message that just finished."""
	global _anthropic_last_content, _openai_last_reasoning_details, _openai_last_reasoning_pin
	if _anthropic_last_content is not None:
		msg["_anthropic_content"] = _anthropic_last_content
		_anthropic_last_content = None
	if _openai_last_reasoning_details is not None:
		msg["_reasoning_details"] = _openai_last_reasoning_details
		_openai_last_reasoning_details = None
	if _openai_last_reasoning_pin is not None:
		msg["_reasoning_pin"] = _openai_last_reasoning_pin
		_openai_last_reasoning_pin = None

def _RunBotTurn():
	"""Generate, early-stop on tool calls, run tools, continue until a final answer."""
	global prompt, messages, responses, redo_count, rando_lvl, chat_files, r, model
	global _anthropic_last_content, _openai_last_reasoning_details, _openai_last_reasoning_pin
	prompt += "\n"+bot_name_c
	rando_lvl -= rando_sub
	temp = min(1.0, max(rando_lvl, rando_min))

	redo_count = 0
	r = 0
	tc_errors = 0
	turn_parts = []
	max_rounds = 24
	aiui_funcs = { "GenImageIPA": GenImageIPA, "GenImage": GenImage }
	stream_open = False
	# Compact once per turn (before any generate / tool round).
	try:
		context_compact_mod.RunCompactIfNeeded()
	except Exception as e:
		ipc.LogException("context_compact", e)
	# After compact notice may have been appended — collapse only this turn's bot msgs.
	turn_msg_start = len(messages)
	ipc.SendEvent("turn_start", {})

	def _LiveBlockPatch(partial):
		"""Show closed think/tool blocks while a GenReply is still running."""
		partial = (partial or "").strip()
		if not partial:
			return
		_SendTurnPatch(turn_parts + [partial])

	try:
		for _round in range(max_rounds):
			if ipc.StopRequested():
				break

			if model is None and not text_backend_mod.IsRemoteText():
				LoadModel()
			in_tokens = None if text_backend_mod.IsRemoteText() else TokenizeChat()
			# Token streaming follows gen_mode. Tool turns still use an internal streamer
			# when streaming is off so early-stop works and closed XML blocks can patch the UI.
			want_ui_stream = gen_mode != 0
			want_stream = want_ui_stream or bool(enable_tooluse)
			cmb_response = _NormalizeBotText(GenReply(
				in_tokens, temp,
				stream=want_stream,
				reuse_stream=stream_open and want_stream,
				ui_stream=want_ui_stream,
				on_block=_LiveBlockPatch if (want_stream and not want_ui_stream) else None,
			))
			cmb_response = StripEmptyThinkTags(cmb_response)
			if want_stream:
				stream_open = True
			if cmb_response == "":
				try:
					context_compact_mod.EmitContextUsageAfterGeneration()
				except Exception as e:
					ipc.LogException("context_usage after generation", e)
				break

			has_tool, preface, call_items, leftover = _ParseToolCalls(cmb_response)

			# Final assistant text (no tool call) — append and finish.
			if (not enable_tooluse) or (not has_tool):
				msg = chat.StampMsg({"role": bot_name, "content": cmb_response})
				_AttachRemoteSidecars(msg)
				messages.append(msg)
				turn_parts.append(cmb_response)
				_SendTurnPatch(turn_parts)
				try:
					context_compact_mod.EmitContextUsageAfterGeneration()
				except Exception as e:
					ipc.LogException("context_usage after generation", e)
				break

			pending_preface = ""
			if preface:
				preface = StripEmptyThinkTags(preface)
				if preface:
					turn_parts.append(preface)
					pending_preface = preface
					_SendTurnPatch(turn_parts)

			# Run every complete tool call block from this generation (early-stop usually
			# yields one; multiple are handled without dropping later calls).
			for item in call_items:
				if ipc.StopRequested():
					break
				if item.get("kind") == "text":
					text_part = StripEmptyThinkTags(item.get("text") or "")
					if not text_part:
						continue
					stored = text_part
					if pending_preface:
						stored = pending_preface + "\n\n" + text_part
						pending_preface = ""
					turn_parts.append(text_part)
					messages.append(chat.StampMsg({"role": bot_name, "content": stored}))
					continue

				raw_inner = item.get("raw") or item.get("json") or ""

				def _OnToolResult(tool_name, tc_result, is_err, call_id=""):
					payload = {"name": tool_name, "content": tc_result}
					if call_id:
						payload["id"] = call_id
					if is_err:
						payload["error"] = True
					tr_data = json.dumps(payload)
					messages.append(chat.StampMsg({
						"role": bot_name,
						"content": "<tool_response>"+tr_data+"</tool_response>",
					}))
					turn_parts.append("<tool_response>" + SafeString(tr_data) + "</tool_response>")
					_SendTurnPatch(turn_parts)
					try:
						context_compact_mod.EmitContextUsageAfterTools()
					except Exception as e:
						ipc.LogException("context_usage after tool", e)

				# Show the call block first, then append responses in call order.
				from lib import tool_calls as _tc
				_preview = _tc.ParseToolCallInner(raw_inner)
				display_raw = _tc.SerializeToolCallInner(_preview) if _preview else (raw_inner or "")
				call_xml = "<tool_call>" + display_raw + "</tool_call>"
				body = call_xml
				if pending_preface:
					body = pending_preface + "\n\n" + call_xml
					pending_preface = ""
				turn_parts.append("<tool_call>" + SafeString(display_raw) + "</tool_call>")
				_SendTurnPatch(turn_parts)
				tc_msg = chat.StampMsg({"role": bot_name, "content": body})
				_AttachRemoteSidecars(tc_msg)
				messages.append(tc_msg)
				try:
					context_compact_mod.EmitContextUsageAfterGeneration()
				except Exception as e:
					ipc.LogException("context_usage after generation", e)

				_display2, _results, block_errs = _RunToolCallBlock(
					raw_inner, aiui_funcs, on_result=_OnToolResult,
				)
				tc_errors += block_errs

				if tc_errors > 4:
					turn_parts.append("ERROR: bad tool calls")
					break

			if pending_preface:
				# No tool_call/text item consumed the think preface (stopped or empty).
				preface_msg = chat.StampMsg({"role": bot_name, "content": pending_preface})
				_AttachRemoteSidecars(preface_msg)
				messages.append(preface_msg)
				pending_preface = ""

			if tc_errors > 4:
				break

			# Text after the last tool call in the same generation (rare with early-stop).
			if leftover:
				leftover = StripEmptyThinkTags(leftover)
				if leftover:
					messages.append(chat.StampMsg({"role": bot_name, "content": leftover}))
					turn_parts.append(leftover)
					_SendTurnPatch(turn_parts)
					break
			# Continue generating so the model sees the tool response(s) in context.

		cmb_response = "\n\n".join([p for p in turn_parts if p]).strip()
		_CollapseAssistantRun(turn_msg_start)
		# Only messages added after compact (turn_msg_start..) are redo-able —
		# never delete the compact notice that was appended before the turn.
		redo_count = max(0, len(messages) - turn_msg_start)
		if cmb_response == "":
			ipc.SendEvent("turn_end", {})
			PrintSleep("BOT_NOANIM:")
			chat_files = []
			PrunePrompt()
			return

		cmb_response, tts_response = _FormatTurnDisplay(cmb_response)
		anim_done = False

		if do_talk and tts_ready and tts_response != '' and not ipc.StopRequested():
			try:
				soft = int(audio_max_chars or 0)
				if soft > 0 and len(tts_response) > soft:
					ipc.LogErr(
						f"WARN: chat TTS skipped "
						f"(text {len(tts_response)} chars exceeds audio_max_chars={soft})"
					)
				else:
					def _DoTts():
						if tts_mode == 0:
							TextToSpeechSys(NormalizeText(tts_response), chat_wav, voice_ps)
						else:
							TextToSpeechAI(tts_response, chat_wav, speaker_embeddings, voice_vol, voice_rate, voice_ps)

					tts_ok, _ = chat.RunStoppable(_DoTts)
					if tts_ok and do_anim and not ipc.StopRequested():
						anim_ok, anim_res = chat.RunStoppable(AnimAvatar)
						anim_done = bool(anim_ok and anim_res)
					# On STOP, skip playback of possibly truncated audio/video.
					if not ipc.StopRequested() and not (do_anim and anim_done):
						PrintSleep('PLAY_SPEECH:'+chat_wav)
			except Exception as e:
				ipc.LogException("chat_tts", e)
				anim_done = False

		# On STOP (or soft abort) still apply the formatted partial so the UI keeps it.
		# Empty bot events used to re-enable SEND while leaving streamed DOM alone, but
		# the streaming stub kept class `hidden` and the empty handler removed it.
		ipc.SendEvent("turn_end", {})
		if ipc.StopRequested():
			PrintSleep("BOT_NOANIM:" + (cmb_response.replace("\n", "[AI_UI_BR]") if cmb_response else ""))
		elif do_anim and anim_done:
			PrintSleep("BOT_OUTPUT:"+cmb_response.replace("\n", "[AI_UI_BR]"))
		else:
			PrintSleep("BOT_NOANIM:"+cmb_response.replace("\n", "[AI_UI_BR]"))

		chat_files = []
		PrunePrompt()

	except Exception as e:
		ipc.LogException("_RunBotTurn", e)
		ipc.SendEvent("turn_end", {})
		err = f"Generation failed: {e}"
		PrintSleep("BOT_NOANIM:" + err.replace("\n", "[AI_UI_BR]"))
		chat_files = []
		PrunePrompt()

def _SyncTemplateFlags():
	global use_chat_template, use_aiui_template, aiui_tool_instructions
	use_chat_template = True if temp_form != "none" else False
	use_aiui_template = temp_form in ("aiui", "aiui_notoolprompt")
	aiui_tool_instructions = temp_form == "aiui"

def _ApplyTemplateFormat(raw):
	global temp_form, tokenizer
	temp_form = _NormTempForm(raw)
	_SyncTemplateFlags()
	if tokenizer is not None:
		SyncTokenizerTemplate()
	GetUserNames()

def _ApplyAttachedFiles(p):
	"""Replace E.chat_files from cmd params (paths / URLs). Also updates mm_processor."""
	global chat_files, mm_processor
	from lib.media_types import ClassifyPath, NormMmProcessor, IsHttpUrl, NormalizeAttachRef

	file_names = p.get("files") or []
	if "mm_processor" in p:
		mm_processor = NormMmProcessor(p.get("mm_processor"))
	if isinstance(file_names, str):
		file_names = file_names.replace("\\", "/").split("[AI_UI_BR]")
	chat_files = []
	for file_name in file_names:
		file_name = NormalizeAttachRef(str(file_name).replace("\\", "/"))
		if file_name.strip() == '':
			continue
		file_type = ClassifyPath(file_name)
		if IsHttpUrl(file_name):
			# Remote ref - never try to read as a local text file.
			chat_files.append({"file": file_name, "type": file_type, "content": "url"})
			continue
		file_text = None
		if file_type == "text":
			try:
				file_text = ReadTextFile(file_name)
			except Exception:
				file_text = None
				file_type = "file"
		if file_text:
			chat_files.append({"file": file_name, "type": "text", "content": file_text})
		elif os.path.isfile(file_name):
			kind = file_type if file_type != "text" else "file"
			chat_files.append({"file": file_name, "type": kind, "content": kind})
		else:
			# Non-existent local path / bare string - keep as text so it is not dropped.
			chat_files.append({"file": file_name, "type": "text", "content": file_name})


def _MmContentPart(chat_file):
	"""Build one multimodal content part from a chat_files entry."""
	kind = str(chat_file.get("type") or "file")
	src = chat_file.get("file")
	payload = chat_file.get("content")
	if kind == "text":
		text = payload if isinstance(payload, str) and payload != "url" else src
		if text is None or text == "":
			return None
		return {"type": "text", "text": str(text)}
	if payload == "url" or (isinstance(src, str) and (src.startswith("http://") or src.startswith("https://"))):
		# Qwen-style keys accept http(s) URLs in the media field.
		if kind == "image":
			return {"type": "image", "image": src}
		if kind == "video":
			return {"type": "video", "video": src}
		if kind == "audio":
			return {"type": "audio", "audio": src}
		# Webpage / unknown URL - pass through as text so the model still sees it.
		return {"type": "text", "text": str(src)}
	if kind in ("image", "video", "audio") and src:
		return {"type": kind, kind: src}
	if src:
		return {"type": "text", "text": str(src)}
	return None


def HandleCmd(msg):
	global init_prompt, messages, prompt, chat_files
	global voice_key, voice_vol, voice_rate, voice_ps, talk_mode, tts_mode, anim_mode, avatar_img
	global msg_mem, gen_max, tool_result_max, gen_min, do_sample, num_beams, beam_groups
	global compact_at, compact_summary_max, git_status_context
	global rando_lvl, prompt_p, top_k, top_p, typical_p, rep_penalty, frequency_penalty, presence_penalty, rando_min
	global reasoning_effort, mm_processor
	global gen_mode, format_mode, enable_tooluse, hf_access_token
	global text_backend, remote_base_url, remote_model, remote_api_key, remote_provider, remote_tool_format, remote_prompt_cache
	global temp_form, tool_mode, tool_packs, firecrawl_api_key
	global use_chat_template, use_aiui_template
	global avatar_img, img_gen_mode, redo_count
	global active_agent, enabled_rules, enabled_skills, workspace_dir
	global last_usage_stale

	name = msg.get("name") or ""
	p = _CmdParams(msg)

	if name == "close_chat":
		return "exit"
	elif name == "print_prompt":
		ipc.LogErr('----- START PROMPT -----')
		ipc.LogErr(prompt)
		ipc.LogErr('------ END PROMPT ------')
		ipc.SendEvent("prompt_dump", {"text": prompt})
	elif name == "compose_prompt":
		req_agent = str(p.get("agent", active_agent) or "")
		req_rules = p.get("enabled_rules") if "enabled_rules" in p else enabled_rules
		req_skills = p.get("enabled_skills") if "enabled_skills" in p else enabled_skills
		req_ws = workspace_dir
		if "workspace" in p:
			req_ws = str(p.get("workspace", "") or "").rstrip('/\\')
		apply = p.get("apply", 1) not in (0, False, "0", "false")
		save = p.get("save", 0) not in (0, False, "0", "false") or apply
		snap = prompt_files.SendPromptFilesEvent(
			aiui_home, req_ws, req_agent, req_rules, req_skills, apply=apply,
		)
		if save:
			active_agent = snap.get("agent", "")
			enabled_rules = snap.get("enabled_rules")
			enabled_skills = snap.get("enabled_skills")
			if "workspace" in p:
				workspace_dir = req_ws
		if apply and prompt_p < 2:
			chat.ApplySystemPrompt(snap.get("composed") or "")
			PrunePrompt()
	elif name == "update_prompt":
		init_prompt = str(p.get("prompt", p.get("init_prompt", ""))).replace("[AI_UI_BR]", "\n").replace("\r", '')
		if prompt_p < 2:
			PrunePrompt()
			old = ""
			if len(messages) > 0 and str(messages[0].get("role") or "").lower() == "system":
				old = str(messages[0].get("content") or "")
			if len(messages) > 0:
				messages[0] = chat.StampMsg({"role":"system", "content":init_prompt})
			else:
				messages = [chat.StampMsg({"role":"system", "content":init_prompt})]
			if old != init_prompt:
				remote_tools_mod.WarnIfAnthropicPrefixMayBreak("system prompt")
		PrunePrompt()
	elif name == "clear_chat":
		InitPrompt(True)
		rando_lvl = rando_min
	elif name == "compact_now":
		try:
			ok, err = context_compact_mod.RunCompactForced()
			if not ok:
				ipc.SendEvent("compact", {"ok": False, "error": err or "Compact failed."})
		except Exception as e:
			ipc.LogException("compact_now", e)
			ipc.SendEvent("compact", {"ok": False, "error": f"Compact failed: {e}"})
	elif name == "export_chat":
		ipc.SendEvent("chat_export", {
			"messages": messages,
			"init_prompt": init_prompt,
			"compaction": context_compact_mod.GetCompactState(),
		})
	elif name == "import_chat":
		raw = p.get("messages")
		if not isinstance(raw, list):
			raw = []
		messages = []
		for m in raw:
			if isinstance(m, dict) and isinstance(m.get("content"), str):
				m = dict(m)
				m["content"] = m["content"].replace("[AI_UI_BR]", "\n")
			if isinstance(m, dict):
				chat.StampMsg(m)
			messages.append(m)
		if len(messages) > 0 and isinstance(messages[0], dict) and messages[0].get("role") == "system":
			init_prompt = str(messages[0].get("content") or "")
		elif "init_prompt" in p:
			init_prompt = str(p.get("init_prompt") or "").replace("[AI_UI_BR]", "\n")
			if init_prompt != "":
				messages = [chat.StampMsg({"role": "system", "content": init_prompt})] + [
					m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")
				]
		if "compaction" in p and isinstance(p.get("compaction"), dict):
			context_compact_mod.ApplyCompactState(p.get("compaction"))
		else:
			context_compact_mod.ResetCompactState()
		PrunePrompt()
		redo_count = chat.DeriveRedoCount(messages)
		ipc.SendEvent("chat_imported", {
			"messages": messages,
			"init_prompt": init_prompt,
			"compaction": context_compact_mod.GetCompactState(),
		})
		# Response usage from the previous session no longer describes this one.
		last_usage_stale = True
		try:
			context_compact_mod.EmitContextUsage(heuristic=True)
		except Exception as e:
			ipc.LogException("context_usage after import_chat", e)
	elif name == "attach_files":
		_ApplyAttachedFiles(p)
	elif name == "config_voice":
		last_amode = anim_mode
		last_avatar = avatar_img
		voice_key = int(p.get("voice_key", p.get("bot_voice", voice_key)))
		voice_vol = float(p.get("voice_vol", p.get("speech_vol", voice_vol)))
		voice_rate = int(p.get("voice_rate", p.get("speech_rate", voice_rate)))
		voice_ps = int(p.get("voice_ps", p.get("pitch_shift", voice_ps)))
		talk_mode = int(p.get("talk_mode", talk_mode))
		tts_mode = int(p.get("tts_mode", tts_mode))
		anim_mode = int(p.get("anim_mode", anim_mode))
		if p.get("avatar_img"):
			avatar_img = str(p.get("avatar_img"))
		SetWorkingDir(anim_mode)
		ApplyTalkMode()
		if anim_mode != last_amode or avatar_img != last_avatar:
			if LoadAvatar():
				PrintSleep("GOT_AVATAR:true")
			else:
				PrintSleep("GOT_AVATAR:false")
		ConfigTTSEngine()
	elif name == "config_ai":
		last_pp = prompt_p
		msg_mem = int(p.get("msg_mem", msg_mem))
		if "compact_at" in p:
			prev_ca = compact_at
			compact_at = int(p.get("compact_at", compact_at))
			if prev_ca > 0 and compact_at <= 0:
				context_compact_mod.OnCompactEnabledChanged(False)
			elif prev_ca <= 0 and compact_at > 0:
				context_compact_mod.OnCompactEnabledChanged(True)
		if "compact_summary_max" in p:
			compact_summary_max = int(p.get("compact_summary_max", compact_summary_max))
		if "git_status_context" in p:
			git_status_context = int(p.get("git_status_context", git_status_context) or 0)
		gen_max = int(p.get("gen_max", p.get("max_res", gen_max)))
		if "max_tool_result" in p or "tool_result_max" in p:
			try:
				n = int(p.get("max_tool_result", p.get("tool_result_max", tool_result_max)))
			except (TypeError, ValueError):
				n = tool_result_max
			tool_result_max = n if n >= 1 else 4_000_000
		gen_min = int(p.get("gen_min", p.get("min_res", gen_min)))
		do_sample = int(p.get("do_sample", do_sample))
		num_beams = int(p.get("num_beams", num_beams))
		beam_groups = int(p.get("beam_groups", beam_groups))
		rando_lvl = float(p.get("rando_lvl", p.get("base_temp", rando_lvl)))
		prompt_p = int(p.get("prompt_p", prompt_p))
		top_k = int(p.get("top_k", top_k))
		top_p = float(p.get("top_p", top_p))
		typical_p = float(p.get("typical_p", typical_p))
		rep_penalty = float(p.get("rep_penalty", rep_penalty))
		if "frequency_penalty" in p or "fp" in p:
			frequency_penalty = float(p.get("frequency_penalty", p.get("fp", frequency_penalty)))
		if "presence_penalty" in p or "prp" in p:
			presence_penalty = float(p.get("presence_penalty", p.get("prp", presence_penalty)))
		rando_min = rando_lvl
		if "reasoning_effort" in p or "enable_thinking" in p:
			reasoning_effort = MergeReasoningEffort(
				p.get("reasoning_effort", reasoning_effort),
				p.get("enable_thinking") if "enable_thinking" in p else None,
			)
		# Settings Apply always wins over boot/model_args locks (use_tool_config).
		if "temp_form" in p or "temp_format" in p:
			_ApplyTemplateFormat(p.get("temp_form", p.get("temp_format", temp_form)))
		if prompt_p != last_pp:
			if init_prompt == 'No prompt': InitPrompt()
			if last_pp == 0:
				if len(messages) > 0 and messages[0]["role"] == "system":
					messages = messages[1::]
			if prompt_p == 0:
				if len(messages) > 0:
					messages[0] = chat.StampMsg({"role":"system", "content":init_prompt})
				else:
					messages = [chat.StampMsg({"role":"system", "content":init_prompt})]
			remote_tools_mod.WarnIfAnthropicPrefixMayBreak("system prompt")
		PrunePrompt()
	elif name == "config_other":
		gen_mode = int(p.get("gen_mode", gen_mode))
		format_mode = int(p.get("format_mode", format_mode))
		enable_tools = int(p.get("enable_tooluse", p.get("enable_tools", enable_tooluse)))
		if enable_tools != enable_tooluse:
			enable_tooluse = enable_tools
			LoadToolFuncs()
			try:
				context_compact_mod.EmitContextUsage(heuristic=True)
			except Exception as e:
				ipc.LogException("context_usage after enable_tooluse", e)
	elif name == "config_remote":
		text_backend = "remote" if str(p.get("text_backend", text_backend) or "local").lower() == "remote" else "local"
		remote_base_url = str(p.get("remote_base_url", remote_base_url) or "")
		remote_model = str(p.get("remote_model", remote_model) or "")
		remote_api_key = str(p.get("remote_api_key", remote_api_key) or "")
		remote_provider = str(p.get("remote_provider", remote_provider) or "")
		_rtf = str(p.get("remote_tool_format", remote_tool_format) or "openai").strip().lower()
		remote_tool_format = _rtf if _rtf == "anthropic" else "openai"
		if "remote_prompt_cache" in p:
			remote_prompt_cache = 0 if int(p.get("remote_prompt_cache", 1) or 0) == 0 else 1
		hf_token = str(p.get("hf_token", hf_access_token))
		if hf_token != hf_access_token:
			hf_access_token = hf_token
			if hf_token != '':
				login(token=hf_token)
		# Usage ring remote/local flag + totals should match the new backend immediately.
		last_usage_stale = True
		try:
			context_compact_mod.EmitContextUsage(heuristic=True)
		except Exception as e:
			ipc.LogException("context_usage after config_remote", e)
	elif name == "config_tools":
		if "workspace" in p:
			workspace_dir = str(p.get("workspace", "") or "").rstrip("/\\")
		if "temp_form" in p or "temp_format" in p:
			_ApplyTemplateFormat(p.get("temp_form", p.get("temp_format", temp_form)))
		else:
			temp_form = _NormTempForm(temp_form)
		if "tool_mode" in p or "tools_file" in p:
			tool_mode = str(p.get("tool_mode", p.get("tools_file", tool_mode)))
		if "tool_packs" in p:
			tool_packs = tool_client.NormalizeToolPacks(p.get("tool_packs"), tool_mode)
		if "firecrawl_api_key" in p:
			firecrawl_api_key = str(p.get("firecrawl_api_key", "") or "")
		# Optional force reconnect for specific MCP packs after settings edits.
		force_ids = p.get("force_mcp_ids") or []
		if isinstance(force_ids, str):
			force_ids = [force_ids] if force_ids.strip() else []
		LoadToolFuncs(force_mcp_ids=force_ids or None)
		# Tool schemas affect the context budget; refresh the usage ring.
		try:
			context_compact_mod.EmitContextUsage(heuristic=True)
		except Exception as e:
			ipc.LogException("context_usage after config_tools", e)
	elif name == "gen_image":
		img_prompt = str(p.get("img_prompt", p.get("prompt_text", "")))
		neg_prompt = str(p.get("neg_prompt", p.get("prompt_neg", "")))
		infer_steps = int(p.get("infer_steps", p.get("inference_steps", 50)))
		guidance = float(p.get("guidance", p.get("guidance_scale", 7.5)))
		safety_check = int(p.get("safety_check", 1))
		img_seed = str(p.get("img_seed", p.get("image_seed", "")))
		img_width = str(p.get("img_width", p.get("image_width", "auto")))
		img_height = str(p.get("img_height", p.get("image_height", "auto")))
		vae_file = str(p.get("vae_file", ""))
		lora_file = str(p.get("lora_file", ""))
		lora_dir = str(p.get("lora_dir", ""))
		lora_scale = float(p.get("lora_scale", 1.0))
		safety_check = True if safety_check > 0 else False
		diff_mode = DiffIMGMode(safety_check, vae_file, lora_file, lora_dir, lora_scale)
		if im_model == None or diff_mode == 2 or (img_gen_mode == False and diff_mode == 1):
			if LoadIMGModel(False, safety_check, vae_file, lora_file, lora_dir, lora_scale):
				img_gen_mode = True
		if im_model != None:
			img_file = GenImage(img_prompt, neg_prompt, infer_steps, guidance, img_seed, img_width, img_height, True)
			PrintSleep("IMG_OUTPUT:"+img_file)
	elif name == "gen_text":
		start_txt = str(p.get("text", p.get("prompt_text", ""))).replace("\r", '')
		gmax = int(p.get("max_len", p.get("gmax", 1000)))
		gmin = int(p.get("min_len", p.get("gmin", 1)))
		dsmp = int(p.get("do_sample", 1))
		numb = int(p.get("num_beams", 1))
		numg = int(p.get("beam_groups", 1))
		temp = float(p.get("temp", 0.8))
		topk = int(p.get("top_k", 50))
		topp = float(p.get("top_p", 1.0))
		typp = float(p.get("typical_p", 1.0))
		repp = float(p.get("rep_penalty", 1.0))
		if tokenizer == None: LoadTokenizer()
		try:
			in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids.to(comp_dev)
			text = GenText(in_tokens, gmin, gmax, dsmp, numb, numg, temp, topk, topp, typp, repp)
			# On STOP keep streamed tokens; empty gen event still restores GENERATE.
			if ipc.StopRequested():
				PrintSleep("GEN_OUTPUT:")
			else:
				PrintSleep("GEN_OUTPUT:"+text.replace("\n", "[AI_UI_BR]"))
		except Exception as e:
			ipc.LogException("gen_text", e)
			PrintSleep("GEN_OUTPUT:ERROR: text gen failed, model may only work as chat bot")
	elif name == "speech_recog":
		try:
			text = SpeechToText(tempfile.gettempdir()+"/recording.wav").strip()
			PrintSleep("ASR_OUTPUT:"+text.replace("\n", " "))
		except Exception as e:
			ipc.LogException("speech_recog", e)
			PrintSleep("ASR_OUTPUT:ERROR")
	elif name == "gen_speech":
		if not tts_ready:
			PrintSleep('TTS_OUTPUT:NO_VOICES')
		else:
			tts_text = str(p.get("text", p.get("tts_text", ""))).replace("\r", '')
			vckey = int(p.get("voice_key", p.get("tts_voice", 0)))
			vcvol = float(p.get("voice_vol", p.get("tts_vol", 1.0)))
			vcrate = int(p.get("voice_rate", p.get("tts_rate", 200)))
			vcpitch = int(p.get("voice_ps", p.get("tts_pitch", 0)))
			ttsmode = int(p.get("tts_mode", 0))
			try:
				if ttsmode == 0:
					synthesizer.setProperty('voice', voices[vckey].id)
					synthesizer.setProperty('volume', vcvol)
					synthesizer.setProperty('rate', vcrate)
					TextToSpeechSys(NormalizeText(tts_text), speech_wav, vcpitch)
					synthesizer.setProperty('voice', voices[voice_key].id)
					synthesizer.setProperty('volume', voice_vol)
					synthesizer.setProperty('rate', voice_rate)
					PrintSleep('TTS_OUTPUT:'+speech_wav)
				else:
					spkr_embs = LoadSpeaker(vckey)
					TextToSpeechAI(tts_text, speech_wav, spkr_embs, vcvol, vcrate, vcpitch, vckey)
					PrintSleep('TTS_OUTPUT:'+speech_wav)
			except Exception as e:
				if tts.IsTtsLengthSkip(e):
					PrintSleep('TTS_OUTPUT:TOO_LONG')
				else:
					ipc.LogException("gen_speech", e)
					PrintSleep("TTS_OUTPUT:ERROR")
	elif name == "read_text":
		if not tts_ready:
			PrintSleep('PLAY_SPEECH:NO_VOICES')
		else:
			tts_text = str(p.get("text", p.get("tts_text", ""))).replace("\r", '')
			try:
				if tts_mode == 0:
					TextToSpeechSys(NormalizeText(tts_text), read_wav, voice_ps)
				else:
					TextToSpeechAI(tts_text, read_wav, speaker_embeddings, voice_vol, voice_rate, voice_ps)
				PrintSleep('PLAY_SPEECH:'+read_wav)
			except Exception as e:
				if tts.IsTtsLengthSkip(e):
					PrintSleep('PLAY_SPEECH:TOO_LONG')
				else:
					ipc.LogException("read_text", e)
					PrintSleep('PLAY_SPEECH:ERROR')
	elif name == "clone_voice":
		clone_model = int(p.get("clone_model", p.get("model", 0)))
		speaker_name = str(p.get("speaker_name", p.get("name", ""))).strip()
		speaker_wav = str(p.get("speaker_wav", p.get("sample", ""))).strip()
		transcript = str(p.get("transcript", "")).strip()
		CleanVRAM()
		PrintSleep('CLONE_OUTPUT:'+CloneVoice(clone_model, speaker_name, speaker_wav, transcript))
		CleanVRAM()
		ReserveVRAM()
	elif name == "update_voices":
		LoadAIVoices()
	elif name == "update_avatar":
		avatar_img = str(p.get("avatar_img", ""))
		if LoadAvatar():
			PrintSleep("GOT_AVATAR:true")
		else:
			PrintSleep("GOT_AVATAR:false")
	elif name == "update_tmode":
		talk_mode = int(p.get("talk_mode", talk_mode))
		ApplyTalkMode()
	elif name == "update_users":
		GetUserNames()
		InitPrompt()
	elif name == "redo_last":
		new_text = str(p.get("text", "")).replace("[AI_UI_BR]", "\n").replace("\r", "").strip()
		if new_text:
			edited = edit_last_mod.ApplyEditLastUser(messages, new_text, user_name)
			if edited is None:
				PrintSleep("BOT_NOANIM:Unable to edit last message.")
			else:
				messages = edited
				last = messages[-1] if messages else None
				if isinstance(last, dict):
					last.pop(git_context_mod.GIT_STATUS_KEY, None)
					git_context_mod.StampUserGitStatus(last)
				PrunePrompt()
				rando_lvl += rando_add
				_RunBotTurn()
		else:
			# Prefer live transcript over redo_count: STOP/empty clears redo_count to 0
			# even when a user turn is still waiting for a reply.
			n = chat.DeriveRedoCount(messages)
			if n == 0 and not chat.LastTurnAwaitingReply(messages):
				PrintSleep("BOT_NOANIM:Unable to redo last response.")
			else:
				if n > 0:
					if n < len(messages):
						messages = messages[:-n]
					elif len(messages) > 0:
						messages = [messages[0]]
				PrunePrompt()
				rando_lvl += rando_add
				_RunBotTurn()
	elif name == "cont_chat":
		if use_chat_template:
			messages.append(git_context_mod.StampUserGitStatus(
				chat.StampMsg({"role":user_name, "content":"continue"})
			))
		_RunBotTurn()
	elif name == "chat":
		# Optional files on the same cmd — avoid a prior attach_files that would
		# emit idle and flip the UI out of STOP before generation starts.
		if "files" in p:
			_ApplyAttachedFiles(p)
		msg_text = str(p.get("text", "")).replace("[AI_UI_BR]", "\n").strip(" \n").replace("\r", '')
		if msg_text == '':
			return None
		mm_msg = None
		if len(chat_files) > 0:
			prompt += "\n"+user_name+roles_append_str
			mm_msg = chat.StampMsg({ "role": user_name, "content": [] })
			for chat_file in chat_files:
				prompt += chat_file["file"] + "\n"
				part = _MmContentPart(chat_file)
				if part is not None:
					mm_msg["content"].append(part)
			prompt = prompt.rstrip("\n")
			if not mm_msg["content"]:
				mm_msg = None

		msgs = msg_text.split("[AIUI_END]")
		for msg_txt in msgs:
			msg_one = msg_txt.strip(" \n")
			if msg_one != '':
				prompt += "\n"+user_name+roles_append_str+msg_one
				if mm_msg == None:
					messages.append(git_context_mod.StampUserGitStatus(
						chat.StampMsg({"role": user_name, "content": msg_one})
					))
				else:
					mm_msg["content"].append({"type": "text", "text": msg_one, "user":True})

		if mm_msg != None:
			git_context_mod.StampUserGitStatus(mm_msg)
			messages.append(mm_msg)
		_RunBotTurn()
	else:
		ipc.LogErr(f"Unknown cmd: {name}")
	return None

while True:
	ipc.SendIdle()
	msg = ipc.Recv()
	if msg is None:
		continue
	mtype = msg.get("type")
	# Leftover stop from a finished turn — ignore quietly
	if mtype == "control":
		continue
	if mtype != "cmd":
		ipc.LogErr(f"Ignoring non-cmd message in main loop: {mtype}")
		continue
	# Fresh command: clear any STOP left from the previous turn
	ipc.ClearStop()
	try:
		result = HandleCmd(msg)
		if result == "exit":
			break
	except Exception as e:
		ipc.LogException("HandleCmd", e)
