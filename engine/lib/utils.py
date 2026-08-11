# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

import gc
import sys
import time
from lib import ipc
from lib.local_ml import EnsureLocalMl, TorchAvailable, MissingLocalMl


def _EmitLegacyBanner(txt_str):
	"""Map old stdout banners to NDJSON events. Returns True if handled."""
	if txt_str.startswith("BOT_NOANIM:"):
		ipc.SendEvent("bot", {"anim": False, "text": txt_str[len("BOT_NOANIM:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("BOT_OUTPUT:"):
		ipc.SendEvent("bot", {"anim": True, "text": txt_str[len("BOT_OUTPUT:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("GEN_OUTPUT:"):
		ipc.SendEvent("gen", {"text": txt_str[len("GEN_OUTPUT:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("IMG_OUTPUT:"):
		ipc.SendEvent("img", {"path": txt_str[len("IMG_OUTPUT:"):]})
		return True
	if txt_str.startswith("CLONE_OUTPUT:"):
		ipc.SendEvent("clone", {"result": txt_str[len("CLONE_OUTPUT:"):]})
		return True
	if txt_str.startswith("ASR_OUTPUT:"):
		ipc.SendEvent("asr", {"text": txt_str[len("ASR_OUTPUT:"):]})
		return True
	if txt_str.startswith("TTS_OUTPUT:"):
		ipc.SendEvent("tts", {"path": txt_str[len("TTS_OUTPUT:"):]})
		return True
	if txt_str.startswith("PLAY_SPEECH:"):
		ipc.SendEvent("play_speech", {"path": txt_str[len("PLAY_SPEECH:"):].strip()})
		return True
	if txt_str.startswith("GOT_AVATAR:"):
		ok = txt_str[len("GOT_AVATAR:"):].strip().lower() == "true"
		ipc.SendEvent("avatar", {"ok": ok})
		return True
	if txt_str.startswith("INIT_DONE:"):
		ipc.SendEvent("prompt", {"kind": "init", "text": txt_str[len("INIT_DONE:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("CLEAR_DONE:"):
		ipc.SendEvent("prompt", {"kind": "clear", "text": txt_str[len("CLEAR_DONE:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("TOOL_FUNCS:"):
		ipc.SendEvent("tools", {"data": txt_str[len("TOOL_FUNCS:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("TOOL_CATALOG:"):
		ipc.SendEvent("tool_catalog", {"data": txt_str[len("TOOL_CATALOG:"):].replace("[AI_UI_BR]", "\n")})
		return True
	if txt_str.startswith("SYS_VOICES:") or txt_str.startswith("AI_VOICES:"):
		mode = "SYS" if txt_str.startswith("SYS_VOICES:") else "AI"
		raw = txt_str.split(":", 1)[1]
		voices = []
		for line in raw.split("\n"):
			line = line.strip()
			if "VOICE_NAME:" not in line or "VOICE_ID:" not in line:
				continue
			parts = line.replace("VOICE_NAME:", "").split("VOICE_ID:")
			if len(parts) != 2:
				continue
			voices.append({"name": parts[0], "id": parts[1]})
		ipc.SendEvent("voices", {"mode": mode, "voices": voices})
		return True
	if txt_str == "[AIUI_STREAM_END]":
		# StreamText sends stream_end itself; ignore legacy
		return True
	return False

def PrintSleep(txt_str, sleep_time=0.1, flush=True):
	if _EmitLegacyBanner(txt_str):
		time.sleep(sleep_time)
		return
	# Library / debug noise stays off the control plane
	print(txt_str, file=sys.stderr, flush=flush)
	time.sleep(sleep_time)

def DefaultVramReserves(have_image_model=False):
	"""Scale min/im reserve defaults from total GPU VRAM.

	Baseline (calibrated on 8GB): min_res=256, im_res=1536.
	16GB doubles; smaller cards scale down. Falls back to the 8GB baseline
	when CUDA is unavailable. Model Setup args still override afterward.
	"""
	base_vram_mb = 8192
	base_min_mb = 256
	base_im_mb = 1536
	total_mb = base_vram_mb
	if TorchAvailable():
		import torch
		if torch.cuda.is_available():
			try:
				_free_b, total_b = torch.cuda.mem_get_info(0)
				total_mb = max(1, int(total_b // (1024 * 1024)))
			except Exception:
				pass
	scale = total_mb / float(base_vram_mb)
	min_r = max(0, int(round(base_min_mb * scale)))
	im_r = max(0, int(round(base_im_mb * scale))) if have_image_model else 0
	return min_r, im_r

def ActiveReserveMb():
	"""MB to hold as a real CUDA tensor via ReserveVRAM.

	- min_res_vram_mb: floor for gen / small on-the-fly models
	- im_res_vram_mb: extra hold only while the image model is not loaded
	- _vram_hold_mb: temporary override (image load uses min_res only)
	"""
	ovr = getattr(E, "_vram_hold_mb", None)
	if ovr is not None:
		return max(0, int(ovr))
	min_r = max(0, int(getattr(E, "min_res_vram_mb", 0) or 0))
	if getattr(E, "im_model", None) is not None:
		return min_r
	im_r = max(0, int(getattr(E, "im_res_vram_mb", 0) or 0))
	return min_r + im_r

def ReserveVRAM():
	if E.reserved_vram != None: return
	need = ActiveReserveMb()
	if need <= 0:
		return
	if not TorchAvailable():
		return
	import torch
	if E.device == "auto" and torch.cuda.is_available():
		E.reserved_vram = torch.empty(256, 1024, need, dtype=torch.float32, device='cuda')

def CleanVRAM():
	E.reserved_vram = None
	gc.collect()
	if not TorchAvailable():
		return
	import torch
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

def _EnsureEngineOnPath():
	"""Keep engine root importable after SetWorkingDir chdir's into an avatar tree."""
	root = str(getattr(E, "work_dir", "") or "").rstrip("/\\")
	if root and root not in sys.path:
		sys.path.insert(0, root)


def _ActiveAvatarFns():
	"""Import only the selected avatar backend (cwd-relative vendored imports)."""
	mode = int(getattr(E, "anim_mode", -1) or 0)
	if mode < 0:
		raise RuntimeError("Animation engine is None")
	EnsureLocalMl("Talking avatar")
	_EnsureEngineOnPath()
	# Vendored packages expect cwd under their tree
	from lib.files import SetWorkingDir
	SetWorkingDir(mode)
	if mode == 0:
		from MakeItTalk.main_end2end import LoadFace, AnimFace
		return LoadFace, AnimFace
	if mode == 1:
		from Wav2Lip.inference import load_face, anim_face
		return load_face, anim_face
	if mode == 2:
		from SadTalker.inference import loadface, animface
		return loadface, animface
	from IMTalker.inference import LoadFace, AnimFace
	return LoadFace, AnimFace


def AudioDurationSecs(wav_path):
	"""Return WAV duration in seconds, or None if unreadable."""
	path = str(wav_path or "")
	if not path:
		return None
	try:
		import wave
		with wave.open(path, "rb") as w:
			rate = float(w.getframerate() or 0)
			frames = int(w.getnframes() or 0)
			if rate > 0 and frames >= 0:
				return frames / rate
	except Exception:
		pass
	try:
		from scipy.io import wavfile
		rate, signal = wavfile.read(path)
		rate = float(rate or 0)
		if rate <= 0:
			return None
		n = int(signal.shape[0]) if hasattr(signal, "shape") else len(signal)
		return n / rate
	except Exception:
		return None


def AnimAvatar():
	"""Generate talking-avatar video for E.chat_wav. Returns False if skipped/failed.

	When anim_max_secs > 0 and the TTS audio is longer than the cap, skip video
	entirely (no partial clip) so the chat path can fall back to PLAY_SPEECH.
	"""
	if int(getattr(E, "anim_mode", -1) or 0) < 0:
		return False
	if ipc.StopRequested():
		return False
	max_secs = float(getattr(E, "anim_max_secs", 30) or 0)
	if max_secs > 0:
		dur = AudioDurationSecs(getattr(E, "chat_wav", None))
		if dur is None:
			ipc.LogErr("WARN: talking avatar skipped (could not read audio duration)")
			return False
		if dur > max_secs:
			ipc.LogErr(
				f"WARN: talking avatar skipped "
				f"(audio {dur:.1f}s exceeds anim_max_secs={max_secs:g})"
			)
			return False

	load_fn, anim_fn = _ActiveAvatarFns()
	E.CleanVRAM()
	# load_fn unused here; anim_fn is AnimFace / anim_face / animface
	result = anim_fn(E.chat_wav)
	E.CleanVRAM()
	E.ReserveVRAM()
	return result


def LoadAvatar():
	if int(getattr(E, "anim_mode", -1) or 0) < 0:
		return True
	if not TorchAvailable():
		ipc.LogErr("Talking avatar skipped (local ML stack not installed)")
		return False
	try:
		load_fn, _anim_fn = _ActiveAvatarFns()
	except MissingLocalMl as e:
		ipc.LogErr(str(e))
		return False
	except Exception as e:
		ipc.LogErr(f"Talking avatar load failed ({e})")
		return False
	E.CleanVRAM()
	try:
		result = load_fn(E.avatar_img)
	except Exception as e:
		ipc.LogErr(f"Talking avatar load failed ({e})")
		E.CleanVRAM()
		E.ReserveVRAM()
		return False
	E.CleanVRAM()
	E.ReserveVRAM()
	if not result:
		ipc.LogErr("Talking avatar load failed (face detection / coeffs)")
	return result
