# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

import os
import sys
import time
import json
import gc
import glob
import random
import threading
import tempfile
import importlib
import wave

from lib.local_ml import EnsureLocalMl, TorchAvailable
from lib.text_norm import NormalizeText

# Filled by _EnsureTtsMl() / first AI-TTS use.
np = None
torch = None
nn = None
SpeechT5Processor = None
SpeechT5ForTextToSpeech = None
SpeechT5HifiGan = None
librosa = None
pyttsx3 = None
sf = None
wavfile = None
EncoderClassifier = None
load_audio = None
_tts_ml_ready = False

# Absolute safety ceiling when audio_max_chars is 0 (unlimited soft cap).
_MAX_TTS_CHARS = 100_000


def GuardTtsText(text_str):
	"""Apply audio_max_chars soft cap and hard ceiling.

	When audio_max_chars > 0 and text is longer, raise RuntimeError so callers
	skip TTS entirely (chat / Speech Gen / read-aloud). When soft cap is 0,
	truncate only at _MAX_TTS_CHARS.
	"""
	text_str = str(text_str or "")
	soft = int(getattr(E, "audio_max_chars", 4000) or 0)
	if soft > 0 and len(text_str) > soft:
		raise RuntimeError(
			f"TTS skipped (text {len(text_str)} chars exceeds audio_max_chars={soft})"
		)
	if len(text_str) > _MAX_TTS_CHARS:
		text_str = text_str[:_MAX_TTS_CHARS]
	return text_str


def IsTtsLengthSkip(exc):
	"""True when GuardTtsText refused input (audio_max_chars soft cap)."""
	return isinstance(exc, RuntimeError) and "audio_max_chars" in str(exc)


def _EnsureTtsMl(feature="Text-to-speech"):
	"""Import torch / SpeechT5 / audio deps for AI TTS paths."""
	global np, torch, nn, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
	global librosa, sf, wavfile, EncoderClassifier, load_audio, _tts_ml_ready
	if _tts_ml_ready:
		return
	EnsureLocalMl(feature)
	import numpy as _np
	import torch as _torch
	import torch.nn as _nn
	from transformers import (
		SpeechT5Processor as _SpeechT5Processor,
		SpeechT5ForTextToSpeech as _SpeechT5ForTextToSpeech,
		SpeechT5HifiGan as _SpeechT5HifiGan,
	)
	import librosa as _librosa
	import soundfile as _sf
	from scipy.io import wavfile as _wavfile
	from speechbrain.inference import EncoderClassifier as _EncoderClassifier
	from lib.av import load_audio as _load_audio
	np = _np
	torch = _torch
	nn = _nn
	SpeechT5Processor = _SpeechT5Processor
	SpeechT5ForTextToSpeech = _SpeechT5ForTextToSpeech
	SpeechT5HifiGan = _SpeechT5HifiGan
	librosa = _librosa
	sf = _sf
	wavfile = _wavfile
	EncoderClassifier = _EncoderClassifier
	load_audio = _load_audio
	EnsureTorchaudioLoad()
	_tts_ml_ready = True


def _EnsurePyttsx3():
	global pyttsx3
	if pyttsx3 is not None:
		return
	import pyttsx3 as _pyttsx3
	pyttsx3 = _pyttsx3

def LoadKokoro():
	global kokoro
	try:
		import kokoro as _kokoro
		kokoro = _kokoro
		E.kokoro_loaded = True
	except Exception as e:
		sys.exit(f"ERROR: failed to load Kokoro, try 'pip install kokoro' and ensure espeak-ng is installed on your system\n\nException: {e}")


def LoadNeuTTS():
	global NeuTTSAir
	try:
		from lib.neutts import NeuTTSAir as _NeuTTSAir
		NeuTTSAir = _NeuTTSAir
		E.neutts_loaded = True
	except Exception as e:
		sys.exit(f"ERROR: failed to load NeuTTSAir, try 'pip install neucodec' and ensure espeak-ng is installed on your system\n\nException: {e}")


def LoadChatTTS():
	global ChatTTS
	try:
		import ChatTTS as _ChatTTS
		ChatTTS = _ChatTTS
		E.chattts_loaded = True
	except Exception as e:
		sys.exit(f"ERROR: failed to load ChatTTS, try 'pip install chattts'\n\nException: {e}")


def LoadVoxCPM():
	global VoxCPM
	try:
		from voxcpm import VoxCPM as _VoxCPM
		VoxCPM = _VoxCPM
		E.voxcpm_loaded = True
	except Exception as e:
		sys.exit(f"ERROR: failed to load VoxCPM, try 'pip install voxcpm'\n\nException: {e}")


def LoadKitten():
	global KittenTTS
	try:
		from kittentts import KittenTTS as _KittenTTS
		KittenTTS = _KittenTTS
		E.kitten_loaded = True
	except Exception as e:
		sys.exit(
			"ERROR: failed to load KittenTTS. On Python ≥3.13 install from git main "
			"(release wheels pin misaki / Python <3.13):\n"
			"  pip install onnxruntime soundfile numpy huggingface_hub phonemizer espeakng_loader\n"
			"  PIP_IGNORE_REQUIRES_PYTHON=1 pip install \"git+https://github.com/KittenML/KittenTTS.git\"\n\n"
			f"Exception: {e}"
		)


def _VoxSampleRate():
	"""VoxCPM-0.5B is 16kHz; VoxCPM1.5/2 report their rate on the inner model."""
	inner = getattr(E.tts_model, "tts_model", None)
	sr = getattr(inner, "sample_rate", None) if inner is not None else None
	try:
		return int(sr) if sr else 16000
	except (TypeError, ValueError):
		return 16000


def _Audio8SampleRate():
	cfg = getattr(E.tts_model, "config", None)
	sr = getattr(cfg, "codec_sample_rate", None) if cfg is not None else None
	try:
		return int(sr) if sr else 44100
	except (TypeError, ValueError):
		return 44100


def _Audio8Device():
	dev = str(getattr(E, "comp_dev", "cpu") or "cpu")
	if dev == "cuda" and torch.cuda.is_available():
		return "cuda"
	return "cpu"


def _Audio8PrecomputeRope(length, head_dim, base):
	"""fp32 RoPE tables - Audio8's shipped helper hardcodes bf16 and can be NaN on some stacks."""
	frequencies = 1.0 / (
		base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim)
	)
	phases = torch.outer(torch.arange(length).float(), frequencies)
	values = torch.polar(torch.ones_like(phases), phases)
	return torch.stack((values.real, values.imag), dim=-1)


def _Audio8FixRope(model):
	"""Replace Audio8 bf16 RoPE buffers with finite fp32 tables after load."""
	cfg = model.config
	device = next(model.parameters()).device
	model.freqs_cis = _Audio8PrecomputeRope(
		cfg.max_seq_len, cfg.head_dim, cfg.rope_base
	).to(device)
	model.fast_freqs_cis = _Audio8PrecomputeRope(
		cfg.num_codebooks, cfg.head_dim, cfg.rope_base
	).to(device)


def _Audio8Generate(txt, ref_wav=None, ref_text=None):
	"""Run Audio8 TTS; optional zero-shot clone via reference wav + transcript."""
	proc = E.audio8_processor
	kwargs = {"text": [txt], "return_tensors": "pt"}
	if ref_wav and ref_text:
		kwargs["reference_audio"] = [ref_wav]
		kwargs["reference_text"] = [ref_text]
	inputs = proc(**kwargs)
	device = _Audio8Device()
	inputs = {name: value.to(device) for name, value in inputs.items()}
	try:
		max_new = max(1, int(getattr(E, "audio_max_new_tokens", 4096) or 4096))
	except (TypeError, ValueError):
		max_new = 4096
	with torch.inference_mode():
		output = E.tts_model.generate(
			**inputs,
			max_new_tokens=max_new,
			temperature=0.8,
			top_p=0.95,
			top_k=50,
			do_sample=True,
			return_dict_in_generate=True,
		)
		waveforms, waveform_lengths = E.tts_model.decode_audio(output.codes)
	finished = True
	try:
		finished = bool(output.finished[0])
	except Exception:
		pass
	if not finished:
		from lib import ipc
		ipc.LogErr(
			f"Audio8 TTS hit max_new_tokens={max_new} without EOS "
			f"(code_len={int(output.code_lengths[0]) if hasattr(output, 'code_lengths') else '?'})"
		)
	audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
	return audio, _Audio8SampleRate()


def EnsureTorchaudioLoad():
	"""torchaudio 2.9+ routes load() through torchcodec.

	Order: stock torchcodec → optional NPP preload if libs are already on disk
	(venv nvidia-npp-* or a CUDA toolkit) → soundfile fallback for file paths.
	Does not require any NVIDIA package to be installed.
	"""
	try:
		import torchaudio
	except ImportError:
		return
	if getattr(torchaudio, "_aiui_load_patched", False):
		return

	def _torchcodec_ok():
		try:
			from torchcodec.decoders import AudioDecoder  # noqa: F401
			return True
		except Exception:
			return False

	if _torchcodec_ok():
		return

	def _load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True,
			format=None, buffer_size=4096, backend=None):
		if not isinstance(uri, (str, bytes, os.PathLike)):
			raise RuntimeError("AI UI torchaudio.load fallback only supports file paths")
		frames = -1 if num_frames is None or num_frames < 0 else int(num_frames)
		data, sr = sf.read(
			os.fspath(uri), dtype="float32", always_2d=True,
			start=int(frame_offset or 0), frames=frames
		)
		audio = torch.from_numpy(np.ascontiguousarray(data.T if channels_first else data))
		return audio, int(sr)

	torchaudio.load = _load
	torchaudio._aiui_load_patched = True


def LoadSpeaker(voice_index):
	_EnsureTtsMl()
	result = None
 
	if E.smodel_type == 0:
		result = torch.tensor(np.load(E.work_dir+"/embeddings/"+E.ai_voices[voice_index])).unsqueeze(0).to(E.comp_dev)
	elif E.smodel_type == 1:
		if voice_index > 0:
			result = E.ReadTextFile(E.work_dir+"/embeddings/ChatTTS/"+E.ai_voices[voice_index])
	elif E.smodel_type == 4:
		result = torch.tensor(np.load(E.work_dir+"/embeddings/NeuTTS/"+E.ai_voices[voice_index])).to(E.comp_dev)
 
	return result


def LoadSysVoices():
 
	E.voices = E.synthesizer.getProperty('voices')
	voice_list = 'SYS_VOICES:'
 
	for voice in E.voices:
		voice_list += "VOICE_NAME:"+voice.name+"VOICE_ID:"+voice.id+"\n"

	if voice_list != 'SYS_VOICES:': E.PrintSleep(voice_list)


def LoadAIVoices(speaker_name=''):
	speaker_index = 0

	if E.smodel_type == 0:
		E.ai_voices = glob.glob1(E.work_dir+"/embeddings/", '*.npy')
		E.ai_voices.sort()
	elif E.smodel_type == 1:
		E.ai_voices = glob.glob1(E.work_dir+"/embeddings/ChatTTS/", '*.txt')
		E.ai_voices.sort()
		E.ai_voices.insert(0, "Random")
	elif E.smodel_type == 2:
		E.ai_voices = E.ReadTextFile(E.work_dir+"/embeddings/Kokoro/voices.txt").replace("\r", '').split("\n")
		E.ai_voices.sort()
	elif E.smodel_type == 3:
		E.ai_voices = glob.glob1(E.work_dir+"/embeddings/VoxCPM/", '*.wav')
		E.ai_voices.sort()
	elif E.smodel_type == 4:
		E.ai_voices = glob.glob1(E.work_dir+"/embeddings/NeuTTS/", '*.npy')
		E.ai_voices.sort()
	elif E.smodel_type == 5:
		# Prefer voices from a loaded Kitten model; fall back to shipped list.
		voices = None
		if E.kitten_loaded and E.tts_model is not None and E.smodel_type == E.last_smt_val:
			voices = getattr(E.tts_model, "available_voices", None)
		if voices:
			E.ai_voices = list(voices)
		else:
			E.ai_voices = E.ReadTextFile(E.work_dir+"/embeddings/Kitten/voices.txt").replace("\r", '').split("\n")
		E.ai_voices = [v for v in E.ai_voices if str(v).strip()]
		E.ai_voices.sort()
	elif E.smodel_type == 6:
		E.ai_voices = glob.glob1(E.work_dir+"/embeddings/Audio8/", '*.wav')
		E.ai_voices.sort()
 
	voice_list = 'AI_VOICES:'
 
	for i in range(0,len(E.ai_voices)):
		if E.ai_voices[i] == speaker_name: speaker_index = i
		voice_list += "VOICE_NAME:"+E.ai_voices[i]+"VOICE_ID:"+str(i)+"\n"
 
	if voice_list != 'AI_VOICES:': E.PrintSleep(voice_list)
  
	return speaker_index


def ConfigTTSEngine():
	E.tts_ready = True
 
	if E.tts_mode == 0:
		if len(E.voices) > 0:
			E.synthesizer.setProperty('voice', E.voices[E.voice_key].id)
			E.synthesizer.setProperty('volume', E.voice_vol)
			E.synthesizer.setProperty('rate', E.voice_rate)
		else:
			print("ERROR: no system voices could be found")
			E.tts_ready = False
	else:
		if len(E.ai_voices) > 0:
				E.speaker_embeddings = LoadSpeaker(E.voice_key)
		else:
			print("ERROR: no voice embeddings could be found")
			E.tts_ready = False


def LoadTTSVoices():
	try:
		_EnsurePyttsx3()
	except ImportError as e:
		from lib import ipc
		ipc.LogErr(f"System TTS unavailable ({e}). Install pyttsx3 (included in requirements-lite.txt).")
		return
	if E.synthesizer == None:
		E.synthesizer = pyttsx3.init()
		if E.synthesizer is None:
			return

	LoadSysVoices()
	LoadAIVoices()
	if E.tts_mode > 0:
		try:
			_EnsureTtsMl()
		except Exception as e:
			from lib import ipc
			ipc.LogErr(f"AI TTS unavailable ({e})")
			return
	ConfigTTSEngine()


def LoadTTSEngine(force_ai_mode=None, tts_mid=None, tts_cid=None):
	_EnsureTtsMl()
 
	if tts_mid == None: tts_mid = E.tts_model_id
	if tts_cid == None: tts_cid = E.vocoder_model
  
	if (force_ai_mode == True or E.tts_mode > 0) and (E.tts_model == None or E.smodel_type != E.last_smt_val):
		if E.smodel_type == 0:
			if tts_mid == '':
				E.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
				E.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(E.comp_dev)
			else:
				E.processor = SpeechT5Processor.from_pretrained(tts_mid)
				E.tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_mid).to(E.comp_dev)
			if tts_cid == '':
				E.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(E.comp_dev)
			else:
				E.vocoder = SpeechT5HifiGan.from_pretrained(tts_cid).to(E.comp_dev)
		elif E.smodel_type == 1:
			if not E.chattts_loaded: LoadChatTTS()
			E.tts_model = ChatTTS.Chat()
			if tts_mid == '':
				E.tts_model.load(compile=E.torch_compile, source="huggingface")
			else:
				E.tts_model.load(compile=E.torch_compile, source="custom", custom_path=tts_mid)
		elif E.smodel_type == 2:
			if not E.kokoro_loaded: LoadKokoro()
			E.tts_model = kokoro.KModel()
		elif E.smodel_type == 3:
			if not E.voxcpm_loaded: LoadVoxCPM()
			if tts_mid == '':
				E.tts_model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B", load_denoiser=False)
			else:
				E.tts_model = VoxCPM.from_pretrained(tts_mid, load_denoiser=False)
		elif E.smodel_type == 4:
			if not E.neutts_loaded: LoadNeuTTS()
			backbone_id = "neuphonic/neutts-air" if tts_mid == '' else tts_mid
			codec_id = "neuphonic/neucodec" if tts_cid == '' else tts_cid
			E.tts_model = NeuTTSAir(backbone_repo=backbone_id, backbone_device=E.comp_dev, codec_repo=codec_id, codec_device=E.comp_dev)
		elif E.smodel_type == 5:
			if not E.kitten_loaded: LoadKitten()
			model_id = "KittenML/kitten-tts-mini-0.8" if tts_mid == '' else tts_mid
			# Optional CUDA backend when available; fall back to default providers.
			if E.comp_dev == "cuda":
				try:
					E.tts_model = KittenTTS(model_id, backend="cuda")
				except Exception:
					E.tts_model = KittenTTS(model_id)
			else:
				E.tts_model = KittenTTS(model_id)
			# Refresh voice list from the loaded model when possible.
			voices = getattr(E.tts_model, "available_voices", None)
			if voices:
				E.ai_voices = sorted(v for v in voices if str(v).strip())
		elif E.smodel_type == 6:
			from transformers import AutoModel, AutoProcessor
			model_id = "Audio8/Audio8-TTS-Preview-0.6b" if tts_mid == '' else tts_mid
			device = _Audio8Device()
			dtype = torch.bfloat16 if device == "cuda" else torch.float32
			E.audio8_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
			try:
				E.tts_model = AutoModel.from_pretrained(
					model_id, trust_remote_code=True, dtype=dtype
				)
			except TypeError:
				E.tts_model = AutoModel.from_pretrained(
					model_id, trust_remote_code=True, torch_dtype=dtype
				)
			E.tts_model = E.tts_model.eval().to(device)
			_Audio8FixRope(E.tts_model)
			E.audio8_loaded = True
 
	E.last_smt_val = E.smodel_type
	if E.smodel_type == 5 and E.kitten_loaded:
		LoadAIVoices()
	ConfigTTSEngine()


def ApplyTalkMode():
	anim_ok = int(getattr(E, "anim_mode", -1) or 0) >= 0
	if E.talk_mode == 0:
		E.do_talk = True
		E.do_anim = anim_ok
	elif E.talk_mode == 1:
		E.do_talk = True
		E.do_anim = False
	else:
		E.do_talk = False
		E.do_anim = False


def _ResampleAudio(origin_sample_rate, origin_audio, new_sample_rate):
	_EnsureTtsMl()
	from scipy.signal import resample as _resample
	new_samps = int(len(origin_audio) * new_sample_rate / origin_sample_rate)
	return _resample(origin_audio, new_samps)


def CloneVoice(clone_model, speaker_name, speaker_wav, transcript=None):
	_EnsureTtsMl()
	embeddings = []
	signal = []
 
	try:
		if clone_model == 0:
			fs, signal = wavfile.read(speaker_wav)
			if fs != 16000:
				signal = _ResampleAudio(fs, signal, 16000)
			else:
				signal = torch.tensor(signal)
		elif clone_model == 1:
			if transcript == None or transcript == '':
				return 'ERROR:Transcript text is required!'
			signal = load_audio(speaker_wav, 24000)
			#if fs != 24000:
			#	signal = load_audio(speaker_wav, 24000)
			#else:
			#	signal, fs = load_audio(speaker_wav)
		elif clone_model == 3 or clone_model == 4 or clone_model == 6:
			if transcript == None or transcript == '':
				return 'ERROR:Transcript text is required!'
			if clone_model == 3:
				signal, sr = librosa.load(speaker_wav, sr=16000, mono=True)
			elif clone_model == 6:
				signal, sr = librosa.load(speaker_wav, sr=None, mono=True)
	except Exception as e:
		from lib import ipc
		ipc.LogException("clone_voice_read", e)
		return 'ERROR:There was an error reading the voice sample.'

	try:
		if clone_model == 0:
   
			ec_model_id = "speechbrain/spkrec-xvect-voxceleb"
   
			if torch.cuda.is_available():
				classifier = EncoderClassifier.from_hparams(source=ec_model_id, run_opts={"device": "cuda"})
			else:
				classifier = EncoderClassifier.from_hparams(source=ec_model_id)

			with torch.no_grad():
				embeddings = classifier.encode_batch(signal)
				embeddings = torch.nn.functional.normalize(embeddings, dim=2)
				embeddings = embeddings.squeeze().cpu().numpy()
		else:
			if E.tts_mode != 1 or E.smodel_type != clone_model or E.tts_model == None or E.smodel_type != E.last_smt_val:
				if clone_model != 3 and clone_model != 6:
					lsmt = E.smodel_type
					E.smodel_type = clone_model
					LoadTTSEngine(True, '', '')
					E.smodel_type = lsmt
   
			if clone_model == 1:
				embeddings = E.tts_model.sample_audio_speaker(signal)
			elif clone_model == 4:
				embeddings = E.tts_model.encode_reference(speaker_wav).cpu().numpy()
	except Exception as e:
		from lib import ipc
		ipc.LogException("clone_voice_model", e)
		return 'ERROR:There was an error with the cloning model.'
 
	try:
		if clone_model == 3:
			ts_dir = E.work_dir+"/embeddings/VoxCPM/transcripts/"
			if not os.path.exists(ts_dir): os.mkdir(ts_dir)
			E.WriteTextFile(ts_dir+speaker_name+".txt", transcript)
			sf.write(E.work_dir+"/embeddings/VoxCPM/"+speaker_name+".wav", signal, 16000)
			voice_id = LoadAIVoices(speaker_name+'.wav')
		elif clone_model == 6:
			base = E.work_dir+"/embeddings/Audio8/"
			ts_dir = base+"transcripts/"
			os.makedirs(ts_dir, exist_ok=True)
			E.WriteTextFile(ts_dir+speaker_name+".txt", transcript)
			sf.write(base+speaker_name+".wav", signal, int(sr) if sr else 44100)
			voice_id = LoadAIVoices(speaker_name+'.wav')
		elif len(embeddings) > 0:
			if clone_model == 0:
				np.save(E.work_dir+"/embeddings/"+speaker_name+".npy", embeddings)
				voice_id = LoadAIVoices(speaker_name+'.npy')
			elif clone_model == 1:
			 #ref_txt = tts_model.infer(transcript, refine_text_only=True)
				ts_dir = E.work_dir+"/embeddings/ChatTTS/transcripts/"
				if not os.path.exists(ts_dir): os.mkdir(ts_dir)
				E.WriteTextFile(ts_dir+speaker_name+".txt", transcript)
				E.WriteTextFile(E.work_dir+"/embeddings/ChatTTS/"+speaker_name+".txt", embeddings)
				voice_id = LoadAIVoices(speaker_name+'.txt')
			elif clone_model == 4:
				ts_dir = E.work_dir+"/embeddings/NeuTTS/transcripts/"
				if not os.path.exists(ts_dir): os.mkdir(ts_dir)
				E.WriteTextFile(ts_dir+speaker_name+".txt", transcript)
				np.save(E.work_dir+"/embeddings/NeuTTS/"+speaker_name+".npy", embeddings)
				voice_id = LoadAIVoices(speaker_name+'.npy')
		else:
			return 'ERROR:There was an error generating the embeddings.'
	except Exception as e:
		from lib import ipc
		ipc.LogException("clone_voice_save", e)
		return 'ERROR:There was an error saving the voice file.'
 
	return speaker_name+':'+str(voice_id)


def ModifySound(sound_wav, vcvol, vcrate, vcpitch):
	_EnsureTtsMl()
	if vcrate != 200 or vcpitch != 0 or vcvol < 1.0:
		y, sr = librosa.load(sound_wav)
	else:
		return
  
	if vcpitch != 0:
		y = librosa.effects.pitch_shift(y, sr=sr, n_steps=vcpitch)
 
	if vcrate != 200:
		if vcrate < 200:
			rate_flt = 0.5 + ((float(vcrate) / 200.0) * 0.5)
		else:
			rate_flt = 1.0 + ((float(vcrate)-200.0) / 200.0)
		y = librosa.effects.time_stretch(y, rate=rate_flt)
  
	if vcvol < 1.0: y *= vcvol
 
	sf.write(sound_wav, y, sr)


def AddEpsilonNoise(sound_wav):
	_EnsureTtsMl()
	y, sr = librosa.load(sound_wav)
	y += random.uniform(-1., 1.) * 0.01
	sf.write(sound_wav, y, sr)


def SentenceSplit(txt, min_sl=64, max_sl=256):
	result = []
	got_sent = False
	sent = ''
	sl = 0
 
	for i, c in enumerate(txt):
		if c == '.' or c == '?' or c == '!':
			if sl < min_sl:
				sl += 1
			else:
				got_sent = True
				for l in range(i+1, i+8):
					if len(txt) > l and (txt[l] == '.' or txt[l] == '?' or txt[l] == '!'):
						got_sent = False
		elif sl >= max_sl and (c == ',' or c == ':' or c == ';' or c == ' '):
			got_sent = True
		else:
			sl += 1
   
		sent += c
    
		if got_sent:
			result.append(sent.strip())
			sent = ''
			sl = 0
			got_sent = False
 
	sent = sent.strip()
	if sent != '': result.append(sent)
    
	return result


def TextToSpeechAI(text_str, wav_file, speaker_data, speech_vol, speech_rate, speech_ps, voice_index=None):
	_EnsureTtsMl()
	LoadTTSEngine(True)
	if voice_index == None: voice_index = E.voice_key
	text_chunks = []
	wav_files = []

	text_str = GuardTtsText(text_str)

	if E.smodel_type != 3: text_str = NormalizeText(text_str)
 
	if len(text_str) > 256:
		text_chunks = SentenceSplit(text_str)
	else:
		text_chunks = [text_str]

	from lib import ipc
	for i, txt in enumerate(text_chunks):
		if ipc.StopRequested():
			break
		wav_part_file = E.StripEnd(wav_file, ".wav") + "_chunk"+str(i)+".wav"
		if E.smodel_type == 0:
			inputs = E.processor(text=txt, return_tensors="pt").to(E.comp_dev)
			speech = E.tts_model.generate_speech(inputs["input_ids"], speaker_data, vocoder=E.vocoder)
			sf.write(wav_part_file, speech.cpu().numpy(), samplerate=16000)
			wav_files.append(wav_part_file)
		elif E.smodel_type == 1:
			if speaker_data == None:
				embeddings = E.tts_model.sample_random_speaker()
				E.WriteTextFile(E.work_dir+"/embeddings/ChatTTS/random.tmp", embeddings)
				params_ic = ChatTTS.Chat.InferCodeParams(spk_emb = embeddings)
			else:
				ts_file = E.work_dir+"/embeddings/ChatTTS/transcripts/" + E.ai_voices[voice_index]
				if os.path.isfile(ts_file):
					params_ic = ChatTTS.Chat.InferCodeParams(spk_smp=speaker_data, txt_smp=E.ReadTextFile(ts_file))
				else:
					params_ic = ChatTTS.Chat.InferCodeParams(spk_emb = speaker_data)
			speech = E.tts_model.infer(txt+" [uv_break]", skip_refine_text=True, params_infer_code=params_ic)
			sf.write(wav_part_file, speech[0], samplerate=24000)
			wav_files.append(wav_part_file)
		elif E.smodel_type == 2 and E.kokoro_loaded:
			tts_pipeline = kokoro.KPipeline(lang_code=E.ai_voices[voice_index][0], model=E.tts_model)
			tts_generator = tts_pipeline(txt, voice=E.ai_voices[voice_index], split_pattern='')
			for i, (gs, ps, speech) in enumerate(tts_generator):
				sf.write(wav_part_file, speech, samplerate=24000)
				wav_files.append(wav_part_file)
				break
		elif E.smodel_type == 3 and E.voxcpm_loaded:
			wav_path = E.work_dir+"/embeddings/VoxCPM/" + E.ai_voices[voice_index]
			ts_text = E.ReadTextFile(E.work_dir+"/embeddings/VoxCPM/transcripts/" + E.StripEnd(E.ai_voices[voice_index], ".wav") + ".txt")
			speech = E.tts_model.generate(
			 text=txt,
			 prompt_wav_path=wav_path,
			 prompt_text=ts_text,
			 cfg_value=2.0,
			 inference_timesteps=10,
			 normalize=True,
			 denoise=False,
			 retry_badcase=True,
			 retry_badcase_max_times=1,
			 retry_badcase_ratio_threshold=6.0
			)
			sf.write(wav_part_file, speech, samplerate=_VoxSampleRate())
			wav_files.append(wav_part_file)
		elif E.smodel_type == 4 and E.neutts_loaded:
			ts_file = E.work_dir+"/embeddings/NeuTTS/transcripts/" + E.StripEnd(E.ai_voices[voice_index], ".npy") + ".txt"
			speech = E.tts_model.infer(txt, speaker_data, E.ReadTextFile(ts_file))
			sf.write(wav_part_file, speech, samplerate=24000)
			wav_files.append(wav_part_file)
		elif E.smodel_type == 5 and E.kitten_loaded:
			speech = E.tts_model.generate(txt, voice=E.ai_voices[voice_index])
			sf.write(wav_part_file, speech, samplerate=24000)
			wav_files.append(wav_part_file)
		elif E.smodel_type == 6 and E.audio8_loaded:
			wav_path = E.work_dir+"/embeddings/Audio8/" + E.ai_voices[voice_index]
			ts_text = E.ReadTextFile(
				E.work_dir+"/embeddings/Audio8/transcripts/"
				+ E.StripEnd(E.ai_voices[voice_index], ".wav") + ".txt"
			)
			speech, sr = _Audio8Generate(txt, ref_wav=wav_path, ref_text=ts_text)
			sf.write(wav_part_file, speech, samplerate=sr)
			wav_files.append(wav_part_file)
		else:
			sys.exit("ERROR: no TTS model available for text-to-speech")
 
	if len(wav_files) > 1:
		merged_wav = []
		for wav_part in wav_files:
			f = wave.open(wav_part, 'rb')
			merged_wav.append([f.getparams(), f.readframes(f.getnframes())])
			f.close()
		wav_out = wave.open(wav_file, 'wb')
		wav_out.setparams(merged_wav[0][0])
		for i in range(len(merged_wav)):
			wav_out.writeframes(merged_wav[i][1])
		wav_out.close()
	elif len(wav_files) > 0:
		if os.path.isfile(wav_file): os.remove(wav_file)
		os.rename(wav_files[0], wav_file)
	else:
		if ipc.StopRequested():
			return
		sys.exit("ERROR: failed to generate speech audio")
 
	ModifySound(wav_file, speech_vol, speech_rate, speech_ps)


def TextToSpeechSys(text_str, wav_file, speech_ps):
	_EnsurePyttsx3()
	text_str = GuardTtsText(text_str)
	if speech_ps != 0:
		_EnsureTtsMl()
	E.synthesizer.save_to_file("<pitch middle='"+str(speech_ps)+"'/>"+text_str, wav_file)
	E.synthesizer.runAndWait()
	AddEpsilonNoise(wav_file)

