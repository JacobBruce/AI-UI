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

from lib.local_ml import EnsureLocalMl

torch = None
AutoProcessor = None
AutoModelForSpeechSeq2Seq = None
pipeline = None
wavfile = None
resample = None
_asr_ml_ready = False


def _EnsureAsrMl():
	global torch, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, wavfile, resample, _asr_ml_ready
	if _asr_ml_ready:
		return
	EnsureLocalMl("Speech recognition")
	import torch as _torch
	from transformers import AutoProcessor as _AutoProcessor, AutoModelForSpeechSeq2Seq as _AutoModelForSpeechSeq2Seq, pipeline as _pipeline
	from scipy.io import wavfile as _wavfile
	from scipy.signal import resample as _resample
	torch = _torch
	AutoProcessor = _AutoProcessor
	AutoModelForSpeechSeq2Seq = _AutoModelForSpeechSeq2Seq
	pipeline = _pipeline
	wavfile = _wavfile
	resample = _resample
	_asr_ml_ready = True


def LoadSRProcessor():
	_EnsureAsrMl()
	E.sr_processor = AutoProcessor.from_pretrained(E.sr_model_id, trust_remote_code=E.custom_token_code)


def LoadSRModel():
	_EnsureAsrMl()
	try:
		if E.sr_processor == None: LoadSRProcessor()
  
		if E.speech_recog_class != None:
			model_loader = getattr(importlib.import_module("transformers"), E.speech_recog_class)
		else:
			model_loader = AutoModelForSpeechSeq2Seq
  
		sr_torch_dtype = torch.float16 if E.comp_dev == "cuda" else torch.float32
		sr_device = None if E.device == "auto" else E.comp_dev
   
		if E.device == "auto":
			E.sr_model = model_loader.from_pretrained(E.sr_model_id, torch_dtype=sr_torch_dtype, attn_implementation=E.attn_implementation, use_safetensors=E.use_safetensors, trust_remote_code=E.custom_model_code, device_map="auto")
		elif E.device == "cuda":
			E.sr_model = model_loader.from_pretrained(E.sr_model_id, torch_dtype=sr_torch_dtype, attn_implementation=E.attn_implementation, use_safetensors=E.use_safetensors, trust_remote_code=E.custom_model_code).to(E.device)
		else:
			E.sr_model = model_loader.from_pretrained(E.sr_model_id, torch_dtype=sr_torch_dtype, attn_implementation=E.attn_implementation, low_cpu_mem_usage=True, use_safetensors=E.use_safetensors, trust_remote_code=E.custom_model_code).to(E.device)

		E.sr_pipe = pipeline(
		 "automatic-speech-recognition",
		 model=E.sr_model,
		 tokenizer=E.sr_processor.tokenizer,
		 feature_extractor=E.sr_processor.feature_extractor,
		 chunk_length_s=E.chunk_audio_secs,
		 torch_dtype=sr_torch_dtype,
		 device=sr_device
		)
	except Exception as e:
		sys.exit(f"ERROR: failed to load speech recognition model ({e})")


def ResampleSpeech(origin_audio, origin_sr, new_sr):
	_EnsureAsrMl()
	new_samps = int(len(origin_audio) * new_sr/origin_sr)
	return resample(origin_audio, new_samps)


def SpeechToText(speech_file):
	_EnsureAsrMl()
	if E.sr_model == None: LoadSRModel()
	sr, signal = wavfile.read(speech_file)
	try:
		if sr != E.sr_processor.feature_extractor.sampling_rate:
			signal = ResampleSpeech(signal, sr, E.sr_processor.feature_extractor.sampling_rate)
	except Exception as e:
		raise RuntimeError(f"failed to resample audio recording ({e})") from e
	result = E.sr_pipe(signal)
	return result["text"]
