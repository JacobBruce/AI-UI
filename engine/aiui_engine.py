"""
 # Copyright 2023 Bitfreak Software
 # All Rights Reserved.

"""

import sys, time, glob, os, gc
import threading
import ctypes
import json
import wave
import tempfile
import importlib
import librosa
import torch
import random
import pyttsx3
import ChatTTS
import transformers
import numpy as np
import torch.nn as nn
import soundfile as sf
from peft import PeftModel
from datetime import datetime
from huggingface_hub import login
from scipy.io import wavfile
from scipy.signal import resample
from tools.av import load_audio
from tools.text_norm import NormalizeText
from tools.rp_funcs import SaveGame, LoadGame, NewGame
from urllib.request import urlopen
from qwen_vl_utils import process_vision_info
from speechbrain.inference import EncoderClassifier
from MakeItTalk.main_end2end import ResampleAudio, LoadFace, AnimFace
from Wav2Lip.inference import load_face, anim_face
from SadTalker.inference import loadface, animface
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from diffusers.utils import load_image
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline, QwenImagePipeline, FluxPipeline, GGUFQuantizationConfig
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, SD3Transformer2DModel, FluxTransformer2DModel, QwenImageTransformer2DModel
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForSpeechSeq2Seq
from transformers import CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel, Qwen2_5_VLForConditionalGeneration

kokoro_retry = False
kokoro_loaded = False

try:
	from kokoro import KModel, KPipeline
	kokoro_loaded = True
except:
	espeak_lib_win = "C:/Program Files/eSpeak NG/libespeak-ng.dll"
	espeak_lib_mac = "/opt/homebrew/lib/libespeak-ng.dylib"

	if os.path.isfile(espeak_lib_win):
		EspeakWrapper.set_library(espeak_lib_win)
		kokoro_retry = True
	elif os.path.isfile(espeak_lib_mac):
		EspeakWrapper.set_library(espeak_lib_mac)
		kokoro_retry = True

if kokoro_retry:
	try:
		from kokoro import KModel, KPipeline
		kokoro_loaded = True
	except:
		print("ERROR: failed to load Kokoro, ensure espeak-ng is installed on your system", flush=True)
		time.sleep(0.1)

###### Config/Vars ######
torch.set_float32_matmul_precision('high')
sys.stdout.reconfigure(encoding='utf-8')

print("APP_CONFIG:")
work_dir = input().rstrip('/')
model_id = input().rstrip('/')
im_model_id = input().rstrip('/')
tts_model_id = input().rstrip('/')
sr_model_id = input().rstrip('/')
model_type = int(input())
imodel_type = int(input())
smodel_type = int(input())
model_args = input().split(',')
device = input()
start_meth = input()
avatar_img = input()

print("OTHER_CONFIG:")
gen_mode = int(input())
format_mode = int(input())
enable_tooluse = int(input())
hf_access_token = input()

if hf_access_token != '':
	login(token=hf_access_token)

if not os.path.isdir(work_dir):
	sys.exit("ERROR: cannot find engine folder at "+work_dir)

print("TOOL_CONFIG:")
temp_form = input()
tool_mode = input()

print("AI_CONFIG:")
msg_mem = int(input())
gen_max = int(input())
gen_min = int(input())
do_sample = int(input())
num_beams = int(input())
beam_groups = int(input())
rando_lvl = float(input())
prompt_p = int(input())
top_k = int(input())
top_p = float(input())
typical_p = float(input())
rep_penalty = float(input())
rando_min = rando_lvl
rando_add = 0.05
rando_sub = 0.01

user_name = ''
bot_name = ''
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
rag_files = []
end_strings = None
tool_funcs = None
tool_func = None
use_tool_config = True

min_res_vram_mb = 800
ext_res_vram_mb = 0 if im_model_id == '' else 3000
reserve_vram_mb = min_res_vram_mb + ext_res_vram_mb
reserved_vram = None

tokenizer_model = ''
vocoder_model = ''
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
process_vision_func = None
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
im_torch_dtype = torch.float16
torch_dtype = "auto"

use_chat_template = True if temp_form != "none" else False
use_tool_template = True if temp_form == "tool" else False

if imodel_type > 0: im_att_slicing = False
comp_dev = 'cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu'
compile_mode = 'default' if comp_dev == 'cpu' else 'max-autotune'
seed_gen = torch.Generator(device=comp_dev)

for arg in model_args:
	carg = arg.strip(" ").replace("\\", "/")
	if carg.startswith("tokenizer_model="):
		tokenizer_model = carg.replace("tokenizer_model=", "", 1)
	elif carg.startswith("model_adapter="):
		model_adapter = carg.replace("model_adapter=", "", 1)
	elif carg.startswith("im_lora_file="):
		im_lora_file = carg.replace("im_lora_file=", "", 1)
	elif carg.startswith("im_lora_dir="):
		im_lora_dir = carg.replace("im_lora_dir=", "", 1)
	elif carg.startswith("im_lora_scale="):
		im_lora_scale = float(carg.replace("im_lora_scale=", "", 1))
	elif carg.startswith("im_vae_file="):
		im_vae_file = carg.replace("im_vae_file=", "", 1)
	elif carg.startswith("im_unet_file="):
		im_unet_file = carg.replace("im_unet_file=", "", 1)
	elif carg.startswith("im_tran_file="):
		im_tran_file = carg.replace("im_tran_file=", "", 1)
	elif carg.startswith("im_enc_file="):
		im_enc_file = carg.replace("im_enc_file=", "", 1)
	elif carg.startswith("im_enc2_file="):
		im_enc2_file = carg.replace("im_enc2_file=", "", 1)
	elif carg.startswith("im_enc3_file="):
		im_enc3_file = carg.replace("im_enc3_file=", "", 1)
	elif carg.startswith("im_variant="):
		im_variant = carg.replace("im_variant=", "", 1)
	elif carg.startswith("im_config_file="):
		im_config_file = carg.replace("im_config_file=", "", 1)
	elif carg.startswith("im_model_class="):
		im_model_class = carg.replace("im_model_class=", "", 1)
	elif carg.startswith("text_lang_class="):
		text_lang_class = carg.replace("text_lang_class=", "", 1)
	elif carg.startswith("multi_modal_class="):
		multi_modal_class = carg.replace("multi_modal_class=", "", 1)
	elif carg.startswith("speech_recog_class="):
		speech_recog_class = carg.replace("speech_recog_class=", "", 1)
	elif carg.startswith("vocoder_model="):
		vocoder_model = carg.replace("vocoder_model=", "", 1)
	elif carg.startswith("process_vision_func="):
		process_vision_func = carg.replace("process_vision_func=", "", 1)
	elif carg.startswith("attn_implementation="):
		attn_implementation = carg.replace("attn_implementation=", "", 1)
	elif carg.startswith("roles_append_str="):
		roles_append_str = carg.replace("roles_append_str=", "", 1).replace("\\n", "\n")
	elif carg.startswith("torch_dtype="):
		dtype = carg.replace("torch_dtype=", "", 1).lower()
		if dtype == "float32":
			torch_dtype = torch.float32
		elif dtype == "float16":
			torch_dtype = torch.float16
		elif dtype == "bfloat16":
			torch_dtype = torch.bfloat16
	elif carg.startswith("im_torch_dtype="):
		im_dtype = carg.replace("im_torch_dtype=", "", 1).lower()
		if im_dtype == "float32":
			im_torch_dtype = torch.float32
		elif im_dtype == "float16":
			im_torch_dtype = torch.float16
		elif im_dtype == "bfloat16":
			im_torch_dtype = torch.bfloat16
	elif carg.startswith("use_safetensors="):
		use_sts = carg.replace("use_safetensors=", "", 1)
		if use_sts.lower() == "true" or use_sts == "1":
			use_safetensors = True
		else:
			use_safetensors = False
	elif carg.startswith("gguf_file="):
		gguf_file = carg.replace("gguf_file=", "", 1).replace("\\n", "\n")
	elif carg.startswith("im_use_safetensors="):
		use_sts = carg.replace("im_use_safetensors=", "", 1)
		if use_sts.lower() == "true" or use_sts == "1":
			im_use_safetensors = True
		else:
			im_use_safetensors = False
	elif carg.startswith("load_in_8bit="):
		load_8bit = carg.replace("load_in_8bit=", "", 1)
		if load_8bit.lower() == "true" or load_8bit == "1":
			load_in_8bit = True
	elif carg.startswith("im_load_in_8bit="):
		load_8bit = carg.replace("im_load_in_8bit=", "", 1)
		if load_8bit.lower() == "true" or load_8bit == "1":
			im_load_in_8bit = True
	elif carg.startswith("custom_model_code="):
		custom_code = carg.replace("custom_model_code=", "", 1)
		if custom_code.lower() == "true" or custom_code == "1":
			custom_model_code = True
	elif carg.startswith("custom_token_code="):
		custom_code = carg.replace("custom_token_code=", "", 1)
		if custom_code.lower() == "true" or custom_code == "1":
			custom_token_code = True
	elif carg.startswith("im_from_single_file="):
		from_single_file = carg.replace("im_from_single_file=", "", 1)
		if from_single_file.lower() == "true" or from_single_file == "1":
			im_from_single_file = True
	elif carg.startswith("im_local_files="):
		from_local_files = carg.replace("im_local_files=", "", 1)
		if from_local_files.lower() == "true" or from_local_files == "1":
			im_local_files = True
	elif carg.startswith("im_cpu_offload="):
		cpu_offload = carg.replace("im_cpu_offload=", "", 1)
		if cpu_offload.lower() == "false" or cpu_offload == "0":
			im_cpu_offload = False
	elif carg.startswith("im_att_slicing="):
		att_slicing = carg.replace("im_att_slicing=", "", 1)
		if att_slicing.lower() == "false" or att_slicing == "0":
			im_att_slicing = False
		else:
			im_att_slicing = True
	elif carg.startswith("im_safety_check="):
		safety_check = carg.replace("im_safety_check=", "", 1)
		if safety_check.lower() == "false" or safety_check == "0":
			im_safety_check = False
	elif carg.startswith("use_chat_template="):
		use_chat_temp = carg.replace("use_chat_template=", "", 1)
		if use_chat_temp.lower() == "true" or use_chat_temp == "1":
			use_chat_template = True
		use_tool_config = False
	elif carg.startswith("use_tool_template="):
		use_tool_temp = carg.replace("use_tool_template=", "", 1)
		if use_tool_temp.lower() == "true" or use_tool_temp == "1":
			use_tool_template = True
			use_chat_template = True
		use_tool_config = False
	elif carg.startswith("skip_special_tokens="):
		skip_spec_tkns = carg.replace("skip_special_tokens=", "", 1)
		if skip_spec_tkns.lower() == "false" or skip_spec_tkns == "0":
			skip_special_tokens = False
	elif carg.startswith("cleanup_token_space="):
		clean_spaces = carg.replace("cleanup_token_space=", "", 1)
		if clean_spaces.lower() == "false" or clean_spaces == "0":
			cleanup_token_space = False
	elif carg.startswith("torch_compile="):
		compile_torch = carg.replace("torch_compile=", "", 1)
		if compile_torch.lower() == "true" or compile_torch == "1":
			torch_compile = True
	elif carg.startswith("chunk_audio_secs="):
		chunk_audio_secs = int(carg.replace("chunk_audio_secs=", "", 1))
	elif carg.startswith("reserve_vram_mb="):
		ext_res_vram_mb = int(carg.replace("reserve_vram_mb=", "", 1))
		reserve_vram_mb = min_res_vram_mb + ext_res_vram_mb
	elif carg.startswith("min_res_vram_mb="):
		min_res_vram_mb = int(carg.replace("min_res_vram_mb=", "", 1))
		reserve_vram_mb = min_res_vram_mb + ext_res_vram_mb

def ReserveVRAM():
	global reserved_vram
	if reserve_vram_mb <= 0 or reserved_vram != None: return
	if device == "auto" and torch.cuda.is_available():
		reserved_vram = torch.cuda.FloatTensor(256,1024,reserve_vram_mb)
			
def CleanVRAM():
	global reserved_vram
	reserved_vram = None
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		
def CountFiles(dir_path):
	count = 0
	for path in os.listdir(dir_path):
		if os.path.isfile(os.path.join(dir_path, path)):
			count += 1
	return count

def NewFileName(dir_path, name_str):
	if not os.path.exists(dir_path): os.mkdir(dir_path)
	file_num = CountFiles(dir_path)
	file_name = name_str.replace("#NUM", str(file_num))
	while os.path.isfile(dir_path+file_name):
		file_num += 1
		file_name = name_str.replace("#NUM", str(file_num))
	return { 'num': file_num, 'file': dir_path+file_name }
	
def SetWorkingDir(face_anim_mode):
	if face_anim_mode == 0:
		os.chdir(work_dir + '/MakeItTalk/')
	elif face_anim_mode == 1:
		os.chdir(work_dir + '/Wav2Lip/')
	else:
		os.chdir(work_dir + '/SadTalker/')

def ReadTextFile(txt_file, txt_encoding='utf-8'):
	f = open(txt_file, "rb")
	result = f.read().decode(txt_encoding)
	f.close()
	return result

def WriteTextFile(txt_file, txt_str, txt_encoding='utf-8'):
	f = open(txt_file, "wb")
	f.write(txt_str.encode(txt_encoding))
	f.close()

def SafeString(txt):
	result = txt
	unsafe_chars = {
		'#': 'AI_UI_AMPnum;',
		'$': 'AI_UI_AMPdollar;',
		'+': 'AI_UI_AMPplus;',
		'-': 'AI_UI_AMPminus;',
		'~': 'AI_UI_AMPtilde;',
		'.': 'AI_UI_AMPperiod;',
		'*': 'AI_UI_AMPast;',
		'`': 'AI_UI_AMPgrave;',
		'!': 'AI_UI_AMPexcl;',
		'|': 'AI_UI_AMPverbar;',
		'(': 'AI_UI_AMPlpar;',
		')': 'AI_UI_AMPrpar;',
		'[': 'AI_UI_AMPlsqb;',
		']': 'AI_UI_AMPrsqb;',
		'<': 'AI_UI_AMPlt;',
		'>': 'AI_UI_AMPgt;'
	}
	result = result.replace('_', 'AI_UI_AMPlowbar;')
	for unsafe_char, safe_str in unsafe_chars.items():
		result = result.replace(unsafe_char, safe_str)
	return result

def StripEnd(txt, ss):
	return txt[:-len(ss)] if txt.endswith(ss) else txt

def StripStart(txt, ss):
	return txt[len(ss):] if txt.startswith(ss) else txt

def LoadToolFuncs():
	global tool_funcs, tool_func
	if enable_tooluse:
		if tool_mode == "roleplay":
			tool_module = "tools.rp_funcs"
		elif tool_mode == "research":
			tool_module = "tools.search_funcs"
		elif tool_mode == "math":
			tool_module = "tools.math_funcs"
		elif tool_mode == "merge":
			tool_module = "tools.merge_funcs"
		else:
			tool_module = "tools.tool_funcs"
		tool_funcs = getattr(importlib.import_module(tool_module), "GetToolFuncs")()
		tool_func = getattr(importlib.import_module(tool_module), "CallToolFunc")
		tools_str = ''
		for f in tool_funcs:
			tools_str += json.dumps(transformers.utils.get_json_schema(f))+','
		print("TOOL_FUNCS:"+tools_str.rstrip(',').replace("\n", "[AI_UI_BR]"), flush=True)
		time.sleep(0.1)

###### TTS Stuff ######

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

print("VOICE_CONFIG:")
voice_key = int(input())
voice_vol = float(input())
voice_rate = int(input())
voice_ps = int(input())
talk_mode = int(input())
tts_mode = int(input())
anim_mode = int(input())

last_smt_val = smodel_type

SetWorkingDir(anim_mode)
	
def LoadSpeaker(voice_index):
	result = None
	
	if smodel_type == 0:
		result = torch.tensor(np.load(work_dir+"/embeddings/"+ai_voices[voice_index])).unsqueeze(0).to(comp_dev)
	elif smodel_type == 1:
		if voice_index > 0:
			result = ReadTextFile(work_dir+"/embeddings/ChatTTS/"+ai_voices[voice_index])
	
	return result
	
def LoadSysVoices():
	global voices
	
	voices = synthesizer.getProperty('voices')
	voice_list = 'SYS_VOICES:'
	
	for voice in voices:
		voice_list += "VOICE_NAME:"+voice.name+"VOICE_ID:"+voice.id+"\n"

	if voice_list != 'SYS_VOICES:':
		print(voice_list, flush=True)
		time.sleep(0.1)

def LoadAIVoices(speaker_name=''):
	global ai_voices
	speaker_index = 0

	if smodel_type == 0:
		ai_voices = glob.glob1(work_dir+"/embeddings/", '*.npy')
		ai_voices.sort()
	elif smodel_type == 1:
		ai_voices = glob.glob1(work_dir+"/embeddings/ChatTTS/", '*.txt')
		ai_voices.sort()
		ai_voices.insert(0, "Random")
	elif smodel_type == 2:
		ai_voices = ReadTextFile(work_dir+"/embeddings/Kokoro/voices.txt").replace("\r", '').split("\n")
		ai_voices.sort()
	
	voice_list = 'AI_VOICES:'
	
	for i in range(0,len(ai_voices)):
		if ai_voices[i] == speaker_name: speaker_index = i
		voice_list += "VOICE_NAME:"+ai_voices[i]+"VOICE_ID:"+str(i)+"\n"
	
	if voice_list != 'AI_VOICES:':
		print(voice_list, flush=True)
		time.sleep(0.1)
		
	return speaker_index

def ConfigTTSEngine():
	global synthesizer, speaker_embeddings, tts_ready
	tts_ready = True
	
	if tts_mode == 0:
		if len(voices) > 0:
			synthesizer.setProperty('voice', voices[voice_key].id)
			synthesizer.setProperty('volume', voice_vol)
			synthesizer.setProperty('rate', voice_rate)
		else:
			print("ERROR: no system voices could be found")
			tts_ready = False
	else:
		if len(ai_voices) > 0:
				speaker_embeddings = LoadSpeaker(voice_key)
		else:
			print("ERROR: no voice embeddings could be found")
			tts_ready = False
	
def LoadTTSVoices():
	global synthesizer
	if synthesizer == None:
		synthesizer = pyttsx3.init()
	
	LoadSysVoices()
	LoadAIVoices()	
	ConfigTTSEngine()

def LoadTTSEngine(force_ai_mode=None):
	global tts_model, processor, vocoder, last_smt_val
	
	if (force_ai_mode == True or tts_mode > 0) and (tts_model == None or smodel_type != last_smt_val):
		if smodel_type == 0:
			if tts_model_id == '':
				processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
				tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(comp_dev)
			else:
				processor = SpeechT5Processor.from_pretrained(tts_model_id)
				tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_id).to(comp_dev)
			if vocoder_model == '':
				vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(comp_dev)
			else:
				vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model).to(comp_dev)
		elif smodel_type == 1:
			tts_model = ChatTTS.Chat()
			if tts_model_id == '':
				tts_model.load(compile=torch_compile, source="huggingface")
			else:
				tts_model.load(compile=torch_compile, source="custom", custom_path=tts_model_id)
		elif smodel_type == 2:
			if not kokoro_loaded:
				print("ERROR: failed to load Kokoro, ensure espeak-ng is installed on your system", flush=True)
				time.sleep(0.1)
				return
			tts_model = KModel()
	
	last_smt_val = smodel_type
	ConfigTTSEngine()

def ApplyTalkMode():
	global do_talk, do_anim
	if talk_mode == 0:
		do_talk = True
		do_anim = True
	elif talk_mode == 1:
		do_talk = True
		do_anim = False
	else:
		do_talk = False
		do_anim = False
		
def CloneVoice(clone_model, speaker_name, speaker_wav, transcript=None):
	global smodel_type
	embeddings = []
	signal = []
	
	try:
		if clone_model == 0:
			fs, signal = wavfile.read(speaker_wav)
			if fs != 16000:
				signal = ResampleAudio(fs, signal, 16000)
			else:
				signal = torch.tensor(signal)
		else:
			if transcript == None or transcript == '':
				return 'ERROR:Transcript text is required!'
			signal = load_audio(speaker_wav, 24000)
			#if fs != 24000:
			#	signal = load_audio(speaker_wav, 24000)
			#else:
			#	signal, fs = load_audio(speaker_wav)
	except:
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
			if tts_mode != 1 or smodel_type != 1 or tts_model == None or smodel_type != last_smt_val:
				lsmt = smodel_type
				smodel_type = 1
				LoadTTSEngine(True)
				smodel_type = lsmt
			
			embeddings = tts_model.sample_audio_speaker(signal)
	except:
		return 'ERROR:There was an error with the cloning model.'
			
	try:
		if len(embeddings) > 0:
			if clone_model == 0:
				np.save(work_dir+"/embeddings/"+speaker_name+".npy", embeddings)
				voice_id = LoadAIVoices(speaker_name+'.npy')
			else:
				#ref_txt = tts_model.infer(transcript, refine_text_only=True)
				ts_dir = work_dir+"/embeddings/ChatTTS/transcripts/"
				if not os.path.exists(ts_dir): os.mkdir(ts_dir)
				WriteTextFile(ts_dir+speaker_name+".txt", transcript)
				WriteTextFile(work_dir+"/embeddings/ChatTTS/"+speaker_name+".txt", embeddings)
				voice_id = LoadAIVoices(speaker_name+'.txt')
		else:
			return 'ERROR:There was an error generating the embeddings.'
	except:
		return 'ERROR:There was an error saving the voice file.'
	
	return speaker_name+':'+str(voice_id)

def ModifySound(sound_wav, vcvol, vcrate, vcpitch):
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
	LoadTTSEngine(True)
	if voice_index == None: voice_index = voice_key
	text_chunks = []
	wav_files = []
	
	if len(text_str) > 256:
		text_chunks = SentenceSplit(text_str)
	else:
		text_chunks = [text_str]
	
	for i, txt in enumerate(text_chunks):
		wav_part_file = StripEnd(wav_file, ".wav") + "_chunk"+str(i)+".wav"
		if smodel_type == 0:
			inputs = processor(text=txt, return_tensors="pt").to(comp_dev)
			speech = tts_model.generate_speech(inputs["input_ids"], speaker_data, vocoder=vocoder)
			sf.write(wav_part_file, speech.cpu().numpy(), samplerate=16000)
			wav_files.append(wav_part_file)
		elif smodel_type == 1:
			if speaker_data == None:
				embeddings = tts_model.sample_random_speaker()
				WriteTextFile(work_dir+"/embeddings/ChatTTS/random.tmp", embeddings)
				params_ic = ChatTTS.Chat.InferCodeParams(spk_emb = embeddings)
			else:
				ts_file = work_dir+"/embeddings/ChatTTS/transcripts/" + ai_voices[voice_index]
				if os.path.isfile(ts_file):
					params_ic = ChatTTS.Chat.InferCodeParams(spk_smp=speaker_data, txt_smp=ReadTextFile(ts_file))
				else:
					params_ic = ChatTTS.Chat.InferCodeParams(spk_emb = speaker_data)
			speech = tts_model.infer(txt+" [uv_break]", skip_refine_text=True, params_infer_code=params_ic)
			sf.write(wav_part_file, speech[0], samplerate=24000)
			wav_files.append(wav_part_file)
		elif smodel_type == 2 and kokoro_loaded:
			tts_pipeline = KPipeline(lang_code=ai_voices[voice_index][0], model=tts_model)
			tts_generator = tts_pipeline(txt, voice=ai_voices[voice_index], split_pattern='')
			for i, (gs, ps, speech) in enumerate(tts_generator):
				sf.write(wav_part_file, speech, samplerate=24000)
				wav_files.append(wav_part_file)
				break
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
		sys.exit("ERROR: failed to generate speech audio")
	
	ModifySound(wav_file, speech_vol, speech_rate, speech_ps)

def TextToSpeechSys(text_str, wav_file, speech_ps):
	synthesizer.save_to_file("<pitch middle='"+str(speech_ps)+"'/>"+text_str, wav_file)
	synthesizer.runAndWait()
	AddEpsilonNoise(wav_file)
	
###### SPEECH RECOGNITION STUFF ######

sr_model = None
sr_pipe = None
sr_processor = None

def LoadSRProcessor():
	global sr_processor
	sr_processor = AutoProcessor.from_pretrained(sr_model_id, trust_remote_code=custom_token_code)

def LoadSRModel():
	global sr_model, sr_pipe
	try:
		if sr_processor == None: LoadSRProcessor()
		
		if speech_recog_class != None:
			model_loader = getattr(importlib.import_module("transformers"), speech_recog_class)
		else:
			model_loader = AutoModelForSpeechSeq2Seq
		
		sr_torch_dtype = torch.float16 if comp_dev == "cuda" else torch.float32
		sr_device = None if device == "auto" else comp_dev
			
		if device == "auto":
			sr_model = model_loader.from_pretrained(sr_model_id, torch_dtype=sr_torch_dtype, attn_implementation=attn_implementation, use_safetensors=use_safetensors, trust_remote_code=custom_model_code, device_map="auto")
		elif device == "cuda":
			sr_model = model_loader.from_pretrained(sr_model_id, torch_dtype=sr_torch_dtype, attn_implementation=attn_implementation, use_safetensors=use_safetensors, trust_remote_code=custom_model_code).to(device)
		else:
			sr_model = model_loader.from_pretrained(sr_model_id, torch_dtype=sr_torch_dtype, attn_implementation=attn_implementation, low_cpu_mem_usage=True, use_safetensors=use_safetensors, trust_remote_code=custom_model_code).to(device)

		sr_pipe = pipeline(
			"automatic-speech-recognition",
			model=sr_model,
			tokenizer=sr_processor.tokenizer,
			feature_extractor=sr_processor.feature_extractor,
			chunk_length_s=chunk_audio_secs,
			torch_dtype=sr_torch_dtype,
			device=sr_device
		)
	except Exception as e:
		sys.exit(f"ERROR: failed to load speech recognition model ({e})")

def ResampleSpeech(origin_audio, origin_sr, new_sr):
	new_samps = int(len(origin_audio) * new_sr/origin_sr)
	return resample(origin_audio, new_samps)

def SpeechToText(speech_file):
	if sr_model == None: LoadSRModel()
	sr, signal = wavfile.read(speech_file)
	try:
		if sr != sr_processor.feature_extractor.sampling_rate:
			signal = ResampleSpeech(signal, sr, sr_processor.feature_extractor.sampling_rate)
	except:
		print("ERROR: failed to resample audio recording", flush=True)
		time.sleep(0.1)
	result = sr_pipe(signal)
	return result["text"]

###### TEXT MODEL STUFF ######

transformers.logging.set_verbosity_error()
#set_seed(int(str(time.time()).replace('.', '')))

model = None
tokenizer = None

def LoadTokenizer():
	global tokenizer

	try:
		if model_type == 2:
			if tokenizer_model == '':
				tokenizer = AutoProcessor.from_pretrained(model_id, gguf_file=gguf_file, trust_remote_code=custom_token_code)
			else:
				tokenizer = AutoProcessor.from_pretrained(tokenizer_model, gguf_file=gguf_file, trust_remote_code=custom_token_code)
		else:
			if tokenizer_model == '':
				tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file, trust_remote_code=custom_token_code)
			else:
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, gguf_file=gguf_file, trust_remote_code=custom_token_code)
	except Exception as e:
		sys.exit(f"ERROR: failed to load text tokenizer ({e})")

def LoadModel():
	global model
	#reserve VRAM for other models
	ReserveVRAM()
	
	if tokenizer == None: LoadTokenizer()
	
	model_loader = AutoModelForCausalLM
	
	if model_type == 0:
		if text_lang_class != None:
			model_loader = getattr(importlib.import_module("transformers"), text_lang_class)
	elif model_type == 1:
		model_loader = AutoModel
	elif model_type == 2:
		if multi_modal_class != None:
			model_loader = getattr(importlib.import_module("transformers"), multi_modal_class)
		else:
			model_loader = AutoModelForImageTextToText
	
	if device == "auto":
		model = model_loader.from_pretrained(model_id, gguf_file=gguf_file, torch_dtype=torch_dtype, attn_implementation=attn_implementation, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, trust_remote_code=custom_model_code, device_map="auto")
	elif device == "cuda":
		model = model_loader.from_pretrained(model_id, gguf_file=gguf_file, torch_dtype=torch_dtype, attn_implementation=attn_implementation, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, trust_remote_code=custom_model_code).to(device)
	else:
		model = model_loader.from_pretrained(model_id, gguf_file=gguf_file, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation=attn_implementation, use_safetensors=use_safetensors, trust_remote_code=custom_model_code).to(device)
	
	if model_adapter != '':
		model.load_adapter(model_adapter)

	model.eval()

	if torch.__version__ >= "2" and torch_compile:
		model = torch.compile(model, mode=compile_mode)

class TextGenThread(threading.Thread):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.exception = None
		self.result = None

	def run(self):
		try:
			with torch.no_grad():
				self.result = self._target(*self._args, **self._kwargs)
		except Exception as e:
			self.exception = e

	def get_id(self):
		if hasattr(self, '_thread_id'): return self._thread_id
		for id, thread in threading._active.items():
			if thread is self: return id

	def force_kill(self):
		thread_id = self.get_id()
		res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
		if res > 1:
			ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
			sys.exit('ERROR: failed to kill TextGenThread')

class StoppingCriteriaKeys(StoppingCriteria):
	def __init__(self, keywords_ids:list):
		self.keywords = keywords_ids

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		if len(self.keywords) >= len(input_ids[0]): return False
		key_num = -len(self.keywords)
		for keyword in self.keywords:
			if input_ids[0][key_num] != keyword: return False
			key_num += 1
		return True

def GenStopStrings():
	global end_strings
	if end_strings != None: return
	end_strings = []
	if use_chat_template == True:
		if hasattr(tokenizer, 'eos_token') and isinstance(tokenizer.eos_token, str):
			if tokenizer.eos_token == "<|endoftext|>":
				end_strings.append(tokenizer.eos_token)
			else:
				end_strings += [tokenizer.eos_token, "<|endoftext|>"]
		else:
			end_strings.append("<|endoftext|>")
	else:
		if hasattr(tokenizer, 'eos_token') and isinstance(tokenizer.eos_token, str):
			if tokenizer.eos_token == "<|endoftext|>":
				end_strings += [tokenizer.eos_token, "\n"+user_name_c]
			else:
				end_strings += [tokenizer.eos_token, "<|endoftext|>", "\n"+user_name_c]
		else:
			end_strings += ["<|endoftext|>", "\n"+user_name_c]

def TokenizeChat():
	global rag_files
	if tokenizer == None: LoadTokenizer()
	if use_chat_template:
		if use_tool_template and tool_funcs != None:
			tool_use = "tool_use" if hasattr(tokenizer, "chat_template") and (not type(tokenizer.chat_template) is str) else None
			result = tokenizer.apply_chat_template(messages, tools=tool_funcs, chat_template=tool_use, add_generation_prompt=True, return_dict=True, return_tensors="pt")
			result = {k: v.to(comp_dev) for k, v in result.items()}
		elif len(rag_files) > 0 and len(chat_files) == 0:
			for rag_file in rag_files:
				if "text" in rag_file: continue
				try:
					resp_data = urlopen(rag_file["url"])
					if resp_data.code == 200:
						rag_file["text"] = resp_data.read().decode('utf-8')
				except:
					continue
			result = tokenizer.apply_chat_template(conversation=messages, documents=rag_files, chat_template="rag", tokenize=True, add_generation_prompt=True, return_tensors="pt").to(comp_dev)
		elif model_type == 2:
			result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
			vision_processor = process_vision_info
			if process_vision_func != None:
				lib_func = process_vision_func.split(".")
				if len(lib_func) == 2:
					vision_processor = getattr(importlib.import_module(lib_func[0]), lib_func[1])
				else:
					sys.exit("ERROR: process_vision_func has invalid value: "+process_vision_func)
			image_inputs, video_inputs = vision_processor(messages)
			result = tokenizer(
				text=[result],
				images=image_inputs,
				videos=video_inputs,
				padding=True,
				return_tensors="pt"
			).to(comp_dev)
		else:
			result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(comp_dev)
	else:
		if comp_dev == "cuda":
			result = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
		else:
			result = tokenizer(prompt, return_tensors="pt").input_ids
	
	return result

def StreamText(txt_streamer, gen_kwargs):
	global model
	
	gen_text = ""
	stream_state = "STREAMING"
	thread = TextGenThread(target=model.generate, kwargs=gen_kwargs)
	thread.start()
	time.sleep(0.5)
	
	if thread.exception:
		thread.join()
		return "ERROR: failed to start text stream thread"
	
	for new_text in txt_streamer:
		if thread.exception:
			print(f"ERROR: model failed to generate text ({thread.exception})", flush=True)
			break
		elif stream_state == "STREAMING":
			nobr_str = new_text.replace("\n", "[AI_UI_BR]").replace("\r", "")
			gen_text += nobr_str
			print("STREAM_WORD:"+nobr_str, flush=True)
		else:
			thread.force_kill()
			break
		stream_state = input()
	
	thread.join()
	
	time.sleep(0.1)
	print("[AIUI_STREAM_END]", flush=True)
	time.sleep(0.1)
	
	if thread.result == None:
		return gen_text
	else:
		return thread.result

def GenReply(it, rl):
	global model, tokenizer
	if model == None: LoadModel()
	CleanVRAM()
	GenStopStrings()
	
	if gen_mode == 0:
		txt_streamer = None
	else:
		print("TEXT_STREAM:", flush=True)
		txt_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
	
	if isinstance(it, torch.Tensor) or type(it) is list:
		gen_kwargs = dict(
			inputs=it, do_sample=do_sample, num_beams=num_beams, num_beam_groups=beam_groups,
			tokenizer=tokenizer, streamer=txt_streamer, min_new_tokens=gen_min, max_new_tokens=gen_max,
			temperature=rl, top_k=top_k, top_p=top_p, typical_p=typical_p, repetition_penalty=rep_penalty, stop_strings=end_strings
		)
	elif type(it) is dict:
		it['do_sample'] = do_sample
		it['num_beams'] = num_beams
		it['num_beam_groups'] = beam_groups
		it['tokenizer'] = tokenizer
		it['streamer'] = txt_streamer
		it['min_new_tokens'] = gen_min
		it['max_new_tokens'] = gen_max
		it['temperature'] = rl
		it['top_k'] = top_k
		it['top_p'] = top_p
		it['typical_p'] = typical_p
		it['repetition_penalty'] = rep_penalty
		it['stop_strings'] = end_strings
		gen_kwargs = it
	else:
		sys.exit("ERROR: unexpected input token format")
	
	if txt_streamer == None:
		with torch.no_grad():
			try:
				output = model.generate(**gen_kwargs)
			except:
				del gen_kwargs['stop_strings']
				output = model.generate(**gen_kwargs)
	else:
		output = StreamText(txt_streamer, gen_kwargs)
	
	CleanVRAM()
	ReserveVRAM()
	
	if type(output) is str:
		return output
	elif isinstance(it, torch.Tensor) or type(it) is list:
		return tokenizer.decode(output[0][len(it[0]):], skip_special_tokens=skip_special_tokens, clean_up_tokenization_space=cleanup_token_space)
	else:
		return tokenizer.decode(output[0][it["input_ids"].shape[-1]:], skip_special_tokens=skip_special_tokens, clean_up_tokenization_space=cleanup_token_space)

def GenText(it, nt, mt, ds, nb, ng, rl, tk, tp, ty, rp):
	global model, tokenizer
	if model == None: LoadModel()
	CleanVRAM()
	
	if gen_mode == 0:
		txt_streamer = None
	else:
		print("GEN_STREAM:", flush=True)
		txt_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
	
	gen_kwargs = dict(
		inputs=it, do_sample=ds, num_beams=nb, num_beam_groups=ng,
		tokenizer=tokenizer, streamer=txt_streamer, min_new_tokens=nt, max_new_tokens=mt,
		temperature=rl, top_k=tk, top_p=tp, typical_p=ty, repetition_penalty=rp
	)
	
	if txt_streamer == None:
		with torch.no_grad():
			output = model.generate(**gen_kwargs)
	else:
		output = StreamText(txt_streamer, gen_kwargs)
	
	CleanVRAM()
	ReserveVRAM()
	
	if type(output) is str:
		return output
	else:
		return tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)

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
	
def DiffIMGMode(safety_check, vae_file, lora_file, lora_dir, lora_scale):
	global last_sc_val, last_vf_val, last_lf_val, last_ld_val, last_ls_val
	if safety_check != last_sc_val or vae_file != last_vf_val or lora_file != last_lf_val or lora_dir != last_ld_val or lora_scale != last_ls_val:
		result = 2
	else:
		result = int(safety_check != im_safety_check or vae_file != im_vae_file or lora_file != im_lora_file or lora_dir != im_lora_dir or lora_scale != im_lora_scale)
	last_sc_val = safety_check
	last_vf_val = vae_file
	last_lf_val = lora_file
	last_ld_val = lora_dir
	last_ls_val = lora_scale
	return result

def LoadIMGModule(module_name, module_file, module_loader, module_list, bnb_config):
	if module_loader == None:
		sys.exit(f"ERROR: got unexpected model file ({module_file})")
	if module_file.lower().endswith(".gguf"):
		quant_config = GGUFQuantizationConfig(compute_dtype=im_torch_dtype)
		if module_name in module_list: module_list.remove(module_name)
	elif module_name in module_list:
		quant_config = bnb_config
		module_list.remove(module_name)
	else:
		quant_config = None
	return module_loader.from_single_file(
		module_file,
		torch_dtype = im_torch_dtype,
		local_files_only = im_local_files,
		quantization_config = quant_config
	)

def LoadIMGModel(exit_on_error=True, safety_check=None, vae_file=None, lora_file=None, lora_dir=None, lora_scale=None):
	global im_model, reserve_vram_mb
	orig_rvram = reserve_vram_mb
	im_model = None
	vae_model = None
	result = True
	
	safety_check = im_safety_check if safety_check == None else safety_check
	vae_file = im_vae_file if vae_file == None else vae_file
	lora_file = im_lora_file if lora_file == None else lora_file
	lora_dir = im_lora_dir if lora_dir == None else lora_dir
	lora_scale = im_lora_scale if lora_scale == None else lora_scale
	
	CleanVRAM()
	if reserve_vram_mb > min_res_vram_mb:
		reserve_vram_mb = min_res_vram_mb
	ReserveVRAM()

	if im_model_class != None:
		model_loader = getattr(importlib.import_module("diffusers"), im_model_class)
		ivae_loader = AutoencoderKL
		itran_loader = None
		ienc_loader = [CLIPTextModel, None, None]
		comps_to_quant = []
	elif imodel_type == 0:
		model_loader = StableDiffusionPipeline
		ivae_loader = AutoencoderKL
		itran_loader = None
		ienc_loader = [CLIPTextModel, None, None]
		comps_to_quant = ["unet"]
	elif imodel_type == 1:
		model_loader = StableDiffusionXLPipeline
		ivae_loader = AutoencoderKL
		itran_loader = None
		ienc_loader = [CLIPTextModel, CLIPTextModelWithProjection, None]
		comps_to_quant = ["unet", "text_encoder_2"]
	elif imodel_type == 2:
		model_loader = StableDiffusion3Pipeline
		ivae_loader = AutoencoderKL
		itran_loader = SD3Transformer2DModel
		ienc_loader = [CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel]
		comps_to_quant = ["transformer", "text_encoder_2", "text_encoder_3"]
	elif imodel_type == 3:
		model_loader = FluxPipeline
		ivae_loader = AutoencoderKL
		itran_loader = FluxTransformer2DModel
		ienc_loader = [CLIPTextModel, T5EncoderModel, None]
		comps_to_quant = ["transformer", "text_encoder_2"]
	else:
		model_loader = QwenImagePipeline
		ivae_loader = AutoencoderKLQwenImage
		itran_loader = QwenImageTransformer2DModel
		ienc_loader = [Qwen2_5_VLForConditionalGeneration, None, None]
		comps_to_quant = ["transformer", "text_encoder"]
	
	try:
		tbnb_config = TransformersBitsAndBytesConfig(load_in_8bit=True) if im_load_in_8bit else None
		dbnb_config = DiffusersBitsAndBytesConfig(load_in_8bit=True) if im_load_in_8bit else None

		gen_kwargs = dict(
			torch_dtype=im_torch_dtype,
			variant=im_variant, use_safetensors=im_use_safetensors,
			local_files_only=im_local_files, original_config_file=im_config_file
		)
		
		if im_enc_file != '':
			gen_kwargs["text_encoder"] = LoadIMGModule("text_encoder", im_enc_file, ienc_loader[0], comps_to_quant, tbnb_config)
		if im_enc2_file != '':
			gen_kwargs["text_encoder_2"] = LoadIMGModule("text_encoder_2", im_enc2_file, ienc_loader[1], comps_to_quant, tbnb_config)
		if im_enc3_file != '':
			gen_kwargs["text_encoder_3"] = LoadIMGModule("text_encoder_3", im_enc3_file, ienc_loader[2], comps_to_quant, tbnb_config)
		
		if vae_file != '':
			gen_kwargs["vae"] = LoadIMGModule("vae", vae_file, ivae_loader, comps_to_quant, dbnb_config)
		
		if im_unet_file != '':
			gen_kwargs["unet"] = LoadIMGModule("unet", im_unet_file, UNet2DConditionModel, comps_to_quant, dbnb_config)

		elif im_tran_file != '':
			if itran_loader == None:
				if im_model_class == None:
					sys.exit(f"ERROR: {model_loader.__name__} does not use a transformer model")
				else:
					sys.exit("ERROR: cannot use im_tran_file with im_model_class")
			gen_kwargs["transformer"] = LoadIMGModule("transformer", im_tran_file, itran_loader, comps_to_quant, dbnb_config)
		
		if not safety_check: gen_kwargs["safety_checker"] = None
		
		if im_load_in_8bit and len(comps_to_quant) > 0:
			gen_kwargs["quantization_config"] = PipelineQuantizationConfig(
				quant_backend = "bitsandbytes_8bit",
				quant_kwargs = { "load_in8bit": True },
				components_to_quantize = comps_to_quant
			)
		
		if im_from_single_file or im_model_id.endswith(".safetensors") or im_model_id.endswith(".ckpt"):
			im_model = model_loader.from_single_file(im_model_id, **gen_kwargs)
		else:
			im_model = model_loader.from_pretrained(im_model_id, **gen_kwargs)
		
		if device == "auto":
			if torch.cuda.is_available() and not im_cpu_offload:
				im_model = im_model.to("cuda")
		else:
			im_model = im_model.to(device)
		
		if lora_file != '':
			im_model.load_lora_weights(lora_dir, weight_name=lora_file)
			im_model.fuse_lora(lora_scale=lora_scale)
		
		if torch_compile:
			im_model.vae.to(memory_format=torch.channels_last)
			im_model.vae.decode = torch.compile(im_model.vae.decode, mode=compile_mode)
			if hasattr(im_model, "transformer"):
				im_model.transformer.to(memory_format=torch.channels_last)
				im_model.transformer = torch.compile(im_model.transformer, mode=compile_mode)
			elif hasattr(im_model, "unet"):
				im_model.unet.to(memory_format=torch.channels_last)
				im_model.unet = torch.compile(im_model.unet, mode=compile_mode)
		
		if im_cpu_offload:
			im_model.enable_model_cpu_offload()
			
		if im_att_slicing:
			im_model.enable_attention_slicing("max")
		
	except Exception as e:
		reserve_vram_mb = orig_rvram
		im_model = None
		result = False
		if exit_on_error:
			sys.exit(f"ERROR: failed to load image model ({e})")
		else:
			print(f"ERROR: failed to load image model ({e})", flush=True)
			time.sleep(0.1)
	
	CleanVRAM()
	ReserveVRAM()
	
	return result

def GenImage(image_prompt, neg_prompt="NONE", infer_steps=50, guidance=7.5, img_seed="NONE", img_width="auto", img_height="auto", igen_mode=False):
	global im_model, img_gen_mode, orig_shdlr

	if image_prompt == '': return "ERROR: Invalid prompt text."
	
	CleanVRAM()

	while True:
		if im_model == None or (igen_mode == False and img_gen_mode == True):
			img_gen_mode = False
			if not LoadIMGModel(False):
				result = "ERROR: There was an error loading the image generation model."
				break
			
		try:
			if orig_shdlr != None:
				im_model.unload_ip_adapter()
				im_model.scheduler = orig_shdlr
				orig_shdlr = None
			
			clean_prompt = image_prompt.strip(' ').replace("\n", ' ').replace("\r", '')
			if neg_prompt == "NONE": neg_prompt = default_neg_prompt
			
			gen_kwargs = dict(
				prompt=clean_prompt, negative_prompt=neg_prompt,
				num_inference_steps=infer_steps, guidance_scale=guidance
			)
			
			if img_width != "auto" and img_height != "auto":
				gen_kwargs["width"] = int(img_width)
				gen_kwargs["height"] = int(img_height)
			
			if img_seed != "NONE":
				gen_kwargs["generator"] = seed_gen.manual_seed(int(img_seed))

			with torch.inference_mode(): output = im_model(**gen_kwargs)
		
		except:
			result = "ERROR: There was an error generating the image with your settings."
			break

		try:
			result = NewFileName(work_dir+"/ai_images/", "image_#NUM.png")
			result = result["file"]
			output.images[0].save(result)
		except:
			result = "ERROR: Failed to save the image. Run the app as an admin."
		
		break

	CleanVRAM()
	ReserveVRAM()
	
	return result

def GenImageIPA(image_prompt, guide_image=None, is_person=False):
	global im_model, orig_shdlr
	
	if image_prompt == '': return "ERROR: prompt is empty"
	
	if is_person:
		neg_prompt = default_neg_prompt + ', ' + person_neg_prompt
	else:
		neg_prompt = default_neg_prompt
	
	if guide_image == None:
		return GenImage(image_prompt, neg_prompt)

	CleanVRAM()
	
	if im_model == None and not LoadIMGModel(False):
		return "ERROR: failed to load image model"
	
	if orig_shdlr == None:
		try:
			if is_person:
				ipa_model_id = "h94/IP-Adapter"
				ipa_subfolder = "models"
				ipa_ckpt = "ip-adapter-full-face_sd15.safetensors"
				if imodel_type == 1:
					#ipa_subfolder = "sdxl_models"
					#ipa_ckpt = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
					return "ERROR: SDXL Face IP-Adapter not yet supported"
				elif imodel_type > 1:
					return "ERROR: Face IP-Adapter not yet supported"
			else:
				ipa_model_id = "h94/IP-Adapter"
				ipa_subfolder = "models"
				ipa_ckpt = "ip-adapter_sd15.safetensors"
				if imodel_type == 1:
					ipa_subfolder = "sdxl_models"
					ipa_ckpt = "ip-adapter_sdxl.safetensors"
				elif imodel_type > 1:
					return "ERROR: IP-Adapter not yet supported"
			
			orig_shdlr = im_model.scheduler
			im_model.scheduler = DDIMScheduler.from_config(im_model.scheduler.config)
			im_model.load_ip_adapter(ipa_model_id, subfolder=ipa_subfolder, weight_name=ipa_ckpt)
			im_model.set_ip_adapter_scale(0.7)
			if im_cpu_offload: im_model.enable_model_cpu_offload()
		except:
			return "ERROR: failed to load IP-Adapter"

	try:
		clean_prompt = image_prompt.strip(' ').replace("\n", ' ').replace("\r", '')
		with torch.inference_mode():
			ip_image = load_image(guide_image)
			output = im_model(prompt=clean_prompt, negative_prompt=neg_prompt, num_inference_steps=50, ip_adapter_image=ip_image)
	except:
		return "ERROR: failed to generate image"

	try:
		result = NewFileName(work_dir+"/ai_images/", "image_#NUM.png")
		output.images[0].save(result["file"])
	except:
		return "ERROR: failed to save image"

	CleanVRAM()
	ReserveVRAM()
	
	return result["file"]

###### CHAT STUFF ######
	
def GetUserNames():
	global user_name, bot_name, user_name_c, bot_name_c, end_strings
	
	print("YOUR_NAME:")
	user_name = input()
	print("BOT_NAME:")
	bot_name = input()
	
	if use_chat_template:
		user_name_c = user_name.rstrip()
		bot_name_c = bot_name.rstrip()
	else:
		user_name_c = user_name+roles_append_str.rstrip()
		bot_name_c = bot_name+roles_append_str.rstrip()

	if tokenizer != None:
		GenStopStrings()
	else:
		end_strings = None

def InitPrompt(have_prompt=False):
	global messages, prompt, init_prompt, redo_count
	redo_count = 0
	if have_prompt and init_prompt != '':
		prompt = init_prompt
	else:
		prompt = 'Chat log between '+user_name+' and '+bot_name+' on {dt.month}/{dt.day}/{dt.year}'.format(dt = datetime.utcnow())
		init_prompt = prompt
	messages = [{"role":"system", "content":init_prompt}]
	if prompt_p == 2:
		prompt = ''
		init_prompt = 'No prompt'
		messages = []
	if have_prompt:
		print("CLEAR_DONE:"+init_prompt.replace("\n", "[AI_UI_BR]"), flush=True)
	else:
		print("INIT_DONE:"+init_prompt.replace("\n", "[AI_UI_BR]"), flush=True)
	time.sleep(0.1)
	
def PrunePrompt():
	global messages, prompt
	i = len(messages) - msg_mem
	if i > 0: messages = messages[i::]
	prompt = ''
	i = 0
	l = len(messages)
	if prompt_p == 1 and len(messages) >= msg_mem:
		if "role" in messages[0] and messages[0]["role"].lower() != 'system':
			messages[0] =  {"role":"system", "content":init_prompt}
	while i < l and "role" in messages[i] and "content" in messages[i]:
		if isinstance(messages[i]["content"], str):
			prompt += messages[i]["role"] + roles_append_str + messages[i]["content"] + "\n"
		else:
			prompt += messages[i]["role"] + roles_append_str
			for msg_con in messages[i]["content"]:
				if "user" in msg_con:
					prompt += messages[i]["role"] + roles_append_str + msg_con["text"] + "\n"
				else:
					for msg_key in msg_con:
						if msg_key != "type":
							prompt += msg_con[msg_key] + "\n"
							break
		i += 1
	prompt = prompt.rstrip("\n")
	if prompt_p < 2 and prompt == '': prompt = init_prompt

def AnimAvatar():
	CleanVRAM()
	if anim_mode == 0:
		result = AnimFace(chat_wav)
	elif anim_mode == 1:
		result = anim_face(chat_wav)
	else:
		result = animface(chat_wav)
	CleanVRAM()
	ReserveVRAM()
	return result
	
def LoadAvatar():
	CleanVRAM()
	if anim_mode == 0:
		result = LoadFace(avatar_img)
	elif anim_mode == 1:
		result = load_face(avatar_img)
	else:
		result = loadface(avatar_img)
	CleanVRAM()
	ReserveVRAM()
	return result

def LoadModels():
	if start_meth == "text":
		LoadModel()
	elif start_meth == "image":
		LoadIMGModel()
	elif start_meth == "speech":
		LoadTTSEngine()
	elif start_meth == "sr":
		LoadSRModel()
	elif start_meth == "all":
		LoadTTSEngine()
		LoadSRModel()
		LoadModel()
		LoadIMGModel()

LoadToolFuncs()
LoadTTSVoices()
LoadModels()
LoadAvatar()
ApplyTalkMode()
GetUserNames()
InitPrompt()

while (True):

	time.sleep(0.1)
	print("HUMAN_INPUT:", flush=True)
	msg = input().replace("[AI_UI_BR]", "\n").strip(" \n").replace("\r", '')

	if (msg == ''):
		continue
	elif msg == "close_chat":
		break
	elif msg == "new_game":
		print("NEW_PROMPT:")
		init_prompt = input()
		InitPrompt(True)
		NewGame()
		continue
	elif msg == "save_game":
		print("SAVE_FILE:")
		SaveGame(input(), messages, init_prompt)
		continue
	elif msg == "load_game":
		print("SAVE_FILE:")
		save_data = LoadGame(input())
		init_prompt = save_data['sys_prompt']
		InitPrompt(True)
		messages = save_data['messages']
		PrunePrompt()
		if len(messages) > 0:
			last_msg = messages[-1]
			if "role" in last_msg and last_msg['role'] == bot_name:
				if "content" in last_msg and isinstance(last_msg["content"], str):
					print("BOT_NOANIM:"+last_msg["content"].replace("\n", "[AI_UI_BR]"), flush=True)
				else:
					print("BOT_NOANIM:Game loaded.", flush=True)
			else:
				print("BOT_NOANIM:Game loaded. WARNING: Roles have changed!", flush=True)
		else:
			print("BOT_NOANIM:Game loaded.", flush=True)
		time.sleep(0.1)
		continue
	elif msg == "print_prompt":
		print('----- START PROMPT -----')
		print(prompt)
		print('------ END PROMPT ------')
		time.sleep(0.1)
		continue
	elif msg == "update_prompt":
		print("NEW_PROMPT:")
		init_prompt = input().replace("[AI_UI_BR]", "\n").replace("\r", '')
		if prompt_p < 2:
			PrunePrompt()
			if len(messages) > 0:
				messages[0] = {"role":"system", "content":init_prompt}
			else:
				messages = [{"role":"system", "content":init_prompt}]
		PrunePrompt()
		continue
	elif msg == "clear_chat":
		InitPrompt(True)
		rando_lvl = rando_min
		continue
	elif msg == "attach_docs":
		print("CHAT_FILES:")
		file_names = input().split("[AI_UI_BR]")
		rag_files = []
		for file_name in file_names:
			if file_name.strip() == '': continue
			try:
				if file_name.startswith("http:") or file_name.startswith("https:"):
					rag_files.append({"title": file_name,  "url": file_name})
				else:
					rag_files.append({"title": file_name,  "text": ReadTextFile(file_name)})
			except:
				print("ERROR: failed to read text from "+file_name, flush=True)
		time.sleep(0.1)
		continue
	elif msg == "attach_files":
		print("CHAT_FILES:")
		file_names = input().replace("\\", "/").split("[AI_UI_BR]")
		chat_files = []
		for file_name in file_names:
			if file_name.strip() == '': continue
			lfn = file_name.lower()
			file_text = ''
			if lfn.endswith(".jpg") or lfn.endswith(".jpeg") or lfn.endswith(".png") or lfn.endswith(".bmp") or lfn.endswith(".gif") or lfn.endswith(".webp") or lfn.endswith(".tiff"):
				file_type = "image"
			elif lfn.endswith(".mp4") or lfn.endswith(".avi") or lfn.endswith(".mkv") or lfn.endswith(".wmv") or lfn.endswith(".webm") or lfn.endswith(".mov") or lfn.endswith(".flv"):
				file_type = "video"
			elif lfn.endswith(".mp3") or lfn.endswith(".wav") or lfn.endswith(".ogg") or lfn.endswith(".wma") or lfn.endswith(".flac") or lfn.endswith(".acc") or lfn.endswith(".m4a"):
				file_type = "audio"
			else:
				file_type = "text"
				try:
					file_text = ReadTextFile(file_name)
				except:
					file_text = ''
					file_type = "file"
			if file_text == '':
				if file_name.startswith("http:") or file_name.startswith("https:"):
					chat_files.append({"file":file_name, "type":file_type, "content":"url"})
				elif os.path.isfile(file_name):
					chat_files.append({"file":file_name, "type":file_type, "content":file_type})
			else:
				chat_files.append({"file":file_name, "type":file_type, "content":file_text})
		continue
	elif msg == "config_voice":
		print("VOICE_CONFIG:")
		last_amode = anim_mode
		voice_key = int(input())
		voice_vol = float(input())
		voice_rate = int(input())
		voice_ps = int(input())
		talk_mode = int(input())
		tts_mode = int(input())
		anim_mode = int(input())
		SetWorkingDir(anim_mode)
		if anim_mode != last_amode:
			if LoadAvatar():
				print("GOT_AVATAR:true", flush=True)
			else:
				print("GOT_AVATAR:false", flush=True)
			time.sleep(0.1)
		ConfigTTSEngine()
		continue
	elif msg == "config_ai":
		print("AI_CONFIG:")
		last_pp = prompt_p
		msg_mem = int(input())
		gen_max = int(input())
		gen_min = int(input())
		do_sample = int(input())
		num_beams = int(input())
		beam_groups = int(input())
		rando_lvl = float(input())
		prompt_p = int(input())
		top_k = int(input())
		top_p = float(input())
		typical_p = float(input())
		rep_penalty = float(input())
		rando_min = rando_lvl
		if prompt_p != last_pp:
			if init_prompt == 'No prompt': InitPrompt()
			if last_pp == 0:
				if len(messages) > 0 and messages[0]["role"] == "system":
					messages = messages[1::]
			if prompt_p == 0:
				if len(messages) > 0:
					messages[0] = {"role":"system", "content":init_prompt}
				else:
					messages = [{"role":"system", "content":init_prompt}]
		PrunePrompt()
		continue
	elif msg == "config_other":
		print("OTHER_CONFIG:")
		gen_mode = int(input())
		format_mode = int(input())
		enable_tools = int(input())
		hf_token = input()
		if hf_token != hf_access_token:
			hf_access_token = hf_token
			if hf_token != '': login(token=hf_token)
		if enable_tools != enable_tooluse:
			enable_tooluse = enable_tools
			LoadToolFuncs()
		continue
	elif msg == "config_tools":
		print("TOOL_CONFIG:")
		end_strings = None
		temp_form = input()
		tool_mode = input()
		if use_tool_config:
			use_chat_template = True if temp_form != "none" else False
			use_tool_template = True if temp_form == "tool" else False
		LoadToolFuncs()
		continue
	elif msg == "gen_image":
		print("IMAGE_PROMPT:")
		img_prompt = input()
		neg_prompt = input()
		infer_steps = int(input())
		guidance = float(input())
		safety_check = int(input())
		img_seed = input()
		img_width = input()
		img_height = input()
		vae_file = input()
		lora_file = input()
		lora_dir = input()
		lora_scale = float(input())
		safety_check = True if safety_check > 0 else False
		diff_mode = DiffIMGMode(safety_check, vae_file, lora_file, lora_dir, lora_scale)
		if im_model == None or diff_mode == 2 or (img_gen_mode == False and diff_mode == 1):
			if LoadIMGModel(False, safety_check, vae_file, lora_file, lora_dir, lora_scale):
				img_gen_mode = True
		if im_model != None:
			img_file = GenImage(img_prompt, neg_prompt, infer_steps, guidance, img_seed, img_width, img_height, True)
			print("IMG_OUTPUT:"+img_file, flush=True)
			time.sleep(0.1)
		continue
	elif msg == "gen_text":
		print("START_TEXT:")
		start_txt = input().replace("[AI_UI_BR]", "\n").replace("\r", '')
		gmax = int(input())
		gmin = int(input())
		dsmp = int(input())
		numb = int(input())
		numg = int(input())
		temp = float(input())
		topk = int(input())
		topp = float(input())
		typp = float(input())
		repp = float(input())
		if tokenizer == None: LoadTokenizer()
		in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids.to(comp_dev)
		text = GenText(in_tokens, gmin, gmax, dsmp, numb, numg, temp, topk, topp, typp, repp)
		print("GEN_OUTPUT:"+text.replace("\n", "[AI_UI_BR]"), flush=True)
		time.sleep(0.1)
		continue
	elif msg == "speech_recog":
		try:
			text = SpeechToText(tempfile.gettempdir()+"/recording.wav").strip()
			print("ASR_OUTPUT:"+text.replace("\n", " "), flush=True)
		except:
			print("ASR_OUTPUT:ERROR", flush=True)
		time.sleep(0.1)
		continue
	elif msg == "gen_speech":
		if not tts_ready:
			print('TTS_OUTPUT:NO_VOICES', flush=True)
			time.sleep(0.1)
			continue
		print("TTS_TEXT:")
		tts_text = NormalizeText(input().replace("[AI_UI_BR]", "\n"))
		vckey = int(input())
		vcvol = float(input())
		vcrate = int(input())
		vcpitch = int(input())
		ttsmode = int(input())
		try:
			if ttsmode == 0:
				synthesizer.setProperty('voice', voices[vckey].id)
				synthesizer.setProperty('volume', vcvol)
				synthesizer.setProperty('rate', vcrate)
				TextToSpeechSys(tts_text, speech_wav, vcpitch)
				synthesizer.setProperty('voice', voices[voice_key].id)
				synthesizer.setProperty('volume', voice_vol)
				synthesizer.setProperty('rate', voice_rate)
			else:
				spkr_embs = LoadSpeaker(vckey)
				TextToSpeechAI(tts_text, speech_wav, spkr_embs, vcvol, vcrate, vcpitch, vckey)
			print('TTS_OUTPUT:'+speech_wav, flush=True)
		except:
			print('TTS_OUTPUT:ERROR', flush=True)
		time.sleep(0.1)
		continue
	elif msg == "read_text":
		if not tts_ready:
			print('PLAY_SPEECH:NO_VOICES', flush=True)
			time.sleep(0.1)
			continue
		print("SPEAK_TEXT:")
		tts_text = NormalizeText(input().replace("[AI_UI_BR]", "\n"))
		try:
			if tts_mode == 0:
				TextToSpeechSys(tts_text, read_wav, voice_ps)
			else:
				TextToSpeechAI(tts_text, read_wav, speaker_embeddings, voice_vol, voice_rate, voice_ps)
			print('PLAY_SPEECH:'+read_wav, flush=True)
		except:
			print('PLAY_SPEECH:ERROR', flush=True)
		time.sleep(0.1)
		continue
	elif msg == "clone_voice":
		print("CLONE_CONFIG:")
		clone_model = input()
		speaker_name = input().strip()
		speaker_wav = input().strip()
		transcript = input().strip()
		CleanVRAM()
		print('CLONE_OUTPUT:'+CloneVoice(clone_model, speaker_name, speaker_wav, transcript))
		time.sleep(0.1)
		CleanVRAM()
		ReserveVRAM()
		continue
	elif msg == "update_voices":
		LoadAIVoices()
		continue
	elif msg == "update_avatar":
		print("AVATAR_IMG:")
		avatar_img = input()
		if LoadAvatar():
			print("GOT_AVATAR:true", flush=True)
		else:
			print("GOT_AVATAR:false", flush=True)
		time.sleep(0.1)
		continue
	elif msg == "update_tmode":
		print("TALK_MODE:")
		talk_mode = int(input())
		ApplyTalkMode()
		continue
	elif msg == "update_users":
		GetUserNames()
		InitPrompt()
		continue
	elif msg == "redo_last":
		if redo_count == 0:
			print("BOT_NOANIM:Unable to redo last response.", flush=True)
			time.sleep(0.1)
			continue
		if redo_count < len(messages):
			messages = messages[:-redo_count]
		elif len(messages) > 0:
			messages = [messages[0]]
		PrunePrompt()
		rando_lvl += rando_add
	elif msg == "cont_chat":
		if use_chat_template:
			messages.append({"role":user_name, "content":"continue"})
	else:
		mm_msg = None
		if len(chat_files) > 0:
			prompt += "\n"+user_name+roles_append_str
			mm_msg = { "role": user_name, "content": [] }
			for chat_file in chat_files:
				prompt += chat_file["file"] + "\n"
				if chat_file["type"] == "text":
					mm_msg["content"].append({ "type": "text", "text": chat_file["content"] })
				else:
					mm_msg["content"].append({ "type": chat_file["type"], chat_file["content"]: chat_file["file"] })
			prompt = prompt.rstrip("\n")
		
		msgs = msg.split("[AIUI_END]")
		for msg_txt in msgs:
			msg = msg_txt.strip(" \n")
			if msg != '':
				prompt += "\n"+user_name+roles_append_str+msg
				if mm_msg == None:
					messages.append({"role": user_name, "content": msg})
				else:
					mm_msg["content"].append({"type": "text", "text": msg, "user":True})
		
		if mm_msg != None: messages.append(mm_msg)
	
	prompt += "\n"+bot_name_c
	rando_lvl -= rando_sub
	temp = min(1.0, max(rando_lvl, rando_min))
	
	in_tokens = TokenizeChat()
	response = GenReply(in_tokens, temp).replace("\r", '')
	responses = response.split("\n")

	redo_count = 0
	r = 0

	while (r < len(responses)):

		if use_chat_template:
			cmb_response = response.strip(" \n")
		else:
			cmb_response = responses[r].strip(" ")
			
			while (r+1 < len(responses)):
				nxt_response = responses[r+1].strip(" ")
				if nxt_response == '':
					cmb_response += "\n"
					r += 1
					continue
				if nxt_response.startswith(user_name_c):
					break
				else:
					if len(nxt_response) <= len(user_name_c) or len(nxt_response) <= len(bot_name_c):
						if user_name_c.startswith(nxt_response) or bot_name_c.startswith(nxt_response):
							break
					cmb_response += "\n" + StripStart(nxt_response, bot_name_c).lstrip(' ')
					r += 1

			cmb_response = StripStart(cmb_response.strip(" \n"), bot_name_c).lstrip(' ')
		
		if cmb_response == '':
			print("BOT_NOANIM:", flush=True)
			time.sleep(0.1)
			break
		
		tc_response = ''
		tc_errors = 0
		
		while enable_tooluse:
			if not "<tool_call>" in cmb_response:
				messages.append({"role":bot_name, "content":cmb_response})
				tc_response += cmb_response
				redo_count += 1
				break
			
			tc_think = False
			tc_result = None
			tc_list = cmb_response.split("<tool_call>")
			tc_resp = ''
			
			for tc_str in tc_list:
				tc_parts = tc_str.strip().split("</tool_call>")
				if len(tc_parts) < 1 or len(tc_parts) > 2: continue
				
				if not '"name"' in tc_parts[0]:
					if tc_parts[0] != '': tc_resp += tc_parts[0] + "\n"
					if len(tc_parts) > 1 and tc_parts[1] != '':
						tc_resp += tc_parts[1] + "\n"
						messages.append({"role":bot_name, "content":tc_parts[1]})
						redo_count += 1
					continue
				
				if tool_mode != "roleplay":
					tc_resp += "<tool_call>" + SafeString(tc_parts[0]) + "</tool_call>\n"
				
				tr_data = ''
				tc_data = json.loads(tc_parts[0])
				tc_arg_key = "arguments"
				tc_error = False
				tool_call = None
				redo_count += 1
				
				if "arguments" in tc_data and "name" in tc_data:
					tool_call = { "name": tc_data["name"], "arguments": tc_data["arguments"] }
				elif "parameters" in tc_data and "name" in tc_data:
					tool_call = { "name": tc_data["name"], "arguments": tc_data["parameters"] }
					tc_arg_key = "parameters"
				
				if tool_call != None:
					tc_result = tool_func(tc_data["name"], tc_data[tc_arg_key], { 'GenImageIPA': GenImageIPA })
				elif "name" in tc_data:
					tc_result = tool_func(tc_data["name"], {}, { 'GenImageIPA': GenImageIPA })
					tool_call = { "name": tc_data["name"], "arguments": {} }
				else:
					tc_error = True
					tc_errors += 1
				
				if use_tool_template and not tc_error:
					messages.append({"role": bot_name, "tool_calls": [{"type": "function", "function": tool_call}]})
				else:
					messages.append({"role": bot_name, "content": tc_parts[0]})
				
				if tc_error:
					tc_error = True
					tc_errors += 1
					redo_count += 1
					
					tr_data = '{"name": "unknown", "content": "ERROR: unknown tool call format"}'
					
					if use_tool_template:
						messages.append({"role": "tool", "name": "unknown", "content": "ERROR: unknown tool call format"})
					else:
						messages.append({"role": bot_name, "content": "<tool_response>"+tr_data+"</tool_response>"})
				
				elif tc_result != None:
						
					if isinstance(tc_result, int) or isinstance(tc_result, float) or isinstance(tc_result, bool):
						tc_result = str(tc_result)
					elif not isinstance(tc_result, str):
						tc_result = json.dumps(tc_result)
					
					if tc_result.startswith("ERROR:"):
						tc_error = True
						tc_errors += 1
					
					tr_data = '{"name": "'+tc_data["name"]+'", "content": "'+tc_result+'"}'
					
					if tool_mode == "roleplay" and (tc_data["name"] == "show_image" or tc_data["name"] == "set_bio" or tc_data["name"] == "set_scene") and not tc_error:
					
						prompt_key = "prompt" if tc_data["name"] == "show_image" else "summary"
						if tc_result != '' and tc_arg_key in tc_data and prompt_key in tc_data[tc_arg_key]:
							tc_resp += "<aiui_image>"+tc_result+"AIUI_IMG_TXT="+tc_data[tc_arg_key][prompt_key]+"</aiui_image>\n"
					else:
						tc_think = True
						redo_count += 1
						
						if use_tool_template:
							messages.append({"role": "tool", "name": tc_data["name"], "content": tc_result})
						else:
							tc_result = tc_result.replace('\\"', '"').replace('"', '\\"')
							messages.append({"role": bot_name, "content": "<tool_response>"+tr_data+"</tool_response>"})
							#NOTE: the role should probably be 'tool' or 'system' in this case but most chat templates seem to ignore those roles here
				
				if tr_data != '' and tool_mode != "roleplay":
					tc_resp += "<tool_response>" + SafeString(tr_data) + "</tool_response>\n"
				
				if len(tc_parts) > 1 and tc_parts[1].strip() != '':
					tc_resp += tc_parts[1] + "\n"
					messages.append({"role":bot_name, "content":tc_parts[1]})
					redo_count += 1

			tc_resp = tc_resp.strip()
			if tc_resp != '': tc_response += tc_resp + "\n"
			
			if tc_errors > 4 or (tc_error and tc_response != ''): 
				if tc_response == '': tc_response = "ERROR: bad tool calls"
				break
			
			if tc_think or tc_response == '':
				in_tokens = TokenizeChat()
				cmb_response = GenReply(in_tokens, temp).replace("\r", '').strip(" \n")
				continue
			break
		
		if tc_response != '':
			cmb_response = tc_response
		else:
			messages.append({"role":bot_name, "content":cmb_response})
			redo_count += 1
		
		tts_response = cmb_response
		anim_done = False
		think_tags = { "<thoughts>": "</thoughts>", "<think>": "</think>" }
		tool_tags = { "<tool_call>": "</tool_call>", "<tool_response>": "</tool_response>" }
		img_tags = { "<ai_image>": "</ai_image>", "<aiui_image>": "</aiui_image>" }

		for xml_open_tag, xml_close_tag in think_tags.items():
			while xml_open_tag in cmb_response and xml_close_tag in cmb_response:
				tag_start = cmb_response.index(xml_open_tag) + len(xml_open_tag)
				tag_end = cmb_response.index(xml_close_tag)
				if tag_start >= tag_end:
					cmb_response = cmb_response.replace(tag_end, tag_end.replace('<', 'AI_UI_AMPlt;', 1).replace('>', 'AI_UI_AMPgt;', 1), 1)
					continue
				inner_txt = cmb_response[tag_start:tag_end]
				tts_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, '', 1)
				cmb_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, "AIUI_THINK_START"+SafeString(inner_txt)+"AIUI_THINK_END", 1)

		for xml_open_tag, xml_close_tag in tool_tags.items():
			while xml_open_tag in cmb_response and xml_close_tag in cmb_response:
				tag_start = cmb_response.index(xml_open_tag) + len(xml_open_tag)
				tag_end = cmb_response.index(xml_close_tag)
				if tag_start >= tag_end:
					cmb_response = cmb_response.replace(tag_end, tag_end.replace('<', 'AI_UI_AMPlt;', 1).replace('>', 'AI_UI_AMPgt;', 1), 1)
					continue
				inner_txt = cmb_response[tag_start:tag_end]
				tts_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, '', 1)
				if xml_open_tag == "<tool_call>":
					cmb_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, "AIUI_TC_START"+inner_txt+"AIUI_TC_END", 1)
				else:
					cmb_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, "AIUI_TR_START"+inner_txt+"AIUI_TR_END", 1)

		for xml_open_tag, xml_close_tag in img_tags.items():
			while xml_open_tag in cmb_response and xml_close_tag in cmb_response:
				tag_start = cmb_response.index(xml_open_tag) + len(xml_open_tag)
				tag_end = cmb_response.index(xml_close_tag)
				if tag_start >= tag_end:
					cmb_response = cmb_response.replace(tag_end, tag_end.replace('<', 'AI_UI_AMPlt;', 1).replace('>', 'AI_UI_AMPgt;', 1), 1)
					continue
				inner_txt = cmb_response[tag_start:tag_end]
				if xml_open_tag == "<ai_image>":
					img_file = GenImage(inner_txt)
					if img_file.startswith("ERROR:"):
						tts_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, img_file, 1)
						cmb_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, img_file, 1)
						break
					else:
						tts_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, 'Image prompt: '+inner_txt, 1)
						cmb_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, "AIUI_IMG_SRC="+img_file+"AIUI_IMG_TXT="+inner_txt+"AIUI_IMG_END", 1)
				else:
					inner_parts = inner_txt.split("AIUI_IMG_TXT=")
					inner_prompt = "invalid image prompt detected" if len(inner_parts) < 2 else inner_parts[1]
					tts_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, 'Image prompt: '+inner_prompt, 1)
					cmb_response = cmb_response.replace(xml_open_tag+inner_txt+xml_close_tag, "AIUI_IMG_SRC="+inner_txt+"AIUI_IMG_END", 1)

		if do_talk and tts_ready and tts_response != '':
			tts_response = NormalizeText(tts_response)
			try:
				if tts_mode == 0:
					TextToSpeechSys(tts_response, chat_wav, voice_ps)
				else:
					TextToSpeechAI(tts_response, chat_wav, speaker_embeddings, voice_vol, voice_rate, voice_ps)
				
				if do_anim: anim_done = AnimAvatar()
				if not (do_anim and anim_done):
					print('PLAY_SPEECH:'+chat_wav, flush=True)
					time.sleep(0.1)
			except:
				anim_done = False

		if do_anim and anim_done:
			print("BOT_OUTPUT:"+cmb_response.replace("\n", "[AI_UI_BR]"), flush=True)
		else:
			print("BOT_NOANIM:"+cmb_response.replace("\n", "[AI_UI_BR]"), flush=True)

		time.sleep(0.1)
		r += 1
		break #NOTE: disabled multiple bot responses for now
	
	chat_files = []
	PrunePrompt()
