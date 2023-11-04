"""
 # Copyright 2023 Bitfreak Software
 # All Rights Reserved.

"""

import sys
import time
import os, gc, glob
import numpy as np
import soundfile as sf
import librosa
import torch
import random
import pyttsx3
import transformers
import torch.nn as nn
from peft import PeftModel
from datetime import datetime
from scipy.io import wavfile
from speechbrain.pretrained import EncoderClassifier
from MakeItTalk.main_end2end import ResampleAudio, LoadFace, AnimFace
from Wav2Lip.inference import load_face, anim_face
from SadTalker.inference import loadface, animface
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from transformers import pipeline, GPTNeoForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
	
###### Config/Vars ######
sys.stdout.reconfigure(encoding='utf-8')

print("APP_CONFIG:")
work_dir = input().rstrip('/')
model_id = input().rstrip('/')
sd_model_id = input().rstrip('/')
tts_model_id = input().rstrip('/')
vocoder_id = input().rstrip('/')
model_type = int(input())
imodel_type = int(input())
smodel_type = int(input())
model_args = input().split(',')
device = input()
start_meth = input()
avatar_img = input()

if not os.path.isdir(work_dir):
	sys.exit("ERROR: cannot find engine folder at "+work_dir)

print("AI_CONFIG:")
msg_mem = int(input())
gen_max = int(input())
gen_min = int(input())
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
user_name_dc = ''
bot_name_dc = ''
user_name_dcs = ''
bot_name_dcs = ''

prompt = ''
init_prompt = ''
last_res = ''
msg = ''
r = 0

messages = []
responses = []
stop_ids = []
stop_criteria = []

min_res_vram_mb = 800
ext_res_vram_mb = 0 if sd_model_id == '' else 3000
reserve_vram_mb = min_res_vram_mb + ext_res_vram_mb
reserved_vram = None

model_adapter = ''
sd_lora_file = ''
sd_lora_dir = ''
sd_vae_file = ''
sd_safety_check = True
sd_cpu_offload = True
sd_att_slicing = True
sd_from_single_file = False
sd_config_file = None
sd_use_safetensors = None
use_safetensors = None
load_in_8bit = False
custom_model_code = False
custom_token_code = False
use_chat_template = False
sd_torch_dtype = torch.float16
torch_dtype = torch.float16

if imodel_type == 1: sd_att_slicing = False

for arg in model_args:
	carg = arg.strip(" ").replace("\\", "/")
	if carg.startswith("model_adapter="):
		model_adapter = carg.replace("model_adapter=", "", 1)
	elif carg.startswith("sd_lora_file="):
		sd_lora_file = carg.replace("sd_lora_file=", "", 1)
	elif carg.startswith("sd_lora_dir="):
		sd_lora_dir = carg.replace("sd_lora_dir=", "", 1)
	elif carg.startswith("sd_vae_file="):
		sd_vae_file = carg.replace("sd_vae_file=", "", 1)
	elif carg.startswith("sd_config_file="):
		sd_config_file = carg.replace("sd_config_file=", "", 1)
	elif carg.startswith("torch_dtype="):
		dtype = carg.replace("torch_dtype=", "", 1).lower()
		if dtype == "float32":
			torch_dtype = torch.float32
		elif dtype == "float16":
			torch_dtype = torch.float16
		elif dtype == "bfloat16":
			torch_dtype = torch.bfloat16
	elif carg.startswith("sd_torch_dtype="):
		sd_dtype = carg.replace("sd_torch_dtype=", "", 1).lower()
		if sd_dtype == "float32":
			sd_torch_dtype = torch.float32
		elif sd_dtype == "float16":
			sd_torch_dtype = torch.float16
		elif sd_dtype == "bfloat16":
			sd_torch_dtype = torch.bfloat16
	elif carg.startswith("use_safetensors="):
		use_sts = carg.replace("use_safetensors=", "", 1)
		if use_sts.lower() == "true" or use_sts == "1":
			use_safetensors = True
		else:
			use_safetensors = False
	elif carg.startswith("sd_use_safetensors="):
		use_sts = carg.replace("sd_use_safetensors=", "", 1)
		if use_sts.lower() == "true" or use_sts == "1":
			sd_use_safetensors = True
		else:
			sd_use_safetensors = False
	elif carg.startswith("load_in_8bit="):
		load_8bit = carg.replace("load_in_8bit=", "", 1)
		if load_8bit.lower() == "true" or load_8bit == "1":
			load_in_8bit = True
	elif carg.startswith("custom_model_code="):
		custom_code = carg.replace("custom_model_code=", "", 1)
		if custom_code.lower() == "true" or custom_code == "1":
			custom_model_code = True
	elif carg.startswith("custom_token_code="):
		custom_code = carg.replace("custom_token_code=", "", 1)
		if custom_code.lower() == "true" or custom_code == "1":
			custom_token_code = True
	elif carg.startswith("sd_from_single_file="):
		from_single_file = carg.replace("sd_from_single_file=", "", 1)
		if from_single_file.lower() == "true" or from_single_file == "1":
			sd_from_single_file = True
	elif carg.startswith("sd_cpu_offload="):
		cpu_offload = carg.replace("sd_cpu_offload=", "", 1)
		if cpu_offload.lower() == "false" or cpu_offload == "0":
			sd_cpu_offload = False
	elif carg.startswith("sd_att_slicing="):
		att_slicing = carg.replace("sd_att_slicing=", "", 1)
		if att_slicing.lower() == "false" or att_slicing == "0":
			sd_att_slicing = False
		else:
			sd_att_slicing = True
	elif carg.startswith("sd_safety_check="):
		safety_check = carg.replace("sd_safety_check=", "", 1)
		if safety_check.lower() == "false" or safety_check == "0":
			sd_safety_check = False
	elif carg.startswith("apply_chat_template="):
		use_chat_temp = carg.replace("apply_chat_template=", "", 1)
		if use_chat_temp.lower() == "true" or use_chat_temp == "1":
			use_chat_template = True
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
	
def SetWorkingDir(face_anim_mode):
	if face_anim_mode == 0:
		os.chdir(work_dir + '/MakeItTalk/')
	elif face_anim_mode == 1:
		os.chdir(work_dir + '/Wav2Lip/')
	else:
		os.chdir(work_dir + '/SadTalker/')

###### TTS Stuff ######

audio_dir = work_dir+'/ai_audio/'
if not os.path.exists(audio_dir): os.mkdir(audio_dir)

speech_wav = audio_dir+'speech.wav'
chat_wav = audio_dir+'chat.wav'
read_wav = audio_dir+'read.wav'

synthesizer = None
speaker_embeddings = None
processor = None
tts_model = None
vocoder = None
tts_ready = False
do_talk = True
do_anim = True
voices = []
t5_voices = []

print("VOICE_CONFIG:")
voice_key = int(input())
voice_vol = float(input())
voice_rate = int(input())
voice_ps = int(input())
talk_mode = int(input())
tts_mode = int(input())
anim_mode = int(input())

SetWorkingDir(anim_mode)
	
def LoadSpeaker(voice_dataset):
	return torch.tensor(np.load(voice_dataset)).unsqueeze(0)
	
def LoadSysVoices():
	global voices
	
	voices = synthesizer.getProperty('voices')
	voice_list = 'SYS_VOICES:'
	
	for voice in voices:
		voice_list += "VOICE_NAME:"+voice.name+"VOICE_ID:"+voice.id+"\n"

	if voice_list != 'SYS_VOICES:':
		print(voice_list, flush=True)
		time.sleep(0.1)

def LoadT5Voices(speaker_name=''):
	global t5_voices
	speaker_index = 0

	t5_voices = glob.glob1(work_dir+"/embeddings/", '*.npy')
	t5_voices.sort()
	voice_list = 'T5_VOICES:'
	
	for i in range(0,len(t5_voices)):
		if t5_voices[i] == speaker_name: speaker_index = i
		voice_list += "VOICE_NAME:"+t5_voices[i]+"VOICE_ID:"+str(i)+"\n"
	
	if voice_list != 'T5_VOICES:':
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
		if len(t5_voices) > 0:
			speaker_embeddings = LoadSpeaker(work_dir + "/embeddings/"+t5_voices[voice_key])
		else:
			print("ERROR: no voice embeddings could be found")
			tts_ready = False
	
def LoadTTSVoices():
	global synthesizer
	if synthesizer == None:
		synthesizer = pyttsx3.init()
	
	LoadSysVoices()
	LoadT5Voices()	
	ConfigTTSEngine()
		
def LoadTTSEngine(force_mode1=False):
	global processor, tts_model, vocoder
	
	if (tts_mode == 1 and tts_model == None) or force_mode1:
		processor = SpeechT5Processor.from_pretrained(tts_model_id)
		tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_id)
		vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
		
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
		
def CloneVoice(speaker_name, speaker_wav):
	embeddings = []
	try:
		ec_model_id = "speechbrain/spkrec-xvect-voxceleb"
		if torch.cuda.is_available():
			classifier = EncoderClassifier.from_hparams(source=ec_model_id, run_opts={"device": "cuda"})
		else:
			classifier = EncoderClassifier.from_hparams(source=ec_model_id)
			
		fs, signal = wavfile.read(speaker_wav)

		if fs != 16000:
			signal = ResampleAudio(fs, signal, 16000)
		else:
			signal = torch.tensor(signal)

		with torch.no_grad():
			embeddings = classifier.encode_batch(signal)
			embeddings = torch.nn.functional.normalize(embeddings, dim=2)
			embeddings = embeddings.squeeze().cpu().numpy()
	except:
		return 'ERROR:There was an error with the cloning model.'
			
	try:
		if len(embeddings) > 0:
			np.save(work_dir+"/embeddings/"+speaker_name+".npy", embeddings)
		else:
			return 'ERROR:There was an error generating the embeddings.'
	except:
		return 'ERROR:There was an error saving the voice file.'
	
	voice_id = LoadT5Voices(speaker_name+'.npy')
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
					
def CleanTextForTTS(txt):
	return txt.replace(' < ', ' less than ').replace(' > ', ' greater than ').replace('<', ' ').replace('>', ' ')

###### GPT Stuff ######

transformers.logging.set_verbosity_error()
#set_seed(int(str(time.time()).replace('.', '')))

model = None
tokenizer = None

def LoadTokenizer():
	global tokenizer

	try:
		if (model_type == 0):
			tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=custom_token_code)
		elif (model_type == 1):
			tokenizer = AutoTokenizer.from_pretrained(model_id)
		elif (model_type == 2):
			tokenizer = LlamaTokenizer.from_pretrained(model_id)
		elif (model_type == 3 or model_type == 4):
			tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
	except:
		if start_meth == "text" or start_meth == "all":
			sys.exit("ERROR: failed to load text tokenizer")

def LoadModel():
	global model
	#reserve VRAM for other models
	ReserveVRAM()
	
	if tokenizer == None: LoadTokenizer()

	if (model_type == 0):
		if device == "auto":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, trust_remote_code=custom_model_code, device_map="auto")
		elif device == "cuda":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, trust_remote_code=custom_model_code).to(device)
		else:
			model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=use_safetensors, trust_remote_code=custom_model_code).to(device)

	elif (model_type == 1):
		if device == "auto":
			model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, pad_token_id=tokenizer.eos_token_id, device_map="auto")
		elif device == "cuda":
			model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, pad_token_id=tokenizer.eos_token_id).to(device)
		else:
			model = GPTNeoForCausalLM.from_pretrained(model_id, pad_token_id=tokenizer.eos_token_id, use_safetensors=use_safetensors, low_cpu_mem_usage=True).to(device)

	elif (model_type == 2):
		if device == "auto":
			model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit, device_map="auto")
		elif device == "cuda":
			model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, load_in_8bit=load_in_8bit).to(device)
		else:
			model = LlamaForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=use_safetensors, device_map={"": device})

	elif (model_type == 3):
		if device == "auto":
			model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, trust_remote_code=True, device_map="auto")
		elif device == "cuda":
			model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=use_safetensors, trust_remote_code=True).to(device)
		else:
			model = AutoModel.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=use_safetensors, trust_remote_code=True).to(device)

	elif (model_type == 4):
		if device == "auto":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=use_safetensors, trust_remote_code=True, device_map="auto")
		elif device == "cuda":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=use_safetensors, trust_remote_code=True).to(device)
		else:
			model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=use_safetensors, trust_remote_code=True).to(device)
			
	if model_adapter != '':
		model.load_adapter(model_adapter)

	model.eval()

	#if torch.__version__ >= "2" and sys.platform != "win32":
	#	model = torch.compile(model)

###### STABLE DIFF STUFF ######

sd_model = None
sd_gen_mode = False

last_sc_val = sd_safety_check
last_vf_val = sd_vae_file
last_lf_val = sd_lora_file
last_ld_val = sd_lora_dir
	
def DiffSDMode(safety_check, vae_file, lora_file, lora_dir):
	global last_sc_val, last_vf_val, last_lf_val, last_ld_val
	if safety_check != last_sc_val or vae_file != last_vf_val or lora_file != last_lf_val or lora_dir != last_ld_val:
		result = 2
	else:
		result = int(safety_check != sd_safety_check or vae_file != sd_vae_file or lora_file != sd_lora_file or lora_dir != sd_lora_dir)
	last_sc_val = safety_check
	last_vf_val = vae_file
	last_lf_val = lora_file
	last_ld_val = lora_dir
	return result

def LoadSDModel(exit_on_error=True, safety_check=None, vae_file=None, lora_file=None, lora_dir=None):
	global sd_model, reserve_vram_mb
	orig_rvram = reserve_vram_mb
	sd_model = None
	vae_model = None
	result = True
	
	safety_check = sd_safety_check if safety_check == None else safety_check
	vae_file = sd_vae_file if vae_file == None else vae_file
	lora_file = sd_lora_file if lora_file == None else lora_file
	lora_dir = sd_lora_dir if lora_dir == None else lora_dir
	
	CleanVRAM()
	if reserve_vram_mb > min_res_vram_mb:
		reserve_vram_mb = min_res_vram_mb
	ReserveVRAM()

	try:
		if sd_from_single_file or sd_model_id.endswith(".safetensors") or sd_model_id.endswith(".ckpt"):
			if vae_file != '':
				vae_model = AutoencoderKL.from_single_file(vae_file, torch_dtype=sd_torch_dtype)
			if imodel_type == 0:
				sd_model = StableDiffusionPipeline.from_single_file(sd_model_id, vae=vae_model, torch_dtype=sd_torch_dtype, use_safetensors=sd_use_safetensors, original_config_file=sd_config_file, load_safety_checker=safety_check)
			else:
				sd_model = StableDiffusionXLPipeline.from_single_file(sd_model_id, vae=vae_model, torch_dtype=sd_torch_dtype, use_safetensors=sd_use_safetensors, original_config_file=sd_config_file, load_safety_checker=safety_check)
		else:
			if safety_check:
				if imodel_type == 0:
					sd_model = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=sd_torch_dtype, use_safetensors=sd_use_safetensors)
				else:
					sd_model = StableDiffusionXLPipeline.from_pretrained(sd_model_id, torch_dtype=sd_torch_dtype, use_safetensors=sd_use_safetensors)
			elif imodel_type == 0:
				sd_model = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=sd_torch_dtype, use_safetensors=sd_use_safetensors, safety_checker=None)
			else:
				sd_model = StableDiffusionXLPipeline.from_pretrained(sd_model_id, torch_dtype=sd_torch_dtype, use_safetensors=sd_use_safetensors, safety_checker=None)
		
		if device == "auto":
			if torch.cuda.is_available():
				sd_model = sd_model.to("cuda")
		else:
			sd_model = sd_model.to(device)
		
		if lora_file != '':
			sd_model.load_lora_weights(lora_dir, weight_name=lora_file)
		
		if sd_cpu_offload:
			sd_model.enable_model_cpu_offload()
			
		if sd_att_slicing:
			sd_model.enable_attention_slicing("max")
	except:
		reserve_vram_mb = orig_rvram
		sd_model = None
		result = False
		if exit_on_error:
			sys.exit("ERROR: failed to load stable diffusion model")
		else:
			print("ERROR: failed to load stable diffusion model", flush=True)
			time.sleep(0.1)
	
	CleanVRAM()
	ReserveVRAM()
	
	return result

def GenImage(image_prompt, neg_prompt="NONE", infer_steps=50, guidance=7.5, img_width="auto", img_height="auto", gen_mode=False):
	global sd_model, sd_gen_mode

	if image_prompt == '': return -1
	
	CleanVRAM()

	while True:
		if sd_model == None or (gen_mode == False and sd_gen_mode == True):
			sd_gen_mode = False
			if not LoadSDModel(False):
				img_num = -2
				break
			
		try:
			clean_prompt = image_prompt.strip(' ').replace("\n", ' ').replace("\r", '')

			with torch.inference_mode():
				if neg_prompt == "NONE":
					if img_width == "auto" or img_height == "auto":
						output = sd_model(prompt=clean_prompt, num_inference_steps=infer_steps, guidance_scale=guidance)
					else:
						output = sd_model(prompt=clean_prompt, num_inference_steps=infer_steps, guidance_scale=guidance, width=int(img_width), height=int(img_height))
				else:
					if img_width == "auto" or img_height == "auto":
						output = sd_model(prompt=clean_prompt, negative_prompt=neg_prompt, num_inference_steps=infer_steps, guidance_scale=guidance)
					else:
						output = sd_model(prompt=clean_prompt, negative_prompt=neg_prompt, num_inference_steps=infer_steps, guidance_scale=guidance, width=int(img_width), height=int(img_height))
		except:
			print("ERROR: failed to generate image", flush=True)
			time.sleep(0.1)
			img_num = -3
			break

		try:
			img_dir = work_dir+"/ai_images/"
			if os.path.exists(img_dir):
				img_num = CountFiles(img_dir)
			else:
				os.mkdir(img_dir)
				img_num = 0

			output.images[0].save(img_dir+"image_"+str(img_num)+".png")
		except:
			print("ERROR: failed to save image", flush=True)
			time.sleep(0.1)
			img_num = -4
		
		break

	CleanVRAM()
	ReserveVRAM()
	
	return img_num

###### CHAT STUFF ######

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

def StripEnd(txt, ss):
	return txt[:-len(ss)] if txt.endswith(ss) else txt

def GetUserNames():
	global user_name, bot_name, user_name_dc, bot_name_dc, user_name_dcs, bot_name_dcs, stop_ids

	print("YOUR_NAME:")
	user_name = input()
	print("BOT_NAME:")
	bot_name = input()
	user_name_dc = user_name+':'
	bot_name_dc = bot_name+':'
	user_name_dcs = user_name_dc+' '
	bot_name_dcs = bot_name_dc+' '

	if tokenizer != None:
		stop_ids = tokenizer.encode("\n"+user_name_dc)

def InitPrompt(have_prompt=False):
	global messages, prompt, init_prompt
	if have_prompt:
		prompt = init_prompt
	else:
		prompt = 'Chat log between '+user_name+' and '+bot_name+' on '+datetime.utcnow().strftime("%m/%d/%Y")+"\n"
		init_prompt = prompt
	messages = [{"role":"system", "content":init_prompt}]
	if prompt_p == 1:
		messages = []
	if prompt_p == 2:
		prompt = ''
		init_prompt = 'No prompt'
		messages = []
	if have_prompt:
		print("CLEAR_DONE:"+init_prompt, flush=True)
	else:
		print("INIT_DONE:"+init_prompt, flush=True)
	time.sleep(0.1)
	
def PrunePrompt():
	global messages, prompt
	i = len(messages) - msg_mem
	if i > 0: messages = messages[i::]
	prompt = ''
	i = 0
	l = len(messages)
	while i < l:
		if i+1 < l:
			prompt += messages[i]["role"] + ": " + messages[i]["content"] + "\n"
		else:
			prompt += messages[i]["role"] + ": " + messages[i]["content"]
		i += 1
	if prompt_p == 1:
		if prompt == '':
			prompt = init_prompt
		else:
			prompt = init_prompt+"\n"+prompt

def GenText(it, rl):
	global model, tokenizer, stop_ids
	CleanVRAM()
	if model == None: LoadModel()
	if len(stop_ids) == 0: stop_ids = tokenizer.encode("\n"+user_name_dc)
	stop_criteria = StoppingCriteriaList([StoppingCriteriaKeys(stop_ids)])
	with torch.no_grad():
		out_tokens = model.generate(do_sample=True, inputs=it, min_new_tokens=gen_min, max_new_tokens=gen_max, temperature=rl, 
			top_k=top_k, top_p=top_p, typical_p=typical_p, repetition_penalty=rep_penalty, stopping_criteria=stop_criteria)
	CleanVRAM()
	ReserveVRAM()
	return StripEnd(tokenizer.decode(out_tokens[0], skip_special_tokens=True), '<|endoftext|>')

def GenNoStop(it, nt, mt, rl, tk, tp, ty, rp):
	global model, tokenizer
	CleanVRAM()
	if model == None: LoadModel()
	with torch.no_grad():
		out_tokens = model.generate(do_sample=True, inputs=it, min_new_tokens=nt, max_new_tokens=mt,
			temperature=rl, top_k=tk, top_p=tp, typical_p=ty, repetition_penalty=rp)
	CleanVRAM()
	ReserveVRAM()
	return StripEnd(tokenizer.decode(out_tokens[0], skip_special_tokens=True), '<|endoftext|>')

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
		LoadSDModel()
	elif start_meth == "speech":
		LoadTTSEngine()
	elif start_meth == "all":
		LoadTTSEngine()
		LoadModel()
		LoadSDModel()

LoadTTSVoices()
LoadModels()
LoadAvatar()
ApplyTalkMode()
GetUserNames()
InitPrompt()

while (True):

	print("HUMAN_INPUT:")
	msg = input().replace("[AI_UI_BR]", "\n").strip(" \n\t").replace("\r", '')

	if (msg == ''):
		continue	
	elif msg == "close_chat":
		break
	elif msg == "print_prompt":
		print('----- START PROMPT -----')
		print(prompt)
		print('------ END PROMPT ------')
		time.sleep(0.1)
		continue
	elif msg == "update_prompt":
		print("NEW_PROMPT:")
		init_prompt = input().replace("[AI_UI_BR]", "\n").replace("\r", '')
		if prompt_p == 0:
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
	elif msg == "config_voice":
		print("VOICE_CONFIG:")
		last_ttsm = tts_mode
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
		if tts_mode != last_ttsm: LoadTTSEngine()
		ConfigTTSEngine()
		continue
	elif msg == "config_ai":
		print("AI_CONFIG:")
		last_pp = prompt_p
		msg_mem = int(input())
		gen_max = int(input())
		gen_min = int(input())
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
	elif msg == "gen_image":
		print("IMAGE_PROMPT:")
		img_prompt = input()
		neg_prompt = input()
		infer_steps = int(input())
		guidance = float(input())
		safety_check = int(input())
		img_width = input()
		img_height = input()
		vae_file = input()
		lora_file = input()
		lora_dir = input()
		img_num = -5
		safety_check = True if safety_check > 0 else False
		diff_mode = DiffSDMode(safety_check, vae_file, lora_file, lora_dir)
		if sd_model == None or diff_mode == 2 or (sd_gen_mode == False and diff_mode == 1):
			if LoadSDModel(False, safety_check, vae_file, lora_file, lora_dir):
				sd_gen_mode = True
			else:
				img_num = -2
		if img_num != -2:
			img_num = GenImage(img_prompt, neg_prompt, infer_steps, guidance, img_width, img_height, True)
		print("IMG_OUTPUT:"+str(img_num), flush=True)
		time.sleep(0.1)
		continue
	elif msg == "gen_text":
		print("START_TEXT:")
		start_txt = input().replace("[AI_UI_BR]", "\n").replace("\r", '')
		gmax = int(input())
		gmin = int(input())
		temp = float(input())
		topk = int(input())
		topp = float(input())
		typp = float(input())
		repp = float(input())
		if tokenizer == None: LoadTokenizer()
		if device != "cpu" and torch.cuda.is_available():
			in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids.to("cuda")
		else:
			in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids
		text = GenNoStop(in_tokens, gmin, gmax, temp, topk, topp, typp, repp)
		print("GEN_OUTPUT:"+text.replace("\n", "[AI_UI_BR]"), flush=True)
		time.sleep(0.1)
		continue
	elif msg == "gen_speech":
		if not tts_ready:
			print('TTS_OUTPUT:NO_VOICES', flush=True)
			time.sleep(0.1)
			continue
		print("TTS_TEXT:")
		tts_text = CleanTextForTTS(input().replace("[AI_UI_BR]", "\n"))
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
				synthesizer.save_to_file("<pitch middle='"+str(vcpitch)+"'/>"+tts_text, speech_wav)
				synthesizer.runAndWait()
				AddEpsilonNoise(speech_wav)
				synthesizer.setProperty('voice', voices[voice_key].id)
				synthesizer.setProperty('volume', voice_vol)
				synthesizer.setProperty('rate', voice_rate)
			else:
				if tts_model == None: LoadTTSEngine(True)
				inputs = processor(text=tts_text, return_tensors="pt")
				spkr_embs = LoadSpeaker(work_dir + "/embeddings/"+t5_voices[vckey])
				speech = tts_model.generate_speech(inputs["input_ids"], spkr_embs, vocoder=vocoder)
				sf.write(speech_wav, speech.numpy(), samplerate=16000)
				ModifySound(speech_wav, vcvol, vcrate, vcpitch)
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
		tts_text = CleanTextForTTS(input().replace("[AI_UI_BR]", "\n"))
		try:
			if tts_mode == 0:
				synthesizer.save_to_file("<pitch middle='"+str(voice_ps)+"'/>"+tts_text, read_wav)
				synthesizer.runAndWait()
				AddEpsilonNoise(read_wav)
			else:
				if tts_model == None: LoadTTSEngine()
				inputs = processor(text=tts_text, return_tensors="pt")
				speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
				sf.write(read_wav, speech.numpy(), samplerate=16000)
				ModifySound(read_wav, voice_vol, voice_rate, voice_ps)
			print('PLAY_SPEECH:'+read_wav, flush=True)
		except:
			print('PLAY_SPEECH:ERROR', flush=True)
		time.sleep(0.1)
		continue
	elif msg == "clone_voice":
		print("CLONE_CONFIG:")
		speaker_name = input()
		speaker_wav = input()
		CleanVRAM()
		print('CLONE_OUTPUT:'+CloneVoice(speaker_name, speaker_wav))
		time.sleep(0.1)
		CleanVRAM()
		ReserveVRAM()
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
		messages = messages[:-1:]
		PrunePrompt()
		prompt += "\n"+bot_name_dc
		rando_lvl += rando_add
	elif msg == "cont_chat":
		prompt += "\n"+bot_name_dc
		if use_chat_template:
			messages.append({"role":user_name, "content":"continue"})
	else:
		prompt += "\n"+user_name_dcs+msg+"\n"+bot_name_dc
		messages.append({"role":user_name, "content":msg})

	rando_lvl -= rando_sub
	temp = min(1.0, max(rando_lvl, rando_min))
	
	if tokenizer == None: LoadTokenizer()

	if use_chat_template:
		if device != "cpu" and torch.cuda.is_available():
			in_tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
		else:
			in_tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")
			
		last_prompt = tokenizer.decode(in_tokens[0], skip_special_tokens=True)
	else:
		if device != "cpu" and torch.cuda.is_available():
			in_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
		else:
			in_tokens = tokenizer(prompt, return_tensors="pt").input_ids
			
		last_prompt = prompt[:-len(bot_name_dc)]
		
	text = GenText(in_tokens, temp)
	text = StripEnd(text, "\n"+user_name_dc).replace("\r", '')
	responses = text.replace(last_prompt, '', 1).split("\n")
	
	#print("RAW OUTPUT: "+text, flush=True)
	#time.sleep(0.1)
	
	got_res = False
	last_res = ''
	r = 0

	while (r < len(responses)):
		cmb_response = responses[r].strip(" \t")

		if not use_chat_template:
			while (r+1 < len(responses)):
				nxt_response = responses[r+1]
				if nxt_response == '':
					cmb_response += "\n"
					r += 1
					continue
				if nxt_response.startswith(bot_name_dc) or nxt_response.startswith(user_name_dc) or nxt_response.split(' ')[0].endswith(':'):
					break
				else:
					if len(nxt_response) < len(user_name_dcs) or len(nxt_response) < len(bot_name_dcs):
						if user_name_dc.startswith(nxt_response) or bot_name_dc.startswith(nxt_response):
							break
					cmb_response += "\n" + nxt_response
					r += 1

			cmb_response = cmb_response.strip(" \n\t")
			try_again = False
		
			if cmb_response == '':
				if r+1 < len(responses):
					r += 1
					continue
				else:
					try_again = True
			elif cmb_response.startswith(bot_name_dc):
				cmb_response = cmb_response.replace(bot_name_dc, '', 1).lstrip(' ')
				if last_res != cmb_response:
					if cmb_response == '':
						try_again = True
					else:
						got_res = True
				else:
					rando_lvl += rando_add
					try_again = True

			if not got_res and try_again:
				rando_lvl += rando_add
				if rando_lvl <= 1.0:
					text = GenText(in_tokens, rando_lvl)
					text = StripEnd(text, "\n"+user_name_dc).replace("\r", '')
					responses = text.replace(prompt[:-len(bot_name_dc)], '', 1).split("\n")
					r = 0
					continue
				else:
					break

		last_res = cmb_response
		tts_response = cmb_response
		messages.append({"role":bot_name, "content":cmb_response})
		anim_done = False

		while "[CODE]" in cmb_response and "[/CODE]" in cmb_response:
			tag_start = cmb_response.index("[CODE]") + 6
			tag_end = cmb_response.index("[/CODE]")
			if tag_start >= tag_end: break
			code_txt = cmb_response[tag_start:tag_end]
			code_enc = code_txt.replace("\t", "[AI_UI_TAB]").replace("\n", "[AI_UI_LF]")
			cmb_response = cmb_response.replace("[CODE]"+code_txt+"[/CODE]", "[CODE START]"+code_enc+"[CODE END]")

		while sd_model_id != "" and "[AI_IMG]" in cmb_response and "[/AI_IMG]" in cmb_response:
			tag_start = cmb_response.index("[AI_IMG]") + 8
			tag_end = cmb_response.index("[/AI_IMG]")
			if tag_start >= tag_end: break
			img_prompt = cmb_response[tag_start:tag_end]
			img_num = GenImage(img_prompt)
			if img_num > -1:
				tts_response = cmb_response.replace("[AI_IMG]"+img_prompt+"[/AI_IMG]", 'Image description: '+img_prompt)
				cmb_response = cmb_response.replace("[AI_IMG]"+img_prompt+"[/AI_IMG]", "[AI_IMG NUM_"+str(img_num)+"_CHAT_IMG]"+img_prompt+"[AI_IMG END]")
			else:
				break

		if do_talk and tts_ready and tts_response != '':
			tts_response = CleanTextForTTS(tts_response)
			if tts_mode == 0:
				synthesizer.save_to_file("<pitch middle='"+str(voice_ps)+"'/>"+tts_response, chat_wav)
				synthesizer.runAndWait()
				AddEpsilonNoise(chat_wav)
			else:
				if tts_model == None: LoadTTSEngine()
				inputs = processor(text=tts_response, return_tensors="pt")
				speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
				sf.write(chat_wav, speech.numpy(), samplerate=16000)
				ModifySound(chat_wav, voice_vol, voice_rate, voice_ps)
				
			if do_anim: anim_done = AnimAvatar()
			if not (do_anim and anim_done):
				print('PLAY_SPEECH:'+chat_wav, flush=True)
				time.sleep(0.1)

		if do_anim and anim_done:
			print("BOT_OUTPUT:"+cmb_response.replace("\n", "[AI_UI_BR]"), flush=True)
		else:
			print("BOT_NOANIM:"+cmb_response.replace("\n", "[AI_UI_BR]"), flush=True)

		time.sleep(0.1)
		got_res = True
		r += 1
		break #NOTE: disabled multiple bot responses for now

	if not got_res:
		print("BOT_NOANIM:. . .", flush=True)
		messages.append({"role":bot_name, "content":". . ."})
		time.sleep(0.1)

	PrunePrompt()

#TODO: allow user to type multiple responses before bot replies
