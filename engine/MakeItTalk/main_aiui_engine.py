"""
 # Copyright 2023 Bitfreak Software
 # Copyright 2020 Adobe
 # All Rights Reserved.

"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import time
import os, gc, glob
import numpy as np
import cv2
import argparse
import torch
import pickle
import shutil
import pyttsx3
import transformers
import torch.nn as nn
import util.utils as util
from peft import PeftModel
from datetime import datetime
from scipy.signal import savgol_filter
from thirdparty.face_alignment import FaceAlignment
from thirdparty.resemblyzer.speaker_emb import get_spk_emb
from src.approaches.train_image_translation import Image_translation_block
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
from src.approaches.train_audio2landmark import Audio2landmark_model
from diffusers import StableDiffusionPipeline
from transformers import pipeline, GPTNeoForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

###### Config/Vars ######
sys.stdout.reconfigure(encoding='utf-8')

print("APP_CONFIG:")
work_dir = input().rstrip('/')
model_id = input().rstrip('/')
sd_model_id = input().rstrip('/')
model_type = int(input())
model_args = input().split(',')
device = input()
start_meth = input()
avatar_img = input()

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

sd_vram_mb = 0 if sd_model_id == '' else 2900
reserve_vram_mb = 300 + sd_vram_mb
reserved_vram = None

model_adapter = ''
sd_safety_check = True
sd_use_float16 = True
load_in_8bit = False
custom_model_code = False
custom_token_code = False
torch_dtype=torch.float16

for arg in model_args:
	carg = arg.strip(" ")
	if carg.startswith("model_adapter="):
		model_adapter = carg.replace("model_adapter=", "", 1)
	elif carg.startswith("torch_dtype="):
		dtype = carg.replace("torch_dtype=", "", 1).lower()
		if dtype == "float32":
			torch_dtype = torch.float32
		elif dtype == "float16":
			torch_dtype = torch.float16
		elif dtype == "bfloat16":
			torch_dtype = torch.bfloat16
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
	elif carg.startswith("sd_safety_check="):
		sd_safety_check = carg.replace("sd_safety_check=", "", 1)
	elif carg.startswith("sd_use_float16="):
		sd_use_float16 = carg.replace("sd_use_float16=", "", 1)
	elif carg.startswith("reserve_vram_mb="):
		reserve_vram_mb = int(carg.replace("reserve_vram_mb=", "", 1))
		if reserve_vram_mb > 300:
			sd_vram_mb = reserve_vram_mb - 300
		else:
			sd_vram_mb = 0

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

###### TTS Stuff ######

do_talk = True
do_anim = True
speech_wav = work_dir+'/MakeItTalk/examples/speech.wav'
read_wav = work_dir+'/MakeItTalk/examples/read.wav'

synthesizer = pyttsx3.init()
voices = synthesizer.getProperty('voices')
voice_list = '';

if len(voices) < 1:
	sys.exit("ERROR: could not find any text-to-speech voices installed on this system")
for voice in voices:
    voice_list += "VOICE_NAME:"+voice.name+"VOICE_ID:"+voice.id+"\n"

print(voice_list, flush=True)
time.sleep(0.1)

print("VOICE_CONFIG:")
voice_key = int(input())
voice_vol = float(input())
voice_rate = int(input())
voice_ps = int(input())
talk_mode = int(input())

synthesizer.setProperty('voice', voices[voice_key].id)
synthesizer.setProperty('rate', voice_rate)
synthesizer.setProperty('volume', voice_vol)

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
		elif (model_type == 3):
			tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
		elif (model_type == 4):
			tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
	except:
		if start_meth == "text" or start_meth == "both":
			sys.exit("ERROR: failed to load text tokenizer")

def LoadModel():
	global model
	#reserve VRAM for other models
	ReserveVRAM()
	
	if tokenizer == None: LoadTokenizer()

	if (model_type == 0):
		if device == "auto":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, trust_remote_code=custom_model_code, device_map="auto")
		elif device == "cuda":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, trust_remote_code=custom_model_code).to(device)
		else:
			model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, trust_remote_code=custom_model_code).to(device)

	elif (model_type == 1):
		if device == "auto":
			model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, pad_token_id=tokenizer.eos_token_id, device_map="auto")
		elif device == "cuda":
			model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, pad_token_id=tokenizer.eos_token_id).to(device)
		else:
			model = GPTNeoForCausalLM.from_pretrained(model_id, pad_token_id=tokenizer.eos_token_id, low_cpu_mem_usage=True).to(device)

	elif (model_type == 2):
		if device == "cuda" or device == "auto":
			if device == "auto":
				model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, device_map="auto")
			else:
				model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit).to(device)
			if model_adapter != '':
				model = PeftModel.from_pretrained(model, model_adapter, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit)
		else:
			model = LlamaForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, device_map={"": device})
			if model_adapter != '':
				model = PeftModel.from_pretrained(model, model_adapter, device_map={"": device})

	elif (model_type == 3):
		if device == "auto":
			model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True, device_map="auto")
		elif device == "cuda":
			model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
		else:
			model = AutoModel.from_pretrained(model_id, low_cpu_mem_usage=True, trust_remote_code=True).to(device)

	elif (model_type == 4):
		if device == "auto":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
		elif device == "cuda":
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
		else:
			model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, trust_remote_code=True).to(device)

	model.eval()

	if torch.__version__ >= "2" and sys.platform != "win32":
		model = torch.compile(model)

###### STABLE DIFF STUFF ######

sd_model = None

def LoadSDModel(exit_on_error=True):
	global sd_model, reserve_vram_mb
	orig_rvram = reserve_vram_mb

	try:
		if reserve_vram_mb > sd_vram_mb:
			reserve_vram_mb -= sd_vram_mb

		if sd_safety_check == True or sd_safety_check.lower() == "true" or sd_safety_check == "1":
			if sd_use_float16 == True or sd_use_float16.lower() == "true" or sd_use_float16 == "1":
				sd_model = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16, revision="fp16")
			else:
				sd_model = StableDiffusionPipeline.from_pretrained(sd_model_id)
		elif sd_use_float16 == True or sd_use_float16.lower() == "true" or sd_use_float16 == "1":
			sd_model = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16, revision="fp16", safety_checker=None)
		else:
			sd_model = StableDiffusionPipeline.from_pretrained(sd_model_id, safety_checker=None)
		
		sd_model.enable_model_cpu_offload()
		sd_model.enable_attention_slicing("max")
	except:
		reserve_vram_mb = orig_rvram
		sd_model = None
		CleanVRAM()
		ReserveVRAM()
		if exit_on_error:
			sys.exit("ERROR: failed to load stable diffusion model")
		else:
			print("ERROR: failed to load stable diffusion model")
		return False
	
	return True

def GenImage(image_prompt, neg_prompt="NONE", infer_steps=50, guidance=7.5, img_width="auto", img_height="auto"):
	global sd_model

	if image_prompt == '': return -1
	
	CleanVRAM()

	if sd_model == None:
		if not LoadSDModel(False): return -1
		
	try:
		clean_prompt = image_prompt.strip(' ').replace("\n", ' ').replace("\r", '')

		with torch.inference_mode():
			if neg_prompt == "NONE":
				if img_width == "auto" or img_height == "auto":
					output = sd_model(prompt=clean_prompt, num_inference_steps=infer_steps, guidance_scale=guidance)
				else:
					output = sd_model(prompt=clean_prompt, num_inference_steps=infer_steps, guidance_scale=guidance, width=img_width, height=img_height)
			else:
				if img_width == "auto" or img_height == "auto":
					output = sd_model(prompt=clean_prompt, negative_prompt=neg_prompt, num_inference_steps=infer_steps, guidance_scale=guidance)
				else:
					output = sd_model(prompt=clean_prompt, negative_prompt=neg_prompt, num_inference_steps=infer_steps, guidance_scale=guidance, width=img_width, height=img_height)

		img_dir = work_dir+"/ai_images/"
		if os.path.exists(img_dir):
			img_num = CountFiles(img_dir)
		else:
			os.mkdir(img_dir)
			img_num = 0

		output.images[0].save(img_dir+"image_"+str(img_num)+".png")
	except:
		print("ERROR: failed to generate or save image")
		return -1

	CleanVRAM()
	ReserveVRAM()
	
	return img_num

###### MAKEITTALK STUFF ######

ADD_NAIVE_EYE = True
CLOSE_INPUT_FACE_MOUTH = False
AMP_LIP_SHAPE_X = 2.
AMP_LIP_SHAPE_Y = 2.
AMP_HEAD_POSE_MOTION = 0.4
CKPT_DIR = work_dir + '/MakeItTalk/examples/ckpt/'
MIT_DEV = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='face.jpg')
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth')
parser.add_argument('--load_i2i_name', type=str, default='examples/ckpt/ckpt_116_i2i_state.pt')

parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[])
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')
parser.add_argument('--comp_dev', type=str, default=MIT_DEV)

parser.add_argument('--test_end2end', default=False, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')

opt_parser = parser.parse_args()

got_avatar = False
face_img = None
face_shape = []

def LoadAvatar():
	global got_avatar, face_img, face_shape

	CleanVRAM()
	got_avatar = False
	print('Processing avatar image ...')

	try:
		''' STEP 1: preprocess input single image '''
		if (os.path.exists(avatar_img)):
			face_img = cv2.imread(avatar_img)
		else:
			face_img = cv2.imread(opt_parser.jpg)

		with torch.no_grad():
			predictor = FaceAlignment(ckpt_dir=CKPT_DIR, device=MIT_DEV, flip_input=True)
			shapes = predictor.get_landmarks_from_image(face_img)

		if (not shapes or len(shapes) != 1):
			#print('ERROR: Cannot detect face landmarks.')
			return

		face_shape = shapes[0]
		if (opt_parser.close_input_face_mouth):
			util.close_input_face_mouth(face_shape)

		''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
		face_shape[49:54, 1] += 1.
		face_shape[55:60, 1] -= 1.
		face_shape[[37,38,43,44], 1] -=2
		face_shape[[40,41,46,47], 1] +=2
	except:
		return

	del predictor
	CleanVRAM()
	ReserveVRAM()
	got_avatar = True
		
def AnimFace():
	global model, face_img

	if not got_avatar: return False

	CleanVRAM()
	
	''' STEP 2: normalize face as input to audio branch '''
	shape_3d, scale, shift = util.norm_input_face(np.copy(face_shape))

	''' STEP 3: Generate audio data as input to audio branch '''
	# audio real data
	au_data = []
	au_emb = []
	anim_done = False
	
	while True:
		try:
			if os.path.isfile('examples/speech.wav') and os.path.getsize('examples/speech.wav') > 0:
				os.system('ffmpeg -y -loglevel error -i examples/speech.wav -ar 16000 examples/tmp.wav')
				shutil.copyfile('examples/tmp.wav', 'examples/speech.wav')
			else:
				print("ERROR: could not open speech.wav")
				break
		except:
			print("ERROR: ffmpeg issue")

		try:
			# au embedding
			me, ae = get_spk_emb('examples/speech.wav')
			au_emb.append(me.reshape(-1))

			print('Processing audio file', 'examples/speech.wav')
			c = AutoVC_mel_Convertor('examples')

			au_data_i = c.convert_single_wav_to_autovc_input(audio_filename='examples/speech.wav',
				   autovc_model_path=opt_parser.load_AUTOVC_name)
			au_data += au_data_i
		except:
			print("ERROR: autovc issue")
			break

		# landmark fake placeholder
		fl_data = []
		rot_tran, rot_quat, anchor_t_shape = [], [], []
		for au, info in au_data:
			au_length = au.shape[0]
			fl = np.zeros(shape=(au_length, 68 * 3))
			fl_data.append((fl, info))
			rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
			rot_quat.append(np.zeros(shape=(au_length, 4)))
			anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

		if (os.path.exists('examples/dump/random_val_fl.pickle')):
			os.remove('examples/dump/random_val_fl.pickle')
		if (os.path.exists('examples/dump/random_val_fl_interp.pickle')):
			os.remove('examples/dump/random_val_fl_interp.pickle')
		if (os.path.exists('examples/dump/random_val_au.pickle')):
			os.remove('examples/dump/random_val_au.pickle')
		if (os.path.exists('examples/dump/random_val_gaze.pickle')):
			os.remove('examples/dump/random_val_gaze.pickle')

		with open('examples/dump/random_val_fl.pickle', 'wb') as fp:
			pickle.dump(fl_data, fp)
		with open('examples/dump/random_val_au.pickle', 'wb') as fp:
			pickle.dump(au_data, fp)
		with open('examples/dump/random_val_gaze.pickle', 'wb') as fp:
			gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
			pickle.dump(gaze, fp)

		''' STEP 4: RUN audio->landmark network'''
		a2l_model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
		try:
			if (len(opt_parser.reuse_train_emb_list) == 0):
				a2l_model.test(au_emb=au_emb)
			else:
				a2l_model.test(au_emb=None)
		except:
			print("ERROR: Audio2landmark_model.test() issue")
			break

		''' STEP 5: de-normalize the output to the original image scale '''
		fls = glob.glob1('examples', 'pred_fls_*.txt')
		fls.sort()

		for i in range(0,len(fls)):
			fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
			fl[:, :, 0:2] = -fl[:, :, 0:2]
			fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

			if (ADD_NAIVE_EYE):
				try:
					fl = util.add_naive_eye(fl)
				except:
					print("ERROR: add_naive_eye() issue") 

			# additional smooth
			fl = fl.reshape((-1, 204))
			try:
				fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
				fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
			except:
				print("ERROR: savgol_filter() issue") 
			fl = fl.reshape((-1, 68, 3))

			''' STEP 6: Imag2image translation '''
			i2i_model = Image_translation_block(opt_parser, single_test=True)
			with torch.no_grad():
				i2i_model.single_test(jpg=face_img, fls=fl, filename=fls[i], prefix='face')
				print('finish image2image gen', flush=True)
			os.remove(os.path.join('examples', fls[i]))
			
		anim_done = True
		break

	if (os.path.isfile('examples/tmp.wav')):
		os.remove('examples/tmp.wav')
	
	i2i_model = None
	a2l_model = None
	CleanVRAM()
	ReserveVRAM()
		
	return anim_done

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

	stop_ids = tokenizer.encode("\n"+user_name_dc)

def InitPrompt(have_prompt=False):
	global messages, prompt, init_prompt
	if have_prompt:
		prompt = init_prompt
	else:
		prompt = 'Chat log between '+user_name+' and '+bot_name+' on '+datetime.utcnow().strftime("%m/%d/%Y")+"\n"
		init_prompt = prompt
	messages = [init_prompt]
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
			prompt += messages[i] + "\n"
		else:
			prompt += messages[i]
		i += 1
	if prompt_p == 1:
		if prompt == '':
			prompt = init_prompt
		else:
			prompt = init_prompt+"\n"+prompt

def GenText(it, rl):
	global model, tokenizer
	CleanVRAM()
	if model == None: LoadModel()
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

if start_meth == "text":
	LoadModel()
elif start_meth == "image":
	LoadSDModel()
elif start_meth == "both":
	LoadModel()
	LoadSDModel()

LoadTokenizer()
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
				messages[0] = init_prompt
			else:
				messages = [init_prompt]
		PrunePrompt()
		continue
	elif msg == "clear_chat":
		InitPrompt(True)
		rando_lvl = rando_min
		continue
	elif msg == "config_voice":
		print("VOICE_CONFIG:")
		voice_key = int(input())
		voice_vol = float(input())
		voice_rate = int(input())
		voice_ps = int(input())
		talk_mode = int(input())
		synthesizer.setProperty('voice', voices[voice_key].id)
		synthesizer.setProperty('volume', voice_vol)
		synthesizer.setProperty('rate', voice_rate)
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
				if len(messages) > 0 and messages[0] == init_prompt:
					messages = messages[1::]
			if prompt_p == 0:
				if len(messages) > 0:
					messages[0] = init_prompt
				else:
					messages = [init_prompt]
		PrunePrompt()
		continue
	elif msg == "gen_image":
		print("IMAGE_PROMPT:")
		img_prompt = input()
		neg_prompt = input()
		infer_steps = int(input())
		guidance = float(input())
		img_width = input()
		img_height = input()
		img_num = GenImage(img_prompt, neg_prompt, infer_steps, guidance, img_width, img_height)
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
		if device != "cpu" and torch.cuda.is_available():
			in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids.to("cuda")
		else:
			in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids
		text = GenNoStop(in_tokens, gmin, gmax, temp, topk, topp, typp, repp)
		print("GEN_OUTPUT:"+text.replace("\n", "[AI_UI_BR]"), flush=True)
		time.sleep(0.1)
		continue
	elif msg == "read_text":
		print("TTS_TEXT:")
		tts_text = CleanTextForTTS(input().replace("[AI_UI_BR]", "\n"))
		synthesizer.save_to_file("<pitch middle='"+str(voice_ps)+"'/>"+tts_text, read_wav)
		synthesizer.runAndWait()
		print('PLAY_SPEECH:'+read_wav, flush=True)
		time.sleep(0.1)
		continue
	elif msg == "update_avatar":
		print("AVATAR_IMG:")
		avatar_img = input()
		LoadAvatar()
		if got_avatar:
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
	else:
		prompt += "\n"+user_name_dcs+msg+"\n"+bot_name_dc
		messages.append(user_name_dcs+msg)

	rando_lvl -= rando_sub
	temp = min(1.0, max(rando_lvl, rando_min))

	if device != "cpu" and torch.cuda.is_available():
		in_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
	else:
		in_tokens = tokenizer(prompt, return_tensors="pt").input_ids

	text = GenText(in_tokens, temp)
	#print("RAW OUTPUT: "+text, flush=True)
	#time.sleep(0.1)

	text = StripEnd(text, "\n"+user_name_dc).replace("\r", '')
	responses = text.replace(prompt[:-len(bot_name_dc)], '', 1).split("\n")
	got_res = False
	last_res = ''
	r = 0

	while (r < len(responses)):
		cmb_response = responses[r].strip(" \t")

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
		messages.append(bot_name_dcs + cmb_response)
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

		if do_talk and tts_response != '':
			tts_response = CleanTextForTTS(tts_response)
			synthesizer.save_to_file("<pitch middle='"+str(voice_ps)+"'/>"+tts_response, speech_wav)
			synthesizer.runAndWait()
			if do_anim: anim_done = AnimFace()
			if not (do_anim and anim_done):
				print('PLAY_SPEECH:'+speech_wav, flush=True)
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
		messages.append(bot_name_dcs + ". . .")
		time.sleep(0.1)

	PrunePrompt()

#TODO: allow user to type multiple responses before bot replies
