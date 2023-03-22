"""
 # Copyright 2023 Bitfreak Software
 # Copyright 2020 Adobe
 # All Rights Reserved.

"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import time
import os, glob
import numpy as np
import cv2
import argparse
import torch
import pickle
import face_alignment
import shutil
import pyttsx3
import transformers
import util.utils as util
from datetime import datetime
from scipy.signal import savgol_filter
from thirdparty.resemblyzer.speaker_emb import get_spk_emb
from src.approaches.train_image_translation import Image_translation_block
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
from src.approaches.train_audio2landmark import Audio2landmark_model
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

###### Config/Vars ######

print("APP_CONFIG:")
work_dir = input().rstrip('/')
model_id = input().rstrip('/')
model_type = int(input())
device = input()
avatar_img = input()

print("AI_CONFIG:")
msg_mem = int(input())
gen_max = int(input())
gen_min = int(input())
rando_lvl = float(input())
prompt_p = int(input())
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
last_prompt = ''
last_res = ''
msg = ''
r = 0

messages = []
responses = []
stop_ids = []
stop_criteria = []

###### TTS Stuff ######

do_talk = True
speech_wav = work_dir+'/MakeItTalk/examples/speech.wav'

synthesizer = pyttsx3.init()
voices = synthesizer.getProperty('voices')
voice_list = '';

if len(voices) < 1:
	sys.exit("Error: could not find any text-to-speech voices installed on this system")
for voice in voices:
    voice_list += "VOICE_NAME:"+voice.name+"VOICE_ID:"+voice.id+"\n"

print(voice_list, flush=True)
time.sleep(0.1)

print("VOICE_CONFIG:")
voice_key = int(input())
voice_vol = float(input())
voice_rate = int(input())
voice_ps = int(input())

synthesizer.setProperty('voice', voices[voice_key].id)
synthesizer.setProperty('rate', voice_rate)
synthesizer.setProperty('volume', voice_vol)

###### GPT Stuff ######

transformers.logging.set_verbosity_error()
#set_seed(int(str(time.time()).replace('.', '')))

if (model_type == 0):
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(model_id, pad_token_id=tokenizer.eos_token_id).to(device)
else:
	tokenizer = GPT2Tokenizer.from_pretrained(model_id)
	model = GPTNeoForCausalLM.from_pretrained(model_id, pad_token_id=tokenizer.eos_token_id).to(device)

###### MAKEITTALK STUFF ######

ADD_NAIVE_EYE = True
CLOSE_INPUT_FACE_MOUTH = False
AMP_LIP_SHAPE_X = 2.
AMP_LIP_SHAPE_Y = 2.
AMP_HEAD_POSE_MOTION = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='face.jpg')
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')

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
		
def AnimFace():
	''' STEP 1: preprocess input single image '''
	img = None
	mit_dev = "cuda" if torch.cuda.is_available() else "cpu"
	if (os.path.exists(avatar_img)):
		img = cv2.imread(avatar_img)
	else:
		img = cv2.imread(opt_parser.jpg)
	predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=mit_dev, flip_input=True)
	shapes = predictor.get_landmarks(img)
	if (not shapes or len(shapes) != 1):
		print('Cannot detect face landmarks. Exit.')
		exit(-1)
	shape_3d = shapes[0]

	if(opt_parser.close_input_face_mouth):
		util.close_input_face_mouth(shape_3d)

	''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
	shape_3d[49:54, 1] += 1.
	shape_3d[55:60, 1] -= 1.
	shape_3d[[37,38,43,44], 1] -=2
	shape_3d[[40,41,46,47], 1] +=2

	''' STEP 2: normalize face as input to audio branch '''
	shape_3d, scale, shift = util.norm_input_face(shape_3d)

	''' STEP 3: Generate audio data as input to audio branch '''
	# audio real data
	au_data = []
	au_emb = []
	
	while True:
		try:
			os.system('ffmpeg -y -loglevel error -i examples/speech.wav -ar 16000 examples/tmp.wav')
			shutil.copyfile('examples/tmp.wav', 'examples/speech.wav')
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
		model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
		try:
			if(len(opt_parser.reuse_train_emb_list) == 0):
				model.test(au_emb=au_emb)
			else:
				model.test(au_emb=None)
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
			model = Image_translation_block(opt_parser, single_test=True)
			with torch.no_grad():
				model.single_test(jpg=img, fls=fl, filename=fls[i], prefix='face')
				print('finish image2image gen', flush=True)
			os.remove(os.path.join('examples', fls[i]))
		break

	if (os.path.isfile('examples/tmp.wav')):
		os.remove('examples/tmp.wav')
		
###### CHAT STUFF ######
		
class StoppingCriteriaKeys(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
	
def StripEnd(txt, ss):
	if txt.endswith(ss):
		txt = txt[:-len(ss)]
	return txt

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

	stop_ids = [tokenizer.encode("\n"+user_name_dcs)[0]]

def InitPrompt():
	global messages, prompt, init_prompt
	prompt = 'Chat log between '+user_name+' and '+bot_name+' on '+datetime.utcnow().strftime("%m/%d/%Y")+"\n"
	init_prompt = prompt
	messages = [init_prompt]
	if prompt_p == 1:
		messages = []
	if prompt_p == 2:
		prompt = ''
		init_prompt = ''
		messages = []
	print("INIT_DONE: "+init_prompt, flush=True)
	time.sleep(0.1)
	
def PrunePrompt():
	global messages, prompt
	i = len(messages) - msg_mem
	if i > 0:
		messages = messages[i::]
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
		prompt = init_prompt+"\n"+prompt

def GenText(m, tn, it, nt, mt, rl, ds):
	stop_criteria = StoppingCriteriaList([StoppingCriteriaKeys(stop_ids)])
	out_tokens = m.generate(it, do_sample=ds, min_new_tokens=nt, max_new_tokens=mt, temperature=rl, stopping_criteria=stop_criteria)
	return StripEnd(tn.decode(out_tokens[0]), '<|endoftext|>')

def GenSimple(m, tn, it, nt, mt, rl, ds):
	out_tokens = m.generate(it, do_sample=ds, min_new_tokens=nt, max_new_tokens=mt, temperature=rl)
	return StripEnd(tn.decode(out_tokens[0]), '<|endoftext|>')

GetUserNames()
InitPrompt()

while (True):

	print("HUMAN_INPUT:")
	msg = input().replace("[AI_UI_BR]", "\n").strip(" \n\t").replace("\r", '')

	if (msg == ''):
		continue	
	if msg == "close_chat":
		break
	if msg == "print_prompt":
		print('----- START PROMPT -----')
		print(prompt)
		print('------ END PROMPT ------')
		time.sleep(0.1)
		continue
	if msg == "update_prompt":
		print("NEW_PROMPT:")
		init_prompt = input().replace("[AI_UI_BR]", "\n").replace("\r", '')
		if prompt_p == 0:
			PrunePrompt()
			if len(messages) > 0:
				messages[0] = init_prompt+"\n"
			else:
				messages = [init_prompt+"\n"]
		PrunePrompt()
		continue
	if msg == "clear_chat":
		InitPrompt()
		continue
	if msg == "config_voice":
		print("VOICE_CONFIG:")
		voice_key = int(input())
		voice_vol = float(input())
		voice_rate = int(input())
		voice_ps = int(input())
		synthesizer.setProperty('voice', voices[voice_key].id)
		synthesizer.setProperty('volume', voice_vol)
		synthesizer.setProperty('rate', voice_rate)
		continue
	if msg == "config_ai":
		print("AI_CONFIG:")
		last_pp = prompt_p
		msg_mem = int(input())
		gen_max = int(input())
		gen_min = int(input())
		rando_lvl = float(input())
		prompt_p = int(input())
		rando_min = rando_lvl
		if prompt_p != last_pp:
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
	if msg == "gen_text":
		print("START_TEXT:")
		start_txt = input().replace("[AI_UI_BR]", "\n").replace("\r", '')
		gmax = int(input())
		gmin = int(input())
		temp = float(input())
		in_tokens = tokenizer(start_txt, return_tensors="pt").input_ids
		text = GenSimple(model, tokenizer, in_tokens, gmin, gmax, temp, True)
		print("GEN_OUTPUT:"+text.replace("\n", "[AI_UI_BR]"), flush=True)
		time.sleep(0.1)
		continue
	if msg == "update_avatar":
		print("AVATAR_IMG:")
		avatar_img = input()
		continue
	if msg == "update_users":
		GetUserNames()
		InitPrompt()
		continue
	if msg == "redo_last":
		prompt = last_prompt
		messages = messages[:-1:]
		rando_lvl += rando_add
	else:
		if msg == "cont_chat":
			prompt += "\n"+bot_name_dc
		else:
			prompt += "\n"+user_name_dcs+msg+"\n"+bot_name_dc
			messages.append(user_name_dcs+msg)
			last_prompt = prompt

	rando_lvl -= rando_sub
	temp = min(1.0, max(rando_lvl, rando_min))

	in_tokens = tokenizer(prompt, return_tensors="pt").input_ids
	text = GenText(model, tokenizer, in_tokens, gen_min, gen_max, temp, True)
	
	print("RAW OUTPUT: "+text)

	text = StripEnd(text, "\n"+user_name_dc).replace("\r", '')
	responses = text.replace(prompt[:-len(bot_name_dc)], '', 1).split("\n")
	got_res = False
	last_res = ''
	r = 0

	while (r < len(responses)):
		cmb_response = responses[r].strip(" \t")

		while (r+1 < len(responses)):
			nxt_response = responses[r+1].strip(" \t")
			if nxt_response == '':
				cmb_response += "\n"
				r += 1
				continue
			if nxt_response.startswith(bot_name_dc) or nxt_response.startswith(user_name_dc) or nxt_response.split(' ')[0].endswith(':'):
				break
			else:
				cmb_response += "\n" + nxt_response
				r += 1

		cmb_response = cmb_response.strip(" \n\t")
		try_again = False
		if cmb_response == '':
			if got_res:
				r += 1
				continue
			else:
				try_again = True
		if cmb_response.startswith(bot_name_dc):
			cmb_response = cmb_response.replace(bot_name_dc, '', 1).lstrip(' ')
			if last_res != cmb_response:
				if cmb_response == '':
					try_again = True
				else:
					got_res = True
			else:
				rando_lvl += rando_add
				break
		else:
			if got_res:
				break
			else:
				try_again = True

		if try_again:
			if got_res:
				break
			rando_lvl += rando_add
			if rando_lvl <= 1.0:
				text = GenText(model, tokenizer, in_tokens, gen_min, gen_max, rando_lvl, True)
				text = StripEnd(text, "\n"+user_name_dc).replace("\r", '')
				responses = text.replace(prompt[:-len(bot_name_dc)], '', 1).split("\n")
				r = 0
				continue
			else:
				break

		last_res = cmb_response
		messages.append(bot_name_dcs + cmb_response)
		
		if do_talk:
			synthesizer.save_to_file("<pitch middle='"+str(voice_ps)+"'/>"+cmb_response, speech_wav)
			synthesizer.runAndWait()
			AnimFace()

		print("BOT_OUTPUT:"+cmb_response.replace("\n", "[AI_UI_BR]"), flush=True)
		time.sleep(0.1)
		r += 1
		
		#NOTE: disabled multiple bot responses for now
		break

	PrunePrompt()

#TODO: allow user to type multiple responses before bot replies

synthesizer.stop()
