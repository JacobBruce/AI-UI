"""
 # Copyright 2023 Bitfreak Software
 # Copyright 2020 Adobe
 # All Rights Reserved.

"""

import sys
sys.path.append('MakeItTalk/thirdparty/AdaptiveWingLoss')
sys.path.append('MakeItTalk')
import os, glob
import numpy as np
import cv2
import argparse
import torch
import pickle
import shutil
from util import utils
from scipy.io import wavfile
from scipy.signal import savgol_filter, resample
from thirdparty.face_alignment import FaceAlignment
from thirdparty.resemblyzer.speaker_emb import get_spk_emb
from src.approaches.train_image_translation import Image_translation_block
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
from src.approaches.train_audio2landmark import Audio2landmark_model
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

ADD_NAIVE_EYE = True
CLOSE_INPUT_FACE_MOUTH = False
AMP_LIP_SHAPE_X = 2.
AMP_LIP_SHAPE_Y = 2.
AMP_HEAD_POSE_MOTION = 0.4
CKPT_DIR = 'examples/ckpt/'
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

def ResampleAudio(origin_sample_rate, origin_audio, new_sample_rate):
	new_samps = int(len(origin_audio) * new_sample_rate/origin_sample_rate)
	return torch.tensor(resample(origin_audio, new_samps))

def LoadFace(avatar_img):
	global got_avatar, face_img, face_shape

	got_avatar = False
	print('Processing avatar image ...')

	try:
		''' STEP 1: preprocess input single image '''
		if (os.path.exists(avatar_img)):
			face_img = cv2.imread(avatar_img)
		else:
			return False

		with torch.no_grad():
			predictor = FaceAlignment(ckpt_dir=CKPT_DIR, device=MIT_DEV, flip_input=True)
			shapes = predictor.get_landmarks_from_image(face_img)

		if (not shapes or len(shapes) != 1):
			#print('ERROR: Cannot detect face landmarks.')
			return False

		face_shape = shapes[0]
		if (opt_parser.close_input_face_mouth):
			utils.close_input_face_mouth(face_shape)

		''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
		face_shape[49:54, 1] += 1.
		face_shape[55:60, 1] -= 1.
		face_shape[[37,38,43,44], 1] -=2
		face_shape[[40,41,46,47], 1] +=2
	except:
		return False

	del predictor
	got_avatar = True
	return True
		
def AnimFace(chat_wav):
	global face_img

	if not got_avatar: return False
	
	''' STEP 2: normalize face as input to audio branch '''
	shape_3d, scale, shift = utils.norm_input_face(np.copy(face_shape))

	''' STEP 3: Generate audio data as input to audio branch '''
	# audio real data
	au_data = []
	au_emb = []
	anim_done = False
	
	while True:
		try:
			if os.path.isfile(chat_wav) and os.path.getsize(chat_wav) > 0:
				fs, signal = wavfile.read(chat_wav)
				if fs != 16000:
					signal = ResampleAudio(fs, signal, 16000)
					wavfile.write('examples/tmp.wav', 16000, signal.numpy().astype(np.int16))
				else:
					shutil.copyfile(chat_wav, 'examples/tmp.wav')
			else:
				print("ERROR: could not open "+chat_wav)
				break
		except:
			print("ERROR: audio resampling issue")
			break

		#try:
		# au embedding
		me, ae = get_spk_emb('examples/tmp.wav')
		au_emb.append(me.reshape(-1))

		print('Processing audio file', 'examples/tmp.wav')
		c = AutoVC_mel_Convertor('examples')

		au_data_i = c.convert_single_wav_to_autovc_input(audio_filename='examples/tmp.wav',
			   autovc_model_path=opt_parser.load_AUTOVC_name)
		au_data += au_data_i
		#except:
		#	print("ERROR: autovc issue")
		#	break

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
					fl = utils.add_naive_eye(fl)
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
				i2i_model.single_test(jpg=face_img, fls=fl, prefix='face')
				print('finish image2image gen', flush=True)
			os.remove(os.path.join('examples', fls[i]))
			
		anim_done = True
		break
		
	''' STEP 7: merge video and audio '''
	if os.path.isfile('examples/out.mp4') and os.path.isfile('examples/tmp.wav'):
		video_clip = VideoFileClip('examples/out.mp4')
		audio_clip = AudioFileClip('examples/tmp.wav')
		
		video_clip.audio = CompositeAudioClip([audio_clip])
		video_clip.write_videofile('examples/face_pred_fls_speech_audio_embed.mp4')

	if (os.path.isfile('examples/out.mp4')): os.remove('examples/out.mp4')
	if (os.path.isfile('examples/tmp.wav')): os.remove('examples/tmp.wav')
	
	i2i_model = None
	a2l_model = None
		
	return anim_done
