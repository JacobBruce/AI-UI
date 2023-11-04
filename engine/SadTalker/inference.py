import sys
sys.path.append('SadTalker')
from glob import glob
import shutil
import torch
from time import strftime
import os, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

parser = ArgumentParser()
parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
parser.add_argument("--result_dir", default='./results', help="path to output")
parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
parser.add_argument("--cpu", dest="cpu", action="store_true") 
parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
parser.add_argument("--preprocess", default='resize', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 

# net structure and parameters
parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
parser.add_argument('--init_path', type=str, default=None, help='Useless')
parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

# default renderer parameters
parser.add_argument('--focal', type=float, default=1015.)
parser.add_argument('--center', type=float, default=112.)
parser.add_argument('--camera_d', type=float, default=10.)
parser.add_argument('--z_near', type=float, default=5.)
parser.add_argument('--z_far', type=float, default=15.)

args = parser.parse_args()

os.makedirs('./SadTalker/results', exist_ok=True)

face_img = ''
crop_info = None
crop_pic_path = ''
first_coeff_path = ''

sadtalker_paths = init_path(args.checkpoint_dir, 'src/config', args.size, False, args.preprocess)

if torch.cuda.is_available() and not args.cpu:
	args.device = "cuda"
else:
	args.device = "cpu"
	
def loadface(avatar_img):
	global face_img, first_coeff_path, crop_pic_path, crop_info
	
	face_img = ''
	print('Processing avatar image ...')
	
	try:
		preprocess_model = CropAndExtract(sadtalker_paths, args.device)

		#crop image and extract 3dmm from image
		first_frame_dir = os.path.join(args.result_dir, 'first_frame_dir')
		os.makedirs(first_frame_dir, exist_ok=True)

		print('3DMM Extraction for source image')
		first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(avatar_img, first_frame_dir, args.preprocess,\
																				 source_image_flag=True, pic_size=args.size)
		if first_coeff_path is None:
			print("ERROR: Can't get the coeffs of the input")
			return False
	except:
		return False
		
	face_img = avatar_img
	return True

def animface(chat_wav):
	if face_img == '': return False

	save_dir = os.path.join(args.result_dir, 'temp')
	os.makedirs(save_dir, exist_ok=True)
	pose_style = args.pose_style
	batch_size = args.batch_size
	input_yaw_list = args.input_yaw
	input_pitch_list = args.input_pitch
	input_roll_list = args.input_roll
	ref_eyeblink = args.ref_eyeblink
	ref_pose = args.ref_pose

	try:
		audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
	except:
		print("ERROR: Audio2Coeff init issue")
		return False
		
	try:
		animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)
	except:
		print("ERROR: AnimateFromCoeff init issue")
		return False

	ref_eyeblink_coeff_path=None
	ref_pose_coeff_path=None

	try: #audio2ceoff
		batch = get_data(first_coeff_path, chat_wav, args.device, ref_eyeblink_coeff_path, still=args.still)
		coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
	except:
		print("ERROR: SadTalker audio2ceoff issue")
		return False

	# 3dface render
	#if args.face3dvis:
	#	from src.face3d.visualize import gen_composed_video
	#	gen_composed_video(args, args.device, first_coeff_path, coeff_path, chat_wav, os.path.join(save_dir, '3dface.mp4'))

	try: #coeff2video
		data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, chat_wav, 
									batch_size, input_yaw_list, input_pitch_list, input_roll_list,
									expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)

		result = animate_from_coeff.generate(data, save_dir, face_img, crop_info, \
									enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
	except:
		print("ERROR: SadTalker coeff2video issue")
		return False

	shutil.move(result, os.path.join(args.result_dir, 'face_pred_fls_speech_audio_embed.mp4'))
	#print('The generated video is named:', 'face_pred_fls_speech_audio_embed.mp4')

	if not args.verbose:
		shutil.rmtree(save_dir)
	
	return True
