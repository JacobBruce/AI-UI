import sys
sys.path.append('Wav2Lip')
from os import listdir, path
import numpy as np
import scipy, cv2, os, argparse, audio
import json, random, string, shutil
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
from moviepy.editor import VideoFileClip, AudioFileClip

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,  help='Name of saved checkpoint to load weights from',
					default='checkpoints/wav2lip_gan.pth')
					
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25.)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)
					
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

got_avatar = False
face_img = None
face_frame = []
face_det_results = []

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ResampleAudio(origin_sample_rate, origin_audio, new_sample_rate):
	new_samps = int(len(origin_audio) * new_sample_rate/origin_sample_rate)
	return torch.tensor(scipy.signal.resample(origin_audio, new_samps))

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				print('ERROR: Image too big to run face detection on GPU. Please resize the image.')
				return []
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			#cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			print('ERROR: Face not detected! Ensure the image contains a face.')
			return []

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(fframe, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		frame_to_save = fframe.copy()
		face, coords = face_det_results[0].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Loading Wav2Lip checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()
	
def load_face(avatar_img):
	global got_avatar, face_det_results, face_frame
	
	got_avatar = False
	print('Processing avatar image ...')
	
	try:
		if not os.path.isfile(avatar_img): return False
		face_frame = cv2.imread(avatar_img)
		face_det_results = face_detect([face_frame])
		if len(face_det_results) > 0:
			got_avatar = True
			return True
		else:
			return False
	except:
		return False

def anim_face(chat_wav):
	if not got_avatar: return False
	
	try:
		if os.path.isfile(chat_wav) and os.path.getsize(chat_wav) > 0:
			fs, signal = scipy.io.wavfile.read(chat_wav)
			if fs != 16000:
				signal = ResampleAudio(fs, signal, 16000)
				scipy.io.wavfile.write('temp/tmp.wav', 16000, signal.numpy().astype(np.int16))
			else:
				shutil.copyfile(chat_wav, 'temp/tmp.wav')
		else:
			print("ERROR: could not open "+chat_wav)
			return False
	except:
		print("ERROR: audio resampling issue")
		return False

	try:
		fps = args.fps
		batch_size = args.wav2lip_batch_size
		
		wav = audio.load_wav('temp/tmp.wav', 16000)
		mel = audio.melspectrogram(wav)

		if np.isnan(mel.reshape(-1)).sum() > 0:
			print('ERROR: Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
			return False

		mel_chunks = []
		mel_idx_multiplier = 80./fps 
		i = 0
		while 1:
			start_idx = int(i * mel_idx_multiplier)
			if start_idx + mel_step_size > len(mel[0]):
				mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
				break
			mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
			i += 1
	except:
		print('ERROR: Wav2Lip melspectrogram issue')
		return False

	try:
		gen = datagen(face_frame.copy(), mel_chunks)
	except:
		print('ERROR: Wav2Lip datagen() issue')
		return False

	try:
		model = load_model(args.checkpoint_path)
	except:
		print('ERROR: Wav2Lip load_model() issue')
		return False
			
	try:
		frame_h, frame_w = face_frame.shape[:-1]
		out = cv2.VideoWriter('temp/result.mp4', cv2.VideoWriter_fourcc(*'mjpg'), fps, (frame_w, frame_h))
		
		for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
			
			for p, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

				f[y1:y2, x1:x2] = p
				out.write(f)

		out.release()
	except:
		print('ERROR: video writing issue')
		return False
	
	if os.path.isfile('temp/result.mp4') and os.path.isfile('temp/tmp.wav'):
		video_clip = VideoFileClip('temp/result.mp4')
		audio_clip = AudioFileClip('temp/tmp.wav')
		
		final_clip = video_clip.set_audio(audio_clip)
		final_clip.write_videofile('results/face_pred_fls_speech_audio_embed.mp4')

	if (os.path.isfile('temp/result.mp4')): os.remove('temp/result.mp4')
	if (os.path.isfile('temp/tmp.wav')): os.remove('temp/tmp.wav')

	return True
