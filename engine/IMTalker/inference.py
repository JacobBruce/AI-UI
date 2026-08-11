import os
import sys
import shutil
import tempfile
import subprocess

_IMT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _IMT_ROOT not in sys.path:
	sys.path.insert(0, _IMT_ROOT)

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import librosa
import face_alignment
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

from generator.FM import FMGenerator
from renderer.models import IMTRenderer

OUTPUT_MP4 = os.path.join(_IMT_ROOT, "results", "face_pred_fls_speech_audio_embed.mp4")

_agent = None
_face_pil = None
_got_avatar = False


class AppConfig:
	def __init__(self):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.seed = 42
		self.fix_noise_seed = False
		self.renderer_path = os.path.join(_IMT_ROOT, "checkpoints", "renderer.ckpt")
		self.generator_path = os.path.join(_IMT_ROOT, "checkpoints", "generator.ckpt")
		self.wav2vec_model_path = os.path.join(_IMT_ROOT, "checkpoints", "wav2vec2-base-960h")
		self.input_size = 512
		self.input_nc = 3
		self.fps = 25.0
		self.rank = "cuda" if self.device == "cuda" else "cpu"
		self.sampling_rate = 16000
		self.audio_marcing = 2
		self.wav2vec_sec = 2.0
		self.attention_window = 5
		self.only_last_features = True
		self.audio_dropout_prob = 0.1
		self.style_dim = 512
		self.dim_a = 512
		self.dim_h = 512
		self.dim_e = 7
		self.dim_motion = 32
		self.dim_c = 32
		self.dim_w = 32
		self.fmt_depth = 8
		self.num_heads = 8
		self.mlp_ratio = 4.0
		self.no_learned_pe = False
		self.num_prev_frames = 10
		self.max_grad_norm = 1.0
		self.ode_atol = 1e-5
		self.ode_rtol = 1e-5
		self.nfe = 10
		self.torchdiffeq_ode_method = "euler"
		self.a_cfg_scale = 2.0
		self.swin_res_threshold = 128
		self.window_size = 8
		self.ref_path = None
		self.pose_path = None
		self.gaze_path = None
		self.aud_path = None
		self.crop = False
		self.source_path = None
		self.driving_path = None


class DataProcessor:
	def __init__(self, opt):
		self.opt = opt
		self.fps = opt.fps
		self.sampling_rate = opt.sampling_rate
		self.fa = face_alignment.FaceAlignment(
			face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False
		)
		local_path = opt.wav2vec_model_path
		if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
			self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
				local_path, local_files_only=True
			)
		else:
			self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
				"facebook/wav2vec2-base-960h"
			)
		size = opt.input_size
		self.transform = transforms.Compose([
			transforms.Resize((size, size)),
			transforms.ToTensor(),
		])

	def process_img(self, img):
		img_arr = np.array(img)
		if img_arr.ndim == 2:
			img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
		elif img_arr.shape[2] == 4:
			img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
		h, w = img_arr.shape[:2]
		try:
			bboxes = self.fa.face_detector.detect_from_image(img_arr)
		except Exception as e:
			print(f"Face detection failed: {e}")
			bboxes = None
		valid_bboxes = []
		if bboxes is not None:
			valid_bboxes = [
				(int(x1), int(y1), int(x2), int(y2), score)
				for (x1, y1, x2, y2, score) in bboxes if score > 0.5
			]
		if not valid_bboxes:
			print("Warning: No face detected. Using center crop.")
			cx, cy = w // 2, h // 2
			half = min(w, h) // 2
			x1_new, x2_new = cx - half, cx + half
			y1_new, y2_new = cy - half, cy + half
		else:
			x1, y1, x2, y2, _ = valid_bboxes[0]
			cx = (x1 + x2) // 2
			cy = (y1 + y2) // 2
			w_face = x2 - x1
			h_face = y2 - y1
			half_side = int(max(w_face, h_face) * 0.8)
			x1_new = cx - half_side
			y1_new = cy - half_side
			x2_new = cx + half_side
			y2_new = cy + half_side
			if x1_new < 0:
				x2_new += (0 - x1_new)
				x1_new = 0
			if y1_new < 0:
				y2_new += (0 - y1_new)
				y1_new = 0
			if x2_new > w:
				x1_new -= (x2_new - w)
				x2_new = w
			if y2_new > h:
				y1_new -= (y2_new - h)
				y2_new = h
			x1_new = max(0, x1_new)
			y1_new = max(0, y1_new)
			x2_new = min(w, x2_new)
			y2_new = min(h, y2_new)
			curr_w = x2_new - x1_new
			curr_h = y2_new - y1_new
			min_side = min(curr_w, curr_h)
			x2_new = x1_new + min_side
			y2_new = y1_new + min_side
		crop_img = img_arr[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
		crop_pil = Image.fromarray(crop_img)
		return crop_pil.resize((self.opt.input_size, self.opt.input_size))

	def process_audio(self, path):
		speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
		return self.wav2vec_preprocessor(
			speech_array, sampling_rate=sampling_rate, return_tensors="pt"
		).input_values[0]


class InferenceAgent:
	def __init__(self, opt):
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		self.opt = opt
		self.device = opt.device
		self.data_processor = DataProcessor(opt)
		print("Loading IMTalker models...")
		self.renderer = IMTRenderer(self.opt).to(self.device)
		self.generator = FMGenerator(self.opt).to(self.device)
		if not os.path.exists(self.opt.renderer_path) or not os.path.exists(self.opt.generator_path):
			raise FileNotFoundError("IMTalker checkpoints not found.")
		self._load_ckpt(self.renderer, self.opt.renderer_path, "gen.")
		self._load_fm_ckpt(self.generator, self.opt.generator_path)
		self.renderer.eval()
		self.generator.eval()
		print("IMTalker models loaded.")

	def _load_ckpt(self, model, path, prefix="gen."):
		checkpoint = torch.load(path, map_location="cpu", weights_only=False)
		state_dict = checkpoint.get("state_dict", checkpoint)
		clean_state_dict = {
			k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)
		}
		model.load_state_dict(clean_state_dict, strict=False)

	def _load_fm_ckpt(self, model, path):
		checkpoint = torch.load(path, map_location="cpu", weights_only=False)
		state_dict = checkpoint.get("state_dict", checkpoint)
		if "model" in state_dict:
			state_dict = state_dict["model"]
		prefix = "model."
		clean_dict = {
			k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
		}
		with torch.no_grad():
			for name, param in model.named_parameters():
				if name in clean_dict:
					param.copy_(clean_dict[name].to(self.device))

	def save_video(self, vid_tensor, fps, audio_path=None):
		with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
			raw_path = tmp.name
		if vid_tensor.dim() == 4:
			vid = vid_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
		else:
			vid = vid_tensor.detach().cpu().numpy()
		if vid.min() < 0:
			vid = (vid + 1) / 2
		vid = np.clip(vid, 0, 1)
		vid = (vid * 255).astype(np.uint8)
		height, width = vid.shape[1], vid.shape[2]
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
		for frame in vid:
			writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		writer.release()
		if not audio_path:
			return raw_path
		with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
			final_path = tmp_out.name
		cmd = [
			"ffmpeg", "-y",
			"-i", raw_path,
			"-i", audio_path,
			"-c:v", "libx264", "-pix_fmt", "yuv420p",
			"-c:a", "aac", "-shortest",
			final_path,
		]
		subprocess.call(cmd)
		if os.path.exists(raw_path):
			os.remove(raw_path)
		return final_path

	@torch.no_grad()
	def run_audio_inference(self, img_pil, aud_path, crop=False, seed=42, nfe=10, cfg_scale=2.0):
		if crop:
			s_pil = self.data_processor.process_img(img_pil)
		else:
			s_pil = img_pil.resize((self.opt.input_size, self.opt.input_size))
		s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
		a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
		data = {
			"s": s_tensor, "a": a_tensor,
			"pose": None, "cam": None, "gaze": None, "ref_x": None,
		}
		f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
		t_lat = self.renderer.latent_token_encoder(s_tensor)
		if isinstance(t_lat, tuple):
			t_lat = t_lat[0]
		data["ref_x"] = t_lat
		torch.manual_seed(seed)
		sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
		d_hat = []
		T = sample.shape[1]
		ta_r = self.renderer.adapt(t_lat, g_r)
		m_r = self.renderer.latent_token_decoder(ta_r)
		for t in range(T):
			ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
			m_c = self.renderer.latent_token_decoder(ta_c)
			out_frame = self.renderer.decode(m_c, m_r, f_r)
			d_hat.append(out_frame)
		vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)
		return self.save_video(vid_tensor, self.opt.fps, aud_path)


def _EnsureAgent():
	global _agent
	if _agent is None:
		_agent = InferenceAgent(AppConfig())
	return _agent


def LoadFace(avatar_img):
	global _face_pil, _got_avatar
	_got_avatar = False
	_face_pil = None
	print("Processing avatar image (IMTalker) ...")
	try:
		if not os.path.isfile(avatar_img):
			return False
		_EnsureAgent()
		_face_pil = Image.open(avatar_img).convert("RGB")
		_got_avatar = True
		return True
	except Exception as e:
		print(f"IMTalker LoadFace failed: {e}")
		return False


def AnimFace(chat_wav):
	global _face_pil
	if not _got_avatar or _face_pil is None:
		return False
	try:
		if not os.path.isfile(chat_wav) or os.path.getsize(chat_wav) <= 0:
			print("ERROR: could not open " + chat_wav)
			return False
		agent = _EnsureAgent()
		tmp_mp4 = agent.run_audio_inference(
			_face_pil, chat_wav,
			crop=agent.opt.crop,
			seed=agent.opt.seed,
			nfe=agent.opt.nfe,
			cfg_scale=agent.opt.a_cfg_scale,
		)
		os.makedirs(os.path.dirname(OUTPUT_MP4), exist_ok=True)
		if os.path.exists(OUTPUT_MP4):
			os.remove(OUTPUT_MP4)
		shutil.move(tmp_mp4, OUTPUT_MP4)
		return True
	except Exception as e:
		print(f"IMTalker AnimFace failed: {e}")
		import traceback
		traceback.print_exc()
		return False
