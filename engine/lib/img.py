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
from urllib.request import urlopen

from lib.local_ml import EnsureLocalMl, ResolveTorchDtype

torch = None
load_image = None
PipelineQuantizationConfig = None
DiffusersBitsAndBytesConfig = None
TransformersBitsAndBytesConfig = None
StableDiffusionPipeline = None
StableDiffusionXLPipeline = None
StableDiffusion3Pipeline = None
QwenImagePipeline = None
FluxPipeline = None
GGUFQuantizationConfig = None
AutoencoderKL = None
AutoencoderKLQwenImage = None
DDIMScheduler = None
UNet2DConditionModel = None
SD3Transformer2DModel = None
FluxTransformer2DModel = None
QwenImageTransformer2DModel = None
CLIPTextModel = None
CLIPTextModelWithProjection = None
T5EncoderModel = None
Qwen2_5_VLForConditionalGeneration = None
_img_ml_ready = False


def _EnsureImgMl():
	global torch, load_image, PipelineQuantizationConfig
	global DiffusersBitsAndBytesConfig, TransformersBitsAndBytesConfig
	global StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
	global QwenImagePipeline, FluxPipeline, GGUFQuantizationConfig
	global AutoencoderKL, AutoencoderKLQwenImage, DDIMScheduler, UNet2DConditionModel
	global SD3Transformer2DModel, FluxTransformer2DModel, QwenImageTransformer2DModel
	global CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel
	global Qwen2_5_VLForConditionalGeneration, _img_ml_ready
	if _img_ml_ready:
		return
	EnsureLocalMl("Image generation")
	import torch as _torch
	from diffusers.utils import load_image as _load_image
	from diffusers.quantizers import PipelineQuantizationConfig as _PipelineQuantizationConfig
	from diffusers import BitsAndBytesConfig as _DiffusersBitsAndBytesConfig
	from transformers import BitsAndBytesConfig as _TransformersBitsAndBytesConfig
	from diffusers import (
		StableDiffusionPipeline as _StableDiffusionPipeline,
		StableDiffusionXLPipeline as _StableDiffusionXLPipeline,
		StableDiffusion3Pipeline as _StableDiffusion3Pipeline,
		QwenImagePipeline as _QwenImagePipeline,
		FluxPipeline as _FluxPipeline,
		GGUFQuantizationConfig as _GGUFQuantizationConfig,
		AutoencoderKL as _AutoencoderKL,
		AutoencoderKLQwenImage as _AutoencoderKLQwenImage,
		DDIMScheduler as _DDIMScheduler,
		UNet2DConditionModel as _UNet2DConditionModel,
		SD3Transformer2DModel as _SD3Transformer2DModel,
		FluxTransformer2DModel as _FluxTransformer2DModel,
		QwenImageTransformer2DModel as _QwenImageTransformer2DModel,
	)
	from transformers import (
		CLIPTextModel as _CLIPTextModel,
		CLIPTextModelWithProjection as _CLIPTextModelWithProjection,
		T5EncoderModel as _T5EncoderModel,
		Qwen2_5_VLForConditionalGeneration as _Qwen2_5_VLForConditionalGeneration,
	)
	torch = _torch
	load_image = _load_image
	PipelineQuantizationConfig = _PipelineQuantizationConfig
	DiffusersBitsAndBytesConfig = _DiffusersBitsAndBytesConfig
	TransformersBitsAndBytesConfig = _TransformersBitsAndBytesConfig
	StableDiffusionPipeline = _StableDiffusionPipeline
	StableDiffusionXLPipeline = _StableDiffusionXLPipeline
	StableDiffusion3Pipeline = _StableDiffusion3Pipeline
	QwenImagePipeline = _QwenImagePipeline
	FluxPipeline = _FluxPipeline
	GGUFQuantizationConfig = _GGUFQuantizationConfig
	AutoencoderKL = _AutoencoderKL
	AutoencoderKLQwenImage = _AutoencoderKLQwenImage
	DDIMScheduler = _DDIMScheduler
	UNet2DConditionModel = _UNet2DConditionModel
	SD3Transformer2DModel = _SD3Transformer2DModel
	FluxTransformer2DModel = _FluxTransformer2DModel
	QwenImageTransformer2DModel = _QwenImageTransformer2DModel
	CLIPTextModel = _CLIPTextModel
	CLIPTextModelWithProjection = _CLIPTextModelWithProjection
	T5EncoderModel = _T5EncoderModel
	Qwen2_5_VLForConditionalGeneration = _Qwen2_5_VLForConditionalGeneration
	if E is not None and (getattr(E, "im_torch_dtype", None) is None or isinstance(E.im_torch_dtype, str)):
		E.im_torch_dtype = ResolveTorchDtype(getattr(E, "im_torch_dtype", None) or "float16", default="float16")
	if E is not None and getattr(E, "seed_gen", None) is None:
		E.comp_dev = 'cuda' if getattr(E, "device", "auto") != 'cpu' and torch.cuda.is_available() else 'cpu'
		E.seed_gen = torch.Generator(device=E.comp_dev)
	_img_ml_ready = True


def DiffIMGMode(safety_check, vae_file, lora_file, lora_dir, lora_scale):
	if safety_check != E.last_sc_val or vae_file != E.last_vf_val or lora_file != E.last_lf_val or lora_dir != E.last_ld_val or lora_scale != E.last_ls_val:
		result = 2
	else:
		result = int(safety_check != E.im_safety_check or vae_file != E.im_vae_file or lora_file != E.im_lora_file or lora_dir != E.im_lora_dir or lora_scale != E.im_lora_scale)
	E.last_sc_val = safety_check
	E.last_vf_val = vae_file
	E.last_lf_val = lora_file
	E.last_ld_val = lora_dir
	E.last_ls_val = lora_scale
	return result


def LoadIMGModule(module_name, module_file, module_loader, module_list, bnb_config):
	_EnsureImgMl()
	if module_loader == None:
		sys.exit(f"ERROR: got unexpected model file ({module_file})")
	if module_file.lower().endswith(".gguf"):
		quant_config = GGUFQuantizationConfig(compute_dtype=E.im_torch_dtype)
		if module_name in module_list: module_list.remove(module_name)
	elif module_name in module_list:
		quant_config = bnb_config
		module_list.remove(module_name)
	else:
		quant_config = None
	return module_loader.from_single_file(
	 module_file,
	 torch_dtype = E.im_torch_dtype,
	 local_files_only = E.im_local_files,
	 quantization_config = quant_config
	)


def LoadIMGModel(exit_on_error=True, safety_check=None, vae_file=None, lora_file=None, lora_dir=None, lora_scale=None):
	_EnsureImgMl()
	E.im_model = None
	vae_model = None
	result = True
 
	safety_check = E.im_safety_check if safety_check == None else safety_check
	vae_file = E.im_vae_file if vae_file == None else vae_file
	lora_file = E.im_lora_file if lora_file == None else lora_file
	lora_dir = E.im_lora_dir if lora_dir == None else lora_dir
	lora_scale = E.im_lora_scale if lora_scale == None else lora_scale
 
	# Free the im_res hold so the image model can use that VRAM; keep min_res only.
	E.CleanVRAM()
	E._vram_hold_mb = E.min_res_vram_mb
	E.ReserveVRAM()

	if E.im_model_class != None:
		model_loader = getattr(importlib.import_module("diffusers"), E.im_model_class)
		ivae_loader = AutoencoderKL
		itran_loader = None
		ienc_loader = [CLIPTextModel, None, None]
		comps_to_quant = []
	elif E.imodel_type == 0:
		model_loader = StableDiffusionPipeline
		ivae_loader = AutoencoderKL
		itran_loader = None
		ienc_loader = [CLIPTextModel, None, None]
		comps_to_quant = ["unet"]
	elif E.imodel_type == 1:
		model_loader = StableDiffusionXLPipeline
		ivae_loader = AutoencoderKL
		itran_loader = None
		ienc_loader = [CLIPTextModel, CLIPTextModelWithProjection, None]
		comps_to_quant = ["unet", "text_encoder_2"]
	elif E.imodel_type == 2:
		model_loader = StableDiffusion3Pipeline
		ivae_loader = AutoencoderKL
		itran_loader = SD3Transformer2DModel
		ienc_loader = [CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel]
		comps_to_quant = ["transformer", "text_encoder_2", "text_encoder_3"]
	elif E.imodel_type == 3:
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
		tbnb_config = TransformersBitsAndBytesConfig(load_in_8bit=True) if E.im_load_in_8bit else None
		dbnb_config = DiffusersBitsAndBytesConfig(load_in_8bit=True) if E.im_load_in_8bit else None

		gen_kwargs = dict(
		 torch_dtype=E.im_torch_dtype,
		 variant=E.im_variant, use_safetensors=E.im_use_safetensors,
		 local_files_only=E.im_local_files, original_config_file=E.im_config_file
		)
  
		if E.im_enc_file != '':
			gen_kwargs["text_encoder"] = LoadIMGModule("text_encoder", E.im_enc_file, ienc_loader[0], comps_to_quant, tbnb_config)
		if E.im_enc2_file != '':
			gen_kwargs["text_encoder_2"] = LoadIMGModule("text_encoder_2", E.im_enc2_file, ienc_loader[1], comps_to_quant, tbnb_config)
		if E.im_enc3_file != '':
			gen_kwargs["text_encoder_3"] = LoadIMGModule("text_encoder_3", E.im_enc3_file, ienc_loader[2], comps_to_quant, tbnb_config)
  
		if vae_file != '':
			gen_kwargs["vae"] = LoadIMGModule("vae", vae_file, ivae_loader, comps_to_quant, dbnb_config)
  
		if E.im_unet_file != '':
			gen_kwargs["unet"] = LoadIMGModule("unet", E.im_unet_file, UNet2DConditionModel, comps_to_quant, dbnb_config)

		elif E.im_tran_file != '':
			if itran_loader == None:
				if E.im_model_class == None:
					sys.exit(f"ERROR: {model_loader.__name__} does not use a transformer model")
				else:
					sys.exit("ERROR: cannot use im_tran_file with im_model_class")
			gen_kwargs["transformer"] = LoadIMGModule("transformer", E.im_tran_file, itran_loader, comps_to_quant, dbnb_config)
  
		if not safety_check: gen_kwargs["safety_checker"] = None
  
		if E.im_load_in_8bit and len(comps_to_quant) > 0:
			gen_kwargs["quantization_config"] = PipelineQuantizationConfig(
			 quant_backend = "bitsandbytes_8bit",
			 quant_kwargs = { "load_in_8bit": True },
			 components_to_quantize = comps_to_quant
			)
  
		if E.im_from_single_file or E.im_model_id.endswith(".safetensors") or E.im_model_id.endswith(".ckpt"):
			E.im_model = model_loader.from_single_file(E.im_model_id, **gen_kwargs)
		else:
			E.im_model = model_loader.from_pretrained(E.im_model_id, **gen_kwargs)
  
		if E.device == "auto":
			if torch.cuda.is_available() and not E.im_cpu_offload:
				E.im_model = E.im_model.to("cuda")
		else:
			E.im_model = E.im_model.to(E.device)
  
		if lora_file != '':
			E.im_model.load_lora_weights(lora_dir, weight_name=lora_file)
			E.im_model.fuse_lora(lora_scale=lora_scale)
  
		if E.torch_compile:
			E.im_model.vae.to(memory_format=torch.channels_last)
			E.im_model.vae.decode = torch.compile(E.im_model.vae.decode, mode=E.compile_mode)
			if hasattr(E.im_model, "transformer"):
				E.im_model.transformer.to(memory_format=torch.channels_last)
				E.im_model.transformer = torch.compile(E.im_model.transformer, mode=E.compile_mode)
			elif hasattr(E.im_model, "unet"):
				E.im_model.unet.to(memory_format=torch.channels_last)
				E.im_model.unet = torch.compile(E.im_model.unet, mode=E.compile_mode)
  
		if E.im_cpu_offload:
			E.im_model.enable_model_cpu_offload()
   
		if E.im_att_slicing:
			E.im_model.enable_attention_slicing("max")
  
	except Exception as e:
		E.im_model = None
		result = False
		E._im_load_error = str(e)
		from lib import ipc
		ipc.LogException("LoadIMGModel", e)
		if exit_on_error:
			sys.exit(f"ERROR: failed to load image model ({e})")
		else:
			E.PrintSleep(f"ERROR: failed to load image model ({e})")
 
	# Success: im_model set and ActiveReserveMb is min_res. Failure: restore min+im hold.
	E._vram_hold_mb = None
	E.CleanVRAM()
	E.ReserveVRAM()
 
	return result


def GenImage(image_prompt, neg_prompt="NONE", infer_steps=50, guidance=7.5, img_seed="NONE", img_width="auto", img_height="auto", igen_mode=False):
	_EnsureImgMl()

	if image_prompt == '': return "ERROR: Invalid prompt text."
 
	E.CleanVRAM()

	while True:
		if E.im_model == None or (igen_mode == False and E.img_gen_mode == True):
			E.img_gen_mode = False
			E._im_load_error = ""
			if not LoadIMGModel(False):
				detail = str(getattr(E, "_im_load_error", "") or "").strip()
				result = "ERROR: There was an error loading the image generation model."
				if detail:
					result = f"{result} ({detail})"
				break
   
		try:
			if E.orig_shdlr != None:
				E.im_model.unload_ip_adapter()
				E.im_model.scheduler = E.orig_shdlr
				E.orig_shdlr = None
   
			clean_prompt = image_prompt.strip(' ').replace("\n", ' ').replace("\r", '')
			if neg_prompt == "NONE": neg_prompt = E.default_neg_prompt
   
			gen_kwargs = dict(
			 prompt=clean_prompt, negative_prompt=neg_prompt,
			 num_inference_steps=infer_steps, guidance_scale=guidance
			)
   
			if img_width != "auto" and img_height != "auto":
				gen_kwargs["width"] = int(img_width)
				gen_kwargs["height"] = int(img_height)
   
			if img_seed != "NONE":
				gen_kwargs["generator"] = E.seed_gen.manual_seed(int(img_seed))

			with torch.inference_mode(): output = E.im_model(**gen_kwargs)
  
		except:
			result = "ERROR: There was an error generating the image with your settings."
			break

		try:
			result = E.NewFileName(E.work_dir+"/ai_images/", "image_#NUM.png")
			result = result["file"]
			output.images[0].save(result)
		except:
			result = "ERROR: Failed to save the image. Run the app as an admin."
  
		break

	E.CleanVRAM()
	E.ReserveVRAM()
 
	return result


def GenImageIPA(image_prompt, guide_image=None, is_person=False):
	_EnsureImgMl()
 
	if image_prompt == '': return "ERROR: prompt is empty"
 
	if is_person:
		neg_prompt = E.default_neg_prompt + ', ' + E.person_neg_prompt
	else:
		neg_prompt = E.default_neg_prompt
 
	if guide_image == None:
		return GenImage(image_prompt, neg_prompt)

	E.CleanVRAM()
 
	if E.im_model == None and not LoadIMGModel(False):
		detail = str(getattr(E, "_im_load_error", "") or "").strip()
		msg = "ERROR: failed to load image model"
		return f"{msg} ({detail})" if detail else msg
 
	if E.orig_shdlr == None:
		try:
			if is_person:
				ipa_model_id = "h94/IP-Adapter"
				ipa_subfolder = "models"
				ipa_ckpt = "ip-adapter-full-face_sd15.safetensors"
				if E.imodel_type == 1:
				 #ipa_subfolder = "sdxl_models"
				 #ipa_ckpt = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
					return "ERROR: SDXL Face IP-Adapter not yet supported"
				elif E.imodel_type > 1:
					return "ERROR: Face IP-Adapter not yet supported"
			else:
				ipa_model_id = "h94/IP-Adapter"
				ipa_subfolder = "models"
				ipa_ckpt = "ip-adapter_sd15.safetensors"
				if E.imodel_type == 1:
					ipa_subfolder = "sdxl_models"
					ipa_ckpt = "ip-adapter_sdxl.safetensors"
				elif E.imodel_type > 1:
					return "ERROR: IP-Adapter not yet supported"
   
			E.orig_shdlr = E.im_model.scheduler
			E.im_model.scheduler = DDIMScheduler.from_config(E.im_model.scheduler.config)
			E.im_model.load_ip_adapter(ipa_model_id, subfolder=ipa_subfolder, weight_name=ipa_ckpt)
			E.im_model.set_ip_adapter_scale(0.7)
			if E.im_cpu_offload: E.im_model.enable_model_cpu_offload()
		except:
			return "ERROR: failed to load IP-Adapter"

	try:
		clean_prompt = image_prompt.strip(' ').replace("\n", ' ').replace("\r", '')
		with torch.inference_mode():
			ip_image = load_image(guide_image)
			output = E.im_model(prompt=clean_prompt, negative_prompt=neg_prompt, num_inference_steps=50, ip_adapter_image=ip_image)
	except:
		return "ERROR: failed to generate image"

	try:
		result = E.NewFileName(E.work_dir+"/ai_images/", "image_#NUM.png")
		output.images[0].save(result["file"])
	except:
		return "ERROR: failed to save image"

	E.CleanVRAM()
	E.ReserveVRAM()
 
	return result["file"]

