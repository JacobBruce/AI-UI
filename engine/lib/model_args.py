# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

from lib.local_ml import ResolveTorchDtype, TorchAvailable


def ApplyModelArgs(model_args):
	"""Parse comma-separated model_args flags into engine attributes."""
	for arg in model_args:
		carg = arg.strip(" ").replace("\\", "/")
		if carg.startswith("tokenizer_model="):
			E.tokenizer_model = carg.replace("tokenizer_model=", "", 1)
		elif carg.startswith("model_adapter="):
			E.model_adapter = carg.replace("model_adapter=", "", 1)
		elif carg.startswith("im_lora_file="):
			E.im_lora_file = carg.replace("im_lora_file=", "", 1)
		elif carg.startswith("im_lora_dir="):
			E.im_lora_dir = carg.replace("im_lora_dir=", "", 1)
		elif carg.startswith("im_lora_scale="):
			E.im_lora_scale = float(carg.replace("im_lora_scale=", "", 1))
		elif carg.startswith("im_vae_file="):
			E.im_vae_file = carg.replace("im_vae_file=", "", 1)
		elif carg.startswith("im_unet_file="):
			E.im_unet_file = carg.replace("im_unet_file=", "", 1)
		elif carg.startswith("im_tran_file="):
			E.im_tran_file = carg.replace("im_tran_file=", "", 1)
		elif carg.startswith("im_enc_file="):
			E.im_enc_file = carg.replace("im_enc_file=", "", 1)
		elif carg.startswith("im_enc2_file="):
			E.im_enc2_file = carg.replace("im_enc2_file=", "", 1)
		elif carg.startswith("im_enc3_file="):
			E.im_enc3_file = carg.replace("im_enc3_file=", "", 1)
		elif carg.startswith("im_variant="):
			E.im_variant = carg.replace("im_variant=", "", 1)
		elif carg.startswith("im_config_file="):
			E.im_config_file = carg.replace("im_config_file=", "", 1)
		elif carg.startswith("im_model_class="):
			E.im_model_class = carg.replace("im_model_class=", "", 1)
		elif carg.startswith("text_lang_class="):
			E.text_lang_class = carg.replace("text_lang_class=", "", 1)
		elif carg.startswith("multi_modal_class="):
			E.multi_modal_class = carg.replace("multi_modal_class=", "", 1)
		elif carg.startswith("speech_recog_class="):
			E.speech_recog_class = carg.replace("speech_recog_class=", "", 1)
		elif carg.startswith("vocoder_model="):
			E.vocoder_model = carg.replace("vocoder_model=", "", 1)
		elif carg.startswith("attn_implementation="):
			E.attn_implementation = carg.replace("attn_implementation=", "", 1)
		elif carg.startswith("roles_append_str="):
			E.roles_append_str = carg.replace("roles_append_str=", "", 1).replace("\\n", "\n")
		elif carg.startswith("torch_dtype="):
			name = carg.replace("torch_dtype=", "", 1).lower()
			E.torch_dtype = ResolveTorchDtype(name, default="auto") if TorchAvailable() else name
		elif carg.startswith("im_torch_dtype="):
			name = carg.replace("im_torch_dtype=", "", 1).lower()
			E.im_torch_dtype = ResolveTorchDtype(name, default="float16") if TorchAvailable() else name
		elif carg.startswith("use_safetensors="):
			use_sts = carg.replace("use_safetensors=", "", 1)
			E.use_safetensors = (use_sts.lower() == "true" or use_sts == "1")
		elif carg.startswith("gguf_file="):
			E.gguf_file = carg.replace("gguf_file=", "", 1).replace("\\n", "\n")
		elif carg.startswith("im_use_safetensors="):
			use_sts = carg.replace("im_use_safetensors=", "", 1)
			E.im_use_safetensors = (use_sts.lower() == "true" or use_sts == "1")
		elif carg.startswith("load_in_8bit="):
			load_8bit = carg.replace("load_in_8bit=", "", 1)
			if load_8bit.lower() == "true" or load_8bit == "1":
				E.load_in_8bit = True
		elif carg.startswith("im_load_in_8bit="):
			load_8bit = carg.replace("im_load_in_8bit=", "", 1)
			if load_8bit.lower() == "true" or load_8bit == "1":
				E.im_load_in_8bit = True
		elif carg.startswith("custom_model_code="):
			custom_code = carg.replace("custom_model_code=", "", 1)
			if custom_code.lower() == "true" or custom_code == "1":
				E.custom_model_code = True
		elif carg.startswith("custom_token_code="):
			custom_code = carg.replace("custom_token_code=", "", 1)
			if custom_code.lower() == "true" or custom_code == "1":
				E.custom_token_code = True
		elif carg.startswith("im_from_single_file="):
			from_single_file = carg.replace("im_from_single_file=", "", 1)
			if from_single_file.lower() == "true" or from_single_file == "1":
				E.im_from_single_file = True
		elif carg.startswith("im_local_files="):
			from_local_files = carg.replace("im_local_files=", "", 1)
			if from_local_files.lower() == "true" or from_local_files == "1":
				E.im_local_files = True
		elif carg.startswith("im_cpu_offload="):
			cpu_offload = carg.replace("im_cpu_offload=", "", 1)
			if cpu_offload.lower() == "false" or cpu_offload == "0":
				E.im_cpu_offload = False
		elif carg.startswith("im_att_slicing="):
			att_slicing = carg.replace("im_att_slicing=", "", 1)
			if att_slicing.lower() == "false" or att_slicing == "0":
				E.im_att_slicing = False
			else:
				E.im_att_slicing = True
		elif carg.startswith("im_safety_check="):
			safety_check = carg.replace("im_safety_check=", "", 1)
			if safety_check.lower() == "false" or safety_check == "0":
				E.im_safety_check = False
		elif carg.startswith("use_chat_template="):
			use_chat_temp = carg.replace("use_chat_template=", "", 1)
			if use_chat_temp.lower() == "true" or use_chat_temp == "1":
				E.use_chat_template = True
			E.use_tool_config = False
		elif carg.startswith("use_aiui_template="):
			use_aiui = carg.replace("use_aiui_template=", "", 1)
			if use_aiui.lower() == "true" or use_aiui == "1":
				E.use_aiui_template = True
				E.use_chat_template = True
				E.temp_form = "aiui"
			E.use_tool_config = False
		elif carg.startswith("skip_special_tokens="):
			skip_spec_tkns = carg.replace("skip_special_tokens=", "", 1)
			if skip_spec_tkns.lower() == "false" or skip_spec_tkns == "0":
				E.skip_special_tokens = False
		elif carg.startswith("cleanup_token_space="):
			clean_spaces = carg.replace("cleanup_token_space=", "", 1)
			if clean_spaces.lower() == "false" or clean_spaces == "0":
				E.cleanup_token_space = False
		elif carg.startswith("torch_compile="):
			compile_torch = carg.replace("torch_compile=", "", 1)
			if compile_torch.lower() == "true" or compile_torch == "1":
				E.torch_compile = True
		elif carg.startswith("audio_max_new_tokens="):
			try:
				E.audio_max_new_tokens = max(1, int(carg.replace("audio_max_new_tokens=", "", 1)))
			except ValueError:
				pass
		elif carg.startswith("audio_max_chars="):
			try:
				E.audio_max_chars = max(0, int(carg.replace("audio_max_chars=", "", 1)))
			except ValueError:
				pass
		elif carg.startswith("anim_max_secs="):
			try:
				E.anim_max_secs = max(0.0, float(carg.replace("anim_max_secs=", "", 1)))
			except ValueError:
				pass
		elif carg.startswith("chunk_audio_secs="):
			try:
				E.chunk_audio_secs = int(carg.replace("chunk_audio_secs=", "", 1))
			except ValueError:
				pass
		elif carg.startswith("im_res_vram_mb="):
			try:
				E.im_res_vram_mb = int(carg.replace("im_res_vram_mb=", "", 1))
			except ValueError:
				pass
		elif carg.startswith("min_res_vram_mb="):
			try:
				E.min_res_vram_mb = int(carg.replace("min_res_vram_mb=", "", 1))
			except ValueError:
				pass

def LoadModels():
	from lib import text_backend
	from lib import ipc
	skip_text = text_backend.IsRemoteText()
	if E.start_meth == "text":
		if skip_text:
			ipc.LogErr("Remote text backend active - skipping local text model load")
		elif not str(getattr(E, "model_id", "") or "").strip():
			ipc.LogErr(
				"WARN: Text Backend is Local but Model Directory is empty - "
				"set a local model, or switch Text Backend to Remote (lite installs)"
			)
		else:
			E.LoadModel()
	elif E.start_meth == "image":
		E.LoadIMGModel()
	elif E.start_meth == "speech":
		E.LoadTTSEngine()
	elif E.start_meth == "sr":
		E.LoadSRModel()
	elif E.start_meth == "all":
		E.LoadTTSEngine()
		E.LoadSRModel()
		if skip_text:
			ipc.LogErr("Remote text backend active - skipping local text model load")
		elif not str(getattr(E, "model_id", "") or "").strip():
			ipc.LogErr(
				"WARN: Text Backend is Local but Model Directory is empty - "
				"set a local model, or switch Text Backend to Remote (lite installs)"
			)
		else:
			E.LoadModel()
		E.LoadIMGModel()
	elif E.start_meth == "none":
		if E.model_id and not skip_text:
			E.LoadTokenizer()
	# Hold VRAM for image / gen headroom even when the text model loads later.
	E.ReserveVRAM()
