import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import { StripFileProto, PathToFileURL, ShowInFolder, DisableButtons, StartGenerating, StopGenerating } from './util.js';

export function CopyGenImg() {
	let gen_img = document.getElementById('gen_img');
	if (gen_img === null) {
		ipcRenderer.send('show-alert', { msg: "There is no image to copy!" });
	} else {
		gen_img = StripFileProto($(gen_img).prop('src'));
		ipcRenderer.send('copy-image', { img: gen_img });
	}
}

export function OpenImgDir() {
	const gen_img = document.getElementById('gen_img');
	ShowInFolder(gen_img);
}

export function GenImage() {
	if (state.thinking || state.generating) return;
	if ($('#img_btn').prop('disabled')) return;
	let prompt_txt = $('#img_text_inp').val();
	let prompt_neg = $('#img_text_neg').val();
	if (prompt_txt == '') return;
	if (prompt_neg == '') prompt_neg = 'NONE';
	DisableButtons(true);
	$('#img_result').html('Generating image ...');
	const infer_steps = $('#infer_steps').val();
	const guide_scale = $('#guidance').val();
	const safety_check = $('#safety_check').find(':selected').val();
	const vae_file = $('#vae_file').val();
	const lora_file = $('#lora_file').val();
	const lora_dir = $('#lora_dir').val();
	const lora_scale = $('#lora_scale').val();
	let img_seed = $('#img_seed').val().trim();
	let img_width = 'auto';
	let img_height = 'auto';
	if ($('#is_select').find(':selected').val() == 'custom') {
		img_width = $('#img_width').val();
		img_height = $('#img_height').val();
	}
	if (img_seed == '') {
		img_seed = 'NONE';
	} else if (Number.isInteger(parseInt(img_seed, 10))) {
		img_seed = parseInt(img_seed, 10).toString();
	} else {
		img_seed = parseInt(img_seed.toLowerCase().replace(/[^a-z0-9]/g, ''), 36);
		img_seed = Number.isInteger(img_seed) ? img_seed.toString() : '0';
	}
	ipcRenderer.send('gen-image', {
		img_prompt: prompt_txt, neg_prompt: prompt_neg, steps: infer_steps, guidance: guide_scale,
		seed: img_seed, width: img_width, height: img_height, check: safety_check,
		vae_file: vae_file, lora_file: lora_file, lora_dir: lora_dir, lora_scale: lora_scale
	});
	StartGenerating('#imggen_box .generating');
}

export function InitImage() {
	ipcRenderer.on('img-result', (event, payload) => {
		DisableButtons(false);
		if (payload.img.startsWith("ERROR:")) {
			$('#img_result').html(payload.img);
		} else {
			$('#img_result').html("<img id='gen_img' src='"+PathToFileURL(payload.img)+"'>");
		}
		StopGenerating('#imggen_box .generating');
	});
}
