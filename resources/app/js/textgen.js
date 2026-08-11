import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import { EncodeHTML, DisableButtons, StartGenerating, StopGenerating } from './util.js';
import { ToggleAudio } from './chat.js';

export function ToggleGenBox() {
	if (state.gen_exp) {
		$('#gen_exp_box').hide();
		$('#gen_text_inp').prop('disabled', false);
		$('#gen_text_inp').focus();
	} else {
		$('#gen_exp_box').show();
		$('#gen_text_inp').prop('disabled', true);
		$('#gen_area_inp').focus();
	}
	state.gen_exp = !state.gen_exp;
}

export function CopyGenTxt() {
	const gen_txt = $('#gen_result').html().replaceAll("<br>", "\n");
	ipcRenderer.send('copy-text', { txt: gen_txt });
}

export function ReadGenTxt() {
	if ($('#read_txt_btn').html() == 'TEXT TO SPEECH') {
		if (state.thinking || state.generating) {
			ipcRenderer.send('show-alert', { msg: 'The engine is busy with another task.' });
			return;
		}
		const gen_txt = $('#gen_result').html().replaceAll("<br>", "[AI_UI_BR]");
		if (gen_txt != '') $('#read_txt_btn').html('STOP READING');
		ipcRenderer.send('read-text', { txt: gen_txt });
	} else if ($('#read_txt_btn').html() == 'STOP READING') {
		$('#read_txt_btn').html('CONTINUE READING');
		ToggleAudio();
	} else {
		$('#read_txt_btn').html('STOP READING');
		ToggleAudio();
	}
}

export function GenText() {
	if ($('#gen_btn').html() == 'STOP') {
		$('#gen_btn').prop('disabled', true);
		ipcRenderer.invoke('stop-stream');
		return;
	}
	if (state.thinking || state.generating) return;
	if ($('#gen_btn').prop('disabled')) return;
	let start_txt = '';
	if (state.gen_exp) {
		start_txt = $('#gen_area_inp').val().replaceAll("\n", "[AI_UI_BR]");
	} else {
		start_txt = $('#gen_text_inp').val();
	}
	if (start_txt == '') return;
	DisableButtons(true, state.gen_mode != 1);
	$('#gen_result').html('Generating text ...');
	const max_genl = $('#gen_max_len').val();
	const min_genl = $('#gen_min_len').val();
	const gen_numb = $('#gen_num_beams').val();
	const gen_numg = $('#gen_beam_groups').val();
	const gen_temp = $('#gen_temp').val();
	const gen_topk = $('#gen_top_k').val();
	const gen_topp = $('#gen_top_p').val();
	const gen_typp = $('#gen_typ_p').val();
	const gen_repp = $('#gen_rep_p').val();
	const gen_dosp = $('#gds_select').find(':selected').val();
	ipcRenderer.send('gen-text', {
		txt: start_txt, max: max_genl, min: min_genl,
		do_sp: gen_dosp, num_b: gen_numb, num_g: gen_numg, temp: gen_temp,
		top_k: gen_topk, top_p: gen_topp, typ_p: gen_typp, rep_p: gen_repp
	});
	StartGenerating('#txtgen_box .generating');
}

export function InitTextgen() {
	ipcRenderer.on('txtgen-stream-start', (event, payload) => {
		state.streaming_tgrb = $('#gen_result');
		state.streaming_tgrb.html('');
	});

	ipcRenderer.on('txtgen-stream-text', (event, payload) => {
		const txt = EncodeHTML(payload.txt).replaceAll("[AI_UI_BR]", "<br>");
		if (state.streaming_tgrb.html() == '') {
			if (payload.txt.trim() != '')
				state.streaming_tgrb.html(txt+' ');
		} else {
			state.streaming_tgrb.html(state.streaming_tgrb.html()+txt+' ');
		}
	});

	ipcRenderer.on('gen-result', (event, payload) => {
		DisableButtons(false);
		// Empty payload after STOP - keep streamed tokens already in the box
		if (payload.txt.trim() != '') {
			const txt_html = EncodeHTML(payload.txt).replaceAll("[AI_UI_BR]", "<br>");
			$('#gen_result').html(txt_html);
		}
		$('#read_txt_btn').html('TEXT TO SPEECH');
		StopGenerating('#txtgen_box .generating');
	});
}
