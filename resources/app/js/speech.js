import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import {
	EncodeHTML, CleanFilename, RandInt, ShowInFolder, PathToFileURL,
	DisableButtons, StartGenerating, StopGenerating
} from './util.js';

export function ToggleTTSBox() {
	if (state.tts_exp) {
		$('#tts_exp_box').hide();
		$('#tts_text_inp').prop('disabled', false);
		$('#tts_text_inp').focus();
	} else {
		$('#tts_exp_box').show();
		$('#tts_text_inp').prop('disabled', true);
		$('#tts_area_inp').focus();
	}
	state.tts_exp = !state.tts_exp;
}

export function OpenWavDir() {
	ShowInFolder($('#speech_wav').prop('src'));
}

export function OpenTTSDir() {
	ShowInFolder('TTS_EMBEDS');
}

export function OpenAIADir() {
	ShowInFolder('AI_AUDIO');
}

export function SaveVoice() {
	const voice_name = CleanFilename($('#voice_name').val()).replaceAll(' ', '_');
	if (voice_name == '') {
		ipcRenderer.send('show-alert', { msg: "A valid voice name is required!" });
		return;
	}
	const src_voice = state.script_dir+"/embeddings/ChatTTS/random.tmp";
	const dest_voice = state.script_dir+"/embeddings/ChatTTS/"+voice_name+".txt";
	ipcRenderer.send('save-voice', { dest_file:dest_voice, src_file:src_voice });
}

export function CloneVoice() {
	if (state.thinking || state.generating) return;
	if ($('#clone_voice_btn').prop('disabled')) return;
	const cmodel = $('#vclone_mode').find(':selected').val();
	const tstext = $('#transcript').val();
	const vsample = $('#voice_sample').val();
	const vname = CleanFilename($('#voice_name').val()).replaceAll(' ', '_');
	if (vsample == '' || vname == '') {
		ipcRenderer.send('show-alert', { msg: "A valid voice sample and voice name is required!" });
		return;
	}
	DisableButtons(true);
	$('#tts_result').html('Cloning voice ...');
	ipcRenderer.send('clone-voice', { model:cmodel, sample:vsample, name:vname, transcript:tstext });
	StartGenerating('#ttsgen_box .generating');
}

export function GenSpeech() {
	if (state.thinking || state.generating) return;
	if ($('#tts_btn').prop('disabled')) return;
	let speak_txt = '';
	if (state.tts_exp) {
		speak_txt = $('#tts_area_inp').val().replaceAll("\n", "[AI_UI_BR]");
	} else {
		speak_txt = $('#tts_text_inp').val();
	}
	if (speak_txt == '') return;
	DisableButtons(true);
	$('#tts_result').html('Generating audio ...');
	state.last_gen_mode = $('#ttsgen_mode').find(':selected').val();
	state.last_gen_voice = $('#tts_voices').find(':selected').val();
	state.tts_voice = state.last_gen_voice;
	const spvol = $('#tts_speech_vol').val();
	const srate = $('#tts_speech_rate').val();
	const spitch = $('#tts_speech_pitch').val();
	ipcRenderer.send('gen-speech', { tts_txt:speak_txt, voice:state.last_gen_voice, vol:spvol, rate:srate, pitch:spitch, engine:state.last_gen_mode });
	StartGenerating('#ttsgen_box .generating');
}

export function InitSpeech() {
	ipcRenderer.on('clone-result', (event, payload) => {
		if (!payload.voice.startsWith('ERROR:')) {
			const voice_info = payload.voice.split(':');
			if (voice_info.length == 2) {
				if ($('#ttsgen_mode').find(':selected').val() > 0) {
					if ($("#tts_voices option[value='"+voice_info[1]+"']").length > 0)
						$('#tts_voices').val(voice_info[1]);
				}
				$('#tts_result').html(EncodeHTML(voice_info[0].replaceAll('_', ' '))+' has been added to the list of voices.');
			} else {
				$('#tts_result').html('An unexpected error occurred.');
			}
		} else {
			$('#tts_result').html('The voice could not be cloned. '+payload.voice.replace('ERROR:', ''));
		}
		DisableButtons(false);
		StopGenerating('#ttsgen_box .generating');
	});

	ipcRenderer.on('tts-result', (event, payload) => {
		if (payload.wav == 'ERROR') {
			$('#tts_result').html('There was an error generating the speech.');
		} else if (payload.wav == 'TOO_LONG') {
			$('#tts_result').html('Text is too long for speech generation. Shorten the input or raise audio_max_chars in Model Setup.');
		} else if (payload.wav.startsWith('ERROR:')) {
			$('#tts_result').html(payload.wav);
		} else if (payload.wav == 'NO_VOICES') {
			$('#tts_result').html('There are no voices available for the selected speech engine.');
		} else {
			let wav_html = "<div class='center'><audio controls><source id='speech_wav' src='"+PathToFileURL(payload.wav)+"?rnd="+RandInt()+"' type='audio/wav'></audio>"+
				"<br><br><button id='wav_dir_btn' class='btn cfg_btn' onclick='OpenWavDir()'>OPEN WAV FILE</button>";
			if (state.smodel_type == 1 && state.last_gen_mode == 1 && state.last_gen_voice == 0) {
				wav_html += "&nbsp;<button id='save_voice_btn' class='btn cfg_btn' onclick='SaveVoice()'>SAVE VOICE</button> "+
					"<p><small>To save a voice you must enter a Voice Name in the Voice Cloning settings.</small></p></div>";
			} else {
				wav_html += "</div>";
			}
			$('#tts_result').html(wav_html);
		}
		DisableButtons(false);
		StopGenerating('#ttsgen_box .generating');
	});

	ipcRenderer.on('asr-result', (event, payload) => {
		let targ_input = '#text_inp';
		$('#loading_box').hide();
		if (payload.txt == 'ERROR') {
			ipcRenderer.send('show-alert', { type: "error", msg: "There was an error converting the speech to text." });
			return;
		} else if (!$('#txtgen_app').is(':hidden')) {
			targ_input = '#gen_text_inp';
		} else if (!$('#ttsgen_app').is(':hidden')) {
			targ_input = '#tts_text_inp';
		} else if (!$('#imggen_app').is(':hidden')) {
			targ_input = '#img_text_inp';
		} else if (!$('#console_tab').is(':hidden')) {
			targ_input = '#cmd_text_inp';
		} else {
			if (state.chat_exp) targ_input = '#area_inp';
			if (state.enable_asasro == 1) state.asr_pending = true;
		}
		let inp_txt = $(targ_input).val().trim();
		if (inp_txt != '') inp_txt += ' ';
		$(targ_input).val(inp_txt+payload.txt);
	});
}
