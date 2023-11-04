window.$ = window.jQuery = require('jquery');
const { ipcRenderer } = require('electron');

const pedit_img_html = '<button id="edit_btn" onclick="ToggleEditPrompt()"><img class="edit_img" src="./img/edit_w.png" width="15" /></button>';
const default_ip = "Chat log between HUMAN_NAME and BOT_NAME on DATE";
const pygmalion_ip = "BOT_NAME's Persona: A helpful AI assistant who can use the [AI_IMG] tag to generate images from a description.\n<START>\nHUMAN_NAME: show me an image of a cyberpunk cityscape\nBOT_NAME: [AI_IMG]cyberpunk cityscape[/AI_IMG]";
const bbcode_ip = "Use the [CODE] tag to post a code snippet. Example: [CODE]int abc = 123;[/CODE]\nUse the [AI_IMG] tag to generate an image from a description using AI. Example: [AI_IMG]cute kitten[/AI_IMG]";
var ip_vals = { chat:'', pygmalion:'', bbcode:'' };

var avatar_vid = null;
var speech_audio = null;
var loop_timer = null;
var dot_count = 0;
var rnd_int = 0;
var active_dl = 0;
var total_dls = 0;
var cmd_index = 0;
var btn_state = true;
var thinking = false;
var generating = false;
var chat_exp = false;
var gen_exp = false;
var tts_exp = false;
var adv_show = false;
var sys_voices = [];
var t5_voices = [];
var cmd_list = [];
var think_targ = '';
var prompt = '';

var human_name = 'Human';
var bot_name = 'Bot';
var bot_voice = 0;
var speech_vol = 1.0;
var speech_rate = 200;
var pitch_shift = 0;
var talk_mode = 0;
var anim_mode = 0;
var tts_mode = 0;
var tts_voice = 0;

var max_mm = 5;
var max_rl = 50;
var min_rl = 1;;
var b_temp = 0.8;
var prompt_p = 0;
var top_k = 50;
var top_p = 1.0;
var typical_p = 1.0;
var rep_penalty = 1.0;

var model_type = 0;
var imodel_type = 0;
var smodel_type = 0;
var comp_dev = 'auto';
var start_meth = 'text';
var script_dir = '';
var model_args = ''
var model_dir = '';
var sd_model = '';
var sd_model = '';
var tts_model = '';
var voc_model = '';
var python_bin = '';
var avatar_mp4 = '';
var avatar_img = '';

function RandInt(max_val=9999999) {
	let rnd = Math.floor(Math.random() * max_val);
	while (rnd_int == rnd)
		rnd = Math.floor(Math.random() * max_val);
	rnd_int = rnd;
	return rnd;
}

function PlayVideo() {
	if ($('#other_vid').is(":hidden")) {
		$('#other_source').prop('src', avatar_mp4+'?rnd='+RandInt());
		avatar_vid = document.getElementById('other_vid');
	} else {
		$('#avatar_source').prop('src', avatar_mp4+'?rnd='+RandInt());
		avatar_vid = document.getElementById('avatar_vid');
	}
	avatar_vid.oncanplaythrough = function() {
		if ($('#other_vid').is(":hidden")) {
			$('#avatar_vid').hide();
		} else {
			$('#other_vid').hide();
		}
		$(avatar_vid).show();
		avatar_vid.play();
	};
	avatar_vid.load();
}

function PlayAudio(audio_file) {
	if (speech_audio !== null && !speech_audio.paused) speech_audio.pause();
	speech_audio = new Audio('file://'+audio_file+'?rnd='+RandInt());
	speech_audio.onended = function() { $('#read_txt_btn').html('TEXT TO SPEECH'); };
	speech_audio.play();
}

function ToggleAudio() {
	if (speech_audio === null) return;
	if (speech_audio.paused) {
		speech_audio.play();
	} else {
		speech_audio.pause();
	}
}

function EncodeHTML(txt) {
	return txt.replaceAll('&', '&amp;').
		replaceAll('<', '&lt;').
		replaceAll('>', '&gt;').
		replaceAll('"', '&quot;').
		replaceAll('\'', '&apos;').
		replaceAll('/', '&#47;').
		replaceAll('\\', '&#92;').
		replaceAll('\t', '&nbsp;&nbsp;');
}

function ConvertAIBB(txt) {
	return txt.replaceAll("[AI_UI_BR]", "<br>").replaceAll("[AI_UI_TAB]", "\t").replaceAll("[AI_UI_LF]", "\n").
		replaceAll("[CODE START]", '</p><pre class="code_box">').replaceAll("[CODE END]", '</pre><p class="msg_p">').
		replaceAll("[AI_IMG NUM_", '</p><img class="chat_img" src="file://'+script_dir+'/ai_images/image_').
		replaceAll("_CHAT_IMG]", '.png" title="').replaceAll("[AI_IMG END]", '" onclick="ShowInFolder(this)"><p class="msg_p">');
}

function ConvertBB(txt) {
	return txt.replaceAll("[AI_UI_TAB]", "\t").replaceAll("[AI_UI_LF]", "\n").
		replaceAll('[CODE START]', '</p><pre class="code_box">').replaceAll('[CODE END]', '</pre><p class="msg_p">');
}

function EncodeBB(bbcode) {
	let result = bbcode;
	while (result.includes('[CODE]') && result.includes('[/CODE]')) {
		const tag_start = result.indexOf("[CODE]") + 6;
		const tag_end = result.indexOf("[/CODE]");
		if (tag_start >= tag_end) break;
		const code_txt = result.substring(tag_start, tag_end);
		const code_enc = code_txt.replaceAll("\t", "[AI_UI_TAB]").replaceAll("\n", "[AI_UI_LF]");
		result = result.replace('[CODE]'+code_txt+'[/CODE]', '[CODE START]'+code_enc+'[CODE END]');
	}
	return result;
}

function TrimLast(str, chr) {
	if (str.endsWith(chr)) {
		return str.slice(0, -1);
	} else {
		return str;
	}
}

function TrimStart(str, chr) {
	if (str.startsWith(chr)) {
		return str.substr(1);
	} else {
		return str;
	}
}

function TrimEnds(str, chr) {
	return TrimLast(TrimStart(str, chr), chr);
}

function CleanFilename(filename) {
	return TrimEnds(TrimEnds(TrimEnds(filename.replaceAll('\\', '').replaceAll('/', '').replaceAll('?', '').replaceAll('#', '').
		replaceAll('%', '').replaceAll('&', '').replaceAll('{', '').replaceAll('}', '').replaceAll('<', '').replaceAll('>', '').
		replaceAll('$', '').replaceAll('!', '').replaceAll('@', '').replaceAll(':', '').replaceAll('*', '').replaceAll('+', '').
		replaceAll('=', '').replaceAll('|', '').replaceAll('`', '').replaceAll('"', '').replaceAll("'", '').trim(), '.'),'-'),'_');
}

// ---- PAGES ----

function HidePages() {
	$('#chat_app').hide();
	$('#txtgen_app').hide();
	$('#imggen_app').hide();
	$('#ttsgen_app').hide();
	$('#console_tab').hide();
	$('#config_tab').hide();
	$('#help_tab').hide();
	$('#about_tab').hide();
}

function SelectPage(btn) {
	$('.menu_btn').removeClass('menu_sel');
	btn.classList.add('menu_sel');
	HidePages();
}

function ChatPage(btn) {
	SelectPage(btn);
	$('#chat_app').show();
}

function TextPage(btn) {
	SelectPage(btn);
	$('#txtgen_app').show();
}

function ImagePage(btn) {
	SelectPage(btn);
	$('#imggen_app').show();
}

function SpeechPage(btn) {
	SelectPage(btn);
	$('#ttsgen_app').show();
}

function ConfigPage(btn) {
	SelectPage(btn);
	$('#config_tab').show();
}

function ConsolePage(btn) {
	SelectPage(btn);
	$('#console_tab').show();
}

function HelpPage(btn) {
	SelectPage(btn);
	$('#help_tab').show();
}

function AboutPage(btn) {
	SelectPage(btn);
	$('#about_tab').show();
}

// ---- CHAT ----

function ThinkAnim() {
	const targ = (thinking) ? '#thinking' : think_targ;
	switch (dot_count) {
	case 0: 
		$(targ).html('');
		dot_count++;
		break;
	case 1:
		$(targ).html('.');
		dot_count++;
		break;
	case 2:
		$(targ).html('..');
		dot_count++;
		break;
	case 3:
		$(targ).html('...');
		dot_count++;
		break;
	case 4:
		$(targ).html('..');
		dot_count++;
		break;
	case 5:
		$(targ).html('.');
		dot_count = 0;
		break;
	}
}

function StartThinking() {
	thinking = true;
	let scroll_box = $('#chat_log');
	scroll_box.animate({scrollTop: scroll_box.prop("scrollHeight")}, 300, 'linear', function() {
		$('#thinking').html('');
		$('#thinking').show();
		dot_count = 1;
		loop_timer = setInterval(ThinkAnim, 1000);
	});
}

function StopThinking() {
	thinking = false;
	let scroll_box = $('#chat_log');
	scroll_box.animate({scrollTop: scroll_box.prop("scrollHeight")});
	$('#thinking').hide();
	clearInterval(loop_timer);
}

function DisExtraButtons(bool_val) {
	$('#redo_btn').prop('disabled', bool_val);
	$('#cont_btn').prop('disabled', bool_val);
	$('#clear_btn').prop('disabled', bool_val);
}

function DisableButtons(bool_val, load_state=false) {
	let new_state = bool_val;
	if (load_state) {
		new_state = bool_val ? true : btn_state;
	} else {
		btn_state = bool_val;
	}
	DisExtraButtons(new_state);
	$('#apply_btn').prop('disabled', bool_val);
	$('#gen_btn').prop('disabled', bool_val);
	$('#tts_btn').prop('disabled', bool_val);
	$('#clone_voice_btn').prop('disabled', bool_val);
	$('#img_btn').prop('disabled', bool_val);
	$('#edit_btn').prop('disabled', bool_val);
	$('.config_btn').prop('disabled', bool_val);
}

ipcRenderer.on('bot-msg', (event, payload) => {
	const bot_msg = ConvertAIBB(EncodeHTML(payload.msg.trim()));
	const msg_html = '<div class="bmsg_box"><div class="bot_msg"><p class="msg_p">'+bot_msg+'</p></div></div>';
	$('#chat_log').append(msg_html.replaceAll('<p class="msg_p"></p>', ''));
	$('.msg_pad').remove();
	DisableButtons(false);
	StopThinking();
	if (payload.got_vid && talk_mode == 0) {
		$('#avatar_img').hide();
		PlayVideo();
	}
});

function SendMsg() {
	if (thinking || generating) return;//TODO: remove this for multi msg from user
	if ($('#send_btn').prop('disabled')) return;
	let message = '';
	let html_msg = '';
	if (chat_exp) {
		message = $('#area_inp').val().trim();
		html_msg = ConvertBB(EncodeHTML(EncodeBB(message)).replaceAll("\n", "<br>"));
		message = message.replaceAll("\n", "[AI_UI_BR]");
		$('#area_inp').val('');
	} else {
		message = $('#text_inp').val().trim();
		html_msg = ConvertBB(EncodeHTML(EncodeBB(message)));
		$('#text_inp').val('');
	}
	if (message == '') return;
	html_msg = '<div class="umsg_box"><div class="chat_msg"><p class="msg_p">'+html_msg+'</p></div></div>';
	$('#chat_log').append(html_msg.replaceAll('<p class="msg_p"></p>', ''));
	$('#edit_dialog').hide();
	ipcRenderer.send('send-msg', { msg: message });
	DisableButtons(true);
	StartThinking();
}

function RedoLast() {
	$('.bmsg_box').last().remove();
	if ($('#chat_log').children().last().hasClass('bmsg_box'))
		$('#chat_log').append('<div class="msg_pad"></div>');
	DisableButtons(true);
	ipcRenderer.send('send-msg', { msg: 'redo_last' });
	StartThinking();
}

function ContChat() {
	DisableButtons(true);
	ipcRenderer.send('send-msg', { msg: 'cont_chat' });
	$('#chat_log').append('<div class="msg_pad"></div>');
	StartThinking();
}

function ClearChat() {
	btn_state = true;
	DisExtraButtons(true);
	$('#chat_log').html('');
	ipcRenderer.send('send-msg', { msg: 'clear_chat' });
}

function ToggleMsgBox() {
	if (chat_exp) {
		$('#area_inp').hide();
		$('#text_inp').prop('disabled', false);
		$('#text_inp').focus();
	} else {
		$('#area_inp').show();
		$('#text_inp').prop('disabled', true);
		$('#area_inp').focus();
	}
	chat_exp = !chat_exp;
}

// ---- TEXT GEN ----

function StartGenerating(targ) {
	generating = true;
	think_targ = targ;
	$(think_targ).html('');
	$(think_targ).show();
	dot_count = 1;
	loop_timer = setInterval(ThinkAnim, 1000);
}

function StopGenerating(targ='.generating') {
	generating = false;
	$(targ).hide();
	clearInterval(loop_timer);
}

function ToggleGenBox() {
	if (gen_exp) {
		$('#gen_area_inp').hide();
		$('#gen_text_inp').prop('disabled', false);
		$('#gen_text_inp').focus();
	} else {
		$('#gen_area_inp').show();
		$('#gen_text_inp').prop('disabled', true);
		$('#gen_area_inp').focus();
	}
	gen_exp = !gen_exp;
}

function ToggleTTSBox() {
	if (tts_exp) {
		$('#tts_area_inp').hide();
		$('#tts_text_inp').prop('disabled', false);
		$('#tts_text_inp').focus();
	} else {
		$('#tts_area_inp').show();
		$('#tts_text_inp').prop('disabled', true);
		$('#tts_area_inp').focus();
	}
	tts_exp = !tts_exp;
}

function CopyGenTxt() {
	const gen_txt = $('#gen_result').html().replaceAll("<br>", "\n");
	ipcRenderer.send('copy-text', { txt: gen_txt });
}

function ReadGenTxt() {
	if ($('#read_txt_btn').html() == 'TEXT TO SPEECH') {
		if (thinking || generating) {
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

function GenText() {
	if (thinking || generating) return;
	if ($('#gen_btn').prop('disabled')) return;
	let start_txt = '';
	if (gen_exp) {
		start_txt = $('#gen_area_inp').val().replaceAll("\n", "[AI_UI_BR]");
	} else {
		start_txt = $('#gen_text_inp').val();
	}
	if (start_txt == '') return;
	DisableButtons(true, true);
	$('#gen_result').html('Generating text ...');
	const max_genl = $('#gen_max_len').val();
	const min_genl = $('#gen_min_len').val();
	const gen_temp = $('#gen_temp').val();
	const gen_topk = $('#gen_top_k').val();
	const gen_topp = $('#gen_top_p').val();
	const gen_typp = $('#gen_typ_p').val();
	const gen_repp = $('#gen_rep_p').val();
	ipcRenderer.send('gen-text', {
		txt: start_txt, max: max_genl, min: min_genl, temp: gen_temp,
		top_k: gen_topk, top_p: gen_topp, typ_p: gen_typp, rep_p: gen_repp
	});
	StartGenerating('#txtgen_box .generating');
}

ipcRenderer.on('gen-result', (event, payload) => {
	const txt_html = EncodeHTML(payload.txt).replaceAll("[AI_UI_BR]", "<br>");
	DisableButtons(false, true);
	$('#gen_result').html(txt_html);
	$('#read_txt_btn').html('TEXT TO SPEECH');
	StopGenerating('#txtgen_box .generating');
});

// ---- SPEECH GEN ----

function OpenWavDir() {
	ShowInFolder($('#speech_wav').prop('src'));
}

function OpenTTSDir() {
	ShowInFolder('TTS_EMBEDS');
}

function CloneVoice() {
	if (thinking || generating) return;
	if ($('#clone_voice_btn').prop('disabled')) return;
	const vsample = $('#voice_sample').val();
	const vname = CleanFilename($('#voice_name').val()).replaceAll(' ', '_');
	if (vsample == '' || vname == '') {
		ipcRenderer.send('show-alert', { msg: "A valid voice sample and voice name is required!" });
		return;
	}
	DisableButtons(true, true);
	$('#tts_result').html('Cloning voice ...');
	ipcRenderer.send('clone-voice', { sample:vsample, name:vname });
	StartGenerating('#ttsgen_box .generating');
}

ipcRenderer.on('clone-result', (event, payload) => {
	if (!payload.voice.startsWith('ERROR:')) {
		const voice_info = payload.voice.split(':');
		if (voice_info.length == 2) {
			if ($('#ttsgen_mode').find(':selected').val() == 1) $('#tts_voices').val(voice_info[1]);
			$('#tts_result').html(EncodeHTML(voice_info[0].replaceAll('_', ' '))+' has been added to the list of voices.');
		} else {
			$('#tts_result').html('An unexpected error occurred.');
		}
	} else {
		$('#tts_result').html('The voice could not be cloned. '+payload.voice.replace('ERROR:', ''));
	}
	DisableButtons(false, true);
	StopGenerating('#ttsgen_box .generating');
});

function GenSpeech() {
	if (thinking || generating) return;
	if ($('#tts_btn').prop('disabled')) return;
	let speak_txt = '';
	if (tts_exp) {
		speak_txt = $('#tts_area_inp').val().replaceAll("\n", "[AI_UI_BR]");
	} else {
		speak_txt = $('#tts_text_inp').val();
	}
	if (speak_txt == '') return;
	DisableButtons(true, true);
	$('#tts_result').html('Generating audio ...');
	const genmode = $('#ttsgen_mode').find(':selected').val();
	const svoice = $('#tts_voices').find(':selected').val();
	const spvol = $('#tts_speech_vol').val();
	const srate = $('#tts_speech_rate').val();
	const spitch = $('#tts_speech_pitch').val();
	ipcRenderer.send('gen-speech', { tts_txt:speak_txt, voice:svoice, vol:spvol, rate:srate, pitch:spitch, engine:genmode });
	StartGenerating('#ttsgen_box .generating');
}

ipcRenderer.on('tts-result', (event, payload) => {
	if (payload.wav == 'ERROR') {
		$('#tts_result').html('There was an error generating the speech.');
	} else if (payload.wav == 'NO_VOICES') {
		$('#tts_result').html('There are no voices available for the selected speech engine.');
	} else {
		const wav_html = "<center><audio controls><source id='speech_wav' src='file://"+payload.wav+"?rnd="+RandInt()+"' type='audio/wav'></audio>"+
			"<br><br><button id='wav_dir_btn' class='btn cfg_btn' onclick='OpenWavDir()'>OPEN WAV FILE</button></center>";
		$('#tts_result').html(wav_html);
	}
	DisableButtons(false, true);
	StopGenerating('#ttsgen_box .generating');
});

// ---- IMAGE GEN ----

function CopyGenImg() {
	let gen_img = document.getElementById('gen_img');
	if (gen_img === null) {
		ipcRenderer.send('show-alert', { msg: "There is no image to copy!" });
	} else {
		gen_img = $(gen_img).prop('src').
			replace('file:///', '').replace('file://', '');
		ipcRenderer.send('copy-image', { img: gen_img });
	}
}

function OpenImgDir() {
	const gen_img = document.getElementById('gen_img');
	ShowInFolder(gen_img);
}

function GenImage() {
	if (thinking || generating) return;
	if ($('#img_btn').prop('disabled')) return;
	let prompt_txt = $('#img_text_inp').val();
	let prompt_neg = $('#img_text_neg').val();
	if (prompt_txt == '') return;
	if (prompt_neg == '') prompt_neg = 'NONE';
	DisableButtons(true, true);
	$('#img_result').html('Generating image ...');
	const infer_steps = $('#infer_steps').val();
	const guide_scale = $('#guidance').val();
	const safety_check = $('#safety_check').find(':selected').val();
	const vae_file = $('#vae_file').val();
	const lora_file = $('#lora_file').val();
	const lora_dir = $('#lora_dir').val();
	let img_width = 'auto';
	let img_height = 'auto';
	if ($('#is_select').find(':selected').val() == 'custom') {
		img_width = $('#img_width').val();
		img_height = $('#img_height').val();
	}
	ipcRenderer.send('gen-image', {
		img_prompt: prompt_txt, neg_prompt: prompt_neg, steps: infer_steps, 
		guidance: guide_scale, width: img_width, height: img_height, check: safety_check,
		vae_file: vae_file, lora_file: lora_file, lora_dir: lora_dir
	});
	StartGenerating('#imggen_box .generating');
}

ipcRenderer.on('img-result', (event, payload) => {
	const img_html = "<img id='gen_img' src='file://"+script_dir+"/ai_images/image_"+payload.img+".png'>";
	DisableButtons(false, true);
	if (payload.img >= 0) {
		$('#img_result').html(img_html);
	} else if (payload.img == -1) {
		$('#img_result').html("Invalid prompt text.");
	} else if (payload.img == -2) {
		$('#img_result').html("There was an error loading the image generation model.");
	} else if (payload.img == -3) {
		$('#img_result').html("There was an error generating the image with your settings.");
	} else if (payload.img == -4) {
		$('#img_result').html("There was an error saving the image. Run the app as an admin.");
	} else {
		$('#img_result').html("An unexpected error occurred. Try again.");
	}
	StopGenerating('#imggen_box .generating');
});

// ---- CONSOLE ----

function CmdHistory(shift, val) {
	if (cmd_list.length == 0) return val;
	cmd_index += shift;
	if (cmd_index >= cmd_list.length)
		cmd_index = cmd_list.length - 1;
	if (cmd_index < 0) cmd_index = 0;
	return cmd_list[cmd_index];
}

function RunCmd() {
	let cmd_txt = $('#cmd_text_inp').val().trim();
	if (cmd_txt != '') {
		$('#cmd_btn').prop('disabled', true);
		$('#cmd_text_inp').prop('disabled', true);
		ipcRenderer.send('run-cmd', { cmd: cmd_txt });
	}
}

ipcRenderer.on('cmd-sent', (event, payload) => {
	if (payload) {
		cmd_index = cmd_list.length + 1;
		cmd_list.push($('#cmd_text_inp').val().trim());
		$('#cmd_text_inp').val('');
	}
	$('#cmd_btn').prop('disabled', false);
	$('#cmd_text_inp').prop('disabled', false);
	$('#cmd_text_inp').focus();
});

ipcRenderer.on('append-log', (event, payload) => {
	$('#log_box').append('<pre class="console_log">'+EncodeHTML(payload.msg)+'</pre>');
	$('#console_box').scrollTop($('#console_box').prop("scrollHeight"));
});

// ---- APP ----

ipcRenderer.on('init-ui', (event, payload) => {
	if (payload.state !== false) {
		const d = new Date();
		const curr_date = (d.getMonth()+1)+'/'+d.getDate()+'/'+d.getFullYear();
		const pyg_ip = pygmalion_ip.replaceAll('HUMAN_NAME', human_name).replaceAll('BOT_NAME', bot_name).replaceAll('DATE', curr_date);
		if (payload.state == 'AI_UI_DEFAULT') {
			prompt = default_ip.replaceAll('HUMAN_NAME', human_name).replaceAll('BOT_NAME', bot_name).replaceAll('DATE', curr_date);
			ip_vals = { chat:prompt, pygmalion:pyg_ip, bbcode:bbcode_ip };
		} else {
			prompt = payload.state;
		}
		$('#prompt_txta').val(prompt);
		$('#ip_select').val('default');
		$('#chat_log').html('<div class="prompt"><p id="init_prompt" class="prompt_txt">'+
			EncodeHTML(prompt)+' '+pedit_img_html+'</p></div>');
		$('#gen_result').html('');
		$('#img_result').html('');
	} else {
		$('#send_btn').prop('disabled', true);
		$('#read_txt_btn').prop('disabled', true);
		DisableButtons(true);
		StopThinking();
		StopGenerating();
	}
});

ipcRenderer.on('ai-ready', (event, payload) => {
	$('#send_btn').prop('disabled', false);
	$('#read_txt_btn').prop('disabled', false);
	DisableButtons(false, true);
	StopThinking();
	StopGenerating();
});

ipcRenderer.on('add-voices', (event, payload) => {
	let ttsgen_sel = false;
	if (payload.mode == 'SYS') {
		sys_voices = [];
		if ($('#ttsgen_mode').find(':selected').val() == 0) {
			$('#tts_voices').empty();
			ttsgen_sel = true;
		}
	} else {
		t5_voices = [];
		if ($('#ttsgen_mode').find(':selected').val() == 1) {
			$('#tts_voices').empty();
			ttsgen_sel = true;
		}
	}
	
	for (let i=0; i<payload.names.length; ++i) {
		const voice_name = payload.names[i].replaceAll('_', ' ');
		if (payload.set) {
			let prop_str = (i == bot_voice) ? i+'" selected' : i+'"';
			$('#voices').append('<option value="'+prop_str+'>'+voice_name+'</option>');
		}
		if (ttsgen_sel) {
			let prop_str = (i == tts_voice) ? i+'" selected' : i+'"';
			$('#tts_voices').append('<option value="'+prop_str+'>'+voice_name+'</option>');
		}
		if (payload.mode == 'SYS') {
			sys_voices.push(voice_name);
		} else {
			t5_voices.push(voice_name);
		}
	}
});

ipcRenderer.on('clear-voices', (event, payload) => {
	$('#voices').empty();
});

ipcRenderer.on('play-audio', (event, payload) => {
	if (payload.file == 'ERROR') {
		ipcRenderer.send('show-alert', { type: "error", msg: "There was an error generating the speech." });
	} else if (payload.file == 'NO_VOICES') {
		ipcRenderer.send('show-alert', { type: "error", msg: "There are no voices available for the selected speech engine." });
	} else {
		PlayAudio(payload.file);
	}
});

ipcRenderer.on('got-avatar', (event, payload) => {
	$('#loading_box').hide();
	if (payload.got != "true") {
		ipcRenderer.send('show-alert', { type: "error", msg: "Face animation engine cannot detect a face in the avatar image!" });
	}
});

ipcRenderer.on('got-file', (event, payload) => {
	if (payload.init) {
		active_dl = 1;
		total_dls = payload.remaining;
		$('#loading_msg').html('Downloading file 1 of '+total_dls+' ...');
		$('#loading_box').show();
	} else if (payload.remaining > 0) {
		$('#loading_msg').html('Downloading file '+(++active_dl)+' of '+total_dls+' ...');
	}
});

ipcRenderer.on('got-models', (event, payload) => {
	let hideMsgBox = true;
	if (payload.failed == 0) {
		$('#loading_msg').html('Finished downloading all model files.');
	} else if (payload.retry) {
		if (payload.failed > 0) {
			$('#loading_msg').html('Failed to download '+payload.failed+' files. Retrying ...');
		} else {
			$('#loading_msg').html('Failed to download model files. Retrying ...');
		}
		let mode_sel = $('#anim_mode').find(':selected').val();
		ipcRenderer.send('check-models', { mode: mode_sel });
		hideMsgBox = false;
	} else if (payload.failed > 0) {
		$('#loading_msg').html('Failed to download '+payload.failed+' files.');
	} else {
		$('#loading_msg').html('Download failed due to an unexpected error.');
	}
	if (hideMsgBox) {
		setTimeout(function() {
			$('#loading_box').hide();
			$('#loading_msg').html('Processing avatar image... please wait.');
		}, 1500);
	}
});

ipcRenderer.on('prompt-msg', (event, payload) => {
	$('#init_prompt').html(payload.msg);
	$('#gen_result').html(payload.msg);
	$('#img_result').html(payload.msg);
});

ipcRenderer.on('load-config', (event, payload) => {
	const chat_config = payload.configs.chat;
	const app_config = payload.configs.app;
	const ai_config = payload.configs.ai;
	const gen_config = payload.configs.gen;
	
	human_name = chat_config.human_name;
	bot_name = chat_config.bot_name;
	bot_voice = chat_config.bot_voice;
	speech_vol = chat_config.speech_vol;
	speech_rate = chat_config.speech_rate;
	pitch_shift = chat_config.pitch_shift;
	talk_mode = chat_config.talk_mode;
	tts_mode = chat_config.tts_mode;
	anim_mode = chat_config.anim_mode;

	avatar_img = app_config.avatar_img;
	script_dir = app_config.script_dir;
	python_bin = app_config.python_bin;
	model_dir = app_config.model_dir;
	sd_model = app_config.sd_model;
	tts_model = app_config.tts_model;
	voc_model = app_config.voc_model;
	model_args = app_config.model_args;
	model_type = app_config.model_type;
	imodel_type = app_config.imodel_type;
	smodel_type = app_config.smodel_type;
	comp_dev = app_config.comp_dev;
	start_meth = app_config.start_meth;

	max_mm = ai_config.msg_mem;
	max_rl = ai_config.max_res;
	min_rl = ai_config.min_res;
	b_temp = ai_config.base_temp;
	prompt_p = ai_config.prompt_p;
	top_k = ai_config.top_k;
	top_p = ai_config.top_p;
	typical_p = ai_config.typical_p;
	rep_penalty = ai_config.rep_penalty;
	
	tts_voice = gen_config.tts_voice;
	
	if (anim_mode == 0) {
		avatar_mp4 = 'file://'+script_dir+'/MakeItTalk/examples/face_pred_fls_speech_audio_embed.mp4';
	} else if (anim_mode == 1) {
		avatar_mp4 = 'file://'+script_dir+'/Wav2Lip/results/face_pred_fls_speech_audio_embed.mp4';
	} else {
		avatar_mp4 = 'file://'+script_dir+'/SadTalker/results/face_pred_fls_speech_audio_embed.mp4';
	}

	if (payload.skip_inputs) return;

	$('#user_name').val(human_name);
	$('#bot_name').val(bot_name);
	$('#tts_mode').val(tts_mode);
	$('#anim_mode').val(anim_mode);
	$('#voices').val(bot_voice);
	$('#speech_vol').val(speech_vol);
	$('#speech_rate').val(speech_rate);
	$('#speech_pitch').val(pitch_shift);

	$('#avatar_img').prop('src', 'file://'+avatar_img);
	$('#script_dir').val(script_dir);
	$('#python_bin').val(python_bin);
	$('#model_dir').val(model_dir);
	$('#sd_model').val(sd_model);
	$('#tts_model').val(tts_model);
	$('#voc_model').val(voc_model);
	$('#model_args').val(model_args);
	$('#tmodel_select').val(model_type);
	$('#imodel_select').val(imodel_type);
	$('#smodel_select').val(smodel_type);
	$('#device_select').val(comp_dev);
	$('#startup_select').val(start_meth);

	$('#max_msg_mem').val(max_mm);
	$('#max_res_len').val(max_rl);
	$('#min_res_len').val(min_rl);
	$('#base_temp').val(b_temp);
	$('#pp_select').val(prompt_p);
	$('#top_k').val(top_k);
	$('#top_p').val(top_p);
	$('#typical_p').val(typical_p);
	$('#rep_penalty').val(rep_penalty);

	$('#ttsgen_mode').val(gen_config.tts_mode);
	$('#tts_voices').val(gen_config.tts_voice);
	$('#tts_speech_vol').val(gen_config.tts_vol);
	$('#tts_speech_rate').val(gen_config.tts_rate);
	$('#tts_speech_pitch').val(gen_config.tts_pitch);

	$('#gen_max_len').val(gen_config.max_len);
	$('#gen_min_len').val(gen_config.min_len);
	$('#gen_temp').val(gen_config.temp);
	$('#gen_top_k').val(gen_config.top_k);
	$('#gen_top_p').val(gen_config.top_p);
	$('#gen_typ_p').val(gen_config.typical_p);
	$('#gen_rep_p').val(gen_config.rep_penalty);
	
	$('#infer_steps').val(gen_config.inference_steps);
	$('#guidance').val(gen_config.guidance_scale);
	$('#safety_check').val(gen_config.safety_check);
	$('#vae_file').val(gen_config.vae_file);
	$('#lora_file').val(gen_config.lora_file);
	$('#lora_dir').val(gen_config.lora_dir);
	
	if (gen_config.image_width != 'auto' && gen_config.image_height != 'auto') {
		$('#is_select').val('custom');
		$('#img_width').val(gen_config.image_width);
		$('#img_height').val(gen_config.image_height);
		$('#is_input_box').show();
	}
	
	$('#tmode_img').prop('src', './img/talk_'+talk_mode+'.png');
});

function ApplySettings() {
	let sdir = TrimLast($('#script_dir').val().trim().replaceAll('\\', '/'), '/');
	let mdir = TrimLast($('#model_dir').val().trim().replaceAll('\\', '/'), '/');
	let sddir = TrimLast($('#sd_model').val().trim().replaceAll('\\', '/'), '/');
	let ttsdir = TrimLast($('#tts_model').val().trim().replaceAll('\\', '/'), '/');
	let vocdir = TrimLast($('#voc_model').val().trim().replaceAll('\\', '/'), '/');
	let pbin = $('#python_bin').val().trim().replaceAll('\\', '/');
	let margs = $('#model_args').val().replaceAll('\\', '/');
	let mtype = $('#tmodel_select').find(':selected').val();
	let itype = $('#imodel_select').find(':selected').val();
	let stype = $('#smodel_select').find(':selected').val();
	let cdev = $('#device_select').find(':selected').val();
	let smeth = $('#startup_select').find(':selected').val();
	if (script_dir != sdir || python_bin != pbin || model_dir != mdir || sd_model != sddir || tts_model != ttsdir || voc_model != vocdir || 
	model_args != margs || model_type != mtype || imodel_type != itype || smodel_type != stype || comp_dev != cdev || start_meth != smeth) {
		ipcRenderer.send('config-app', {
			script_dir: sdir, python_bin: pbin, model_dir: mdir, sd_model: sddir, tts_model: ttsdir, voc_model: vocdir, 
			model_args: margs, model_type: mtype, imodel_type: itype, smodel_type: stype, comp_dev: cdev, start_meth: smeth
		});
	}
}

function ApplyConfig() {
	if (adv_show) {
		const maxmm = $('#max_msg_mem').val();
		const maxrl = $('#max_res_len').val();
		const minrl = $('#min_res_len').val();
		const btemp = $('#base_temp').val();
		const topk = $('#top_k').val();
		const topp = $('#top_p').val();
		const typp = $('#typical_p').val();
		const repp = $('#rep_penalty').val();
		const promptp = $('#pp_select').find(':selected').val();
		if (max_mm != maxmm || max_rl != maxrl || min_rl != minrl || b_temp != btemp || 
		topk != top_k || topp != top_p || typp != typical_p || repp != rep_penalty || prompt_p != promptp) {
			ipcRenderer.send('config-ai', {
				max_mmem: maxmm, max_rlen: maxrl, min_rlen: minrl, temp: btemp, 
				tk: topk, tp: topp, typ: typp, rp: repp, pp: promptp
			});
			ipcRenderer.send('show-alert', { msg: 'New settings applied.' });
		}
	} else {
		const hname = $('#user_name').val();
		const bname = $('#bot_name').val();
		const spvol = $('#speech_vol').val();
		const srate = $('#speech_rate').val();
		const spitch = $('#speech_pitch').val();
		const newvc = $('#voices').find(':selected').val();
		const newtts = $('#tts_mode').find(':selected').val();
		const newam = $('#anim_mode').find(':selected').val();
		let alert_msg = true;

		if (human_name != hname || bot_name != bname) {
			ipcRenderer.send('update-users', { human: hname, bot: bname });
			alert_msg = false;
		}
		
		if (bot_voice != newvc || speech_vol != spvol || speech_rate != srate || pitch_shift != spitch || tts_mode != newtts || anim_mode != newam) {
			ipcRenderer.send('config-voice', { voice:newvc, vol:spvol, rate:srate, pitch:spitch, engine:newtts, amode:newam });
			if (anim_mode != newam) {
				$('#loading_box').show();
			} else if (alert_msg) {
				ipcRenderer.send('show-alert', { msg: 'New settings applied.' });
			}
		}
	}
}

function ToggleConfig() {
	if (adv_show) {
		$('#adv_btn').html('AI SETTINGS');
		$('#adv_config').hide();
		$('#basic_config').show();
	} else {
		$('#adv_btn').html('MAIN SETTINGS');
		$('#basic_config').hide();
		$('#adv_config').show();
	}
	adv_show = !adv_show;
}

function ShowInstallHelp() {
	if ($('#install_txt').is(":hidden")) {
		$('#install_txt').show();
		$('#setupargs_txt').hide();
		$('#inst_help_btn').css('color', '#c4c4cf');
		$('#args_help_btn').css('color', '#868687');
	}
}

function ShowSetupHelp() {
	if ($('#setupargs_txt').is(":hidden")) {
		$('#setupargs_txt').show();
		$('#install_txt').hide();
		$('#args_help_btn').css('color', '#c4c4cf');
		$('#inst_help_btn').css('color', '#868687');
	}
}

function ToggleEditPrompt() {
	if ($('#edit_dialog').is(":hidden")) {
		$('#edit_dialog').show();
		$('#prompt_txta').focus();
	} else {
		$('#edit_dialog').hide();
	}
}

function ApplyNewPrompt() {
	prompt = $('#prompt_txta').val();
	p_html = EncodeHTML(prompt).replaceAll("\n", "<br>");
	ipcRenderer.send('update-prompt', prompt.replaceAll("\n", "[AI_UI_BR]"));
	$('#init_prompt').html(p_html+' '+pedit_img_html);
	ToggleEditPrompt();
}

function ChangeTalkMode() {
	if (++talk_mode>2) talk_mode = 0;
	$('#tmode_img').prop('src', './img/talk_'+talk_mode+'.png');
	ipcRenderer.send('update-talk-mode', talk_mode);
}

function ChangeAvatar() {
	window.aiuiAPI.openAvatar().then(result => {
		if (result === false) return;
		$('#loading_box').show();
		avatar_img = result.replaceAll('\\', '/');
		ipcRenderer.send('update-avatar', avatar_img);
		$('#avatar_img').prop('src', 'file://'+avatar_img);
		$('#avatar_vid').hide();
		$('#other_vid').hide();
		$('#avatar_img').show();
	});
}

function OpenSample(elem) {
	window.aiuiAPI.openSample().then(result => {
		if (result === false) return;
		$('#'+elem).val(result.replaceAll('\\', '/'));
	});
}

function OpenFileFolder(elem) {
	window.aiuiAPI.openFileFolder().then(result => {
		if (result === false) return;
		$('#'+elem).val(result.replaceAll('\\', '/'));
	});
}

function OpenFolder(elem) {
	window.aiuiAPI.openFolder().then(result => {
		if (result === false) return;
		$('#'+elem).val(result.replaceAll('\\', '/'));
	});
}

function OpenFile(elem, trim_path=false) {
	window.aiuiAPI.openFile().then(result => {
		if (result === false) return;
		result = result.replaceAll('\\', '/');
		if (trim_path) {
			const last_slash = result.lastIndexOf('/');
			if (last_slash >= 0)
				result = result.substr(last_slash+1);
		}
		$('#'+elem).val(result);
	});
}

function SetAppVersion() {
	window.aiuiAPI.getAppVersion().then(result => {
		$('#app-version').html(result);
		$('#aiui-version').html(result);
	});
}

function ShowInFolder(elem) {
	let file_path = '';
	
	if (elem === null) {
		file_path = script_dir+'/ai_images/';
	} else if (typeof elem == "string") {
		if (elem == 'TTS_EMBEDS') {
			file_path = script_dir+'/embeddings/';
		} else {
			file_path = elem.replace('file:///', '').replace('file://', '');
			file_path = file_path.substr(0, file_path.indexOf('?'));
		}
	} else {
		file_path = $(elem).prop('src').
			replace('file:///', '').replace('file://', '');
	}

	ipcRenderer.send('show-in-dir', file_path);
}

function RestartScript() {
	ipcRenderer.invoke('restart-script');
}

function StartScript() {
	ipcRenderer.invoke('start-script');
}

function QuitApp() {
	ipcRenderer.invoke('close-app');
}

$(document).ready(function() {
	SetAppVersion();
	StartScript();

	$("#text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			SendMsg();
		}
	});
	
	$("#gen_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenText();
		}
	});
	
	$("#tts_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenSpeech();
		}
	});
	
	$("#img_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenImage();
		}
	});
	
	$("#img_text_neg").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenImage();
		}
	});
	
	$("#cmd_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			RunCmd();
		} else if (e.key === 'Arrow Up' || e.keyCode === 38) {
			e.preventDefault();
			$(this).val(CmdHistory(-1, $(this).val()));
		} else if (e.key === 'Arrow Down' || e.keyCode === 40) {
			e.preventDefault();
			$(this).val(CmdHistory(1, $(this).val()));
		}
	});

	$('#area_inp').on('keydown', function(e) {
		if (e.key === 'Tab' || e.keyCode == 9) {
			e.preventDefault();
			let s = this.selectionStart;
			$(this).val(function(i, v) {
				return v.substring(0, s) + "\t" + v.substring(this.selectionEnd)
			});
			this.selectionEnd = s + 1;
		}
	});
	
	$('#avatar_img').on('load', function() {
		if ($(this).prop('naturalWidth') != 256 || $(this).prop('naturalHeight') != 256) {
			ipcRenderer.send('show-alert', { msg: 'Avatar images must have a resolution of 256x256' });
		}
	});
	
	$('#ip_select').on('change', function() {
		let ip_sel = $(this).find(':selected').val();
		switch (ip_sel) {
		case 'default':
			$('#prompt_txta').val(ip_vals.chat);
			break;
		case 'pygmalion':
			$('#prompt_txta').val(ip_vals.pygmalion);
			break;
		case 'bbcode':
			$('#prompt_txta').val(ip_vals.bbcode);
			break;
		}
	});
	
	$('#is_select').on('change', function() {
		let is_sel = $(this).find(':selected').val();
		switch (is_sel) {
		case 'auto':
			$('#is_input_box').hide();
			break;
		case 'custom':
			$('#is_input_box').show();
			break;
		}
	});
	
	$('#tts_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		voice_arr = (mode_sel == 0) ? sys_voices : t5_voices;
		$('#voices').empty();
		for (let i=0; i<voice_arr.length; ++i) {
			$('#voices').append('<option value="'+i+'">'+voice_arr[i]+'</option>');
		}
	});
	
	$('#ttsgen_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		voice_arr = (mode_sel == 0) ? sys_voices : t5_voices;
		$('#tts_voices').empty();
		for (let i=0; i<voice_arr.length; ++i) {
			const voice_name = voice_arr[i].replaceAll('_', ' ');
			$('#tts_voices').append('<option value="'+i+'">'+voice_name+'</option>');
		}
	});
	
	$('#anim_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		ipcRenderer.send('check-models', { mode: mode_sel });
	});
});