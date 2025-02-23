window.$ = window.jQuery = require('jquery');
const { ipcRenderer, webUtils } = require('electron');
const hljs = require('highlight.js');

const pedit_img_html = '<button id="edit_btn" onclick="ToggleEditPrompt()"><img class="edit_img" src="./img/edit_w.png" width="15" /></button>';
const default_ip = "Chat log between HUMAN_NAME and BOT_NAME on DATE";
const pygmalion_ip = "BOT_NAME's Persona: A helpful AI assistant who can use the [AI_IMG] tag to generate images from a description.\n<START>\nHUMAN_NAME: show me an image of a cyberpunk cityscape\nBOT_NAME: [AI_IMG]cyberpunk cityscape[/AI_IMG]";
const thinking_ip = "You are a helpful AI assistant who can think step by step before answering questions. You can put your thoughts inside the <think> and </think> XML tags to help you think through a problem before providing an answer. Your thoughts should always be contained between those XML tags so they can be hidden and separated from your actual responses.";
const thinking2_ip = "You are a helpful AI assistant who can think step by step before answering questions. You can put your thoughts inside the <thoughts> and </thoughts> XML tags to help you think through a problem before providing an answer. Your thoughts should always be contained between those XML tags so they can be hidden and separated from your actual responses.";
const bbcode_ip = "You can use BBCode tags to format text and embed media such as images and videos into your messages.\nUse [b], [i], [u], and [s] to make text bold, italic, underlined, or striked-through. Example: [b]this is bold text[/b]\nUse [h1], [h2], [h3], [h4], [h5], and [h6] for headings. Example: [h3]this is a heading[/h3]\nUse the [hr] tag to insert a horizontal rule. Example: this is above the line[hr]this is below\nUse the [center] tag to horizontally center text and other content. Example [center]this is centered text[/center]\nUse the [quote] tag for quoting text. Example: [quote]this is a quote[/quote]\nUse the [spoiler] tag for hiding spoiler text and other content. Example: [spoiler]this is a spoiler[/spoiler]\nUse the [pre] tag to post preformatted text. Example: [pre]this is preformatted text[/pre]\nUse the [ol] and [ul] tags for ordered and unordered lists. Use [li] or [*] for list items. Example: [ol][*]item1[*]item2[/ol]\nUse the [url] tag to post a link. Example: [url=http://test.com]this is a link[/url]\nUse the [code] tag to post a code snippet. Example: [code=C++]int abc = 123;[/code]\nUse the [video] and [audio] tags to post a video or audio file. Example: [video]http://test.com/vid.mp4[/video]\nUse the [youtube] tag to post a youtube video with just the video ID. Example: [youtube]5eqRuVp65eY[/youtube]\nUse the [img] tag to post an image. Example: [img]http://test.com/pic.png[/img]\nUse the [ai_img] tag to generate an image from a description using AI. Example: [ai_img]cute kitten[/ai_img]";
var tooluse_ip = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:\n<tools>{tool_funcs}</tools>\n\nFor each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>{\"arguments\": <args-dict>, \"name\": <function-name>}</tool_call>";
var tooluse2_ip = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n\nIf you decide to invoke any of the function(s), you MUST put it in this format:\n<tool_call>{\"arguments\": <args-dict>, \"name\": <function-name>}</tool_call>\n\nHere is a list of functions in JSON format that you can invoke:\n{tool_funcs}";
var ip_vals = { chat:'', pygmalion:'', think:'', think2:'', tools:'', tools2:'', bbcode:'' };

const bb_code_tags = [
	"B", "b", "I", "i", "U", "u", "S", "s", "OL", "ol", "UL", "ul", "LI", "li", "IMG", "img", "PRE", "pre", 
	"H1", "h1", "H2", "h2", "H3", "h3", "H4", "h4", "H5", "h5", "H6", "h6", "URL", "url", "CODE", "code", 
	"CENTER", "center", "SPOILER", "spoiler", "QUOTE", "quote", "AUDIO", "audio", "VIDEO", "video", "YOUTUBE", "youtube"
];

const bb_data_tags = ["URL", "url", "CODE", "code", "QUOTE", "quote", "SPOILER", "spoiler"];

const alpha_num_regex = /^[a-z0-9]+$/i;
var held_keys = [null,null,null];

var avatar_vid = null;
var speech_audio = null;
var audio_context = null;
var microphone_stream = null;
var audio_processor_node = null;
var loop_timer = null;
var dot_count = 0;
var rnd_int = 0;
var active_dl = 0;
var total_dls = 0;
var cmd_index = 0;
var max_ip_len = 256;
var last_gen_mode = 0;
var last_gen_voice = 0;
var btn_state = true;
var recording = false;
var thinking = false;
var generating = false;
var asr_pending = false;
var chat_exp = false;
var gen_exp = false;
var tts_exp = false;
var adv_show = false;
var sys_voices = [];
var ai_voices = [];
var cmd_list = [];
var attachments = [];
var tool_funcs = '';
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
var im_model = '';
var tts_model = '';
var sr_model = '';
var python_bin = '';
var avatar_mp4 = '';
var avatar_img = '';

var enable_bbcode = 1;
var enable_tooluse = 1;
var enable_devmode = 0;
var enable_asasro = 0;
var start_rec_keys = '';
var stop_rec_keys = '';

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

function FloatTo16BitPCM(output, offset, input) {
	for (let i = 0; i < input.length; i++, offset+=2){
		let s = Math.max(-1, Math.min(1, input[i]));
		output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
	}
}

function WriteString(view, offset, string) {
	for (let i = 0; i < string.length; i++) {
		view.setUint8(offset + i, string.charCodeAt(i));
	}
}

function EncodeWAV(samples) {
	let buffer = new ArrayBuffer(44 + samples.length * 2);
	let view = new DataView(buffer);

	/* RIFF identifier */
	WriteString(view, 0, 'RIFF');
	/* file length */
	view.setUint32(4, 36 + samples.length * 2, true);
	/* RIFF type */
	WriteString(view, 8, 'WAVE');
	/* format chunk identifier */
	WriteString(view, 12, 'fmt ');
	/* format chunk length */
	view.setUint32(16, 16, true);
	/* sample format (raw) */
	view.setUint16(20, 1, true);
	/* channel count */
	view.setUint16(22, 1, true);
	/* sample rate */
	view.setUint32(24, audio_context.sampleRate, true);
	/* byte rate (sample rate * block align) */
	view.setUint32(28, audio_context.sampleRate * 2, true);
	/* block align (channel count * bytes per sample) */
	view.setUint16(32, 2, true);
	/* bits per sample */
	view.setUint16(34, 16, true);
	/* data chunk identifier */
	WriteString(view, 36, 'data');
	/* data chunk length */
	view.setUint32(40, samples.length * 2, true);

	FloatTo16BitPCM(view, 44, samples);

	return view;
}

function StartMicrophone() {
	audio_context = new AudioContext();

	navigator.mediaDevices.getUserMedia({audio:true, video:false}).then((stream) => {
		StartRecording(stream);
	}).catch((err) => {
		ipcRenderer.send('show-alert', { type: "error", msg: 'Failed to get microphone audio stream.' });
	});
}

function SaveRecording(sample_buff) {
	recording = false;
	$('#loading_msg').html('Running speech recognition model ...');
	if (sample_buff.length > 0) {
		let dataView = EncodeWAV(sample_buff);
		ipcRenderer.send('save-recording', dataView);
	} else {
		$('#loading_box').hide();
	}
}

function StopRecording() {
	if (audio_processor_node !== null && recording)
		audio_processor_node.port.postMessage({msg:"stop"});
	recording = false;
}

function InitAudioProcessor() {
	audio_processor_node = new AudioWorkletNode(audio_context, 'worklet-processor', {});
	microphone_stream.connect(audio_processor_node);
	
	audio_processor_node.port.onmessage = (e) => {
		if (e.data.msg == "anim") {
			const circleRad = Math.min(5.0, e.data.sample * 10.0) + 3.0;
			$('#rec_circle').attr("r", Math.round(circleRad));
		} else if (e.data.msg == "stop") {
			SaveRecording(e.data.buffer);
		} else if (e.data.msg == "start") {
			$('#loading_msg').html('<svg id="rec_svg"><circle id="rec_circle" r="3" /></svg> Recording. Stop speaking for a few seconds to end recording.');
		}
	};
}

function StartRecording(stream=null) {
	recording = true;
	if (stream !== null) {
		microphone_stream = audio_context.createMediaStreamSource(stream);

		audio_context.audioWorklet.addModule('./abp.js').then(() => {
			InitAudioProcessor();
		}).catch((err) => {
			recording = false;
			$('#loading_box').hide();
			ipcRenderer.send('show-alert', { type: "error", msg: 'Failed to start recording.' });
		});
	} else {
		InitAudioProcessor();
	}
	$('#loading_msg').html('Speak into your microphone to begin recording.');
	$('#loading_box').show();
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

function EncodeAIXML(txt) {
	return txt.replaceAll('<thoughts>', '[AIUI_THOUGHTS]').replaceAll('</thoughts>', '[AIUI_THOUGHTS END]').
		replaceAll('<think>', '[AIUI_THOUGHTS]').replaceAll('</think>', '[AIUI_THOUGHTS END]');
}

function ConvertBB(txt) {
	return txt.replaceAll("[AI_UI_TAB]", "\t").replaceAll("[AI_UI_BR]", "<br>").
		replaceAll('[HR]', '</p><hr class="chat_hr"><p class="msg_p">').
		replaceAll('[B]', '<b>').replaceAll('[B END]', '</b>').
		replaceAll('[I]', '<i>').replaceAll('[I END]', '</i>').
		replaceAll('[U]', '<u>').replaceAll('[U END]', '</u>').
		replaceAll('[S]', '<s>').replaceAll('[S END]', '</s>').
		replaceAll("[H1]", '</p><h1>').replaceAll('[H1 END]', '</h1><p class="msg_p">').
		replaceAll("[H2]", '</p><h2>').replaceAll('[H2 END]', '</h2><p class="msg_p">').
		replaceAll("[H3]", '</p><h3>').replaceAll('[H3 END]', '</h3><p class="msg_p">').
		replaceAll("[H4]", '</p><h4>').replaceAll('[H4 END]', '</h4><p class="msg_p">').
		replaceAll("[H5]", '</p><h5>').replaceAll('[H5 END]', '</h5><p class="msg_p">').
		replaceAll("[H6]", '</p><h6>').replaceAll('[H6 END]', '</h6><p class="msg_p">').
		replaceAll("[OL]", '</p><ol>').replaceAll('[OL END]', '</ol><p class="msg_p">').
		replaceAll("[UL]", '</p><ul>').replaceAll('[UL END]', '</ul><p class="msg_p">').
		replaceAll("[LI]", '<li>').replaceAll('[LI END]', '</li>').
		replaceAll('[PRE]', '</p><pre>').replaceAll('[PRE END]', '</pre><p class="msg_p">').
		replaceAll('[QUOTE]', '</p><blockquote>').replaceAll('[QUOTE END]', '</blockquote><p class="msg_p">').
		replaceAll('[SPOILER]', '</p><details><summary onclick="ToggleDetails(this)">click to show spoiler</summary><hr class="chat_hr">').
		replaceAll('[SPOILER=', '</p><details><summary onclick="ToggleDetails(this)">').replaceAll('_SPOILER]', '</summary><hr class="chat_hr">').
		replaceAll('[SPOILER END]', '</details><p class="msg_p">').
		replaceAll('[CENTER]', '</p><center>').replaceAll('[CENTER END]', '</center><p class="msg_p">').
		replaceAll("[IMG]", '</p><img class="chat_img" src="').replaceAll('[IMG END]', '" onclick="ShowImage(this)"><p class="msg_p">').
		replaceAll('[CODE]', '</p><pre class="code_box"><code>').replaceAll('[CODE END]', '</code></pre><p class="msg_p">').
		replaceAll('[CODE=', '</p><pre class="code_box"><code class="language-').replaceAll('_CODE]', '">').
		replaceAll('[URL=', '<a href="').replaceAll('_URL]', '" target="_blank">').replaceAll('[URL END]', '</a>').
		replaceAll('[VIDEO=', '</p><video class="chat_vid" controls><source type="video/').replaceAll('_VIDEO]', '" src="').
		replaceAll('[AUDIO=', '</p><audio class="chat_aud" controls><source type="audio/').replaceAll('_AUDIO]', '" src="').
		replaceAll('[VIDEO END]', '"></video><p class="msg_p">').replaceAll('[AUDIO END]', '"></audio><p class="msg_p">').
		replaceAll('[YOUTUBE]', '</p><iframe class="yt_vid" src="https://www.youtube.com/embed/').replaceAll('[YOUTUBE END]', '"></iframe><p class="msg_p">');
}

function ConvertAIBB(txt) {
	let result = txt.replaceAll('[AIUI_THOUGHTS]', '</p><details><summary onclick="ToggleDetails(this)">click to show AI thoughts</summary><hr class="chat_hr">').
		replaceAll('[AIUI_THOUGHTS END]', '</details><p class="msg_p">');
	if (enable_bbcode == 1) {
		result = ConvertBB(result).replaceAll("[AI_IMG NUM_", '</p><img class="chat_img" src="file://'+script_dir+'/ai_images/image_').
		replaceAll("_CHAT_IMG]", '.png" title="').replaceAll("[AI_IMG END]", '" onclick="ShowInFolder(this)"><p class="msg_p">');
	}
	return result;
}

function EncodeBB(bbcode) {
	let result = bbcode.replaceAll("[hr]", "[HR]");
	
	for (let i=0; i < bb_code_tags.length; ++i)
	{
		const bbc_tag = bb_code_tags[i];
		const bbc_open_tag = "[" + bbc_tag + "]";
		const bbc_close_tag = "[/" + bbc_tag + "]";
		const bbc_upper_tag = bbc_tag.toUpperCase();
		
		while (result.includes(bbc_open_tag) && result.includes(bbc_close_tag)) {
			const tag_start = result.indexOf(bbc_open_tag) + bbc_open_tag.length;
			const tag_end = result.indexOf(bbc_close_tag);
			if (tag_start >= tag_end) break;
			const inner_txt = result.substring(tag_start, tag_end);
			let inner_enc = inner_txt.replaceAll("\t", "[AI_UI_TAB]").replaceAll("\n", "[AI_UI_BR]");
			if (bbc_upper_tag == "OL" || bbc_upper_tag == "UL") inner_enc = inner_enc.replaceAll("[*]", "[LI]");
			if (bbc_upper_tag == "URL") {
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, "["+bbc_upper_tag+"="+inner_txt+"_"+bbc_upper_tag+"]"+inner_enc+"["+bbc_upper_tag+" END]");
			} else if (bbc_upper_tag == "VIDEO") {
				const vid_link = inner_txt.toLowerCase().trim();
				let tag_data = "mp4";
				if (vid_link.endsWith("webm")) {
					tag_data = "webm";
				} else if (vid_link.endsWith("ogg")) {
					tag_data = "ogg";
				}
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, "["+bbc_upper_tag+"="+tag_data+"_"+bbc_upper_tag+"]"+inner_enc+"["+bbc_upper_tag+" END]");
			} else if (bbc_upper_tag == "AUDIO") {
				const aud_link = inner_txt.toLowerCase().trim();
				let tag_data = "mp3";
				if (aud_link.endsWith("wav")) {
					tag_data = "wav";
				} else if (aud_link.endsWith("ogg")) {
					tag_data = "ogg";
				}
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, "["+bbc_upper_tag+"="+tag_data+"_"+bbc_upper_tag+"]"+inner_enc+"["+bbc_upper_tag+" END]");
			} else {
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, "["+bbc_upper_tag+"]"+inner_enc+"["+bbc_upper_tag+" END]");
			}
		}
	}
	
	for (let i=0; i < bb_data_tags.length; ++i)
	{
		const bbc_tag = bb_data_tags[i];
		const bbc_data_tag = "[" + bbc_tag + "=";
		const bbc_close_tag = "[/" + bbc_tag + "]";
		const bbc_upper_tag = bbc_tag.toUpperCase();
		
		while (result.includes(bbc_data_tag) && result.includes(bbc_close_tag)) {
			const data_start = result.indexOf(bbc_data_tag) + bbc_data_tag.length;
			const tag_start = data_start + result.substring(data_start).indexOf(']');
			const tag_data = result.substring(data_start, tag_start);
			const tag_end = result.indexOf(bbc_close_tag);
			if (tag_start+1 >= tag_end) break;
			const inner_txt = result.substring(tag_start+1, tag_end);
			if (bbc_upper_tag == "QUOTE") {
				result = result.replace(bbc_data_tag+tag_data+"]"+inner_txt+bbc_close_tag, "["+bbc_upper_tag+"]"+inner_txt+"[AI_UI_BR]\t- "+tag_data+"["+bbc_upper_tag+" END]");
			} else {
				result = result.replace(bbc_data_tag+tag_data+"]"+inner_txt+bbc_close_tag, "["+bbc_upper_tag+"="+tag_data+"_"+bbc_upper_tag+"]"+inner_txt+"["+bbc_upper_tag+" END]");
			}
			break;
		}
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
	$('#send_btn').prop('disabled', bool_val);
	$('#apply_btn').prop('disabled', bool_val);
	$('#gen_btn').prop('disabled', bool_val);
	$('#tts_btn').prop('disabled', bool_val);
	$('#read_txt_btn').prop('disabled', bool_val);
	$('#clone_voice_btn').prop('disabled', bool_val);
	$('#img_btn').prop('disabled', bool_val);
	$('#edit_btn').prop('disabled', bool_val);
	$('#attach_btn').prop('disabled', bool_val);
	$('#mic_btn').prop('disabled', bool_val);
	$('.config_btn').prop('disabled', bool_val);
}

ipcRenderer.on('bot-msg', (event, payload) => {
	let bot_msg = payload.msg.trim();
	if (enable_bbcode == 0) {
		bot_msg = ConvertAIBB(EncodeHTML(EncodeAIXML(bot_msg)));
	} else {
		bot_msg = ConvertAIBB(EncodeHTML(EncodeAIXML(EncodeBB(bot_msg))));
	}
	const msg_html = '<div class="bmsg_box"><div class="bot_msg"><p class="msg_p">'+bot_msg+'</p></div></div>';
	$('#chat_log').append(msg_html.replaceAll('<p class="msg_p"></p>', '').replaceAll('<p class="msg_p"><br>', '<p class="msg_p">'));
	$("#chat_log div.bmsg_box").last().find("pre.code_box code").each(function () { hljs.highlightElement(this) });
	$('.msg_pad').remove();
	DisableButtons(false);
	StopThinking();
	if (payload.got_vid && talk_mode == 0) {
		$('#avatar_img').hide();
		PlayVideo();
	}
});

function SendMsg() {
	if (thinking || generating) return;
	if ($('#send_btn').prop('disabled')) return;
	let messages = [];
	let message = '';
	let html_msg = '';
	let send_delay = 0;
	if (chat_exp) {
		message = $('#area_inp').val().trim();
		$('#area_inp').val('');
	} else {
		message = $('#text_inp').val().trim();
		$('#text_inp').val('');
	}
	if (message == '') return;
	messages = message.split('[AIUI_END]');
	message = message.replaceAll("\n", "[AI_UI_BR]");
	for (let i=0; i<messages.length; ++i) {
		let msg = messages[i].trim();
		if (msg == '') continue;
		if (chat_exp) {
			if (enable_bbcode == 0) {
				html_msg += '<div class="umsg_box"><div class="chat_msg"><p class="msg_p">'+
							EncodeHTML(msg).replaceAll("\n", "<br>")+'</p>';
			} else {
				html_msg += '<div class="umsg_box"><div class="chat_msg"><p class="msg_p">'+
							ConvertBB(EncodeHTML(EncodeBB(msg)).replaceAll("\n", "<br>"))+'</p>';
			}
		} else {
			if (enable_bbcode == 0) {
				html_msg += '<div class="umsg_box"><div class="chat_msg"><p class="msg_p">'+
							EncodeHTML(msg)+'</p>';
			} else {
				html_msg += '<div class="umsg_box"><div class="chat_msg"><p class="msg_p">'+
							ConvertBB(EncodeHTML(EncodeBB(msg)))+'</p>';
			}
		}
	}
	html_msg = html_msg.replaceAll('<p class="msg_p"></p>', '').replaceAll('<p class="msg_p"><br>', '<p class="msg_p">');
	if (html_msg == '') return;
	if (!$('#attach_dialog').is(":hidden")) {
		ToggleAttachBox();
		if (attachments.length > 0) send_delay = 500;
	}
	if ($("#am_select").find(':selected').val() != "rag") {
		if (attachments.length > 0)
			html_msg += '<h5 class="files_head">ATTACHMENTS:</h5>';
		for (let i=0; i < attachments.length; ++i)
			html_msg += '<div class="msg_file"><small>📎 ' + EncodeHTML(attachments[i]) + '</small></div>';
		ClearAttachments();
	}
	$('#chat_log').append(html_msg+'</div></div>');
	$("#chat_log div.umsg_box").each(function () {
		$(this).find("pre.code_box code").each(function () {
			if ($(this).attr("data-highlighted") != "yes") hljs.highlightElement(this);
		});
	});
	$('#edit_dialog').hide();
	setTimeout(function () {
		DisableButtons(true);
		StartThinking();
		ipcRenderer.send('send-msg', { msg: message });
	}, send_delay);
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
	btn_state = false;
	DisExtraButtons(true);
	$('#chat_log').html('');
	ipcRenderer.send('send-msg', { msg: 'clear_chat' });
}

function ToggleMsgBox() {
	if (chat_exp) {
		$('#exp_box').hide();
		$('#text_inp').prop('disabled', false);
		$('#text_inp').focus();
	} else {
		$('#exp_box').show();
		$('#text_inp').prop('disabled', true);
		$('#area_inp').focus();
	}
	chat_exp = !chat_exp;
}

function ToggleDetails(elem) {
	let summary = $(elem).html();
	if (summary.includes("show")) {
		$(elem).html(summary.replace("show", "hide"));
	} else {
		$(elem).html(summary.replace("hide", "show"));
	}
}

function ToggleRecordBox() {
	if ($('#mic_btn').prop('disabled')) return;
	if (audio_context === null) {
		StartMicrophone();
	} else if (!recording) {
		StartRecording();
	}
}

function ToggleAttachBox() {
	if ($('#attach_dialog').is(":hidden")) {
		$('#attach_dialog').show();
	} else {
		$('#attach_dialog').hide();
		$('#input_box').hide();
		ipcRenderer.send('attach-files', { files: attachments, mode: $("#am_select").find(':selected').val() });
	}
}

function RemoveFile(elem) {
	const attachment = $(elem).parent();
	const fpath = decodeURI(attachment.attr("data-file"));
	attachment.remove();
	for (let i=0; i < attachments.length; ++i)
	{
		if (attachments[i] == fpath) {
			attachments.splice(i, 1);
			break;
		}
	}
}

function AttachFile(file_path, is_link=false) {
	if (file_path === null || file_path == '') return;
	if (attachments.includes(file_path)) return;
	if (attachments.length == 0) {
		$("#attach_box").html('');
	}
	attachments.push(file_path);
	if (is_link) {
		$("#attach_box").append('<div data-file="'+encodeURI(file_path)+'">🔗 '+EncodeHTML(file_path)+' <span onclick="RemoveFile(this)">×</span></div>');
	} else {
		$("#attach_box").append('<div data-file="'+encodeURI(file_path)+'">🗎 '+EncodeHTML(file_path)+' <span onclick="RemoveFile(this)">×</span></div>');
	}
}

function AttachFiles() {
	$("#attach_files").click();
}

function AttachLink() {
	if ($('#input_box').is(":hidden")) {
		$("#input_box").show();
	} else {
		$("#input_box").hide();
	}
}

function ClearAttachments() {
	attachments = []
	if ($('#am_select').find(':selected').val() == "multi") {
		$("#attach_box").html('<small id="attach_ph">drag and drop files here</small>');
	} else {
		$("#attach_box").html('<small id="attach_ph">drag and drop text files here</small>');
	}
}

function FileDragDrop(e) {
	e.preventDefault();
	const files = e.dataTransfer.files;
	for (let i=0; i < files.length; ++i) {
		const fpath = webUtils.getPathForFile(files[i]);
		AttachFile(fpath);
	}
}

function FileDragOver(e) {
	e.preventDefault();
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
		$('#gen_exp_box').hide();
		$('#gen_text_inp').prop('disabled', false);
		$('#gen_text_inp').focus();
	} else {
		$('#gen_exp_box').show();
		$('#gen_text_inp').prop('disabled', true);
		$('#gen_area_inp').focus();
	}
	gen_exp = !gen_exp;
}

function ToggleTTSBox() {
	if (tts_exp) {
		$('#tts_exp_box').hide();
		$('#tts_text_inp').prop('disabled', false);
		$('#tts_text_inp').focus();
	} else {
		$('#tts_exp_box').show();
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

function SaveVoice() {
	const voice_name = CleanFilename($('#voice_name').val()).replaceAll(' ', '_');
	if (voice_name == '') {
		ipcRenderer.send('show-alert', { msg: "A valid voice name is required!" });
		return;
	}
	const src_voice = script_dir+"/embeddings/ChatTTS/random.tmp";
	const dest_voice = script_dir+"/embeddings/ChatTTS/"+voice_name+".txt";
	ipcRenderer.send('save-voice', { dest_file:dest_voice, src_file:src_voice });
}

function CloneVoice() {
	if (thinking || generating) return;
	if ($('#clone_voice_btn').prop('disabled')) return;
	const cmodel = $('#vclone_mode').find(':selected').val();
	const tstext = $('#transcript').val();
	const vsample = $('#voice_sample').val();
	const vname = CleanFilename($('#voice_name').val()).replaceAll(' ', '_');
	if (vsample == '' || vname == '') {
		ipcRenderer.send('show-alert', { msg: "A valid voice sample and voice name is required!" });
		return;
	}
	DisableButtons(true, true);
	$('#tts_result').html('Cloning voice ...');
	ipcRenderer.send('clone-voice', { model:cmodel, sample:vsample, name:vname, transcript:tstext });
	StartGenerating('#ttsgen_box .generating');
}

ipcRenderer.on('clone-result', (event, payload) => {
	if (!payload.voice.startsWith('ERROR:')) {
		const voice_info = payload.voice.split(':');
		if (voice_info.length == 2) {
			if ($('#ttsgen_mode').find(':selected').val() > 0) $('#tts_voices').val(voice_info[1]);
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
	last_gen_mode = $('#ttsgen_mode').find(':selected').val();
	last_gen_voice = $('#tts_voices').find(':selected').val();
	const spvol = $('#tts_speech_vol').val();
	const srate = $('#tts_speech_rate').val();
	const spitch = $('#tts_speech_pitch').val();
	ipcRenderer.send('gen-speech', { tts_txt:speak_txt, voice:last_gen_voice, vol:spvol, rate:srate, pitch:spitch, engine:last_gen_mode });
	StartGenerating('#ttsgen_box .generating');
}

ipcRenderer.on('tts-result', (event, payload) => {
	if (payload.wav == 'ERROR') {
		$('#tts_result').html('There was an error generating the speech.');
	} else if (payload.wav == 'NO_VOICES') {
		$('#tts_result').html('There are no voices available for the selected speech engine.');
	} else {
		let wav_html = "<center><audio controls><source id='speech_wav' src='file://"+payload.wav+"?rnd="+RandInt()+"' type='audio/wav'></audio>"+
			"<br><br><button id='wav_dir_btn' class='btn cfg_btn' onclick='OpenWavDir()'>OPEN WAV FILE</button>";	
		if (smodel_type == 1 && last_gen_mode == 1 && last_gen_voice == 0) {
			wav_html += "&nbsp;<button id='save_voice_btn' class='btn cfg_btn' onclick='SaveVoice()'>SAVE VOICE</button> "+
				"<p><small>To save a voice you must enter a Voice Name in the Voice Cloning settings.</small></p></center>";
		} else {
			wav_html += "</center>";
		}
		$('#tts_result').html(wav_html);
	}
	DisableButtons(false, true);
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
		if (chat_exp) targ_input = '#area_inp';
		if (enable_asasro == 1) asr_pending = true;
	}
	let inp_txt = $(targ_input).val().trim();
	if (inp_txt != '') inp_txt += ' ';
	$(targ_input).val(inp_txt+payload.txt);
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
	const lora_scale = $('#lora_scale').val();
	let img_width = 'auto';
	let img_height = 'auto';
	if ($('#is_select').find(':selected').val() == 'custom') {
		img_width = $('#img_width').val();
		img_height = $('#img_height').val();
	}
	ipcRenderer.send('gen-image', {
		img_prompt: prompt_txt, neg_prompt: prompt_neg, steps: infer_steps, 
		guidance: guide_scale, width: img_width, height: img_height, check: safety_check,
		vae_file: vae_file, lora_file: lora_file, lora_dir: lora_dir, lora_scale: lora_scale
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
		const def_ip = default_ip.replaceAll('HUMAN_NAME', human_name).replaceAll('BOT_NAME', bot_name).replaceAll('DATE', curr_date);
		const pyg_ip = pygmalion_ip.replaceAll('HUMAN_NAME', human_name).replaceAll('BOT_NAME', bot_name).replaceAll('DATE', curr_date);
		const tools_ip = tooluse_ip.replaceAll("{tool_funcs}", tool_funcs);
		const tools2_ip = tooluse2_ip.replaceAll("{tool_funcs}", tool_funcs);
		if (payload.state == 'AI_UI_DEFAULT') {
			prompt = def_ip;
		} else {
			prompt = payload.state.replaceAll("[AI_UI_BR]", "\n");
		}
		ip_vals = { chat:def_ip, pygmalion:pyg_ip, think:thinking_ip, think2:thinking2_ip, tools:tools_ip, tools2:tools2_ip, bbcode:bbcode_ip };
		RefreshPrompt(prompt);
		$('#gen_result').html('');
		$('#img_result').html('');
		$('#tts_result').html('');
	} else {
		$('#loading_box').hide();
		DisableButtons(true);
		StopThinking();
		StopGenerating();
	}
});

ipcRenderer.on('ai-ready', (event, payload) => {
	if (asr_pending) {
		asr_pending = false;
		SendMsg();
	}
	DisableButtons(false, true);
	StopThinking();
	StopGenerating();
});

ipcRenderer.on('start-recording', (event, payload) => {
	ToggleRecordBox();
});

ipcRenderer.on('stop-recording', (event, payload) => {
	StopRecording();
});

ipcRenderer.on('add-voices', (event, payload) => {
	let ttsgen_sel = false;
	let ttsgen_mode = $('#ttsgen_mode').find(':selected').val()
	
	if (payload.mode == 'SYS') {
		sys_voices = [];
		if (ttsgen_mode == 0) {
			$('#tts_voices').empty();
			ttsgen_sel = true;
		}
	} else {
		ai_voices = [];
		if (ttsgen_mode > 0) {
			$('#tts_voices').empty();
			ttsgen_sel = true;
		}
	}
	
	for (let i=0; i<payload.names.length; ++i)
	{
		let voice_name = payload.names[i];
		if (voice_name.startsWith("af_") || voice_name.startsWith("am_") || 
		voice_name.startsWith("bf_") || voice_name.startsWith("bm_") ||
		voice_name.startsWith("ef_") || voice_name.startsWith("em_") ||
		voice_name.startsWith("ff_") || voice_name.startsWith("fm_") ||
		voice_name.startsWith("hf_") || voice_name.startsWith("hm_") ||
		voice_name.startsWith("if_") || voice_name.startsWith("im_") ||
		voice_name.startsWith("jf_") || voice_name.startsWith("jm_") ||
		voice_name.startsWith("pf_") || voice_name.startsWith("pm_") ||
		voice_name.startsWith("zf_") || voice_name.startsWith("zm_")) {
			if (voice_name[0] == 'a') {
				voice_name += ' (American English - ';
			} else if (voice_name[0] == 'b') {
				voice_name += ' (British English - ';
			} else if (voice_name[0] == 'e') {
				voice_name += ' (Spanish - ';
			} else if (voice_name[0] == 'f') {
				voice_name += ' (French - ';
			} else if (voice_name[0] == 'h') {
				voice_name += ' (Hindi - ';
			} else if (voice_name[0] == 'i') {
				voice_name += ' (Italian - ';
			} else if (voice_name[0] == 'j') {
				voice_name += ' (Japanese - ';
			} else if (voice_name[0] == 'p') {
				voice_name += ' (Brazilian Portuguese - ';
			} else if (voice_name[0] == 'z') {
				voice_name += ' (Mandarin Chinese - ';
			} else {
				voice_name += ' (Unknown - ';
			}
			if (voice_name[1] == 'f') {
				voice_name += 'Female)';
			} else {
				voice_name += 'Male)';
			}
			voice_name = voice_name.substring(3);
		}
		voice_name = voice_name.replaceAll('_', ' ');
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
			ai_voices.push(voice_name);
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

ipcRenderer.on('got-tools', (event, payload) => {
	tool_funcs = payload.tools;
	ip_vals.tools = tooluse_ip.replaceAll("{tool_funcs}", tool_funcs);
	ip_vals.tools2 = tooluse2_ip.replaceAll("{tool_funcs}", tool_funcs);
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
		}, 1500);
	}
});

ipcRenderer.on('prompt-msg', (event, payload) => {
	$('#init_prompt').html(payload.msg);
	$('#gen_result').html(payload.msg);
	$('#img_result').html(payload.msg);
	$('#tts_result').html(payload.msg);
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
	im_model = app_config.sd_model;
	tts_model = app_config.tts_model;
	sr_model = app_config.sr_model;
	model_args = app_config.model_args;
	model_type = app_config.model_type;
	imodel_type = app_config.imodel_type;
	smodel_type = app_config.smodel_type;
	comp_dev = app_config.comp_dev;
	start_meth = app_config.start_meth;
	
	enable_bbcode = app_config.enable_bbcode;
	enable_tooluse = app_config.enable_tooluse;
	enable_devmode = app_config.enable_devmode;
	enable_asasro = app_config.enable_asasro;
	start_rec_keys = app_config.start_rec_keys;
	stop_rec_keys = app_config.stop_rec_keys;

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
	
	ipcRenderer.send('show-menubar', enable_devmode);

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
	$('#im_model').val(im_model);
	$('#tts_model').val(tts_model);
	$('#sr_model').val(sr_model);
	$('#model_args').val(model_args);
	$('#tmodel_select').val(model_type);
	$('#imodel_select').val(imodel_type);
	$('#smodel_select').val(smodel_type);
	$('#device_select').val(comp_dev);
	$('#startup_select').val(start_meth);
	
	$('#bbcode_enable').val(enable_bbcode);
	$('#tooluse_enable').val(enable_tooluse);
	$('#devmode_enable').val(enable_devmode);
	$('#asasro_enable').val(enable_asasro);
	$('#start_rec_keys').val(start_rec_keys);
	$('#stop_rec_keys').val(stop_rec_keys);

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
	$('#lora_scale').val(gen_config.lora_scale);
	
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
	let imdir = TrimLast($('#im_model').val().trim().replaceAll('\\', '/'), '/');
	let ttsdir = TrimLast($('#tts_model').val().trim().replaceAll('\\', '/'), '/');
	let srdir = TrimLast($('#sr_model').val().trim().replaceAll('\\', '/'), '/');
	let pbin = $('#python_bin').val().trim().replaceAll('\\', '/');
	let margs = $('#model_args').val().replaceAll('\\', '/');
	let mtype = $('#tmodel_select').find(':selected').val();
	let itype = $('#imodel_select').find(':selected').val();
	let stype = $('#smodel_select').find(':selected').val();
	let cdev = $('#device_select').find(':selected').val();
	let smeth = $('#startup_select').find(':selected').val();
	let ebbcode = $('#bbcode_enable').find(':selected').val();
	let etooluse = $('#tooluse_enable').find(':selected').val();
	let edevmode = $('#devmode_enable').find(':selected').val();
	let easasro = $('#asasro_enable').find(':selected').val();
	let startrk = $('#start_rec_keys').val().trim();
	let stoprk = $('#stop_rec_keys').val().trim();
	if (script_dir != sdir || python_bin != pbin || model_dir != mdir || im_model != imdir || tts_model != ttsdir || sr_model != srdir || 
	model_args != margs || model_type != mtype || imodel_type != itype || smodel_type != stype || comp_dev != cdev || start_meth != smeth ||
	enable_bbcode != ebbcode || enable_tooluse != etooluse) {
		ipcRenderer.send('config-app', {
			script_dir: sdir, python_bin: pbin, model_dir: mdir, sd_model: imdir, tts_model: ttsdir, sr_model: srdir, 
			model_args: margs, model_type: mtype, imodel_type: itype, smodel_type: stype, comp_dev: cdev, start_meth: smeth,
			enable_bbcode: ebbcode, enable_tooluse: etooluse
		});
	} else if (enable_devmode != edevmode || enable_asasro != easasro || start_rec_keys != startrk || stop_rec_keys != stoprk) {
		ipcRenderer.send('config-other', {
			enable_devmode: edevmode, enable_asasro: easasro, start_rec_keys: startrk, stop_rec_keys: stoprk
		});
		enable_devmode = edevmode;
		enable_asasro = easasro;
		start_rec_keys = startrk;
		stop_rec_keys = stoprk;
		ipcRenderer.send('show-menubar', edevmode);
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
				$('#loading_msg').html('Processing avatar image... please wait.');
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

function ShowMainConfig() {
	if ($('#main_settings').is(":hidden")) {
		$('#main_settings').show();
		$('#other_settings').hide();
		$('#main_conf_btn').css('color', '#c4c4cf');
		$('#other_conf_btn').css('color', '#868687');
	}
}

function ShowOtherConfig() {
	if ($('#other_settings').is(":hidden")) {
		$('#other_settings').show();
		$('#main_settings').hide();
		$('#other_conf_btn').css('color', '#c4c4cf');
		$('#main_conf_btn').css('color', '#868687');
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
	let p_html = '';
	if (prompt.length > max_ip_len) {
		p_html = EncodeHTML(prompt.substring(0, max_ip_len)).replaceAll("\n", "<br>")+"...";
	} else {
		p_html = EncodeHTML(prompt).replaceAll("\n", "<br>");
	}
	ipcRenderer.send('update-prompt', prompt.replaceAll("\n", "[AI_UI_BR]"));
	$('#init_prompt').html(p_html+' '+pedit_img_html);
	ToggleEditPrompt();
}

function RefreshPrompt(prompt_txt) {
	$('#prompt_txta').val(prompt_txt);
	$('#ip_select').val('default');
	let p_html = '<div class="prompt"><p id="init_prompt" class="prompt_txt">';
	if (prompt_txt.length > max_ip_len) {
		p_html += EncodeHTML(prompt_txt.substring(0, max_ip_len)).replaceAll("\n", "<br>")+"... "+pedit_img_html+'</p></div>';
	} else {
		p_html += EncodeHTML(prompt_txt).replaceAll("\n", "<br>")+' '+pedit_img_html+'</p></div>';
	}
	$('#chat_log').html(p_html);
}

function ChangeTalkMode() {
	if (++talk_mode>2) talk_mode = 0;
	$('#tmode_img').prop('src', './img/talk_'+talk_mode+'.png');
	ipcRenderer.send('update-talk-mode', talk_mode);
}

function ChangeAvatar() {
	window.aiuiAPI.openAvatar().then(result => {
		if (result === false) return;
		$('#loading_msg').html('Processing avatar image... please wait.');
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

function ShowImage(elem) {
	let file_path = $(elem).prop('src').
		replace('file:///', '').replace('file://', '');
	
	if (file_path != '')
		ipcRenderer.send('show-img', file_path);
}

function UpdateHeldKeys(input_id, got_combo=false) {
	let keyShortcut = '';
	if (held_keys[0] !== null) {
		keyShortcut += held_keys[0]+'+';
	}
	if (held_keys[1] !== null) {
		keyShortcut += held_keys[1]+'+';
	}
	if (held_keys[2] !== null) {
		keyShortcut += held_keys[2];
	}
	if (keyShortcut.endsWith('+'))
		keyShortcut = keyShortcut.substr(0, keyShortcut.length-1);
	
	let inpElem = $(input_id);
	inpElem.val(keyShortcut);

	if (got_combo && inpElem.hasClass("keys_input")) {
		held_keys = [null,null,null];
		$(':focus').blur();
	}
}

function ShortcutKeyDown(input_id, key_str, key_code) {
	let keyName = '';
	if (typeof key_str == "string" && key_str.length == 1 && alpha_num_regex.test(key_str)) {
		held_keys[2] = key_str;
		if (held_keys[0] !== null || held_keys[1] !== null) {
			UpdateHeldKeys(input_id, true);
			return true;
		}
	} else if (key_code == 16 || key_code == 17 || key_code == 18) {
		keyName = key_str.replace("Control", "Ctrl");
	}
	if (keyName != '') {
		if (held_keys[0] === null) {
			held_keys[0] = keyName;
		} else if (held_keys[0] != keyName) {
			held_keys[1] = keyName;
		};
	}
	UpdateHeldKeys(input_id);
	return false;
}

function ShortcutKeyUp(input_id, key_str, key_code) {
	key_str = key_str.replace("Control", "Ctrl");
	if (typeof key_str == "string" && key_str.length == 1 && held_keys[2] === key_str) {
		held_keys[2] = null;
	} else if (held_keys[0] === key_str) {
		held_keys[0] = null;
	} else if (held_keys[1] === key_str) {
		held_keys[1] = null;
	}
	UpdateHeldKeys(input_id);
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

	$("#input_btn").on('click', function () {
		const fpath = $("#user_input").val();
		AttachFile(fpath, true);
		$("#input_box").hide();
	});
	
	$("#user_input").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			$("#input_btn").click();
		}
	});
	
	$("#attach_files").on('change', function(e) {
		for (let i=0; i < e.target.files.length; ++i) {
			AttachFile(webUtils.getPathForFile(e.target.files[i]));
		}
		$(this).val('');
	});
	
	$("#am_select").on('change', function() {
		ClearAttachments();
	});

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
	
	$('#start_rec_keys').on('keydown', function(e) {
		e.preventDefault();
		ShortcutKeyDown('#start_rec_keys', e.key, e.keyCode);
	});
	
	$('#start_rec_keys').on('keyup', function(e) {
		e.preventDefault();
		ShortcutKeyUp('#start_rec_keys', e.key, e.keyCode);
	});
	
	$('#stop_rec_keys').on('keydown', function(e) {
		e.preventDefault();
		ShortcutKeyDown('#stop_rec_keys', e.key, e.keyCode);
	});
	
	$('#stop_rec_keys').on('keyup', function(e) {
		e.preventDefault();
		ShortcutKeyUp('#stop_rec_keys', e.key, e.keyCode);
	});
	
	$('body').on('keydown', function(e) {
		if ($('#config_tab').is(':hidden')) {
			if (ShortcutKeyDown('#held_keys', e.key, e.keyCode)) {
				const heldShortcut = $('#held_keys').val();
				if (heldShortcut == start_rec_keys) {
					ToggleRecordBox();
				} else if (heldShortcut == stop_rec_keys) {
					StopRecording();
				}
			}
		}
	});
	
	$('body').on('keyup', function(e) {
		if ($('#config_tab').is(':hidden'))
			ShortcutKeyUp('#held_keys', e.key, e.keyCode);
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
		case 'think':
			$('#prompt_txta').val(ip_vals.think);
			break;
		case 'think2':
			$('#prompt_txta').val(ip_vals.think2);
			break;
		case 'tools':
			$('#prompt_txta').val(ip_vals.tools);
			break;
		case 'tools2':
			$('#prompt_txta').val(ip_vals.tools2);
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
		voice_arr = (mode_sel == 0) ? sys_voices : ai_voices;
		$('#voices').empty();
		for (let i=0; i<voice_arr.length; ++i) {
			$('#voices').append('<option value="'+i+'">'+voice_arr[i]+'</option>');
		}
	});
	
	$('#ttsgen_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		voice_arr = (mode_sel == 0) ? sys_voices : ai_voices;
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
	
	$('#vclone_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		if (mode_sel == 1) {
			$('#ts_txt_row').show();
		} else {
			$('#ts_txt_row').hide();
		}
	});
});