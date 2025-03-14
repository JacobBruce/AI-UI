const process = require('child_process');
const fs = require('fs');

var CHAT_CONFIG = { human_name: 'Human', bot_name: 'Bot', bot_voice: 0, speech_vol: 1.0, speech_rate: 200, pitch_shift: 0, talk_mode: 0, tts_mode: 0, anim_mode: 0 };
var AI_CONFIG = { msg_mem: 5, max_res: 50, min_res: 1, base_temp: 0.8, prompt_p: 0, top_k: 50, top_p: 1.0, typical_p: 1.0, rep_penalty: 1.0, temp_format: 'none', tools_file: 'custom' };
var APP_CONFIG = {
	avatar_img: '', script_dir: '', python_bin: '', model_dir: '', sd_model: '', tts_model: '', sr_model: '', model_args: '', model_type: 0, imodel_type: 0, smodel_type: 0, comp_dev: 'auto', start_meth: 'text', //main
	enable_bbcode: 1, enable_tooluse: 1, enable_devmode: 0, enable_asasro: 0, start_rec_keys: '', stop_rec_keys: '' //other
};
var GEN_CONFIG = { 
	prompt_text: '', prompt_neg: '', image_width: 'auto', image_height: 'auto', inference_steps: 50, guidance_scale: 7.5, safety_check: 1, vae_file: '', lora_file: '', lora_dir: '', lora_scale: 1.0, //image
	max_len: 50, min_len: 1, temp: 0.8, top_k: 50, top_p: 1.0, typical_p: 1.0, rep_penalty: 1.0, //text
	tts_voice: 0, tts_vol: 1.0, tts_rate: 200, tts_pitch: 0, tts_mode: 0 //speech
};

var ENDL = "\r\n";
var AI_ENGINE = null;
var CB_FUNCS = null;
var CMD_PROC = null;
var CMD_LOCK = false;
var CMD_CWD = '';
var CMD_STATE = 'STOPPED';
var ENGINE_STATE = 'STOPPED';
var CHAT_STATE = 'INIT';
var CHAT_FILES = '';
var SAVE_FILE = '';
var CLONE_MODEL = '';
var CLONE_SAMPLE = '';
var CLONE_NAME = '';
var CLONE_TEXT = '';
var INIT_PROMPT = '';
var TTS_TEXT = '';
var TTS_MODES = ['SYS', 'AI'];

function EngineRunning() {
	return (ENGINE_STATE == 'STARTED' || ENGINE_STATE == 'RESTART');
}

function GetConfigs() {
	return { chat: CHAT_CONFIG, app: APP_CONFIG, ai: AI_CONFIG, gen: GEN_CONFIG };
}

function SetConfigs(configs) {
	for (let prop in CHAT_CONFIG) {
		if (configs.chat.hasOwnProperty(prop))
			CHAT_CONFIG[prop] = configs.chat[prop];
	}
	for (let prop in APP_CONFIG) {
		if (configs.app.hasOwnProperty(prop))
			APP_CONFIG[prop] = configs.app[prop];
	}
	for (let prop in AI_CONFIG) {
		if (configs.ai.hasOwnProperty(prop))
			AI_CONFIG[prop] = configs.ai[prop];
	}
	if (configs.hasOwnProperty('gen')) {
		for (let prop in GEN_CONFIG) {
			if (configs.gen.hasOwnProperty(prop))
				GEN_CONFIG[prop] = configs.gen[prop];
		}
	}
}

function InitAppConfig(avatar_img, engine_dir) {
	APP_CONFIG.avatar_img = avatar_img;
	APP_CONFIG.script_dir = engine_dir;
}

function SetAvatar(img_file) {
	APP_CONFIG.avatar_img = img_file;
}

function SetTalkMode(new_mode) {
	CHAT_CONFIG.talk_mode = new_mode;
}

function SetUsernames(names) {
	CHAT_CONFIG.human_name = names.human;
	CHAT_CONFIG.bot_name = names.bot;
}

function SetCloneVoice(voice) {
	CLONE_MODEL = voice.model;
	CLONE_SAMPLE = voice.sample;
	CLONE_NAME = voice.name;
	CLONE_TEXT = voice.transcript;
}

function SetSaveFile(save_file) {
	SAVE_FILE = save_file;
}

function SetPrompt(new_prompt) {
	INIT_PROMPT = new_prompt;
}

function SetReadText(tts_txt) {
	TTS_TEXT = tts_txt;
}

function SetChatFiles(chat_files) {
	CHAT_FILES = chat_files;
}

function ConfigSpeech(speech_config) {
	CHAT_CONFIG.bot_voice = speech_config.voice;
	CHAT_CONFIG.speech_vol = speech_config.vol;
	CHAT_CONFIG.speech_rate = speech_config.rate;
	CHAT_CONFIG.pitch_shift = speech_config.pitch;
	CHAT_CONFIG.tts_mode = speech_config.engine;
	CHAT_CONFIG.anim_mode = speech_config.amode;
}

function ConfigTools(tools_config) {
	AI_CONFIG.temp_format = tools_config.temp_format;
	AI_CONFIG.tools_file = tools_config.tools_file;
}

function ConfigAI(ai_config) {
	AI_CONFIG.msg_mem = ai_config.max_mmem;
	AI_CONFIG.max_res = ai_config.max_rlen;
	AI_CONFIG.min_res = ai_config.min_rlen;
	AI_CONFIG.base_temp = ai_config.temp;
	AI_CONFIG.prompt_p = ai_config.pp;
	AI_CONFIG.top_k = ai_config.tk;
	AI_CONFIG.top_p = ai_config.tp;
	AI_CONFIG.typical_p = ai_config.typ;
	AI_CONFIG.rep_penalty = ai_config.rp;
}

function ConfigApp(app_config) {
	APP_CONFIG.script_dir = app_config.script_dir;
	APP_CONFIG.python_bin = app_config.python_bin;
	APP_CONFIG.model_dir = app_config.model_dir;
	APP_CONFIG.model_type = app_config.model_type;
	APP_CONFIG.imodel_type = app_config.imodel_type;
	APP_CONFIG.smodel_type = app_config.smodel_type;
	APP_CONFIG.model_args = app_config.model_args;
	APP_CONFIG.sd_model = app_config.sd_model;
	APP_CONFIG.tts_model = app_config.tts_model;
	APP_CONFIG.sr_model = app_config.sr_model;
	APP_CONFIG.comp_dev = app_config.comp_dev;
	APP_CONFIG.start_meth = app_config.start_meth;
	APP_CONFIG.enable_bbcode = app_config.enable_bbcode;
	APP_CONFIG.enable_tooluse = app_config.enable_tooluse;
}

function ConfigOther(app_config) {
	APP_CONFIG.enable_devmode = app_config.enable_devmode;
	APP_CONFIG.enable_asasro = app_config.enable_asasro;
	APP_CONFIG.start_rec_keys = app_config.start_rec_keys;
	APP_CONFIG.stop_rec_keys = app_config.stop_rec_keys;
}

function ConfigGen(gen_config) {
	if (gen_config.hasOwnProperty('img_prompt')) {
		GEN_CONFIG.prompt_text = gen_config.img_prompt;
		GEN_CONFIG.prompt_neg = gen_config.neg_prompt;
		GEN_CONFIG.guidance_scale = gen_config.guidance;
		GEN_CONFIG.inference_steps = gen_config.steps;
		GEN_CONFIG.image_width = gen_config.width;
		GEN_CONFIG.image_height = gen_config.height;
		GEN_CONFIG.safety_check = gen_config.check;
		GEN_CONFIG.vae_file = gen_config.vae_file;
		GEN_CONFIG.lora_file = gen_config.lora_file;
		GEN_CONFIG.lora_dir = gen_config.lora_dir;
		GEN_CONFIG.lora_scale = gen_config.lora_scale;
	} else if (gen_config.hasOwnProperty('tts_txt')) {
		GEN_CONFIG.tts_voice = gen_config.voice;
		GEN_CONFIG.tts_vol = gen_config.vol;
		GEN_CONFIG.tts_rate = gen_config.rate;
		GEN_CONFIG.tts_pitch = gen_config.pitch;
		GEN_CONFIG.tts_mode = gen_config.engine;
	} else {
		GEN_CONFIG.prompt_text = gen_config.txt;
		GEN_CONFIG.max_len = gen_config.max;
		GEN_CONFIG.min_len = gen_config.min;
		GEN_CONFIG.temp = gen_config.temp;
		GEN_CONFIG.top_k = gen_config.top_k;
		GEN_CONFIG.top_p = gen_config.top_p;
		GEN_CONFIG.typical_p = gen_config.typ_p;
		GEN_CONFIG.rep_penalty = gen_config.rep_p;
	}
}

function SetCallbacks(bot_out_func, gen_out_func, tts_out_func, asr_out_func, img_out_func, clone_out_func, ai_ready_func, 
ai_ended_func, add_voices_func, clear_voices_func, play_audio_func, append_log_func, avatar_got_func, got_tools_func) {
	if (CB_FUNCS === null) {
		CB_FUNCS = {
			bot_out: bot_out_func,
			gen_out: gen_out_func,
			tts_out: tts_out_func,
			asr_out: asr_out_func,
			img_out: img_out_func,
			clone_out: clone_out_func,
			ai_ready: ai_ready_func,
			ai_ended: ai_ended_func,
			add_voices: add_voices_func,
			clear_voices: clear_voices_func,
			play_audio: play_audio_func,
			append_log: append_log_func,
			avatar_got: avatar_got_func,
			got_tools: got_tools_func
		};
	}
}

function SetPlatform(platform_name) {
	if (platform_name.startsWith('win')) {
		ENDL = "\r\n";
	} else {
		ENDL = "\n";
	}
}

function LogToConsole(msg) {
	console.log(msg);
	if (CB_FUNCS !== null) 
		CB_FUNCS.append_log(msg);
}

function SendMsg(message, req_state='HUMAN_INPUT') {
	if (AI_ENGINE === null) {
		LogToConsole('ERROR: AI Engine not initialized');
		return;
	}
	if (CHAT_STATE == req_state || req_state == 'ANY') {
		LogToConsole('STDIN: '+message);
		AI_ENGINE.stdin.write(message+ENDL);
	} else {
		LogToConsole('ERROR: wrong state for SendMsg(), CHAT_STATE='+CHAT_STATE);
	}
}

function RunCommand(cmd) {
	const cmd_str = cmd.trim();
	if (cmd_str == '') return false;
	
	if (CMD_STATE == 'STOPPED') {
		
		try {
			CMD_CWD = (CMD_CWD == '') ? APP_CONFIG.script_dir : CMD_CWD;
			const script_file = APP_CONFIG.script_dir + '/cmd_control.py';
			if (fs.existsSync(script_file) && fs.existsSync(APP_CONFIG.python_bin)) {
				CMD_PROC = process.spawn(
					APP_CONFIG.python_bin, ['-u',script_file], 
					{cwd:CMD_CWD, windowsHide:true}
				);
				CMD_STATE = 'STARTED';
			} else {
				throw Error('cannot find files to spawn python process');
			}
		} catch (err) {
			CMD_STATE = 'STOPPED';
			LogToConsole('Failed to run command. Check your settings are correct.');
			return false;
		}
		
		CMD_PROC.stdin.setEncoding('utf-8');
		
		CMD_PROC.stdout.on('data', function (data) {
			const out_str = data.toString().trim();
			if (out_str == 'AIUI_CMD:') {
				LogToConsole(CMD_CWD+'>'+cmd_str);
				if (cmd_str.startsWith('cd ')) {
					CMD_CWD = cmd_str.replace('cd ', '').replaceAll('"', '').trim();
					CMD_PROC.stdin.write("cd"+ENDL);
				} else {
					CMD_PROC.stdin.write(cmd_str+ENDL);
				}
			} else if (out_str == 'AIUI_CWD:') {
				CMD_PROC.stdin.write(CMD_CWD+ENDL);
			} else if (out_str.startsWith('AIUI_CWD:')) {
				CMD_CWD = out_str.replace('AIUI_CWD:', '');
			} else {
				LogToConsole(out_str);
			}
			CMD_LOCK = true;
			setTimeout(function(){ CMD_LOCK = false; }, 1000);
		});
		
		CMD_PROC.stderr.on('data', (data) => {
			LogToConsole(data.toString());
			CMD_LOCK = true;
			setTimeout(function(){ CMD_LOCK = false; }, 1000);
		});
		
		CMD_PROC.on('exit', function (code, signal) {
			CMD_STATE = 'STOPPED';
		});
	} else if (!CMD_LOCK) {
		LogToConsole(CMD_CWD+'>'+cmd_str);
		CMD_PROC.stdin.write(cmd_str+ENDL);
	} else {
		return false;
	}
	
	return true;
}

function StopScript(new_state='STOPPING') {
	LogToConsole('Stopping engine script ...');
	if (ENGINE_STATE == 'STARTED') {
		ENGINE_STATE = new_state;
		AI_ENGINE.kill();
	} else {
		LogToConsole('Engine script already stopped');
		ENGINE_STATE = new_state;
		if (new_state == 'RESTART') return true;
	}
	return false;
}

function StartScript() {

	LogToConsole('Executing engine script ...');
	if (ENGINE_STATE == 'STARTED') {
		LogToConsole('Engine script already running');
		return;
	}
	
	if (CMD_STATE != 'STOPPED') {
		try {
			LogToConsole('Stopping command shell process ...');
			CMD_PROC.kill();
		} catch (err) {
			CMD_STATE = 'STOPPED';
		}
	}

	try {
		const script_file = APP_CONFIG.script_dir + '/aiui_engine.py';
		if (fs.existsSync(script_file) && fs.existsSync(APP_CONFIG.python_bin)) {
			AI_ENGINE = process.spawn(
				APP_CONFIG.python_bin, ['-u',script_file],
				{cwd:APP_CONFIG.script_dir, windowsHide:true}
			);
			ENGINE_STATE = 'STARTED';
		} else {
			throw Error('cannot find files to spawn python process');
		}
	} catch (err) {
		ENGINE_STATE = 'START_FAIL';
		LogToConsole('Failed to start AI Engine');
		CB_FUNCS.ai_ended('Failed to start the AI Engine. Check your settings are correct.');
		return;
	}

	AI_ENGINE.stdin.setEncoding('utf-8');
	
	AI_ENGINE.stdout.on('data', function (data) {
		const out_str = data.toString().trim();
		LogToConsole('STDOUT: '+out_str);
		if (out_str == 'HUMAN_INPUT:') {
			CHAT_STATE = 'HUMAN_INPUT';
			CB_FUNCS.ai_ready();
		} else if (out_str.startsWith('BOT_NOANIM:')) {
			CHAT_STATE = 'BOT_OUTPUT';
			CB_FUNCS.bot_out(out_str.replace('BOT_NOANIM:', ''), false);
		} else if (out_str.startsWith('BOT_OUTPUT:')) {
			CHAT_STATE = 'BOT_OUTPUT';
			CB_FUNCS.bot_out(out_str.replace('BOT_OUTPUT:', ''));
		} else if (out_str.startsWith('GEN_OUTPUT:')) {
			CHAT_STATE = 'GEN_OUTPUT';
			CB_FUNCS.gen_out(out_str.replace('GEN_OUTPUT:', ''));
		} else if (out_str.startsWith('IMG_OUTPUT:')) {
			CHAT_STATE = 'IMG_OUTPUT';
			CB_FUNCS.img_out(out_str.replace('IMG_OUTPUT:', ''));
		} else if (out_str.startsWith('CLONE_OUTPUT:')) {
			CHAT_STATE = 'CLONE_OUTPUT';
			CB_FUNCS.clone_out(out_str.replace('CLONE_OUTPUT:', ''));
		} else if (out_str.startsWith('ASR_OUTPUT:')) {
			CHAT_STATE = 'ASR_OUTPUT';
			CB_FUNCS.asr_out(out_str.replace('ASR_OUTPUT:', ''));
		} else if (out_str.startsWith('TTS_OUTPUT:')) {
			CHAT_STATE = 'TTS_OUTPUT';
			CB_FUNCS.tts_out(out_str.replace('TTS_OUTPUT:', ''));
		} else if (out_str.startsWith('TTS_TEXT:')) {
			CHAT_STATE = 'TTS_TEXT';
			LogToConsole('STDIN: '+TTS_TEXT);
			AI_ENGINE.stdin.write(TTS_TEXT+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.tts_voice);
			AI_ENGINE.stdin.write(GEN_CONFIG.tts_voice+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.tts_vol);
			AI_ENGINE.stdin.write(GEN_CONFIG.tts_vol+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.tts_rate);
			AI_ENGINE.stdin.write(GEN_CONFIG.tts_rate+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.tts_pitch);
			AI_ENGINE.stdin.write(GEN_CONFIG.tts_pitch+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.tts_mode);
			AI_ENGINE.stdin.write(GEN_CONFIG.tts_mode+ENDL);
		} else if (out_str.startsWith('SPEAK_TEXT:')) {
			CHAT_STATE = 'SPEAK_TEXT';
			LogToConsole('STDIN: '+TTS_TEXT);
			AI_ENGINE.stdin.write(TTS_TEXT+ENDL);
		} else if (out_str.startsWith('GOT_AVATAR:')) {
			CHAT_STATE = 'GOT_AVATAR';
			CB_FUNCS.avatar_got(out_str.replace('GOT_AVATAR:', ''));
		} else if (out_str == 'YOUR_NAME:') {
			CHAT_STATE = 'HUMAN_NAME';
			LogToConsole('STDIN: '+CHAT_CONFIG.human_name);
			AI_ENGINE.stdin.write(CHAT_CONFIG.human_name+ENDL);
		} else if (out_str == 'BOT_NAME:') {
			CHAT_STATE = 'BOT_NAME';
			LogToConsole('STDIN: '+CHAT_CONFIG.bot_name);
			AI_ENGINE.stdin.write(CHAT_CONFIG.bot_name+ENDL);
		} else if (out_str == 'NEW_PROMPT:') {
			CHAT_STATE = 'NEW_PROMPT';
			LogToConsole('STDIN: '+INIT_PROMPT);
			AI_ENGINE.stdin.write(INIT_PROMPT+ENDL);
		} else if (out_str == 'SAVE_FILE:') {
			CHAT_STATE = 'SAVE_FILE';
			LogToConsole('STDIN: '+SAVE_FILE);
			AI_ENGINE.stdin.write(SAVE_FILE+ENDL);
		} else if (out_str == 'START_TEXT:') {
			CHAT_STATE = 'START_TEXT';
			LogToConsole('STDIN: '+GEN_CONFIG.prompt_text);
			AI_ENGINE.stdin.write(GEN_CONFIG.prompt_text+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.max_len);
			AI_ENGINE.stdin.write(GEN_CONFIG.max_len+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.min_len);
			AI_ENGINE.stdin.write(GEN_CONFIG.min_len+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.temp);
			AI_ENGINE.stdin.write(GEN_CONFIG.temp+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.top_k);
			AI_ENGINE.stdin.write(GEN_CONFIG.top_k+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.top_p);
			AI_ENGINE.stdin.write(GEN_CONFIG.top_p+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.typical_p);
			AI_ENGINE.stdin.write(GEN_CONFIG.typical_p+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.rep_penalty);
			AI_ENGINE.stdin.write(GEN_CONFIG.rep_penalty+ENDL);
		} else if (out_str == 'IMAGE_PROMPT:') {
			CHAT_STATE = 'IMAGE_PROMPT';
			LogToConsole('STDIN: '+GEN_CONFIG.prompt_text);
			AI_ENGINE.stdin.write(GEN_CONFIG.prompt_text+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.prompt_neg);
			AI_ENGINE.stdin.write(GEN_CONFIG.prompt_neg+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.inference_steps);
			AI_ENGINE.stdin.write(GEN_CONFIG.inference_steps+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.guidance_scale);
			AI_ENGINE.stdin.write(GEN_CONFIG.guidance_scale+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.safety_check);
			AI_ENGINE.stdin.write(GEN_CONFIG.safety_check+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.image_width);
			AI_ENGINE.stdin.write(GEN_CONFIG.image_width+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.image_height);
			AI_ENGINE.stdin.write(GEN_CONFIG.image_height+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.vae_file);
			AI_ENGINE.stdin.write(GEN_CONFIG.vae_file+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.lora_file);
			AI_ENGINE.stdin.write(GEN_CONFIG.lora_file+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.lora_dir);
			AI_ENGINE.stdin.write(GEN_CONFIG.lora_dir+ENDL);
			LogToConsole('STDIN: '+GEN_CONFIG.lora_scale);
			AI_ENGINE.stdin.write(GEN_CONFIG.lora_scale+ENDL);
		} else if (out_str == 'AVATAR_IMG:') {
			CHAT_STATE = 'AVATAR_IMG';
			LogToConsole('STDIN: '+APP_CONFIG.avatar_img);
			AI_ENGINE.stdin.write(APP_CONFIG.avatar_img+ENDL);
		} else if (out_str == 'TALK_MODE:') {
			CHAT_STATE = 'TALK_MODE';
			LogToConsole('STDIN: '+CHAT_CONFIG.talk_mode);
			AI_ENGINE.stdin.write(CHAT_CONFIG.talk_mode+ENDL);
		} else if (out_str == 'CLONE_CONFIG:') {
			CHAT_STATE = 'CLONE_CONFIG';
			LogToConsole('STDIN: '+CLONE_MODEL);
			AI_ENGINE.stdin.write(CLONE_MODEL+ENDL);
			LogToConsole('STDIN: '+CLONE_NAME);
			AI_ENGINE.stdin.write(CLONE_NAME+ENDL);
			LogToConsole('STDIN: '+CLONE_SAMPLE);
			AI_ENGINE.stdin.write(CLONE_SAMPLE+ENDL);
			LogToConsole('STDIN: '+CLONE_TEXT);
			AI_ENGINE.stdin.write(CLONE_TEXT+ENDL);
		} else if (out_str == 'VOICE_CONFIG:') {
			CHAT_STATE = 'VOICE_CONFIG';
			LogToConsole('STDIN: '+CHAT_CONFIG.bot_voice);
			AI_ENGINE.stdin.write(CHAT_CONFIG.bot_voice+ENDL);
			LogToConsole('STDIN: '+CHAT_CONFIG.speech_vol);
			AI_ENGINE.stdin.write(CHAT_CONFIG.speech_vol+ENDL);
			LogToConsole('STDIN: '+CHAT_CONFIG.speech_rate);
			AI_ENGINE.stdin.write(CHAT_CONFIG.speech_rate+ENDL);
			LogToConsole('STDIN: '+CHAT_CONFIG.pitch_shift);
			AI_ENGINE.stdin.write(CHAT_CONFIG.pitch_shift+ENDL);
			LogToConsole('STDIN: '+CHAT_CONFIG.talk_mode);
			AI_ENGINE.stdin.write(CHAT_CONFIG.talk_mode+ENDL);
			LogToConsole('STDIN: '+CHAT_CONFIG.tts_mode);
			AI_ENGINE.stdin.write(CHAT_CONFIG.tts_mode+ENDL);
			LogToConsole('STDIN: '+CHAT_CONFIG.anim_mode);
			AI_ENGINE.stdin.write(CHAT_CONFIG.anim_mode+ENDL);
		} else if (out_str == 'AI_CONFIG:') {
			CHAT_STATE = 'AI_CONFIG';
			LogToConsole('STDIN: '+AI_CONFIG.msg_mem);
			AI_ENGINE.stdin.write(AI_CONFIG.msg_mem+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.max_res);
			AI_ENGINE.stdin.write(AI_CONFIG.max_res+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.min_res);
			AI_ENGINE.stdin.write(AI_CONFIG.min_res+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.base_temp);
			AI_ENGINE.stdin.write(AI_CONFIG.base_temp+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.prompt_p);
			AI_ENGINE.stdin.write(AI_CONFIG.prompt_p+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.top_k);
			AI_ENGINE.stdin.write(AI_CONFIG.top_k+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.top_p);
			AI_ENGINE.stdin.write(AI_CONFIG.top_p+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.typical_p);
			AI_ENGINE.stdin.write(AI_CONFIG.typical_p+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.rep_penalty);
			AI_ENGINE.stdin.write(AI_CONFIG.rep_penalty+ENDL);
		} else if (out_str == 'TOOL_CONFIG:') {
			CHAT_STATE = 'TOOL_CONFIG';
			LogToConsole('STDIN: '+AI_CONFIG.temp_format);
			AI_ENGINE.stdin.write(AI_CONFIG.temp_format+ENDL);
			LogToConsole('STDIN: '+AI_CONFIG.tools_file);
			AI_ENGINE.stdin.write(AI_CONFIG.tools_file+ENDL);
		} else if (out_str == 'APP_CONFIG:') {
			CHAT_STATE = 'APP_CONFIG';
			LogToConsole('STDIN: '+APP_CONFIG.script_dir);
			AI_ENGINE.stdin.write(APP_CONFIG.script_dir+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.model_dir);
			AI_ENGINE.stdin.write(APP_CONFIG.model_dir+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.sd_model);
			AI_ENGINE.stdin.write(APP_CONFIG.sd_model+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.tts_model);
			AI_ENGINE.stdin.write(APP_CONFIG.tts_model+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.sr_model);
			AI_ENGINE.stdin.write(APP_CONFIG.sr_model+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.model_type);
			AI_ENGINE.stdin.write(APP_CONFIG.model_type+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.imodel_type);
			AI_ENGINE.stdin.write(APP_CONFIG.imodel_type+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.smodel_type);
			AI_ENGINE.stdin.write(APP_CONFIG.smodel_type+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.model_args);
			AI_ENGINE.stdin.write(APP_CONFIG.model_args+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.comp_dev);
			AI_ENGINE.stdin.write(APP_CONFIG.comp_dev+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.start_meth);
			AI_ENGINE.stdin.write(APP_CONFIG.start_meth+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.enable_bbcode);
			AI_ENGINE.stdin.write(APP_CONFIG.enable_bbcode+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.enable_tooluse);
			AI_ENGINE.stdin.write(APP_CONFIG.enable_tooluse+ENDL);
			LogToConsole('STDIN: '+APP_CONFIG.avatar_img);
			AI_ENGINE.stdin.write(APP_CONFIG.avatar_img+ENDL);
		} else if (out_str.startsWith('PLAY_SPEECH:')) {
			CHAT_STATE = 'PLAY_SPEECH';
			CB_FUNCS.play_audio(out_str.replace('PLAY_SPEECH:','').trim());
		} else if (out_str.startsWith('INIT_DONE:')) {
			CHAT_STATE = 'INIT_DONE';
			CB_FUNCS.ai_ready('AI_UI_DEFAULT');
		} else if (out_str.startsWith('CLEAR_DONE:')) {
			CHAT_STATE = 'CLEAR_DONE';
			CB_FUNCS.ai_ready(out_str.replace('CLEAR_DONE:', ''));
		} else if (out_str.startsWith('TOOL_FUNCS:')) {
			CHAT_STATE = 'TOOL_FUNCS';
			CB_FUNCS.got_tools(out_str.replace('TOOL_FUNCS:',''));
		} else if (out_str.startsWith('CHAT_FILES:')) {
			CHAT_STATE = 'CHAT_FILES';
			const files_str = CHAT_FILES.join("[AI_UI_BR]");
			LogToConsole('STDIN: '+files_str);
			AI_ENGINE.stdin.write(files_str+ENDL);
		} else if (out_str.startsWith('SYS_VOICES:') || out_str.startsWith('AI_VOICES:')) {
			let voice_mode = out_str.startsWith('SYS_VOICES:') ? 'SYS' : 'AI';
			let do_set = (TTS_MODES[CHAT_CONFIG.tts_mode] == voice_mode);
			if (do_set) CB_FUNCS.clear_voices();
			let voice_list = out_str.replace(voice_mode+'_VOICES:','');
			let voice_names = [];
			const voices = voice_list.trim().split("\n");
			for (let i=0; i<voices.length; i++) {
				const voice = voices[i].trim().replace('VOICE_NAME:','').split('VOICE_ID:');
				if (voice.length != 2) continue;
				if (voice[0].endsWith('.npy') || voice[0].endsWith('.txt')) {
					voice_names.push(voice[0].substring(0, voice[0].length-4));
				} else if (voice[0].endsWith('.pt')) {
					voice_names.push(voice[0].substring(0, voice[0].length-3));
				} else {
					voice_names.push(voice[0]);
				}
			}
			CB_FUNCS.add_voices({names:voice_names, mode:voice_mode, set:do_set});
		}
	});
	
	AI_ENGINE.stderr.on('data', (data) => {
		LogToConsole('STDERR: '+data.toString());
	});
	
	AI_ENGINE.on('exit', function (code, signal) {
		LogToConsole('AI Engine exited with code '+code);
		CHAT_STATE = 'INIT';
		if (ENGINE_STATE == 'RESTART') {
			CB_FUNCS.ai_ended('Restarting AI Engine... please wait.');
			setTimeout(StartScript, 100);
		} else {
			ENGINE_STATE = 'STOPPED';
			if (code != 0) {
				CB_FUNCS.ai_ended('The AI Engine stopped running. Check the Settings tab to restart it.');
			} else {
				CB_FUNCS.ai_ended('The AI Engine is offline. Check the Settings tab to restart it.');
			}
		}
	});
}

module.exports = {
	startScript: StartScript,
	stopScript: StopScript,
	sendMsg: SendMsg,
	setPlatform: SetPlatform,
	setCallbacks: SetCallbacks,
	setUsernames: SetUsernames,
	setPrompt: SetPrompt,
	initAppConfig: InitAppConfig,
	configSpeech: ConfigSpeech,
	configTools: ConfigTools,
	configAI: ConfigAI,
	configApp: ConfigApp,
	configOther: ConfigOther,
	configGen: ConfigGen,
	getConfigs: GetConfigs,
	setConfigs: SetConfigs,
	setAvatar: SetAvatar,
	setTalkMode: SetTalkMode,
	setReadText: SetReadText,
	setChatFiles: SetChatFiles,
	setCloneVoice: SetCloneVoice,
	setSaveFile: SetSaveFile,
	engineRunning: EngineRunning,
	runCommand: RunCommand
};
