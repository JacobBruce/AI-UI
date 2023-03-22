const process = require('child_process');

var AI_ENGINE = null;
var ENGINE_STATE = 'STOPPED';
var CHAT_STATE = 'INIT';
var CHAT_CONFIG = { human_name: 'Human', bot_name: 'Bot', bot_voice: 0, speech_vol: 1.0, speech_rate: 200, pitch_shift: 0 };
var AI_CONFIG = { msg_mem: 5, max_res: 50, min_res: 1, base_temp: 0.8, prompt_p: 0 };
var APP_CONFIG = { avatar_img: '', script_dir: '', python_bin: '', model_dir: '', model_type: 0, comp_dev: 'cpu' };
var GEN_CONFIG = { start_text: '', max_len: 50, min_len: 1, temp: 0.8 };
var MAIN_SCRIPT = 'main_aiui_engine.py';
var DONE_VOICES = false;
var CB_FUNCS = null;
var INIT_PROMPT = '';

function GetConfigs() {
	return { chat: CHAT_CONFIG, app: APP_CONFIG, ai: AI_CONFIG };
}

function SetConfigs(configs) {
	CHAT_CONFIG = configs.chat;
	APP_CONFIG = configs.app;
	AI_CONFIG = configs.ai;
}

function SetAvatar(img_file) {
	APP_CONFIG.avatar_img = img_file;
}

function SetUsernames(names) {
	CHAT_CONFIG.human_name = names.human;
	CHAT_CONFIG.bot_name = names.bot;
}

function SetPrompt(new_prompt) {
	INIT_PROMPT = new_prompt;
}

function ConfigSpeech(speech_config) {
	CHAT_CONFIG.bot_voice = speech_config.voice;
	CHAT_CONFIG.speech_vol = speech_config.vol;
	CHAT_CONFIG.speech_rate = speech_config.rate;
	CHAT_CONFIG.pitch_shift = speech_config.pitch;
}

function ConfigAI(ai_config) {
	AI_CONFIG.msg_mem = ai_config.max_mmem;
	AI_CONFIG.max_res = ai_config.max_rlen;
	AI_CONFIG.min_res = ai_config.min_rlen;
	AI_CONFIG.base_temp = ai_config.temp;
	AI_CONFIG.prompt_p = ai_config.pp;
}

function ConfigApp(app_config) {
	APP_CONFIG.script_dir = app_config.script_dir;
	APP_CONFIG.python_bin = app_config.python_bin;
	APP_CONFIG.model_dir = app_config.model_dir;
	APP_CONFIG.model_type = app_config.model_type;
	APP_CONFIG.comp_dev = app_config.comp_dev;
}

function ConfigGen(gen_config) {
	GEN_CONFIG.start_text = gen_config.txt;
	GEN_CONFIG.max_len = gen_config.max;
	GEN_CONFIG.min_len = gen_config.min;
	GEN_CONFIG.temp = gen_config.temp;
}

function SendMsg(message, req_state='HUMAN_INPUT') {
	if (AI_ENGINE === null) {
		console.log('ERROR: AI Engine not initialized');
		return;
	}
	if (CHAT_STATE == req_state || req_state == 'ANY') {
		console.log('STDIN: '+message);
		AI_ENGINE.stdin.write(message+"\r\n");
	} else {
		console.log('ERROR: wrong state for SendMsg(), CHAT_STATE='+CHAT_STATE);
	}
}

function StopScript(new_state) {
	console.log('Stopping engine script ...');
	if (ENGINE_STATE == 'STARTED') {
		ENGINE_STATE = new_state;
		AI_ENGINE.kill();
	} else {
		console.log('Engine script already stopped');
		ENGINE_STATE = new_state;
		if (new_state == 'RESTART') return true;
	}
	return false;
}

function StartScript(bot_out_func, gen_out_func, ai_ready_func, ai_ended_func, add_voice_func) {

	console.log('Executing engine script ...');
	if (ENGINE_STATE == 'STARTED') {
		console.log('Engine script already running');
		return;
	}
	
	if (CB_FUNCS === null) {
		CB_FUNCS = {
			bot_out: bot_out_func,
			gen_out: gen_out_func,
			ai_ready: ai_ready_func,
			ai_ended: ai_ended_func,
			add_voice: add_voice_func
		};
	}

	try {
		const mit_dir = APP_CONFIG.script_dir + '/MakeItTalk/';
		const script_file = mit_dir + MAIN_SCRIPT;
		AI_ENGINE = process.spawn(APP_CONFIG.python_bin, ['-u',script_file], {cwd:mit_dir});
		ENGINE_STATE = 'STARTED';
	} catch (err) {
		ENGINE_STATE = 'START_FAIL';
		console.log('Failed to start AI Engine');
		CB_FUNCS.ai_ended('Failed to start the AI Engine. Check your settings are correct.');
		return;
	}

	AI_ENGINE.stdin.setEncoding('utf-8');
	
	AI_ENGINE.stdout.on('data', function (data) {
		const out_str = data.toString().trim();
		console.log('STDOUT: '+out_str);
		if (out_str == 'HUMAN_INPUT:') {
			CHAT_STATE = 'HUMAN_INPUT';
			CB_FUNCS.ai_ready(false);
		} else if (out_str.startsWith('BOT_OUTPUT:')) {
			CHAT_STATE = 'BOT_OUTPUT';
			CB_FUNCS.bot_out(out_str.replace('BOT_OUTPUT:', ''));
		} else if (out_str.startsWith('GEN_OUTPUT:')) {
			CHAT_STATE = 'GEN_OUTPUT';
			CB_FUNCS.gen_out(out_str.replace('GEN_OUTPUT:', ''));
		} else if (out_str == 'YOUR_NAME:') {
			CHAT_STATE = 'HUMAN_NAME';
			console.log('STDIN: '+CHAT_CONFIG.human_name);
			AI_ENGINE.stdin.write(CHAT_CONFIG.human_name+"\r\n");
		} else if (out_str == 'BOT_NAME:') {
			CHAT_STATE = 'BOT_NAME';
			console.log('STDIN: '+CHAT_CONFIG.bot_name);
			AI_ENGINE.stdin.write(CHAT_CONFIG.bot_name+"\r\n");
		} else if (out_str == 'NEW_PROMPT:') {
			CHAT_STATE = 'NEW_PROMPT';
			console.log('STDIN: '+INIT_PROMPT);
			AI_ENGINE.stdin.write(INIT_PROMPT+"\r\n");	
		} else if (out_str == 'START_TEXT:') {
			CHAT_STATE = 'START_TEXT';
			console.log('STDIN: '+GEN_CONFIG.start_text);
			AI_ENGINE.stdin.write(GEN_CONFIG.start_text+"\r\n");
			console.log('STDIN: '+GEN_CONFIG.max_len);
			AI_ENGINE.stdin.write(GEN_CONFIG.max_len+"\r\n");
			console.log('STDIN: '+GEN_CONFIG.min_len);
			AI_ENGINE.stdin.write(GEN_CONFIG.min_len+"\r\n");
			console.log('STDIN: '+GEN_CONFIG.temp);
			AI_ENGINE.stdin.write(GEN_CONFIG.temp+"\r\n");
		} else if (out_str == 'AVATAR_IMG:') {
			CHAT_STATE = 'AVATAR_IMG';
			console.log('STDIN: '+APP_CONFIG.avatar_img);
			AI_ENGINE.stdin.write(APP_CONFIG.avatar_img+"\r\n");
		} else if (out_str == 'VOICE_CONFIG:') {
			CHAT_STATE = 'VOICE_CONFIG';
			console.log('STDIN: '+CHAT_CONFIG.bot_voice);
			AI_ENGINE.stdin.write(CHAT_CONFIG.bot_voice+"\r\n");
			console.log('STDIN: '+CHAT_CONFIG.speech_vol);
			AI_ENGINE.stdin.write(CHAT_CONFIG.speech_vol+"\r\n");
			console.log('STDIN: '+CHAT_CONFIG.speech_rate);
			AI_ENGINE.stdin.write(CHAT_CONFIG.speech_rate+"\r\n");
			console.log('STDIN: '+CHAT_CONFIG.pitch_shift);
			AI_ENGINE.stdin.write(CHAT_CONFIG.pitch_shift+"\r\n");
		} else if (out_str == 'AI_CONFIG:') {
			CHAT_STATE = 'AI_CONFIG';
			console.log('STDIN: '+AI_CONFIG.msg_mem);
			AI_ENGINE.stdin.write(AI_CONFIG.msg_mem+"\r\n");
			console.log('STDIN: '+AI_CONFIG.max_res);
			AI_ENGINE.stdin.write(AI_CONFIG.max_res+"\r\n");
			console.log('STDIN: '+AI_CONFIG.min_res);
			AI_ENGINE.stdin.write(AI_CONFIG.min_res+"\r\n");
			console.log('STDIN: '+AI_CONFIG.base_temp);
			AI_ENGINE.stdin.write(AI_CONFIG.base_temp+"\r\n");
			console.log('STDIN: '+AI_CONFIG.prompt_p);
			AI_ENGINE.stdin.write(AI_CONFIG.prompt_p+"\r\n");
		} else if (out_str == 'APP_CONFIG:') {
			CHAT_STATE = 'APP_CONFIG';
			console.log('STDIN: '+APP_CONFIG.script_dir);
			AI_ENGINE.stdin.write(APP_CONFIG.script_dir+"\r\n");
			console.log('STDIN: '+APP_CONFIG.model_dir);
			AI_ENGINE.stdin.write(APP_CONFIG.model_dir+"\r\n");
			console.log('STDIN: '+APP_CONFIG.model_type);
			AI_ENGINE.stdin.write(APP_CONFIG.model_type+"\r\n");
			console.log('STDIN: '+APP_CONFIG.comp_dev);
			AI_ENGINE.stdin.write(APP_CONFIG.comp_dev+"\r\n");
			console.log('STDIN: '+APP_CONFIG.avatar_img);
			AI_ENGINE.stdin.write(APP_CONFIG.avatar_img+"\r\n");
		} else if (out_str.startsWith('INIT_DONE:')) {
			CHAT_STATE = 'INIT_DONE';
			CB_FUNCS.ai_ready();
		} else if (DONE_VOICES == false && out_str.startsWith('VOICE_NAME:')) {
			const voices = out_str.trim().split("\n");
			for (let i=0; i<voices.length; i++) {
				const voice = voices[i].trim().replace('VOICE_NAME:','').split('VOICE_ID:');
				if (voice.length != 2) continue;
				CB_FUNCS.add_voice({ name:voice[0], id:voice[1] });
			}
			DONE_VOICES = true;
		}
	});
	
	AI_ENGINE.stderr.on('data', (data) => {
		console.error('STDERR: '+data.toString());
	});
	
	AI_ENGINE.on('exit', function (code, signal) {
		console.log('AI Engine exited with ' + `code ${code}`);
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
	setUsernames: SetUsernames,
	setPrompt: SetPrompt,
	configSpeech: ConfigSpeech,
	configAI: ConfigAI,
	configApp: ConfigApp,
	configGen: ConfigGen,
	getConfigs: GetConfigs,
	setConfigs: SetConfigs,
	setAvatar: SetAvatar
};