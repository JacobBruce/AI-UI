const process = require('child_process');
const fs = require('fs');
const core = require('./core.js');

var ENDL = "\r\n";
var AI_ENGINE = null;
var CB_FUNCS = null;
var CMD_PROC = null;
var CMD_LOCK = false;
var CMD_CWD = '';
var CMD_STATE = 'STOPPED';
var ENGINE_READY = false;
var ENGINE_STATE = 'STOPPED';
var TTS_MODES = ['SYS', 'AI'];
var GEN_STREAM = false;
var STDOUT_BUF = '';
var BOOT_SENT = false;
var STREAMING = false;
var STOP_SENT = false;
var EXPORT_WAIT = null;
var TOOL_PERM_HANDLER = null;

function EngineRunning() {
	return (ENGINE_STATE == 'STARTED' || ENGINE_STATE == 'RESTART');
}

function SetCallbacks(bot_out_func, gen_out_func, tts_out_func, asr_out_func, img_out_func, clone_out_func, ai_ready_func, ai_ended_func,
add_voices_func, clear_voices_func, play_audio_func, append_log_func, avatar_got_func, got_tools_func, stream_start_func, stream_text_func,
stream_end_func, turn_start_func, turn_patch_func, turn_end_func, prompt_files_func, got_tool_catalog_func, compact_func, context_usage_func) {
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
		got_tools: got_tools_func,
		got_tool_catalog: got_tool_catalog_func || function() {},
		stream_start: stream_start_func,
		stream_text: stream_text_func,
		stream_end: stream_end_func || function() {},
		turn_start: turn_start_func || function() {},
		turn_patch: turn_patch_func || function() {},
		turn_end: turn_end_func || function() {},
		prompt_files: prompt_files_func,
		compact: compact_func || function() {},
		context_usage: context_usage_func || function() {}
	};
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

function SendJson(obj) {
	if (AI_ENGINE === null || ENGINE_STATE != 'STARTED') {
		LogToConsole('ERROR: AI Engine not initialized');
		return false;
	}
	const payload = Object.assign({ v: 1 }, obj);
	const line = JSON.stringify(payload);
	LogToConsole('STDIN: ' + line);
	AI_ENGINE.stdin.write(line + ENDL);
	return true;
}

function SendCmd(name, params) {
	const msg = { type: 'cmd', name: name };
	if (params && typeof params === 'object') {
		Object.assign(msg, params);
	}
	return SendJson(msg);
}

function SendControl(action) {
	if (action === 'stop') STOP_SENT = true;
	return SendJson({ type: 'control', action: action });
}

function SendBoot() {
	BOOT_SENT = true;
	return SendJson({ type: 'boot', config: core.buildBootConfig() });
}

/** Chat text (+ optional attach params). Never maps user text to engine commands. */
function SendMsg(message, extra) {
	const params = Object.assign({ text: message }, extra || {});
	return SendCmd('chat', params);
}

/** Named engine command with params from core.buildCmdParams when omitted. */
function SendEngineCmd(name, params) {
	const p = (params !== undefined) ? params : core.buildCmdParams(name);
	return SendCmd(name, p);
}

function RunCommand(cmd) {
	const cmd_str = cmd.trim();
	if (cmd_str == '') return false;

	if (CMD_STATE == 'STOPPED') {

		try {
			CMD_CWD = (CMD_CWD == '') ? core.APP_CONFIG.script_dir : CMD_CWD;
			const script_file = core.APP_CONFIG.script_dir + '/cmd_control.py';
			if (fs.existsSync(script_file) && fs.existsSync(core.APP_CONFIG.python_bin)) {
				CMD_PROC = process.spawn(
					core.APP_CONFIG.python_bin, ['-u',script_file],
					{cwd:CMD_CWD, windowsHide:true}
				);
				CMD_STATE = 'STARTED';
			} else {
				throw Error('cannot find files to spawn python process');
			}
		} catch (err) {
			CMD_STATE = 'STOPPED';
			LogToConsole('Failed to run command. Check your settings.');
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

function _KillEngineProc(proc, signal) {
	if (!proc || proc.exitCode !== null) return;
	const sig = signal || 'SIGTERM';
	// Best-effort: stop loky/multiprocessing children first (common after hung generate).
	if (process.platform !== 'win32' && proc.pid) {
		try {
			const { execFileSync } = require('child_process');
			const flag = sig === 'SIGKILL' ? '-9' : '-15';
			execFileSync('pkill', [flag, '-P', String(proc.pid)], { stdio: 'ignore' });
		} catch (err) { /* no children or pkill unavailable */ }
	}
	try { proc.kill(sig); } catch (err) { /* already gone */ }
}

function StopScript(new_state='STOPPING') {
	LogToConsole('Stopping engine script ...');
	if (ENGINE_STATE == 'STARTED' || ENGINE_STATE == 'RESTART') {
		const proc = AI_ENGINE;
		ENGINE_STATE = new_state;
		if (proc) {
			_KillEngineProc(proc, 'SIGTERM');
			// CUDA / generate can ignore SIGTERM briefly - escalate so orphans do not linger.
			setTimeout(function() {
				_KillEngineProc(proc, 'SIGKILL');
			}, 2500);
		}
	} else {
		LogToConsole('Engine script already stopped');
		ENGINE_STATE = new_state;
		if (new_state == 'RESTART') return true;
	}
	return false;
}

function HandleEngineMessage(msg) {
	if (!msg || typeof msg !== 'object') return;
	const mtype = msg.type;

	if (mtype === 'hello') {
		LogToConsole('Engine hello protocol=' + (msg.protocol || msg.v));
		if (!BOOT_SENT) SendBoot();
		return;
	}

	if (mtype === 'ready' || mtype === 'idle') {
		STREAMING = false;
		STOP_SENT = false;
		ENGINE_READY = true;
		if (CB_FUNCS) CB_FUNCS.ai_ready();
		return;
	}

	if (mtype === 'stream_start') {
		STREAMING = true;
		STOP_SENT = false;
		GEN_STREAM = (msg.channel === 'gen');
		if (CB_FUNCS) CB_FUNCS.stream_start(GEN_STREAM);
		return;
	}

	if (mtype === 'stream') {
		STREAMING = true;
		GEN_STREAM = (msg.channel === 'gen');
		if (CB_FUNCS) CB_FUNCS.stream_text(msg.delta || '', GEN_STREAM);
		return;
	}

	if (mtype === 'stream_end') {
		STREAMING = false;
		STOP_SENT = false;
		if (CB_FUNCS && CB_FUNCS.stream_end) CB_FUNCS.stream_end(GEN_STREAM);
		LogToConsole('stream_end reason=' + (msg.reason || 'ok'));
		return;
	}

	if (mtype === 'error') {
		let err = 'ENGINE ERROR [' + (msg.where || '?') + ']: ' + (msg.message || '');
		if (msg.traceback) err += '\n' + msg.traceback;
		LogToConsole(err);
		return;
	}

	if (mtype === 'event') {
		HandleEngineEvent(msg);
		return;
	}

	LogToConsole('Unknown engine message type: ' + mtype);
}

function HandleEngineEvent(msg) {
	const name = msg.name;
	const data = msg.data || {};

	if (name === 'bot') {
		const text = (data.text || '').replaceAll('\n', '[AI_UI_BR]');
		if (CB_FUNCS) CB_FUNCS.bot_out(text, !!data.anim);
	} else if (name === 'turn_start') {
		if (CB_FUNCS && CB_FUNCS.turn_start) CB_FUNCS.turn_start();
	} else if (name === 'turn_patch') {
		const text = (data.text || '').replaceAll('\n', '[AI_UI_BR]');
		if (CB_FUNCS && CB_FUNCS.turn_patch) CB_FUNCS.turn_patch(text);
	} else if (name === 'turn_end') {
		if (CB_FUNCS && CB_FUNCS.turn_end) CB_FUNCS.turn_end();
	} else if (name === 'compact') {
		if (CB_FUNCS && CB_FUNCS.compact) CB_FUNCS.compact(data || {});
	} else if (name === 'context_usage') {
		if (CB_FUNCS && CB_FUNCS.context_usage) CB_FUNCS.context_usage(data || {});
	} else if (name === 'gen') {
		const text = (data.text || '').replaceAll('\n', '[AI_UI_BR]');
		if (CB_FUNCS) CB_FUNCS.gen_out(text);
	} else if (name === 'img') {
		if (CB_FUNCS) CB_FUNCS.img_out(data.path || '');
	} else if (name === 'clone') {
		if (CB_FUNCS) CB_FUNCS.clone_out(data.result || '');
	} else if (name === 'asr') {
		if (CB_FUNCS) CB_FUNCS.asr_out(data.text || '');
	} else if (name === 'tts') {
		if (CB_FUNCS) CB_FUNCS.tts_out(data.path || '');
	} else if (name === 'play_speech') {
		if (CB_FUNCS) CB_FUNCS.play_audio((data.path || '').trim());
	} else if (name === 'avatar') {
		if (CB_FUNCS) CB_FUNCS.avatar_got(data.ok ? 'true' : 'false');
	} else if (name === 'prompt') {
		if (data.kind === 'init') {
			if (CB_FUNCS) CB_FUNCS.ai_ready('AI_UI_DEFAULT');
		} else if (data.kind === 'clear') {
			const text = (data.text || '').replaceAll('\n', '[AI_UI_BR]');
			if (CB_FUNCS) CB_FUNCS.ai_ready(text);
		}
	} else if (name === 'tools') {
		if (CB_FUNCS) CB_FUNCS.got_tools((data.data || '').replaceAll('\n', '[AI_UI_BR]'));
	} else if (name === 'tool_catalog') {
		if (CB_FUNCS && CB_FUNCS.got_tool_catalog) CB_FUNCS.got_tool_catalog(data.data || '');
	} else if (name === 'voices') {
		const mode = data.mode || 'SYS';
		const do_set = (TTS_MODES[core.CHAT_CONFIG.tts_mode] == mode);
		if (do_set && CB_FUNCS) CB_FUNCS.clear_voices();
		const voice_names = [];
		const voices = data.voices || [];
		for (let i = 0; i < voices.length; i++) {
			let n = voices[i].name || '';
			if (n.endsWith('.npy') || n.endsWith('.txt') || n.endsWith('.wav')) {
				voice_names.push(n.substring(0, n.length - 4));
			} else if (n.endsWith('.pt')) {
				voice_names.push(n.substring(0, n.length - 3));
			} else {
				voice_names.push(n);
			}
		}
		if (CB_FUNCS) CB_FUNCS.add_voices({ names: voice_names, mode: mode, set: do_set });
	} else if (name === 'prompt_dump') {
		LogToConsole('PROMPT:\n' + (data.text || ''));
	} else if (name === 'prompt_files') {
		if (CB_FUNCS && CB_FUNCS.prompt_files) CB_FUNCS.prompt_files(data);
	} else if (name === 'chat_export') {
		if (EXPORT_WAIT) {
			const done = EXPORT_WAIT;
			EXPORT_WAIT = null;
			done.resolve({
				messages: data.messages || [],
				init_prompt: data.init_prompt || '',
				compaction: (data.compaction != null) ? data.compaction : null
			});
		}
	} else if (name === 'chat_imported') {
		// Renderer rebuilds from the load payload; event acknowledges engine apply.
		LogToConsole('chat_imported messages=' + ((data.messages || []).length));
	} else if (name === 'tool_permission') {
		if (typeof TOOL_PERM_HANDLER === 'function') {
			try {
				TOOL_PERM_HANDLER(data || {});
			} catch (err) {
				LogToConsole('tool_permission handler error: ' + err);
				SendCmd('tool_permission_result', {
					request_id: (data && data.request_id) || '',
					action: 'deny'
				});
			}
		} else {
			SendCmd('tool_permission_result', {
				request_id: (data && data.request_id) || '',
				action: 'deny'
			});
		}
	} else {
		LogToConsole('Unknown engine event: ' + name);
	}
}

function SetToolPermissionHandler(fn) {
	TOOL_PERM_HANDLER = typeof fn === 'function' ? fn : null;
}

function IsStreaming() {
	return STREAMING;
}

function IsReady() {
	return ENGINE_READY;
}

function ExportChat(timeoutMs) {
	if (EXPORT_WAIT) {
		return Promise.reject(new Error('export_chat already pending'));
	}
	if (!EngineRunning()) {
		return Promise.reject(new Error('engine not running'));
	}
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => {
			EXPORT_WAIT = null;
			reject(new Error('export_chat timed out'));
		}, timeoutMs || 15000);
		EXPORT_WAIT = {
			resolve: (data) => { clearTimeout(timer); resolve(data); },
			reject: (err) => { clearTimeout(timer); reject(err); }
		};
		if (!SendEngineCmd('export_chat', {})) {
			EXPORT_WAIT = null;
			clearTimeout(timer);
			reject(new Error('failed to send export_chat'));
		}
	});
}

function OnStdoutChunk(chunk) {
	STDOUT_BUF += chunk.toString();
	let idx;
	while ((idx = STDOUT_BUF.indexOf('\n')) >= 0) {
		let line = STDOUT_BUF.slice(0, idx);
		STDOUT_BUF = STDOUT_BUF.slice(idx + 1);
		if (line.endsWith('\r')) line = line.slice(0, -1);
		line = line.trim();
		if (!line) continue;
		LogToConsole('STDOUT: ' + line);
		try {
			const msg = JSON.parse(line);
			HandleEngineMessage(msg);
		} catch (err) {
			// Library progress (avatar, etc.) may still leak if guard fails; treat as log noise.
			LogToConsole('STDOUT (non-json): ' + line);
		}
	}
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

	STDOUT_BUF = '';
	BOOT_SENT = false;
	STREAMING = false;
	STOP_SENT = false;
	ENGINE_READY = false;

	try {
		const script_file = core.APP_CONFIG.script_dir + '/aiui_engine.py';
		if (fs.existsSync(script_file) && fs.existsSync(core.APP_CONFIG.python_bin)) {
			AI_ENGINE = process.spawn(
				core.APP_CONFIG.python_bin, ['-u',script_file],
				{cwd:core.APP_CONFIG.script_dir, windowsHide:true}
			);
			ENGINE_STATE = 'STARTED';
		} else {
			throw Error('cannot find files to spawn python process');
		}
	} catch (err) {
		ENGINE_STATE = 'START_FAIL';
		LogToConsole('Failed to start AI Engine');
		CB_FUNCS.ai_ended('Failed to start AI Engine. Check Console & Settings.');
		return;
	}

	AI_ENGINE.stdin.setEncoding('utf-8');

	AI_ENGINE.stdout.on('data', OnStdoutChunk);

	AI_ENGINE.stderr.on('data', (data) => {
		LogToConsole('STDERR: '+data.toString());
	});

	AI_ENGINE.on('exit', function (code, signal) {
		const how = (code !== null && code !== undefined)
			? ('code ' + code)
			: ('signal ' + (signal || 'unknown'));
		LogToConsole('AI Engine exited with ' + how);
		STDOUT_BUF = '';
		BOOT_SENT = false;
		STREAMING = false;
		ENGINE_READY = false;
		if (ENGINE_STATE == 'RESTART') {
			CB_FUNCS.ai_ended('Restarting AI Engine ...');
			setTimeout(StartScript, 100);
		} else {
			ENGINE_STATE = 'STOPPED';
			if (code != 0) {
				CB_FUNCS.ai_ended('AI Engine stopped running. Check Console & Settings.');
			} else {
				CB_FUNCS.ai_ended('AI Engine is offline. Restart it from Settings tab.');
			}
		}
	});
}

function ForceStopBot() {
	SendControl('stop');
}

module.exports = {
	startScript: StartScript,
	stopScript: StopScript,
	sendMsg: SendMsg,
	sendCmd: SendCmd,
	sendEngineCmd: SendEngineCmd,
	sendControl: SendControl,
	setPlatform: SetPlatform,
	setCallbacks: SetCallbacks,
	setToolPermissionHandler: SetToolPermissionHandler,
	engineRunning: EngineRunning,
	runCommand: RunCommand,
	forceStopBot: ForceStopBot,
	exportChat: ExportChat,
	isStreaming: IsStreaming,
	isReady: IsReady
};
