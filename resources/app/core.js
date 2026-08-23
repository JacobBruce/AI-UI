var CHAT_CONFIG = { human_name: 'Human', bot_name: 'Bot', bot_voice: 0, speech_vol: 1.0, speech_rate: 200, pitch_shift: 0, talk_mode: 0, tts_mode: 0, anim_mode: -1 };
var AI_CONFIG = {
	msg_mem: 10, max_res: 32000, max_tool_result: 4000000, min_res: 1, do_sample: 1, num_beams: 1, beam_groups: 1, base_temp: 0.8, prompt_p: 0,
	top_k: 50, top_p: 1.0, typical_p: 1.0, rep_penalty: 1.0,
	frequency_penalty: 0, presence_penalty: 0,
	compact_at: 0, compact_tail_tokens: 8000, compact_summary_max: 8192, git_status_context: 0,
	temp_format: 'none',
	enable_thinking: 'auto', reasoning_effort: 'auto',
	tools_file: 'custom', tool_packs: null, firecrawl_api_key: ''
};

/** Installed MCP servers from ~/.aiui/mcp.json (not written into config.json). */
var MCP_CONFIG = { mcpServers: {} };
var MCP_OVERLAY = [];

function DefaultNativePacks() {
	return [
		{ id: 'native:utils', kind: 'native', name: 'Utils', enabled: true, disabled_tools: [], permission_mode: 'auto' },
		{ id: 'native:math', kind: 'native', name: 'Math', enabled: true, disabled_tools: [], permission_mode: 'ask' },
		{ id: 'native:web', kind: 'native', name: 'Web', enabled: true, disabled_tools: [], permission_mode: 'ask' },
		{
			id: 'native:coding',
			kind: 'native',
			name: 'Coding',
			enabled: true,
			disabled_tools: ['run_command'],
			disabled_defaults_applied: ['run_command'],
			permission_mode: 'ask',
			sandbox: true
		},
		{ id: 'native:image', kind: 'native', name: 'Image', enabled: true, disabled_tools: [], permission_mode: 'ask' }
	];
}

function NormalizePermissionMode(raw) {
	return String(raw || 'ask').trim().toLowerCase() === 'auto' ? 'auto' : 'ask';
}

function MigrateNativePacksFromToolsFile(tools_file) {
	const packs = DefaultNativePacks();
	const mode = tools_file == 'roleplay' ? 'custom' : (tools_file || 'custom');
	const enable_math = (mode === 'math' || mode === 'merge' || mode === 'custom');
	const enable_web = (mode === 'research' || mode === 'merge' || mode === 'custom');
	for (const p of packs) {
		if (p.id === 'native:math') p.enabled = enable_math;
		if (p.id === 'native:web') p.enabled = enable_web;
	}
	return packs;
}

function NormalizeDisabledTools(raw) {
	if (!Array.isArray(raw)) return [];
	const out = [];
	const seen = new Set();
	for (const name of raw) {
		const n = String(name || '').trim();
		if (!n || seen.has(n)) continue;
		seen.add(n);
		out.push(n);
	}
	return out;
}

const CODING_OPT_IN_DISABLED = ['run_command'];

function ApplyCodingOptInDisabled(pack) {
	if (!pack || pack.id !== 'native:coding') return pack;
	const disabled = Array.isArray(pack.disabled_tools) ? pack.disabled_tools.slice() : [];
	const applied = Array.isArray(pack.disabled_defaults_applied)
		? pack.disabled_defaults_applied.map((x) => String(x || '').trim()).filter(Boolean)
		: [];
	for (const name of CODING_OPT_IN_DISABLED) {
		if (applied.includes(name)) continue;
		if (!disabled.includes(name) && !disabled.includes('aiui_' + name)) {
			disabled.push(name);
		}
		applied.push(name);
	}
	pack.disabled_tools = NormalizeDisabledTools(disabled);
	pack.disabled_defaults_applied = applied;
	pack.name = 'Coding';
	return pack;
}

function NormalizeNativePacks(packs, tools_file) {
	const defaults = MigrateNativePacksFromToolsFile(tools_file);
	const byId = {};
	if (Array.isArray(packs)) {
		for (const p of packs) {
			if (!p || p.kind === 'mcp') continue;
			const id = String(p.id || '');
			if (!id.startsWith('native:')) continue;
			byId[id] = {
				enabled: !!p.enabled,
				disabled_tools: NormalizeDisabledTools(p.disabled_tools || p.disabledTools),
				permission_mode: NormalizePermissionMode(p.permission_mode || p.permissionMode),
				sandbox: p.sandbox !== false,
				disabled_defaults_applied: Array.isArray(p.disabled_defaults_applied)
					? p.disabled_defaults_applied.slice()
					: []
			};
		}
	}
	return defaults.map((p) => {
		const prev = byId[p.id];
		const out = Object.assign({}, p, {
			enabled: prev ? prev.enabled : p.enabled,
			disabled_tools: prev ? prev.disabled_tools : (p.disabled_tools || []),
			permission_mode: prev ? prev.permission_mode : NormalizePermissionMode(p.permission_mode),
			disabled_defaults_applied: prev
				? (prev.disabled_defaults_applied || [])
				: (p.disabled_defaults_applied || [])
		});
		if (p.id === 'native:coding') {
			out.sandbox = prev ? prev.sandbox !== false : true;
			ApplyCodingOptInDisabled(out);
		}
		return out;
	});
}

function SetMcpConfig(cfg) {
	MCP_CONFIG = cfg && cfg.mcpServers ? cfg : { mcpServers: {} };
}

function GetMcpConfig() {
	return MCP_CONFIG;
}

function GetMcpOverlay() {
	return MCP_OVERLAY.slice();
}

function SetMcpOverlay(overlay) {
	MCP_OVERLAY = Array.isArray(overlay) ? overlay.slice() : [];
}

function ApplyMcpOverlayFromDisk(aiuiHome) {
	const mcpStore = require('./mcp_store.js');
	const home = String(aiuiHome || '');
	if (!home) return;
	const disk = mcpStore.LoadMcpConfig(home);
	MCP_CONFIG = mcpStore.ApplyMcpOverlay(disk, MCP_OVERLAY);
}

function BuildMergedToolPacks() {
	const mcpStore = require('./mcp_store.js');
	const native = NormalizeNativePacks(AI_CONFIG.tool_packs, AI_CONFIG.tools_file);
	const mcp = mcpStore.ServersToPacks(MCP_CONFIG);
	return native.concat(mcp);
}
var APP_CONFIG = {
	avatar_img: '', script_dir: '', python_bin: '', model_dir: '', sd_model: '', tts_model: '', sr_model: '', model_args: '', model_type: 0, imodel_type: 0, smodel_type: 0, comp_dev: 'auto', start_meth: 'none',
	format_mode: 0, gen_mode: 0, enable_tooluse: 1, enable_devmode: 0, enable_asasro: 0, start_rec_keys: '', stop_rec_keys: '', hf_token: '',
	ui_theme: 'modern',
	text_backend: 'local', remote_base_url: '', remote_model: '', remote_api_key: '', remote_provider: '',
	remote_tool_format: 'openai',
	remote_prompt_cache: 1,
	mm_processor: 'auto',
	workspace: '', agent: '', enabled_rules: null, enabled_skills: null,
	last_session: '', last_project: '', recent_sessions: [], recent_projects: []
};
var GEN_CONFIG = {
	prompt_text: '', prompt_neg: '', image_seed: '', image_width: 'auto', image_height: 'auto', inference_steps: 50, guidance_scale: 7.5, safety_check: 1, vae_file: '', lora_file: '', lora_dir: '', lora_scale: 1.0,
	max_len: 1000, min_len: 1, do_sample: 1, num_beams: 1, beam_groups: 1, temp: 0.8, top_k: 50, top_p: 1.0, typical_p: 1.0, rep_penalty: 1.0,
	tts_voice: 0, tts_vol: 1.0, tts_rate: 200, tts_pitch: 0, tts_mode: 0
};

var CLONE_MODEL = '';
var CLONE_SAMPLE = '';
var CLONE_NAME = '';
var CLONE_TEXT = '';
var INIT_PROMPT = '';
var TTS_TEXT = '';
var CHAT_FILES = [];

function GetConfigs() {
	const mcpStore = require('./mcp_store.js');
	const native = NormalizeNativePacks(AI_CONFIG.tool_packs, AI_CONFIG.tools_file);
	const mcpOverlay = mcpStore.ExtractMcpOverlay(mcpStore.ServersToPacks(MCP_CONFIG));
	const ai = Object.assign({}, AI_CONFIG, {
		tool_packs: native.concat(mcpOverlay)
	});
	return { chat: CHAT_CONFIG, app: APP_CONFIG, ai: ai, gen: GEN_CONFIG };
}

function SetConfigs(configs) {
	if (!configs) return;
	for (let prop in CHAT_CONFIG) {
		if (configs.chat && configs.chat.hasOwnProperty(prop))
			CHAT_CONFIG[prop] = configs.chat[prop];
	}
	if (configs.app) {
		for (let prop in APP_CONFIG) {
			if (configs.app.hasOwnProperty(prop))
				APP_CONFIG[prop] = configs.app[prop];
		}
	}
	for (let prop in AI_CONFIG) {
		if (configs.ai && configs.ai.hasOwnProperty(prop))
			AI_CONFIG[prop] = configs.ai[prop];
	}
	if (configs.ai && Array.isArray(configs.ai.tool_packs)) {
		const mcpStore = require('./mcp_store.js');
		MCP_OVERLAY = mcpStore.ExtractMcpOverlay(configs.ai.tool_packs);
	} else {
		MCP_OVERLAY = [];
	}
	if (AI_CONFIG.tools_file == 'roleplay') AI_CONFIG.tools_file = 'custom';
	AI_CONFIG.tool_packs = NormalizeNativePacks(AI_CONFIG.tool_packs, AI_CONFIG.tools_file);
	if (!AI_CONFIG.firecrawl_api_key) AI_CONFIG.firecrawl_api_key = '';
	AI_CONFIG.reasoning_effort = NormReasoningEffort(AI_CONFIG.reasoning_effort);
	AI_CONFIG.git_status_context = Number(AI_CONFIG.git_status_context) ? 1 : 0;
	const mtr = Number(AI_CONFIG.max_tool_result);
	AI_CONFIG.max_tool_result = (Number.isFinite(mtr) && mtr >= 1) ? Math.floor(mtr) : 4000000;
	APP_CONFIG.remote_tool_format = NormRemoteToolFormat(APP_CONFIG.remote_tool_format);
	APP_CONFIG.text_backend = (String(APP_CONFIG.text_backend || 'local') === 'remote') ? 'remote' : 'local';
	if (APP_CONFIG.remote_prompt_cache == null || APP_CONFIG.remote_prompt_cache === '') {
		APP_CONFIG.remote_prompt_cache = 1;
	} else {
		APP_CONFIG.remote_prompt_cache = (Number(APP_CONFIG.remote_prompt_cache) === 0) ? 0 : 1;
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
	CHAT_CONFIG.talk_mode = Math.max(0, Math.min(2, parseInt(new_mode, 10) || 0));
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

function SetPrompt(new_prompt) {
	INIT_PROMPT = new_prompt;
}

function SetReadText(tts_txt) {
	TTS_TEXT = tts_txt;
}

function SetChatFiles(chat_files) {
	CHAT_FILES = chat_files;
}

function SetBotVoice(voice_id) {
	CHAT_CONFIG.bot_voice = voice_id;
}

function ConfigSpeech(speech_config) {
	CHAT_CONFIG.bot_voice = speech_config.voice;
	CHAT_CONFIG.speech_vol = speech_config.vol;
	CHAT_CONFIG.speech_rate = speech_config.rate;
	CHAT_CONFIG.pitch_shift = speech_config.pitch;
	CHAT_CONFIG.tts_mode = speech_config.engine;
	CHAT_CONFIG.anim_mode = speech_config.amode;
}

function NormTempFormat(raw) {
	const t = String(raw || 'none').trim().toLowerCase();
	if (t === 'tool') return 'aiui';
	if (t === 'chat') return 'model';
	if (t === 'none' || t === 'model' || t === 'aiui' || t === 'aiui_notoolprompt') return t;
	return 'none';
}

function NormRemoteToolFormat(raw) {
	const t = String(raw || 'openai').toLowerCase();
	if (t === 'anthropic') return 'anthropic';
	return 'openai';
}

function NormReasoningEffort(raw) {
	const t = String(raw || 'auto').toLowerCase();
	if (t === 'none') return 'auto';
	if (t === 'minimal') return 'low';
	if (t === 'auto' || t === 'low' || t === 'medium' || t === 'high' || t === 'xhigh' || t === 'max') return t;
	return 'auto';
}

function ConfigTools(tools_config) {
	if (tools_config.hasOwnProperty('temp_format')) AI_CONFIG.temp_format = NormTempFormat(tools_config.temp_format);
	if (tools_config.hasOwnProperty('tools_file')) {
		AI_CONFIG.tools_file = tools_config.tools_file == 'roleplay' ? 'custom' : tools_config.tools_file;
	}
	if (tools_config.hasOwnProperty('tool_packs')) {
		// Accept merged packs from UI; persist natives only. MCP enables update MCP_CONFIG separately.
		AI_CONFIG.tool_packs = NormalizeNativePacks(tools_config.tool_packs, AI_CONFIG.tools_file);
	}
	if (tools_config.hasOwnProperty('native_packs')) {
		AI_CONFIG.tool_packs = NormalizeNativePacks(tools_config.native_packs, AI_CONFIG.tools_file);
	}
	if (tools_config.hasOwnProperty('firecrawl_api_key')) {
		AI_CONFIG.firecrawl_api_key = tools_config.firecrawl_api_key || '';
	}
}

function ConfigAI(ai_config) {
	AI_CONFIG.msg_mem = ai_config.max_mmem;
	AI_CONFIG.max_res = ai_config.max_rlen;
	AI_CONFIG.min_res = ai_config.min_rlen;
	AI_CONFIG.do_sample = ai_config.do_sample;
	AI_CONFIG.num_beams = ai_config.num_beams;
	AI_CONFIG.beam_groups = ai_config.beam_groups;
	AI_CONFIG.base_temp = ai_config.temp;
	AI_CONFIG.prompt_p = ai_config.pp;
	AI_CONFIG.top_k = ai_config.tk;
	AI_CONFIG.top_p = ai_config.tp;
	AI_CONFIG.typical_p = ai_config.typ;
	AI_CONFIG.rep_penalty = ai_config.rp;
	if (ai_config.hasOwnProperty('frequency_penalty')) AI_CONFIG.frequency_penalty = Number(ai_config.frequency_penalty);
	if (ai_config.hasOwnProperty('fp')) AI_CONFIG.frequency_penalty = Number(ai_config.fp);
	if (ai_config.hasOwnProperty('presence_penalty')) AI_CONFIG.presence_penalty = Number(ai_config.presence_penalty);
	if (ai_config.hasOwnProperty('prp')) AI_CONFIG.presence_penalty = Number(ai_config.prp);
	if (ai_config.hasOwnProperty('compact_at')) AI_CONFIG.compact_at = Number(ai_config.compact_at);
	if (ai_config.hasOwnProperty('compact_tail_tokens')) AI_CONFIG.compact_tail_tokens = Number(ai_config.compact_tail_tokens);
	if (ai_config.hasOwnProperty('compact_summary_max')) AI_CONFIG.compact_summary_max = Number(ai_config.compact_summary_max);
	if (ai_config.hasOwnProperty('git_status_context')) AI_CONFIG.git_status_context = Number(ai_config.git_status_context) ? 1 : 0;
	if (ai_config.hasOwnProperty('temp_format')) AI_CONFIG.temp_format = NormTempFormat(ai_config.temp_format);
	if (ai_config.hasOwnProperty('enable_thinking')) AI_CONFIG.enable_thinking = String(ai_config.enable_thinking || 'auto');
	if (ai_config.hasOwnProperty('reasoning_effort')) AI_CONFIG.reasoning_effort = NormReasoningEffort(ai_config.reasoning_effort);
	if (ai_config.hasOwnProperty('max_tool_result')) {
		const n = Number(ai_config.max_tool_result);
		AI_CONFIG.max_tool_result = (Number.isFinite(n) && n >= 1) ? Math.floor(n) : 4000000;
	}
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
	if (app_config.hasOwnProperty('workspace')) APP_CONFIG.workspace = app_config.workspace || '';
	if (app_config.hasOwnProperty('mm_processor')) SetMmProcessor(app_config.mm_processor);
}

function SetMmProcessor(raw) {
	const mp = String(raw || 'auto').toLowerCase();
	APP_CONFIG.mm_processor = (mp === 'qwen_vl' || mp === 'janus') ? mp : 'auto';
}

function ConfigOther(app_config, app_other=false) {
	if (app_other) {
		APP_CONFIG.gen_mode = app_config.gen_mode;
		APP_CONFIG.format_mode = app_config.format_mode;
		APP_CONFIG.enable_tooluse = app_config.enable_tooluse;
		if (app_config.hasOwnProperty('workspace')) APP_CONFIG.workspace = app_config.workspace;
	} else {
		APP_CONFIG.enable_devmode = app_config.enable_devmode;
		APP_CONFIG.enable_asasro = app_config.enable_asasro;
		APP_CONFIG.start_rec_keys = app_config.start_rec_keys;
		APP_CONFIG.stop_rec_keys = app_config.stop_rec_keys;
		if (app_config.hasOwnProperty('ui_theme')) {
			APP_CONFIG.ui_theme = String(app_config.ui_theme || 'modern');
		}
	}
}

function ConfigRemote(app_config) {
	if (app_config.hasOwnProperty('text_backend')) {
		APP_CONFIG.text_backend = (String(app_config.text_backend || 'local') === 'remote') ? 'remote' : 'local';
	}
	if (app_config.hasOwnProperty('remote_base_url')) APP_CONFIG.remote_base_url = String(app_config.remote_base_url || '');
	if (app_config.hasOwnProperty('remote_model')) APP_CONFIG.remote_model = String(app_config.remote_model || '');
	if (app_config.hasOwnProperty('remote_api_key')) APP_CONFIG.remote_api_key = String(app_config.remote_api_key || '');
	if (app_config.hasOwnProperty('remote_provider')) APP_CONFIG.remote_provider = String(app_config.remote_provider || '');
	if (app_config.hasOwnProperty('remote_tool_format')) {
		APP_CONFIG.remote_tool_format = NormRemoteToolFormat(app_config.remote_tool_format);
	}
	if (app_config.hasOwnProperty('remote_prompt_cache')) {
		APP_CONFIG.remote_prompt_cache = (Number(app_config.remote_prompt_cache) === 0) ? 0 : 1;
	}
	if (app_config.hasOwnProperty('hf_token')) APP_CONFIG.hf_token = String(app_config.hf_token || '');
}

function ConfigPromptFiles(pf) {
	if (pf.hasOwnProperty('agent')) APP_CONFIG.agent = pf.agent || '';
	if (pf.hasOwnProperty('enabled_rules')) APP_CONFIG.enabled_rules = pf.enabled_rules;
	if (pf.hasOwnProperty('enabled_skills')) APP_CONFIG.enabled_skills = pf.enabled_skills;
	if (pf.hasOwnProperty('workspace')) APP_CONFIG.workspace = pf.workspace || '';
}

function ConfigGen(gen_config) {
	if (gen_config.hasOwnProperty('img_prompt')) {
		GEN_CONFIG.prompt_text = gen_config.img_prompt;
		GEN_CONFIG.prompt_neg = gen_config.neg_prompt;
		GEN_CONFIG.guidance_scale = gen_config.guidance;
		GEN_CONFIG.inference_steps = gen_config.steps;
		GEN_CONFIG.image_seed = gen_config.seed;
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
		GEN_CONFIG.do_sample = gen_config.do_sp;
		GEN_CONFIG.num_beams = gen_config.num_b;
		GEN_CONFIG.beam_groups = gen_config.num_g;
		GEN_CONFIG.temp = gen_config.temp;
		GEN_CONFIG.top_k = gen_config.top_k;
		GEN_CONFIG.top_p = gen_config.top_p;
		GEN_CONFIG.typical_p = gen_config.typ_p;
		GEN_CONFIG.rep_penalty = gen_config.rep_p;
	}
}

/** Single boot payload for NDJSON engine IPC (replaces sequential stdin fields). */
function BuildBootConfig() {
	const { app } = require('electron');
	const path = require('path');
	const aiui_home = path.join(app.getPath('home'), '.aiui');
	return {
		app: {
			script_dir: APP_CONFIG.script_dir,
			model_dir: APP_CONFIG.model_dir,
			sd_model: APP_CONFIG.sd_model,
			tts_model: APP_CONFIG.tts_model,
			sr_model: APP_CONFIG.sr_model,
			model_type: APP_CONFIG.model_type,
			imodel_type: APP_CONFIG.imodel_type,
			smodel_type: APP_CONFIG.smodel_type,
			model_args: APP_CONFIG.model_args,
			comp_dev: APP_CONFIG.comp_dev,
			start_meth: APP_CONFIG.start_meth,
			avatar_img: APP_CONFIG.avatar_img,
			gen_mode: APP_CONFIG.gen_mode,
			format_mode: APP_CONFIG.format_mode,
			enable_tooluse: APP_CONFIG.enable_tooluse,
			hf_token: APP_CONFIG.hf_token,
			text_backend: APP_CONFIG.text_backend || 'local',
			remote_base_url: APP_CONFIG.remote_base_url || '',
			remote_model: APP_CONFIG.remote_model || '',
			remote_api_key: APP_CONFIG.remote_api_key || '',
			remote_provider: APP_CONFIG.remote_provider || '',
			remote_tool_format: NormRemoteToolFormat(APP_CONFIG.remote_tool_format),
			remote_prompt_cache: (Number(APP_CONFIG.remote_prompt_cache) === 0) ? 0 : 1,
			mm_processor: APP_CONFIG.mm_processor || 'auto',
			aiui_home: aiui_home,
			app_version: app.getVersion(),
			workspace: APP_CONFIG.workspace || '',
			agent: APP_CONFIG.agent || '',
			enabled_rules: APP_CONFIG.enabled_rules,
			enabled_skills: APP_CONFIG.enabled_skills
		},
		tools: {
			temp_form: AI_CONFIG.temp_format,
			tool_mode: AI_CONFIG.tools_file,
			tool_packs: BuildMergedToolPacks(),
			firecrawl_api_key: AI_CONFIG.firecrawl_api_key || ''
		},
		ai: {
			msg_mem: AI_CONFIG.msg_mem,
			gen_max: AI_CONFIG.max_res,
			max_tool_result: AI_CONFIG.max_tool_result ?? 4000000,
			gen_min: AI_CONFIG.min_res,
			do_sample: AI_CONFIG.do_sample,
			num_beams: AI_CONFIG.num_beams,
			beam_groups: AI_CONFIG.beam_groups,
			base_temp: AI_CONFIG.base_temp,
			prompt_p: AI_CONFIG.prompt_p,
			top_k: AI_CONFIG.top_k,
			top_p: AI_CONFIG.top_p,
			typical_p: AI_CONFIG.typical_p,
			rep_penalty: AI_CONFIG.rep_penalty,
			frequency_penalty: AI_CONFIG.frequency_penalty ?? 0,
			presence_penalty: AI_CONFIG.presence_penalty ?? 0,
			compact_at: AI_CONFIG.compact_at ?? 0,
			compact_tail_tokens: AI_CONFIG.compact_tail_tokens ?? 8000,
			compact_summary_max: AI_CONFIG.compact_summary_max ?? 8192,
			git_status_context: AI_CONFIG.git_status_context ? 1 : 0,
			enable_thinking: AI_CONFIG.enable_thinking || 'auto',
			reasoning_effort: NormReasoningEffort(AI_CONFIG.reasoning_effort)
		},
		voice: {
			bot_voice: CHAT_CONFIG.bot_voice,
			speech_vol: CHAT_CONFIG.speech_vol,
			speech_rate: CHAT_CONFIG.speech_rate,
			pitch_shift: CHAT_CONFIG.pitch_shift,
			talk_mode: CHAT_CONFIG.talk_mode,
			tts_mode: CHAT_CONFIG.tts_mode,
			anim_mode: CHAT_CONFIG.anim_mode
		},
		chat: {
			human_name: CHAT_CONFIG.human_name,
			bot_name: CHAT_CONFIG.bot_name
		}
	};
}

/** Params for cmds that previously relied on follow-up stdin after a banner. */
function BuildCmdParams(name) {
	switch (name) {
		case 'update_prompt':
			return { prompt: INIT_PROMPT };
		case 'compose_prompt':
			return {
				agent: APP_CONFIG.agent || '',
				enabled_rules: APP_CONFIG.enabled_rules,
				enabled_skills: APP_CONFIG.enabled_skills,
				workspace: APP_CONFIG.workspace || ''
			};
		case 'attach_files':
			return { files: CHAT_FILES.slice(), mm_processor: APP_CONFIG.mm_processor || 'auto' };
		case 'config_voice':
			return {
				bot_voice: CHAT_CONFIG.bot_voice,
				speech_vol: CHAT_CONFIG.speech_vol,
				speech_rate: CHAT_CONFIG.speech_rate,
				pitch_shift: CHAT_CONFIG.pitch_shift,
				talk_mode: CHAT_CONFIG.talk_mode,
				tts_mode: CHAT_CONFIG.tts_mode,
				anim_mode: CHAT_CONFIG.anim_mode,
				avatar_img: APP_CONFIG.avatar_img
			};
		case 'config_ai':
			return {
				msg_mem: AI_CONFIG.msg_mem,
				max_res: AI_CONFIG.max_res,
				max_tool_result: AI_CONFIG.max_tool_result ?? 4000000,
				min_res: AI_CONFIG.min_res,
				do_sample: AI_CONFIG.do_sample,
				num_beams: AI_CONFIG.num_beams,
				beam_groups: AI_CONFIG.beam_groups,
				base_temp: AI_CONFIG.base_temp,
				prompt_p: AI_CONFIG.prompt_p,
				top_k: AI_CONFIG.top_k,
				top_p: AI_CONFIG.top_p,
				typical_p: AI_CONFIG.typical_p,
				rep_penalty: AI_CONFIG.rep_penalty,
				frequency_penalty: AI_CONFIG.frequency_penalty ?? 0,
				presence_penalty: AI_CONFIG.presence_penalty ?? 0,
				compact_at: AI_CONFIG.compact_at ?? 0,
				compact_tail_tokens: AI_CONFIG.compact_tail_tokens ?? 8000,
				compact_summary_max: AI_CONFIG.compact_summary_max ?? 8192,
				git_status_context: AI_CONFIG.git_status_context ? 1 : 0,
				temp_format: AI_CONFIG.temp_format,
				enable_thinking: AI_CONFIG.enable_thinking || 'auto',
				reasoning_effort: NormReasoningEffort(AI_CONFIG.reasoning_effort)
			};
		case 'config_other':
			return {
				gen_mode: APP_CONFIG.gen_mode,
				format_mode: APP_CONFIG.format_mode,
				enable_tooluse: APP_CONFIG.enable_tooluse
			};
		case 'config_remote':
			return {
				text_backend: APP_CONFIG.text_backend || 'local',
				remote_base_url: APP_CONFIG.remote_base_url || '',
				remote_model: APP_CONFIG.remote_model || '',
				remote_api_key: APP_CONFIG.remote_api_key || '',
				remote_provider: APP_CONFIG.remote_provider || '',
				remote_tool_format: NormRemoteToolFormat(APP_CONFIG.remote_tool_format),
				remote_prompt_cache: (Number(APP_CONFIG.remote_prompt_cache) === 0) ? 0 : 1,
				hf_token: APP_CONFIG.hf_token
			};
		case 'config_tools':
			return {
				temp_format: AI_CONFIG.temp_format,
				tools_file: AI_CONFIG.tools_file,
				tool_packs: BuildMergedToolPacks(),
				firecrawl_api_key: AI_CONFIG.firecrawl_api_key || '',
				workspace: APP_CONFIG.workspace || ''
			};
		case 'gen_image':
			return {
				prompt_text: GEN_CONFIG.prompt_text,
				prompt_neg: GEN_CONFIG.prompt_neg,
				inference_steps: GEN_CONFIG.inference_steps,
				guidance_scale: GEN_CONFIG.guidance_scale,
				safety_check: GEN_CONFIG.safety_check,
				image_seed: GEN_CONFIG.image_seed,
				image_width: GEN_CONFIG.image_width,
				image_height: GEN_CONFIG.image_height,
				vae_file: GEN_CONFIG.vae_file,
				lora_file: GEN_CONFIG.lora_file,
				lora_dir: GEN_CONFIG.lora_dir,
				lora_scale: GEN_CONFIG.lora_scale
			};
		case 'gen_text':
			return {
				prompt_text: GEN_CONFIG.prompt_text,
				max_len: GEN_CONFIG.max_len,
				min_len: GEN_CONFIG.min_len,
				do_sample: GEN_CONFIG.do_sample,
				num_beams: GEN_CONFIG.num_beams,
				beam_groups: GEN_CONFIG.beam_groups,
				temp: GEN_CONFIG.temp,
				top_k: GEN_CONFIG.top_k,
				top_p: GEN_CONFIG.top_p,
				typical_p: GEN_CONFIG.typical_p,
				rep_penalty: GEN_CONFIG.rep_penalty
			};
		case 'gen_speech':
			return {
				text: TTS_TEXT,
				tts_voice: GEN_CONFIG.tts_voice,
				tts_vol: GEN_CONFIG.tts_vol,
				tts_rate: GEN_CONFIG.tts_rate,
				tts_pitch: GEN_CONFIG.tts_pitch,
				tts_mode: GEN_CONFIG.tts_mode
			};
		case 'read_text':
			return { text: TTS_TEXT };
		case 'clone_voice':
			return {
				model: CLONE_MODEL,
				name: CLONE_NAME,
				sample: CLONE_SAMPLE,
				transcript: CLONE_TEXT
			};
		case 'update_avatar':
			return { avatar_img: APP_CONFIG.avatar_img };
		case 'update_tmode':
			return { talk_mode: CHAT_CONFIG.talk_mode };
		case 'update_users':
			return {
				human_name: CHAT_CONFIG.human_name,
				bot_name: CHAT_CONFIG.bot_name
			};
		case 'import_chat':
			return { messages: [] };
		default:
			return {};
	}
}

module.exports = {
	getConfigs: GetConfigs,
	setConfigs: SetConfigs,
	getMcpConfig: GetMcpConfig,
	setMcpConfig: SetMcpConfig,
	getMcpOverlay: GetMcpOverlay,
	setMcpOverlay: SetMcpOverlay,
	applyMcpOverlayFromDisk: ApplyMcpOverlayFromDisk,
	buildMergedToolPacks: BuildMergedToolPacks,
	normalizeNativePacks: NormalizeNativePacks,
	initAppConfig: InitAppConfig,
	setAvatar: SetAvatar,
	setTalkMode: SetTalkMode,
	setUsernames: SetUsernames,
	setCloneVoice: SetCloneVoice,
	setPrompt: SetPrompt,
	setReadText: SetReadText,
	setChatFiles: SetChatFiles,
	setBotVoice: SetBotVoice,
	configSpeech: ConfigSpeech,
	configTools: ConfigTools,
	configAI: ConfigAI,
	configApp: ConfigApp,
	setMmProcessor: SetMmProcessor,
	configOther: ConfigOther,
	configRemote: ConfigRemote,
	configPromptFiles: ConfigPromptFiles,
	configGen: ConfigGen,
	buildBootConfig: BuildBootConfig,
	buildCmdParams: BuildCmdParams,
	get CHAT_CONFIG() { return CHAT_CONFIG; },
	get APP_CONFIG() { return APP_CONFIG; },
	get AI_CONFIG() { return AI_CONFIG; },
	get GEN_CONFIG() { return GEN_CONFIG; },
	get MCP_CONFIG() { return MCP_CONFIG; },
	get CLONE_MODEL() { return CLONE_MODEL; },
	get CLONE_SAMPLE() { return CLONE_SAMPLE; },
	get CLONE_NAME() { return CLONE_NAME; },
	get CLONE_TEXT() { return CLONE_TEXT; },
	get INIT_PROMPT() { return INIT_PROMPT; },
	get TTS_TEXT() { return TTS_TEXT; },
	get CHAT_FILES() { return CHAT_FILES; }
};
