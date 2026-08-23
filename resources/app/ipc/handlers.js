const fs = require('fs');
const path = require('path');
const { ipcMain, dialog, clipboard, shell, Menu, MenuItem, nativeImage } = require('electron');
const { PrepareAvatarImage, EnsureAvatarForCore } = require('../avatar_cache.js');

function createBridgeHelpers(getWin) {
	function send(channel, payload) {
		const w = getWin();
		if (w && !w.isDestroyed()) w.webContents.send(channel, payload);
	}

	function ChatAIReady(init_prompt='') {
		if (init_prompt != '') {
			send('init-ui', { state: init_prompt });
		} else {
			send('ai-ready');
		}
	}

	function ChatAIEnded(p_msg) {
		send('init-ui', { state: false });
		send('prompt-msg', { msg: p_msg });
	}

	function ShowBotMsg(message, anim_done=true) {
		send('bot-msg', { msg: message, got_vid: anim_done });
	}

	function ShowGenTxt(gen_txt) {
		send('gen-result', { txt: gen_txt });
	}

	function ShowGenTTS(gen_tts) {
		send('tts-result', { wav: gen_tts });
	}

	function ShowGenASR(gen_asr) {
		send('asr-result', { txt: gen_asr });
	}

	function ShowGenImg(gen_img) {
		send('img-result', { img: gen_img });
	}

	function ShowClonedVoice(gen_voice) {
		send('clone-result', { voice: gen_voice });
	}

	function AppendLog(log_msg) {
		send('append-log', { msg: log_msg });
	}

	function PlayAudio(audio_file) {
		send('play-audio', { file: audio_file });
	}

	function GotAvatar(got_avatar) {
		send('got-avatar', { got: got_avatar });
	}

	function GotTools(tool_funcs) {
		send('got-tools', { tools: tool_funcs });
	}

	function GotToolCatalog(raw) {
		let tools = {};
		let mcp_running = [];
		let mcp_errors = {};
		try {
			const parsed = raw ? JSON.parse(raw) : {};
			if (parsed && typeof parsed === 'object' && parsed.tools && typeof parsed.tools === 'object') {
				tools = parsed.tools;
				mcp_running = Array.isArray(parsed.mcp_running) ? parsed.mcp_running : [];
				mcp_errors = (parsed.mcp_errors && typeof parsed.mcp_errors === 'object') ? parsed.mcp_errors : {};
			} else if (parsed && typeof parsed === 'object') {
				// Legacy: catalog was a bare pack_id -> tools map.
				tools = parsed;
			}
		} catch (err) {
			tools = {};
			mcp_running = [];
			mcp_errors = {};
		}
		send('got-tool-catalog', { catalog: tools, mcp_running, mcp_errors });
	}

	function StreamText(stream_txt, txtgen_stream) {
		if (txtgen_stream) {
			send('txtgen-stream-text', { txt: stream_txt });
		} else {
			send('stream-text', { txt: stream_txt });
		}
	}

	function StreamStart(txtgen_stream) {
		if (txtgen_stream) {
			send('txtgen-stream-start');
		} else {
			send('stream-start');
		}
	}

	function StreamEnd(txtgen_stream) {
		if (txtgen_stream) {
			send('txtgen-stream-end');
		} else {
			send('stream-end');
		}
	}

	function TurnStart() {
		send('turn-start');
	}

	function TurnPatch(text) {
		send('turn-patch', { text: text || '' });
	}

	function TurnEnd() {
		send('turn-end');
	}

	function ContextCompact(data) {
		send('context-compact', data || {});
	}

	function ContextUsage(data) {
		send('context-usage', data || {});
	}

	function AddVoices(payload) {
		send('add-voices', payload);
	}

	function ClearVoices() {
		send('clear-voices');
	}

	function PromptFiles(data) {
		send('prompt-files', data || {});
	}

	return {
		ChatAIReady,
		ChatAIEnded,
		ShowBotMsg,
		ShowGenTxt,
		ShowGenTTS,
		ShowGenASR,
		ShowGenImg,
		ShowClonedVoice,
		AppendLog,
		PlayAudio,
		GotAvatar,
		GotTools,
		GotToolCatalog,
		StreamText,
		StreamStart,
		StreamEnd,
		TurnStart,
		TurnPatch,
		TurnEnd,
		ContextCompact,
		ContextUsage,
		AddVoices,
		ClearVoices,
		PromptFiles
	};
}

function CreateHandlers({ getWin, core, engine, dialogs, checkModelsExist, getConfigPath, sessionOps, sessions, rebuildMenu, saveGlobalConfig }) {
	const mcpStore = require('../mcp_store.js');

	function LoadConfigPayload(skip_inputs) {
		return {
			configs: core.getConfigs(),
			tool_packs: core.buildMergedToolPacks(),
			skip_inputs: !!skip_inputs
		};
	}

	function PersistMcpEnablesFromPacks(packs) {
		if (!Array.isArray(packs)) return;
		const overlay = mcpStore.ExtractMcpOverlay(packs);
		core.setMcpOverlay(overlay);
		core.setMcpConfig(mcpStore.ApplyMcpOverlay(core.getMcpConfig(), overlay));
	}

	function PersistPackAutoAllow(packId) {
		const home = sessions ? sessions.GetAiuiHome() : '';
		if (!home || !packId) return;
		const id = String(packId);
		if (id.startsWith('mcp:')) {
			let mcp = core.getMcpConfig();
			mcp = mcpStore.UpsertPack(mcp, {
				server_id: id.replace(/^mcp:/, ''),
				permission_mode: 'auto'
			});
			core.setMcpConfig(mcp);
		} else {
			const packs = core.buildMergedToolPacks().map((p) => {
				if (p.id === id) return Object.assign({}, p, { permission_mode: 'auto' });
				return p;
			});
			core.configTools({ tool_packs: packs });
		}
		PersistScopedConfig();
		try {
			getWin().webContents.send('load-config', LoadConfigPayload(true));
		} catch (err) { /* window may be gone */ }
	}

	function HandleToolPermission(data) {
		data = data || {};
		try {
			getWin().webContents.send('tool-permission', data);
		} catch (err) {
			engine.sendEngineCmd('tool_permission_result', {
				request_id: data.request_id || '',
				action: 'deny'
			});
		}
	}

	function ResolveToolPermissionResult(payload) {
		payload = payload || {};
		const action = String(payload.action || 'deny').trim().toLowerCase();
		const requestId = payload.request_id || '';
		const packId = payload.pack_id || '';
		if (action === 'auto') PersistPackAutoAllow(packId);
		engine.sendEngineCmd('tool_permission_result', {
			request_id: requestId,
			action: action === 'auto' ? 'auto' : (action === 'allow' ? 'allow' : 'deny')
		});
	}

	function PersistMcpServer(pack) {
		const home = sessions ? sessions.GetAiuiHome() : '';
		if (!home || !pack) return [];
		let mcp = mcpStore.UpsertPack(core.getMcpConfig(), pack);
		const upsertWarn = mcpStore.TakeWarnings(mcp);
		mcp = mcpStore.SaveMcpConfig(home, mcp);
		const saveWarn = mcpStore.TakeWarnings(mcp);
		core.setMcpConfig(mcp);
		return upsertWarn.concat(saveWarn);
	}
	const bridge = createBridgeHelpers(getWin);
	var sessionRestored = false;
	var sessionScopePrepared = false;
	var pendingEngineResume = null;
	var suppressBootPrompt = false;

	async function SnapshotForEngineRestart() {
		let snap = { messages: [], init_prompt: '' };
		try {
			if (engine.engineRunning && engine.engineRunning()) {
				snap = await engine.exportChat();
			}
		} catch (err) { /* use INIT_PROMPT below */ }
		if (!snap.init_prompt && core.INIT_PROMPT) {
			snap.init_prompt = String(core.INIT_PROMPT).replaceAll('[AI_UI_BR]', '\n');
		}
		return snap;
	}

	async function RestartEngine() {
		pendingEngineResume = await SnapshotForEngineRestart();
		suppressBootPrompt = true;
		bridge.ChatAIEnded('Initializing AI Engine... please wait.');
		if (engine.stopScript('RESTART')) engine.startScript();
	}

	const origChatAIReady = bridge.ChatAIReady;
	const origChatAIEnded = bridge.ChatAIEnded;
	bridge.ChatAIReady = function(init_prompt) {
		// Engine process restart: re-apply chat/prompt. Do not run RestoreOnStartup again
		// (that was flipping Home back to last_project, e.g. AI-UI).
		if (pendingEngineResume && sessionOps) {
			const snap = pendingEngineResume;
			pendingEngineResume = null;
			suppressBootPrompt = true;
			sessionOps.ReapplyAfterEngineRestart(snap).then(() => {
				suppressBootPrompt = false;
			}).catch(err => {
				console.error('engine resume failed', err);
				suppressBootPrompt = false;
				origChatAIReady(init_prompt);
			});
			return;
		}
		// ready + prompt:init both call ai_ready; ignore boot defaults until resume finishes.
		if (suppressBootPrompt && (!init_prompt || init_prompt === 'AI_UI_DEFAULT')) {
			return;
		}
		origChatAIReady(init_prompt);
		if (!sessionRestored && sessionOps) {
			sessionRestored = true;
			const resume = sessionScopePrepared
				? sessionOps.ResumeSessionOnStartup()
				: sessionOps.RestoreOnStartup();
			resume.catch(err => {
				console.error('session restore failed', err);
			});
		}
	};
	bridge.ChatAIEnded = function(p_msg) {
		// Keep sessionRestored so RESTART does not re-open last_project.
		origChatAIEnded(p_msg);
	};

	function PersistScopedConfig() {
		// SaveConfig writes project scope when open, otherwise full Home config.
		// SaveActiveConfig alone is a no-op in Home - AI Settings would not stick.
		if (typeof saveGlobalConfig === 'function') saveGlobalConfig();
		else if (sessions) sessions.SaveActiveConfig(core);
	}

	function SetRecShortcuts(start_shortcut, stop_shortcut) {
		let appMenu = Menu.getApplicationMenu();
		let menuItem = appMenu.getMenuItemById('stop-recording');

		if (stop_shortcut != '') {
			if (menuItem === null) {
				appMenu.items[0].submenu.insert(0, new MenuItem({
					id: 'stop-recording',
					role: 'stopSpeaking',
					label: 'Stop Recording',
					accelerator: stop_shortcut,
					registerAccelerator: false,
					click: () => { const w = getWin(); if (w) w.webContents.send('stop-recording'); }
				}));
			} else {
				menuItem.accelerator = stop_shortcut;
			}
		}

		if (start_shortcut == '') return;
		menuItem = appMenu.getMenuItemById('start-recording');

		if (menuItem === null) {
			appMenu.items[0].submenu.insert(0, new MenuItem({
				id: 'start-recording',
				role: 'startSpeaking',
				label: 'Start Recording',
				accelerator: start_shortcut,
				registerAccelerator: false,
				click: () => { const w = getWin(); if (w) w.webContents.send('start-recording'); }
			}));
		} else {
			menuItem.accelerator = start_shortcut;
		}
	}

	ipcMain.on('show-in-dir', (event, payload) => {
		const path = require('path');
		const norm_path = path.normalize(payload);
		if (fs.existsSync(norm_path)) shell.showItemInFolder(norm_path);
	});

	ipcMain.on('show-img', (event, payload) => {
		const path = require('path');
		const norm_path = path.normalize(payload);
		if (fs.existsSync(norm_path)) {
			shell.showItemInFolder(norm_path);
		} else if (payload.startsWith('https:') || payload.startsWith('http:')) {
			shell.openExternal(payload);
		}
	});

	ipcMain.on('copy-text', (event, payload) => {
		if (payload.txt != '') {
			clipboard.writeText(payload.txt);
			dialog.showMessageBox(getWin(), { type: "info", message: "Text copied to clipboard." });
		} else {
			dialog.showMessageBox(getWin(), { type: "info", message: "There is no text to copy!" });
		}
	});

	ipcMain.on('copy-image', (event, payload) => {
		if (payload.img != '') {
			clipboard.writeImage(nativeImage.createFromPath(payload.img));
			dialog.showMessageBox(getWin(), { type: "info", message: "Image copied to clipboard." });
		} else {
			dialog.showMessageBox(getWin(), { type: "info", message: "There is no image to copy!" });
		}
	});

	ipcMain.on('show-alert', (event, payload) => {
		const msg_type = payload.type ? payload.type : 'info';
		dialog.showMessageBox(getWin(), { type: msg_type, message: payload.msg });
	});

	ipcMain.handle('confirm-action', async (event, payload) => {
		payload = payload || {};
		const result = await dialog.showMessageBox(getWin(), {
			type: 'question',
			title: payload.title || 'Confirm Action',
			noLink: true,
			message: payload.msg || 'Proceed?',
			buttons: ['Confirm', 'Cancel']
		});
		return result.response === 0;
	});

	ipcMain.handle('stop-stream', (event) => {
		engine.sendControl('stop');
	});

	ipcMain.handle('tool-permission-result', (event, payload) => {
		ResolveToolPermissionResult(payload);
	});

	ipcMain.on('send-msg', (event, payload) => {
		payload = payload || {};
		if (payload.cmd) {
			engine.sendEngineCmd(payload.cmd);
			return;
		}
		// Fold attachments into the chat cmd (not a separate attach_files first).
		// A prior attach_files would emit idle -> ai-ready and flip STOP back to SEND.
		const extra = {};
		if (Object.prototype.hasOwnProperty.call(payload, 'files')) {
			extra.files = payload.files || [];
			core.setChatFiles(extra.files);
			if (payload.mm_processor != null) {
				extra.mm_processor = payload.mm_processor;
				core.setMmProcessor(payload.mm_processor);
			}
		}
		engine.sendMsg(payload.msg, extra);
	});

	ipcMain.on('gen-image', (event, payload) => {
		core.configGen(payload);
		engine.sendEngineCmd('gen_image');
	});

	ipcMain.on('gen-text', (event, payload) => {
		core.configGen(payload);
		engine.sendEngineCmd('gen_text');
	});

	ipcMain.on('gen-speech', (event, payload) => {
		core.configGen(payload);
		core.setReadText(payload.tts_txt);
		engine.sendEngineCmd('gen_speech');
	});

	ipcMain.on('read-text', (event, payload) => {
		if (payload.txt != '') {
			core.setReadText(payload.txt);
			engine.sendEngineCmd('read_text');
		} else {
			dialog.showMessageBox(getWin(), { type: "info", message: "There is no text to read!" });
		}
	});

	ipcMain.on('save-recording', (event, payload) => {
		const { app } = require('electron');
		const wavPath = app.getPath("temp")+'/recording.wav';
		let wavBuf;
		if (Buffer.isBuffer(payload)) {
			wavBuf = payload;
		} else if (payload instanceof ArrayBuffer) {
			wavBuf = Buffer.from(payload);
		} else if (ArrayBuffer.isView(payload)) {
			wavBuf = Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength);
		} else {
			wavBuf = Buffer.from(payload);
		}
		fs.writeFileSync(wavPath, wavBuf);
		engine.sendEngineCmd('speech_recog');
	});

	ipcMain.on('save-voice', (event, payload) => {
		if (fs.existsSync(payload.src_file)) {
			let voiceData = fs.readFileSync(payload.src_file);
			if (fs.existsSync(payload.dest_file)) {
				dialog.showMessageBox(getWin(), { type: "warning", message: "A voice file with that name already exists." });
			} else {
				try {
					fs.writeFileSync(payload.dest_file, voiceData);
					engine.sendEngineCmd('update_voices');
					dialog.showMessageBox(getWin(), { type: "info", message: "Voice saved, you can now select it from the list of voices." });
				} catch (err) {
					dialog.showMessageBoxSync(getWin(), { type: "warning", message: "Unable to save voice file. Ensure you are running the app as an administrator." });
				}
			}
		} else {
			dialog.showMessageBox(getWin(), { type: "error", message: "Unable to locate voice data!" });
		}
	});

	ipcMain.on('clone-voice', (event, payload) => {
		const workDir = core.getConfigs().app.script_dir;
		core.setCloneVoice(payload);
		let voiceFile = workDir+'/embeddings/'+payload.name+'.npy';
		if (payload.model == 1 || payload.model == 3 || payload.model == 4 || payload.model == 6) {
			if (payload.transcript.trim() == '') {
				bridge.ShowGenTTS("ERROR: Transcript text cannot be empty!");
				return;
			}
			if (payload.model == 1) {
				voiceFile = workDir+'/embeddings/ChatTTS/'+payload.name+'.txt';
			} else if (payload.model == 3) {
				voiceFile = workDir+'/embeddings/VoxCPM/'+payload.name+'.wav';
			} else if (payload.model == 6) {
				voiceFile = workDir+'/embeddings/Audio8/'+payload.name+'.wav';
			} else {
				voiceFile = workDir+'/embeddings/NeuTTS/'+payload.name+'.npy';
			}
		}
		if (fs.existsSync(voiceFile)) {
			dialog.showMessageBox(getWin(), { type: "question", title: "Confirm Action",
			noLink: true, message: "A voice with that name already exists. Overwrite it?", buttons: ["Yes","No"] }).then(value => {
				if (!value.response) {
					engine.sendEngineCmd('clone_voice');
				} else {
					bridge.ShowClonedVoice("ERROR:A voice with that name already exists.");
				}
			});
		} else {
			engine.sendEngineCmd('clone_voice');
		}
	});

	ipcMain.on('run-cmd', (event, payload) => {
		if (engine.engineRunning()) {
			dialog.showMessageBox(getWin(), { type: "question", title: "Engine Running",
			noLink: true, message: "The AI engine must be stopped to run commands. Continue?", buttons: ["Yes","No"] }).then(value => {
				if (!value.response) {
					engine.stopScript();
					setTimeout(function() {
						getWin().webContents.send('cmd-sent', engine.runCommand(payload.cmd));
					}, 1000);
				} else {
					getWin().webContents.send('cmd-sent', false);
				}
			});
		} else {
			getWin().webContents.send('cmd-sent', engine.runCommand(payload.cmd));
		}
	});

	ipcMain.on('config-voice', (event, payload) => {
		if (payload && payload.avatar_img) core.setAvatar(payload.avatar_img);
		core.configSpeech(payload);
		engine.sendEngineCmd('config_voice');
		PersistScopedConfig();
		getWin().webContents.send('load-config', LoadConfigPayload(true));
	});

	ipcMain.on('config-ai', (event, payload) => {
		core.configAI(payload);
		engine.sendEngineCmd('config_ai');
		PersistScopedConfig();
		getWin().webContents.send('load-config', LoadConfigPayload(true));
	});

	ipcMain.on('config-tools', (event, payload) => {
		payload = payload || {};
		let mcpWarn = [];
		if (payload.mcp_server) mcpWarn = PersistMcpServer(payload.mcp_server) || [];
		if (payload.tool_packs) PersistMcpEnablesFromPacks(payload.tool_packs);
		core.configTools(payload);
		const params = Object.assign({}, core.buildCmdParams('config_tools'));
		// Restart MCP after server settings edits so its tool list refreshes.
		if (payload.mcp_server && payload.mcp_server.id) {
			params.force_mcp_ids = [String(payload.mcp_server.id)];
		} else if (Array.isArray(payload.force_mcp_ids)) {
			params.force_mcp_ids = payload.force_mcp_ids;
		}
		engine.sendEngineCmd('config_tools', params);
		PersistScopedConfig();
		getWin().webContents.send('load-config', LoadConfigPayload(true));
		if (mcpWarn.length) {
			dialog.showMessageBox(getWin(), {
				type: 'warning',
				title: 'MCP config',
				message: mcpWarn.join('\n'),
				buttons: ['OK']
			});
		}
	});

	ipcMain.handle('mcp-open-editor', async () => {
		const { shell } = require('electron');
		const path = require('path');
		const fs = require('fs');
		try {
			const home = sessions ? sessions.GetAiuiHome() : '';
			if (!home) return { ok: false, error: 'No AI UI home' };
			mcpStore.LoadMcpConfig(home); // ensure file exists
			let file = String(mcpStore.GetMcpPath(home) || '');
			file = file.replace(/^file:\/\//i, '');
			if (file.startsWith('/') && /^[A-Za-z]:/.test(file.slice(1))) file = file.slice(1);
			file = path.resolve(file);
			if (!fs.existsSync(file)) return { ok: false, error: 'File not found: ' + file, path: file };
			// Reveal in the system file manager - opening via MIME handlers is unreliable
			// (browsers block file://, some editors reject the path).
			shell.showItemInFolder(file);
			return { ok: true, path: file };
		} catch (err) {
			return { ok: false, error: String(err && err.message ? err.message : err) };
		}
	});

	ipcMain.handle('mcp-refresh-packs', async () => {
		const home = sessions ? sessions.GetAiuiHome() : '';
		if (!home) return { ok: false };
		const mcp = mcpStore.LoadMcpConfig(home);
		const warnings = mcpStore.TakeWarnings(mcp);
		const overlay = mcpStore.ExtractMcpOverlay(mcpStore.ServersToPacks(core.getMcpConfig()));
		core.setMcpOverlay(overlay);
		core.setMcpConfig(mcpStore.ApplyMcpOverlay(mcp, overlay));
		const payload = LoadConfigPayload(true);
		getWin().webContents.send('load-config', payload);
		// Push updated pack list to engine without requiring APPLY.
		core.configTools({
			temp_format: core.AI_CONFIG.temp_format,
			tool_packs: payload.tool_packs,
			firecrawl_api_key: core.AI_CONFIG.firecrawl_api_key || ''
		});
		engine.sendEngineCmd('config_tools');
		PersistScopedConfig();
		return { ok: true, tool_packs: payload.tool_packs, warnings };
	});

	ipcMain.on('config-app', (event, payload) => {
		dialog.showMessageBox(getWin(), { type: "question", title: "Confirm Action",
		noLink: true, message: "Updating these settings will restart the AI engine. Proceed?", buttons: ["Confirm","Cancel"] }).then(async value => {
			if (!value.response) {
				if (payload.smodel_type != core.getConfigs().app.smodel_type) core.setBotVoice(0);
				if (payload && sessions) {
					payload.workspace = sessions.workspacePath || '';
				}
				core.configApp(payload);
				PersistScopedConfig();
				getWin().webContents.send('load-config', LoadConfigPayload(true));
				await RestartEngine();
			}
		});
	});

	ipcMain.on('config-app-other', (event, payload) => {
		core.configOther(payload, true);
		engine.sendEngineCmd('config_other');
		PersistScopedConfig();
	});

	ipcMain.on('config-app-remote', (event, payload) => {
		core.configRemote(payload || {});
		engine.sendEngineCmd('config_remote');
		PersistScopedConfig();
	});

	ipcMain.on('config-prompt-files', (event, payload) => {
		core.configPromptFiles(payload || {});
		if (sessions && payload && payload.hasOwnProperty('workspace')) {
			// workspace is owned by Open/Close Project; ignore settings overrides
			core.APP_CONFIG.workspace = sessions.workspacePath || '';
		}
		PersistScopedConfig();
		engine.sendEngineCmd('compose_prompt');
	});

	ipcMain.on('config-other', (event, payload) => {
		core.configOther(payload);
		SetRecShortcuts(payload.start_rec_keys, payload.stop_rec_keys);
		PersistScopedConfig();
	});

	ipcMain.on('show-menubar', () => {
		const win = getWin();
		if (win) win.setMenuBarVisibility(true);
		rebuildMenu();
	});

	ipcMain.on('update-users', (event, payload) => {
		dialog.showMessageBox(getWin(), { type: "question", title: "Confirm Action",
		noLink: true, message: "Updating the user names will clear the chat log. Proceed?", buttons: ["Confirm","Cancel"] }).then(value => {
			if (!value.response) {
				core.setUsernames(payload);
				engine.sendEngineCmd('update_users');
				PersistScopedConfig();
				getWin().webContents.send('load-config', LoadConfigPayload(true));
			}
		});
	});

	ipcMain.on('update-avatar', (event, payload) => {
		core.setAvatar(payload);
		engine.sendEngineCmd('update_avatar');
		PersistScopedConfig();
	});

	ipcMain.handle('prepare-avatar', (event, payload) => {
		payload = payload || {};
		const workDir = String(core.getConfigs().app.script_dir || '').replace(/[/\\]+$/, '');
		const res = PrepareAvatarImage({
			source: (payload.source != null && payload.source !== '') ? payload.source : '',
			mode: payload.mode,
			workDir,
			fallback: core.getConfigs().app.avatar_img || ''
		});
		if (res.ok) core.setAvatar(res.path);
		return res;
	});

	ipcMain.on('update-talk-mode', (event, payload) => {
		core.setTalkMode(payload);
		engine.sendEngineCmd('update_tmode');
		PersistScopedConfig();
	});

	ipcMain.on('update-prompt', (event, payload) => {
		core.setPrompt(payload);
		engine.sendEngineCmd('update_prompt');
	});

	ipcMain.on('attach-files', (event, payload) => {
		payload = payload || {};
		core.setChatFiles(payload.files);
		if (payload.mm_processor != null) {
			core.setMmProcessor(payload.mm_processor);
		}
		engine.sendEngineCmd('attach_files');
		PersistScopedConfig();
	});

	ipcMain.handle('restart-script', (event) => {
		dialog.showMessageBox(getWin(), { type: "question", title: "Confirm Action",
		noLink: true, message: "Restart the AI engine?", buttons: ["Confirm","Cancel"] }).then(async value => {
			if (!value.response) await RestartEngine();
		});
	});

	ipcMain.handle('start-script', async (event) => {
		engine.setToolPermissionHandler(HandleToolPermission);
		engine.setCallbacks(
			bridge.ShowBotMsg, bridge.ShowGenTxt, bridge.ShowGenTTS, bridge.ShowGenASR, bridge.ShowGenImg, bridge.ShowClonedVoice,
			bridge.ChatAIReady, bridge.ChatAIEnded, bridge.AddVoices, bridge.ClearVoices, bridge.PlayAudio, bridge.AppendLog,
			bridge.GotAvatar, bridge.GotTools, bridge.StreamStart, bridge.StreamText, bridge.StreamEnd,
			bridge.TurnStart, bridge.TurnPatch, bridge.TurnEnd,
			bridge.PromptFiles, bridge.GotToolCatalog, bridge.ContextCompact, bridge.ContextUsage
		);
		if (fs.existsSync(getConfigPath())) {
			sessionScopePrepared = false;
			if (sessionOps) {
				try {
					await sessionOps.PrepareScopeOnStartup();
					sessionScopePrepared = true;
				} catch (err) {
					console.error('scope prepare failed', err);
				}
			}
			getWin().webContents.send('load-config', LoadConfigPayload(false));
			const animMode = Number(core.getConfigs().chat.anim_mode);
			if (animMode >= 0) checkModelsExist(animMode);
			engine.startScript();
		} else {
			let workDir = process.cwd().trim().replaceAll('\\', '/');
			if (workDir.endsWith('/')) workDir = workDir.slice(0, -1);
			if (fs.existsSync(workDir+'/engine/ai_images/default_avatar.jpg'))
				core.initAppConfig(workDir+'/engine/ai_images/default_avatar.jpg', workDir+'/engine');
			getWin().webContents.send('load-config', LoadConfigPayload(false));
			bridge.ChatAIEnded('No configuration file found. Visit the Settings tab to get started.');
		}
	});

	return { SetRecShortcuts, bridge };
}

module.exports = { CreateHandlers };
