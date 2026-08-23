const path = require('path');
const fs = require('fs');
const sessions = require('./sessions.js');
const { EnsureAvatarForCore } = require('./avatar_cache.js');

function CreateSessionOps({ core, engine, getWin, rebuildMenu, saveGlobalConfig }) {
	let opChain = Promise.resolve();

	function enqueue(fn) {
		const run = opChain.then(() => fn());
		opChain = run.catch(() => {});
		return run;
	}

	function EngineBusyForSessions() {
		if (engine.isStreaming && engine.isStreaming()) return true;
		if (engine.isReady && !engine.isReady()) return true;
		return false;
	}

	function RejectIfBusy() {
		if (!EngineBusyForSessions()) return null;
		const err = new Error('Wait for the AI engine to finish before changing sessions.');
		err.code = 'engine_busy';
		return err;
	}

	function NotifyUi(channel, payload) {
		const w = getWin();
		if (w && !w.isDestroyed()) w.webContents.send(channel, payload || {});
	}

	function PushContext() {
		const st = sessions.GetState();
		core.APP_CONFIG.workspace = st.workspace || '';
		NotifyUi('session-context', {
			...st,
			recent: core.APP_CONFIG.recent_sessions || [],
			recent_projects: core.APP_CONFIG.recent_projects || []
		});
		if (typeof rebuildMenu === 'function') rebuildMenu();
	}

	function Persist() {
		if (typeof saveGlobalConfig === 'function') saveGlobalConfig();
		PushContext();
	}

	/** Persist current Home or project settings before switching scopes. */
	function PersistActiveScope() {
		if (typeof saveGlobalConfig === 'function') saveGlobalConfig();
		else sessions.SaveActiveConfig(core);
	}

	/**
	 * Host core already holds the active scope's settings; push live-updatable
	 * fields to the running engine so Context Token Budget / usage stay aligned.
	 */
	function SyncLiveEngineConfig() {
		if (!engine.engineRunning || !engine.engineRunning()) return;
		engine.sendEngineCmd('config_ai');
		engine.sendEngineCmd('config_voice');
		engine.sendEngineCmd('config_other');
		engine.sendEngineCmd('config_remote');
		const params = Object.assign({}, core.buildCmdParams('config_tools'));
		params.force_mcp_ids = (params.tool_packs || [])
			.filter((p) => p && p.kind === 'mcp' && p.id)
			.map((p) => String(p.id));
		engine.sendEngineCmd('config_tools', params);
	}

	function PromptFilesPayload(workspace) {
		return {
			agent: core.APP_CONFIG.agent,
			enabled_rules: core.APP_CONFIG.enabled_rules,
			enabled_skills: core.APP_CONFIG.enabled_skills,
			workspace: workspace || ''
		};
	}

	function ComposeFromFiles(workspace) {
		core.configPromptFiles(PromptFilesPayload(workspace));
		engine.sendEngineCmd('compose_prompt');
	}

	function BeginFreshChat() {
		sessions.SetSessionDirOverride('');
		sessions.SetSessionId(sessions.NewSessionId());
		NotifyUi('session-cleared', { compose: true });
		engine.sendEngineCmd('clear_chat');
	}

	async function ResumeLoaded(loaded) {
		sessions.SetSessionId(loaded.meta.id || path.basename(loaded.dir));
		sessions.SetSessionDirOverride(loaded.dir);
		await ImportMessages(loaded.messages, loaded.meta && loaded.meta.compaction);
		return true;
	}

	async function Autosave() {
		return enqueue(async () => {
			if (engine.isStreaming && engine.isStreaming()) return { ok: false, reason: 'streaming' };
			if (!engine.engineRunning()) return { ok: false, reason: 'engine_off' };
			try {
				const data = await engine.exportChat();
				const messages = data.messages || [];
				if (!sessions.HasConversation(messages)) return { ok: true, skipped: true };
				const written = sessions.WriteSessionFiles(core, messages, data.compaction != null ? data.compaction : null);
				if (written.skipped) return { ok: true, skipped: true };
				Persist();
				return { ok: true, dir: written.dir, meta: written.meta };
			} catch (err) {
				return { ok: false, reason: String(err && err.message ? err.message : err) };
			}
		});
	}

	function ReloadHomeScopedConfig() {
		const cfgPath = path.join(sessions.GetAiuiHome(), 'config.json');
		if (fs.existsSync(cfgPath)) {
			try {
				const globalCfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
				sessions.ApplyProjectScopedConfigs(core, {
					chat: globalCfg.chat, ai: globalCfg.ai, gen: globalCfg.gen, app: globalCfg.app
				});
			} catch (err) { /* keep current */ }
		}
		core.APP_CONFIG.workspace = '';
	}

	/** Leave project (if any) and load Home settings - does not resume a Home chat. */
	async function SwitchToHome(opts) {
		const doCompose = !opts || opts.compose !== false;
		if (sessions.workspacePath) {
			PersistActiveScope();
			sessions.ClearProjectResume();
		}
		sessions.SetSessionDirOverride('');
		sessions.SetHomeContext();
		core.APP_CONFIG.last_project = '';
		ReloadHomeScopedConfig();
		EnsureAvatarForCore(core);
		NotifyUi('load-config', {
			configs: core.getConfigs(),
			tool_packs: core.buildMergedToolPacks(),
			skip_inputs: false
		});
		SyncLiveEngineConfig();
		if (doCompose) ComposeFromFiles('');
	}

	async function NewSession() {
		return enqueue(async () => {
			const busy = RejectIfBusy();
			if (busy) throw busy;
			// Always return to Home with a fresh chat (leaves any open project).
			if (engine.engineRunning() && !(engine.isStreaming && engine.isStreaming())) {
				try {
					const data = await engine.exportChat();
					if (sessions.HasConversation(data.messages || [])) {
						sessions.WriteSessionFiles(core, data.messages || [], data.compaction != null ? data.compaction : null);
					}
				} catch (err) { /* continue */ }
			}
			NotifyUi('session-cleared', { compose: true });
			await SwitchToHome({ compose: false });
			sessions.SetSessionId(sessions.NewSessionId());
			sessions.ClearResumePointers(core);
			core.APP_CONFIG.last_session = '';
			core.APP_CONFIG.last_project = '';
			engine.sendEngineCmd('clear_chat');
			ComposeFromFiles('');
			Persist();
			return sessions.GetState();
		});
	}

	/** Fresh chat in the currently open project (does not leave the project). */
	async function NewProjectSession() {
		return enqueue(async () => {
			const busy = RejectIfBusy();
			if (busy) throw busy;
			if (!sessions.workspacePath || !sessions.projectKey) {
				throw new Error('Open a project before starting a new project session.');
			}
			if (engine.engineRunning() && !(engine.isStreaming && engine.isStreaming())) {
				try {
					const data = await engine.exportChat();
					if (sessions.HasConversation(data.messages || [])) {
						sessions.WriteSessionFiles(core, data.messages || [], data.compaction != null ? data.compaction : null);
					}
				} catch (err) { /* continue */ }
			}
			sessions.ClearProjectResume();
			BeginFreshChat();
			ComposeFromFiles(sessions.workspacePath);
			sessions.TouchRecentProject(core, sessions.projectKey);
			Persist();
			return sessions.GetState();
		});
	}

	function DecodeBr(text) {
		return String(text || '').replaceAll('[AI_UI_BR]', '\n');
	}

	function NormalizeMessages(messages) {
		if (!Array.isArray(messages)) return [];
		return messages.map((m) => {
			if (!m || typeof m !== 'object' || typeof m.content !== 'string') return m;
			if (!m.content.includes('[AI_UI_BR]')) return m;
			return { ...m, content: DecodeBr(m.content) };
		});
	}

	async function ImportMessages(messages, compaction) {
		const normalized = NormalizeMessages(messages);
		const payload = { messages: normalized };
		if (compaction && typeof compaction === 'object') {
			payload.compaction = {
				summary: String(compaction.summary || ''),
				summarized_until: Number(
					compaction.summarized_until != null
						? compaction.summarized_until
						: (compaction.compact_until != null ? compaction.compact_until : 0)
				),
				compact_count: Number(
					compaction.compact_count != null ? compaction.compact_count : 0
				)
			};
		}
		engine.sendEngineCmd('import_chat', payload);
		let initPrompt = '';
		if (normalized.length && normalized[0].role === 'system') {
			initPrompt = DecodeBr(normalized[0].content || '');
		}
		core.setPrompt(initPrompt.replaceAll('\n', '[AI_UI_BR]'));
		NotifyUi('session-loaded', {
			messages: normalized,
			init_prompt: initPrompt,
			context: sessions.GetState()
		});
		PushContext();
	}

	/** Re-apply host prompt/chat after an engine process restart (not app startup). */
	async function ReapplyAfterEngineRestart(snap) {
		const messages = (snap && snap.messages) || [];
		const initPrompt = (snap && snap.init_prompt) || '';
		if (sessions.HasConversation(messages)) {
			await ImportMessages(messages, snap.compaction || null);
			NotifyUi('ai-ready', {});
			return;
		}
		const prompt = initPrompt || String(core.INIT_PROMPT || '').replaceAll('[AI_UI_BR]', '\n');
		if (prompt) {
			core.setPrompt(prompt.replaceAll('\n', '[AI_UI_BR]'));
			engine.sendEngineCmd('update_prompt');
			ComposeFromFiles(sessions.workspacePath || '');
			NotifyUi('init-ui', { state: prompt.replaceAll('\n', '[AI_UI_BR]') });
			NotifyUi('ai-ready', {});
			return;
		}
		ComposeFromFiles(sessions.workspacePath || '');
		NotifyUi('ai-ready', {});
	}

	async function ApplyProjectContext(absPath) {
		const { key } = ApplyProjectScopeOnly(absPath);
		EnsureAvatarForCore(core);
		NotifyUi('load-config', {
			configs: core.getConfigs(),
			tool_packs: core.buildMergedToolPacks(),
			skip_inputs: false
		});
		SyncLiveEngineConfig();
		ComposeFromFiles(sessions.workspacePath);
		return key;
	}

	/** Host-only project scope (no engine cmds). Used before boot so LoadAvatar runs once. */
	function ApplyProjectScopeOnly(absPath) {
		PersistActiveScope();
		const ensured = sessions.EnsureProjectDirs(absPath);
		sessions.SetSessionDirOverride('');
		sessions.SetProjectContext(absPath, ensured.key);
		core.APP_CONFIG.workspace = sessions.workspacePath;
		sessions.LoadOrSeedProjectConfig(core, ensured.key);
		sessions.TouchRecentProject(core, ensured.key);
		return ensured;
	}

	/** Pending conversation resume after PrepareScopeOnStartup + engine ready. */
	let startupResume = null;

	/**
	 * Apply Home vs last-project settings to core before the engine boots so
	 * buildBootConfig / first LoadAvatar already match the restored scope.
	 */
	async function PrepareScopeOnStartup() {
		return enqueue(async () => {
			sessions.EnsureHomeLayout();
			sessions.PruneRecent(core);
			sessions.PruneRecentProjects(core);
			if (typeof saveGlobalConfig === 'function') saveGlobalConfig();

			const lastProject = (core.APP_CONFIG.last_project || '').trim();
			const lastSession = (core.APP_CONFIG.last_session || '').trim();
			startupResume = null;

			const lastRec = lastProject ? sessions.ReadProjectRecord(lastProject) : null;
			if (lastRec && fs.existsSync(lastRec.workspace)) {
				try {
					const { key, meta } = ApplyProjectScopeOnly(lastRec.workspace);
					EnsureAvatarForCore(core);
					startupResume = { kind: 'project', key, meta, lastSession };
					return sessions.GetState();
				} catch (err) {
					sessions.SetHomeContext();
					sessions.RemoveRecentProject(core, lastRec.registry);
					core.APP_CONFIG.last_project = '';
					ReloadHomeScopedConfig();
				}
			} else if (lastProject) {
				sessions.RemoveRecentProject(core, lastProject);
				core.APP_CONFIG.last_project = '';
			}

			sessions.SetHomeContext();
			core.APP_CONFIG.workspace = '';
			core.APP_CONFIG.last_project = '';
			EnsureAvatarForCore(core);
			startupResume = { kind: 'home', lastSession };
			return sessions.GetState();
		});
	}

	/** Import last chat / fresh session after engine ready (scope already prepared). */
	async function ResumeSessionOnStartup() {
		return enqueue(async () => {
			const pending = startupResume;
			startupResume = null;
			if (!pending) {
				// Fallback when prepare was skipped: full restore (may double-load avatar).
				return RestoreOnStartupFull();
			}

			if (pending.kind === 'project') {
				await ResumeProjectSession(pending.key, pending.meta, pending.lastSession);
				ComposeFromFiles(sessions.workspacePath);
				Persist();
				return sessions.GetState();
			}

			const lastSession = String(pending.lastSession || '').trim();
			const loaded = sessions.TryLoadConversation(lastSession);
			if (loaded) {
				await ResumeLoaded(loaded);
				ComposeFromFiles('');
				Persist();
				return sessions.GetState();
			}

			if (lastSession) sessions.RemoveRecentSession(core, lastSession);
			sessions.SetSessionDirOverride('');
			sessions.SetSessionId(sessions.NewSessionId());
			core.APP_CONFIG.last_session = '';
			ComposeFromFiles('');
			Persist();
			return sessions.GetState();
		});
	}

	async function RestoreOnStartupFull() {
		sessions.EnsureHomeLayout();
		sessions.PruneRecent(core);
		sessions.PruneRecentProjects(core);
		if (typeof saveGlobalConfig === 'function') saveGlobalConfig();

		const lastProject = (core.APP_CONFIG.last_project || '').trim();
		const lastSession = (core.APP_CONFIG.last_session || '').trim();

		const lastRec = lastProject ? sessions.ReadProjectRecord(lastProject) : null;
		if (lastRec && fs.existsSync(lastRec.workspace)) {
			try {
				const { key, meta } = sessions.EnsureProjectDirs(lastRec.workspace);
				await ApplyProjectContext(lastRec.workspace);
				await ResumeProjectSession(key, meta, lastSession);
				Persist();
				return sessions.GetState();
			} catch (err) {
				sessions.SetHomeContext();
				sessions.RemoveRecentProject(core, lastRec.registry);
				core.APP_CONFIG.last_project = '';
				ReloadHomeScopedConfig();
				EnsureAvatarForCore(core);
			}
		} else if (lastProject) {
			sessions.RemoveRecentProject(core, lastProject);
			core.APP_CONFIG.last_project = '';
		}

		sessions.SetHomeContext();
		core.APP_CONFIG.workspace = '';
		core.APP_CONFIG.last_project = '';
		EnsureAvatarForCore(core);
		const loaded = sessions.TryLoadConversation(lastSession);
		if (loaded) {
			await ResumeLoaded(loaded);
			Persist();
			return sessions.GetState();
		}

		if (lastSession) sessions.RemoveRecentSession(core, lastSession);
		sessions.SetSessionDirOverride('');
		sessions.SetSessionId(sessions.NewSessionId());
		core.APP_CONFIG.last_session = '';
		Persist();
		return sessions.GetState();
	}

	async function RestoreOnStartup() {
		await PrepareScopeOnStartup();
		return ResumeSessionOnStartup();
	}

	async function DropMissingSession(selectedPath) {
		const candidate = sessions.SessionDirCandidate(selectedPath);
		if (!candidate) return false;
		const chatPath = path.join(candidate, 'chat.json');
		if (fs.existsSync(chatPath)) return false;
		if (!sessions.RemoveRecentSession(core, candidate)) return false;
		Persist();
		return true;
	}

	async function ResumeProjectSession(key, meta, preferSessionPath) {
		let resumed = false;
		const prefer = String(preferSessionPath || '').trim();
		if (prefer) {
			const root = sessions.NormPath(path.join(sessions.SessionsRoot(), 'projects', key));
			const loadedPrefer = sessions.TryLoadConversation(prefer);
			if (loadedPrefer) {
				const dir = sessions.NormPath(loadedPrefer.dir);
				if (dir === root || dir.startsWith(root + '/')) {
					resumed = await ResumeLoaded(loadedPrefer);
				}
			}
		}
		if (!resumed && meta && meta.last_session) {
			const sdir = path.join(sessions.SessionsRoot(), 'projects', key, meta.last_session);
			const loaded = sessions.TryLoadConversation(sdir);
			if (loaded) resumed = await ResumeLoaded(loaded);
		}
		if (!resumed) {
			sessions.ClearProjectResume();
			BeginFreshChat();
		} else if (sessions.sessionId) {
			sessions.SetProjectLastSession(sessions.sessionId, key);
		}
		return resumed;
	}

	async function LoadSessionPath(selectedPath) {
		return enqueue(async () => {
			const busy = RejectIfBusy();
			if (busy) throw busy;
			if (engine.engineRunning() && !(engine.isStreaming && engine.isStreaming())) {
				try {
					const data = await engine.exportChat();
					if (sessions.HasConversation(data.messages || [])) {
						sessions.WriteSessionFiles(core, data.messages || [], data.compaction != null ? data.compaction : null);
					}
				} catch (err) { /* continue */ }
			}
			const dir = sessions.ResolveSessionDir(selectedPath);
			if (!dir) {
				if (await DropMissingSession(selectedPath)) {
					throw new Error('That session no longer exists on disk. It was removed from Recent Sessions.');
				}
				throw new Error('Select a session folder or chat.json file');
			}
			const loaded = sessions.TryLoadConversation(dir);
			if (!loaded) {
				if (await DropMissingSession(dir)) {
					throw new Error('That session no longer exists on disk. It was removed from Recent Sessions.');
				}
				throw new Error('That session has no conversation to load');
			}

			// Path under sessions/home vs sessions/projects wins over meta.project_path.
			const loc = sessions.ContextFromSessionDir(loaded.dir);
			if (loc.context === 'project') {
				let projectPath = loc.project_path || '';
				if (!projectPath) {
					const hit = (core.APP_CONFIG.recent_sessions || []).find(
						(r) => r && sessions.NormPath(r.path) === sessions.NormPath(loaded.dir)
					);
					if (hit && hit.project_path) projectPath = sessions.NormPath(hit.project_path);
				}
				if (!projectPath) {
					for (const p of (core.APP_CONFIG.recent_projects || [])) {
						const hit = (p && p.recent_sessions || []).find(
							(r) => r && sessions.NormPath(r.path) === sessions.NormPath(loaded.dir)
						);
						if (hit && hit.project_path) {
							projectPath = sessions.NormPath(hit.project_path);
							break;
						}
						if (hit && p.workspace) {
							projectPath = sessions.NormPath(p.workspace);
							break;
						}
					}
				}
				if (!projectPath) {
					const rec = loc.project_key ? sessions.ReadProjectRecord(loc.project_key) : null;
					if (rec) projectPath = rec.workspace;
				}
				if (!projectPath || !fs.existsSync(projectPath)) {
					sessions.RemoveRecentSession(core, loaded.dir);
					Persist();
					throw new Error('That session\'s project folder is missing. Open the project again from File → Open Project.');
				}
				if (sessions.workspacePath !== projectPath) await ApplyProjectContext(projectPath);
			} else if (sessions.workspacePath) {
				await SwitchToHome();
			}

			await ResumeLoaded(loaded);
			sessions.TouchRecent(core, loaded.dir, {
				title: sessions.TitleFromMessages(loaded.messages),
				updated_at: new Date().toISOString()
			});
			if (loc.context === 'project' && loc.project_key) {
				sessions.SetProjectLastSession(loaded.meta.id || path.basename(loaded.dir), loc.project_key);
			}
			sessions.PruneRecent(core);
			sessions.PruneRecentProjects(core);
			Persist();
			return loaded;
		});
	}

	/** Open a workspace folder (File > Open Project). Creates registry data if needed. */
	async function OpenProject(absPath) {
		return enqueue(async () => {
			const busy = RejectIfBusy();
			if (busy) throw busy;
			if (!absPath || !fs.existsSync(absPath) || !fs.statSync(absPath).isDirectory()) {
				throw new Error('Invalid project folder');
			}
			if (sessions.IsProjectRegistryPath(absPath)) {
				throw new Error('Select the project repository folder, not an AI UI project record.');
			}
			if (engine.engineRunning() && !(engine.isStreaming && engine.isStreaming())) {
				try {
					const data = await engine.exportChat();
					if (sessions.HasConversation(data.messages || [])) {
						sessions.WriteSessionFiles(core, data.messages || [], data.compaction != null ? data.compaction : null);
					}
				} catch (err) { /* continue */ }
			}
			const { key, meta } = sessions.EnsureProjectDirs(absPath);
			await ApplyProjectContext(absPath);
			await ResumeProjectSession(key, meta);
			sessions.PruneRecent(core);
			sessions.PruneRecentProjects(core);
			Persist();
			return sessions.GetState();
		});
	}

	/**
	 * Open from Recent Projects. path is ~/.aiui/projects/<key> (not the repo folder).
	 * Requires existing AI UI project metadata; does not silently recreate wiped records.
	 */
	async function OpenRecentProject(registryPath) {
		return enqueue(async () => {
			const busy = RejectIfBusy();
			if (busy) throw busy;
			const rec = sessions.ReadProjectRecord(registryPath);
			if (!rec) {
				sessions.RemoveRecentProject(core, registryPath);
				Persist();
				throw new Error(
					'AI UI project data for this entry is missing (removed from Recent Projects). '
					+ 'Use File > Open Project to open the repository again.'
				);
			}
			if (!fs.existsSync(rec.workspace) || !fs.statSync(rec.workspace).isDirectory()) {
				sessions.RemoveRecentProject(core, rec.registry);
				Persist();
				throw new Error(
					'That project folder no longer exists on disk. It was removed from Recent Projects.'
				);
			}
			if (engine.engineRunning() && !(engine.isStreaming && engine.isStreaming())) {
				try {
					const data = await engine.exportChat();
					if (sessions.HasConversation(data.messages || [])) {
						sessions.WriteSessionFiles(core, data.messages || [], data.compaction != null ? data.compaction : null);
					}
				} catch (err) { /* continue */ }
			}
			const { key, meta } = sessions.EnsureProjectDirs(rec.workspace);
			await ApplyProjectContext(rec.workspace);
			await ResumeProjectSession(key, meta);
			sessions.PruneRecent(core);
			sessions.PruneRecentProjects(core);
			Persist();
			return sessions.GetState();
		});
	}

	async function CloseProject() {
		return enqueue(async () => {
			const busy = RejectIfBusy();
			if (busy) throw busy;
			if (!sessions.workspacePath) return sessions.GetState();
			if (engine.engineRunning() && !(engine.isStreaming && engine.isStreaming())) {
				try {
					const data = await engine.exportChat();
					if (sessions.HasConversation(data.messages || [])) {
						sessions.WriteSessionFiles(core, data.messages || [], data.compaction != null ? data.compaction : null);
					}
				} catch (err) { /* continue */ }
			}
			await SwitchToHome();
			sessions.PruneRecent(core);
			sessions.PruneRecentProjects(core);

			const last = core.APP_CONFIG.last_session || '';
			const homeLast = last.includes('/sessions/home/') ? sessions.TryLoadConversation(last) : null;
			if (homeLast) {
				await ResumeLoaded(homeLast);
			} else {
				core.APP_CONFIG.last_session = '';
				BeginFreshChat();
			}
			Persist();
			return sessions.GetState();
		});
	}

	return {
		Autosave,
		NewSession,
		NewProjectSession,
		LoadSessionPath,
		OpenProject,
		OpenRecentProject,
		CloseProject,
		PrepareScopeOnStartup,
		ResumeSessionOnStartup,
		RestoreOnStartup,
		ReapplyAfterEngineRestart,
		PushContext,
		ImportMessages
	};
}

module.exports = { CreateSessionOps };
