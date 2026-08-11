const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const RECENT_MAX = 12;

const GLOBAL_ONLY_APP = new Set([
	'script_dir', 'python_bin', 'model_dir', 'sd_model', 'tts_model', 'sr_model', 'model_args',
	'model_type', 'imodel_type', 'smodel_type', 'comp_dev', 'start_meth', 'avatar_img',
	'hf_token', 'enable_devmode', 'enable_asasro', 'start_rec_keys', 'stop_rec_keys', 'ui_theme',
	'text_backend', 'remote_base_url', 'remote_model', 'remote_api_key', 'remote_provider', 'remote_tool_format', 'remote_prompt_cache',
	'workspace', 'last_session', 'last_project', 'recent_sessions', 'recent_projects'
]);

const BOOKKEEPING_APP = ['workspace', 'last_session', 'last_project', 'recent_sessions', 'recent_projects'];

var aiuiHome = '';
var sessionId = '';
var projectKey = '';
var workspacePath = '';
var sessionDirOverride = '';

function NormPath(p) {
	return path.resolve(String(p || '')).replace(/\\/g, '/');
}

function GetAiuiHome() {
	if (!aiuiHome) {
		const { app } = require('electron');
		aiuiHome = path.join(app.getPath('home'), '.aiui');
	}
	return aiuiHome;
}

function SetAiuiHome(home) {
	aiuiHome = home;
}

function ProjectKey(absPath) {
	return crypto.createHash('sha1').update(NormPath(absPath)).digest('hex').slice(0, 16);
}

function NewSessionId() {
	return Date.now().toString(36) + '-' + crypto.randomBytes(4).toString('hex');
}

function EnsureDir(dir) {
	if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
	return dir;
}

function EnsureHomeLayout() {
	const home = GetAiuiHome();
	for (const sub of ['', 'agents', 'rules', 'skills', 'artifacts', 'sessions', 'sessions/home', 'projects']) {
		EnsureDir(sub ? path.join(home, sub) : home);
	}
	return home;
}

function SessionsRoot() {
	return path.join(GetAiuiHome(), 'sessions');
}

function CurrentSessionDir() {
	if (sessionDirOverride) return sessionDirOverride;
	if (!sessionId) return '';
	if (projectKey) return path.join(SessionsRoot(), 'projects', projectKey, sessionId);
	return path.join(SessionsRoot(), 'home', sessionId);
}

function SetSessionDirOverride(dir) {
	sessionDirOverride = dir ? NormPath(dir) : '';
}

function CurrentContextLabel() {
	if (!workspacePath) return 'Home';
	return path.basename(workspacePath.replace(/[\\/]+$/, '')) || 'Project';
}

/** Trust the session folder location over meta.json (meta can be wrong after bugs). */
function ContextFromSessionDir(dir) {
	const norm = NormPath(dir);
	if (!norm) {
		return {
			context: projectKey ? 'project' : 'home',
			project_key: projectKey || '',
			project_path: workspacePath || ''
		};
	}
	const homeRoot = NormPath(path.join(SessionsRoot(), 'home'));
	const projRoot = NormPath(path.join(SessionsRoot(), 'projects'));
	if (norm === homeRoot || norm.startsWith(homeRoot + '/')) {
		return { context: 'home', project_key: '', project_path: '' };
	}
	if (norm === projRoot || norm.startsWith(projRoot + '/')) {
		const key = norm.slice(projRoot.length + 1).split('/')[0] || '';
		const pj = ReadJson(path.join(ProjectDir(key), 'project.json'), {});
		return {
			context: 'project',
			project_key: key,
			project_path: NormPath(pj.path || workspacePath || '')
		};
	}
	return {
		context: projectKey ? 'project' : 'home',
		project_key: projectKey || '',
		project_path: workspacePath || ''
	};
}

function ReadJson(file, fallback) {
	try {
		if (!fs.existsSync(file)) return fallback;
		return JSON.parse(fs.readFileSync(file, 'utf8'));
	} catch (err) {
		return fallback;
	}
}

function WriteJson(file, obj) {
	EnsureDir(path.dirname(file));
	fs.writeFileSync(file, JSON.stringify(obj, null, '\t') + '\n', 'utf8');
}

function ProjectDir(key) {
	return path.join(GetAiuiHome(), 'projects', key || projectKey);
}

function EnsureProjectDirs(absProjectPath) {
	const key = ProjectKey(absProjectPath);
	const pdir = EnsureDir(ProjectDir(key));
	const projAiui = path.join(absProjectPath, '.aiui');
	for (const sub of ['', 'agents', 'rules', 'skills', 'artifacts']) {
		EnsureDir(sub ? path.join(projAiui, sub) : projAiui);
	}
	EnsureDir(path.join(SessionsRoot(), 'projects', key));
	const projectJson = path.join(pdir, 'project.json');
	const meta = ReadJson(projectJson, null) || {};
	meta.path = NormPath(absProjectPath);
	meta.name = path.basename(meta.path);
	meta.key = key;
	if (!meta.last_session) meta.last_session = '';
	WriteJson(projectJson, meta);
	return { key, meta, pdir };
}

function GetProjectScopedConfigs(core) {
	const all = core.getConfigs();
	const appScoped = {};
	for (const k of Object.keys(all.app)) {
		if (!GLOBAL_ONLY_APP.has(k)) appScoped[k] = all.app[k];
	}
	return { chat: { ...all.chat }, ai: { ...all.ai }, gen: { ...all.gen }, app: appScoped };
}

function ApplyProjectScopedConfigs(core, scoped) {
	if (!scoped) return;
	const cur = core.getConfigs();
	const merged = {
		chat: { ...cur.chat, ...(scoped.chat || {}) },
		ai: { ...cur.ai, ...(scoped.ai || {}) },
		gen: { ...cur.gen, ...(scoped.gen || {}) },
		app: { ...cur.app }
	};
	if (scoped.app) {
		for (const k of Object.keys(scoped.app)) {
			if (!GLOBAL_ONLY_APP.has(k)) merged.app[k] = scoped.app[k];
		}
	}
	for (const k of BOOKKEEPING_APP) merged.app[k] = cur.app[k];
	core.setConfigs(merged);
}

function LoadOrSeedProjectConfig(core, key) {
	const cfgPath = path.join(ProjectDir(key), 'config.json');
	if (fs.existsSync(cfgPath)) {
		ApplyProjectScopedConfigs(core, ReadJson(cfgPath, {}));
		return false;
	}
	WriteJson(cfgPath, GetProjectScopedConfigs(core));
	return true;
}

function SaveActiveConfig(core) {
	if (projectKey) {
		WriteJson(path.join(ProjectDir(projectKey), 'config.json'), GetProjectScopedConfigs(core));
	}
}

function TitleFromMessages(messages) {
	if (!Array.isArray(messages)) return 'New session';
	for (let i = 0; i < messages.length; i++) {
		const m = messages[i];
		if (!m || m.role === 'system' || m.role === 'tool' || m.role === 'compact' || m._aiui_compact || m.tool_calls) continue;
		if (typeof m.content === 'string' && m.content.trim()) {
			const line = m.content.trim().split(/\r?\n/)[0].trim();
			return line.length > 60 ? line.slice(0, 57) + '...' : line;
		}
	}
	return 'New session';
}

/** True only when there is real conversation content (not system-prompt-only). */
function HasConversation(messages) {
	if (!Array.isArray(messages) || messages.length === 0) return false;
	for (let i = 0; i < messages.length; i++) {
		const m = messages[i];
		if (!m || typeof m !== 'object' || m.role === 'system') continue;
		if (m.role === 'compact' || m._aiui_compact) continue;
		if (m.tool_calls || m.role === 'tool') return true;
		if (typeof m.content === 'string' && m.content.trim()) return true;
		if (Array.isArray(m.content) && m.content.length) return true;
	}
	return false;
}

function CapRecent(list) {
	if (list.length > RECENT_MAX) list.length = RECENT_MAX;
	return list;
}

function MakeRecentSessionEntry(sessionDir, meta, loc) {
	const dir = NormPath(sessionDir);
	const place = loc || ContextFromSessionDir(dir);
	return {
		path: dir,
		title: (meta && meta.title) || 'Session',
		updated_at: (meta && meta.updated_at) || new Date().toISOString(),
		context: place.context,
		project_key: place.project_key || '',
		project_path: place.project_path || ''
	};
}

/** Drop missing/empty session entries; keep order. */
function PruneSessionEntryList(list) {
	const kept = [];
	for (const r of (list || [])) {
		if (!r || !r.path) continue;
		const chatPath = path.join(r.path, 'chat.json');
		if (!fs.existsSync(chatPath)) continue;
		if (!HasConversation(ReadJson(chatPath, []))) continue;
		const loc = ContextFromSessionDir(r.path);
		kept.push(MakeRecentSessionEntry(r.path, {
			title: r.title || 'Session',
			updated_at: r.updated_at || ''
		}, loc));
	}
	return kept;
}

/** Scan session folders and read titles from meta.json (newest first). */
function ListSessionsInDir(rootDir) {
	const root = String(rootDir || '');
	if (!root || !fs.existsSync(root)) return [];
	let names = [];
	try {
		names = fs.readdirSync(root);
	} catch (err) {
		return [];
	}
	const out = [];
	for (const name of names) {
		const dir = path.join(root, name);
		try {
			if (!fs.statSync(dir).isDirectory()) continue;
		} catch (err) {
			continue;
		}
		const chatPath = path.join(dir, 'chat.json');
		if (!fs.existsSync(chatPath)) continue;
		if (!HasConversation(ReadJson(chatPath, []))) continue;
		const meta = ReadJson(path.join(dir, 'meta.json'), {}) || {};
		out.push({
			path: NormPath(dir),
			title: String(meta.title || 'Session'),
			updated_at: String(meta.updated_at || '')
		});
	}
	out.sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)));
	return out;
}

function ListHomeSessions() {
	return ListSessionsInDir(path.join(SessionsRoot(), 'home'));
}

function ListProjectSessions(key) {
	const k = String(key || '').trim();
	if (!k) return [];
	return ListSessionsInDir(path.join(SessionsRoot(), 'projects', k));
}

function SessionDirCandidate(selectedPath) {
	if (!selectedPath) return '';
	const p = String(selectedPath);
	try {
		if (fs.existsSync(p) && fs.statSync(p).isFile()) {
			return path.basename(p) === 'chat.json' ? path.dirname(p) : '';
		}
	} catch (err) { /* missing path */ }
	return p;
}

function RemoveRecentSession(core, sessionDir) {
	if (!core || !sessionDir) return false;
	const dir = NormPath(sessionDir);
	const appCfg = core.APP_CONFIG;
	let removed = false;
	const prev = Array.isArray(appCfg.recent_sessions) ? appCfg.recent_sessions : [];
	const next = prev.filter(r => r && NormPath(r.path) !== dir);
	if (next.length !== prev.length) {
		removed = true;
		appCfg.recent_sessions = CapRecent(next);
	}
	const projects = Array.isArray(appCfg.recent_projects) ? appCfg.recent_projects : [];
	for (const p of projects) {
		if (!p || !Array.isArray(p.recent_sessions)) continue;
		const before = p.recent_sessions.length;
		p.recent_sessions = p.recent_sessions.filter((r) => r && NormPath(r.path) !== dir);
		if (p.recent_sessions.length !== before) removed = true;
	}
	if (NormPath(appCfg.last_session || '') === dir) {
		appCfg.last_session = (appCfg.recent_sessions && appCfg.recent_sessions[0])
			? appCfg.recent_sessions[0].path
			: '';
	}
	return removed;
}

function ProjectsRoot() {
	return path.join(GetAiuiHome(), 'projects');
}

function IsProjectRegistryPath(absPath) {
	const norm = NormPath(absPath);
	const root = NormPath(ProjectsRoot());
	if (!norm || !root) return false;
	return norm === root || norm.startsWith(root + '/');
}

function ReadProjectRecord(registryOrKey) {
	if (!registryOrKey) return null;
	const raw = String(registryOrKey);
	const regDir = IsProjectRegistryPath(raw)
		? NormPath(raw)
		: NormPath(ProjectDir(raw));
	const metaPath = path.join(regDir, 'project.json');
	if (!fs.existsSync(metaPath)) return null;
	const meta = ReadJson(metaPath, null);
	if (!meta || typeof meta !== 'object') return null;
	const key = String(meta.key || path.basename(regDir) || '').trim();
	const workspace = NormPath(meta.path || '');
	if (!key || !workspace) return null;
	return {
		key,
		registry: regDir,
		workspace,
		name: String(meta.name || path.basename(workspace) || key),
		last_session: String(meta.last_session || ''),
		meta
	};
}

function RemoveRecentProject(core, registryPath) {
	if (!core || !registryPath) return false;
	const norm = NormPath(registryPath);
	const appCfg = core.APP_CONFIG;
	const prev = Array.isArray(appCfg.recent_projects) ? appCfg.recent_projects : [];
	const next = prev.filter((r) => {
		if (!r || !r.path) return false;
		if (NormPath(r.path) === norm) return false;
		if (r.key && NormPath(ProjectDir(r.key)) === norm) return false;
		return true;
	});
	const removed = next.length !== prev.length;
	appCfg.recent_projects = CapRecent(next);
	if (NormPath(appCfg.last_project || '') === norm) appCfg.last_project = '';
	return removed;
}

function TouchRecent(core, sessionDir, meta) {
	const appCfg = core.APP_CONFIG;
	const dir = NormPath(sessionDir);
	const loc = ContextFromSessionDir(dir);
	const entry = MakeRecentSessionEntry(dir, meta, loc);
	appCfg.last_session = dir;

	if (loc.context === 'project' && loc.project_key) {
		TouchRecentProject(core, loc.project_key);
		const projects = Array.isArray(appCfg.recent_projects) ? appCfg.recent_projects : [];
		const proj = projects.find((p) => p && p.key === loc.project_key);
		if (proj) {
			let rs = Array.isArray(proj.recent_sessions) ? proj.recent_sessions.slice() : [];
			rs = rs.filter((r) => r && NormPath(r.path) !== dir);
			rs.unshift(entry);
			proj.recent_sessions = CapRecent(rs);
		}
		return;
	}

	appCfg.last_project = '';
	let recent = Array.isArray(appCfg.recent_sessions) ? appCfg.recent_sessions.slice() : [];
	recent = recent.filter((r) => r && NormPath(r.path) !== dir);
	recent.unshift(entry);
	appCfg.recent_sessions = CapRecent(recent);
}

function PruneRecent(core) {
	const appCfg = core.APP_CONFIG;
	const kept = PruneSessionEntryList(appCfg.recent_sessions).filter((r) => r.context !== 'project');
	appCfg.recent_sessions = CapRecent(kept);
	if (appCfg.last_session) {
		const lastChat = path.join(appCfg.last_session, 'chat.json');
		if (!fs.existsSync(lastChat) || !HasConversation(ReadJson(lastChat, []))) {
			appCfg.last_session = kept.length ? kept[0].path : '';
		}
	}
	return appCfg.recent_sessions;
}

function ClearProjectResume() {
	if (!projectKey) return;
	const pj = path.join(ProjectDir(projectKey), 'project.json');
	const pmeta = ReadJson(pj, {});
	pmeta.last_session = '';
	WriteJson(pj, pmeta);
}

/** Point project resume at this session id (basename under sessions/projects/<key>/). */
function SetProjectLastSession(sessionIdOrDir, key) {
	const k = key || projectKey;
	if (!k) return;
	let sid = String(sessionIdOrDir || '').trim();
	if (!sid) return;
	// Allow a full session dir path.
	if (sid.includes('/') || sid.includes('\\')) sid = path.basename(sid);
	const pj = path.join(ProjectDir(k), 'project.json');
	const pmeta = ReadJson(pj, {});
	pmeta.last_session = sid;
	WriteJson(pj, pmeta);
}

function ClearResumePointers(core) {
	if (core && core.APP_CONFIG) core.APP_CONFIG.last_session = '';
	ClearProjectResume();
}

/** Record a project in Recent Projects. Accepts registry path or project key. */
function TouchRecentProject(core, registryOrKey) {
	if (!core || !registryOrKey) return null;
	const rec = ReadProjectRecord(registryOrKey);
	if (!rec) return null;
	const appCfg = core.APP_CONFIG;
	let recent = Array.isArray(appCfg.recent_projects) ? appCfg.recent_projects.slice() : [];
	const prevHit = recent.find((r) => {
		if (!r || !r.path) return false;
		if (NormPath(r.path) === rec.registry) return true;
		if (r.key && r.key === rec.key) return true;
		return false;
	});
	const nested = CapRecent(Array.isArray(prevHit && prevHit.recent_sessions)
		? prevHit.recent_sessions
		: []);
	recent = recent.filter((r) => {
		if (!r || !r.path) return false;
		if (NormPath(r.path) === rec.registry) return false;
		if (r.key && r.key === rec.key) return false;
		return true;
	});
	recent.unshift({
		path: rec.registry,
		key: rec.key,
		name: rec.name,
		workspace: rec.workspace,
		updated_at: new Date().toISOString(),
		recent_sessions: nested
	});
	appCfg.recent_projects = CapRecent(recent);
	appCfg.last_project = rec.registry;
	return rec;
}

function PruneRecentProjects(core) {
	const appCfg = core.APP_CONFIG;
	const kept = [];
	const seen = new Set();
	for (const r of (appCfg.recent_projects || [])) {
		if (!r || !r.path) continue;
		const rec = ReadProjectRecord(r.path);
		if (!rec) continue;
		if (!fs.existsSync(rec.workspace) || !fs.statSync(rec.workspace).isDirectory()) continue;
		if (seen.has(rec.key)) continue;
		seen.add(rec.key);
		kept.push({
			path: rec.registry,
			key: rec.key,
			name: rec.name,
			workspace: rec.workspace,
			updated_at: r.updated_at || '',
			recent_sessions: CapRecent(PruneSessionEntryList(r.recent_sessions || []))
		});
	}
	appCfg.recent_projects = CapRecent(kept);
	if (appCfg.last_project) {
		const lastRec = ReadProjectRecord(appCfg.last_project);
		if (!lastRec || !fs.existsSync(lastRec.workspace)) {
			appCfg.last_project = kept.length ? kept[0].path : '';
		} else {
			appCfg.last_project = lastRec.registry;
		}
	}
	return appCfg.recent_projects;
}

/** Home (non-project) entries from the pruned recent session list. */
function RecentChatSessions(core) {
	return PruneRecent(core);
}

/**
 * Recent projects with their recent chat sessions nested underneath.
 * Outer order follows recent_projects; inner order is per-project recent_sessions.
 */
function RecentProjectSessionGroups(core) {
	const projects = PruneRecentProjects(core);
	return projects.map((p) => ({
		project: p,
		sessions: Array.isArray(p.recent_sessions) ? p.recent_sessions : []
	}));
}

function NormalizeMessageNewlines(messages) {
	if (!Array.isArray(messages)) return messages;
	return messages.map((m) => {
		if (!m || typeof m !== 'object' || typeof m.content !== 'string') return m;
		if (!m.content.includes('[AI_UI_BR]')) return m;
		return { ...m, content: m.content.replaceAll('[AI_UI_BR]', '\n') };
	});
}

function WriteSessionFiles(core, messages, compaction) {
	if (!HasConversation(messages)) return { dir: '', meta: null, skipped: true };
	if (!sessionId) sessionId = NewSessionId();
	const dir = CurrentSessionDir();
	EnsureDir(dir);
	const loc = ContextFromSessionDir(dir);
	const now = new Date().toISOString();
	const metaPath = path.join(dir, 'meta.json');
	const prev = ReadJson(metaPath, null) || {};
	const meta = {
		id: sessionId,
		title: TitleFromMessages(messages),
		created_at: prev.created_at || now,
		updated_at: now,
		context: loc.context,
		project_key: loc.project_key || null,
		project_path: loc.project_path || null
	};
	// undefined → keep previous; null/empty → clear; object → write (normalized).
	const comp = NormalizeCompaction(
		compaction === undefined
			? (prev.compaction && typeof prev.compaction === 'object' ? prev.compaction : null)
			: compaction
	);
	if (comp) meta.compaction = comp;
	WriteJson(path.join(dir, 'chat.json'), NormalizeMessageNewlines(messages));
	WriteJson(metaPath, meta);
	if (loc.context === 'project' && loc.project_key) {
		SetProjectLastSession(sessionId, loc.project_key);
	}
	TouchRecent(core, dir, meta);
	SaveActiveConfig(core);
	return { dir: NormPath(dir), meta };
}

function NormalizeCompaction(compaction) {
	if (compaction == null || typeof compaction !== 'object') return null;
	const summary = String(compaction.summary || '');
	const count = Number(
		compaction.compact_count != null ? compaction.compact_count : 0
	);
	const until = Number(
		compaction.summarized_until != null
			? compaction.summarized_until
			: (compaction.compact_until != null ? compaction.compact_until : 0)
	);
	if (!summary && !(count > 0) && !(until > 0)) return null;
	return {
		summary,
		summarized_until: until,
		compact_count: count
	};
}

function ResolveSessionDir(selectedPath) {
	if (!selectedPath) return '';
	const p = selectedPath;
	if (fs.existsSync(p) && fs.statSync(p).isFile()) {
		return path.basename(p) === 'chat.json' ? path.dirname(p) : '';
	}
	return fs.existsSync(path.join(p, 'chat.json')) ? p : '';
}

function LoadSessionFromDir(dir) {
	const messages = ReadJson(path.join(dir, 'chat.json'), []);
	if (!Array.isArray(messages)) throw new Error('Invalid chat.json');
	const meta = ReadJson(path.join(dir, 'meta.json'), {});
	sessionId = meta.id || path.basename(dir);
	return { messages, meta, dir: NormPath(dir) };
}

function TryLoadConversation(dir) {
	if (!dir || !fs.existsSync(path.join(dir, 'chat.json'))) return null;
	try {
		const loaded = LoadSessionFromDir(dir);
		return HasConversation(loaded.messages) ? loaded : null;
	} catch (err) {
		return null;
	}
}

function GetState() {
	return {
		session_id: sessionId,
		project_key: projectKey,
		workspace: workspacePath,
		session_dir: CurrentSessionDir().replace(/\\/g, '/'),
		label: CurrentContextLabel()
	};
}

function SetHomeContext() {
	projectKey = '';
	workspacePath = '';
}

function SetProjectContext(absPath, key) {
	workspacePath = NormPath(absPath);
	projectKey = key || ProjectKey(workspacePath);
}

function SetSessionId(id) {
	sessionId = id || '';
	if (!sessionId) sessionDirOverride = '';
}

module.exports = {
	RECENT_MAX,
	GLOBAL_ONLY_APP,
	SetAiuiHome,
	GetAiuiHome,
	EnsureHomeLayout,
	ProjectKey,
	NewSessionId,
	NormPath,
	CurrentSessionDir,
	CurrentContextLabel,
	ContextFromSessionDir,
	SessionsRoot,
	EnsureProjectDirs,
	GetProjectScopedConfigs,
	ApplyProjectScopedConfigs,
	LoadOrSeedProjectConfig,
	SaveActiveConfig,
	HasConversation,
	TitleFromMessages,
	PruneRecent,
	ClearProjectResume,
	SetProjectLastSession,
	ClearResumePointers,
	TouchRecent,
	TouchRecentProject,
	PruneRecentProjects,
	RecentChatSessions,
	RecentProjectSessionGroups,
	ListHomeSessions,
	ListProjectSessions,
	RemoveRecentSession,
	RemoveRecentProject,
	SessionDirCandidate,
	WriteSessionFiles,
	ResolveSessionDir,
	LoadSessionFromDir,
	TryLoadConversation,
	GetState,
	SetHomeContext,
	SetProjectContext,
	SetSessionId,
	SetSessionDirOverride,
	ReadJson,
	WriteJson,
	ProjectDir,
	ProjectsRoot,
	IsProjectRegistryPath,
	ReadProjectRecord,
	get sessionId() { return sessionId; },
	get projectKey() { return projectKey; },
	get workspacePath() { return workspacePath; }
};
