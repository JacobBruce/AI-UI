/** ~/.aiui/mcp.json - installed MCP servers (Cursor-compatible shape + enabled flag). */

const fs = require('fs');
const path = require('path');

function DefaultMcpConfig() {
	return {
		mcpServers: {}
	};
}

function GetMcpPath(aiuiHome) {
	return path.join(aiuiHome, 'mcp.json');
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
	fs.writeFileSync(file, JSON.stringify(obj, null, '\t') + '\n', 'utf8');
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

function NormalizeServer(id, cfg) {
	const c = cfg && typeof cfg === 'object' ? cfg : {};
	const args = Array.isArray(c.args) ? c.args.slice() : [];
	const env = (c.env && typeof c.env === 'object' && !Array.isArray(c.env)) ? Object.assign({}, c.env) : {};
	const pm = String(c.permissionMode || c.permission_mode || 'ask').trim().toLowerCase();
	return {
		command: String(c.command || ''),
		args,
		cwd: String(c.cwd || ''),
		env,
		enabled: !!c.enabled,
		name: String(c.name || id),
		disabledTools: NormalizeDisabledTools(c.disabledTools || c.disabled_tools),
		permissionMode: pm === 'auto' ? 'auto' : 'ask'
	};
}

function NormalizeMcpConfig(raw) {
	const base = DefaultMcpConfig();
	const servers = (raw && raw.mcpServers && typeof raw.mcpServers === 'object') ? raw.mcpServers : null;
	if (!servers) return Object.assign({}, base, { _warnings: [] });
	const out = { mcpServers: {} };
	const warnings = [];
	const seenLower = new Map(); // lower(id) -> kept id
	for (const id of Object.keys(servers)) {
		const key = String(id || '').trim();
		if (!key) {
			warnings.push('Ignored MCP server with empty id');
			continue;
		}
		const lower = key.toLowerCase();
		if (seenLower.has(lower)) {
			warnings.push('Duplicate MCP server id ignored: "' + key + '" (conflicts with "' + seenLower.get(lower) + '")');
			continue;
		}
		seenLower.set(lower, key);
		out.mcpServers[key] = NormalizeServer(key, servers[key]);
	}
	// Empty mcpServers is valid (do not re-seed defaults)
	out._warnings = warnings;
	return out;
}

function TakeWarnings(cfg) {
	const w = (cfg && Array.isArray(cfg._warnings)) ? cfg._warnings.slice() : [];
	if (cfg && cfg._warnings) delete cfg._warnings;
	return w;
}

function LoadMcpConfig(aiuiHome) {
	const file = GetMcpPath(aiuiHome);
	if (!fs.existsSync(file)) {
		const fresh = DefaultMcpConfig();
		WriteJson(file, fresh);
		return fresh;
	}
	const cfg = NormalizeMcpConfig(ReadJson(file, DefaultMcpConfig()));
	const warnings = TakeWarnings(cfg);
	if (warnings.length) cfg._warnings = warnings;
	return cfg;
}

function SaveMcpConfig(aiuiHome, cfg) {
	const normalized = NormalizeMcpConfig(cfg);
	const warnings = TakeWarnings(normalized);
	// Strip non-persistent fields before write
	const toWrite = { mcpServers: normalized.mcpServers };
	WriteJson(GetMcpPath(aiuiHome), toWrite);
	if (warnings.length) normalized._warnings = warnings;
	return normalized;
}

/** Convert mcp.json servers into tool-pack rows for UI/engine. */
function ServersToPacks(mcpConfig) {
	const servers = (mcpConfig && mcpConfig.mcpServers) || {};
	return Object.keys(servers).map((id) => {
		const s = NormalizeServer(id, servers[id]);
		return {
			id: 'mcp:' + id,
			kind: 'mcp',
			name: s.name || id,
			enabled: s.enabled,
			command: s.command,
			args: s.args,
			cwd: s.cwd,
			env: s.env,
			server_id: id,
			disabled_tools: s.disabledTools.slice(),
			permission_mode: s.permissionMode
		};
	});
}

/** Merge edited pack back into mcp.json object. Rejects empty ids. */
function UpsertPack(mcpConfig, pack) {
	const cfg = NormalizeMcpConfig(mcpConfig);
	TakeWarnings(cfg);
	const serverId = String(pack.server_id || (pack.id || '').replace(/^mcp:/, '') || '').trim();
	if (!serverId) return cfg;
	// Case-insensitive conflict with a different key — do not overwrite the other server.
	const lower = serverId.toLowerCase();
	for (const existing of Object.keys(cfg.mcpServers)) {
		if (existing.toLowerCase() === lower && existing !== serverId) {
			cfg._warnings = ['Cannot install MCP server "' + serverId + '": id conflicts with "' + existing + '"'];
			return cfg;
		}
	}
	const prev = cfg.mcpServers[serverId] || {};
	const disabled = pack.hasOwnProperty('disabled_tools')
		? pack.disabled_tools
		: (pack.hasOwnProperty('disabledTools') ? pack.disabledTools : prev.disabledTools);
	const perm = pack.hasOwnProperty('permission_mode')
		? pack.permission_mode
		: (pack.hasOwnProperty('permissionMode') ? pack.permissionMode : prev.permissionMode);
	cfg.mcpServers[serverId] = NormalizeServer(serverId, {
		command: pack.hasOwnProperty('command') ? pack.command : prev.command,
		args: pack.hasOwnProperty('args') ? pack.args : prev.args,
		cwd: pack.hasOwnProperty('cwd') ? pack.cwd : prev.cwd,
		env: pack.hasOwnProperty('env') ? pack.env : prev.env,
		enabled: pack.hasOwnProperty('enabled') ? pack.enabled : prev.enabled,
		name: pack.name || prev.name || serverId,
		disabledTools: disabled,
		permissionMode: perm
	});
	return cfg;
}

function SetServerEnabled(mcpConfig, serverId, enabled) {
	const cfg = NormalizeMcpConfig(mcpConfig);
	const id = String(serverId || '').trim();
	if (!id || !cfg.mcpServers[id]) return cfg;
	cfg.mcpServers[id].enabled = !!enabled;
	return cfg;
}

/**
 * One-time migration: pull MCP entries out of legacy AI tool_packs into mcp.json.
 * Returns { mcpConfig, nativePacks } where nativePacks is a cleaned tool_packs array.
 */
function MigrateFromToolPacks(aiuiHome, toolPacks, existingMcp) {
	const packs = Array.isArray(toolPacks) ? toolPacks : [];
	const mcpPacks = packs.filter((p) => p && p.kind === 'mcp');
	const nativePacks = packs.filter((p) => p && p.kind === 'native');
	let mcpConfig = existingMcp ? NormalizeMcpConfig(existingMcp) : LoadMcpConfig(aiuiHome);
	const fileExisted = fs.existsSync(GetMcpPath(aiuiHome));
	const emptyFile = !fileExisted || Object.keys(mcpConfig.mcpServers || {}).length === 0;

	if (mcpPacks.length > 0 && emptyFile) {
		mcpConfig = { mcpServers: {} };
		const seenLower = new Map();
		const warnings = [];
		for (const p of mcpPacks) {
			const sid = String(p.server_id || String(p.id || '').replace(/^mcp:/, '') || p.name || '').trim();
			if (!sid) continue;
			const lower = sid.toLowerCase();
			if (seenLower.has(lower)) {
				warnings.push('Duplicate MCP server id skipped during migration: "' + sid + '"');
				continue;
			}
			seenLower.set(lower, sid);
			mcpConfig.mcpServers[sid] = NormalizeServer(sid, p);
		}
		SaveMcpConfig(aiuiHome, mcpConfig);
		if (warnings.length) mcpConfig._warnings = warnings;
	}
	return { mcpConfig, nativePacks };
}

module.exports = {
	DefaultMcpConfig,
	GetMcpPath,
	LoadMcpConfig,
	SaveMcpConfig,
	NormalizeMcpConfig,
	TakeWarnings,
	ServersToPacks,
	UpsertPack,
	SetServerEnabled,
	MigrateFromToolPacks
};
