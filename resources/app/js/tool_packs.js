import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import {
	EncodeHTML, EngineBusy, SyncSessionMenuLock, GuardEngineIdle
} from './util.js';

function NormPermMode(raw) {
	return String(raw || 'ask').trim().toLowerCase() === 'auto' ? 'auto' : 'ask';
}

function DefaultNativePacks(tools_file) {
	const mode = tools_file == 'roleplay' ? 'custom' : (tools_file || 'custom');
	const enable_math = (mode === 'math' || mode === 'merge' || mode === 'custom');
	const enable_web = (mode === 'research' || mode === 'merge' || mode === 'custom');
	return [
		{ id: 'native:utils', kind: 'native', name: 'Utils', enabled: true, disabled_tools: [], permission_mode: 'auto' },
		{ id: 'native:math', kind: 'native', name: 'Math', enabled: enable_math, disabled_tools: [], permission_mode: 'ask' },
		{ id: 'native:web', kind: 'native', name: 'Web', enabled: enable_web, disabled_tools: [], permission_mode: 'ask' },
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
	pack.disabled_tools = disabled;
	pack.disabled_defaults_applied = applied;
	pack.name = 'Coding';
	return pack;
}

export function NormalizeMergedPacks(packs, tools_file) {
	const natives = DefaultNativePacks(tools_file);
	const byId = {};
	const mcp = [];
	if (Array.isArray(packs)) {
		for (const p of packs) {
			if (!p) continue;
			if (p.kind === 'mcp') {
				mcp.push({
					id: String(p.id || ''),
					kind: 'mcp',
					name: String(p.name || p.server_id || p.id || 'MCP'),
					enabled: !!p.enabled,
					command: String(p.command || ''),
					args: Array.isArray(p.args) ? p.args.slice() : [],
					cwd: String(p.cwd || ''),
					env: (p.env && typeof p.env === 'object') ? Object.assign({}, p.env) : {},
					server_id: String(p.server_id || String(p.id || '').replace(/^mcp:/, '') || ''),
					disabled_tools: Array.isArray(p.disabled_tools) ? p.disabled_tools.slice() : [],
					permission_mode: NormPermMode(p.permission_mode || p.permissionMode)
				});
			} else {
				byId[String(p.id || '')] = {
					enabled: !!p.enabled,
					disabled_tools: Array.isArray(p.disabled_tools) ? p.disabled_tools.slice() : [],
					permission_mode: NormPermMode(p.permission_mode || p.permissionMode),
					sandbox: p.sandbox !== false,
					disabled_defaults_applied: Array.isArray(p.disabled_defaults_applied)
						? p.disabled_defaults_applied.slice()
						: []
				};
			}
		}
	}
	const nativeOut = natives.map((p) => {
		const prev = byId[p.id];
		const out = Object.assign({}, p, {
			enabled: prev ? prev.enabled : p.enabled,
			disabled_tools: prev ? prev.disabled_tools : (p.disabled_tools || []),
			permission_mode: prev ? prev.permission_mode : NormPermMode(p.permission_mode),
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
	return nativeOut.concat(mcp);
}

function PackStatus(pack) {
	if (!pack || pack.kind !== 'mcp') {
		return { cls: 'pack-native', suffix: '', title: 'Click to edit pack settings', busy: false };
	}
	const running = Array.isArray(state.mcp_running) && state.mcp_running.indexOf(pack.id) >= 0;
	const err = state.mcp_errors && state.mcp_errors[pack.id] ? String(state.mcp_errors[pack.id]) : '';
	const pending = state.mcp_pending && state.mcp_pending[pack.id];
	if (pending === 'starting' && !running) {
		return { cls: 'pack-starting', suffix: ' (starting)', title: 'Starting MCP server...', busy: true };
	}
	if (err && pack.enabled && !running) {
		return { cls: 'pack-error', suffix: ' (error)', title: err, busy: false };
	}
	if (running) {
		return { cls: 'pack-running', suffix: '', title: 'MCP server running - click to edit', busy: false };
	}
	if (pack.enabled) {
		return { cls: 'pack-starting', suffix: '', title: 'Enabled but not running - click to edit', busy: false };
	}
	return { cls: 'pack-stopped', suffix: '', title: 'MCP server stopped - click to edit', busy: false };
}

export function RenderToolPacks() {
	const packs = NormalizeMergedPacks(state.tool_packs, state.tools_file);
	state.tool_packs = packs;
	const $list = $('#tool_packs_list');
	$list.empty();
	const engBusy = EngineBusy();
	packs.forEach((pack, idx) => {
		const kind = pack.kind === 'mcp' ? 'MCP' : 'Native';
		const checked = pack.enabled ? 'checked' : '';
		const st = PackStatus(pack);
		// Only enable-toggles lock while busy - settings dialog stays openable.
		const busy = (st.busy || engBusy) ? 'disabled' : '';
		$list.append(`
			<div class="tool_pack_row" data-idx="${idx}" title="${EncodeHTML(st.title)}">
				<span class="tool_pack_kind">${kind}</span>
				<span class="tool_pack_name ${st.cls}">${EncodeHTML(pack.name + st.suffix)}</span>
				<input type="checkbox" class="tool_pack_enable" data-idx="${idx}" ${checked} ${busy} title="Enable for model">
			</div>
		`);
	});
	UpdatePackBusyControls();
	SyncSessionMenuLock();
}

export function UpdatePackBusyControls() {
	const busy = EngineBusy();
	$('#mcp_refresh_btn').prop('disabled', busy);
	const $save = $('#pack_dialog_save');
	if ($save.length && !$('#pack_dialog').hasClass('hidden')) {
		$save.prop('disabled', busy);
	}
}

export function PushToolPacksNow(opts) {
	opts = opts || {};
	if (!GuardEngineIdle()) return;
	const packs = CollectToolPacksFromDom();
	const payload = {
		temp_format: state.temp_format,
		tool_packs: packs,
		firecrawl_api_key: state.firecrawl_api_key || ''
	};
	if (opts.force_mcp_ids && opts.force_mcp_ids.length) {
		payload.force_mcp_ids = opts.force_mcp_ids;
	}
	ipcRenderer.send('config-tools', payload);
	state.tool_packs = packs;
	state.tool_packs_applied = JSON.parse(JSON.stringify(packs));
}

export function CollectToolPacksFromDom() {
	const packs = NormalizeMergedPacks(state.tool_packs, state.tools_file);
	packs.forEach((pack, idx) => {
		const $cb = $(`.tool_pack_enable[data-idx="${idx}"]`);
		if ($cb.length) pack.enabled = $cb.is(':checked');
	});
	state.tool_packs = packs;
	return packs;
}

var packDialogIdx = -1;
var packDialogMode = '';
var packDialogTab = 'settings';

export function GetPackDialogIdx() {
	return packDialogIdx;
}

export function GetPackDialogTab() {
	return packDialogTab;
}

export function ShowPackDialogTab(tab) {
	packDialogTab = tab === 'tools' ? 'tools' : 'settings';
	if (packDialogTab === 'tools') {
		$('#pack_dialog_settings').hide();
		$('#pack_dialog_tools').removeClass('hidden').show();
		$('#pack_tab_tools_btn').addClass('active').css('color', '#c4c4cf');
		$('#pack_tab_settings_btn').removeClass('active').css('color', '#868687');
		RenderPackToolsList();
	} else {
		$('#pack_dialog_tools').addClass('hidden').hide();
		$('#pack_dialog_settings').show();
		$('#pack_tab_settings_btn').addClass('active').css('color', '#c4c4cf');
		$('#pack_tab_tools_btn').removeClass('active').css('color', '#868687');
	}
}

export function ClosePackDialog() {
	packDialogIdx = -1;
	packDialogMode = '';
	packDialogTab = 'settings';
	$('#pack_dialog').addClass('hidden');
	$('#pack_dialog_settings').empty();
	$('#pack_tools_list').empty();
	$('#pack_tools_note').text('');
	$('#pack_dialog_save').show();
}

function CollectDisabledToolsFromDom() {
	const disabled = [];
	$('#pack_tools_list .pack_tool_enable').each(function() {
		if (!$(this).is(':checked')) disabled.push($(this).attr('data-name'));
	});
	return disabled;
}

export function ApplyDisabledToolsToPack(pack) {
	if (!pack) return pack;
	if ($('#pack_tools_list .pack_tool_enable').length) {
		pack.disabled_tools = CollectDisabledToolsFromDom();
	} else if (!Array.isArray(pack.disabled_tools)) {
		pack.disabled_tools = [];
	}
	return pack;
}

export function RenderPackToolsList() {
	const packs = state.tool_packs || [];
	const pack = packs[packDialogIdx];
	const $list = $('#pack_tools_list');
	$list.empty();
	if (!pack) return;

	const catalog = (state.tool_catalog && state.tool_catalog[pack.id]) || [];
	const disabled = new Set(pack.disabled_tools || []);
	const running = pack.kind === 'mcp' && Array.isArray(state.mcp_running) && state.mcp_running.indexOf(pack.id) >= 0;

	if (!catalog.length) {
		$('#pack_tools_note').text(
			pack.kind === 'mcp'
				? (pack.enabled
					? 'No tools yet. Enable the pack and Apply so the MCP server can start; the tool list loads when it connects.'
					: 'Enable this MCP pack and Apply to start the server and load its tools.')
				: 'No tools found for this pack.'
		);
		return;
	}

	if (pack.kind === 'mcp' && !running) {
		$('#pack_tools_note').text('Showing last known tools (server not running). Unchecked tools stay hidden from the model.');
	} else {
		$('#pack_tools_note').text('Unchecked tools stay installed but are hidden from the model. Hover a name for its description.');
	}
	catalog.forEach((tool) => {
		const name = tool.name || '';
		const on = tool.hasOwnProperty('enabled')
			? !!tool.enabled
			: !(disabled.has(name));
		const checked = on ? 'checked' : '';
		const desc = tool.description || '';
		$list.append(`
			<div class="pack_tool_row">
				<input type="checkbox" class="pack_tool_enable" data-name="${EncodeHTML(name)}" ${checked}>
				<div class="pack_tool_meta">
					<span class="pack_tool_name" title="${EncodeHTML(desc)}">${EncodeHTML(name)}</span>
				</div>
			</div>
		`);
	});
}

function AutoAllowSelectHtml(pack) {
	const isAuto = NormPermMode(pack.permission_mode) === 'auto';
	return `
		<span class="cfg_lbl">Auto-allow Tools <img class="help_icon" width="16" height="16" title="When True, tools from this pack run without a permission prompt. When False, each call asks Deny / Allow / Auto-allow." /></span>
		<select id="pack_auto_allow" class="cfg_drp">
			<option value="1"${isAuto ? ' selected' : ''}>True</option>
			<option value="0"${isAuto ? '' : ' selected'}>False</option>
		</select>
	`;
}

export function OpenPackDialog(idx) {
	const packs = CollectToolPacksFromDom();
	const pack = packs[idx];
	if (!pack) return;
	packDialogIdx = idx;
	packDialogMode = pack.kind === 'mcp' ? 'mcp' : pack.id;
	$('#pack_dialog_title').text(pack.name);
	const $settings = $('#pack_dialog_settings');
	$settings.empty();

	if (pack.kind === 'mcp') {
		$settings.append(`
			<span class="cfg_lbl">Server ID <img class="help_icon" width="16" height="16" title="Key used in ~/.aiui/mcp.json. Read-only here; rename or add servers with SHOW MCP.JSON." /></span>
			<input type="text" id="pack_mcp_id" class="cfg_txt" value="${EncodeHTML(pack.server_id || '')}" readonly>
			<span class="cfg_lbl">Display Name <img class="help_icon" width="16" height="16" title="Label shown in the Tool Packs list." /></span>
			<input type="text" id="pack_mcp_name" class="cfg_txt" value="${EncodeHTML(pack.name || '')}">
			<span class="cfg_lbl">Command <img class="help_icon" width="16" height="16" title="Executable used to start the MCP server (for example a Python path)." /></span>
			<input type="text" id="pack_mcp_command" class="cfg_txt" value="${EncodeHTML(pack.command || '')}" placeholder="/path/to/python">
			<span class="cfg_lbl">Args (JSON array) <img class="help_icon" width="16" height="16" title='Command-line arguments as a JSON array, e.g. ["-m","cait.server"].' /></span>
			<input type="text" id="pack_mcp_args" class="cfg_txt" value="${EncodeHTML(JSON.stringify(pack.args || []))}">
			<span class="cfg_lbl">Working Directory <img class="help_icon" width="16" height="16" title="Optional cwd for the MCP process (the server install). Tokens like \${workspaceFolder} expand at start (open project, or ~/.aiui on Home). Leave blank to use the default." /></span>
			<input type="text" id="pack_mcp_cwd" class="cfg_txt" value="${EncodeHTML(pack.cwd || '')}" placeholder="/path/to/CAIT">
			<span class="cfg_lbl">Env (JSON object) <img class="help_icon" width="16" height="16" title='Environment variables as JSON. \${workspaceFolder} expands at start. Example: {"PYTHONPATH":"/path/to/CAIT","CAIT_WORKSPACE":"\${workspaceFolder}"}.' /></span>
			<input type="text" id="pack_mcp_env" class="cfg_txt" value="${EncodeHTML(JSON.stringify(pack.env || {}))}">
			${AutoAllowSelectHtml(pack)}
		`);
		$('#pack_dialog_save').show();
		$('#pack_dialog_cancel').text('CANCEL');
	} else if (pack.id === 'native:web') {
		$settings.append(`
			<span class="cfg_lbl">Firecrawl API Key <img class="help_icon" width="16" height="16" title="Required for web_search. get_web_page still works without a key via HTTP fallback." /></span>
			<input type="password" id="pack_firecrawl_key" class="cfg_txt" value="${EncodeHTML(state.firecrawl_api_key || '')}" autocomplete="off">
			${AutoAllowSelectHtml(pack)}
		`);
		$('#pack_dialog_save').show();
		$('#pack_dialog_cancel').text('CANCEL');
	} else if (pack.id === 'native:coding') {
		const sandOn = pack.sandbox !== false;
		$settings.append(`
			<span class="cfg_lbl">Sandbox Paths <img class="help_icon" width="16" height="16" title="When True, path-based coding tools may only touch paths under the open project (and ~/.aiui/skills, ~/.aiui/rules, ~/.aiui/agents or ~/.aiui on Home). When False, absolute paths elsewhere are allowed; relative paths still use that root. run_command always starts in the workspace root and is not path-sandboxed." /></span>
			<select id="pack_files_sandbox" class="cfg_drp">
				<option value="1"${sandOn ? ' selected' : ''}>True</option>
				<option value="0"${sandOn ? '' : ' selected'}>False</option>
			</select>
			${AutoAllowSelectHtml(pack)}
		`);
		$('#pack_dialog_save').show();
		$('#pack_dialog_cancel').text('CANCEL');
	} else {
		$settings.append(`${AutoAllowSelectHtml(pack)}`);
		$('#pack_dialog_save').show();
		$('#pack_dialog_cancel').text('CLOSE');
	}

	$('#pack_dialog').removeClass('hidden');
	ShowPackDialogTab('settings');
	UpdatePackBusyControls();
}

export function SavePackDialog() {
	const packs = CollectToolPacksFromDom();
	const pack = packs[packDialogIdx];
	if (!pack) {
		ClosePackDialog();
		return;
	}
	if (!GuardEngineIdle('Wait for the current task to finish before saving pack settings.')) return;
	ApplyDisabledToolsToPack(pack);
	if ($('#pack_auto_allow').length) {
		pack.permission_mode = $('#pack_auto_allow').val() === '1' ? 'auto' : 'ask';
	}
	if ($('#pack_files_sandbox').length) {
		pack.sandbox = $('#pack_files_sandbox').val() === '1';
	}

	if (packDialogMode === 'mcp') {
		let args = pack.args || [];
		let env = pack.env || {};
		if ($('#pack_mcp_args').length) {
			try {
				args = JSON.parse($('#pack_mcp_args').val() || '[]');
				if (!Array.isArray(args)) throw new Error('args');
			} catch (e) {
				ipcRenderer.send('show-alert', { msg: 'MCP args must be a JSON array.' });
				return;
			}
			try {
				env = JSON.parse($('#pack_mcp_env').val() || '{}');
				if (!env || typeof env !== 'object' || Array.isArray(env)) throw new Error('env');
			} catch (e) {
				ipcRenderer.send('show-alert', { msg: 'MCP env must be a JSON object.' });
				return;
			}
			pack.name = $('#pack_mcp_name').val().trim() || pack.name;
			pack.command = $('#pack_mcp_command').val().trim();
			pack.args = args;
			pack.cwd = $('#pack_mcp_cwd').val().trim();
			pack.env = env;
		}
		const mcp_server = {
			id: pack.id,
			kind: 'mcp',
			server_id: pack.server_id,
			name: pack.name,
			command: pack.command,
			args: pack.args,
			cwd: pack.cwd,
			env: pack.env,
			enabled: pack.enabled,
			disabled_tools: pack.disabled_tools || [],
			permission_mode: pack.permission_mode || 'ask'
		};
		state.tool_packs = packs;
		ipcRenderer.send('config-tools', {
			temp_format: state.temp_format,
			tool_packs: packs,
			mcp_server,
			firecrawl_api_key: state.firecrawl_api_key || ''
		});
		state.tool_packs_applied = JSON.parse(JSON.stringify(packs));
		ClosePackDialog();
		ipcRenderer.send('show-alert', { msg: 'Pack settings saved.' });
		return;
	}

	if (packDialogMode === 'native:web' && $('#pack_firecrawl_key').length) {
		state.firecrawl_api_key = $('#pack_firecrawl_key').val().trim();
	}

	state.tool_packs = packs;
	ipcRenderer.send('config-tools', {
		temp_format: state.temp_format,
		tool_packs: packs,
		firecrawl_api_key: state.firecrawl_api_key || ''
	});
	state.tool_packs_applied = JSON.parse(JSON.stringify(packs));
	ClosePackDialog();
	ipcRenderer.send('show-alert', { msg: 'Pack settings saved.' });
}

export function WirePackEvents() {
	$('#tool_packs_list').on('click', '.tool_pack_row', function(e) {
		if ($(e.target).is('input.tool_pack_enable')) return;
		OpenPackDialog(parseInt($(this).attr('data-idx'), 10));
	});
	$('#tool_packs_list').on('click', '.tool_pack_enable', function(e) {
		e.stopPropagation();
	});
	$('#tool_packs_list').on('change', '.tool_pack_enable', function() {
		// Disabled checkboxes should not fire; belt-and-suspenders if busy state lagged.
		if (EngineBusy() || $(this).prop('disabled')) {
			const idx = parseInt($(this).attr('data-idx'), 10);
			const applied = (state.tool_packs_applied || state.tool_packs || [])[idx];
			$(this).prop('checked', !!(applied && applied.enabled));
			return;
		}
		const idx = parseInt($(this).attr('data-idx'), 10);
		const packs = CollectToolPacksFromDom();
		const pack = packs[idx];
		if (!pack) return;
		const forceIds = [];
		if (pack.kind === 'mcp') {
			if (!state.mcp_pending) state.mcp_pending = {};
			if (!state.mcp_errors) state.mcp_errors = {};
			if (pack.enabled) {
				state.mcp_pending[pack.id] = 'starting';
				delete state.mcp_errors[pack.id];
				forceIds.push(pack.id);
			} else {
				delete state.mcp_pending[pack.id];
				delete state.mcp_errors[pack.id];
			}
		}
		RenderToolPacks();
		PushToolPacksNow({ force_mcp_ids: forceIds });
	});
	$('#mcp_refresh_btn').on('click', async function() {
		if (!GuardEngineIdle('Wait for the current task to finish before refreshing packs.')) return;
		try {
			const res = await window.aiuiAPI.refreshMcpPacks();
			if (res && res.tool_packs) {
				state.tool_packs = NormalizeMergedPacks(res.tool_packs, state.tools_file);
				state.tool_packs_applied = JSON.parse(JSON.stringify(state.tool_packs));
				RenderToolPacks();
			}
			if (res && res.warnings && res.warnings.length) {
				ipcRenderer.send('show-alert', {
					type: 'error',
					msg: res.warnings.join('\n')
				});
			} else {
				ipcRenderer.send('show-alert', { msg: 'Tool packs refreshed from mcp.json.' });
			}
		} catch (err) {
			ipcRenderer.send('show-alert', { type: 'error', msg: String(err.message || err) });
		}
	});
	$('#pack_tab_settings_btn').on('click', function() {
		ShowPackDialogTab('settings');
	});
	$('#pack_tab_tools_btn').on('click', function() {
		ShowPackDialogTab('tools');
	});
	$('#pack_dialog_save').on('click', function() {
		SavePackDialog();
	});
	$('#pack_dialog_cancel').on('click', function() {
		ClosePackDialog();
	});
	$('#pack_dialog').on('click', function(e) {
		if (e.target === this) ClosePackDialog();
	});
}
