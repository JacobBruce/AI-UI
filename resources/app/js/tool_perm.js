import { ipcRenderer } from './ipc.js';

let permQueue = [];
let permCurrent = null;
let permBusy = false;
let wired = false;

function EscapeHtml(txt) {
	return String(txt || '')
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;');
}

function BuildToolCallText(data) {
	const preview = data && data.args_preview != null ? String(data.args_preview).trim() : '';
	// Prefer raw preview when it looks like XML tool-call markup.
	if (preview.startsWith('<')) return preview;

	const tool = String((data && data.tool) || '?');
	let args = null;
	if (data && data.arguments != null && typeof data.arguments === 'object') {
		args = data.arguments;
	} else if (preview) {
		try {
			args = JSON.parse(preview);
		} catch (_e) {
			args = preview;
		}
	} else {
		args = {};
	}
	try {
		return JSON.stringify({ name: tool, arguments: args }, null, 2);
	} catch (_e) {
		return '{\n  "name": ' + JSON.stringify(tool) + ',\n  "arguments": ' + String(preview || '{}') + '\n}';
	}
}

/** Auto-detect json/xml (same idea as chat `highlightElement` on code blocks). */
function HighlightToolCode(el, text) {
	const raw = String(text || '');
	const node = el && el.length ? el[0] : el;
	if (!node) return;
	const $el = $(node);
	$el.removeAttr('data-highlighted').attr('class', 'hljs').text(raw);
	const hljs = window.hljs;
	if (hljs && typeof hljs.highlightElement === 'function') {
		try {
			hljs.highlightElement(node);
			return;
		} catch (_e) { /* fall through */ }
	}
	if (hljs && typeof hljs.highlightAuto === 'function') {
		try {
			const result = hljs.highlightAuto(raw, ['json', 'xml']);
			$el.html(result.value || EscapeHtml(raw));
			if (result.language) $el.addClass('language-' + result.language);
			return;
		} catch (_e) { /* fall through */ }
	}
	$el.html(EscapeHtml(raw));
}

function HideToolPermDialog() {
	$('#tool_perm_dialog').addClass('hidden');
	$('#tool_perm_code > code').removeAttr('data-highlighted').attr('class', 'hljs').empty();
	$('#tool_perm_summary').text('');
}

function PresentToolPermission(data) {
	permCurrent = data || {};
	const tool = EscapeHtml(String(permCurrent.tool || '?'));
	const pack = EscapeHtml(String(permCurrent.pack_name || permCurrent.pack_id || 'pack'));
	$('#tool_perm_summary').html('Allow tool <code>' + tool + '</code> from ' + pack + '?');
	HighlightToolCode($('#tool_perm_code > code'), BuildToolCallText(permCurrent));
	$('#tool_perm_dialog').removeClass('hidden');
	$('#tool_perm_allow').trigger('focus');
}

function DrainNextPermission() {
	if (permBusy) return;
	const next = permQueue.shift();
	if (!next) {
		permCurrent = null;
		HideToolPermDialog();
		return;
	}
	PresentToolPermission(next);
}

async function SendPermissionResult(requestId, action, packId) {
	await ipcRenderer.invoke('tool-permission-result', {
		request_id: requestId || '',
		action: action || 'deny',
		pack_id: packId || ''
	});
}

async function ResolveToolPermission(action) {
	if (permBusy || !permCurrent) return;
	permBusy = true;
	const req = permCurrent;
	permCurrent = null;
	HideToolPermDialog();
	try {
		if (action === 'stop') {
			const pending = [req].concat(permQueue);
			permQueue = [];
			for (let i = 0; i < pending.length; ++i) {
				await SendPermissionResult(pending[i].request_id, 'deny', pending[i].pack_id);
			}
			await ipcRenderer.invoke('stop-stream');
		} else {
			await SendPermissionResult(req.request_id, action, req.pack_id);
		}
	} finally {
		permBusy = false;
		DrainNextPermission();
	}
}

export function QueueToolPermission(data) {
	permQueue.push(data || {});
	if (!permCurrent && !permBusy) DrainNextPermission();
}

/** Deny any open/queued prompts (e.g. turn ended / engine stopped). */
export async function DismissToolPermissions() {
	if (!permCurrent && !permQueue.length) return;
	const pending = (permCurrent ? [permCurrent] : []).concat(permQueue);
	permCurrent = null;
	permQueue = [];
	HideToolPermDialog();
	permBusy = true;
	try {
		for (let i = 0; i < pending.length; ++i) {
			await SendPermissionResult(pending[i].request_id, 'deny', pending[i].pack_id);
		}
	} finally {
		permBusy = false;
	}
}

export function WireToolPermEvents() {
	if (wired) return;
	wired = true;

	$('#tool_perm_deny').on('click', function() {
		ResolveToolPermission('deny');
	});
	$('#tool_perm_allow').on('click', function() {
		ResolveToolPermission('allow');
	});
	$('#tool_perm_auto').on('click', function() {
		ResolveToolPermission('auto');
	});
	$('#tool_perm_stop').on('click', function() {
		ResolveToolPermission('stop');
	});

	// Do not dismiss on scrim click — permission needs an explicit choice.

	ipcRenderer.on('tool-permission', (event, payload) => {
		QueueToolPermission(payload || {});
	});

	ipcRenderer.on('ai-ready', () => {
		DismissToolPermissions();
	});
	ipcRenderer.on('init-ui', (event, payload) => {
		if (payload && payload.state === false) DismissToolPermissions();
	});

	$(document).on('keydown.tool_perm', function(e) {
		if ($('#tool_perm_dialog').hasClass('hidden')) return;
		if (e.key === 'Escape') {
			e.preventDefault();
			ResolveToolPermission('deny');
		}
	});
}
