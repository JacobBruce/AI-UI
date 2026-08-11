import { ipcRenderer } from './ipc.js';
import { EncodeHTML, WireOverlayDismiss } from './util.js';

let pickList = [];
let selectedPath = '';
let loading = false;

function FormatSessionWhen(iso) {
	if (!iso) return '';
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return '';
	try {
		return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
	} catch (err) {
		return String(iso).slice(0, 16).replace('T', ' ');
	}
}

export function CloseSessionPicker() {
	selectedPath = '';
	pickList = [];
	loading = false;
	$('#session_dialog').addClass('hidden');
	$('#session_pick_list').empty();
	$('#session_pick_note').text('');
	$('#session_dialog_open').prop('disabled', true);
}

function RenderSessionPickList() {
	const $list = $('#session_pick_list');
	$list.empty();
	if (!pickList.length) {
		$('#session_pick_note').text('No sessions found.');
		$('#session_dialog_open').prop('disabled', true);
		return;
	}
	$('#session_pick_note').text('Select a session. Double-click to open immediately.');
	pickList.forEach((s) => {
		const path = String(s.path || '');
		const title = String(s.title || 'Session').replace(/[\r\n]+/g, ' ').trim() || 'Session';
		const when = FormatSessionWhen(s.updated_at);
		const sel = path && path === selectedPath ? ' selected' : '';
		$list.append(
			'<div class="session_pick_row' + sel + '" data-path="' + EncodeHTML(path) + '" title="' + EncodeHTML(path) + '">' +
				'<span class="session_pick_title">' + EncodeHTML(title) + '</span>' +
				(when ? '<span class="session_pick_date">' + EncodeHTML(when) + '</span>' : '') +
			'</div>'
		);
	});
	$('#session_dialog_open').prop('disabled', !selectedPath);
}

async function OpenSelectedSession() {
	if (loading || !selectedPath) return;
	loading = true;
	$('#session_dialog_open').prop('disabled', true);
	try {
		await ipcRenderer.invoke('session-load', selectedPath);
		CloseSessionPicker();
	} catch (err) {
		loading = false;
		$('#session_dialog_open').prop('disabled', !selectedPath);
		ipcRenderer.send('show-alert', {
			type: 'error',
			msg: String(err && err.message ? err.message : err)
		});
	}
}

export function ShowSessionPicker(payload) {
	const data = payload || {};
	pickList = Array.isArray(data.sessions) ? data.sessions : [];
	selectedPath = '';
	loading = false;
	$('#session_dialog_title').text(data.title || 'Load Session');
	RenderSessionPickList();
	$('#session_dialog').removeClass('hidden');
}

export function InitSessionPicker() {
	WireOverlayDismiss('#session_dialog', CloseSessionPicker);

	$('#session_dialog_cancel').on('click', function() {
		CloseSessionPicker();
	});

	$('#session_dialog_open').on('click', function() {
		OpenSelectedSession();
	});

	$('#session_pick_list').on('click', '.session_pick_row', function() {
		selectedPath = String($(this).attr('data-path') || '');
		$('#session_pick_list .session_pick_row').removeClass('selected');
		$(this).addClass('selected');
		$('#session_dialog_open').prop('disabled', !selectedPath || loading);
	});

	$('#session_pick_list').on('dblclick', '.session_pick_row', function() {
		selectedPath = String($(this).attr('data-path') || '');
		OpenSelectedSession();
	});

	ipcRenderer.on('session-picker', (event, payload) => {
		ShowSessionPicker(payload);
	});

	$(document).on('keydown.sessionPicker', function(e) {
		if (e.key !== 'Escape') return;
		if ($('#session_dialog').hasClass('hidden')) return;
		e.preventDefault();
		CloseSessionPicker();
	});
}
