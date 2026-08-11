import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import { EncodeHTML } from './util.js';

export function ToggleEditPrompt() {
	const $edit = $('#edit_dialog');
	if ($edit.hasClass('hidden')) {
		$('#prompt_files_dialog').addClass('hidden');
		$edit.removeClass('hidden');
		$('#prompt_txta').focus();
	} else {
		$edit.addClass('hidden');
	}
}

export function TogglePromptFiles() {
	$('#prompt_files_dialog').toggleClass('hidden');
}

function PromptPreviewLine(prompt_txt) {
	let one_line = String(prompt_txt || '').replace(/\s+/g, ' ').trim();
	if (one_line.length > state.max_ip_len) {
		one_line = one_line.substring(0, state.max_ip_len) + '...';
	}
	return one_line;
}

export function ApplyNewPrompt() {
	state.prompt = $('#prompt_txta').val();
	ipcRenderer.send('update-prompt', state.prompt.replaceAll("\n", "[AI_UI_BR]"));
	RefreshPrompt(state.prompt);
	ToggleEditPrompt();
}

function CollectEnabledIds(sel) {
	const ids = [];
	$(sel+' input[type=checkbox]:checked').each(function () {
		ids.push($(this).val());
	});
	return ids;
}

function StripMdName(name) {
	return String(name || '').replace(/\.agent\.md$/i, '').replace(/\.md$/i, '');
}

function RenderFileToggles(box_sel, items, enabled, label_key) {
	const $box = $(box_sel);
	$box.empty();
	if (!items || items.length === 0) {
		$box.append('<div class="pf_empty">(none found)</div>');
		return;
	}
	const en = new Set(enabled || []);
	for (let i = 0; i < items.length; ++i) {
		const it = items[i];
		const id = it.id;
		const label = StripMdName(it[label_key] || id);
		const checked = en.has(id) ? ' checked' : '';
		$box.append('<label><input type="checkbox" value="'+EncodeHTML(id)+'"'+checked+'> '+EncodeHTML(label)+'</label>');
	}
}

export function PushPromptFileConfig(apply_compose) {
	state.apply_compose = !!apply_compose;
	state.agent = $('#agent_select').val() || '';
	if ($('#rules_toggles input').length > 0)
		state.enabled_rules = CollectEnabledIds('#rules_toggles');
	if ($('#skills_toggles input').length > 0)
		state.enabled_skills = CollectEnabledIds('#skills_toggles');
	ipcRenderer.send('config-prompt-files', {
		agent: state.agent,
		enabled_rules: state.enabled_rules,
		enabled_skills: state.enabled_skills,
		workspace: state.workspace || ''
	});
}

export function ReloadPromptFromFiles() {
	PushPromptFileConfig(true);
}

function RefreshAgentSelectTitle() {
	const val = $('#agent_select').val() || '';
	const label = val ? ($('#agent_select option:selected').text() || val) : 'None';
	$('#agent_select').attr('title', 'Agent: ' + label);
}

export function OnAgentChanged() {
	RefreshAgentSelectTitle();
	PushPromptFileConfig(true);
}

function ApplyComposedPrompt(composed) {
	if (composed == null) return;
	state.prompt = composed;
	RefreshPrompt(composed);
	ipcRenderer.send('update-prompt', composed.replaceAll("\n", "[AI_UI_BR]"));
}

export function HandlePromptFiles(data) {
	const agents = data.agents || [];
	const rules = data.rules || [];
	const skills = data.skills || [];
	state.agent = data.agent || '';
	state.enabled_rules = data.enabled_rules || [];
	state.enabled_skills = data.enabled_skills || [];

	const cur = $('#agent_select').val();
	$('#agent_select').empty();
	$('#agent_select').append('<option value="">None</option>');
	for (let i = 0; i < agents.length; ++i) {
		$('#agent_select').append('<option value="'+EncodeHTML(agents[i].id)+'">'+EncodeHTML(StripMdName(agents[i].id))+'</option>');
	}
	const want = state.agent;
	const exists = !want || agents.some(a => a.id === want);
	$('#agent_select').val(exists ? want : '');
	if (!exists) state.agent = '';
	RefreshAgentSelectTitle();

	RenderFileToggles('#rules_toggles', rules, state.enabled_rules, 'id');
	RenderFileToggles('#skills_toggles', skills, state.enabled_skills, 'name');

	const composed = data.composed || '';
	const should_apply = state.apply_compose || (!state.prompt_files_ready && composed != '');
	state.prompt_files_ready = true;
	state.apply_compose = false;
	if (should_apply && composed != '') {
		ApplyComposedPrompt(composed);
	}
}

export function ShowPromptChip(show) {
	// Compact status occupies the same sticky slot as the prompt chip.
	if (show && !$('#compact_run_status').hasClass('hidden')) {
		$('#prompt_bar').addClass('hidden');
		return;
	}
	if (show) $('#prompt_bar').removeClass('hidden');
	else $('#prompt_bar').addClass('hidden');
}

export function RefreshPrompt(prompt_txt) {
	$('#prompt_txta').val(prompt_txt);
	const preview = PromptPreviewLine(prompt_txt);
	$('#init_prompt').html(EncodeHTML(preview));
	// Hide the empty chip+edit control until the engine has a real prompt.
	ShowPromptChip(!!String(prompt_txt || '').trim());
}
