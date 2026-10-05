import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import { EncodeHTML } from './util.js';

export function ToggleEditPrompt() {
	const $edit = $('#edit_dialog');
	if ($edit.hasClass('hidden')) {
		$('#prompt_files_dialog').addClass('hidden');
		StopPromptFilesWatch();
		RestoreAppliedEditor();
		$edit.removeClass('hidden');
		$('#prompt_txta').focus();
	} else {
		RestoreAppliedEditor();
		$edit.addClass('hidden');
	}
}

export function TogglePromptFiles() {
	const $d = $('#prompt_files_dialog');
	$d.toggleClass('hidden');
	if ($d.hasClass('hidden')) StopPromptFilesWatch();
	else StartPromptFilesWatch();
}

function PromptPreviewLine(prompt_txt) {
	let one_line = String(prompt_txt || '').replace(/\s+/g, ' ').trim();
	if (one_line.length > state.max_ip_len) {
		one_line = one_line.substring(0, state.max_ip_len) + '...';
	}
	return one_line;
}

export function ApplyNewPrompt() {
	if ($('#rules_toggles input').length)
		state.enabled_rules = CollectEnabledIds('#rules_toggles');
	if ($('#skills_toggles input').length)
		state.enabled_skills = CollectEnabledIds('#skills_toggles');
	const agent = $('#agent_select').val();
	if (agent != null) state.agent = agent || '';
	state.prompt = $('#prompt_txta').val();
	ipcRenderer.send('config-prompt-files', {
		agent: state.agent,
		enabled_rules: state.enabled_rules,
		enabled_skills: state.enabled_skills,
		workspace: state.workspace || '',
		apply: false,
		save: true
	});
	ipcRenderer.send('update-prompt', state.prompt.replaceAll("\n", "[AI_UI_BR]"));
	RefreshPrompt(state.prompt);
	$('#prompt_files_dialog').addClass('hidden');
	StopPromptFilesWatch();
	$('#edit_dialog').addClass('hidden');
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

let fileRules = [];
let fileSkills = [];
let fillTextarea = false;

function RestoreAppliedEditor() {
	$('#prompt_txta').val(state.prompt || '');
	if (fileRules.length || $('#rules_toggles').children().length)
		RenderFileToggles('#rules_toggles', fileRules, state.enabled_rules, 'id');
	if (fileSkills.length || $('#skills_toggles').children().length)
		RenderFileToggles('#skills_toggles', fileSkills, state.enabled_skills, 'name');
}

export function PushPromptFileConfig(mode) {
	const apply = mode === 'apply' || mode === true;
	fillTextarea = mode === 'fill';
	const agent = $('#agent_select').val() || state.agent || '';
	let rules = state.enabled_rules;
	let skills = state.enabled_skills;
	if ($('#rules_toggles input').length)
		rules = CollectEnabledIds('#rules_toggles');
	if ($('#skills_toggles input').length)
		skills = CollectEnabledIds('#skills_toggles');
	if (apply) {
		state.agent = agent;
		state.enabled_rules = rules;
		state.enabled_skills = skills;
	}
	ipcRenderer.send('config-prompt-files', {
		agent: agent,
		enabled_rules: rules,
		enabled_skills: skills,
		workspace: state.workspace || '',
		apply: apply,
		save: apply
	});
}

let promptFilesTimer = null;
let promptFilesListing = null;
let promptFilesBusy = false;
let promptAskDeclined = false;

function StopPromptFilesWatch() {
	if (promptFilesTimer) clearInterval(promptFilesTimer);
	promptFilesTimer = null;
	promptFilesBusy = false;
}

function StartPromptFilesWatch() {
	StopPromptFilesWatch();
	promptAskDeclined = false;
	promptFilesTimer = setInterval(TickPromptFiles, 1000);
	TickPromptFiles();
}

function EntryMap(list) {
	const map = new Map();
	for (const e of list || []) {
		if (e && e.id) map.set(e.id, e.mtime + ':' + e.size);
	}
	return map;
}

function EnabledFilesChanged(prev, next, ids) {
	const before = EntryMap(prev);
	const after = EntryMap(next);
	for (const id of ids || []) {
		if (!after.has(id)) return true;
		if (before.get(id) !== after.get(id)) return true;
	}
	return false;
}

function PromptListingAffects(prev, next) {
	if (!prev) return false;
	if (EnabledFilesChanged(prev.rules, next.rules, state.enabled_rules)) return true;
	if (EnabledFilesChanged(prev.skills, next.skills, state.enabled_skills)) return true;
	if (state.agent && EnabledFilesChanged(prev.agents, next.agents, [state.agent])) return true;
	return false;
}

function ListingKey(listing) {
	return JSON.stringify(listing || {});
}

async function RememberPromptFilesListing() {
	try {
		const listing = await ipcRenderer.invoke('prompt-files-stamp');
		if (listing && typeof listing === 'object') promptFilesListing = listing;
	} catch (e) { /* dialog watch retries */ }
}

async function TickPromptFiles() {
	if ($('#prompt_files_dialog').hasClass('hidden')) {
		StopPromptFilesWatch();
		return;
	}
	if (promptFilesBusy) return;
	let listing = null;
	try {
		listing = await ipcRenderer.invoke('prompt-files-stamp');
	} catch (e) {
		return;
	}
	if (!listing || typeof listing !== 'object') return;
	if (!promptFilesListing) {
		promptFilesListing = listing;
		return;
	}
	if (ListingKey(listing) === ListingKey(promptFilesListing)) return;
	const ask = PromptListingAffects(promptFilesListing, listing);
	promptFilesListing = listing;
	promptFilesBusy = true;
	PushPromptFileConfig('list');
	try {
		if (ask && !promptAskDeclined) {
			const yes = await ipcRenderer.invoke('confirm-action', {
				title: 'Update system prompt?',
				msg: 'A file used by the system prompt changed. Apply the new system prompt now?',
				buttons: ['Yes', 'No']
			});
			if (yes) PushPromptFileConfig('apply');
			else promptAskDeclined = true;
		}
	} catch (e) { /* list already refreshed */ }
	promptFilesBusy = false;
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
	fileRules = rules;
	fileSkills = skills;

	const cur = $('#agent_select').val();
	$('#agent_select').empty();
	$('#agent_select').append('<option value="">None</option>');
	for (let i = 0; i < agents.length; ++i) {
		$('#agent_select').append('<option value="'+EncodeHTML(agents[i].id)+'">'+EncodeHTML(StripMdName(agents[i].id))+'</option>');
	}

	if (data.apply !== false) {
		state.agent = data.agent || '';
		state.enabled_rules = data.enabled_rules || [];
		state.enabled_skills = data.enabled_skills || [];
	}
	const want = (data.apply !== false ? state.agent : (cur || state.agent)) || '';
	const exists = !want || agents.some(a => a.id === want);
	$('#agent_select').val(exists ? want : '');
	if (data.apply !== false && !exists) state.agent = '';
	RefreshAgentSelectTitle();

	const ruleIds = (data.apply === false && $('#rules_toggles input').length)
		? CollectEnabledIds('#rules_toggles')
		: state.enabled_rules;
	const skillIds = (data.apply === false && $('#skills_toggles input').length)
		? CollectEnabledIds('#skills_toggles')
		: state.enabled_skills;
	RenderFileToggles('#rules_toggles', rules, ruleIds, 'id');
	RenderFileToggles('#skills_toggles', skills, skillIds, 'name');

	const composed = data.composed;
	state.prompt_files_ready = true;
	state.apply_compose = false;
	RememberPromptFilesListing();
	if (fillTextarea && data.apply === false && composed != null) {
		$('#prompt_txta').val(composed);
		fillTextarea = false;
	} else if (data.apply !== false && composed != null) {
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
