import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import {
	TrimLast, DisableButtons, StopGenerating, EncodeHTML, VoiceOptionLabel,
	ShortcutKeyDown, ShortcutKeyUp, PathToFileURL, ApplyUiTheme, NormalizeThemeId,
	SetEngineReady, SyncSessionMenuLock, WireOverlayDismiss, NormMmProcessor, WhenDomReady
} from './util.js';
import {
	SendMsg, StopRecording, ToggleRecordBox, PlayAudio, ClearAttachments,
	StopThinking, ToggleInputDialog, UpdateContextUsage, ClearPlaybackReplay, StopPlayback
} from './chat.js';
import { ToggleAttachBox, SyncVisionSelectForBackend } from './attach.js';
import {
	RenderToolPacks, WirePackEvents, CollectToolPacksFromDom, NormalizeMergedPacks,
	GetPackDialogIdx, GetPackDialogTab, ApplyDisabledToolsToPack, RenderPackToolsList
} from './tool_packs.js';
import { WireToolPermEvents } from './tool_perm.js';
import {
	ToggleEditPrompt, TogglePromptFiles, ApplyNewPrompt, ReloadPromptFromFiles,
	OnAgentChanged, HandlePromptFiles, ShowPromptChip, RefreshPrompt, PushPromptFileConfig
} from './prompt_files.js';

export {
	RenderToolPacks, ToggleEditPrompt, TogglePromptFiles, ApplyNewPrompt,
	ReloadPromptFromFiles, OnAgentChanged, ShowPromptChip, RefreshPrompt
};

export const default_ip = "Chat session started on DATE";

export function NormTempFormat(raw) {
	const t = String(raw || 'none').trim().toLowerCase();
	if (t === 'tool') return 'aiui';
	if (t === 'chat') return 'model';
	if (t === 'none' || t === 'model' || t === 'aiui' || t === 'aiui_notoolprompt') return t;
	return 'none';
}

export function NormRemoteToolFormat(raw) {
	const t = String(raw || 'openai').toLowerCase();
	if (t === 'anthropic') return 'anthropic';
	return 'openai';
}

export function NormReasoningEffort(raw) {
	const t = String(raw || 'auto').toLowerCase();
	if (t === 'none') return 'auto';
	if (t === 'minimal') return 'low';
	if (t === 'auto' || t === 'low' || t === 'medium' || t === 'high' || t === 'xhigh' || t === 'max') return t;
	return 'auto';
}

function HidePages() {
	$('#chat_app').hide();
	$('#txtgen_app').hide();
	$('#imggen_app').hide();
	$('#ttsgen_app').hide();
	$('#console_tab').hide();
	$('#config_tab').hide();
	$('#help_tab').hide();
	$('#about_tab').hide();
}

function SelectPage(btn) {
	$('.menu_btn').removeClass('menu_sel');
	btn.classList.add('menu_sel');
	HidePages();
}

export function ChatPage(btn) {
	SelectPage(btn);
	$('#chat_app').show();
}

export function TextPage(btn) {
	SelectPage(btn);
	$('#txtgen_app').show();
}

export function ImagePage(btn) {
	SelectPage(btn);
	$('#imggen_app').show();
}

export function SpeechPage(btn) {
	SelectPage(btn);
	$('#ttsgen_app').show();
}

export function ConfigPage(btn) {
	SelectPage(btn);
	$('#config_tab').show();
}

export function ConsolePage(btn) {
	SelectPage(btn);
	$('#console_tab').show();
}

export function HelpPage(btn) {
	SelectPage(btn);
	$('#help_tab').show();
}

export function AboutPage(btn) {
	SelectPage(btn);
	$('#about_tab').show();
}

function IsEngineStatusError(msg) {
	const s = String(msg || '').toLowerCase();
	if (!s) return false;
	return /\b(error|failed|failure|unable|exception|traceback|denied|invalid|warning|warn)\b/.test(s)
		|| s.indexOf('check console') >= 0;
}

function WorkspaceStatusSvg(kind) {
	const common = 'viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
	if (kind === 'home') {
		return '<svg ' + common + '>'
			+ '<path d="M2.5 7.2 L8 2.6 L13.5 7.2"/>'
			+ '<path d="M4.2 6.8 V13.2 H11.8 V6.8"/>'
			+ '<path d="M6.6 13.2 V9.4 H9.4 V13.2"/>'
			+ '</svg>';
	}
	if (kind === 'folder') {
		return '<svg ' + common + '>'
			+ '<path d="M2.4 4.4 H6.1 L7.4 5.8 H13.6 V12.6 H2.4 Z"/>'
			+ '</svg>';
	}
	if (kind === 'error') {
		return '<svg ' + common + '>'
			+ '<path d="M8 2.4 L14.1 13.2 H1.9 Z"/>'
			+ '<path d="M8 6.2 V9.6"/>'
			+ '<circle cx="8" cy="11.4" r="0.55" fill="currentColor" stroke="none"/>'
			+ '</svg>';
	}
	// info
	return '<svg ' + common + '>'
		+ '<circle cx="8" cy="8" r="5.5"/>'
		+ '<path d="M8 7.2 V11"/>'
		+ '<circle cx="8" cy="5.15" r="0.55" fill="currentColor" stroke="none"/>'
		+ '</svg>';
}

function WorkspaceStatusDetail() {
	if (state.engine_status) return state.engine_status;
	const ws = state.workspace || '';
	const name = state.workspace_label || 'Home';
	if (ws) return 'Workspace: ' + name + '\n' + ws;
	return 'Workspace: Home';
}

function PaintWorkspaceStatusTip() {
	const detail = WorkspaceStatusDetail();
	$('#workspace_status_tip').html(EncodeHTML(detail).replaceAll('\n', '<br>'));
}

export function ToggleWorkspaceStatusTip(ev) {
	if (ev && typeof ev.stopPropagation === 'function') ev.stopPropagation();
	const $tip = $('#workspace_status_tip');
	if ($tip.hasClass('hidden')) {
		PaintWorkspaceStatusTip();
		$tip.removeClass('hidden');
	} else {
		$tip.addClass('hidden');
	}
}

export function HideWorkspaceStatusTip() {
	$('#workspace_status_tip').addClass('hidden');
}

export function SyncWorkspaceStatusLayout() {
	const menu = document.getElementById('main_menu');
	const status = document.getElementById('workspace_label');
	if (!menu || !status) return;
	let used = 0;
	const btns = menu.querySelectorAll('.menu_btn');
	for (let i = 0; i < btns.length; ++i) used += btns[i].offsetWidth;
	const avail = menu.clientWidth - used - 20;
	// Keep a reserved icon footprint; collapse text when the label would be squeezed away.
	status.classList.toggle('compact', avail < 170);
}

function RenderWorkspaceLabel() {
	const shortName = state.workspace_label || 'Home';
	const ws = state.workspace || '';
	const isHome = !ws;
	document.title = 'AI UI - ' + shortName;

	let kind = isHome ? 'home' : 'folder';
	let text = 'Workspace: ' + shortName;
	let tip = ws || 'Home';
	let err = false;

	if (state.engine_status) {
		err = IsEngineStatusError(state.engine_status);
		kind = err ? 'error' : 'info';
		text = state.engine_status;
		tip = state.engine_status;
	}

	const $root = $('#workspace_label');
	const $icon = $('#workspace_status_icon');
	const $text = $('#workspace_status_text');
	const $btn = $('#workspace_status_btn');
	$root.toggleClass('is_error', err);
	$root.toggleClass('is_status', !!state.engine_status);
	$icon.html(WorkspaceStatusSvg(kind));
	$text.text(text);
	$btn.attr('title', tip);
	$btn.attr('aria-label', tip);
	if (!$('#workspace_status_tip').hasClass('hidden')) {
		PaintWorkspaceStatusTip();
	}
	SyncWorkspaceStatusLayout();
}

function UpdateWorkspaceLabel(ctx) {
	const ws = (ctx && ctx.workspace != null) ? ctx.workspace : (state.workspace || '');
	const name = (ctx && ctx.label) ? ctx.label : (ws ? String(ws).split(/[/\\]/).pop() : 'Home');
	state.workspace = ws || '';
	state.workspace_label = name || 'Home';
	RenderWorkspaceLabel();
}

function SetEngineStatus(msg) {
	state.engine_status = String(msg || '').replaceAll('[AI_UI_BR]', ' ').trim();
	RenderWorkspaceLabel();
}

function ClearEngineStatus() {
	if (!state.engine_status) return;
	state.engine_status = '';
	RenderWorkspaceLabel();
}

export function ApplySettings() {
	let sdir = TrimLast($('#script_dir').val().trim().replaceAll('\\', '/'), '/');
	let mdir = TrimLast($('#model_dir').val().trim().replaceAll('\\', '/'), '/');
	let imdir = TrimLast($('#im_model').val().trim().replaceAll('\\', '/'), '/');
	let ttsdir = TrimLast($('#tts_model').val().trim().replaceAll('\\', '/'), '/');
	let srdir = TrimLast($('#sr_model').val().trim().replaceAll('\\', '/'), '/');
	let pbin = $('#python_bin').val().trim().replaceAll('\\', '/');
	let margs = $('#model_args').val().replaceAll('\\', '/');
	let mtype = $('#tmodel_select').find(':selected').val();
	let itype = $('#imodel_select').find(':selected').val();
	let stype = $('#smodel_select').find(':selected').val();
	let cdev = $('#device_select').find(':selected').val();
	let smeth = $('#startup_select').find(':selected').val();
	let formatm = $('#format_mode').find(':selected').val();
	let themeId = NormalizeThemeId($('#ui_theme').find(':selected').val());
	let genmd = $('#txtgen_mode').find(':selected').val();
	let etooluse = $('#tooluse_enable').find(':selected').val();
	let edevmode = $('#devmode_enable').find(':selected').val();
	let easasro = $('#asasro_enable').find(':selected').val();
	let startrk = $('#start_rec_keys').val().trim();
	let stoprk = $('#stop_rec_keys').val().trim();
	let hftoken = $('#hf_access_token').val().trim();
	let textBackend = ($('#text_backend').find(':selected').val() === 'remote') ? 'remote' : 'local';
	let remoteUrl = $('#remote_base_url').val().trim();
	let remoteModel = $('#remote_model').val().trim();
	let remoteKey = $('#remote_api_key').val().trim();
	let remoteProvider = $('#remote_provider').val().trim();
	let remoteToolFormat = NormRemoteToolFormat($('#remote_tool_format').find(':selected').val());
	let remotePromptCache = ($('#remote_prompt_cache').find(':selected').val() === '0') ? 0 : 1;

	if (!$('#main_settings').is(':hidden')) {
		const engine_changed = (state.script_dir != sdir || state.python_bin != pbin || state.model_dir != mdir || state.im_model != imdir || state.tts_model != ttsdir || state.sr_model != srdir ||
		state.model_args != margs || state.model_type != mtype || state.imodel_type != itype || state.smodel_type != stype || state.comp_dev != cdev || state.start_meth != smeth);
		if (engine_changed) {
			ipcRenderer.send('config-app', {
				script_dir: sdir, python_bin: pbin, model_dir: mdir, sd_model: imdir, tts_model: ttsdir, sr_model: srdir,
				model_args: margs, model_type: mtype, imodel_type: itype, smodel_type: stype, comp_dev: cdev, start_meth: smeth,
				workspace: state.workspace || ''
			});
		}
		return;
	}

	if (!$('#remote_settings').is(':hidden')) {
		if (textBackend === 'remote' && !remoteModel) {
			ipcRenderer.send('show-alert', { msg: 'Remote Model ID is required when Text Backend is Remote.' });
			return;
		}
		const remote_changed = (
			state.text_backend != textBackend || state.remote_base_url != remoteUrl || state.remote_model != remoteModel ||
			state.remote_api_key != remoteKey || state.remote_provider != remoteProvider ||
			state.remote_tool_format != remoteToolFormat || state.remote_prompt_cache != remotePromptCache ||
			state.hf_token != hftoken
		);
		if (!remote_changed) return;
		if (state.thinking || state.generating) {
			ipcRenderer.send('show-alert', { msg: 'Wait for the current task to finish.' });
			return;
		}
		if (remoteToolFormat === 'anthropic' && !remoteKey) {
			ipcRenderer.send('show-alert', { msg: 'Remote API Key is required for Anthropic Messages.' });
			return;
		}
		ipcRenderer.send('config-app-remote', {
			text_backend: textBackend,
			remote_base_url: remoteUrl,
			remote_model: remoteModel,
			remote_api_key: remoteKey,
			remote_provider: remoteProvider,
			remote_tool_format: remoteToolFormat,
			remote_prompt_cache: remotePromptCache,
			hf_token: hftoken
		});
		state.text_backend = textBackend;
		state.remote_base_url = remoteUrl;
		state.remote_model = remoteModel;
		state.remote_api_key = remoteKey;
		state.remote_provider = remoteProvider;
		state.remote_tool_format = remoteToolFormat;
		state.remote_prompt_cache = remotePromptCache;
		state.hf_token = hftoken;
		SyncVisionSelectForBackend(textBackend);
		ipcRenderer.send('show-alert', { msg: 'New settings applied.' });
		return;
	}

	// Other Settings
	let new_settings = false;
	if (state.gen_mode != genmd || state.format_mode != formatm || state.enable_tooluse != etooluse) {
		if (state.thinking || state.generating) {
			ipcRenderer.send('show-alert', { msg: 'Wait for the current task to finish.' });
			return;
		}
		ipcRenderer.send('config-app-other', {
			gen_mode: genmd, format_mode: formatm, enable_tooluse: etooluse
		});
		new_settings = true;
		state.gen_mode = genmd;
		state.format_mode = formatm;
		state.enable_tooluse = etooluse;
	}
	if (state.enable_devmode != edevmode || state.enable_asasro != easasro || state.start_rec_keys != startrk || state.stop_rec_keys != stoprk || state.ui_theme != themeId) {
		ipcRenderer.send('config-other', {
			enable_devmode: edevmode, enable_asasro: easasro, start_rec_keys: startrk, stop_rec_keys: stoprk,
			ui_theme: themeId
		});
		new_settings = true;
		state.enable_devmode = edevmode;
		state.enable_asasro = easasro;
		state.start_rec_keys = startrk;
		state.stop_rec_keys = stoprk;
		state.ui_theme = ApplyUiTheme(themeId);
		ipcRenderer.send('show-menubar');
	}
	if (new_settings) ipcRenderer.send('show-alert', { msg: 'New settings applied.' });
}

export function ApplyConfig() {
	if (state.config_state == 1) {
		const maxmm = $('#max_msg_mem').val();
		const maxrl = $('#max_res_len').val();
		const minrl = $('#min_res_len').val();
		const numbm = $('#num_beams').val();
		const numbg = $('#beam_groups').val();
		const btemp = $('#base_temp').val();
		const topk = $('#top_k').val();
		const topp = $('#top_p').val();
		const typp = $('#typical_p').val();
		const repp = $('#rep_penalty').val();
		const freqp = $('#frequency_penalty').val();
		const presp = $('#presence_penalty').val();
		const compactAt = $('#compact_at').val();
		const compactTail = $('#compact_tail_tokens').val();
		const compactSumMax = $('#compact_summary_max').val();
		const dosmp = $('#ds_select').find(':selected').val();
		const promptp = $('#pp_select').find(':selected').val();
		const tempf = $('#template_select').find(':selected').val();
		const ethink = $('#enable_thinking').find(':selected').val();
		const reffort = NormReasoningEffort($('#reasoning_effort').find(':selected').val());
		const caNum = Number(compactAt);
		const rlNum = Number(maxrl);
		if (caNum > 0 && rlNum >= caNum) {
			ipcRenderer.send('show-alert', {
				msg: 'Context Token Budget must be greater than Max Response Length. '
					+ 'Otherwise compaction triggers every turn (response budget cannot fit under the limit).'
			});
			return;
		}
		if (state.max_mm != maxmm || state.max_rl != maxrl || state.min_rl != minrl || state.b_temp != btemp ||
		state.do_sample != dosmp || state.num_beams != numbm || state.beam_groups != numbg ||
		topk != state.top_k || topp != state.top_p || typp != state.typical_p || repp != state.rep_penalty ||
		freqp != state.frequency_penalty || presp != state.presence_penalty ||
		compactAt != state.compact_at || compactTail != state.compact_tail_tokens || compactSumMax != state.compact_summary_max ||
		state.prompt_p != promptp || state.temp_format != tempf ||
		state.enable_thinking != ethink || state.reasoning_effort != reffort) {
			ipcRenderer.send('config-ai', {
				max_mmem: maxmm, max_rlen: maxrl, min_rlen: minrl,
				do_sample: dosmp, num_beams: numbm, beam_groups: numbg,
				temp: btemp, tk: topk, tp: topp, typ: typp, rp: repp, pp: promptp,
				fp: freqp, prp: presp,
				compact_at: compactAt,
				compact_tail_tokens: compactTail,
				compact_summary_max: compactSumMax,
				temp_format: tempf,
				enable_thinking: ethink,
				reasoning_effort: reffort
			});
			state.temp_format = tempf;
			state.enable_thinking = ethink;
			state.reasoning_effort = reffort;
			state.frequency_penalty = freqp;
			state.presence_penalty = presp;
			state.compact_at = compactAt;
			state.compact_tail_tokens = compactTail;
			state.compact_summary_max = compactSumMax;
			state.max_mm = maxmm;
			state.max_rl = maxrl;
			state.min_rl = minrl;
			state.b_temp = btemp;
			state.do_sample = dosmp;
			state.num_beams = numbm;
			state.beam_groups = numbg;
			state.top_k = topk;
			state.top_p = topp;
			state.typical_p = typp;
			state.rep_penalty = repp;
			state.prompt_p = promptp;
			UpdateContextUsage({
				used: state.ctx_used || 0,
				limit: Number(compactAt) || 0,
				gen_max: Number(maxrl) || 0
			});
			let applied = 'New settings applied.';
			if (Number(state.model_type) === 2 && (tempf === 'aiui' || tempf === 'aiui_notoolprompt')) {
				applied += ' Note: multimodal models always use the Model Template so image/video tokens work.';
			}
			ipcRenderer.send('show-alert', { msg: applied });
		}
	} else if (state.config_state == 2) {
		const prevPacks = JSON.stringify(state.tool_packs_applied || state.tool_packs || null);
		const packs = CollectToolPacksFromDom();
		const packsChanged = JSON.stringify(packs) !== prevPacks;
		if (packsChanged) {
			ipcRenderer.send('config-tools', {
				temp_format: state.temp_format,
				tool_packs: packs,
				firecrawl_api_key: state.firecrawl_api_key || ''
			});
			state.tool_packs = packs;
			state.tool_packs_applied = JSON.parse(JSON.stringify(packs));
			ipcRenderer.send('show-alert', { msg: 'New settings applied.' });
		}
	} else {
		const spvol = Number($('#speech_vol').val());
		const srate = Number($('#speech_rate').val());
		const spitch = Number($('#speech_pitch').val());
		const newvc = Number($('#voices').find(':selected').val());
		const newtts = Number($('#tts_mode').find(':selected').val());
		const newam = Number($('#anim_mode').find(':selected').val());
		let alert_msg = true;

		if (Number(state.bot_voice) !== newvc || Number(state.speech_vol) !== spvol ||
		Number(state.speech_rate) !== srate || Number(state.pitch_shift) !== spitch ||
		Number(state.tts_mode) !== newtts || Number(state.anim_mode) !== newam) {
			const voicePayload = { voice:newvc, vol:spvol, rate:srate, pitch:spitch, engine:newtts, amode:newam };
			if (Number(state.anim_mode) !== newam) {
				$('#loading_msg').html('Processing avatar image... please wait.');
				$('#loading_box').show();
				window.aiuiAPI.prepareAvatar({ mode: newam }).then(res => {
					if (!res || !res.ok) {
						$('#loading_box').hide();
						ipcRenderer.send('show-alert', {
							type: 'error',
							msg: (res && res.error) || 'Could not prepare avatar for this animation mode.'
						});
						return;
					}
					if (res.warned) {
						ipcRenderer.send('show-alert', {
							msg: 'Avatar is below the recommended '+res.target+'x'+res.target+' size; quality may suffer.'
						});
					}
					state.avatar_img = res.path.replaceAll('\\', '/');
					state.avatar_mp4 = AvatarMp4ForMode(newam);
					$('#avatar_img').prop('src', AvatarImgSrc(state.avatar_img));
					voicePayload.avatar_img = state.avatar_img;
					ipcRenderer.send('config-voice', voicePayload);
				});
			} else {
				ipcRenderer.send('config-voice', voicePayload);
				if (alert_msg) {
					ipcRenderer.send('show-alert', { msg: 'New settings applied.' });
				}
			}
		}
	}
}

export function ToggleConfig() {
	if (state.config_state == 2) {
		$('#adv_btn').html('AI SETTINGS');
		$('#tool_config').hide();
		$('#basic_config').show();
		state.config_state = 0;
	} else if (state.config_state == 1) {
		$('#adv_btn').html('MAIN SETTINGS');
		$('#adv_config').hide();
		$('#tool_config').show();
		state.config_state = 2;
	} else {
		$('#adv_btn').html('TOOL SETTINGS');
		$('#basic_config').hide();
		$('#adv_config').show();
		state.config_state = 1;
	}
}

function SetSubtabPair(activeSel, inactiveSel) {
	$(activeSel).addClass('subtab_sel');
	$(inactiveSel).removeClass('subtab_sel');
}

export function ShowInstallHelp() {
	if ($('#install_txt').is(":hidden")) {
		$('#install_txt').show();
		$('#setupargs_txt').hide();
		SetSubtabPair('#inst_help_btn', '#args_help_btn');
	}
}

export function ShowSetupHelp() {
	if ($('#setupargs_txt').is(":hidden")) {
		$('#setupargs_txt').show();
		$('#install_txt').hide();
		SetSubtabPair('#args_help_btn', '#inst_help_btn');
	}
}

function SetSettingsSubtab(activeBtn, activePanel) {
	$('#main_settings, #other_settings, #remote_settings').hide();
	$('#main_conf_btn, #other_conf_btn, #remote_conf_btn').removeClass('subtab_sel');
	$(activePanel).show();
	$(activeBtn).addClass('subtab_sel');
}

export function ShowMainConfig() {
	if ($('#main_settings').is(':hidden')) {
		SetSettingsSubtab('#main_conf_btn', '#main_settings');
	}
}

export function ShowOtherConfig() {
	if ($('#other_settings').is(':hidden')) {
		SetSettingsSubtab('#other_conf_btn', '#other_settings');
	}
}

export function ShowRemoteConfig() {
	if ($('#remote_settings').is(':hidden')) {
		SetSettingsSubtab('#remote_conf_btn', '#remote_settings');
	}
}

export { ToggleInputDialog };

export function ChangeTalkMode() {
	if (++state.talk_mode > 2) state.talk_mode = 0;
	$('#tmode_img').prop('src', './img/talk_'+state.talk_mode+'.png');
	ipcRenderer.send('update-talk-mode', state.talk_mode);
}

function AvatarMp4ForMode(mode) {
	const base = state.script_dir;
	const m = Number(mode);
	if (!(m >= 0)) return '';
	if (m == 0) return PathToFileURL(base+'/MakeItTalk/examples/face_pred_fls_speech_audio_embed.mp4');
	if (m == 1) return PathToFileURL(base+'/Wav2Lip/results/face_pred_fls_speech_audio_embed.mp4');
	if (m == 2) return PathToFileURL(base+'/SadTalker/results/face_pred_fls_speech_audio_embed.mp4');
	return PathToFileURL(base+'/IMTalker/results/face_pred_fls_speech_audio_embed.mp4');
}

function FormatDlBytes(n) {
	if (!n || n < 0) return '0 B';
	if (n < 1024) return n + ' B';
	if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
	return (n / 1048576).toFixed(1) + ' MB';
}

function AvatarImgSrc(file_path) {
	// Work cache is always the same path; bust Chromium's file:// image cache.
	return PathToFileURL(file_path) + '?t=' + Date.now();
}

export function ChangeAvatar() {
	window.aiuiAPI.openAvatar().then(picked => {
		if (picked === false) return;
		$('#loading_msg').html('Processing avatar image... please wait.');
		$('#loading_box').show();
		const mode = $('#anim_mode').find(':selected').val();
		window.aiuiAPI.prepareAvatar({ source: picked, mode: mode }).then(res => {
			if (!res || !res.ok) {
				$('#loading_box').hide();
				ipcRenderer.send('show-alert', {
					type: 'error',
					msg: (res && res.error) || 'Could not prepare avatar image.'
				});
				return;
			}
			if (res.warned) {
				ipcRenderer.send('show-alert', {
					msg: 'Avatar is below the recommended '+res.target+'x'+res.target+' size; quality may suffer.'
				});
			}
			state.avatar_img = res.path.replaceAll('\\', '/');
			ipcRenderer.send('update-avatar', state.avatar_img);
			StopPlayback();
			ClearPlaybackReplay();
			$('#avatar_img').prop('src', AvatarImgSrc(state.avatar_img));
			$('#avatar_vid').hide();
			$('#other_vid').hide();
			$('#avatar_img').show();
		});
	});
}

export function OpenSample(elem) {
	window.aiuiAPI.openSample().then(result => {
		if (result === false) return;
		$('#'+elem).val(result.replaceAll('\\', '/'));
	});
}

export function OpenFileFolder(elem) {
	window.aiuiAPI.openFileFolder().then(result => {
		if (result === false) return;
		$('#'+elem).val(result.replaceAll('\\', '/'));
	});
}

export function OpenFolder(elem) {
	window.aiuiAPI.openFolder().then(result => {
		if (result === false) return;
		$('#'+elem).val(result.replaceAll('\\', '/'));
	});
}

export function OpenFile(elem, trim_path=false) {
	window.aiuiAPI.openFile().then(result => {
		if (result === false) return;
		result = result.replaceAll('\\', '/');
		if (trim_path) {
			const last_slash = result.lastIndexOf('/');
			if (last_slash >= 0)
				result = result.substr(last_slash+1);
		}
		$('#'+elem).val(result);
	});
}

export function SetAppVersion() {
	window.aiuiAPI.getAppVersion().then(result => {
		$('#aiui-version').html(result);
	});
	const versions = window.aiuiAPI.versions || {};
	for (const type of ['chrome', 'node', 'electron']) {
		if (versions[type]) $('#' + type + '-version').text(versions[type]);
	}
	if (window.aiuiAPI.platform === 'win32') {
		state.is_windows = true;
	} else if (window.aiuiAPI.platform) {
		state.is_windows = false;
	}
}

export function RestartScript() {
	ClearAttachments();
	SetEngineReady(false);
	ipcRenderer.invoke('restart-script');
}

export function StartScript() {
	SetEngineReady(false);
	ipcRenderer.invoke('start-script');
}

export function QuitApp() {
	ipcRenderer.invoke('close-app');
}

export function InitUi() {
	// Start locked until ai-ready (matches disabled main buttons in index.html).
	SyncSessionMenuLock();
	WhenDomReady(function () {
		RenderWorkspaceLabel();
		SyncWorkspaceStatusLayout();
		if (typeof ResizeObserver !== 'undefined') {
			const menu = document.getElementById('main_menu');
			if (menu) {
				const ro = new ResizeObserver(function () {
					SyncWorkspaceStatusLayout();
				});
				ro.observe(menu);
			}
		} else {
			$(window).on('resize.workspace_status', SyncWorkspaceStatusLayout);
		}
		$(document).on('click.workspace_status', function (e) {
			if ($(e.target).closest('#workspace_label').length) return;
			HideWorkspaceStatusTip();
		});
	});

	ipcRenderer.on('init-ui', (event, payload) => {
		if (payload.state !== false) {
			const d = new Date();
			const curr_date = (d.getMonth()+1)+'/'+d.getDate()+'/'+d.getFullYear();
			const def_ip = default_ip.replaceAll('DATE', () => curr_date);
			if (payload.state == 'AI_UI_DEFAULT') {
				// Keep an already-composed / restored prompt; only use the dated stub when empty.
				if (!state.prompt || !String(state.prompt).trim()) state.prompt = def_ip;
			} else {
				state.prompt = payload.state.replaceAll("[AI_UI_BR]", "\n");
			}
			RefreshPrompt(state.prompt);
			// Engine is up - restore workspace label. Do not wipe gen/tts/img results
			// (those panes are for generation output, not engine lifecycle status).
			ClearEngineStatus();
			// Prompt restore is not idle yet - wait for ai-ready.
			SyncSessionMenuLock();
		} else {
			// Allow prompt-files compose to re-apply after engine restart.
			state.prompt_files_ready = false;
			SetEngineReady(false);
			$('#loading_box').hide();
			DisableButtons(true);
			StopThinking();
			StopGenerating();
		}
	});

	ipcRenderer.on('prompt-files', (event, payload) => {
		HandlePromptFiles(payload || {});
	});

	ipcRenderer.on('ai-ready', (event, payload) => {
		ClearEngineStatus();
		SetEngineReady(true);
		if (state.asr_pending) {
			state.asr_pending = false;
			SendMsg();
		}
		DisableButtons(false);
		StopThinking();
		StopGenerating();
	});

	ipcRenderer.on('set-platform', (event, payload) => {
		if (payload.startsWith('win')) {
			state.is_windows = true;
		} else {
			state.is_windows = false;
		}
	});

	ipcRenderer.on('add-voices', (event, payload) => {
		let ttsgen_sel = false;
		let ttsgen_mode = $('#ttsgen_mode').find(':selected').val()
		// Keep the Speech Gen selection across rebuilds (e.g. Kitten LoadAIVoices).
		let keep_tts_voice = $('#tts_voices').val();
		if (keep_tts_voice == null || keep_tts_voice === '') {
			keep_tts_voice = state.tts_voice;
		}

		if (payload.mode == 'SYS') {
			state.sys_voices = [];
			if (ttsgen_mode == 0) {
				$('#tts_voices').empty();
				ttsgen_sel = true;
			}
		} else {
			state.ai_voices = [];
			if (ttsgen_mode > 0) {
				$('#tts_voices').empty();
				ttsgen_sel = true;
			}
		}

		for (let i=0; i<payload.names.length; ++i)
		{
			let voice_name = payload.names[i];
			if (voice_name.startsWith("af_") || voice_name.startsWith("am_") ||
			voice_name.startsWith("bf_") || voice_name.startsWith("bm_") ||
			voice_name.startsWith("ef_") || voice_name.startsWith("em_") ||
			voice_name.startsWith("ff_") || voice_name.startsWith("fm_") ||
			voice_name.startsWith("hf_") || voice_name.startsWith("hm_") ||
			voice_name.startsWith("if_") || voice_name.startsWith("im_") ||
			voice_name.startsWith("jf_") || voice_name.startsWith("jm_") ||
			voice_name.startsWith("pf_") || voice_name.startsWith("pm_") ||
			voice_name.startsWith("zf_") || voice_name.startsWith("zm_")) {
				if (voice_name[0] == 'a') {
					voice_name += ' (American English - ';
				} else if (voice_name[0] == 'b') {
					voice_name += ' (British English - ';
				} else if (voice_name[0] == 'e') {
					voice_name += ' (Spanish - ';
				} else if (voice_name[0] == 'f') {
					voice_name += ' (French - ';
				} else if (voice_name[0] == 'h') {
					voice_name += ' (Hindi - ';
				} else if (voice_name[0] == 'i') {
					voice_name += ' (Italian - ';
				} else if (voice_name[0] == 'j') {
					voice_name += ' (Japanese - ';
				} else if (voice_name[0] == 'p') {
					voice_name += ' (Portuguese - ';
				} else if (voice_name[0] == 'z') {
					voice_name += ' (Chinese - ';
				} else {
					voice_name += ' (Unknown - ';
				}
				if (voice_name[1] == 'f') {
					voice_name += 'Female)';
				} else {
					voice_name += 'Male)';
				}
				voice_name = voice_name.substring(3);
			}
			voice_name = voice_name.replaceAll('_', ' ');
			const opt_label = EncodeHTML(VoiceOptionLabel(voice_name));
			const opt_title = EncodeHTML(voice_name);
			if (payload.set) {
				let prop_str = (i == state.bot_voice) ? i+'" selected' : i+'"';
				$('#voices').append('<option value="'+prop_str+' title="'+opt_title+'">'+opt_label+'</option>');
			}
			if (ttsgen_sel) {
				$('#tts_voices').append('<option value="'+i+'" title="'+opt_title+'">'+opt_label+'</option>');
			}
			if (payload.mode == 'SYS') {
				state.sys_voices.push(voice_name);
			} else {
				state.ai_voices.push(voice_name);
			}
		}
		if (ttsgen_sel) {
			if ($("#tts_voices option[value='"+keep_tts_voice+"']").length > 0) {
				$('#tts_voices').val(keep_tts_voice);
				state.tts_voice = keep_tts_voice;
			} else if ($("#tts_voices option[value='"+state.tts_voice+"']").length > 0) {
				$('#tts_voices').val(state.tts_voice);
			}
		}
	});

	ipcRenderer.on('clear-voices', (event, payload) => {
		$('#voices').empty();
	});

	ipcRenderer.on('play-audio', (event, payload) => {
		if (payload.file == 'ERROR') {
			ipcRenderer.send('show-alert', { type: "error", msg: "There was an error generating the speech." });
		} else if (payload.file == 'TOO_LONG') {
			ipcRenderer.send('show-alert', {
				type: "error",
				msg: "Text is too long for speech. Shorten it or raise audio_max_chars in Model Setup."
			});
		} else if (payload.file == 'NO_VOICES') {
			ipcRenderer.send('show-alert', { type: "error", msg: "There are no voices available for the selected speech engine." });
		} else {
			PlayAudio(payload.file);
		}
	});

	ipcRenderer.on('got-avatar', (event, payload) => {
		$('#loading_box').hide();
		if (payload.got != "true") {
			ipcRenderer.send('show-alert', { type: "error", msg: "Face animation engine cannot detect a face in the avatar image!" });
		}
	});

	ipcRenderer.on('got-tools', (event, payload) => {
		state.tool_funcs = payload.tools;
	});

	ipcRenderer.on('got-tool-catalog', (event, payload) => {
		state.tool_catalog = (payload && payload.catalog) || {};
		state.mcp_running = (payload && Array.isArray(payload.mcp_running)) ? payload.mcp_running.slice() : [];
		state.mcp_errors = (payload && payload.mcp_errors && typeof payload.mcp_errors === 'object')
			? Object.assign({}, payload.mcp_errors)
			: {};
		// Catalog arrives after sync_servers finishes - clear any in-flight start markers.
		state.mcp_pending = {};
		if ($('#tool_config').length && !$('#tool_config').is(':hidden')) {
			CollectToolPacksFromDom();
		}
		RenderToolPacks();
		if (!$('#pack_dialog').hasClass('hidden') && GetPackDialogTab() === 'tools' && GetPackDialogIdx() >= 0) {
			const packs = state.tool_packs || [];
			const pack = packs[GetPackDialogIdx()];
			if (pack) ApplyDisabledToolsToPack(pack);
			RenderPackToolsList();
		}
	});

	ipcRenderer.on('got-file', (event, payload) => {
		if (payload.init) {
			state.active_dl = 1;
			state.total_dls = payload.total || payload.remaining;
			$('#loading_msg').html('Downloading file 1 of '+state.total_dls+' ...');
			$('#loading_box').show();
		} else if (payload.progress) {
			const n = Math.max(1, state.total_dls - payload.remaining + 1);
			let msg = 'Downloading file '+n+' of '+state.total_dls;
			if (payload.percent >= 0 && payload.total > 0) {
				msg += ': '+FormatDlBytes(payload.received)+' / '+FormatDlBytes(payload.total)+' ('+payload.percent+'%)';
			} else if (payload.received > 0) {
				msg += ': '+FormatDlBytes(payload.received);
			}
			msg += ' ...';
			$('#loading_msg').html(msg);
		} else if (payload.remaining > 0) {
			state.active_dl = Math.max(1, state.total_dls - payload.remaining + 1);
			$('#loading_msg').html('Downloading file '+state.active_dl+' of '+state.total_dls+' ...');
		}
	});

	ipcRenderer.on('got-models', (event, payload) => {
		let hideMsgBox = true;
		if (payload.failed == 0) {
			$('#loading_msg').html('Finished downloading all model files.');
			if (payload.remind_apply) {
				ipcRenderer.send('show-alert', {
					msg: 'Animation engine model download finished. Click Apply to use the selected engine.'
				});
			}
		} else if (payload.retry) {
			if (payload.failed > 0) {
				$('#loading_msg').html('Failed to download '+payload.failed+' files. Retrying ...');
			} else {
				$('#loading_msg').html('Failed to download model files. Retrying ...');
			}
			let mode_sel = $('#anim_mode').find(':selected').val();
			ipcRenderer.send('check-models', { mode: mode_sel });
			hideMsgBox = false;
		} else if (payload.failed > 0) {
			$('#loading_msg').html('Failed to download '+payload.failed+' files.');
		} else {
			$('#loading_msg').html('Download failed due to an unexpected error.');
		}
		if (hideMsgBox) {
			setTimeout(function() {
				$('#loading_box').hide();
			}, 1500);
		}
	});

	ipcRenderer.on('prompt-msg', (event, payload) => {
		SetEngineStatus(payload.msg || '');
	});

	ipcRenderer.on('load-config', (event, payload) => {
		const chat_config = payload.configs.chat;
		const app_config = payload.configs.app;
		const ai_config = payload.configs.ai;
		const gen_config = payload.configs.gen;

		state.human_name = 'user';
		state.bot_name = 'assistant';
		state.bot_voice = chat_config.bot_voice;
		state.speech_vol = chat_config.speech_vol;
		state.speech_rate = chat_config.speech_rate;
		state.pitch_shift = chat_config.pitch_shift;
		state.talk_mode = parseInt(chat_config.talk_mode, 10) || 0;
		state.tts_mode = chat_config.tts_mode;
		state.anim_mode = chat_config.anim_mode;

		state.avatar_img = app_config.avatar_img;
		state.script_dir = app_config.script_dir;
		state.python_bin = app_config.python_bin;
		state.model_dir = app_config.model_dir;
		state.im_model = app_config.sd_model;
		state.tts_model = app_config.tts_model;
		state.sr_model = app_config.sr_model;
		state.model_args = app_config.model_args;
		state.model_type = app_config.model_type;
		state.imodel_type = app_config.imodel_type;
		state.smodel_type = app_config.smodel_type;
		state.comp_dev = app_config.comp_dev;
		state.start_meth = app_config.start_meth;

		state.gen_mode = app_config.gen_mode;
		state.format_mode = app_config.format_mode;
		state.ui_theme = NormalizeThemeId(app_config.ui_theme);
		ApplyUiTheme(state.ui_theme);
		state.enable_tooluse = app_config.enable_tooluse;
		state.enable_devmode = app_config.enable_devmode;
		state.enable_asasro = app_config.enable_asasro;
		state.start_rec_keys = app_config.start_rec_keys;
		state.stop_rec_keys = app_config.stop_rec_keys;
		state.hf_token = app_config.hf_token;
		state.text_backend = (String(app_config.text_backend || 'local') === 'remote') ? 'remote' : 'local';
		state.remote_base_url = app_config.remote_base_url || '';
		state.remote_model = app_config.remote_model || '';
		state.remote_api_key = app_config.remote_api_key || '';
		state.remote_provider = app_config.remote_provider || '';
		state.remote_tool_format = NormRemoteToolFormat(app_config.remote_tool_format);
		state.remote_prompt_cache = (Number(app_config.remote_prompt_cache) === 0) ? 0 : 1;
		state.workspace = app_config.workspace || '';
		state.agent = app_config.agent || '';
		if (app_config.enabled_rules) state.enabled_rules = app_config.enabled_rules;
		if (app_config.enabled_skills) state.enabled_skills = app_config.enabled_skills;

		state.max_mm = ai_config.msg_mem;
		state.max_rl = ai_config.max_res;
		state.min_rl = ai_config.min_res;
		state.do_sample = ai_config.do_sample;
		state.num_beams = ai_config.num_beams;
		state.beam_groups = ai_config.beam_groups;
		state.b_temp = ai_config.base_temp;
		state.prompt_p = ai_config.prompt_p;
		state.top_k = ai_config.top_k;
		state.top_p = ai_config.top_p;
		state.typical_p = ai_config.typical_p;
		state.rep_penalty = ai_config.rep_penalty;
		state.frequency_penalty = ai_config.frequency_penalty ?? 0;
		state.presence_penalty = ai_config.presence_penalty ?? 0;
		state.compact_at = ai_config.compact_at ?? 0;
		state.compact_tail_tokens = ai_config.compact_tail_tokens ?? 8000;
		state.compact_summary_max = ai_config.compact_summary_max ?? 8192;
		state.temp_format = NormTempFormat(ai_config.temp_format);
		state.enable_thinking = String(ai_config.enable_thinking || 'auto');
		state.reasoning_effort = NormReasoningEffort(ai_config.reasoning_effort);
		if (app_config.hasOwnProperty('mm_processor')) {
			state.mm_processor = NormMmProcessor(app_config.mm_processor);
		}
		state.tools_file = ai_config.tools_file;
		const mergedPacks = Array.isArray(payload.tool_packs) ? payload.tool_packs : ai_config.tool_packs;
		state.tool_packs = NormalizeMergedPacks(mergedPacks, ai_config.tools_file);
		state.tool_packs_applied = JSON.parse(JSON.stringify(state.tool_packs));
		state.firecrawl_api_key = ai_config.firecrawl_api_key || '';

		state.tts_voice = gen_config.tts_voice;

		state.avatar_mp4 = AvatarMp4ForMode(state.anim_mode);

		// Live chrome (talk mode button) must stay in sync even when form fields
		// are skipped - otherwise Apply / scope sync quietly desyncs the icon.
		$('#tmode_img').prop('src', './img/talk_'+state.talk_mode+'.png');

		ipcRenderer.send('show-menubar');
		UpdateWorkspaceLabel({
			label: state.workspace ? state.workspace.split(/[/\\]/).pop() : 'Home',
			workspace: state.workspace || ''
		});

		if (payload.skip_inputs) return;

		$('#tts_mode').val(state.tts_mode);
		$('#anim_mode').val(state.anim_mode);
		$('#voices').val(state.bot_voice);
		$('#speech_vol').val(state.speech_vol);
		$('#speech_rate').val(state.speech_rate);
		$('#speech_pitch').val(state.pitch_shift);

		$('#avatar_img').prop('src', PathToFileURL(state.avatar_img));
		$('#script_dir').val(state.script_dir);
		$('#python_bin').val(state.python_bin);
		$('#model_dir').val(state.model_dir);
		$('#im_model').val(state.im_model);
		$('#tts_model').val(state.tts_model);
		$('#sr_model').val(state.sr_model);
		$('#model_args').val(state.model_args);
		$('#tmodel_select').val(state.model_type);
		$('#imodel_select').val(state.imodel_type);
		$('#smodel_select').val(state.smodel_type);
		$('#device_select').val(state.comp_dev);
		$('#startup_select').val(state.start_meth);

		$('#txtgen_mode').val(state.gen_mode);
		$('#format_mode').val(state.format_mode);
		$('#ui_theme').val(state.ui_theme || 'modern');
		$('#tooluse_enable').val(state.enable_tooluse);
		$('#devmode_enable').val(state.enable_devmode);
		$('#asasro_enable').val(state.enable_asasro);
		$('#start_rec_keys').val(state.start_rec_keys);
		$('#stop_rec_keys').val(state.stop_rec_keys);
		$('#hf_access_token').val(state.hf_token);
		$('#text_backend').val(state.text_backend || 'local');
		$('#remote_base_url').val(state.remote_base_url || '');
		$('#remote_model').val(state.remote_model || '');
		$('#remote_api_key').val(state.remote_api_key || '');
		$('#remote_provider').val(state.remote_provider || '');
		$('#remote_tool_format').val(state.remote_tool_format || 'openai');
		$('#remote_prompt_cache').val(String((Number(state.remote_prompt_cache) === 0) ? 0 : 1));

		$('#max_msg_mem').val(state.max_mm);
		$('#max_res_len').val(state.max_rl);
		$('#min_res_len').val(state.min_rl);
		$('#ds_select').val(state.do_sample);
		$('#num_beams').val(state.num_beams);
		$('#beam_groups').val(state.beam_groups);
		$('#base_temp').val(state.b_temp);
		$('#pp_select').val(state.prompt_p);
		$('#top_k').val(state.top_k);
		$('#top_p').val(state.top_p);
		$('#typical_p').val(state.typical_p);
		$('#rep_penalty').val(state.rep_penalty);
		$('#frequency_penalty').val(state.frequency_penalty ?? 0);
		$('#presence_penalty').val(state.presence_penalty ?? 0);
		$('#compact_at').val(state.compact_at ?? 0);
		$('#compact_tail_tokens').val(state.compact_tail_tokens ?? 8000);
		$('#compact_summary_max').val(state.compact_summary_max ?? 8192);
		UpdateContextUsage({
			used: state.ctx_used || 0,
			limit: Number(state.compact_at) || 0,
			gen_max: Number(state.max_rl) || 0
		});
		$('#template_select').val(NormTempFormat(state.temp_format));
		$('#enable_thinking').val(state.enable_thinking || 'auto');
		$('#reasoning_effort').val(NormReasoningEffort(state.reasoning_effort));
		$('#am_select').val(state.mm_processor || 'auto');
		SyncVisionSelectForBackend(state.text_backend);
		RenderToolPacks();

		$('#ttsgen_mode').val(gen_config.tts_mode);
		$('#tts_voices').val(gen_config.tts_voice);
		$('#tts_speech_vol').val(gen_config.tts_vol);
		$('#tts_speech_rate').val(gen_config.tts_rate);
		$('#tts_speech_pitch').val(gen_config.tts_pitch);

		$('#gen_max_len').val(gen_config.max_len);
		$('#gen_min_len').val(gen_config.min_len);
		$('#gds_select').val(gen_config.do_sample);
		$('#gen_num_beams').val(gen_config.num_beams);
		$('#gen_beam_groups').val(gen_config.beam_groups);
		$('#gen_temp').val(gen_config.temp);
		$('#gen_top_k').val(gen_config.top_k);
		$('#gen_top_p').val(gen_config.top_p);
		$('#gen_typ_p').val(gen_config.typical_p);
		$('#gen_rep_p').val(gen_config.rep_penalty);

		$('#infer_steps').val(gen_config.inference_steps);
		$('#guidance').val(gen_config.guidance_scale);
		$('#safety_check').val(gen_config.safety_check);
		$('#vae_file').val(gen_config.vae_file);
		$('#lora_file').val(gen_config.lora_file);
		$('#lora_dir').val(gen_config.lora_dir);
		$('#lora_scale').val(gen_config.lora_scale);

		if (gen_config.image_width != 'auto' && gen_config.image_height != 'auto') {
			$('#is_select').val('custom');
			$('#img_width').val(gen_config.image_width);
			$('#img_height').val(gen_config.image_height);
			$('#is_input_box').show();
		}
	});
}

export function WireUiEvents() {
	ipcRenderer.on('session-context', (event, payload) => {
		state.workspace = (payload && payload.workspace) || '';
		UpdateWorkspaceLabel(payload || {});
	});

	$('#ui_theme').on('change', function() {
		ApplyUiTheme($(this).val());
	});

	$('#start_rec_keys').on('keydown', function(e) {
		e.preventDefault();
		ShortcutKeyDown('#start_rec_keys', e.key, e.keyCode);
	});

	$('#start_rec_keys').on('keyup', function(e) {
		e.preventDefault();
		ShortcutKeyUp('#start_rec_keys', e.key, e.keyCode);
	});

	$('#stop_rec_keys').on('keydown', function(e) {
		e.preventDefault();
		ShortcutKeyDown('#stop_rec_keys', e.key, e.keyCode);
	});

	$('#stop_rec_keys').on('keyup', function(e) {
		e.preventDefault();
		ShortcutKeyUp('#stop_rec_keys', e.key, e.keyCode);
	});

	$('body').on('keydown', function(e) {
		if ($('#workspace_status_tip').length && !$('#workspace_status_tip').hasClass('hidden') && e.key === 'Escape') {
			e.preventDefault();
			HideWorkspaceStatusTip();
			return;
		}
		if ($('#config_tab').is(':hidden')) {
			if (ShortcutKeyDown('#held_keys', e.key, e.keyCode)) {
				const heldShortcut = $('#held_keys').val();
				if (heldShortcut == state.start_rec_keys) {
					ToggleRecordBox();
				} else if (heldShortcut == state.stop_rec_keys) {
					StopRecording();
				}
			}
		}
	});

	$('body').on('keyup', function(e) {
		if ($('#config_tab').is(':hidden'))
			ShortcutKeyUp('#held_keys', e.key, e.keyCode);
	});

	$('#agent_select').on('change', function() {
		OnAgentChanged();
	});

	$('#rules_toggles, #skills_toggles').on('change', 'input[type=checkbox]', function() {
		PushPromptFileConfig(true);
	});

	$('#is_select').on('change', function() {
		switch ($(this).find(':selected').val()) {
		case 'auto':
			$('#is_input_box').hide();
			break;
		case 'custom':
			$('#is_input_box').show();
			break;
		}
	});

	WirePackEvents();
	WireToolPermEvents();

	$('#mcp_edit_file_btn').on('click', async function() {
		try {
			const res = await window.aiuiAPI.openMcpEditor();
			if (!res || !res.ok) {
				ipcRenderer.send('show-alert', {
					type: 'error',
					msg: (res && res.error) || 'Could not show mcp.json in the file manager'
				});
			}
		} catch (err) {
			ipcRenderer.send('show-alert', { type: 'error', msg: String(err.message || err) });
		}
	});

	WireOverlayDismiss('#attach_dialog', ToggleAttachBox);
	WireOverlayDismiss('#edit_dialog', ToggleEditPrompt);
	WireOverlayDismiss('#prompt_files_dialog', TogglePromptFiles);

	$(document).on('click', function() {
		$('#ctx_usage_tip').addClass('hidden');
	});

	$('#tts_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		let voice_arr = (mode_sel == 0) ? state.sys_voices : state.ai_voices;
		$('#voices').empty();
		for (let i=0; i<voice_arr.length; ++i) {
			const full = String(voice_arr[i] || '');
			$('#voices').append(
				'<option value="'+i+'" title="'+EncodeHTML(full)+'">'+EncodeHTML(VoiceOptionLabel(full))+'</option>'
			);
		}
	});

	$('#ttsgen_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		let voice_arr = (mode_sel == 0) ? state.sys_voices : state.ai_voices;
		const keep = state.tts_voice;
		$('#tts_voices').empty();
		for (let i=0; i<voice_arr.length; ++i) {
			const full = String(voice_arr[i] || '').replaceAll('_', ' ');
			$('#tts_voices').append(
				'<option value="'+i+'" title="'+EncodeHTML(full)+'">'+EncodeHTML(VoiceOptionLabel(full))+'</option>'
			);
		}
		if ($("#tts_voices option[value='"+keep+"']").length > 0) {
			$('#tts_voices').val(keep);
		}
	});

	$('#anim_mode').on('change', function() {
		let mode_sel = $(this).find(':selected').val();
		if (!(Number(mode_sel) >= 0)) return;
		ipcRenderer.send('check-models', { mode: mode_sel, remind_apply: true });
	});

	$('#vclone_mode').on('change', function() {
		if ($(this).find(':selected').val() > 0) {
			$('#ts_txt_row').show();
		} else {
			$('#ts_txt_row').hide();
		}
	});
}
