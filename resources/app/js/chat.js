import { ipcRenderer } from './ipc.js';
import { state } from './state.js';
import {
	EncodeHTML, ConvertBB, EncodeBB, WrapAiXmlTags, StripEmptyThinkTags,
	DecodeAiSafeString, ExtractAiuiBlocks, RestoreAiuiBlocks, SplitSealedOpen, FindOpenAiBlock,
	RandInt, DisableButtons, DisExtraButtons, ThinkAnim, PathToFileURL, GuardEngineIdle, WhenDomReady
} from './util.js';
import {
	ToggleAttachBox, ClearAttachments, AttachFile,
	AttachFiles, AttachLink, RemoveFile, FileDragDrop, FileDragOver,
	FormatAttachmentsBlock
} from './attach.js';

// Re-export attach API for window bindings / older imports.
export {
	ToggleAttachBox, ClearAttachments, AttachFile, AttachFiles, AttachLink,
	RemoveFile, FileDragDrop, FileDragOver
};

/** Rich markdown/BBCode for the newest N transcript messages on session load. */
const REPLAY_RICH_LIMIT = 32;
let chatStickBottom = true;
let chatScrollLock = false;

/** Last clip that can be replayed: video (mp4 has audio) or audio-only TTS — never both. */
let lastReplay = null;

function SetPlaybackUi(playing) {
	const btn = $('#stop_play_btn');
	if (playing) {
		btn.html('Stop Playback').removeClass('hidden');
	} else if (lastReplay) {
		btn.html('Start Playback').removeClass('hidden');
	} else {
		btn.html('Stop Playback').addClass('hidden');
	}
}

function IsMediaPlaying() {
	const audioOn = state.speech_audio !== null && !state.speech_audio.paused;
	const vids = [document.getElementById('avatar_vid'), document.getElementById('other_vid')];
	for (let i = 0; i < vids.length; i++) {
		const vid = vids[i];
		if (vid && !vid.paused && !vid.ended) return true;
	}
	return audioOn;
}

function RefreshPlaybackUi() {
	SetPlaybackUi(IsMediaPlaying());
}

/** Bumped on stop so a pending dual-buffer `oncanplaythrough` cannot restart playback. */
let avatarPlayGen = 0;

function ClearVideoHandlers(vid) {
	if (!vid) return;
	vid.oncanplaythrough = null;
	vid.onplay = null;
	vid.onended = null;
	vid.onpause = null;
}

function StopVideos() {
	avatarPlayGen++;
	const vids = [document.getElementById('avatar_vid'), document.getElementById('other_vid')];
	for (let i = 0; i < vids.length; i++) {
		const vid = vids[i];
		if (!vid) continue;
		ClearVideoHandlers(vid);
		vid.pause();
		$(vid).hide();
	}
	$('#avatar_img').show();
}

export function ClearPlaybackReplay() {
	lastReplay = null;
	if (!IsMediaPlaying()) SetPlaybackUi(false);
}

export function StopPlayback() {
	if (state.speech_audio !== null) {
		state.speech_audio.onplay = null;
		state.speech_audio.onended = null;
		state.speech_audio.onpause = null;
		state.speech_audio.pause();
		state.speech_audio = null;
		$('#read_txt_btn').html('TEXT TO SPEECH');
	}
	StopVideos();
	SetPlaybackUi(false);
}

export function StartPlayback() {
	if (!lastReplay) return;
	if (lastReplay.kind === 'video') {
		PlayVideo();
	} else if (lastReplay.kind === 'audio' && lastReplay.file) {
		PlayAudio(lastReplay.file);
	}
}

export function TogglePlayback() {
	if (IsMediaPlaying()) StopPlayback();
	else StartPlayback();
}

export function PlayVideo() {
	lastReplay = { kind: 'video' };
	const playGen = avatarPlayGen;
	if ($('#other_vid').is(":hidden")) {
		$('#other_source').prop('src', state.avatar_mp4+'?rnd='+RandInt());
		state.avatar_vid = document.getElementById('other_vid');
	} else {
		$('#avatar_source').prop('src', state.avatar_mp4+'?rnd='+RandInt());
		state.avatar_vid = document.getElementById('avatar_vid');
	}
	// Dual-buffer: load into the hidden element, then swap visibility when ready
	// so the previous clip (or still) is not torn down mid-decode (avoids flicker).
	state.avatar_vid.oncanplaythrough = function() {
		if (playGen !== avatarPlayGen) return;
		if ($('#other_vid').is(":hidden")) {
			$('#avatar_vid').hide();
		} else {
			$('#other_vid').hide();
		}
		$('#avatar_img').hide();
		$(state.avatar_vid).show();
		state.avatar_vid.onplay = function() { SetPlaybackUi(true); };
		state.avatar_vid.onended = function() { RefreshPlaybackUi(); };
		state.avatar_vid.onpause = function() { RefreshPlaybackUi(); };
		state.avatar_vid.play();
	};
	state.avatar_vid.load();
}

export function PlayAudio(audio_file) {
	lastReplay = { kind: 'audio', file: audio_file };
	if (state.speech_audio !== null && !state.speech_audio.paused) state.speech_audio.pause();
	state.speech_audio = new Audio(PathToFileURL(audio_file)+'?rnd='+RandInt());
	state.speech_audio.onplay = function() { SetPlaybackUi(true); };
	state.speech_audio.onended = function() {
		$('#read_txt_btn').html('TEXT TO SPEECH');
		RefreshPlaybackUi();
	};
	state.speech_audio.onpause = function() { RefreshPlaybackUi(); };
	state.speech_audio.play();
}

export function ToggleAudio() {
	if (state.speech_audio === null) return;
	if (state.speech_audio.paused) {
		state.speech_audio.play();
	} else {
		state.speech_audio.pause();
	}
}

function FloatTo16BitPCM(output, offset, input) {
	for (let i = 0; i < input.length; i++, offset+=2){
		let s = Math.max(-1, Math.min(1, input[i]));
		output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
	}
}

function WriteString(view, offset, string) {
	for (let i = 0; i < string.length; i++) {
		view.setUint8(offset + i, string.charCodeAt(i));
	}
}

function EncodeWAV(samples) {
	let buffer = new ArrayBuffer(44 + samples.length * 2);
	let view = new DataView(buffer);

	WriteString(view, 0, 'RIFF');
	view.setUint32(4, 36 + samples.length * 2, true);
	WriteString(view, 8, 'WAVE');
	WriteString(view, 12, 'fmt ');
	view.setUint32(16, 16, true);
	view.setUint16(20, 1, true);
	view.setUint16(22, 1, true);
	view.setUint32(24, state.audio_context.sampleRate, true);
	view.setUint32(28, state.audio_context.sampleRate * 2, true);
	view.setUint16(32, 2, true);
	view.setUint16(34, 16, true);
	WriteString(view, 36, 'data');
	view.setUint32(40, samples.length * 2, true);

	FloatTo16BitPCM(view, 44, samples);

	return view;
}

function StartMicrophone() {
	state.audio_context = new AudioContext();

	navigator.mediaDevices.getUserMedia({audio:true, video:false}).then((stream) => {
		StartRecording(stream);
	}).catch((err) => {
		ipcRenderer.send('show-alert', { type: "error", msg: 'Failed to get microphone audio stream.' });
	});
}

function SaveRecording(sample_buff) {
	state.recording = false;
	$('#loading_msg').html('Running speech recognition model ...');
	if (sample_buff.length > 0) {
		let dataView = EncodeWAV(sample_buff);
		ipcRenderer.send('save-recording', dataView);
	} else {
		$('#loading_box').hide();
	}
}

export function StopRecording() {
	if (state.audio_processor_node !== null && state.recording)
		state.audio_processor_node.port.postMessage({msg:"stop"});
	state.recording = false;
}

function InitAudioProcessor() {
	state.audio_processor_node = new AudioWorkletNode(state.audio_context, 'worklet-processor', {});
	state.microphone_stream.connect(state.audio_processor_node);

	state.audio_processor_node.port.onmessage = (e) => {
		if (e.data.msg == "anim") {
			const circleRad = Math.min(5.0, e.data.sample * 10.0) + 3.0;
			$('#rec_circle').attr("r", Math.round(circleRad));
		} else if (e.data.msg == "stop") {
			SaveRecording(e.data.buffer);
		} else if (e.data.msg == "start") {
			$('#loading_msg').html('<svg id="rec_svg"><circle id="rec_circle" r="3" /></svg> Recording. Stop speaking for a few seconds to end recording.');
		}
	};
}

function StartRecording(stream=null) {
	state.recording = true;
	if (stream !== null) {
		state.microphone_stream = state.audio_context.createMediaStreamSource(stream);

		state.audio_context.audioWorklet.addModule('./abp.js').then(() => {
			InitAudioProcessor();
		}).catch((err) => {
			state.recording = false;
			$('#loading_box').hide();
			ipcRenderer.send('show-alert', { type: "error", msg: 'Failed to start recording.' });
		});
	} else {
		InitAudioProcessor();
	}
	$('#loading_msg').html('Speak into your microphone to begin recording.');
	$('#loading_box').show();
}

export function ScrollChatToBottom(smooth, on_done, force) {
	let scroll_box = $('#chat_log');
	const el = scroll_box[0];
	if (!el) {
		if (on_done) on_done();
		return;
	}
	if (!force && !chatStickBottom) {
		SyncChatScrollBtn();
		if (on_done) on_done();
		return;
	}
	chatStickBottom = true;
	chatScrollLock = true;
	// jumpToEnd so prior animate callbacks still run and clear the lock.
	scroll_box.stop(true, true);
	void el.offsetHeight;
	const top = Math.max(0, el.scrollHeight - el.clientHeight);
	const Finish = function () {
		chatScrollLock = false;
		chatStickBottom = true;
		SyncChatScrollBtn();
		if (on_done) on_done();
	};
	if (smooth) {
		scroll_box.animate({scrollTop: top}, 300, 'linear', function() {
			void el.offsetHeight;
			el.scrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
			Finish();
		});
	} else {
		el.scrollTop = top;
		Finish();
	}
}

export function ScrollChatBottom() {
	ScrollChatToBottom(true, null, true);
}

function IsChatNearBottom(el, px) {
	if (!el) return true;
	const slack = (px == null) ? 80 : px;
	return (el.scrollHeight - el.scrollTop - el.clientHeight) <= slack;
}

function SyncChatScrollBtn() {
	const el = document.getElementById('chat_log');
	const btn = document.getElementById('chat_scroll_btn');
	if (!el || !btn) return;
	const near = IsChatNearBottom(el);
	if (!chatScrollLock) chatStickBottom = near;
	const canScroll = el.scrollHeight > el.clientHeight + 8;
	const show = canScroll && !near;
	btn.classList.toggle('hidden', !show);
	btn.setAttribute('aria-hidden', show ? 'false' : 'true');
}

function OnChatLogUserScroll() {
	const el = document.getElementById('chat_log');
	if (!el) return;
	// User gesture wins over in-flight programmatic scroll.
	if (!IsChatNearBottom(el)) {
		chatScrollLock = false;
		chatStickBottom = false;
	}
	SyncChatScrollBtn();
}

function WireChatScrollUi() {
	const el = document.getElementById('chat_log');
	if (!el || el._aiuiScrollWired) return;
	el._aiuiScrollWired = true;
	el.addEventListener('scroll', OnChatLogUserScroll, { passive: true });
	el.addEventListener('wheel', function () {
		requestAnimationFrame(OnChatLogUserScroll);
	}, { passive: true });
	if (typeof ResizeObserver === 'function') {
		const ro = new ResizeObserver(function () { SyncChatScrollBtn(); });
		ro.observe(el);
	}
}

export function StartThinking() {
	state.thinking = true;
	if (typeof window.RenderToolPacks === 'function') {
		try { window.RenderToolPacks(); } catch (e) { /* ignore */ }
	}
	$('#chat_log').addClass('chat_think_pad');
	ScrollChatToBottom(true, function() {
		$('#thinking').html('');
		$('#thinking').show();
		state.dot_count = 1;
		state.loop_timer = setInterval(ThinkAnim, 1000);
	}, true);
}

export function StopThinking() {
	state.thinking = false;
	$('#chat_log').removeClass('chat_think_pad');
	ScrollChatToBottom(true);
	$('#thinking').hide();
	clearInterval(state.loop_timer);
	if (typeof window.RenderToolPacks === 'function') {
		try { window.RenderToolPacks(); } catch (e) { /* ignore */ }
	}
}

export async function SendMsg() {
	if ($('#send_btn').html() == 'STOP') {
		$('#send_btn').prop('disabled', true);
		ipcRenderer.invoke('stop-stream');
		return;
	}
	if (state.thinking || state.generating) return;
	if ($('#send_btn').prop('disabled')) return;
	let messages = [];
	let message = '';
	let html_msg = '';
	if (state.chat_exp) {
		message = $('#area_inp').val().trim();
	} else {
		message = $('#text_inp').val().trim();
	}
	if (message == '') return;
	const fitWarn = OversizedSendWarning(message);
	if (fitWarn) {
		const ok = await ipcRenderer.invoke('confirm-action', {
			title: 'Message may not fit',
			msg: fitWarn,
		});
		if (!ok) return;
	}
	if (state.chat_exp) {
		$('#area_inp').val('');
	} else {
		$('#text_inp').val('');
	}
	messages = message.split('[AIUI_END]');
	message = message.replaceAll("\n", "[AI_UI_BR]");
	for (let i=0; i<messages.length; ++i) {
		let msg = messages[i].trim();
		if (msg == '') continue;
		html_msg += '<div class="msg_box"><div class="chat_msg">';
		if (state.format_mode == 1) {
			html_msg += ConvertBB(EncodeHTML(EncodeBB(msg)));
		} else if (state.format_mode == 2) {
			html_msg += state.md.render(msg);
		} else {
			html_msg += '<p>'+EncodeHTML(msg).replaceAll("\n", "<br>")+'</p>';
		}
		if (state.chat_exp && state.format_mode < 2) msg = msg;
		if (i+1 < messages.length) html_msg += "</div></div>";
	}
	if (html_msg == '') return;
	const filesSnapshot = state.attachments.slice();
	const mmProcessor = state.mm_processor || 'auto';
	html_msg += FormatAttachmentsBlock(filesSnapshot);
	ClearAttachments();
	$('#chat_log').append(html_msg+'</div></div>');
	if (messages.length > 1) {
		let msg_boxes = $("#chat_log div.msg_box").slice(-messages.length);
		if (state.format_mode == 1) {
			msg_boxes.each(function () {
				$(this).find("div.chat_msg p").each(function () {
					if ($(this).html().trim() == '') this.remove();
				});
			});
		}
		msg_boxes.each(function () {
			$(this).find("pre.code_box code").each(function () {
				if ($(this).attr("data-highlighted") != "yes") hljs.highlightElement(this);
			});
		});
	} else {
		let last_cmb = $("#chat_log div.msg_box").last();
		last_cmb.find("div.chat_msg p").each(function () { if ($(this).html().trim() == '') this.remove(); });
		last_cmb.find("pre.code_box code").each(function () {
			if ($(this).attr("data-highlighted") != "yes") hljs.highlightElement(this);
		});
	}
	$('#edit_dialog').addClass('hidden');
	DisableButtons(true, state.gen_mode != 1);
	chatStickBottom = true;
	StartThinking();
	ipcRenderer.send('send-msg', {
		msg: message,
		files: filesSnapshot,
		mm_processor: mmProcessor
	});
}

export async function RedoLast() {
	const ok = await ipcRenderer.invoke('confirm-action', {
		msg: 'Regenerate the last reply? The current last assistant message will be removed.'
	});
	if (!ok) return;
	const $last = $('#chat_log > .msg_box').last();
	if ($last.length && $last.children('.bot_msg').length) {
		$last.remove();
	}
	DisableButtons(true, state.gen_mode != 1);
	ipcRenderer.send('send-msg', { cmd: 'redo_last' });
	chatStickBottom = true;
	StartThinking();
}

export function ContChat() {
	DisableButtons(true, state.gen_mode != 1);
	ipcRenderer.send('send-msg', { cmd: 'cont_chat' });
	chatStickBottom = true;
	StartThinking();
}

function GetTurnDisplayRaw() {
	const base = String(state.turnPatchBase || '');
	const seg = String(state.turnStreamSegment || '');
	if (base && seg) return base + '\n\n' + seg;
	return base || seg;
}

const STREAM_PAINT_MS = 200;
let streamPaintTimer = null;
let streamPaintRaf = null;
let streamPaintPending = false;
let streamLastPaintAt = 0;
let streamSealedRaw = '';
let streamSealedHtml = '';
let toolRunTimer = null;
let toolRunStartedAt = 0;
let toolRunName = '';
let thinkRunTimer = null;
let thinkRunStartedAt = 0;

function ResetStreamPaintState() {
	streamPaintPending = false;
	streamLastPaintAt = 0;
	streamSealedRaw = '';
	streamSealedHtml = '';
	ClearToolRunTimer();
	ClearThinkRunTimer();
	if (streamPaintTimer) {
		clearTimeout(streamPaintTimer);
		streamPaintTimer = null;
	}
	if (streamPaintRaf) {
		cancelAnimationFrame(streamPaintRaf);
		streamPaintRaf = null;
	}
}

function CountOccurrences(s, sub) {
	let n = 0;
	let pos = 0;
	while ((pos = s.indexOf(sub, pos)) >= 0) {
		n++;
		pos += sub.length;
	}
	return n;
}

/** True when the turn has a closed tool_call still waiting on tool_response. */
function TurnAwaitingTool(raw) {
	const s = String(raw || '');
	const tc = CountOccurrences(s, 'AIUI_TC_END') + CountOccurrences(s, '</tool_call>');
	const tr = CountOccurrences(s, 'AIUI_TR_END') + CountOccurrences(s, '</tool_response>');
	return tc > tr;
}

function DecodeAiChrPlaceholders(txt) {
	return String(txt || '').replace(/AI_UI_C(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
}

function LastToolCallName(raw) {
	const s = String(raw || '');
	let inner = '';
	const aiui = s.lastIndexOf('AIUI_TC_START');
	if (aiui >= 0) {
		const end = s.indexOf('AIUI_TC_END', aiui);
		if (end > aiui) inner = s.slice(aiui + 'AIUI_TC_START'.length, end);
	} else {
		const open = s.lastIndexOf('<tool_call>');
		if (open >= 0) {
			const end = s.indexOf('</tool_call>', open);
			if (end > open) inner = s.slice(open + '<tool_call>'.length, end);
		}
	}
	if (!inner) return 'tool';
	try {
		const obj = JSON.parse(DecodeAiChrPlaceholders(inner));
		const name = String((obj && obj.name) || '').trim();
		return name || 'tool';
	} catch (e) {
		return 'tool';
	}
}

function FormatToolRunElapsed(ms) {
	const sec = Math.max(0, Math.floor(ms / 1000));
	if (sec < 60) return sec + 's';
	const m = Math.floor(sec / 60);
	const s = sec % 60;
	return m + 'm ' + s + 's';
}

function ToolRunStatusInner() {
	const elapsed = FormatToolRunElapsed(Date.now() - (toolRunStartedAt || Date.now()));
	const name = toolRunName || 'tool';
	return '<span class="tool_run_pulse" aria-hidden="true"></span>' +
		'Running <code>' + EncodeHTML(name) + '</code> (' + elapsed + ')...';
}

function ToolRunStatusHtml() {
	return '<div id="tool_run_status" class="tool_run_status" aria-live="polite">' +
		ToolRunStatusInner() + '</div>';
}

function RefreshToolRunStatus() {
	const el = document.getElementById('tool_run_status');
	if (!el) return;
	el.innerHTML = ToolRunStatusInner();
}

function EnsureToolRunTimer(name) {
	if (!toolRunTimer) {
		toolRunStartedAt = Date.now();
		toolRunName = name || 'tool';
		toolRunTimer = setInterval(RefreshToolRunStatus, 1000);
	} else if (name && name !== toolRunName) {
		toolRunStartedAt = Date.now();
		toolRunName = name;
	}
	RefreshToolRunStatus();
}

function ClearToolRunTimer() {
	if (toolRunTimer) {
		clearInterval(toolRunTimer);
		toolRunTimer = null;
	}
	toolRunStartedAt = 0;
	toolRunName = '';
}

function ThinkRunStatusInner() {
	const elapsed = FormatToolRunElapsed(Date.now() - (thinkRunStartedAt || Date.now()));
	return '<span class="tool_run_pulse" aria-hidden="true"></span>' +
		'Thinking (' + elapsed + ')...';
}

function RefreshThinkRunStatus() {
	const el = document.querySelector('.ai_think_live > summary');
	if (!el) return;
	el.innerHTML = ThinkRunStatusInner();
}

function EnsureThinkRunTimer() {
	if (!thinkRunTimer) {
		thinkRunStartedAt = Date.now();
		thinkRunTimer = setInterval(RefreshThinkRunStatus, 1000);
	}
	RefreshThinkRunStatus();
}

function ClearThinkRunTimer() {
	if (thinkRunTimer) {
		clearInterval(thinkRunTimer);
		thinkRunTimer = null;
	}
	thinkRunStartedAt = 0;
}

/** Live open think: details stays open; pulse lives in summary; body is escaped text. */
function LiveThinkDetailsHtml(bodyEscaped) {
	return '<details class="ai_act ai_think ai_think_live" open>' +
		'<summary class="think_run_summary" onclick="event.preventDefault()">' +
		ThinkRunStatusInner() + '</summary><hr>' +
		'<div class="details_body details_think">' + bodyEscaped + '</div></details>';
}

/** Open-tail HTML: live think details when an unclosed think tag is present. */
function FormatLiveOpenTail(open) {
	const s = String(open || '');
	if (!s) {
		ClearThinkRunTimer();
		return '';
	}
	const block = FindOpenAiBlock(s);
	if (!block || block.kind !== 'think') {
		ClearThinkRunTimer();
		return FormatPlainTail(s);
	}
	EnsureThinkRunTimer();
	const before = s.slice(0, block.at);
	const inner = s.slice(block.at + block.open.length);
	return FormatPlainTail(before) + LiveThinkDetailsHtml(EncodeHTML(inner));
}

function FormatPlainTail(open) {
	if (!open) return '';
	return EncodeHTML(open).replaceAll('\n', '<br>');
}

/** Live stream: rich-format sealed prefix only; plain-escape the open tail. */
function FormatLiveBotHtml(raw) {
	const { sealed, open } = SplitSealedOpen(raw);
	let sealedHtml = '';
	if (sealed) {
		if (sealed !== streamSealedRaw) {
			streamSealedRaw = sealed;
			streamSealedHtml = FormatBotHtml(sealed, true);
		}
		sealedHtml = streamSealedHtml;
	} else {
		streamSealedRaw = '';
		streamSealedHtml = '';
	}
	return StripEmptyParagraphs(sealedHtml + FormatLiveOpenTail(open));
}

function StickLiveThinkScroll() {
	const body = document.querySelector('#chat_log .ai_think_live .details_body');
	if (body) body.scrollTop = body.scrollHeight;
}

function PaintStreamBox(finalize) {
	if (!state.streaming_bmbl || !state.streaming_bmbl.length) return;
	streamPaintPending = false;
	streamLastPaintAt = Date.now();
	const raw = state.turnBox ? GetTurnDisplayRaw() : String(state.turnStreamRaw || '');
	const withCompact = PrependPendingCompact(raw);
	let html = finalize ? FormatBotHtml(withCompact, true) : FormatLiveBotHtml(withCompact);
	if (finalize) {
		streamSealedRaw = '';
		streamSealedHtml = '';
		ClearThinkRunTimer();
	}
	const awaiting = TurnAwaitingTool(raw);
	if (awaiting) {
		EnsureToolRunTimer(LastToolCallName(raw));
		html += ToolRunStatusHtml();
	} else {
		ClearToolRunTimer();
	}
	state.streaming_bmbl.html('<div class="bot_msg">'+html+'</div>');
	if (html.trim() !== '') state.streaming_bmbl.removeClass('hidden').show();
	if (finalize) FinalizeBotBox(state.streaming_bmbl);
	else GroupAgentActionDetails(state.streaming_bmbl);
	if (!finalize) StickLiveThinkScroll();
	ScrollChatToBottom(false);
}

function ScheduleStreamPaint(finalize) {
	if (finalize) {
		if (streamPaintTimer) {
			clearTimeout(streamPaintTimer);
			streamPaintTimer = null;
		}
		if (streamPaintRaf) {
			cancelAnimationFrame(streamPaintRaf);
			streamPaintRaf = null;
		}
		streamPaintPending = false;
		PaintStreamBox(true);
		return;
	}
	streamPaintPending = true;
	const elapsed = Date.now() - streamLastPaintAt;
	const runPaint = () => {
		streamPaintRaf = null;
		if (streamPaintPending) PaintStreamBox(false);
	};
	if (elapsed >= STREAM_PAINT_MS) {
		if (!streamPaintRaf) streamPaintRaf = requestAnimationFrame(runPaint);
		return;
	}
	if (!streamPaintTimer) {
		streamPaintTimer = setTimeout(() => {
			streamPaintTimer = null;
			if (streamPaintPending && !streamPaintRaf) {
				streamPaintRaf = requestAnimationFrame(runPaint);
			}
		}, STREAM_PAINT_MS - elapsed);
	}
}

function UpdateTurnBox(finalize=false) {
	if (!state.streaming_bmbl || !state.streaming_bmbl.length) return;
	ScheduleStreamPaint(!!finalize);
}

/** Drop paragraphs that only contain whitespace / breaks (BBCode/markdown leftovers). */
function StripEmptyParagraphs(html) {
	return String(html || '').replace(/<p>(?:\s|&nbsp;|<br\s*\/?>)*<\/p>/gi, '');
}

function IsEmptyParagraph(el) {
	const $el = $(el);
	if ($el.find('img, video, audio, iframe, details, pre, table, svg, object, embed').length) return false;
	return $el.text().replace(/\u00a0/g, ' ').trim() === '';
}

function FormatBotHtml(raw, rich=true) {
	let bot_msg = StripEmptyThinkTags(String(raw || '').replaceAll("[AI_UI_TAB]", "\t").replaceAll("[AI_UI_BR]", "\n").trim());
	if (bot_msg === '') return '';
	const wrapped = WrapAiXmlTags(bot_msg);
	// Keep think/tool bodies out of markdown/BB so indent and JSON escapes survive.
	const { text, blocks } = ExtractAiuiBlocks(wrapped);
	let html = '';
	if (rich && state.format_mode == 1) {
		html = ConvertBB(EncodeHTML(EncodeBB(text)));
	} else if (rich && state.format_mode == 2 && state.md) {
		html = state.md.render(text);
	} else {
		html = EncodeHTML(text).replaceAll("\n", "<br>");
		if (html.indexOf('<p>') !== 0) {
			html = '<p>' + html + '</p>';
		}
	}
	html = RestoreAiuiBlocks(html, blocks);
	return StripEmptyParagraphs(DecodeAiSafeString(html));
}

function FormatMsgHtml(text, as_bot, rich) {
	if (as_bot) return FormatBotHtml(text, rich);
	let body = '';
	const prepared = text;
	if (rich && state.format_mode == 1) {
		body = ConvertBB(EncodeHTML(EncodeBB(prepared)));
	} else if (rich && state.format_mode == 2 && state.md) {
		body = state.md.render(prepared);
	} else {
		body = '<p>' + EncodeHTML(prepared).replaceAll("\n", "<br>") + '</p>';
	}
	return DecodeAiSafeString(body);
}

/** Flatten string or multimodal content parts for session replay. */
function FlattenMessageContent(content) {
	if (typeof content === 'string') {
		return { text: content, files: [] };
	}
	if (!Array.isArray(content)) {
		return { text: content == null ? '' : String(content), files: [] };
	}
	const texts = [];
	const files = [];
	for (let i = 0; i < content.length; ++i) {
		const part = content[i];
		if (typeof part === 'string') {
			if (part) texts.push(part);
			continue;
		}
		if (!part || typeof part !== 'object') continue;
		const t = String(part.type || '');
		if (t === 'text') {
			const tx = String(part.text || '');
			if (tx) texts.push(tx);
			continue;
		}
		let path = '';
		if (t === 'image' || t === 'image_url') {
			const iu = part.image_url;
			path = part.image || part.url || (iu && (typeof iu === 'string' ? iu : iu.url)) || '';
		} else if (t === 'video' || t === 'video_url') {
			const vu = part.video_url;
			path = part.video || part.url || (vu && (typeof vu === 'string' ? vu : vu.url)) || '';
		} else if (t === 'audio' || t === 'file') {
			path = part.audio || part.file || part.url || '';
		}
		path = String(path || '').trim();
		if (path) files.push(path);
	}
	return { text: texts.join('\n'), files };
}

function FormatReplayAttachments(files) {
	return FormatAttachmentsBlock(files);
}

export function ReplayChatMessages(messages, init_prompt) {
	const decodeBr = (t) => String(t || '').replaceAll('[AI_UI_BR]', '\n');
	const prompt_txt = decodeBr(
		(init_prompt != null && init_prompt !== '')
			? init_prompt
			: (messages && messages[0] && messages[0].role === 'system' ? (messages[0].content || '') : (state.prompt || ''))
	);
	state.prompt = prompt_txt;
	$('#prompt_txta').val(prompt_txt);
	const preview = (prompt_txt || '').trim().split(/\r?\n/).map(s => s.trim()).filter(Boolean)[0] || 'New session';
	$('#init_prompt').html(EncodeHTML(preview));
	$('#prompt_bar').removeClass('hidden');
	let html = '';
	const list = Array.isArray(messages) ? messages : [];
	const display = [];
	let pendingCompact = null;
	for (let i = 0; i < list.length; ++i) {
		const m = list[i];
		if (!m || typeof m !== 'object') continue;
		const role = m.role;
		if (role === 'system') continue;
		if (role === 'compact' || m._aiui_compact) {
			pendingCompact = m;
			continue;
		}
		if (role === 'tool') {
			display.push({ kind: 'tool', m });
			continue;
		}
		if (m.tool_calls) {
			display.push({ kind: 'tool_calls', m });
			continue;
		}
		const flat = FlattenMessageContent(m.content);
		const text = decodeBr(flat.text);
		if (!text.trim() && (!flat.files || !flat.files.length)) continue;
		const item = {
			kind: 'chat',
			text,
			files: flat.files || [],
			as_bot: (role === 'assistant')
		};
		if (item.as_bot && pendingCompact) {
			item.compact = pendingCompact;
			pendingCompact = null;
		}
		display.push(item);
	}
	if (pendingCompact) {
		display.push({ kind: 'compact', m: pendingCompact });
	}
	const collapsed = [];
	for (let i = 0; i < display.length; ++i) {
		const item = display[i];
		if (item.kind === 'chat' && item.as_bot && collapsed.length > 0) {
			const prev = collapsed[collapsed.length - 1];
			if (prev.kind === 'chat' && prev.as_bot && !(prev.files && prev.files.length) && !(item.files && item.files.length) && !item.compact) {
				prev.text = prev.text + '\n\n' + item.text;
				continue;
			}
		}
		collapsed.push(item);
	}
	const richFrom = Math.max(0, collapsed.length - REPLAY_RICH_LIMIT);
	for (let i = 0; i < collapsed.length; ++i) {
		const item = collapsed[i];
		const rich = i >= richFrom;
		if (item.kind === 'compact') {
			// Orphan compact (no following assistant) — still show inside a bot box.
			html += '<div class="msg_box"><div class="bot_msg">'
				+ FormatBotHtml(CompactMarkerRaw(item.m.content || '', item.m.compact_count), rich)
				+ '</div></div>';
			continue;
		}
		if (item.kind === 'tool') {
			const t = decodeBr(typeof item.m.content === 'string' ? item.m.content : JSON.stringify(item.m.content || ''));
			html += '<div class="msg_box"><div class="bot_msg"><div class="dark_bg"><small>tool'+(item.m.name ? ' - '+EncodeHTML(item.m.name) : '')+'</small><pre class="tool_pre">'+EncodeHTML(t)+'</pre></div></div></div>';
			continue;
		}
		if (item.kind === 'tool_calls') {
			html += '<div class="msg_box"><div class="bot_msg"><div class="dark_bg"><small>tool_call</small><pre class="tool_pre">'+EncodeHTML(JSON.stringify(item.m.tool_calls, null, 2))+'</pre></div></div></div>';
			continue;
		}
		const cls = item.as_bot ? 'bot_msg' : 'chat_msg';
		let body = '';
		if (item.as_bot) {
			const raw = (item.compact
				? CompactMarkerRaw(item.compact.content || '', item.compact.compact_count) + '\n\n'
				: '') + item.text;
			body = FormatBotHtml(raw, rich);
		} else {
			body = FormatMsgHtml(item.text, false, rich);
		}
		html += '<div class="msg_box"><div class="'+cls+'">'+body+FormatReplayAttachments(item.files)+'</div></div>';
	}
	$('#chat_log').children().not('#prompt_bar, #compact_run_status').remove();
	$('#chat_log').append(html);
	FinalizeBotBox($('#chat_log'));
	WireChatScrollUi();
	ScrollChatToBottom(false, null, true);
}

/** Raw markers so FormatBotHtml / ConvertAITags render like think details. */
function CompactMarkerRaw(summary, compactCount) {
	const body = String(summary || '').trim() || '(empty summary)';
	return 'AIUI_COMPACT_START\n\n' + body + 'AIUI_COMPACT_END';
}

function PrependPendingCompact(raw) {
	if (!state.pendingCompact) return String(raw || '');
	const p = state.pendingCompact;
	return CompactMarkerRaw(p.summary || '', p.compact_count) + '\n\n' + String(raw || '');
}

function RequestAutosave() {
	// Only after generation has finished (never mid-stream).
	if (state.thinking || state.generating) return;
	if ($('#send_btn').html() == 'STOP') return;
	ipcRenderer.invoke('session-autosave').catch(() => {});
}

export function ToggleMsgBox() {
	if (state.chat_exp) {
		$('#exp_box').hide();
		$('#text_inp').prop('disabled', false);
		$('#text_inp').focus();
	} else {
		$('#exp_box').show();
		$('#text_inp').prop('disabled', true);
		$('#area_inp').focus();
	}
	state.chat_exp = !state.chat_exp;
}

export function ToggleDetails(elem) {
	let summary = $(elem).html();
	if (summary.includes("show")) {
		$(elem).html(summary.replace("show", "hide"));
	} else {
		$(elem).html(summary.replace("hide", "show"));
	}
}

export function ToggleRecordBox() {
	if ($('#mic_btn').prop('disabled')) return;
	if (state.audio_context === null) {
		StartMicrophone();
	} else if (!state.recording) {
		StartRecording();
	}
}

export function ToggleInputDialog(click_func, dialog_title, placeholder_txt='') {
	if ($('#input_dialog').is(":hidden")) {
		$('#in_dialog_head').html(dialog_title);
		$('#input_txta').attr("placeholder", placeholder_txt);
		$('#in_accept_btn').off('click').on('click', function(e) {
			e.preventDefault();
			ToggleInputDialog();
			click_func();
		});
		$('#input_dialog').show();
		$('#input_txta').focus();
	} else {
		$('#input_dialog').hide();
	}
}

function FinalizeBotBox(box, rich=true) {
	if (!box || !box.length) return;
	box.find('div.bot_msg p').each(function () { if (IsEmptyParagraph(this)) this.remove(); });
	GroupAgentActionDetails(box);
	box.find("pre.code_box code").each(function () {
		if ($(this).attr("data-highlighted") != "yes") hljs.highlightElement(this);
	});
	box.find('img').on('load', function () { ScrollChatToBottom(false); });
}

const AGENT_ACT_MIN = 4; // group when more than 3 sequential agent details

function IsAgentActDetails(el) {
	return el && el.nodeType === 1 && el.tagName === 'DETAILS'
		&& el.classList.contains('ai_act')
		&& !el.classList.contains('agent_actions')
		&& !el.classList.contains('ai_think_live');
}

function IsSkippableBetweenDetails(node) {
	if (!node) return true;
	if (node.nodeType === 3) return !String(node.textContent || '').trim();
	if (node.nodeType !== 1) return true;
	const tag = node.tagName;
	if (tag === 'BR' || tag === 'HR') return true;
	if (tag === 'P' && IsEmptyParagraph(node)) return true;
	return false;
}

function UnwrapAgentActionGroups(msgEl) {
	msgEl.querySelectorAll('details.agent_actions').forEach(function (wrap) {
		const parent = wrap.parentNode;
		if (!parent) return;
		Array.from(wrap.children).forEach(function (child) {
			if (child.tagName === 'DETAILS' && child.classList.contains('ai_act')) {
				parent.insertBefore(child, wrap);
			}
		});
		wrap.remove();
	});
}

function GroupAgentActionDetails(box) {
	if (!box || !box.length) return;
	box.find('div.bot_msg').each(function () {
		const msg = this;
		UnwrapAgentActionGroups(msg);
		const nodes = Array.from(msg.childNodes);
		let run = [];

		function FlushRun() {
			if (run.length >= AGENT_ACT_MIN) {
				const wrap = document.createElement('details');
				wrap.className = 'agent_actions';
				const sum = document.createElement('summary');
				sum.setAttribute('onclick', 'ToggleDetails(this)');
				sum.textContent = 'click to show ' + run.length + ' agent actions';
				wrap.appendChild(sum);
				wrap.appendChild(document.createElement('hr'));
				const first = run[0];
				first.parentNode.insertBefore(wrap, first);
				run.forEach(function (d) { wrap.appendChild(d); });
			}
			run = [];
		}

		for (let i = 0; i < nodes.length; i++) {
			const node = nodes[i];
			if (IsAgentActDetails(node)) {
				run.push(node);
			} else if (run.length && IsSkippableBetweenDetails(node)) {
				continue;
			} else {
				FlushRun();
			}
		}
		FlushRun();
		msg.querySelectorAll(':scope > p').forEach(function (p) {
			if (IsEmptyParagraph(p)) p.remove();
		});
	});
}

const CTX_USAGE_R = 26;
const CTX_USAGE_C = 2 * Math.PI * CTX_USAGE_R;

function FormatTokenCount(n) {
	n = Math.max(0, Number(n) || 0);
	if (n >= 1000000) {
		const s = (n / 1000000).toFixed(1);
		return (s.endsWith('.0') ? s.slice(0, -2) : s) + 'M';
	}
	if (n >= 1000) {
		const s = (n / 1000).toFixed(1);
		return (s.endsWith('.0') ? s.slice(0, -2) : s) + 'k';
	}
	return String(Math.round(n));
}

function HeuristicTokensFromText(text) {
	return Math.max(0, Math.floor((String(text || '').length + 3) / 4));
}

function OversizedSendWarning(text) {
	const limit = Number(state.compact_at) || 0;
	if (limit <= 0) return '';
	const msgTok = HeuristicTokensFromText(text);
	const sys = Number(state.ctx_system) || 0;
	const tools = Number(state.ctx_tools) || 0;
	const sum = Number(state.ctx_summary) || 0;
	const overhead = sys + tools + sum;
	const tail = Number(state.compact_tail_tokens) || 8000;
	const needed = overhead + msgTok;
	if (msgTok > limit || needed > limit) {
		let msg = 'This message is about ' + FormatTokenCount(msgTok) + ' tokens.';
		if (overhead > 0) {
			msg += ' With the system prompt, tools, and any summary ('
				+ FormatTokenCount(overhead) + '), that is about '
				+ FormatTokenCount(needed) + ' tokens.';
		}
		msg += ' The Context Token Budget is ' + FormatTokenCount(limit)
			+ '. Send anyway? The model may not see the full message.';
		return msg;
	}
	if (msgTok > tail) {
		return 'This message is about ' + FormatTokenCount(msgTok)
			+ ' tokens, larger than Compact Recent Tail ('
			+ FormatTokenCount(tail)
			+ '). After compaction the last user message is always kept, '
			+ 'but there may be little room for other context. Send anyway?';
	}
	return '';
}

function FormatUsagePct(part, whole) {
	whole = Number(whole) || 0;
	part = Math.max(0, Number(part) || 0);
	if (whole <= 0) return '0%';
	const pct = Math.min(100, (part / whole) * 100);
	const s = pct.toFixed(1);
	return (s.endsWith('.0') ? s.slice(0, -2) : s) + '%';
}

function CtxUsageHoverText() {
	const used = state.ctx_used || 0;
	const limit = state.ctx_limit || 0;
	const gen = state.ctx_gen_max || state.max_rl || 0;
	if (limit <= 0) return 'Context compaction off';
	const free = Math.max(0, limit - used);
	let line = 'Context usage: ' + FormatTokenCount(used) + ' / ' + FormatTokenCount(limit)
		+ ' tokens (' + FormatUsagePct(used, limit) + ')';
	line += '\nFree headroom: ' + FormatTokenCount(free);
	if (gen > 0) line += ' (max reply: ' + FormatTokenCount(gen) + ')';
	return line;
}

function CtxUsageTipHead() {
	const used = state.ctx_used || 0;
	const limit = state.ctx_limit || 0;
	if (limit <= 0) return 'Context compaction off';
	return 'Context usage: ' + FormatTokenCount(used) + ' / ' + FormatTokenCount(limit)
		+ ' (' + FormatUsagePct(used, limit) + ')';
}

function CtxUsageBarSegments() {
	const limit = state.ctx_limit || 0;
	if (limit <= 0) return [];
	if (state.ctx_remote) {
		const used = Math.max(0, state.ctx_used || 0);
		const free = Math.max(0, limit - used);
		return [
			{ key: 'used', n: used },
			{ key: 'free', n: free }
		];
	}
	const sys = Math.max(0, state.ctx_system || 0);
	const sum = Math.max(0, state.ctx_summary || 0);
	const tools = Math.max(0, state.ctx_tools || 0);
	const conv = Math.max(0, state.ctx_conversation || 0);
	const free = (state.ctx_free != null && state.ctx_free !== '')
		? Math.max(0, Number(state.ctx_free) || 0)
		: Math.max(0, limit - (state.ctx_used || 0));
	return [
		{ key: 'system', n: sys },
		{ key: 'tools', n: tools },
		{ key: 'summary', n: sum },
		{ key: 'conversation', n: conv },
		{ key: 'free', n: free }
	];
}

function CtxUsageBarHtml() {
	const limit = state.ctx_limit || 0;
	if (limit <= 0) return '';
	const segs = CtxUsageBarSegments().filter(s => s.n > 0 || s.key === 'free');
	let html = '<div class="ctx_usage_bar" title="Context budget">';
	for (const s of segs) {
		const pct = Math.max(0, Math.min(100, (s.n / limit) * 100));
		if (pct <= 0 && s.key !== 'free') continue;
		html += '<span class="ctx_bar_seg ctx_bar_' + s.key + '" style="width:' + pct + '%"></span>';
	}
	html += '</div>';
	return html;
}

function CtxUsageTipRows() {
	const used = state.ctx_used || 0;
	const limit = state.ctx_limit || 0;
	const gen = state.ctx_gen_max || state.max_rl || 0;
	if (limit <= 0) return [];
	const free = (state.ctx_free != null && state.ctx_free !== '')
		? Math.max(0, Number(state.ctx_free) || 0)
		: Math.max(0, limit - used);
	const rows = [];
	if (!state.ctx_remote) {
		const sys = state.ctx_system || 0;
		const sum = state.ctx_summary || 0;
		const conv = state.ctx_conversation || 0;
		const tools = state.ctx_tools || 0;
		if (sys > 0) {
			rows.push({
				key: 'system',
				label: 'System prompt',
				value: FormatTokenCount(sys) + ' (' + FormatUsagePct(sys, limit) + ')'
			});
		}
		if (tools > 0) {
			rows.push({
				key: 'tools',
				label: 'Tool definitions',
				value: FormatTokenCount(tools) + ' (' + FormatUsagePct(tools, limit) + ')'
			});
		}
		if (sum > 0) {
			rows.push({
				key: 'summary',
				label: 'Context summary',
				value: FormatTokenCount(sum) + ' (' + FormatUsagePct(sum, limit) + ')'
			});
		}
		rows.push({
			key: 'conversation',
			label: 'Conversation',
			value: FormatTokenCount(conv) + ' (' + FormatUsagePct(conv, limit) + ')'
		});
		rows.push({
			key: 'free',
			label: 'Free headroom',
			value: FormatTokenCount(free) + ' (' + FormatUsagePct(free, limit) + ')'
		});
	} else {
		rows.push({
			key: 'free',
			label: 'Free headroom',
			value: FormatTokenCount(free)
				+ (gen > 0 ? ' (max reply: ' + FormatTokenCount(gen) + ')' : '')
		});
	}
	return rows;
}

function PaintCtxUsageTip() {
	const head = CtxUsageTipHead();
	const rows = CtxUsageTipRows();
	let html = '<div class="ctx_tip_head">' + EncodeHTML(head) + '</div>';
	if ((state.ctx_limit || 0) > 0) html += CtxUsageBarHtml();
	if (rows.length) {
		html += '<div class="ctx_tip_lines">';
		for (const r of rows) {
			html += '<div class="ctx_tip_line"><span class="ctx_tip_lbl ctx_lbl_' + r.key + '">'
				+ EncodeHTML(r.label) + ':</span> ' + EncodeHTML(r.value) + '</div>';
		}
		html += '</div>';
	}
	$('#ctx_usage_tip_body').html(html);
}

export function SetCtxCompacting(active) {
	const $status = $('#compact_run_status');
	const $prompt = $('#prompt_bar');
	if (!$status.length) return;
	if (active) {
		$prompt.addClass('hidden');
		$status.removeClass('hidden');
	} else {
		$status.addClass('hidden');
		if (String(state.prompt || '').trim()) {
			$prompt.removeClass('hidden');
		}
	}
}

export function UpdateContextUsage(payload) {
	payload = payload || {};
	if (Object.prototype.hasOwnProperty.call(payload, 'compacting')) {
		SetCtxCompacting(!!payload.compacting);
		// Pulse-only events omit used/limit — keep the last ring values.
		if (!Object.prototype.hasOwnProperty.call(payload, 'used')
			&& !Object.prototype.hasOwnProperty.call(payload, 'limit')) {
			return;
		}
	}

	const used = Math.max(0, Number(payload.used) || 0);
	const limit = Math.max(0, Number(payload.limit) || 0);
	const gen = Math.max(0, Number(payload.gen_max) || 0);
	state.ctx_used = used;
	state.ctx_limit = limit;
	if (gen > 0) state.ctx_gen_max = gen;
	state.ctx_remote = !!payload.remote;
	if (!state.ctx_remote) {
		state.ctx_system = Math.max(0, Number(payload.system) || 0);
		state.ctx_summary = Math.max(0, Number(payload.summary) || 0);
		state.ctx_conversation = Math.max(0, Number(payload.conversation) || 0);
		state.ctx_tools = Math.max(0, Number(payload.tools) || 0);
		state.ctx_free = (payload.free != null)
			? Math.max(0, Number(payload.free) || 0)
			: Math.max(0, limit - used);
	} else {
		state.ctx_system = 0;
		state.ctx_summary = 0;
		state.ctx_conversation = 0;
		state.ctx_tools = 0;
		state.ctx_free = Math.max(0, limit - used);
	}

	const $btn = $('#ctx_usage_btn');
	const $arc = $('#ctx_usage_arc');
	const $tip = $('#ctx_usage_tip');
	if (limit <= 0) {
		$btn.addClass('hidden');
		$tip.addClass('hidden');
		SetCtxCompacting(false);
		return;
	}
	$btn.removeClass('hidden');
	const pct = Math.max(0, Math.min(1, used / limit));
	$arc.css({
		'stroke-dasharray': String(CTX_USAGE_C),
		'stroke-dashoffset': String(CTX_USAGE_C * (1 - pct))
	});
	$btn.attr('title', CtxUsageHoverText());
	PaintCtxUsageTip();
}

export function ToggleCtxUsageTip(ev) {
	if (ev && typeof ev.stopPropagation === 'function') ev.stopPropagation();
	const $btn = $('#ctx_usage_btn');
	if ($btn.hasClass('hidden')) return;
	const $tip = $('#ctx_usage_tip');
	if ($tip.hasClass('hidden')) {
		PaintCtxUsageTip();
		$tip.removeClass('hidden');
	} else {
		$tip.addClass('hidden');
	}
}

export function CompactNow(ev) {
	if (ev && typeof ev.stopPropagation === 'function') ev.stopPropagation();
	if (!GuardEngineIdle('Wait for the current task to finish before compacting.')) return;
	if (!(state.ctx_limit > 0)) {
		ipcRenderer.send('show-alert', { type: 'error', msg: 'Set Context Token Budget above 0 first.' });
		return;
	}
	$('#ctx_usage_tip').addClass('hidden');
	state._manualCompact = true;
	DisableButtons(true);
	DisExtraButtons(true);
	StartThinking();
	ipcRenderer.send('send-msg', { cmd: 'compact_now' });
}

function AppendOrphanCompact(summary, compactCount) {
	const html = '<div class="msg_box"><div class="bot_msg">'
		+ FormatBotHtml(CompactMarkerRaw(summary, compactCount), true)
		+ '</div></div>';
	$('#chat_log').append(html);
	FinalizeBotBox($('#chat_log div.msg_box').last());
	// Manual compact still has thinking=true until ai-ready; save directly.
	ipcRenderer.invoke('session-autosave').catch(() => {});
}

export function InitChat() {
	// IPC handlers register immediately; DOM wiring waits until nodes exist.
	WhenDomReady(function () {
		WireChatScrollUi();
		SyncChatScrollBtn();
		UpdateContextUsage({ used: 0, limit: 0, gen_max: 0 });
		$('#ctx_usage_tip').on('click', function(e) {
			e.stopPropagation();
		});
	});

	ipcRenderer.on('context-usage', (event, payload) => {
		UpdateContextUsage(payload || {});
	});

	ipcRenderer.on('context-compact', (event, payload) => {
		if (payload && payload.ok === false) {
			state.pendingCompact = null;
			state._manualCompact = false;
			SetCtxCompacting(false);
			ipcRenderer.send('show-alert', {
				type: 'error',
				msg: (payload && payload.error) || 'Compact failed.'
			});
			return;
		}
		state.pendingCompact = {
			summary: (payload && payload.summary) || '',
			compact_count: payload && payload.compact_count
		};
		// Mid-turn compact merges into the streaming bot box.
		if (state.turnActive && state.streaming_bmbl && state.streaming_bmbl.length) {
			state._manualCompact = false;
			PaintStreamBox(false);
			return;
		}
		// Manual compact has no following bot turn — show its own box.
		// Auto compact fires before turn_start; keep pending for merge there.
		if (state._manualCompact) {
			state._manualCompact = false;
			AppendOrphanCompact(state.pendingCompact.summary, state.pendingCompact.compact_count);
			state.pendingCompact = null;
		}
	});

	ipcRenderer.on('turn-start', () => {
		state.turnActive = true;
		state.turnBox = true;
		state.turnPatchBase = '';
		state.turnStreamSegment = '';
		state.turnStreamRaw = '';
		ResetStreamPaintState();
		const msg_html = '<div class="msg_box hidden"><div class="bot_msg"></div></div>';
		$('#chat_log').append(msg_html);
		state.streaming_bmbl = $("#chat_log div.msg_box").last();
		state.streaming_bmpf = null;
		if (state.pendingCompact) {
			PaintStreamBox(false);
		}
	});

	ipcRenderer.on('turn-patch', (event, payload) => {
		state.turnPatchBase = String(payload.text || '').replaceAll("[AI_UI_BR]", "\n");
		state.turnStreamSegment = '';
		streamSealedRaw = '';
		streamSealedHtml = '';
		if (!state.streaming_bmbl || !state.streaming_bmbl.length) return;
		UpdateTurnBox(true);
	});

	ipcRenderer.on('turn-end', () => {
		// turnActive stays true until bot-msg applies the final formatted turn.
	});

	ipcRenderer.on('stream-start', (event, payload) => {
		if (state.turnActive && state.streaming_bmbl && state.streaming_bmbl.length) {
			return;
		}
		state.turnActive = true;
		state.turnStreamRaw = '';
		ResetStreamPaintState();
		const msg_html = '<div class="msg_box hidden"><div class="bot_msg"></div></div>';
		$('#chat_log').append(msg_html);
		state.streaming_bmbl = $("#chat_log div.msg_box").last();
		state.streaming_bmpf = null;
	});

	ipcRenderer.on('stream-text', (event, payload) => {
		const delta = String(payload.txt || '').replaceAll("[AI_UI_BR]", "\n");
		if (state.turnBox) {
			state.turnStreamSegment += delta;
			ScheduleStreamPaint(false);
			return;
		}
		state.turnStreamRaw += delta;
		if (!state.streaming_bmbl || !state.streaming_bmbl.length) return;
		ScheduleStreamPaint(false);
	});

	ipcRenderer.on('stream-end', () => {
		if (state.turnBox) {
			UpdateTurnBox(true);
			return;
		}
		if (!state.streaming_bmbl || !state.streaming_bmbl.length) return;
		ScheduleStreamPaint(true);
	});

	ipcRenderer.on('bot-msg', (event, payload) => {
		ResetStreamPaintState();
		let last_bmb = state.streaming_bmbl;
		const bot_msg = FormatBotHtml(PrependPendingCompact(payload.msg), true);
		state.pendingCompact = null;
		if (bot_msg !== '') {
			if (last_bmb !== null && last_bmb.length)  {
				last_bmb.html('<div class="bot_msg">'+bot_msg+'</div>');
				last_bmb.removeClass('hidden').show();
			} else {
				$('#chat_log').append('<div class="msg_box"><div class="bot_msg">'+bot_msg+'</div></div>');
				last_bmb = $("#chat_log div.msg_box").last();
			}
			FinalizeBotBox(last_bmb);
		} else if (last_bmb !== null && last_bmb.length) {
			const streamed = String(state.turnStreamRaw || '').trim();
			const painted = last_bmb.find('.bot_msg').text().replace(/\u00a0/g, ' ').trim();
			if (streamed !== '' || painted !== '') {
				// STOP / empty final event with tokens already on screen - keep them.
				last_bmb.removeClass('hidden').show();
				FinalizeBotBox(last_bmb);
			} else if (last_bmb.hasClass('hidden')) {
				// Unused streaming stub (no tokens) - drop so REDO targets a real message.
				last_bmb.remove();
			}
		}
		state.turnActive = false;
		state.turnBox = false;
		state.turnPatchBase = '';
		state.turnStreamSegment = '';
		state.turnStreamRaw = '';
		state.streaming_bmbl = null;
		DisableButtons(false);
		StopThinking();
		if (payload.got_vid && state.talk_mode == 0) {
			PlayVideo();
		}
		RequestAutosave();
	});

	ipcRenderer.on('session-loaded', (event, payload) => {
		ReplayChatMessages(payload.messages || [], payload.init_prompt || '');
		DisableButtons(false);
		StopThinking();
	});

	ipcRenderer.on('session-cleared', (event, payload) => {
		state.apply_compose = true;
		state.streaming_bmbl = null;
		state.streaming_bmpf = null;
		$('#chat_log').children().not('#prompt_bar, #compact_run_status').remove();
		const preview = (state.prompt || '').trim().split(/\r?\n/).map(s => s.trim()).filter(Boolean)[0] || 'New session';
		$('#init_prompt').html(EncodeHTML(preview));
		if (String(state.prompt || '').trim()) $('#prompt_bar').removeClass('hidden');
		DisableButtons(false);
		StopThinking();
	});

	ipcRenderer.on('start-recording', (event, payload) => {
		ToggleRecordBox();
	});

	ipcRenderer.on('stop-recording', (event, payload) => {
		StopRecording();
	});
}
