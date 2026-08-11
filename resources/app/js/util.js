import { ipcRenderer } from './ipc.js';
import { state } from './state.js';

export const UI_THEMES = [
	{ id: 'classic', name: 'Classic' },
	{ id: 'modern', name: 'Modern' },
];

export function NormalizeThemeId(raw) {
	const id = String(raw || 'modern').trim().toLowerCase();
	return UI_THEMES.some((t) => t.id === id) ? id : 'modern';
}

/** Run after the document body exists (bundles may load from head or end of body). */
export function WhenDomReady(fn) {
	if (typeof fn !== 'function') return;
	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', fn, { once: true });
	} else {
		fn();
	}
}

/** Swap the active theme stylesheet and remember it for the next cold start. */
export function ApplyUiTheme(themeId) {
	const id = NormalizeThemeId(themeId);
	const link = document.getElementById('theme_css');
	if (link) link.href = './css/themes/' + id + '.css';
	try {
		localStorage.setItem('aiui_theme', id);
	} catch (e) { /* ignore */ }
	return id;
}

export const bb_code_tags = [
	"B", "b", "I", "i", "U", "u", "S", "s", "OL", "ol", "UL", "ul", "LI", "li", "IMG", "img", "URL", "url",
	"H1", "h1", "H2", "h2", "H3", "h3", "H4", "h4", "H5", "h5", "H6", "h6", "PRE", "pre", "CENTER", "center",
	"SPOILER", "spoiler", "QUOTE", "quote", "AUDIO", "audio", "VIDEO", "video", "YOUTUBE", "youtube"
];

export const bb_data_tags = ["CODE", "code", "QUOTE", "quote", "SPOILER", "spoiler", "URL", "url"];

export function RandInt(max_val=9999999) {
	let rnd = Math.floor(Math.random() * max_val);
	while (state.rnd_int == rnd)
		rnd = Math.floor(Math.random() * max_val);
	state.rnd_int = rnd;
	return rnd;
}

export function CountOccurences(str, sub_str, start=0) {
	let count = 0;
	let pos = start;

	while ((pos = str.indexOf(sub_str, pos)) !== -1) {
		count++;
		pos += sub_str.length;
	}

	return count;
}

export function TrimLast(str, chr) {
	if (str.endsWith(chr)) {
		return str.slice(0, -1);
	} else {
		return str;
	}
}

export function TrimStart(str, chr) {
	if (str.startsWith(chr)) {
		return str.substr(1);
	} else {
		return str;
	}
}

export function TrimEnds(str, chr) {
	return TrimLast(TrimStart(str, chr), chr);
}

export function CleanFilename(filename) {
	return TrimEnds(TrimEnds(TrimEnds(filename.replaceAll('\\', '').replaceAll('/', '').replaceAll('?', '').replaceAll('#', '').
		replaceAll('%', '').replaceAll('&', '').replaceAll('{', '').replaceAll('}', '').replaceAll('<', '').replaceAll('>', '').
		replaceAll('$', '').replaceAll('!', '').replaceAll('@', '').replaceAll(':', '').replaceAll('*', '').replaceAll('+', '').
		replaceAll('=', '').replaceAll('|', '').replaceAll('`', '').replaceAll('"', '').replaceAll("'", '').trim(), '.'),'-'),'_');
}

export function StripFileProto(filename) {
	if (state.is_windows) {
		return filename.replace('file:///', '').replace('file://', '');
	} else {
		return filename.replace('file://', '');
	}
}

/** Build a file:// URL that works on Windows (file:///C:/...) and Unix. */
export function PathToFileURL(file_path) {
	const path = String(file_path || '').replace(/\\/g, '/');
	if (state.is_windows) {
		return 'file:///' + path;
	}
	return 'file://' + path;
}

export function EncodeHTML(txt) {
	return txt.replaceAll('&', '&amp;').
		replaceAll('<', '&lt;').
		replaceAll('>', '&gt;').
		replaceAll('(', '&lpar;').
		replaceAll(')', '&rpar;').
		replaceAll('#', '&num;').
		replaceAll('$', '&dollar;').
		replaceAll(':', '&colon;').
		replaceAll('"', '&quot;').
		replaceAll('\'', '&apos;').
		replaceAll('/', '&#47;').
		replaceAll('\\', '&#92;').
		replaceAll('\t', '&Tab;');
}

/** Truncate voice labels for #voices / #tts_voices (native <select> ignores CSS ellipsis). */
export function VoiceOptionLabel(name, maxLen = 28) {
	const s = String(name || '');
	if (s.length <= maxLen) return s;
	return s.slice(0, Math.max(1, maxLen - 1)) + '…';
}

/** Match engine SafeString() so think-block content survives markdown rendering. */
export function SafeAiString(txt) {
	// Placeholders AI_UI_C{N}; - no #/&. Decode to &#N; after EncodeHTML (which mangles #).
	const codes = {
		'#': 35, '$': 36, '+': 43, '-': 45, '~': 126, '.': 46, '*': 42,
		'`': 96, '!': 33, '|': 124, '(': 40, ')': 41, '[': 91, ']': 93,
		'<': 60, '>': 62
	};
	let result = String(txt || '').replaceAll('_', 'AI_UI_C95;');
	for (const ch of Object.keys(codes)) {
		result = result.replaceAll(ch, 'AI_UI_C' + codes[ch] + ';');
	}
	return result;
}

/** Turn SafeString / SafeAiString placeholders into numeric HTML entities. */
export function DecodeAiSafeString(txt) {
	return String(txt || '')
		.replace(/AI_UI_C(\d+);/g, '&#$1;')
		// Legacy AI_UI_AMP* placeholders (named or AI_UI_AMP#N;).
		.replaceAll('AI_UI_AMP', '&');
}

/** Match engine StripEmptyThinkTags - drop empty / whitespace-only think blocks. */
export function StripEmptyThinkTags(txt) {
	let s = String(txt || '');
	let prev = null;
	while (prev !== s) {
		prev = s;
		s = s.replace(/<(think|thoughts)>\s*<\/\1>/gi, '');
		s = s.replace(/AIUI_THINK_START\s*AIUI_THINK_END/g, '');
	}
	return s;
}

const STREAM_XML_PAIRS = [
	['<thoughts>', '</thoughts>'],
	['<think>', '</think>'],
	['<tool_call>', '</tool_call>'],
	['<tool_response>', '</tool_response>']
];

/** Kind for a STREAM_XML_PAIRS open tag. */
function StreamXmlKind(open) {
	if (open === '<think>' || open === '<thoughts>') return 'think';
	if (open === '<tool_call>') return 'tc';
	return 'tr';
}

/**
 * Earliest unclosed think/tool block in txt, or null.
 * { kind, open, close, at } — at is the index of the open tag.
 */
export function FindOpenAiBlock(txt) {
	const s = String(txt || '');
	let best = null;
	for (let p = 0; p < STREAM_XML_PAIRS.length; ++p) {
		const open = STREAM_XML_PAIRS[p][0];
		const close = STREAM_XML_PAIRS[p][1];
		let pos = 0;
		while (true) {
			const i = s.indexOf(open, pos);
			if (i < 0) break;
			const j = s.indexOf(close, i + open.length);
			if (j < 0) {
				if (!best || i < best.at) {
					best = { kind: StreamXmlKind(open), open, close, at: i };
				}
				break;
			}
			pos = j + close.length;
		}
	}
	return best;
}

/** Index of the earliest unclosed think/tool open tag, or -1 if all closed. */
export function FindOpenAiBlockStart(txt) {
	const block = FindOpenAiBlock(txt);
	return block ? block.at : -1;
}

/** End offset after the last fully closed think/tool block in s, or -1. */
function EndOfLastClosedAiBlock(s) {
	let end = -1;
	for (let p = 0; p < STREAM_XML_PAIRS.length; ++p) {
		const open = STREAM_XML_PAIRS[p][0];
		const close = STREAM_XML_PAIRS[p][1];
		let pos = 0;
		while (true) {
			const i = s.indexOf(open, pos);
			if (i < 0) break;
			const j = s.indexOf(close, i + open.length);
			if (j < 0) break;
			end = Math.max(end, j + close.length);
			pos = j + close.length;
		}
	}
	return end;
}

/**
 * Split live stream text into a rich-format sealed prefix and a plain open tail.
 * Sealed: closed think/tool blocks and prose through the last paragraph break (\n\n).
 * Open: incomplete XML block and/or the current paragraph (shown escaped until sealed).
 */
export function SplitSealedOpen(txt) {
	const s = String(txt || '');
	if (!s) return { sealed: '', open: '' };

	const openAt = FindOpenAiBlockStart(s);
	const region = openAt < 0 ? s : s.slice(0, openAt);
	const para = region.lastIndexOf('\n\n');
	const blockEnd = EndOfLastClosedAiBlock(region);
	const paraEnd = para >= 0 ? para + 2 : -1;
	const sealEnd = Math.max(paraEnd, blockEnd);

	if (sealEnd <= 0) return { sealed: '', open: s };
	return { sealed: s.slice(0, sealEnd), open: s.slice(sealEnd) };
}

/** Undo SafeString / SafeAiString placeholders back to plain text. */
export function UndoSafeAiString(txt) {
	return String(txt || '').replace(/AI_UI_C(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
}

/** Pretty-print JSON tool payloads for details bodies (real newlines + indent). */
export function PrettyToolJson(inner) {
	const s = UndoSafeAiString(String(inner || '')).trim();
	if (!s) return '';
	try {
		return JSON.stringify(JSON.parse(s), null, 2);
	} catch (_e) {
		return s;
	}
}

/**
 * Convert raw model XML tags (stored in chat history) into AIUI_* markers the
 * renderer already understands. Live replies get this treatment in the engine;
 * session replay must do it here.
 *
 * Tool bodies stay raw here — pretty-print/HTML happen after markdown so indent
 * is not stripped by md.render / <br> conversion.
 */
export function WrapAiXmlTags(txt) {
	let result = StripEmptyThinkTags(txt);
	const pairs = [
		{ open: '<thoughts>', close: '</thoughts>', kind: 'think' },
		{ open: '<think>', close: '</think>', kind: 'think' },
		{ open: '<tool_call>', close: '</tool_call>', kind: 'tc' },
		{ open: '<tool_response>', close: '</tool_response>', kind: 'tr' }
	];
	for (let p = 0; p < pairs.length; ++p) {
		const { open, close, kind } = pairs[p];
		while (result.includes(open) && result.includes(close)) {
			const start = result.indexOf(open) + open.length;
			const end = result.indexOf(close, start);
			if (end < start) break;
			const inner = result.slice(start, end);
			let repl;
			if (kind === 'think') {
				repl = FormatThinkMarkers(inner);
			}
			else if (kind === 'tc') repl = 'AIUI_TC_START' + inner + 'AIUI_TC_END';
			else repl = 'AIUI_TR_START' + inner + 'AIUI_TR_END';
			result = result.slice(0, start - open.length) + repl + result.slice(end + close.length);
		}
	}
	return StripEmptyThinkTags(result);
}

/**
 * Wrap think body in AIUI markers. Leading single \\n would become <br> under
 * markdown breaks:true and glue the first paragraph to the <hr>; force \\n\\n.
 */
export function FormatThinkMarkers(inner) {
	const body = String(inner || '').replace(/^\n+/, '').replace(/\s+$/, '');
	if (!body) return '';
	return 'AIUI_THINK_START\n\n' + SafeAiString(body) + 'AIUI_THINK_END';
}

const AIUI_MARKER_PAIRS = [
	{ start: 'AIUI_THINK_START', end: 'AIUI_THINK_END', kind: 'think' },
	{ start: 'AIUI_COMPACT_START', end: 'AIUI_COMPACT_END', kind: 'compact' },
	{ start: 'AIUI_TC_START', end: 'AIUI_TC_END', kind: 'tc' },
	{ start: 'AIUI_TR_START', end: 'AIUI_TR_END', kind: 'tr' }
];

/** Pull AIUI marker blocks out so markdown/BBCode cannot mangle their bodies. */
export function ExtractAiuiBlocks(txt) {
	let text = String(txt || '');
	const blocks = [];
	for (let p = 0; p < AIUI_MARKER_PAIRS.length; ++p) {
		const { start, end, kind } = AIUI_MARKER_PAIRS[p];
		while (text.includes(start) && text.includes(end)) {
			const i = text.indexOf(start);
			const j = text.indexOf(end, i + start.length);
			if (j < 0) break;
			const inner = text.slice(i + start.length, j);
			const id = blocks.length;
			blocks.push({ kind, inner });
			// No underscores — markdown would treat _BLOCK_ as emphasis.
			text = text.slice(0, i) + '\n\nAIUIBLOCK' + id + 'X\n\n' + text.slice(j + end.length);
		}
	}
	return { text, blocks };
}

export function BuildDetailsHtml(kind, inner) {
	let summary = 'click to show';
	let detailsClass = 'ai_act';
	let bodyClass = 'details_body';
	let body = '';
	if (kind === 'think') {
		summary = 'click to show reasoning';
		detailsClass += ' ai_think';
		bodyClass += ' details_think';
		body = EncodeHTML(UndoSafeAiString(inner).replace(/^\n+/, '').replace(/\s+$/, ''));
	} else if (kind === 'compact') {
		summary = 'click to show context summary';
		detailsClass = 'ai_compact';
		bodyClass += ' details_compact';
		body = EncodeHTML(UndoSafeAiString(inner).replace(/^\n+/, '').replace(/\s+$/, ''));
	} else if (kind === 'tc') {
		summary = 'click to show tool call';
		detailsClass += ' ai_tool_call';
		bodyClass += ' details_tool';
		body = EncodeHTML(PrettyToolJson(inner));
	} else {
		summary = 'click to show tool response';
		detailsClass += ' ai_tool_response';
		bodyClass += ' details_tool';
		body = EncodeHTML(PrettyToolJson(inner));
	}
	return (
		'<details class="' + detailsClass + '">' +
		'<summary onclick="ToggleDetails(this)">' + summary + '</summary><hr>' +
		'<div class="' + bodyClass + '">' + body + '</div></details>'
	);
}

export function RestoreAiuiBlocks(html, blocks) {
	let out = String(html || '');
	for (let i = 0; i < blocks.length; ++i) {
		const ph = 'AIUIBLOCK' + i + 'X';
		const blockHtml = BuildDetailsHtml(blocks[i].kind, blocks[i].inner);
		out = out.replace(new RegExp('<p>\\s*' + ph + '\\s*</p>', 'g'), blockHtml);
		out = out.replaceAll(ph, blockHtml);
	}
	return out;
}

/** Legacy marker→details rewrite (used when blocks were not extracted first). */
export function ConvertAITags(txt) {
	const { text, blocks } = ExtractAiuiBlocks(txt);
	if (!blocks.length) return text;
	return RestoreAiuiBlocks(text, blocks);
}

export function ConvertBB(txt) {
	return ('<p>' + txt.replaceAll('[HR]', '</p><hr><p>').
		replaceAll('[BBC_START_B]', '<b>').replaceAll('[BBC_END_B]', '</b>').
		replaceAll('[BBC_START_I]', '<i>').replaceAll('[BBC_END_I]', '</i>').
		replaceAll('[BBC_START_U]', '<u>').replaceAll('[BBC_END_U]', '</u>').
		replaceAll('[BBC_START_S]', '<s>').replaceAll('[BBC_END_S]', '</s>').
		replaceAll('[BBC_START_H1]', '</p><h1>').replaceAll('[BBC_END_H1]', '</h1><p>').
		replaceAll('[BBC_START_H2]', '</p><h2>').replaceAll('[BBC_END_H2]', '</h2><p>').
		replaceAll('[BBC_START_H3]', '</p><h3>').replaceAll('[BBC_END_H3]', '</h3><p>').
		replaceAll('[BBC_START_H4]', '</p><h4>').replaceAll('[BBC_END_H4]', '</h4><p>').
		replaceAll('[BBC_START_H5]', '</p><h5>').replaceAll('[BBC_END_H5]', '</h5><p>').
		replaceAll('[BBC_START_H6]', '</p><h6>').replaceAll('[BBC_END_H6]', '</h6><p>').
		replaceAll('[BBC_START_OL]', '</p><ol>').replaceAll('[BBC_END_OL]', '</ol><p>').
		replaceAll('[BBC_START_UL]', '</p><ul>').replaceAll('[BBC_END_UL]', '</ul><p>').
		replaceAll('[BBC_START_LI]', '<li><p>').replaceAll('[BBC_END_LI]', '</p></li>').
		replaceAll('[BBC_START_PRE]', '</p><pre>').replaceAll('[BBC_END_PRE]', '</pre><p>').
		replaceAll('[BBC_START_QUOTE]', '</p><blockquote><p>').replaceAll('[BBC_END_QUOTE]', '</p></blockquote><p>').
		replaceAll('[BBC_START_SPOILER]', '</p><details><summary onclick="ToggleDetails(this)">click to show spoiler</summary><hr><p>').
		replaceAll('[BBC_START_SPOILER=', '</p><details><summary onclick="ToggleDetails(this)">').replaceAll('BBC_EOT_SPOILER]', '</summary><hr><p>').
		replaceAll('[BBC_END_SPOILER]', '</p></details><p>').
		replaceAll('[BBC_START_CENTER]', '</p><div class="center"><p>').replaceAll('[BBC_END_CENTER]', '</p></div><p>').
		replaceAll('[BBC_START_IMG]', '</p><img src="').replaceAll('[BBC_END_IMG]', '" onclick="ShowImage(this)"><p>').
		replaceAll('[BBC_START_CODE]', '</p><pre class="code_box"><code>').replaceAll('[BBC_END_CODE]', '</code></pre><p>').
		replaceAll('[BBC_START_CODE=', '</p><pre class="code_box"><code class="language-').replaceAll('BBC_EOT_CODE]', '">').
		replaceAll('[BBC_START_URL=', '<a href="').replaceAll('BBC_EOT_URL]', '" target="_blank">').replaceAll('[BBC_END_URL]', '</a>').
		replaceAll('[BBC_START_VIDEO=', '</p><video class="chat_vid" controls><source type="video/').replaceAll('BBC_EOT_VIDEO]', '" src="').
		replaceAll('[BBC_START_AUDIO=', '</p><audio class="chat_aud" controls><source type="audio/').replaceAll('BBC_EOT_AUDIO]', '" src="').
		replaceAll('[BBC_END_VIDEO]', '"></video><p>').replaceAll('[BBC_END_AUDIO]', '"></audio><p>').
		replaceAll('[BBC_START_YOUTUBE]', '</p><iframe class="yt_vid" src="https://www.youtube.com/embed/').replaceAll('[BBC_END_YOUTUBE]', '"></iframe><p>').
		replaceAll('[BBC_LEFT_BRACKET_', '[').replaceAll('_BBC_RIGHT_BRACKET]', ']') + '</p>').
		replaceAll("\n\n", "</p><p>").replaceAll("\n", "<br>").replaceAll("[AI_UI_TAB]", "\t").replaceAll("[AI_UI_BR]", "\n");
}

export function EncodeBB(bbcode) {
	let result = bbcode.replaceAll("[hr]", "[HR]").replaceAll("[CODE]", "[CODE=auto]").replaceAll("[code]", "[code=auto]");

	for (let i=0; i < bb_data_tags.length; ++i)
	{
		const bbc_tag = bb_data_tags[i];
		const bbc_open_tag = "[" + bbc_tag + "]";
		const bbc_data_tag = "[" + bbc_tag + "=";
		const bbc_close_tag = "[/" + bbc_tag + "]";
		const bbc_upper_tag = bbc_tag.toUpperCase();
		const bbc_encoded_tag = "[BBC_START_"+bbc_upper_tag;
		const bbc_encoded_end = "[BBC_END_"+bbc_upper_tag+"]";
		const bbc_encoded_eot = "BBC_EOT_"+bbc_upper_tag+"]";
		let data_start = 0;

		while (result.includes(bbc_data_tag) && result.includes(bbc_close_tag)) {
			data_start = result.indexOf(bbc_data_tag, data_start) + bbc_data_tag.length;
			const tag_start = data_start + result.substring(data_start).indexOf(']');
			const tag_data = result.substring(data_start, tag_start);
			let tag_end = result.indexOf(bbc_close_tag, tag_start);
			if (tag_start+1 >= tag_end) break;
			let inner_txt = result.substring(tag_start+1, tag_end);
			let tags_skipped = 0;
			while (tags_skipped < CountOccurences(inner_txt, bbc_open_tag) + CountOccurences(inner_txt, bbc_data_tag)) {
				let last_end = tag_end;
				tag_end = result.indexOf(bbc_close_tag, tag_end + bbc_close_tag.length);
				if (tag_start >= tag_end) {
					inner_txt = result.substring(tag_start+1, last_end);
					break;
				} else {
					tags_skipped++;
					inner_txt = result.substring(tag_start+1, tag_end);
				}
			}
			let inner_enc = inner_txt;
			if (bbc_upper_tag == "CODE") {
				inner_enc = TrimStart(inner_enc, "\n").replaceAll('[CODE=auto]', '[CODE]').replaceAll('[code=auto]', '[code]').
							replaceAll('[', '[BBC_LEFT_BRACKET_').replaceAll(']', '_BBC_RIGHT_BRACKET]');
			}
			inner_enc = inner_enc.replaceAll("\t", "[AI_UI_TAB]").replaceAll("\n", "[AI_UI_BR]");
			if (bbc_upper_tag == "QUOTE") {
				result = result.replace(bbc_data_tag+tag_data+"]"+inner_txt+bbc_close_tag, bbc_encoded_tag+"]"+inner_enc+"\n\n - "+tag_data+bbc_encoded_end);
			} else if (bbc_upper_tag == "CODE" && tag_data == "auto") {
				result = result.replace(bbc_data_tag+tag_data+"]"+inner_txt+bbc_close_tag, bbc_encoded_tag+"]"+inner_enc+bbc_encoded_end);
			} else {
				result = result.replace(bbc_data_tag+tag_data+"]"+inner_txt+bbc_close_tag, bbc_encoded_tag+"="+tag_data+bbc_encoded_eot+inner_enc+bbc_encoded_end);
			}
		}
	}

	for (let i=0; i < bb_code_tags.length; ++i)
	{
		const bbc_tag = bb_code_tags[i];
		const bbc_open_tag = "[" + bbc_tag + "]";
		const bbc_data_tag = "[" + bbc_tag + "=";
		const bbc_close_tag = "[/" + bbc_tag + "]";
		const bbc_upper_tag = bbc_tag.toUpperCase();
		const bbc_encoded_tag = "[BBC_START_"+bbc_upper_tag;
		const bbc_encoded_end = "[BBC_END_"+bbc_upper_tag+"]";
		const bbc_encoded_eot = "BBC_EOT_"+bbc_upper_tag+"]";
		let tag_start = 0;

		while (result.includes(bbc_open_tag) && result.includes(bbc_close_tag)) {
			tag_start = result.indexOf(bbc_open_tag, tag_start) + bbc_open_tag.length;
			let tag_end = result.indexOf(bbc_close_tag, tag_start);
			if (tag_start >= tag_end) break;
			let inner_txt = result.substring(tag_start, tag_end);
			let tags_skipped = 0;
			while (tags_skipped < CountOccurences(inner_txt, bbc_open_tag) + CountOccurences(inner_txt, bbc_data_tag)) {
				let last_end = tag_end;
				tag_end = result.indexOf(bbc_close_tag, tag_end + bbc_close_tag.length);
				if (tag_start >= tag_end) {
					inner_txt = result.substring(tag_start, last_end);
					break;
				} else {
					tags_skipped++;
					inner_txt = result.substring(tag_start, tag_end);
				}
			}
			let inner_enc = inner_txt;
			if (bbc_upper_tag == "PRE") inner_enc = TrimStart(inner_enc, "\n");
			inner_enc = inner_enc.replaceAll("\t", "[AI_UI_TAB]").replaceAll("\n", "[AI_UI_BR]");
			if (bbc_upper_tag == "OL" || bbc_upper_tag == "UL") inner_enc = inner_enc.replaceAll("[*]", "[BBC_START_LI]");
			if (bbc_upper_tag == "URL") {
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, bbc_encoded_tag+"="+inner_txt+bbc_encoded_eot+inner_enc+bbc_encoded_end);
			} else if (bbc_upper_tag == "VIDEO") {
				const vid_link = inner_txt.toLowerCase().trim();
				let tag_data = "mp4";
				if (vid_link.endsWith("webm")) {
					tag_data = "webm";
				} else if (vid_link.endsWith("ogg")) {
					tag_data = "ogg";
				}
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, bbc_encoded_tag+"="+tag_data+bbc_encoded_eot+inner_enc+bbc_encoded_end);
			} else if (bbc_upper_tag == "AUDIO") {
				const aud_link = inner_txt.toLowerCase().trim();
				let tag_data = "mp3";
				if (aud_link.endsWith("wav")) {
					tag_data = "wav";
				} else if (aud_link.endsWith("ogg")) {
					tag_data = "ogg";
				}
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, bbc_encoded_tag+"="+tag_data+bbc_encoded_eot+inner_enc+bbc_encoded_end);
			} else {
				result = result.replace(bbc_open_tag+inner_txt+bbc_close_tag, bbc_encoded_tag+"]"+inner_enc+bbc_encoded_end);
			}
		}
	}

	return result.replaceAll("[CODE=auto]", "[CODE]").replaceAll("[code=auto]", "[code]");
}

export function ThinkAnim() {
	const targ = (state.thinking) ? '#thinking' : state.think_targ;
	switch (state.dot_count) {
	case 0:
		$(targ).html('');
		state.dot_count++;
		break;
	case 1:
		$(targ).html('.');
		state.dot_count++;
		break;
	case 2:
		$(targ).html('..');
		state.dot_count++;
		break;
	case 3:
		$(targ).html('...');
		state.dot_count++;
		break;
	case 4:
		$(targ).html('..');
		state.dot_count++;
		break;
	case 5:
		$(targ).html('.');
		state.dot_count = 0;
		break;
	}
}

export function StartGenerating(targ) {
	state.generating = true;
	state.think_targ = targ;
	$(state.think_targ).html('');
	$(state.think_targ).show();
	state.dot_count = 1;
	state.loop_timer = setInterval(ThinkAnim, 1000);
	if (typeof window.RenderToolPacks === 'function') {
		try { window.RenderToolPacks(); } catch (e) { /* ignore */ }
	}
}

export function StopGenerating(targ='.generating') {
	state.generating = false;
	$(targ).hide();
	clearInterval(state.loop_timer);
	if (typeof window.RenderToolPacks === 'function') {
		try { window.RenderToolPacks(); } catch (e) { /* ignore */ }
	}
}

export function DisExtraButtons(bool_val) {
	if (bool_val) {
		$('#redo_btn').prop('disabled', true);
		$('#cont_btn').prop('disabled', true);
		return;
	}
	const hasMsg = $('#chat_log > .msg_box').filter(function () {
		const $b = $(this);
		// Ignore empty hidden streaming stubs (turn started, no reply yet).
		if ($b.hasClass('hidden') && !$.trim($b.text())) return false;
		return true;
	}).length > 0;
	$('#redo_btn').prop('disabled', !hasMsg);
	$('#cont_btn').prop('disabled', !hasMsg);
}

export function DisableButtons(bool_val, dis_send_btn=true) {
	if (!bool_val) {
		$('#send_btn').html('SEND');
		$('#gen_btn').html('GENERATE');
		$('#send_btn').prop('disabled', false);
		$('#gen_btn').prop('disabled', false);
	} else {
		let btn_id = '#send_btn';
		if ($('#chat_app').is(":hidden")) {
			$(btn_id).prop('disabled', true);
			btn_id = '#gen_btn';
		} else {
			$('#gen_btn').prop('disabled', true);
		}
		if (dis_send_btn) {
			$(btn_id).prop('disabled', true);
		} else {
			$(btn_id).html('STOP');
		}
	}
	DisExtraButtons(bool_val);
	$('#apply_btn').prop('disabled', bool_val);
	$('#tts_btn').prop('disabled', bool_val);
	$('#read_txt_btn').prop('disabled', bool_val);
	$('#clone_voice_btn').prop('disabled', bool_val);
	$('#img_btn').prop('disabled', bool_val);
	$('#edit_btn').prop('disabled', bool_val);
	$('#mic_btn').prop('disabled', bool_val);
	$('#attach_btn').prop('disabled', bool_val);
	$('#attach_box_btn').prop('disabled', bool_val);
	$('#in_accept_btn').prop('disabled', bool_val);
	$('.config_btn').prop('disabled', bool_val);
	$('#ctx_compact_btn').prop('disabled', bool_val);
	$('#agent_select').prop('disabled', bool_val || !state.engine_ready);
	// Lock/unlock pack enable toggles (+ SAVE if dialog open) with engine busy state.
	if (typeof window.RenderToolPacks === 'function') {
		try { window.RenderToolPacks(); } catch (e) { /* ignore */ }
	}
}

export function ShowInFolder(elem) {
	let file_path = '';

	if (elem === null) {
		file_path = state.script_dir+'/ai_images/';
	} else if (typeof elem == "string") {
		if (elem == 'TTS_EMBEDS') {
			file_path = state.script_dir+'/embeddings/';
		} else if (elem == 'AI_AUDIO') {
			file_path = state.script_dir+'/ai_audio/';
		} else {
			file_path = StripFileProto(elem);
			file_path = file_path.substr(0, file_path.indexOf('?'));
		}
	} else {
		file_path = StripFileProto($(elem).prop('src'));
	}

	ipcRenderer.send('show-in-dir', file_path);
}

export function ShowImage(elem) {
	let file_path = StripFileProto($(elem).prop('src'));

	if (file_path != '')
		ipcRenderer.send('show-img', file_path);
}

export function UpdateHeldKeys(input_id, got_combo=false) {
	let keyShortcut = '';
	if (state.held_keys[0] !== null) {
		keyShortcut += state.held_keys[0]+'+';
	}
	if (state.held_keys[1] !== null) {
		keyShortcut += state.held_keys[1]+'+';
	}
	if (state.held_keys[2] !== null) {
		keyShortcut += state.held_keys[2];
	}
	if (keyShortcut.endsWith('+'))
		keyShortcut = keyShortcut.substr(0, keyShortcut.length-1);

	let inpElem = $(input_id);
	inpElem.val(keyShortcut);

	if (got_combo && inpElem.hasClass("keys_input")) {
		state.held_keys = [null,null,null];
		$(':focus').blur();
	}
}

export function ShortcutKeyDown(input_id, key_str, key_code) {
	let keyName = '';
	if (typeof key_str == "string" && key_str.length == 1 && state.alpha_num_regex.test(key_str)) {
		state.held_keys[2] = key_str;
		if (state.held_keys[0] !== null || state.held_keys[1] !== null) {
			UpdateHeldKeys(input_id, true);
			return true;
		}
	} else if (key_code == 16 || key_code == 17 || key_code == 18) {
		keyName = key_str.replace("Control", "Ctrl");
	}
	if (keyName != '') {
		if (state.held_keys[0] === null) {
			state.held_keys[0] = keyName;
		} else if (state.held_keys[0] != keyName) {
			state.held_keys[1] = keyName;
		};
	}
	UpdateHeldKeys(input_id);
	return false;
}

export function ShortcutKeyUp(input_id, key_str, key_code) {
	key_str = key_str.replace("Control", "Ctrl");
	if (typeof key_str == "string" && key_str.length == 1 && state.held_keys[2] === key_str) {
		state.held_keys[2] = null;
	} else if (state.held_keys[0] === key_str) {
		state.held_keys[0] = null;
	} else if (state.held_keys[1] === key_str) {
		state.held_keys[1] = null;
	}
	UpdateHeldKeys(input_id);
}

/** True while initializing or a chat/gen task is in progress. */
export function EngineBusy() {
	if (!state.engine_ready) return true;
	if (state.thinking || state.generating) return true;
	if ($('#send_btn').html() === 'STOP') return true;
	// DisableButtons(true) locks apply before thinking/generating flags flip.
	if ($('#apply_btn').prop('disabled')) return true;
	return false;
}

export function SyncSessionMenuLock() {
	ipcRenderer.send('set-session-menu-lock', { locked: EngineBusy() });
}

export function SetEngineReady(ready) {
	const next = !!ready;
	if (state.engine_ready === next) {
		SyncSessionMenuLock();
		$('#agent_select').prop('disabled', EngineBusy());
		return;
	}
	state.engine_ready = next;
	$('#agent_select').prop('disabled', EngineBusy());
	if (typeof window.RenderToolPacks === 'function') {
		try { window.RenderToolPacks(); } catch (e) { /* ignore */ }
	} else {
		SyncSessionMenuLock();
	}
}

export function GuardEngineIdle(msg) {
	if (!EngineBusy()) return true;
	ipcRenderer.send('show-alert', {
		msg: msg || 'Wait for the current task to finish before changing MCP settings.'
	});
	return false;
}

export function NormMmProcessor(raw) {
	const mp = String(raw || 'auto').trim().toLowerCase();
	return (mp === 'qwen_vl' || mp === 'janus') ? mp : 'auto';
}

/** Click-outside on overlay scrim (target === element) calls onClose. */
export function WireOverlayDismiss(sel, onClose) {
	$(sel).on('click', function(e) {
		if (e.target === this && !$(this).hasClass('hidden')) onClose();
	});
}
