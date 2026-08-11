import { state } from './js/state.js';
import { webUtils } from './js/ipc.js';
import {
	ShowInFolder, ShowImage, ShortcutKeyDown, ShortcutKeyUp
} from './js/util.js';
import {
	SendMsg, RedoLast, ContChat, ToggleMsgBox, ToggleDetails,
	ToggleRecordBox, ToggleAttachBox, AttachFiles, AttachLink, ClearAttachments,
	AttachFile, FileDragDrop, FileDragOver, RemoveFile,
	StopRecording, StopPlayback, TogglePlayback, InitChat, ToggleCtxUsageTip, CompactNow, ScrollChatBottom
} from './js/chat.js';
import {
	GenText, ToggleGenBox, CopyGenTxt, ReadGenTxt, InitTextgen
} from './js/textgen.js';
import {
	GenSpeech, ToggleTTSBox, CloneVoice, OpenWavDir, OpenTTSDir, OpenAIADir, SaveVoice, InitSpeech
} from './js/speech.js';
import { GenImage, CopyGenImg, OpenImgDir, InitImage } from './js/image.js';
import { RunCmd, CmdHistory, InitConsole } from './js/console.js';
import {
	ChatPage, TextPage, ImagePage, SpeechPage, ConfigPage, ConsolePage, HelpPage, AboutPage,
	ApplySettings, ApplyConfig, ToggleConfig, ShowMainConfig, ShowOtherConfig, ShowRemoteConfig,
	ShowInstallHelp, ShowSetupHelp, ToggleInputDialog, ToggleEditPrompt, TogglePromptFiles, ApplyNewPrompt,
	ReloadPromptFromFiles, OnAgentChanged,
	ChangeTalkMode, ChangeAvatar, OpenSample, OpenFile, OpenFolder, OpenFileFolder,
	SetAppVersion, RestartScript, StartScript, QuitApp,
	InitUi, WireUiEvents, RenderToolPacks, ToggleWorkspaceStatusTip
} from './js/ui.js';
import { InitSessionPicker } from './js/session_picker.js';

const hljs = window.hljs;
const markdownit = window.markdownit;
const footNotes = window.markdownitFootnote;
const taskLists = window.markdownitTaskLists;
const markdownitEmoji = window.markdownitEmoji;
const mk = window.markdownitKatex;

Object.assign(window, {
	ChatPage, TextPage, ImagePage, SpeechPage, ConfigPage, ConsolePage, HelpPage, AboutPage,
	SendMsg, RedoLast, ContChat, ToggleMsgBox, ToggleDetails,
	ToggleRecordBox, ToggleAttachBox, AttachFiles, AttachLink, ClearAttachments,
	RemoveFile, FileDragDrop, FileDragOver, ToggleCtxUsageTip, CompactNow, ScrollChatBottom,
	ToggleInputDialog, ToggleEditPrompt, TogglePromptFiles, ApplyNewPrompt, ReloadPromptFromFiles, OnAgentChanged,
	ChangeTalkMode, ChangeAvatar, StopPlayback, TogglePlayback,
	GenText, ToggleGenBox, CopyGenTxt, ReadGenTxt,
	GenSpeech, ToggleTTSBox, CloneVoice, OpenWavDir, OpenTTSDir, OpenAIADir, SaveVoice,
	GenImage, CopyGenImg, OpenImgDir,
	RunCmd,
	ApplySettings, ApplyConfig, ToggleConfig,
	ShowMainConfig, ShowOtherConfig, ShowRemoteConfig, ShowInstallHelp, ShowSetupHelp,
	OpenSample, OpenFile, OpenFolder, OpenFileFolder,
	RestartScript, QuitApp,
	ShowInFolder, ShowImage,
	RenderToolPacks, ToggleWorkspaceStatusTip
});

// IPC listeners must register before StartScript (in ready). Bundles load at end of
// <body>, so the DOM already exists here; Init* DOM wiring still uses WhenDomReady.
InitChat();
InitTextgen();
InitSpeech();
InitImage();
InitConsole();
InitUi();
InitSessionPicker();

$(document).ready(function() {
	SetAppVersion();
	StartScript();

	state.md = markdownit({
		html:         false,
		xhtmlOut:     false,
		linkify:      false,
		typographer:  false,
		breaks:       true,

		highlight: function (str, lang) {
			if (lang && hljs.getLanguage(lang)) {
				try {
					// language-* + data-highlighted so FinalizeBotBox / send-path
					// highlightElement does not re-run highlightAuto
					return '<pre class="code_box"><code class="hljs language-' + lang +
						'" data-highlighted="yes">' +
						hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
						'</code></pre>';
				} catch (__) {}
			}

			return '<pre class="code_box"><code class="hljs">' + state.md.utils.escapeHtml(str) + '</code></pre>';
		}
	}).use(mk, { throwOnError: false, errorColor: " #CC0000" }).
	use(markdownitEmoji, { shortcuts: {}}).use(taskLists).use(footNotes);

	// Tooltip from alt when Markdown omits an explicit title: ![alt](url)
	const defaultImageRender = state.md.renderer.rules.image
		|| ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
	state.md.renderer.rules.image = (tokens, idx, options, env, self) => {
		const token = tokens[idx];
		const alt = self.renderInlineAsText(token.children || [], options, env);
		if (alt) {
			const titleIdx = token.attrIndex('title');
			if (titleIdx < 0) token.attrPush(['title', alt]);
			else if (!token.attrs[titleIdx][1]) token.attrs[titleIdx][1] = alt;
		}
		return defaultImageRender(tokens, idx, options, env, self);
	};

	$("#input_btn").on('click', function () {
		const fpath = $("#user_input").val();
		AttachFile(fpath, true);
		$("#input_box").hide();
	});

	$("#user_input").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			$("#input_btn").click();
		}
	});

	$("#attach_files").on('change', function(e) {
		for (let i=0; i < e.target.files.length; ++i) {
			AttachFile(webUtils.getPathForFile(e.target.files[i]));
		}
		$(this).val('');
	});

	$("#text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			SendMsg();
		}
	});

	$("#gen_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenText();
		}
	});

	$("#tts_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenSpeech();
		}
	});

	$("#img_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenImage();
		}
	});

	$("#img_text_neg").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			GenImage();
		}
	});

	$("#cmd_text_inp").on('keyup', function (e) {
		if (e.key === 'Enter' || e.keyCode === 13) {
			RunCmd();
		} else if (e.key === 'Arrow Up' || e.keyCode === 38) {
			e.preventDefault();
			$(this).val(CmdHistory(-1, $(this).val()));
		} else if (e.key === 'Arrow Down' || e.keyCode === 40) {
			e.preventDefault();
			$(this).val(CmdHistory(1, $(this).val()));
		}
	});

	$('#area_inp').on('keydown', function(e) {
		if (e.key === 'Tab' || e.keyCode == 9) {
			e.preventDefault();
			let s = this.selectionStart;
			$(this).val(function(i, v) {
				return v.substring(0, s) + "\t" + v.substring(this.selectionEnd)
			});
			this.selectionEnd = s + 1;
		}
	});

	WireUiEvents();
});
