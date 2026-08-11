import { ipcRenderer, webUtils } from './ipc.js';
import { state } from './state.js';
import { EncodeHTML, NormMmProcessor } from './util.js';

const link_svg_html = '<svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>';

const file_svg_html = '<svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"></path><path d="M14 2v4a2 2 0 0 0 2 2h4"></path><path d="M8 13h8"></path><path d="M8 17h8"></path><path d="M8 9h2"></path></svg>';

export function AttachmentChipHtml(path) {
	return '<div class="msg_file"><small>&#x1F4CE; ' + EncodeHTML(path) + '</small></div>';
}

export function FormatAttachmentsBlock(paths) {
	if (!paths || !paths.length) return '';
	let html = '<h5 class="files_head">ATTACHMENTS:</h5>';
	for (let i = 0; i < paths.length; ++i) html += AttachmentChipHtml(paths[i]);
	return html;
}

/** Local Vision family only; remote uses OpenAI-style parts (disable dropdown). */
export function SyncVisionSelectForBackend(textBackend) {
	const remote = (textBackend != null ? textBackend : state.text_backend) === 'remote';
	$('#am_select').prop('disabled', remote);
	if (remote) {
		$('#am_select').attr('title', 'Vision family applies to local models only. Remote uses OpenAI-style image/video parts.');
	} else {
		$('#am_select').removeAttr('title');
	}
}

function SyncAttachFilesToEngine() {
	state.mm_processor = NormMmProcessor($('#am_select').find(':selected').val() || state.mm_processor);
	ipcRenderer.send('attach-files', {
		files: state.attachments.slice(),
		mm_processor: state.mm_processor
	});
}

export function CloseAttachDialog() {
	const $d = $('#attach_dialog');
	if (!$d.length || $d.hasClass('hidden')) return;
	$d.addClass('hidden');
	$('#input_box').hide();
	SyncAttachFilesToEngine();
}

export function ToggleAttachBox() {
	const $d = $('#attach_dialog');
	if ($d.hasClass('hidden')) {
		SyncVisionSelectForBackend();
		$('#am_select').val(state.mm_processor || 'auto');
		$d.removeClass('hidden');
	} else {
		CloseAttachDialog();
	}
}

export function RemoveFile(elem) {
	const attachment = $(elem).parent();
	const fpath = decodeURI(attachment.attr('data-file'));
	attachment.remove();
	for (let i = 0; i < state.attachments.length; ++i) {
		if (state.attachments[i] == fpath) {
			state.attachments.splice(i, 1);
			break;
		}
	}
}

export function AttachFile(file_path, is_link=false) {
	if (file_path === null || file_path == '') return;
	if (state.attachments.includes(file_path)) return;
	if (state.attachments.length == 0) {
		$('#attach_box').html('');
	}
	state.attachments.push(file_path);
	const attach_icon = (is_link ? link_svg_html : file_svg_html) + '&nbsp;';
	$('#attach_box').append(
		'<div data-file="' + encodeURI(file_path) + '">' + attach_icon + EncodeHTML(file_path) +
		' <span onclick="RemoveFile(this)">&times;</span></div>'
	);
}

export function AttachFiles() {
	$('#attach_files').click();
}

export function AttachLink() {
	if ($('#input_box').is(':hidden')) $('#input_box').show();
	else $('#input_box').hide();
}

export function ClearAttachments() {
	state.attachments = [];
	$('#attach_box').html('<small id="attach_ph">drag and drop files here</small>');
}

export function FileDragDrop(e) {
	e.preventDefault();
	const files = e.dataTransfer.files;
	for (let i = 0; i < files.length; ++i) {
		AttachFile(webUtils.getPathForFile(files[i]));
	}
}

export function FileDragOver(e) {
	e.preventDefault();
}
