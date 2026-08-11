import { ipcRenderer } from './ipc.js';
import { state } from './state.js';

/** Cap console DOM nodes so a long session does not lag the UI. */
const MAX_CONSOLE_LOGS = 500;
let consoleLogCount = 1; // initial "Application started" <pre>

export function CmdHistory(shift, val) {
	if (state.cmd_list.length == 0) return val;
	state.cmd_index += shift;
	if (state.cmd_index >= state.cmd_list.length)
		state.cmd_index = state.cmd_list.length - 1;
	if (state.cmd_index < 0) state.cmd_index = 0;
	return state.cmd_list[state.cmd_index];
}

export function RunCmd() {
	let cmd_txt = $('#cmd_text_inp').val().trim();
	if (cmd_txt != '') {
		$('#cmd_btn').prop('disabled', true);
		$('#cmd_text_inp').prop('disabled', true);
		ipcRenderer.send('run-cmd', { cmd: cmd_txt });
	}
}

function AppendConsoleLog(msg) {
	const logBox = document.getElementById('log_box');
	if (!logBox) return;
	const pre = document.createElement('pre');
	pre.className = 'console_log';
	pre.textContent = msg;
	logBox.appendChild(pre);
	consoleLogCount++;
	while (consoleLogCount > MAX_CONSOLE_LOGS && logBox.firstChild) {
		logBox.removeChild(logBox.firstChild);
		consoleLogCount--;
	}
	$('#console_box').scrollTop($('#console_box').prop("scrollHeight"));
}

export function InitConsole() {
	ipcRenderer.on('cmd-sent', (event, payload) => {
		if (payload) {
			state.cmd_index = state.cmd_list.length + 1;
			state.cmd_list.push($('#cmd_text_inp').val().trim());
			$('#cmd_text_inp').val('');
		}
		$('#cmd_btn').prop('disabled', false);
		$('#cmd_text_inp').prop('disabled', false);
		$('#cmd_text_inp').focus();
	});

	ipcRenderer.on('append-log', (event, payload) => {
		AppendConsoleLog(payload.msg);
	});
}
