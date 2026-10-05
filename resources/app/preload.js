const { contextBridge, ipcRenderer, webUtils } = require('electron');

const validSend = [
	'show-alert', 'save-recording', 'send-msg', 'attach-files',
	'copy-text', 'copy-text-silent', 'read-text', 'gen-text', 'save-voice', 'clone-voice', 'gen-speech',
	'copy-image', 'gen-image', 'run-cmd', 'check-models', 'show-menubar',
	'config-app', 'config-app-other', 'config-app-remote', 'config-other', 'config-ai', 'config-tools',
	'config-prompt-files',
	'update-users', 'config-voice', 'update-prompt', 'update-talk-mode', 'update-avatar',
	'show-in-dir', 'show-img', 'set-session-menu-lock'
];

const validInvoke = [
	'stop-stream', 'restart-script', 'start-script', 'close-app',
	'dialog:openAvatar', 'dialog:openSample', 'dialog:openFile',
	'dialog:openFolder', 'dialog:openFileFolder', 'getAppVersion',
	'prepare-avatar',
	'session-autosave', 'session-new', 'session-load', 'session-open-project',
	'session-close-project', 'session-get-context',
	'prompt-files-stamp',
	'mcp-open-editor', 'mcp-refresh-packs', 'confirm-action',
	'tool-permission-result'
];

const validReceive = [
	'txtgen-stream-start', 'txtgen-stream-text', 'txtgen-stream-end', 'stream-start', 'stream-text', 'stream-end',
	'turn-start', 'turn-patch', 'turn-end', 'bot-msg', 'context-compact', 'context-usage',
	'gen-result', 'clone-result', 'tts-result', 'asr-result', 'img-result', 'cmd-sent',
	'append-log', 'init-ui', 'ai-ready', 'set-platform', 'start-recording', 'stop-recording',
	'add-voices', 'clear-voices', 'play-audio', 'got-avatar', 'got-tools', 'got-tool-catalog', 'got-file',
	'got-models', 'prompt-msg', 'load-config', 'prompt-files',
	'session-context', 'session-loaded', 'session-cleared', 'session-picker', 'tool-permission',
	'set-side-panel'
];

contextBridge.exposeInMainWorld('aiuiAPI', {
	openAvatar: () => ipcRenderer.invoke('dialog:openAvatar'),
	openSample: () => ipcRenderer.invoke('dialog:openSample'),
	openFile: () => ipcRenderer.invoke('dialog:openFile'),
	openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
	openFileFolder: () => ipcRenderer.invoke('dialog:openFileFolder'),
	getAppVersion: () => ipcRenderer.invoke('getAppVersion'),
	prepareAvatar: (payload) => ipcRenderer.invoke('prepare-avatar', payload),
	openMcpEditor: () => ipcRenderer.invoke('mcp-open-editor'),
	refreshMcpPacks: () => ipcRenderer.invoke('mcp-refresh-packs'),
	confirmAction: (payload) => ipcRenderer.invoke('confirm-action', payload),

	getPathForFile: (file) => webUtils.getPathForFile(file),

	send: (channel, ...args) => {
		if (validSend.includes(channel)) ipcRenderer.send(channel, ...args);
	},

	invoke: (channel, ...args) => {
		if (validInvoke.includes(channel)) return ipcRenderer.invoke(channel, ...args);
		return Promise.reject(new Error('Invalid invoke channel: ' + channel));
	},

	on: (channel, listener) => {
		if (!validReceive.includes(channel)) return;
		// Drop the IpcRendererEvent - it cannot cross the context bridge.
		const wrapped = (_event, ...args) => listener(null, ...args);
		ipcRenderer.on(channel, wrapped);
	},

	versions: {
		chrome: process.versions.chrome,
		node: process.versions.node,
		electron: process.versions.electron
	},

	platform: process.platform
});
