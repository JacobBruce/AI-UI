const { app, BrowserWindow, dialog, clipboard, ipcMain, shell } = require('electron');
const fs = require('fs');
const path = require('path');
const core = require('./core.js');

var win = null;
var winState = 0;

function closeWindow() {
	dialog.showMessageBox(win, { type: "question", title: "Confirm Action", 
	noLink: true, message: "Exit the app?", buttons: ["Quit","Cancel"] }).then(value => {
		if (value.response) {
			winState = 0;
		} else {
			try {
				fs.writeFileSync('./config.json', JSON.stringify(core.getConfigs()));
			} catch (err) {
				dialog.showMessageBoxSync(win, { type: "warning", message: "Unable to save config file. Ensure you are running the app as an administrator." });
			}
			win.close();
			app.quit();
		}
	});
}

function createWindow() {
	win = new BrowserWindow({
		width: 1280,
		height: 800,
		resizable: true,
		fullscreen: false,
		backgroundColor: '#1e1921',
		accessibleTitle: 'AI_UI',
		title: 'AI UI',
		icon: __dirname + '/logo_icon.ico',
		webPreferences: {
			nodeIntegration: true,
			contextIsolation: false,
			preload: path.join(__dirname, 'preload.js')
		}
	});

	win.removeMenu();
	win.loadFile('./index.html');
	
	win.on('close', (e) => {
		if (winState == 0) {
			winState = 1;
			e.preventDefault();
			closeWindow();
		}
	});

	win.webContents.setWindowOpenHandler(({ url }) => {
		if (url.startsWith('https:') || url.startsWith('http:')) {
			shell.openExternal(url);
			return { action: 'deny' };
		}
		return { action: 'allow' };
	});
}

async function handleJpgOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openFile'], filters:[{ name: 'JPG Image', extensions: ['jpg', 'jpeg'] }]});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function handleFileOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openFile']});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function handleDirOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openDirectory']});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

function getAppVersion() {
	return app.getVersion();
}

ipcMain.on('copy-text', (event, payload) => {
	if (payload.txt != '') {
		clipboard.writeText(payload.txt);
		dialog.showMessageBox(win, { type: "info", message: "Text copied to clipboard." });
	} else {
		dialog.showMessageBox(win, { type: "info", message: "There is no text to copy!" });
	}
});

ipcMain.on('show-alert', (event, payload) => {
	dialog.showMessageBox(win, { type: "info", message: payload.msg });
});

// ---- ENGINE INTERCOM ----

function ChatAIReady(init_chat=true) {
	if (init_chat) {
		win.webContents.send('init-ui', { state: true });
	} else {
		win.webContents.send('ai-ready');
	}
}

function ChatAIEnded(p_msg) {
	win.webContents.send('init-ui', { state: false });
	win.webContents.send('prompt-msg', { msg: p_msg });
}

function ShowBotMsg(message) {
	win.webContents.send('bot-msg', { msg: message });
}

function ShowGenTxt(gen_txt) {
	win.webContents.send('gen-result', { txt: gen_txt });
}

ipcMain.on('send-msg', (event, payload) => {
	core.sendMsg(payload.msg);
});

ipcMain.on('gen-text', (event, payload) => {
	core.configGen(payload);
	core.sendMsg('gen_text');
});

// ---- CONFIG STUFF ----

function AddVoice(payload) {
	win.webContents.send('add-voice', payload);
}

ipcMain.on('config-voice', (event, payload) => {
	core.configSpeech(payload);
	core.sendMsg('config_voice');
	win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:true});
});

ipcMain.on('config-ai', (event, payload) => {
	core.configAI(payload);
	core.sendMsg('config_ai');
	win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:true});
});

ipcMain.on('config-app', (event, payload) => {
	dialog.showMessageBox(win, { type: "question", title: "Confirm Action", 
	noLink: true, message: "Updating these settings will restart the AI engine. Proceed?", buttons: ["Confirm","Cancel"] }).then(value => {
		if (!value.response) {
			core.configApp(payload);
			ChatAIEnded('Initializing AI Engine... please wait.');
			win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:true});
			if (core.stopScript('RESTART')) {
				core.startScript(ShowBotMsg, ShowGenTxt, ChatAIReady, ChatAIEnded, AddVoice);
			}
		}
	});
});

ipcMain.on('update-users', (event, payload) => {
	dialog.showMessageBox(win, { type: "question", title: "Confirm Action", 
	noLink: true, message: "Updating the user names will clear the chat log. Proceed?", buttons: ["Confirm","Cancel"] }).then(value => {
		if (!value.response) {
			core.setUsernames(payload);
			core.sendMsg('update_users');
			win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:true});
		}
	});
});

ipcMain.on('update-avatar', (event, payload) => {
	core.setAvatar(payload);
	core.sendMsg('update_avatar');
});

ipcMain.on('update-prompt', (event, payload) => {
	core.setPrompt(payload);
	core.sendMsg('update_prompt');
});

// ---- INIT/END STUFF ----

app.whenReady().then(() => {
	createWindow();

	ipcMain.handle('dialog:openAvatar', handleJpgOpen);
	ipcMain.handle('dialog:openFile', handleFileOpen);
	ipcMain.handle('dialog:openFolder', handleDirOpen);
	ipcMain.handle('getAppVersion', getAppVersion);

	app.on('activate', () => {
		if (BrowserWindow.getAllWindows().length === 0) {
			createWindow();
		}
	});
});

app.on('window-all-closed', () => {
	if (process.platform !== 'darwin') {
		app.quit();
	}
});

ipcMain.handle('restart-script', (event) => {
	dialog.showMessageBox(win, { type: "question", title: "Confirm Action", 
	noLink: true, message: "Restart the AI engine?", buttons: ["Confirm","Cancel"] }).then(value => {
		if (!value.response) {
			ChatAIEnded('Initializing AI Engine... please wait.');
			if (core.stopScript('RESTART')) {
				core.startScript(ShowBotMsg, ShowGenTxt, ChatAIReady, ChatAIEnded, AddVoice);
			}
		}
	});
});

ipcMain.handle('start-script', (event) => {
	if (fs.existsSync('./config.json')) {
		try {
			core.setConfigs(JSON.parse(fs.readFileSync('./config.json')));
			win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:false});
		} catch (err) {
			ChatAIEnded('Error reading config file');
			return;
		}
		core.startScript(ShowBotMsg, ShowGenTxt, ChatAIReady, ChatAIEnded, AddVoice);
	} else {
		ChatAIEnded('No configuration file found. Visit the Settings tab to get started.');
	}
});

ipcMain.handle('close-app', (event) => {
	closeWindow();
});