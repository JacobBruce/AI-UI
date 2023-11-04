const { app, BrowserWindow, dialog, clipboard, ipcMain, shell, Menu, MenuItem } = require('electron');
const fs = require('fs');
const path = require('path');
const core = require('./core.js');
const https = require('https');
const AdmZip = require("adm-zip");

var win = null;
var winState = 0;

var currentDownload = 0;
var failedDownloads = 0;
var getURLs = [];

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
			spellcheck: true,
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

	win.webContents.on('context-menu', (event, params) => {
		const menu = new Menu();
		let itemCount = 0;

		// Add each spelling suggestion to context menu
		for (const suggestion of params.dictionarySuggestions) {
			menu.append(new MenuItem({
				label: suggestion,
				click: () => win.webContents.replaceMisspelling(suggestion)
			}));
			itemCount++;
		}

		// Allow users to add the misspelled word to the dictionary
		if (params.misspelledWord) {
			menu.append(new MenuItem({
				label: 'Add to dictionary',
				click: () => win.webContents.session.addWordToSpellCheckerDictionary(params.misspelledWord)
			}));
			itemCount++;
		}

		if (itemCount > 0) menu.popup();
	})
}

function getAppVersion() {
	return app.getVersion();
}

async function handleJpgOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openFile'], filters:[{ name: 'JPG Image', extensions: ['jpg', 'jpeg'] }]});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function handleWavOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openFile'], filters:[{ name: 'Wave Sound', extensions: ['wav'] }]});
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

async function handleMixOpen() {
	let response = dialog.showMessageBoxSync(win, { type: "question", title: "File or Folder", 
	noLink: true, message: "Open a file or a folder?", buttons: ["File","Folder"] });
	
	if (response) {
		return await handleDirOpen();
	} else {
		return await handleFileOpen();
	}
}

ipcMain.on('show-in-dir', (event, payload) => {
	const norm_path = path.normalize(payload);
	if (fs.existsSync(norm_path)) shell.showItemInFolder(norm_path);
});

ipcMain.on('copy-text', (event, payload) => {
	if (payload.txt != '') {
		clipboard.writeText(payload.txt);
		dialog.showMessageBox(win, { type: "info", message: "Text copied to clipboard." });
	} else {
		dialog.showMessageBox(win, { type: "info", message: "There is no text to copy!" });
	}
});

ipcMain.on('copy-image', (event, payload) => {
	if (payload.img != '') {
		clipboard.writeImage(payload.img);
		dialog.showMessageBox(win, { type: "info", message: "Image copied to clipboard." });
	} else {
		dialog.showMessageBox(win, { type: "info", message: "There is no image to copy!" });
	}
});

ipcMain.on('show-alert', (event, payload) => {
	const msg_type = payload.type ? payload.type : 'info';
	dialog.showMessageBox(win, { type: msg_type, message: payload.msg });
});

// ---- FILE DOWNLOADING ----

function ExtractZipFile(zip_file, dest_dir) {
	try {
		let zipFile = new AdmZip(zip_file);
		zipFile.extractAllTo(dest_dir, true);
		return true;
	} catch (err) {
		return false;
	}
}

function CheckDirsExist(dirs) {
	for (let i=0; i<dirs.length; ++i) {
		if (!fs.existsSync(dirs[i])) {
			fs.mkdirSync(dirs[i]);
		}
	}
}

function CheckDownload(dest, ext, cb) {
	if (ext) {
		if (!ExtractZipFile(dest, ext))
			dialog.showMessageBox(win, { type: "error", message: "Failed to extract "+dest+" into "+ext });
		fs.unlink(dest);
	}
	if (cb) cb();
}

function DownloadFile(url, dest, ext, cb) {  
	let request = https.get(url, function(response) {
		let file = fs.createWriteStream(dest);
		response.pipe(file);
		file.on("finish", () => {
			file.close(function() {
				CheckDownload(dest, ext, cb);
			});
		});
	}).on('error', function(err) {
		fs.unlink(dest);
		if (cb) cb(err.message);
	});
};

function DownloadDone(err_msg='') {
	--currentDownload;
	if (err_msg != '') ++failedDownloads;
	win.webContents.send('got-file', { remaining: currentDownload, init: false });
	if (currentDownload <= 0) {
		if (failedDownloads > 0) {
			dialog.showMessageBox(win, { type: "question", title: "Download Failed", 
			noLink: true, message: "Failed to download "+failedDownloads+" model file(s), ensure you are connected to the internet. Retry?", buttons: ["Yes","No"] }).then(value => {
				win.webContents.send('got-models', { failed: failedDownloads, retry: !value.response });
			});
		} else {
			win.webContents.send('got-models', { failed: 0, retry: false });
		}
	} else {
		DownloadModels();
	}
}

function DownloadModels(urls=false) {
	const workDir = core.getConfigs().app.script_dir;
	if (urls !== false) {
		if (currentDownload > 0) {
			dialog.showMessageBox(win, { type: "warning", message: "Download already in progress!" });
			win.webContents.send('got-models', { failed: -1, retry: false });
			return;
		}
		getURLs = urls;
		failedDownloads = 0;
		currentDownload = urls.length;
		win.webContents.send('got-file', { remaining: urls.length, init: true });
	} else if (currentDownload <= 0 || currentDownload > getURLs.length) {
		win.webContents.send('got-models', { failed: -1, retry: false });
		return;
	}
	const i = currentDownload - 1;
	if (getURLs[i].extract == 0) {
		DownloadFile(getURLs[i].src, workDir+getURLs[i].dest, false, DownloadDone);
	} else {
		if (fs.existsSync(workDir+getURLs[i].dest)) fs.unlink(workDir+getURLs[i].dest);
		DownloadFile(getURLs[i].src, workDir+getURLs[i].dest, workDir+getURLs[i].extract.dir, DownloadDone);
	}
}

function CheckModelsExist(anim_mode) {
	let urls = {};
	let missingFiles = [];
	let target = '';
	const workDir = core.getConfigs().app.script_dir;
	const modelDirs = [
		workDir + "/MakeItTalk/examples/",
		workDir + "/MakeItTalk/examples/ckpt/",
		workDir + "/MakeItTalk/examples/dump/",
		workDir + "/Wav2Lip/checkpoints/",
		workDir + "/SadTalker/checkpoints/",
		workDir + "/SadTalker/gfpgan/",
		workDir + "/SadTalker/gfpgan/weights/"
	];
	if (!fs.existsSync(workDir)) {
		dialog.showMessageBox(win, { type: "error", message: "Unable to find Engine Folder. Check the app settings." });
		return;
	}
	CheckDirsExist(modelDirs);
	if (fs.existsSync(workDir+'/model_urls.json')) {
		try {
			urls = JSON.parse(fs.readFileSync(workDir+'/model_urls.json'));
		} catch (err) {
			dialog.showMessageBox(win, { type: "error", message: "Error reading model_urls.json file!" });
			return;
		}
	} else {
		dialog.showMessageBox(win, { type: "error", message: "Could not find model_urls.json file!" });
		return;
	}
	if (anim_mode == 0) {
		urls = urls.MIT;
		target = "MakeItTalk";
	} else if (anim_mode == 1) {
		urls = urls.W2L;
		target = "Wav2Lip";
	} else {
		urls = urls.ST;
		target = "SadTalker";
	}
	for (let i=0; i<urls.length; ++i) {
		if (urls[i].extract == 0) {
			if (!fs.existsSync(workDir+urls[i].dest)) missingFiles.push(urls[i]);
		} else {
			for (let f=0; f<urls[i].extract.check.length; ++f) {
				if (!fs.existsSync(workDir+urls[i].extract.check[f])) {
					missingFiles.push(urls[i]);
					break;
				}
			}
		}
	}
	if (missingFiles.length > 0) {
		dialog.showMessageBox(win, { type: "question", title: "Missing Models", 
		noLink: true, message: target+" model files are missing. Do you want to download them now?", buttons: ["Yes","No"] }).then(value => {
			if (!value.response) DownloadModels(missingFiles);
		});
	}
}

ipcMain.on('check-models', (event, payload) => {
	CheckModelsExist(payload.mode);
});

// ---- ENGINE INTERCOM ----

function ChatAIReady(init_prompt='') {
	if (init_prompt != '') {
		win.webContents.send('init-ui', { state: init_prompt });
	} else {
		win.webContents.send('ai-ready');
	}
}

function ChatAIEnded(p_msg) {
	win.webContents.send('init-ui', { state: false });
	win.webContents.send('prompt-msg', { msg: p_msg });
}

function ShowBotMsg(message, anim_done=true) {
	win.webContents.send('bot-msg', { msg: message, got_vid: anim_done });
}

function ShowGenTxt(gen_txt) {
	win.webContents.send('gen-result', { txt: gen_txt });
}

function ShowGenTTS(gen_tts) {
	win.webContents.send('tts-result', { wav: gen_tts });
}

function ShowGenImg(gen_img) {
	win.webContents.send('img-result', { img: gen_img });
}

function ShowClonedVoice(gen_voice) {
	win.webContents.send('clone-result', { voice: gen_voice });
}

function AppendLog(log_msg) {
	win.webContents.send('append-log', { msg: log_msg });
}

function PlayAudio(audio_file) {
	win.webContents.send('play-audio', { file: audio_file });
}

function GotAvatar(got_avatar) {
	win.webContents.send('got-avatar', { got: got_avatar });
}

ipcMain.on('send-msg', (event, payload) => {
	core.sendMsg(payload.msg);
});

ipcMain.on('gen-image', (event, payload) => {
	core.configGen(payload);
	core.sendMsg('gen_image');
});

ipcMain.on('gen-text', (event, payload) => {
	core.configGen(payload);
	core.sendMsg('gen_text');
});

ipcMain.on('gen-speech', (event, payload) => {
	core.configGen(payload);
	core.setReadText(payload.tts_txt);
	core.sendMsg('gen_speech');
});

ipcMain.on('read-text', (event, payload) => {
	if (payload.txt != '') {
		core.setReadText(payload.txt);
		core.sendMsg('read_text');
	} else {
		dialog.showMessageBox(win, { type: "info", message: "There is no text to read!" });
	}
});

ipcMain.on('clone-voice', (event, payload) => {
	const workDir = core.getConfigs().app.script_dir;
	core.setCloneVoice(payload);
	if (fs.existsSync(workDir+'/embeddings/'+payload.name+'.npy')) {
		dialog.showMessageBox(win, { type: "question", title: "Confirm Action", 
		noLink: true, message: "A voice with that name already exists. Overwrite it?", buttons: ["Yes","No"] }).then(value => {
			if (!value.response) {
				core.sendMsg('clone_voice');
			} else {
				ShowClonedVoice("ERROR:A voice with that name already exists.");
			}
		});
	} else {
		core.sendMsg('clone_voice');
	}
});

ipcMain.on('run-cmd', (event, payload) => {
	if (core.engineRunning()) {
		dialog.showMessageBox(win, { type: "question", title: "Engine Running", 
		noLink: true, message: "The AI engine must be stopped to run commands. Continue?", buttons: ["Yes","No"] }).then(value => {
			if (!value.response) {
				core.stopScript();
				setTimeout(function() {
					win.webContents.send('cmd-sent', core.runCommand(payload.cmd));
				}, 1000);
			} else {
				win.webContents.send('cmd-sent', false);
			}
		});
	} else {
		win.webContents.send('cmd-sent', core.runCommand(payload.cmd));
	}
});

// ---- CONFIG STUFF ----

function AddVoices(payload) {
	win.webContents.send('add-voices', payload);
}

function ClearVoices() {
	win.webContents.send('clear-voices');
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
				core.startScript();
			}
			let customModel = false;
			if (payload.model_type == 3 || payload.model_type == 4) {
				customModel = true;
			} else {
				let args = payload.model_args.split(',');
				for (let i=0; i<args.length; i++) {
					const arg = args[i].trim();
					if (arg.startsWith('custom_model_code=') || arg.startsWith('custom_token_code=')) {
						let val = arg.substring(arg.indexOf('=')).toLowerCase();
						if (arg == 'false' || arg == '0') continue;
						customModel = true;
						break;
					}
				}
			}
			/*if (customModel && payload.model_dir.length > 1 && (payload.model_dir[1] == ':' || fs.existsSync(payload.model_dir))) {
				dialog.showMessageBox(win, { type: "warning", message: "It appears you are using a model with custom code, this may not load from a local directory." });
			}*/
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

ipcMain.on('update-talk-mode', (event, payload) => {
	core.setTalkMode(payload);
	core.sendMsg('update_tmode');
});

ipcMain.on('update-prompt', (event, payload) => {
	core.setPrompt(payload);
	core.sendMsg('update_prompt');
});

// ---- INIT/END STUFF ----

app.whenReady().then(() => {
	createWindow();

	ipcMain.handle('dialog:openAvatar', handleJpgOpen);
	ipcMain.handle('dialog:openSample', handleWavOpen);
	ipcMain.handle('dialog:openFile', handleFileOpen);
	ipcMain.handle('dialog:openFolder', handleDirOpen);
	ipcMain.handle('dialog:openFileFolder', handleMixOpen);
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
				core.startScript();
			}
		}
	});
});

ipcMain.handle('start-script', (event) => {
	core.setCallbacks(ShowBotMsg, ShowGenTxt, ShowGenTTS, ShowGenImg, ShowClonedVoice, ChatAIReady, ChatAIEnded, AddVoices, ClearVoices, PlayAudio, AppendLog, GotAvatar);
	if (fs.existsSync('./config.json')) {
		try {
			core.setConfigs(JSON.parse(fs.readFileSync('./config.json')));
			win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:false});
		} catch (err) {
			ChatAIEnded('Error reading config file');
			return;
		}
		CheckModelsExist(core.getConfigs().chat.anim_mode);
		core.startScript();
	} else {
		let workDir = process.cwd().trim().replaceAll('\\', '/');
		if (workDir.endsWith('/')) workDir = workDir.slice(0, -1);
		if (fs.existsSync(workDir+'/engine/ai_images/default_avatar.jpg'))
			core.initAppConfig(workDir+'/engine/ai_images/default_avatar.jpg', workDir+'/engine');
		win.webContents.send('load-config', {configs:core.getConfigs(), skip_inputs:false});
		ChatAIEnded('No configuration file found. Visit the Settings tab to get started.');
	}
});

ipcMain.handle('close-app', (event) => {
	closeWindow();
});