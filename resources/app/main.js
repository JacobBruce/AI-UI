const { app, BrowserWindow, dialog, ipcMain, shell, Menu, MenuItem } = require('electron');
const fs = require('fs');
const path = require('path');
const core = require('./core.js');
const engine = require('./engine.js');
const sessions = require('./sessions.js');
const mcpStore = require('./mcp_store.js');
const { CreateSessionOps } = require('./session_ops.js');
const { BuildAppMenu, ApplyAppMenuBar } = require('./menu.js');
const { CreateDownloads } = require('./ipc/downloads.js');
const { CreateHandlers } = require('./ipc/handlers.js');

var win = null;
var winState = 0;
var SetRecShortcuts = null;
var sessionOps = null;
var downloadsOps = null;
/** File menu session/project items locked while engine initializes or is busy. */
var sessionMenuLocked = true;

function GetWin() {
	return win;
}

function GetAiuiHome() {
	return path.join(app.getPath('home'), '.aiui');
}

function GetConfigPath() {
	return path.join(GetAiuiHome(), 'config.json');
}

function EnsureAiuiHome() {
	sessions.SetAiuiHome(GetAiuiHome());
	return sessions.EnsureHomeLayout();
}

function LoadMcpConfigOnly() {
	core.setMcpConfig(mcpStore.LoadMcpConfig(GetAiuiHome()));
}

function LoadConfig() {
	EnsureAiuiHome();
	const home = GetAiuiHome();
	const cfgPath = GetConfigPath();
	if (!fs.existsSync(cfgPath)) {
		LoadMcpConfigOnly();
		return false;
	}
	try {
		const raw = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
		const rawPacks = raw.ai && raw.ai.tool_packs;
		let mcpCfg = mcpStore.LoadMcpConfig(home);
		if (Array.isArray(rawPacks) && rawPacks.some((p) => p && p.kind === 'mcp' && p.command)) {
			const migrated = mcpStore.MigrateFromToolPacks(home, rawPacks, mcpCfg);
			mcpCfg = migrated.mcpConfig;
		}
		core.setConfigs(raw);
		core.applyMcpOverlayFromDisk(home);
		return true;
	} catch (err) {
		dialog.showMessageBoxSync({ type: "error", message: "Error reading config file!" });
		LoadMcpConfigOnly();
		return false;
	}
}

function SaveConfig() {
	EnsureAiuiHome();
	sessions.SaveActiveConfig(core);
	const cfgPath = GetConfigPath();
	const cur = core.getConfigs();
	if (sessions.projectKey) {
		let base = { chat: cur.chat, ai: cur.ai, gen: cur.gen, app: {} };
		try {
			if (fs.existsSync(cfgPath)) base = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
		} catch (err) { /* use defaults below */ }
		if (!base.app) base.app = {};
		for (const k of sessions.GLOBAL_ONLY_APP) {
			if (Object.prototype.hasOwnProperty.call(cur.app, k)) base.app[k] = cur.app[k];
		}
		base.app.last_session = cur.app.last_session || '';
		base.app.last_project = cur.app.last_project || '';
		base.app.recent_sessions = cur.app.recent_sessions || [];
		base.app.recent_projects = cur.app.recent_projects || [];
		base.app.workspace = '';
		fs.writeFileSync(cfgPath, JSON.stringify(base, null, '\t'));
	} else {
		const all = core.getConfigs();
		all.app.workspace = '';
		fs.writeFileSync(cfgPath, JSON.stringify(all, null, '\t'));
	}
}

function RebuildMenu() {
	BuildAppMenu({ sessionOps, getWin: GetWin, core, sessionLocked: sessionMenuLocked, saveGlobalConfig: SaveConfig });
}

function SetSessionMenuLock(locked) {
	const next = !!locked;
	if (next === sessionMenuLocked) return;
	sessionMenuLocked = next;
	RebuildMenu();
}

function CloseWindow() {
	dialog.showMessageBox(win, { type: "question", title: "Confirm Action",
	noLink: true, message: "Exit the app?", buttons: ["Quit","Cancel"] }).then(async value => {
		if (value.response) {
			winState = 0;
		} else {
			try {
				if (downloadsOps && typeof downloadsOps.CancelDownloads === 'function') {
					downloadsOps.CancelDownloads();
				}
			} catch (err) { /* ignore */ }
			try {
				// Hung generate can ignore SIGTERM; escalate so the Python process
				// does not outlive the UI (common after a stuck tool-call stop).
				engine.stopScript('STOPPING');
			} catch (err) { /* ignore */ }
			try {
				if (sessionOps) await sessionOps.Autosave({ force: false });
				SaveConfig();
			} catch (err) {
				dialog.showMessageBoxSync(win, { type: "warning", message: "Unable to save config/session. Ensure you are running the app as an administrator." });
			}
			win.close();
			app.quit();
		}
	});
}

async function HandleAvatarOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {
		properties: ['openFile'],
		filters: [{ name: 'Image', extensions: ['jpg', 'jpeg', 'png'] }]
	});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function HandleWavOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openFile'], filters:[{ name: 'Wave Sound', extensions: ['wav'] }]});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function HandleFileOpen(filters=[{ name: 'All Files', extensions: ['*'] }]) {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openFile'], filters:filters});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function HandleFileSave(filters=[{ name: 'All Files', extensions: ['*'] }]) {
	const { canceled, filePath } = await dialog.showSaveDialog(win, {properties: ['showOverwriteConfirmation'], filters:filters});
	if (canceled) {
		return false;
	} else {
		return filePath;
	}
}

async function HandleDirOpen() {
	const { canceled, filePaths } = await dialog.showOpenDialog(win, {properties: ['openDirectory']});
	if (canceled) {
		return false;
	} else {
		return filePaths[0];
	}
}

async function HandleMixOpen() {
	let response = dialog.showMessageBoxSync(win, { type: "question", title: "File or Folder",
	noLink: true, message: "Open a file or a folder?", buttons: ["File","Folder"] });

	if (response) {
		return await HandleDirOpen();
	} else {
		return await HandleFileOpen();
	}
}

function GetAppVersion() {
	return app.getVersion();
}

function CreateWindow() {
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
			nodeIntegration: false,
			contextIsolation: true,
			sandbox: true,
			preload: path.join(__dirname, 'preload.js')
		}
	});

	RebuildMenu();
	ApplyAppMenuBar(win, core);
	win.loadFile('./index.html');

	function RestoreMenuBar() {
		ApplyAppMenuBar(win, core);
	}
	// Linux: View → Toggle Full Screen dismisses the GTK menu first, so Electron
	// snapshots the bar as hidden and never restores it on exit (F11-only is fine).
	for (const ev of ['enter-full-screen', 'leave-full-screen', 'enter-html-full-screen', 'leave-html-full-screen']) {
		win.on(ev, () => { setTimeout(RestoreMenuBar, 0); });
	}

	win.on('close', (e) => {
		if (winState == 0) {
			winState = 1;
			e.preventDefault();
			CloseWindow();
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
		const devMode = !!Number(core.APP_CONFIG && core.APP_CONFIG.enable_devmode);

		for (const suggestion of params.dictionarySuggestions) {
			menu.append(new MenuItem({
				label: suggestion,
				click: () => win.webContents.replaceMisspelling(suggestion)
			}));
			itemCount++;
		}

		if (params.misspelledWord) {
			menu.append(new MenuItem({
				label: 'Add to dictionary',
				click: () => win.webContents.session.addWordToSpellCheckerDictionary(params.misspelledWord)
			}));
			itemCount++;
		}

		if (devMode) {
			if (itemCount > 0) menu.append(new MenuItem({ type: 'separator' }));
			menu.append(new MenuItem({
				label: 'Inspect Element',
				click: () => win.webContents.inspectElement(params.x, params.y)
			}));
			itemCount++;
		}

		if (itemCount > 0) menu.append(new MenuItem({ type: 'separator' }));
		menu.append(new MenuItem({
			label: 'Reset View',
			click: () => {
				if (core && core.APP_CONFIG) {
					core.APP_CONFIG.show_app_menu = 1;
					core.APP_CONFIG.show_side_panel = 1;
				}
				SaveConfig();
				ApplyAppMenuBar(win, core);
				RebuildMenu();
				if (win && !win.isDestroyed()) win.webContents.send('set-side-panel', 1);
			}
		}));
		menu.popup();
	});

	win.webContents.on('did-finish-load', () => {
		win.webContents.send('set-platform', process.platform);
		const show = (core.APP_CONFIG && Number(core.APP_CONFIG.show_side_panel) === 0) ? 0 : 1;
		win.webContents.send('set-side-panel', show);
	});
}

function RegisterIpc() {
	sessionOps = CreateSessionOps({
		core,
		engine,
		getWin: GetWin,
		dialog,
		rebuildMenu: RebuildMenu,
		saveGlobalConfig: SaveConfig
	});

	const downloads = CreateDownloads({ getWin: GetWin, core, ipcMain });
	downloadsOps = downloads;
	const handlers = CreateHandlers({
		getWin: GetWin,
		core,
		engine,
		dialogs: { HandleFileOpen, HandleFileSave },
		checkModelsExist: downloads.CheckModelsExist,
		getConfigPath: GetConfigPath,
		sessionOps,
		sessions,
		rebuildMenu: RebuildMenu,
		saveGlobalConfig: SaveConfig
	});
	SetRecShortcuts = handlers.SetRecShortcuts;

	ipcMain.handle('dialog:openAvatar', HandleAvatarOpen);
	ipcMain.handle('dialog:openSample', HandleWavOpen);
	ipcMain.handle('dialog:openFile', HandleFileOpen);
	ipcMain.handle('dialog:openFolder', HandleDirOpen);
	ipcMain.handle('dialog:openFileFolder', HandleMixOpen);
	ipcMain.handle('getAppVersion', GetAppVersion);
	ipcMain.handle('close-app', (event) => {
		CloseWindow();
	});
	ipcMain.handle('session-autosave', async () => sessionOps.Autosave({ force: false }));
	ipcMain.handle('session-new', async () => sessionOps.NewSession());
	ipcMain.handle('session-load', async (event, selectedPath) => sessionOps.LoadSessionPath(selectedPath));
	ipcMain.handle('session-open-project', async (event, projectPath) => sessionOps.OpenProject(projectPath));
	ipcMain.handle('session-close-project', async () => sessionOps.CloseProject());
	ipcMain.handle('session-get-context', async () => {
		const st = sessions.GetState();
		return { ...st, recent: core.APP_CONFIG.recent_sessions || [] };
	});
	ipcMain.on('set-session-menu-lock', (event, payload) => {
		SetSessionMenuLock(payload && payload.locked);
	});
}

app.whenReady().then(() => {
	LoadConfig();
	engine.setPlatform(process.platform);
	RegisterIpc();
	CreateWindow();

	app.on('activate', () => {
		if (BrowserWindow.getAllWindows().length === 0) {
			CreateWindow();
		}
	});
});

app.on('window-all-closed', () => {
	if (process.platform !== 'darwin') {
		app.quit();
	}
});
