const { Menu, shell } = require('electron');
const path = require('path');
const sessions = require('./sessions.js');

function ShowMenuError(getWin, err) {
	const { dialog } = require('electron');
	dialog.showMessageBox(getWin(), { type: 'error', message: String(err && err.message ? err.message : err) });
}

function SessionMenuLabel(s) {
	let t = String((s && s.title) || 'Session').replace(/[\r\n]+/g, ' ').trim();
	if (!t) t = 'Session';
	if (t.length > 72) t = t.slice(0, 69) + '...';
	return t;
}

/** Open in-app session picker (titles from meta.json). */
function OpenSessionPicker(getWin, scope) {
	const w = getWin();
	if (!w || w.isDestroyed()) return;
	const isProject = scope === 'project';
	const list = isProject
		? sessions.ListProjectSessions(sessions.projectKey)
		: sessions.ListHomeSessions();
	w.webContents.send('session-picker', {
		scope: isProject ? 'project' : 'home',
		title: isProject ? 'Load Project Session' : 'Load Chat Session',
		sessions: list
	});
}

function BuildAppMenu({ sessionOps, getWin, core, sessionLocked }) {
	const st = sessions.GetState();
	const canUse = !sessionLocked;
	const projectOpen = !!st.workspace;
	const recentChats = sessions.RecentChatSessions(core);
	const projectGroups = sessions.RecentProjectSessionGroups(core);

	const recentChatItems = recentChats.length
		? recentChats.map((r) => ({
			label: SessionMenuLabel(r),
			enabled: canUse,
			click: async () => {
				try { await sessionOps.LoadSessionPath(r.path); }
				catch (err) { ShowMenuError(getWin, err); }
			}
		}))
		: [{ label: '(No recent chat sessions)', enabled: false }];

	const recentProjectSessionItems = projectGroups.length
		? projectGroups.map((g) => {
			const name = g.project.name || path.basename(g.project.workspace || g.project.path);
			const sessItems = g.sessions.length
				? g.sessions.map((r) => ({
					label: SessionMenuLabel(r),
					enabled: canUse,
					click: async () => {
						try { await sessionOps.LoadSessionPath(r.path); }
						catch (err) { ShowMenuError(getWin, err); }
					}
				}))
				: [{ label: '(No recent sessions)', enabled: false }];
			return {
				label: name,
				enabled: canUse,
				submenu: sessItems
			};
		})
		: [{ label: '(No recent project sessions)', enabled: false }];

	const fileSub = [
		{
			label: 'New Chat Session',
			accelerator: 'CmdOrCtrl+N',
			enabled: canUse,
			click: async () => {
				try { await sessionOps.NewSession(); }
				catch (err) { ShowMenuError(getWin, err); }
			}
		},
		{
			label: 'Load Chat Session…',
			accelerator: 'CmdOrCtrl+O',
			enabled: canUse,
			click: () => OpenSessionPicker(getWin, 'home')
		},
		{
			label: 'Recent Chat Sessions',
			enabled: canUse,
			submenu: recentChatItems
		},
		{ type: 'separator' },
		{
			label: 'Open Project…',
			accelerator: 'CmdOrCtrl+Shift+O',
			enabled: canUse,
			click: async () => {
				const { dialog } = require('electron');
				const { canceled, filePaths } = await dialog.showOpenDialog(getWin(), {
					properties: ['openDirectory']
				});
				if (canceled || !filePaths || !filePaths[0]) return;
				try { await sessionOps.OpenProject(filePaths[0]); }
				catch (err) { ShowMenuError(getWin, err); }
			}
		},
		{
			label: 'New Project Session',
			accelerator: 'CmdOrCtrl+Shift+N',
			enabled: canUse && projectOpen,
			click: async () => {
				try { await sessionOps.NewProjectSession(); }
				catch (err) { ShowMenuError(getWin, err); }
			}
		},
		{
			label: 'Load Project Session…',
			enabled: canUse && projectOpen,
			click: () => OpenSessionPicker(getWin, 'project')
		},
		{
			label: 'Recent Project Sessions',
			enabled: canUse,
			submenu: recentProjectSessionItems
		},
		{
			id: 'close-project',
			label: 'Close Project',
			enabled: canUse && projectOpen,
			click: async () => {
				try { await sessionOps.CloseProject(); }
				catch (err) { ShowMenuError(getWin, err); }
			}
		}
	];

	const startShortcut = (core.getConfigs().app.start_rec_keys || '').trim();
	const stopShortcut = (core.getConfigs().app.stop_rec_keys || '').trim();
	if (startShortcut || stopShortcut) {
		fileSub.push({ type: 'separator' });
		if (startShortcut) {
			fileSub.push({
				id: 'start-recording',
				label: 'Start Recording',
				accelerator: startShortcut,
				registerAccelerator: false,
				enabled: canUse,
				click: () => { const w = getWin(); if (w) w.webContents.send('start-recording'); }
			});
		}
		if (stopShortcut) {
			fileSub.push({
				id: 'stop-recording',
				label: 'Stop Recording',
				accelerator: stopShortcut,
				registerAccelerator: false,
				enabled: canUse,
				click: () => { const w = getWin(); if (w) w.webContents.send('stop-recording'); }
			});
		}
	}
	fileSub.push({ type: 'separator' });
	// Quit/Close stays available while the engine is busy or initializing.
	fileSub.push({ role: process.platform === 'darwin' ? 'close' : 'quit' });

	const viewSub = [];
	const devMode = !!Number(core.APP_CONFIG && core.APP_CONFIG.enable_devmode);
	if (devMode) {
		viewSub.push({ role: 'toggleDevTools' });
		viewSub.push({ type: 'separator' });
	}
	viewSub.push(
		{ role: 'togglefullscreen' },
		{ type: 'separator' },
		{ role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }
	);

	const template = [
		{
			label: 'File',
			submenu: fileSub
		},
		{
			label: 'Edit',
			submenu: [
				{ role: 'undo' }, { role: 'redo' }, { type: 'separator' },
				{ role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }
			]
		},
		{
			label: 'View',
			submenu: viewSub
		},
		{
			label: 'Help',
			submenu: [
				{
					label: 'Github Issues',
					click: () => { shell.openExternal('https://github.com/JacobBruce/AI-UI/issues'); }
				}
			]
		}
	];

	if (process.platform === 'darwin') {
		template.unshift({
			label: 'AI UI',
			submenu: [
				{ role: 'about' },
				{ type: 'separator' },
				{ role: 'services' },
				{ type: 'separator' },
				{ role: 'hide' },
				{ role: 'hideOthers' },
				{ role: 'unhide' },
				{ type: 'separator' },
				{ role: 'quit' }
			]
		});
	}

	const menu = Menu.buildFromTemplate(template);
	Menu.setApplicationMenu(menu);
	return menu;
}

module.exports = { BuildAppMenu };
