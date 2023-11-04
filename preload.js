window.aiuiAPI = {
	openAvatar: () => ipcRenderer.invoke('dialog:openAvatar'),
	openSample: () => ipcRenderer.invoke('dialog:openSample'),
	openFile: () => ipcRenderer.invoke('dialog:openFile'),
	openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
	openFileFolder: () => ipcRenderer.invoke('dialog:openFileFolder'),
	getAppVersion: () => ipcRenderer.invoke('getAppVersion')
}

window.addEventListener('DOMContentLoaded', () => {
	const replaceText = (selector, text) => {
		const element = document.getElementById(selector);
		if (element) element.innerText = text;
	}

	for (const type of ['chrome', 'node', 'electron']) {
		replaceText(`${type}-version`, process.versions[type]);
	}
});