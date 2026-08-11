const { Readable, Transform, pipeline } = require('stream');
const fs = require('fs');
const path = require('path');
const { dialog } = require('electron');
const AdmZip = require('adm-zip');

function CreateDownloads({ getWin, core, ipcMain }) {
	let currentDownload = 0;
	let failedDownloads = 0;
	let getURLs = [];
	let totalDownloads = 0;
	let abortController = null;
	let closing = false;
	let downloadRemindApply = false;

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
				fs.mkdirSync(dirs[i], { recursive: true });
			}
		}
	}

	function SafeUnlink(path_name) {
		try {
			if (fs.existsSync(path_name)) fs.unlinkSync(path_name);
		} catch (err) {
			/* ignore missing/locked file */
		}
	}

	function CheckDownload(dest, ext, cb) {
		if (ext) {
			if (!ExtractZipFile(dest, ext))
				dialog.showMessageBox(getWin(), { type: "error", message: "Failed to extract "+dest+" into "+ext });
			SafeUnlink(dest);
		}
		if (cb) cb();
	}

	function SendProgress(extra) {
		const win = getWin();
		if (!win || win.isDestroyed()) return;
		win.webContents.send('got-file', Object.assign({
			remaining: currentDownload,
			init: false,
			total: totalDownloads
		}, extra || {}));
	}

	function IsAbortError(err) {
		if (!err) return false;
		const name = err.name || '';
		const msg = String(err.message || err || '');
		return name === 'AbortError' || /aborted|abort/i.test(msg);
	}

	function DownloadFile(url, dest, ext, cb) {
		const name = path.basename(dest);
		const ctrl = abortController;
		const signal = ctrl ? ctrl.signal : undefined;
		fetch(url, signal ? { signal } : undefined).then(response => {
			if (closing || (signal && signal.aborted)) {
				SafeUnlink(dest);
				if (cb) cb("aborted");
				return;
			}
			if (response.ok && response.body) {
				const total = Number(response.headers.get('content-length')) || 0;
				let received = 0;
				let lastSend = 0;
				SendProgress({
					progress: true,
					name,
					received: 0,
					total,
					percent: total > 0 ? 0 : -1
				});
				const file = fs.createWriteStream(dest);
				const counter = new Transform({
					transform(chunk, enc, callback) {
						if (closing || (signal && signal.aborted)) {
							callback(new Error("aborted"));
							return;
						}
						received += chunk.length;
						const now = Date.now();
						if (now - lastSend >= 250) {
							lastSend = now;
							const percent = total > 0 ? Math.min(100, Math.floor((received * 100) / total)) : -1;
							SendProgress({
								progress: true,
								name,
								received,
								total,
								percent
							});
						}
						callback(null, chunk);
					},
					flush(callback) {
						const percent = total > 0 ? 100 : -1;
						SendProgress({
							progress: true,
							name,
							received,
							total,
							percent
						});
						callback();
					}
				});
				pipeline(Readable.fromWeb(response.body), counter, file, (err) => {
					if (err) {
						SafeUnlink(dest);
						if (cb) cb(err.message || "stream failed");
						return;
					}
					if (closing || (signal && signal.aborted)) {
						SafeUnlink(dest);
						if (cb) cb("aborted");
						return;
					}
					CheckDownload(dest, ext, cb);
				});
			} else {
				if (cb) cb("bad response");
			}
		}).catch((err) => {
			SafeUnlink(dest);
			if (cb) cb(IsAbortError(err) ? "aborted" : (err.message || String(err)));
		});
	}

	function DownloadDone(err_msg) {
		--currentDownload;
		if (closing) {
			if (currentDownload <= 0) {
				currentDownload = 0;
				getURLs = [];
				totalDownloads = 0;
				abortController = null;
			}
			return;
		}
		if (err_msg && err_msg !== "aborted") ++failedDownloads;
		SendProgress({});
		if (currentDownload <= 0) {
			if (failedDownloads > 0) {
				dialog.showMessageBox(getWin(), { type: "question", title: "Download Failed",
				noLink: true, message: "Failed to download "+failedDownloads+" model file(s), ensure you are connected to the internet. Retry?", buttons: ["Yes","No"] }).then(value => {
					getWin().webContents.send('got-models', { failed: failedDownloads, retry: !value.response });
					if (value.response) downloadRemindApply = false;
				});
			} else {
				getWin().webContents.send('got-models', {
					failed: 0,
					retry: false,
					remind_apply: downloadRemindApply
				});
				downloadRemindApply = false;
			}
		} else {
			DownloadModels();
		}
	}

	function CancelDownloads() {
		closing = true;
		downloadRemindApply = false;
		if (abortController) {
			try { abortController.abort(); } catch (err) { /* ignore */ }
			abortController = null;
		}
		currentDownload = 0;
		failedDownloads = 0;
		getURLs = [];
		totalDownloads = 0;
	}

	function DownloadModels(urls=false) {
		const workDir = core.getConfigs().app.script_dir;
		if (urls !== false) {
			if (currentDownload > 0) {
				dialog.showMessageBox(getWin(), { type: "warning", message: "Download already in progress!" });
				getWin().webContents.send('got-models', { failed: -1, retry: false });
				return;
			}
			closing = false;
			abortController = new AbortController();
			getURLs = urls;
			failedDownloads = 0;
			currentDownload = urls.length;
			totalDownloads = urls.length;
			getWin().webContents.send('got-file', { remaining: urls.length, total: urls.length, init: true });
		} else if (closing || currentDownload <= 0 || currentDownload > getURLs.length) {
			getWin().webContents.send('got-models', { failed: -1, retry: false });
			return;
		}
		const i = currentDownload - 1;
		const destDir = path.dirname(workDir+getURLs[i].dest);
		CheckDirsExist([destDir]);
		if (getURLs[i].extract == 0) {
			DownloadFile(getURLs[i].src, workDir+getURLs[i].dest, false, DownloadDone);
		} else {
			SafeUnlink(workDir+getURLs[i].dest);
			DownloadFile(getURLs[i].src, workDir+getURLs[i].dest, workDir+getURLs[i].extract.dir, DownloadDone);
		}
	}

	function CheckModelsExist(anim_mode) {
		let urls = {};
		let missingFiles = [];
		let target = '';
		const mode = Number(anim_mode);
		if (!(mode >= 0)) {
			downloadRemindApply = false;
			return;
		}
		const workDir = core.getConfigs().app.script_dir;
		const modelDirs = [
			workDir + "/MakeItTalk/examples/",
			workDir + "/MakeItTalk/examples/ckpt/",
			workDir + "/MakeItTalk/examples/dump/",
			workDir + "/Wav2Lip/checkpoints/",
			workDir + "/SadTalker/checkpoints/",
			workDir + "/SadTalker/gfpgan/",
			workDir + "/SadTalker/gfpgan/weights/",
			workDir + "/IMTalker/checkpoints/",
			workDir + "/IMTalker/checkpoints/wav2vec2-base-960h/",
			workDir + "/IMTalker/results/"
		];

		if (!fs.existsSync(workDir)) {
			dialog.showMessageBox(getWin(), { type: "error", message: "Unable to find Engine Folder. Check the app settings." });
			return;
		}
		CheckDirsExist(modelDirs);

		if (fs.existsSync(workDir+'/model_urls.json')) {
			try {
				urls = JSON.parse(fs.readFileSync(workDir+'/model_urls.json'));
			} catch (err) {
				dialog.showMessageBox(getWin(), { type: "error", message: "Error reading model_urls.json file!" });
				return;
			}
		} else {
			dialog.showMessageBox(getWin(), { type: "error", message: "Could not find model_urls.json file!" });
			return;
		}

		if (mode == 0) {
			urls = urls.MIT;
			target = "MakeItTalk";
		} else if (mode == 1) {
			urls = urls.W2L;
			target = "Wav2Lip";
		} else if (mode == 2) {
			urls = urls.ST;
			target = "SadTalker";
		} else {
			urls = urls.IMT;
			target = "IMTalker";
		}

		if (!Array.isArray(urls)) {
			dialog.showMessageBox(getWin(), { type: "error", message: "No download URLs found for "+target+"." });
			return;
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
			dialog.showMessageBox(getWin(), { type: "question", title: "Missing Models",
			noLink: true, message: target+" model file(s) are missing. Do you want to download them now?", buttons: ["Yes","No"] }).then(value => {
				if (!value.response) {
					DownloadModels(missingFiles);
				} else {
					downloadRemindApply = false;
				}
			});
		} else {
			downloadRemindApply = false;
		}
	}

	ipcMain.on('check-models', (event, payload) => {
		payload = payload || {};
		downloadRemindApply = !!payload.remind_apply;
		CheckModelsExist(payload.mode);
	});

	return { CheckModelsExist, CancelDownloads };
}

module.exports = { CreateDownloads };
