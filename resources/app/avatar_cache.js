const fs = require('fs');
const path = require('path');
const { nativeImage } = require('electron');

function AvatarTargetSize(animMode) {
	return Number(animMode) === 3 ? 512 : 256;
}

function AvatarSrcPath(aiDir) {
	return path.join(aiDir, 'avatar_src.jpg');
}

function AvatarSizedPath(aiDir, targetSize) {
	return path.join(aiDir, 'avatar_' + targetSize + 'x' + targetSize + '.jpg');
}

function SafeUnlink(filePath) {
	try {
		if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
	} catch (err) {
		/* ignore */
	}
}

/** Drop sized work copies so they are rebuilt from a new master. */
function InvalidateSizedAvatars(aiDir) {
	if (!fs.existsSync(aiDir)) return;
	let names = [];
	try {
		names = fs.readdirSync(aiDir);
	} catch (err) {
		return;
	}
	for (let i = 0; i < names.length; i++) {
		const name = names[i];
		if (/^avatar_\d+x\d+\.jpg$/i.test(name) || name === 'avatar.jpg') {
			SafeUnlink(path.join(aiDir, name));
		}
	}
}

function IsJpegPath(filePath) {
	const ext = path.extname(String(filePath || '')).toLowerCase();
	return ext === '.jpg' || ext === '.jpeg';
}

function WriteMasterFromPath(sourcePath, srcCache) {
	if (IsJpegPath(sourcePath)) {
		fs.copyFileSync(sourcePath, srcCache);
		return { ok: true };
	}
	const picked = nativeImage.createFromPath(sourcePath);
	if (picked.isEmpty()) {
		return { ok: false, error: 'Could not load avatar image.' };
	}
	fs.writeFileSync(srcCache, picked.toJPEG(100));
	return { ok: true };
}

function BuildSizedFromMaster(srcCache, workCache, targetSize) {
	const master = nativeImage.createFromPath(srcCache);
	if (master.isEmpty()) {
		return { ok: false, error: 'Could not load avatar image.' };
	}
	const { width, height } = master.getSize();
	const warned = width < targetSize || height < targetSize;
	if (width === targetSize && height === targetSize) {
		fs.copyFileSync(srcCache, workCache);
	} else {
		const scale = Math.max(targetSize / width, targetSize / height);
		const rw = Math.max(targetSize, Math.round(width * scale));
		const rh = Math.max(targetSize, Math.round(height * scale));
		const resized = master.resize({ width: rw, height: rh, quality: 'best' });
		const x = Math.max(0, Math.floor((rw - targetSize) / 2));
		const y = Math.max(0, Math.floor((rh - targetSize) / 2));
		const cropped = resized.crop({ x, y, width: targetSize, height: targetSize });
		fs.writeFileSync(workCache, cropped.toJPEG(100));
	}
	return {
		ok: true,
		path: workCache.replaceAll('\\', '/'),
		warned,
		width,
		height,
		target: targetSize
	};
}

/**
 * Ensure master + sized work file for anim_mode.
 * Sized files are created only when missing (or after a new source pick).
 * @param {object} opts
 * @param {string} [opts.source] - new pick path; updates avatar_src and invalidates sized caches
 * @param {number|string} opts.mode - anim_mode
 * @param {string} opts.workDir - engine script_dir
 * @param {string} [opts.fallback] - existing avatar_img if src missing
 */
function PrepareAvatarImage(opts) {
	opts = opts || {};
	const workDir = String(opts.workDir || '').replace(/[/\\]+$/, '');
	if (!workDir) return { ok: false, error: 'Engine folder not set.' };
	const targetSize = AvatarTargetSize(opts.mode);
	const aiDir = path.join(workDir, 'ai_images');
	const srcCache = AvatarSrcPath(aiDir);
	const workCache = AvatarSizedPath(aiDir, targetSize);
	const absSrcCache = path.resolve(srcCache);
	const absWorkCache = path.resolve(workCache);

	try {
		if (!fs.existsSync(aiDir)) fs.mkdirSync(aiDir, { recursive: true });

		const sourcePath = opts.source != null && opts.source !== '' ? String(opts.source) : '';
		if (sourcePath && fs.existsSync(sourcePath)) {
			const absMaster = path.resolve(sourcePath);
			const isCachePath = absMaster === absSrcCache
				|| absMaster === absWorkCache
				|| /^avatar_\d+x\d+\.jpg$/i.test(path.basename(sourcePath));
			if (!isCachePath) {
				const wrote = WriteMasterFromPath(sourcePath, srcCache);
				if (!wrote.ok) return wrote;
				InvalidateSizedAvatars(aiDir);
			} else if (!fs.existsSync(srcCache)) {
				const wrote = WriteMasterFromPath(sourcePath, srcCache);
				if (!wrote.ok) return wrote;
			}
		} else if (!fs.existsSync(srcCache)) {
			const fallback = String(opts.fallback || '');
			if (fallback && fs.existsSync(fallback)) {
				const wrote = WriteMasterFromPath(fallback, srcCache);
				if (!wrote.ok) return wrote;
			} else {
				return { ok: false, error: 'No avatar image to prepare.' };
			}
		}

		if (fs.existsSync(workCache)) {
			return {
				ok: true,
				path: workCache.replaceAll('\\', '/'),
				warned: false,
				width: targetSize,
				height: targetSize,
				target: targetSize,
				cached: true
			};
		}

		return BuildSizedFromMaster(srcCache, workCache, targetSize);
	} catch (err) {
		return { ok: false, error: String(err && err.message ? err.message : err) };
	}
}

/** Point core avatar_img at the sized work file for the current anim_mode. */
function EnsureAvatarForCore(core) {
	const cfg = core.getConfigs();
	const workDir = String((cfg.app && cfg.app.script_dir) || '').replace(/[/\\]+$/, '');
	const mode = (cfg.chat && cfg.chat.anim_mode != null) ? cfg.chat.anim_mode : 0;
	const res = PrepareAvatarImage({
		mode,
		workDir,
		fallback: (cfg.app && cfg.app.avatar_img) || ''
	});
	if (res.ok && typeof core.setAvatar === 'function') {
		core.setAvatar(res.path);
	}
	return res;
}

module.exports = {
	AvatarTargetSize,
	PrepareAvatarImage,
	EnsureAvatarForCore,
	InvalidateSizedAvatars
};
