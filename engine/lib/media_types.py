"""Shared media extension / MIME helpers for attach + remote encoding."""

from __future__ import annotations

import mimetypes
import os

IMAGE_EXTS = {
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".png": "image/png",
	".gif": "image/gif",
	".webp": "image/webp",
	".bmp": "image/bmp",
	".tiff": "image/tiff",
	".tif": "image/tiff",
}

VIDEO_EXTS = {
	".mp4": "video/mp4",
	".webm": "video/webm",
	".mkv": "video/x-matroska",
	".mov": "video/quicktime",
	".avi": "video/x-msvideo",
	".wmv": "video/x-ms-wmv",
	".flv": "video/x-flv",
	".mpeg": "video/mpeg",
	".mpg": "video/mpeg",
}

AUDIO_EXTS = {
	".mp3": "audio/mpeg",
	".wav": "audio/wav",
	".ogg": "audio/ogg",
	".wma": "audio/x-ms-wma",
	".flac": "audio/flac",
	".aac": "audio/aac",
	".m4a": "audio/mp4",
}

# Remote data-URL size caps (videos are often much larger than images).
REMOTE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
REMOTE_VIDEO_MAX_BYTES = 100 * 1024 * 1024


def ExtOf(path_or_url: str) -> str:
	return os.path.splitext(str(path_or_url or "").split("?", 1)[0])[1].lower()


def ImageMime(path_or_url: str) -> str:
	ext = ExtOf(path_or_url)
	if ext in IMAGE_EXTS:
		return IMAGE_EXTS[ext]
	guess, _ = mimetypes.guess_type(str(path_or_url or ""))
	if guess and guess.startswith("image/"):
		return guess
	return "image/jpeg"


def VideoMime(path_or_url: str) -> str:
	ext = ExtOf(path_or_url)
	if ext in VIDEO_EXTS:
		return VIDEO_EXTS[ext]
	guess, _ = mimetypes.guess_type(str(path_or_url or ""))
	if guess and guess.startswith("video/"):
		return guess
	return "video/mp4"


def ClassifyPath(path: str) -> str:
	"""Return attach kind: image | video | audio | text (caller may refine text->file)."""
	ext = ExtOf(path)
	if ext in IMAGE_EXTS:
		return "image"
	if ext in VIDEO_EXTS:
		return "video"
	if ext in AUDIO_EXTS:
		return "audio"
	return "text"


def IsHttpUrl(path_or_url: str) -> bool:
	s = str(path_or_url or "").strip().lower()
	return s.startswith("http://") or s.startswith("https://")


def NormalizeAttachRef(path_or_url: str) -> str:
	"""Normalize user attach refs; prepend https:// for bare www. links."""
	s = str(path_or_url or "").strip()
	if not s:
		return s
	if IsHttpUrl(s):
		return s
	if s.lower().startswith("www."):
		return "https://" + s
	return s


def NormMmProcessor(raw) -> str:
	mp = str(raw or "auto").strip().lower()
	return mp if mp in ("qwen_vl", "janus") else "auto"
