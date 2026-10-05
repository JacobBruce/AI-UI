"""Per-user-message git working-tree snapshot for the model wire.

Stored on the message as `_git_status` (chat.json) and merged into `content`
only when building the model request. The UI paints `content` and never sees it.
"""

from __future__ import annotations

import os
import subprocess

E = None

GIT_STATUS_KEY = "_git_status"

_SKIP_DIR_NAMES = {".git", "node_modules", "vendor", ".venv", "__pycache__"}
_MAX_STATUS_LINES = 60
_GIT_TIMEOUT = 3


def bind(engine_module):
	global E
	E = engine_module


def CaptureGitContext():
	"""Return porcelain snapshot text, or None when disabled / not a repo / clean."""
	if E is None:
		return None
	if not int(getattr(E, "git_status_context", 0) or 0):
		return None
	root = str(getattr(E, "workspace_dir", "") or "").strip()
	if not root or not os.path.isdir(root):
		return None
	if not _git_is_work_tree(root):
		return None
	branch = _git_text(root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "HEAD"
	raw = _git_text(root, ["status", "--porcelain", "--untracked-files=normal"])
	if raw is None:
		return None
	lines, omitted = _filter_status_lines(raw.splitlines())
	if not lines and not omitted:
		return None
	out = ["# Working tree (git status)", f"branch: {branch}"]
	if not lines and omitted:
		out.append(f"(no listed paths; omitted {omitted} under {', '.join(sorted(_SKIP_DIR_NAMES))})")
	else:
		out.extend(lines[:_MAX_STATUS_LINES])
		extra = max(0, len(lines) - _MAX_STATUS_LINES)
		note = []
		if extra:
			note.append(f"{extra} more")
		if omitted:
			note.append(f"{omitted} omitted ({', '.join(sorted(_SKIP_DIR_NAMES))})")
		if note:
			out.append("… " + ", ".join(note))
	return "\n".join(out)


def StampUserGitStatus(msg):
	"""Snapshot git status onto a user message once (in-place)."""
	if E is None or not isinstance(msg, dict):
		return msg
	if getattr(E, "_compact_bypass", False):
		return msg
	if not int(getattr(E, "git_status_context", 0) or 0):
		return msg
	if msg.get(GIT_STATUS_KEY):
		return msg
	block = CaptureGitContext()
	if block:
		msg[GIT_STATUS_KEY] = block
	return msg


def _IsUserMsg(msg):
	role = str(msg.get("role") or "").lower()
	user = str(getattr(E, "user_name", "user") or "user").lower()
	return role in ("user", user)


def _AppendGitBlock(content, block):
	if isinstance(content, str):
		body = content.rstrip()
		return body + "\n\n" + block if body else block
	if isinstance(content, list):
		parts = list(content)
		parts.append({"type": "text", "text": block})
		return parts
	if content in (None, ""):
		return block
	return str(content).rstrip() + "\n\n" + block


def AttachGitContext(messages):
	"""Copy-on-write: append each stored `_git_status` to that user message.

	Replay is per-message and byte-stable so later turns do not edit earlier
	prefixes. Setting off, summarizer side-channel, and compact handoff
	(`_aiui_summary`) skip attach. No user field → leave content unchanged.
	"""
	if E is None:
		return messages
	if getattr(E, "_compact_bypass", False):
		return messages
	if not int(getattr(E, "git_status_context", 0) or 0):
		return messages
	raw = list(messages or [])
	out = []
	changed = False
	for m in raw:
		if not isinstance(m, dict) or not _IsUserMsg(m) or m.get("_aiui_summary"):
			out.append(m)
			continue
		block = m.get(GIT_STATUS_KEY)
		if not block:
			out.append(m)
			continue
		msg = dict(m)
		msg["content"] = _AppendGitBlock(msg.get("content"), str(block))
		out.append(msg)
		changed = True
	return out if changed else messages


def _git_is_work_tree(root):
	text = _git_text(root, ["rev-parse", "--is-inside-work-tree"])
	return str(text or "").strip().lower() == "true"


def _git_text(root, args):
	try:
		proc = subprocess.run(
			["git", "-C", root, "-c", "color.ui=false", *args],
			capture_output=True,
			text=True,
			timeout=_GIT_TIMEOUT,
			errors="replace",
			env={**os.environ, "GIT_OPTIONAL_LOCKS": "1", "GIT_TERMINAL_PROMPT": "0"},
		)
	except (OSError, subprocess.TimeoutExpired):
		return None
	if proc.returncode != 0:
		return None
	return proc.stdout or ""


def _status_path(line):
	s = (line or "").rstrip("\n")
	if len(s) < 4:
		return ""
	body = s[3:]
	if " -> " in body:
		body = body.split(" -> ", 1)[-1]
	return body.strip().strip('"')


def _omit_status_path(path):
	parts = [p for p in path.replace("\\", "/").split("/") if p]
	if not parts:
		return False
	if len(parts) == 1 and parts[0] in _SKIP_DIR_NAMES:
		return False
	return any(p in _SKIP_DIR_NAMES for p in parts)


def _filter_status_lines(raw_lines):
	kept = []
	omitted = 0
	for line in raw_lines:
		if not str(line).strip():
			continue
		path = _status_path(line)
		if path and _omit_status_path(path):
			omitted += 1
			continue
		kept.append(line.rstrip("\n"))
	return kept, omitted
