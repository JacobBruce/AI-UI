"""Ephemeral git working-tree snapshot for the model wire (not stored in chat)."""

from __future__ import annotations

import os
import subprocess

E = None

_SKIP_DIR_NAMES = {".git", "node_modules", "vendor", ".venv", "__pycache__"}
_MAX_STATUS_LINES = 60
_GIT_TIMEOUT = 3


def bind(engine_module):
	global E
	E = engine_module


def ClearGitContext():
	if E is None:
		return
	E._git_context_block = None
	E._git_context_frozen = False


def EnsureGitContext(*, freeze=None):
	"""Keep one snapshot per user turn; refresh when idle / counting outside a turn.

	freeze=True  — capture now and reuse until freeze=False (bot turn).
	freeze=False — allow the next Ensure to recapture (turn end).
	omit freeze  — capture unless a turn snapshot is frozen.
	"""
	if E is None:
		return
	if freeze is True:
		E._git_context_frozen = True
		CaptureGitContext()
		return
	if freeze is False:
		E._git_context_frozen = False
		return
	if getattr(E, "_git_context_frozen", False):
		return
	CaptureGitContext()


def CaptureGitContext():
	"""Refresh porcelain status. No-op when disabled / not a repo / clean tree."""
	if E is None:
		return
	E._git_context_block = None
	if not int(getattr(E, "git_status_context", 0) or 0):
		return
	root = str(getattr(E, "workspace_dir", "") or "").strip()
	if not root or not os.path.isdir(root):
		return
	if not _git_is_work_tree(root):
		return
	branch = _git_text(root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "HEAD"
	raw = _git_text(root, ["status", "--porcelain", "--untracked-files=normal"])
	if raw is None:
		return
	lines, omitted = _filter_status_lines(raw.splitlines())
	if not lines and not omitted:
		return
	out = ["# Working tree (git status, this turn only)", f"branch: {branch}"]
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
	E._git_context_block = "\n".join(out)


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
	"""Copy-on-write: append the snapshot to the last user message on the wire.

	Keeps the system prompt (and earlier history) stable for prefix caching.
	No user message yet (empty chat / usage ring) → skip; do not put it on system.
	"""
	if E is None:
		return messages
	# Summarizer side-channel uses a synthetic transcript; do not inject there.
	if getattr(E, "_compact_bypass", False):
		return messages
	EnsureGitContext()
	block = getattr(E, "_git_context_block", None)
	if not block:
		return messages
	raw = list(messages or [])
	idx = None
	for i in range(len(raw) - 1, -1, -1):
		m = raw[i]
		if isinstance(m, dict) and _IsUserMsg(m):
			idx = i
			break
	if idx is None:
		return messages
	out = []
	for i, m in enumerate(raw):
		if i != idx or not isinstance(m, dict):
			out.append(m)
			continue
		msg = dict(m)
		msg["content"] = _AppendGitBlock(msg.get("content"), block)
		out.append(msg)
	return out


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
