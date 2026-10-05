"""Persisted per-message git status: stamp, replay, stay out of UI content."""

import os
import subprocess
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..")
if _ENGINE not in sys.path:
	sys.path.insert(0, _ENGINE)

from lib import context_compact as cc
from lib import git_context as gc


class _E:
	git_status_context = 1
	workspace_dir = ""
	user_name = "user"
	_compact_bypass = False
	compact_at = 0
	compact_summary = ""
	compact_until = 0
	compact_count = 0
	messages = None
	aiui_home = ""


def _bind(tmp_path=None, **kw):
	e = _E()
	e.messages = []
	if tmp_path is not None:
		e.workspace_dir = str(tmp_path)
		e.aiui_home = str(tmp_path)
	for k, v in kw.items():
		setattr(e, k, v)
	gc.bind(e)
	cc.bind(e)
	return e


def _git_repo(tmp_path):
	env = {**os.environ, "GIT_OPTIONAL_LOCKS": "1", "GIT_TERMINAL_PROMPT": "0"}
	subprocess.run(
		["git", "init", "-b", "main"],
		cwd=tmp_path, check=True, capture_output=True, env=env,
	)
	subprocess.run(
		["git", "config", "user.email", "t@t"],
		cwd=tmp_path, check=True, capture_output=True, env=env,
	)
	subprocess.run(
		["git", "config", "user.name", "t"],
		cwd=tmp_path, check=True, capture_output=True, env=env,
	)
	(tmp_path / "dirty.txt").write_text("x\n", encoding="utf-8")


def test_stamp_stores_field_not_content(tmp_path):
	_git_repo(tmp_path)
	_bind(tmp_path)
	msg = {"role": "user", "content": "hello"}
	gc.StampUserGitStatus(msg)
	assert "hello" == msg["content"]
	assert gc.GIT_STATUS_KEY in msg
	assert "dirty.txt" in msg[gc.GIT_STATUS_KEY]
	assert "Working tree (git status)" in msg[gc.GIT_STATUS_KEY]


def test_stamp_skips_clean_tree(tmp_path):
	env = {**os.environ, "GIT_OPTIONAL_LOCKS": "1", "GIT_TERMINAL_PROMPT": "0"}
	subprocess.run(
		["git", "init", "-b", "main"],
		cwd=tmp_path, check=True, capture_output=True, env=env,
	)
	_bind(tmp_path)
	msg = {"role": "user", "content": "hello"}
	gc.StampUserGitStatus(msg)
	assert gc.GIT_STATUS_KEY not in msg


def test_attach_replays_every_stored_user():
	_bind()
	old = {"role": "user", "content": "first", gc.GIT_STATUS_KEY: "GIT-A"}
	asst = {"role": "assistant", "content": "ok"}
	new = {"role": "user", "content": "second", gc.GIT_STATUS_KEY: "GIT-B"}
	out = gc.AttachGitContext([old, asst, new])
	assert out[0]["content"] == "first\n\nGIT-A"
	assert out[1]["content"] == "ok"
	assert out[2]["content"] == "second\n\nGIT-B"
	assert old["content"] == "first"
	assert new["content"] == "second"


def test_attach_off_or_summary_skips():
	e = _bind(git_status_context=0)
	msg = {"role": "user", "content": "hi", gc.GIT_STATUS_KEY: "GIT"}
	assert gc.AttachGitContext([msg])[0]["content"] == "hi"
	e.git_status_context = 1
	e._compact_bypass = True
	assert gc.AttachGitContext([msg])[0]["content"] == "hi"
	e._compact_bypass = False
	summary = {"role": "user", "content": "handoff", "_aiui_summary": True, gc.GIT_STATUS_KEY: "GIT"}
	assert gc.AttachGitContext([summary])[0]["content"] == "handoff"


def test_sanitize_strips_field_after_merge():
	e = _bind(compact_at=0)
	e.messages = [
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "hello", gc.GIT_STATUS_KEY: "GIT-SNAP"},
	]
	from lib import git_context
	git_context.bind(e)
	wire = cc.BuildModelMessages(sanitize=True)
	user = [m for m in wire if m.get("role") == "user"][0]
	assert "GIT-SNAP" in user["content"]
	assert gc.GIT_STATUS_KEY not in user
	assert e.messages[1]["content"] == "hello"
	assert e.messages[1][gc.GIT_STATUS_KEY] == "GIT-SNAP"


def test_stamp_does_not_overwrite():
	_bind()
	msg = {"role": "user", "content": "hi", gc.GIT_STATUS_KEY: "FROZEN"}
	gc.StampUserGitStatus(msg)
	assert msg[gc.GIT_STATUS_KEY] == "FROZEN"
