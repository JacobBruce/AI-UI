"""glob_files / grep_files: ripgrep when present, Python fallback, opt-in exclude."""

import importlib.util
import os
import sys

import pytest

_ENGINE = os.path.join(os.path.dirname(__file__), "..")
if _ENGINE not in sys.path:
	sys.path.insert(0, _ENGINE)

from lib import fs_sandbox


def _load_pack():
	path = os.path.join(_ENGINE, "lib", "native_tools", "coding_pack.py")
	spec = importlib.util.spec_from_file_location("coding_pack_grep_test", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


pack = _load_pack()


class _E:
	workspace_dir = ""
	aiui_home = ""
	tool_packs = [{"id": "native:coding", "sandbox": True}]


def _bind(tmp_path):
	e = _E()
	e.workspace_dir = str(tmp_path)
	e.aiui_home = str(tmp_path)
	fs_sandbox.bind(e)
	return e


def _tree(tmp_path):
	(tmp_path / "keep.py").write_text("alpha_token = 1\n", encoding="utf-8")
	(tmp_path / "ignored.py").write_text("alpha_token = 2\n", encoding="utf-8")
	(tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
	vend = tmp_path / "vendor" / "lib"
	vend.mkdir(parents=True)
	(vend / "skip.py").write_text("alpha_token = 3\n", encoding="utf-8")
	src = tmp_path / "src"
	src.mkdir()
	(src / "nested.py").write_text("alpha_token = 4\n", encoding="utf-8")


def test_find_rg_prefers_system_binary():
	rg = pack._find_rg()
	if rg:
		assert "@vscode/ripgrep" not in rg.replace("\\", "/")


def test_list_tools_points_at_status():
	tools = {t["name"]: t for t in pack.list_tools()}
	for name in ("grep_files", "glob_files"):
		desc = tools[name]["description"]
		assert "aiui_status" in desc
		assert "regex.engine" in desc or "regex.glob" in desc
		props = tools[name]["inputSchema"]["properties"]
		assert "exclude" in props
		assert "no_ignore" not in props
	pat = tools["grep_files"]["inputSchema"]["properties"]["pattern"]["description"]
	assert "aiui_status" in pat
	sr = tools["search_replace"]["description"]
	assert "not a regex" in sr.lower()


def test_regex_engine_info_matches_find_rg():
	info = pack.RegexEngineInfo()
	rg = pack._find_rg()
	assert "default_exclude" in info
	assert info["default_exclude"] == pack._SKIP_DIR_NAMES
	if rg:
		assert info["engine"] == "ripgrep"
		assert info["syntax"] == "rust"
		assert info["glob"] == "gitignore"
		assert info["ignore"] == "gitignore"
		assert info["binary"] == rg
	else:
		assert info["engine"] == "python"
		assert info["syntax"] == "python-re"
		assert info["glob"] == "stdlib"
		assert info["ignore"] == "default_exclude"


def test_rg_grep_respects_gitignore_by_default(tmp_path):
	if not pack._find_rg():
		pytest.skip("ripgrep not installed")
	_bind(tmp_path)
	_tree(tmp_path)
	out = pack._grep_files("alpha_token")
	assert "keep.py" in out
	assert "nested.py" in out
	assert "ignored.py" not in out
	assert "skip.py" in out
	narrow = pack._grep_files("alpha_token", glob_pat="**/*.py")
	assert "keep.py" in narrow
	assert "nested.py" in narrow
	assert "ignored.py" not in narrow
	assert "skip.py" in narrow


def test_rg_glob_respects_gitignore_by_default(tmp_path):
	if not pack._find_rg():
		pytest.skip("ripgrep not installed")
	_bind(tmp_path)
	_tree(tmp_path)
	out = pack._glob_files("**/*.py")
	assert "keep.py" in out
	assert "nested.py" in out
	assert "ignored.py" not in out
	assert "skip.py" in out


def test_exclude_skips_named_dirs(tmp_path):
	_bind(tmp_path)
	_tree(tmp_path)
	grep = pack._grep_files("alpha_token", exclude=["vendor"])
	assert "keep.py" in grep
	assert "nested.py" in grep
	assert "skip.py" not in grep
	listed = pack._glob_files("**/*.py", exclude=["vendor"])
	assert "keep.py" in listed
	assert "nested.py" in listed
	assert "skip.py" not in listed
	# Literal path component in the glob keeps that directory.
	kept = pack._glob_files("vendor/**", exclude=["vendor"])
	assert "skip.py" in kept
	grep_kept = pack._grep_files("alpha_token", glob_pat="vendor/**", exclude=["vendor"])
	assert "skip.py" in grep_kept


def test_exclude_skips_gitignore(tmp_path):
	if not pack._find_rg():
		pytest.skip("ripgrep not installed")
	_bind(tmp_path)
	_tree(tmp_path)
	assert "ignored.py" not in pack._grep_files("alpha_token")
	grep = pack._grep_files("alpha_token", exclude=[])
	assert "ignored.py" in grep
	assert "keep.py" in grep
	listed = pack._glob_files("**/*.py", exclude=[])
	assert "ignored.py" in listed
	grep = pack._grep_files("alpha_token", exclude=["vendor"])
	assert "ignored.py" in grep
	assert "skip.py" not in grep


def test_python_fallback_skips_default_exclude(tmp_path, monkeypatch):
	_bind(tmp_path)
	_tree(tmp_path)
	monkeypatch.setattr(pack, "_find_rg", lambda: None)
	grep = pack._grep_files("alpha_token")
	assert "keep.py" in grep
	assert "ignored.py" in grep
	assert "skip.py" not in grep
	listed = pack._glob_files("**/*.py")
	assert "keep.py" in listed
	assert "ignored.py" in listed
	assert "skip.py" not in listed
	opened = pack._glob_files("**/*.py", exclude=[])
	assert "skip.py" in opened
	assert "ignored.py" in opened
	skipped = pack._glob_files("**/*.py", exclude=["vendor"])
	assert "skip.py" not in skipped
	assert "ignored.py" in skipped
	grep_skip = pack._grep_files("alpha_token", exclude=["vendor"])
	assert "skip.py" not in grep_skip
	assert "ignored.py" in grep_skip


def test_rg_start_failure_does_not_python_fallback(tmp_path, monkeypatch):
	_bind(tmp_path)
	_tree(tmp_path)
	monkeypatch.setattr(pack, "_find_rg", lambda: "/usr/bin/rg")
	monkeypatch.setattr(pack, "_glob_files_rg", lambda *a, **k: "ERROR: ripgrep failed to start")
	monkeypatch.setattr(pack, "_grep_files_rg", lambda *a, **k: "ERROR: ripgrep failed to start")
	listed = pack._glob_files("**/*.py")
	assert listed.startswith("ERROR: ripgrep")
	grep = pack._grep_files("alpha_token")
	assert grep.startswith("ERROR: ripgrep")


def test_call_tool_passes_exclude(tmp_path):
	_bind(tmp_path)
	_tree(tmp_path)
	listed = pack.call_tool("glob_files", {
		"pattern": "**/*.py",
		"exclude": ["vendor"],
	})
	assert "skip.py" not in listed
	assert "keep.py" in listed
	grep = pack.call_tool("grep_files", {
		"pattern": "alpha_token",
		"exclude": ["vendor"],
	})
	assert "skip.py" not in grep
	assert "keep.py" in grep
	if pack._find_rg():
		assert "ignored.py" in grep
		open_listed = pack.call_tool("glob_files", {
			"pattern": "**/*.py",
			"exclude": [],
		})
		assert "ignored.py" in open_listed


def test_exclude_rejects_wrong_types(tmp_path):
	_bind(tmp_path)
	_tree(tmp_path)
	for bad in (True, 1, {"vendor": True}, "{}"):
		out = pack._glob_files("**/*.py", exclude=bad)
		assert out.startswith("ERROR: exclude"), bad
		out = pack._grep_files("alpha_token", exclude=bad)
		assert out.startswith("ERROR: exclude"), bad
	listed = pack._glob_files("**/*.py", exclude="vendor")
	assert "skip.py" not in listed
	assert "keep.py" in listed


def test_exclude_accepts_json_array_string(tmp_path):
	_bind(tmp_path)
	_tree(tmp_path)
	listed = pack._glob_files("**/*.py", exclude='["vendor"]')
	assert "skip.py" not in listed
	assert "keep.py" in listed
	listed = pack.call_tool("glob_files", {
		"pattern": "**/*.py",
		"exclude": '["vendor"]',
	})
	assert "skip.py" not in listed
	assert "keep.py" in listed


def test_invalid_regex(tmp_path):
	_bind(tmp_path)
	(tmp_path / "a.py").write_text("x\n", encoding="utf-8")
	out = pack._grep_files("(")
	assert out.startswith("ERROR: invalid regex")


def test_one_level_glob_does_not_match_nested(tmp_path):
	_bind(tmp_path)
	src = tmp_path / "src"
	src.mkdir()
	(src / "top.py").write_text("x\n", encoding="utf-8")
	nested = src / "a"
	nested.mkdir()
	(nested / "deep.py").write_text("x\n", encoding="utf-8")
	listed = pack._glob_files("src/*.py")
	assert "src/top.py" in listed.replace("\\", "/")
	assert "deep.py" not in listed
	grep = pack._grep_files("x", glob_pat="src/*.py")
	assert "src/top.py" in grep.replace("\\", "/")
	assert "deep.py" not in grep
	if pack._find_rg():
		py = pack._glob_files_python("src/*.py")
		assert "deep.py" not in py
		assert "src/top.py" in py.replace("\\", "/")


def test_brace_dir_glob(tmp_path):
	_bind(tmp_path)
	(tmp_path / "src").mkdir()
	(tmp_path / "lib").mkdir()
	(tmp_path / "src" / "a.py").write_text("x\n", encoding="utf-8")
	(tmp_path / "lib" / "b.py").write_text("x\n", encoding="utf-8")
	(tmp_path / "other.py").write_text("x\n", encoding="utf-8")
	listed = pack._glob_files("{src,lib}/**/*.py")
	assert "src/a.py" in listed.replace("\\", "/")
	assert "lib/b.py" in listed.replace("\\", "/")
	assert "other.py" not in listed


def test_hidden_yml_glob(tmp_path):
	_bind(tmp_path)
	wf = tmp_path / ".github" / "workflows"
	wf.mkdir(parents=True)
	(wf / "ci.yml").write_text("on: push\n", encoding="utf-8")
	listed = pack._glob_files("**/*.yml")
	assert "ci.yml" in listed
