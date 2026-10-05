"""write_file prev_line / next_line neighbor checks."""

import importlib.util
import os
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..")
if _ENGINE not in sys.path:
	sys.path.insert(0, _ENGINE)

from lib import fs_sandbox


def _load_pack():
	path = os.path.join(_ENGINE, "lib", "native_tools", "coding_pack.py")
	spec = importlib.util.spec_from_file_location("coding_pack_write_test", path)
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


def _src(tmp_path):
	p = tmp_path / "src.py"
	p.write_text("def foo():\n    a = 1\n    return a\n", encoding="utf-8")
	return "src.py"


def test_insert_with_neighbors(tmp_path):
	_bind(tmp_path)
	_src(tmp_path)
	out = pack._write_file(
		"src.py",
		"    b = 2\n",
		start_line=3,
		prev_line="    a = 1",
		next_line="    return a",
	)
	assert out.startswith("OK:")
	text = (tmp_path / "src.py").read_text(encoding="utf-8")
	assert "    b = 2\n    return a\n" in text


def test_insert_mismatch_does_not_write(tmp_path):
	_bind(tmp_path)
	_src(tmp_path)
	before = (tmp_path / "src.py").read_text(encoding="utf-8")
	out = pack._write_file(
		"src.py",
		"    b = 2\n",
		start_line=3,
		prev_line="    a = 1",
		next_line="    return b",
	)
	assert out.startswith("ERROR: next_line mismatch")
	assert (tmp_path / "src.py").read_text(encoding="utf-8") == before


def test_replace_range_neighbors(tmp_path):
	_bind(tmp_path)
	_src(tmp_path)
	out = pack._write_file(
		"src.py",
		"    return 1\n",
		start_line=2,
		end_line=3,
		prev_line="def foo():",
	)
	assert out.startswith("OK:")
	assert (tmp_path / "src.py").read_text(encoding="utf-8") == "def foo():\n    return 1\n"


def test_anchors_require_start_line(tmp_path):
	_bind(tmp_path)
	_src(tmp_path)
	out = pack._write_file("src.py", "x\n", prev_line="def foo():")
	assert "require start_line" in out


def test_schema_has_anchor_props():
	tools = {t["name"]: t for t in pack.list_tools()}
	props = tools["write_file"]["inputSchema"]["properties"]
	assert "prev_line" in props
	assert "next_line" in props
	assert "prev_line" in tools["write_file"]["description"]


def test_anchor_rejects_non_string(tmp_path):
	_bind(tmp_path)
	_src(tmp_path)
	before = (tmp_path / "src.py").read_text(encoding="utf-8")
	out = pack._write_file("src.py", "    b = 2\n", start_line=3, prev_line=True)
	assert "prev_line must be a string" in out
	assert (tmp_path / "src.py").read_text(encoding="utf-8") == before
	out = pack._write_file("src.py", True)
	assert "content must be a string" in out
	assert (tmp_path / "src.py").read_text(encoding="utf-8") == before


def test_read_last_line(tmp_path):
	_bind(tmp_path)
	(tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
	out = pack._read_file("notes.txt", start_line=-1)
	assert out == "3|three"
	out = pack._read_file("notes.txt", start_line=-1, end_line=-1)
	assert out == "3|three"


def test_read_last_line_huge_file_tails(tmp_path, monkeypatch):
	_bind(tmp_path)
	p = tmp_path / "big.txt"
	p.write_text("head\nLAST\n", encoding="utf-8")
	monkeypatch.setattr(pack, "_MAX_FILE_BYTES", 4)
	out = pack._read_file("big.txt", start_line=-1)
	assert out.endswith("|LAST")
	assert out.startswith("-1|")
