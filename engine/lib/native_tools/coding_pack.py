"""Native Coding pack - glob / grep / read / write / replace / run command (optional path sandbox)."""

import glob
import os
import re
import subprocess

from lib import fs_sandbox

PACK_ID = "native:coding"
PACK_NAME = "Coding"
PACK_KIND = "native"

_MAX_GREP_HITS = 200
_MAX_GLOB = 500
_MAX_FILE_BYTES = 2_000_000
_MAX_CMD_CHARS = 200_000
_DEFAULT_CMD_TIMEOUT = 60
_MAX_CMD_TIMEOUT = 300

# Opt-in tools: start disabled until the user enables them in pack Tools.
DEFAULT_DISABLED_TOOLS = ["run_command"]

TOOLS = [
	{
		"name": "glob_files",
		"description": (
			"Lists files matching a glob pattern (e.g. '**/*.py' or "
			"'/path/to/dir/**/*.pdf'). Relative patterns use the open project "
			"folder, or ~/.aiui on Home. Absolute patterns are allowed when "
			"Sandbox Paths is disabled."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"pattern": {
					"type": "string",
					"description": "Glob pattern (relative to the workspace root, or absolute when sandbox is off).",
				},
			},
			"required": ["pattern"],
			"additionalProperties": False,
		},
	},
	{
		"name": "grep_files",
		"description": (
			"Searches file contents for a regex pattern. Returns matching lines with "
			"file paths and line numbers. The optional glob may be relative to the "
			"workspace root, or absolute when Sandbox Paths is disabled."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"pattern": {
					"type": "string",
					"description": "Regular expression to search for.",
				},
				"glob": {
					"type": "string",
					"description": "Optional glob to limit which files are scanned (default '**/*'; absolute allowed when sandbox is off).",
					"default": "**/*",
				},
				"case_insensitive": {
					"type": "boolean",
					"description": "If true, match case-insensitively. Optional (default=false).",
					"default": False,
				},
			},
			"required": ["pattern"],
			"additionalProperties": False,
		},
	},
	{
		"name": "read_file",
		"description": (
			"Reads a text file. By default returns the entire file (size-capped). "
			"Optional start_line / end_line are 1-based inclusive; use -1 for the last line. "
			"If only start_line is set, that single line is returned. "
			"Each line is prefixed with N| (line number) for editing."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "File path relative to the workspace root, or absolute when allowed.",
				},
				"start_line": {
					"type": "integer",
					"description": "Optional 1-based start line (-1 = last line).",
				},
				"end_line": {
					"type": "integer",
					"description": "Optional 1-based end line inclusive (-1 = last line).",
				},
			},
			"required": ["path"],
			"additionalProperties": False,
		},
	},
	{
		"name": "write_file",
		"description": (
			"Writes a text file. With no line params, creates or overwrites the whole file "
			"(parent dirs are created as needed). With start_line only (1-based, -1 = before "
			"last line; last+1 = append), inserts content before that line without overwriting. "
			"With both start_line and end_line (inclusive, -1 = last line), replaces that range; "
			"empty content deletes the range. end_line alone is invalid."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "File path relative to the workspace root, or absolute when allowed.",
				},
				"content": {
					"type": "string",
					"description": "Full file body, insert text, or replacement for a line range (empty deletes a range).",
				},
				"start_line": {
					"type": "integer",
					"description": "1-based line: alone = insert before; with end_line = range start (-1 = last line).",
				},
				"end_line": {
					"type": "integer",
					"description": "Optional 1-based end of range to replace inclusive (-1 = last line). Requires start_line.",
				},
			},
			"required": ["path", "content"],
			"additionalProperties": False,
		},
	},
	{
		"name": "search_replace",
		"description": (
			"Replaces an exact string occurrence in one file. Fails if old_string is "
			"not found or is not unique when replace_all is false."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "File path relative to the workspace root, or absolute when allowed.",
				},
				"old_string": {
					"type": "string",
					"description": "Exact text to find.",
				},
				"new_string": {
					"type": "string",
					"description": "Replacement text.",
				},
				"replace_all": {
					"type": "boolean",
					"description": "If true, replace every occurrence. Optional (default=false).",
					"default": False,
				},
			},
			"required": ["path", "old_string", "new_string"],
			"additionalProperties": False,
		},
	},
	{
		"name": "run_command",
		"description": (
			"Runs a shell command in the workspace root (open project, or ~/.aiui on Home). "
			"Use for mkdir, rm, mv, ls, git, tests, builds, and similar. "
			"Returns exit code plus stdout/stderr (size-capped). "
			"Disabled by default — enable it under pack Tools before the model can call it."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"command": {
					"type": "string",
					"description": "Shell command to run (e.g. 'mkdir -p src' or 'ls -la').",
				},
				"timeout": {
					"type": "integer",
					"description": f"Optional timeout in seconds (default {_DEFAULT_CMD_TIMEOUT}, max {_MAX_CMD_TIMEOUT}).",
					"default": _DEFAULT_CMD_TIMEOUT,
				},
			},
			"required": ["command"],
			"additionalProperties": False,
		},
	},
]

def list_tools():
	return list(TOOLS)

_SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__"}

def _rel(root, abs_path):
	try:
		return os.path.relpath(abs_path, root).replace("\\", "/")
	except Exception:
		return abs_path.replace("\\", "/")

def _norm_glob_pattern(pattern: str) -> str:
	raw = (pattern or "").strip() or "**/*"
	return os.path.expanduser(raw)

def _is_abs_or_home(pattern: str) -> bool:
	p = (pattern or "").strip()
	return p.startswith("~") or os.path.isabs(p)

def _literal_glob_prefix(pattern: str) -> str:
	"""Text before the first glob magic character (* ? [)."""
	for i, ch in enumerate(pattern):
		if ch in "*?[":
			return pattern[:i].rstrip("/\\")
	return pattern.rstrip("/\\")

def _prepare_glob(pattern: str):
	"""Return (glob_query, display_absolute, err)."""
	raw = _norm_glob_pattern(pattern)
	root = fs_sandbox.GetFilesRoot()
	display_absolute = _is_abs_or_home((pattern or "").strip()) or os.path.isabs(raw)
	query = raw if os.path.isabs(raw) else os.path.join(root, raw)

	# Early sandbox gate so we do not walk the whole filesystem then filter.
	prefix = _literal_glob_prefix(query) or root
	_, err = fs_sandbox.ResolveInRoot(prefix, must_exist=False, allow_new=True)
	if err:
		return None, False, err
	return query, display_absolute, None

def _path_has_skipped_dir(path: str) -> bool:
	parts = path.replace("\\", "/").split("/")
	return any(p in _SKIP_DIR_NAMES for p in parts)

def _collect_glob(pattern: str):
	"""Return (paths, display_absolute, truncated, err). Paths are absolute."""
	query, display_absolute, err = _prepare_glob(pattern)
	if err:
		return None, False, False, err

	root = fs_sandbox.GetFilesRoot()
	sandbox = fs_sandbox.IsSandboxEnabled()
	matches = []
	truncated = False
	for path in glob.glob(query, recursive=True):
		if not os.path.isfile(path):
			continue
		if _path_has_skipped_dir(path):
			continue
		if sandbox:
			_, deny = fs_sandbox.ResolveInRoot(path, must_exist=False, allow_new=False)
			if deny:
				continue
		matches.append(os.path.abspath(path))
		if len(matches) >= _MAX_GLOB:
			truncated = True
			break
	matches.sort()
	return matches, display_absolute, truncated, None

def _format_glob_paths(paths, display_absolute):
	root = fs_sandbox.GetFilesRoot()
	if display_absolute:
		return [p.replace("\\", "/") for p in paths]
	return [_rel(root, p) for p in paths]

def _glob_files(pattern: str) -> str:
	paths, display_absolute, truncated, err = _collect_glob(pattern)
	if err:
		return err
	if not paths:
		return "(no matches)"
	out = "\n".join(_format_glob_paths(paths, display_absolute))
	if truncated:
		out += f"\n... truncated at {_MAX_GLOB} matches"
	return out

def _grep_files(pattern: str, glob_pat: str = "**/*", case_insensitive: bool = False) -> str:
	paths, display_absolute, _, err = _collect_glob(glob_pat or "**/*")
	if err:
		return err
	try:
		flags = re.MULTILINE
		if case_insensitive:
			flags |= re.IGNORECASE
		rx = re.compile(pattern, flags)
	except re.error as e:
		return f"ERROR: invalid regex ({e})"

	root = fs_sandbox.GetFilesRoot()
	hits = []
	for full in paths:
		display = full.replace("\\", "/") if display_absolute else _rel(root, full)
		try:
			if os.path.getsize(full) > _MAX_FILE_BYTES:
				continue
			with open(full, "r", encoding="utf-8", errors="replace") as f:
				for i, line in enumerate(f, 1):
					if rx.search(line):
						hits.append(f"{display}:{i}:{line.rstrip()}")
						if len(hits) >= _MAX_GREP_HITS:
							return "\n".join(hits) + f"\n... truncated at {_MAX_GREP_HITS} hits"
		except Exception:
			continue
	return "\n".join(hits) if hits else "(no matches)"

def _parse_line_arg(raw, label):
	if raw is None or raw == "":
		return None, None
	try:
		n = int(raw)
	except (TypeError, ValueError):
		return None, f"ERROR: {label} must be an integer"
	if n == 0 or n < -1:
		return None, f"ERROR: {label} must be >= 1 or -1 (last line)"
	return n, None

def _resolve_line(n, nlines):
	"""Map 1-based / -1 to 0-based index. Returns (idx, err)."""
	if nlines <= 0:
		return None, "ERROR: file has no lines"
	if n == -1:
		return nlines - 1, None
	if n < 1 or n > nlines:
		return None, f"ERROR: line {n} out of range (file has {nlines} line(s))"
	return n - 1, None

def _resolve_insert_line(n, nlines):
	"""Map 1-based / -1 to 0-based insert index (before that line). Allows nlines+1 = append."""
	if n == -1:
		return max(nlines - 1, 0), None
	if nlines <= 0:
		if n == 1:
			return 0, None
		return None, "ERROR: line 1 is the only valid insert position in an empty file"
	if n < 1 or n > nlines + 1:
		return None, f"ERROR: line {n} out of range (insert allows 1..{nlines + 1}, or -1)"
	return n - 1, None

def _detect_newline(text):
	if "\r\n" in text:
		return "\r\n"
	if "\r" in text:
		return "\r"
	return "\n"

def _load_text_file(abs_path, *, allow_empty=False):
	try:
		size = os.path.getsize(abs_path)
	except Exception as e:
		return None, f"ERROR: failed to stat file ({e})"
	if size > _MAX_FILE_BYTES:
		return None, f"ERROR: file exceeds {_MAX_FILE_BYTES} byte limit"
	try:
		with open(abs_path, "rb") as f:
			raw = f.read()
	except Exception as e:
		return None, f"ERROR: failed to read file ({e})"
	if b"\x00" in raw:
		return None, "ERROR: binary file (NUL byte detected)"
	try:
		text = raw.decode("utf-8")
	except UnicodeDecodeError:
		text = raw.decode("utf-8", errors="replace")
	if not allow_empty and text == "" and size == 0:
		return {"text": "", "lines": [], "newline": "\n"}, None
	newline = _detect_newline(text)
	# Keep line endings out of line list; splitlines() drops endings.
	lines = text.splitlines()
	# Preserve a final empty line if the file ended with a newline.
	if text.endswith("\n") or text.endswith("\r"):
		# splitlines already yields no trailing empty for "a\n"; for "a\n\n" yields ['a', ''].
		# Files that are only "\n" → [''].
		pass
	return {"text": text, "lines": lines, "newline": newline}, None

def _format_numbered(lines, start_idx):
	"""lines is a list of line strings; start_idx is 0-based index of first line."""
	out = []
	for i, line in enumerate(lines):
		out.append(f"{start_idx + i + 1}|{line}")
	return "\n".join(out) if out else "(empty)"

def _read_file(path: str, start_line=None, end_line=None) -> str:
	abs_path, err = fs_sandbox.ResolveInRoot(path, must_exist=True, allow_new=False)
	if err:
		return err
	if not os.path.isfile(abs_path):
		return "ERROR: path is not a file"
	loaded, err = _load_text_file(abs_path)
	if err:
		return err
	lines = loaded["lines"]
	nlines = len(lines)

	start_raw, err = _parse_line_arg(start_line, "start_line")
	if err:
		return err
	end_raw, err = _parse_line_arg(end_line, "end_line")
	if err:
		return err

	if start_raw is None and end_raw is None:
		return _format_numbered(lines, 0)

	if start_raw is None and end_raw is not None:
		return "ERROR: start_line is required when end_line is set"

	# Only start_line - single line
	if start_raw is not None and end_raw is None:
		idx, err = _resolve_line(start_raw, nlines)
		if err:
			return err
		return _format_numbered([lines[idx]], idx)

	s_idx, err = _resolve_line(start_raw, nlines)
	if err:
		return err
	e_idx, err = _resolve_line(end_raw, nlines)
	if err:
		return err
	if s_idx > e_idx:
		return "ERROR: start_line is after end_line"
	return _format_numbered(lines[s_idx:e_idx + 1], s_idx)

def _write_file(path: str, content, start_line=None, end_line=None) -> str:
	if content is None:
		return "ERROR: content is required"
	content = str(content)

	start_raw, err = _parse_line_arg(start_line, "start_line")
	if err:
		return err
	end_raw, err = _parse_line_arg(end_line, "end_line")
	if err:
		return err
	if end_raw is not None and start_raw is None:
		return "ERROR: start_line is required when end_line is set"

	root = fs_sandbox.GetFilesRoot()

	# Full create / overwrite
	if start_raw is None:
		abs_path, err = fs_sandbox.ResolveInRoot(path, must_exist=False, allow_new=True)
		if err:
			return err
		if abs_path and os.path.isdir(abs_path):
			return "ERROR: path is a directory"
		if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
			return f"ERROR: content exceeds {_MAX_FILE_BYTES} byte limit"
		try:
			parent = os.path.dirname(abs_path)
			if parent:
				os.makedirs(parent, exist_ok=True)
			with open(abs_path, "w", encoding="utf-8", newline="") as f:
				f.write(content)
		except Exception as e:
			return f"ERROR: failed to write file ({e})"
		return f"OK: wrote {_rel(root, abs_path)} ({len(content)} chars)"

	# Insert or range replace - file must exist
	abs_path, err = fs_sandbox.ResolveInRoot(path, must_exist=True, allow_new=False)
	if err:
		return err
	if not os.path.isfile(abs_path):
		return "ERROR: path is not a file"
	loaded, err = _load_text_file(abs_path)
	if err:
		return err
	lines = list(loaded["lines"])
	nlines = len(lines)
	newline = loaded["newline"]

	# start_line only - insert before that line (no overwrite)
	if end_raw is None:
		if content == "":
			return "ERROR: content is required for insert (use both line params with empty content to delete)"
		ins_idx, err = _resolve_insert_line(start_raw, nlines)
		if err:
			return err
		repl = content.splitlines()
		new_lines = lines[:ins_idx] + repl + lines[ins_idx:]
		if nlines > 0 and ins_idx >= nlines:
			action = f"appended {len(repl)} line(s)"
		else:
			action = f"inserted {len(repl)} line(s) before line {ins_idx + 1}"
	else:
		s_idx, err = _resolve_line(start_raw, nlines)
		if err:
			return err
		e_idx, err = _resolve_line(end_raw, nlines)
		if err:
			return err
		if s_idx > e_idx:
			return "ERROR: start_line is after end_line"

		if content == "":
			new_lines = lines[:s_idx] + lines[e_idx + 1:]
			action = f"deleted lines {s_idx + 1}-{e_idx + 1}"
		else:
			repl = content.splitlines()
			new_lines = lines[:s_idx] + repl + lines[e_idx + 1:]
			action = f"replaced lines {s_idx + 1}-{e_idx + 1} with {len(repl)} line(s)"

	new_text = newline.join(new_lines)
	# If original ended with a newline, keep a trailing newline for non-empty files.
	if loaded["text"].endswith(("\n", "\r")) and new_lines:
		new_text += newline
	elif not new_lines:
		new_text = ""

	if len(new_text.encode("utf-8")) > _MAX_FILE_BYTES:
		return f"ERROR: resulting file would exceed {_MAX_FILE_BYTES} byte limit"
	try:
		with open(abs_path, "w", encoding="utf-8", newline="") as f:
			f.write(new_text)
	except Exception as e:
		return f"ERROR: failed to write file ({e})"
	return f"OK: {action} in {_rel(root, abs_path)}"

def _search_replace(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
	abs_path, err = fs_sandbox.ResolveInRoot(path, must_exist=True, allow_new=False)
	if err:
		return err
	if not os.path.isfile(abs_path):
		return "ERROR: path is not a file"
	try:
		with open(abs_path, "r", encoding="utf-8") as f:
			text = f.read()
	except Exception as e:
		return f"ERROR: failed to read file ({e})"
	count = text.count(old_string)
	if count == 0:
		return "ERROR: old_string not found"
	if not replace_all and count > 1:
		return f"ERROR: old_string found {count} times; set replace_all=true or provide a unique string"
	if replace_all:
		new_text = text.replace(old_string, new_string)
		n = count
	else:
		new_text = text.replace(old_string, new_string, 1)
		n = 1
	try:
		with open(abs_path, "w", encoding="utf-8") as f:
			f.write(new_text)
	except Exception as e:
		return f"ERROR: failed to write file ({e})"
	return f"OK: replaced {n} occurrence(s) in {_rel(fs_sandbox.GetFilesRoot(), abs_path)}"

def _run_command(command: str, timeout=None) -> str:
	cmd = str(command or "").strip()
	if not cmd:
		return "ERROR: command is empty"
	try:
		sec = float(timeout) if timeout is not None and timeout != "" else _DEFAULT_CMD_TIMEOUT
	except (TypeError, ValueError):
		return "ERROR: timeout must be a number"
	if sec <= 0:
		return "ERROR: timeout must be positive"
	sec = min(sec, float(_MAX_CMD_TIMEOUT))

	root = fs_sandbox.GetFilesRoot()
	try:
		proc = subprocess.run(
			cmd,
			shell=True,
			cwd=root,
			capture_output=True,
			text=True,
			timeout=sec,
			errors="replace",
		)
	except subprocess.TimeoutExpired as e:
		out = (e.stdout or "") + (e.stderr or "")
		out = out[:_MAX_CMD_CHARS]
		return f"ERROR: command timed out after {int(sec)}s\n{out}".rstrip()
	except Exception as e:
		return f"ERROR: failed to run command ({e})"

	parts = [f"exit_code: {proc.returncode}", f"cwd: {root}"]
	stdout = proc.stdout or ""
	stderr = proc.stderr or ""
	if len(stdout) > _MAX_CMD_CHARS:
		stdout = stdout[:_MAX_CMD_CHARS] + f"\n... truncated at {_MAX_CMD_CHARS} chars"
	if len(stderr) > _MAX_CMD_CHARS:
		stderr = stderr[:_MAX_CMD_CHARS] + f"\n... truncated at {_MAX_CMD_CHARS} chars"
	if stdout:
		parts.append("--- stdout ---\n" + stdout.rstrip("\n"))
	if stderr:
		parts.append("--- stderr ---\n" + stderr.rstrip("\n"))
	if not stdout and not stderr:
		parts.append("(no output)")
	return "\n".join(parts)

def call_tool(name, arguments=None):
	arguments = arguments or {}
	try:
		if name == "glob_files":
			return _glob_files(arguments.get("pattern", ""))
		if name == "grep_files":
			return _grep_files(
				arguments.get("pattern", ""),
				arguments.get("glob", "**/*"),
				bool(arguments.get("case_insensitive", False)),
			)
		if name == "read_file":
			return _read_file(
				arguments.get("path", ""),
				arguments.get("start_line"),
				arguments.get("end_line"),
			)
		if name == "write_file":
			return _write_file(
				arguments.get("path", ""),
				arguments.get("content"),
				arguments.get("start_line"),
				arguments.get("end_line"),
			)
		if name == "search_replace":
			return _search_replace(
				arguments.get("path", ""),
				arguments.get("old_string", ""),
				arguments.get("new_string", ""),
				bool(arguments.get("replace_all", False)),
			)
		if name == "run_command":
			return _run_command(
				arguments.get("command", ""),
				arguments.get("timeout"),
			)
		return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"
