"""Native Coding pack - glob / grep / read / write / replace / run command (optional path sandbox)."""

import fnmatch
import glob as globmod
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import PurePosixPath

from lib import fs_sandbox

PACK_ID = "native:coding"
PACK_NAME = "Coding"
PACK_KIND = "native"

_MAX_GREP_HITS = 100
_MAX_GREP_BYTES = 32_000
_MAX_GREP_LINE = 500
_MAX_GREP_FILES = 2000
_DEFAULT_GLOB = 100
_HARD_MAX_GLOB = 2000
_MAX_FILE_BYTES = 2_000_000
_MAX_CMD_CHARS = 200_000
_DEFAULT_CMD_TIMEOUT = 60
_MAX_CMD_TIMEOUT = 300
_RG_TIMEOUT = 60
_RG_BIN_CANDIDATES = ("/usr/bin/rg", "/bin/rg", "/usr/local/bin/rg")
_SKIP_DIR_NAMES = [
	"__pycache__", ".mypy_cache", ".pytest_cache",
	".git", ".venv", "venv", "node_modules", "vendor",
	"bin", "build", "dist", "target",
]

_BINARY_EXTS = frozenset({
	".o", ".obj", ".a", ".lib", ".so", ".dylib", ".dll", ".out",
	".AppImage", ".deb", ".rpm", ".msi", ".dmg", ".pkg", ".mpkg", ".app", ".exe",
	".pt", ".cpk", ".ckpt", ".pth", ".pkl", ".pickle", ".safetensors",
	".pdb", ".map", ".bin", ".dat", ".pcm", ".mo", ".pak",
	".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tif", ".tiff", ".avif",
	".woff", ".woff2", ".ttf", ".otf", ".eot",
	".pdf", ".zip", ".7z", ".gz", ".bz", ".bz2", ".xz", ".zst", ".tar", ".rar",
	".wasm", ".npz", ".pyc", ".pyo", ".class", ".jar", ".arc",
	".mp3", ".wma", ".weba", ".wav", ".ogg", ".flac", ".aac", ".m4a",
	".mp4", ".wmv", ".webm", ".avi", ".flv", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v",
})

_BRACE_RE = re.compile(r"\{([^{}]+)\}")

_EXCLUDE_PROP = {
	"type": "array",
	"items": {"type": "string"},
	"description": (
		"Directory names to skip (e.g. ['node_modules', 'vendor']). "
		"Passing this list also skips .gitignore (use [] to search ignored "
		"files without extra skips). A name that is a literal path component "
		"in the glob is kept (e.g. glob 'vendor/**' still searches vendor)."
	),
}

# Opt-in tools: start disabled until the user enables them in pack Tools.
DEFAULT_DISABLED_TOOLS = ["run_command"]

TOOLS = [
	{
		"name": "glob_files",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Lists files matching a glob pattern (e.g. '**/*.py' or "
			"'/path/to/dir/**/*.pdf'). Relative patterns use the open project "
			"folder, or ~/.aiui on Home. With Sandbox Paths on, absolute paths may "
			"also target ~/.aiui/skills, ~/.aiui/rules, and ~/.aiui/agents; other "
			"absolute paths need Sandbox Paths off. Brace globs like '**/*.{cpp,h}' are expanded. "
			"Respects .gitignore when using ripgrep unless exclude is set. "
			"The Python fallback skips regex.default_exclude (see aiui_status). "
			"Capped at max_results (default 100). "
			"Call aiui_status and read regex.engine / regex.glob for the live glob flavor."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"pattern": {
					"type": "string",
					"description": (
						"Glob pattern (relative to the workspace root, or absolute when allowed). "
						"Flavor depends on the engine — see aiui_status regex.glob."
					),
				},
				"max_results": {
					"type": "integer",
					"description": f"Maximum paths to return (default {_DEFAULT_GLOB}, max {_HARD_MAX_GLOB}).",
					"default": _DEFAULT_GLOB,
				},
				"exclude": _EXCLUDE_PROP,
			},
			"required": ["pattern"],
			"additionalProperties": False,
		},
	},
	{
		"name": "grep_files",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Searches file contents for a regex. Returns matching lines as path:line:text. "
			"Skips binaries and caps output. Brace globs like '**/*.{cpp,h}' are expanded. "
			"Respects .gitignore when using ripgrep unless exclude is set. "
			"The Python fallback skips regex.default_exclude (see aiui_status). "
			"Call aiui_status and read regex.engine for the live regex engine and glob flavor."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"pattern": {
					"type": "string",
					"description": "Regex. Call aiui_status (regex.engine) if you need the dialect (ripgrep/Rust vs Python re).",
				},
				"glob": {
					"type": "string",
					"description": (
						"Optional glob to limit which files are scanned (default '**/*'; "
						"absolute allowed under the sandbox). Supports {cpp,h} braces. "
						"Glob flavor depends on the engine — see aiui_status regex.engine / regex.glob."
					),
					"default": "**/*",
				},
				"case_insensitive": {
					"type": "boolean",
					"description": "If true, match case-insensitively. Optional (default=false).",
					"default": False,
				},
				"exclude": _EXCLUDE_PROP,
			},
			"required": ["pattern"],
			"additionalProperties": False,
		},
	},
	{
		"name": "read_file",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Reads a text file without loading it all into memory. "
			"By default returns the file up to a size cap (says when truncated). "
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
		"annotations": {"readOnlyHint": False},
		"description": (
			"Writes a text file. With no line params, creates or overwrites the whole file "
			"(parent dirs are created as needed). With start_line only (1-based, -1 = before "
			"last line; last+1 = append), inserts content before that line without overwriting. "
			"With both start_line and end_line (inclusive, -1 = last line), replaces that range; "
			"empty content deletes the range. end_line alone is invalid. "
			"For line inserts/replaces, pass prev_line and/or next_line (exact text of the line "
			"before and after the edit, without the N| prefix from read_file). The write is "
			"refused if those lines do not match. Prefer both when editing source."
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
				"prev_line": {
					"type": "string",
					"description": (
						"Exact text of the line immediately before the insert/replace "
						"(no N| prefix). Optional; recommended for source edits. Requires start_line."
					),
				},
				"next_line": {
					"type": "string",
					"description": (
						"Exact text of the line immediately after the insert point or replace range "
						"(no N| prefix). Optional; recommended for source edits. Requires start_line."
					),
				},
			},
			"required": ["path", "content"],
			"additionalProperties": False,
		},
	},
	{
		"name": "search_replace",
		"annotations": {"readOnlyHint": False},
		"description": (
			"Replaces an exact substring in one file (not a regex). Fails if old_string is "
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
					"description": "Exact text to find (literal substring, not a regex).",
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
		"annotations": {"readOnlyHint": False},
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

def _find_rg():
	"""Prefer a system ripgrep over a Cursor/VS Code bundled binary on PATH."""
	if os.name != "nt":
		for path in _RG_BIN_CANDIDATES:
			if os.path.isfile(path) and os.access(path, os.X_OK):
				return path
	found = shutil.which("rg") or (shutil.which("rg.exe") if os.name == "nt" else None)
	if found and "@vscode/ripgrep" in found.replace("\\", "/"):
		return None
	return found


def RegexEngineInfo():
	"""Live glob_files / grep_files engine for aiui_status."""
	info = {"default_exclude": list(_SKIP_DIR_NAMES)}
	rg = _find_rg()
	if rg:
		info.update({
			"engine": "ripgrep",
			"syntax": "rust",
			"glob": "gitignore",
			"ignore": "gitignore",
			"binary": rg,
		})
		return info
	info.update({
		"engine": "python",
		"syntax": "python-re",
		"glob": "stdlib",
		"ignore": "default_exclude",
	})
	return info


def list_tools():
	return list(TOOLS)

def _rel(root, abs_path):
	try:
		return os.path.relpath(abs_path, root).replace("\\", "/")
	except Exception:
		return abs_path.replace("\\", "/")

def _expand_braces(pattern: str):
	"""Expand bash-style {a,b} groups (comma required). Nested groups are expanded iteratively."""
	if not pattern or "{" not in pattern:
		return [pattern]
	out = [pattern]
	changed = True
	while changed:
		changed = False
		nxt = []
		for pat in out:
			m = _BRACE_RE.search(pat)
			if not m or "," not in m.group(1):
				nxt.append(pat)
				continue
			pre, post = pat[:m.start()], pat[m.end():]
			for part in m.group(1).split(","):
				nxt.append(pre + part.strip() + post)
			changed = True
		out = nxt
	return out


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

def _parse_exclude(raw):
	"""None if omit (keep gitignore). Else a set of skip names (possibly empty).

	Returns (skip, err). err is an ERROR: string when the value is the wrong type.
	"""
	if raw is None:
		return None, None
	if isinstance(raw, bool):
		return None, (
			"ERROR: exclude must be a list of directory names "
			"or a comma-separated string"
		)
	if isinstance(raw, str):
		from lib.tool_calls import TryParseJsonArray
		parsed = TryParseJsonArray(raw)
		if parsed is not None:
			return _parse_exclude(parsed)
		s = raw.strip()
		if s.startswith("{") or s.startswith("["):
			return None, (
				"ERROR: exclude must be a list of directory names "
				"or a comma-separated string"
			)
		return {p.strip() for p in s.split(",") if p.strip()}, None
	if isinstance(raw, (list, tuple, set)):
		names = set()
		for x in raw:
			if isinstance(x, bool) or not isinstance(x, str):
				return None, "ERROR: exclude entries must be strings"
			n = x.strip()
			if n:
				names.add(n)
		return names, None
	return None, (
		"ERROR: exclude must be a list of directory names "
		"or a comma-separated string"
	)


def _skip_dirs(pattern, exclude):
	"""Skip names from exclude, minus literal glob path components (vendor/**).

	None = exclude omitted (ripgrep should still honor gitignore).
	Returns (skip, err).
	"""
	skip, err = _parse_exclude(exclude)
	if err:
		return None, err
	if skip is None:
		return None, None
	if not skip:
		return set(), None
	parts = {p for p in (pattern or "").replace("\\", "/").split("/") if p and p != "**"}
	if os.name == "nt":
		folded = {p.casefold() for p in parts}
		return {n for n in skip if n.casefold() not in folded}, None
	return {n for n in skip if n not in parts}, None


def _python_skip(pattern, exclude):
	"""Python walk skips: caller exclude, or _SKIP_DIR_NAMES when omitted."""
	skip, err = _skip_dirs(pattern, exclude)
	if err:
		return None, err
	if skip is None:
		return _skip_dirs(pattern, list(_SKIP_DIR_NAMES))
	return skip, None

def _name_in(name, skip):
	if not skip:
		return False
	if os.name == "nt":
		return name.casefold() in {s.casefold() for s in skip}
	return name in skip

def _rel_excluded(rel, skip):
	if not skip:
		return False
	parts = [p for p in (rel or "").replace("\\", "/").split("/") if p and p not in (".", "..")]
	return any(_name_in(p, skip) for p in parts)

def _iter_files(start, recursive, skip=()):
	"""Yield file paths under start, not descending into exclude-named dirs."""
	skip = set(skip or ())
	if os.path.isfile(start):
		yield start
		return
	if not os.path.isdir(start):
		return
	if not recursive:
		try:
			names = os.listdir(start)
		except OSError:
			return
		for name in names:
			if _name_in(name, skip):
				continue
			path = os.path.join(start, name)
			if os.path.isfile(path):
				yield path
		return
	for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
		if skip:
			dirnames[:] = [d for d in dirnames if not _name_in(d, skip)]
		for name in filenames:
			if _name_in(name, skip):
				continue
			yield os.path.join(dirpath, name)

def _glob_path_regex(query, recursive):
	"""stdlib glob.translate (3.13+). None on older Python — caller uses glob.iglob."""
	translate = getattr(globmod, "translate", None)
	if not callable(translate):
		return None
	try:
		return re.compile(translate(query, recursive=recursive, include_hidden=True))
	except TypeError:
		try:
			return re.compile(translate(query, recursive=recursive))
		except (re.error, TypeError, ValueError):
			return None
	except (re.error, ValueError):
		return None

def _parse_max_results(raw):
	if raw is None or raw == "":
		return _DEFAULT_GLOB, None
	try:
		n = int(raw)
	except (TypeError, ValueError):
		return None, "ERROR: max_results must be an integer"
	if n < 1:
		return None, "ERROR: max_results must be >= 1"
	return min(n, _HARD_MAX_GLOB), None

def _accept_glob_path(path, sandbox):
	if not os.path.isfile(path):
		return False
	if sandbox:
		_, deny = fs_sandbox.ResolveInRoot(path, must_exist=False, allow_new=False)
		if deny:
			return False
	return True

def _collect_glob_one(pattern: str, max_results=None, skip=None):
	"""Return (paths, display_absolute, truncated, err). Paths are absolute."""
	query, display_absolute, err = _prepare_glob(pattern)
	if err:
		return None, False, False, err

	sandbox = fs_sandbox.IsSandboxEnabled()
	recursive = "**" in query.replace("\\", "/")
	cap = max_results
	rx = _glob_path_regex(query, recursive)
	matches = []
	truncated = False
	skip = set(skip or ())
	root = fs_sandbox.GetFilesRoot()

	if rx is not None:
		start = _literal_glob_prefix(query) or root
		if not os.path.isdir(start):
			start = os.path.dirname(start) or root
		for path in _iter_files(start, recursive, skip):
			if not rx.match(path) and not rx.match(os.path.abspath(path)):
				continue
			if not _accept_glob_path(path, sandbox):
				continue
			matches.append(os.path.abspath(path))
			if recursive and cap is not None and len(matches) >= cap:
				truncated = True
				break
	else:
		for path in globmod.iglob(query, recursive=recursive):
			if not _accept_glob_path(path, sandbox):
				continue
			if skip and _rel_excluded(_rel(root, os.path.abspath(path)), skip):
				continue
			matches.append(os.path.abspath(path))
			if recursive and cap is not None and len(matches) >= cap:
				truncated = True
				break

	matches.sort()
	if not recursive and cap is not None and len(matches) > cap:
		matches = matches[:cap]
		truncated = True
	return matches, display_absolute, truncated, None


def _collect_glob(pattern: str, max_results=None, exclude=None):
	"""Like _collect_glob_one, after expanding {a,b} brace groups."""
	raw = (pattern or "").strip() or "**/*"
	skip, err = _python_skip(raw, exclude)
	if err:
		return None, False, False, err
	skip = skip or set()
	pats = _expand_braces(raw)
	if len(pats) == 1:
		return _collect_glob_one(pats[0], max_results=max_results, skip=skip)

	seen = set()
	matches = []
	truncated = False
	display_absolute = False
	any_ok = False
	last_err = None
	for pat in pats:
		remain = None if max_results is None else max(0, max_results - len(matches))
		if max_results is not None and remain == 0:
			truncated = True
			break
		paths, da, trunc, err = _collect_glob_one(pat, max_results=remain, skip=skip)
		if err:
			last_err = err
			continue
		any_ok = True
		display_absolute = display_absolute or da
		if trunc:
			truncated = True
		for p in paths:
			if p in seen:
				continue
			seen.add(p)
			matches.append(p)
			if max_results is not None and len(matches) >= max_results:
				truncated = True
				break
		if truncated and max_results is not None:
			break
	if not any_ok and last_err:
		return None, False, False, last_err
	matches.sort()
	return matches, display_absolute, truncated, None

def _format_glob_paths(paths, display_absolute):
	root = fs_sandbox.GetFilesRoot()
	if display_absolute:
		return [p.replace("\\", "/") for p in paths]
	return [_rel(root, p) for p in paths]

def _format_glob_result(paths, display_absolute, truncated, cap):
	if not paths:
		return "(no matches)"
	paths = sorted(paths)
	out = "\n".join(_format_glob_paths(paths, display_absolute))
	if truncated:
		out += (
			f"\n... truncated at {cap} matches "
			"(raise max_results or narrow the pattern)"
		)
	return out

def _glob_files_python(pattern: str, max_results=None, exclude=None) -> str:
	cap, err = _parse_max_results(max_results)
	if err:
		return err
	paths, display_absolute, truncated, err = _collect_glob(
		pattern, max_results=cap, exclude=exclude,
	)
	if err:
		return err
	return _format_glob_result(paths, display_absolute, truncated, cap)

def _looks_binary(path: str) -> bool:
	ext = os.path.splitext(path)[1].lower()
	if ext in _BINARY_EXTS:
		return True
	try:
		with open(path, "rb") as f:
			return b"\x00" in f.read(8192)
	except OSError:
		return True


def _clip_grep_line(line: str) -> str:
	text = line.rstrip("\r\n")
	if len(text) > _MAX_GREP_LINE:
		return text[:_MAX_GREP_LINE] + "…"
	return text


def _rel_matches_globs(rel, globs):
	"""Match workspace-relative paths against include globs. Empty globs = all.

	CLI --glob on ripgrep *whitelists* and overrides .gitignore, so include
	patterns are applied here after rg has already honored ignore files.
	Uses pathlib.full_match so `src/*.py` is one directory level, not nested.
	"""
	if not globs:
		return True
	rel = (rel or "").replace("\\", "/").lstrip("./")
	if not rel:
		return False
	path = PurePosixPath(rel)
	full_match = getattr(path, "full_match", None)
	for g in globs:
		g = (g or "").replace("\\", "/")
		if g in ("", "*", "**", "**/*"):
			return True
		if g.startswith("./"):
			g = g[2:]
		try:
			if callable(full_match):
				if full_match(g):
					return True
				if g.startswith("**/") and full_match(g[3:]):
					return True
			elif fnmatch.fnmatch(rel, g) or ("**" in g and path.match(g)):
				return True
		except ValueError:
			continue
	return False


def _glob_relative_to_root(query, root):
	"""Pattern relative to the workspace root, or None to accept every file."""
	q = os.path.abspath(query).replace("\\", "/")
	r = os.path.abspath(root).replace("\\", "/")
	rest = q[len(r) + 1:] if q.startswith(r + "/") else ""
	if rest in ("", "*", "**", "**/*"):
		return None
	return rest or None


def _rg_targets(glob_pat):
	"""Return (targets, globs, display_absolute, err) for an rg search root.

	Brace groups are expanded first so `{src,lib}/**/*.py` searches both dirs.
	Include globs are workspace-relative (`src/*.py`), not remainders after the
	literal prefix (`*.py`), so one-level patterns do not match nested files.
	"""
	glob_pat = (glob_pat or "").strip() or "**/*"
	pats = _expand_braces(glob_pat)
	root = fs_sandbox.GetFilesRoot()
	display_absolute = False
	targets = []
	seen_t = set()
	globs = []
	seen_g = set()
	last_err = None
	any_ok = False
	for pat in pats:
		query, abs_disp, err = _prepare_glob(pat)
		if err:
			last_err = err
			continue
		any_ok = True
		if abs_disp:
			display_absolute = True
		start = _literal_glob_prefix(query) or root
		if os.path.isfile(start) or os.path.isdir(start):
			key = os.path.abspath(start)
			if key not in seen_t:
				seen_t.add(key)
				targets.append(key)
		rel_g = _glob_relative_to_root(query, root)
		if rel_g and rel_g not in seen_g:
			seen_g.add(rel_g)
			globs.append(rel_g)
	if not any_ok:
		return None, None, False, last_err or "(no matches)"
	if not targets:
		return None, None, display_absolute, "(no matches)"
	return targets, globs, display_absolute, None


def _rg_exclude_args(skip):
	"""None skip → honor gitignore. Else --no-ignore plus negation globs."""
	if skip is None:
		return []
	args = ["--no-ignore"]
	for name in sorted(skip):
		args.extend(["--glob", f"!{name}"])
	return args


def _rg_foreach(cmd, root, on_line):
	"""Call on_line(line) for each stdout line; False stops.

	Wall-clock timeout from Popen (even if stdout never yields). Stderr is drained
	on a side thread so a full pipe cannot deadlock stdout.
	Returns None if rg could not be started, or (code, stderr, timed_out).
	"""
	try:
		proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			errors="replace",
			cwd=root,
		)
	except OSError:
		return None
	timed_out = threading.Event()
	finished = threading.Event()
	stderr_buf = []

	def _drain_err():
		try:
			if proc.stderr:
				stderr_buf.append(proc.stderr.read() or "")
		except OSError:
			pass

	def _kill_after():
		if not finished.wait(_RG_TIMEOUT):
			timed_out.set()
			try:
				proc.kill()
			except OSError:
				pass

	threading.Thread(target=_drain_err, daemon=True).start()
	threading.Thread(target=_kill_after, daemon=True).start()
	stopped = False
	try:
		stdout = proc.stdout
		if stdout is None:
			try:
				proc.kill()
			except OSError:
				pass
			return None
		for raw in stdout:
			if on_line(raw.rstrip("\r\n")) is False:
				stopped = True
				break
		if stopped or timed_out.is_set():
			try:
				proc.kill()
			except OSError:
				pass
		try:
			proc.wait(timeout=1)
		except subprocess.TimeoutExpired:
			try:
				proc.kill()
			except OSError:
				pass
			try:
				proc.wait(timeout=1)
			except subprocess.TimeoutExpired:
				pass
	except OSError:
		try:
			proc.kill()
		except OSError:
			pass
		return None
	finally:
		finished.set()
	return proc.returncode, "".join(stderr_buf), timed_out.is_set()


def _format_grep_hit(display, line_no, text):
	return f"{display}:{line_no}:{_clip_grep_line(text)}"


def _grep_finish(hits, capped, files_trunc=False):
	if not hits:
		if files_trunc:
			return (
				f"(no matches in first {_MAX_GREP_FILES} files; "
				"narrow glob, e.g. **/*.{cpp,h})"
			)
		return "(no matches)"
	out = "\n".join(hits)
	if capped:
		out += (
			f"\n... {capped} "
			"(narrow glob e.g. **/*.{cpp,h}, or a more specific pattern)"
		)
	elif files_trunc:
		out += f"\n... scanned first {_MAX_GREP_FILES} files"
	return out


def _glob_files_rg(rg, pattern, max_results, exclude=None):
	cap, err = _parse_max_results(max_results)
	if err:
		return err
	targets, globs, display_absolute, err = _rg_targets(pattern)
	if err:
		return err
	skip, err = _skip_dirs(pattern, exclude)
	if err:
		return err
	root = fs_sandbox.GetFilesRoot()
	sandbox = fs_sandbox.IsSandboxEnabled()
	cmd = [rg, "--files", "--no-config", "--hidden", "--no-require-git"]
	cmd.extend(_rg_exclude_args(skip))
	cmd.append("--")
	cmd.extend(targets)

	paths = []
	truncated = False

	def take(full):
		nonlocal truncated
		if not full:
			return True
		if not os.path.isabs(full):
			full = os.path.join(root, full)
		full = os.path.abspath(full)
		rel = _rel(root, full)
		if not _rel_matches_globs(rel, globs):
			return True
		if skip and _rel_excluded(rel, skip):
			return True
		if not _accept_glob_path(full, sandbox):
			return True
		paths.append(full)
		if len(paths) >= cap:
			truncated = True
			return False
		return True

	got = _rg_foreach(cmd, root, take)
	if got is None:
		return "ERROR: ripgrep failed to start"
	code, stderr, timed_out = got
	if timed_out and not paths:
		return f"ERROR: ripgrep timed out after {_RG_TIMEOUT}s"
	if timed_out:
		truncated = True
	if code not in (0, 1, None) and not paths:
		msg = (stderr or "").strip() or f"ripgrep exited {code}"
		return f"ERROR: ripgrep failed ({msg})"
	return _format_glob_result(paths, display_absolute, truncated, cap)


def _glob_files(pattern: str, max_results=None, exclude=None) -> str:
	rg = _find_rg()
	if rg:
		return _glob_files_rg(rg, pattern, max_results, exclude)
	return _glob_files_python(pattern, max_results, exclude)


def _grep_files_rg(rg, pattern, glob_pat, case_insensitive, exclude=None):
	"""Ripgrep search. Returns a result string."""
	targets, globs, display_absolute, err = _rg_targets(glob_pat)
	if err:
		return err
	if targets is None:
		return "(no matches)"
	skip, err = _skip_dirs(glob_pat, exclude)
	if err:
		return err
	root = fs_sandbox.GetFilesRoot()
	sandbox = fs_sandbox.IsSandboxEnabled()
	cmd = [
		rg, "--json", "--no-config", "--hidden", "--no-require-git",
		"--max-columns", str(_MAX_GREP_LINE), "--max-columns-preview",
		"--max-filesize", str(_MAX_FILE_BYTES),
		"-e", pattern,
	]
	if case_insensitive:
		cmd.append("-i")
	cmd.extend(_rg_exclude_args(skip))
	cmd.append("--")
	cmd.extend(targets)

	hits = []
	hit_chars = 0
	capped = None

	def take(raw):
		nonlocal capped, hit_chars
		try:
			ev = json.loads(raw)
		except json.JSONDecodeError:
			return True
		if ev.get("type") != "match":
			return True
		data = ev.get("data") or {}
		full = ((data.get("path") or {}).get("text")) or ""
		if not full:
			return True
		full = os.path.abspath(full)
		rel = _rel(root, full)
		if not _rel_matches_globs(rel, globs):
			return True
		if skip and _rel_excluded(rel, skip):
			return True
		if sandbox:
			_, deny = fs_sandbox.ResolveInRoot(full, must_exist=False, allow_new=False)
			if deny:
				return True
		line_no = data.get("line_number") or 0
		text = ((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
		display = full.replace("\\", "/") if display_absolute else _rel(root, full)
		row = _format_grep_hit(display, line_no, text)
		hits.append(row)
		hit_chars += len(row) + 1
		if len(hits) >= _MAX_GREP_HITS:
			capped = f"truncated at {_MAX_GREP_HITS} hits"
			return False
		if hit_chars >= _MAX_GREP_BYTES:
			capped = f"truncated at {_MAX_GREP_BYTES} bytes"
			return False
		return True

	got = _rg_foreach(cmd, root, take)
	if got is None:
		return "ERROR: ripgrep failed to start"
	code, stderr, timed_out = got
	if timed_out and not hits:
		return f"ERROR: ripgrep timed out after {_RG_TIMEOUT}s"
	if timed_out:
		capped = capped or f"truncated (ripgrep timeout after {_RG_TIMEOUT}s)"
	if code not in (0, 1, None) and not hits:
		msg = (stderr or "").strip() or f"ripgrep exited {code}"
		low = msg.lower()
		if "regex" in low or "syntax" in low or "parse error" in low:
			return f"ERROR: invalid regex ({msg})"
		return f"ERROR: ripgrep failed ({msg})"
	return _grep_finish(hits, capped)


def _grep_files_python(pattern, glob_pat, case_insensitive, exclude=None) -> str:
	paths, display_absolute, files_trunc, err = _collect_glob(
		glob_pat or "**/*", max_results=_MAX_GREP_FILES, exclude=exclude,
	)
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
	hit_chars = 0
	capped = None
	for full in paths:
		display = full.replace("\\", "/") if display_absolute else _rel(root, full)
		try:
			if os.path.getsize(full) > _MAX_FILE_BYTES:
				continue
			if _looks_binary(full):
				continue
			with open(full, "r", encoding="utf-8", errors="replace") as f:
				for i, line in enumerate(f, 1):
					if not rx.search(line):
						continue
					row = _format_grep_hit(display, i, line)
					hits.append(row)
					hit_chars += len(row) + 1
					if len(hits) >= _MAX_GREP_HITS:
						capped = f"truncated at {_MAX_GREP_HITS} hits"
						break
					if hit_chars >= _MAX_GREP_BYTES:
						capped = f"truncated at {_MAX_GREP_BYTES} bytes"
						break
		except Exception:
			continue
		if capped:
			break
	return _grep_finish(hits, capped, files_trunc=files_trunc)


def _grep_files(
	pattern: str,
	glob_pat: str = "**/*",
	case_insensitive: bool = False,
	exclude=None,
) -> str:
	if not str(pattern or ""):
		return "ERROR: pattern is empty"
	rg = _find_rg()
	if rg:
		return _grep_files_rg(rg, pattern, glob_pat, case_insensitive, exclude)
	return _grep_files_python(pattern, glob_pat, case_insensitive, exclude)

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

def _file_has_nul(abs_path):
	try:
		with open(abs_path, "rb") as f:
			return b"\0" in f.read(8192)
	except OSError:
		return True

def _tail_last_line(abs_path):
	"""Last line without terminator. Reads at most _MAX_FILE_BYTES from the end."""
	try:
		size = os.path.getsize(abs_path)
	except OSError as e:
		return None, f"ERROR: failed to read file ({e})"
	if size <= 0:
		return "", None
	with open(abs_path, "rb") as f:
		end = size
		if end >= 2:
			f.seek(end - 2)
			tail = f.read(2)
			if tail == b"\r\n":
				end -= 2
			elif tail[-1:] in (b"\n", b"\r"):
				end -= 1
		elif end == 1:
			f.seek(0)
			if f.read(1) in (b"\n", b"\r"):
				end = 0
		if end <= 0:
			return "", None
		start = max(0, end - _MAX_FILE_BYTES)
		f.seek(start)
		data = f.read(end - start)
	nl = max(data.rfind(b"\n"), data.rfind(b"\r"))
	if nl >= 0:
		data = data[nl + 1:]
	return data.decode("utf-8", errors="replace"), None


def _read_last_line(abs_path):
	"""Return (line_no, text, err). line_no is -1 when the file is over the size cap."""
	try:
		size = os.path.getsize(abs_path)
	except OSError as e:
		return None, None, f"ERROR: failed to read file ({e})"
	if size <= 0:
		return 0, "", None
	if size <= _MAX_FILE_BYTES:
		try:
			with open(abs_path, encoding="utf-8", errors="replace") as f:
				text = f.read()
		except OSError as e:
			return None, None, f"ERROR: failed to read file ({e})"
		lines = text.splitlines()
		if not lines:
			return 0, "", None
		return len(lines), lines[-1], None
	text, err = _tail_last_line(abs_path)
	if err:
		return None, None, err
	return -1, text, None


def _iter_text_lines(abs_path):
	with open(abs_path, encoding="utf-8", errors="replace") as f:
		for i, line in enumerate(f, 1):
			yield i, line.rstrip("\r\n")

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
	if _file_has_nul(abs_path):
		return "ERROR: binary file (NUL byte detected)"

	start_raw, err = _parse_line_arg(start_line, "start_line")
	if err:
		return err
	end_raw, err = _parse_line_arg(end_line, "end_line")
	if err:
		return err

	if start_raw is None and end_raw is not None:
		return "ERROR: start_line is required when end_line is set"
	if (
		start_raw is not None and end_raw is not None
		and start_raw != -1 and end_raw != -1 and end_raw < start_raw
	):
		return "ERROR: start_line is after end_line"

	if start_raw is None:
		win_start, win_end = 1, None
		last_only = False
	elif end_raw is None:
		last_only = start_raw == -1
		win_start = win_end = start_raw
	elif start_raw == -1 and end_raw == -1:
		last_only = True
		win_start = win_end = -1
	elif start_raw == -1:
		last_only = True
		win_start, win_end = -1, end_raw
	elif end_raw == -1:
		last_only = False
		win_start, win_end = start_raw, None
	else:
		last_only = False
		win_start, win_end = start_raw, end_raw

	last_no = 0
	last_text = ""
	parts = []
	used = 0
	truncated = False
	started = False
	if last_only:
		last_no, last_text, err = _read_last_line(abs_path)
		if err:
			return err
		if last_no == 0 and last_text == "":
			return "ERROR: file has no lines" if start_raw is not None else "(empty)"
		if end_raw not in (None, -1) and end_raw != last_no:
			if last_no < 0:
				return f"ERROR: file exceeds {_MAX_FILE_BYTES} byte limit"
			if end_raw > last_no:
				return f"ERROR: line {end_raw} out of range (file has {last_no} line(s))"
			return "ERROR: start_line is after end_line"
		return f"{last_no}|{last_text}"

	try:
		for lineno, line in _iter_text_lines(abs_path):
			last_no = lineno
			if lineno < win_start:
				continue
			if win_end is not None and lineno > win_end:
				break
			piece = f"{lineno}|{line}"
			add = len(piece.encode("utf-8")) + (1 if parts else 0)
			if parts and used + add > _MAX_FILE_BYTES:
				truncated = True
				break
			parts.append(piece)
			used += add
			started = True
			if win_end is not None and lineno >= win_end:
				break
	except OSError as e:
		return f"ERROR: failed to read file ({e})"

	if last_no == 0:
		return "(empty)" if start_raw is None else "ERROR: file has no lines"
	if win_start > 0 and not started:
		return f"ERROR: line {win_start} out of range (file has {last_no} line(s))"
	if win_end is not None and win_end > last_no and not truncated:
		return f"ERROR: line {win_end} out of range (file has {last_no} line(s))"
	text = "\n".join(parts) if parts else "(empty)"
	if truncated:
		text += f"\n... truncated at {_MAX_FILE_BYTES} bytes"
	return text

def _clip_anchor(text):
	s = text if isinstance(text, str) else str(text)
	if len(s) > 80:
		return s[:80] + "…"
	return s

def _require_str(raw, label, *, optional=False):
	"""None if omitted (optional). Reject bool/number so XML coerce cannot spoof a line."""
	if raw is None:
		if optional:
			return None, None
		return None, f"ERROR: {label} is required"
	if isinstance(raw, bool) or not isinstance(raw, str):
		return None, f"ERROR: {label} must be a string"
	return raw, None

def _anchor_line(raw, label):
	"""None if omitted. Single line of exact text (trailing newline ignored)."""
	s, err = _require_str(raw, label, optional=True)
	if err or s is None:
		return None, err
	if s.endswith("\r\n"):
		s = s[:-2]
	elif s.endswith("\n") or s.endswith("\r"):
		s = s[:-1]
	if "\n" in s or "\r" in s:
		return None, f"ERROR: {label} must be a single line"
	return s, None

def _check_neighbors(lines, prev_idx, next_idx, prev_want, next_want):
	n = len(lines)
	if prev_want is not None:
		if prev_idx is None or prev_idx < 0 or prev_idx >= n:
			return "ERROR: prev_line was set but there is no line before this edit"
		got = lines[prev_idx]
		if got != prev_want:
			return (
				"ERROR: prev_line mismatch "
				f"(wanted {_clip_anchor(prev_want)!r}, file has {_clip_anchor(got)!r})"
			)
	if next_want is not None:
		if next_idx is None or next_idx < 0 or next_idx >= n:
			return "ERROR: next_line was set but there is no line after this edit"
		got = lines[next_idx]
		if got != next_want:
			return (
				"ERROR: next_line mismatch "
				f"(wanted {_clip_anchor(next_want)!r}, file has {_clip_anchor(got)!r})"
			)
	return None

def _write_file(path: str, content, start_line=None, end_line=None, prev_line=None, next_line=None) -> str:
	content, err = _require_str(content, "content")
	if err:
		return err

	start_raw, err = _parse_line_arg(start_line, "start_line")
	if err:
		return err
	end_raw, err = _parse_line_arg(end_line, "end_line")
	if err:
		return err
	if end_raw is not None and start_raw is None:
		return "ERROR: start_line is required when end_line is set"
	prev_want, err = _anchor_line(prev_line, "prev_line")
	if err:
		return err
	next_want, err = _anchor_line(next_line, "next_line")
	if err:
		return err
	if start_raw is None and (prev_want is not None or next_want is not None):
		return "ERROR: prev_line/next_line require start_line"

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
		prev_idx = ins_idx - 1 if ins_idx > 0 else None
		next_idx = ins_idx if ins_idx < nlines else None
		err = _check_neighbors(lines, prev_idx, next_idx, prev_want, next_want)
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
		prev_idx = s_idx - 1 if s_idx > 0 else None
		next_idx = e_idx + 1 if e_idx + 1 < nlines else None
		err = _check_neighbors(lines, prev_idx, next_idx, prev_want, next_want)
		if err:
			return err

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
			return _glob_files(
				arguments.get("pattern", ""),
				arguments.get("max_results"),
				arguments.get("exclude"),
			)
		if name == "grep_files":
			return _grep_files(
				arguments.get("pattern", ""),
				arguments.get("glob", "**/*"),
				bool(arguments.get("case_insensitive", False)),
				arguments.get("exclude"),
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
				arguments.get("prev_line"),
				arguments.get("next_line"),
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
