"""Path sandbox for native Coding pack path tools.

Root is the open project workspace when set, otherwise ~/.aiui (Home).
When the Coding pack has sandbox=False, absolute paths outside the root are allowed;
relative paths still resolve against the workspace/home root.
run_command uses the root as cwd but does not enforce path sandbox on the command itself.
"""

import os
from pathlib import Path

E = None

def bind(engine_module):
	global E
	E = engine_module

def GetFilesRoot():
	"""Absolute default root for Files tools (project or ~/.aiui)."""
	ws = ""
	home = ""
	if E is not None:
		ws = str(getattr(E, "workspace_dir", "") or "").strip()
		home = str(getattr(E, "aiui_home", "") or "").strip()
	if not home:
		home = os.path.join(os.path.expanduser("~"), ".aiui")
	root = ws if ws else home
	root = os.path.abspath(os.path.expanduser(root))
	os.makedirs(root, exist_ok=True)
	return root

def IsSandboxEnabled():
	"""Files pack sandbox flag (default True)."""
	packs = getattr(E, "tool_packs", None) if E is not None else None
	if isinstance(packs, list):
		for p in packs:
			if p and p.get("id") == "native:coding":
				if "sandbox" in p:
					return bool(p.get("sandbox"))
				if "sandbox_enabled" in p:
					return bool(p.get("sandbox_enabled"))
				return True
	return True

def ResolveInRoot(path_str, *, must_exist=False, allow_new=True):
	"""Resolve path_str, optionally enforcing the sandbox root.

	Returns (abs_path, None) on success or (None, error_string) on failure.
	"""
	root = GetFilesRoot()
	root_path = Path(root).resolve()
	raw = str(path_str or "").strip()
	if not raw:
		return None, "ERROR: path is empty"

	candidate = Path(raw).expanduser()
	if not candidate.is_absolute():
		candidate = root_path / candidate
	try:
		if allow_new and not candidate.exists():
			parent = candidate.parent.resolve()
			resolved = (parent / candidate.name)
		else:
			resolved = candidate.resolve()
	except Exception as e:
		return None, f"ERROR: invalid path ({e})"

	if IsSandboxEnabled():
		try:
			resolved.relative_to(root_path)
		except ValueError:
			return None, "ERROR: path is outside the workspace sandbox"

		if resolved.exists() or resolved.is_symlink():
			try:
				real = Path(os.path.realpath(str(resolved)))
				real.relative_to(Path(os.path.realpath(str(root_path))))
			except ValueError:
				return None, "ERROR: path is outside the workspace sandbox"
			resolved = real

	if must_exist and not resolved.exists():
		return None, "ERROR: path does not exist"
	return str(resolved), None
