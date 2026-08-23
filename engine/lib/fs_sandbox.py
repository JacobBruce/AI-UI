"""Path sandbox for native Coding pack path tools.

Root is the open project workspace when set, otherwise ~/.aiui (Home).
When a project is open, ~/.aiui/skills, ~/.aiui/rules, and ~/.aiui/agents are
also allowed so the agent can read (and update) global prompt files. The rest
of ~/.aiui (config, sessions, keys) stays blocked.
When the Coding pack has sandbox=False, absolute paths outside the root are allowed;
relative paths still resolve against the workspace/home root.
run_command uses the root as cwd but does not enforce path sandbox on the command itself.
"""

import os
from pathlib import Path

E = None

_PROMPT_SUBDIRS = ("skills", "rules", "agents")


def bind(engine_module):
	global E
	E = engine_module


def GetAiuiHome():
	home = ""
	if E is not None:
		home = str(getattr(E, "aiui_home", "") or "").strip()
	if not home:
		home = os.path.join(os.path.expanduser("~"), ".aiui")
	return os.path.abspath(os.path.expanduser(home))


def GetFilesRoot():
	"""Absolute default root for Files tools (project or ~/.aiui)."""
	ws = ""
	if E is not None:
		ws = str(getattr(E, "workspace_dir", "") or "").strip()
	root = ws if ws else GetAiuiHome()
	root = os.path.abspath(os.path.expanduser(root))
	os.makedirs(root, exist_ok=True)
	return root


def ExtraSandboxRoots():
	"""Global prompt dirs allowed in addition to the workspace when a project is open."""
	if E is None:
		return []
	ws = str(getattr(E, "workspace_dir", "") or "").strip()
	if not ws:
		return []
	home = GetAiuiHome()
	return [os.path.join(home, name) for name in _PROMPT_SUBDIRS]


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


def _SandboxRoots():
	roots = [Path(GetFilesRoot()).resolve()]
	for extra in ExtraSandboxRoots():
		roots.append(Path(extra).expanduser().resolve())
	return roots


def _UnderRoots(path, roots):
	for root in roots:
		try:
			path.relative_to(root)
			return True
		except ValueError:
			continue
	return False


def ResolveInRoot(path_str, *, must_exist=False, allow_new=True):
	"""Resolve path_str, optionally enforcing the sandbox roots.

	Returns (abs_path, None) on success or (None, error_string) on failure.
	"""
	raw = str(path_str or "").strip()
	if not raw:
		return None, "ERROR: path is empty"

	root_path = Path(GetFilesRoot()).resolve()
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
		roots = _SandboxRoots()
		if not _UnderRoots(resolved, roots):
			return None, "ERROR: path is outside the workspace sandbox"

		if resolved.exists() or resolved.is_symlink():
			real = Path(os.path.realpath(str(resolved)))
			real_roots = [Path(os.path.realpath(str(r))) for r in roots]
			if not _UnderRoots(real, real_roots):
				return None, "ERROR: path is outside the workspace sandbox"
			resolved = real

	if must_exist and not resolved.exists():
		return None, "ERROR: path does not exist"
	return str(resolved), None
