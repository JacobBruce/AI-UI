# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

import os
import re
from lib import ipc

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _SafeRead(path):
	try:
		with open(path, "r", encoding="utf-8") as f:
			return f.read()
	except Exception as e:
		ipc.LogErr(f"WARN: could not read {path}: {e}")
		return None


def _ParseFrontmatter(text):
	"""Return (meta_dict, body). meta may be empty."""
	m = _FRONTMATTER_RE.match(text)
	if not m:
		return {}, text
	meta = {}
	for line in m.group(1).splitlines():
		if ":" not in line:
			continue
		key, val = line.split(":", 1)
		meta[key.strip().lower()] = val.strip().strip('"').strip("'")
	return meta, text[m.end():]


def _Roots(aiui_home, workspace=""):
	roots = []
	if aiui_home and os.path.isdir(aiui_home):
		roots.append(aiui_home)
	ws = (workspace or "").strip()
	if ws:
		proj = os.path.join(ws, ".aiui")
		if os.path.isdir(proj):
			roots.append(proj)
	return roots


def _ListMd(dir_path):
	"""Return sorted list of (id, abs_path) for .md files in dir."""
	out = []
	if not dir_path or not os.path.isdir(dir_path):
		return out
	try:
		names = os.listdir(dir_path)
	except Exception as e:
		ipc.LogErr(f"WARN: could not list {dir_path}: {e}")
		return out
	for name in sorted(names):
		if not name.endswith(".md"):
			continue
		path = os.path.join(dir_path, name)
		if os.path.isfile(path):
			out.append((name, path))
	return out


def _ListSkillEntries(dir_path):
	"""Flat skills/*.md and Cursor-style skills/*/SKILL.md → (id, path).

	Folder skills use the directory name as id. Flat skills keep the filename
	(including .md) so existing enabled_skills configs stay valid.
	"""
	out = []
	if not dir_path or not os.path.isdir(dir_path):
		return out
	try:
		names = os.listdir(dir_path)
	except Exception as e:
		ipc.LogErr(f"WARN: could not list {dir_path}: {e}")
		return out
	for name in sorted(names):
		path = os.path.join(dir_path, name)
		if os.path.isfile(path) and name.endswith(".md"):
			out.append((name, path))
		elif os.path.isdir(path):
			skill_md = os.path.join(path, "SKILL.md")
			if os.path.isfile(skill_md):
				out.append((name, skill_md))
	return out


def _MergeFiles(kind, aiui_home, workspace=""):
	"""Global then project; project id wins on clash."""
	by_id = {}
	for root in _Roots(aiui_home, workspace):
		for fid, path in _ListMd(os.path.join(root, kind)):
			by_id[fid] = path
	return by_id


def _MergeSkills(aiui_home, workspace=""):
	"""Global then project; project id wins on clash."""
	by_id = {}
	for root in _Roots(aiui_home, workspace):
		for fid, path in _ListSkillEntries(os.path.join(root, "skills")):
			by_id[fid] = path
	return by_id


def _FallbackSkillDesc(body):
	"""First non-empty, non-heading line when frontmatter has no description."""
	for line in (body or "").splitlines():
		s = line.strip()
		if not s or s.startswith("#"):
			continue
		if len(s) > 200:
			return s[:197] + "..."
		return s
	return ""


def ListAgents(aiui_home, workspace=""):
	return [{"id": fid, "path": path} for fid, path in sorted(_MergeFiles("agents", aiui_home, workspace).items())]


def ListRules(aiui_home, workspace=""):
	return [{"id": fid, "path": path} for fid, path in sorted(_MergeFiles("rules", aiui_home, workspace).items())]


def ListSkills(aiui_home, workspace=""):
	skills = []
	for fid, path in sorted(_MergeSkills(aiui_home, workspace).items()):
		text = _SafeRead(path)
		if text is None:
			continue
		meta, body = _ParseFrontmatter(text)
		# Folder id is already the directory name; flat id is filename.md.
		fallback_name = fid[:-3] if fid.lower().endswith(".md") else fid
		name = (meta.get("name") or "").strip() or fallback_name
		desc = (meta.get("description") or "").strip()
		# Naive frontmatter leaves YAML folded scalars as ">-" / "|".
		if desc in ("", ">", ">-", "|", "|-", "|+"):
			desc = _FallbackSkillDesc(body)
		skills.append({"id": fid, "name": name, "description": desc, "path": path})
	return skills


def LoadAgent(agent_id, aiui_home, workspace=""):
	if not agent_id:
		return ""
	agents = _MergeFiles("agents", aiui_home, workspace)
	path = agents.get(agent_id)
	if not path or not os.path.isfile(path):
		return ""
	text = _SafeRead(path)
	if text is None:
		return ""
	_meta, body = _ParseFrontmatter(text)
	return body.strip()


def LoadEnabledRules(enabled_ids, aiui_home, workspace=""):
	rules = _MergeFiles("rules", aiui_home, workspace)
	if enabled_ids is None:
		ids = list(rules.keys())
	else:
		ids = [i for i in enabled_ids if i in rules]
	parts = []
	for fid in sorted(ids):
		text = _SafeRead(rules[fid])
		if text is None:
			continue
		_meta, body = _ParseFrontmatter(text)
		body = body.strip()
		if body:
			parts.append(body)
	return parts


def SkillCatalogText(skills, enabled_ids):
	if enabled_ids is None:
		chosen = skills
	else:
		en = set(enabled_ids)
		chosen = [s for s in skills if s["id"] in en]
	if not chosen:
		return ""
	lines = [
		"# Available skills",
		"",
		"Use a skill when needed by reading its file (loaded on demand). "
		"Folder skills may reference related files beside SKILL.md - follow those links as needed.",
		"",
	]
	for s in chosen:
		skill_path = os.path.abspath(s.get("path") or s["id"])
		line = f"- **{s['name']}** (`{skill_path}`)"
		if s.get("description"):
			line += f": {s['description']}"
		lines.append(line)
	return "\n".join(lines)


def _AbsRoot(aiui_home, workspace=""):
	"""Sandbox / tool cwd root: open project, else ~/.aiui."""
	ws = (workspace or "").strip()
	if ws:
		return os.path.abspath(os.path.expanduser(ws))
	home = (aiui_home or "").strip()
	if not home:
		home = os.path.join(os.path.expanduser("~"), ".aiui")
	return os.path.abspath(os.path.expanduser(home))


def ArtifactsDir(aiui_home, workspace=""):
	"""Disposable files: project/.aiui/artifacts, or ~/.aiui/artifacts on Home."""
	ws = (workspace or "").strip()
	if ws:
		return os.path.join(_AbsRoot(aiui_home, workspace), ".aiui", "artifacts")
	return os.path.join(_AbsRoot(aiui_home, workspace), "artifacts")


def EnvironmentPromptBlock(aiui_home, workspace=""):
	"""Short always-on paths so agents know where to put disposable files."""
	root = _AbsRoot(aiui_home, workspace)
	arts = ArtifactsDir(aiui_home, workspace)
	return "\n".join([
		"# Environment\n",
		f"Workspace root: {root}",
		f"Artifacts directory: {arts}",
		"\nUse the artifacts directory for disposable scripts, notes, and other files that should not live in the project tree (or when there is no project). Prefer writing project source under the workspace root.",
	])


def ComposeSystemPrompt(aiui_home, workspace="", agent_id="", enabled_rules=None, enabled_skills=None):
	"""Agent → Rules → Skills (metadata) → Environment. Tools stay outside this compose."""
	parts = []
	agent_body = LoadAgent(agent_id, aiui_home, workspace)
	if agent_body:
		parts.append(agent_body)
	for rule in LoadEnabledRules(enabled_rules, aiui_home, workspace):
		parts.append(rule)
	skills = ListSkills(aiui_home, workspace)
	catalog = SkillCatalogText(skills, enabled_skills)
	if catalog:
		parts.append(catalog)
	parts.append(EnvironmentPromptBlock(aiui_home, workspace))
	return "\n\n".join(parts).strip()


def PromptFilesSnapshot(aiui_home, workspace="", agent_id="", enabled_rules=None, enabled_skills=None):
	agents = ListAgents(aiui_home, workspace)
	rules = ListRules(aiui_home, workspace)
	skills = ListSkills(aiui_home, workspace)
	# Default: all enabled when config has null
	if enabled_rules is None:
		enabled_rules = [r["id"] for r in rules]
	if enabled_skills is None:
		enabled_skills = [s["id"] for s in skills]
	# Drop stale ids
	rule_ids = {r["id"] for r in rules}
	skill_ids = {s["id"] for s in skills}
	enabled_rules = [i for i in enabled_rules if i in rule_ids]
	enabled_skills = [i for i in enabled_skills if i in skill_ids]
	agent_ok = any(a["id"] == agent_id for a in agents) if agent_id else True
	if agent_id and not agent_ok:
		agent_id = ""
	composed = ComposeSystemPrompt(aiui_home, workspace, agent_id, enabled_rules, enabled_skills)
	return {
		"agents": [{"id": a["id"]} for a in agents],
		"rules": [{"id": r["id"]} for r in rules],
		"skills": [{"id": s["id"], "name": s["name"], "description": s["description"]} for s in skills],
		"agent": agent_id,
		"enabled_rules": enabled_rules,
		"enabled_skills": enabled_skills,
		"composed": composed,
	}


def SendPromptFilesEvent(aiui_home, workspace="", agent_id="", enabled_rules=None, enabled_skills=None):
	snap = PromptFilesSnapshot(aiui_home, workspace, agent_id, enabled_rules, enabled_skills)
	ipc.SendEvent("prompt_files", snap)
	return snap
