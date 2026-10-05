"""Locate and rewrite the last typed user turn for in-bubble edit + redo."""

from lib import context_compact


def _UserNames(user_name="user"):
	name = str(user_name or "user").lower()
	return ("user", name)


def IsUserRole(msg, user_name="user"):
	if not isinstance(msg, dict):
		return False
	role = str(msg.get("role") or "").lower()
	return role in _UserNames(user_name)


def _TextParts(content):
	if not isinstance(content, list):
		return []
	parts = []
	for i, part in enumerate(content):
		if not isinstance(part, dict):
			continue
		kind = str(part.get("type") or "text")
		if kind != "text" and "text" not in part:
			continue
		if "text" not in part:
			continue
		parts.append(i)
	return parts


def IsContinueStub(msg, user_name="user"):
	"""True for ContChat's hidden user turn (`content` is exactly `continue`)."""
	if not IsUserRole(msg, user_name):
		return False
	content = msg.get("content")
	if isinstance(content, str):
		return content.strip() == "continue"
	idxs = _TextParts(content)
	if not idxs:
		return False
	texts = [str(content[i].get("text") or "").strip() for i in idxs]
	return texts == ["continue"]


def LastEditableUserIndex(messages, user_name="user"):
	"""Index of the last typed user message (skip compact notices and continue stubs)."""
	raw = list(messages or [])
	for i in range(len(raw) - 1, -1, -1):
		m = raw[i]
		if not isinstance(m, dict):
			continue
		if context_compact.IsCompactNotice(m):
			continue
		if not IsUserRole(m, user_name):
			continue
		if IsContinueStub(m, user_name):
			continue
		return i
	return -1


def SetUserMessageText(msg, text):
	"""Replace string content or the last text part; keep media parts."""
	if not isinstance(msg, dict):
		return msg
	text = str(text or "")
	content = msg.get("content")
	if isinstance(content, list):
		idxs = _TextParts(content)
		user_idxs = [i for i in idxs if content[i].get("user")]
		target = (user_idxs or idxs)
		if target:
			content[target[-1]]["text"] = text
			content[target[-1]]["type"] = str(content[target[-1]].get("type") or "text")
		else:
			content.append({"type": "text", "text": text, "user": True})
		return msg
	msg["content"] = text
	return msg


def ApplyEditLastUser(messages, text, user_name="user"):
	"""Replace last typed user text and drop every turn after it.

	Returns a new list (same message objects up to the user), or None.
	"""
	text = str(text or "").replace("\r", "").strip()
	if not text:
		return None
	idx = LastEditableUserIndex(messages, user_name)
	if idx < 0:
		return None
	SetUserMessageText(messages[idx], text)
	return list(messages[: idx + 1])
