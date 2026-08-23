# Bound to aiui_engine module (shared mutable engine state).
E = None

def bind(engine_module):
	global E
	E = engine_module

import os
import sys
import time
import json
import gc
import glob
import random
import re
import threading
import tempfile
import importlib
import wave

def CountFiles(dir_path):
	count = 0
	for path in os.listdir(dir_path):
		if os.path.isfile(os.path.join(dir_path, path)):
			count += 1
	return count


def NewFileName(dir_path, name_str):
	if not os.path.exists(dir_path): os.mkdir(dir_path)
	file_num = CountFiles(dir_path)
	file_name = name_str.replace("#NUM", str(file_num))
	while os.path.isfile(dir_path+file_name):
		file_num += 1
		file_name = name_str.replace("#NUM", str(file_num))
	return { 'num': file_num, 'file': dir_path+file_name }


def SetWorkingDir(face_anim_mode):
	root = str(getattr(E, "work_dir", "") or "").rstrip("/\\")
	mode = int(face_anim_mode)
	if mode < 0:
		if root:
			os.chdir(root)
		return
	if mode == 0:
		os.chdir(os.path.join(root, "MakeItTalk"))
	elif mode == 1:
		os.chdir(os.path.join(root, "Wav2Lip"))
	elif mode == 2:
		os.chdir(os.path.join(root, "SadTalker"))
	else:
		os.chdir(os.path.join(root, "IMTalker"))


def ReadTextFile(txt_file, txt_encoding='utf-8'):
	if os.path.isfile(txt_file):
		f = open(txt_file, "rb")
		result = f.read().decode(txt_encoding)
		f.close()
		return result
	else:
		return None


def WriteTextFile(txt_file, txt_str, txt_encoding='utf-8'):
	f = open(txt_file, "wb")
	f.write(txt_str.encode(txt_encoding))
	f.close()


def SafeString(txt):
	"""Escape markdown/HTML-sensitive chars for chat display.

	Placeholders are AI_UI_C{codepoint}; (no #/&). The renderer turns them into
	&#N; after EncodeHTML/markdown so they are not double-escaped. EncodeHTML
	itself rewrites # → &num;, so AI_UI_AMP#N; placeholders cannot work.
	"""
	result = str(txt or '')
	# '_' first so later placeholders may contain underscores safely (one-pass replace).
	result = result.replace('_', 'AI_UI_C95;')
	unsafe_chars = {
		'#': 35, '$': 36, '+': 43, '-': 45, '~': 126, '.': 46, '*': 42,
		'`': 96, '!': 33, '|': 124, '(': 40, ')': 41, '[': 91, ']': 93,
		'<': 60, '>': 62,
	}
	for ch, code in unsafe_chars.items():
		result = result.replace(ch, 'AI_UI_C'+str(code)+';')
	return result


def FormatThinkMarkers(inner_txt):
	"""Wrap think body for the chat UI.

	A single leading newline becomes <br> under markdown breaks:true and glues
	the first paragraph to the <hr>; normalize to a paragraph break instead.
	"""
	import re
	body = re.sub(r'^\n+', '', str(inner_txt or '')).rstrip()
	if not body:
		return ''
	return 'AIUI_THINK_START\n\n' + SafeString(body) + 'AIUI_THINK_END'


def FormatToolMarkers(inner_txt, kind="tc"):
	"""Wrap tool call/response body for the chat UI (pretty-print/HTML is client-side).

	Do not SafeString here — tool XML may already be SafeString'd when built, and
	the renderer undoes placeholders before JSON.parse / EncodeHTML.
	"""
	body = str(inner_txt or '').strip()
	if kind == "tr":
		return 'AIUI_TR_START' + body + 'AIUI_TR_END'
	return 'AIUI_TC_START' + body + 'AIUI_TC_END'


def StripEmptyThinkTags(txt):
	"""Remove empty / whitespace-only think blocks (raw XML or AIUI markers)."""
	s = str(txt or '')
	prev = None
	while prev != s:
		prev = s
		s = re.sub(r'<(think|thoughts)>\s*</\1>', '', s, flags=re.IGNORECASE)
		s = re.sub(r'AIUI_THINK_START\s*AIUI_THINK_END', '', s)
	return CollapseDuplicateThinkBlocks(s)


_THINK_BLOCK_RE = re.compile(
	r"<(think|thoughts)>(.*?)</\1>",
	re.IGNORECASE | re.DOTALL,
)


def CollapseDuplicateThinkBlocks(txt):
	"""Merge adjacent <think> copies (native reasoning wrap + the same XML in content)."""
	s = str(txt or "")
	while True:
		blocks = list(_THINK_BLOCK_RE.finditer(s))
		changed = False
		for i in range(len(blocks) - 1):
			a, b = blocks[i], blocks[i + 1]
			if s[a.end():b.start()].strip():
				continue
			na = re.sub(r"\s+", " ", a.group(2)).strip()
			nb = re.sub(r"\s+", " ", b.group(2)).strip()
			if not (na and nb and (na == nb or na.startswith(nb) or nb.startswith(na))):
				continue
			keep = b.group(0) if len(nb) > len(na) else a.group(0)
			s = s[:a.start()] + keep + s[b.end():]
			changed = True
			break
		if not changed:
			return s


_OPEN_THINK_TAGS = (("<think>", "</think>"), ("<thoughts>", "</thoughts>"))


def CloseOrUnwrapOpenThink(txt):
	"""Seal or drop a trailing unclosed `<think>` / `<thoughts>` wrapper.

	Native reasoning is wrapped in these tags for the UI. If the stream ends
	(or Stop) with no close tag and no tool XML after the open tag, unwrap so
	the body is the visible reply. If a tool block sits inside the open think,
	insert the close tag before that block.
	"""
	s = str(txt or "")
	if not s:
		return s
	low = s.lower()
	best = None
	for open_t, close_t in _OPEN_THINK_TAGS:
		pos = 0
		lo, lc = open_t.lower(), close_t.lower()
		while True:
			i = low.find(lo, pos)
			if i < 0:
				break
			j = low.find(lc, i + len(open_t))
			if j < 0:
				if best is None or i < best[2]:
					best = (open_t, close_t, i)
				break
			pos = j + len(close_t)
	if best is None:
		return s
	open_t, close_t, at = best
	insert_at = None
	for marker in ("<tool_call>", "<tool_response>"):
		k = s.find(marker, at)
		if k >= 0 and (insert_at is None or k < insert_at):
			insert_at = k
	if insert_at is not None:
		head = s[:insert_at].rstrip()
		return head + "\n" + close_t + "\n\n" + s[insert_at:]
	body = s[at + len(open_t):]
	if body.startswith("\n"):
		body = body[1:]
	return s[:at] + body


# Shared EOS / turn closers: trailing trim (display) and local ID fallback when
# primary eos_token_id is missing. Mid-message mentions are left alone by trim.
EOS_MARKERS = ("<|endoftext|>", "<|role_end|>", "<|turn_end|>")


def TrimTrailingChatClosers(txt):
	"""Strip known EOS markers only when they are a suffix of the reply."""
	s = str(txt or "")
	while True:
		t = s.rstrip()
		cut = None
		for marker in EOS_MARKERS:
			if t.endswith(marker):
				cut = marker
				break
		if cut is None:
			return t
		s = t[: -len(cut)]


def StripEnd(txt, ss):
	return txt[:-len(ss)] if txt.endswith(ss) else txt


def StripStart(txt, ss):
	return txt[len(ss):] if txt.startswith(ss) else txt
