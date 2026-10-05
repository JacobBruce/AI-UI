"""Edit-last-user: skip continue stubs, keep media, drop trailing turns."""

import os
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..")
if _ENGINE not in sys.path:
	sys.path.insert(0, _ENGINE)

from lib import edit_last as el


def test_last_user_plain():
	msgs = [
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "one"},
		{"role": "assistant", "content": "a1"},
		{"role": "user", "content": "two"},
		{"role": "assistant", "content": "a2"},
	]
	out = el.ApplyEditLastUser(msgs, "two edited")
	assert out is not None
	assert len(out) == 4
	assert out[-1]["content"] == "two edited"
	assert out[-2]["content"] == "a1"


def test_skip_continue_stub_and_tools():
	msgs = [
		{"role": "user", "content": "do it"},
		{"role": "assistant", "content": "call"},
		{"role": "tool", "content": "{}"},
		{"role": "assistant", "content": "done"},
		{"role": "user", "content": "continue"},
		{"role": "assistant", "content": "more"},
	]
	out = el.ApplyEditLastUser(msgs, "do it now")
	assert [m["role"] for m in out] == ["user"]
	assert out[0]["content"] == "do it now"


def test_keep_compact_before_user():
	msgs = [
		{"role": "user", "content": "old"},
		{"role": "compact", "content": "sum", "_aiui_compact": True},
		{"role": "user", "content": "new"},
		{"role": "assistant", "content": "ans"},
	]
	out = el.ApplyEditLastUser(msgs, "newer")
	assert [m["role"] for m in out] == ["user", "compact", "user"]
	assert out[-1]["content"] == "newer"


def test_drop_compact_after_user():
	msgs = [
		{"role": "user", "content": "ask"},
		{"role": "compact", "content": "sum", "_aiui_compact": True},
		{"role": "assistant", "content": "ans"},
	]
	out = el.ApplyEditLastUser(msgs, "ask again")
	assert [m["role"] for m in out] == ["user"]
	assert out[0]["content"] == "ask again"


def test_multimodal_replaces_last_text_keeps_image():
	msg = {
		"role": "user",
		"content": [
			{"type": "image", "image": "/tmp/a.png"},
			{"type": "text", "text": "what is this", "user": True},
		],
	}
	el.SetUserMessageText(msg, "describe it")
	assert msg["content"][0]["image"] == "/tmp/a.png"
	assert msg["content"][1]["text"] == "describe it"


def test_empty_or_missing_returns_none():
	assert el.ApplyEditLastUser([], "x") is None
	assert el.ApplyEditLastUser([{"role": "assistant", "content": "a"}], "x") is None
	assert el.ApplyEditLastUser([{"role": "user", "content": "hi"}], "  ") is None
