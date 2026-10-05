"""ParseToolCallInner: JSON vs XML by first non-whitespace char; XML value unwrap."""

import importlib.util
import os
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..")
if _ENGINE not in sys.path:
	sys.path.insert(0, _ENGINE)

from lib.tool_calls import (
	CoerceArgsToSchema,
	ParseToolCallInner,
	_CoerceArgValue,
	_UnwrapXmlValue,
)


def test_first_char_json_object():
	calls = ParseToolCallInner('\n{"name": "glob_files", "arguments": {"pattern": "*.py"}}\n')
	assert len(calls) == 1
	assert calls[0]["name"] == "glob_files"
	assert calls[0]["arguments"]["pattern"] == "*.py"


def test_first_char_json_array():
	raw = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
	calls = ParseToolCallInner(raw)
	assert [c["name"] for c in calls] == ["a", "b"]


def test_first_char_xml():
	raw = """
<function=write_file>
<parameter=path>
src/a.py
</parameter>
<parameter=content>
print(1)
</parameter>
</function>
"""
	calls = ParseToolCallInner(raw)
	assert calls[0]["name"] == "write_file"
	assert calls[0]["arguments"]["path"] == "src/a.py"
	assert calls[0]["arguments"]["content"] == "print(1)"


def test_broken_json_is_not_xml():
	assert ParseToolCallInner('{"name": ') == []


def test_json_file_body_stays_string():
	raw = """<function=write_file>
<parameter=content>
{"a": 1}
</parameter>
</function>"""
	calls = ParseToolCallInner(raw)
	assert calls[0]["arguments"]["content"] == '{"a": 1}'


def test_xml_bool_and_number_coerce():
	raw = """<function=grep_files>
<parameter=pattern>x</parameter>
<parameter=case_insensitive>true</parameter>
<parameter=max_results>8</parameter>
</function>"""
	args = ParseToolCallInner(raw)[0]["arguments"]
	assert args["case_insensitive"] is True
	assert args["max_results"] == 8


def test_unwrap_one_newline_each_side():
	assert _UnwrapXmlValue("\nvalue_1\n") == "value_1"
	assert _UnwrapXmlValue("value_1") == "value_1"
	assert _UnwrapXmlValue("\nline1\nline2\n") == "line1\nline2"
	assert _UnwrapXmlValue("\nline1\nline2\n\n") == "line1\nline2\n"
	assert _UnwrapXmlValue("  spaced  ") == "  spaced  "


def test_coerce_does_not_strip_inner_spaces():
	assert _CoerceArgValue("\n  indented\n") == "  indented"


def test_xml_json_array_stays_string_until_schema():
	raw = """<function=glob_files>
<parameter=pattern>**/*.py</parameter>
<parameter=exclude>["vendor"]</parameter>
</function>"""
	args = ParseToolCallInner(raw)[0]["arguments"]
	assert args["exclude"] == '["vendor"]'
	schema = {
		"type": "object",
		"properties": {
			"pattern": {"type": "string"},
			"exclude": {"type": "array", "items": {"type": "string"}},
		},
	}
	got = CoerceArgsToSchema(args, schema)
	assert got["exclude"] == ["vendor"]
	assert got["pattern"] == "**/*.py"


def test_schema_coerce_leaves_json_file_body():
	args = {"path": "a.json", "content": '{"a": 1}'}
	schema = {
		"properties": {
			"path": {"type": "string"},
			"content": {"type": "string"},
		},
	}
	got = CoerceArgsToSchema(args, schema)
	assert got["content"] == '{"a": 1}'


def test_chat_history_query_accepts_json_array_string():
	path = os.path.join(_ENGINE, "lib", "native_tools", "utils_pack.py")
	spec = importlib.util.spec_from_file_location("utils_pack_query_test", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	terms, err = mod._norm_query('["compact", "pin"]')
	assert err is None
	assert terms == ["compact", "pin"]
	terms, err = mod._norm_query("compact")
	assert err is None
	assert terms == ["compact"]


def test_chat_history_query_ignores_role_matches_tool_name():
	path = os.path.join(_ENGINE, "lib", "native_tools", "utils_pack.py")
	spec = importlib.util.spec_from_file_location("utils_pack_query_hit_test", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	user = {"role": "user", "content": "hello"}
	assert mod._query_hits(user, ["user"]) is False
	assert mod._query_hits(user, ["hello"]) is True
	call = {
		"role": "assistant",
		"content": "",
		"tool_calls": [{"function": {"name": "glob_files", "arguments": "{}"}}],
	}
	assert mod._query_hits(call, ["glob_files"]) is True
	named = {"role": "tool", "name": "grep_files", "content": "ok"}
	assert mod._query_hits(named, ["grep_files"]) is True


def test_utils_call_tool_errors_are_dicts():
	path = os.path.join(_ENGINE, "lib", "native_tools", "utils_pack.py")
	spec = importlib.util.spec_from_file_location("utils_pack_err_test", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	got = mod.call_tool("not_a_tool")
	assert isinstance(got, dict)
	assert "error" in got
	assert not str(got["error"]).startswith("ERROR:")
