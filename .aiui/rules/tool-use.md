# Tool calling

You may use `<tool_call>` `</tool_call>` tags to call tool functions when necessary.

Optional natural-language reasoning may appear **before** a tool call.

Do not put commentary after `</tool_call>` in the same reply - wait for the tool response(s), then continue.

## JSON or XML

The form is the first non-whitespace character inside `<tool_call>`: `{` or `[` is JSON, `<` is XML.

**JSON** for short or simple arguments (paths, flags, small patterns). JSON argument strings must be valid JSON (`\"`, `\\`, newlines as `\n`).

**XML** when an argument is long or multiline (file bodies, patches, including JSON files). Parameter text is copied as-is: real newlines, quotes, braces, and backslashes, with no escaping.

Do not put `</tool_call>` in any argument; it always ends the call. Also do not put `</parameter>` or `</function>` in a value when writing XML-style tool calls.

### XML parameter whitespace

One newline right after the opening tag and one newline right before the closing tag are ignored (so the usual multiline layout works).

Every other character is part of the value, including any whitespace. Do not add extra indent just to nest the text under the XML tags.

```
<parameter=path>
src/a.py
</parameter>
```

That is the string `src/a.py`. To keep a trailing newline in a file body, leave a blank line before `</parameter>`.

`true`, `false`, `null`, and bare numbers are parsed as those types. Inline is also fine: `<parameter=ascending>true</parameter>`.

## When to use one function vs many

**Prefer a single-function tool call** when steps must run in order or later work depends on an earlier result.

**Only call multiple functions in one `<tool_call>` when it is safe to run them at the same time.** Safe means:

- The calls do not depend on each other's outputs
- They do not fight over the same file, resource, or mutable state
- Order of completion does not change correctness (the engine may run them in parallel)

If you are unsure, use one function, wait for `<tool_response>`, then call the next.

## JSON object (one function)

```
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

## JSON array (multiple functions)

Use this only for independent, parallel-safe work:

```
<tool_call>
[
  {"name": <function-name>, "arguments": <args-json-object>},
  {"name": <function-name>, "arguments": <args-json-object>}
]
</tool_call>
```

## XML function / parameter (one or more functions)

Same safety rule: multiple `<function>` blocks only when parallel execution is safe.

```
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for
the second parameter
</parameter>
</function>
<function=other_function_name>
<parameter=x>
1
</parameter>
</function>
</tool_call>
```

## Tool response

Each function result is returned as:

```
<tool_response>{"name": <function-name>, "content": <response-string>}</tool_response>
```

When several functions run in one turn, you receive one `<tool_response>` per function. Read every response before deciding the next step.
