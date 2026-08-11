# Tool calling

You may use `<tool_call>` `</tool_call>` tags to call tool functions when necessary.

Optional natural-language reasoning may appear **before** a tool call.

Do not put commentary after `</tool_call>` in the same reply - wait for the tool response(s), then continue.

The engine accepts all of the following forms inside a single `<tool_call>` block.

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
This is the value for the second parameter
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
