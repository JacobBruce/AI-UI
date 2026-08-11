"""Native Math pack - asteval-backed calculation tools."""

PACK_ID = "native:math"
PACK_NAME = "Math"
PACK_KIND = "native"

_aeval = None

TOOLS = [
	{
		"name": "reset_aeval",
		"description": "Resets the asteval interpreter, clearing the symbol table.",
		"inputSchema": {
			"type": "object",
			"properties": {},
			"additionalProperties": False,
		},
	},
	{
		"name": "get_value",
		"description": (
			"Returns the value of a variable in the asteval symbol table as a string. "
			"Returns an error string if the symbol name does not exist."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"sym_name": {
					"type": "string",
					"description": "A symbol name. The value of that variable will be returned if it exists.",
				},
			},
			"required": ["sym_name"],
			"additionalProperties": False,
		},
	},
	{
		"name": "aeval",
		"description": (
			"Safely evaluates Python-like code using asteval. Useful for solving math-related problems. "
			"Many functions from the math and numpy modules are available. "
			"These features are disabled: importing, printing, while loops."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"expression": {
					"type": "string",
					"description": "The string of Python-like code to be evaluated.",
				},
				"sym_name": {
					"type": "string",
					"description": "A symbol name. Forces the value of that variable to be returned. Optional.",
					"default": "",
				},
				"reset": {
					"type": "boolean",
					"description": "If true the interpreter is reset before evaluating the expression. Optional (default=False).",
					"default": False,
				},
			},
			"required": ["expression"],
			"additionalProperties": False,
		},
	},
]

def _EnsureAeval():
	global _aeval
	if _aeval is not None:
		return _aeval
	from asteval import Interpreter
	_aeval = Interpreter(use_numpy=True, no_while=True, no_print=True)
	return _aeval

def _reset_aeval():
	global _aeval
	from asteval import Interpreter
	_aeval = Interpreter(use_numpy=True, no_while=True, no_print=True)
	return "OK"

def _get_value(sym_name: str) -> str:
	ae = _EnsureAeval()
	if sym_name in ae.symtable:
		val = ae.symtable[sym_name]
		return val if isinstance(val, str) else str(val)
	return "ERROR: symbol not found"

def _aeval_expr(expression: str, sym_name: str = "", reset: bool = False) -> str:
	if reset:
		_reset_aeval()
	ae = _EnsureAeval()
	result = ae(expression)
	if len(ae.error) > 0:
		return "ERROR: " + str(ae.error_msg or "unknown error")
	if sym_name and sym_name in ae.symtable:
		val = ae.symtable[sym_name]
		return val if isinstance(val, str) else str(val)
	if isinstance(result, str):
		return result
	return str(result)

def configure(_cfg=None):
	"""No pack-level config for Math."""
	return None

def list_tools():
	return list(TOOLS)

def call_tool(name, arguments=None):
	arguments = arguments or {}
	try:
		if name == "reset_aeval":
			return _reset_aeval()
		if name == "get_value":
			return _get_value(**arguments)
		if name == "aeval":
			return _aeval_expr(**arguments)
		return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"
