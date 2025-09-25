import math
from asteval import Interpreter

aeval_interpreter = Interpreter(use_numpy=True, no_while=True, no_print=True)

# MATH TOOL FUNCTIONS
#TODO: add sympy functions to asteval

def reset_aeval():
	"""Resets the asteval interpreter, clearing the symbol table."""
	global aeval_interpreter
	aeval_interpreter = Interpreter(use_numpy=True, no_while=True, no_print=True)

def get_value(sym_name: str) -> str:
	"""
	Returns the value of a variable in the asteval symbol table as a string.
	Returns an error string if the symbol name does not exist.
    
	Args:
		sym_name: A symbol name. The value of that variable will be returned if it exists.
	"""
	if sym_name in aeval_interpreter.symtable:
		if type(aeval_interpreter.symtable[sym_name]) is str:
			return aeval_interpreter.symtable[sym_name]
		else:
			return str(aeval_interpreter.symtable[sym_name])
	else:
		return "ERROR: symbol not found"

def aeval(expression: str, sym_name: str='', reset: bool=False) -> str:
	"""
	Safely evaluates Python-like code using asteval.
	Useful for solving math-related problems.
	Many functions from the math and numpy modules are available.
	These features are disabled: importing, printing, while loops.
    
	Args:
		expression: The string of Python-like code to be evaluated.
		sym_name: A symbol name. Forces the value of that variable to be returned. Optional.
		reset: If true the interpreter is reset before evaluating the expression. Optional (default=False)
	"""
	if reset: reset_aeval()
	
	result = aeval_interpreter(expression)
	
	if len(aeval_interpreter.error) > 0:
		return "ERROR: " + aeval_interpreter.error_msg
	elif sym_name != '' and sym_name in aeval_interpreter.symtable:
		if type(aeval_interpreter.symtable[sym_name]) is str:
			return aeval_interpreter.symtable[sym_name]
		else:
			return str(aeval_interpreter.symtable[sym_name])
	elif type(result) is str:
		return result
	else:
		return str(result)

# FUNCTIONS FOR AIUI ENGINE

def CallToolFunc(func_name, func_args, aiui_funcs):
	try:
		if func_name == "aeval":
			return aeval(**func_args)
		elif func_name == "get_value":
			return get_value(**func_args)
		elif func_name == "reset_aeval":
			return reset_aeval()
		else:
			return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"

def GetToolFuncs():
	return [reset_aeval, get_value, aeval]
