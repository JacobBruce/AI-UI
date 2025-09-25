from . import search_funcs, math_funcs, tool_funcs

# FUNCTIONS FOR AIUI ENGINE

def CallToolFunc(func_name, func_args, aiui_funcs):
	if hasattr(search_funcs, func_name):
		return search_funcs.CallToolFunc(func_name, func_args, aiui_funcs)
	elif hasattr(math_funcs, func_name):
		return math_funcs.CallToolFunc(func_name, func_args, aiui_funcs)
	elif hasattr(tool_funcs, func_name):
		return tool_funcs.CallToolFunc(func_name, func_args, aiui_funcs)
	else:
		return "ERROR: unknown function"

def GetToolFuncs():
	return search_funcs.GetToolFuncs() + math_funcs.GetToolFuncs() + tool_funcs.GetToolFuncs()
