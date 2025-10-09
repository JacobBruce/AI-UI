import time
from datetime import datetime

AIUI_FUNCS = None
IMG_FILE = ''

# CUSTOM TOOL FUNCTIONS

"""
Some tool functions are already implemented below to demonstrate how they are created.
You can replace these functions with your own or add to the existing tool functions.
"""

def gen_image(prompt: str, neg_prompt: str="NONE", infer_steps: int=50, guidance: float=7.5):
	"""
	Generates an image from a prompt using AI and displays it to the user.
    
	Args:
		prompt: The image prompt, usually a brief description of the image.
		neg_prompt: The negative prompt, useful for reducing visual issues. Optional.
		infer_steps: The number of inference steps. Optional (default=50, max=100)
		guidance: The prompt guidance scale. Optional (default=7.5)
	"""
	global IMG_FILE
	IMG_FILE = AIUI_FUNCS['GenImage'](prompt, neg_prompt, max(0, min(100, infer_steps)), abs(guidance))

def utc_date() -> str:
	"""Get the current UTC date as a string."""
	return '{dt.year}-{dt.month}-{dt.day}'.format(dt = datetime.utcnow())

def utc_time() -> str:
	"""Get the current UTC time as a string."""
	return datetime.utcnow().strftime("%H:%M:%S")

def local_date() -> str:
	"""Get the current date as a string."""
	return '{dt.year}-{dt.month}-{dt.day}'.format(dt = datetime.now())

def local_time() -> str:
	"""Get the current time as a string."""
	return datetime.now().strftime("%H:%M:%S")

def local_timezone() -> str:
	"""Get the name of the local timezone as a string."""
	return time.tzname[time.daylight]

# FUNCTIONS FOR AIUI ENGINE

def CallToolFunc(func_name, func_args, aiui_funcs):
	global AIUI_FUNCS
	try:
		AIUI_FUNCS = aiui_funcs
		if func_name == "gen_image":
			gen_image(**func_args)
			return IMG_FILE
		elif func_name == "utc_date":
			return utc_date()
		elif func_name == "utc_time":
			return utc_time()
		elif func_name == "local_date":
			return local_date()
		elif func_name == "local_time":
			return local_time()
		elif func_name == "local_timezone":
			return local_timezone()
		else:
			return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"

def GetToolFuncs():
	return [gen_image, utc_date, utc_time, local_date, local_time, local_timezone]
