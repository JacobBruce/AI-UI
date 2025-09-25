import time
from datetime import datetime

# CUSTOM TOOL FUNCTIONS

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
	try:
		if func_name == "utc_date":
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
	return [utc_date, utc_time, local_date, local_time, local_timezone]
