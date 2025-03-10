import math 

# MATH TOOL FUNCTIONS

def mul(a: float, b: float) -> float:
	"""
	Performs multiplication on two numbers
    
	Args:
		a: The number on the left side of the operator
		b: The number on the right side of the operator
	"""
	return a * b

def sum(a: float, b: float) -> float:
	"""
	Performs addition on two numbers
    
	Args:
		a: The number on the left side of the operator
		b: The number on the right side of the operator
	"""
	return a + b

def sub(a: float, b: float) -> float:
	"""
	Performs subtraction on two numbers
    
	Args:
		a: The number on the left side of the operator
		b: The number on the right side of the operator
	"""
	return a - b

def div(a: float, b: float) -> float:
	"""
	Performs division on two numbers
    
	Args:
		a: The number on the left side of the operator
		b: The number on the right side of the operator
	"""
	return a / b

def mod(a: float, b: float) -> float:
	"""
	Performs modulo division on two numbers
    
	Args:
		a: The number on the left side of the operator
		b: The number on the right side of the operator
	"""
	return math.fmod(a, b)

def pow(a: float, b: float) -> float:
	"""
	Returns the base number raised to a power
    
	Args:
		a: The base number
		b: The exponent number
	"""
	return math.pow(a, b)

def min(a: float, b: float) -> float:
	"""
	Returns the smallest of two numbers
    
	Args:
		a: The first number
		b: The second number
	"""
	return a if a < b else b

def max(a: float, b: float) -> float:
	"""
	Returns the largest of two numbers
    
	Args:
		a: The first number
		b: The second number
	"""
	return a if a > b else b

def abs(x: float) -> float:
	"""
	Returns the absolute value of a number
    
	Args:
		x: The input number
	"""
	return math.fabs(x)

def exp(x: float) -> float:
	"""
	Returns e raised to the power of a number
    
	Args:
		x: The input number
	"""
	return math.exp(x)

def sin(x: float) -> float:
	"""
	Returns the sine of a number in radians
    
	Args:
		x: The input number
	"""
	return math.sin(x)

def cos(x: float) -> float:
	"""
	Returns the cosine of a number in radians
    
	Args:
		x: The input number
	"""
	return math.cos(x)

def tan(x: float) -> float:
	"""
	Returns the tangent of a number in radians
    
	Args:
		x: The input number
	"""
	return math.tan(x)

def sqrt(x: float) -> float:
	"""
	Returns the square root of a number
    
	Args:
		x: The input number
	"""
	return math.sqrt(x)

def rnd(x: float) -> float:
	"""
	Returns a number rounded to the nearest integer
    
	Args:
		x: The input number
	"""
	return round(x)

def floor(x: float) -> float:
	"""
	Returns the largest integer less than or equal to a number
    
	Args:
		x: The input number
	"""
	return math.floor(x)

def ceil(x: float) -> float:
	"""
	Returns the smallest integer greater than or equal to a number
    
	Args:
		x: The input number
	"""
	return math.ceil(x)

def log(x: float) -> float:
	"""
	Returns the natural logarithm of a number
    
	Args:
		x: The input number
	"""
	return math.log(x)

def log2(x: float) -> float:
	"""
	Returns the base 2 logarithm of a number
    
	Args:
		x: The input number
	"""
	return math.log2(x)

def log10(x: float) -> float:
	"""
	Returns the base 10 logarithm of a number
    
	Args:
		x: The input number
	"""
	return math.log10(x)

# FUNCTIONS FOR AIUI ENGINE

def CallToolFunc(func_name, func_args, aiui_funcs):
	try:
		if 'a' in func_args and isinstance(func_args['a'], str):
			func_args['a'] = float(func_args['a'])
		if 'b' in func_args and isinstance(func_args['b'], str):
			func_args['b'] = float(func_args['b'])
		if 'x' in func_args and isinstance(func_args['x'], str):
			func_args['x'] = float(func_args['x'])
		if func_name == "mul":
			return mul(func_args['a'], func_args['b'])
		elif func_name == "sum":
			return sum(func_args['a'], func_args['b'])
		elif func_name == "sub":
			return sub(func_args['a'], func_args['b'])
		elif func_name == "div":
			return div(func_args['a'], func_args['b'])
		elif func_name == "mod":
			return mod(func_args['a'], func_args['b'])
		elif func_name == "pow":
			return pow(func_args['a'], func_args['b'])
		elif func_name == "min":
			return min(func_args['a'], func_args['b'])
		elif func_name == "max":
			return max(func_args['a'], func_args['b'])
		elif func_name == "abs":
			return abs(func_args['x'])
		elif func_name == "exp":
			return exp(func_args['x'])
		elif func_name == "sin":
			return sin(func_args['x'])
		elif func_name == "cos":
			return cos(func_args['x'])
		elif func_name == "tan":
			return tan(func_args['x'])
		elif func_name == "sqrt":
			return sqrt(func_args['x'])
		elif func_name == "rnd":
			return rnd(func_args['x'])
		elif func_name == "floor":
			return floor(func_args['x'])
		elif func_name == "ceil":
			return ceil(func_args['x'])
		elif func_name == "log":
			return log(func_args['x'])
		elif func_name == "log2":
			return log2(func_args['x'])
		elif func_name == "log10":
			return log10(func_args['x'])
		else:
			return "ERROR: unknown function"
	except:
		return "ERROR: invalid argument"

def GetToolFuncs():
	return [mul, sum, sub, div, mod, pow, min, max, abs, exp, sin, cos, tan, sqrt, rnd, floor, ceil, log, log2, log10]
