import sys, os, time
from subprocess import Popen

exec_path = sys.executable.replace('\\', '/')
python_path = exec_path[:exec_path.rfind('/')]
os.environ["PATH"] = python_path + os.pathsep + os.environ.get("PATH", "")

cmd_str = input("AIUI_CMD:")

try:
	os.chdir(input("AIUI_CWD:"))
except:
	print("AIUI_CWD:"+os.getcwd(), flush=True)
	sys.exit("ERROR: invalid working directory specified.")

print("AIUI_CWD:"+os.getcwd(), flush=True)
time.sleep(0.1)

if cmd_str != "cd":
	try:
		# shell=True so the console can run shell commands; cwd already set via os.chdir.
		# (Previously shell=sys.executable — a truthy string — which accidentally meant shell=True.)
		proc = Popen(cmd_str, shell=True, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, text=True)
		proc.wait()
	except:
		print("ERROR: failed to run the command shell process.")
