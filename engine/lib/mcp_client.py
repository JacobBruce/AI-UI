"""Long-lived MCP stdio client sessions for AI UI tool packs.

Sessions run inside a dedicated asyncio loop thread. Each server is owned by a
lifecycle task that enters stdio_client / ClientSession context managers and
waits on a hold event - closing sets the event so contexts exit naturally.
Calling AsyncExitStack.aclose() from outside those tasks left anyio CancelScope
in a tight _deliver_cancellation loop (~100% of one core).
"""

import asyncio
import os
import threading
import time

_loop = None
_loop_thread = None
_sessions = {}  # pack_id -> {session, task, hold, tools, schemas, cfg}
_schema_cache = {}  # pack_id -> [schema, ...] last known tools
_errors = {}  # pack_id -> last connect error string
_lock = threading.Lock()


class _PrefixedErrlog:
	"""Pipe-backed stderr target; prefixes each line for the Console tab.

	Subprocess stderr is bound to the pipe write fd (write() is not used), so a
	reader thread is required to label lines as they arrive.
	"""

	def __init__(self, prefix):
		self._prefix = str(prefix or "[MCP] ")
		r_fd, w_fd = os.pipe()
		self._read_fd = r_fd
		self._write_fd = w_fd
		self._closed = False
		self._thread = threading.Thread(target=self._pump, name="mcp-errlog", daemon=True)
		self._thread.start()

	def fileno(self):
		return self._write_fd

	def write(self, data):
		if self._closed:
			return 0
		if isinstance(data, str):
			data = data.encode("utf-8", errors="replace")
		return os.write(self._write_fd, data)

	def flush(self):
		return None

	def close(self):
		if self._closed:
			return
		self._closed = True
		try:
			os.close(self._write_fd)
		except Exception:
			pass
		self._thread.join(timeout=2)
		try:
			os.close(self._read_fd)
		except Exception:
			pass

	def _pump(self):
		from lib import ipc
		buf = b""
		try:
			while True:
				chunk = os.read(self._read_fd, 4096)
				if not chunk:
					break
				buf += chunk
				while b"\n" in buf:
					line, buf = buf.split(b"\n", 1)
					text = line.decode("utf-8", errors="replace").rstrip("\r")
					if text:
						ipc.LogErr(self._prefix + text)
			if buf.strip():
				ipc.LogErr(self._prefix + buf.decode("utf-8", errors="replace").rstrip())
		except Exception:
			pass


def _mcp_log_prefix(cfg):
	name = str((cfg or {}).get("name") or (cfg or {}).get("id") or "mcp").strip() or "mcp"
	# Keep console labels short and stable.
	if name.startswith("mcp:"):
		name = name[4:]
	return f"[MCP:{name}] "

def _ensure_loop():
	global _loop, _loop_thread
	if _loop is not None and _loop.is_running():
		return _loop

	ready = threading.Event()

	def _run():
		global _loop
		_loop = asyncio.new_event_loop()
		asyncio.set_event_loop(_loop)
		ready.set()
		_loop.run_forever()

	_loop_thread = threading.Thread(target=_run, name="aiui-mcp-loop", daemon=True)
	_loop_thread.start()
	ready.wait(timeout=5)
	deadline = time.time() + 5
	while _loop is None or not _loop.is_running():
		if time.time() > deadline:
			raise RuntimeError("MCP event loop failed to start")
		time.sleep(0.01)
	return _loop

def _shutdown_loop():
	"""Stop the MCP loop thread so a poisoned anyio CancelScope cannot linger."""
	global _loop, _loop_thread
	loop = _loop
	thread = _loop_thread
	_loop = None
	_loop_thread = None
	if loop is None:
		return
	try:
		loop.call_soon_threadsafe(loop.stop)
	except Exception:
		pass
	if thread is not None and thread.is_alive():
		thread.join(timeout=5)
	try:
		loop.close()
	except Exception:
		pass

def _await(coro, timeout=120):
	loop = _ensure_loop()
	fut = asyncio.run_coroutine_threadsafe(coro, loop)
	return fut.result(timeout=timeout)

def _tool_to_schema(tool):
	schema = getattr(tool, "inputSchema", None) or {}
	if not isinstance(schema, dict):
		schema = {}
	return {
		"name": tool.name,
		"description": getattr(tool, "description", None) or "",
		"inputSchema": schema,
	}

def ServerKey(pack_id):
	"""mcp:cait → cait (used as tool-name prefix)."""
	s = str(pack_id or "").strip()
	if s.startswith("mcp:"):
		s = s[4:]
	return s.strip() or "mcp"

def SanitizeId(server_id):
	"""Safe prefix for tool names (alnum + underscore)."""
	parts = []
	for ch in str(server_id or ""):
		if ch.isalnum() or ch == "_":
			parts.append(ch)
		else:
			parts.append("_")
	out = "".join(parts).strip("_")
	return out or "mcp"

def PrefixedToolName(pack_id, tool_name):
	"""Expose MCP tools as <server_id>_<tool> (e.g. cait_read_file)."""
	bare = str(tool_name or "").strip()
	return SanitizeId(ServerKey(pack_id)) + "_" + bare

def _schemas_for_pack(pack_id, tools):
	"""Build HF/UI schemas with disambiguated names; keep bare name for MCP call_tool."""
	out = []
	for tool in tools or []:
		item = _tool_to_schema(tool)
		bare = item.get("name") or ""
		if not bare:
			continue
		item["_mcp_name"] = bare
		item["name"] = PrefixedToolName(pack_id, bare)
		item["_pack_id"] = pack_id
		out.append(item)
	return out

def _result_to_text(result):
	if result is None:
		return ""
	if isinstance(result, str):
		return result
	structured = getattr(result, "structuredContent", None)
	if structured is not None:
		import json
		try:
			return json.dumps(structured)
		except Exception:
			pass
	parts = []
	content = getattr(result, "content", None) or []
	for item in content:
		text = getattr(item, "text", None)
		if text is not None:
			parts.append(text)
		else:
			parts.append(str(item))
	text = "\n".join(parts).strip()
	is_error = bool(getattr(result, "isError", False))
	if is_error and text and not text.startswith("ERROR:"):
		return "ERROR: " + text
	return text if text else ("ERROR: empty MCP tool result" if is_error else "")

def _parse_server_params(cfg):
	from mcp.client.stdio import StdioServerParameters

	command = str(cfg.get("command", "") or "").strip()
	if not command:
		raise ValueError("MCP server command is empty")

	args = cfg.get("args") or []
	if isinstance(args, str):
		import json
		args = json.loads(args) if args.strip() else []
	if not isinstance(args, list):
		args = list(args)

	env = cfg.get("env") or {}
	if isinstance(env, str):
		import json
		env = json.loads(env) if env.strip() else {}
	if not isinstance(env, dict):
		env = {}
	env_str = {str(k): str(v) for k, v in env.items()} if env else None

	cwd = cfg.get("cwd") or None
	if isinstance(cwd, str) and not cwd.strip():
		cwd = None

	return StdioServerParameters(
		command=command,
		args=[str(a) for a in args],
		env=env_str,
		cwd=cwd,
	)

async def _session_lifecycle(cfg, hold, ready):
	"""Own stdio + ClientSession until hold is set, then exit contexts cleanly."""
	from mcp import ClientSession
	from mcp.client.stdio import stdio_client

	params = _parse_server_params(cfg)
	errlog = _PrefixedErrlog(_mcp_log_prefix(cfg))
	try:
		async with stdio_client(params, errlog=errlog) as (read, write):
			async with ClientSession(read, write) as session:
				await session.initialize()
				listed = await session.list_tools()
				tools = list(getattr(listed, "tools", None) or [])
				pack_id = str(cfg.get("id", "") or "")
				schemas = _schemas_for_pack(pack_id, tools)
				ready.set_result({
					"session": session,
					"tools": tools,
					"schemas": schemas,
				})
				await hold.wait()
	except Exception as e:
		if not ready.done():
			ready.set_exception(e)
		else:
			# Already handed out the session; swallow exit-time noise.
			pass
	finally:
		errlog.close()

async def _connect_one(cfg):
	hold = asyncio.Event()
	ready = asyncio.get_running_loop().create_future()
	task = asyncio.create_task(_session_lifecycle(cfg, hold, ready), name="mcp-session")
	try:
		info = await asyncio.wait_for(asyncio.shield(ready), timeout=55)
	except Exception:
		hold.set()
		task.cancel()
		try:
			await asyncio.wait_for(task, timeout=10)
		except Exception:
			pass
		raise
	return {
		"session": info["session"],
		"tools": info["tools"],
		"schemas": info["schemas"],
		"cfg": dict(cfg),
		"hold": hold,
		"task": task,
	}

async def _close_session(info):
	hold = info.get("hold")
	task = info.get("task")
	if hold is not None:
		hold.set()
	if task is None:
		return
	try:
		await asyncio.wait_for(task, timeout=30)
	except Exception:
		task.cancel()
		try:
			await asyncio.wait_for(task, timeout=5)
		except Exception:
			pass

def _cache_schemas(pack_id, schemas):
	_schema_cache[pack_id] = [dict(s) for s in (schemas or [])]

def _cfg_same(a, b):
	a = a or {}
	b = b or {}
	return (
		a.get("command") == b.get("command")
		and a.get("args") == b.get("args")
		and a.get("env") == b.get("env")
		and a.get("cwd") == b.get("cwd")
	)

def _close_infos(closing):
	for info in closing:
		try:
			_await(_close_session(info), timeout=35)
		except Exception:
			pass
	# If nothing remains, recycle the loop so a stuck CancelScope cannot burn CPU.
	with _lock:
		empty = not _sessions
	if empty:
		_shutdown_loop()

def close_all():
	with _lock:
		closing = list(_sessions.values())
		_sessions.clear()
		_errors.clear()
	_close_infos(closing)

def running_ids():
	with _lock:
		return list(_sessions.keys())

def error_map():
	with _lock:
		return dict(_errors)

def is_running(pack_id):
	with _lock:
		return pack_id in _sessions

def sync_servers(server_cfgs, force_ids=None):
	"""Connect enabled MCP servers; disconnect anything no longer enabled.

	force_ids: pack ids that must reconnect even if config looks unchanged.
	Does not hold the session lock across awaits (avoids event-loop stalls).
	"""
	force_ids = set(force_ids or [])
	wanted = {}
	for cfg in server_cfgs or []:
		if not cfg.get("enabled"):
			continue
		pack_id = str(cfg.get("id", "") or "")
		if not pack_id:
			continue
		wanted[pack_id] = cfg

	with _lock:
		to_close = []
		for pack_id in list(_sessions.keys()):
			if pack_id not in wanted:
				info = _sessions.pop(pack_id, None)
				if info is not None:
					to_close.append(info)
				_errors.pop(pack_id, None)
		for pack_id in list(_errors.keys()):
			if pack_id not in wanted:
				_errors.pop(pack_id, None)
		snapshot = {pid: (info.get("cfg"), info) for pid, info in _sessions.items()}

	_close_infos(to_close)

	from lib import ipc
	errors = []
	for pack_id, cfg in wanted.items():
		prev_cfg, prev = snapshot.get(pack_id, (None, None))
		need_connect = pack_id in force_ids or prev is None or not _cfg_same(prev_cfg, cfg)
		if not need_connect:
			continue

		if prev is not None:
			with _lock:
				cur = _sessions.pop(pack_id, None)
			if cur is not None:
				_close_infos([cur])

		try:
			ipc.LogErr(f"Engine: connecting MCP {pack_id} ...")
			info = _await(_connect_one(cfg), timeout=60)
			_cache_schemas(pack_id, info.get("schemas"))
			with _lock:
				_sessions[pack_id] = info
				_errors.pop(pack_id, None)
			ipc.LogErr(f"Engine: MCP {pack_id} ready")
		except Exception as e:
			with _lock:
				_errors[pack_id] = str(e)
			errors.append(f"{pack_id}: {e}")
			ipc.LogErr(f"Engine: MCP {pack_id} failed ({e})")
	return errors

def list_enabled_schemas():
	with _lock:
		out = []
		for pack_id, info in _sessions.items():
			for schema in info.get("schemas") or []:
				item = dict(schema)
				item["_pack_id"] = pack_id
				out.append(item)
		return out

def list_pack_schemas(pack_id, use_cache=True):
	"""Return live schemas if connected, else last cached list."""
	with _lock:
		info = _sessions.get(pack_id)
		if info is not None:
			return [dict(s) for s in (info.get("schemas") or [])]
		cached = _schema_cache.get(pack_id) if use_cache else None
		return [dict(s) for s in (cached or [])]

def call_tool(name, arguments=None, allowed_pack_ids=None):
	"""Dispatch by prefixed name (cait_read_file); call MCP with bare tool name."""
	arguments = arguments or {}
	want = str(name or "").strip()
	if not want:
		return None
	allowed = set(allowed_pack_ids) if allowed_pack_ids is not None else None
	with _lock:
		targets = list(_sessions.items())
	for pack_id, info in targets:
		if allowed is not None and pack_id not in allowed:
			continue
		bare = None
		for schema in info.get("schemas") or []:
			if schema.get("name") == want:
				bare = schema.get("_mcp_name") or ""
				break
		if not bare:
			prefix = SanitizeId(ServerKey(pack_id)) + "_"
			if want.startswith(prefix):
				bare = want[len(prefix):]
			else:
				continue
		if not bare:
			continue
		try:
			result = _await(info["session"].call_tool(bare, arguments), timeout=180)
			return _result_to_text(result)
		except Exception as e:
			return f"ERROR: MCP call failed ({e})"
	return None
