"""Versioned NDJSON IPC between Electron host and aiui_engine.

Stdout is control-plane only (one JSON object per line).
Debug / library noise should go to stderr via LogErr / traceback helpers.

Writes go through a background writer thread so a full Windows pipe (parent not
draining stdout/stderr) cannot freeze the engine mid-boot. That matched a
Windows-only hang that cleared when the UI later pumped I/O (e.g. Apply).
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback

PROTOCOL_VERSION = 1

_msg_queue: queue.Queue = queue.Queue()
_reader_started = False
_reader_lock = threading.Lock()
_stop_event = threading.Event()
_active_gen_thread = None
_active_gen_lock = threading.Lock()
# Real process stdout — control plane only after InstallStdoutGuard().
_real_stdout = None

# Serialized writes; never block the engine main thread on a full pipe.
_write_queue: queue.Queue = queue.Queue()
_writer_started = False
_writer_lock = threading.Lock()
_queued_stderr = None


class _QueuedStderr:
	"""File-like stderr that queues writes so print() cannot stall the engine."""

	encoding = "utf-8"
	errors = "replace"
	name = "<aiui-queued-stderr>"

	def write(self, data):
		if not data:
			return 0
		text = data if isinstance(data, str) else data.decode("utf-8", "replace")
		_EnsureWriter()
		_write_queue.put(("err", text))
		return len(data)

	def flush(self):
		return None

	def isatty(self):
		return False


def ProtocolVersion():
	return PROTOCOL_VERSION


def InstallStdoutGuard():
	"""Route bare print()/sys.stdout writes through the queued stderr writer.

	SadTalker / MakeItTalk / Wav2Lip / tqdm-style progress all print to stdout by
	default; those must not share the Electron control pipe. Queuing also avoids
	Windows pipe deadlocks when Electron is slow to drain stderr.
	"""
	global _real_stdout, _queued_stderr
	if _real_stdout is not None:
		return
	_real_stdout = sys.stdout
	_EnsureWriter()
	if _queued_stderr is None:
		_queued_stderr = _QueuedStderr()
	sys.stderr = _queued_stderr
	sys.stdout = _queued_stderr


def ClearStop():
	_stop_event.clear()


def StopRequested():
	return _stop_event.is_set()


def RequestStop():
	_stop_event.set()
	try:
		from lib import text_backend
		text_backend.FireStopCallbacks()
	except Exception:
		pass
	with _active_gen_lock:
		thread = _active_gen_thread
	if thread is not None and hasattr(thread, "ForceKill"):
		try:
			thread.ForceKill()
		except Exception:
			pass


def SetActiveGenThread(thread):
	with _active_gen_lock:
		global _active_gen_thread
		_active_gen_thread = thread


def _EnsureWriter():
	global _writer_started
	with _writer_lock:
		if _writer_started:
			return
		t = threading.Thread(target=_WriterLoop, name="aiui-ipc-writer", daemon=True)
		t.start()
		_writer_started = True


def _WriterLoop():
	"""Drain queued stdout/stderr writes; may block here without stalling boot."""
	real_err = sys.__stderr__
	while True:
		kind, payload = _write_queue.get()
		try:
			if kind == "err":
				err = real_err if real_err is not None else sys.__stderr__
				if err is None:
					continue
				err.write(payload)
				err.flush()
			elif kind == "out":
				out = _real_stdout if _real_stdout is not None else sys.__stdout__
				if out is None:
					continue
				try:
					out.write(payload)
				except UnicodeEncodeError:
					out.write(payload.encode("utf-8", "replace").decode("utf-8", "replace"))
				out.flush()
		except Exception:
			# Never kill the writer; next line may still get through.
			pass


def LogErr(msg):
	_EnsureWriter()
	_write_queue.put(("err", str(msg) + "\n"))


def LogException(where, exc=None):
	tb = traceback.format_exc()
	LogErr(f"ERROR [{where}]: {exc if exc is not None else ''}\n{tb}")
	SendError(where, str(exc) if exc is not None else "unknown error", tb)


def Send(obj):
	"""Queue one NDJSON message for the real stdout (control plane)."""
	if not isinstance(obj, dict):
		raise TypeError("IPC Send requires a dict")
	payload = dict(obj)
	payload.setdefault("v", PROTOCOL_VERSION)
	try:
		line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
		# Fail fast if surrogates would break UTF-8 on the writer thread.
		line.encode("utf-8")
	except UnicodeEncodeError:
		line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
	_EnsureWriter()
	_write_queue.put(("out", line + "\n"))


def SendHello():
	InstallStdoutGuard()
	Send({"type": "hello", "protocol": PROTOCOL_VERSION})


def SendReady(kind="idle"):
	"""kind: ready (after boot) or idle (awaiting next cmd)."""
	Send({"type": "ready" if kind == "ready" else "idle"})


def SendIdle():
	Send({"type": "idle"})


def SendEvent(name, data=None, **extra):
	msg = {"type": "event", "name": name}
	if data is not None:
		msg["data"] = data
	msg.update(extra)
	Send(msg)


def SendError(where, message, tb=None):
	msg = {"type": "error", "where": where, "message": message}
	if tb:
		msg["traceback"] = tb
	Send(msg)


def SendStreamStart(channel="chat"):
	Send({"type": "stream_start", "channel": channel})


def SendStreamDelta(delta, channel="chat"):
	# Skip empty deltas — streamers often emit them and they only noise the console.
	if not delta:
		return
	Send({"type": "stream", "channel": channel, "delta": delta})


def SendStreamEnd(reason="ok", channel="chat"):
	Send({"type": "stream_end", "channel": channel, "reason": reason})


def _ReaderLoop():
	while True:
		try:
			line = sys.stdin.readline()
		except Exception as e:
			LogErr(f"IPC stdin read failed: {e}")
			break
		if line == "":
			# EOF
			_msg_queue.put({"v": PROTOCOL_VERSION, "type": "cmd", "name": "close_chat"})
			break
		line = line.strip()
		if not line:
			continue
		try:
			msg = json.loads(line)
		except json.JSONDecodeError as e:
			LogErr(f"IPC bad JSON from host: {e}: {line[:200]}")
			continue
		if not isinstance(msg, dict):
			LogErr(f"IPC ignoring non-object message: {type(msg)}")
			continue
		if msg.get("type") == "control" and msg.get("action") == "stop":
			RequestStop()
			# Enqueue so a blocked Recv() wakes; StreamText polls StopRequested().
			_msg_queue.put(msg)
			continue
		# Resolve tool permission waits on the reader thread so the main
		# turn (blocked in RequirePermission) does not deadlock on Recv.
		if msg.get("type") == "cmd" and msg.get("name") == "tool_permission_result":
			try:
				from lib import tool_permissions
				params = msg.get("params") if isinstance(msg.get("params"), dict) else None
				if params is None:
					skip = {"v", "type", "name", "params"}
					params = {k: v for k, v in msg.items() if k not in skip}
				tool_permissions.ResolvePermission(
					params.get("request_id"),
					params.get("action"),
				)
			except Exception as e:
				LogErr(f"tool_permission_result failed: {e}")
			continue
		_msg_queue.put(msg)


def StartReader():
	global _reader_started
	with _reader_lock:
		if _reader_started:
			return
		t = threading.Thread(target=_ReaderLoop, name="aiui-ipc-reader", daemon=True)
		t.start()
		_reader_started = True


def Recv(timeout=None):
	"""Block until next host message (dict). timeout in seconds or None."""
	StartReader()
	if timeout is None:
		return _msg_queue.get()
	try:
		return _msg_queue.get(timeout=timeout)
	except queue.Empty:
		return None


def WaitBoot():
	"""Receive boot message; return config dict."""
	StartReader()
	while True:
		msg = Recv()
		if msg.get("type") == "boot":
			cfg = msg.get("config")
			if not isinstance(cfg, dict):
				raise RuntimeError("boot message missing config object")
			return cfg
		if msg.get("type") == "control" and msg.get("action") == "stop":
			continue
		LogErr(f"IPC ignoring message before boot: {msg.get('type')}")
