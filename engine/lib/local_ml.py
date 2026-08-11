"""Lazy local ML stack (torch / transformers / etc.).

API-only installs use requirements-lite.txt and never import these packages.
Full local features need requirements.txt (includes lite + the ML stack).
"""

from __future__ import annotations

_ready = False


class MissingLocalMl(RuntimeError):
	def __init__(self, feature: str = "This feature"):
		super().__init__(
			f"{feature} requires the full engine stack. Install with:\n"
			"  pip install -r engine/requirements.txt\n"
			"(API-only installs can use requirements-lite.txt.)"
		)


def TorchAvailable() -> bool:
	try:
		import torch  # noqa: F401
		return True
	except ImportError:
		return False


def EnsureLocalMl(feature: str = "Local ML"):
	"""Import torch/transformers once; raise MissingLocalMl if absent."""
	global _ready
	if _ready:
		return
	try:
		import torch  # noqa: F401
		import transformers
	except ImportError as e:
		raise MissingLocalMl(feature) from e
	try:
		transformers.logging.set_verbosity_error()
	except Exception:
		pass
	_ready = True


def ResolveTorchDtype(name, default="auto"):
	"""Map a dtype name string to a torch.dtype (or 'auto')."""
	EnsureLocalMl("Model dtype")
	import torch
	key = str(name or default).strip().lower()
	if key in ("", "auto"):
		return "auto"
	if key == "float32":
		return torch.float32
	if key == "float16":
		return torch.float16
	if key == "bfloat16":
		return torch.bfloat16
	return default
