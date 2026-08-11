#!/usr/bin/env python3
"""AI UI engine install / upgrade helper (Windows + Linux).

Interactive by default; pass flags for non-interactive use.

Examples:
  python install.py
  python install.py --venv ~/aiui-venv --full --torch cu128
  python install.py --venv C:\\venv --lite --yes
  python install.py --upgrade --yes
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


MIN_PY = (3, 12)
MAX_PY = (3, 14)
TORCH_CU_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")


def IsWindows() -> bool:
	return os.name == "nt" or platform.system().lower().startswith("win")


def AppRoot() -> Path:
	return Path(__file__).resolve().parent


def EngineDir(root: Path | None = None) -> Path:
	return (root or AppRoot()) / "engine"


def AiuiHome() -> Path:
	return Path.home() / ".aiui"


def ConfigPath() -> Path:
	return AiuiHome() / "config.json"


def DefaultVenvPath() -> Path:
	if IsWindows():
		return Path.home() / "aiui-venv"
	return AppRoot() / "venv"


def VenvPython(venv: Path) -> Path:
	if IsWindows():
		return venv / "Scripts" / "python.exe"
	py3 = venv / "bin" / "python3"
	if py3.exists():
		return py3
	return venv / "bin" / "python"


def Print(msg: str = "") -> None:
	print(msg, flush=True)


def Die(msg: str, code: int = 1) -> None:
	Print(f"ERROR: {msg}")
	sys.exit(code)


def Prompt(msg: str, default: str | None = None) -> str:
	suffix = f" [{default}]" if default is not None and default != "" else ""
	try:
		raw = input(f"{msg}{suffix}: ").strip()
	except EOFError:
		raw = ""
	if not raw and default is not None:
		return default
	return raw


def PromptYesNo(msg: str, default: bool = True) -> bool:
	hint = "Y/n" if default else "y/N"
	raw = Prompt(f"{msg} ({hint})", "y" if default else "n").lower()
	if raw in ("y", "yes"):
		return True
	if raw in ("n", "no"):
		return False
	if raw == "":
		return default
	return default


def PromptChoice(msg: str, choices: list[str], default: str) -> str:
	labels = "/".join(choices)
	while True:
		raw = Prompt(f"{msg} ({labels})", default).lower().strip()
		if raw in choices:
			return raw
		Print(f"  Please choose one of: {labels}")


def CheckHostPython() -> None:
	ver = sys.version_info[:2]
	if ver < MIN_PY or ver > MAX_PY:
		Print(
			f"WARN: this interpreter is Python {ver[0]}.{ver[1]}; "
			f"AI UI targets {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]}."
		)
	else:
		Print(f"Host Python: {ver[0]}.{ver[1]}.{sys.version_info[2]}")


def ResolveVenv(args: argparse.Namespace, interactive: bool) -> Path:
	if args.venv:
		return Path(args.venv).expanduser().resolve()
	if not interactive:
		Die("--venv is required in non-interactive mode")
	default = str(DefaultVenvPath())
	Print("\nVirtual environment")
	Print("  Enter a path to create, or an existing venv directory.")
	raw = Prompt("Venv path", default)
	if not raw:
		Die("venv path is required")
	return Path(raw).expanduser().resolve()


def EnsureVenv(venv: Path, *, dry_run: bool) -> Path:
	py = VenvPython(venv)
	if py.exists():
		Print(f"Using existing venv: {venv}")
		return py
	Print(f"Creating venv: {venv}")
	if dry_run:
		Print(f"  [dry-run] {sys.executable} -m venv {venv}")
		return py
	venv.parent.mkdir(parents=True, exist_ok=True)
	subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
	py = VenvPython(venv)
	if not py.exists():
		Die(f"venv created but interpreter not found: {py}")
	return py


def Pip(py: Path, *pip_args: str, dry_run: bool) -> None:
	cmd = [str(py), "-m", "pip", *pip_args]
	Print("  $ " + " ".join(cmd))
	if dry_run:
		return
	subprocess.check_call(cmd)


def InstallTorch(py: Path, kind: str, *, dry_run: bool) -> None:
	if kind == "skip":
		Print("Skipping PyTorch install.")
		return
	index = TORCH_CU_INDEX if kind == "cu128" else TORCH_CPU_INDEX
	Print(f"Installing PyTorch ({kind})…")
	Pip(py, "install", *TORCH_PACKAGES, "--index-url", index, dry_run=dry_run)


def InstallRequirements(py: Path, mode: str, engine: Path, *, dry_run: bool) -> None:
	req = engine / ("requirements-lite.txt" if mode == "lite" else "requirements.txt")
	if not req.is_file():
		Die(f"requirements file not found: {req}")
	Print(f"Installing engine packages ({mode}) from {req.name}…")
	Pip(py, "install", "-U", "pip", "setuptools", "wheel", dry_run=dry_run)
	Pip(py, "install", "-U", "-r", str(req), dry_run=dry_run)


def BundlesMissing(root: Path) -> bool:
	app = root / "resources" / "app"
	needed = [
		app / "app.bundle.js",
		app / "vendor.bundle.js",
		app / "node_modules",
	]
	return any(not p.exists() for p in needed)


def MaybeNpmInstall(root: Path, *, do_npm: bool | None, interactive: bool, dry_run: bool) -> None:
	app = root / "resources" / "app"
	missing = BundlesMissing(root)
	if do_npm is False:
		return
	if do_npm is None:
		if not missing:
			Print("App bundles / node_modules present — skipping npm install.")
			return
		if not interactive:
			Print(
				"WARN: app bundles or node_modules missing; "
				"re-run with --npm or install Node and run npm install in resources/app."
			)
			return
		if not PromptYesNo("App JS bundles or node_modules look missing. Run npm install in resources/app?", True):
			return
	npm = shutil.which("npm")
	if not npm:
		Print("WARN: npm not found on PATH — install Node.js and re-run with --npm.")
		return
	Print(f"Running npm install in {app}…")
	cmd = [npm, "install"]
	Print("  $ " + " ".join(cmd))
	if dry_run:
		return
	subprocess.check_call(cmd, cwd=str(app))


def BundledAiui(root: Path | None = None) -> Path:
	return (root or AppRoot()) / ".aiui"


def _CopyFileIfMissing(src: Path, dest: Path, *, dry_run: bool) -> str:
	"""Copy src → dest when dest is missing. Returns 'copied' | 'skipped' | 'dry'."""
	if dest.exists():
		return "skipped"
	if dry_run:
		Print(f"  [dry-run] copy {src} → {dest}")
		return "dry"
	dest.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src, dest)
	return "copied"


def SeedAiuiHome(
	root: Path,
	*,
	mode: str,
	dry_run: bool,
) -> None:
	"""Install bundled .aiui defaults into ~/.aiui.

	mode: 'copy' (merge, never overwrite), 'move' (same then remove bundled tree), 'skip'.
	Skips config.json so install WritePythonBin stays authoritative.
	"""
	if mode == "skip":
		return
	src = BundledAiui(root)
	dest = AiuiHome()
	if not src.is_dir():
		Print(f"WARN: no bundled .aiui at {src} — nothing to seed")
		return

	Print(f"Seeding {dest} from {src} ({mode}; existing files kept)")
	copied = skipped = 0
	for path in sorted(src.rglob("*")):
		if not path.is_file():
			continue
		rel = path.relative_to(src)
		if rel.as_posix() == "config.json":
			continue
		status = _CopyFileIfMissing(path, dest / rel, dry_run=dry_run)
		if status == "copied":
			copied += 1
			Print(f"  + {rel.as_posix()}")
		elif status == "dry":
			copied += 1
		else:
			skipped += 1

	Print(f"  done: {copied} new, {skipped} already present")
	if mode != "move":
		return
	if dry_run:
		Print(f"  [dry-run] would remove bundled {src}")
		return
	try:
		shutil.rmtree(src)
		Print(f"  removed bundled {src}")
	except Exception as e:
		Print(f"WARN: could not remove bundled .aiui ({e})")


def ResolveSeedAiui(args: argparse.Namespace, interactive: bool, root: Path) -> str:
	"""Return 'copy' | 'move' | 'skip'."""
	if getattr(args, "no_seed_aiui", False):
		return "skip"
	if getattr(args, "seed_aiui_move", False):
		return "move"
	if getattr(args, "seed_aiui", False):
		return "copy"
	src = BundledAiui(root)
	if not src.is_dir():
		return "skip"
	if not interactive:
		return "copy"
	Print("\nBundled prompt files")
	Print(f"  Found {src}")
	Print(f"  Target {AiuiHome()} (agents / rules / skills / etc.)")
	Print("  copy = merge into ~/.aiui, keep package copy (recommended)")
	Print("  move = merge into ~/.aiui, then delete the package .aiui folder")
	Print("  skip = leave ~/.aiui unchanged")
	return PromptChoice("Install bundled .aiui defaults?", ["copy", "move", "skip"], "copy")


def WritePythonBin(py: Path, engine: Path, *, dry_run: bool) -> None:
	"""Write python_bin into ~/.aiui/config.json.

	Only sets script_dir when creating a new config (first-launch auto-detect
	is skipped once any config.json exists). Existing configs keep Engine Folder.
	"""
	home = AiuiHome()
	cfg_path = ConfigPath()
	py_s = str(py).replace("\\", "/")
	engine_s = str(engine.resolve()).replace("\\", "/")
	avatar = engine / "ai_images" / "default_avatar.jpg"
	avatar_s = str(avatar.resolve()).replace("\\", "/") if avatar.is_file() else ""

	created = not cfg_path.is_file()
	data: dict = {}
	if not created:
		try:
			data = json.loads(cfg_path.read_text(encoding="utf-8"))
		except Exception as e:
			Die(f"could not read {cfg_path}: {e}")
		if not isinstance(data, dict):
			Die(f"{cfg_path} is not a JSON object")

	app = data.setdefault("app", {})
	if not isinstance(app, dict):
		Die(f"{cfg_path}: app section must be an object")
	app["python_bin"] = py_s
	if created:
		app["script_dir"] = engine_s
		if avatar_s and not app.get("avatar_img"):
			app["avatar_img"] = avatar_s

	Print(f"Updating {cfg_path}")
	Print(f"  python_bin = {py_s}")
	if created:
		Print(f"  script_dir = {engine_s}  (new config; required so first launch can find the engine)")
	if dry_run:
		Print("  [dry-run] not written")
		return
	home.mkdir(parents=True, exist_ok=True)
	cfg_path.write_text(json.dumps(data, indent="\t") + "\n", encoding="utf-8")


def LinuxSandboxNote(root: Path) -> None:
	if IsWindows():
		return
	sandbox = root / "chrome-sandbox"
	if sandbox.is_file():
		Print(
			"\nLinux note: if Electron fails to start with a sandbox error, you may need:\n"
			f"  sudo chown root:root {sandbox}\n"
			f"  sudo chmod 4755 {sandbox}"
		)


def PrintSummary(py: Path, mode: str, root: Path) -> None:
	Print("\n=== Done ===")
	Print(f"Interpreter: {py}")
	Print(f"Stack:       {mode}")
	if IsWindows():
		exe = root / "AI_UI.exe"
		Print(f"Launch:      {exe if exe.is_file() else 'AI_UI.exe (from the install folder)'}")
	else:
		exe = root / "electron"
		Print(f"Launch:      {exe if exe.is_file() else './electron'}  (from the install folder)")
	Print("Settings → Python Binary should already point at the venv interpreter.")
	Print("")
	Print("Recommended: install CAIT (MCP toolkit) for richer tools in AI UI.")
	Print("Available at https://github.com/JacobBruce/CAIT or https://smithery.ai/servers/bitfreak/cait")
	Print("Then add it under Settings → Tool Packs (SHOW MCP.JSON) or enable it when listed.")


def ParseArgs(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Install or upgrade the AI UI Python engine environment.",
	)
	p.add_argument("--venv", help="Path to create or reuse as the virtual environment")
	g = p.add_mutually_exclusive_group()
	g.add_argument("--lite", action="store_true", help="Install requirements-lite.txt only")
	g.add_argument("--full", action="store_true", help="Install full requirements.txt (local models)")
	p.add_argument(
		"--torch",
		choices=("cu128", "cpu", "skip"),
		help="PyTorch build for full installs (default: ask, or cu128 with --yes)",
	)
	p.add_argument("--upgrade", action="store_true", help="Reinstall/upgrade packages (same as a normal run)")
	p.add_argument("--npm", action="store_true", help="Force npm install in resources/app")
	p.add_argument("--no-npm", action="store_true", help="Never run npm install")
	p.add_argument("--no-config", action="store_true", help="Do not write ~/.aiui/config.json")
	p.add_argument(
		"--seed-aiui",
		action="store_true",
		help="Copy bundled .aiui into ~/.aiui (merge; never overwrite)",
	)
	p.add_argument(
		"--seed-aiui-move",
		action="store_true",
		help="Like --seed-aiui, then remove the package .aiui folder",
	)
	p.add_argument("--no-seed-aiui", action="store_true", help="Do not seed ~/.aiui from bundled .aiui")
	p.add_argument("--yes", "-y", action="store_true", help="Non-interactive; accept defaults")
	p.add_argument("--dry-run", action="store_true", help="Print actions without running them")
	return p.parse_args(argv)


def ResolveMode(args: argparse.Namespace, interactive: bool) -> str:
	if args.lite:
		return "lite"
	if args.full:
		return "full"
	if not interactive:
		return "lite"
	Print("\nInstall size")
	Print("  lite = remote APIs + tools (no local torch/transformers)")
	Print("  full = local chat / image / TTS / ASR / avatar stack")
	return PromptChoice("Which stack?", ["lite", "full"], "lite")


def ResolveTorch(args: argparse.Namespace, mode: str, interactive: bool) -> str:
	if mode == "lite":
		return "skip"
	if args.torch:
		return args.torch
	if not interactive:
		return "cu128"
	Print("\nPyTorch (full install only; install before other packages)")
	Print("  cu128 = NVIDIA CUDA 12.8 wheels (recommended if you have an NVIDIA GPU)")
	Print("  cpu   = CPU-only wheels")
	Print("  skip  = already installed in this venv")
	return PromptChoice("Torch build?", ["cu128", "cpu", "skip"], "cu128")


def Main(argv: list[str] | None = None) -> int:
	args = ParseArgs(argv)
	interactive = not args.yes
	root = AppRoot()
	engine = EngineDir(root)

	Print("AI UI install helper")
	Print(f"App root: {root}")
	CheckHostPython()

	if not engine.is_dir() or not (engine / "aiui_engine.py").is_file():
		Die(f"engine folder not found under {root} (expected engine/aiui_engine.py)")

	venv = ResolveVenv(args, interactive)
	mode = ResolveMode(args, interactive)
	torch_kind = ResolveTorch(args, mode, interactive)
	if args.no_npm:
		do_npm: bool | None = False
	elif args.npm:
		do_npm = True
	else:
		do_npm = None
	seed_mode = ResolveSeedAiui(args, interactive, root)

	Print("\nPlan")
	Print(f"  venv:   {venv}")
	Print(f"  stack:  {mode}")
	Print(f"  torch:  {torch_kind}")
	Print(f"  npm:    {'force' if do_npm is True else 'never' if do_npm is False else 'if missing'}")
	Print(f"  config: {'skip' if args.no_config else 'write python_bin → ~/.aiui/config.json'}")
	Print(f"  .aiui:  {seed_mode}")
	if args.dry_run:
		Print("  mode:   dry-run")
	if interactive and not PromptYesNo("\nProceed?", True):
		Print("Cancelled.")
		return 0

	py = EnsureVenv(venv, dry_run=args.dry_run)
	if not args.dry_run and not py.exists():
		Die(f"interpreter not found: {py}")

	if not args.dry_run:
		try:
			out = subprocess.check_output(
				[str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
				text=True,
			).strip()
			major, minor = (int(x) for x in out.split("."))
			if (major, minor) < MIN_PY or (major, minor) > MAX_PY:
				Print(
					f"WARN: venv Python is {out}; "
					f"AI UI targets {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]}."
				)
		except Exception as e:
			Print(f"WARN: could not check venv Python version ({e})")

	if mode == "full":
		InstallTorch(py, torch_kind, dry_run=args.dry_run)
	InstallRequirements(py, mode, engine, dry_run=args.dry_run)
	MaybeNpmInstall(root, do_npm=do_npm, interactive=interactive, dry_run=args.dry_run)

	if not args.no_config:
		WritePythonBin(py if not args.dry_run else VenvPython(venv), engine, dry_run=args.dry_run)
	SeedAiuiHome(root, mode=seed_mode, dry_run=args.dry_run)

	LinuxSandboxNote(root)
	PrintSummary(py if not args.dry_run else VenvPython(venv), mode, root)
	return 0


if __name__ == "__main__":
	try:
		sys.exit(Main())
	except KeyboardInterrupt:
		Print("\nCancelled.")
		sys.exit(130)
	except subprocess.CalledProcessError as e:
		Die(f"command failed with exit code {e.returncode}")
