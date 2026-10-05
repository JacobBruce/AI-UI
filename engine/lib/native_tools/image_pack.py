"""Native Image pack - GenImage / GenImageIPA wrappers."""

import os

PACK_ID = "native:image"
PACK_NAME = "Image"
PACK_KIND = "native"

TOOLS = [
	{
		"name": "gen_image",
		"annotations": {"readOnlyHint": False},
		"description": (
			"Generates an image from a text prompt using the configured local image model. "
			"Returns the saved image file as an absolute path on success. To show it in chat, "
			"embed that exact absolute path with Markdown (![alt](/full/path.png)) or "
			"BBCode ([img]/full/path.png[/img]) — do not use a relative filename."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"prompt": {
					"type": "string",
					"description": "Image generation prompt.",
				},
				"neg_prompt": {
					"type": "string",
					"description": "Optional negative prompt. Use NONE for the default.",
					"default": "NONE",
				},
				"infer_steps": {
					"type": "integer",
					"description": "Inference steps. Optional (default=50).",
					"default": 50,
				},
				"guidance": {
					"type": "number",
					"description": "Guidance scale. Optional (default=7.5).",
					"default": 7.5,
				},
				"img_seed": {
					"type": "string",
					"description": "Optional seed string, or NONE for random.",
					"default": "NONE",
				},
				"img_width": {
					"type": "string",
					"description": "Width in pixels, or auto. Optional.",
					"default": "auto",
				},
				"img_height": {
					"type": "string",
					"description": "Height in pixels, or auto. Optional.",
					"default": "auto",
				},
			},
			"required": ["prompt"],
			"additionalProperties": False,
		},
	},
	{
		"name": "gen_image_ipa",
		"annotations": {"readOnlyHint": False},
		"description": (
			"Generates an image guided by a reference image via IP-Adapter when guide_image "
			"is set; otherwise behaves like gen_image. Use is_person=true for face-oriented adapters. "
			"Returns the saved image file as an absolute path on success. To show it in chat, "
			"embed that exact absolute path with Markdown (![alt](/full/path.png)) or "
			"BBCode ([img]/full/path.png[/img]) — do not use a relative filename."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"prompt": {
					"type": "string",
					"description": "Image generation prompt.",
				},
				"guide_image": {
					"type": "string",
					"description": "Optional path to a guide/reference image for IP-Adapter.",
				},
				"is_person": {
					"type": "boolean",
					"description": "If true, use face IP-Adapter settings when available. Optional (default=false).",
					"default": False,
				},
			},
			"required": ["prompt"],
			"additionalProperties": False,
		},
	},
]

def list_tools():
	return list(TOOLS)

def _AbsPathResult(result):
	if isinstance(result, str) and result and not result.startswith("ERROR:"):
		try:
			return os.path.abspath(result)
		except Exception:
			return result
	return result

def call_tool(name, arguments=None, aiui_funcs=None):
	arguments = arguments or {}
	aiui_funcs = aiui_funcs or {}
	try:
		if name == "gen_image":
			fn = aiui_funcs.get("GenImage")
			if fn is None:
				return "ERROR: GenImage is not available"
			return _AbsPathResult(fn(
				arguments.get("prompt", ""),
				arguments.get("neg_prompt", "NONE"),
				int(arguments.get("infer_steps", 50) or 50),
				float(arguments.get("guidance", 7.5) or 7.5),
				arguments.get("img_seed", "NONE"),
				arguments.get("img_width", "auto"),
				arguments.get("img_height", "auto"),
			))
		if name == "gen_image_ipa":
			fn = aiui_funcs.get("GenImageIPA")
			if fn is None:
				return "ERROR: GenImageIPA is not available"
			guide = arguments.get("guide_image")
			if isinstance(guide, str) and not guide.strip():
				guide = None
			return _AbsPathResult(fn(
				arguments.get("prompt", ""),
				guide,
				bool(arguments.get("is_person", False)),
			))
		return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"
