"""Built-in native tool packs with MCP-shaped schemas."""

from . import math_pack
from . import web_pack
from . import coding_pack
from . import image_pack

# Registry order is the default UI / dispatch priority.
NATIVE_PACKS = {
	math_pack.PACK_ID: math_pack,
	web_pack.PACK_ID: web_pack,
	coding_pack.PACK_ID: coding_pack,
	image_pack.PACK_ID: image_pack,
}

def list_native_pack_defs():
	return [
		{"id": math_pack.PACK_ID, "kind": "native", "name": math_pack.PACK_NAME},
		{"id": web_pack.PACK_ID, "kind": "native", "name": web_pack.PACK_NAME},
		{"id": coding_pack.PACK_ID, "kind": "native", "name": coding_pack.PACK_NAME},
		{"id": image_pack.PACK_ID, "kind": "native", "name": image_pack.PACK_NAME},
	]
