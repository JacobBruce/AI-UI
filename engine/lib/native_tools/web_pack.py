"""Native Web pack - Firecrawl search/scrape with HTTP markdown fallback."""

import re
import urllib.error
import urllib.request

from html_to_markdown import convert, ConversionOptions

PACK_ID = "native:web"
PACK_NAME = "Web"
PACK_KIND = "native"

_user_agent = "AI UI (github.com/JacobBruce/AI-UI)"
_fc_api_key = ""
_firecrawl = None
# Keep tool results from flooding chat context (~100KB UTF-8).
_MAX_PAGE_BYTES = 100_000
_DATA_URI_RE = re.compile(
	r"data:[^\s\"'<>]*;base64,[A-Za-z0-9+/=\s]+",
	re.IGNORECASE,
)

TOOLS = [
	{
		"name": "web_search",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Performs a web search via Firecrawl. Supports news and images via search_mode, "
			"and academic or GitHub-focused results via category. "
			"Common query operators are supported, for example: '\"ping pong\" -site:reddit.com'."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "The search query",
				},
				"search_mode": {
					"type": "string",
					"description": "Valid values are 'web', 'news', and 'images'. Optional (default='web').",
					"enum": ["web", "news", "images"],
					"default": "web",
				},
				"category": {
					"type": "string",
					"description": (
						"Optional Firecrawl category. Use 'research' for academic sites "
						"(arXiv, Nature, IEEE, PubMed, etc) or 'github' for GitHub content. "
						"Leave empty for a general search."
					),
					"default": "",
				},
				"location": {
					"type": "string",
					"description": "Set a location (e.g. 'Australia') to get geo-targeted results. Optional.",
					"default": "",
				},
				"max_results": {
					"type": "integer",
					"description": "The maximum number of search results. Optional (default=10).",
					"default": 10,
				},
			},
			"required": ["query"],
			"additionalProperties": False,
		},
	},
	{
		"name": "get_web_page",
		"annotations": {"readOnlyHint": True},
		"description": (
			"Returns the contents of a web page as HTML or Markdown. "
			"Uses Firecrawl when an API key is configured; otherwise falls back to HTTP fetch + markdown conversion. "
			"Responses are capped at ~100KB; prefer a more specific URL for large pages."
		),
		"inputSchema": {
			"type": "object",
			"properties": {
				"url": {
					"type": "string",
					"description": "The URL of the web page",
				},
				"format": {
					"type": "string",
					"description": "Valid values are 'html' and 'markdown'. Optional (default='markdown').",
					"enum": ["html", "markdown"],
					"default": "markdown",
				},
			},
			"required": ["url"],
			"additionalProperties": False,
		},
	},
]

def configure(cfg=None):
	"""Refresh Firecrawl client from app/tools config."""
	global _fc_api_key, _firecrawl
	cfg = cfg or {}
	key = str(cfg.get("firecrawl_api_key", "") or "").strip()
	if key == "YOUR-API-KEY":
		key = ""
	_fc_api_key = key
	_firecrawl = None
	if _fc_api_key:
		try:
			from firecrawl import Firecrawl
			_firecrawl = Firecrawl(api_key=_fc_api_key)
		except Exception:
			_firecrawl = None

def list_tools():
	# web_search requires Firecrawl; get_web_page always available (HTTP fallback).
	if _firecrawl is None:
		return [t for t in TOOLS if t["name"] == "get_web_page"]
	return list(TOOLS)

def _read_url(url):
	req = urllib.request.Request(url, headers={"User-Agent": _user_agent})
	try:
		response = urllib.request.urlopen(req, timeout=30)
		if response.code == 200:
			return {"txt": response.read().decode("utf-8", errors="replace"), "err": False}
		return {"txt": f"ERROR: got unexpected status code ({response.code})", "err": True}
	except urllib.error.HTTPError as e:
		return {"txt": f"HTTP ERROR ({e.code}): {e.reason}", "err": True}
	except urllib.error.URLError as e:
		return {"txt": f"URL ERROR: {e.reason}", "err": True}

def _format_search_results(results, search_mode):
	result_str = "SEARCH RESULTS:\n\n"
	if search_mode == "web" and hasattr(results, "web"):
		for result in results.web:
			result_str += "URL: " + str(getattr(result, "url", "") or "") + "\n"
			result_str += "Title: " + str(getattr(result, "title", "") or "") + "\n"
			result_str += "Description: " + str(getattr(result, "description", "") or "") + "\n\n"
	elif search_mode == "news" and hasattr(results, "news"):
		for result in results.news:
			result_str += "URL: " + str(getattr(result, "url", "") or "") + "\n"
			result_str += "Title: " + str(getattr(result, "title", "") or "") + "\n"
			result_str += "Snippet: " + str(getattr(result, "snippet", "") or "") + "\n"
			result_str += "Date: " + str(getattr(result, "date", "") or "") + "\n\n"
	elif search_mode == "images" and hasattr(results, "images"):
		for result in results.images:
			result_str += "URL: " + result.url + "\n"
			if result.image_url is not None:
				result_str += "Image URL: " + result.image_url + "\n"
			if result.image_width is not None:
				result_str += "Image Width: " + str(result.image_width) + "\n"
			if result.image_height is not None:
				result_str += "Image Height: " + str(result.image_height) + "\n"
			result_str += "Title: " + result.title + "\n\n"
	else:
		return "ERROR: got unexpected response"
	if result_str == "SEARCH RESULTS:\n\n":
		result_str += "Nothing Found"
	return result_str.rstrip("\n")

def _web_search(query: str, search_mode: str = "web", category: str = "", location: str = "", max_results: int = 10) -> str:
	if _firecrawl is None:
		return "ERROR: Firecrawl API key not configured"
	if str(query).strip() == "":
		return "ERROR: search query is empty"
	search_mode = (search_mode or "web").lower()
	if search_mode not in ("web", "news", "images"):
		return "ERROR: invalid search_mode"
	category = (category or "").strip().lower()
	kwargs = {
		"query": query,
		"limit": min(int(max_results or 10), 99),
	}
	if category in ("research", "github"):
		kwargs["categories"] = [category]
	else:
		kwargs["sources"] = [search_mode]
		if location:
			kwargs["location"] = location
	results = _firecrawl.search(**kwargs)
	if isinstance(results, str):
		return results
	# Category searches return web-shaped results.
	mode_for_fmt = "web" if category in ("research", "github") else search_mode
	return _format_search_results(results, mode_for_fmt)

def _CapPageText(text):
	"""Strip data URIs and truncate oversized page text for tool responses."""
	if text is None:
		return ""
	text = _DATA_URI_RE.sub("[omitted data URI]", str(text))
	encoded = text.encode("utf-8")
	if len(encoded) <= _MAX_PAGE_BYTES:
		return text
	cut = encoded[:_MAX_PAGE_BYTES].decode("utf-8", errors="ignore")
	return (
		cut
		+ f"\n\n[truncated: showing {_MAX_PAGE_BYTES} of {len(encoded)} bytes. "
		+ "Use a more specific URL or save large pages outside the tool response.]"
	)

def _get_web_page(url: str, format: str = "markdown") -> str:
	clean_url = str(url).strip()
	format_low = str(format or "markdown").lower()
	if clean_url == "":
		return "ERROR: URL argument is empty"
	if format_low not in ("html", "markdown"):
		return "ERROR: invalid format"
	if _firecrawl is not None:
		result = _firecrawl.scrape(clean_url, formats=[format_low])
		if isinstance(result, str):
			return _CapPageText(result)
		if format_low == "markdown" and hasattr(result, "markdown"):
			return _CapPageText(result.markdown)
		if format_low == "html" and hasattr(result, "html"):
			return _CapPageText(result.html)
		return "ERROR: got unexpected response"
	result = _read_url(clean_url)
	if format_low == "html" or result["err"]:
		return _CapPageText(result["txt"])
	return _CapPageText(convert(
		result["txt"],
		ConversionOptions(
			output_format="markdown",
			list_indent_width=2,
			extract_metadata=False,
		),
	).content)

def call_tool(name, arguments=None):
	arguments = arguments or {}
	try:
		if name == "web_search":
			return _web_search(**arguments)
		if name == "get_web_page":
			return _get_web_page(**arguments)
		return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"
