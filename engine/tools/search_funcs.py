import os
import json
import arxiv
import wikipediaapi
import pymupdf
import shutil
import urllib.request
from firecrawl import Firecrawl
from html_to_markdown import convert_to_markdown

# set to your Firecrawl API Key
fc_api_key = "YOUR-API-KEY"

# set to True to use Firecrawl for website scraping
fc_scraping = False

firecrawl = Firecrawl(api_key=fc_api_key)

user_agent = 'AI UI (github.com/JacobBruce/AI-UI)'

arxiv_client = arxiv.Client()

wiki_client = wikipediaapi.Wikipedia(
    user_agent=user_agent,
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    language='en'
)

def download_file(url, file_path):
	req = urllib.request.Request(url=url, headers={'User-Agent': user_agent})
	try:
		with urllib.request.urlopen(req) as response:
			with open(file_path, 'wb') as out_file:
				shutil.copyfileobj(response, out_file)
	except:
		return False
	return True

def read_url(url):
	req = urllib.request.Request(url, headers={'User-Agent': user_agent})
	try:
		response = urllib.request.urlopen(req)
		if response.code == 200:
			return { 'txt': response.read().decode("utf-8"), 'err': False }
		else:
			return { 'txt': f"ERROR: got unexpected status code ({response.code})", 'err': True }
	except urllib.error.HTTPError as e:
		return { 'txt': f"HTTP ERROR ({e.code}): {e.reason}", 'err': True }
	except urllib.error.URLError as e:
		return { 'txt': f"URL ERROR: {e.reason}", 'err': True }

# SEARCH TOOL FUNCTIONS

def web_search(query: str, search_mode: str='web', location: str='', max_results: int=10) -> str:
	"""
	Performs a general web search. Can also specifically search for news and images.
	Common query operators are supported, for example: '"ping pong" -site:reddit.com'
	That query would return results with the phrase "ping pong", excluding results from Reddit.
    
	Args:
		query: The search query
		search_mode: Valid values are 'web', 'news', and 'images'. Optional (default='web')
		location: Set a location (e.g. 'Australia') to get geo-targeted results. Optional.
		max_results: The maximum number of search results. Optional (default=10)
	"""
	if query.strip() == '': return "ERROR: search query is empty"
	locale_str = None if location == '' else location
	
	result_str = "SEARCH RESULTS:\n\n"
	
	results = firecrawl.search(
		query = query,
		limit = min(max_results, 99),
		sources = [search_mode],
		location = locale_str
	)
	
	if type(results) is str: return results
	
	if search_mode == 'web' and hasattr(results, 'web'):
		for result in results.web:
			result_str += "URL: "+result.url+"\n"
			result_str += "Title: "+result.title+"\n"
			result_str += "Description: "+result.description+"\n\n"
	elif search_mode == 'news' and hasattr(results, 'news'):
		for result in results.news:
			result_str += "URL: "+result.url+"\n"
			result_str += "Title: "+result.title+"\n"
			result_str += "Snippet: "+result.snippet+"\n"
			result_str += "Date: "+result.date+"\n\n"
	elif search_mode == 'images' and hasattr(results, 'images'):
		for result in results.images:
			result_str += "URL: "+result.url+"\n"
			if result.image_url != None: result_str += "Image URL: "+result.image_url+"\n"
			if result.image_width != None: result_str += "Image Width: "+str(result.image_width)+"\n"
			if result.image_height != None: result_str += "Image Height: "+str(result.image_height)+"\n"
			result_str += "Title: "+result.title+"\n\n"
	else:
		return "ERROR: got unexpected response"
	
	if result_str == "SEARCH RESULTS:\n\n":
		result_str += "Nothing Found"
	
	return result_str.rstrip("\n")
	
def science_search(query: str, max_results: int=10) -> str:
	"""
	Performs a search of academic and research websites (arXiv, Nature, IEEE, PubMed, etc).
    
	Args:
		query: The search query
		max_results: The maximum number of search results. Optional (default=10)
	"""
	if query.strip() == '': return "ERROR: search query is empty"
	
	result_str = "SEARCH RESULTS:\n\n"
	
	results = firecrawl.search(
		query = query,
		limit = min(max_results, 99),
		categories = ['research']
	)
	
	if type(results) is str: return results
	
	if hasattr(results, 'web'):
		for result in results.web:
			result_str += "URL: "+result.url+"\n"
			result_str += "Title: "+result.title+"\n"
			result_str += "Description: "+result.description+"\n\n"
	else:
		return "ERROR: got unexpected response"
	
	if result_str == "SEARCH RESULTS:\n\n":
		result_str += "Nothing Found"
	
	return result_str.rstrip("\n")

def github_search(query: str, max_results: int=10) -> str:
	"""
	Performs a search of GitHub repositories, code, issues, and documentation.
    
	Args:
		query: The search query
		max_results: The maximum number of search results. Optional (default=10)
	"""
	if query.strip() == '': return "ERROR: search query is empty"
	
	result_str = "SEARCH RESULTS:\n\n"
	
	results = firecrawl.search(
		query = query,
		limit = min(max_results, 99),
		categories = ['github']
	)
	
	if type(results) is str: return results
	
	if hasattr(results, 'web'):
		for result in results.web:
			result_str += "URL: "+result.url+"\n"
			result_str += "Title: "+result.title+"\n"
			result_str += "Description: "+result.description+"\n\n"
	else:
		return "ERROR: unexpected response format"
	
	if result_str == "SEARCH RESULTS:\n\n":
		result_str += "Nothing Found"
	
	return result_str.rstrip("\n")

def arxiv_search(query: str, max_results: int=10) -> str:
	"""
	Performs a search of arXiv, an online archive of scientific papers.
    
	Args:
		query: The search query
		max_results: The maximum number of search results. Optional (default=10)
	"""
	if query.strip() == '': return "ERROR: search query is empty"
	
	result_str = "SEARCH RESULTS:\n\n"
	
	search = arxiv.Search(
		query = query,
		max_results = min(max_results, 99),
		sort_by = arxiv.SortCriterion.Relevance
	)
	
	for result in arxiv_client.results(search):
		id_parts = result.entry_id.split('/')
		paper_id = id_parts[len(id_parts)-1]
		result_str += "ID: "+paper_id+"\n"
		result_str += "Title: "+result.title+"\n"
		result_str += "Published: {year}-{month}-{day}".format(year=result.published.year, month=result.published.month, day=result.published.day)+"\n"
		result_str += "Summary: "+result.summary+"\n\n"
	
	if result_str == "SEARCH RESULTS:\n\n":
		result_str += "Nothing Found"
	
	return result_str.rstrip("\n")

def get_arxiv_paper(paper_id: str) -> str:
	"""
	Returns the text of an arXiv paper.
    
	Args:
		paper_id: The paper ID
	"""
	paper_dir = "./papers/"
	if not os.path.exists(paper_dir): os.mkdir(paper_dir)
	paper = next(arxiv_client.results(arxiv.Search(id_list=[paper_id])))
	paper.download_pdf(dirpath=paper_dir, filename=paper_id+".pdf")
	pdf_doc = pymupdf.open(paper_dir+paper_id+".pdf")
	pdf_txt = ''
	
	for page in pdf_doc:
		pdf_txt += page.get_text()+"\n\n"
	
	return pdf_txt.strip("\n")

def get_file_text(url: str) -> str:
	"""
	Returns the text of any file (e.g. source code files).
	Also supports reading text from PDF documents.
    
	Args:
		url: The URL of the file containing text
	"""
	clean_url = url.strip()
	if clean_url == '': return "ERROR: URL argument is empty"
	
	if clean_url.lower().endswith(".pdf"):
		paper_dir = "./papers/"
		file_name = clean_url.split('/')
		file_name = file_name[len(file_name)-1]
		file_path = os.path.join(paper_dir, file_name)
		if not os.path.exists(paper_dir): os.mkdir(paper_dir)
		
		if download_file(clean_url, file_path):
			if os.path.isfile(file_path):
				pdf_doc = pymupdf.open(file_path)
				pdf_txt = ''
				for page in pdf_doc:
					pdf_txt += page.get_text()+"\n\n"
			else:
				return "ERROR: failed to save file"
		else:
			return "ERROR: failed to download file"
		
		return pdf_txt.strip("\n")
	else:
		result = read_url(clean_url)
		return result['txt']

def get_web_page(url: str, format: str='markdown') -> str:
	"""
	Returns the contents of a web page as HTML or Markdown.
    
	Args:
		url: The URL of the web page
		format: Valid values are 'html' and 'markdown'. Optional (default='markdown')
	"""
	clean_url = url.strip()
	format_low = format.lower()
	if clean_url == '': return "ERROR: URL argument is empty"
	
	if fc_scraping:
		result = firecrawl.scrape(clean_url, formats=[format_low])
	
		if type(result) is str:
			return result
		elif format_low == 'markdown' and hasattr(result, 'markdown'):
			return result.markdown
		elif format_low == 'html' and hasattr(result, 'html'):
			return result.html
		else:
			return "ERROR: got unexpected response"
	
	result = read_url(clean_url)
	
	if format_low == "html" or result['err']:
		return result['txt']
	else:
		return convert_to_markdown(result['txt'],
			list_indent_width = 2,
			whitespace_mode = "strict",
			preprocessing_preset = "minimal",
			preprocess_html = True,
			remove_navigation = False,
			autolinks = False
		)

def get_wiki_page(page_name: str, only_summary: bool=True) -> str:
	"""
	Returns the text of a Wikipedia page.
    
	Args:
		page_name: The name of the Wikipedia page
		only_summary: Set to false to return the full page text instead of just the summary. Optional (default=True)
	"""
	if page_name.strip() == '': return "ERROR: page name is empty"
	
	wiki_page = wiki_client.page(page_name.replace(' ', '_'))
	
	if wiki_page.exists():
		if only_summary and not (len(wiki_page.summary) < len(page_name)+16 and wiki_page.summary.endswith("may refer to:")):
			return wiki_page.summary
		else:
			return wiki_page.text
	else:
		return "ERROR: no page named "+page_name
	
# FUNCTIONS FOR AIUI ENGINE

def CallToolFunc(func_name, func_args, aiui_funcs):
	try:
		if func_name == "web_search":
			return web_search(**func_args)
		elif func_name == "science_search":
			return science_search(**func_args)
		elif func_name == "github_search":
			return github_search(**func_args)
		elif func_name == "arxiv_search":
			return arxiv_search(**func_args)
		elif func_name == "get_arxiv_paper":
			return get_arxiv_paper(**func_args)
		elif func_name == "get_file_text":
			return get_file_text(**func_args)
		elif func_name == "get_web_page":
			return get_web_page(**func_args)
		elif func_name == "get_wiki_page":
			return get_wiki_page(**func_args)
		else:
			return "ERROR: unknown function"
	except Exception as e:
		return f"ERROR: an exception occured ({e})"

def GetToolFuncs():
	if fc_api_key == '' or fc_api_key == "YOUR-API-KEY":
		return [arxiv_search, get_arxiv_paper, get_file_text, get_web_page, get_wiki_page]
	else:
		return [web_search, science_search, github_search, arxiv_search, get_arxiv_paper, get_file_text, get_web_page, get_wiki_page]
