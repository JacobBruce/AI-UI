import os
import arxiv
import wikipediaapi
import pymupdf

arxiv_client = arxiv.Client()

wiki_client = wikipediaapi.Wikipedia(
    user_agent='AI UI (github.com/JacobBruce/AI-UI)',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    language='en'
)

# SEARCH TOOL FUNCTIONS

def arxiv_search(query: str, max_results: int=10) -> str:
	"""
	Performs a search of arXiv, an online archive of scientific papers.
    
	Args:
		query: The search query
		max_results: The maximum number of search results. Optional (default=10)
	"""
	result_str = "SEARCH RESULTS:\n\n"
	
	search = arxiv.Search(
		query = query,
		max_results = max_results,
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

def get_wiki_page(page_name: str, only_summary: bool=True) -> str:
	"""
	Returns the text of a Wikipedia page.
    
	Args:
		page_name: The name of the Wikipedia page
		only_summary: Set to false to return the full page text instead of just the summary. Optional (default=True)
	"""
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
		if func_name == "arxiv_search":
			if "max_results" in func_args:
				return arxiv_search(func_args['query'], func_args['max_results'])
			else:
				return arxiv_search(func_args['query'])
		elif func_name == "get_arxiv_paper":
			return get_arxiv_paper(func_args['paper_id'])
		elif func_name == "get_wiki_page":
			if "only_summary" in func_args:
				return get_wiki_page(func_args['page_name'], func_args['only_summary'])
			else:
				return get_wiki_page(func_args['page_name'])
		else:
			return "ERROR: unknown function"
	except:
		return "ERROR: request failed"

def GetToolFuncs():
	return [arxiv_search, get_arxiv_paper, get_wiki_page]
