# creating tools 
# 1) web_search
# 2) scrape_url

from langchain.tools import tool
import requests
from bs4 import BeautifulSoup  #its a web scrapping tool
from tavily import TavilyClient  #fetch data from tavily
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()


tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# abhi yeh tool web searches provide krega based on out request.
@tool
def web_search(query: str) -> str: 
    """"Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""

    results=tavily.search(query=query,max_results=5)

    out=[]

    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n\n") 

    return "\n------\n".join(out)




# second tool un web searches ki web site pr jaakr data scrapes krega which are relevant 

@tool
def scrape_url(url:str)->str:
    """Scrape and return clean text content from the given URL for deeper understanding."""

    try:
        response=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        soup=BeautifulSoup(response.text,
        "html.parser")
        for tag in soup(['script','style',"nav","footer"]):
            tag.decompose()

        return soup.get_text(separator=" ",strip=True)[:2000]  #returning first 2000 characters of the text content


    except Exception as e:
        return f"Error scraping the URL: {str(e)}"

