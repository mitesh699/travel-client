# extract_brave_results.py
import httpx
import os
from dotenv import load_dotenv
import json

load_dotenv()
brave_api_key = os.getenv("BRAVE_API_KEY")

def search_brave(query: str, count: int = 5):
    """Perform a search using Brave Search API and print formatted results"""
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {
        "q": query,
        "count": count,
        "result_filter": "web"  # You can also use "web,news,videos" for multiple types
    }
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key
    }
    
    response = httpx.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        # Print query information
        original_query = data.get("query", {}).get("original", "")
        print(f"Search results for: {original_query}\n")
        
        # Extract and print web results
        if "web" in data and "results" in data["web"]:
            web_results = data["web"]["results"]
            print(f"Found {len(web_results)} web results\n")
            
            for i, result in enumerate(web_results, 1):
                title = result.get("title", "No Title")
                url = result.get("url", "")
                description = result.get("description", "No description available")
                
                print(f"{i}. {title}")
                print(f"   URL: {url}")
                print(f"   {description}")
                print("")
        
        # Extract news results if available
        if "news" in data and "results" in data["news"]:
            news_results = data["news"]["results"]
            print(f"Found {len(news_results)} news results\n")
            
            for i, result in enumerate(news_results, 1):
                title = result.get("title", "No Title")
                url = result.get("url", "")
                age = result.get("age", "")
                
                print(f"{i}. {title}")
                print(f"   URL: {url}")
                if age:
                    print(f"   Published: {age}")
                print("")
                
        # Return the full data for further processing
        return data
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    query = input("Enter your search query: ")
    search_brave(query)