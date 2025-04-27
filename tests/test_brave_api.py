# test_brave_api.py
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
brave_api_key = os.getenv("BRAVE_API_KEY")

url = "https://api.search.brave.com/res/v1/web/search"
params = {
    "q": "travel to paris",
    "count": 2
}

headers = {
    "Accept": "application/json",
    "X-Subscription-Token": brave_api_key
}

response = httpx.get(url, params=params, headers=headers)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("Success!")
else:
    print(f"Error: {response.text}")