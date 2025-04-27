# src/search.py
from mcp.server.fastmcp import FastMCP
import logging
import httpx
import os
import asyncio # Added for sleep
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging (use INFO for less noise in production, DEBUG for development)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("search")
logger.info(f"Search MCP server initialized with name: search")

# Constants for Brave Search API
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_LOCAL_API_URL = "https://api.search.brave.com/res/v1/local/pois"

# Get API key from environment variables
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
if not BRAVE_API_KEY:
    logger.warning("BRAVE_API_KEY not set in environment variables")

@mcp.tool()
async def search_travel_info(query: str, category: str = "general") -> str:
    """Performs a general web search for travel information using Brave Search API.

    Args:
        query: The search query (e.g., "travel tips for Paris").
        category: A hint for the type of information (e.g., 'restaurants', 'attractions', 'general').

    Returns:
        Formatted string with top search results, or an error message.
    """
    logger.info(f"Tool called: search_travel_info for query: {query}, category: {category}")

    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not configured, using fallback response")
        # Fallback to simulated response if API key is not available
        return f"Search results for '{query}' in category '{category}':\n\n1. Example Result 1...\n2. Example Result 2..."

    try:
        # Prepare request parameters
        params = {
            "q": query,
            "count": 5,  # Limit to 5 results for readability
            "result_filter": "web"
        }
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }

        # Make the API request
        async with httpx.AsyncClient() as client:
            response = await client.get(
                BRAVE_SEARCH_API_URL,
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status() # Raise HTTP errors

        # Parse the response
        data = response.json()

        # Format the results
        results = []
        if "web" in data and "results" in data["web"]:
            results.append(f"Travel information for '{query}':\n")
            for i, result in enumerate(data["web"]["results"][:5], 1):
                title = result.get("title", "No Title")
                url = result.get("url", "")
                description = result.get("description", "").replace('...', '').strip()

                results.append(f"{i}. **{title}**")
                if url: results.append(f"   URL: [{url}]({url})") # Format URL as link
                if description: results.append(f"   *{description}*")
            return "\n".join(results)
        else:
            logger.warning(f"No web results found in Brave Search for query: {query}")
            return f"No travel information found for '{query}'"

    except httpx.HTTPStatusError as http_err:
         logger.error(f"Brave Search API HTTP error in search_travel_info: {http_err}")
         return f"Error searching for travel information: {http_err.response.status_code}"
    except Exception as e:
        logger.error(f"Error in search_travel_info: {str(e)}", exc_info=True)
        return f"Error searching for travel information: {str(e)}"

@mcp.tool()
async def find_flights(origin: str, destination: str, date: str = None) -> str:
    """
    Fallback Flight Search: Performs a web search for flights using Brave Search API.
    Note: This provides general search results, not structured flight offers.

    Args:
        origin: Departure city or airport code.
        destination: Arrival city or airport code.
        date: Optional departure date (YYYY-MM-DD).

    Returns:
        Formatted string with top search results related to flights, or an error message.
    """
    logger.info(f"Tool called: find_flights (Brave Fallback) from {origin} to {destination}, date: {date}")

    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not configured, using fallback response")
        # Fallback to simulated response
        date_info = f" on {date}" if date else ""
        return f"""Fallback Flight Search Results for {origin} to {destination}{date_info}:

1. **Google Flights** - Find cheap flights...
   [https://www.google.com/flights?q=flights+from+{origin}+to+{destination}{'+on+'+date if date else ''}](https://www.google.com/flights?q=flights+from+{origin}+to+{destination}{'+on+'+date if date else ''})
   *Use Google Flights to explore flight options...*

2. **Skyscanner** - Compare cheap flights...
   [https://www.skyscanner.com/transport/flights/{origin.lower()}/{destination.lower()}/{date.replace('-', '') if date else ''}](https://www.skyscanner.com/transport/flights/{origin.lower()}/{destination.lower()}/{date.replace('-', '') if date else ''})
   *Compare flights from all major airlines and travel agents...*
"""

    try:
        search_query = f"flights from {origin} to {destination}"
        if date: search_query += f" on {date}"

        params = {"q": search_query, "count": 5, "result_filter": "web"}
        headers = {
            "Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": BRAVE_API_KEY
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(BRAVE_SEARCH_API_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()

        data = response.json()
        results = []
        if "web" in data and "results" in data["web"]:
            results.append(f"Flight search web results for {origin} to {destination}:")
            for i, result in enumerate(data["web"]["results"][:5], 1):
                title = result.get("title", "No Title")
                url = result.get("url", "")
                description = result.get("description", "").replace('...','').strip()
                results.append(f"\n{i}. **{title}**")
                if url: results.append(f"   URL: [{url}]({url})")
                if description: results.append(f"   *{description}*")
            if not data["web"]["results"]:
                 results.append("\nNo specific flight links found. Try searching directly on airline or booking sites.")
        else:
            results.append("\nNo web results found for flights.")

        return "\n".join(results)

    except httpx.HTTPStatusError as http_err:
         logger.error(f"Brave Search API HTTP error in find_flights: {http_err}")
         return f"Error searching for flights via fallback: {http_err.response.status_code}"
    except Exception as e:
        logger.error(f"Error in find_flights (fallback): {str(e)}", exc_info=True)
        return f"Error searching for flights via fallback: {str(e)}"

# --- TripAdvisor Fallback Function ---
@mcp.tool()
async def get_tripadvisor_data(search_query: str, location: str) -> str:
    """
    Fallback: Searches Brave for TripAdvisor links based on a specific query.
    Formats results to highlight relevant TA pages.

    Args:
        search_query: The specific search term (e.g., "TripAdvisor best mexican restaurants San Diego").
        location: The general location name for context and manual search link.

    Returns:
        Formatted string with relevant TripAdvisor links or general web results.
    """
    logger.info(f"Tool called: get_tripadvisor_data (fallback) for query: '{search_query}' in location: {location}")

    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not configured, using fallback response")
        fallback_link = f"https://www.tripadvisor.com/Search?q={location.replace(' ', '+')}"
        return f"# TripAdvisor Fallback\nBrave API key missing. Please search manually.\n[Browse on TripAdvisor]({fallback_link})"

    try:
        params = {"q": search_query, "count": 10} # Get a few more results to filter
        headers = {
            "Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": BRAVE_API_KEY
        }

        async with httpx.AsyncClient() as client:
            logger.debug(f"Calling Brave Search API with query: {search_query}")
            response = await client.get(BRAVE_SEARCH_API_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status() # Raise HTTP errors

        data = response.json()
        formatted_output = [f"# TripAdvisor Fallback Search Results for {location}"]

        if "web" in data and "results" in data["web"]:
            results = data["web"]["results"]
            # Filter more specifically for TripAdvisor reviews/pages
            ta_results = [
                r for r in results
                if "tripadvisor.com" in r.get("url", "").lower() and
                any(k in r.get("url", "").lower() for k in ["restaurant_review", "hotel_review", "attraction_review", "tourism", "/Restaurants-", "/Hotels-", "/Attractions-"])
            ]

            if ta_results:
                formatted_output.append("\n## Relevant TripAdvisor Pages Found:")
                for i, r in enumerate(ta_results[:7], 1): # Show up to 7 relevant links
                    title = r.get('title','Link')
                    url = r.get('url','')
                    description = r.get('description','').replace('...', '').strip()
                    formatted_output.append(f"{i}. **{title}**")
                    if url: formatted_output.append(f"   [{url}]({url})") # Show URL as link text
                    if description: formatted_output.append(f"   *{description}*")
            else:
                formatted_output.append("\nNo specific TripAdvisor review/listing pages found via fallback search.")
                # Optionally add top 3 general web results if no TA links found
                if results:
                    formatted_output.append("\n## Top General Web Results:")
                    for i, r in enumerate(results[:3], 1):
                         title = r.get('title','Link')
                         url = r.get('url','')
                         description = r.get('description','').replace('...', '').strip()
                         formatted_output.append(f"{i}. **{title}**")
                         if url: formatted_output.append(f"   [{url}]({url})")
                         if description: formatted_output.append(f"   *{description}*")
        else:
            formatted_output.append("\nNo web results found via fallback search.")

        # Always add the manual search link
        manual_search_link = f"https://www.tripadvisor.com/Search?q={location.replace(' ', '+')}"
        formatted_output.append(f"\n[Browse more on TripAdvisor]({manual_search_link})")
        return "\n\n".join(formatted_output)

    except httpx.HTTPStatusError as http_err:
         logger.error(f"Brave Search API HTTP error in get_tripadvisor_data: {http_err}")
         fallback_link = f"https://www.tripadvisor.com/Search?q={location.replace(' ', '+')}"
         return f"# TripAdvisor Fallback\nError fetching data from Brave: {http_err.response.status_code}\n[Browse manually]({fallback_link})"
    except Exception as e:
        logger.error(f"Error in get_tripadvisor_data: {str(e)}", exc_info=True)
        fallback_link = f"https://www.tripadvisor.com/Search?q={location.replace(' ', '+')}"
        return f"# TripAdvisor Fallback\nError searching for TripAdvisor info: {str(e)}\n[Browse manually]({fallback_link})"


# --- REVISED search_local_businesses ---
@mcp.tool()
async def search_local_businesses(query: str, count: int = 5) -> str:
    """
    Search for local businesses (POIs) using Brave Local API, with fallback to web search.

    Args:
        query: Search query (e.g., "coffee shops near Pike Place Market Seattle")
        count: Number of results to return (max 20 for Local API)

    Returns:
        Text representation of local business search results or web search fallback.
    """
    logger.info(f"Tool called: search_local_businesses for query: {query}")

    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not configured, using fallback response")
        return f"Local business results for '{query}':\n\n1. Mock Business...\n2. Another Mock Place..." # Simple fallback

    try:
        # --- Attempt 1: Use Brave Local Search API ---
        logger.debug(f"Attempting Brave Local Search API for: {query}")
        local_params = {"q": query, "count": min(count * 2, 20)} # Request slightly more for filtering
        headers = {
            "Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": BRAVE_API_KEY
        }
        local_results = []
        response_local = None
        async with httpx.AsyncClient() as client:
            response_local = await client.get(
                BRAVE_LOCAL_API_URL, params=local_params, headers=headers, timeout=10
            )

        if response_local and response_local.status_code == 200:
            data_local = response_local.json()
            if "results" in data_local:
                local_results = data_local["results"]
                logger.info(f"Found {len(local_results)} results using Brave Local API.")
        elif response_local: # Log errors other than rate limits (429 handled implicitly by not having results)
             if response_local.status_code != 429:
                 logger.warning(f"Brave Local API error: {response_local.status_code} - {response_local.text}. Will fallback to web search.")
        else:
             logger.warning("Brave Local API call failed unexpectedly. Will fallback to web search.")


        # Format Local API results if found
        if local_results:
            results = [f"Local business results for '{query}':\n"]
            added_count = 0
            for result in local_results:
                if added_count >= count: break # Stop after adding 'count' results

                title = result.get("name", "No Name")
                # Basic address formatting
                address_dict = result.get("address", {})
                address_parts = [
                    address_dict.get("streetAddress"),
                    address_dict.get("addressLocality"),
                    address_dict.get("addressRegion"),
                    address_dict.get("postalCode")
                ]
                address = ", ".join(filter(None, address_parts)) or "Address not available"

                rating_info = ""
                if result.get("rating"):
                    rating_value = result["rating"].get("ratingValue", "N/A")
                    review_count = result["rating"].get("reviewCount", "0")
                    rating_info = f"Rating: {rating_value}/5 ({review_count} reviews)"

                results.append(f"{added_count + 1}. **{title}**")
                results.append(f"   {address}")
                if rating_info: results.append(f"   {rating_info}")
                if result.get("telephone"): results.append(f"   Phone: {result['telephone']}")
                if result.get("url"): results.append(f"   [Website]({result['url']})") # Format URL as link
                results.append("") # Add blank line for separation
                added_count += 1
            return "\n".join(results)

        # --- Attempt 2: Fallback to Web Search (if Local API failed or no results) ---
        else:
            logger.info(f"No valid results from Brave Local API for '{query}'. Falling back to web search.")
            # *** ADD DELAY BEFORE FALLBACK CALL ***
            logger.debug("Waiting 1.1 seconds before fallback web search to avoid rate limit...")
            await asyncio.sleep(1.1) # Add delay
            # *** END DELAY ***

            # Use search_travel_info, trying to infer category
            category_hint = "businesses" # Default
            if "restaurant" in query.lower(): category_hint = "restaurants"
            elif "hotel" in query.lower(): category_hint = "hotels"
            elif "attraction" in query.lower() or "museum" in query.lower() or "park" in query.lower(): category_hint = "attractions"
            elif "coffee" in query.lower(): category_hint = "coffee shops"

            logger.info(f"Calling search_travel_info as fallback with query='{query}', category='{category_hint}'")
            fallback_result = await search_travel_info(query, category=category_hint)
            return f"Could not find specific local results, showing general web search results for '{query}':\n\n{fallback_result}"

    except httpx.HTTPStatusError as http_err:
         logger.error(f"Brave Search API HTTP error in search_local_businesses: {http_err}")
         return f"Error searching for local businesses: {http_err.response.status_code}"
    except Exception as e:
        logger.error(f"Error in search_local_businesses: {str(e)}", exc_info=True)
        return f"Error searching for local businesses: {str(e)}"


# --- Main execution block ---
if __name__ == "__main__":
    import sys
    transport = "stdio"
    if len(sys.argv) > 1: transport = sys.argv[1]
    logger.info(f"Starting search MCP server with transport: {transport}")
    try:
        mcp.run(transport=transport) # Use run for stdio
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")