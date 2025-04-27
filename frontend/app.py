import asyncio
import json
import logging
import os
import re
import sys
import traceback
from datetime import date, datetime, timedelta
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# MCP client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file
load_dotenv(override=True)

# --- Setup Logger ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
GMAPS_PLACE_URL_BASE = "google.com/maps/search/?api=1&query=...&query_place_id=...?q="  # Base for search links
GMAPS_PLACE_ID_URL_BASE = "https://www.google.com/maps/search/?api=1&query=...&query_place_id=...:"  # Base for Place ID links
GMAPS_DIRECTIONS_URL_BASE = "https://www.google.com/maps/place/?q=place_id:ChIJYfYBZQHyloARXBIgn69QrHQ"  # Base for Directions

# List of US states/territories (used for entity extraction hinting)
US_STATES_TERRITORIES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "AS", "DC", "FM", "GU", "MH", "MP", "PW", "PR", "VI",
    "USA", "U.S.A", "UNITED STATES",
}

# --- Airline Name to Code Mapping ---
AIRLINE_NAME_TO_CODE = {
    "delta": "DL", "delta air lines": "DL",
    "united": "UA", "united airlines": "UA",
    "american": "AA", "american airlines": "AA",
    "southwest": "WN", "southwest airlines": "WN",
    "jetblue": "B6", "jetblue airways": "B6",
    "alaska": "AS", "alaska airlines": "AS",
    "spirit": "NK", "spirit airlines": "NK",
    "frontier": "F9", "frontier airlines": "F9",
    "british airways": "BA", "ba": "BA",
    "air france": "AF",
    "lufthansa": "LH",
    "klm": "KL",
    "emirates": "EK",
    "qatar airways": "QR",
    "virgin atlantic": "VS",
}



# Ensure place_keywords_map is defined here
place_keywords_map = {
    "restaurants": "restaurants",
    "food": "restaurants",
    "dining": "restaurants",
    "eats": "restaurants",
    "bars": "bars",
    "pubs": "bars",
    "cafes": "cafes",
    "coffee shops": "cafes",
    "bakeries": "bakeries",
    "museums": "museums",
    "art gallery": "museums",
    "galleries": "museums",
    "landmarks": "attractions",
    "tourist attractions": "attractions",
    "sights": "attractions",
    "points of interest": "attractions",
    "parks": "parks",
    "gardens": "parks",
    "shopping": "shopping",
    "malls": "shopping",
    "stores": "shopping",
    "markets": "markets", # Keep separate from shopping malls
    "farmers market": "markets",
    "hotels": "hotels", # Add if needed, though Accommodations tool usually handles this
    "accommodations": "hotels",
    "lodging": "hotels",
    # Add more mappings as needed
}

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi there! 👋 I'm your AI travel assistant. "
                "How can I help you plan your trip?"
            ),
        }
    ]

if "search_options" not in st.session_state:
    st.session_state.search_options = {
        "airbnb": {
            "adults": 2, "children": 0, "infants": 0, "pets": 0,
            "min_price": None, "max_price": None,
        },
        "flight": {
            "travel_class": "ECONOMY", "non_stop": False, "max_price": None,
        },
        "gmaps": {"max_results": 5},
        "weather": {"days": 3, "aqi": "yes", "alerts": "yes"}, # Added weather defaults
    }
# Ensure sub-keys exist if 'search_options' was already partially initialized
elif "gmaps" not in st.session_state.search_options:
    st.session_state.search_options["gmaps"] = {"max_results": 5}
elif "weather" not in st.session_state.search_options:
    st.session_state.search_options["weather"] = {
        "days": 3, "aqi": "yes", "alerts": "yes"
    }


# --- Sidebar UI ---
with st.sidebar:
    st.header("Travel Planning Options")

    st.subheader("General Options")
    if st.button("Reset Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi there! 👋 I'm your AI travel assistant. How can I help?",
            }
        ]
        # Reset options to default structure
        st.session_state.search_options = {
            "airbnb": {
                "adults": 2, "children": 0, "infants": 0, "pets": 0,
                "min_price": None, "max_price": None,
            },
            "flight": {
                "travel_class": "ECONOMY", "non_stop": False, "max_price": None,
            },
            "gmaps": {"max_results": 5},
            "weather": {"days": 3, "aqi": "yes", "alerts": "yes"}, # Reset weather
        }
        st.rerun()

    st.subheader("Advanced Search Options")

    with st.expander("Airbnb Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            adults = st.number_input(
                "Adults", 1, 16, st.session_state.search_options["airbnb"]["adults"]
            )
            children = st.number_input(
                "Children", 0, 5, st.session_state.search_options["airbnb"]["children"]
            )
        with col2:
            infants = st.number_input(
                "Infants", 0, 5, st.session_state.search_options["airbnb"]["infants"]
            )
            pets = st.number_input(
                "Pets", 0, 5, st.session_state.search_options["airbnb"]["pets"]
            )

        current_min_price = st.session_state.search_options["airbnb"].get("min_price") or 0
        current_max_price = st.session_state.search_options["airbnb"].get("max_price") or 1000
        price_range = st.slider(
            "Price Range ($)", 0, 1000, (current_min_price, current_max_price), 50
        )

        if st.button("Save Airbnb Settings"):
            st.session_state.search_options["airbnb"] = {
                "adults": adults,
                "children": children,
                "infants": infants,
                "pets": pets,
                "min_price": price_range[0] if price_range[0] > 0 else None,
                "max_price": price_range[1] if price_range[1] < 1000 else None,
            }
            st.success("Airbnb settings saved!")

    with st.expander("Flight Search Options"):
        current_travel_class = st.session_state.search_options["flight"].get(
            "travel_class", "ECONOMY"
        )
        class_options = ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
        class_index = (
            class_options.index(current_travel_class)
            if current_travel_class in class_options
            else 0
        )
        travel_class = st.selectbox(
            "Travel Class", class_options, index=class_index
        )
        if st.button("Save Flight Settings"):
            st.session_state.search_options["flight"]["travel_class"] = travel_class
            st.success("Flight settings saved!")

    with st.expander("Google Maps Options"):
        gmaps_max_results = st.slider(
            "Max Places to Show per Search",
            min_value=1,
            max_value=10,
            value=st.session_state.search_options["gmaps"]["max_results"],
            step=1,
            help=(
                "Controls how many results from Google Maps searches "
                "(restaurants, attractions, etc.) are shown."
            ),
        )
        if st.button("Save Google Maps Settings"):
            st.session_state.search_options["gmaps"]["max_results"] = gmaps_max_results
            st.success("Google Maps settings saved!")

    # Added Weather Options
    with st.expander("Weather Options"):
        weather_days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=10,  # WeatherAPI free allows up to 10
            value=st.session_state.search_options["weather"]["days"],
            step=1,
            help="Number of forecast days to retrieve (max 10 for free tier).",
        )
        weather_aqi = st.checkbox(
            "Include Air Quality (AQI)",
            value=st.session_state.search_options["weather"]["aqi"] == "yes",
        )
        weather_alerts = st.checkbox(
            "Include Weather Alerts",
            value=st.session_state.search_options["weather"]["alerts"] == "yes",
        )
        if st.button("Save Weather Settings"):
            st.session_state.search_options["weather"] = {
                "days": weather_days,
                "aqi": "yes" if weather_aqi else "no",
                "alerts": "yes" if weather_alerts else "no",
            }
            st.success("Weather settings saved!")

    st.subheader("Example Queries")
    st.markdown(
        """
        - Plan a 5-day trip to Seattle in July. Include flights from SFO, a pet-friendly Airbnb near Pike Place Market, weather forecast, and some museum recommendations.
        - What's the weather like in Paris, France right now?
        - Find business class flights from LHR to JFK for next Monday.
        - What are some highly-rated Italian restaurants in Boston? Show me the current weather there too.
        - Show me hotels near Disneyland California for 2 adults and 2 children. What's the weather forecast?
        - What's the driving time from San Jose to Los Angeles?
        - Get the weather forecast and AQI for Mumbai, India.
        """
    )  # Added Mumbai example


# Page layout
st.title("🗺️ MCP-Powered Travel Assistant")
#st.write(
#    "Using Groq LLM, Google Maps, Amadeus, Airbnb, Search, "
#    "and a custom WeatherAPI server."
#)

# --- Helper Functions ---

def extract_mcp_result_text(result) -> Union[str, None]:
    """
    Extract text content from MCP result.

    Prioritizes the .text attribute within result.content if it's structured
    like [TextContent(text="...")].
    Returns a string (potentially JSON) or None.
    """
    if not result:
        logger.debug("extract_mcp_result_text received None, returning None.")
        return None

    if hasattr(result, "content") and result.content:
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            first_item = content[0]
            if hasattr(first_item, "text") and isinstance(first_item.text, str):
                logger.debug("Extracting text from result.content[0].text.")
                return first_item.text
            else:
                logger.warning(
                    f"result.content[0] ({type(first_item)}) lacks .text "
                    "attribute. Falling back to str(content)."
                )
                return str(content)
        elif hasattr(content, "text"):
            logger.debug("Extracting text from result.content.text.")
            return content.text
        else:
            logger.debug("Returning string representation of result.content.")
            return str(content)
    elif hasattr(result, "text"):
        logger.debug("Extracting text from result.text.")
        return result.text
    elif isinstance(result, str):
        logger.debug("Returning result as it is a string.")
        return result

    logger.warning(
        f"Could not extract text reliably from result ({type(result)}). "
        "Returning str(result)."
    )
    return str(result)


def format_price_label(label: str) -> str:
    """Cleans up Airbnb price label."""
    if not label or not isinstance(label, str):
        return "Price info unavailable"

    cleaned = re.sub(r'\s+', ' ', label).strip()
    # Attempt to normalize "X for Y nights" variations
    cleaned = re.sub(
        r'(\d+)\s*f\s*o\s*r\s*(\d+)\s*n\s*i\s*g\s*h\s*t\s*s?',
        r'$\1 for \2 nights',
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r'(\$\s?[\d,]+(?:\.\d+)?)\s*for\s*(\d+)\s*nights?', # Added optional decimal for price
        r'\1 for \2 nights',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r'originally\s*', r'originally ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r'(\d+)\s*x\s*(\d+)\s*nights?:',
        r'\1 x \2 nights:',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r'\$\s*', '$', cleaned)
    cleaned = re.sub(r'(?<=\d),(?=\d)', '', cleaned) # Remove commas in numbers
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Handle overly long labels which might indicate parsing issues
    if len(cleaned) > 70 and '\n' in label: # Increased threshold slightly
        first_part = cleaned.split()[0]
        if '$' in first_part:
             return f"Price: {first_part} (details may be unclear)"
        else:
             # If first part doesn't look like a price, return more generic message
             return "Price info may be unclear"

    # If only a number like "$XXXX" remains, label it
    if re.match(r'^\$\d+(?:,\d+)*(?:\.\d+)?$', cleaned):
        return f"Total price: {cleaned}"

    return cleaned if cleaned else "Price info unavailable"


def format_rating_label(label: str) -> str:
    """Cleans up Airbnb rating label."""
    if not label or not isinstance(label, str):
        return "No rating"

    cleaned = re.sub(r'\s+', ' ', label).strip()
    # Removed aggressive space removal as it might merge things incorrectly

    rating_match = re.search(r'(\d\.?\d*)\s?out\s?of\s?5', cleaned, re.IGNORECASE)
    reviews_match = re.search(r'(\d+)\s?review', cleaned, re.IGNORECASE)

    rating_part = f"{rating_match.group(1)}/5" if rating_match else "No rating"
    reviews_part = f"({reviews_match.group(1)} reviews)" if reviews_match else ""

    if rating_match:
        return f"{rating_part} {reviews_part}".strip()
    else:
        # Try a simpler rating match at the start (e.g., "4.8 (10 reviews)")
        # Match digits, potential dot, digits, then optional space and parenthesis
        simple_rating_match = re.search(r'^(\d\.\d+|\d)\s*\(?', cleaned)
        if simple_rating_match:
            rating_part = f"{simple_rating_match.group(1)}/5"
            # Try to find reviews number separately if not captured above
            if not reviews_match:
                reviews_match = re.search(r'\((\d+)\s?review', cleaned, re.IGNORECASE)
                reviews_part = f"({reviews_match.group(1)} reviews)" if reviews_match else ""
            return f"{rating_part} {reviews_part}".strip()

    return "No rating" # Default if no patterns match


def get_airport_code(location_str: str) -> Union[str, None]:
    """Converts location string to IATA airport code if possible."""
    if not location_str or not isinstance(location_str, str):
        return None

    location_upper = location_str.strip().upper()

    # Already a 3-letter code?
    if len(location_upper) == 3 and location_upper.isalpha():
        return location_upper

    # Check for format "City (XYZ)"
    match_city_code = re.match(
        r'([a-zA-Z\s.\'-]+)\s*\(\s*([A-Z]{3})\s*\)', location_str.strip()
    )
    if match_city_code:
        return match_city_code.group(2).upper()

    # Known mappings (expand as needed)
    airport_codes = {
        "san francisco": "SFO", "sfo": "SFO",
        "seattle": "SEA", "seatac": "SEA", "sea": "SEA",
        "new york": "JFK", "nyc": "JFK", "new york city": "JFK", "jfk": "JFK",
        "laguardia": "LGA", "newark": "EWR",
        "los angeles": "LAX", "la": "LAX", "lax": "LAX",
        "london": "LHR", "heathrow": "LHR", "gatwick": "LGW", "lon": "LON",
        "paris": "CDG", "charles de gaulle": "CDG", "orly": "ORY", "par": "PAR",
        "chicago": "ORD", "o'hare": "ORD", "midway": "MDW", "ord": "ORD",
        "boston": "BOS", "logan": "BOS", "bos": "BOS",
        "austin": "AUS", "aus": "AUS",
        "san jose": "SJC", "sjc": "SJC",
        "san diego": "SAN", "san": "SAN",
        "denver": "DEN", "den": "DEN",
        "portland": "PDX", "pdx": "PDX",
        "cedar city": "CDC", "cdc": "CDC",
        "st. george": "SGU", "sgu": "SGU",
        "springdale": "SGU", # Near Zion
        "zion national park": "SGU", # Map park to nearest airport
        "disneyland": "SNA", "disneyland california": "SNA", # Map landmark to nearest airport
        "pike place market": "SEA", # Map landmark to nearest airport
        "menlo park": "SFO", # Assuming SFO is closest major
        "mumbai": "BOM", # Added Mumbai
    }
    location_lower = location_str.lower().strip()

    # Specific landmark/area checks first
    if "zion" in location_lower:
        return airport_codes["zion national park"]
    if "disneyland" in location_lower:
        return airport_codes["disneyland"]
    if "pike place" in location_lower:
        return airport_codes["pike place market"]

    # Direct lookup
    if location_lower in airport_codes:
        return airport_codes[location_lower]

    # Check if input *is* already a code in the values (e.g., user typed "LHR")
    # Check if input looks like a code (3 alphanumeric chars) - slightly looser
    if location_upper in airport_codes.values() or (
        len(location_upper) == 3 and location_upper.isalnum()
    ):
        return location_upper

    logger.warning(f"Could not map location '{location_str}' to an airport code.")
    return None


def map_airline_name_to_code(name_or_code: str) -> Union[str, None]:
    """Maps airline name or code to IATA airline code."""
    if not name_or_code or not isinstance(name_or_code, str):
        return None

    name_or_code_upper = name_or_code.strip().upper()

    # Already a 2-letter/digit code?
    if len(name_or_code_upper) == 2 and name_or_code_upper.isalnum():
        return name_or_code_upper

    # Lookup by name
    name_lower = name_or_code.strip().lower()
    return AIRLINE_NAME_TO_CODE.get(name_lower)



def extract_markdown_links(text: str) -> Dict[str, str]:
    """Finds all markdown links in text and returns a dict of {URL: Full Markdown Link}."""
    # Regex to find markdown links: [Link Text](URL)
    # It captures the full link and the URL separately
    # Allows for URLs without http/https prefix if they start with common domains or paths
    pattern = r'(\[([^\]]+)\]\(((?:https?://|www\.|/)[^\)\s]+)\))'
    links = {}
    try:
        for match in re.findall(pattern, text):
            full_link_markdown = match[0]
            url = match[2]
            if url not in links: # Store the first instance of each URL
                links[url] = full_link_markdown
    except Exception as e:
        # Log error if regex fails unexpectedly
        logger.error(f"Error extracting markdown links: {e}", exc_info=True)
    return links


def celsius_to_fahrenheit(celsius: Optional[Any]) -> Optional[int]:
    """Converts Celsius (float/int/numeric str) to Fahrenheit and rounds."""
    if celsius is None:
        return None
    try:
        # Use 9.0/5.0 for float division
        return int(round(float(celsius) * 9.0 / 5.0 + 32))
    except (TypeError, ValueError) as e:
        logger.error(
            f"Failed to convert Celsius value '{celsius}' ({type(celsius)}) "
            f"to Fahrenheit: {e}"
        )
        return None


# --- LLM Setup ---

def get_llm():
    """Initialize and return the Groq LLM."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        # model_name = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
        # Use Llama 3 70b for potentially better link inclusion / instruction following
        model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        if not api_key:
            st.error("GROQ_API_KEY environment variable not set.")
            return None
        return ChatGroq(api_key=api_key, model_name=model_name, temperature=0.1)
    except Exception as e:
        st.error(f"LLM Init Error: {e}")
        return None


# --- System Prompt (Updated for Custom Weather Tool & Link Emphasis) ---

def create_system_prompt() -> str:
    """Create a custom system prompt reflecting the custom weather tool."""
    return """
You are a sophisticated AI travel assistant designed to help users plan trips by leveraging specialized tools.

Available Tools:
- Weather Information: Provides current conditions, forecasts (up to 10 days), future weather predictions (14-300 days out), AQI, and alerts via a custom WeatherAPI server. Specify forecast days if needed.
- Accommodation Search: Finds Airbnb listings based on location, dates, group size, and price via @openbnb/mcp-server-airbnb. Provides details and links.
- Flight Search: Finds flight options using Amadeus data (preferred) or Brave search (fallback). Provides price, times, airline, stops, and search links. Can filter by preferred airline if specified.
- Google Maps Place Search: Finds places like restaurants, attractions, museums, coffee shops, etc., using Google Maps. Provides name, address, rating, reviews, and map links.
- Google Maps Directions/Routing: Can calculate driving times or provide directions (if explicitly requested - *currently triggered by specific keywords like 'driving time'*). Provides details and map links.
- Local Business Search (Fallback): Finds local businesses using Brave search (used if Google Maps search is insufficient or fails). May provide website links.
- General Travel Search: Performs general web searches for travel tips, best times to visit, etc., using Brave search. May provide website links.

Response Guidelines:
1.  **Strict Context Adherence:** ***IMPORTANT: Base your response ONLY on the user's query and the specific tool data provided in the CONTEXT section below. Do NOT mention tools, capabilities, or information sources if they are not present in the CONTEXT for the current request.***
2.  **Synthesize Information:** Combine information from the provided CONTEXT into a cohesive response or itinerary. Don't just list raw tool outputs.
3.  **Address All Aspects:** Ensure all parts of the user's query (location, dates, activities, flights, accommodation, preferred airline etc.) are addressed *using the provided tool CONTEXT*.
4.  **Prioritize & Select:** Do not list *all* results from tools in the CONTEXT. Select the most relevant (e.g., top 3-5 flight options sorted by price, top 3-5 hotels/restaurants/attractions). Summarize findings. If an airline filter was applied and results are shown, state that clearly. Extract specific recommendations (names, details) for requested categories (restaurants, museums, etc.) directly from the Google Maps Search or Travel Info context *if available*.
5.  **Format Clearly:** Use markdown formatting (bolding, bullet points, headers) for readability. Structure itineraries logically (e.g., by day).
6.  **Include Key Details:** For flights: price, airline, times, duration, stops. For accommodations: name, price, rating. For Google Maps places: name, address, rating. For Weather: Summarize the key information provided in the formatted weather context (current temp/condition, forecast highlights, future prediction if applicable). Mention AQI/alerts if present.
7.  **Actionable Links:** You **MUST** include the primary actionable links (e.g., flight search link, accommodation search link, main place/direction links, specific listing links) when they are provided in the CONTEXT. Ensure they are correctly formatted as Markdown links (e.g., `[Link Text](URL)`). Do not omit them in your summary.
8.  **Weather Integration:** Use the weather information provided in the CONTEXT. The custom tool handles both US and international locations.
9.  **Handle Errors Gracefully:** If the CONTEXT for a specific tool contains error notes ('[Tool Error/No Results]', 'ERROR_MCP:', 'ERROR_TOOL:') acknowledge this briefly *only if that tool was directly relevant to the user's specific request*. Construct the best possible response using the information that *is* available. Mention if fallback search results were used.
10. **Be Conversational:** Maintain a helpful, friendly, and informative tone, *while strictly adhering to the provided CONTEXT*.

Combine the information gathered from the tools provided in the CONTEXT to create the best possible travel plan or answer for the user, following these guidelines.
"""


# --- LLM-based Entity Extraction ---

async def extract_entities_with_llm(query: str) -> dict:
    """Uses the Groq LLM to extract travel entities from the user query."""
    llm = get_llm()
    default_entities = {
        "primary_destination": None, "flight_origin": None, "flight_destination": None,
        "weather_location_query": None, "start_date": None, "end_date": None,
        "num_adults": 1, "num_children": 0, "num_infants": 0, "num_pets": 0,
        "preferred_airline": None, "specific_requests": [], "future_weather_date": None
    }
    if not llm:
        st.error("LLM not available for entity extraction.")
        return default_entities

    # Added future_weather_date field
    json_schema = """
{
    "primary_destination": "Primary destination city, region, or specific landmark.",
    "flight_origin": "Departure city or 3-letter airport code.",
    "flight_destination": "Arrival city or 3-letter airport code.",
    "weather_location_query": "The location string suitable for a weather query (e.g., 'Seattle', 'Paris, France', '90210'). If primary_destination is specific (e.g. landmark), use nearest city/airport.",
    "start_date": "Start date in 'YYYY-MM-DD' format.",
    "end_date": "End date in 'YYYY-MM-DD' format.",
    "num_adults": "Number of adults. Default 1.",
    "num_children": "Number of children. Default 0.",
    "num_infants": "Number of infants. Default 0.",
    "num_pets": "Number of pets. Default 0.",
    "preferred_airline": "Preferred airline name/code.",
    "specific_requests": ["List distinct user requirements like 'pet-friendly Airbnb', 'visit Louvre', 'weather forecast', 'driving time', 'AQI', 'future weather for DATE'],
    "future_weather_date": "Specific future date 'YYYY-MM-DD' if requested for weather (e.g., 'weather on May 15th'). MUST be between 14 and 300 days from today."
}
"""
    current_date_info = (
        f"Today's date is {datetime.now().strftime('%Y-%m-%d, %A')}. "
        f"Current Year: {datetime.now().year}."
    )
    prompt_text = f"""
Analyze the user travel query STRICTLY based on {current_date_info}.
Extract key entities according to the JSON schema below.
Infer future dates. Format dates 'YYYY-MM-DD'. Calculate end dates from duration.
Use 3-letter airport codes if possible for flights.
For `weather_location_query`, use the most appropriate location identifier from the query for the weather API.
Extract group size numbers. List specific needs/activities/details requested.
If a specific future date for weather is requested (14-300 days out), extract it into `future_weather_date`.
Return *only* the valid JSON object.

User Query: "{query}"
JSON Schema: {json_schema}
Extracted JSON:
"""
    messages = [
        SystemMessage(content="Extract travel entities into JSON. Return only JSON."),
        HumanMessage(content=prompt_text)
    ]
    llm_output = 'N/A'
    try:
        logger.info("Invoking LLM for entity extraction...")
        response = await llm.ainvoke(messages)
        llm_output = response.content.strip()
        logger.debug(f"--- DEBUG: LLM Extraction Raw Output ---\n{llm_output}")

        # Extract JSON block more robustly
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("LLM did not return a recognizable JSON object.")
        json_string = llm_output[json_start:json_end]

        parsed_entities = json.loads(json_string)
        final_entities = {
            "primary_destination": parsed_entities.get("primary_destination"),
            "flight_origin": parsed_entities.get("flight_origin"),
            "flight_destination": parsed_entities.get("flight_destination"),
            "weather_location_query": parsed_entities.get("weather_location_query"),
            "start_date": parsed_entities.get("start_date"),
            "end_date": parsed_entities.get("end_date"),
            "num_adults": int(parsed_entities.get("num_adults", 1)),
            "num_children": int(parsed_entities.get("num_children", 0)),
            "num_infants": int(parsed_entities.get("num_infants", 0)),
            "num_pets": int(parsed_entities.get("num_pets", 0)),
            "preferred_airline": parsed_entities.get("preferred_airline"),
            "specific_requests": parsed_entities.get("specific_requests", []),
            "future_weather_date": parsed_entities.get("future_weather_date"),
        }

        # Ensure at least one adult if children/infants present
        if (final_entities["num_adults"] < 1 and
                (final_entities["num_children"] > 0 or final_entities["num_infants"] > 0)):
            final_entities["num_adults"] = 1

        # Post-process locations to codes
        origin_input = final_entities.get("flight_origin")
        dest_input = final_entities.get("flight_destination")
        final_entities["flight_origin"] = get_airport_code(origin_input) if origin_input else None
        final_entities["flight_destination"] = (
            get_airport_code(dest_input) if dest_input else (
                get_airport_code(final_entities.get("primary_destination"))
                if final_entities.get("primary_destination") else None
            )
        )

        # Basic validation for future_weather_date if present
        if fw_date := final_entities.get("future_weather_date"):
            try:
                target_date = datetime.strptime(fw_date, "%Y-%m-%d").date()
                today = date.today()
                min_future = today + timedelta(days=14)
                max_future = today + timedelta(days=300)
                if not (min_future <= target_date <= max_future):
                    logger.warning(
                        f"Extracted future_weather_date '{fw_date}' is outside "
                        "the valid range (14-300 days). Clearing it."
                    )
                    final_entities["future_weather_date"] = None
            except ValueError:
                logger.warning(
                    f"Extracted future_weather_date '{fw_date}' is not a "
                    "valid YYYY-MM-DD date. Clearing it."
                )
                final_entities["future_weather_date"] = None

        logger.info(f"--- DEBUG: Final Extracted Entities (Post-Processing) ---\n{final_entities}")
        return final_entities

    except json.JSONDecodeError as json_err:
        st.error(f"Error parsing LLM JSON output: {json_err}")
        logger.error(
            f"LLM entity extraction failed - JSON parsing. Raw Output:\n{llm_output}",
            exc_info=True
        )
        return default_entities
    except Exception as e:
        st.error(f"Error during LLM entity extraction: {e}")
        logger.error(
            f"LLM entity extraction failed. Raw Output:\n{llm_output}", exc_info=True
        )
        return default_entities


# --- MCP Tool Calling Base Function ---

async def call_mcp_tool_base(
    server_script: str,
    tool_name: str,
    tool_params: dict,
    server_command: str = "python",
    args_prefix: list = None,  # Use None as default for mutable list
    env: dict | None = None,
    connect_timeout: float = 60.0,
    tool_call_timeout: float = 180.0,
) -> str:
    """Base function to call any stdio MCP server script using async/await."""
    if args_prefix is None:
        args_prefix = [] # Initialize here

    log_command_str = ""
    script_full_path = None
    try:
        command = server_command
        args = []
        # Assuming src is one level up from frontend directory where app.py resides
        base_dir = os.path.join(os.path.dirname(__file__), "..", "src")

        if server_command == "python":
            python_executable = sys.executable
            command = python_executable
            script_full_path = os.path.abspath(os.path.join(base_dir, server_script))
            # Fallback: check relative to current working directory
            if not os.path.exists(script_full_path):
                script_full_path = os.path.abspath(os.path.join("src", server_script))
            if not os.path.exists(script_full_path):
                 raise FileNotFoundError(f"Cannot find Python script: {server_script} (looked in {base_dir} and src/)")
            args = [script_full_path, "stdio"]
            log_command_str = f"{command} {' '.join(args)}"
        elif server_command == "npx":
            command = "npx"
            args = args_prefix # e.g., ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
            log_command_str = f"{command} {' '.join(args)}"
        else: # Treat server_script as the command itself
            command = server_script
            args = args_prefix
            log_command_str = f"{command} {' '.join(args)}"

        logger.info(f"--- DEBUG: MCP Launch Command for {tool_name} ---")
        logger.info(f"   Command: {log_command_str}")
        logger.info(f"   Params: {json.dumps(tool_params, default=str)}")

        server_params = StdioServerParameters(command=command, args=args, env=env)

        async with stdio_client(server_params) as client_tuple:
            if client_tuple is None:
                raise ConnectionError("Failed to establish MCP connection")

            read, write = client_tuple
            async with ClientSession(read, write) as session:
                logger.debug(f"Initializing MCP session for {tool_name}...")
                await asyncio.wait_for(session.initialize(), timeout=connect_timeout)
                logger.debug(f"MCP Session for {tool_name} initialized. Calling tool...")
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, tool_params),
                    timeout=tool_call_timeout,
                )
                logger.debug(f"MCP Tool call {tool_name} completed.")
                return extract_mcp_result_text(result) # Returns string (JSON or text)

    except FileNotFoundError as fnf_error:
        st.error(f"MCP Script Error: {fnf_error}")
        logger.error(f"MCP Script not found: {fnf_error}", exc_info=True)
        return f"ERROR_MCP: Could not locate script needed for {tool_name}."
    except ConnectionError as conn_err:
        st.error(f"MCP Connection Error for '{log_command_str}': {conn_err}")
        logger.error(f"MCP Connection Error for {tool_name}: {conn_err}", exc_info=True)
        return f"ERROR_MCP: Connection Error for {tool_name}: {conn_err}"
    except asyncio.TimeoutError:
        st.error(f"MCP operation for {tool_name} via '{log_command_str}' timed out.")
        logger.error(f"MCP Timeout for {tool_name}", exc_info=True)
        return f"ERROR_MCP: Operation timed out for {tool_name}."
    except Exception as e:
        st.error(f"Error calling {tool_name} MCP tool via '{log_command_str}': {str(e)}")
        logger.error(f"MCP call failed for {tool_name}", exc_info=True)
        return f"ERROR_MCP: Error calling {tool_name}: {str(e)}"




# --- NEW/UPDATED Weather Wrappers ---

async def call_custom_weather_forecast_tool(
    location: str, days: int = 3, aqi: str = "yes", alerts: str = "yes"
) -> str:
    """Calls the custom WeatherAPI MCP server's forecast tool."""
    if not location:
        return "ERROR_MCP: Location missing for weather forecast search."
    logger.info(f"Calling custom weather forecast tool for {location}, days={days}")
    tool_params = {"location": location, "days": days, "aqi": aqi, "alerts": alerts}
    return await call_mcp_tool_base(
        "custom_weatherapi_mcp.py", "get_weatherapi_forecast", tool_params
    )


async def call_custom_future_weather_tool(location: str, date: str) -> str:
    """Calls the custom WeatherAPI MCP server's future weather tool."""
    if not location:
        return "ERROR_MCP: Location missing for future weather search."
    if not date:
        return "ERROR_MCP: Date missing for future weather search."
    logger.info(f"Calling custom future weather tool for {location}, date={date}")
    tool_params = {"location": location, "date": date}
    return await call_mcp_tool_base(
        "custom_weatherapi_mcp.py", "get_future_weather", tool_params
    )


# --- Other Tool Wrappers (Mostly Unchanged, Formatting Adjusted) ---

async def call_airbnb_tool_with_params(params: dict) -> str:
    """Calls the Airbnb MCP server via npx (async)."""
    location = params.get('location')
    if not location:
        return "ERROR_MCP: Location missing for Airbnb search."

    logger.info(f"Calling Airbnb tool for {location}")
    tool_name = "airbnb_search"
    tool_params = {
        "location": params.get("location"),
        "checkin": params.get("check_in"),
        "checkout": params.get("check_out"),
        "adults": params.get("adults"),
        "children": params.get("children"),
        "infants": params.get("infants"),
        "pets": params.get("pets"),
        "minPrice": params.get("min_price"),
        "maxPrice": params.get("max_price"),
    }
    # Remove keys with None values
    tool_params = {k: v for k, v in tool_params.items() if v is not None}
    npx_args = ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]

    # Result formatting is now handled by enhance_results
    return await call_mcp_tool_base(
        server_script="@openbnb/mcp-server-airbnb", # This is the package name for npx
        tool_name=tool_name,
        tool_params=tool_params,
        server_command="npx",
        args_prefix=npx_args,
        tool_call_timeout=240.0, # Longer timeout for Airbnb
    )


async def call_amadeus_mcp_tool(params: dict) -> str:
    """Calls the Amadeus MCP server (async)."""
    origin_code = params.get("origin") # Expecting codes already
    dest_code = params.get("destination")
    if not origin_code or not dest_code:
        return "ERROR_MCP: Origin/Destination code missing for Amadeus."

    logger.info(f"Calling Amadeus tool for {origin_code} -> {dest_code}")
    tool_params = {
        "origin": origin_code,
        "destination": dest_code,
        "departure_date": params.get("departure_date"),
        "return_date": params.get("return_date"),
        "adults": params.get("adults", 1),
        "travel_class": params.get("travel_class"),
        "included_airline_codes": params.get("included_airline_codes"),
    }
    tool_params = {k: v for k, v in tool_params.items() if v is not None}
    return await call_mcp_tool_base(
        "flight_service.py", "get_flight_offers", tool_params, tool_call_timeout=240.0
    )


async def call_flights_tool(origin: str, destination: str, date: str = None) -> str:
    """Fallback: Calls the search MCP server's find_flights tool (async)."""
    if not origin or not destination:
        return "ERROR_MCP: Missing origin/destination for fallback flight search."

    logger.warning(f"Using fallback flight search (Brave Search) for {origin} to {destination}")
    tool_params = {"origin": origin, "destination": destination, "date": date}
    tool_params = {k: v for k, v in tool_params.items() if v is not None}
    return await call_mcp_tool_base("search.py", "find_flights", tool_params)


async def call_search_tool(query: str, category: str = "general") -> str:
    """Calls the search MCP server's search_travel_info tool (async)."""
    logger.info(f"Calling search tool for query: {query}, category: {category}")
    tool_params = {"query": query, "category": category}
    return await call_mcp_tool_base(
        "search.py", "search_travel_info", tool_params
    )


async def call_search_local_businesses(query: str, count: int = 5) -> str:
    """Calls the search MCP server's search_local_businesses tool (async)."""
    logger.info(f"Calling local search tool for query: {query}")
    tool_params = {"query": query, "count": count}
    return await call_mcp_tool_base(
        "search.py", "search_local_businesses", tool_params
    )


async def call_gmaps_search_places_tool(
    query: str,
    location_dict: Optional[Dict[str, float]] = None,
    radius: Optional[int] = None,
) -> str:
    """Calls the Google Maps MCP server's maps_search_places tool (async)."""
    if not query:
        return "ERROR_MCP: Query missing for Google Maps search."
    logger.info(
        f"Calling Google Maps search tool for query: {query}, "
        f"location: {location_dict}, radius: {radius}"
    )
    tool_params = {"query": query, "location": location_dict, "radius": radius}
    tool_params = {k: v for k, v in tool_params.items() if v is not None}
    return await call_mcp_tool_base(
        "gmaps_service.py", "maps_search_places", tool_params
    )


async def call_gmaps_directions_tool(
    origin: str, destination: str, mode: str = "driving"
) -> str:
    """Calls the Google Maps MCP server's maps_directions tool (async)."""
    if not origin or not destination:
        return "ERROR_MCP: Origin/Destination missing for directions."
    logger.info(
        f"Calling Google Maps directions tool from '{origin}' to '{destination}', "
        f"mode: {mode}"
    )
    tool_params = {"origin": origin, "destination": destination, "mode": mode}
    tool_params = {k: v for k, v in tool_params.items() if v is not None}
    return await call_mcp_tool_base(
        "gmaps_service.py", "maps_directions", tool_params
    )


async def call_gmaps_place_details_tool(place_id: str) -> str:
    """Calls the Google Maps MCP server's maps_place_details tool (async)."""
    if not place_id:
        return "ERROR_MCP: Place ID missing for place details."
    logger.info(f"Calling Google Maps place details tool for place_id: {place_id}")
    tool_params = {"place_id": place_id}
    return await call_mcp_tool_base(
        "gmaps_service.py", "maps_place_details", tool_params
    )


# --- Result Formatting/Enhancement ---

def format_gmaps_places_results(
    places_json_str: Union[str, None], query_context: str, max_results: int = 5
) -> str:
    """Formats Google Maps place search results (expects JSON string input)."""
    places_list = None
    if not places_json_str:
        return f"No Google Maps data received for '{query_context}'."
    if places_json_str.startswith("ERROR_MCP"):
        return f"Could not find results for '{query_context}' via Google Maps ({places_json_str})."

    try:
        parsed_data = json.loads(places_json_str)
        if isinstance(parsed_data, list):
            places_list = parsed_data
            logger.debug("Successfully parsed Google Maps JSON string to list.")
        else:
            logger.error(
                f"Parsed Google Maps JSON string is not a list for '{query_context}'. "
                f"Type: {type(parsed_data)}."
            )
            return (
                f"Error processing Google Maps results (unexpected format: "
                f"{type(parsed_data).__name__}) for '{query_context}'."
            )
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Google Maps result JSON string: {places_json_str[:500]}...")
        return f"Could not find results for '{query_context}' via Google Maps (Invalid JSON data format)."
    except Exception as e:
        logger.error(f"Unexpected error parsing Google Maps results: {e}", exc_info=True)
        return f"Error processing Google Maps results (parsing error) for '{query_context}'."

    # This check seems redundant given the try/except block, but keeping for safety
    if places_list is None:
        return f"Internal error formatting Google Maps results for '{query_context}'."

    if not places_list:
        return f"No specific results found for '{query_context}' via Google Maps."

    places_to_show = places_list[:max_results]
    formatted_output = [
        f"Found {len(places_to_show)} result(s) for '{query_context}' via Google Maps:"
    ]

    for i, place in enumerate(places_to_show, 1):
        if not isinstance(place, dict):
            logger.warning(f"Skipping non-dictionary item in places_list: {place}")
            continue

        name = place.get("name", "Unnamed Place")
        address = place.get("formatted_address", "Address not available")
        rating = place.get("rating")
        reviews = place.get("user_ratings_total", 0)
        place_id = place.get("place_id")
        loc_data = place.get("location") # Often nested under 'geometry'
        if not loc_data and 'geometry' in place and isinstance(place['geometry'], dict):
             loc_data = place['geometry'].get('location')

        lat = loc_data.get("lat") if isinstance(loc_data, dict) else None
        lng = loc_data.get("lng") if isinstance(loc_data, dict) else None

        formatted_output.append(f"\n{i}. **{name}**")
        formatted_output.append(f"   Address: {address}")

        if rating is not None and isinstance(rating, (int, float)) and rating > 0:
            formatted_output.append(f"   Rating: {rating}/5 ({reviews} reviews)")

        # Construct Google Maps link
        gmaps_link = None
        if place_id:
            gmaps_link = f"{GMAPS_PLACE_ID_URL_BASE}{place_id}"
        elif lat is not None and lng is not None:
            gmaps_link = f"{GMAPS_PLACE_URL_BASE}?q={lat},{lng}+{quote_plus(name)}"
        elif address:
            gmaps_link = f"{GMAPS_PLACE_URL_BASE}?q={quote_plus(address)}"

        if gmaps_link:
            formatted_output.append(f"   [View on Google Maps]({gmaps_link})")

    if len(places_list) > max_results:
        formatted_output.append(
            f"\n*Showing top {max_results} results. More might be available.*"
        )

    return "\n".join(formatted_output)


def enhance_results(
    result_text: str, result_type: str, location: Optional[str] = None
) -> str:
    """Add links and improve formatting of results (excluding custom weather)."""
    try:
        if not isinstance(result_text, str):
            result_text = str(result_text)

        enhanced_text = result_text

        # Basic cleanup/error pass
        if (not enhanced_text or enhanced_text.strip() == "" or
                enhanced_text.startswith("ERROR_") or
                enhanced_text.startswith("Error:") or
                enhanced_text.startswith("No results found") or
                enhanced_text.startswith("No flight offers found") or
                enhanced_text == "[]"):
            if enhanced_text == "[]" and (
                "Google Maps" in result_type or "Accommodations" in result_type
            ):
                return f"No specific results found via {result_type}."
            elif enhanced_text == "[]":
                return f"No results found via {result_type}."
            # Return error messages or empty indicators as is
            return enhanced_text

        # Remove control characters and normalize newlines
        enhanced_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', enhanced_text)
        enhanced_text = enhanced_text.replace("\\n", "\n")

        loc_str = location if location else "location" # Used for Airbnb search link

        # --- Skip Custom Weather Formatting ---
        # Custom weather tool results are assumed to be pre-formatted strings
        if (result_type.startswith("Weather Forecast (Custom)") or
                result_type.startswith("Future Weather (Custom)")):
            logger.debug(
                f"Skipping enhance_results for pre-formatted custom weather tool: {result_type}"
            )
            return result_text

        # --- Accommodation Formatting (Airbnb) ---
        if result_type == "Accommodations":
            try:
                data = json.loads(result_text) # Expects JSON string from the tool
                search_url = data.get(
                    "searchUrl",
                    f"https://www.airbnb.com/s/{quote_plus(loc_str)}/homes" # Use quote_plus for location in URL
                )
                search_results = data.get("searchResults", [])
                output_lines = [
                    f"Found {len(search_results)} accommodations near {loc_str} "
                    f"([View Full Search on Airbnb]({search_url}))" # Changed link text
                ]

                if not search_results:
                    output_lines.append("\nNo specific listings found matching criteria.")
                else:
                    # Show top 5 results
                    for i, listing_data in enumerate(search_results[:5], 1):
                        if not isinstance(listing_data, dict):
                            continue

                        # Safely extract nested data with defaults
                        listing_id = listing_data.get("id", "")
                        listing_url = listing_data.get(
                            "url",
                            f"https://www.airbnb.com/rooms/{listing_id}" if listing_id else ""
                        )

                        name = "Unnamed Listing"
                        # Simplify name extraction - adapt if structure differs
                        try:
                            # Attempt standard path
                            name_data = listing_data.get("demandStayListing", {}) \
                                            .get("description", {}) \
                                            .get("name", {}) \
                                            .get("localizedStringWithTranslationPreference", None)
                            # Fallback to simpler name path if available
                            if not name_data:
                                name_data = listing_data.get("name", None)

                            if name_data:
                                name = name_data

                        except AttributeError:
                             pass # Keep default name if path doesn't exist
                        except Exception as name_ex: # Catch other unexpected errors during name extraction
                             logger.warning(f"Error extracting Airbnb listing name: {name_ex}")
                             pass

                        price_text = "Price info unavailable"
                        try:
                             # Try primary line first
                             price_label_raw = listing_data.get("structuredDisplayPrice", {}) \
                                                .get("primaryLine", {}) \
                                                .get("accessibilityLabel", None)
                             # Fallback to secondary line if primary fails
                             if not price_label_raw:
                                 price_label_raw = listing_data.get("structuredDisplayPrice", {}) \
                                                     .get("secondaryLine", {}) \
                                                     .get("price", None) # Secondary line might just have 'price'
                             # Final fallback to just price if available
                             if not price_label_raw:
                                 price_label_raw = listing_data.get("structuredDisplayPrice", {}) \
                                                     .get("primaryLine", {}) \
                                                     .get("price", None)


                             if price_label_raw:
                                 price_text = format_price_label(str(price_label_raw))
                        except AttributeError:
                             pass # Keep default price text
                        except Exception as price_ex:
                            logger.warning(f"Error extracting Airbnb price: {price_ex}")
                            pass

                        rating_text = "No rating"
                        rating_label_raw = listing_data.get("avgRatingA11yLabel")
                        if rating_label_raw:
                            rating_text = format_rating_label(str(rating_label_raw))
                        # Fallback: sometimes rating is just a number
                        elif "avgRatingLocalized" in listing_data and listing_data["avgRatingLocalized"]:
                             try:
                                 rating_num = float(str(listing_data["avgRatingLocalized"]).split()[0]) # e.g. "4.8 stars" -> 4.8
                                 reviews_count = listing_data.get("reviewsCount", 0)
                                 rating_text = f"{rating_num:.1f}/5 ({reviews_count} reviews)"
                             except (ValueError, TypeError, IndexError):
                                 pass # Stick with "No rating" if conversion fails

                        room_type = ""
                        try:
                            # Try primary line first
                            room_type_data = listing_data.get("structuredContent", {}).get("primaryLine", [{}])[0].get("body", "")
                            # Fallback: sometimes under 'title'
                            if not room_type_data:
                                room_type_data = listing_data.get("title", "")
                            room_type = room_type_data.strip() if room_type_data else ""

                        except (AttributeError, IndexError, TypeError):
                             pass
                        except Exception as room_ex:
                            logger.warning(f"Error extracting Airbnb room type: {room_ex}")
                            pass


                        output_lines.append(f"\n**Option {i}: {name}**")
                        output_lines.append(f"   Price: {price_text}")
                        output_lines.append(f"   Rating: {rating_text}")
                        if room_type:
                            output_lines.append(f"   Type: {room_type}") # Changed label for clarity
                        # if listing_id: # ID less useful to end-user
                        #     output_lines.append(f"   Listing ID: {listing_id}")
                        if listing_url:
                            output_lines.append(f"   [View on Airbnb]({listing_url})")

                enhanced_text = "\n".join(output_lines).strip()

            except json.JSONDecodeError:
                logger.error(f"Failed to parse Airbnb JSON: {result_text[:500]}...")
                enhanced_text = "Error: Could not process accommodation results (invalid JSON)."
            except Exception as e:
                logger.error(f"Error formatting Airbnb: {e}", exc_info=True)
                enhanced_text = f"Error formatting accommodation results. Raw:\n{result_text[:1000]}..."

        # --- Flight Formatting ---
        elif result_type == "Flights":
            lines = enhanced_text.split('\n')
            enhanced_lines = []
            current_option_lines = []
            origin_code, dest_code = None, None
            departure_date_str = "" # Try to extract date for link

            # Try to extract origin/dest/date codes for Skyscanner link
            header_match = re.search(
                r'(?:from|for)\s+(\S{3})\s+to\s+(\S{3})(?:\s+on\s+([\d\-]+))?', enhanced_text, re.IGNORECASE
            )
            if header_match:
                g1 = header_match.group(1).upper()
                g2 = header_match.group(2).upper()
                g3 = header_match.group(3) # Date YYYY-MM-DD
                if len(g1) == 3 and g1.isalnum():
                    origin_code = g1
                if len(g2) == 3 and g2.isalnum():
                    dest_code = g2
                if g3:
                    try:
                        # Convert YYYY-MM-DD to YYMMDD for Skyscanner
                        departure_date_obj = datetime.strptime(g3, "%Y-%m-%d")
                        departure_date_str = departure_date_obj.strftime("%y%m%d")
                    except ValueError:
                        pass # Ignore if date format is wrong

            # General link if codes found but no options listed (e.g., just header returned)
            general_skyscanner_link = None
            if origin_code and dest_code and not any("Flight Option" in line or re.match(r'\s*\d+\.\s*Flight Option', line) for line in lines):
                 skyscanner_url = (
                     f"https://www.skyscanner.com/transport/flights/"
                     f"{origin_code.lower()}/{dest_code.lower()}/{departure_date_str if departure_date_str else ''}"
                 )
                 general_skyscanner_link = f"\n[Search this route on Skyscanner]({skyscanner_url})"


            for line in lines:
                is_new_option = (
                    "Flight Option" in line or
                    re.match(r'\s*\d+\.\s*Flight Option', line)
                )
                if is_new_option:
                    # Add Skyscanner link to the *previous* option if codes were found
                    if current_option_lines and origin_code and dest_code:
                        skyscanner_url = (
                            f"https://www.skyscanner.com/transport/flights/"
                            f"{origin_code.lower()}/{dest_code.lower()}/{departure_date_str if departure_date_str else ''}" # Add date if available
                        )
                        current_option_lines.append(
                            f"   [Search this route on Skyscanner]({skyscanner_url})"
                        )
                        enhanced_lines.extend(current_option_lines)
                        enhanced_lines.append("") # Add blank line between options
                    current_option_lines = [line.strip()] # Start new option
                elif current_option_lines:
                    # Add subsequent lines indented under the current option
                    current_option_lines.append(f"   {line.strip()}")
                else:
                    # Handle header lines before the first option
                    enhanced_lines.append(line)

            # Add Skyscanner link to the *last* option
            if current_option_lines and origin_code and dest_code:
                skyscanner_url = (
                    f"https://www.skyscanner.com/transport/flights/"
                    f"{origin_code.lower()}/{dest_code.lower()}/{departure_date_str if departure_date_str else ''}"
                )
                current_option_lines.append(
                    f"   [Search this route on Skyscanner]({skyscanner_url})"
                )
                enhanced_lines.extend(current_option_lines)

            # Add general link if generated
            if general_skyscanner_link:
                enhanced_lines.append(general_skyscanner_link)

            # Add general note if not already present
            if "**Note**: Prices and availability" not in enhanced_text:
                enhanced_lines.append("\n**Note**: Prices and availability are subject to change.")

            enhanced_text = '\n'.join(enhanced_lines).strip()

        # --- Driving Directions Formatting ---
        elif result_type == "Driving Directions":
            try:
                directions_data = json.loads(result_text) # Expects JSON string
                if not isinstance(directions_data, list) or not directions_data:
                    enhanced_text = "No driving routes found."
                else:
                    # Assume the first route is the primary one
                    route = directions_data[0]
                    summary = route.get("summary", "N/A")
                    legs = route.get("legs", [])

                    if legs:
                        first_leg = legs[0]
                        distance = first_leg.get("distance", {}).get("text", "N/A")
                        duration = first_leg.get("duration", {}).get("text", "N/A")
                        start_addr = first_leg.get("start_address", "N/A")
                        end_addr = first_leg.get("end_address", "N/A")

                        # Create Google Maps directions link
                        origin_param = quote_plus(start_addr)
                        dest_param = quote_plus(end_addr)
                        maps_link = (
                            f"{GMAPS_DIRECTIONS_URL_BASE}?saddr={origin_param}"
                            f"&daddr={dest_param}&dirflg=d" # 'd' for driving
                        )

                        output_lines = [
                            f"Driving Directions ({summary}):",
                            f"- From: {start_addr}",
                            f"- To: {end_addr}",
                            f"- Distance: {distance}",
                            f"- Duration: {duration} (typical traffic)", # Add note about traffic
                            f"- [View Route on Google Maps]({maps_link})",
                        ]
                        enhanced_text = "\n".join(output_lines)
                    else:
                        enhanced_text = "Route details not available."
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON for driving directions.")
                enhanced_text = "Error processing driving directions data."
            except Exception as e:
                logger.error(f"Error formatting directions: {e}", exc_info=True)
                enhanced_text = "Error formatting driving directions."

        # --- General URL Formatting (for Travel Info, Local Search etc.) ---
        # Avoid re-linking Gmaps results already formatted by format_gmaps_places_results
        elif "Google Maps Search" not in result_type:
            # Find bare URLs (not already in markdown links or code blocks)
            # Make pattern slightly more robust against trailing punctuation attached to URL
            bare_url_pattern = r'(?<![`\[\(])(https?://(?:www\.)?[^\s\'\"`<>\[\]\(\)\{\}]+[^\s\'\"`<>\[\]\(\)\{\}\.\,])' # Avoid ending in common punctuation
            enhanced_text = re.sub(bare_url_pattern, r'[Visit Website](\1)', enhanced_text)

            # Clean up patterns like "URL: [Visit Website](...)"
            enhanced_text = re.sub(
                r'\bURL:\s*\[Visit Website\]\((https?://[^\)]+)\)',
                r'[Visit Website](\1)',
                enhanced_text,
                flags=re.IGNORECASE
            )
            # Clean up links at the start of a line if preceded by only whitespace
            enhanced_text = re.sub(
                r'^\s*\[Visit Website\]\((https?://[^\)]+)\)',
                r'[Visit Website](\1)',
                enhanced_text,
                flags=re.MULTILINE
             )

            # Relabel Google Maps links if accidentally caught by general formatter
            enhanced_text = enhanced_text.replace(
                f"[Visit Website]({GMAPS_PLACE_ID_URL_BASE}",
                f"[View on Google Maps]({GMAPS_PLACE_ID_URL_BASE}"
            )
            enhanced_text = enhanced_text.replace(
                f"[Visit Website]({GMAPS_PLACE_URL_BASE}",
                f"[View on Google Maps]({GMAPS_PLACE_URL_BASE}"
            )
            enhanced_text = enhanced_text.replace(
                f"[Visit Website]({GMAPS_DIRECTIONS_URL_BASE}",
                f"[View Route on Google Maps]({GMAPS_DIRECTIONS_URL_BASE}"
            )

        # Final cleanup: remove excessive newlines and trailing spaces
        enhanced_text = re.sub(r'\n\s*\n', '\n\n', enhanced_text).strip()
        # Remove trailing whitespace from each line
        enhanced_text = "\n".join([line.rstrip() for line in enhanced_text.splitlines()])
        # Reduce multiple spaces within lines (be careful not to merge intended spaces)
        enhanced_text = "\n".join(
            [re.sub(r'(?<=\S)\s{2,}(?=\S)', ' ', line) for line in enhanced_text.splitlines()]
        )

        return enhanced_text

    except Exception as e:
        logger.warning(f"Error enhancing '{result_type}' results: {e}", exc_info=True)
        # Return original text if enhancement fails
        return result_text


# --- determine_needs function ---

async def determine_needs(query: str, parsed_entities: dict) -> Tuple[dict, Set[int]]:
    """Determines which tools are needed based on query and parsed entities."""
    needs = {
        "flights": False,
        "accommodation": False,
        "gmaps_search_queries": [],
        "local_search_query": None, # Currently seems unused? Keep structure.
        "general_search": False,
        "weather": False,
        "needs_driving_info": False,
        "needs_place_details": {}, # Store details needed {place_name: True}
    }
    handled_request_indices = set()
    query_lower = query.lower()
    # Ensure specific_requests are strings and lowercase
    specific_requests = [
        str(req).lower() for req in parsed_entities.get("specific_requests", []) if req is not None
    ]

    location_context = (
        parsed_entities.get("primary_destination") or
        parsed_entities.get("flight_destination")
    )
    weather_location = parsed_entities.get("weather_location_query") or location_context

    # --- Define Keywords and Maps ---
    accommodation_keywords = {
        "airbnb", "hotel", "stay", "accommodation", "motel", "inn", "hostel", "lodging"
    }
    # Map common request keywords to preferred Google Maps search terms
    place_keywords_map = {
        "restaurant": "restaurants", "restaurants": "restaurants", "food": "restaurants",
        "foods": "restaurants", "eat": "restaurants", "dining": "restaurants",
        "cuisine": "restaurants", "seafood": "seafood restaurants", "pizza": "pizza places",
        "sushi": "sushi restaurants", "steakhouse": "steakhouses",
        "vegetarian": "vegetarian restaurants", "vegan": "vegan restaurants",
        "brunch": "brunch spots", "dinner": "dinner restaurants", "lunch": "lunch spots",
        "bakery": "bakeries", "bar": "bars", "pub": "pubs",
        "attraction": "attractions", "attractions": "attractions", "museum": "museums",
        "museums": "museums", "activity": "activities", "activities": "activities",
        "things to do": "things to do", "landmark": "landmarks", "landmarks": "landmarks",
        "park": "parks", "parks": "parks", "visit": "places to visit", "see": "places to see",
        "tour": "tours", "tours": "tours", "gallery": "art galleries",
        "galleries": "art galleries", "science": "science attractions",
        "history": "historical sites", "art": "art attractions",
        "recommendations": "recommendations", "suggestions": "suggestions",
        "point of interest": "points of interest", "points of interest": "points of interest",
        "viewpoint": "scenic viewpoints", "scenic viewpoint": "scenic viewpoints",
        "viewpoints": "scenic viewpoints", "hike": "hiking trails", "hiking": "hiking trails",
        "easy hike": "easy hiking trails", "moderate hike": "moderate hiking trails",
        "coffee shop": "coffee shops", "coffee shops": "coffee shops", "cafe": "cafes",
        "cafes": "cafes", "coffee": "coffee shops", "tea": "tea shops",
        "bookstore": "bookstores", "book store": "bookstores", "bookstores": "bookstores",
        "independent bookstore": "independent bookstores", "shop": "shops", "shops": "shops",
        "store": "stores", "stores": "stores", "market": "markets", "markets": "markets",
        "quirky small town": "quirky small towns", "town": "towns", "small town": "small towns",
    }
    all_place_keywords = set(place_keywords_map.keys()) | set(place_keywords_map.values()) | \
                         {"place", "spot", "spots", "places"}
    # Keywords indicating needs handled by non-place-search tools
    non_place_keywords = {"driving time", "weather", "forecast", "flights", "hotels",
                          "airbnb", "permit", "aqi"} | accommodation_keywords
    route_context_keywords = {
        "along the route", "on the way", "to stop at", "driving from",
        "stop between", "on the drive", "road trip"
    }

    # 1. Check for Weather, Flights, Accommodation Needs from query/entities
    # Check for weather keywords OR a request for future weather date
    weather_keywords = {"weather", "forecast", "aqi", "future weather", "temperature", "air quality"}
    weather_query_keywords = {" weather", " forecast", " temperature", " air quality", " aqi"}
    if weather_location and (
            any(k in req for req in specific_requests for k in weather_keywords) or
            any(k in query_lower for k in weather_query_keywords) or
            parsed_entities.get("future_weather_date") # Explicit future date request
        ):
        needs["weather"] = True

    flight_query_keywords = {" flight", " fly ", " airline"}
    if (parsed_entities.get("flight_origin") or
            parsed_entities.get("flight_destination") or # Need dest even if origin missing
            parsed_entities.get("preferred_airline") or
            any(s in query_lower for s in flight_query_keywords)):
        needs["flights"] = True

    accom_query_keywords = {" airbnb", " accommodation", " stay ", " hotel"}
    if (any(k in req for req in specific_requests for k in accommodation_keywords) or
            any(s in query_lower for s in accom_query_keywords)):
        needs["accommodation"] = True

    # 2. Determine Google Maps Search Needs / Driving / Details from specific_requests
    place_queries_set = set()
    for i, req in enumerate(specific_requests):
        # Skip if handled by other core needs already determined
        if any(acc_kw in req for acc_kw in accommodation_keywords):
            handled_request_indices.add(i)
            continue
        if any(w_kw in req for w_kw in weather_keywords):
            handled_request_indices.add(i)
            continue
        # Skip general route context, handle driving time separately
        if any(route_kw in req for route_kw in route_context_keywords):
            # Don't mark as handled yet, might need general search
            continue
        if any(drive_kw in req for drive_kw in ["driving time", "drive time", "how long to drive"]):
            needs['needs_driving_info'] = True
            handled_request_indices.add(i)
            continue
        # Example specific exclusion (adjust as needed)
        if "permit" in req and "landing" in req: # e.g., Angels Landing permit info
             handled_request_indices.add(i)
             continue

        # --- Try to Generate Google Maps Place Queries ---
        gmaps_query_generated = False
        if location_context:
            # Clean up common request phrasing
            req_processed = req.replace(" recommendations", "").replace(" suggestions", "")
            req_processed = req_processed.replace("a place for ", "").replace("places for ", "").strip()
            req_processed = req_processed.replace("highly-rated ", "").replace("highly rated ", "") # Remove rating prefix for matching

            # Check for requests for details about a specific place
            detail_match = re.search(r'(?:opening hours for|details for|info on)\s+(.+)', req_processed)
            # Check for requests to visit/see a specific place
            visit_match = re.search(r'(?:visit|see|go to|check out|recommend|suggest)\s+(.+)', req_processed)
            # Check if the request itself looks like a named place
            # Simplified pattern: Multiple capitalized words or ends with common type
            named_place_pattern = r'([\w\s\'\-]+(?:[A-Z][\w\s\'\-]+){1,}|[\w\s\'\-]+(?: Art Museum| Museum| Center| Gallery| Park| Market| Square| Trail| Point| Falls| Lodge| Landing| Observatory))$'
            named_place_match = re.match(named_place_pattern, req_processed, re.IGNORECASE)


            potential_place_name = None
            is_detail_request = False
            if detail_match:
                potential_place_name = detail_match.group(1).strip()
                is_detail_request = True
            elif visit_match:
                potential_place_name = visit_match.group(1).strip()
            elif named_place_match:
                potential_place_name = named_place_match.group(0).strip() # Use full match

            if potential_place_name:
                # Heuristics to decide if it's a specific named entity vs. a category
                has_multiple_caps = bool(re.search(r'[A-Z][a-z]+\s+[A-Z]', potential_place_name)) or len(potential_place_name.split()) > 2
                ends_with_type = potential_place_name.split()[-1].lower() in [
                    "museum", "center", "gallery", "park", "market", "square",
                    "trail", "point", "falls", "lodge", "landing", "observatory", "restaurant", "cafe", "shop" # Add more common endings
                ]
                known_landmarks = [
                    "eiffel tower", "space needle", "freedom trail", "pike place market",
                    "north end", "art museum", "of science", "yosemite valley lodge",
                    "angels landing", "louvre" # Be careful this doesn't overlap with permit request
                ]
                is_likely_named_entity = (
                    has_multiple_caps or ends_with_type or
                    any(landmark in potential_place_name.lower() for landmark in known_landmarks)
                )
                # Check if it's *just* a generic type word
                is_generic_type = any(
                    kw == potential_place_name.lower() or f"{kw}s" == potential_place_name.lower()
                    for kw in all_place_keywords
                ) and len(potential_place_name.split()) == 1 # Only generic if single word

                # Special case: avoid matching things like "italian restaurants" as named entity
                if potential_place_name.lower().endswith(" restaurants") or \
                   potential_place_name.lower().endswith(" museums") or \
                   potential_place_name.lower().endswith(" shops"):
                    is_likely_named_entity = False


                if is_likely_named_entity and not is_generic_type:
                    gmaps_query = f"{potential_place_name}, {location_context}"
                    place_queries_set.add(gmaps_query)
                    gmaps_query_generated = True
                    if is_detail_request:
                        # Mark that details are needed for this specific place name
                        needs["needs_place_details"][potential_place_name] = True
                    logger.debug(f"Identified '{potential_place_name}' as named entity for Gmaps search.")


            # If not identified as a specific named entity, try matching patterns like "X near Y"
            if not gmaps_query_generated:
                context_match = re.search(
                    r'(?:good|any|some)?\s*' # Optional adjectives
                    r'([\w\s\'\-]+?)'             # Place type phrase (non-greedy)
                    r'\s+(?:near|in|at)\s+'       # Preposition
                    r'(.+)',                      # Context location within primary location
                    req_processed
                )
                if context_match:
                    place_type_phrase = context_match.group(1).strip()
                    near_context = context_match.group(2).strip()
                    mapped_type = None
                    best_match_keyword = ""
                    # Find the best (longest) keyword match in the phrase
                    for keyword, preferred_term in place_keywords_map.items():
                        # Use word boundaries for safer matching
                        if re.search(rf'\b{re.escape(keyword)}\b', place_type_phrase, re.IGNORECASE):
                            if len(keyword) > len(best_match_keyword):
                                mapped_type = preferred_term
                                best_match_keyword = keyword
                    # Direct match check (e.g., phrase is exactly "restaurants")
                    if place_type_phrase.lower() in place_keywords_map and len(place_type_phrase) > len(best_match_keyword):
                         mapped_type = place_keywords_map[place_type_phrase.lower()]

                    if mapped_type and mapped_type not in non_place_keywords:
                        attribute = ""
                        # Check for attributes in original request, not just the processed phrase
                        if "outdoor seating" in req: attribute = " with outdoor seating"
                        if "good reviews" in req: attribute += " with good reviews"

                        gmaps_query = f"{mapped_type} near {near_context}, {location_context}{attribute}"
                        place_queries_set.add(gmaps_query)
                        gmaps_query_generated = True
                        logger.debug(f"Identified '{mapped_type}' near '{near_context}' for Gmaps search.")


            # If still no query, check for simple "find [type]" requests
            if not gmaps_query_generated:
                 # Match "find/any/recommendations for [type]" or just "[type]"
                 # Ensure the type contains a keyword from our map
                 type_only_match = re.search(
                     r'(?:find|any|some|recommendations? for|suggestions? for|suggest|recommend|'
                     r'local|nearby|easy|moderate|quirky|unique|independent|good|best)\s+'
                     r'([\w\s\'\-]+)', # The type phrase
                     req_processed
                 )
                 simple_type_match = re.match(r'^([\w\s\'\-]+)$', req_processed) # Just the type

                 potential_type_phrase = None
                 if type_only_match:
                     potential_type_phrase = type_only_match.group(1).strip()
                     potential_type_phrase = potential_type_phrase.replace(" recommendations","").replace(" suggestions","")
                 elif simple_type_match:
                     potential_type_phrase = simple_type_match.group(1).strip()

                 if potential_type_phrase:
                     mapped_type = None
                     best_match_keyword = ""
                     # Find best keyword match
                     for keyword, preferred_term in place_keywords_map.items():
                         # Use word boundaries
                         if re.search(rf'\b{re.escape(keyword)}\b', potential_type_phrase, re.IGNORECASE):
                             if len(keyword) > len(best_match_keyword):
                                 mapped_type = preferred_term
                                 best_match_keyword = keyword
                     # Direct match check
                     if potential_type_phrase.lower() in place_keywords_map and len(potential_type_phrase) > len(best_match_keyword):
                          mapped_type = place_keywords_map[potential_type_phrase.lower()]

                     if mapped_type and mapped_type not in non_place_keywords:
                         gmaps_query = f"{mapped_type} in {location_context}"
                         place_queries_set.add(gmaps_query)
                         gmaps_query_generated = True
                         logger.debug(f"Identified simple type '{mapped_type}' for Gmaps search.")


            if gmaps_query_generated:
                handled_request_indices.add(i)

    needs['gmaps_search_queries'] = list(place_queries_set)

    # 3. Determine General Search Need
    is_planning_request = any(s in query_lower for s in ["plan", "itinerary", "trip", " guide"])
    # Check if any specific tool relevant to location/travel was triggered
    specific_place_or_travel_tool_triggered = (
        needs["flights"] or needs["accommodation"] or
        bool(needs["gmaps_search_queries"]) or needs['needs_driving_info'] or
        bool(needs['needs_place_details']) # Added place details check
    )

    # Check if there are unhandled specific requests (that aren't just keywords for other tools)
    unhandled_requests = False
    handled_keywords_in_search = non_place_keywords | all_place_keywords | {"hours", "details", "aqi", "near", "in"} # Added near/in

    unhandled_search_terms = []
    for i, req in enumerate(specific_requests):
        if i not in handled_request_indices:
            # Check if this unhandled request contains *only* keywords already covered or very generic terms
            # Remove covered keywords and see if anything substantial is left
            temp_req = req
            for kw in handled_keywords_in_search:
                 temp_req = temp_req.replace(kw, "")
            temp_req = re.sub(r'\s+', ' ', temp_req).strip() # Clean up remaining whitespace

            # If something meaningful remains, consider it unhandled
            if len(temp_req) > 3: # Arbitrary threshold for meaningful leftover text
                 unhandled_requests = True
                 unhandled_search_terms.append(req) # Keep original request for search query
                 logger.debug(f"Request deemed unhandled by specific tools: '{req}' (remaining: '{temp_req}')")
                 # Don't break, collect all unhandled terms for the search query


    # Trigger general search if:
    # - It's an explicit planning request OR
    # - A location was given but no specific tools were triggered OR
    # - There are unhandled requests
    if is_planning_request or \
       (location_context and not specific_place_or_travel_tool_triggered and not needs["weather"]) or \
       unhandled_requests: # Simplified: Trigger if location given and no specific tools (excl weather) triggered
        # Add a condition to AVOID general search if *only* weather was requested
        # (and it wasn't part of a larger planning request or unhandled items)
        only_weather_requested = (
            needs["weather"] and
            not needs["flights"] and not needs["accommodation"] and
            not needs["gmaps_search_queries"] and not needs['needs_driving_info'] and
            not needs['needs_place_details'] and not is_planning_request and
            not unhandled_requests
        )
        if not only_weather_requested:
             needs["general_search"] = True
             needs["unhandled_search_terms"] = unhandled_search_terms # Pass terms for query construction


    logger.debug(f"DEBUG (determine_needs): Final calculated needs: {needs}")
    logger.debug(f"DEBUG (determine_needs): Handled request indices: {handled_request_indices}")
    return needs, handled_request_indices


# --- ASYNC Query Processing Logic (Using Custom Weather Server) ---

async def process_query_async(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Processes user query async: extracts entities, determines needs,
    calls tools (incl. custom weather), synthesizes response, ensures links.
    """
    logger.info("--- Starting process_query_async ---")
    query_lower = query.lower()

    # 1. Extract Entities
    parsed_entities = await extract_entities_with_llm(query)

    primary_location_name = parsed_entities.get("primary_destination")
    weather_location = parsed_entities.get("weather_location_query") or primary_location_name
    flight_origin_code = parsed_entities.get("flight_origin")
    flight_destination_code = parsed_entities.get("flight_destination")
    location_context = primary_location_name or flight_destination_code
    future_weather_date = parsed_entities.get("future_weather_date")

    # Basic validation
    if not location_context and not weather_location:
        st.warning("Could not determine a destination or weather location.")
        return "Sorry, I couldn't figure out the destination. Could you please specify?", []
    if not location_context:
        location_context = weather_location # Fallback

    # Determine driving origin (can be improved)
    driving_origin = "Current Location" # Default assumes Gmaps API can handle this
    match_driving = re.search(r'driving from\s+(.+?)\s+(?:to|towards)\s+(.+)', query_lower, re.IGNORECASE)
    if match_driving:
        extracted_origin = match_driving.group(1).strip()
        # Override default only if extracted origin is specific
        if extracted_origin.lower() not in ["current location", "here", "my location"]:
            driving_origin = extracted_origin
        # Potentially override destination context based on 'to' phrase
        extracted_dest = match_driving.group(2).strip()
        if extracted_dest and (not location_context or extracted_dest.lower() != location_context.lower()):
             # Prefer longer/more specific destination from query if different
             if len(extracted_dest) > len(location_context or ""):
                 logger.info(f"Overriding location context based on 'driving to': '{extracted_dest}'")
                 location_context = extracted_dest


    # 2. Determine Tool Needs
    needs, handled_request_indices = await determine_needs(query, parsed_entities)
    logger.info(f"--- Determined Needs ---: {needs}")
    logger.info(f"--- Handled Request Indices ---: {handled_request_indices}")

    # 3. Schedule Tool Coroutines
    logger.info("--- Scheduling Tool Calls ---")
    tasks_to_run_coroutines = []
    task_map = {}
    tool_order = [ # Preferred order
        "Weather", "Flights", "Accommodations", "Driving Directions",
        "Google Maps Search", "Local Search", "Travel Info"
    ]
    flight_params_stored = {}

    # --- Weather Scheduling ---
    weather_scheduled = False
    if needs.get("weather") and weather_location:
        weather_opts = st.session_state.search_options["weather"]
        if future_weather_date: # Check for specific future date first
            logger.debug(f"   Scheduling Future Weather (Custom): loc='{weather_location}', date='{future_weather_date}'")
            tasks_to_run_coroutines.append(call_custom_future_weather_tool(location=weather_location, date=future_weather_date))
            task_map[len(tasks_to_run_coroutines) - 1] = "Future Weather (Custom)"
            weather_scheduled = True
        else: # Default to forecast
            forecast_params = {"location": weather_location, **weather_opts}
            logger.debug(f"   Scheduling Weather Forecast (Custom): {forecast_params}")
            tasks_to_run_coroutines.append(call_custom_weather_forecast_tool(**forecast_params))
            task_map[len(tasks_to_run_coroutines) - 1] = "Weather Forecast (Custom)"
            weather_scheduled = True
    if "Weather" in tool_order and not weather_scheduled: tool_order.remove("Weather")

    # --- Flights Scheduling (Amadeus preferred, fallback handled later) ---
    flights_scheduled_primary = False
    if needs.get("flights"):
        if flight_origin_code and flight_destination_code:
            flight_params = {
                "origin": flight_origin_code, "destination": flight_destination_code,
                "adults": parsed_entities.get("num_adults", 1),
                "departure_date": parsed_entities.get("start_date"),
                "return_date": parsed_entities.get("end_date"),
                "travel_class": st.session_state.search_options["flight"].get("travel_class", "ECONOMY"),
            }
            preferred_airline_name = parsed_entities.get("preferred_airline")
            airline_code = map_airline_name_to_code(preferred_airline_name) if preferred_airline_name else None
            if airline_code: flight_params["included_airline_codes"] = airline_code
            flight_params = {k: v for k, v in flight_params.items() if v is not None}

            flight_params_stored = flight_params.copy() # Store for potential fallback
            logger.debug(f"   Scheduling Flights (Amadeus): {flight_params}")
            tasks_to_run_coroutines.append(call_amadeus_mcp_tool(flight_params))
            task_map[len(tasks_to_run_coroutines) - 1] = "Flights"
            flights_scheduled_primary = True
        else:
             logger.warning("Cannot schedule primary flight search: Missing origin/destination codes.")
             # Fallback will be attempted later if primary fails *or* wasn't scheduled but needed

    if "Flights" in tool_order and not needs.get("flights"): tool_order.remove("Flights") # Remove if not needed at all

    # --- Accommodation Scheduling ---
    accom_scheduled = False
    if needs.get("accommodation") and location_context:
         accom_location = location_context # Start with main context
         # Check specific requests for "near X" refinement
         for req in parsed_entities.get("specific_requests", []):
             req_str = str(req).lower()
             match = re.search(r'(?:airbnb|accommodation|stay|hotel)\s+near\s+([\w\s,]+)', req_str)
             if match:
                 specific_near = match.group(1).strip()
                 accom_location = f"{specific_near}, {location_context}" # Refine location
                 logger.debug(f"Using specific accommodation location: {accom_location}")
                 break

         # Check budget from query
         budget_per_night = None
         budget_match_night = re.search( r'(?:budget|under|less than)\s+\$?(\d+)\s*(?:per night|a night)', query_lower)
         if budget_match_night: budget_per_night = int(budget_match_night.group(1))

         accom_params = {
             "location": accom_location,
             "adults": parsed_entities.get("num_adults", 1),
             "children": parsed_entities.get("num_children", 0),
             "infants": parsed_entities.get("num_infants", 0),
             "pets": int(parsed_entities.get("num_pets", 0)), # Ensure integer
             "min_price": st.session_state.search_options["airbnb"].get("min_price"),
             "max_price": budget_per_night or st.session_state.search_options["airbnb"].get("max_price"), # Prioritize query budget
             "check_in": parsed_entities.get("start_date"),
             "check_out": parsed_entities.get("end_date"),
         }
         accom_params = {k: v for k, v in accom_params.items() if v is not None}
         if "pets" in accom_params and accom_params["pets"] < 0: accom_params["pets"] = 0 # Sanity check

         logger.debug(f"   Scheduling Accommodation (Airbnb): {accom_params}")
         tasks_to_run_coroutines.append(call_airbnb_tool_with_params(accom_params))
         task_map[len(tasks_to_run_coroutines) - 1] = "Accommodations"
         accom_scheduled = True
    if "Accommodations" in tool_order and not accom_scheduled: tool_order.remove("Accommodations")

    # --- Driving Directions Scheduling ---
    driving_scheduled = False
    if needs.get("needs_driving_info") and location_context:
         driving_dest = location_context
         logger.debug(f"   Scheduling Driving Directions: From '{driving_origin}' To '{driving_dest}'")
         tasks_to_run_coroutines.append(call_gmaps_directions_tool(origin=driving_origin, destination=driving_dest))
         task_map[len(tasks_to_run_coroutines) - 1] = "Driving Directions"
         driving_scheduled = True
    if "Driving Directions" in tool_order and not driving_scheduled: tool_order.remove("Driving Directions")

    # --- Google Maps Search Scheduling ---
    gmaps_scheduled = False
    gmaps_queries = needs.get("gmaps_search_queries", [])
    if gmaps_queries:
        gmaps_tool_label_base = "Google Maps Search"
        place_keywords_map_rev = {v: k for k, v in place_keywords_map.items()} # For finding original keyword from preferred term
        for gmaps_query in gmaps_queries:
            search_params_gmaps = {"query": gmaps_query}
            # Determine label for context
            query_parts = gmaps_query.split(',')[0].split(' near ')[0].split(' in ')[0].strip()
            type_label = query_parts
            best_match_len = 0
            for key, val in place_keywords_map.items(): # Check preferred terms first
                if val in query_parts.lower() and len(val) > best_match_len:
                    type_label = val; best_match_len = len(val)
            if best_match_len == 0: # Check original keywords if no preferred term matched well
                for key, val in place_keywords_map.items():
                    if key in query_parts.lower() and len(key) > best_match_len:
                         type_label = val; best_match_len = len(key) # Use preferred term as label

            tool_label_gmaps = f"{gmaps_tool_label_base} ({type_label})"
            logger.debug(f"   Scheduling {tool_label_gmaps}: {search_params_gmaps}")
            tasks_to_run_coroutines.append(call_gmaps_search_places_tool(**search_params_gmaps))
            task_map[len(tasks_to_run_coroutines) - 1] = tool_label_gmaps
            gmaps_scheduled = True
    if "Google Maps Search" in tool_order and not gmaps_scheduled: tool_order.remove("Google Maps Search")

    # --- General Search Scheduling ---
    general_search_scheduled = False
    if needs.get("general_search") and location_context:
         search_query = f"travel planning tips activities itinerary for {location_context}"
         unhandled_terms = needs.get("unhandled_search_terms", [])
         if unhandled_terms:
             search_query += f" including information about: {' and '.join(unhandled_terms)}"
         # Add driving time explicitly if needed but not scheduled via Gmaps
         elif needs.get("needs_driving_info") and not driving_scheduled:
             search_query += " including driving time"

         search_params = {"query": search_query, "category": "general"}
         logger.debug(f"   Scheduling General Search: {search_params}")
         tasks_to_run_coroutines.append(call_search_tool(**search_params))
         task_map[len(tasks_to_run_coroutines) - 1] = "Travel Info"
         general_search_scheduled = True
    if "Travel Info" in tool_order and not general_search_scheduled: tool_order.remove("Travel Info")
    # Remove Local Search if unused, seems like general search covers its purpose now
    if "Local Search" in tool_order: tool_order.remove("Local Search")


    # --- Determine Final Execution Order ---
    task_indices = list(task_map.keys())
    tool_order_base = [t.split(" (")[0] for t in tool_order]
    def get_sort_key(index):
        tool_label = task_map[index]
        base_tool_name = tool_label.split(" (")[0]
        try: return tool_order_base.index(base_tool_name)
        except ValueError: return float('inf')
    task_indices.sort(key=get_sort_key)
    tasks_to_run_ordered = [tasks_to_run_coroutines[i] for i in task_indices]
    task_map_ordered = {new_idx: task_map[old_idx] for new_idx, old_idx in enumerate(task_indices)}
    tasks_to_run_coroutines = tasks_to_run_ordered
    task_map = task_map_ordered
    final_tool_order = [task_map[i] for i in range(len(tasks_to_run_coroutines))]
    logger.info(f"Final scheduled tool order for execution: {final_tool_order}")


    # 4. Execute Tool Coroutines
    if not tasks_to_run_coroutines:
        logger.info("No tools scheduled for execution.")
        results_list = []
    else:
        logger.info(f"Executing {len(tasks_to_run_coroutines)} tasks...")
        results_list = await asyncio.gather(*tasks_to_run_coroutines, return_exceptions=True)
        logger.info("--- Task execution finished ---")


    # 5. Process Results & Handle Fallbacks
    logger.info("--- Processing Tool Results ---")
    raw_results = {}
    for i, result in enumerate(results_list):
        tool_name = task_map.get(i, f"Unknown Task {i}")
        if isinstance(result, Exception):
            raw_results[tool_name] = f"ERROR_TOOL: Tool call failed for {tool_name}: {result}"
            logger.error(f"Exception for tool '{tool_name}': {result}", exc_info=True)
            st.warning(f"Tool '{tool_name}' failed: {result}") # Show warning in UI too
        else:
            raw_results[tool_name] = result if result is not None else ""
            # Basic check for empty results that aren't errors
            if not raw_results[tool_name] or raw_results[tool_name] == "[]":
                 logger.info(f"   Result OK (Empty): {tool_name}")
            else:
                 logger.info(f"   Result OK: {tool_name} (Length: {len(raw_results[tool_name])})")


    # --- Flight Fallback Logic ---
    flight_result_str = raw_results.get("Flights", "")
    # Conditions for fallback:
    # - Flights were needed.
    # - EITHER Primary Amadeus wasn't scheduled (missing codes) OR Amadeus ran but failed/returned empty.
    # - We have string representations of origin/destination for fallback tool.
    amadeus_failed_or_empty = flights_scheduled_primary and \
                              (not flight_result_str or flight_result_str.startswith("ERROR_") or \
                               flight_result_str == "[]" or "No flight offers found" in flight_result_str)
    amadeus_not_scheduled = needs.get("flights") and not flights_scheduled_primary

    if amadeus_failed_or_empty or amadeus_not_scheduled:
        logger.warning(f"Flight condition met for fallback check (Amadeus Failed/Empty: {amadeus_failed_or_empty}, Amadeus Not Scheduled: {amadeus_not_scheduled}).")
        # Get origin/dest strings (prefer parsed inputs, fallback to codes/context)
        f_origin_str = parsed_entities.get("flight_origin") or flight_origin_code
        f_dest_str = parsed_entities.get("flight_destination") or flight_destination_code or location_context

        if f_origin_str and f_dest_str:
            logger.warning(f"Attempting fallback flight search for {f_origin_str} -> {f_dest_str}")
            try:
                fallback_flight_result = await call_flights_tool(
                    origin=f_origin_str,
                    destination=f_dest_str,
                    date=parsed_entities.get("start_date") # Pass date if available
                )
                # Update the result, adding the fallback note
                raw_results["Flights"] = f"[Using Fallback Search Results]\n{fallback_flight_result}"
                logger.info(f"   Fallback Flight Result OK: {len(raw_results.get('Flights', ''))} chars")
                # Ensure 'Flights' is in the final order if it wasn't before (e.g., Amadeus not scheduled)
                if "Flights" not in final_tool_order: final_tool_order.append("Flights") # Add to end if missing
            except Exception as fallback_err:
                 logger.error(f"Fallback flight search failed: {fallback_err}", exc_info=True)
                 raw_results["Flights"] = f"ERROR_TOOL: Fallback flight search also failed: {fallback_err}"
        else:
            logger.error("Cannot run flight fallback: Missing origin or destination strings.")
            raw_results["Flights"] = "ERROR_TOOL: Flight search failed, missing origin/destination for fallback."


    # 6. Format and Order Results for LLM Context
    ordered_results: List[Tuple[str, str]] = []
    gmaps_max_results_setting = st.session_state.search_options["gmaps"]["max_results"]
    logger.info(f"--- Final Ordered Results for LLM Context (Pre-Enhance/Format): {final_tool_order} ---")

    # Build context string and simultaneously extract ALL links from formatted results
    context_message_parts = ["CONTEXT:\nHere is the information gathered from the tools:\n"]
    all_context_links = {} # Stores {URL: Full Markdown Link}

    # Re-sort the final_tool_order based on the tool_order preference list again
    # This ensures results appear in the context in the desired logical flow
    try:
        # Define tool_order_base here based on your preferred order list 'tool_order'
        tool_order_base = [t.split(" (")[0] for t in tool_order] # Assuming tool_order is defined earlier
        # Sort based on the index of the base tool name in the preferred list
        final_tool_order.sort(key=lambda name: tool_order_base.index(name.split(" (")[0]) if name.split(" (")[0] in tool_order_base else float('inf'))
        logger.info(f"Re-sorted final_tool_order for context: {final_tool_order}")
    except Exception as sort_err:
        logger.error(f"Error re-sorting final_tool_order: {sort_err}. Proceeding with unsorted order.", exc_info=True) # Sorts in-place based on original preference

    for name in final_tool_order:
        if name not in raw_results or not raw_results[name]: # Skip tools not run or empty
            continue

        logger.debug(f"Processing result for LLM Context: {name}")
        result_data = raw_results[name]
        formatted_result = "" # To store the result after enhancement/formatting

        # Apply formatting (calls enhance_results, format_gmaps_places_results)
        location_for_enhance = location_context or weather_location or "Unknown Location"
        is_fallback = isinstance(result_data, str) and result_data.startswith("[Using Fallback Search Results]")
        result_text_content = result_data.replace("[Using Fallback Search Results]\n", "") if is_fallback else result_data
        result_type = name.split(" (")[0] # Base type like "Flights", "Google Maps Search"

        try:
            if name.startswith("Google Maps Search"):
                query_ctx_match = re.search(r'\((.*?)\)$', name)
                query_ctx = query_ctx_match.group(1) if query_ctx_match else name
                formatted_result = format_gmaps_places_results(
                    result_text_content, query_ctx, gmaps_max_results_setting
                )
            elif name.startswith("Weather") or name == "Driving Directions" or name == "Accommodations" or name == "Flights" or name == "Travel Info":
                 # These types have specific handling or general URL linking in enhance_results
                 formatted_result = enhance_results(result_text_content, result_type, location_for_enhance)
            else: # Fallback for any unexpected tool names - treat as general text
                 formatted_result = enhance_results(result_text_content, "General", location_for_enhance)

        except Exception as format_err:
             logger.error(f"Error formatting result for tool '{name}': {format_err}", exc_info=True)
             formatted_result = f"[Error formatting result for {name}]\nRaw:\n{result_data[:500]}..." # Pass error info to LLM

        # Add fallback note if necessary (ensure it's not duplicated)
        final_result_for_llm = formatted_result
        if is_fallback and not formatted_result.startswith("[Using Fallback Search Results]"):
             final_result_for_llm = f"[Using Fallback Search Results]\n{formatted_result}"

        ordered_results.append((name, final_result_for_llm)) # Store for potential display/debug

        # --- Prepare context piece for LLM ---
        result_str_llm = str(final_result_for_llm)
        is_error_llm = result_str_llm.startswith("ERROR_") or "[Tool Error/No Results]" in result_str_llm or "Error:" in result_str_llm or "[Error formatting result" in result_str_llm
        tool_note_llm = ""
        result_preview_llm = result_str_llm

        if is_fallback: tool_note_llm += "[Using Fallback Search Results] "
        if is_error_llm:
            tool_note_llm += "[Tool Error/No Results] "
            result_preview_llm = f"({name} Error - Data not usable)" # Simplify error for LLM

        MAX_RESULT_LEN_CTX = 4000 # Context limit per tool result
        if len(result_preview_llm) > MAX_RESULT_LEN_CTX:
            result_preview_llm = result_preview_llm[:MAX_RESULT_LEN_CTX] + "... (truncated)"
            tool_note_llm += "[Result Truncated] "

        context_message_parts.append(
            f"--- Start of {name} Data ---\n"
            f"{tool_note_llm}{result_preview_llm}\n" # Use the potentially truncated/error-noted version
            f"--- End {name} Data ---"
        )

        # --- Extract links from the fully formatted result ---
        # Use the helper function defined earlier (extract_markdown_links)
        # Extract from final_result_for_llm BEFORE truncation for context
        if not is_error_llm: # Don't try to extract links from error messages
            found_links = extract_markdown_links(final_result_for_llm)
            for url, md_link in found_links.items():
                 if url not in all_context_links:
                     all_context_links[url] = md_link
        # --- End link extraction ---


    # 7. Synthesize Final Response using LLM
    llm = get_llm()
    if not llm:
        st.error("LLM could not be initialized.")
        return "Sorry, I encountered an internal error.", ordered_results # Return results for debug

    system_prompt_text = create_system_prompt()
    messages = [SystemMessage(content=system_prompt_text)]

    # Add limited history
    history_messages = st.session_state.messages[-4:-1]
    for msg in history_messages:
        role = msg["role"]
        content = str(msg.get("content", ""))
        if role == "user": messages.append(HumanMessage(content=content))
        elif role == "assistant": messages.append(AIMessage(content=content))

    # Add current query
    messages.append(HumanMessage(content=query))

    # Assemble and Add Context
    if not any(part for part in context_message_parts[1:]): # Check if any tool data was added
         context_message_parts.append("No specific tool results were available for this query.")
    context_message = "\n\n".join(context_message_parts)
    logger.debug(f"--- DEBUG: LLM Synthesis Context Message Start ---\n{context_message[:1500]}...")
    logger.debug(f"--- DEBUG: LLM Synthesis Context Message End ---\n...{context_message[-1500:]}")
    logger.debug(f"--- DEBUG: Total Context Length Approx: {len(context_message)} ---")
    messages.append(HumanMessage(content=context_message))

    # --- LLM Call and Link Check/Append ---
    final_answer = ""
    key_links_to_ensure_md = {} # To store {URL: Markdown} for check

    try:
        logger.info("Invoking LLM for final synthesis...")
        response = await llm.ainvoke(messages)
        final_answer = response.content
        logger.info("--- LLM Synthesis Complete ---")

        # --- REFINED: Post-LLM Link Check and Append ---
        logger.debug("Checking LLM response for link inclusion (strict markdown check)...")
        links_to_append = []
        final_answer_str_for_check = str(final_answer)

        # Identify key links from the gathered context links
        try:
            skyscanner_md = next((md for url, md in all_context_links.items() if "skyscanner.com/transport/flights" in url), None)
            if skyscanner_md: key_links_to_ensure_md[next(url for url, md in all_context_links.items() if md == skyscanner_md)] = skyscanner_md
            airbnb_search_md = next((md for url, md in all_context_links.items() if "airbnb.com/s/" in url and "/homes" in url), None)
            if airbnb_search_md: key_links_to_ensure_md[next(url for url, md in all_context_links.items() if md == airbnb_search_md)] = airbnb_search_md
            airbnb_listing_md = next((md for url, md in all_context_links.items() if "airbnb.com/rooms/" in url), None)
            if airbnb_listing_md: key_links_to_ensure_md[next(url for url, md in all_context_links.items() if md == airbnb_listing_md)] = airbnb_listing_md
            gmaps_direction_md = next((md for url, md in all_context_links.items() if GMAPS_DIRECTIONS_URL_BASE in url), None)
            if gmaps_direction_md: key_links_to_ensure_md[next(url for url, md in all_context_links.items() if md == gmaps_direction_md)] = gmaps_direction_md
            gmaps_place_md = next((md for url, md in all_context_links.items() if GMAPS_PLACE_ID_URL_BASE in url or GMAPS_PLACE_URL_BASE in url), None)
            if gmaps_place_md: key_links_to_ensure_md[next(url for url, md in all_context_links.items() if md == gmaps_place_md)] = gmaps_place_md
        except StopIteration: logger.warning("Could not find one or more key links during iteration.")
        except Exception as link_find_ex: logger.error(f"Error finding key links: {link_find_ex}", exc_info=True)

        logger.debug(f"Key Markdown links identified for checking: {list(key_links_to_ensure_md.values())}")

        # Check if the FULL MARKDOWN string is missing
        for url, markdown_to_check in key_links_to_ensure_md.items():
            if markdown_to_check not in final_answer_str_for_check:
                 logger.info(f"LLM response missing key MARKDOWN link: {markdown_to_check}. Will append.")
                 if markdown_to_check not in links_to_append:
                     links_to_append.append(markdown_to_check)

        # Append if needed
        if links_to_append:
            if "**Relevant Links:**" not in final_answer: final_answer += "\n\n**Relevant Links:**\n"
            else: final_answer += "\n"
            final_answer += "\n".join(f"- {link}" for link in links_to_append)
            logger.info(f"Appended {len(links_to_append)} missing markdown links.")
        else:
            logger.debug("No key markdown links were missing from the LLM response.")
        # --- End REFINED Link Check ---

        return final_answer, ordered_results # Return potentially modified answer

    except Exception as synthesis_error:
        logger.error(f"LLM synthesis failed: {synthesis_error}", exc_info=True)
        st.error(f"Error during final response generation: {synthesis_error}")
        # --- Error Handling ---
        error_response_text = ("I gathered some information but encountered an issue generating the final response. "
                               "Please check the logs or try again.")
        if not final_answer: final_answer = error_response_text # Use error text if LLM failed before assigning

        # Attempt to append links even on synthesis failure
        if all_context_links and key_links_to_ensure_md:
             links_to_append_on_error = []
             final_answer_str_on_error = str(final_answer)
             for url, markdown_to_check in key_links_to_ensure_md.items():
                  if markdown_to_check not in final_answer_str_on_error:
                       logger.info(f"LLM synthesis failed. Appending missing markdown link: {markdown_to_check}")
                       if markdown_to_check not in links_to_append_on_error:
                            links_to_append_on_error.append(markdown_to_check)
             if links_to_append_on_error:
                  links_section_header = "\n\n**Relevant Links (from gathered data):**\n"
                  if "**Relevant Links" not in final_answer: final_answer += links_section_header
                  else: final_answer += "\n"
                  final_answer += "\n".join(f"- {link}" for link in links_to_append_on_error)

        return final_answer, ordered_results

# --- Streamlit UI Logic ---

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Ensure content is treated as a string
        content_str = str(message.get("content", ""))
        # Allow markdown formatting, including links
        st.markdown(content_str, unsafe_allow_html=True)

# Get user input
if prompt := st.chat_input("What are your travel plans? (e.g., 'Plan a 3-day trip to Paris')"):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query and display assistant response
    with st.spinner("Planning your trip using available tools... This may take a minute..."):
        try:
            # Run the async processing function
            # tool_results_for_display is kept in case needed later, but not displayed by default
            final_answer, tool_results_for_display = asyncio.run(process_query_async(prompt))

            # Add assistant response to state and display it
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                final_answer_str = str(final_answer)
                # Ensure links are displayed correctly in markdown
                st.markdown(final_answer_str, unsafe_allow_html=True)

            # --- REMOVED RAW DATA DISPLAY ---
            # The expander for "View Raw Data Sources" has been removed as requested.
            # The `tool_results_for_display` variable still contains the data if needed for debugging
            # or future features, but it's not shown to the user in the chat interface.
            # ---

        except Exception as final_error:
            # Catch unexpected errors during the async processing
            error_msg = f"An critical error occurred during processing: {str(final_error)}"
            st.error(error_msg)
            detailed_error = traceback.format_exc()
            logger.critical("Critical error in main processing loop", exc_info=True)
            st.code(detailed_error) # Show traceback in UI for debugging

            # Add an error message to the chat history
            error_response = (
                "I encountered a critical issue while processing your request. "
                "Please check the error details above or try simplifying your query."
            )
            st.session_state.messages.append({"role": "assistant", "content": error_response})
            # Rerun to display the error message added to the history
            st.rerun()