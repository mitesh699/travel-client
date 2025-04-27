# travel-client/src/flight_service.py (Regenerated with airline filtering and previous changes)
from mcp.server.fastmcp import FastMCP
from amadeus import Client, ResponseError
import os
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import json
# Add this import if not already present
from typing import List, Dict, Any
import sys # Added for sys.exit

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Log to console
        logging.FileHandler('amadeus_mcp.log', mode='a') # Append mode
    ]
)
logger = logging.getLogger('amadeus_mcp')

# Initialize FastMCP server
mcp = FastMCP("amadeus")
logger.info("Amadeus MCP server initialized with name: amadeus")

# Load environment variables
load_dotenv(override=True)

# Dictionary of airport codes (Keep this updated or consider a more robust lookup)
AIRPORT_CODES = {
    "san francisco": "SFO", "sfo": "SFO",
    "seattle": "SEA", "seatac": "SEA", "sea": "SEA",
    "new york": "JFK", "nyc": "JFK", "jfk": "JFK", "new york city": "JFK",
    "laguardia": "LGA", "newark": "EWR",
    "chicago": "ORD", "ord": "ORD", "o'hare": "ORD", "midway": "MDW",
    "boston": "BOS", "bos": "BOS", "logan": "BOS",
    "los angeles": "LAX", "la": "LAX", "lax": "LAX",
    "miami": "MIA", "mia": "MIA",
    "las vegas": "LAS", "las": "LAS",
    "denver": "DEN", "den": "DEN",
    "atlanta": "ATL", "atl": "ATL",
    "dallas": "DFW", "dfw": "DFW", "dallas fort worth": "DFW",
    "washington": "IAD", "dc": "IAD", "iad": "IAD", "dulles": "IAD", "reagan": "DCA",
    "london": "LHR", "lhr": "LHR", "heathrow": "LHR", "gatwick": "LGW", "lon": "LON",
    "paris": "CDG", "cdg": "CDG", "charles de gaulle": "CDG", "orly": "ORY", "par": "PAR",
    "austin": "AUS", "aus": "AUS",
    # Add more as needed
}

# Initialize Amadeus client
amadeus = None
try:
    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Missing Amadeus credentials. Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in .env")
    amadeus = Client(
        client_id=client_id,
        client_secret=client_secret,
    )
    logger.info("Amadeus client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Amadeus client: {str(e)}", exc_info=True)
    amadeus = None

def get_airport_code(location):
    """Convert a location name to an airport code"""
    if not location: return None
    if isinstance(location, str) and len(location) == 3 and location.isalpha(): return location.upper()
    location_lower = location.lower().strip()
    if location_lower in AIRPORT_CODES: return AIRPORT_CODES[location_lower]
    if location.upper() in AIRPORT_CODES.values(): return location.upper()
    logger.warning(f"Could not find airport code for '{location}'")
    return None

def format_duration(duration_str):
    """Formats ISO 8601 duration like PT2H42M into 2h 42m"""
    if not duration_str or not duration_str.startswith("PT"): return duration_str
    try:
        duration_str = duration_str[2:]; hours = 0; minutes = 0
        h_index = duration_str.find('H')
        if h_index != -1: hours = int(duration_str[:h_index]); duration_str = duration_str[h_index+1:]
        m_index = duration_str.find('M')
        if m_index != -1: minutes = int(duration_str[:m_index])
        parts = [];
        if hours > 0: parts.append(f"{hours}h")
        if minutes > 0: parts.append(f"{minutes}m")
        return " ".join(parts) if parts else "0m"
    except Exception: logger.warning(f"Could not parse duration: {duration_str}", exc_info=True); return duration_str

def format_flight_results(flight_offers: List[Dict[str, Any]], origin_code: str, dest_code: str, carriers_dict: Dict[str, str]) -> str:
    """Format flight search results into readable text, including airline names."""
    if not flight_offers:
        return f"No flight offers found from {origin_code} to {dest_code}."

    offers_to_display_count = min(len(flight_offers), 6) # Show up to 6
    output = [f"Found {len(flight_offers)} flight options from {origin_code} to {dest_code} (sorted by price, showing top {offers_to_display_count}):"]

    for i, offer in enumerate(flight_offers[:offers_to_display_count], 1): # Loop up to 6
        price = offer.get("price", {}); currency = price.get("currency", "USD")
        total_price = price.get("total", "N/A"); grand_total = price.get("grandTotal", total_price)
        output.append(f"\n{i}. Flight Option - {currency} {grand_total}")

        for j, itinerary in enumerate(offer.get("itineraries", [])):
            direction = "Outbound" if j == 0 else "Return"
            duration_formatted = format_duration(itinerary.get("duration", ""))
            output.append(f"   {direction} Journey ({duration_formatted}):")

            for k, segment in enumerate(itinerary.get("segments", [])):
                dep = segment.get("departure", {}); arr = segment.get("arrival", {})
                carrier_code = segment.get("carrierCode", "")
                # --- Look up airline name ---
                airline_name = carriers_dict.get(carrier_code, carrier_code) # Default to code if name not found
                flight_num = segment.get("number", "")
                aircraft_code = segment.get("aircraft", {}).get("code", "")
                dep_code = dep.get("iataCode", ""); arr_code = arr.get("iataCode", "")
                dep_term = f" (Term {dep.get('terminal', '')})" if dep.get('terminal') else ""
                arr_term = f" (Term {arr.get('terminal', '')})" if arr.get('terminal') else ""
                try:
                    dep_dt = datetime.fromisoformat(dep.get("at", "").replace("Z", "+00:00"))
                    arr_dt = datetime.fromisoformat(arr.get("at", "").replace("Z", "+00:00"))
                    dep_str = dep_dt.strftime("%a, %b %d %H:%M"); arr_str = arr_dt.strftime("%a, %b %d %H:%M")
                    time_info = f"{dep_str} -> {arr_str}"
                except Exception: time_info = f"{dep.get('at', '')} -> {arr.get('at', '')}"
                # --- Include airline name in output ---
                output.append(f"     {k+1}. {airline_name} ({carrier_code}{flight_num}): {dep_code}{dep_term} -> {arr_code}{arr_term} (Aircraft: {aircraft_code})")
                output.append(f"        {time_info}")

    output.append("\n**Note**: Prices and availability are subject to change. Please verify details directly with airlines or booking sites.")
    return "\n".join(output)

# --- Updated function ---
@mcp.tool()
async def get_flight_offers(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str = None,
    adults: int = 1,
    travel_class: str = None,
    included_airline_codes: str = None # New parameter for airline filtering
) -> str:
    """Search for flight options between cities, sorted by price, including airline names and optional airline filtering.

    Args:
        origin: Origin city or airport code
        destination: Destination city or airport code
        departure_date: Departure date (YYYY-MM-DD)
        return_date: Return date (YYYY-MM-DD) for round trips
        adults: Number of adult travelers
        travel_class: Cabin class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
        included_airline_codes: Comma-separated IATA codes of airlines to include (e.g., 'DL,UA')

    Returns:
        Flight information in text format or an error message.
    """
    log_prefix = f"Flights {origin}->{destination} ({departure_date})"
    logger.info(f"{log_prefix} - Search initiated.")

    if not amadeus:
        logger.error("{log_prefix} - Error: Amadeus client not initialized.")
        return "Error: Amadeus client not initialized. Check credentials."

    origin_code = get_airport_code(origin); dest_code = get_airport_code(destination)
    if not origin_code: return f"Error: Could not find airport code for origin '{origin}'."
    if not dest_code: return f"Error: Could not find airport code for destination '{destination}'."

    try:
        datetime.strptime(departure_date, '%Y-%m-%d')
        if return_date: datetime.strptime(return_date, '%Y-%m-%d')
    except ValueError: logger.error(f"{log_prefix} - Invalid date format. Got dep: {departure_date}, ret: {return_date}"); return "Error: Invalid date format. Please use YYYY-MM-DD."

    params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": dest_code,
        "departureDate": departure_date,
        "adults": adults,
        "currencyCode": "USD",
        "max": 50 # Request more results for sorting/filtering
    }
    if return_date: params["returnDate"] = return_date
    if travel_class and travel_class in ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]: params["travelClass"] = travel_class
    else: params["travelClass"] = "ECONOMY"; logger.warning(f"{log_prefix} - Invalid/missing travel_class '{travel_class}', defaulting to ECONOMY.")

    # --- Add airline filter parameter ---
    if included_airline_codes:
        params["includedAirlineCodes"] = included_airline_codes
        logger.info(f"{log_prefix} - Applying airline filter: {included_airline_codes}")
    # ----------------------------------

    logger.info(f"{log_prefix} - Calling Amadeus API with params: {params}")
    try:
        response = await asyncio.to_thread( amadeus.shopping.flight_offers_search.get, **params )

        flight_offers = response.data
        if flight_offers:
            def get_price_key(offer):
                try: price_str = offer.get('price', {}).get('grandTotal', offer.get('price', {}).get('total')); return float(price_str) if price_str else float('inf')
                except (ValueError, TypeError): return float('inf')
            flight_offers.sort(key=get_price_key)
            logger.info(f"{log_prefix} - Sorted {len(flight_offers)} offers by price.")

        # Extract carrier dictionary for names
        carriers_dict = response.result.get("dictionaries", {}).get("carriers", {})

        logger.info(f"{log_prefix} - Successfully received and processed {len(flight_offers)} offers.")
        return format_flight_results(flight_offers, origin_code, dest_code, carriers_dict)

    except ResponseError as e:
        error_details = f"Status Code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}"
        response_data = None; error_messages = []
        if hasattr(e, 'response') and hasattr(e.response, 'data'): response_data = e.response.data
        if response_data and 'errors' in response_data:
            error_messages = [f"{err.get('code', 'UNK')}: {err.get('detail', 'No detail')}" for err in response_data['errors']]
            error_details += f" | Errors: {'; '.join(error_messages)}"
        logger.error(f"{log_prefix} - Amadeus API ResponseError: {error_details}", exc_info=True)
        return f"Error searching flights: Amadeus API Error ({error_messages[0] if error_messages else 'Details unavailable'})"
    except Exception as e:
        logger.error(f"{log_prefix} - Unexpected error during flight search: {str(e)}", exc_info=True)
        return f"Error searching flights: An unexpected error occurred ({type(e).__name__})."

if __name__ == "__main__":
    # import sys # Already imported
    transport = "stdio";
    if len(sys.argv) > 1: transport = sys.argv[1]
    logger.info(f"Starting Amadeus MCP server with transport: {transport}")
    try:
        if amadeus: mcp.run(transport=transport)
        else: logger.error("Cannot start MCP server: Amadeus client failed to initialize."); sys.exit(1)
    except Exception as e: logger.critical(f"Fatal error running Amadeus MCP server: {str(e)}", exc_info=True); sys.exit(1)