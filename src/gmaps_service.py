# src/gmaps_service.py

from mcp.server.fastmcp import FastMCP
import logging
import os
import sys
import json # Added for JSON serialization
from dotenv import load_dotenv
import googlemaps # Import the library
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Use INFO for less noise, DEBUG for more detail
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server with the name from the README
mcp = FastMCP("google-maps")
logger.info(f"Google Maps MCP server initialized with name: google-maps")


# Get API key and initialize Google Maps client
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps_client = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logger.info("Google Maps client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Google Maps client: {e}", exc_info=True)
else:
    logger.warning("GOOGLE_MAPS_API_KEY not set in environment variables. Google Maps tools will not function.")

# --- Helper Function to handle API errors ---
def handle_gmaps_exception(e, tool_name):
    """Standard way to log and return errors from Google Maps API calls."""
    if isinstance(e, googlemaps.exceptions.ApiError):
        logger.error(f"Google Maps API Error in {tool_name}: Status {e.status}, Message: {str(e)}")
        return f"ERROR_MCP: Google Maps API Error ({e.status}): {str(e)}"
    elif isinstance(e, googlemaps.exceptions.HTTPError):
        logger.error(f"Google Maps HTTP Error in {tool_name}: {e}", exc_info=True)
        return f"ERROR_MCP: Google Maps Connection Error: {e}"
    elif isinstance(e, googlemaps.exceptions.Timeout):
        logger.error(f"Google Maps Timeout in {tool_name}", exc_info=True)
        return f"ERROR_MCP: Google Maps operation timed out."
    else:
        logger.error(f"Unexpected error in {tool_name}: {e}", exc_info=True)
        return f"ERROR_MCP: Unexpected error in {tool_name}: {str(e)}"

# --- Helper to safely serialize results ---
def safe_json_dumps(data: Any, tool_name: str) -> str:
    """Serializes data to JSON string, returning error string on failure."""
    try:
        return json.dumps(data)
    except TypeError as e:
        logger.error(f"Failed to serialize result for {tool_name} to JSON: {e}")
        return f"ERROR_MCP: Failed to format {tool_name} result as JSON."


# --- Google Maps Tools ---

@mcp.tool()
async def maps_geocode(address: str) -> str:
    """
    Convert address to coordinates (latitude, longitude).

    Args:
        address (string): The street address or place name to geocode.

    Returns:
        string: JSON string of {'location': {'latitude': float, 'longitude': float}, 'formatted_address': str, 'place_id': str} or error string.
    """
    logger.info(f"Tool called: maps_geocode for address='{address}'")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."

    try:
        geocode_result = gmaps_client.geocode(address)
        if geocode_result and len(geocode_result) > 0:
            first_result = geocode_result[0]
            location = first_result.get("geometry", {}).get("location", {})
            formatted_address = first_result.get("formatted_address")
            place_id = first_result.get("place_id")
            lat = location.get('lat')
            lng = location.get('lng')

            if lat is not None and lng is not None:
                result_dict = {
                    "location": {"latitude": lat, "longitude": lng},
                    "formatted_address": formatted_address,
                    "place_id": place_id,
                }
                return safe_json_dumps(result_dict, "maps_geocode")
            else:
                return "ERROR_MCP: Geocoding succeeded but location coordinates missing."
        else:
            return f"ERROR_MCP: Geocoding failed for '{address}'. No results found."
    except Exception as e:
        return handle_gmaps_exception(e, "maps_geocode")

@mcp.tool()
async def maps_reverse_geocode(latitude: float, longitude: float) -> str:
    """
    Convert coordinates (latitude, longitude) to address.

    Args:
        latitude (number): The latitude coordinate.
        longitude (number): The longitude coordinate.

    Returns:
        string: JSON string of {'formatted_address': str, 'place_id': str, 'address_components': list} or error string.
    """
    logger.info(f"Tool called: maps_reverse_geocode for lat={latitude}, lon={longitude}")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."

    try:
        reverse_geocode_result = gmaps_client.reverse_geocode((latitude, longitude))
        if reverse_geocode_result and len(reverse_geocode_result) > 0:
            first_result = reverse_geocode_result[0]
            result_dict = {
                "formatted_address": first_result.get("formatted_address"),
                "place_id": first_result.get("place_id"),
                "address_components": first_result.get("address_components"),
            }
            return safe_json_dumps(result_dict, "maps_reverse_geocode")
        else:
            return f"ERROR_MCP: Reverse geocoding failed for ({latitude}, {longitude}). No results found."
    except Exception as e:
        return handle_gmaps_exception(e, "maps_reverse_geocode")


@mcp.tool()
async def maps_search_places(query: str, location: Optional[Dict[str, float]] = None, radius: Optional[int] = None) -> str:
    """
    Search for places using Google Places API (Text Search).

    Args:
        query (string): The search text (e.g., "mexican restaurants", "MoPOP Seattle").
        location (optional): {'latitude': float, 'longitude': float} to bias results.
        radius (optional): Distance in meters (max 50000) to bias results. 'location' must also be provided.

    Returns:
        string: JSON string representing List[Dict] of places (name, formatted_address, location, place_id, rating, user_ratings_total) or error string. Returns '[]' for zero results.
    """
    logger.info(f"Tool called: maps_search_places for query='{query}', location='{location}', radius='{radius}'")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."

    try:
        # Convert location dict to tuple if provided
        location_tuple = (location['latitude'], location['longitude']) if location else None
        if radius and not location_tuple:
            return "ERROR_MCP: 'radius' requires 'location' to be provided."

        places_result = gmaps_client.places(
            query=query,
            location=location_tuple,
            radius=radius
        )

        status = places_result.get('status')
        results = places_result.get('results', [])

        if status == 'OK':
            logger.info(f"Google Places API returned status: OK. Found {len(results)} raw results.")
            # Prepare the list of simplified place dictionaries
            places_list = [
                {
                    "name": place.get("name"),
                    "formatted_address": place.get("formatted_address"),
                    "location": place.get("geometry", {}).get("location"), # Returns {'lat': float, 'lng': float}
                    "place_id": place.get("place_id"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total")
                } for place in results
            ]
            return safe_json_dumps(places_list, "maps_search_places")
        elif status == 'ZERO_RESULTS':
             logger.info(f"Google Places API returned status: ZERO_RESULTS for query: {query}")
             return "[]" # Return an empty JSON list string
        else:
            error_msg = places_result.get('error_message', 'Status: ' + status)
            logger.error(f"Google Places API error: {error_msg}")
            return f"ERROR_MCP: Google Maps Places API error: {status}"

    except Exception as e:
        return handle_gmaps_exception(e, "maps_search_places")


@mcp.tool()
async def maps_place_details(place_id: str) -> str:
    """
    Get detailed information about a specific place using its Place ID.

    Args:
        place_id (string): The Google Place ID to retrieve details for.

    Returns:
        string: JSON string of detailed place information (name, address, contact, rating, reviews, hours, etc.) or error string.
    """
    logger.info(f"Tool called: maps_place_details for place_id='{place_id}'")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."

    try:
        # Define fields to request - customize as needed
        fields = [
            'name', 'formatted_address', 'place_id', 'geometry', 'rating',
            'user_ratings_total', 'international_phone_number', 'website',
            'opening_hours', 'price_level', 'types', 'review' # Request some reviews
        ]
        details_result = gmaps_client.place(place_id=place_id, fields=fields)

        status = details_result.get('status')
        result = details_result.get('result')

        if status == 'OK':
            logger.info(f"Google Place Details API returned status: OK for place_id: {place_id}")
            return safe_json_dumps(result, "maps_place_details") # Return the result dict as JSON string
        else:
            error_msg = details_result.get('error_message', 'Status: ' + status)
            logger.error(f"Google Place Details API error: {error_msg}")
            return f"ERROR_MCP: Google Maps Place Details API error: {status}"

    except Exception as e:
        return handle_gmaps_exception(e, "maps_place_details")


@mcp.tool()
async def maps_distance_matrix(origins: List[str], destinations: List[str], mode: Optional[str] = "driving") -> str:
    """
    Calculate travel distances and durations between multiple origins and destinations.

    Args:
        origins (List[str]): List of starting addresses or coordinates (lat,lng).
        destinations (List[str]): List of ending addresses or coordinates (lat,lng).
        mode (optional): Travel mode ('driving', 'walking', 'bicycling', 'transit'). Defaults to 'driving'.

    Returns:
        string: JSON string of the raw Distance Matrix API response containing rows/elements with distance/duration, or error string.
    """
    logger.info(f"Tool called: maps_distance_matrix")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."
    if not origins or not destinations: return "ERROR_MCP: Origins and destinations lists cannot be empty."
    valid_modes = ["driving", "walking", "bicycling", "transit"]
    if mode not in valid_modes: return f"ERROR_MCP: Invalid mode '{mode}'. Must be one of {valid_modes}."

    try:
        matrix_result = gmaps_client.distance_matrix(
            origins=origins,
            destinations=destinations,
            mode=mode
        )
        status = matrix_result.get('status')
        if status == 'OK':
             logger.info(f"Distance Matrix API returned status: OK.")
             return safe_json_dumps(matrix_result, "maps_distance_matrix") # Return the full response dict as JSON string
        else:
             error_msg = matrix_result.get('error_message', 'Status: ' + status)
             logger.error(f"Google Distance Matrix API error: {error_msg}")
             return f"ERROR_MCP: Google Distance Matrix API error: {status}"

    except Exception as e:
        return handle_gmaps_exception(e, "maps_distance_matrix")


@mcp.tool()
async def maps_elevation(locations: List[Dict[str, float]]) -> str:
    """
    Get elevation data for one or more locations.

    Args:
        locations (List[Dict]): List of location dictionaries, e.g., [{'latitude': 40.71, 'longitude': -74.00}, ...]

    Returns:
        string: JSON string representing List[Dict] of results (elevation, location, resolution), or error string.
    """
    logger.info(f"Tool called: maps_elevation")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."
    if not locations: return "ERROR_MCP: Locations list cannot be empty."

    # Convert list of dicts to list of tuples expected by the library
    location_tuples = []
    try:
         for loc in locations:
              if 'latitude' in loc and 'longitude' in loc:
                   location_tuples.append((loc['latitude'], loc['longitude']))
              else:
                   raise ValueError("Missing 'latitude' or 'longitude' in location dict")
    except (TypeError, ValueError):
         return "ERROR_MCP: Invalid format in locations list. Each item must be a dict with 'latitude' and 'longitude'."

    try:
        elevation_result = gmaps_client.elevation(locations=location_tuples)
        # The API returns a list of results directly if successful
        if isinstance(elevation_result, list):
             logger.info(f"Elevation API returned {len(elevation_result)} results.")
             return safe_json_dumps(elevation_result, "maps_elevation") # Return the list as JSON string
        else:
             status = elevation_result.get('status', 'UNKNOWN_ERROR')
             error_msg = elevation_result.get('error_message', 'Failed to get elevation data.')
             logger.error(f"Google Elevation API error: {error_msg}")
             return f"ERROR_MCP: Google Elevation API error: {status}"

    except Exception as e:
        return handle_gmaps_exception(e, "maps_elevation")

@mcp.tool()
async def maps_directions(origin: str, destination: str, mode: Optional[str] = "driving") -> str:
    """
    Get directions between two points.

    Args:
        origin (string): Starting address, Place ID (place_id:...), or coordinates (lat,lng).
        destination (string): Ending address, Place ID (place_id:...), or coordinates (lat,lng).
        mode (optional): Travel mode ('driving', 'walking', 'bicycling', 'transit'). Defaults to 'driving'.

    Returns:
        string: JSON string representing List[Dict] containing route details (legs, steps, distance, duration, etc.) or error string. Returns '[]' for no route found.
    """
    logger.info(f"Tool called: maps_directions from '{origin}' to '{destination}'")
    if not gmaps_client: return "ERROR_MCP: Google Maps client not initialized."
    valid_modes = ["driving", "walking", "bicycling", "transit"]
    if mode not in valid_modes: return f"ERROR_MCP: Invalid mode '{mode}'. Must be one of {valid_modes}."

    try:
        directions_result = gmaps_client.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            departure_time=datetime.now() # Optional: use current time for traffic/transit
        )
        # The API returns a list of routes (usually just one)
        if isinstance(directions_result, list):
            if directions_result: # List is not empty
                logger.info(f"Directions API returned {len(directions_result)} route(s).")
                return safe_json_dumps(directions_result, "maps_directions") # Return the list of routes as JSON string
            else: # Empty list means no route found
                logger.info(f"Directions API returned status: ZERO_RESULTS.")
                return "[]" # Return empty JSON list string for zero results
        elif isinstance(directions_result, dict) and directions_result.get('status') != 'OK':
            status = directions_result.get('status', 'UNKNOWN_ERROR')
            error_msg = directions_result.get('error_message', f"Failed to get directions ({status}).")
            logger.error(f"Google Directions API error: {error_msg}")
            return f"ERROR_MCP: Google Directions API error: {status}"
        else:
             logger.error(f"Unexpected Directions API response format: {type(directions_result)}")
             return f"ERROR_MCP: Unexpected response format from Directions API."

    except Exception as e:
        return handle_gmaps_exception(e, "maps_directions")


# --- Main execution block ---
if __name__ == "__main__":
    transport = "stdio"
    if len(sys.argv) > 1:
        transport = sys.argv[1]

    logger.info(f"Starting Google Maps MCP server with transport: {transport}")
    if not gmaps_client:
        logger.critical("Google Maps client failed to initialize. MCP Server cannot function properly. Exiting.")
        sys.exit(1) # Exit if client isn't ready

    try:
        mcp.run(transport=transport)
    except Exception as e:
        logger.critical(f"Error running Google Maps MCP server: {str(e)}", exc_info=True)
        sys.exit(1)