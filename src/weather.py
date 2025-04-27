# src/weather.py

from mcp.server.fastmcp import FastMCP
import logging
import os
import sys
import json
import httpx
from dotenv import load_dotenv
import googlemaps
from typing import Optional, Dict, Tuple, List, Any
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# --- Constants ---
COORDINATES = {
    "sfo": (37.6213, -122.3790), "san francisco": (37.6213, -122.3790),
    "sea": (47.4502, -122.3088), "seattle": (47.4502, -122.3088),
    "lax": (33.9416, -118.4085), "los angeles": (33.9416, -118.4085),
    "jfk": (40.6413, -73.7781), "new york": (40.6413, -73.7781), "nyc": (40.6413, -73.7781),
    "den": (39.8561, -104.6737), "denver": (39.8561, -104.6737),
    "bos": (42.3656, -71.0096), "boston": (42.3656, -71.0096),
    "ord": (41.9742, -87.9073), "chicago": (41.9742, -87.9073),
    "pdx": (45.5898, -122.5951), "portland": (45.5898, -122.5951),
    "san": (32.7338, -117.1933), "san diego": (32.7338, -117.1933),
    "aus": (30.1975, -97.6664), "austin": (30.1975, -97.6664),
    "mariposa, ca": (37.4849, -119.9663), # Near Yosemite
    "yosemite valley": (37.7456, -119.5936), # Inside Yosemite
    "springdale, ut": (37.1891, -112.9980), # Near Zion
    "st. george, ut": (37.0952, -113.5759), # Near Zion, SGU area
}
NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "(TravelPlannerApp/1.0, contact@example.com)")

# --- Initialize Google Maps Client ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps_client = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logging.info("Weather Service: Google Maps client initialized for geocoding fallback.")
    except Exception as e:
        logging.error(f"Weather Service: Failed to initialize Google Maps client: {e}", exc_info=True)
else:
    logging.warning("Weather Service: GOOGLE_MAPS_API_KEY not set. Geocoding fallback for weather locations will not work.")

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO) # Set desired level (INFO, DEBUG, etc.)
logger.propagate = False

# --- MCP Server Setup ---
mcp = FastMCP("weather")
logger.info("Weather MCP server initialized")

# --- Helper Functions ---
async def get_nws_grid_url(lat: float, lon: float) -> Optional[str]:
    """Gets the NWS API grid forecast URL for given coordinates."""
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Requesting NWS points URL: {points_url}")
            response = await client.get(points_url, headers=headers, follow_redirects=True, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            grid_url = data.get("properties", {}).get("forecastGridData")
            logger.debug(f"Received grid_url from NWS points API: {grid_url}")
            return grid_url
    except Exception as e:
        logger.error(f"Error getting NWS grid URL: {e}", exc_info=True)
    return None

async def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """Finds coordinates for a location name using internal dict or Google Geocoding fallback."""
    location_lower = location_name.lower().strip()
    logger.debug(f"Attempting to geocode: {location_lower}")
    if location_lower in COORDINATES:
        logger.info(f"Found '{location_lower}' in internal coordinate dictionary.")
        return COORDINATES[location_lower]
    if gmaps_client:
        logger.warning(f"Location '{location_lower}' not in internal dict. Attempting Google Geocoding fallback...")
        try:
            geocode_result = gmaps_client.geocode(location_name)
            if geocode_result and len(geocode_result) > 0:
                coords = geocode_result[0].get("geometry", {}).get("location", {})
                lat = coords.get('lat'); lng = coords.get('lng')
                if lat is not None and lng is not None:
                     logger.info(f"Geocoding fallback successful for '{location_name}': ({lat}, {lng})")
                     return (lat, lng)
                else: logger.error(f"Geocoding fallback for '{location_name}' succeeded but coordinates missing.")
            else: logger.error(f"Geocoding fallback failed for '{location_name}': No results found.")
        except Exception as e: logger.error(f"Error during Google Geocoding fallback for '{location_name}': {e}", exc_info=True)
    else: logger.warning("Google Maps client not available for geocoding fallback.")
    logger.error(f"Could not find coordinates for '{location_name}' via internal dict or Google Geocoding.")
    return None

def celsius_to_fahrenheit(celsius: Optional[Any]) -> Optional[int]:
    """Converts Celsius (float/int/numeric str) to Fahrenheit and rounds."""
    if celsius is None: return None
    try: return int(round(float(celsius) * 9.0/5.0 + 32))
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert Celsius value '{celsius}' ({type(celsius)}) to Fahrenheit: {e}")
        return None

def format_summary_from_grid_properties(properties: Dict[str, Any], location: str) -> str:
    """Formats a summary forecast string from NWS gridpoint properties."""
    summary_parts = [f"Summary forecast for {location.upper()}:"]
    try:
        temp_data = properties.get("temperature", {})
        temp_val_orig = temp_data.get("value"); temp_unit = temp_data.get("unitCode", "").upper()
        logger.debug(f"NWS Summary Data for {location} - Current Temp Raw: Value={temp_val_orig}, Unit='{temp_unit}'")
        if temp_val_orig is not None:
            is_fahrenheit = "DEGF" in temp_unit or "FAHRENHEIT" in temp_unit
            temp_val_f = int(round(float(temp_val_orig))) if is_fahrenheit else celsius_to_fahrenheit(temp_val_orig)
            if temp_val_f is not None: summary_parts.append(f"- Current Temp: {temp_val_f}°F")

        max_temp_data = properties.get("maxTemperature", {}); max_temps = max_temp_data.get("values", [])
        if max_temps and max_temps[0].get("value") is not None:
             max_val_orig = max_temps[0]["value"]; max_unit = max_temp_data.get("unitCode", "").upper()
             logger.debug(f"NWS Summary Data for {location} - Max Temp Raw: Value={max_val_orig}, Unit='{max_unit}'")
             is_fahrenheit = "DEGF" in max_unit or "FAHRENHEIT" in max_unit
             max_val_f = int(round(float(max_val_orig))) if is_fahrenheit else celsius_to_fahrenheit(max_val_orig)
             if max_val_f is not None: summary_parts.append(f"- Upcoming Max Temp: {max_val_f}°F")

        min_temp_data = properties.get("minTemperature", {}); min_temps = min_temp_data.get("values", [])
        if min_temps and min_temps[0].get("value") is not None:
             min_val_orig = min_temps[0]["value"]; min_unit = min_temp_data.get("unitCode", "").upper()
             logger.debug(f"NWS Summary Data for {location} - Min Temp Raw: Value={min_val_orig}, Unit='{min_unit}'")
             is_fahrenheit = "DEGF" in min_unit or "FAHRENHEIT" in min_unit
             min_val_f = int(round(float(min_val_orig))) if is_fahrenheit else celsius_to_fahrenheit(min_val_orig)
             if min_val_f is not None: summary_parts.append(f"- Upcoming Min Temp: {min_val_f}°F")

        weather_data = properties.get("weather", {}); weather_conditions = weather_data.get("values", [])
        if weather_conditions and weather_conditions[0].get("value"):
            conditions_list = weather_conditions[0]['value']
            if conditions_list and isinstance(conditions_list, list) and len(conditions_list) > 0:
                 primary_condition = conditions_list[0].get("weather")
                 if primary_condition and isinstance(primary_condition, str):
                     summary_parts.append(f"- General Conditions: {primary_condition.capitalize()}")
                 else: logger.debug(f"Primary condition value was None or not a string for {location}. Skipping.")

        if len(summary_parts) == 1: return f"Basic weather grid data retrieved for {location}, but no specific summary details found."
        return "\n".join(summary_parts)
    except Exception as e:
        logger.error(f"Error formatting summary from grid properties for {location}: {e}", exc_info=True)
        return f"Error processing summary weather data for {location}."


# --- MCP Tool (Updated Formatting Logic) ---
@mcp.tool()
async def get_weather_forecast(location: str, date: Optional[str] = None) -> str:
    """
    Gets the 7-day weather forecast for a US location using the NWS API.
    Falls back to a basic summary if detailed periods are unavailable.
    If a date (YYYY-MM-DD) is provided, it attempts to return the forecast for that specific day (only from detailed forecast).
    """
    logger.info(f"Tool called: get_weather_forecast for location: {location}, date: {date}")

    coords = await geocode_location(location)
    if not coords: return f"Error: Could not find coordinates for US location '{location}'. Please provide a known US city or airport code."

    base_grid_url = await get_nws_grid_url(coords[0], coords[1])

    if not base_grid_url or not isinstance(base_grid_url, str) or not base_grid_url.startswith("https://api.weather.gov"):
        logger.error(f"Invalid or missing base_grid_url received: {base_grid_url}")
        return f"Error: Could not retrieve a valid NWS forecast grid URL for {location}."

    forecast_url = base_grid_url + "/forecast"
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/ld+json"}

    periods_found = False # Flag to track if detailed periods were processed

    try:
        logger.info(f"Attempting to fetch detailed forecast from: {forecast_url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(forecast_url, headers=headers, follow_redirects=True, timeout=15.0)

            if response.status_code == 404:
                 logger.warning(f"NWS detailed forecast API returned 404 for {forecast_url}. Trying base gridpoint fallback.")
            elif response.status_code == 200:
                data = response.json()
                periods = data.get("properties", {}).get("periods", [])
                if periods:
                    logger.info(f"Successfully fetched {len(periods)} detailed forecast periods.")
                    periods_found = True
                    # --- Format Detailed Periods (CORRECTED) ---
                    target_date_obj = None
                    if date:
                        try: target_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                        except ValueError: logger.warning(f"Invalid date format: {date}. Returning full forecast.")

                    day_forecast: Dict[str, List[str]] = {} # Explicit type hint
                    for period in periods:
                        start_time_str = period.get("startTime"); period_date = None
                        if start_time_str:
                            try: period_date = datetime.fromisoformat(start_time_str).date()
                            except ValueError: pass
                        # Apply date filter if requested
                        if target_date_obj and period_date != target_date_obj: continue

                        name = period.get("name", "Unknown")
                        temp = period.get("temperature", "N/A"); temp_unit = period.get("temperatureUnit", "F")
                        wind_speed = period.get("windSpeed", "N/A"); wind_dir = period.get("windDirection", "")
                        short_forecast = period.get("shortForecast", "No description")

                        # Group by logical day name (e.g., "Saturday")
                        day_name_match = re.match(r"^(Today|Tonight|Overnight|\w+day)", name) # Match common day starts
                        day_name = day_name_match.group(1) if day_name_match else "Other" # Group others if name format unexpected

                        if day_name not in day_forecast: day_forecast[day_name] = []
                        forecast_line = f"{name}: Temp: {temp}°{temp_unit}, Wind: {wind_speed} {wind_dir}. {short_forecast}."
                        day_forecast[day_name].append(forecast_line)

                    if not day_forecast:
                         # This occurs if periods were found but none matched the date filter
                         periods_found = False
                         logger.warning(f"No detailed forecast periods found for specified date {date} in {location}.")
                    else:
                        # *** FIXED Output Formatting Logic ***
                        output_lines = [f"7-Day Weather Forecast Summary for {location.upper()}:"] # Start with header
                        day_separator = "\n---\n"
                        processed_days = set()
                        day_order = ["Today", "Tonight", "Overnight", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                        # Ensure keys are sorted correctly according to day_order
                        sorted_day_keys = sorted(day_forecast.keys(), key=lambda d: day_order.index(d) if d in day_order else 99)

                        for day in sorted_day_keys:
                            if day in day_forecast and day not in processed_days:
                                # Check if it's a night period that should be grouped
                                is_night = "Night" in day or "Overnight" in day
                                corresponding_day = day.replace(" Night", "").replace("Overnight", "Today") # Simple mapping

                                # Only add header if it's a new day or a standalone night
                                if not is_night or corresponding_day not in processed_days:
                                     output_lines.append(f"## {day}")

                                output_lines.extend([f"  - {line}" for line in day_forecast[day]]) # Indent forecast lines
                                processed_days.add(day)

                                # Handle corresponding night/day if not already processed
                                if not is_night:
                                     night_name = day + " Night"
                                     if night_name in day_forecast and night_name not in processed_days:
                                         # output_lines.append(f"**{night_name}:**") # Optional night header
                                         output_lines.extend([f"  - {line}" for line in day_forecast[night_name]])
                                         processed_days.add(night_name)
                                # If it *was* a night, ensure its day gets processed if available later
                                # (Handled by the loop structure and processed_days check)

                        return "\n".join(output_lines) # Join all collected lines
                    # --- End Detailed Formatting ---
                else: # periods list was empty
                    logger.warning(f"NWS detailed forecast had empty 'periods' list for {forecast_url}. Trying base gridpoint fallback.")
            else:
                response.raise_for_status()

        # If periods_found is still False, proceed to fallback
        if not periods_found:
            logger.info(f"Falling back to fetch base gridpoint data from: {base_grid_url}")
            headers_base = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
            async with httpx.AsyncClient() as client:
                response_base = await client.get(base_grid_url, headers=headers_base, follow_redirects=True, timeout=15.0)
                if response_base.status_code == 200:
                    try:
                        data_base = response_base.json()
                        properties = data_base.get("properties")
                        if properties: return format_summary_from_grid_properties(properties, location)
                        else:
                            logger.warning(f"Base gridpoint response for {base_grid_url} missing 'properties'.")
                            return f"Detailed forecast unavailable. Basic grid data retrieved for {location} but lacks summary properties."
                    except json.JSONDecodeError:
                         logger.error(f"Failed to decode JSON from base gridpoint URL: {base_grid_url}")
                         return f"Detailed forecast unavailable, and failed to process summary data for {location}."
                else:
                    logger.error(f"Base gridpoint URL {base_grid_url} returned status {response_base.status_code}: {response_base.text}")
                    return f"Detailed forecast unavailable, and failed to retrieve summary data for {location} (HTTP {response_base.status_code})."

    # Exception Handling for the outer try block
    except httpx.RequestError as exc:
        failed_url = exc.request.url if hasattr(exc, 'request') else forecast_url
        logger.error(f"HTTP error contacting NWS forecast API ({failed_url}): {exc}", exc_info=True)
        return f"Error: Could not connect to weather service for {location}."
    except httpx.HTTPStatusError as exc:
        logger.error(f"NWS forecast API returned status {exc.response.status_code} for {exc.request.url}: {exc.response.text}")
        return f"Error: Weather service returned status {exc.response.status_code} for {location}."
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from NWS forecast API: {forecast_url}")
        return f"Error: Failed to process weather data for {location}."
    except Exception as e:
        logger.error(f"Unexpected error getting weather forecast: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while fetching weather for {location}."


# --- Main execution block ---
if __name__ == "__main__":
    transport = "stdio"
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    logger.info(f"Starting weather MCP server with transport: {transport}")
    try:
        mcp.run(transport=transport)
    except Exception as e:
        logger.critical(f"Error running Weather MCP server: {str(e)}", exc_info=True)
        sys.exit(1)