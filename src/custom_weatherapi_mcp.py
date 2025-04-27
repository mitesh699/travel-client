# src/custom_weatherapi_mcp.py

import os
import sys
import json
import httpx
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Union # Added Union
from datetime import datetime, timedelta, date

# Add project root to path if necessary
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp library not found. Please install langchain-mcp-adapters.")
    sys.exit(1)

# Load environment variables
load_dotenv(override=True)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("custom_weatherapi_mcp") # Specific logger name

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_BASE_URL = "http://api.weatherapi.com/v1"

if not WEATHER_API_KEY:
    logger.critical("WEATHER_API_KEY not found in environment variables. Server cannot function.")
    # sys.exit(1) # Uncomment to exit if key is missing

# --- MCP Server Initialization ---
# Use a distinct name for your custom server
mcp = FastMCP("custom_weatherapi")
logger.info(f"Custom WeatherAPI MCP server initialized with name: {mcp.name}")

# --- Helper Function for API Calls ---
async def _call_weather_api(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to call WeatherAPI.com"""
    if not WEATHER_API_KEY:
        raise ValueError("API Key not configured")

    # Add API key to parameters
    params["key"] = WEATHER_API_KEY
    url = f"{WEATHER_API_BASE_URL}/{endpoint}"

    async with httpx.AsyncClient() as client:
        try:
            logger.debug(f"Calling WeatherAPI: {url} with params: {params}")
            response = await client.get(url, params=params, timeout=20.0)
            response.raise_for_status() # Raise HTTP errors (4xx, 5xx)
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                 error_json = e.response.json()
                 error_detail = error_json.get("error", {}).get("message", e.response.text)
                 logger.error(f"HTTP Error calling WeatherAPI ({e.response.status_code}): {error_detail}")
                 # Return specific error structure if available
                 return {"error": {"code": error_json.get("error", {}).get("code", e.response.status_code), "message": error_detail}}
            except Exception:
                 logger.error(f"HTTP Error calling WeatherAPI ({e.response.status_code}): {error_detail}")
                 # Return generic error structure
                 return {"error": {"code": e.response.status_code, "message": error_detail}}
        except httpx.RequestError as e:
            logger.error(f"Request Error calling WeatherAPI: {e}")
            return {"error": {"code": 503, "message": f"Could not connect to WeatherAPI: {e}"}} # 503 Service Unavailable
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error for WeatherAPI response: {e}")
            return {"error": {"code": 500, "message": "Failed to parse WeatherAPI response."}}
        except Exception as e:
            logger.error(f"Unexpected error calling WeatherAPI: {e}", exc_info=True)
            return {"error": {"code": 500, "message": f"An unexpected error occurred: {e}"}}

# --- Helper Function for Celsius to Fahrenheit ---
def celsius_to_fahrenheit(celsius: Optional[Union[float, int, str]]) -> Optional[int]:
    """Converts Celsius to Fahrenheit and rounds."""
    if celsius is None: return None
    try: return int(round(float(celsius) * 9.0/5.0 + 32))
    except (TypeError, ValueError): return None

# --- Tool Definitions ---

@mcp.tool()
async def get_weatherapi_forecast(
    location: str,
    days: int = 3,
    aqi: str = "yes",
    alerts: str = "yes"
) -> str:
    """
    Gets the weather forecast (current + future days) for a location using WeatherAPI.com.

    Args:
        location (string): City name, zip code, lat/lon, etc.
        days (int): Number of forecast days (1-14). Defaults to 3.
        aqi (string): Include Air Quality Index data ('yes' or 'no'). Defaults to 'yes'.
        alerts (string): Include weather alerts ('yes' or 'no'). Defaults to 'yes'.

    Returns:
        string: Formatted weather forecast information or an error message string starting with ERROR_MCP:.
    """
    logger.info(f"Tool call: get_weatherapi_forecast('{location}', days={days}, aqi='{aqi}', alerts='{alerts}')")
    if not 1 <= days <= 14:
        return "ERROR_MCP: Invalid number of days. Must be between 1 and 14."

    params = {"q": location, "days": days, "aqi": aqi, "alerts": alerts}

    try:
        api_data = await _call_weather_api("forecast.json", params)

        if "error" in api_data:
             error_info = api_data["error"]
             logger.error(f"WeatherAPI forecast error for '{location}': {error_info.get('message')}")
             return f"ERROR_MCP: WeatherAPI Error {error_info.get('code', '')} - {error_info.get('message', 'Unknown error')}"

        # --- Format the response ---
        output_lines = []
        loc_info = api_data.get("location", {})
        current = api_data.get("current", {})
        forecast = api_data.get("forecast", {}).get("forecastday", [])
        alerts_data = api_data.get("alerts", {}).get("alert", [])
        current_aqi = current.get("air_quality", {})

        output_lines.append(f"## Weather Report for {loc_info.get('name', location)}, {loc_info.get('region', '')} {loc_info.get('country', '')}")
        output_lines.append(f"_(Local Time: {loc_info.get('localtime', 'N/A')})_")

        # Current Conditions
        if current:
            temp_c = current.get('temp_c'); temp_f = current.get('temp_f')
            feels_c = current.get('feelslike_c'); feels_f = current.get('feelslike_f')
            output_lines.append("\n### Current Conditions:")
            output_lines.append(f"- **Temp:** {temp_c}°C / {temp_f}°F" if temp_c is not None else "- **Temp:** N/A")
            output_lines.append(f"- **Feels Like:** {feels_c}°C / {feels_f}°F" if feels_c is not None else "- **Feels Like:** N/A")
            output_lines.append(f"- **Condition:** {current.get('condition', {}).get('text', 'N/A')}")
            output_lines.append(f"- **Wind:** {current.get('wind_kph')} kph ({current.get('wind_mph')} mph) from {current.get('wind_dir', 'N/A')}")
            output_lines.append(f"- **Humidity:** {current.get('humidity')}%")
            output_lines.append(f"- **Precip:** {current.get('precip_mm')} mm")
            if current_aqi and aqi == 'yes':
                 output_lines.append(f"- **Air Quality (US EPA Index):** {current_aqi.get('us-epa-index', 'N/A')}")

        # Forecast
        if forecast:
            output_lines.append(f"\n### Forecast ({len(forecast)} days):")
            for day_data in forecast:
                day_info = day_data.get("day", {})
                astro_info = day_data.get("astro", {})
                min_c = day_info.get('mintemp_c'); max_c = day_info.get('maxtemp_c')
                min_f = day_info.get('mintemp_f'); max_f = day_info.get('maxtemp_f')
                output_lines.append(f"\n**{day_data.get('date', 'Unknown Date')}:**")
                output_lines.append(f"  - **Temp:** {min_c}°C - {max_c}°C / {min_f}°F - {max_f}°F" if min_c is not None else "  - **Temp:** N/A")
                output_lines.append(f"  - **Condition:** {day_info.get('condition', {}).get('text', 'N/A')}")
                output_lines.append(f"  - **Max Wind:** {day_info.get('maxwind_kph')} kph / {day_info.get('maxwind_mph')} mph")
                output_lines.append(f"  - **Total Precip:** {day_info.get('totalprecip_mm')} mm")
                output_lines.append(f"  - **Rain Chance:** {day_info.get('daily_chance_of_rain')}% | **Snow Chance:** {day_info.get('daily_chance_of_snow')}%")
                output_lines.append(f"  - **Sunrise:** {astro_info.get('sunrise', 'N/A')} | **Sunset:** {astro_info.get('sunset', 'N/A')}")

        # Alerts
        if alerts_data and alerts == 'yes':
            output_lines.append("\n### Active Weather Alerts:")
            for alert in alerts_data[:2]: # Show top 2 alerts
                 output_lines.append(f"- **{alert.get('event', 'Alert')}:** {alert.get('headline', 'No headline').strip()}")

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Unexpected error in get_weatherapi_forecast formatting: {e}", exc_info=True)
        return f"ERROR_MCP: Unexpected error processing forecast: {e}"


@mcp.tool()
async def get_future_weather(location: str, date: str) -> str:
    """
    Gets future weather for a location on a specific date using WeatherAPI.com.
    Date must be between 14 and 300 days from today.

    Args:
        location (string): City name, zip code, lat/lon, etc.
        date (string): The future date in YYYY-MM-DD format.

    Returns:
        string: Formatted future weather information or an error message string starting with ERROR_MCP:.
    """
    logger.info(f"Tool call: get_future_weather('{location}', date='{date}')")

    # Basic date format validation
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        # Optional: Add range validation (14-300 days from today)
        today = datetime.now().date()
        min_future_date = today + timedelta(days=14)
        max_future_date = today + timedelta(days=300)
        if not (min_future_date <= target_date <= max_future_date):
             return f"ERROR_MCP: Date must be between {min_future_date.isoformat()} and {max_future_date.isoformat()}."
    except ValueError:
        return "ERROR_MCP: Invalid date format. Please use YYYY-MM-DD."

    params = {"q": location, "dt": date}
    try:
        api_data = await _call_weather_api("future.json", params)

        if "error" in api_data:
            error_info = api_data["error"]
            logger.error(f"WeatherAPI future error for '{location}' on {date}: {error_info.get('message')}")
            return f"ERROR_MCP: WeatherAPI Error {error_info.get('code', '')} - {error_info.get('message', 'Unknown error')}"

        # --- Format the response ---
        output_lines = []
        loc_info = api_data.get("location", {})
        forecast = api_data.get("forecast", {}).get("forecastday", [])

        output_lines.append(f"## Future Weather for {loc_info.get('name', location)}, {loc_info.get('region', '')} {loc_info.get('country', '')}")

        if forecast:
            day_data = forecast[0] # Future API returns only one day
            day_info = day_data.get("day", {})
            astro_info = day_data.get("astro", {})
            min_c = day_info.get('mintemp_c'); max_c = day_info.get('maxtemp_c')
            min_f = day_info.get('mintemp_f'); max_f = day_info.get('maxtemp_f')
            avg_c = day_info.get('avgtemp_c'); avg_f = day_info.get('avgtemp_f')

            output_lines.append(f"\n**On {day_data.get('date', date)}:**")
            output_lines.append(f"  - **Avg Temp:** {avg_c}°C / {avg_f}°F" if avg_c is not None else "  - **Avg Temp:** N/A")
            output_lines.append(f"  - **Max Temp:** {max_c}°C / {max_f}°F" if max_c is not None else "  - **Max Temp:** N/A")
            output_lines.append(f"  - **Min Temp:** {min_c}°C / {min_f}°F" if min_c is not None else "  - **Min Temp:** N/A")
            output_lines.append(f"  - **Condition:** {day_info.get('condition', {}).get('text', 'N/A')}")
            output_lines.append(f"  - **Max Wind:** {day_info.get('maxwind_kph')} kph / {day_info.get('maxwind_mph')} mph")
            output_lines.append(f"  - **Total Precip:** {day_info.get('totalprecip_mm')} mm")
            output_lines.append(f"  - **Sunrise:** {astro_info.get('sunrise', 'N/A')} | **Sunset:** {astro_info.get('sunset', 'N/A')}")
            # Hourly data for future endpoint is less common but could be added if needed
        else:
             output_lines.append(f"\nNo future weather data found for {date}.")

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Unexpected error in get_future_weather formatting: {e}", exc_info=True)
        return f"ERROR_MCP: Unexpected error processing future weather: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    transport = "stdio"
    if len(sys.argv) > 1:
        transport = sys.argv[1]

    logger.info(f"Starting Custom WeatherAPI MCP server ('{mcp.name}') with transport: {transport}")
    if not WEATHER_API_KEY:
         logger.warning("WEATHER_API_KEY not set. Weather API calls will fail.")
         # sys.exit(1) # Uncomment to prevent server start without key

    try:
        mcp.run(transport=transport)
    except Exception as e:
        logger.critical(f"Error running Custom WeatherAPI MCP server: {str(e)}", exc_info=True)
        sys.exit(1)