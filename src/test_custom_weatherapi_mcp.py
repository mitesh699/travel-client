# test_custom_weatherapi_mcp.py

import asyncio
import os
import sys
import json
import traceback
from dotenv import load_dotenv
from typing import Dict, Any, Union
from datetime import datetime, timedelta

# Add project root to path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession, StdioServerParameters
    # Import helper from frontend app - adjust path if needed
    try:
        frontend_path = os.path.join(project_root, 'frontend')
        if frontend_path not in sys.path:
             sys.path.insert(0, frontend_path)
        # Only need extract_mcp_result_text as formatting is done server-side now
        from app import extract_mcp_result_text
    except ImportError:
        print("Warning: Could not import extract_mcp_result_text from frontend/app.py.")
        # Define dummy function
        def extract_mcp_result_text(result) -> Union[str, None]:
             # Basic fallback: convert the whole result object to string
             # This might not be ideal if the server returns complex objects
             if hasattr(result, 'content') and result.content and hasattr(result.content[0], 'text'):
                 return result.content[0].text
             return str(result)

except ImportError as e:
    print(f"Error importing MCP library or frontend helper: {e}")
    print("Please ensure mcp-sdk is installed and paths are correct.")
    sys.exit(1)

# Load environment variables (needed for API key by the server)
load_dotenv(override=True)

async def run_custom_weather_test():
    """Connects to the custom WeatherAPI MCP server and tests its tools."""
    print("Starting Custom WeatherAPI MCP client test...")
    print("-" * 30)

    # --- Configuration for the custom Python server ---
    server_script_name = "custom_weatherapi_mcp.py" # The new server script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), server_script_name))
    python_executable = sys.executable # Use the current python environment

    if not os.path.exists(script_path):
        print(f"ERROR: Cannot find server script at {script_path}")
        return
    if not os.getenv("WEATHER_API_KEY"):
         print("ERROR: WEATHER_API_KEY environment variable not set. Cannot run test.")
         return

    server_params = StdioServerParameters(
        command=python_executable,
        args=[script_path, "stdio"] # Run the python script via stdio
    )

    try:
        # Connect to the server
        async with stdio_client(server_params) as client_tuple:
            if client_tuple is None:
                print("ERROR: Failed to establish MCP connection to custom WeatherAPI server.")
                return
            read, write = client_tuple

            async with ClientSession(read, write) as session:
                print("Initializing MCP session with custom WeatherAPI server...")
                await asyncio.wait_for(session.initialize(), timeout=30.0)
                print("Session initialized successfully.")

                # --- Test Case 1: Forecast Tool ---
                forecast_tool = "get_weatherapi_forecast"
                forecast_params = {"location": "London", "days": 4, "aqi": "yes", "alerts": "no"} # Test params
                print(f"\n[Test Case 1] Calling tool '{forecast_tool}' with params: {forecast_params}")
                try:
                    result_1_obj = await asyncio.wait_for(session.call_tool(forecast_tool, forecast_params), timeout=60.0)
                    # Extract the formatted string returned by the tool
                    result_1_text = extract_mcp_result_text(result_1_obj)
                    print(f"\n--- Result 1 ({forecast_tool}) ---")
                    print(result_1_text)
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling {forecast_tool}: {tool_err}")
                     traceback.print_exc()

                # --- Test Case 2: Future Weather Tool ---
                future_tool = "get_future_weather"
                # Calculate a date 20 days in the future dynamically
                future_date = (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
                future_params = {"location": "Tokyo", "date": future_date}
                print(f"\n[Test Case 2] Calling tool '{future_tool}' with params: {future_params}")
                try:
                    result_2_obj = await asyncio.wait_for(session.call_tool(future_tool, future_params), timeout=60.0)
                    # Extract the formatted string returned by the tool
                    result_2_text = extract_mcp_result_text(result_2_obj)
                    print(f"\n--- Result 2 ({future_tool}) ---")
                    print(result_2_text)
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling {future_tool}: {tool_err}")
                     traceback.print_exc()

                # --- Test Case 3: Future Weather Tool (Invalid Date Range) ---
                invalid_future_params = {"location": "Paris", "date": "2020-01-01"} # Date too far in past
                print(f"\n[Test Case 3] Calling tool '{future_tool}' with params: {invalid_future_params}")
                try:
                    result_3_obj = await asyncio.wait_for(session.call_tool(future_tool, invalid_future_params), timeout=60.0)
                    # Extract the formatted string returned by the tool
                    result_3_text = extract_mcp_result_text(result_3_obj)
                    print(f"\n--- Result 3 ({future_tool} - Invalid Date Expected) ---")
                    print(result_3_text) # Should show an ERROR_MCP message from the server
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling {future_tool} with invalid date (expected if server handles it): {tool_err}")

                # --- Test Case 4: Forecast Tool (Location Not Found) ---
                forecast_params_invalid = {"location": "InvalidPlaceNameXYZ123"}
                print(f"\n[Test Case 4] Calling tool '{forecast_tool}' with params: {forecast_params_invalid}")
                try:
                    result_4_obj = await asyncio.wait_for(session.call_tool(forecast_tool, forecast_params_invalid), timeout=60.0)
                    # Extract the formatted string returned by the tool
                    result_4_text = extract_mcp_result_text(result_4_obj)
                    print(f"\n--- Result 4 ({forecast_tool} - Invalid Location Expected) ---")
                    print(result_4_text) # Should show an ERROR_MCP message
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling {forecast_tool} with invalid location (expected if server handles it): {tool_err}")


    except asyncio.TimeoutError as te:
        print(f"\nERROR: Operation timed out: {te}. Check if the server script started correctly.")
    except ConnectionRefusedError:
        print("\nERROR: Connection refused. Is the server script (`custom_weatherapi_mcp.py stdio`) running?")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_custom_weather_test())