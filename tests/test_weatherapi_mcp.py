# test_weatherapi_mcp.py

import asyncio
import os
import sys
import json
import traceback
from dotenv import load_dotenv
from typing import Dict, Any, Union # Added Union

# Add project root to path if necessary (adjust path as needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession, StdioServerParameters
    # Import helpers from frontend app to display results nicely
    # Ensure frontend/app.py is accessible or copy the function
    try:
        # Assuming app.py is in ../frontend relative to this script in src/
        frontend_path = os.path.join(project_root, 'frontend')
        if frontend_path not in sys.path:
             sys.path.insert(0, frontend_path)
        # Import BOTH functions needed
        from app import enhance_results, extract_mcp_result_text # <--- IMPORT ADDED HERE
    except ImportError:
        print("Warning: Could not import helpers from frontend/app.py. Raw JSON will be shown.")
        # Define dummy functions if import fails
        def enhance_results(result_text: str, result_type: str, location: str | None = None) -> str:
            return result_text
        # Added dummy for extract_mcp_result_text
        def extract_mcp_result_text(result) -> Union[str, None]:
             # Basic fallback: convert the whole result object to string
             return str(result)

except ImportError as e:
    print(f"Error importing MCP library or frontend helper: {e}")
    print("Please ensure mcp-sdk is installed and paths are correct.")
    sys.exit(1)

# Load environment variables (specifically WEATHER_API_KEY)
load_dotenv(override=True)

async def run_weatherapi_test():
    """Connects to the WeatherAPI MCP server and tests the get_weather tool."""
    print("Starting WeatherAPI MCP client test...")
    print("-" * 30)

    # --- Configuration for the npx server ---
    server_command = "npx"
    # Args match how app.py calls it
    args_prefix = ["-y", "@swonixs/weatherapi-mcp"]
    tool_name_to_call = "get_weather" # Tool name from the MCP server

    # Pass WEATHER_API_KEY via environment
    weather_api_key = os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        print("ERROR: WEATHER_API_KEY environment variable not set. Cannot run test.")
        return
    env_vars = {"WEATHER_API_KEY": weather_api_key}

    server_params = StdioServerParameters(
        command=server_command,
        args=args_prefix,
        env=env_vars # Pass API key environment variable
    )

    try:
        # Connect to the server using stdio_client (NO connect_timeout here)
        async with stdio_client(server_params) as client_tuple:
            if client_tuple is None:
                print("ERROR: Failed to establish MCP connection to WeatherAPI server.")
                return
            read, write = client_tuple

            # Create client session (NO session_timeout here)
            async with ClientSession(read, write) as session:
                print("Initializing MCP session with WeatherAPI server...")
                # Increased init timeout
                await asyncio.wait_for(session.initialize(), timeout=60.0)
                print("Session initialized successfully.")

                # --- Test Case 1: International Location ---
                test_params_1: Dict[str, Any] = {"location": "Mumbai, India"}
                print(f"\n[Test Case 1] Calling tool '{tool_name_to_call}' with params: {test_params_1}")
                try:
                    # Get the result object
                    result_1_obj = await asyncio.wait_for(session.call_tool(tool_name_to_call, test_params_1), timeout=60.0)
                    # *** Use extract_mcp_result_text ***
                    result_1_raw_text = extract_mcp_result_text(result_1_obj)

                    print("\n--- Result 1 (Raw Extracted Text) ---")
                    print(result_1_raw_text)

                    # Use enhance_results on the extracted text
                    formatted_result_1 = enhance_results(result_1_raw_text, "Weather (WeatherAPI)", test_params_1["location"])
                    print("\n--- Result 1 (Formatted) ---")
                    print(formatted_result_1)
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling tool for Mumbai: {tool_err}")
                     traceback.print_exc()


                # --- Test Case 2: US Location ---
                test_params_2: Dict[str, Any] = {"location": "New York"}
                print(f"\n[Test Case 2] Calling tool '{tool_name_to_call}' with params: {test_params_2}")
                try:
                    result_2_obj = await asyncio.wait_for(session.call_tool(tool_name_to_call, test_params_2), timeout=60.0)
                     # *** Use extract_mcp_result_text ***
                    result_2_raw_text = extract_mcp_result_text(result_2_obj)

                    print("\n--- Result 2 (Raw Extracted Text) ---")
                    print(result_2_raw_text)

                     # Use enhance_results on the extracted text
                    formatted_result_2 = enhance_results(result_2_raw_text, "Weather (WeatherAPI)", test_params_2["location"])
                    print("\n--- Result 2 (Formatted) ---")
                    print(formatted_result_2)
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling tool for New York: {tool_err}")
                     traceback.print_exc()

                # --- Test Case 3: Invalid Location (Optional) ---
                test_params_3: Dict[str, Any] = {"location": "InvalidPlaceNameXYZ"}
                print(f"\n[Test Case 3] Calling tool '{tool_name_to_call}' with params: {test_params_3}")
                try:
                    result_3_obj = await asyncio.wait_for(session.call_tool(tool_name_to_call, test_params_3), timeout=60.0)
                     # *** Use extract_mcp_result_text ***
                    result_3_raw_text = extract_mcp_result_text(result_3_obj)

                    print("\n--- Result 3 (Raw Extracted Text - Expect Error/No Data) ---")
                    print(result_3_raw_text)

                    # Use enhance_results on the extracted text (should handle errors)
                    formatted_result_3 = enhance_results(result_3_raw_text, "Weather (WeatherAPI)", test_params_3["location"])
                    print("\n--- Result 3 (Formatted) ---")
                    print(formatted_result_3)
                    print("-" * 26)
                except Exception as tool_err:
                     print(f"ERROR calling tool for InvalidPlaceNameXYZ (expected): {tool_err}")

    except asyncio.TimeoutError as te:
        print(f"\nERROR: Operation timed out: {te}. Check if npx command ran successfully and server started.")
    except ConnectionRefusedError:
        print("\nERROR: Connection refused. Could not connect to the npx server process.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure correct asyncio event loop policy for Windows if needed
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_weatherapi_test())