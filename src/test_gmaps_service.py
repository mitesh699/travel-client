# src/test_gmaps_service.py

import asyncio
import os
import sys
import traceback # Added for detailed error printing
from dotenv import load_dotenv
from typing import Dict, Any # Added for type hinting

# Adjust the path to ensure the MCP client library is accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession
    from mcp import StdioServerParameters
except ImportError as e:
     print(f"Error importing MCP library: {e}")
     print("Please ensure the MCP library is installed and the path is correct.")
     sys.exit(1)

# Load environment variables (needed for API key by the server)
load_dotenv()

async def run_test():
    """Connects to the gmaps_service MCP server and calls the maps_search_places tool."""
    print("Starting Google Maps MCP client test...")
    print("-" * 30)

    # --- Configuration ---
    server_script_name = "gmaps_service.py" # Ensure this points to your updated script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), server_script_name))
    python_executable = sys.executable # Use the same python that runs this script

    if not os.path.exists(script_path):
        print(f"ERROR: Cannot find server script at {script_path}")
        return

    # Define how to start the server
    server_params = StdioServerParameters(
        command=python_executable,
        args=[script_path, "stdio"] # Pass 'stdio' argument
    )

    try:
        # Connect to the server
        async with stdio_client(server_params) as client_tuple:
            if client_tuple is None:
                print("ERROR: Failed to establish MCP connection.")
                return
            read, write = client_tuple

            async with ClientSession(read, write) as session:
                print("Initializing MCP session...")
                await asyncio.wait_for(session.initialize(), timeout=30)
                print("Session initialized successfully.")

                # --- Define Test Parameters ---
                # CORRECTED TOOL NAME and removed 'count'
                tool_name = "maps_search_places"

                test_params_1: Dict[str, Any] = {
                    "query": "Coffee shops near Eiffel Tower Paris",
                    # 'radius' could be added if needed, requires 'location' dict
                }

                print(f"\n[Test Case 1] Calling tool '{tool_name}' with params: {test_params_1}")
                result_1 = await asyncio.wait_for(session.call_tool(tool_name, test_params_1), timeout=60)
                print(f"\n--- Result 1 ({tool_name} - Coffee) ---")
                print(result_1) # Print raw result
                print("--- End of Result 1 ---")

                # --- Test Case 2: Restaurants ---
                # CORRECTED TOOL NAME and removed 'count'
                test_params_2: Dict[str, Any] = {
                    "query": "Best New York City restaurants",
                }
                print(f"\n[Test Case 2] Calling tool '{tool_name}' with params: {test_params_2}")
                result_2 = await asyncio.wait_for(session.call_tool(tool_name, test_params_2), timeout=60)
                print(f"\n--- Result 2 ({tool_name} - Restaurants) ---") # Corrected label
                print(result_2)
                print("--- End of Result 2 ---")

                # --- Test Case 3: No Results Expected ---
                # CORRECTED TOOL NAME and removed 'count', removed 'location' string
                test_params_3: Dict[str, Any] = {
                    "query": "Xyzzyqux Sklorgnif Restaurant nowhere",
                    # 'location': {'latitude': 71.7, 'longitude': -42.6} # Example if lat/lon known
                }
                print(f"\n[Test Case 3] Calling tool '{tool_name}' with params: {test_params_3}")
                result_3 = await asyncio.wait_for(session.call_tool(tool_name, test_params_3), timeout=60)
                print(f"\n--- Result 3 ({tool_name} - Zero Results Expected) ---") # Corrected label
                print(result_3) # Should now return an empty list [] for zero results
                print("--- End of Result 3 ---")

    except asyncio.TimeoutError:
        print("\nERROR: Operation timed out. Check if the server script started correctly and isn't blocked.")
    except ConnectionRefusedError:
        print("\nERROR: Connection refused. Is the server script (`gmaps_service.py stdio`) running?")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_test())