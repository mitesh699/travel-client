# test_nws_weather.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_weather_tools():
    """Test the updated Weather MCP server tools"""
    print("Testing Weather MCP server...")

    # Create server parameters for connecting to the Weather MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["src/weather.py", "stdio"] # Ensure this path is correct
    )

    try:
        # Connect to the server using stdio
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Connected to Weather MCP server")

                # Initialize the session
                await session.initialize()
                print("Session initialized")

                # List available tools
                tools_result = await session.list_tools()
                print(f"Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    # Use getattr to safely access description and format it
                    description = getattr(tool, 'description', 'No description available.')
                    # Simple formatting for display
                    formatted_desc = '\n    '.join(description.split('\n'))
                    print(f"- {tool.name}: \n    {formatted_desc}\n")

                # --- Removed the get_alerts test section ---

                # Test get_weather_forecast
                print("\n=== TESTING WEATHER FORECAST ===")

                # Test for a few different locations
                locations = ["San Francisco", "New York", "Chicago"]

                for location in locations:
                    print(f"\nForecast for {location}:")
                    # Call get_weather_forecast tool
                    forecast_result = await session.call_tool(
                        "get_weather_forecast",
                        {"location": location}
                    )

                    # Format and print the result
                    if hasattr(forecast_result, "content"):
                        # Iterate through content items if it's a list
                        content_items = forecast_result.content if isinstance(forecast_result.content, list) else [forecast_result.content]
                        for content in content_items:
                            if hasattr(content, "text"):
                                print(content.text)
                            else:
                                # Print the content item directly if no text attribute
                                print(content)
                    else:
                        # Print the result directly if no content attribute
                        print(forecast_result)

    except Exception as e:
        print(f"Error testing Weather MCP server: {str(e)}")
        # Optionally print traceback for more debugging info
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather_tools())