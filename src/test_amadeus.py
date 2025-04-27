import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from datetime import datetime, timedelta

async def test_amadeus_mcp():
    """Test the Amadeus MCP server"""
    try:
        # Set up parameters for connecting to the server
        server_params = StdioServerParameters(
            command="python",
            args=["src/flight_service.py", "stdio"]
        )
        
        print("Connecting to Amadeus MCP server...")
        
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("Session initialized")
                
                # List available tools
                tools_result = await session.list_tools()
                print(f"Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    print(f"- {tool.name}: {tool.description}")
                
                # Calculate dates for a near-future trip
                departure_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
                return_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
                
                # Test flight search
                print("\n=== TESTING FLIGHT SEARCH ===")
                search_result = await session.call_tool(
                    "get_flight_offers", 
                    {
                        "origin": "JFK",
                        "destination": "ORD",
                        "departure_date": departure_date,
                        "return_date": return_date
                    }
                )
                
                # Print the result
                if hasattr(search_result, "content"):
                    for content in search_result.content:
                        if hasattr(content, "text"):
                            result_text = content.text
                            print(result_text[:1000] + "..." if len(result_text) > 1000 else result_text)
                else:
                    print("Unexpected result format:", search_result)
                
    except Exception as e:
        print(f"Error testing Amadeus MCP server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_amadeus_mcp())