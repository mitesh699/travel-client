# test_apify_mcp.py
import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

async def test_apify_mcp():
    print("Testing Apify MCP server...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["src/apify.py", "stdio"]
    )
    
    try:
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                tools_result = await session.list_tools()
                print(f"Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    print(f"- {tool.name}: {tool.description}")
                
                # Test TripAdvisor search
                print("\n=== TESTING TRIPADVISOR INFO ===")
                search_result = await session.call_tool(
                    "get_tripadvisor_info", 
                    {"location": "Chicago"}
                )
                
                # Format and print the result
                if hasattr(search_result, "content"):
                    for content in search_result.content:
                        if hasattr(content, "text"):
                            result_text = content.text
                            print(result_text[:1000] + "..." if len(result_text) > 1000 else result_text)
                else:
                    print(json.dumps(search_result, indent=2))
                    
    except Exception as e:
        print(f"Error testing Apify MCP server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_apify_mcp())