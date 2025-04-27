# test_airbnb_mcp.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

async def test_airbnb_tools():
    """Test your custom Airbnb MCP server implementation"""
    print("Testing custom Airbnb MCP server...")
    
    # Connect to your custom implementation
    server_params = StdioServerParameters(
        command="python",
        args=["src/airbnb.py", "stdio"] # Use your actual path
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                # Test find_accommodations
                location = "Miami"
                params = {
                    "location": location,
                    "adults": 2,
                    "children": 0,
                    "check_in": "2025-05-01",
                    "check_out": "2025-05-05"
                }
                
                print(f"\nTesting find_accommodations for {location}...")
                result = await session.call_tool("find_accommodations", params)
                
                # Print result
                if hasattr(result, "content"):
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            text = content_item.text
                            print(text[:500] + "..." if len(text) > 500 else text)
                            
                            # Try to extract a listing ID for the next test
                            listing_id = None
                            lines = text.split('\n')
                            for line in lines:
                                if "Listing ID:" in line:
                                    listing_id = line.split("Listing ID:")[1].strip()
                                    break
                
                # Test get_listing_details if we found an ID
                if listing_id:
                    print(f"\nTesting get_listing_details for ID: {listing_id}")
                    details_result = await session.call_tool("get_listing_details", {"listing_id": listing_id})
                    
                    if hasattr(details_result, "content"):
                        for content_item in details_result.content:
                            if hasattr(content_item, "text"):
                                details_text = content_item.text
                                print(details_text[:500] + "..." if len(details_text) > 500 else details_text)
                else:
                    print("No listing ID found to test details")
                    
    except Exception as e:
        print(f"Error testing Airbnb MCP server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_airbnb_tools())