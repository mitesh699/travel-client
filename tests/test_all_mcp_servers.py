# test_all_mcp_servers.py
import httpx
import json

def test_mcp_debug_endpoint():
    """Test the MCP debug endpoint to verify all servers are initialized"""
    print("Testing MCP debug endpoint...")
    
    try:
        response = httpx.get("http://localhost:8000/mcp/debug/status")
        if response.status_code == 200:
            print("MCP debug endpoint is working!")
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_mcp_debug_endpoint()