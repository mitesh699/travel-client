# test_complete.py
import httpx
import json
import asyncio
import sys
import time

def test_api_server():
    """Test if the main API server is running"""
    try:
        response = httpx.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Main API server is running")
            return True
        else:
            print(f"❌ Error: API server returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: Could not connect to API server: {str(e)}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = httpx.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint is working!")
            data = response.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Servers: {', '.join(data.get('servers', []))}")
            return True
        else:
            print(f"❌ Error: Health endpoint returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: Could not connect to health endpoint: {str(e)}")
        return False

def test_mcp_debug_endpoint():
    """Test the MCP debug endpoint to verify all servers are initialized"""
    try:
        response = httpx.get("http://localhost:8000/mcp/debug/status", timeout=5)
        if response.status_code == 200:
            print("✅ MCP debug endpoint is working!")
            data = response.json()
            
            # Print server status
            for server_name, status in data.items():
                initialized = status.get("initialized", False)
                tools = status.get("tools", [])
                
                if initialized:
                    print(f"  ✅ {server_name}: Initialized with {len(tools)} tools")
                    for tool in tools:
                        print(f"     - {tool.get('name')}: {tool.get('description', '')[:50]}...")
                else:
                    print(f"  ❌ {server_name}: Not initialized")
            
            return data
        else:
            print(f"❌ Error: MCP debug endpoint returned status code {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: Could not connect to MCP debug endpoint: {str(e)}")
        return None

def test_travel_endpoints():
    """Test the REST API travel endpoints"""
    endpoints = {
        "weather": {"url": "http://localhost:8000/travel/weather", "params": {"location": "San Francisco"}},
        "accommodations": {"url": "http://localhost:8000/travel/accommodations", "params": {"location": "Barcelona"}},
        "flights": {"url": "http://localhost:8000/travel/flights", "params": {"origin": "New York", "destination": "Paris"}},
        "search": {"url": "http://localhost:8000/travel/search", "params": {"query": "best time to visit Tokyo"}}
    }
    
    results = {}
    
    for name, config in endpoints.items():
        print(f"Testing {name} endpoint...")
        try:
            response = httpx.get(config["url"], params=config["params"], timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {name} endpoint works!")
                results[name] = True
            else:
                print(f"  ❌ Error: {name} endpoint returned status code {response.status_code}")
                results[name] = False
        except Exception as e:
            print(f"  ❌ Error: Could not connect to {name} endpoint: {str(e)}")
            results[name] = False
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
    
    return results

def main():
    """Run all tests"""
    print("=== TESTING TRAVEL PLANNER API ===\n")
    
    # Test API server first
    if not test_api_server():
        print("\n❌ API server not running. Please start it first with: python server.py")
        return
    
    print("\n=== SERVER HEALTH CHECK ===\n")
    test_health_endpoint()
    
    print("\n=== MCP SERVER STATUS ===\n")
    test_mcp_debug_endpoint()
    
    print("\n=== TESTING REST API ENDPOINTS ===\n")
    test_travel_endpoints()
    
    print("\n=== TESTING COMPLETE ===\n")
    print("If all endpoints are working, you can now start the Streamlit app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()