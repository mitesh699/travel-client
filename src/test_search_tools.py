# test_search_tools.py
import asyncio
from search import search_travel_info, find_flights, search_local_businesses

async def test_search_tools():
    # Test travel info search
    travel_result = await search_travel_info("best time to visit Paris")
    print("=== TRAVEL INFO SEARCH ===")
    print(travel_result)
    print("\n")
    
    # Test flight search
    flight_result = await find_flights("New York", "Paris")
    print("=== FLIGHT SEARCH ===")
    print(flight_result)
    print("\n")
    
    # Test local business search
    local_result = await search_local_businesses("restaurants in San Francisco")
    print("=== LOCAL BUSINESS SEARCH ===")
    print(local_result)

if __name__ == "__main__":
    asyncio.run(test_search_tools())