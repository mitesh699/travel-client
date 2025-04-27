from fastapi import APIRouter, HTTPException, Query
from main import app
import logging
from typing import Optional

# Create a router with a travel tag
router = APIRouter(prefix="/travel", tags=["Travel"])

logger = logging.getLogger(__name__)

@router.get("/debug/tools", summary="List available MCP tools")
async def list_tools():
    """List all available MCP tools across all servers."""
    tools = []
    
    # Weather server tools
    tools.append({
        "name": "get_weather_forecast",
        "description": "Get weather forecast for a location"
    })
    
    # Airbnb server tools
    tools.append({
        "name": "find_accommodations",
        "description": "Find Airbnb accommodations in a location"
    })
    
    # Search server tools
    tools.append({
        "name": "search_travel_info",
        "description": "Search for travel information"
    })
    tools.append({
        "name": "find_flights",
        "description": "Search for flights between two locations"
    })
    
    return {"available_tools": tools, "count": len(tools)}

@router.get("/weather", summary="Get weather information")
async def get_weather(location: str = Query(..., description="Location name")):
    """Get weather information for a location."""
    try:
        return {
            "location": f"{location}, USA",
            "current": {
                "temperature": "22",
                "unit": "C",
                "description": "Partly Cloudy",
                "wind": "10 km/h NW",
                "humidity": 65
            },
            "forecast": [
                {
                    "date": "2025-04-16",
                    "conditions": {
                        "temperature": "22",
                        "unit": "C",
                        "description": "Partly Cloudy",
                        "wind": "10 km/h NW",
                        "humidity": 65
                    }
                },
                {
                    "date": "2025-04-17",
                    "conditions": {
                        "temperature": "24",
                        "unit": "C",
                        "description": "Sunny",
                        "wind": "8 km/h W",
                        "humidity": 60
                    }
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error in weather endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/accommodations", summary="Get accommodation options")
async def get_accommodations(
    location: str = Query(..., description="Location name"),
    check_in: Optional[str] = Query(None, description="Check-in date (YYYY-MM-DD)"),
    check_out: Optional[str] = Query(None, description="Check-out date (YYYY-MM-DD)")
):
    """Get accommodation options for a location."""
    return [
        {
            "name": "Grand Hotel Central",
            "price": "$180 per night",
            "rating": "4.5/5",
            "amenities": ["WiFi", "Pool", "Breakfast"],
            "location": "City Center",
            "url": ""
        },
        {
            "name": "Cozy Downtown Apartment",
            "price": "$120 per night",
            "rating": "4.2/5",
            "amenities": ["WiFi", "Kitchen", "Washer"],
            "location": "Downtown",
            "url": ""
        }
    ]

@router.get("/flights", summary="Get flight options")
async def get_flights(
    origin: str = Query(..., description="Origin city or airport code"),
    destination: str = Query(..., description="Destination city or airport code"),
    date: Optional[str] = Query(None, description="Travel date (YYYY-MM-DD)")
):
    """Get flight options between two locations."""
    return [
        {
            "airline": "Delta Airlines",
            "flight_number": "DL 123",
            "departure": {
                "airport": "JFK",
                "time": "08:30 AM",
                "date": ""
            },
            "arrival": {
                "airport": "SFO",
                "time": "11:45 AM",
                "date": ""
            },
            "price": "$450",
            "duration": "",
            "stops": 0,
            "url": ""
        },
        {
            "airline": "British Airways",
            "flight_number": "BA 789",
            "departure": {
                "airport": "JFK",
                "time": "7:45 PM",
                "date": ""
            },
            "arrival": {
                "airport": "CDG",
                "time": "9:20 AM",
                "date": ""
            },
            "price": "$480",
            "duration": "",
            "stops": 1,
            "url": ""
        }
    ]

@router.get("/search", summary="Search travel information")
async def search_travel_info(
    query: str = Query(..., description="Search query"),
    category: str = Query("general", description="Search category")
):
    """Search for travel information."""
    return f"Search results for '{query}' in category '{category}' would be shown here."

# Include the router in the main application
app.include_router(router)
