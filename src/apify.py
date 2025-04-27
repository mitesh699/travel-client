from mcp.server.fastmcp import FastMCP
import logging
import asyncio
import os
import json
from apify_client import ApifyClient
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("apify")
logger.info(f"Apify MCP server initialized with name: apify")

# Get Apify token directly
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "apify_api_QlaJibd3nbCgWZNRReVF5XrRJpP5wZ03NTIO")
logger.info(f"APIFY_TOKEN set: {'Yes' if APIFY_TOKEN else 'No'}")

# Initialize Apify client
apify_client = ApifyClient(APIFY_TOKEN)

@mcp.tool()
async def get_tripadvisor_info(location: str) -> str:
    """Get TripAdvisor information about restaurants, hotels, and attractions in a location.
    
    Args:
        location: Name of the location (e.g., "Chicago")
    """
    logger.info(f"Tool called: get_tripadvisor_info for location: {location}")
    
    try:
        # Prepare the Actor input
        run_input = {
            "currency": "USD",
            "includeAiReviewsSummary": True,
            "includeAttractions": True,
            "includeHotels": True,
            "includeNearbyResults": False,
            "includePriceOffers": False,
            "includeRestaurants": True,
            "includeTags": True,
            "includeVacationRentals": False,
            "language": "en",
            "locationFullName": location,
            "maxItemsPerQuery": 10
        }
        
        # Run the Actor and wait for it to finish
        logger.info(f"Running Apify actor: maxcopell/tripadvisor for {location}")
        run = apify_client.actor("maxcopell/tripadvisor").call(run_input=run_input)
        
        # Check if run was successful
        if not run or not run.get("defaultDatasetId"):
            return f"Error running TripAdvisor actor: No dataset returned"
        
        # Fetch results from the dataset
        dataset_items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
        
        # Format the results
        if not dataset_items:
            return f"No TripAdvisor results found for {location}"
        
        result_data = dataset_items[0]  # Get the first item which should contain all results
        formatted_output = [f"# TripAdvisor information for {location}"]
        
        # Process restaurants
        if "restaurants" in result_data and result_data["restaurants"]:
            restaurants = result_data["restaurants"]
            formatted_output.append("\n## Top Restaurants:")
            
            for i, restaurant in enumerate(restaurants[:5], 1):
                name = restaurant.get("name", "Unnamed Restaurant")
                rating = restaurant.get("rating", "No rating")
                reviews = restaurant.get("reviewsCount", "0")
                cuisine = restaurant.get("cuisine", [])
                cuisine_str = ", ".join(cuisine) if cuisine else "Various cuisines"
                price_level = restaurant.get("priceLevel", "")
                url = restaurant.get("url", "")
                
                # Add AI review summary if available
                ai_review = restaurant.get("aiReviewSummary", "")
                
                restaurant_info = [
                    f"{i}. **{name}** - {rating}/5 ({reviews} reviews)",
                    f"   Cuisine: {cuisine_str}",
                    f"   Price Level: {price_level}",
                ]
                
                if url:
                    restaurant_info.append(f"   [View on TripAdvisor]({url})")
                
                if ai_review:
                    restaurant_info.append(f"   AI Review Summary: {ai_review}")
                
                formatted_output.append("\n".join(restaurant_info))
        
        # Process hotels
        if "hotels" in result_data and result_data["hotels"]:
            hotels = result_data["hotels"]
            formatted_output.append("\n## Top Hotels:")
            
            for i, hotel in enumerate(hotels[:5], 1):
                name = hotel.get("name", "Unnamed Hotel")
                rating = hotel.get("rating", "No rating")
                reviews = hotel.get("reviewsCount", "0")
                price = hotel.get("price", "")
                url = hotel.get("url", "")
                
                # Add AI review summary if available
                ai_review = hotel.get("aiReviewSummary", "")
                
                hotel_info = [
                    f"{i}. **{name}** - {rating}/5 ({reviews} reviews)",
                    f"   Price: {price}",
                ]
                
                if url:
                    hotel_info.append(f"   [View on TripAdvisor]({url})")
                
                if ai_review:
                    hotel_info.append(f"   AI Review Summary: {ai_review}")
                
                formatted_output.append("\n".join(hotel_info))
        
        # Process attractions
        if "attractions" in result_data and result_data["attractions"]:
            attractions = result_data["attractions"]
            formatted_output.append("\n## Top Attractions:")
            
            for i, attraction in enumerate(attractions[:5], 1):
                name = attraction.get("name", "Unnamed Attraction")
                rating = attraction.get("rating", "No rating")
                reviews = attraction.get("reviewsCount", "0")
                category = attraction.get("category", "")
                url = attraction.get("url", "")
                
                # Add AI review summary if available
                ai_review = attraction.get("aiReviewSummary", "")
                
                attraction_info = [
                    f"{i}. **{name}** - {rating}/5 ({reviews} reviews)",
                    f"   Category: {category}",
                ]
                
                if url:
                    attraction_info.append(f"   [View on TripAdvisor]({url})")
                
                if ai_review:
                    attraction_info.append(f"   AI Review Summary: {ai_review}")
                
                formatted_output.append("\n".join(attraction_info))
        
        return "\n\n".join(formatted_output)
        
    except Exception as e:
        error_msg = f"Error in get_tripadvisor_info: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error getting TripAdvisor information: {str(e)}"

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    transport = "sse"
    
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    
    logger.info(f"Starting Apify MCP server with transport: {transport}")
    
    # Run the server with the specified transport
    try:
        if transport == "sse":
            async def run_server():
                await mcp.run_sse_async()
                
            asyncio.run(run_server())
        else:
            mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")