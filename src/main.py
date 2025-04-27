# main.py (Regenerated)
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from mcp.server.sse import SseServerTransport
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import asyncio
import logging
import os
import json
from starlette.middleware.cors import CORSMiddleware

# Import MCP servers (excluding airbnb)
from weather import mcp as weather_mcp
# Removed: from airbnb import mcp as airbnb_mcp
from search import mcp as search_mcp
from apify import mcp as apify_mcp
from flight_service import mcp as amadeus_mcp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application with metadata
app = FastAPI(
    title="Travel Planner API Backend", # Updated title slightly
    description="A backend API managing MCP servers for a travel planning application.",
    version="0.1.1", # Incremented version
)

# Add CORS middleware to allow connections from Streamlit or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create SSE transport instance for handling server-sent events
# Note: If only stdio clients are used by app.py now, SSE might be unnecessary
# But we keep it for potential future use or if other clients need it.
sse = SseServerTransport("/mcp/messages")
logger.info("Initialized SSE transport at /mcp/messages")

# Handle SSE message posting (Keep if SSE is potentially needed)
@app.post("/mcp/messages", status_code=202, include_in_schema=False) # Hide from docs if internal
async def handle_post_message(request: Request):
    """Endpoint to handle MCP message posting via SSE transport"""
    logger.info("Received message post request via SSE")
    try:
        # Make sure sse object is properly initialized and handles request
        return await sse.handle_post_message(request.scope, request.receive, request._send)
    except Exception as e:
        logger.error(f"Error handling message post: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error handling message: {str(e)}")

@app.get("/mcp", tags=["MCP"], include_in_schema=False) # Hide from docs if internal/deprecated
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server (primarily for SSE clients).
    NOTE: If app.py only uses stdio_client, this endpoint might not be directly used by it.
    """
    # Get the server param from query string or default to weather
    server_type = request.query_params.get("server", "weather")
    logger.info(f"MCP SSE connection request: server_type={server_type}")

    # Select the appropriate MCP server
    selected_mcp = None
    if server_type == "weather":
        selected_mcp = weather_mcp
    # Removed: elif server_type == "airbnb": selected_mcp = airbnb_mcp
    elif server_type == "search":
        selected_mcp = search_mcp
    elif server_type == "apify":
        selected_mcp = apify_mcp
    elif server_type == "amadeus":
        selected_mcp = amadeus_mcp
    else:
        logger.warning(f"Requested unknown MCP server type via SSE: {server_type}")
        raise HTTPException(status_code=404, detail=f"Unknown MCP server type: {server_type}")

    # Prevent request timeout for SSE
    response = Response(media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no" # Useful for Nginx buffering issues

    # Connect via SSE
    try:
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            # Create initialization options
            init_options = selected_mcp._mcp_server.create_initialization_options()
            logger.debug(f"Initialization options for {server_type} (SSE): {init_options}")

            # Run the selected MCP server over SSE
            await selected_mcp._mcp_server.run(
                read_stream,
                write_stream,
                init_options,
            )
    except Exception as e:
        logger.error(f"Error in MCP SSE server connection for {server_type}: {str(e)}", exc_info=True)
        # Don't raise HTTPException within SSE stream, just log and exit
        logger.info(f"Closing SSE connection for {server_type} due to error.")


@app.get("/mcp/debug/status", tags=["MCP Debug"])
async def debug_mcp_status():
    """Return the status of MCP servers managed by this backend."""
    # Initialize status dictionary excluding airbnb
    status = {
        "weather": {"name": "weather", "initialized": False, "tools": []},
        "search": {"name": "search", "initialized": False, "tools": []},
        "apify": {"name": "apify", "initialized": False, "tools": []},
        "amadeus": {"name": "amadeus", "initialized": False, "tools": []}
    }

    servers_to_check = {
        "weather": weather_mcp,
        "search": search_mcp,
        "apify": apify_mcp,
        "amadeus": amadeus_mcp
    }

    tool_definitions = {
         "weather": [{"name": "get_weather_forecast", "description": "Get weather forecast for a location."}],
         "search": [
            {"name": "search_travel_info", "description": "Search for travel information."},
            {"name": "find_flights", "description": "Search for flights between two locations (fallback)."},
            {"name": "search_local_businesses", "description": "Search for local businesses and attractions."}
         ],
         "apify": [{"name": "get_tripadvisor_info", "description": "Get TripAdvisor info via Apify."}],
         "amadeus": [{"name": "get_flight_offers", "description": "Search for flight offers via Amadeus."}]
    }

    for server_name, mcp_instance in servers_to_check.items():
        try:
            # Check if the underlying MCP server object exists and has a name
            if hasattr(mcp_instance, '_mcp_server') and hasattr(mcp_instance._mcp_server, 'name'):
                status[server_name]["name"] = mcp_instance._mcp_server.name
                status[server_name]["initialized"] = True # Assume initialized if object exists
                # Use predefined tool definitions as inspecting live tools can be complex
                status[server_name]["tools"] = tool_definitions.get(server_name, [])
            else:
                 logger.warning(f"MCP instance for '{server_name}' seems malformed or not fully initialized.")

        except Exception as e:
            logger.error(f"Error getting status for '{server_name}' server: {str(e)}")
            status[server_name]["error"] = str(e) # Add error field

    # Removed airbnb status check block

    return status

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    # Return list excluding airbnb
    managed_servers = ["weather", "search", "apify", "amadeus"]
    return {"status": "ok", "managed_servers": managed_servers}

@app.get("/", tags=["General"], response_class=HTMLResponse)
async def homepage():
    """Root endpoint that returns a simple HTML welcome page"""
    # Removed Airbnb link from HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Travel Planner API Backend</title>
        <style>
            body { font-family: sans-serif; line-height: 1.6; padding: 2em; }
            h1, h2 { color: #333; }
            ul { list-style: none; padding: 0; }
            li { margin-bottom: 0.5em; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Travel Planner API Backend</h1>
        <p>This API backend manages and exposes several MCP servers for travel planning.</p>
        <p>The primary interaction may happen via STDIO clients directly from orchestrators (like Streamlit app), or potentially via SSE.</p>

        <h2>Managed MCP Servers</h2>
        <p>This backend manages the following servers (though they might be invoked directly via STDIO):</p>
        <ul>
            <li>Weather MCP Server (<code>weather</code>): Get weather forecasts.</li>
            <li>Search MCP Server (<code>search</code>): General search, fallback flights, local businesses.</li>
            <li>Apify MCP Server (<code>apify</code>): Get TripAdvisor information.</li>
            <li>Amadeus MCP Server (<code>amadeus</code>): Search for flights.</li>
        </ul>
        <p><i>Note: Airbnb MCP functionality is now handled directly by the client application.</i></p>

        <h2>Debugging & Monitoring</h2>
        <ul>
            <li><a href="/mcp/debug/status">MCP Server Status</a> - Check status of servers managed by this backend.</li>
            <li><a href="/health">Health Check</a> - Simple health check endpoint.</li>
        </ul>

        <p>API documentation (auto-generated by FastAPI): <a href="/docs">/docs</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Import routes at the end (if routes.py defines additional endpoints)
# If routes.py only contained mock endpoints, you might not need it anymore
try:
    import routes
    logger.info("Loaded additional routes from routes.py")
except ImportError:
    logger.info("No additional routes found in routes.py or routes.py not present.")