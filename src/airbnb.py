# airbnb.py
from mcp.server.fastmcp import FastMCP
import logging
import subprocess
import json
import asyncio
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("airbnb")
logger.info(f"Airbnb MCP server initialized with name: airbnb")

async def run_airbnb_cli_command(command_args: list[str]) -> tuple[str | None, str | None]:
    """Runs the npx command and captures stdout/stderr directly."""
    logger.debug(f"Running command: {' '.join(command_args)}")
    process = await asyncio.create_subprocess_exec(
        *command_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout_list = []
    stderr_list = []

    # Read stdout and stderr concurrently
    async def read_stream(stream, output_list):
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            output_list.append(decoded_line)
            # Optional: Log lines as they come in for debugging hangs
            # logger.debug(f"Stream Output: {decoded_line}")
        await asyncio.sleep(0.01) # Yield control briefly

    try:
        await asyncio.gather(
            read_stream(process.stdout, stdout_list),
            read_stream(process.stderr, stderr_list)
        )
        await process.wait() # Wait for the process to terminate

        stdout = "\n".join(stdout_list)
        stderr = "\n".join(stderr_list)

        if process.returncode != 0:
            logger.error(f"Airbnb CLI command failed with code {process.returncode}. Stderr: {stderr}")
            return None, stderr
        else:
            logger.debug(f"Airbnb CLI command succeeded. Stdout length: {len(stdout)}")
            # Log beginning of stdout for verification
            # logger.debug(f"Stdout sample: {stdout[:200]}...")
            return stdout, None

    except asyncio.TimeoutError:
         logger.error("Airbnb CLI command timed out.")
         process.terminate() # Terminate the hung process
         await process.wait()
         return None, "Command timed out"
    except Exception as e:
        logger.error(f"Exception running Airbnb CLI command: {e}")
        try:
             process.terminate()
             await process.wait()
        except ProcessLookupError:
             pass # Process might have already finished with error
        return None, f"Exception: {e}"


def format_price_label(label: str) -> str:
     """Cleans up the garbled price accessibility label."""
     if not label:
          return ""
     # Remove excessive whitespace and newlines
     cleaned = re.sub(r'\s+', ' ', label).strip()
     # Reconstruct price string if it was split (basic reconstruction)
     # This is fragile and depends on consistent patterns
     cleaned = re.sub(r'(\$\s?\d+)\s*f\s*o\s*r\s*(\d+)\s*n\s*i\s*g\s*h\s*t\s*s', r'\1 for \2 nights', cleaned, flags=re.IGNORECASE)
     cleaned = re.sub(r'o\s*r\s*i\s*g\s*i\s*n\s*a\s*l\s*l\s*y', r'originally', cleaned, flags=re.IGNORECASE)
     cleaned = re.sub(r'(\d+)\s*x\s*(\d+)\s*n\s*i\s*g\s*h\s*t\s*s\s*:', r'\1 x \2 nights:', cleaned, flags=re.IGNORECASE)

     # Fallback: return the cleaned version even if reconstruction failed
     return cleaned

def format_rating_label(label: str) -> str:
     """Cleans up the rating accessibility label."""
     if not label:
          return ""
     cleaned = re.sub(r'\s+', ' ', label).strip()
     # Example: "4.97 out of 5 average rating, 31 reviews" -> Keep as is after cleaning space
     return cleaned


@mcp.tool()
async def find_accommodations(
    location: str,
    check_in: str = None,
    check_out: str = None,
    adults: int = 2,
    children: int = 0,
    infants: int = 0,
    pets: int = 0,
    min_price: int = None,
    max_price: int = None
) -> str:
    """Find Airbnb accommodations in a location. (Uses @openbnb/mcp-server-airbnb CLI)"""
    logger.info(f"Tool called: find_accommodations for location: {location}")

    try:
        # Create parameters for the Airbnb search CLI
        cmd_args = ["npx", "-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt", "--cli", "search"]
        cmd_args.extend(["--location", location, "--adults", str(adults), "--children", str(children), "--infants", str(infants), "--pets", str(pets)])
        if check_in: cmd_args.extend(["--checkin", check_in])
        if check_out: cmd_args.extend(["--checkout", check_out])
        if min_price is not None: cmd_args.extend(["--minPrice", str(min_price)]) # Pass 0 if needed
        if max_price is not None: cmd_args.extend(["--maxPrice", str(max_price)])

        # Run command and capture stdout/stderr
        results_json, error_output = await run_airbnb_cli_command(cmd_args)

        if error_output or not results_json:
            logger.error(f"Error running Airbnb search via CLI: {error_output}")
            return f"Error searching Airbnb: {error_output or 'No output received'}"

        # Parse the JSON output from stdout
        try:
            results = json.loads(results_json)
            search_url = results.get("searchUrl", f"https://www.airbnb.com/s/{location.replace(' ', '-')}/homes") # Fallback URL
            search_results = results.get("searchResults", [])


            # Format the results into a nice text representation
            formatted_results = [f"Found {len(search_results)} accommodations in {location} ([View Search]({search_url}))"]

            for i, listing_data in enumerate(search_results[:5], 1):  # Limit to first 5 listings
                listing = listing_data.get("listing", {}) # Actual listing data might be nested
                name = listing.get("name", listing_data.get("title", "Unnamed listing")) # Use title if name missing
                # Price parsing
                price_text = "Unknown price"
                structured_price = listing_data.get("structuredDisplayPrice", {})
                if structured_price:
                     primary_line = structured_price.get("primaryLine", {})
                     price_label = primary_line.get("accessibilityLabel", "")
                     if price_label:
                          price_text = format_price_label(price_label) # Clean the label
                     else:
                          # Fallback if label missing but price exists
                          price_val = primary_line.get("price")
                          if price_val: price_text = price_val

                # Rating parsing
                rating_text = "No rating"
                rating_label = listing_data.get("avgRatingA11yLabel", "")
                if rating_label:
                     rating_text = format_rating_label(rating_label)
                elif listing.get("avgRating"): # Fallback
                     rating_text = f"{listing.get('avgRating')}/5"


                room_type = listing.get("roomType", "")
                listing_id = listing.get("id", listing_data.get("id")) # Get ID from listing or parent


                formatted_results.append(f"\n{i}. {name}")
                formatted_results.append(f"   Price: {price_text}")
                formatted_results.append(f"   Rating: {rating_text}")
                if room_type:
                    formatted_results.append(f"   Room Type: {room_type}")

                # Add link to the listing
                if listing_id:
                    formatted_results.append(f"   Listing ID: {listing_id}")
                    formatted_results.append(f"   [View on Airbnb](https://www.airbnb.com/rooms/{listing_id})")

            if not search_results:
                 formatted_results.append("\nNo specific listings found matching your criteria.")

            return "\n".join(formatted_results)

        except json.JSONDecodeError as json_err:
            logger.error(f"Error parsing results JSON from Airbnb CLI: {json_err}. Raw output: {results_json[:500]}...")
            return f"Error parsing Airbnb search results."
        except Exception as parse_err:
            logger.error(f"Error formatting Airbnb results: {parse_err}")
            # Return raw JSON if formatting fails but parsing worked
            if results_json:
                 return f"Found results, but error during formatting. Raw data:\n{results_json[:1000]}..."
            return "Error formatting Airbnb results."


    except Exception as e:
        error_msg = f"Error in find_accommodations: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error finding accommodations: {str(e)}"

@mcp.tool()
async def get_listing_details(listing_id: str) -> str:
    """Get detailed information about a specific Airbnb listing. (Uses @openbnb/mcp-server-airbnb CLI)"""
    logger.info(f"Tool called: get_listing_details for listing_id: {listing_id}")

    try:
        # Create parameters for the Airbnb listing details CLI
        cmd_args = ["npx", "-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt", "--cli", "details"]
        cmd_args.extend(["--id", listing_id])

        # Run command and capture stdout/stderr
        results_json, error_output = await run_airbnb_cli_command(cmd_args)

        if error_output or not results_json:
            logger.error(f"Error running Airbnb listing details via CLI: {error_output}")
            return f"Error getting listing details: {error_output or 'No output received'}"


        # Parse the JSON output from stdout
        try:
            result = json.loads(results_json)

            # Format the results into a nice text representation
            name = result.get("name", "Unnamed listing")
            price_rate = result.get("price", {}).get("rate", "Unknown price")
            rating = result.get("rating", "No rating")
            reviews_count = result.get("reviewsCount", "0")
            location = result.get("location", {}).get("address", "Unknown location")
            host_name = result.get("host", {}).get("name", "Unknown host")

            # Format amenities
            amenities = result.get("amenities", [])
            amenities_text = ", ".join(amenities[:10])
            if len(amenities) > 10:
                amenities_text += f", and {len(amenities) - 10} more"

            # Description
            description = result.get("description", "No description available.")
            description_preview = description[:500] + ('...' if len(description) > 500 else '')


            # House rules
            house_rules = result.get("houseRules", [])
            house_rules_text = ""
            if house_rules:
                house_rules_text = "\n   - ".join(house_rules[:5])
                if len(house_rules) > 5:
                    house_rules_text += f"\n   - and {len(house_rules) - 5} more rules"
            else:
                 house_rules_text = "No specific house rules provided."

            # Build the formatted response
            formatted_result = [
                f"## Details for '{name}' (ID: {listing_id})",
                f"\n**Price:** {price_rate}",
                f"**Rating:** {rating}/5 ({reviews_count} reviews)",
                f"**Location:** {location}",
                f"**Host:** {host_name}",
                f"\n**Description:**\n{description_preview}",
                f"\n**Top Amenities:**\n{amenities_text}",
                f"\n**House Rules:**\n{house_rules_text}",
                f"\n[View Full Listing on Airbnb](https://www.airbnb.com/rooms/{listing_id})"
            ]

            return "\n".join(formatted_result)

        except json.JSONDecodeError as json_err:
            logger.error(f"Error parsing results JSON from Airbnb details CLI: {json_err}. Raw output: {results_json[:500]}...")
            return f"Error parsing Airbnb listing details."
        except Exception as parse_err:
             logger.error(f"Error formatting listing details: {parse_err}")
             if results_json:
                  return f"Found details, but error during formatting. Raw data:\n{results_json[:1000]}..."
             return "Error formatting listing details."

    except Exception as e:
        error_msg = f"Error in get_listing_details: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error getting listing details: {str(e)}"

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    transport = "stdio" # Default to stdio for direct testing simplicity

    if len(sys.argv) > 1:
        transport = sys.argv[1]

    logger.info(f"Starting airbnb MCP server with transport: {transport}")

    # Run the server with the specified transport
    try:
        if transport == "sse":
            async def run_server():
                await mcp.run_sse_async()
            asyncio.run(run_server())
        else:
            # Assuming FastMCP supports run with stdio
            mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")