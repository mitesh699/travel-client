import uvicorn
import os
import logging
from main import app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variable configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

def run():
    """Start the FastAPI server with uvicorn"""
    logger.info(f"Starting FastAPI server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="debug")

if __name__ == "__main__":
    run()