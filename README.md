# Travel Client Project

This project is a travel planning application built with Python.

## Structure

-   `frontend/`: Contains the Streamlit frontend application (`app.py`).
-   `src/`: Contains the backend server (`server.py`) and various service integrations (Amadeus, Google Maps, Weather, etc.).
-   `main.py`: Entry point for the application (likely starts the server or frontend).
-   `old_scripts/`: Contains older or unused versions of application scripts.

## Setup

1.  Install dependencies: `uv pip install -r requirements.txt` (Assuming you have a requirements file, otherwise adjust)
2.  Set up environment variables in a `.env` file (see `.env.example` if one exists).
3.  Run the backend server: `python src/server.py`
4.  Run the frontend: `streamlit run frontend/app.py`

## Usage

[Add instructions on how to use the application here]