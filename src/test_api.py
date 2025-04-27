# test_api.py - Test the REST API endpoints
import httpx
import json

def test_weather_endpoint():
    """Test the weather endpoint"""
    try:
        response = httpx.get(
            "http://localhost:8000/travel/weather",
            params={"location": "San Francisco"},
            timeout=10
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_accommodations_endpoint():
    """Test the accommodations endpoint"""
    try:
        response = httpx.get(
            "http://localhost:8000/travel/accommodations",
            params={"location": "Barcelona"},
            timeout=10
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_flights_endpoint():
    """Test the flights endpoint"""
    try:
        response = httpx.get(
            "http://localhost:8000/travel/flights",
            params={"origin": "New York", "destination": "Paris"},
            timeout=10
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Testing weather endpoint...")
    test_weather_endpoint()
    
    print("\nTesting accommodations endpoint...")
    test_accommodations_endpoint()
    
    print("\nTesting flights endpoint...")
    test_flights_endpoint()