"""Local Google ADK agents and deterministic travel tools."""

from typing import Any

from google.adk.agents import LlmAgent

from groq_adk_model import GroqLlm


def search_places(city: str) -> dict[str, Any]:
    places = {
        "goa": [
            {"name": "Baga Beach", "category": "Beach", "distance": "12 km from Panaji"},
            {"name": "Fort Aguada", "category": "Historic site", "distance": "8 km from Panaji"},
            {"name": "Dudhsagar Falls", "category": "Nature", "distance": "60 km from Panaji"},
        ],
        "hyderabad": [
            {"name": "Charminar", "category": "Historic site", "distance": "City centre"},
            {"name": "Golconda Fort", "category": "Historic site", "distance": "11 km from centre"},
            {"name": "Hussain Sagar", "category": "Lake", "distance": "5 km from centre"},
        ],
    }
    return {"city": city, "places": places.get(city.lower(), places["goa"])}


def search_hotels(city: str, area: str = "") -> dict[str, Any]:
    return {
        "city": city,
        "area": area or "central area",
        "hotels": [
            {"name": f"{city.title()} Central Stay", "price": "₹2,500/night", "rating": 4.4},
            {"name": f"{city.title()} Comfort Suites", "price": "₹3,800/night", "rating": 4.6},
            {"name": f"{city.title()} Garden Hotel", "price": "₹4,600/night", "rating": 4.7},
        ],
    }


def search_restaurants(location: str, radius_km: float = 2.0) -> dict[str, Any]:
    location_key = location.lower()
    restaurants_by_city = {
        "goa": [
            {"name": "Sakana", "cuisine": "Goan seafood", "distance": "0.4 km", "rating": 4.6},
            {"name": "Mum's Kitchen", "cuisine": "Goan", "distance": "0.8 km", "rating": 4.5},
            {"name": "Bean Me Up", "cuisine": "Vegetarian", "distance": "1.3 km", "rating": 4.4},
        ],
        "hyderabad": [
            {"name": "Bawarchi", "cuisine": "Hyderabadi", "distance": "0.5 km", "rating": 4.5},
            {"name": "Shah Ghouse", "cuisine": "Biryani", "distance": "0.9 km", "rating": 4.4},
            {"name": "Chutneys", "cuisine": "South Indian", "distance": "1.4 km", "rating": 4.3},
        ],
    }
    city = next(
        (name for name in restaurants_by_city if name in location_key),
        "goa",
    )
    return {
        "location": location,
        "radius_km": radius_km,
        "city": city.title(),
        "restaurants": restaurants_by_city[city],
    }


def build_agents() -> LlmAgent:
    travel = LlmAgent(
        name="travel_agent",
        model=GroqLlm(),
        description="Finds attractions and builds travel itineraries.",
        instruction=(
            "You are the Travel Agent. Use search_places for attractions and itineraries. "
            "Return concrete demo results to the orchestrator."
        ),
        tools=[search_places],
    )
    hotel = LlmAgent(
        name="hotel_agent",
        model=GroqLlm(),
        description="Finds hotels and nearby restaurants.",
        instruction=(
            "You are the Hotel Agent. Use search_hotels to select a suitable hotel. "
            "If nearby restaurants are requested, also use search_restaurants and return "
            "one complete combined plan. State that the data is demo data."
        ),
        tools=[search_hotels, search_restaurants],
    )
    food = LlmAgent(
        name="food_agent",
        model=GroqLlm(),
        description="Finds restaurants near a supplied hotel or location.",
        instruction=(
            "You are the Food Agent. Use search_restaurants for the requested location "
            "and radius. Return concrete demo recommendations."
        ),
        tools=[search_restaurants],
    )
    return LlmAgent(
        name="orchestrator",
        model=GroqLlm(),
        instruction=(
            "You are the travel orchestrator. Read the user's request and delegate to "
            "the appropriate local specialist agent. Use travel_agent for attractions, "
            "hotel_agent for hotels, and food_agent for restaurants. For hotel plus "
            "nearby restaurant requests, delegate to hotel_agent, which can complete both "
            "parts. Synthesize a concise final answer and state that data is demo data."
        ),
        sub_agents=[travel, hotel, food],
    )
