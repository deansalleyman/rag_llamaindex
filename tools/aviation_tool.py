import os
import datetime

import requests
from dotenv import load_dotenv

load_dotenv()
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

AVIATIONSTACK_URL = "https://api.aviationstack.com/v1/flights"


def _error(message: str) -> str:
    """Format an error as a tool result the agent should relay to the user.

    We return this string (rather than raising) so the agent reports the real
    problem instead of silently falling back to made-up flight data.
    """
    return (
        f"ERROR: {message} "
        "Report this problem to the user. Do NOT invent, guess, or use placeholder "
        "flight data."
    )


def get_flights(origin: str, destination: str, flight_date: str = None) -> str:
    """A tool that looks up flights for a route using the AviationStack API.

    Args:
        origin: Departure airport IATA code, e.g. 'JFK'.
        destination: Arrival airport IATA code, e.g. 'LAX'.
        flight_date: Flight date as 'YYYY-MM-DD'. Defaults to today's date.
    """
    if not AVIATIONSTACK_API_KEY:
        return _error(
            "Missing AVIATIONSTACK_API_KEY. Add AVIATIONSTACK_API_KEY=... to your .env file."
        )

    # Default to today's date if the caller doesn't pass one
    if not flight_date:
        flight_date = datetime.date.today().isoformat()

    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "dep_iata": origin,
        "arr_iata": destination,
        "flight_date": flight_date,
    }

    try:
        resp = requests.get(AVIATIONSTACK_URL, params=params, timeout=30)
    except requests.RequestException as exc:
        return _error(f"Could not reach AviationStack: {exc}")

    # Surface API errors (bad key, plan limits, etc.) with the real message
    if not resp.ok:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text[:500]
        return _error(f"AviationStack API error (HTTP {resp.status_code}): {detail}")

    try:
        body = resp.json()
    except ValueError:
        return _error(f"AviationStack returned a non-JSON response: {resp.text[:300]}")

    # A successful HTTP call can still carry an API-level error payload
    if isinstance(body, dict) and body.get("error"):
        return _error(f"AviationStack API error: {body['error']}")

    flights = body.get("data", []) if isinstance(body, dict) else []
    if not flights:
        return f"No flights found from {origin} to {destination} on {flight_date}."

    lines = [f"Flights from {origin} to {destination} on {flight_date}:"]
    for flight in flights:
        airline = (flight.get("airline") or {}).get("name", "Unknown airline")
        number = (flight.get("flight") or {}).get("iata", "?")
        departure = flight.get("departure") or {}
        arrival = flight.get("arrival") or {}
        status = flight.get("flight_status", "unknown")
        lines.append(
            f"- {airline} {number}: "
            f"{departure.get('airport', origin)} ({departure.get('iata', origin)}) -> "
            f"{arrival.get('airport', destination)} ({arrival.get('iata', destination)}) "
            f"[{status}]"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_flights("JFK", "LAX"))
