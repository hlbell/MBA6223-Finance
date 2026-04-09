"""Geographic cost-of-living lookup using Zippopotam.us + Teleport API."""
from __future__ import annotations
import asyncio
import httpx

# Columbus, OH Teleport COL score used as the baseline index = 100
# Determined empirically; adjust if Teleport changes their scoring.
_BASELINE_COL_SCORE = 4.5  # Columbus OH typical Teleport COL score

# Fallback COL indices for common retirement cities (index 100 = baseline)
# Used when Teleport doesn't cover a particular urban area.
_FALLBACK_COL = {
    "phoenix": 97,
    "asheville": 94,
    "tucson": 88,
    "raleigh": 102,
    "san antonio": 91,
    "miami": 123,
    "new york": 175,
    "los angeles": 155,
    "chicago": 112,
    "houston": 96,
    "dallas": 101,
    "denver": 118,
    "seattle": 142,
    "boston": 160,
    "atlanta": 105,
    "tampa": 108,
    "orlando": 103,
    "nashville": 107,
    "charlotte": 104,
    "austin": 112,
    "columbus": 100,
    "cleveland": 90,
    "cincinnati": 92,
    "indianapolis": 91,
    "minneapolis": 110,
    "st. louis": 90,
    "kansas city": 93,
    "pittsburgh": 95,
    "detroit": 88,
    "memphis": 85,
    "new orleans": 98,
    "las vegas": 106,
    "salt lake city": 111,
    "portland": 130,
    "san francisco": 185,
    "san diego": 155,
    "san jose": 180,
}


async def get_city_from_zip(zip_code: str) -> tuple[str, str] | None:
    """Return (city_name, state_abbreviation) for a US zip code, or None."""
    zip_code = zip_code.strip()[:5]
    if len(zip_code) < 5:
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"https://api.zippopotam.us/us/{zip_code}")
            if resp.status_code != 200:
                return None
            data = resp.json()
            place = data["places"][0]
            return place["place name"], place["state abbreviation"]
    except Exception:
        return None


async def get_col_index(city: str, state: str) -> float | None:
    """
    Return a COL index (100 = Columbus OH baseline) for a city.
    Tries Teleport API first; falls back to hardcoded table; returns None if unknown.
    """
    # Try Teleport
    teleport_score = await _teleport_col_score(city, state)
    if teleport_score is not None:
        return round((teleport_score / _BASELINE_COL_SCORE) * 100, 1)

    # Fallback: check hardcoded table
    key = city.lower().strip()
    for fallback_city, idx in _FALLBACK_COL.items():
        if fallback_city in key or key in fallback_city:
            return float(idx)

    return None


async def _teleport_col_score(city: str, state: str) -> float | None:
    """
    4-step Teleport lookup:
      1. Search city name → get city item href
      2. Resolve city → urban area href
      3. Fetch urban area scores
      4. Extract 'Cost of Living' category score
    Returns raw Teleport score (0-10) or None.
    """
    search_query = f"{city}, {state}"
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
        ) as client:
            # Step 1: Search for city
            search_resp = await client.get(
                "https://api.teleport.org/api/cities/",
                params={"search": search_query, "limit": 5},
            )
            if search_resp.status_code != 200:
                return None

            search_data = search_resp.json()
            results = (
                search_data.get("_embedded", {})
                .get("city:search-results", [])
            )
            if not results:
                return None

            # Step 2: Get city detail URL
            city_href = results[0].get("_links", {}).get("city:item", {}).get("href")
            if not city_href:
                return None

            # Step 3: Fetch city and extract urban area link
            city_resp = await client.get(
                city_href,
                params={"embed": "city:urban_area"},
            )
            if city_resp.status_code != 200:
                return None

            city_data = city_resp.json()
            ua_embedded = (
                city_data.get("_embedded", {})
                .get("city:urban_area", {})
            )
            ua_href = (
                ua_embedded.get("_links", {})
                .get("self", {})
                .get("href")
            )
            if not ua_href:
                return None

            # Step 4: Fetch urban area scores
            scores_resp = await client.get(f"{ua_href}scores/")
            if scores_resp.status_code != 200:
                return None

            scores_data = scores_resp.json()
            categories = scores_data.get("categories", [])
            for cat in categories:
                if "cost of living" in cat.get("name", "").lower():
                    return cat.get("score_out_of_10")

            return None

    except Exception:
        return None


async def lookup_zip_full(zip_code: str) -> tuple[str, str, float | None] | None:
    """
    Full lookup: zip → (city, state, col_index).
    Returns None if zip is invalid.
    """
    result = await get_city_from_zip(zip_code)
    if result is None:
        return None
    city, state = result
    col_idx = await get_col_index(city, state)
    return city, state, col_idx


async def lookup_both_zips(
    current_zip: str, target_zip: str
) -> tuple[
    tuple[str, str, float | None] | None,
    tuple[str, str, float | None] | None,
]:
    """Concurrently look up both zip codes."""
    current_task = asyncio.create_task(lookup_zip_full(current_zip))
    target_task = asyncio.create_task(lookup_zip_full(target_zip))
    current, target = await asyncio.gather(current_task, target_task)
    return current, target
