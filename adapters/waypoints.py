"""
ATHR NASA M20 Waypoints Adapter
Searches Perseverance rover waypoints and provides the complete
source waypoint collection for mission-route validation.

Source: NASA Mars 2020 MMGIS GeoJSON (updated as the rover drives)
Coordinates: source-provided latitude and longitude, east-positive.
Distance: spherical great-circle approximation.
"""

import json
import math
import threading
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

SOURCE_URL = (
    "https://mars.nasa.gov/mmgis-maps/"
    "M20/Layers/json/M20_waypoints.json"
)

MARS_RADIUS_KM = 3396.19
TIMEOUT = 30
CACHE_TTL_S = 6 * 3600          # the rover keeps driving, so refresh periodically
MAX_BYTES = 50 * 1024 * 1024    # refuse unexpectedly large responses

_lock = threading.Lock()
_cache = {
    "points": None,
    "total": 0,
    "fetched": 0.0,
    "stale": False,
}


def _download():
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "ATHR-Mars-Exploration-Platform/0.1"},
    )

    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read(MAX_BYTES + 1)
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not download NASA waypoints: {exc}") from exc

    if len(raw) > MAX_BYTES:
        raise RuntimeError("NASA waypoint file is larger than expected.")

    # json errors are ValueError subclasses; the API maps ValueError to
    # "bad input" (422), so convert them to RuntimeError (upstream problem).
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"NASA returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise RuntimeError("Unexpected NASA GeoJSON format.")

    features = payload.get("features")
    if not isinstance(features, list):
        raise RuntimeError("NASA GeoJSON has no valid features list.")

    # Parse once: keep only features with usable coordinates.
    points = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        try:
            lat = float(props["lat"])
            lon = float(props["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90:
            points.append((lat, lon, props))

    return points, len(features)


def load_waypoints():
    """Return (points, total_features, fetched_at_epoch, stale)."""
    with _lock:
        age = time.time() - _cache["fetched"]
        if _cache["points"] is None or age > CACHE_TTL_S:
            try:
                points, total = _download()
                _cache.update(points=points, total=total, fetched=time.time(), stale=False)
            except RuntimeError:
                if _cache["points"] is None:
                    raise                      # nothing to fall back on
                _cache["stale"] = True         # serve the old copy, flagged
        return _cache["points"], _cache["total"], _cache["fetched"], _cache["stale"]


def great_circle_distance_km(lat1, lon1, lat2, lon2):
    """Spherical great-circle distance in kilometers."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = lat2_rad - lat1_rad

    # Normalize longitude difference across the 0/360 seam.
    delta_lon = math.radians(((lon2 - lon1 + 180) % 360) - 180)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    a = min(1.0, max(0.0, a))

    return 2 * MARS_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearby_waypoints(
    latitude: float,
    longitude: float,
    radius_km: float,
    limit: int = 100,
):
    """Return NASA Perseverance waypoints within radius_km, nearest first.

    Source coordinates are kept as supplied; no convention conversion is applied.
    """
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not math.isfinite(longitude) or not 0 <= longitude < 360:
        raise ValueError("Longitude must be in the range 0 to less than 360.")

    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Radius must be greater than zero.")

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("Limit must be an integer from 1 to 500.")

    points, total, fetched, stale = load_waypoints()
    matches = []

    for point_lat, point_lon, props in points:
        distance = great_circle_distance_km(latitude, longitude, point_lat, point_lon)
        if distance <= radius_km:
            matches.append({
                "site": props.get("site"),
                "drive": props.get("drive"),
                "sol": props.get("sol"),
                "rmc": props.get("RMC"),
                "latitude": point_lat,
                "longitude": point_lon % 360,
                "distance_km": round(distance, 3),
                "elevation_geoid_m": props.get("elev_geoid"),
                "traverse_total_m": props.get("dist_total_m"),
                "note": props.get("Note"),
            })

    matches.sort(key=lambda item: item["distance_km"])

    return {
        "source": "NASA Mars 2020 Waypoints",
        "source_url": SOURCE_URL,
        "method": f"Spherical great-circle approximation (R = {MARS_RADIUS_KM} km)",
        "fetched_at": datetime.fromtimestamp(fetched, timezone.utc).isoformat(),
        "stale": stale,
        "total_source_features": total,
        "usable_features": len(points),
        "matching_count": len(matches),
        "returned_count": min(len(matches), limit),
        "radius_km": radius_km,
        "waypoints": matches[:limit],
    }


def get_mission_route():
    """Return the complete Perseverance route, ordered by NASA's cumulative
    traverse distance (dist_total_m), or an explicit rejection.

    The route is accepted (route_order_valid = True) only when ALL of these hold:
      - every source feature has usable coordinates,
      - every usable point has a finite, non-negative dist_total_m,
      - all dist_total_m values are unique (a strict, unambiguous order),
      - at least two points exist.

    Order is never inferred from proximity, Sol, Site or Drive.
    When validation fails, route_points is empty and route_status explains why,
    so a partial or ambiguous route is never drawn.
    """
    points, total, fetched, stale = load_waypoints()

    ordered = []
    missing_traverse = 0

    for point_lat, point_lon, props in points:
        raw = props.get("dist_total_m")

        try:
            traverse = float(raw)
        except (TypeError, ValueError):
            missing_traverse += 1
            continue

        if not math.isfinite(traverse) or traverse < 0:
            missing_traverse += 1
            continue

        ordered.append({
            "site": props.get("site"),
            "drive": props.get("drive"),
            "sol": props.get("sol"),
            "latitude": point_lat,
            "longitude": point_lon % 360,
            "traverse_total_m": traverse,
            "elevation_geoid_m": props.get("elev_geoid"),
        })

    ordered.sort(key=lambda item: item["traverse_total_m"])

    unusable_features = total - len(points)
    duplicate_traverse = sum(
        1
        for previous, current in zip(ordered, ordered[1:])
        if current["traverse_total_m"] == previous["traverse_total_m"]
    )

    problems = []
    if unusable_features > 0:
        problems.append(
            f"{unusable_features} source features have no usable coordinates"
        )
    if missing_traverse > 0:
        problems.append(
            f"{missing_traverse} points have no valid cumulative traverse distance"
        )
    if duplicate_traverse > 0:
        problems.append(
            f"{duplicate_traverse} points share a traverse distance with another point"
        )
    if len(ordered) < 2:
        problems.append("fewer than 2 ordered points")

    valid = not problems

    return {
        "source": "NASA Mars 2020 Waypoints",
        "source_url": SOURCE_URL,
        "fetched_at": datetime.fromtimestamp(fetched, timezone.utc).isoformat(),
        "stale": stale,
        "ordering": "traverse_total_m ascending",
        "total_source_features": total,
        "usable_features": len(points),
        "ordered_count": len(ordered) if valid else 0,
        "skipped_unordered_count": missing_traverse,
        "duplicate_traverse_count": duplicate_traverse,
        "unusable_coordinate_count": unusable_features,
        "route_order_valid": valid,
        "route_status": (
            "Route order verified from NASA cumulative traverse distance."
            if valid
            else "Route not drawn: " + "; ".join(problems) + "."
        ),
        "route_points": ordered if valid else [],
    }


if __name__ == "__main__":
    print("ATHR NASA M20 Waypoints Adapter Test")
    print("=" * 45)

    try:
        result = find_nearby_waypoints(latitude=18.4, longitude=77.5, radius_km=10)

        print("Source:", result["source"])
        print("Fetched at:", result["fetched_at"], "| stale:", result["stale"])
        print("Source features / usable:", result["total_source_features"], "/", result["usable_features"])
        print("Matching waypoints:", result["matching_count"])
        print("Returned:", result["returned_count"])

        for point in result["waypoints"][:5]:
            print(
                f"Sol {point['sol']} | Site {point['site']} | "
                f"Drive {point['drive']} | Distance {point['distance_km']} km"
            )

        route = get_mission_route()
        print("Route valid:", route["route_order_valid"], "|", route["route_status"])
        print("\n[PASS] NASA waypoint adapter completed.")

    except Exception as exc:
        print("\n[ERROR]", exc)