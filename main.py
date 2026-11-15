import json
import math
import os
import re
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from pathlib import Path
from fastapi.responses import FileResponse
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, APIRouter, HTTPException, Query
from adapters.mola import (
    read_elevation,
    read_slope,
    InsufficientDataError,
)
from adapters.waypoints import (
    find_nearby_waypoints,
    load_waypoints,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from assistant import build_assistant_router  # M7

VERSION = "0.1.0"
MILESTONE = "M7"

# Comma-separated list, e.g. ATHR_CORS_ORIGINS="https://athr.example.com"
DEFAULT_ORIGINS = "http://127.0.0.1:5500,http://localhost:5500"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ATHR_CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]

TREK_SEARCH_URL = "https://trek.nasa.gov/mars/TrekServices/ws/index/eq/searchItems"
TREK_PROJECTION = "urn:ogc:def:crs:EPSG::104905"
TREK_TIMEOUT_S = 20

app = FastAPI(
    title="ATHR API",
    description="Mars Exploration Platform Backend",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "HEAD", "POST"],   # POST is needed for /discover
    allow_headers=["Content-Type"],
    # allow_credentials omitted: the API uses no cookies, so it is not needed.
)

api = APIRouter(prefix="/api")


class DiscoverRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=0, lt=360)
    radius_km: float = Field(gt=0)
    limit: int = Field(default=100, ge=1, le=500)


class MissionRouteRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=0, lt=360)
    radius_km: float = Field(default=25, gt=0, le=20000)

class Health(BaseModel):
    status: str
    service: str
    version: str
    milestone: str


@api.post("/discover", tags=["discovery"])
def discover(request: DiscoverRequest):
    try:
        waypoints = find_nearby_waypoints(
            latitude=request.lat,
            longitude=request.lon,
            radius_km=request.radius_km,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Waypoint discovery failed: {exc}",
        )

    return {
        "status": "ok",
        "location": {
            "latitude": request.lat,
            "longitude": request.lon,
            "radius_km": request.radius_km,
        },
        "mission_waypoints": waypoints,
    }


# ============================================================
# M6 — NASA Perseverance Mission Route
# Ordering: Sol -> Site -> Drive
# Geographic gap threshold: 200 meters
# ============================================================

M6_GAP_THRESHOLD_M = 200.0
MARS_RADIUS_M = 3_396_190.0


def _m6_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _m6_distance_m(a, b):
    """Spherical great-circle distance in meters between two route points."""
    lat1 = math.radians(a["latitude"])
    lat2 = math.radians(b["latitude"])
    dlat = lat2 - lat1

    lon1 = math.radians(a["longitude"])
    lon2 = math.radians(b["longitude"])
    dlon = lon2 - lon1

    # Shortest longitude difference (handles the 0/360 seam).
    dlon = (dlon + math.pi) % (2 * math.pi) - math.pi

    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    h = min(1.0, max(0.0, h))

    return 2 * MARS_RADIUS_M * math.asin(math.sqrt(h))


def _m6_build_route():
    """
    Build the Perseverance route from the complete NASA waypoint collection.

    - Order: Sol, then Site, then Drive (ascending). dist_total_m is NOT used
      for ordering.
    - Every usable point is kept (no point is dropped, zero-distance points
      included).
    - The route is accepted only if every point has numeric Sol/Site/Drive and
      no two points share the same (Sol, Site, Drive) key.
    - route_points: all ordered points.
      route_segments: consecutive runs of points; a new segment starts
      wherever two consecutive points are more than M6_GAP_THRESHOLD_M apart.
    """
    points, total, fetched, stale = load_waypoints()

    keyed = []
    missing_key = 0

    for point_lat, point_lon, props in points:
        sol = _m6_number(props.get("sol"))
        site = _m6_number(props.get("site"))
        drive = _m6_number(props.get("drive"))

        if sol is None or site is None or drive is None:
            missing_key += 1
            continue

        keyed.append((
            (sol, site, drive),
            {
                "sol": props.get("sol"),
                "site": props.get("site"),
                "drive": props.get("drive"),
                "latitude": point_lat,
                "longitude": point_lon % 360,
                "traverse_total_m": _m6_number(props.get("dist_total_m")),
                "elevation_geoid_m": props.get("elev_geoid"),
            },
        ))

    keyed.sort(key=lambda item: item[0])

    duplicate_keys = sum(
        1
        for previous, current in zip(keyed, keyed[1:])
        if current[0] == previous[0]
    )

    unusable_features = total - len(points)

    problems = []
    if missing_key > 0:
        problems.append(
            f"{missing_key} points have no numeric Sol/Site/Drive ordering key"
        )
    if duplicate_keys > 0:
        problems.append(
            f"{duplicate_keys} points share a Sol/Site/Drive key with another point"
        )
    if len(keyed) < 2:
        problems.append("fewer than 2 ordered points")

    valid = not problems

    route_points = [point for _, point in keyed] if valid else []

    segments = []
    gaps_m = []

    if route_points:
        current = [route_points[0]]
        for previous, point in zip(route_points, route_points[1:]):
            distance = _m6_distance_m(previous, point)
            if distance > M6_GAP_THRESHOLD_M:
                gaps_m.append(distance)
                segments.append(current)
                current = [point]
            else:
                current.append(point)
        segments.append(current)

    if valid:
        status = "Route ordered by Sol, Site, Drive."
        if unusable_features > 0:
            status += (
                f" {unusable_features} source features without usable "
                "coordinates are not part of the route."
            )
    else:
        status = "Route not drawn: " + "; ".join(problems) + "."

    return {
        "source": "NASA Mars 2020 Waypoints",
        "source_url": "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json",
        "fetched_at": datetime.fromtimestamp(fetched, timezone.utc).isoformat(),
        "stale": stale,
        "ordering": "sol, site, drive ascending",
        "total_source_features": total,
        "usable_features": len(points),
        "unusable_coordinate_count": unusable_features,
        "missing_order_key_count": missing_key,
        "duplicate_order_key_count": duplicate_keys,
        "route_order_valid": valid,
        "route_status": status,
        "route_point_count": len(route_points),
        "gap_threshold_m": M6_GAP_THRESHOLD_M,
        "gap_count": len(gaps_m),
        "max_gap_m": round(max(gaps_m), 3) if gaps_m else 0.0,
        "route_segment_count": len(segments),
        "drawable_segment_count": sum(1 for seg in segments if len(seg) >= 2),
        "route_points": route_points,
        "route_segments": segments,
    }


@api.post("/mission-route", tags=["missions"])
def get_mission_route_endpoint(request: MissionRouteRequest):
    """
    Return the complete NASA Mars 2020 waypoint collection as an ordered route.

    The selected location is returned as context only; the radius does NOT
    filter the route. Points are ordered by Sol -> Site -> Drive and the route
    is split into segments wherever consecutive points are more than 200 m apart.
    """
    try:
        result = _m6_build_route()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"NASA mission route retrieval failed: {exc}",
        )

    return {
        "status": "ok",
        "location": {
            "latitude": request.lat,
            "longitude": request.lon,
            "coverage_radius_km": request.radius_km,
        },
        **result,
    }


# ---------- Name search (NASA/JPL Mars Trek nomenclature) ----------

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _center_from_geometry(value):
    """
    Return (lat, lon) from a bbox or WKT geometry, or None.
    Accepts: "w,s,e,n" / [w, s, e, n] / "lon,lat" / WKT such as
    "POINT(77.68 18.40)" or "POLYGON((x y, x y, ...))".
    Assumes x = longitude, y = latitude.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        numbers = [float(n) for n in value if isinstance(n, (int, float))]
        text_is_wkt = False
    else:
        text = str(value)
        numbers = [float(n) for n in _NUMBER.findall(text)]
        text_is_wkt = "(" in text

    if text_is_wkt:
        if len(numbers) < 2 or len(numbers) % 2:
            return None
        xs, ys = numbers[0::2], numbers[1::2]
    elif len(numbers) == 4:
        west, south, east, north = numbers
        xs, ys = [west, east], [south, north]
    elif len(numbers) == 2:
        xs, ys = [numbers[0]], [numbers[1]]
    else:
        return None

    lon = (min(xs) + max(xs)) / 2
    lat = (min(ys) + max(ys)) / 2
    return lat, lon


def _extract_docs(payload):
    """Find the list of result records in the Trek response."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("docs", "items", "results"):
        if isinstance(payload.get(key), list):
            return payload[key]
    inner = payload.get("response")
    if isinstance(inner, dict):
        return _extract_docs(inner)
    return []


def _parse_doc(doc):
    if not isinstance(doc, dict):
        return None

    name = doc.get("name") or doc.get("title") or doc.get("label")
    if not name:
        return None

    lat = lon = None

    # Direct coordinate fields, if present.
    try:
        if doc.get("latitude") is not None and doc.get("longitude") is not None:
            lat, lon = float(doc["latitude"]), float(doc["longitude"])
    except (TypeError, ValueError):
        lat = lon = None

    # Otherwise derive from the bounding box / geometry.
    if lat is None:
        for key in ("bbox", "boundingBox", "geometry", "shape", "footprint"):
            center = _center_from_geometry(doc.get(key))
            if center:
                lat, lon = center
                break

    if lat is None or lon is None or not -90 <= lat <= 90:
        return None

    return {
        "name": str(name),
        "latitude": round(lat, 6),
        "longitude": round(lon % 360, 6),   # ATHR uses east-positive 0-360
        "item_type": doc.get("itemType") or doc.get("type") or doc.get("productType"),
        "source": "NASA/JPL Mars Trek",
    }


@api.get("/search", tags=["discovery"])
def search_location(
    q: str = Query(..., min_length=2, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    raw: bool = Query(default=False, description="Return the unprocessed Trek payload (debugging)."),
):
    """
    Search Mars nomenclature features using NASA/JPL Mars Trek.
    Coordinates are derived from the returned feature bounding box.
    """
    params = {
        "key": q,
        "proj": TREK_PROJECTION,
        "start": 0,
        "rows": limit,
    }
    request = Request(
        f"{TREK_SEARCH_URL}?{urlencode(params)}",
        headers={"User-Agent": "ATHR-Mars-Explorer/0.1"},
    )

    try:
        with urlopen(request, timeout=TREK_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Mars Trek search failed (HTTP {exc.code}).",
        )
    except (URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Mars Trek search unreachable: {getattr(exc, 'reason', exc)}",
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
            status_code=502,
            detail="Mars Trek returned an unreadable response.",
        )

    if raw:
        return payload

    results = []
    seen = set()
    for doc in _extract_docs(payload):
        # Accept geographic nomenclature records only.
        if not isinstance(doc, dict) or doc.get("itemType") != "nomenclature":
            continue
        item = _parse_doc(doc)
        if not item:
            continue
        key = (item["name"].lower(), item["latitude"], item["longitude"])
        if key in seen:
            continue
        seen.add(key)
        results.append(item)

    return {
        "status": "ok",
        "query": q,
        "count": len(results),
        "results": results,
    }


@api.get("/health", response_model=Health, tags=["system"])
def health() -> Health:
    return Health(
        status="ok",
        service="ATHR API",
        version=VERSION,
        milestone=MILESTONE,
    )

@api.get("/elevation", tags=["data"])
def get_elevation(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=0, lt=360),
):
    """
    Retrieve Mars elevation from NASA MOLA MEGDR 32ppd.
    Uses HTTP Range requests; does not download the full image.
    """
    try:
        result = read_elevation(lat, lon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"MOLA elevation request failed: {exc}",
        )

    return {
        "status": "ok",
        "location": {
            "latitude": result.latitude,
            "longitude": result.longitude,
        },
        "elevation": {
            "value_m": result.elevation_m,
            "datum": result.datum,
        },
        "pixel": {
            "line": result.line,
            "sample": result.sample,
        },
        "source": {
            "name": result.source,
            "product": result.data_product,
            "resolution_m": result.resolution_m,
            "method": result.method,
        },
    }


@api.get("/slope", tags=["data"])
def get_slope(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=0, lt=360),
):
    """
    Local slope estimate at one location from neighbouring MOLA MEGDR 32ppd
    pixels. Rejects the estimate if any neighbouring pixel was interpolated
    (PDS counts == 0). This is NOT a global slope layer.
    """
    try:
        result = read_slope(lat, lon)
    except ValueError as exc:   # includes InsufficientDataError
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"MOLA slope request failed: {exc}",
        )

    return {
        "status": "ok",
        "location": {
            "latitude": result.latitude,
            "longitude": result.longitude,
        },
        "slope": {
            "value_degrees": result.slope_degrees,
            "dz_dx": result.dz_dx,
            "dz_dy": result.dz_dy,
            "scope": "local estimate, not a global slope layer",
        },
        "elevation": {
            "center_value_m": result.center_elevation_m,
            "datum": result.datum,
        },
        "pixel": {
            "line": result.line,
            "sample": result.sample,
        },
        "source": {
            "name": result.source,
            "product": result.data_product,
            "resolution_m": result.resolution_m,
            "method": result.method,
        },
        "quality": {
            "counts_validated": True,
            "counts_valid": result.counts_valid,
            "counts_total": result.counts_total,
        },
    }


@api.get("/hazards", tags=["analysis"])
def get_hazards(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=0, lt=360),
):
    """
    Initial ATHR hazard assessment.
    Uses NASA MOLA elevation and local slope when valid.
    Insufficient measurements are reported explicitly.
    No safety classification is inferred from missing data.
    """

    # 1. Retrieve elevation
    try:
        elevation = read_elevation(lat, lon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Hazard elevation retrieval failed: {exc}",
        )

    # 2. Retrieve local slope
    try:
        slope = read_slope(lat, lon)

    except InsufficientDataError as exc:
        return {
            "status": "partial",
            "assessment": {
                "classification": "insufficient_data",
                "message": (
                    "Insufficient real MOLA measurements "
                    "for a reliable local slope estimate."
                ),
                "safety_status": "undetermined",
            },
            "location": {
                "latitude": lat,
                "longitude": lon,
            },
            "elevation": {
                "value_m": elevation.elevation_m,
                "datum": elevation.datum,
                "available": True,
            },
            "slope": {
                "available": False,
                "reason": str(exc),
            },
            "source": elevation.source,
            "data_product": elevation.data_product,
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Hazard slope retrieval failed: {exc}",
        )

    # 3. Measurements are available, but hazard thresholds
    # must be scientifically validated before classification.
    return {
        "status": "ok",
        "assessment": {
            "classification": "pending_thresholds",
            "message": (
                "Measurements are available. "
                "Hazard classification requires validated thresholds."
            ),
            "safety_status": "undetermined",
        },
        "location": {
            "latitude": lat,
            "longitude": lon,
        },
        "elevation": {
            "value_m": elevation.elevation_m,
            "datum": elevation.datum,
            "available": True,
        },
        "slope": {
            "value_degrees": slope.slope_degrees,
            "available": True,
            "counts_valid": slope.counts_valid,
            "counts_total": slope.counts_total,
            "method": slope.method,
        },
        "source": elevation.source,
        "data_product": elevation.data_product,
    }


app.include_router(api)
app.include_router(build_assistant_router(_m6_build_route))  # M7: /api/assistant
_INDEX = Path(__file__).parent / "index.html"

@app.get("/", include_in_schema=False)
def serve_frontend():
    if not _INDEX.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(_INDEX)
# Run: uvicorn main:app --reload --port 8000
