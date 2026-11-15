"""
ATHR MOLA 32ppd Adapter
Reads single elevation samples from NASA PDS with HTTP Range requests and
estimates local slope from the 3x3 neighbourhood of a pixel, using the PDS
counts image to reject interpolated (no-data) pixels.
No full IMG download required.
"""

import http.client
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = (
    "https://pds-geosciences.wustl.edu/"
    "mgs/urn-nasa-pds-mgs_mola_topography_derived/meg032/"
)
LABEL_URL = BASE_URL + "megt90n000fb.lbl"
IMAGE_URL = BASE_URL + "megt90n000fb.img"

# M4: counts = number of real MOLA shots that contributed to each pixel.
# 0 means the pixel was interpolated, so it is not a real measurement.
COUNTS_LABEL_URL = BASE_URL + "megc90n000fb.lbl"
COUNTS_IMAGE_URL = BASE_URL + "megc90n000fb.img"

LINES = 5760
SAMPLES = 11520
RECORD_BYTES = 23040          # SAMPLES * SAMPLE_BYTES
SAMPLE_BYTES = 2              # 16-bit, big-endian

RESOLUTION = 32.0             # pixels per degree
LINE_OFFSET = 2880.5
SAMPLE_OFFSET = 5760.5
CENTER_LONGITUDE = 180.0

# M4: ground size of one pixel (~1852 m at the equator for 32 ppd)
MARS_RADIUS_KM = 3396.19
PIXEL_M = 2.0 * math.pi * MARS_RADIUS_KM * 1000.0 / (360.0 * RESOLUTION)

# East-west pixel spacing shrinks with cos(lat); slope is unreliable near poles
MAX_SLOPE_LATITUDE = 87.0

HEADERS = {"User-Agent": "ATHR-Mars-Exploration-Platform/0.1"}
TIMEOUT = 30


class InsufficientDataError(ValueError):
    """Data exists but is not valid for the calculation (HTTP 422 in the API)."""


@dataclass(frozen=True)
class MolaResult:
    elevation_m: int
    latitude: float
    longitude: float
    line: int
    sample: int
    source: str
    resolution_m: float
    datum: str
    method: str
    data_product: str


@dataclass(frozen=True)
class SlopeResult:
    slope_degrees: float
    dz_dx: float                  # east-positive, metres per metre
    dz_dy: float                  # north-positive, metres per metre
    center_elevation_m: int
    latitude: float
    longitude: float
    line: int
    sample: int
    source: str
    resolution_m: float
    datum: str
    method: str
    data_product: str
    counts_valid: int
    counts_total: int


def coordinate_to_index(latitude: float, longitude: float) -> tuple[int, int]:
    """Return the 1-based (line, sample) of the pixel containing the point.

    Integer values are pixel centers in the MEGDR map convention, so the
    containing pixel is the nearest integer. floor(x + 0.5) is used instead
    of round() so that exact .5 boundaries resolve the same way every time
    (round() uses banker's rounding).
    """
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    longitude %= 360  # accepts -180..180 as well as 0..360

    line = math.floor(LINE_OFFSET - latitude * RESOLUTION + 0.5)
    sample = math.floor(SAMPLE_OFFSET + (longitude - CENTER_LONGITUDE) * RESOLUTION + 0.5)

    line = min(max(line, 1), LINES)             # poles
    sample = (sample - 1) % SAMPLES + 1         # wrap at the 0/360 seam
    return line, sample


# ----------------------------------------------------------------------
# Low-level reading
# ----------------------------------------------------------------------

def _fetch_bytes(url: str, start: int, length: int) -> bytes:
    """Fetch exactly `length` bytes at byte offset `start` using a Range request."""
    end = start + length - 1

    request = Request(url, headers={**HEADERS, "Range": f"bytes={start}-{end}"})

    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            # Check BEFORE reading: a server that ignores Range answers 200
            # with the whole ~130 MB file.
            if response.status != 206:
                raise RuntimeError(
                    f"Expected HTTP 206, got {response.status}. "
                    "The server may not support Range requests."
                )
            content_range = response.headers.get("Content-Range", "")
            raw = response.read(length + 1)
    except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"MOLA PDS request failed: {exc}") from exc

    if not content_range.startswith(f"bytes {start}-{end}/"):
        raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
    if len(raw) != length:
        raise RuntimeError(f"Expected {length} bytes, received {len(raw)}.")

    return raw


@lru_cache(maxsize=4096)
def _read_sample(line: int, sample: int) -> int:
    """Single elevation sample (M3 behaviour, unchanged)."""
    start = (line - 1) * RECORD_BYTES + (sample - 1) * SAMPLE_BYTES
    raw = _fetch_bytes(IMAGE_URL, start, SAMPLE_BYTES)
    return int.from_bytes(raw, byteorder="big", signed=True)


@lru_cache(maxsize=2048)
def _read_run(url: str, signed: bool, line: int, sample: int, count: int) -> tuple:
    """`count` consecutive samples on one line; sample is 1-based."""
    start = (line - 1) * RECORD_BYTES + (sample - 1) * SAMPLE_BYTES
    raw = _fetch_bytes(url, start, count * SAMPLE_BYTES)
    return tuple(
        int.from_bytes(raw[i:i + SAMPLE_BYTES], byteorder="big", signed=signed)
        for i in range(0, len(raw), SAMPLE_BYTES)
    )


def _read_row(url: str, signed: bool, line: int, first_sample: int, count: int) -> list:
    """Like _read_run, but wraps across the 0/360 longitude seam."""
    first = (first_sample - 1) % SAMPLES + 1

    if first + count - 1 <= SAMPLES:
        return list(_read_run(url, signed, line, first, count))

    head = SAMPLES - first + 1
    return (
        list(_read_run(url, signed, line, first, head))
        + list(_read_run(url, signed, line, 1, count - head))
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def read_elevation(latitude: float, longitude: float) -> MolaResult:
    line, sample = coordinate_to_index(latitude, longitude)
    return MolaResult(
        elevation_m=_read_sample(line, sample),
        latitude=latitude,
        longitude=longitude % 360,
        line=line,
        sample=sample,
        source="NASA PDS Geosciences Node",
        resolution_m=1853.0,  # ~1.85 km per pixel at the equator
        datum="Mars areoid (not the reference ellipsoid)",
        method="Nearest pixel of the gridded MOLA MEGDR product",
        data_product="MGS MOLA MEGDR 32ppd",
    )


def read_slope(latitude: float, longitude: float) -> SlopeResult:
    """Local slope in degrees from the 3x3 MOLA neighbourhood (Horn method).

    Raises InsufficientDataError (a ValueError) if any of the 9 pixels was
    interpolated (counts == 0), since slope from filled data is not trustworthy.
    """
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if abs(latitude) > MAX_SLOPE_LATITUDE:
        raise InsufficientDataError(
            f"Slope is not reliable poleward of {MAX_SLOPE_LATITUDE:g} degrees."
        )

    line, sample = coordinate_to_index(latitude, longitude)

    if line < 2 or line > LINES - 1:
        raise InsufficientDataError("Location is too close to the map edge for slope.")

    rows = (line - 1, line, line + 1)  # north -> south

    elev = [_read_row(IMAGE_URL, True, r, sample - 1, 3) for r in rows]
    counts = [_read_row(COUNTS_IMAGE_URL, False, r, sample - 1, 3) for r in rows]

    valid = sum(1 for row in counts for c in row if c > 0)
    if valid < 9:
        raise InsufficientDataError(
            f"Insufficient MOLA coverage: only {valid}/9 neighbouring pixels "
            "contain real measurements."
        )

    (a, b, c), (d, _e, f), (g, h, i) = elev

    # Latitude of the centre pixel (integer line = pixel centre, 1-based)
    center_lat = (LINE_OFFSET - line) / RESOLUTION
    dx = PIXEL_M * math.cos(math.radians(center_lat))
    dy = PIXEL_M

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)   # east +
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * dy)   # north +

    slope_deg = math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))

    return SlopeResult(
        slope_degrees=slope_deg,
        dz_dx=dz_dx,
        dz_dy=dz_dy,
        center_elevation_m=elev[1][1],
        latitude=latitude,
        longitude=longitude % 360,
        line=line,
        sample=sample,
        source="NASA PDS Geosciences Node",
        resolution_m=1853.0,
        datum="Mars areoid (not the reference ellipsoid)",
        method="Horn 3x3 gradient on neighbouring MOLA pixels (local estimate)",
        data_product="MGS MOLA MEGDR 32ppd",
        counts_valid=valid,
        counts_total=9,
    )


def verify_label(label_url: str = LABEL_URL) -> None:
    """Optional startup check: compare the constants above with the PDS label."""
    request = Request(label_url, headers=HEADERS)
    with urlopen(request, timeout=TIMEOUT) as response:
        text = response.read(200_000).decode("ascii", "replace")

    expected = {
        "LINES": LINES,
        "LINE_SAMPLES": SAMPLES,
        "MAP_RESOLUTION": RESOLUTION,
        "LINE_PROJECTION_OFFSET": LINE_OFFSET,
        "SAMPLE_PROJECTION_OFFSET": SAMPLE_OFFSET,
        "CENTER_LONGITUDE": CENTER_LONGITUDE,
    }
    for key, want in expected.items():
        match = re.search(rf"^\s*{key}\s*=\s*(-?[\d.]+)", text, re.MULTILINE)
        if not match or float(match.group(1)) != want:
            found = match.group(1) if match else "missing"
            raise RuntimeError(f"Label mismatch for {key}: expected {want}, found {found}")


if __name__ == "__main__":
    print("ATHR MOLA 32ppd Adapter Test")
    print("=" * 40)

    try:
        verify_label()
        print("Elevation label constants verified.")
        verify_label(COUNTS_LABEL_URL)
        print("Counts label constants verified.")

        result = read_elevation(18.4, 77.5)  # Jezero Crater area
        print(f"Latitude:    {result.latitude}")
        print(f"Longitude:   {result.longitude}")
        print(f"Pixel:       line {result.line}, sample {result.sample}")
        print(f"Elevation:   {result.elevation_m} m")
        print(f"Datum:       {result.datum}")
        print(f"Resolution:  ~{result.resolution_m} m")
        print(f"Method:      {result.method}")
        print(f"Product:     {result.data_product}")

        print("-" * 40)
        slope = read_slope(18.4, 77.5)
        print(f"Slope:       {slope.slope_degrees:.3f} degrees")
        print(f"Counts:      {slope.counts_valid}/{slope.counts_total} real pixels")
        print(f"Method:      {slope.method}")
    except Exception as exc:
        print(f"[ERROR] {exc}")