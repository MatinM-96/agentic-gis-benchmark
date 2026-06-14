from langchain.tools import tool
from functools import lru_cache
from pathlib import Path
import requests
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ORS_API_KEY

from backend.mcp.client.tools_discovery import (
    get_database_catalog,
    validate_sql_via_mcp,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
    get_available_databases,
)
from backend.mcp.client.tools_steps import (
    execute_query_step,
    execute_filter_step,
    execute_select_columns_step,
    execute_spatial_join_step,
    execute_attribute_join_step,
    execute_buffer_step,
    execute_nearest_step,
    execute_distance_step,
    execute_aggregate_step,
    execute_sort_step,
    execute_limit_step,
    execute_clip_step,
    execute_union_step,
    execute_intersection_step,
    execute_difference_step,
    execute_dissolve_step,
    execute_centroid_step,
    execute_transform_crs_step,
)


def _safe_json(data) -> str:
    """Serialize to JSON string, always safe for the agent to parse."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _err_json(step: int | None, action: str, msg: str) -> str:
    """Consistent error JSON returned to agent when a tool fails."""
    return _safe_json({"executed": False, "step": step, "action": action, "error": str(msg)})


@lru_cache(maxsize=64)
def _fetch_kommune_geometry(name: str) -> tuple[str, dict, tuple[float, float, float, float]]:
    r = requests.get(
        "https://api.kartverket.no/kommuneinfo/v1/sok",
        params={"knavn": name},
        timeout=10,
    )
    r.raise_for_status()
    kommuner = r.json().get("kommuner", [])
    if not kommuner:
        raise ValueError(f"No municipality found for '{name}'")

    kommune_nr = kommuner[0]["kommunenummer"]

    r = requests.get(
        f"https://api.kartverket.no/kommuneinfo/v1/kommuner/{kommune_nr}/omrade",
        timeout=10,
    )
    r.raise_for_status()
    omrade = r.json()["omrade"]
    coords = omrade["coordinates"]

    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for polygon in coords:
        for ring in polygon:
            for lon, lat in ring:
                minx = min(minx, lon)
                miny = min(miny, lat)
                maxx = max(maxx, lon)
                maxy = max(maxy, lat)

    return kommune_nr, omrade, (minx, miny, maxx, maxy)








# --------------------------------------------------
# Discovery tools
# --------------------------------------------------

@tool
async def tool_get_available_databases() -> str:
    """Get all available database names."""
    return _safe_json(await get_available_databases())


@tool
async def tool_get_available_schemas(databaseName: str) -> str:
    """Get available schemas for a given database."""
    return _safe_json(await get_available_schemas(databaseName=databaseName))


@tool
async def tool_get_available_tables(databaseName: str, schemaName: str) -> str:
    """Get available tables for a given schema in a given database."""
    return _safe_json(await get_available_tables(databaseName=databaseName, schemaName=schemaName))


@tool
async def tool_get_table_schema(databaseName: str, schemaName: str, table_name: str) -> str:
    """Get column names and data types for a specific table."""
    return _safe_json(await get_table_schema(databaseName=databaseName, schemaName=schemaName, table_name=table_name))


@tool
async def tool_get_database_catalog(
    databaseName: str | None = None,
    schemaName: str | None = None,
) -> str:
    """Get database catalog with schemas, tables, geometry columns and SRID info."""
    return _safe_json(await get_database_catalog(databaseName=databaseName, schemaName=schemaName))



# --------------------------------------------------
# Validation tool
# --------------------------------------------------

@tool
async def tool_validate_sql(sql: str, databaseName: str, step_number: int) -> str:
    """Validate SQL syntax using EXPLAIN through MCP and return JSON."""
    return _safe_json(
        await validate_sql_via_mcp(
            sql=sql,
            databaseName=databaseName,
            step_number=step_number,
        )
    )





# --------------------------------------------------
# Step tools — all catch exceptions and return JSON
# --------------------------------------------------

@tool
async def tool_execute_query_step(run_id: str, step: dict) -> str:
    """Execute one validated SQL query step."""
    try:
        return _safe_json(await execute_query_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "query", str(e))


@tool
async def tool_execute_filter_step(run_id: str, step: dict) -> str:
    """Filter rows from a previous step."""
    try:
        return _safe_json(await execute_filter_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "filter", str(e))


@tool
async def tool_execute_select_columns_step(run_id: str, step: dict) -> str:
    """Select specific columns from a previous step."""
    try:
        return _safe_json(await execute_select_columns_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "select_columns", str(e))


@tool
async def tool_execute_spatial_join_step(run_id: str, step: dict) -> str:
    """Spatial join between two previous step results."""
    try:
        return _safe_json(await execute_spatial_join_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "spatial_join", str(e))


@tool
async def tool_execute_attribute_join_step(run_id: str, step: dict) -> str:
    """Attribute join between two previous step results."""
    try:
        return _safe_json(await execute_attribute_join_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "attribute_join", str(e))


@tool
async def tool_execute_buffer_step(run_id: str, step: dict) -> str:
    """Buffer geometries from a previous step."""
    try:
        return _safe_json(await execute_buffer_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "buffer", str(e))


@tool
async def tool_execute_nearest_step(run_id: str, step: dict) -> str:
    """Find nearest geometry from another step."""
    try:
        return _safe_json(await execute_nearest_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "nearest", str(e))


@tool
async def tool_execute_distance_step(run_id: str, step: dict) -> str:
    """Compute nearest distance to another step."""
    try:
        return _safe_json(await execute_distance_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "distance", str(e))


@tool
async def tool_execute_aggregate_step(run_id: str, step: dict) -> str:
    """Aggregate rows from a previous step."""
    try:
        return _safe_json(await execute_aggregate_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "aggregate", str(e))


@tool
async def tool_execute_sort_step(run_id: str, step: dict) -> str:
    """Sort rows from a previous step."""
    try:
        return _safe_json(await execute_sort_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "sort", str(e))


@tool
async def tool_execute_limit_step(run_id: str, step: dict) -> str:
    """Limit rows from a previous step."""
    try:
        return _safe_json(await execute_limit_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "limit", str(e))


@tool
async def tool_execute_clip_step(run_id: str, step: dict) -> str:
    """Clip one geometry layer by another."""
    try:
        return _safe_json(await execute_clip_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "clip", str(e))


@tool
async def tool_execute_union_step(run_id: str, step: dict) -> str:
    """Union all geometries from a previous step."""
    try:
        return _safe_json(await execute_union_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "union", str(e))


@tool
async def tool_execute_intersection_step(run_id: str, step: dict) -> str:
    """Overlay intersection between two geometry layers."""
    try:
        return _safe_json(await execute_intersection_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "intersection", str(e))


@tool
async def tool_execute_difference_step(run_id: str, step: dict) -> str:
    """Overlay difference between two geometry layers."""
    try:
        return _safe_json(await execute_difference_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "difference", str(e))


@tool
async def tool_execute_dissolve_step(run_id: str, step: dict) -> str:
    """Dissolve geometries by group."""
    try:
        return _safe_json(await execute_dissolve_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "dissolve", str(e))


@tool
async def tool_execute_centroid_step(run_id: str, step: dict) -> str:
    """Compute centroids for a geometry layer."""
    try:
        return _safe_json(await execute_centroid_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "centroid", str(e))


@tool
async def tool_execute_transform_crs_step(run_id: str, step: dict) -> str:
    """Transform geometries to another CRS."""
    try:
        return _safe_json(await execute_transform_crs_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "transform_crs", str(e))


# --------------------------------------------------
# Routing tool — OpenRouteService
# --------------------------------------------------

_ORS_PROFILES = {
    "foot-walking", "foot-hiking",
    "driving-car", "driving-hgv",
    "cycling-regular", "cycling-road", "cycling-mountain", "cycling-electric",
    "wheelchair",
}

_ORS_BASE = "https://api.openrouteservice.org/v2/directions"
_POINT_WKT_RE = re.compile(
    r"POINT\s*\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)


def _coerce_coordinate_pair(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except Exception:
        return None


def _extract_coords_from_text(value: str | None) -> list[float] | None:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    match = _POINT_WKT_RE.fullmatch(text)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        coords = _coerce_coordinate_pair(parsed)
        if coords is not None:
            return coords

    compact = text.replace(",", " ")
    parts = [part for part in compact.split() if part]
    if len(parts) == 2:
        try:
            return [float(parts[0]), float(parts[1])]
        except Exception:
            return None

    return None


@lru_cache(maxsize=256)
def _geocode_place(place: str) -> tuple[float, float] | None:
    if not place or not place.strip():
        return None
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "agentic-gis-benchmark/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        return None
    return float(hits[0]["lon"]), float(hits[0]["lat"])


@tool
def tool_get_route(
    coordinates: list | None = None,
    profile: str = "foot-walking",
    origin: str | None = None,
    destination: str | None = None,
) -> str:
    """
    Calculate a route between two or more waypoints using OpenRouteService.

    Returns a GeoJSON FeatureCollection with the route line geometry and
    summary properties (distance in metres, duration in seconds).

    Args:
        coordinates: Optional list of [longitude, latitude] pairs, e.g.
                     [[10.74, 59.91], [10.75, 59.92]].
                     At least two points are required if origin/destination are not provided.
        profile: Travel mode. One of:
                 foot-walking (default), foot-hiking,
                 driving-car, driving-hgv,
                 cycling-regular, cycling-road, cycling-mountain, cycling-electric,
                 wheelchair.
        origin: Optional start place name for geocoding, e.g. "Oslo S".
        destination: Optional end place name for geocoding, e.g. "Rådhuset i Oslo".
    """
    if not ORS_API_KEY:
        return _safe_json({"error": "ORS_API_KEY is not configured."})

    if (not isinstance(coordinates, list) or len(coordinates) < 2) and origin and destination:
        origin_coords = _extract_coords_from_text(origin)
        destination_coords = _extract_coords_from_text(destination)

        try:
            if origin_coords is None:
                origin_coords = _geocode_place(origin)
            if destination_coords is None:
                destination_coords = _geocode_place(destination)
        except Exception as e:
            return _safe_json({"error": f"Route geocoding failed: {e}"})

        if not origin_coords or not destination_coords:
            return _safe_json({"error": f"Could not geocode route endpoints: {origin} -> {destination}"})

        coordinates = [list(origin_coords), list(destination_coords)]
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return _safe_json({"error": "Provide either coordinates or both origin and destination."})

    normalized_coordinates: list[list[float]] = []
    for item in coordinates:
        coords = _coerce_coordinate_pair(item)
        if coords is None:
            return _safe_json({"error": f"Invalid route coordinate: {item}"})
        normalized_coordinates.append(coords)

    if profile not in _ORS_PROFILES:
        profile = "foot-walking"

    url = f"{_ORS_BASE}/{profile}/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {"coordinates": normalized_coordinates}

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        return _safe_json({"error": f"ORS HTTP {resp.status_code}: {resp.text[:300]}"})
    except Exception as e:
        return _safe_json({"error": str(e)})

    # Surface the summary in each feature's properties so the agent can read it
    features = data.get("features") or []
    for feat in features:
        props = feat.get("properties") or {}
        summary = props.get("summary") or {}
        feat["properties"] = {
            **props,
            "distance_m": summary.get("distance"),
            "duration_s": summary.get("duration"),
        }

    return _safe_json({
        "executed": True,
        "profile": profile,
        "geojson": data,
        "geom_col": "geometry",
        "rows": [
            {
                "geometry": feat["geometry"],
                **{k: v for k, v in (feat.get("properties") or {}).items()},
            }
            for feat in features
        ],
        "row_count": len(features),
    })


_ORS_ISOCHRONES_BASE = "https://api.openrouteservice.org/v2/isochrones"
_ORS_GEOCODE_SEARCH = "https://api.openrouteservice.org/geocode/search"
_ORS_GEOCODE_REVERSE = "https://api.openrouteservice.org/geocode/reverse"


@tool
def tool_get_isochrone(
    coordinates: list | None = None,
    minutes: int = 10,
    profile: str = "foot-walking",
) -> str:
    """
    Calculate an isochrone (reachable area within N minutes) using OpenRouteService.

    Returns a GeoJSON FeatureCollection polygon showing the area reachable
    within the given travel time from the given location.

    Args:
        coordinates: [longitude, latitude] of the starting point, e.g. [10.74, 59.91].
        minutes: Travel time in minutes (default 10).
        profile: Travel mode — same options as tool_get_route (default foot-walking).
    """
    if not ORS_API_KEY:
        return _safe_json({"error": "ORS_API_KEY is not configured."})

    center: list[float] | None = None
    if isinstance(coordinates, list):
        center = _coerce_coordinate_pair(coordinates)
        if center is None and coordinates:
            center = _coerce_coordinate_pair(coordinates[0])
    if center is None:
        return _safe_json({"error": "Isochrone requires one [lon, lat] coordinate pair."})

    if profile not in _ORS_PROFILES:
        profile = "foot-walking"

    try:
        resp = requests.post(
            f"{_ORS_ISOCHRONES_BASE}/{profile}",
            headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
            json={"locations": [center], "range": [int(minutes) * 60]},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        return _safe_json({"error": f"ORS isochrones HTTP {resp.status_code}: {resp.text[:300]}"})
    except Exception as e:
        return _safe_json({"error": str(e)})

    features = data.get("features") or []
    for feat in features:
        props = feat.get("properties") or {}
        feat["properties"] = {**props, "minutes": minutes, "profile": profile}

    return _safe_json({
        "executed": True,
        "profile": profile,
        "geojson": data,
        "geom_col": "geometry",
        "rows": [
            {"geometry": feat["geometry"], "minutes": minutes, "profile": profile}
            for feat in features
        ],
        "row_count": len(features),
        "ors_kind": "isochrone",
    })


@tool
def tool_get_place_info(
    coordinates: list | None = None,
    text: str | None = None,
) -> str:
    """
    Look up a place by coordinates (reverse geocode) or by name/text (forward geocode).

    Args:
        coordinates: [longitude, latitude] to reverse-geocode, e.g. [10.74, 59.91].
        text: Place name or address to forward-geocode, e.g. "Oslo S".
    """
    if not ORS_API_KEY:
        return _safe_json({"error": "ORS_API_KEY is not configured."})

    try:
        if isinstance(coordinates, list):
            center = _coerce_coordinate_pair(coordinates)
            if center is None and coordinates:
                center = _coerce_coordinate_pair(coordinates[0])
            if center is not None:
                resp = requests.get(
                    _ORS_GEOCODE_REVERSE,
                    params={"api_key": ORS_API_KEY, "point.lon": center[0], "point.lat": center[1], "size": 1},
                    timeout=10,
                )
                resp.raise_for_status()
                features = resp.json().get("features") or []
                label = ((features[0] or {}).get("properties") or {}).get("label") if features else None
                return _safe_json({
                    "executed": True,
                    "label": label,
                    "coordinates": center,
                    "rows": [{"label": label, "coordinates": center}],
                    "row_count": 1,
                    "ors_kind": "place",
                })

        if text and str(text).strip():
            resp = requests.get(
                _ORS_GEOCODE_SEARCH,
                params={"api_key": ORS_API_KEY, "text": str(text).strip(), "size": 1, "boundary.country": "NO"},
                timeout=10,
            )
            resp.raise_for_status()
            features = resp.json().get("features") or []
            if features:
                props = (features[0] or {}).get("properties") or {}
                coords = _coerce_coordinate_pair(((features[0] or {}).get("geometry") or {}).get("coordinates"))
                label = props.get("label") or props.get("name") or text
                return _safe_json({
                    "executed": True,
                    "label": label,
                    "coordinates": coords,
                    "rows": [{"label": label, "coordinates": coords}],
                    "row_count": 1,
                    "ors_kind": "place",
                })
            return _safe_json({"executed": False, "error": f"No results for '{text}'"})

    except requests.HTTPError as e:
        return _safe_json({"error": f"ORS place lookup HTTP {e.response.status_code}: {e.response.text[:300]}"})
    except Exception as e:
        return _safe_json({"error": str(e)})

    return _safe_json({"error": "Provide coordinates or text for place lookup."})
