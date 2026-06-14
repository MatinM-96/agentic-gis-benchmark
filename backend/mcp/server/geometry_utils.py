
import json

import geopandas as gpd
import pandas as pd
from shapely import wkb as shapely_wkb, wkt as shapely_wkt




def _decode_geom(value):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, bytes):
        return shapely_wkb.loads(value)
    if isinstance(value, memoryview):
        return shapely_wkb.loads(bytes(value))
    if isinstance(value, str):
        s = value.strip()
        # Strip hex prefixes from PostgreSQL
        if s.startswith("\\x") or s.startswith("0x"):
            s = s[2:]
        # Try WKB hex first — must be even length and all hex chars
        if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
            try:
                return shapely_wkb.loads(bytes.fromhex(s))
            except Exception:
                pass
        # Try GeoJSON
        if s.startswith("{"):
            try:
                from shapely.geometry import shape
                return shape(json.loads(s))
            except Exception:
                pass
        # Fall back to WKT
        try:
            return shapely_wkt.loads(s)
        except Exception:
            return value
    return value

def _to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()



















def _is_geojson_geom(value) -> bool:
    """Return True if value is a GeoJSON geometry string or dict."""
    if isinstance(value, dict):
        return "type" in value and "coordinates" in value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{"):
            try:
                d = json.loads(s)
                return isinstance(d, dict) and "type" in d and "coordinates" in d
            except Exception:
                pass
    return False


def _detect_geom_col(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    preferred = ["geom", "geometry", "wkb_geom", "omrade_union", "omrade_union_valid"]
    for col in preferred:
        if col in df.columns:
            sample = df[col].dropna()
            if sample.empty:
                continue
            val = sample.iloc[0]
            if _is_geojson_geom(val):
                return col
            try:
                geom = _decode_geom(val)
                if hasattr(geom, "geom_type"):
                    return col
            except Exception:
                pass

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        val = sample.iloc[0]
        if _is_geojson_geom(val):
            return col
        try:
            geom = _decode_geom(val)
            if hasattr(geom, "geom_type"):
                return col
        except Exception:
            continue

    return None












def _sjoin_nearest_return_right(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    distance_col: str = "distance",
) -> list[dict]:
    """
    Perform a nearest join but return rows whose geometry is taken from right_gdf
    (the "found" features), not from left_gdf (the reference / AOI).

    This is correct for queries like "nearest footpath to AOI":
    - distance is measured between full left and full right geometries (Shapely default)
    - the result row carries the right-side geometry so the MAP shows the found feature

    index_right in the joined frame maps each left row to the nearest right row's index.
    We use that index to look up the right geometry explicitly.
    """
    if left_gdf.empty or right_gdf.empty:
        return []

    joined = gpd.sjoin_nearest(left_gdf, right_gdf, how="left", distance_col=distance_col)
    if joined.empty or "index_right" not in joined.columns:
        return joined.to_dict("records")

    right_geom_col = right_gdf.geometry.name
    left_geom_col = left_gdf.geometry.name

    # Map each matched right index → right geometry
    matched_geoms = joined["index_right"].map(right_gdf[right_geom_col])

    out = joined.copy()
    # Write right geometry into the output; overwrite left geometry in-place
    # so the saved rows carry the found feature, not the reference polygon.
    out[right_geom_col] = matched_geoms
    if left_geom_col != right_geom_col and left_geom_col in out.columns:
        out = out.drop(columns=[left_geom_col])

    return out.to_dict("records")


def _to_gdf(
    rows: list[dict],
    geom_col: str | None,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs=crs)

    if not geom_col:
        raise ValueError(
            "Geometry column is required for geometry operations. "
            f"Available columns: {list(df.columns)}"
        )

    detected_geom = geom_col

    if detected_geom not in df.columns:
        raise ValueError(
            f"Geometry column '{detected_geom}' not found. Available: {list(df.columns)}"
        )

    df = df.copy()
    df[detected_geom] = df[detected_geom].apply(_decode_geom)

    valid_mask = df[detected_geom].apply(lambda g: g is not None and hasattr(g, "geom_type"))
    invalid_count = int((~valid_mask).sum())
    if invalid_count > 0:
        raise ValueError(
            f"Invalid geometry values in column '{detected_geom}': "
            f"{invalid_count} of {len(df)} rows could not be decoded."
        )

    return gpd.GeoDataFrame(df, geometry=detected_geom, crs=crs)
