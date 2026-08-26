"""Functions for working with GeoJSON"""

from typing import Any, Iterator

import numpy as np


def features(geojson: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield features in a parsed GeoJSON"""
    if geojson["type"] == "Feature":
        yield geojson
    elif geojson["type"] == "FeatureCollection":
        yield from geojson["features"]


def get_feature_point(feature: dict[str, Any]) -> np.ndarray:
    """Return the point location (lat, lon, hae) from a GeoJSON feature."""
    coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
    if feature["geometry"]["type"] != "Point" or coordinates.shape != (3,):
        raise ValueError("Only 3D Point features are supported")
    # swap lon/lat
    return coordinates[[1, 0, 2]]
