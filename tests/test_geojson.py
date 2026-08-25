import numpy as np
import pytest

from sarkit_assurance import _geojson


def test_features():
    single = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0, 0, 0]},
    }
    assert len(list(_geojson.features(single))) == 1

    assert len(list(_geojson.features(single["geometry"]))) == 0

    multi = {"type": "FeatureCollection", "features": [single, single, single]}
    assert len(list(_geojson.features(multi))) == 3


def test_get_feature_point():
    lon, lat, hae = [1.1, 2.2, 3.3]
    good_feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, hae]},
    }
    assert np.allclose(_geojson.get_feature_point(good_feature), [lat, lon, hae])


@pytest.mark.parametrize(
    "bad_feature",
    [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 1]},
        },
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[0, 1, 2], [3, 4, 5]]},
        },
    ],
)
def test_bad_get_feature_point(bad_feature):
    with pytest.raises(ValueError, match="Only 3D Point features"):
        _geojson.get_feature_point(bad_feature)
