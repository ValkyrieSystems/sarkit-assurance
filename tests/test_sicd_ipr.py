import filecmp
import json

import numpy as np
import sarkit.sicd as sksicd
import sarkit.wgs84

import tests.utils
from sarkit_assurance import sicd_ipr


def make_feature_collection_geojson(sicd, include_bad=False):
    with open(sicd, "rb") as f, sksicd.NitfReader(f) as r:
        ew = sksicd.ElementWrapper(r.metadata.xmltree.getroot())

    ecef_points = {"scp": ew["GeoData"]["SCP"]["ECF"]}

    assert "ValidData" in ew["GeoData"]
    another_ll = ew["GeoData"]["ValidData"][0]
    ecef_points["valid_vertex"] = sarkit.wgs84.geodetic_to_cartesian(
        [*another_ll, ew["GeoData"]["SCP"]["LLH"][2]]
    )

    if include_bad:
        bad_ecef = -ew["GeoData"]["SCP"]["ECF"]
        _, _, success = sksicd.scene_to_image(r.metadata.xmltree, bad_ecef)
        assert not success
        ecef_points["no_project"] = bad_ecef

        far_rc = 2 * np.array([ew["ImageData"][x] for x in ("NumRows", "NumCols")])
        far_iac = sksicd.rowcol_to_xrowycol(r.metadata.xmltree, far_rc)
        far_ecef, _, success = sksicd.image_to_ground_plane(
            r.metadata.xmltree,
            far_iac,
            ew["GeoData"]["SCP"]["ECF"],
            sarkit.wgs84.up(ew["GeoData"]["SCP"]["LLH"]),
        )
        assert success
        ecef_points["projects_but_invalid"] = far_ecef

    def make_feature(name, ecef_pt):
        lat, lon, hae = sarkit.wgs84.cartesian_to_geodetic(ecef_pt)
        return {
            "type": "Feature",
            "id": name,
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, hae],
            },
        }

    return {
        "type": "FeatureCollection",
        "features": [make_feature(k, v) for k, v in ecef_points.items()],
    }


@np.errstate(all="ignore")
def test_main(example_sicd_cf8, tmp_path):
    geo = make_feature_collection_geojson(example_sicd_cf8, include_bad=True)

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out"
    outdir.mkdir()

    sicd_ipr.main([str(example_sicd_cf8), str(target_geojson), str(outdir)])
    outfile = outdir / "sicd_ipr.json"
    assert outfile.is_file()
    ipr_results = json.loads(outfile.read_text())

    expected_features_by_id = {f["id"]: f for f in geo["features"]}
    features_by_id = {f["id"]: f for f in ipr_results["features"]}

    assert features_by_id.keys() == expected_features_by_id.keys()
    for fid, feat in features_by_id.items():
        expfeat = expected_features_by_id[fid]
        assert feat["geometry"] == expfeat["geometry"]

        if fid == "no_project":
            assert not feat["properties"]["valid"]
            assert feat["properties"]["message"] == "Failed to project"
        elif fid == "projects_but_invalid":
            assert not feat["properties"]["valid"]
            assert feat["properties"]["message"] == "Too far outside ValidData"
        else:
            assert feat["properties"]["valid"]


def test_main_unsupported_chip_extent(example_sicd_cf8, tmp_path):
    geo = make_feature_collection_geojson(example_sicd_cf8)

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out"
    outdir.mkdir()

    sicd_ipr.main(
        [
            str(example_sicd_cf8),
            str(target_geojson),
            str(outdir),
            "--search-size-pixels",
            "999999999999",
        ]
    )
    outfile = outdir / "sicd_ipr.json"
    assert outfile.is_file()
    ipr_results = json.loads(outfile.read_text())
    assert all(not f["properties"]["valid"] for f in ipr_results["features"])
    assert any(
        f["properties"]["message"] == "Chip extent not supported by image"
        for f in ipr_results["features"]
    )


def test_main_iterative_search(example_sicd_cf8, tmp_path):
    geo = make_feature_collection_geojson(example_sicd_cf8)

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out-one"
    outdir.mkdir()
    sicd_ipr.main(
        [
            str(example_sicd_cf8),
            str(target_geojson),
            str(outdir),
            "--search-size-pixels",
            "24",
        ]
    )

    outdir2 = tmp_path / "out-two"
    outdir2.mkdir()
    sicd_ipr.main(
        [
            str(example_sicd_cf8),
            str(target_geojson),
            str(outdir2),
            "--search-size-pixels",
            "24",
            "0",
        ]
    )

    def get_valid_offsets(iprdir):
        ipr_results = json.loads((iprdir / "sicd_ipr.json").read_text())
        return np.array(
            [
                x["properties"]["observed_location_offset_xrowycol"]
                for x in ipr_results["features"]
                if x["properties"]["valid"]
            ]
        )

    # random data finds targets at two different random offsets
    offsets = get_valid_offsets(outdir)
    assert np.abs(np.diff(offsets, axis=0)).max() > 1.0

    # with a tiny (non-existent) second search size, the second iteration should just be the bulk shift
    offsets2 = get_valid_offsets(outdir2)
    assert np.abs(np.diff(offsets2, axis=0)).max() < 1.0


def test_main_smartopen(example_sicd_cf8, tmp_path):
    geo = make_feature_collection_geojson(example_sicd_cf8, include_bad=False)

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out-local"
    outdir.mkdir()
    sicd_ipr.main([str(example_sicd_cf8), str(target_geojson), str(outdir)])

    outdir2 = tmp_path / "out-remote"
    outdir2.mkdir()
    with tests.utils.static_http_server(example_sicd_cf8.parent) as server_url:
        sicd_ipr.main(
            [f"{server_url}/{example_sicd_cf8.name}", str(target_geojson), str(outdir2)]
        )

    assert filecmp.cmp(
        outdir / "sicd_ipr.json", outdir2 / "sicd_ipr.json", shallow=False
    )
