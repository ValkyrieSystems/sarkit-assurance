import json
import shutil
import subprocess
import sys

import lxml.html
import sarkit.sidd as sksidd
import sarkit.wgs84

import sarkit_assurance.sidd_chip_to_html as scth
import tests.utils


def test_all_features():
    single = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0, 0, 0]},
    }
    assert len(scth._get_all_features(single)) == 1

    assert len(scth._get_all_features(single["geometry"])) == 0

    multi = {"type": "FeatureCollection", "features": [single, single, single]}
    assert len(scth._get_all_features(multi)) == 3


def _get_ref_pt_llh(xmltree):
    ref_pt_ecef = sksidd.ElementWrapper(
        xmltree.getroot().findall("./{*}Measurement//{*}ReferencePoint")[0]
    )["ECEF"]
    ref_pt_llh = sarkit.wgs84.cartesian_to_geodetic(ref_pt_ecef)
    ref_pt_llh = ref_pt_llh[[1, 0, 2]]
    return ref_pt_llh


def test_sidd_chip_to_html(tmp_path, multi_sidd):
    with open(multi_sidd, "rb") as file, sksidd.NitfReader(file) as reader:
        num_images = len(reader.metadata.images)
        xmltree = reader.metadata.images[0].xmltree

    ref_pt_llh = _get_ref_pt_llh(xmltree)
    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": ref_pt_llh.tolist(),
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    # features outside the image should be ignored
                    "coordinates": (ref_pt_llh + 20).tolist(),
                },
            },
        ],
    }
    geojson_file.write_text(json.dumps(geojson))
    html_file = tmp_path / "out.html"

    assert scth.main([str(multi_sidd), str(geojson_file), str(html_file)]) == 0

    root = lxml.html.parse(html_file)
    assert len(root.findall("./body/div/div")) == num_images * (
        len(geojson["features"]) - 1
    )


def test_no_intersection(tmp_path, multi_sidd):
    with open(multi_sidd, "rb") as file, sksidd.NitfReader(file) as reader:
        xmltree = reader.metadata.images[0].xmltree

    ref_pt_llh = _get_ref_pt_llh(xmltree)
    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    # features outside the image should be ignored
                    "coordinates": (ref_pt_llh + 20).tolist(),
                },
            },
        ],
    }
    geojson_file.write_text(json.dumps(geojson))
    html_file = tmp_path / "out.html"

    assert scth.main([str(multi_sidd), str(geojson_file), str(html_file)]) == 1

    assert not html_file.exists()


def test_smart_open(tmp_path, multi_sidd):
    with open(multi_sidd, "rb") as file, sksidd.NitfReader(file) as reader:
        xmltree = reader.metadata.images[0].xmltree

    ref_pt_llh = _get_ref_pt_llh(xmltree)
    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": ref_pt_llh.tolist(),
                },
            },
        ],
    }
    geojson_file.write_text(json.dumps(geojson))
    html_file = tmp_path / "out.html"

    shutil.copyfile(multi_sidd, tmp_path / multi_sidd.name)

    with tests.utils.static_http_server(tmp_path) as server_url:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.sidd_chip_to_html",
                f"{server_url}/{multi_sidd.name}",
                f"{server_url}/{geojson_file.name}",
                html_file,
            ],
        )
        assert html_file.exists()
