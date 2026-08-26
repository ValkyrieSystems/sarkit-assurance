import json
import shutil
import subprocess
import sys

import lxml.html
import pytest
import sarkit.sicd as sksicd

import sarkit_assurance.sicd_chip_to_html as scth
import tests.utils


@pytest.mark.parametrize("remove_validdata", (False, True))
def test_sicd_chip_to_html(tmp_path, example_sicd, remove_validdata):
    with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
        xmltree = reader.metadata.xmltree

        if remove_validdata:
            mod_sicd = tmp_path / "mod.sicd"
            with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
                ew = sksicd.ElementWrapper(reader.metadata.xmltree.getroot())
                del ew["ImageData"]["ValidData"]
                del ew["GeoData"]["ValidData"]

                with (
                    mod_sicd.open("wb") as file,
                    sksicd.NitfWriter(file, reader.metadata) as writer,
                ):
                    writer.write_image(reader.read_image())
            sicd_filename = mod_sicd
        else:
            sicd_filename = example_sicd

    ew = sksicd.ElementWrapper(xmltree.getroot())

    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": ew["GeoData"]["SCP"]["LLH"].tolist(),
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    # features outside the image should be ignored
                    "coordinates": (ew["GeoData"]["SCP"]["LLH"] + 20).tolist(),
                },
            },
        ],
    }
    for icp in ew["GeoData"]["ImageCorners"]:
        geojson["features"].append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [icp[1], icp[0], ew["GeoData"]["SCP"]["LLH"][2]],
                },
            }
        )
    geojson_file.write_text(json.dumps(geojson))
    html_file = tmp_path / "out.html"

    assert scth.main([str(sicd_filename), str(geojson_file), str(html_file)]) == 0

    root = lxml.html.parse(html_file)
    assert len(root.findall("./body/div/div")) == len(geojson["features"]) - 1


def test_sicd_chip_to_html_no_intersection(tmp_path, example_sicd):
    with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
        xmltree = reader.metadata.xmltree

    ew = sksicd.ElementWrapper(xmltree.getroot())

    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    # features outside the image should be ignored
                    "coordinates": (ew["GeoData"]["SCP"]["LLH"] + 20).tolist(),
                },
            },
        ],
    }
    geojson_file.write_text(json.dumps(geojson))
    html_file = tmp_path / "out.html"

    assert scth.main([str(example_sicd), str(geojson_file), str(html_file)]) == 1

    assert not html_file.exists()


def test_sicd_chip_to_html_smart_open(tmp_path, example_sicd):
    with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
        xmltree = reader.metadata.xmltree

    ew = sksicd.ElementWrapper(xmltree.getroot())

    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": (ew["GeoData"]["SCP"]["LLH"]).tolist(),
                },
            },
        ],
    }
    geojson_file.write_text(json.dumps(geojson))
    html_file = tmp_path / "out.html"

    shutil.copyfile(example_sicd, tmp_path / example_sicd.name)

    with tests.utils.static_http_server(tmp_path) as server_url:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.sicd_chip_to_html",
                f"{server_url}/{example_sicd.name}",
                f"{server_url}/{geojson_file.name}",
                html_file,
            ],
        )
        assert html_file.exists()
