import filecmp
import json
import pathlib

import lxml.etree
import numpy as np
import pytest
import sarkit.cphd as skcphd

import tests.utils
from sarkit_assurance import cphd_ipr

DATAPATH = pathlib.Path(__file__).parents[1] / "data"


def make_scenario(
    in_cphd, out_cphd, *, include_channel_imagearea=False, include_bad=False
):
    with open(in_cphd, "rb") as f, skcphd.Reader(f) as r:
        ew = skcphd.ElementWrapper(r.metadata.xmltree.getroot())

        ch_id = ew["Channel"]["RefChId"]
        imgarea = ew["SceneCoordinates"]["ImageArea"]
        imgarea.pop("Polygon", None)  # remove Polygon if it exists for simplicity
        ch_params = ew["Channel"].find("Parameters", Identifier=ch_id)
        ch_params.pop("ImageArea", None)

        if include_channel_imagearea:
            for xyname, xyval in imgarea.items():
                ch_params["ImageArea"][xyname] = xyval / 2

        with open(out_cphd, "wb") as f, skcphd.Writer(f, r.metadata) as w:
            sig, pvps = r.read_channel(ch_id)
            w.write_signal(ch_id, sig)
            w.write_pvp(ch_id, pvps)
            # assume we don't need support arrays for now...

    relevant_imgarea = ch_params["ImageArea"] if include_channel_imagearea else imgarea
    in_iac = sum(x for x in relevant_imgarea.values()) / 2
    out_iac = relevant_imgarea["X2Y2"] + [cphd_ipr.NUM_IAC_PAD + 0.1, 0.0]

    llh_points = {"inside_imgarea": skcphd.iac_to_llh(ew.elem.getroottree(), in_iac)}

    if include_bad:
        llh_points["outside_imgarea"] = skcphd.iac_to_llh(
            ew.elem.getroottree(), out_iac
        )

    def make_feature(name, latlonhae_pt):
        lat, lon, hae = latlonhae_pt
        return {
            "type": "Feature",
            "id": name,
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, hae],
            },
        }

    geo = {
        "type": "FeatureCollection",
        "features": [make_feature(k, v) for k, v in llh_points.items()],
    }
    return geo


@pytest.mark.parametrize("include_channel_imagearea", (True, False))
def test_main(multichan_cphd, tmp_path, include_channel_imagearea):
    modified_cphd = tmp_path / "modified.cphd"
    geo = make_scenario(
        multichan_cphd,
        modified_cphd,
        include_channel_imagearea=include_channel_imagearea,
        include_bad=True,
    )

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out"
    outdir.mkdir()

    cphd_ipr.main([str(modified_cphd), str(target_geojson), str(outdir)])
    outfile = outdir / "cphd_ipr.json"
    assert outfile.is_file()
    ipr_results = json.loads(outfile.read_text())

    expected_features_by_id = {f["id"]: f for f in geo["features"]}
    features_by_id = {f["id"]: f for f in ipr_results["features"]}

    assert features_by_id.keys() == expected_features_by_id.keys()
    for fid, feat in features_by_id.items():
        expfeat = expected_features_by_id[fid]
        assert feat["geometry"] == expfeat["geometry"]

        if fid == "outside_imgarea":
            assert not feat["properties"]["valid"]
            assert feat["properties"]["message"] == "Too far outside ImageArea"
        else:
            assert feat["properties"]["valid"]


def test_main_smartopen(multichan_cphd, tmp_path):
    modified_cphd = tmp_path / "modified.cphd"
    geo = make_scenario(
        multichan_cphd,
        modified_cphd,
        include_channel_imagearea=False,
        include_bad=False,
    )

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out-local"
    outdir.mkdir()
    cphd_ipr.main([str(modified_cphd), str(target_geojson), str(outdir)])

    outdir2 = tmp_path / "out-remote"
    outdir2.mkdir()
    with tests.utils.static_http_server(modified_cphd.parent) as server_url:
        cphd_ipr.main(
            [f"{server_url}/{modified_cphd.name}", str(target_geojson), str(outdir2)]
        )

    assert filecmp.cmp(
        outdir / "cphd_ipr.json", outdir2 / "cphd_ipr.json", shallow=False
    )


def test_ecef_to_scene_transform():
    cphd_xml = lxml.etree.parse(DATAPATH / "example-cphd-1.1.0.xml")
    ew = skcphd.ElementWrapper(cphd_xml.getroot())
    iarp_ecf = ew["SceneCoordinates"]["IARP"]["ECF"]

    assert cphd_ipr.ecef_to_scene_transform(cphd_xml, iarp_ecf) == pytest.approx(0.0)

    refplane = ew["SceneCoordinates"]["ReferenceSurface"].get("Planar", None)
    assert refplane is not None
    refsurf_normal = np.cross(refplane["uIAX"], refplane["uIAY"])

    # moving along normal doesn't change the point
    assert cphd_ipr.ecef_to_scene_transform(
        cphd_xml, iarp_ecf + 24 * refsurf_normal
    ) == pytest.approx(0.0)

    perturb = 8 * refplane["uIAX"] - 2 * refplane["uIAY"]
    assert cphd_ipr.ecef_to_scene_transform(
        cphd_xml, iarp_ecf + perturb
    ) == pytest.approx([8.0, -2.0])
