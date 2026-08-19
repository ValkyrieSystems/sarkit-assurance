import copy
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


def make_scenario(in_cphd, out_cphd, *, include_bad=False):
    with open(in_cphd, "rb") as f, skcphd.Reader(f) as r:
        ew = skcphd.ElementWrapper(r.metadata.xmltree.getroot())

        imgarea = ew["SceneCoordinates"]["ImageArea"]
        imgarea.pop("Polygon", None)  # remove Polygon if it exists for simplicity
        chan_imgarea_shape = (imgarea["X2Y2"] - imgarea["X1Y1"]) / 3

        # for simplicity, the data offsets will be shared across channels; which is not allowed by the spec
        ch_ids = ("top_left_area", "middle_area", "no_chan_area")
        ew["Channel"]["RefChId"] = ch_ids[-1]
        basis_datachan = copy.deepcopy(ew["Data"]["Channel"][0])
        del ew["Data"]["Channel"]
        basis_chparm = copy.deepcopy(ew["Channel"]["Parameters"][0])
        del ew["Channel"]["Parameters"]
        for n, ch_id in enumerate(ch_ids):
            new_datachan = copy.deepcopy(basis_datachan)
            new_datachan["Identifier"] = ch_id
            ew["Data"].add("Channel", new_datachan)

            new_chparm = copy.deepcopy(basis_chparm)
            new_chparm["Identifier"] = new_datachan["Identifier"]
            ew["Channel"].add("Parameters", new_chparm)

            new_chparm.pop("ImageArea", None)
            if n < 2:
                # only add channel image areas to first 2
                new_chparm["ImageArea"]["X1Y1"] = (
                    imgarea["X1Y1"] + n * chan_imgarea_shape
                )
                new_chparm["ImageArea"]["X2Y2"] = (
                    imgarea["X1Y1"] + (n + 1) * chan_imgarea_shape
                )

        with open(out_cphd, "wb") as f, skcphd.Writer(f, r.metadata) as w:
            sig, pvps = r.read_channel(ch_id)
            w.write_signal(ch_id, sig)
            w.write_pvp(ch_id, pvps)
            # assume we don't need support arrays for now...

    llh_points = {}
    for n, ch_id in enumerate(ch_ids):
        llh_points[ch_id] = skcphd.iac_to_llh(
            ew.elem.getroottree(), imgarea["X1Y1"] + (0.5 + n) * chan_imgarea_shape
        )

    if include_bad:
        out_iac = imgarea["X2Y2"] + [cphd_ipr.NUM_IAC_PAD + 0.1, 0.0]
        llh_points["outside_all"] = skcphd.iac_to_llh(ew.elem.getroottree(), out_iac)

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


@pytest.fixture(scope="session")
def multichan_scenario(tmp_path_factory, multichan_cphd):
    tmp_cphd = tmp_path_factory.mktemp("data") / "scenario.cphd"
    geo = make_scenario(multichan_cphd, tmp_cphd, include_bad=True)
    return (tmp_cphd, geo)


@pytest.mark.parametrize(
    "chan_args,expected_chids",
    [
        ([], ("top_left_area", "middle_area", "no_chan_area")),
        (["--ref-chan"], ("no_chan_area",)),
        (["--chan", "top_left_area", "middle_area"], ("top_left_area", "middle_area")),
        (["--chan", "top_left_area", "--ref-chan"], ("top_left_area", "no_chan_area")),
    ],
)
def test_main(multichan_scenario, tmp_path, chan_args, expected_chids):
    scenario_cphd, geo = multichan_scenario
    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out"
    outdir.mkdir()

    cphd_ipr.main([str(scenario_cphd), str(target_geojson), str(outdir)] + chan_args)
    outfile = outdir / "cphd_ipr.json"
    assert outfile.is_file()
    ipr_results = json.loads(outfile.read_text())

    expected_features_by_id = {f["id"]: f for f in geo["features"]}
    features_by_id = {f["id"]: f for f in ipr_results["features"]}

    assert features_by_id.keys() == expected_features_by_id.keys()
    expected_plot_slugs = []  # (index, ch_id)
    for fid, feat in features_by_id.items():
        expfeat = expected_features_by_id[fid]
        assert feat["geometry"] == expfeat["geometry"]
        assert sorted(
            [x["ch_id"] for x in feat["properties"]["per_channel_results"]]
        ) == sorted(expected_chids)

        for chanresult in feat["properties"]["per_channel_results"]:
            if fid == "outside_all":
                assert not chanresult["valid"]
            else:
                assert chanresult["valid"] == (
                    chanresult["ch_id"] in (fid, "no_chan_area")
                )
            if chanresult["valid"]:
                expected_plot_slugs.append(
                    (feat["properties"]["index"], chanresult["ch_id"])
                )
            else:
                assert chanresult["message"] == "Too far outside ImageArea"

    actual_plot_names = [x.name for x in outdir.glob("cphd_ipr*.html")]
    assert len(actual_plot_names) == len(expected_plot_slugs)
    for index, ch_id in expected_plot_slugs:
        assert (
            sum(
                [
                    x.startswith(f"cphd_ipr{index}") and x.endswith(f"-{ch_id}.html")
                    for x in actual_plot_names
                ]
            )
            == 1
        )


def test_main_smartopen(multichan_cphd, tmp_path):
    modified_cphd = tmp_path / "modified.cphd"
    geo = make_scenario(
        multichan_cphd,
        modified_cphd,
        include_bad=False,
    )

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out-local"
    outdir.mkdir()
    cphd_ipr.main([str(modified_cphd), str(target_geojson), str(outdir), "--ref-chan"])

    outdir2 = tmp_path / "out-remote"
    outdir2.mkdir()
    with tests.utils.static_http_server(modified_cphd.parent) as server_url:
        cphd_ipr.main(
            [
                f"{server_url}/{modified_cphd.name}",
                str(target_geojson),
                str(outdir2),
                "--ref-chan",
            ]
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
