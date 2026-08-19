import filecmp
import json
import re

import numpy as np
import pytest
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
    expected_plot_indices = []
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
            expected_plot_indices.append(feat["properties"]["index"])

    def get_feature_index(n):
        return int(re.fullmatch(r"sicd_ipr(?P<idx>\d+).*\.html", n).group("idx"))

    actual_plot_indices = [
        get_feature_index(x.name) for x in outdir.glob("sicd_ipr*.html")
    ]
    assert actual_plot_indices == expected_plot_indices


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


def test_main_chipped_sicd(example_sicd_cf8, tmp_path):
    geo = make_feature_collection_geojson(example_sicd_cf8, include_bad=False)

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    outdir = tmp_path / "out"
    outdir.mkdir()
    sicd_ipr.main([str(example_sicd_cf8), str(target_geojson), str(outdir)])
    outfile = outdir / "sicd_ipr.json"
    assert outfile.is_file()
    ipr_results = json.loads(outfile.read_text())

    tgtprops = ipr_results["features"][1]["properties"]
    assert tgtprops["valid"]
    tgt_xrowycol = np.array(tgtprops["projected_location_xrowycol"])

    chipsize = sicd_ipr.NOM_CHIP_EDGE_PX + 10
    chip_sicd = tmp_path / "chip.sicd"
    with example_sicd_cf8.open("rb") as f, sksicd.NitfReader(f) as r:
        tgt_rcglob = sksicd.xrowycol_to_rowcol(r.metadata.xmltree, tgt_xrowycol)
        chip, chipxml = r.read_sub_image(
            int(tgt_rcglob[0] - chipsize // 2),
            int(tgt_rcglob[1] - chipsize // 2),
            int(tgt_rcglob[0] + chipsize // 2),
            int(tgt_rcglob[1] + chipsize // 2),
        )
        assert chip.shape == (chipsize, chipsize)
        r.metadata.xmltree = chipxml
        with chip_sicd.open("wb") as fw, sksicd.NitfWriter(fw, r.metadata) as w:
            w.write_image(chip)

    chipoutdir = tmp_path / "out_chip"
    chipoutdir.mkdir()
    sicd_ipr.main([str(chip_sicd), str(target_geojson), str(chipoutdir)])
    outfile = chipoutdir / "sicd_ipr.json"
    assert outfile.is_file()
    ipr_results_chip = json.loads(outfile.read_text())

    # not in chip
    assert not ipr_results_chip["features"][0]["properties"]["valid"]

    # chip results match global
    assert ipr_results_chip["features"][1] == ipr_results["features"][1]


def test_main_pixeltypes(example_sicd_cf8, tmp_path):
    geo = make_feature_collection_geojson(example_sicd_cf8, include_bad=False)
    geo["features"] = [x for x in geo["features"] if x["id"] == "scp"]  # only keep SCP
    assert len(geo["features"]) == 1

    target_geojson = tmp_path / "targets.geojson"
    target_geojson.write_text(json.dumps(geo))

    # write equivalent SICDs in all PixelTypes
    chipsize = sicd_ipr.NOM_CHIP_EDGE_PX + 10

    rng = np.random.default_rng(123456)
    amp = rng.integers(0, 255, (chipsize, chipsize))
    phase = rng.choice([0, 64, 128, 192], amp.shape)
    sig = amp * np.exp(1j * phase / 256 * 2 * np.pi)

    ampphs = np.zeros(
        (chipsize, chipsize), dtype=sksicd.PIXEL_TYPES["AMP8I_PHS8I"]["dtype"]
    )
    ampphs["amp"] = amp
    ampphs["phase"] = phase

    reim16 = np.zeros(
        (chipsize, chipsize), dtype=sksicd.PIXEL_TYPES["RE16I_IM16I"]["dtype"]
    )
    reim16["real"] = sig.real
    reim16["imag"] = sig.imag

    reim32f = np.round(sig).astype(sksicd.PIXEL_TYPES["RE32F_IM32F"]["dtype"])

    equivalent_sicds = {
        "AMP8I_PHS8I": (ampphs, tmp_path / "AMP8I_PHS8I.sicd"),
        "RE16I_IM16I": (reim16, tmp_path / "RE16I_IM16I.sicd"),
        "RE32F_IM32F": (reim32f, tmp_path / "RE32F_IM32F.sicd"),
    }
    with example_sicd_cf8.open("rb") as f, sksicd.NitfReader(f) as r:
        tgt_rcglob = sksicd.xrowycol_to_rowcol(r.metadata.xmltree, [0.0, 0.0])
        chip, chipxml = r.read_sub_image(
            int(tgt_rcglob[0] - chipsize // 2),
            int(tgt_rcglob[1] - chipsize // 2),
            int(tgt_rcglob[0] + chipsize // 2),
            int(tgt_rcglob[1] + chipsize // 2),
        )
        assert chip.shape == (chipsize, chipsize)
        r.metadata.xmltree = chipxml
        for pixeltype, (sig, sicdfile) in equivalent_sicds.items():
            chipxml.find(".//{*}PixelType").text = pixeltype
            with sicdfile.open("wb") as fw, sksicd.NitfWriter(fw, r.metadata) as w:
                w.write_image(sig)

    results = {}
    for pixeltype, (_, sicdfile) in equivalent_sicds.items():
        outdir = tmp_path / f"out-{pixeltype}"
        outdir.mkdir()
        sicd_ipr.main([str(sicdfile), str(target_geojson), str(outdir)])
        outfile = outdir / "sicd_ipr.json"
        assert outfile.is_file()
        results[pixeltype] = json.loads(outfile.read_text())

    def compare_results(a, b):
        for fa, fb in zip(a["features"], b["features"]):
            if "id" in fa:
                assert fa["id"] == fb["id"]
            else:
                assert "id" not in fb
            faprops = fa["properties"]
            fbprops = fb["properties"]
            assert faprops["valid"] == fbprops["valid"]
            if faprops["valid"]:
                for prop in ("observed_location_offset_xrowycol", "peak_power"):
                    assert faprops[prop] == pytest.approx(fbprops[prop])

    compare_results(results["AMP8I_PHS8I"], results["RE16I_IM16I"])
    compare_results(results["AMP8I_PHS8I"], results["RE32F_IM32F"])
