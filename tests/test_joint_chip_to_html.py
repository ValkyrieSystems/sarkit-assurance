import numpy as np
import sarkit.sicd as sksicd

import sarkit_assurance.joint_chip_to_html
import tests.utils


def assert_output_dir(output_dir):
    # assumes output_dir was empty to start
    expected_names = {"geo.json", "sicd_chip.html", "sidd_chip.html"}
    actual_names = {x.name for x in output_dir.iterdir()}
    assert expected_names == actual_names


def test_joint_chipping(tmp_path, example_sicd, example_sidd):
    with example_sicd.open("rb") as f, sksicd.NitfReader(f) as r:
        sicd_xmltree = r.metadata.xmltree
        input_sicd_data = r.read_image()

        center_row = input_sicd_data.shape[0] // 2
        center_col = input_sicd_data.shape[1] // 2
        input_sicd_data[center_row - 10, center_col - 10] = (
            input_sicd_data["real"].max() - 1,
            input_sicd_data["imag"].max() - 1,
        )
        input_sicd_data[514, 519] = (
            input_sicd_data["real"].max(),
            input_sicd_data["imag"].max(),
        )

    sec = {"security": {"clas": "U"}}
    sicd_meta = sksicd.NitfMetadata(
        xmltree=sicd_xmltree,
        file_header_part={"ostaid": "nowhere"} | sec,
        im_subheader_part={"isorce": "this sensor"} | sec,
        de_subheader_part=sec,
    )

    tmp_sicd = tmp_path / "mod_example_sicd"
    with open(tmp_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta) as w:
        w.write_image(input_sicd_data)

    outdir = tmp_path / "out"
    outdir.mkdir()
    sarkit_assurance.joint_chip_to_html.main(
        [str(tmp_sicd), str(example_sidd), str(outdir)]
    )

    assert_output_dir(outdir)
    html = (outdir / "sicd_chip.html").read_text(encoding="utf-8")
    assert "514.00 519.00" in html


def test_joint_chipping_outside_valid_data(tmp_path, example_sicd, example_sidd):
    with example_sicd.open("rb") as f, sksicd.NitfReader(f) as r:
        sicd_xmltree = r.metadata.xmltree
        input_sicd_data = r.read_image()
        sicd_ew = sksicd.ElementWrapper(sicd_xmltree.getroot())

        center_row = input_sicd_data.shape[0] // 2
        center_col = input_sicd_data.shape[1] // 2
        input_sicd_data[center_row - 10, center_col - 10] = (
            input_sicd_data["real"].max() - 1,
            input_sicd_data["imag"].max() - 1,
        )
        input_sicd_data[500, 500] = (
            input_sicd_data["real"].max(),
            input_sicd_data["imag"].max(),
        )
        # Make point (500, 500) outside the SICD valid data so new max is (center_row-10, center_col-10)
        new_valid_data = np.array([[1400, 1189], [4000, 1800], [4000, 500]])
        sicd_ew["ImageData"]["ValidData"] = new_valid_data

    sec = {"security": {"clas": "U"}}
    sicd_meta = sksicd.NitfMetadata(
        xmltree=sicd_xmltree,
        file_header_part={"ostaid": "nowhere"} | sec,
        im_subheader_part={"isorce": "this sensor"} | sec,
        de_subheader_part=sec,
    )

    tmp_sicd = tmp_path / "mod_example_sicd"
    with open(tmp_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta) as w:
        w.write_image(input_sicd_data)

    outdir = tmp_path / "out"
    outdir.mkdir()
    sarkit_assurance.joint_chip_to_html.main(
        [str(tmp_sicd), str(example_sidd), str(outdir)]
    )

    assert_output_dir(outdir)
    html = (outdir / "sicd_chip.html").read_text(encoding="utf-8")
    assert f"{center_row - 10:.2f} {center_col - 10:.2f}" in html


def test_smart_open(tmp_path, example_sicd, example_sidd):
    with (
        tests.utils.static_http_server(example_sicd.parent) as sicd_server,
        tests.utils.static_http_server(example_sidd.parent) as sidd_server,
    ):
        sarkit_assurance.joint_chip_to_html.main(
            [
                f"{sicd_server}/{example_sicd.name}",
                f"{sidd_server}/{example_sidd.name}",
                str(tmp_path),
            ]
        )
        assert_output_dir(tmp_path)
