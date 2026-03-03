import copy
import filecmp
import pathlib
import subprocess
import sys

import lxml.etree
import numpy as np
import pytest
import sarkit.sidd as sksidd
from PIL import Image

import tests.utils

DATAPATH = pathlib.Path(__file__).parents[1] / "data"


def _image(sidd_xmltree):
    xml_helper = sksidd.XmlHelper(sidd_xmltree)
    rows = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
    cols = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")
    basis = Image.effect_mandelbrot((cols, rows), (-1.024, -0.768, 1.024, 0.768), 100)
    im = Image.merge("RGB", (basis, basis.rotate(120), basis.rotate(240)))
    return im


def make_thumb(in_sidd, img_num, out_thumb, expected_max_num_pixels):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.sidd_thumb",
            str(in_sidd),
            out_thumb,
            "--image-number",
            str(img_num),
            "--num-mebipixels",
            str(expected_max_num_pixels / 2**20),
        ],
    )


# try to trigger PIL thumbnail first stage resize and not
@pytest.mark.parametrize("expected_max_num_pixels", [2**10, 2**16])
def test_main(tmp_path, expected_max_num_pixels):
    test_sidd = tmp_path / "out.sidd"
    sidd_xml = DATAPATH / "example-sidd-3.0.0.xml"
    expected_img_modes = []

    # MONO8I
    basis_etree0 = lxml.etree.parse(sidd_xml)
    basis_array0 = np.asarray(_image(basis_etree0).convert(mode="L"))
    expected_img_modes.append("L")

    # MONO16I
    basis_etree1 = lxml.etree.parse(sidd_xml)
    basis_etree1.find("./{*}Display/{*}PixelType").text = "MONO16I"
    basis_array1 = (
        np.asarray(_image(basis_etree1).convert(mode="L")).astype(np.uint16) << 8
    )
    expected_img_modes.append("I;16")

    def _set_3_bands(tree):
        ew = sksidd.ElementWrapper(tree.getroot())

        try:
            ew["Display"]["NumBands"] = 3
        except KeyError:
            # SIDD 1.0
            return

        ew["Display"].add(
            "NonInteractiveProcessing",
            copy.deepcopy(ew["Display"]["NonInteractiveProcessing"][0]),
        )
        ew["Display"].add(
            "NonInteractiveProcessing",
            copy.deepcopy(ew["Display"]["NonInteractiveProcessing"][0]),
        )
        ew["Display"]["NonInteractiveProcessing"][1]["@band"] = "2"
        ew["Display"]["NonInteractiveProcessing"][2]["@band"] = "3"
        ew["Display"].add(
            "InteractiveProcessing",
            copy.deepcopy(ew["Display"]["InteractiveProcessing"][0]),
        )
        ew["Display"].add(
            "InteractiveProcessing",
            copy.deepcopy(ew["Display"]["InteractiveProcessing"][0]),
        )
        ew["Display"]["InteractiveProcessing"][1]["@band"] = "2"
        ew["Display"]["InteractiveProcessing"][2]["@band"] = "3"

    # RGB24I
    basis_etree2 = lxml.etree.parse(sidd_xml)
    basis_etree2.find("./{*}Display/{*}PixelType").text = "RGB24I"
    _set_3_bands(basis_etree2)
    basis_array2 = (
        np.asarray(_image(basis_etree2))
        .view(sksidd.PIXEL_TYPES["RGB24I"]["dtype"])
        .squeeze()
    )
    expected_img_modes.append("RGB")

    # RGB8LU
    basis_etree3 = lxml.etree.parse(sidd_xml)
    img3 = _image(basis_etree3).convert("P", palette=Image.Palette.ADAPTIVE)
    basis_array3 = np.asarray(img3)
    basis_etree3.find("./{*}Display/{*}PixelType").text = "RGB8LU"
    _set_3_bands(basis_etree3)
    lookup_table3 = (
        np.asarray(img3.getpalette())
        .astype(np.uint8)
        .reshape(-1, 3)
        .view(sksidd.PIXEL_TYPES["RGB24I"]["dtype"])
        .squeeze()
    )
    expected_img_modes.append("RGB")

    # MONO8LU - 8bit LUT
    basis_etree4 = lxml.etree.parse(sidd_xml)
    basis_array4 = np.asarray(_image(basis_etree4).convert(mode="L"))
    basis_etree4.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    lookup_table4 = np.arange(256, dtype=np.uint8)
    expected_img_modes.append("L")

    # MONO8LU - 16bit LUT
    basis_etree5 = lxml.etree.parse(sidd_xml)
    basis_array5 = np.asarray(_image(basis_etree5).convert(mode="L"))
    basis_etree5.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    lookup_table5 = (np.arange(256, dtype=np.uint16) << 8) + np.arange(
        256, dtype=np.uint16
    )[::-1]
    expected_img_modes.append("I;16")

    sec = sksidd.NitfSecurityFields(clas="U")
    write_metadata = sksidd.NitfMetadata(
        file_header_part=sksidd.NitfFileHeaderPart(ostaid="UNKNOWN", security=sec)
    )
    write_metadata.images.extend(
        [
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree0,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree1,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree2,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree3,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
                lookup_table=lookup_table3,
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree4,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
                lookup_table=lookup_table4,
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree5,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
                lookup_table=lookup_table5,
            ),
        ]
    )

    with test_sidd.open("wb") as file:
        with sksidd.NitfWriter(file, write_metadata) as writer:
            writer.write_image(0, basis_array0)
            writer.write_image(1, basis_array1)
            writer.write_image(2, basis_array2)
            writer.write_image(3, basis_array3)
            writer.write_image(4, basis_array4)
            writer.write_image(5, basis_array5)

    with tests.utils.static_http_server(test_sidd.parent) as server_url:
        for index in range(len(write_metadata.images)):
            out_thumb = tmp_path / f"thumb_{index}.png"
            make_thumb(test_sidd, index, out_thumb, expected_max_num_pixels)

            out_thumb_remote = tmp_path / f"remotethumb_{index}.png"
            make_thumb(
                f"{server_url}/{test_sidd.name}",
                index,
                out_thumb_remote,
                expected_max_num_pixels,
            )

            assert filecmp.cmp(out_thumb, out_thumb_remote, shallow=False)

            img = Image.open(out_thumb)
            assert img.format == "PNG"
            assert 0 < np.prod(img.size) <= expected_max_num_pixels
            assert img.mode == expected_img_modes[index]
