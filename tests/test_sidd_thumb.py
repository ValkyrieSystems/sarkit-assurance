import filecmp
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import sarkit.sidd as sksidd
from PIL import Image

import tests.utils

DATAPATH = pathlib.Path(__file__).parents[1] / "data"


def make_thumb(in_sidd, out_thumb, expected_max_num_pixels=2**20, img_num=None):
    img_num_args = [f"--image-number={x}" for x in img_num] if img_num else []
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.sidd_thumb",
            str(in_sidd),
            str(out_thumb),
            "--num-mebipixels",
            str(expected_max_num_pixels / 2**20),
        ]
        + img_num_args,
    )


def get_expected_img_modes(sidd_file):
    with open(sidd_file, "rb") as f, sksidd.NitfReader(f) as r:
        img_metas = r.metadata.images

    def lookup_type(img_meta):
        px_type = img_meta.xmltree.findtext("{*}Display/{*}PixelType")
        lut_type = (
            None if img_meta.lookup_table is None else img_meta.lookup_table.dtype
        )
        return {
            ("MONO8I", None): "L",
            ("MONO8LU", np.dtype("uint8")): "L",
            ("MONO8LU", np.dtype("uint16")): "I;16",
            ("MONO16I", None): "I;16",
            ("RGB8LU", sksidd.PIXEL_TYPES["RGB24I"]["dtype"]): "RGB",
            ("RGB24I", None): "RGB",
        }[(px_type, lut_type)]

    return [lookup_type(x) for x in img_metas]


# try to trigger PIL thumbnail first stage resize and not
@pytest.mark.parametrize("expected_max_num_pixels", [2**10, 2**16])
def test_main(tmp_path, expected_max_num_pixels, multi_sidd):
    expected_img_modes = get_expected_img_modes(multi_sidd)

    make_thumb(multi_sidd, str(tmp_path / "thumb_{num}.png"), expected_max_num_pixels)
    with tests.utils.static_http_server(multi_sidd.parent) as server_url:
        make_thumb(
            f"{server_url}/{multi_sidd.name}",
            str(tmp_path / "remotethumb_{num}.png"),
            expected_max_num_pixels,
        )

    for index, expected_img_mode in enumerate(expected_img_modes):
        out_thumb = tmp_path / f"thumb_{index}.png"
        out_thumb_remote = tmp_path / f"remotethumb_{index}.png"
        assert filecmp.cmp(out_thumb, out_thumb_remote, shallow=False)

        img = Image.open(out_thumb)
        assert img.format == "PNG"
        assert 0 < np.prod(img.size) <= expected_max_num_pixels
        assert img.mode == expected_img_mode


def test_bad_image_number(tmp_path, multi_sidd):
    bad_img_num = len(get_expected_img_modes(multi_sidd)) * 10
    with pytest.raises(subprocess.CalledProcessError):
        make_thumb(multi_sidd, str(tmp_path / "{num}.png"), img_num=[0, bad_img_num])
    assert len(list(tmp_path.iterdir())) == 0


def test_multi_img(tmp_path, multi_sidd):
    expected_img_modes = get_expected_img_modes(multi_sidd)
    name_pattern = "{num}.png"
    # don't specify image_number
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    make_thumb(multi_sidd, auto_dir / name_pattern)

    expected_names = {
        name_pattern.format(num=x) for x in range(len(expected_img_modes))
    }
    actual_names = {x.name for x in auto_dir.iterdir()}
    assert expected_names == actual_names

    # specify image_number
    manual_img_nums = [0, 2, 4]
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    make_thumb(multi_sidd, manual_dir / name_pattern, img_num=manual_img_nums)

    expected_names = {name_pattern.format(num=x) for x in manual_img_nums}
    actual_names = {x.name for x in manual_dir.iterdir()}
    assert expected_names == actual_names

    for name in actual_names:
        assert filecmp.cmp(auto_dir / name, manual_dir / name, shallow=False)


def test_multi_img_clobber(tmp_path, multi_sidd):
    with pytest.raises(subprocess.CalledProcessError):
        make_thumb(multi_sidd, tmp_path / "no_num.png")
