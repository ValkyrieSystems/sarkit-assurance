import subprocess
import sys

import numpy as np
from PIL import Image

import tests.utils


def test_nominal(tmp_path, example_cphd):
    out_thumb = tmp_path / "out.png"
    expected_max_num_pixels = 2**10
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_thumb",
            str(example_cphd),
            out_thumb,
            "--channel-id",
            "1",
            "--num-mebipixels",
            str(expected_max_num_pixels / 2**20),
        ],
    )
    img = Image.open(out_thumb)
    assert img.format == "PNG"
    assert img.mode == "L"  # 8 bit grayscale
    assert 0 < np.prod(img.size) <= expected_max_num_pixels


def test_smart_open(tmp_path, example_cphd):
    with tests.utils.static_http_server(example_cphd.parent) as server_url:
        out_thumb = tmp_path / "out.png"
        expected_max_num_pixels = 2**10
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.cphd_thumb",
                f"{server_url}/{example_cphd.name}",
                str(out_thumb),
                "--channel-id",
                "1",
                "--num-mebipixels",
                str(expected_max_num_pixels / 2**20),
            ],
        )
        img = Image.open(out_thumb)
        assert img.format == "PNG"
        assert img.mode == "L"  # 8 bit grayscale
        assert 0 < np.prod(img.size) <= expected_max_num_pixels
