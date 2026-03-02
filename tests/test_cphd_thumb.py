import filecmp
import subprocess
import sys

import numpy as np
import pytest
import sarkit.cphd as skcphd
from PIL import Image

import tests.utils
from sarkit_assurance import names


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
                "--num-mebipixels",
                str(expected_max_num_pixels / 2**20),
            ],
        )
        img = Image.open(out_thumb)
        assert img.format == "PNG"
        assert img.mode == "L"  # 8 bit grayscale
        assert 0 < np.prod(img.size) <= expected_max_num_pixels


def test_multichan(tmp_path, multichan_cphd):
    with multichan_cphd.open("rb") as f, skcphd.Reader(f) as r:
        ch_ids = [
            x.text
            for x in r.metadata.xmltree.findall("{*}Data/{*}Channel/{*}Identifier")
        ]

    # don't specify ch_ids
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_thumb",
            str(multichan_cphd),
            str(tmp_path / "{ch_id}_auto.png"),
        ],
    )
    actual_files = set(x.name for x in tmp_path.iterdir())
    assert len(actual_files) == len(ch_ids)

    # specify ch_ids
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_thumb",
            str(multichan_cphd),
            "--channel-id",
            ch_ids[0],
            "--channel-id",
            ch_ids[2],
            str(tmp_path / "{ch_id}_manual.png"),
        ],
    )
    assert len(list(tmp_path.iterdir())) == len(ch_ids) + 2
    assert filecmp.cmp(
        (tmp_path / f"{names.sanitize_name(ch_ids[0])}_auto.png"),
        (tmp_path / f"{names.sanitize_name(ch_ids[0])}_manual.png"),
        shallow=False,
    )
    assert filecmp.cmp(
        (tmp_path / f"{names.sanitize_name(ch_ids[2])}_auto.png"),
        (tmp_path / f"{names.sanitize_name(ch_ids[2])}_manual.png"),
        shallow=False,
    )


def test_multichan_clobber(tmp_path, multichan_cphd):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.cphd_thumb",
                str(multichan_cphd),
                str(tmp_path / "missing_chid.png"),
            ],
        )
