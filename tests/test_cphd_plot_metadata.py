import subprocess
import sys

import numpy as np
import pytest
import sarkit.cphd as skcphd

import sarkit_assurance.cphd_plot_metadata
import sarkit_assurance.names
import tests.utils


def test_main(tmp_path, multichan_cphd):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            "-q",
        ],
        cwd=tmp_path,
    )
    assert len(list(tmp_path.glob("*.html"))) > 0


def test_main_output_dir(tmp_path, multichan_cphd):
    outdir = tmp_path / "metadata_plots"
    assert not outdir.is_dir()
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(outdir),
            "-q",
        ],
        cwd=tmp_path,
    )
    assert outdir.is_dir()
    assert len(list(outdir.glob("*.html"))) > 0


def test_main_prefix(tmp_path, multichan_cphd):
    prefix = "expected_prefix_"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            "-q",
            "-p",
            prefix,
        ],
        cwd=tmp_path,
    )
    for file in tmp_path.glob("*.html"):
        assert file.name.startswith(prefix)


def test_main_concatenate(tmp_path, multichan_cphd):
    separate_dir = tmp_path / "separate"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(separate_dir),
            "-q",
        ]
    )
    assert len(list(separate_dir.glob("*.html"))) > 1

    concat_dir = tmp_path / "concatenated"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(concat_dir),
            "-qc",
        ]
    )
    assert len(list(concat_dir.glob("*.html"))) == 1


def test_main_plot_fixed(tmp_path, multichan_cphd):

    with open(multichan_cphd, "rb") as f, skcphd.Reader(f) as r:
        pvps = r.read_pvps(r.metadata.xmltree.findtext(".//{*}RefChId"))
    assert any(np.unique(pvps[name], axis=0).shape[0] == 1 for name in pvps.dtype.names)

    no_fixed_dir = tmp_path / "no_fixed"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(no_fixed_dir),
            "-q",
        ]
    )
    fixed_dir = tmp_path / "fixed"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(fixed_dir),
            "-q",
            "--plot-fixed",
        ]
    )
    assert len(list(fixed_dir.glob("*.html"))) > len(list(no_fixed_dir.glob("*.html")))


def test_main_channel_args(tmp_path, multichan_cphd):
    with open(multichan_cphd, "rb") as f, skcphd.Reader(f) as r:
        ref_channel = r.metadata.xmltree.findtext(".//{*}RefChId")
        all_channels = [
            x.text
            for x in r.metadata.xmltree.findall("{*}Data/{*}Channel/{*}Identifier")
        ]
    for chan_args, expected_channels in [
        ([], all_channels),
        (["--ref-chan"], [ref_channel]),
        ([f"--chan={ref_channel}"], [ref_channel]),
        (["--chan"] + all_channels, all_channels),
    ]:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.cphd_plot_metadata",
                str(multichan_cphd),
                str(tmp_path),
                "-q",
                "--prefix",
                "",
            ]
            + chan_args
        )
        expected_channels_sanitized = set(
            sarkit_assurance.names.sanitize_name(x) for x in expected_channels
        )
        assert all(
            file.stem.removeprefix("pvp_").startswith(
                tuple(expected_channels_sanitized)
            )
            for file in tmp_path.glob("pvp_*.html")
        )
        assert all(
            list(tmp_path.glob(f"pvp_{chan}*.html"))
            for chan in expected_channels_sanitized
        )


def test_main_bad_channel(tmp_path, multichan_cphd):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.cphd_plot_metadata",
                str(multichan_cphd),
                str(tmp_path),
                "-q",
                "--chan",
                "NOT_A_CHANNEL",
            ]
        )


def test_smart_open(tmp_path, multichan_cphd):
    with tests.utils.static_http_server(multichan_cphd.parent) as server_url:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.cphd_plot_metadata",
                f"{server_url}/{multichan_cphd.name}",
                str(tmp_path),
                "-q",
            ],
        )
    assert len(list(tmp_path.glob("*.html"))) > 0


def test_main_all_support_arrays(tmp_path, multichan_cphd):
    default_dir = tmp_path / "default"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(default_dir),
            "-qc",
            "--ref-chan",
        ]
    )

    all_dir = tmp_path / "all"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(all_dir),
            "-qc",
            "--ref-chan",
            "--all-support-arrays",
        ]
    )

    dta_chan_dir = tmp_path / "dta_chan"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.cphd_plot_metadata",
            str(multichan_cphd),
            str(dta_chan_dir),
            "-qc",
            "--chan=1_copy/1",
        ]
    )
    default_file = next(iter(default_dir.glob("*.html")))
    all_file = next(iter(all_dir.glob("*.html")))
    dta_chan_file = next(iter(dta_chan_dir.glob("*.html")))
    assert all_file.stat().st_size > default_file.stat().st_size
    assert dta_chan_file.stat().st_size > default_file.stat().st_size


def test_available_figures(multichan_cphd):
    with multichan_cphd.open("rb") as f:
        plotter = sarkit_assurance.cphd_plot_metadata.Plotter(f, multichan_cphd.name)
        available_figs = plotter.make_available_figures()
    all_plotters = {x.__name__ for x in plotter.plotters}
    assert not set(available_figs).difference(all_plotters)
    assert not set(all_plotters).difference(available_figs)
