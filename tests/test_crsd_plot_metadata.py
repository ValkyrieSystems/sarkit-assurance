import subprocess
import sys
import uuid

import numpy as np
import pytest
import sarkit.crsd as skcrsd

import sarkit_assurance.crsd_plot_metadata
import tests.utils
from sarkit_assurance.crsd_plot_metadata import main


@pytest.mark.parametrize(
    ("fixture_name", "test_args"),
    [
        (
            "example_crsdsar",
            ["--ref-seq"],
        ),
        (
            "example_crsdsar",
            ["--ref-chan"],
        ),
        (
            "multi_crsdsar",
            ["--ref-seq", "--ref-chan"],
        ),
        (
            "multi_crsdrcv",
            ["--ref-chan"],
        ),
        (
            "multi_crsdtx",
            ["--ref-seq"],
        ),
    ],
)
def test_main(tmp_path, fixture_name, test_args, request):
    file = request.getfixturevalue(fixture_name)

    with file.open("rb"):
        assert not main([str(file), str(tmp_path), "-q"] + test_args)
    assert len(list(tmp_path.glob("*.html"))) > 0


@pytest.mark.parametrize(
    ("fixture_name", "test_args"),
    [
        (
            "multi_crsdrcv",
            ["--ref-seq", "--ref-chan"],
        ),
        (
            "multi_crsdrcv",
            ["--seq", "027_056723_IW2", "--ref-chan"],
        ),
        (
            "multi_crsdtx",
            ["--ref-chan", "--ref-seq"],
        ),
        (
            "multi_crsdtx",
            ["--ref-seq", "--chan", "027_056723_IW2"],
        ),
    ],
)
def test_main_bad_arg_list(tmp_path, fixture_name, test_args, request):
    file = request.getfixturevalue(fixture_name)

    with file.open("rb"):
        with pytest.raises(ValueError):
            main([str(file), str(tmp_path), "-q"] + test_args)


def test_main_output_dir(tmp_path, example_crsdsar):
    outdir = tmp_path / "metadata_plots"
    assert not outdir.is_dir()
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(example_crsdsar),
            str(outdir),
            "-q",
        ],
        cwd=tmp_path,
    )
    assert outdir.is_dir()
    assert len(list(outdir.glob("*.html"))) > 0


def test_main_prefix(tmp_path, example_crsdsar):
    prefix = "expected_prefix_"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(example_crsdsar),
            "--ref-chan",
            "--ref-seq",
            "-q",
            "-p",
            prefix,
        ],
        cwd=tmp_path,
    )
    for file in tmp_path.glob("*.html"):
        assert file.name.startswith(prefix)


def test_main_concatenate(tmp_path, example_crsdsar):
    separate_dir = tmp_path / "separate"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(example_crsdsar),
            str(separate_dir),
            "--ref-chan",
            "--ref-seq",
            "-q",
        ]
    )
    assert len(list(separate_dir.glob("*.html"))) > 1

    concat_dir = tmp_path / "concatenated"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(example_crsdsar),
            str(concat_dir),
            "--ref-chan",
            "--ref-seq",
            "-qc",
        ]
    )
    assert len(list(concat_dir.glob("*.html"))) == 1


@pytest.mark.parametrize(
    "chan_args, expected_channels",
    [
        (
            ["--ref-seq", "--ref-chan"],
            ["027_056723_IW2"],
        ),
        (
            ["--ref-seq", "--chan=027_056723_IW1"],
            ["027_056723_IW1", "027_056723_IW2"],
        ),
        (
            [
                "--ref-seq",
                "--chan",
                "027_056723_IW1",
                "027_056723_IW2",
                "027_056722_IW3",
            ],
            ["027_056723_IW1", "027_056723_IW2", "027_056722_IW3"],
        ),
    ],
)
def test_main_channel_args(tmp_path, multi_crsdsar, chan_args, expected_channels):
    with multi_crsdsar.open("rb"):
        assert not main(
            [str(multi_crsdsar), str(tmp_path), "-q", "--prefix", ""] + chan_args
        )
    expected_channels_sanitized = set(
        sarkit_assurance.names.sanitize_name(x) for x in expected_channels
    )
    assert all(
        file.stem.removeprefix("ref_tf_").startswith(tuple(expected_channels_sanitized))
        for file in tmp_path.glob("ref_tf_*.html")
    )
    assert all(
        file.stem.removeprefix("pvp_").startswith(tuple(expected_channels_sanitized))
        for file in tmp_path.glob("pvp_*.html")
    )


def test_main_bad_channel(tmp_path, multi_crsdsar):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.crsd_plot_metadata",
                str(multi_crsdsar),
                str(tmp_path),
                "--chan",
                "NOT_A_CHANNEL",
                "--ref-seq",
            ]
        )


def test_main_bad_chan_arg_list(tmp_path, multi_crsdsar):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.crsd_plot_metadata",
                str(multi_crsdsar),
                str(tmp_path),
                "--ref-chan",
                "--chan",
                "NOT_A_CHANNEL",
            ]
        )


@pytest.mark.parametrize(
    "seq_args, expected_sequences",
    [
        (
            ["--ref-seq", "--ref-chan"],
            ["027_056723_IW2"],
        ),
        (
            ["--seq=027_056722_IW3", "--ref-chan"],
            ["027_056723_IW2", "027_056722_IW3"],
        ),
        (
            [
                "--seq",
                "027_056723_IW1",
                "027_056723_IW2",
                "027_056722_IW3",
                "--ref-chan",
            ],
            ["027_056723_IW1", "027_056723_IW2", "027_056722_IW3"],
        ),
    ],
)
def test_main_sequence_args(tmp_path, multi_crsdsar, seq_args, expected_sequences):
    with multi_crsdsar.open("rb"):
        assert not main(
            [str(multi_crsdsar), str(tmp_path), "-q", "--prefix", ""] + seq_args
        )
    expected_sequences_sanitized = set(
        sarkit_assurance.names.sanitize_name(x) for x in expected_sequences
    )
    assert all(
        file.stem.removeprefix("ppp_").startswith(tuple(expected_sequences_sanitized))
        for file in tmp_path.glob("ppp_*.html")
    )


def test_main_bad_sequence(tmp_path, multi_crsdsar):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.crsd_plot_metadata",
                str(multi_crsdsar),
                str(tmp_path),
                "--seq",
                "NOT_A_SEQUENCE",
            ]
        )


def test_main_bad_seq_arg_list(tmp_path, multi_crsdsar):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.crsd_plot_metadata",
                str(multi_crsdsar),
                str(tmp_path),
                "--ref-seq",
                "--seq",
                "NOT_A_SEQUENCE",
            ]
        )


def test_main_plot_fixed_pvps(tmp_path, multi_crsdrcv):
    with open(multi_crsdrcv, "rb") as f, skcrsd.Reader(f) as r:
        pvps = r.read_pvps(r.metadata.xmltree.findtext(".//{*}RefChId"))
    assert any(np.unique(pvps[name], axis=0).shape[0] == 1 for name in pvps.dtype.names)

    no_fixed_dir = tmp_path / "no_fixed"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(multi_crsdrcv),
            str(no_fixed_dir),
            "--ref-chan",
            "-q",
        ]
    )
    fixed_dir = tmp_path / "fixed"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(multi_crsdrcv),
            str(fixed_dir),
            "--ref-chan",
            "-q",
            "--plot-fixed",
        ]
    )
    assert len(list(fixed_dir.glob("*.html"))) > len(list(no_fixed_dir.glob("*.html")))


def test_main_plot_fixed_ppps(tmp_path, multi_crsdtx):
    with open(multi_crsdtx, "rb") as f, skcrsd.Reader(f) as r:
        ppps = r.read_ppps(r.metadata.xmltree.findtext(".//{*}RefTxId"))
    assert any(np.unique(ppps[name], axis=0).shape[0] == 1 for name in ppps.dtype.names)

    no_fixed_dir = tmp_path / "no_fixed"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(multi_crsdtx),
            str(no_fixed_dir),
            "--ref-seq",
            "-q",
        ]
    )
    fixed_dir = tmp_path / "fixed"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(multi_crsdtx),
            str(fixed_dir),
            "--ref-seq",
            "-q",
            "--plot-fixed",
        ]
    )
    assert len(list(fixed_dir.glob("*.html"))) > len(list(no_fixed_dir.glob("*.html")))


def test_smart_open(tmp_path, example_crsdsar):
    with tests.utils.static_http_server(example_crsdsar.parent) as server_url:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.crsd_plot_metadata",
                f"{server_url}/{example_crsdsar.name}",
                str(tmp_path),
                "--ref-chan",
                "--ref-seq",
                "-q",
            ],
        )
    assert len(list(tmp_path.glob("*.html"))) > 0


def test_available_figures(example_crsdsar):
    with example_crsdsar.open("rb") as f:
        plotter = sarkit_assurance.crsd_plot_metadata.Plotter(
            f,
            example_crsdsar.name,
            channels=["SyntheticChannel"],
            sequences=["SyntheticChannel"],
        )
        available_figs = plotter.make_available_figures()
    all_plotters = {x.__name__ for x in plotter.plotters}
    assert not set(available_figs).difference(all_plotters)
    assert not set(all_plotters).difference(available_figs)


def test_plot_dwell_dta(tmp_path, example_crsdsar):
    with example_crsdsar.open("rb") as f, skcrsd.Reader(f) as r:
        assert (
            r.metadata.xmltree.find(".//{*}SARImage/{*}DwellTimes/{*}Polynomials")
            is not None
        )

        dta_id = str(uuid.uuid4())
        elem_format = "COD=F4;DT=F4;"
        crsd_with_dta = skcrsd.ElementWrapper(r.metadata.xmltree.getroot())
        for chan_param in crsd_with_dta["Channel"]["Parameters"]:
            chan_param["SARImage"]["DwellTimes"] = {"Array": {"DTAId": dta_id}}
        crsd_with_dta["Data"]["Support"].add(
            "SupportArray",
            {
                "SAId": dta_id,
                "NumRows": 24,
                "NumCols": 8,
                "BytesPerElement": 8,
                "ArrayByteOffset": 0,  # doesn't matter for this
            },
        )
        crsd_with_dta["SupportArray"].add(
            "DwellTimeArray",
            {
                "Identifier": dta_id,
                "ElementFormat": elem_format,
                "X0": 0,
                "Y0": 0,
                "XSS": 1,
                "YSS": 1,
            },
        )
        crsd_with_dta = tmp_path / "has_dta.crsd"
        sequence_id = r.metadata.xmltree.findtext(
            "{*}TxSequence/{*}Parameters/{*}Identifier"
        )
        channel_id = r.metadata.xmltree.findtext(
            "{*}Channel/{*}Parameters/{*}Identifier"
        )
        with crsd_with_dta.open("wb") as fw, skcrsd.Writer(fw, r.metadata) as w:
            w.write_ppp(sequence_id, r.read_ppps(sequence_id))
            w.write_pvp(channel_id, r.read_pvps(channel_id))

    with crsd_with_dta.open("rb") as f:
        plotter = sarkit_assurance.crsd_plot_metadata.Plotter(
            f,
            crsd_with_dta.name,
            channels=["SyntheticChannel"],
            sequences=["SyntheticChannel"],
        )
    figs = plotter.plot_dwell()
    assert figs
