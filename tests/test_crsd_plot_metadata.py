import subprocess
import sys
import uuid

import pytest
import sarkit.crsd as skcrsd

import sarkit_assurance.crsd_plot_metadata
import tests.utils
from sarkit_assurance.crsd_plot_metadata import main


def test_main(tmp_path, example_crsdsar):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sarkit_assurance.crsd_plot_metadata",
            str(example_crsdsar),
            "-q",
        ],
        cwd=tmp_path,
    )
    assert len(list(tmp_path.glob("*.html"))) > 0


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
            "-qc",
        ]
    )
    assert len(list(concat_dir.glob("*.html"))) == 1


@pytest.mark.parametrize(
    "chan_args, expected_channels",
    [
        (
            [],
            ["027_056723_IW1", "027_056723_IW2", "027_056722_IW3"],
        ),
        (
            ["--ref-chan"],
            ["027_056723_IW2"],
        ),
        (
            ["--chan=027_056723_IW2"],
            ["027_056723_IW2"],
        ),
        (
            ["--chan", "027_056723_IW1", "027_056723_IW2", "027_056722_IW3"],
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
            ]
        )


def test_main_bad_arg_list(tmp_path, multi_crsdsar):
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


def test_smart_open(tmp_path, example_crsdsar):
    with tests.utils.static_http_server(example_crsdsar.parent) as server_url:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.crsd_plot_metadata",
                f"{server_url}/{example_crsdsar.name}",
                str(tmp_path),
                "-q",
            ],
        )
    assert len(list(tmp_path.glob("*.html"))) > 0


def test_available_figures(example_crsdsar):
    with example_crsdsar.open("rb") as f:
        plotter = sarkit_assurance.crsd_plot_metadata.Plotter(f, example_crsdsar.name)
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
        plotter = sarkit_assurance.crsd_plot_metadata.Plotter(f, crsd_with_dta.name)
    figs = plotter.plot_dwell()
    assert figs
