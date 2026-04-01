import subprocess
import sys

import lxml.etree
import numpy as np
import numpy.testing as npt
import pytest
import sarkit.sicd as sksicd

import sarkit_assurance.sicd_plot_metadata as plot_metadata
import tests.utils


def test_smart_open(tmp_path, example_sicd):
    with tests.utils.static_http_server(example_sicd.parent) as server_url:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "sarkit_assurance.sicd_plot_metadata",
                f"{server_url}/{example_sicd.name}",
                str(tmp_path),
                "-q",
            ],
        )
    assert len(list(tmp_path.glob("*.html"))) > 0


@pytest.mark.parametrize("args", [[], ["--sample-data"]])
def test_main(tmp_path, args, sicd_xml):
    plot_metadata.main([str(sicd_xml), str(tmp_path), "-qc"] + args)
    assert len(list(tmp_path.glob("*.html"))) == 1


@pytest.mark.parametrize("args", [[], ["--sample-data"]])
def test_main_sicd(tmp_path, example_sicd, args):
    plot_metadata.main([str(example_sicd), str(tmp_path), "-qc"] + args)
    assert len(list(tmp_path.glob("*.html"))) == 1


def test_main_output_dir(tmp_path, sicd_xml):
    outdir = tmp_path / "metadata_plots"
    assert not outdir.is_dir()
    plot_metadata.main([str(sicd_xml), str(outdir), "-q"])
    assert outdir.is_dir()
    assert len(list(outdir.glob("*.html"))) > 0


def test_main_concatenate(tmp_path, sicd_xml):
    separate_dir = tmp_path / "separate"
    plot_metadata.main([str(sicd_xml), str(separate_dir), "-q"])
    assert len(list(separate_dir.glob("*.html"))) > 1

    concat_dir = tmp_path / "concatenated"
    plot_metadata.main([str(sicd_xml), str(concat_dir), "-qc"])
    assert len(list(concat_dir.glob("*.html"))) == 1


def test_main_prefix(tmp_path, sicd_xml):
    prefix = "expected_prefix_"
    plot_metadata.main([str(sicd_xml), str(tmp_path), "-q", "-p", prefix])
    for file in tmp_path.glob("*.html"):
        assert file.name.startswith(prefix)


def test_tgt_spatial_freq_support_missing_poly(sicd_xml, tmp_path):
    sicd_ew = sksicd.ElementWrapper(lxml.etree.parse(sicd_xml).getroot())
    del sicd_ew["Grid"]["Row"]["DeltaKCOAPoly"]
    del sicd_ew["Grid"]["Col"]["DeltaKCOAPoly"]
    tmp_sicd = tmp_path / "sicd.xml"
    tmp_sicd.write_bytes(lxml.etree.tostring(sicd_ew.elem))
    plt = plot_metadata.Plotter(tmp_sicd, "sicd.xml")
    result = plt.plot_spatial_frequency_support()
    assert result


def test_available_figures(example_sicd):
    with example_sicd.open("rb") as f:
        plotter = plot_metadata.Plotter(f, example_sicd.name, use_sample_data=True)
        available_figs = plotter.make_available_figures()
    all_plotters = {x.__name__ for x in plotter.plotters}
    assert not set(available_figs).difference(all_plotters)
    assert not set(all_plotters).difference(available_figs)


def test_downsample_last_dim():
    indata = np.array([[0, 10, 20], [1, 11, 21], [2, 12, 22], [3, 13, 23]])
    expect = np.array([[5, 20], [6, 21], [7, 22], [8, 23]])
    result = plot_metadata.downsample_last_dim(indata, 2)
    npt.assert_array_equal(result, expect)
