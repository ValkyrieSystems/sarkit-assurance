import pathlib

import numpy as np
import pytest
import sarkit.cphd as skcphd
from lxml import etree

DATAPATH = pathlib.Path(__file__).parents[1] / "data"

good_cphd_xml_path = DATAPATH / "example-cphd-1.0.1.xml"


def _random_array(shape, dtype, reshape=True):
    rng = np.random.default_rng()
    retval = np.frombuffer(
        rng.bytes(np.prod(shape) * dtype.itemsize), dtype=dtype
    ).copy()

    def _zerofill(arr):
        if arr.dtype.names is None:
            arr[~np.isfinite(arr)] = 0
        else:
            for name in arr.dtype.names:
                _zerofill(arr[name])

    _zerofill(retval)
    return retval.reshape(shape) if reshape else retval


@pytest.fixture(scope="session", params=["CI2", "CI4", "CF8"])
def example_cphd(tmp_path_factory, request):
    cphd_etree = etree.parse(good_cphd_xml_path)
    ew = skcphd.ElementWrapper(cphd_etree.getroot())
    ew["Data"]["SignalArrayFormat"] = request.param
    cphd_plan = skcphd.Metadata(
        xmltree=cphd_etree,
    )

    assert int(cphd_etree.findtext("{*}Data/{*}NumCPHDChannels")) == 1
    num_vectors = ew["Data"]["Channel"][0]["NumVectors"]
    num_samples = ew["Data"]["Channel"][0]["NumSamples"]
    sig_dtype = skcphd.binary_format_string_to_dtype(request.param)
    signal = _random_array((num_vectors, num_samples), sig_dtype)
    tmp_cphd = (
        tmp_path_factory.mktemp("data") / good_cphd_xml_path.with_suffix(".cphd").name
    )
    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, cphd_plan) as cw:
        cw.write_signal("1", signal)
        # don't care about PVPs yet
    yield tmp_cphd
