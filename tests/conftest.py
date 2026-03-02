import copy
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


def make_cphd(tmp_path_factory, sig_format):
    cphd_etree = etree.parse(good_cphd_xml_path)
    ew = skcphd.ElementWrapper(cphd_etree.getroot())
    ew["Data"]["SignalArrayFormat"] = sig_format
    cphd_plan = skcphd.Metadata(
        xmltree=cphd_etree,
    )

    assert int(cphd_etree.findtext("{*}Data/{*}NumCPHDChannels")) == 1
    num_vectors = ew["Data"]["Channel"][0]["NumVectors"]
    num_samples = ew["Data"]["Channel"][0]["NumSamples"]
    sig_dtype = skcphd.binary_format_string_to_dtype(sig_format)
    signal = _random_array((num_vectors, num_samples), sig_dtype)
    tmp_cphd = (
        tmp_path_factory.mktemp("data") / good_cphd_xml_path.with_suffix(".cphd").name
    )
    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, cphd_plan) as cw:
        cw.write_signal("1", signal)
        # don't care about PVPs yet
    return tmp_cphd


@pytest.fixture(scope="session", params=["CI2", "CI4", "CF8"])
def example_cphd(tmp_path_factory, request):
    yield make_cphd(tmp_path_factory, request.param)


@pytest.fixture(scope="session")
def multichan_cphd(tmp_path_factory):
    onechan_cphd = make_cphd(tmp_path_factory, "CI2")

    with onechan_cphd.open("rb") as f, skcphd.Reader(f) as r:
        ch_id = r.metadata.xmltree.findtext(".//{*}RefChId")
        sig, pvps = r.read_channel(ch_id)

    newmeta = copy.deepcopy(r.metadata)
    cphdew = skcphd.ElementWrapper(newmeta.xmltree.getroot())

    # for simplicity, the data offsets will be shared across channels; which is not allowed by the spec
    num_chans = 3
    ch_ids = [ch_id]
    for chan in range(num_chans - 1):
        ch_ids.append(f"{ch_id}_copy/{chan}")
        new_datachan = copy.deepcopy(cphdew["Data"]["Channel"][0])
        new_datachan["Identifier"] = ch_ids[-1]
        cphdew["Data"].add("Channel", new_datachan)

        new_chparm = copy.deepcopy(cphdew["Channel"]["Parameters"][0])
        new_chparm["Identifier"] = new_datachan["Identifier"]
        cphdew["Data"].add("Channel", new_datachan)

    tmp_cphd = tmp_path_factory.mktemp("data") / "multichannel.cphd"

    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, newmeta) as cw:
        cw.write_pvp(ch_id, pvps)
        cw.write_signal(ch_id, sig)
    yield tmp_cphd
