import argparse

import lxml.etree
import pytest
import sarkit.cphd as skcphd

from sarkit_assurance import _cli

CHAN_IDS = ["a", "b", "c", "d", "e"]


@pytest.fixture
def multichan_cphdxml():
    """Make a minimal, non-schema-conforming CPHD XML with channel identifiers"""
    ns = next(iter(skcphd.VERSION_INFO))
    ew = skcphd.ElementWrapper(lxml.etree.Element(lxml.etree.QName(ns, "CPHD")))
    ew["Channel"]["RefChId"] = CHAN_IDS[0]
    ew["Channel"]["Parameters"] = [{"Identifier": n} for n in CHAN_IDS]
    ew["Data"]["NumCPHDChannels"] = len(CHAN_IDS)
    ew["Data"]["Channel"] = [{"Identifier": n} for n in CHAN_IDS]
    return ew.elem.getroottree()


@pytest.mark.parametrize(
    "chan_args,expected_chids",
    [
        ([], CHAN_IDS),
        (["--ref-chan"], ["a"]),
        (["--chan", "a", "--ref-chan"], ["a"]),
        (["--chan", "b", "c", "d"], ["b", "c", "d"]),
        (["--chan", "b", "c", "b", "c", "--ref-chan"], ["a", "b", "c"]),
    ],
)
def test_cphd_chan_args(chan_args, expected_chids, multichan_cphdxml):
    parser = argparse.ArgumentParser()
    _cli.add_cphd_chan_arg_group(parser)
    config = parser.parse_args(chan_args)
    actual_chids = _cli.selected_cphd_channels(multichan_cphdxml, config)
    assert actual_chids == expected_chids


def test_cphd_chan_args_bad(multichan_cphdxml):
    parser = argparse.ArgumentParser()
    _cli.add_cphd_chan_arg_group(parser)
    not_a_chid = "INVALID"
    config = parser.parse_args(["--chan", not_a_chid])
    with pytest.raises(ValueError, match=f"Unrecognized.*{not_a_chid}"):
        _cli.selected_cphd_channels(multichan_cphdxml, config)
