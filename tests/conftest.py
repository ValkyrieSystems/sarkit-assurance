import copy
import pathlib

import lxml.etree
import numpy as np
import pytest
import sarkit.cphd as skcphd
import sarkit.sidd as sksidd
import scipy.constants
from lxml import etree
from PIL import Image

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
    xmlhelp = skcphd.XmlHelper(cphd_etree)
    ew["Data"]["SignalArrayFormat"] = sig_format
    cphd_plan = skcphd.Metadata(
        xmltree=cphd_etree,
    )

    assert int(cphd_etree.findtext("{*}Data/{*}NumCPHDChannels")) == 1
    num_vectors = ew["Data"]["Channel"][0]["NumVectors"]
    num_samples = ew["Data"]["Channel"][0]["NumSamples"]

    # Make signal array
    sig_dtype = skcphd.binary_format_string_to_dtype(sig_format)
    signal = _random_array((num_vectors, num_samples), sig_dtype)

    # Make PVPs
    pvp_dtype = skcphd.get_pvp_dtype(cphd_etree)
    pvps = np.zeros((num_vectors), dtype=pvp_dtype)
    pvps["TxTime"] = np.linspace(
        xmlhelp.load(".//{*}TxTime1"),
        xmlhelp.load(".//{*}TxTime2"),
        num_vectors,
        endpoint=True,
    )
    arppos = xmlhelp.load(".//{*}ARPPos")
    arpvel = xmlhelp.load(".//{*}ARPVel")
    t_ref = xmlhelp.load(".//{*}ReferenceTime")

    arppoly = np.stack([(arppos - t_ref * arpvel), arpvel])

    fx1 = xmlhelp.load(".//{*}FxMin")
    fx2 = xmlhelp.load(".//{*}FxMax")
    pvps["FX1"][:] = fx1
    pvps["FX2"][:] = fx2
    pvps["SC0"] = fx1
    pvps["SCSS"] = (fx2 - fx1) / (num_samples - 1)
    pvps["TOA1"][:] = xmlhelp.load(".//{*}TOAMin")
    pvps["TOA2"][:] = xmlhelp.load(".//{*}TOAMax")

    pvps["TxPos"] = np.polynomial.polynomial.polyval(pvps["TxTime"], arppoly).T
    pvps["TxVel"] = np.polynomial.polynomial.polyval(
        pvps["TxTime"], np.polynomial.polynomial.polyder(arppoly)
    ).T

    pvps["RcvTime"] = (
        pvps["TxTime"]
        + 2.0 * xmlhelp.load(".//{*}SlantRange") / scipy.constants.speed_of_light
    )
    pvps["RcvPos"] = np.polynomial.polynomial.polyval(pvps["RcvTime"], arppoly).T
    pvps["RcvVel"] = np.polynomial.polynomial.polyval(
        pvps["RcvTime"], np.polynomial.polynomial.polyder(arppoly)
    ).T

    srp = xmlhelp.load(".//{*}SRP/{*}ECF")
    pvps["SRPPos"] = srp

    pvps["SIGNAL"] = 1

    tmp_cphd = (
        tmp_path_factory.mktemp("data") / good_cphd_xml_path.with_suffix(".cphd").name
    )
    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, cphd_plan) as cw:
        cw.write_signal("1", signal)
        cw.write_pvp("1", pvps)
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
        cphdew["Channel"].add("Parameters", new_chparm)

    # make channel image areas non-contiguous
    for index, chan_params in enumerate(cphdew["Channel"]["Parameters"]):
        low = -500 + 200 * index
        high = low + 100
        chan_params["ImageArea"]["X1Y1"] = (low, low)
        chan_params["ImageArea"]["X2Y2"] = (high, high)
        chan_params["ImageArea"]["Polygon"] = [
            (low, low),
            (low, high),
            (high, high),
            (high, low),
        ]

    tmp_cphd = tmp_path_factory.mktemp("data") / "multichannel.cphd"

    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, newmeta) as cw:
        cw.write_pvp(ch_id, pvps)
        cw.write_signal(ch_id, sig)
    yield tmp_cphd


def _image(sidd_xmltree):
    xml_helper = sksidd.XmlHelper(sidd_xmltree)
    rows = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Row")
    cols = xml_helper.load("./{*}Measurement/{*}PixelFootprint/{*}Col")
    basis = Image.effect_mandelbrot((cols, rows), (-1.024, -0.768, 1.024, 0.768), 100)
    im = Image.merge("RGB", (basis, basis.rotate(120), basis.rotate(240)))
    return im


@pytest.fixture(scope="session")
def multi_sidd(tmp_path_factory):
    sidd_xml = DATAPATH / "example-sidd-3.0.0.xml"
    expected_img_modes = []

    # MONO8I
    basis_etree0 = lxml.etree.parse(sidd_xml)
    basis_array0 = np.asarray(_image(basis_etree0).convert(mode="L"))
    expected_img_modes.append("L")

    # MONO16I
    basis_etree1 = lxml.etree.parse(sidd_xml)
    basis_etree1.find("./{*}Display/{*}PixelType").text = "MONO16I"
    basis_array1 = (
        np.asarray(_image(basis_etree1).convert(mode="L")).astype(np.uint16) << 4
    )
    expected_img_modes.append("I;16")

    def _set_3_bands(tree):
        ew = sksidd.ElementWrapper(tree.getroot())

        try:
            ew["Display"]["NumBands"] = 3
        except KeyError:
            # SIDD 1.0
            return

        ew["Display"].add(
            "NonInteractiveProcessing",
            copy.deepcopy(ew["Display"]["NonInteractiveProcessing"][0]),
        )
        ew["Display"].add(
            "NonInteractiveProcessing",
            copy.deepcopy(ew["Display"]["NonInteractiveProcessing"][0]),
        )
        ew["Display"]["NonInteractiveProcessing"][1]["@band"] = "2"
        ew["Display"]["NonInteractiveProcessing"][2]["@band"] = "3"
        ew["Display"].add(
            "InteractiveProcessing",
            copy.deepcopy(ew["Display"]["InteractiveProcessing"][0]),
        )
        ew["Display"].add(
            "InteractiveProcessing",
            copy.deepcopy(ew["Display"]["InteractiveProcessing"][0]),
        )
        ew["Display"]["InteractiveProcessing"][1]["@band"] = "2"
        ew["Display"]["InteractiveProcessing"][2]["@band"] = "3"

    # RGB24I
    basis_etree2 = lxml.etree.parse(sidd_xml)
    basis_etree2.find("./{*}Display/{*}PixelType").text = "RGB24I"
    _set_3_bands(basis_etree2)
    basis_array2 = (
        np.asarray(_image(basis_etree2))
        .view(sksidd.PIXEL_TYPES["RGB24I"]["dtype"])
        .squeeze()
    )
    expected_img_modes.append("RGB")

    # RGB8LU
    basis_etree3 = lxml.etree.parse(sidd_xml)
    img3 = _image(basis_etree3).convert("P", palette=Image.Palette.ADAPTIVE)
    basis_array3 = np.asarray(img3)
    basis_etree3.find("./{*}Display/{*}PixelType").text = "RGB8LU"
    _set_3_bands(basis_etree3)
    lookup_table3 = (
        np.asarray(img3.getpalette())
        .astype(np.uint8)
        .reshape(-1, 3)
        .view(sksidd.PIXEL_TYPES["RGB24I"]["dtype"])
        .squeeze()
    )
    expected_img_modes.append("RGB")

    # MONO8LU - 8bit LUT
    basis_etree4 = lxml.etree.parse(sidd_xml)
    basis_array4 = np.asarray(_image(basis_etree4).convert(mode="L"))
    basis_etree4.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    lookup_table4 = np.arange(256, dtype=np.uint8)
    expected_img_modes.append("L")

    # MONO8LU - 16bit LUT
    basis_etree5 = lxml.etree.parse(sidd_xml)
    basis_array5 = np.asarray(_image(basis_etree5).convert(mode="L"))
    basis_etree5.find("./{*}Display/{*}PixelType").text = "MONO8LU"
    lookup_table5 = np.arange(256, dtype=np.uint16) << 4
    expected_img_modes.append("I;16")

    sec = sksidd.NitfSecurityFields(clas="U")
    write_metadata = sksidd.NitfMetadata(
        file_header_part=sksidd.NitfFileHeaderPart(ostaid="UNKNOWN", security=sec)
    )
    write_metadata.images.extend(
        [
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree0,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree1,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree2,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree3,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
                lookup_table=lookup_table3,
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree4,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
                lookup_table=lookup_table4,
            ),
            sksidd.NitfProductImageMetadata(
                xmltree=basis_etree5,
                im_subheader_part=sksidd.NitfImSubheaderPart(security=sec),
                de_subheader_part=sksidd.NitfDeSubheaderPart(security=sec),
                lookup_table=lookup_table5,
            ),
        ]
    )

    tmp_sidd = tmp_path_factory.mktemp("data") / "multi.sidd"
    with tmp_sidd.open("wb") as file:
        with sksidd.NitfWriter(file, write_metadata) as writer:
            writer.write_image(0, basis_array0)
            writer.write_image(1, basis_array1)
            writer.write_image(2, basis_array2)
            writer.write_image(3, basis_array3)
            writer.write_image(4, basis_array4)
            writer.write_image(5, basis_array5)
    yield tmp_sidd
