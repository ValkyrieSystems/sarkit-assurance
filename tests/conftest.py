import copy
import pathlib

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import pytest
import sarkit.cphd as skcphd
import sarkit.crsd as skcrsd
import sarkit.sicd as sksicd
import sarkit.sidd as sksidd
import scipy.constants
from PIL import Image

DATAPATH = pathlib.Path(__file__).parents[1] / "data"

good_cphd_xml_path = DATAPATH / "example-cphd-1.1.0.xml"
good_crsd_xml_path = DATAPATH / "example-crsd-1.0.xml"
multi_crsdsar_xml_path = DATAPATH / "example-multi-crsdsar-1.0.xml"
good_sicd_xml_path = DATAPATH / "example-sicd-1.4.0.xml"


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


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


def _remove(root, pattern):
    if (elem := root.find(pattern)) is not None:
        elem.getparent().remove(elem)
    else:
        print(f"Cannot find {pattern=}")


def _replace_error(crsd_etree, sensor_type):
    sar_error = crsd_etree.find("{*}ErrorParameters/{*}SARImage")
    elem_ns = lxml.etree.QName(sar_error).namespace
    retval = copy.deepcopy(sar_error.find("{*}Monostatic"))
    retval.tag = f"{{{elem_ns}}}{sensor_type}Sensor"
    sar_error.addnext(retval)
    helper = skcrsd.XmlHelper(crsd_etree)
    ndx = {"Tx": 0, "Rcv": 1}[sensor_type]
    helper.set_elem(
        retval.find(".//{*}TimeFreqCov"),
        skcrsd.MtxType((3, 3)).parse_elem(retval.find(".//{*}TimeFreqCov"))[
            [ndx, 2], :
        ][:, [ndx, 2]],
    )
    time_decorr = copy.deepcopy(retval.find(f".//{{*}}{{{sensor_type}}}TimeDecorr"))
    if time_decorr is not None:
        time_decorr.tag = f"{{{elem_ns}}}TimeDecorr"
        _remove(retval, "{*}TxTimeDecorr")
        _remove(retval, "{*}RcvTimeDecorr")
        retval.find(".//{*}ClockFreqDecorr").addprevious(time_decorr)
    sar_error.getparent().remove(sar_error)


def _repack_support_arrays(crsd_etree):
    offset = 0
    for array in crsd_etree.findall("{*}Data/{*}Support/{*}SupportArray"):
        array.find("{*}ArrayByteOffset").text = str(offset)
        offset += (
            int(array.findtext("{*}NumRows"))
            * int(array.findtext("{*}NumCols"))
            * int(array.findtext("{*}BytesPerElement"))
        )
    return offset


def make_cphd(tmp_path_factory, sig_format):
    cphd_etree = lxml.etree.parse(good_cphd_xml_path)
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

    # make  support array
    sa_dtype = skcphd.binary_format_string_to_dtype("Gain=F4;Phase=F4;")
    sa = np.zeros((21, 21), sa_dtype)
    sa[:, 0] = (np.pi, np.e)
    sa[0, :] = (np.pi, np.e)
    sa[10, :] = (np.pi, np.e)

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

    tx_acz = unit(srp - pvps["TxPos"])
    rcv_acz = unit(srp - pvps["RcvPos"])
    tx_acx = unit(np.cross(pvps["TxVel"], tx_acz))
    rcv_acx = unit(np.cross(pvps["RcvVel"], rcv_acz))
    tx_acy = unit(np.cross(tx_acz, tx_acx))
    rcv_acy = unit(np.cross(rcv_acz, rcv_acx))
    pvps["TxACY"] = tx_acy
    pvps["TxACX"] = tx_acx
    pvps["TxEB"] = 0
    pvps["RcvACY"] = rcv_acy
    pvps["RcvACX"] = rcv_acx
    pvps["RcvEB"] = 0

    pvps["SIGNAL"] = 1

    tmp_cphd = (
        tmp_path_factory.mktemp("data") / good_cphd_xml_path.with_suffix(".cphd").name
    )
    with open(tmp_cphd, "wb") as f, skcphd.Writer(f, cphd_plan) as cw:
        cw.write_signal("1", signal)
        cw.write_pvp("1", pvps)
        cw.write_support_array("transmit_array", sa)
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
        sa = r.read_support_array("transmit_array")

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

    # attach DwellTimeArray to single channel only
    new_chparm["DwellTimes"]["DTAId"] = "dwell_array"

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
        cw.write_support_array("transmit_array", sa)
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


@pytest.fixture
def sicd_xml():
    return good_sicd_xml_path


@pytest.fixture(scope="session")
def example_sicd(tmp_path_factory):
    sicd_etree = lxml.etree.parse(good_sicd_xml_path)
    tmp_sicd = (
        tmp_path_factory.mktemp("data") / good_sicd_xml_path.with_suffix(".sicd").name
    )
    sec = {"security": {"clas": "U"}}
    sicd_meta = sksicd.NitfMetadata(
        xmltree=sicd_etree,
        file_header_part={"ostaid": "nowhere"} | sec,
        im_subheader_part={"isorce": "this sensor"} | sec,
        de_subheader_part=sec,
    )
    nrows = int(sicd_etree.findtext("{*}ImageData/{*}NumRows"))
    ncols = int(sicd_etree.findtext("{*}ImageData/{*}NumCols"))
    pixel_type = sicd_etree.findtext("{*}ImageData/{*}PixelType")
    dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"]
    with open(tmp_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta) as w:
        w.write_image(_random_array((nrows, ncols), dtype))
    yield tmp_sicd


@pytest.fixture(scope="session")
def example_crsdsar(tmp_path_factory):
    crsd_etree = lxml.etree.parse(good_crsd_xml_path)
    xmlhelp = skcrsd.XmlHelper(crsd_etree)

    pvp_dtype = skcrsd.get_pvp_dtype(crsd_etree)

    assert crsd_etree.findtext("./{*}Data/{*}Receive/{*}SignalArrayFormat") == "CI2"
    signal_dtype = skcrsd.binary_format_string_to_dtype(
        crsd_etree.findtext("./{*}Data/{*}Receive/{*}SignalArrayFormat")
    )
    rng = np.random.default_rng(123456)
    num_pulses = xmlhelp.load("./{*}Data/{*}Transmit/{*}TxSequence/{*}NumPulses")
    num_vectors = xmlhelp.load("./{*}Data/{*}Receive/{*}Channel/{*}NumVectors")
    num_samples = xmlhelp.load("./{*}Data/{*}Receive/{*}Channel/{*}NumSamples")
    signal = (
        rng.integers(-128, 127, (num_vectors, num_samples, 2), dtype=np.int8)
        .view(signal_dtype)
        .squeeze()
    )

    pvps = np.zeros((num_vectors), dtype=pvp_dtype)
    ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(crsd_etree))
    tx_ref_time = xmlhelp.load("{*}ReferenceGeometry/{*}TxParameters/{*}Time")
    txtime = np.interp(
        np.arange(num_pulses),
        [
            0,
            xmlhelp.load(".//{*}RefVectorPulseIndex"),
            num_pulses - 1,
        ],
        [
            xmlhelp.load(".//{*}TxTime1"),
            tx_ref_time,
            xmlhelp.load(".//{*}TxTime2"),
        ],
    )
    ppps["TxTime"]["Int"] = np.floor(txtime)
    ppps["TxTime"]["Frac"] = txtime % 1

    txpos = xmlhelp.load("{*}ReferenceGeometry/{*}TxParameters/{*}APCPos")
    txvel = xmlhelp.load("{*}ReferenceGeometry/{*}TxParameters/{*}APCVel")

    tx_pos_poly = np.stack([(txpos - tx_ref_time * txvel), txvel])

    fx1 = xmlhelp.load(".//{*}FxMin")
    fx2 = xmlhelp.load(".//{*}FxMax")
    tx_bw = xmlhelp.load(".//{*}FxBW")
    ppps["FX1"][:] = fx1
    ppps["FX2"][:] = fx2
    ppps["TXmt"][:] = xmlhelp.load(".//{*}TXmtMin")
    ppps["TxRadInt"][:] = xmlhelp.load(".//{*}TxRefRadIntensity")
    ppps["FxRate"] = tx_bw / ppps["TXmt"]
    ppps["FxFreq0"][:] = xmlhelp.load(".//{*}FxC")

    ppps["TxPos"] = np.polynomial.polynomial.polyval(txtime, tx_pos_poly).T
    ppps["TxPos"][xmlhelp.load(".//{*}RefVectorPulseIndex")] = txpos
    ppps["TxVel"] = txvel

    tx_refpt = xmlhelp.load(".//{*}TxRefPoint/{*}ECF")
    tx_acz = unit(tx_refpt - ppps["TxPos"])
    tx_acx = unit(np.cross(ppps["TxVel"], tx_acz))
    tx_acy = unit(np.cross(tx_acz, tx_acx))
    ppps["TxACX"] = tx_acx
    ppps["TxACY"] = tx_acy
    ppps["TxEB"] = 0

    rcvstart = np.interp(
        np.arange(num_vectors),
        [
            0,
            xmlhelp.load(".//{*}RefVectorIndex"),
            num_vectors - 1,
        ],
        [
            xmlhelp.load(".//{*}RcvStartTime1"),
            xmlhelp.load("./{*}ReferenceGeometry/{*}RcvParameters/{*}Time"),
            xmlhelp.load(".//{*}RcvStartTime2"),
        ],
    )
    fs = xmlhelp.load("{*}Channel/{*}Parameters/{*}Fs")
    rcvstart = np.round((rcvstart - rcvstart[0]) * fs) / fs + rcvstart[0]
    pvps["RcvStart"]["Int"] = np.floor(rcvstart)
    pvps["RcvStart"]["Frac"] = rcvstart % 1

    rcvpos = xmlhelp.load("{*}ReferenceGeometry/{*}RcvParameters/{*}APCPos")
    rcvvel = xmlhelp.load("{*}ReferenceGeometry/{*}RcvParameters/{*}APCVel")
    rcv_ref_time = xmlhelp.load("{*}ReferenceGeometry/{*}RcvParameters/{*}Time")

    rcv_pos_poly = np.stack([(rcvpos - rcv_ref_time * rcvvel), rcvvel])
    pvps["RcvPos"] = np.polynomial.polynomial.polyval(rcvstart, rcv_pos_poly).T
    pvps["RcvPos"][xmlhelp.load(".//{*}RefVectorIndex")] = rcvpos
    pvps["RcvVel"] = rcvvel
    pvps["SIGNAL"] = 1
    pvps["RefFreq"] = xmlhelp.load("{*}Channel/{*}Parameters/{*}F0Ref")
    pvps["TxPulseIndex"] = np.arange(pvps.size)
    pvps["FRCV1"] = xmlhelp.load(".//{*}FrcvMin")
    pvps["FRCV2"] = xmlhelp.load(".//{*}FrcvMax")
    pvps["AmpSF"] = 1.0
    pvps["DFIC0"][1] = -10
    pvps["FICRate"][1] = 10

    rcv_refpt = xmlhelp.load(".//{*}RcvRefPoint/{*}ECF")
    rcv_acz = unit(rcv_refpt - pvps["RcvPos"])
    rcv_acx = unit(np.cross(pvps["RcvVel"], rcv_acz))
    rcv_acy = unit(np.cross(rcv_acz, rcv_acx))
    pvps["RcvACX"] = rcv_acx
    pvps["RcvACY"] = rcv_acy
    pvps["RcvEB"] = 0

    tmp_crsd = (
        tmp_path_factory.mktemp("data") / good_crsd_xml_path.with_suffix(".crsd").name
    )
    sequence_id = crsd_etree.findtext("{*}TxSequence/{*}Parameters/{*}Identifier")
    channel_id = crsd_etree.findtext("{*}Channel/{*}Parameters/{*}Identifier")
    new_meta = skcrsd.Metadata(
        xmltree=crsd_etree,
    )
    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        cw.write_ppp(sequence_id, ppps)
        cw.write_pvp(channel_id, pvps)
        cw.write_signal(channel_id, signal)
    yield tmp_crsd


@pytest.fixture(scope="session")
def multi_crsdsar(tmp_path_factory):
    crsd_etree = lxml.etree.parse(multi_crsdsar_xml_path)
    root = crsd_etree.getroot()
    crsd_ew = skcrsd.ElementWrapper(root)
    pvp_dtype = skcrsd.get_pvp_dtype(crsd_etree)
    assert crsd_ew["Data"]["Receive"]["SignalArrayFormat"] == "CI2"
    signal_dtype = skcrsd.binary_format_string_to_dtype(
        crsd_ew["Data"]["Receive"]["SignalArrayFormat"]
    )
    rng = np.random.default_rng(123456)

    ref_ch_id = crsd_ew["Channel"]["RefChId"]
    channel_ids = [x.text for x in root.findall(".//{*}ChId")]
    tx_ref_time = crsd_ew["ReferenceGeometry"]["TxParameters"]["Time"]
    txpos = crsd_ew["ReferenceGeometry"]["TxParameters"]["APCPos"]
    txvel = crsd_ew["ReferenceGeometry"]["TxParameters"]["APCVel"]
    tx_pos_poly = np.stack([(txpos - tx_ref_time * txvel), txvel])
    rcv_ref_time = crsd_ew["ReferenceGeometry"]["RcvParameters"]["Time"]
    rcvpos = crsd_ew["ReferenceGeometry"]["RcvParameters"]["APCPos"]
    rcvvel = crsd_ew["ReferenceGeometry"]["RcvParameters"]["APCVel"]
    rcv_pos_poly = np.stack([(rcvpos - rcv_ref_time * rcvvel), rcvvel])
    ant = crsd_ew["Antenna"]

    def compute_vh_pol(pos, acx, acy, ref_pt, ant_pol_ref, txrcv_pol_ref):
        xr = 1 if txrcv_pol_ref.elem.tag.endswith("TxPolarization") else -1
        amph, ampv, phaseh, phasev = skcrsd.compute_h_v_pol_parameters(
            pos,
            acx,
            acy,
            ref_pt,
            xr,
            ant_pol_ref["AmpX"],
            ant_pol_ref["AmpY"],
            ant_pol_ref["PhaseX"],
            ant_pol_ref["PhaseY"],
        )
        txrcv_pol_ref["AmpH"] = amph
        txrcv_pol_ref["AmpV"] = ampv
        txrcv_pol_ref["PhaseH"] = phaseh
        txrcv_pol_ref["PhaseV"] = phasev

    # Loop over channels
    tmp_crsd = (
        tmp_path_factory.mktemp("data")
        / multi_crsdsar_xml_path.with_suffix(".crsd").name
    )
    new_meta = skcrsd.Metadata(xmltree=crsd_etree)
    pxp_dict_list = []
    for ch_id in channel_ids:
        chan_param = crsd_ew["Channel"].find("Parameters", Identifier=ch_id)
        tx_id = chan_param["SARImage"]["TxId"]
        txseq_param = crsd_ew["TxSequence"].find("Parameters", Identifier=tx_id)
        data_txseq = crsd_ew["Data"]["Transmit"].find("TxSequence", TxId=tx_id)
        num_pulses = data_txseq["NumPulses"]
        data_rcv = crsd_ew["Data"]["Receive"].find("Channel", ChId=ch_id)
        num_vectors = data_rcv["NumVectors"]
        num_samples = data_rcv["NumSamples"]

        ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(crsd_etree))
        pulse_nums = [0, num_pulses - 1]
        tx_times = [txseq_param["TxTime1"], txseq_param["TxTime2"]]
        if ch_id == ref_ch_id:
            pulse_nums.insert(1, chan_param["SARImage"]["RefVectorPulseIndex"])
            tx_times.insert(1, tx_ref_time)
        txtime = np.interp(np.arange(num_pulses), pulse_nums, tx_times)
        ppps["TxTime"]["Int"] = np.floor(txtime)
        ppps["TxTime"]["Frac"] = txtime % 1
        ppps["FX1"][:] = txseq_param["FxC"] - txseq_param["FxBW"] / 2
        ppps["FX2"][:] = txseq_param["FxC"] + txseq_param["FxBW"] / 2
        ppps["TXmt"][:] = txseq_param["TXmtMin"]
        ppps["TxRadInt"][:] = txseq_param["TxRefRadIntensity"]
        ppps["FxRate"][:] = txseq_param["FxBW"] / txseq_param["TXmtMin"]
        ppps["FxFreq0"][:] = txseq_param["FxC"]
        ppps["TxPos"] = npp.polyval(txtime, tx_pos_poly).T
        if ch_id == ref_ch_id:
            ppps["TxPos"][chan_param["SARImage"]["RefVectorPulseIndex"]] = txpos
        ppps["TxVel"] = txvel
        tx_ref_pt = txseq_param["TxRefPoint"]["ECF"]
        los = tx_ref_pt - ppps["TxPos"]
        cross_track = unit(np.cross(los, txvel))
        ppps["TxACX"][...] = unit(cross_track)
        ppps["TxACY"][...] = unit(txvel)
        ref_pulse_index = txseq_param["RefPulseIndex"]
        tx_apat_id = txseq_param["TxAPATId"]
        tx_ant_pol_ref = ant.find("AntPattern", Identifier=tx_apat_id)["AntPolRef"]
        compute_vh_pol(
            ppps["TxPos"][ref_pulse_index],
            ppps["TxACX"][ref_pulse_index],
            ppps["TxACY"][ref_pulse_index],
            tx_ref_pt,
            tx_ant_pol_ref,
            txseq_param["TxPolarization"],
        )

        pvps = np.zeros((num_vectors), dtype=pvp_dtype)
        vector_nums = [0, num_vectors - 1]
        rcv_times = [chan_param["RcvStartTime1"], chan_param["RcvStartTime2"]]
        if ch_id == ref_ch_id:
            vector_nums.insert(1, chan_param["RefVectorIndex"])
            rcv_times.insert(1, rcv_ref_time)
        rcvstart = np.interp(np.arange(num_vectors), vector_nums, rcv_times)
        fs = chan_param["Fs"]
        rcvstart = np.round((rcvstart - rcvstart[0]) * fs) / fs + rcvstart[0]
        pvps["RcvStart"]["Int"] = np.floor(rcvstart)
        pvps["RcvStart"]["Frac"] = rcvstart % 1
        pvps["RcvPos"] = npp.polyval(rcvstart, rcv_pos_poly).T
        if ch_id == ref_ch_id:
            pvps["RcvPos"][chan_param["RefVectorIndex"]] = rcvpos
        pvps["RcvVel"] = rcvvel
        pvps["SIGNAL"] = 1
        pvps["RefFreq"] = chan_param["F0Ref"]
        pvps["TxPulseIndex"] = np.arange(pvps.size)
        pvps["FRCV1"] = chan_param["FrcvMin"]
        pvps["FRCV2"] = chan_param["FrcvMax"]
        pvps["AmpSF"] = 1.0
        pvps["DFIC0"][1] = -10
        pvps["FICRate"][1] = 10
        rcv_ref_pt = chan_param["RcvRefPoint"]["ECF"]
        los = rcv_ref_pt - pvps["RcvPos"]
        cross_track = np.cross(los, rcvvel)
        pvps["RcvACX"][...] = unit(cross_track)
        pvps["RcvACY"][...] = unit(rcvvel)
        ref_vector_index = chan_param["RefVectorIndex"]
        rcv_apat_id = chan_param["RcvAPATId"]
        rcv_ant_pol_ref = ant.find("AntPattern", Identifier=rcv_apat_id)["AntPolRef"]
        compute_vh_pol(
            pvps["RcvPos"][ref_vector_index],
            pvps["RcvACX"][ref_vector_index],
            pvps["RcvACY"][ref_vector_index],
            rcv_ref_pt,
            rcv_ant_pol_ref,
            chan_param["RcvPolarization"],
        )
        ref_vector_pulse_index = chan_param["SARImage"]["RefVectorPulseIndex"]
        compute_vh_pol(
            ppps["TxPos"][ref_vector_pulse_index],
            ppps["TxACX"][ref_vector_pulse_index],
            ppps["TxACY"][ref_vector_pulse_index],
            rcv_ref_pt,
            tx_ant_pol_ref,
            chan_param["SARImage"]["TxPolarization"],
        )

        pxp_dict_list.append(
            {
                "txid": tx_id,
                "chid": ch_id,
                "nvec": num_vectors,
                "nsamp": num_samples,
                "ppps": ppps,
                "pvps": pvps,
            }
        )

    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        for pxp_dict in pxp_dict_list:
            cw.write_ppp(pxp_dict["txid"], pxp_dict["ppps"])
            cw.write_pvp(pxp_dict["chid"], pxp_dict["pvps"])
            signal = (
                rng.integers(
                    -128, 127, (pxp_dict["nvec"], pxp_dict["nsamp"], 2), dtype=np.int8
                )
                .view(signal_dtype)
                .squeeze()
            )
            cw.write_signal(pxp_dict["chid"], signal)
    yield tmp_crsd


@pytest.fixture(scope="session")
def multi_crsdtx(tmp_path_factory, multi_crsdsar):
    with multi_crsdsar.open("rb") as f, skcrsd.Reader(f) as cr:
        crsd_etree = cr.metadata.xmltree
        crsd_ew = skcrsd.ElementWrapper(crsd_etree.getroot())
        sequence_ids = [
            param["Identifier"] for param in crsd_ew["TxSequence"]["Parameters"]
        ]
        ppp_dict_list = []
        for seqid in sequence_ids:
            ppp_dict_list.append(
                {
                    "txid": seqid,
                    "ppps": cr.read_ppps(seqid),
                }
            )

    crsd_etree.find(".//{*}RefPulseIndex").text = crsd_etree.find(
        ".//{*}RefVectorPulseIndex"
    ).text
    ns = lxml.etree.QName(crsd_etree.getroot()).namespace
    crsd_etree.getroot().tag = f"{{{ns}}}CRSDtx"
    _remove(crsd_etree, "{*}SARInfo")
    _remove(crsd_etree, "{*}ReceiveInfo")
    _remove(crsd_etree, "{*}Global/{*}Receive")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ExtendedArea")
    _remove(crsd_etree, "{*}SceneCoordinates/{*}ImageGrid")
    _remove(crsd_etree, "{*}Data/{*}Receive")
    _remove(crsd_etree, "{*}Channel")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}RcvParameters")
    _remove(crsd_etree, "{*}DwellPolynomials")
    _remove(crsd_etree, "{*}PVP")
    _replace_error(crsd_etree, "Tx")
    tmp_crsd = (
        tmp_path_factory.mktemp("data") / good_crsd_xml_path.with_suffix(".crsd").name
    )

    new_meta = skcrsd.Metadata(
        xmltree=crsd_etree,
    )
    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        for ppp_dict in ppp_dict_list:
            cw.write_ppp(ppp_dict["txid"], ppp_dict["ppps"])
    yield tmp_crsd


@pytest.fixture(scope="session")
def multi_crsdrcv(tmp_path_factory, multi_crsdsar):
    with multi_crsdsar.open("rb") as f, skcrsd.Reader(f) as cr:
        crsd_etree = cr.metadata.xmltree
        crsd_ew = skcrsd.ElementWrapper(crsd_etree.getroot())
        channel_ids = [
            param["Identifier"] for param in crsd_ew["Channel"]["Parameters"]
        ]
        pvps_sig_dict_list = []
        for chan_id in channel_ids:
            pvps_sig_dict_list.append(
                {
                    "chan_id": chan_id,
                    "pvps": cr.read_pvps(chan_id),
                    "signal": cr.read_signal(chan_id),
                }
            )
    ns = lxml.etree.QName(crsd_etree.getroot()).namespace
    crsd_etree.getroot().tag = f"{{{ns}}}CRSDrcv"
    _remove(crsd_etree, "{*}SARInfo")
    _remove(crsd_etree, "{*}TransmitInfo")
    _remove(crsd_etree, "{*}Global/{*}Transmit")
    _remove(crsd_etree, "{*}Data/{*}Transmit")
    _remove(crsd_etree, "{*}TxSequence")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}SARImage")
    _remove(crsd_etree, "{*}ReferenceGeometry/{*}TxParameters")
    _remove(crsd_etree, "{*}DwellPolynomials")
    for chan_param in crsd_etree.findall("{*}Channel/{*}Parameters"):
        chan_param.remove(chan_param.find("{*}SARImage"))

    fx_ids = [
        x.text
        for x in crsd_etree.findall("{*}SupportArray/{*}FxResponseArray/{*}Identifier")
    ]
    xm_ids = [
        x.text for x in crsd_etree.findall("{*}SupportArray/{*}XMArray/{*}Identifier")
    ]
    _remove(crsd_etree, "{*}SupportArray/{*}FxResponseArray")
    for x in fx_ids + xm_ids:
        _remove(
            crsd_etree,
            f"{{*}}Data/{{*}}Support/{{*}}SupportArray[{{*}}SAId='{x}']",
        )
    nsa = crsd_etree.find("{*}Data/{*}Support/{*}NumSupportArrays")
    nsa.text = str(int(nsa.text) - len(fx_ids + xm_ids))
    _repack_support_arrays(crsd_etree)
    _remove(crsd_etree, "{*}PPP")
    tx_pulse_index_offset = int(crsd_etree.findtext("{*}PVP/{*}TxPulseIndex/{*}Offset"))
    _remove(crsd_etree, "{*}PVP/{*}TxPulseIndex")
    for pvp_offset in crsd_etree.findall("{*}PVP/*/{*}Offset"):
        if int(pvp_offset.text) > tx_pulse_index_offset:
            pvp_offset.text = str(int(pvp_offset.text) - 1)
    new_num_bytes_pvps = (
        int(crsd_etree.findtext("{*}Data/{*}Receive/{*}NumBytesPVP")) - 8
    )
    crsd_etree.find("{*}Data/{*}Receive/{*}NumBytesPVP").text = str(new_num_bytes_pvps)
    _replace_error(crsd_etree, "Rcv")

    # Make PVPs without TxPulseIndex
    new_pvp_dtype = skcrsd.get_pvp_dtype(crsd_etree)
    for pvps_sig_dict in pvps_sig_dict_list:
        new_pvps = np.zeros(pvps_sig_dict["pvps"].shape, new_pvp_dtype)
        for field in new_pvp_dtype.fields:
            new_pvps[field] = pvps_sig_dict["pvps"][field]
        pvps_sig_dict["pvps"] = new_pvps

    offset = 0
    for pvps_sig_dict in pvps_sig_dict_list:
        data_rcv = crsd_ew["Data"]["Receive"].find(
            "Channel", ChId=pvps_sig_dict["chan_id"]
        )
        data_rcv["PVPArrayByteOffset"] = str(offset)
        offset += data_rcv["NumVectors"] * crsd_ew["Data"]["Receive"]["NumBytesPVP"]

    tmp_crsd = (
        tmp_path_factory.mktemp("data") / good_crsd_xml_path.with_suffix(".crsd").name
    )
    new_meta = skcrsd.Metadata(
        xmltree=crsd_etree,
    )
    with open(tmp_crsd, "wb") as f, skcrsd.Writer(f, new_meta) as cw:
        for pvps_sig_dict in pvps_sig_dict_list:
            cw.write_pvp(pvps_sig_dict["chan_id"], pvps_sig_dict["pvps"])
            cw.write_signal(pvps_sig_dict["chan_id"], pvps_sig_dict["signal"])
    yield tmp_crsd
