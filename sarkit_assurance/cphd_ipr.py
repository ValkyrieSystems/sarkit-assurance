"""Perform IPR analysis on CPHD Targets"""

import argparse
import bisect
import json
import pathlib
from collections.abc import Sequence
from typing import Any

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt
import sarkit.cphd as skcphd
import sarkit.wgs84
import sarkit_processing.remocomp as skp_remo
import scipy.fft
import shapely

from . import _ipr, names

try:
    from smart_open import open
except ImportError:
    pass

NUM_IAC_PAD = 10
NOM_CHIP_EDGE_PX = 256
NOMINAL_OSR = 1.25


def ecef_to_scene_transform(
    cphd_xmltree: lxml.etree.ElementTree, tgt_ecef: npt.ArrayLike
) -> np.ndarray:
    """Return the IAC coordinates where ``tgt_ecef`` orthogonally projects onto the ReferenceSurface

    Parameters
    ----------
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML
    tgt_ecef : (3,) array_like
        Target location in WGS 84 cartesian coordinates with X, Y, Z components in meters

    Returns
    -------
    tgt_iac : tuple, shape=(2,)
        Location where ``tgt_ecef`` orthogonally projects onto the ReferenceSurface in IAC coordinates
        with IAX, IAY components in meters.
    """
    tgt_ecef = np.asarray(tgt_ecef)

    max_iters = 15
    max_err = 1e-5  # meters
    step_x = 1.0  # Initial step size in scene coords
    step_y = 1.0  # Initial step size in scene coords

    # Initialize three calibration points near origin of scene coord sys
    scene_cal_points = np.array([[0.0, 0.0], [step_x, 0.0], [0.0, step_y]])

    # Finds step size in scene coords that produces 1-meter change in ECEF
    ecef_cal_points = skcphd.iac_to_ecf(cphd_xmltree, scene_cal_points)
    step_x /= float(np.linalg.norm(ecef_cal_points[1, :] - ecef_cal_points[0, :]))
    step_y /= float(np.linalg.norm(ecef_cal_points[2, :] - ecef_cal_points[0, :]))

    # Pick new calibration points based on new step sizes
    scene_cal_points = np.array([[0.0, 0.0], [step_x, 0.0], [0.0, step_y]])

    for _ in range(max_iters):
        ecef_cal_points = skcphd.iac_to_ecf(cphd_xmltree, scene_cal_points)

        # Compute local partial derivates of ECEF w.r.t. scene x and y coordinates
        der_x = (ecef_cal_points[1, :] - ecef_cal_points[0, :]) / step_x
        der_y = (ecef_cal_points[2, :] - ecef_cal_points[0, :]) / step_y

        # Convert target_ecef to ESTIMATED scene_coords
        tgt_scene_x = (
            np.dot(der_x, tgt_ecef - ecef_cal_points[0, :]) + scene_cal_points[0, 0]
        )
        tgt_scene_y = (
            np.dot(der_y, tgt_ecef - ecef_cal_points[0, :]) + scene_cal_points[0, 1]
        )

        # Convert scene_coords back to ecef (using exact transform - result will be in scene plane)
        est_ecef = skcphd.iac_to_ecf(cphd_xmltree, [tgt_scene_x, tgt_scene_y])

        # Find error, defined as distance between tgt_ecef and a line normal to scene plane at est_ecef
        ecef_normal = np.cross(der_x, der_y) / np.linalg.norm(
            np.cross(der_x, der_y)
        )  # Unit vector normal to scene
        dist = np.linalg.norm(
            (tgt_ecef - est_ecef)
            - np.dot(tgt_ecef - est_ecef, ecef_normal) * ecef_normal
        )

        if dist < max_err:
            return np.array([tgt_scene_x, tgt_scene_y])

        # Re-initialize three scene points near last estimated tgt location
        scene_cal_points = np.array(
            [
                [tgt_scene_x, tgt_scene_y],
                [tgt_scene_x + step_x, tgt_scene_y],
                [tgt_scene_x, tgt_scene_y + step_y],
            ]
        )

    raise RuntimeError(f"Failed to converge to {max_err} after {max_iters} iterations")


def _get_imagearea_poly(imgarea_ew, *, use_polygon=True):
    region = shapely.box(*imgarea_ew["X1Y1"], *imgarea_ew["X2Y2"])
    if use_polygon and (poly := imgarea_ew.get("Polygon", None)) is not None:
        region = region.intersection(shapely.Polygon(poly))
    return region


def get_channel_image_area(
    cphd_xmltree: lxml.etree.ElementTree, ch_id: str
) -> shapely.Polygon:
    # TODO: move to sarkit
    ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
    ia_poly = _get_imagearea_poly(ew["SceneCoordinates"]["ImageArea"])
    chan_param_ew = ew["Channel"].find("Parameters", Identifier=ch_id)
    if "ImageArea" in chan_param_ew:
        ia_poly = _get_imagearea_poly(chan_param_ew["ImageArea"])
    return ia_poly


def get_scene_image_area(cphd_xmltree: lxml.etree.ElementTree) -> shapely.Polygon:
    # TODO: move to sarkit
    ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
    return _get_imagearea_poly(ew["SceneCoordinates"]["ImageArea"])


def get_extended_image_area(
    cphd_xmltree: lxml.etree.ElementTree,
) -> shapely.Polygon | None:
    # TODO: move to sarkit
    ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
    imgarea_ew = ew["SceneCoordinates"].get("ExtendedArea", None)
    return None if imgarea_ew is None else _get_imagearea_poly(imgarea_ew)


def analyze(
    cphd_reader: skcphd.Reader,
    geojson: dict[str, Any],
    ch_id: str,
    outdir: pathlib.Path,
    *,
    search_size_px: None | int = None,
) -> None:
    """Perform IPR analysis of CPHD targets.

    TODO: finish docstring
    """

    # Find target image coordinates and ensure they are (nearly) within ImageArea
    cphd_xmltree = cphd_reader.metadata.xmltree
    imagearea_iac = get_channel_image_area(cphd_xmltree, ch_id)
    padded_imagearea_iac = imagearea_iac.buffer(NUM_IAC_PAD, quad_segs=4)
    features_to_analyze = []
    target_locs = []  # tuple of (ecef, iac)
    for index, feature in enumerate(_get_all_features(geojson)):
        feature_props: dict[str, Any] = {"index": index}
        if (orig_properties := feature.get("properties")) is not None:
            feature_props["original_properties"] = orig_properties
        feature["properties"] = feature_props

        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
        if feature["geometry"]["type"] != "Point" or coordinates.shape != (3,):
            raise ValueError("Only 3D Point features are supported")

        coord_llh = [coordinates[1], coordinates[0], coordinates[2]]
        coord_ecef = sarkit.wgs84.geodetic_to_cartesian(coord_llh)
        try:
            coord_iac = ecef_to_scene_transform(cphd_xmltree, coord_ecef)
        except RuntimeError as exc:
            feature["properties"].update(valid=False, message=str(exc))
            continue

        if not shapely.contains_xy(padded_imagearea_iac, coord_iac[0], coord_iac[1]):
            feature_props.update(valid=False, message="Too far outside ImageArea")
            continue
        feature_props.update(valid=True, projected_location_iac=coord_iac)
        features_to_analyze.append(feature)
        target_locs.append((coord_ecef, coord_iac))

    sgn = int(cphd_xmltree.findtext("{*}Global/{*}SGN"))
    cphd_table_info = [
        ("Ch_ID", ch_id),
        ("Global/SGN", str(sgn)),
    ]
    context_geoms = [
        ("Channel Image Area", shapely.get_coordinates(imagearea_iac.exterior)),
    ]
    scene_imagearea = get_scene_image_area(cphd_xmltree)
    if not shapely.equals(imagearea_iac, scene_imagearea):
        context_geoms.append(
            ("Scene Image Area", shapely.get_coordinates(scene_imagearea))
        )
    extended_imagearea = get_extended_image_area(cphd_xmltree)
    if extended_imagearea is not None:
        context_geoms.append(
            ("Extended Image Area", shapely.get_coordinates(extended_imagearea))
        )
    for feature, (chip, spacing_rc, bw_rc) in zip(
        features_to_analyze,
        extract_chips(
            cphd_reader,
            target_locs,
            ch_id,
        ),
        strict=True,
    ):
        if chip is None:
            feature["properties"].update(
                valid=False, message="No vectors found that support this target"
            )
            continue

        est_offset_rc, peak_power, iprparts = _ipr.estimate_peak(
            chip, search_dist=search_size_px
        )
        est_offset_si = est_offset_rc * spacing_rc  # in SI units (0: s, 1: Hz)
        feature["properties"].update(
            valid=True,
            observed_toa_offset_sec=est_offset_si[0],
            observed_dopplerfreq_offset_hz=est_offset_si[1],
            peak_power=peak_power,
        )
        iprparts.chip = _ipr.downsample_chip(iprparts.chip)
        customize_spatial_axes(iprparts, spacing_rc)
        k_iprparts = _ipr.create_spectral_chip(iprparts, sgn)
        customize_spatialfreq_axes(k_iprparts, spacing_rc)
        target_info = [
            ("Target Peak", f"{10 * np.log10(peak_power):.6f} dB"),
            ("", ""),
            ("TOA Offset", f"{est_offset_si[0] * 1e9:.6f} nsec"),
            ("RF BW", f"{bw_rc[0] * 1e-6:.6f} MHz"),
            ("Offset x BW", f"{bw_rc[0] * est_offset_si[0]:.6f}"),
            ("", ""),
            ("fdop Offset", f"{est_offset_si[1]:.6f} m"),
            ("T Dwell", f"{bw_rc[1]:.6f} sec"),
            ("Offset x Dwell", f"{bw_rc[1] * est_offset_si[1]:.6f}"),
            ("", ""),
        ]
        fig = _ipr.plot_ipr(
            iprparts,
            k_iprparts,
            spacing_rc,
            bw_rc,
            target_info + cphd_table_info,
            _ipr.TargetContext(
                geoms=[
                    (
                        "Projected Target",
                        np.atleast_2d(feature["properties"]["projected_location_iac"]),
                    )
                ]
                + context_geoms,
                row_label="IAX (m)",
                col_label="IAY (m)",
            ),
        )
        figstem = f"cphd_ipr{feature['properties']['index']}"
        figtitle = f"CPHD IPR Analysis - Feature #{feature['properties']['index']}"
        if "id" in feature:
            figstem += f"-{names.sanitize_name(feature['id'])}"
            figtitle += f": {feature['id']}"
        fig.update_layout(title_text=figtitle)
        fig.write_html(outdir / f"{figstem}.html")

    (outdir / "cphd_ipr.json").write_text(
        json.dumps(geojson, cls=_ipr.NdArrJSONEncoder, indent=4)
    )


def extract_chips(
    cphd_reader: skcphd.Reader,
    target_locs: Sequence[tuple[np.ndarray, np.ndarray]],
    ch_id: str,
):
    """Iterator that yields image chips and supporting metadata for requested target locations.

    Implementation uses a simple 2D FFT after re-doing motion compensation and assumes a somewhat constant FX
    bandwidth and RcvTime sampling interval across the target's dwell.

    TODO: more docstrings / return typehints
    """
    all_pvps = cphd_reader.read_pvps(ch_id)
    ref_times = skcphd.compute_t_ref_from_pvps(all_pvps)

    sgn = int(cphd_reader.metadata.xmltree.findtext("{*}Global/{*}SGN"))
    for tgt_ecef, (tgt_iax, tgt_iay) in target_locs:
        t_cod, t_dwell = compute_dwelltimes_using_poly(
            ch_id, tgt_iax, tgt_iay, cphd_reader.metadata.xmltree
        )
        t_start = t_cod - t_dwell / 2
        t_end = t_cod + t_dwell / 2

        start_vector = bisect.bisect_right(ref_times, t_start)  # leftmost > t_start
        past_stop_vector = bisect.bisect_left(ref_times, t_end)  # rightmost < t_end + 1
        if past_stop_vector - start_vector < 1:
            yield None
            continue

        this_signal, these_pvps = skp_remo.remocomp_cphd_chan(
            cphd_reader,
            ch_id,
            tgt_ecef,
            start_vector=start_vector,
            stop_vector=past_stop_vector,
        )

        # Correct for antenna gain
        tx_gain, tx_phase = compute_oneway_gain_and_phase_for_vectors(
            cphd_reader.metadata.xmltree, ch_id, "Tx", these_pvps, tgt_ecef
        )
        rcv_gain, rcv_phase = compute_oneway_gain_and_phase_for_vectors(
            cphd_reader.metadata.xmltree, ch_id, "Rcv", these_pvps, tgt_ecef
        )
        # Convert from dB gain to linear correction
        inv_ant_gain = (
            np.power(10.0, -(tx_gain + rcv_gain) / 20.0)
            * np.exp(-1j * (tx_phase + rcv_phase) * np.pi * 2.0)
        ).astype(np.complex64)
        this_signal *= inv_ant_gain[:, np.newaxis]

        condition_signal_in_place(this_signal, these_pvps)

        nfft1 = scipy.fft.next_fast_len(int(NOMINAL_OSR * this_signal.shape[-1]))
        this_signal = _ipr._fft_ops(
            nfft1, NOM_CHIP_EDGE_PX, 1.0, -sgn, True, True, this_signal
        )
        this_signal = np.transpose(this_signal)
        nfft2 = scipy.fft.next_fast_len(int(NOMINAL_OSR * this_signal.shape[-1]))
        this_signal = _ipr._fft_ops(
            nfft2, NOM_CHIP_EDGE_PX, 1.0, -sgn, True, True, this_signal
        )

        spacing_rc = 1.0 / np.array(
            [
                np.mean(these_pvps["SCSS"]) * nfft1,
                np.mean(np.diff(these_pvps["RcvTime"])) * nfft2,
            ]
        )
        bw_rc = np.array(
            [
                np.mean(these_pvps["FX2"] - these_pvps["FX1"]),
                (these_pvps["RcvTime"][-1] - these_pvps["RcvTime"][0]),
            ]
        )
        yield this_signal, spacing_rc, bw_rc


def condition_signal_in_place(sig_array, pvps):
    """Condition signal array by zeroing out unwanted portions."""
    has_signal = "SIGNAL" in pvps.dtype.names
    for sigvec, pvp in zip(sig_array, pvps, strict=True):
        if has_signal and pvp["SIGNAL"] != 1:
            sigvec[:] = 0.0
        else:
            start_sample = int(np.round((pvp["FX1"] - pvp["SC0"]) / pvp["SCSS"]))
            end_sample = int(np.round((pvp["FX2"] - pvp["SC0"]) / pvp["SCSS"]))
            sigvec[:start_sample] = 0.0
            sigvec[end_sample + 1 :] = 0.0


def compute_dwelltimes_using_poly(
    ch_id: str,
    iax: npt.ArrayLike,
    iay: npt.ArrayLike,
    cphd_xmltree: lxml.etree.ElementTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute center of dwell times and dwell times for scene points using polynomials.

    Parameters
    ----------
    ch_id : str
        Channel unique identifier
    iax, iay : array_like
        Image area coordinates (in meters) of the scene points for which to compute the dwell times
    cphd_xmltree : lxml.etree.ElementTree
        CPHD XML

    Returns
    -------
    t_cod : ndarray
        Center of dwell times (sec) for the scene points relative to the CollectionStart time
    t_dwell : ndarray
       Dwell times (sec) for which the channel signal array contains the echo signals from the scene points
    """
    # TODO: move to sarkit
    iax, iay = np.broadcast_arrays(iax, iay)

    ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
    chan_dt = ew["Channel"].find("Parameters", Identifier=ch_id)["DwellTimes"]
    cod_poly = ew["Dwell"].find("CODTime", Identifier=chan_dt["CODId"])["CODTimePoly"]
    dwell_poly = ew["Dwell"].find("DwellTime", Identifier=chan_dt["DwellId"])[
        "DwellTimePoly"
    ]
    t_cod = npp.polyval2d(iax, iay, cod_poly)
    t_dwell = npp.polyval2d(iax, iay, dwell_poly)
    return t_cod, t_dwell


def compute_oneway_gain_and_phase_for_vectors(
    xmltree,
    ch_id,
    txrcv,
    pvps,
    tgt_ecef,
):
    ew = skcphd.ElementWrapper(xmltree.getroot())
    chpar = ew["Channel"].find("Parameters", Identifier=ch_id)
    apc_id = chpar["Antenna"][f"{txrcv}APCId"]
    apat_id = chpar["Antenna"][f"{txrcv}APATId"]
    acf_id = ew["Antenna"].find("AntPhaseCenter", Identifier=apc_id)["ACFId"]

    acf = ew["Antenna"].find("AntCoordFrame", Identifier=acf_id)
    apat = ew["Antenna"].find("AntPattern", Identifier=apat_id)

    los = tgt_ecef - pvps[f"{txrcv}Pos"]
    ulos = los / np.linalg.norm(los, axis=-1, keepdims=True)

    t = pvps[f"{txrcv}Time"]
    acx = np.moveaxis(npp.polyval(t, acf["XAxisPoly"]), 0, -1)
    acy = np.moveaxis(npp.polyval(t, acf["YAxisPoly"]), 0, -1)

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    uacz = unit(np.cross(acx, acy))
    u = unit(unit(acx) + unit(acy))
    v = np.cross(uacz, u)
    acx_norm = unit(u - v)
    acy_norm = unit(u + v)

    dcx = np.vecdot(ulos, acx_norm)
    dcy = np.vecdot(ulos, acy_norm)

    eb_dcx = npp.polyval(t, apat["EB"]["DCXPoly"])
    eb_dcy = npp.polyval(t, apat["EB"]["DCYPoly"])

    delta_dcx = dcx - eb_dcx
    delta_dcy = dcy - eb_dcy
    g_a = npp.polyval2d(delta_dcx, delta_dcy, apat["Array"]["GainPoly"])
    p_a = npp.polyval2d(delta_dcx, delta_dcy, apat["Array"]["PhasePoly"])

    g_e = npp.polyval2d(dcx, dcy, apat["Element"]["GainPoly"])
    p_e = npp.polyval2d(dcx, dcy, apat["Element"]["PhasePoly"])

    # TODO: ask about r**2/ref_range losses
    return g_a + g_e, p_a + p_e


def customize_spatial_axes(iprparts: _ipr.IprParts, spacing_rc: Sequence[float]):
    # scale by Row/Col spacing, change out generic labels
    ns_per_s = 1e9
    iprparts.chip.row.rescale(spacing_rc[0] * ns_per_s, name="Range: ΔTOA", units="ns")
    iprparts.chip.col.rescale(spacing_rc[1], name="Azimuth: Δfdop", units="Hz")

    # Change out generic labels
    iprparts.vs_row.domain.name = "Range: ΔTOA"
    iprparts.vs_col.domain.name = "Azimuth: Δfdop"


def customize_spatialfreq_axes(k_iprparts: _ipr.IprParts, spacing_rc: Sequence[float]):
    # Scale to sampling frequency, change out generic labels
    k_iprparts.chip.row.rescale(1 / spacing_rc[0], name="RF Freq", units="Hz")
    k_iprparts.chip.col.rescale(1 / spacing_rc[1], name="Slow Time", units="sec")
    k_iprparts.vs_row.domain.rescale(1 / spacing_rc[0], name="RF Freq", units="Hz")
    k_iprparts.vs_col.domain.rescale(1 / spacing_rc[1], name="Slow Time", units="sec")


def _get_all_features(geojson):
    """Iterate over each feature in a GeoJSON"""
    # TODO: move this to a common place
    if geojson["type"] == "Feature":
        return [geojson]
    elif geojson["type"] == "FeatureCollection":
        return geojson["features"]
    return []


def main(args=None):
    parser = argparse.ArgumentParser(description="Analyze target IPRs in a CPHD")
    parser.add_argument(
        "cphd_file", help="Input CPHD file (must have Antenna metadata)"
    )
    parser.add_argument("geojson_file", help="Input GeoJSON file")
    parser.add_argument("out_dir", help="Directory to store results", type=pathlib.Path)
    config = parser.parse_args(args)

    with open(config.geojson_file, "rb") as file:
        geo = json.load(file)

    with open(config.cphd_file, "rb") as f, skcphd.Reader(f) as r:
        if r.metadata.xmltree.find("{*}Antenna") is None:
            raise ValueError("CPHD must have antenna metadata")

        # TODO: figure out what we want to do about multi-channel
        ch_id = r.metadata.xmltree.findtext("{*}Channel/{*}RefChId")
        analyze(r, geo, ch_id, config.out_dir)


if __name__ == "__main__":
    main()
