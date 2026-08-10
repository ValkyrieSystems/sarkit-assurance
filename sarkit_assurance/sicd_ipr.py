"""Perform IPR analysis on SICD Targets"""

import argparse
import json
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt
import sarkit.sicd as sksicd
import sarkit.wgs84
import sarkit_processing.sicd_pixel_type as skp_sicdpx
import shapely

from . import _ipr, _sicd_utils, names

try:
    from smart_open import open
except ImportError:
    pass

NUM_SIDELOBES_VALIDDATA_PAD = 10
NOM_CHIP_EDGE_PX = 256


class UnsupportedChipError(Exception):
    """Chip unsupported by SICD"""


def get_padded_validdata(ew: sksicd.ElementWrapper) -> shapely.Polygon:
    """Return a SICD's ValidData polygon in xrowycol, padded out a number of resolution cells"""
    sidelobe_size_m = 1.0 / np.array(
        [ew["Grid"][rc]["ImpRespBW"] for rc in ("Row", "Col")]
    )

    def rowcol_to_iprs(rc):
        return sksicd.rowcol_to_xrowycol(ew.elem.getroottree(), rc) / sidelobe_size_m

    validdata_rowcol = _sicd_utils.get_validdata_polygon(ew)
    validdata_iprs = shapely.transform(validdata_rowcol, rowcol_to_iprs)
    padded_validdata_iprs = validdata_iprs.buffer(
        NUM_SIDELOBES_VALIDDATA_PAD, quad_segs=4
    )
    padded_validdata_xrowycol = shapely.transform(
        padded_validdata_iprs, lambda c: c * sidelobe_size_m
    )
    return padded_validdata_xrowycol


def analyze(
    sicd_reader: sksicd.NitfReader,
    geojson: dict[str, Any],
    outdir: pathlib.Path,
    *,
    search_sizes_px: None | Sequence[int] = None,
) -> None:
    """Generate and write SICD IPR analysis artifacts to a directory for targets described in a GeoJSON.

    Targets may be refined iteratively.
    Each iteration will be moved by the median geolocation error from the previous iteration.
    Multiple iterations with decreasing search sizes can be used to be more likely to find the correct target when there
    is mainly a bulk offset to geolocation for all targets.

    Parameters
    ----------
    sicd_reader : sarkit.sicd.NitfReader
        Open SICD reader object
    geojson : dict of {str : any}
        Parsed GeoJSON object containing 3D point features to analyze
    outdir : pathlib.Path
        Path to write output files to
    search_sizes_px : sequence of int or None, optional
        Number of pixels away from the expected position to search in each dimension.
        Multiple iterations with decreasing search sizes can be used to more likely find the correct target when
        there is mainly a bulk geolocation offset.
        If ``None``, a single iteration using default-sized chips is performed.
    """

    # Make sure targets project near the valid region
    sicd_xmltree = sicd_reader.metadata.xmltree
    ew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    padded_validdata_xrowycol = get_padded_validdata(ew)
    features_to_analyze = []
    for index, feature in enumerate(_get_all_features(geojson)):
        feature_props: dict[str, Any] = {"index": index}
        if (orig_properties := feature.get("properties")) is not None:
            feature_props["original_properties"] = orig_properties
        feature["properties"] = feature_props
        coord_llh = _ipr.get_feature_point(feature)
        coord_ecef = sarkit.wgs84.geodetic_to_cartesian(coord_llh)
        coord_xrowycol, _, success = sksicd.scene_to_image(sicd_xmltree, coord_ecef)
        if not success:
            feature_props.update(valid=False, message="Failed to project")
            continue
        if not padded_validdata_xrowycol.contains(shapely.Point(coord_xrowycol)):
            feature_props.update(valid=False, message="Too far outside ValidData")
            continue
        feature_props.update(valid=True, projected_location_xrowycol=coord_xrowycol)
        features_to_analyze.append(feature)

    ss_rc = [ew["Grid"][d]["SS"] for d in ("Row", "Col")]
    bw_rc = [ew["Grid"][d]["ImpRespBW"] for d in ("Row", "Col")]
    labels_rc = {
        "RGAZIM": ("Range", "Azimuth"),
        "RGZERO": ("Range", "Azimuth"),
        "XRGYCR": ("Range", "Cross Range"),
        "XCTYAT": ("Cross Track", "Along Track"),
        "PLANE": ("U", "V"),
    }.get(ew["Grid"]["Type"], ("\N{UP DOWN ARROW}", "\N{LEFT RIGHT ARROW}"))

    sicd_table_info = get_table_info(ew)
    image_extent_rc = shapely.box(
        -0.5, -0.5, ew["ImageData"]["NumRows"] - 0.5, ew["ImageData"]["NumCols"] - 0.5
    )
    image_extent_rc = shapely.get_coordinates(image_extent_rc.exterior) + [
        ew["ImageData"]["FirstRow"],
        ew["ImageData"]["FirstCol"],
    ]
    image_extent_xrowycol = sksicd.rowcol_to_xrowycol(sicd_xmltree, image_extent_rc)

    context_geoms = [
        ("Image Extent", image_extent_xrowycol),
    ]
    if "ValidData" in ew["ImageData"]:
        vdata_rc = ew["ImageData"]["ValidData"]
        vdata_rc = shapely.buffer(shapely.Polygon(vdata_rc), 0.5)
        vdata_xrowycol = sksicd.rowcol_to_xrowycol(
            sicd_xmltree, shapely.get_coordinates(vdata_rc)
        )
        context_geoms.append(("ValidData", vdata_xrowycol))

    search_offset = np.zeros(2)
    search_sizes = search_sizes_px or (None,)
    for iter_index, search_size_px in enumerate(search_sizes):
        is_last_iter = iter_index + 1 == len(search_sizes)
        this_iter_offsets = []
        for feature in features_to_analyze:
            try:
                est_loc_xrowycol, peak_power, iprparts = chip_and_estimate_peak(
                    sicd_reader,
                    feature["properties"]["projected_location_xrowycol"]
                    + search_offset,
                    search_size_px=search_size_px,
                )
                offset_xrowycol = (
                    est_loc_xrowycol
                    - feature["properties"]["projected_location_xrowycol"]
                )
                this_iter_offsets.append(offset_xrowycol)
                if is_last_iter:
                    feature["properties"].update(
                        valid=True,
                        observed_location_offset_xrowycol=offset_xrowycol,
                        peak_power=peak_power,
                    )
                    iprparts.chip = _ipr.downsample_chip(iprparts.chip)
                    customize_spatial_axes(iprparts, search_offset, ss_rc, labels_rc)
                    k_iprparts = _ipr.create_spectral_chip(
                        iprparts, ew["Grid"]["Row"]["Sgn"]
                    )
                    customize_spatialfreq_axes(k_iprparts, ss_rc, labels_rc)
                    target_info = [
                        ("Target Peak", f"{10 * np.log10(peak_power):.6f} dB"),
                        ("", ""),
                        ("xrow Offset", f"{offset_xrowycol[0]:.6f} m"),
                        ("Krow BW", f"{bw_rc[0]:.6f} cyc/m"),
                        ("Offset x BW", f"{bw_rc[0] * offset_xrowycol[0]:.6f}"),
                        ("", ""),
                        ("ycol Offset", f"{offset_xrowycol[1]:.6f} m"),
                        ("Kcol BW", f"{bw_rc[1]:.6f} cyc/m"),
                        ("Offset x BW", f"{bw_rc[1] * offset_xrowycol[1]:.6f}"),
                        ("", ""),
                    ]
                    this_target_context = _ipr.TargetContext(
                        geoms=[
                            (
                                "Projected Target",
                                np.atleast_2d(
                                    feature["properties"]["projected_location_xrowycol"]
                                ),
                            ),
                            ("Estimated Target", np.atleast_2d(est_loc_xrowycol)),
                        ]
                        + context_geoms,
                        row_label=f"{labels_rc[0]}: xrow (m)",
                        col_label=f"{labels_rc[1]}: ycol (m)",
                    )
                    fig = _ipr.plot_ipr(
                        iprparts,
                        k_iprparts,
                        ss_rc,
                        bw_rc,
                        table_info=target_info + sicd_table_info,
                        target_context=this_target_context,
                    )
                    figstem = f"sicd_ipr{feature['properties']['index']}"
                    figtitle = (
                        f"SICD IPR Analysis - Feature #{feature['properties']['index']}"
                    )
                    if "id" in feature:
                        figstem += f"-{names.sanitize_name(feature['id'])}"
                        figtitle += f": {feature['id']}"
                    fig.update_layout(title_text=figtitle)
                    fig.write_html(outdir / f"{figstem}.html")
            except UnsupportedChipError as exc:
                feature["properties"].update(valid=False, message=str(exc))

        if not this_iter_offsets:
            # no targets were found
            break
        search_offset = np.median(this_iter_offsets, axis=0)

    (outdir / "sicd_ipr.json").write_text(
        json.dumps(geojson, cls=_ipr.NdArrJSONEncoder, indent=4)
    )


def chip_and_estimate_peak(
    sicd_reader: sksicd.NitfReader,
    chip_center_xrowycol: npt.ArrayLike,
    *,
    search_size_px: int | None = None,
) -> tuple[np.ndarray, float, _ipr.IprParts]:
    """Chip a SICD around a center point then estimate a nearby peak power and location inside of it.

    Parameters
    ----------
    sicd_reader : sarkit.sicd.NitfReader
        Open SICD reader object
    chip_center_xrowycol : array_like
        (xrow, ycol) coordinate of chip center in meters
    search_size_px : int or None, optional
        Number of pixels away from the chip center to search in each dimension.
        If ``None``, the entire chip is searched.

    Returns
    -------
    est_loc_xrowycol : ndarray
        (xrow, ycol) coordinate of estimated peak location in meters
    peak_power : float
        Estimated peak power
    iprparts : IprParts
        Byproducts of the peak finding that may be useful for downstream analysis/plotting

    Raises
    ------
    UnsupportedChipError
        If the requested chip extent is not supported by the SICD.
    """
    chip_edge_px = NOM_CHIP_EDGE_PX
    if search_size_px is not None:
        chip_edge_px = max(
            chip_edge_px, 2 ** (int(np.ceil(np.log2(search_size_px + 1)) + 1))
        )
    chip_size = np.array([chip_edge_px, chip_edge_px])
    chip_center_xrowycol = np.asarray(chip_center_xrowycol)
    chip_center_rcglob = sksicd.xrowycol_to_rowcol(
        sicd_reader.metadata.xmltree, chip_center_xrowycol
    )
    imdata_ew = sksicd.ElementWrapper(sicd_reader.metadata.xmltree.find("{*}ImageData"))
    chip_center_rc = chip_center_rcglob - (imdata_ew["FirstRow"], imdata_ew["FirstCol"])
    chip_center_rc_int = np.round(chip_center_rc).astype(np.int64)
    chip_center_rc_frac = chip_center_rc - chip_center_rc_int

    start = chip_center_rc_int - chip_size // 2
    end = start + chip_size
    sicd_shape = [imdata_ew["NumRows"], imdata_ew["NumCols"]]

    if np.any(start < 0) or np.any(end > sicd_shape):
        raise UnsupportedChipError("Chip extent not supported by image")

    chip, chipxml = sicd_reader.read_sub_image(*start, *end)
    chip, chipxml = skp_sicdpx.sicd_as_re32f_im32f(chip, chipxml)
    grid_ew = sksicd.ElementWrapper(chipxml.find("{*}Grid"))
    delta_kctr = np.array(
        [
            npp.polyval2d(
                chip_center_xrowycol[0],
                chip_center_xrowycol[1],
                grid_ew[d].get("DeltaKCOAPoly", [[0.0]]),
            )
            for d in ("Row", "Col")
        ]
    )
    ss_rc = [grid_ew[d]["SS"] for d in ("Row", "Col")]
    phase = [
        -delta_kctr[ndx]
        * (np.arange(chip.shape[ndx]) - chip.shape[ndx] // 2)
        * ss_rc[ndx]
        for ndx in range(2)
    ]

    basebanded = chip.copy()
    basebanded *= np.exp(1j * 2 * np.pi * (phase[0][:, np.newaxis] + phase[1]))

    est_offset_rc, peak_power, iprparts = _ipr.estimate_peak(
        basebanded, offset_rc=chip_center_rc_frac, search_dist=search_size_px
    )
    est_loc_xrowycol = chip_center_xrowycol + est_offset_rc * ss_rc
    return est_loc_xrowycol, peak_power, iprparts


def customize_spatial_axes(
    iprparts: _ipr.IprParts,
    search_offset_rc: np.ndarray,
    spacing_rc: Sequence[float],
    gridlabels_rc: tuple[str, str],
):
    # reference chip to projected location, scale by Row/Col spacing, change out generic labels
    iprparts.chip.row.rescale(
        spacing_rc[0], name=f"{gridlabels_rc[0]}: Δxrow", units="m"
    )
    iprparts.chip.row.x0 += search_offset_rc[0]
    iprparts.chip.col.rescale(
        spacing_rc[1], name=f"{gridlabels_rc[1]}: Δycol", units="m"
    )
    iprparts.chip.col.x0 += search_offset_rc[1]

    # Change out generic labels
    iprparts.vs_row.domain.name = f"{gridlabels_rc[0]}: Δxrow"
    iprparts.vs_col.domain.name = f"{gridlabels_rc[1]}: Δycol"


def customize_spatialfreq_axes(
    k_iprparts: _ipr.IprParts,
    spacing_rc: Sequence[float],
    gridlabels_rc: tuple[str, str],
):
    # Scale to sampling frequency, change out generic labels
    k_iprparts.chip.row.rescale(1 / spacing_rc[0], units="cyc/m")
    k_iprparts.chip.col.rescale(1 / spacing_rc[1], units="cyc/m")
    k_iprparts.vs_row.domain.rescale(1 / spacing_rc[0], units="cyc/m")
    k_iprparts.vs_col.domain.rescale(1 / spacing_rc[1], units="cyc/m")
    k_iprparts.vs_row.domain.name = (
        f"{gridlabels_rc[0]} Spatial Freq: " + k_iprparts.vs_row.domain.name
    )
    k_iprparts.vs_col.domain.name = (
        f"{gridlabels_rc[1]} Spatial Freq: " + k_iprparts.vs_col.domain.name
    )


def get_table_info(ew: sksicd.ElementWrapper) -> list[tuple[str, str]]:
    info = [
        ("Grid/Type", ew["Grid"]["Type"]),
        ("Sgn", ew["Grid"]["Row"]["Sgn"]),
    ]
    for rc, arrow in [("Col", "\N{LEFT RIGHT ARROW}"), ("Row", "\N{UP DOWN ARROW}")]:
        info.append(
            (f"{arrow} window", ew["Grid"][rc].get("WgtType", {}).get("WindowName"))
        )
    for fname in ("STBeamComp", "ImageBeamComp", "AzAutofocus", "RgAutofocus"):
        info.append((fname, ew["ImageFormation"][fname]))
    return info


def _get_all_features(geojson):
    """Iterate over each feature in a GeoJSON"""
    # TODO: move this to a common place
    if geojson["type"] == "Feature":
        return [geojson]
    elif geojson["type"] == "FeatureCollection":
        return geojson["features"]
    return []


def main(args=None):
    parser = argparse.ArgumentParser(description="Analyze target IPRs in a SICD")
    parser.add_argument("sicd_file", help="Input SICD file")
    parser.add_argument("geojson_file", help="Input GeoJSON file")
    parser.add_argument("out_dir", help="Directory to store results", type=pathlib.Path)
    parser.add_argument(
        "--search-size-pixels",
        type=int,
        metavar="SSP",
        nargs="+",
        help=(
            "Number of pixels away from the expected position to search in each dimension. "
            "Multiple iterations with decreasing search sizes can be used to more likely find the correct target when "
            "there is mainly a bulk geolocation offset. "
            f"If unspecified, a single iteration using {NOM_CHIP_EDGE_PX} x {NOM_CHIP_EDGE_PX} chips is performed."
        ),
    )
    config = parser.parse_args(args)

    with open(config.geojson_file, "rb") as file:
        geo = json.load(file)

    with open(config.sicd_file, "rb") as f, sksicd.NitfReader(f) as r:
        analyze(r, geo, config.out_dir, search_sizes_px=config.search_size_pixels)


if __name__ == "__main__":
    main()
