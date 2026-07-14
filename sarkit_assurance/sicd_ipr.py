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
import shapely

from . import _ipr, _sicd_utils

try:
    from smart_open import open
except ImportError:
    pass

NUM_SIDELOBES_VALIDDATA_PAD = 10
NOM_CHIP_EDGE_PX = 256


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
    """Perform IPR analysis of SICD targets.

    Targets may be refined iteratively.
    Each iteration will be moved by the median geolocation error from the previous iteration.
    Multiple iterations with decreasing search sizes can be used to be more likely to find the correct target when there
    is mainly a bulk offset to geolocation for all targets.

    TODO: finish docstring
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

        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
        if feature["geometry"]["type"] != "Point" or coordinates.shape != (3,):
            raise ValueError("Only 3D Point features are supported")

        coord_llh = [coordinates[1], coordinates[0], coordinates[2]]
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

    search_offset = np.zeros(2)
    for search_size_px in search_sizes_px or (None,):
        for feature in features_to_analyze:
            chip_and_measure(
                sicd_reader,
                feature,
                search_size_px=search_size_px,
                search_offset=search_offset,
            )

        features_to_analyze = [
            feature for feature in features_to_analyze if feature["properties"]["valid"]
        ]
        if features_to_analyze:
            search_offset = np.median(
                [
                    f["properties"]["observed_location_offset_xrowycol"]
                    for f in features_to_analyze
                ],
                axis=0,
            )

    (outdir / "sicd_ipr.json").write_text(
        json.dumps(geojson, cls=_ipr.NdArrJSONEncoder, indent=4)
    )


def chip_and_measure(
    sicd_reader: sksicd.NitfReader,
    feature: dict[str, Any],
    *,
    search_size_px: int | None = None,
    search_offset: npt.ArrayLike = (0, 0),
) -> None:
    """Perform IPR analysis for a feature

    TODO: docstrings
    """
    chip_edge_px = NOM_CHIP_EDGE_PX
    if search_size_px is not None:
        chip_edge_px = max(
            chip_edge_px, 2 ** (int(np.ceil(np.log2(search_size_px + 1)) + 1))
        )
    chip_size = np.array([chip_edge_px, chip_edge_px])

    search_offset = np.asarray(search_offset)

    feature_props = feature["properties"]
    proj_loc_xrowycol: np.ndarray = feature_props["projected_location_xrowycol"]
    search_center_xrowycol = proj_loc_xrowycol + search_offset
    search_center_rc = sksicd.xrowycol_to_rowcol(
        sicd_reader.metadata.xmltree, search_center_xrowycol
    )
    search_center_rc_int = np.round(search_center_rc).astype(np.int64)
    search_center_rc_frac = search_center_rc - search_center_rc_int

    # TODO: handle chipped SICDs
    start = search_center_rc_int - chip_size // 2
    end = start + chip_size
    imdata_ew = sksicd.ElementWrapper(sicd_reader.metadata.xmltree.find("{*}ImageData"))
    sicd_shape = [imdata_ew["NumRows"], imdata_ew["NumCols"]]

    if np.any(start < 0) or np.any(end > sicd_shape):
        feature_props.update(valid=False, message="Chip extent not supported by image")
        return

    chip, chipxml = sicd_reader.read_sub_image(*start, *end)
    # TODO: pixel type handling
    grid_ew = sksicd.ElementWrapper(chipxml.find("{*}Grid"))
    delta_kctr = np.array(
        [
            npp.polyval2d(
                proj_loc_xrowycol[0],
                proj_loc_xrowycol[1],
                grid_ew[d].get("DeltaKCOAPoly", [[0.0]]),
            )
            for d in ("Row", "Col")
        ]
    )
    rc_ss = [grid_ew[d]["SS"] for d in ("Row", "Col")]
    phase = [
        -delta_kctr[ndx]
        * (np.arange(chip.shape[ndx]) - chip.shape[ndx] // 2)
        * rc_ss[ndx]
        for ndx in range(2)
    ]

    basebanded = chip.copy()
    basebanded *= np.exp(1j * 2 * np.pi * (phase[0][:, np.newaxis] + phase[1]))

    ipr_measurements = _ipr.analyze_complex_ipr(
        basebanded, search_center_rc_frac, search_dist=search_size_px
    )

    feature_props["observed_location_offset_xrowycol"] = (
        search_offset + ipr_measurements["offset_rc"] * rc_ss
    )
    feature_props["valid"] = True
    feature_props["peak_power"] = ipr_measurements["peak_power"]


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
