import argparse
import json
import pathlib

import numpy as np
import sarkit.sicd as sksicd
import sarkit.sidd as sksidd
import sarkit.wgs84
import shapely

from . import _sicd_utils, sicd_chip_to_html, sidd_chip_to_html


def _get_shared_valid_data(data, sicd_xmltree, sidd_xmltree):
    sicd_ew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    sidd_ew = sksidd.ElementWrapper(sidd_xmltree.getroot())

    sidd_vd_rc = sidd_ew["Measurement"]["ValidData"]
    sidd_vd_ecf = sksidd.calculations.pixel_to_ecef(sidd_xmltree, sidd_vd_rc)
    proj_sidd_vd_xrow_ycol, _, success = sksicd.scene_to_image(
        sicd_xmltree, sidd_vd_ecf
    )
    assert success

    proj_sidd_vd_rc = sksicd.xrowycol_to_rowcol(sicd_xmltree, proj_sidd_vd_xrow_ycol)
    proj_sidd_poly = shapely.Polygon(proj_sidd_vd_rc)
    if (
        np.ptp(np.asarray(proj_sidd_poly.bounds).reshape((2, 2)), axis=0).min()
        > 2 * sidd_chip_to_html.NOMINAL_CHIP_SIZE_PX
    ):
        proj_sidd_poly = proj_sidd_poly.buffer(
            -0.5 * sidd_chip_to_html.NOMINAL_CHIP_SIZE_PX
        )
    sicd_vd_poly = _sicd_utils.get_validdata_polygon(sicd_ew)
    shared_valid_region_poly = proj_sidd_poly.intersection(sicd_vd_poly)

    rows, cols = np.indices(data.shape)
    mask = shapely.contains_xy(shared_valid_region_poly, rows, cols)
    return np.ma.masked_array(data, np.logical_not(mask))


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Chip brightest pixel supported in both a SICD and SIDD."
    )
    parser.add_argument(
        "input_sicd_filename", type=pathlib.Path, help="Input SICD file"
    )
    parser.add_argument(
        "input_sidd_filename",
        type=pathlib.Path,
        help="Input SIDD file (must be 2.0 or 3.0)",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="Directory where output HTMLs will be placed",
    )
    config = parser.parse_args(args)

    with config.input_sidd_filename.open("rb") as f, sksidd.NitfReader(f) as r:
        # TODO: This tool uses only ValidData from the first SIDD image
        sidd_xmltree = r.metadata.images[0].xmltree

    with config.input_sicd_filename.open("rb") as f, sksicd.NitfReader(f) as r:
        sicd_xmltree = r.metadata.xmltree
        input_sicd_data = r.read_image()
        sicd_ew = sksicd.ElementWrapper(sicd_xmltree.getroot())

    if sicd_ew["ImageData"]["PixelType"] == "RE16I_IM16I":
        input_sicd_data = (
            input_sicd_data["real"].astype(np.float32)
            + 1j * input_sicd_data["imag"].astype(np.float32)
        ).astype(np.complex64)
    elif sicd_ew["ImageData"]["PixelType"] == "AMP8I_PHS8I":
        # TODO: Handle 8-bit amp/phase
        raise NotImplementedError("SICDs with PixelType=AMP8I_PHS8I are not supported")

    masked_sicd_data = _get_shared_valid_data(
        input_sicd_data, sicd_xmltree, sidd_xmltree
    )

    max_rc = np.unravel_index(np.abs(masked_sicd_data).argmax(), masked_sicd_data.shape)
    max_xryc = sksicd.rowcol_to_xrowycol(sicd_xmltree, max_rc)

    # Use RadarCollection.Area.Plane metadata if available, default to SCP
    if sicd_ew["RadarCollection"]["Area"]["Plane"] is not None:
        ref_pt = sicd_ew["RadarCollection"]["Area"]["Plane"]["RefPt"]["ECF"]
        ux = sicd_ew["RadarCollection"]["Area"]["Plane"]["XDir"]["UVectECF"]
        uy = sicd_ew["RadarCollection"]["Area"]["Plane"]["YDir"]["UVectECF"]
        unorm = np.cross(ux, uy)
    else:
        ref_pt = sicd_ew["GeoData"]["SCP"]["ECF"]
        unorm = sarkit.wgs84.up(sicd_ew["GeoData"]["SCP"]["LLH"])

    max_ecef, _, success = sksicd.image_to_ground_plane(
        sicd_xmltree, max_xryc, ref_pt, unorm
    )
    assert success
    max_llh = sarkit.wgs84.cartesian_to_geodetic(max_ecef)
    # GeoJSON order is (Lon, Lat, HAE)
    max_llh = max_llh[[1, 0, 2]]

    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": max_llh.tolist(),
        },
    }
    geojson_file = config.output_dir / "geo.json"
    sicd_chip_file = config.output_dir / "sicd_chip.html"
    sidd_chip_file = config.output_dir / "sidd_chip.html"
    with open(geojson_file, "w") as f:
        json.dump(geojson, f, indent=2)

    sicd_chip_to_html.main(
        [str(config.input_sicd_filename), str(geojson_file), str(sicd_chip_file)]
    )
    sidd_chip_to_html.main(
        [str(config.input_sidd_filename), str(geojson_file), str(sidd_chip_file)]
    )


if __name__ == "__main__":
    main()  # pragma: no cover
