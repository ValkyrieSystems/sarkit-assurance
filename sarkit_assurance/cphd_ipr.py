"""Perform IPR analysis on CPHD Targets"""

import argparse
import json
import pathlib
from typing import Any

import lxml.etree
import numpy as np
import numpy.typing as npt
import sarkit.cphd as skcphd
import sarkit.wgs84
import shapely

from . import _ipr

try:
    from smart_open import open
except ImportError:
    pass

NUM_IAC_PAD = 10


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
    step_x /= np.linalg.norm(ecef_cal_points[1, :] - ecef_cal_points[0, :])
    step_y /= np.linalg.norm(ecef_cal_points[2, :] - ecef_cal_points[0, :])

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


def get_channel_image_area(
    cphd_xmltree: lxml.etree.ElementTree, ch_id: str
) -> shapely.Polygon:
    # TODO: move to sarkit
    ew = skcphd.ElementWrapper(cphd_xmltree.getroot())

    def get_imagearea_poly(imgarea_ew):
        region = shapely.box(*imgarea_ew["X1Y1"], *imgarea_ew["X2Y2"])
        if (poly := imgarea_ew.get("Polygon", None)) is not None:
            region = region.intersection(shapely.Polygon(poly))
        return region

    ia_poly = get_imagearea_poly(ew["SceneCoordinates"]["ImageArea"])
    chan_param_ew = ew["Channel"].find("Parameters", Identifier=ch_id)
    if "ImageArea" in chan_param_ew:
        ia_poly = get_imagearea_poly(chan_param_ew["ImageArea"])
    return ia_poly


def analyze(
    cphd_reader: skcphd.Reader,
    geojson: dict[str, Any],
    ch_id: str,
    outdir: pathlib.Path,
) -> None:
    """Perform IPR analysis of CPHD targets.

    TODO: finish docstring
    """

    # Find target image coordinates and ensure they are (nearly) within ImageArea
    cphd_xmltree = cphd_reader.metadata.xmltree
    imagearea_iac = get_channel_image_area(cphd_xmltree, ch_id)
    padded_imagearea_iac = imagearea_iac.buffer(NUM_IAC_PAD, quad_segs=4)
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

    (outdir / "cphd_ipr.json").write_text(
        json.dumps(geojson, cls=_ipr.NdArrJSONEncoder, indent=4)
    )


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
    parser.add_argument("cphd_file", help="Input CPHD file")
    parser.add_argument("geojson_file", help="Input GeoJSON file")
    parser.add_argument("out_dir", help="Directory to store results", type=pathlib.Path)
    config = parser.parse_args(args)

    with open(config.geojson_file, "rb") as file:
        geo = json.load(file)

    with open(config.cphd_file, "rb") as f, skcphd.Reader(f) as r:
        # TODO: figure out what we want to do about multi-channel
        ch_id = r.metadata.xmltree.findtext("{*}Channel/{*}RefChId")
        analyze(r, geo, ch_id, config.out_dir)


if __name__ == "__main__":
    main()
