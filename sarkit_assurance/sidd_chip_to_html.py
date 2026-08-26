import argparse
import html
import json
import logging
import pathlib
import sys

import numpy as np
import numpy.polynomial.polynomial as npp
import plotly.graph_objects as go
import plotly.offline
import sarkit.sidd as sksidd
import sarkit.wgs84
import shapely.geometry as shg

from . import _geojson, _remap

try:
    from smart_open import open
except ImportError:
    pass

NOMINAL_CHIP_SIZE_PX = 512


def _ref_pt_row_col(sidd_ew):
    cs_type = sksidd.get_coordinate_system_type(sidd_ew.elem)
    if cs_type == sksidd.calculations.CoordinateSystem.PGD:
        return sidd_ew["Measurement"]["PlaneProjection"]["ReferencePoint"]["Point"]
    elif cs_type == sksidd.calculations.CoordinateSystem.PFGD:
        return sidd_ew["Measurement"]["PolynomialProjection"]["ReferencePoint"]["Point"]
    else:
        raise NotImplementedError(cs_type)


def _rg_rgrate_from_pos_vel_tgt(pos, vel, tgt):
    rg = np.linalg.norm(pos - tgt, axis=-1)
    u_tgt = (pos - tgt) / rg[..., np.newaxis]
    rg_rate = np.vecdot(vel, u_tgt)
    return np.stack([rg, rg_rate], axis=-1)


def _pixel_location_to_plane_proj_coa_time(sidd_ew, row_col):
    reference_pixel = np.asarray(_ref_pt_row_col(sidd_ew))
    sample_spacing = sidd_ew["Measurement"]["PlaneProjection"]["SampleSpacing"]
    row_delta = np.asarray(row_col)[..., 0] - reference_pixel[0]
    col_delta = np.asarray(row_col)[..., 1] - reference_pixel[1]
    row_meters_from_scene_center = row_delta * sample_spacing[0]
    col_meters_from_scene_center = col_delta * sample_spacing[1]

    time_coa_poly = sidd_ew["Measurement"]["PlaneProjection"]["TimeCOAPoly"]
    return npp.polyval2d(
        row_meters_from_scene_center, col_meters_from_scene_center, time_coa_poly
    )


def _get_pos_vel_at_pixel_coa(sidd_ew, row_col):
    arp_poly = sidd_ew["Measurement"]["ARPPoly"]

    cs_type = sksidd.get_coordinate_system_type(sidd_ew.elem)
    if cs_type == sksidd.calculations.CoordinateSystem.PGD:
        coa_time = _pixel_location_to_plane_proj_coa_time(sidd_ew, row_col)
    elif cs_type == sksidd.calculations.CoordinateSystem.PFGD:
        # This is at best a rough estimate
        coa_time = (
            sidd_ew["ExploitationFeatures"]["Collection"][0]["Information"][
                "CollectionDuration"
            ]
            / 2.0
        )

    arp_pos = np.moveaxis(npp.polyval(coa_time, arp_poly), 0, -1)
    arp_vel = np.moveaxis(npp.polyval(coa_time, npp.polyder(arp_poly, 1)), 0, -1)

    return arp_pos, arp_vel


def _get_valid_index_data_polygon(sidd_ew):
    valid_data = sidd_ew["Measurement"].get("ValidData", None)

    if valid_data is None:
        num_rows = sidd_ew["Measurement"]["PixelFootprint"][0]
        num_cols = sidd_ew["Measurement"]["PixelFootprint"][1]
        valid_data = [
            (0, 0),
            (0, num_cols - 1),
            (num_rows - 1, num_cols - 1),
            (num_rows - 1, 0),
        ]

    return shg.Polygon(valid_data)


def _proj_ecef_to_image(sidd_ew, tgt_ecef):
    image_shape = np.array(sidd_ew["Measurement"]["PixelFootprint"])

    ref_pt_row_col = _ref_pt_row_col(sidd_ew)
    curr_pt = ref_pt_row_col
    max_iter = 11
    for _ in range(max_iter):
        pos, vel = _get_pos_vel_at_pixel_coa(sidd_ew, curr_pt)
        tgt_r_rdot = _rg_rgrate_from_pos_vel_tgt(pos, vel, tgt_ecef)
        plus_row = curr_pt + [1, 0]
        plus_col = curr_pt + [0, 1]
        row_col_ecef = sksidd.pixel_to_ecef(
            sidd_ew.elem.getroottree(), np.stack([curr_pt, plus_row, plus_col], axis=0)
        )
        r_rdot_row_col = _rg_rgrate_from_pos_vel_tgt(pos, vel, row_col_ecef)
        d_r_rdot_d_row_col = r_rdot_row_col[1:] - r_rdot_row_col[0]
        d_row_col_d_r_rdot = np.linalg.inv(d_r_rdot_d_row_col)
        tgt_delta_r_rdot = tgt_r_rdot - r_rdot_row_col[0]
        delta_row_col = tgt_delta_r_rdot @ d_row_col_d_r_rdot
        curr_pt = np.clip(curr_pt + delta_row_col, [0, 0], image_shape)
        if np.linalg.norm(delta_row_col) < 0.1:
            success = True
            break
    else:
        success = False

    return curr_pt, success


def create_sidd_chip_plot(image, ew, feature):
    """Plotly figure containing pixels covering a GeoJSON feature

    Parameters
    ----------
    image : ndarray
        SIDD image from which to chip
    ew : sksidd.ElementWrapper
        SIDD metadata corresponding to `image`
    feature : dict
        GeoJSON Feature

    Returns
    -------
    plotly.graph_objects.Figure
    dict
        GeoJSON Feature
    """
    try:
        coord_llh = _geojson.get_feature_point(feature)
    except ValueError as exc:
        logging.warning(exc)
        return None

    coord_ecef = sarkit.wgs84.geodetic_to_cartesian(coord_llh)

    image_shape = np.array(ew["Measurement"]["PixelFootprint"])

    image_grid_index, success = _proj_ecef_to_image(ew, coord_ecef)
    if not success:
        logging.debug("Feature not in image")
        return None

    start_rc = np.floor(image_grid_index - NOMINAL_CHIP_SIZE_PX // 2).astype(np.int64)
    stop_rc = np.ceil(start_rc + NOMINAL_CHIP_SIZE_PX).astype(np.int64)
    start_rc = np.maximum(start_rc, 0)
    stop_rc = np.minimum(stop_rc, image_shape)

    # skip if too close to the edge
    if np.min(stop_rc - start_rc, axis=0) < NOMINAL_CHIP_SIZE_PX / 4:
        return None

    subimage = image[start_rc[0] : stop_rc[0], start_rc[1] : stop_rc[1], ...]
    if np.issubdtype(subimage.dtype, np.integer) and subimage.dtype.itemsize == 2:
        subimage = _remap.simple_log_remap(subimage, min_low_relative=1e-3).astype(
            np.uint8
        )

    fig = go.Figure()
    if len(subimage.shape) == 2:
        fig.add_heatmap(
            z=subimage,
            x0=start_rc[1],
            y0=start_rc[0],
            dx=1,
            dy=1,
            showscale=False,
            colorscale="gray",
            name="image",
        )
    else:
        fig.add_image(
            z=subimage,
            x0=start_rc[1],
            y0=start_rc[0],
            dx=1,
            dy=1,
            name="image",
        )
    fig.add_scatter(
        x=image_grid_index[1:],
        y=image_grid_index[:1],
        mode="markers",
        marker_color="red",
        name="expected location",
    )
    fig.update_xaxes(title_text="Image Column (px)")
    fig.update_yaxes(title_text="Image Row (px)", autorange="reversed")

    title_text = f"pixel row/col: {image_grid_index[0]:0.2f} {image_grid_index[1]:0.2f}"
    valid_data_polygon = _get_valid_index_data_polygon(ew)
    if not valid_data_polygon.contains(shg.Point(image_grid_index)):
        title_text += " [outside ValidData]"

    fig.update_layout(title_text=title_text)
    return fig, feature


def write_html_file(html_filename, image_plot_dict, extra_metadata):
    """Create an HTML file containing plots

    Parameters
    ----------
    html_filename : str or path-like
    image_plot_dict : dict
        dictionary of image number to plots
    extra_metadata : dict
        extra metadata to put at top of the html
    """
    style = """
        <style>
        table {
            border-collapse: collapse;
        }
        table, tr, td {
            border: 1px solid black;
        }
        </style>
    """
    htmllines = [
        "<html>",
        '<head><meta charset="utf-8"/></head>',
        style,
        "<body>",
        f'<script type="text/javascript">{plotly.offline.offline.get_plotlyjs()}</script>',
    ]
    if extra_metadata:
        htmllines.append("<table>")
        for key, value in extra_metadata.items():
            htmllines.append(
                f"<tr><td><b><pre>{html.escape(key)}</pre></b></td><td><pre>{html.escape(value)}</pre></td></tr>"
            )
        htmllines.append("</table>")

    for image, plots in image_plot_dict.items():
        htmllines.append("<div>")
        htmllines.append(f"{image} plots:")
        htmllines.append("</div>")
        htmllines.append('<div style="display:flex;flex-wrap:wrap">')
        for fig, feature in plots:
            htmllines.append(
                '<div style="display:flex;flex-direction:column;border: 1px dotted grey;">'
            )

            htmllines.append('<div style="width:600;">')
            htmllines.append("<pre>")
            htmllines.append(json.dumps(feature, indent=2))
            htmllines.append("</pre>")
            htmllines.append("</div>")

            htmllines.append("<div>")
            htmllines.append(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    default_width=600,
                    default_height=600,
                )
            )
            htmllines.append("</div>")

            htmllines.append("</div>")

        htmllines.append("</div>")

    htmllines.append("</body>")
    htmllines.append("</html>")

    pathlib.Path(html_filename).write_text("\n".join(htmllines), encoding="utf=8")


def _get_image(reader, image_num):
    img_meta = reader.metadata.images[image_num]
    arr = reader.read_image(image_num)
    px_type = img_meta.xmltree.findtext("{*}Display/{*}PixelType")
    if px_type == "MONO8I":
        img = arr
    elif px_type == "MONO16I":
        img = arr
    elif px_type == "MONO8LU":
        lut = img_meta.lookup_table
        img = lut[arr]
    elif px_type == "RGB24I":
        img = arr[..., np.newaxis].view(np.uint8)
    elif px_type == "RGB8LU":
        lut = img_meta.lookup_table
        img = lut[arr][..., np.newaxis].view(np.uint8)
    return img


def main(args=None):
    parser = argparse.ArgumentParser(description="Plot GeoJSON Features")
    parser.add_argument("sidd_file", help="Input SIDD file (must be 2.0 or 3.0)")
    parser.add_argument("geojson_file", help="Input GeoJSON file")
    parser.add_argument("output_html_file", help="Output HTML file")
    config = parser.parse_args(args)

    with open(config.geojson_file, "rb") as file:
        geo = json.load(file)

    image_plot_dict = dict()
    plot_exists = False
    with open(config.sidd_file, "rb") as file, sksidd.NitfReader(file) as reader:
        num_images = len(reader.metadata.images)
        for image_num in range(num_images):
            plots = []
            xmltree = reader.metadata.images[image_num].xmltree
            sidd_ew = sksidd.ElementWrapper(xmltree.getroot())
            image = _get_image(reader, image_num)
            for feature in _geojson.features(geo):
                plot = create_sidd_chip_plot(image, sidd_ew, feature)
                if plot is None:
                    continue
                plot_exists = True
                plots.append(plot)
            image_plot_dict[f"Image {image_num + 1}"] = plots

    if not plot_exists:
        logging.error("No plots created")
        return 1

    extra_metadata = {
        "SIDD File": str(config.sidd_file),
        "GeoJSON File": str(config.geojson_file),
    }
    write_html_file(config.output_html_file, image_plot_dict, extra_metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
