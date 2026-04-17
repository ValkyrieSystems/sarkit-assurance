import argparse
import html
import json
import logging
import pathlib
import sys
import typing

import numpy as np
import plotly.graph_objects as go
import plotly.offline
import sarkit.sicd as sksicd
import sarkit.wgs84
import shapely.geometry as shg

from sarkit_assurance import _remap

try:
    from smart_open import open
except ImportError:
    pass

NOMINAL_CHIP_SIZE_PX = 512


def _scp_centered_coord_to_global_index(ew, scp_centered_coord):
    spacing = np.array([ew["Grid"]["Row"]["SS"], ew["Grid"]["Col"]["SS"]])
    origin = ew["ImageData"]["SCPPixel"]
    return scp_centered_coord / spacing + origin


def _get_valid_index_data_polygon(ew):
    valid_data = ew["ImageData"].get("ValidData", None)

    if valid_data is None:
        valid_data = [
            (0, 0),
            (0, ew["ImageData"]["NumCols"] - 1),
            (ew["ImageData"]["NumRows"], ew["ImageData"]["NumCols"] - 1),
            (ew["ImageData"]["NumRows"], 0),
        ]

    return shg.Polygon(valid_data)


def create_sicd_chip_plot(reader: sksicd.NitfReader, feature: dict):
    """Plotly figure containing pixels covering a GeoJSON feature

    Parameters
    ----------
    reader : sarkit.sicd.NitfReader
        open SICD reader object
    feature : dict
        GeoJSON Feature

    Returns
    -------
    plotly.graph_objects.Figure
    dict
        GeoJSON Feature
    """
    coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
    if feature["geometry"]["type"] != "Point" or coordinates.shape != (3,):
        logging.warning("Only 3D Point features are supported")
        return None

    coord_llh = [coordinates[1], coordinates[0], coordinates[2]]
    coord_ecef = sarkit.wgs84.geodetic_to_cartesian(coord_llh)

    xmltree = reader.metadata.xmltree
    ew = sksicd.ElementWrapper(xmltree.getroot())
    first_pixel = np.array([ew["ImageData"]["FirstRow"], ew["ImageData"]["FirstCol"]])
    image_shape = np.array([ew["ImageData"]["NumRows"], ew["ImageData"]["NumCols"]])

    image_grid_loc, delta_gp, success = sksicd.scene_to_image(xmltree, coord_ecef)
    if not success:
        logging.debug("Feature not in image")
        return None

    image_grid_index = _scp_centered_coord_to_global_index(ew, image_grid_loc)

    start_rc = np.floor(
        image_grid_index - first_pixel - NOMINAL_CHIP_SIZE_PX // 2
    ).astype(np.int64)
    stop_rc = np.ceil(start_rc + NOMINAL_CHIP_SIZE_PX).astype(np.int64)
    start_rc = np.maximum(start_rc, 0)
    stop_rc = np.minimum(stop_rc, image_shape)

    # skip if too close to the edge
    if np.min(stop_rc - start_rc, axis=0) < NOMINAL_CHIP_SIZE_PX / 4:
        return None

    subimage, subimage_xmltree = reader.read_sub_image(
        start_row=start_rc[0],
        start_col=start_rc[1],
        stop_row=stop_rc[0],
        stop_col=stop_rc[1],
    )

    subew = sksicd.ElementWrapper(subimage_xmltree.getroot())
    sub_loc, _, _ = sksicd.scene_to_image(subimage_xmltree, coord_ecef)
    sub_index = _scp_centered_coord_to_global_index(subew, sub_loc)

    fig = go.Figure()
    fig.add_heatmap(
        z=_remap.simple_sicd_remap(subimage),
        x0=-(sub_index[1] - subew["ImageData"]["FirstCol"])
        * subew["Grid"]["Col"]["SS"],
        y0=-(sub_index[0] - subew["ImageData"]["FirstRow"])
        * subew["Grid"]["Row"]["SS"],
        dx=subew["Grid"]["Col"]["SS"],
        dy=subew["Grid"]["Row"]["SS"],
        showscale=False,
        colorscale="gray",
        name="image",
    )
    fig.add_scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker_color="red",
        name="expected location",
    )
    fig.update_xaxes(title_text="distance from expected (m)")
    fig.update_yaxes(title_text="distance from expected (m)", autorange="reversed")

    title_text = f"pixel row/col: {sub_index[0]:0.2f} {sub_index[1]:0.2f}"
    valid_data_polygon = _get_valid_index_data_polygon(subew)
    if not valid_data_polygon.contains(shg.Point(sub_index)):
        title_text += " [outside ValidData]"

    fig.update_layout(title_text=title_text)
    return fig, feature


def write_html_file(
    html_filename: str | pathlib.Path,
    plots: typing.Iterable[tuple[plotly.graph_objects.Figure, dict]],
    extra_metadata: dict,
) -> None:
    """Create an HTML file containing plots

    Properties
    ----------
    html_filename : str or path-like
    plots : iterable
        Tuples of (plotly figure, GeoJSON feature dictionary)
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

    with open(html_filename, "w") as file:
        file.write("\n".join(htmllines))


def _get_all_features(geojson):
    """Iterate over each feature in a GeoJSON"""
    if geojson["type"] == "Feature":
        return [geojson]
    elif geojson["type"] == "FeatureCollection":
        return geojson["features"]
    return []


def main(args=None):
    parser = argparse.ArgumentParser(description="Plot GeoJSON Features")
    parser.add_argument("sicd_file", help="Input SICD file")
    parser.add_argument("geojson_file", help="Input GeoJSON file")
    parser.add_argument("output_html_file", help="Output HTML file")
    config = parser.parse_args(args)

    with open(config.geojson_file, "rb") as file:
        geo = json.load(file)

    plots = []
    with open(config.sicd_file, "rb") as file, sksicd.NitfReader(file) as reader:
        for feature in _get_all_features(geo):
            plot = create_sicd_chip_plot(reader, feature)
            if plot is None:
                continue
            plots.append(plot)

    if not plots:
        logging.error("No plots created")
        return 1

    extra_metadata = {
        "SICD File": str(config.sicd_file),
        "GeoJSON File": str(config.geojson_file),
    }
    write_html_file(config.output_html_file, plots, extra_metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
