import argparse
import json
import pathlib

import numpy as np
import sarkit.sicd as sksicd
import sarkit.wgs84

from sarkit_assurance import sicd_chip_to_html, sidd_chip_to_html


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Chip brightest pixel supported in both a SICD and SIDD."
    )
    parser.add_argument("input_sicd_filename", type=pathlib.Path)
    parser.add_argument("input_sidd_filename", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    config = parser.parse_args(args)

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
            raise NotImplementedError(
                "SICDs with PixelType=AMP8I_PHS8I are not supported"
            )

        # TODO: Ensure that the bright point is contained within both SICD and SIDD valid data

        max_rc = np.unravel_index(
            np.abs(input_sicd_data).argmax(), input_sicd_data.shape
        )
        max_xryc = sksicd.rowcol_to_xrowycol(sicd_xmltree, max_rc)

        # Use RadarCollection.Area.Plane metadata if available, default to SCP
        if sicd_ew["RadarCollection"]["Area"]["Plane"] is not None:
            ref_pt = sicd_ew["RadarCollection"]["Area"]["Plane"]["RefPt"]["ECF"]
            ux = sicd_ew["RadarCollection"]["Area"]["Plane"]["XDir"]["UVectECF"]
            uy = sicd_ew["RadarCollection"]["Area"]["Plane"]["YDir"]["UVectECF"]
            unorm = np.cross(ux, uy)
        else:
            ref_pt = sicd_ew["GeoData"]["SCP"]["ECF"]
            unorm = ref_pt / np.linalg.norm(ref_pt)

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
        geojson_file = config.output_path / "geo.json"
        sicd_chip_file = config.output_path / "sicd_chip.html"
        sidd_chip_file = config.output_path / "sidd_chip.html"
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
